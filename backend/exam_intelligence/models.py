"""
Exam-intelligence data model for BacPilot AI.

Design notes:
- Reference tables (BacSection, Subject, CurriculumEra, Chapter, Concept) are small
  and seeded once. They define the curriculum space.
- Exam tables encode the *structure* of the Tunisian Bac (section/session/year/era,
  exercise -> question tree, points/barème, corrections, rubrics).
- EmbeddingChunk denormalizes filter columns on purpose: we filter BEFORE vector
  search, so subject/section/era/year must live on the chunk row.
- Student tables capture the learner's state for personalization + readiness.

Nullability and uniqueness are called out per field. Everything that can be
auto-detected by ingestion is nullable, because ingestion is allowed to be unsure.
"""

from django.conf import settings
from django.db import models
from pgvector.django import VectorField


# --------------------------------------------------------------------------- #
# Choices
# --------------------------------------------------------------------------- #

class Session(models.TextChoices):
    PRINCIPALE = "principale", "Session principale"
    CONTROLE = "controle", "Session de contrôle"


class Difficulty(models.TextChoices):
    EASY = "easy", "Facile"
    MEDIUM = "medium", "Moyen"
    HARD = "hard", "Difficile"


class RelevanceStatus(models.TextChoices):
    RELEVANT = "relevant", "Pertinent (programme actuel)"
    PARTIALLY = "partially", "Partiellement pertinent"
    OUTDATED = "outdated", "Hors programme actuel"
    UNREVIEWED = "unreviewed", "Non revu"


class ReviewStatus(models.TextChoices):
    PENDING = "pending", "En attente de revue"
    APPROVED = "approved", "Approuvé"
    REJECTED = "rejected", "Rejeté"


class Reliability(models.TextChoices):
    HIGH = "high", "Élevée"
    MEDIUM = "medium", "Moyenne"
    LOW = "low", "Faible"


class Frequency(models.TextChoices):
    """Exam frequency: how often a concept appears across exam years (used by tags)."""
    RARE = "rare", "Rare"
    OCCASIONAL = "occasional", "Occasionnel"
    FREQUENT = "frequent", "Fréquent"


class MistakeFrequency(models.TextChoices):
    """Mistake frequency: how often students make a given mistake.

    Distinct vocabulary from Frequency (exam frequency). Aligned with the JSON
    schema enum used by seed_data so example exercises load without violating
    the field choices.
    """
    RARE = "rare", "Rare"
    OCCASIONAL = "occasional", "Occasionnel"
    COMMON = "common", "Fréquent"
    VERY_COMMON = "very_common", "Très fréquent"


class Language(models.TextChoices):
    FR = "fr", "Français"
    AR = "ar", "العربية"
    DARJA = "darja", "Darija tunisienne"


class EmbeddingStatus(models.TextChoices):
    """Lifecycle of a chunk's vector.

    pending = text + metadata stored, no vector yet (default).
    mock    = deterministic local placeholder vector (NOT a real embedding).
    ready   = real embedding produced by an embedding model.
    """
    PENDING = "pending", "En attente"
    MOCK = "mock", "Mock déterministe"
    READY = "ready", "Prêt"


# --------------------------------------------------------------------------- #
# Reference / curriculum
# --------------------------------------------------------------------------- #

class BacSection(models.Model):
    """A Bac track. Scope: MATH, SC_EXP. Extensible later."""
    code = models.CharField(max_length=16, unique=True)          # e.g. "MATH", "SC_EXP"
    name_fr = models.CharField(max_length=120)
    name_ar = models.CharField(max_length=120, blank=True)

    def __str__(self):
        return self.name_fr


class Subject(models.Model):
    """A subject (matière). Distinct from section: a section contains several subjects."""
    code = models.CharField(max_length=16, unique=True)          # "MATH", "PHYSIQUE", "SVT"
    name_fr = models.CharField(max_length=120)
    name_ar = models.CharField(max_length=120, blank=True)

    def __str__(self):
        return self.name_fr


class SectionSubject(models.Model):
    """
    Coefficient lives on the (section, subject) pair, because Maths has a different
    coefficient in the Math section than in the Sciences section.
    NOTE: coefficients here are PLACEHOLDERS — verify against the official Bac.
    """
    section = models.ForeignKey(BacSection, on_delete=models.CASCADE, related_name="section_subjects")
    subject = models.ForeignKey(Subject, on_delete=models.CASCADE, related_name="subject_sections")
    coefficient = models.DecimalField(max_digits=4, decimal_places=2)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["section", "subject"], name="uniq_section_subject"),
        ]

    def __str__(self):
        return f"{self.section.code}/{self.subject.code} (coef {self.coefficient})"


class CurriculumEra(models.Model):
    """
    A curriculum period. Drives relevance weighting (Part 6).
    end_year null => current era. relevance_weight in [0, 1].
    Era boundaries MUST be validated with a Tunisian Bac teacher.
    """
    code = models.CharField(max_length=32, unique=True)         # "ERA_2016_2025"
    label = models.CharField(max_length=120)
    start_year = models.PositiveSmallIntegerField()
    end_year = models.PositiveSmallIntegerField(null=True, blank=True)
    relevance_weight = models.DecimalField(max_digits=3, decimal_places=2, default=1.0)

    class Meta:
        ordering = ["-start_year"]

    def __str__(self):
        end = self.end_year or "présent"
        return f"{self.label} ({self.start_year}–{end})"


class Chapter(models.Model):
    """A chapter within a subject for a given era (curriculum reshapes chapters)."""
    subject = models.ForeignKey(Subject, on_delete=models.CASCADE, related_name="chapters")
    era = models.ForeignKey(CurriculumEra, on_delete=models.PROTECT, related_name="chapters")
    code = models.CharField(max_length=48)                      # "PROBA", "COMPLEXES"
    name_fr = models.CharField(max_length=160)
    name_ar = models.CharField(max_length=160, blank=True)
    order = models.PositiveSmallIntegerField(default=0)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["subject", "era", "code"], name="uniq_chapter"),
        ]
        ordering = ["subject", "order"]

    def __str__(self):
        return f"{self.subject.code}:{self.code}"


class Concept(models.Model):
    """A concept (or sub-concept via parent) inside a chapter."""
    chapter = models.ForeignKey(Chapter, on_delete=models.CASCADE, related_name="concepts")
    parent = models.ForeignKey("self", on_delete=models.CASCADE, null=True, blank=True,
                               related_name="children")
    code = models.CharField(max_length=64)                      # "loi_binomiale"
    name_fr = models.CharField(max_length=200)
    name_ar = models.CharField(max_length=200, blank=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["chapter", "code"], name="uniq_concept"),
        ]

    def __str__(self):
        return self.code


# --------------------------------------------------------------------------- #
# Source documents & exams
# --------------------------------------------------------------------------- #

class SourceDocument(models.Model):
    """A raw ingested file (an exam or a correction). Detections are nullable: ingestion may be unsure."""
    DOC_TYPES = [("exam", "Énoncé"), ("correction", "Correction")]

    file = models.CharField(max_length=512)                     # path or storage URL
    original_filename = models.CharField(max_length=512)
    doc_type = models.CharField(max_length=16, choices=DOC_TYPES)
    is_scanned = models.BooleanField(default=False)

    detected_year = models.PositiveSmallIntegerField(null=True, blank=True)
    detected_session = models.CharField(max_length=16, choices=Session.choices, null=True, blank=True)
    detected_section = models.ForeignKey(BacSection, on_delete=models.SET_NULL, null=True, blank=True)
    detected_subject = models.ForeignKey(Subject, on_delete=models.SET_NULL, null=True, blank=True)

    ocr_engine = models.CharField(max_length=32, blank=True)    # "pdfplumber", "tesseract", "mathpix"
    confidence_score = models.DecimalField(max_digits=4, decimal_places=3, null=True, blank=True)

    review_status = models.CharField(max_length=16, choices=ReviewStatus.choices,
                                     default=ReviewStatus.PENDING)
    reviewed_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
                                    null=True, blank=True, related_name="reviewed_documents")

    raw_text = models.TextField(blank=True)                    # never destroyed
    cleaned_text = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=["review_status"]),
            models.Index(fields=["detected_year", "detected_subject"]),
        ]

    def __str__(self):
        return f"{self.original_filename} [{self.doc_type}]"


class Exam(models.Model):
    """One exam = one (section, subject, year, session) slot."""
    section = models.ForeignKey(BacSection, on_delete=models.PROTECT, related_name="exams")
    subject = models.ForeignKey(Subject, on_delete=models.PROTECT, related_name="exams")
    year = models.PositiveSmallIntegerField()
    session = models.CharField(max_length=16, choices=Session.choices)
    era = models.ForeignKey(CurriculumEra, on_delete=models.PROTECT, related_name="exams")
    source_document = models.ForeignKey(SourceDocument, on_delete=models.SET_NULL,
                                        null=True, blank=True, related_name="exams")

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["section", "subject", "year", "session"],
                                    name="uniq_exam_slot"),
        ]
        indexes = [models.Index(fields=["subject", "year"])]

    def __str__(self):
        return f"Bac {self.year} {self.session} {self.section.code}/{self.subject.code}"


class ExamExercise(models.Model):
    """An exercise within an exam. Difficulty/relevance default unset until tagged/reviewed."""
    exam = models.ForeignKey(Exam, on_delete=models.CASCADE, related_name="exercises")
    number = models.PositiveSmallIntegerField()
    title = models.CharField(max_length=200, blank=True)
    intro_text = models.TextField(blank=True)                  # shared statement before sub-questions
    total_points = models.DecimalField(max_digits=4, decimal_places=2, null=True, blank=True)

    difficulty = models.CharField(max_length=8, choices=Difficulty.choices, null=True, blank=True)
    estimated_minutes = models.PositiveSmallIntegerField(null=True, blank=True)
    relevance_status = models.CharField(max_length=12, choices=RelevanceStatus.choices,
                                        default=RelevanceStatus.UNREVIEWED)
    # Per-exercise weight override (teacher can set; otherwise derived from era).
    relevance_weight = models.DecimalField(max_digits=3, decimal_places=2, null=True, blank=True)
    validated_by_teacher = models.BooleanField(default=False)

    chapters = models.ManyToManyField(Chapter, blank=True, related_name="exercises")
    concepts = models.ManyToManyField(Concept, blank=True, related_name="exercises")

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["exam", "number"], name="uniq_exercise_in_exam"),
        ]
        ordering = ["exam", "number"]

    def __str__(self):
        return f"{self.exam} — Ex.{self.number}"

    @property
    def effective_relevance_weight(self):
        if self.relevance_weight is not None:
            return self.relevance_weight
        return self.exam.era.relevance_weight


class ExamQuestion(models.Model):
    """A question (possibly nested via parent) inside an exercise."""
    exercise = models.ForeignKey(ExamExercise, on_delete=models.CASCADE, related_name="questions")
    parent = models.ForeignKey("self", on_delete=models.CASCADE, null=True, blank=True,
                               related_name="subquestions")
    number = models.CharField(max_length=16)                   # "1", "2.a"
    text = models.TextField()
    points = models.DecimalField(max_digits=4, decimal_places=2, null=True, blank=True)
    order = models.PositiveSmallIntegerField(default=0)
    concepts = models.ManyToManyField(Concept, blank=True, related_name="questions")

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["exercise", "number"], name="uniq_question_in_exercise"),
        ]
        ordering = ["exercise", "order"]

    def __str__(self):
        return f"{self.exercise} Q{self.number}"


class Correction(models.Model):
    """
    A correction, attached at question OR exercise level (at least one must be set).
    author_type distinguishes official vs teacher vs AI draft.
    """
    AUTHOR_TYPES = [("official", "Officielle"), ("teacher", "Enseignant"), ("ai_draft", "Brouillon IA")]

    question = models.ForeignKey(ExamQuestion, on_delete=models.CASCADE, null=True, blank=True,
                                 related_name="corrections")
    exercise = models.ForeignKey(ExamExercise, on_delete=models.CASCADE, null=True, blank=True,
                                 related_name="corrections")
    source_document = models.ForeignKey(SourceDocument, on_delete=models.SET_NULL,
                                        null=True, blank=True, related_name="corrections")
    text = models.TextField()
    is_official = models.BooleanField(default=False)
    author_type = models.CharField(max_length=16, choices=AUTHOR_TYPES, default="official")
    reliability = models.CharField(max_length=8, choices=Reliability.choices, default=Reliability.HIGH)

    class Meta:
        constraints = [
            models.CheckConstraint(
                name="correction_targets_question_or_exercise",
                check=models.Q(question__isnull=False) | models.Q(exercise__isnull=False),
            ),
        ]

    def __str__(self):
        return f"Correction ({self.author_type}) — {self.question or self.exercise}"


class RubricItem(models.Model):
    """A scoreable step in a correction. Keywords help the correction engine pre-detect the step."""
    correction = models.ForeignKey(Correction, on_delete=models.CASCADE, related_name="rubric_items")
    description = models.CharField(max_length=300)
    points = models.DecimalField(max_digits=4, decimal_places=2)
    order = models.PositiveSmallIntegerField(default=0)
    keywords = models.JSONField(default=list, blank=True)      # ["binomiale", "C(n,k)"]

    class Meta:
        ordering = ["correction", "order"]

    def __str__(self):
        return f"{self.description} ({self.points} pts)"


class CommonMistake(models.Model):
    """A recurring student mistake, linkable to a concept and/or a specific exercise."""
    concept = models.ForeignKey(Concept, on_delete=models.CASCADE, null=True, blank=True,
                                related_name="common_mistakes")
    exercise = models.ForeignKey(ExamExercise, on_delete=models.CASCADE, null=True, blank=True,
                                 related_name="common_mistakes")
    description_fr = models.CharField(max_length=300)
    description_ar = models.CharField(max_length=300, blank=True)
    why_it_happens = models.TextField(blank=True)
    correct_approach = models.TextField(blank=True)
    frequency = models.CharField(max_length=12, choices=MistakeFrequency.choices,
                                 default=MistakeFrequency.COMMON)

    def __str__(self):
        return self.description_fr


class ExerciseTag(models.Model):
    """Flexible tagging on top of structured concept links. source = auto|reviewed."""
    TAG_TYPES = [
        ("skill", "Type de compétence"), ("method", "Méthode requise"),
        ("difficulty", "Difficulté"), ("language", "Langue"),
        ("relevance", "Pertinence"), ("frequency", "Fréquence"), ("other", "Autre"),
    ]
    SOURCES = [("auto", "Automatique"), ("reviewed", "Validé")]

    exercise = models.ForeignKey(ExamExercise, on_delete=models.CASCADE, related_name="tags")
    tag_type = models.CharField(max_length=16, choices=TAG_TYPES)
    value = models.CharField(max_length=120)
    source = models.CharField(max_length=10, choices=SOURCES, default="auto")

    class Meta:
        indexes = [models.Index(fields=["tag_type", "value"])]
        constraints = [
            models.UniqueConstraint(fields=["exercise", "tag_type", "value"], name="uniq_exercise_tag"),
        ]

    def __str__(self):
        return f"{self.tag_type}={self.value}"


# --------------------------------------------------------------------------- #
# Embeddings (vector DB lives in Postgres via pgvector)
# --------------------------------------------------------------------------- #

class EmbeddingChunk(models.Model):
    """
    A retrievable chunk. Filter columns are DENORMALIZED so we can filter before
    vector search. source_object_type/id is a lightweight generic link back to the
    originating row (question/correction/etc.) without a hard FK.
    """
    CONTENT_TYPES = [
        ("lesson", "Résumé de cours"), ("exercise", "Énoncé d'exercice"),
        ("question", "Question"), ("correction", "Correction"),
        ("rubric", "Barème"), ("mistake", "Erreur fréquente"),
        ("combined", "Contexte combiné de l'exercice"),
    ]

    content = models.TextField()
    content_type = models.CharField(max_length=16, choices=CONTENT_TYPES)
    # Nullable: chunks are created with text+metadata first, embedded later. This is
    # the production-correct flow (decouple ingestion from the embedding job).
    embedding = VectorField(dimensions=1536, null=True, blank=True)
    embedding_status = models.CharField(max_length=12, choices=EmbeddingStatus.choices,
                                        default=EmbeddingStatus.PENDING, db_index=True)
    model_name = models.CharField(max_length=64, blank=True)    # which embedding model produced this
    language = models.CharField(max_length=8, choices=Language.choices, blank=True, default="")

    # Denormalized filter columns:
    subject = models.ForeignKey(Subject, on_delete=models.CASCADE, null=True, related_name="chunks")
    section = models.ForeignKey(BacSection, on_delete=models.CASCADE, null=True, related_name="chunks")
    chapter = models.ForeignKey(Chapter, on_delete=models.SET_NULL, null=True, related_name="chunks")
    era = models.ForeignKey(CurriculumEra, on_delete=models.SET_NULL, null=True, related_name="chunks")
    year = models.PositiveSmallIntegerField(null=True, blank=True)
    relevance_weight = models.DecimalField(max_digits=3, decimal_places=2, default=1.0)
    relevance_status = models.CharField(max_length=12, choices=RelevanceStatus.choices,
                                        default=RelevanceStatus.UNREVIEWED)

    source_object_type = models.CharField(max_length=32, blank=True)   # "ExamQuestion"
    source_object_id = models.PositiveIntegerField(null=True, blank=True)

    # Postgres full-text vector for the keyword half of hybrid search (populated on save/ingest).
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        # The pgvector HNSW index on `embedding` is NOT tracked in model state:
        # it is created in migration 0001 as a Postgres-only database operation.
        # Keeping it out of state lets non-Postgres backends (SQLite, used for
        # local checks) alter this table without trying to recreate an index whose
        # `USING hnsw ... WITH (...)` SQL they cannot parse.
        indexes = [
            models.Index(fields=["content_type"]),
            models.Index(fields=["subject", "section"]),
            models.Index(fields=["year"]),
            models.Index(fields=["relevance_status"]),
        ]

    def __str__(self):
        return f"[{self.content_type}] {self.content[:60]}"


# --------------------------------------------------------------------------- #
# Student state & interactions
# --------------------------------------------------------------------------- #

class StudentAttempt(models.Model):
    """A student's attempt at a question/exercise (or a freeform submission)."""
    INPUT_TYPES = [("mcq", "QCM"), ("text", "Réponse rédigée"), ("image", "Photo/PDF")]

    student = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
                                related_name="attempts")
    question = models.ForeignKey(ExamQuestion, on_delete=models.SET_NULL, null=True, blank=True,
                                 related_name="attempts")
    exercise = models.ForeignKey(ExamExercise, on_delete=models.SET_NULL, null=True, blank=True,
                                 related_name="attempts")
    input_type = models.CharField(max_length=8, choices=INPUT_TYPES)
    raw_answer = models.TextField(blank=True)

    estimated_score = models.DecimalField(max_digits=5, decimal_places=2, null=True, blank=True)
    max_score = models.DecimalField(max_digits=5, decimal_places=2, null=True, blank=True)
    is_correct = models.BooleanField(null=True, blank=True)
    mistakes = models.JSONField(default=list, blank=True)      # structured mistake list
    feedback = models.JSONField(default=dict, blank=True)

    is_diagnostic = models.BooleanField(default=False)
    is_mock = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [models.Index(fields=["student", "created_at"])]

    def __str__(self):
        return f"Attempt by {self.student} ({self.input_type})"


class AIInteraction(models.Model):
    """Audit + eval log of every tutor/correction call. Never skip writing this."""
    MODES = [
        ("explain_fr", "Explication FR"), ("explain_ar", "Explication AR"),
        ("explain_darja", "Explication Darja"), ("hint_only", "Indice"),
        ("full_correction", "Correction complète"), ("bac_style_answer", "Réponse type Bac"),
        ("identify_mistakes", "Repérage d'erreurs"), ("generate_similar", "Exercice similaire"),
        ("revision_summary", "Fiche de révision"), ("explain_points_lost", "Points perdus"),
        ("tutor_answer", "Réponse tuteur"),
    ]

    student = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
                                null=True, blank=True, related_name="ai_interactions")
    mode = models.CharField(max_length=24, choices=MODES)
    language = models.CharField(max_length=8, choices=Language.choices, default=Language.FR)
    query = models.TextField()
    retrieved_chunk_ids = models.JSONField(default=list, blank=True)
    response = models.TextField()
    citations = models.JSONField(default=list, blank=True)
    used_sources = models.BooleanField(default=False)
    provider = models.CharField(max_length=32, blank=True)
    model_name = models.CharField(max_length=64, blank=True)
    retrieval_mode = models.CharField(max_length=48, blank=True)
    refused = models.BooleanField(default=False)
    refusal_reason = models.TextField(blank=True)
    warnings = models.JSONField(default=list, blank=True)
    tokens = models.PositiveIntegerField(null=True, blank=True)
    confidence = models.DecimalField(max_digits=4, decimal_places=3, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [models.Index(fields=["mode", "created_at"])]

    def __str__(self):
        return f"{self.mode} @ {self.created_at:%Y-%m-%d}"


class KnowledgeState(models.Model):
    """Per-student, per-concept mastery. Powers readiness + recommendations."""
    student = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
                                related_name="knowledge_states")
    concept = models.ForeignKey(Concept, on_delete=models.CASCADE, related_name="knowledge_states")
    mastery = models.DecimalField(max_digits=4, decimal_places=3, default=0)   # 0..1
    attempts = models.PositiveIntegerField(default=0)
    correct = models.PositiveIntegerField(default=0)
    last_practiced = models.DateTimeField(null=True, blank=True)
    decay_applied_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["student", "concept"], name="uniq_student_concept"),
        ]

    def __str__(self):
        return f"{self.student} · {self.concept.code} = {self.mastery}"


class MistakeNotebookEntry(models.Model):
    """An entry in a student's mistake notebook, optionally linked to a known CommonMistake."""
    student = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
                                related_name="mistake_notebook")
    common_mistake = models.ForeignKey(CommonMistake, on_delete=models.SET_NULL,
                                       null=True, blank=True, related_name="notebook_entries")
    concept = models.ForeignKey(Concept, on_delete=models.SET_NULL, null=True, blank=True,
                                related_name="notebook_entries")
    attempt = models.ForeignKey(StudentAttempt, on_delete=models.SET_NULL, null=True, blank=True,
                                related_name="notebook_entries")
    note = models.TextField(blank=True)
    resolved = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [models.Index(fields=["student", "resolved"])]

    def __str__(self):
        return f"Notebook[{self.student}] {self.concept or self.common_mistake}"

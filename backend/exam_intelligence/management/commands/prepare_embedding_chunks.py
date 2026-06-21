"""
Create EmbeddingChunk records from loaded exercises/questions/corrections so the
future RAG system has data to embed.

NO paid APIs are called. By default chunks are created with status 'pending' and
NO vector (text + metadata only) — the production-correct flow: decouple chunk
creation from the embedding job.

Optional `--mock` fills a DETERMINISTIC local placeholder vector (status 'mock').
These are NOT real embeddings and must not be treated as semantically meaningful;
they exist only so downstream code paths can be exercised without a paid API.

Chunk types produced:
  exercise   (exercise_statement)
  question   (question_text)
  correction (correction_text)
  rubric     (rubric_text)
  mistake    (common_mistake_text)
  combined   (combined_exercise_context)

Idempotent: each chunk is upserted by (source_object_type, source_object_id,
content_type). Running twice does not duplicate.

Usage:
    python manage.py prepare_embedding_chunks
    python manage.py prepare_embedding_chunks --mock
"""

import hashlib
import struct

from django.core.management.base import BaseCommand
from django.db import transaction

from backend.exam_intelligence.models import (
    Correction, EmbeddingChunk, EmbeddingStatus, ExamExercise,
)

MOCK_MODEL_NAME = "mock-deterministic-v1"
EMBEDDING_DIM = 1536
# Assumption: all current seed content is French. Document in README. Override per
# exercise later when language detection / multilingual content exists.
DEFAULT_LANGUAGE = "fr"


def mock_embedding(text: str, dim: int = EMBEDDING_DIM) -> list[float]:
    """Deterministic pseudo-vector from text. Reproducible, NOT a real embedding."""
    values: list[float] = []
    counter = 0
    while len(values) < dim:
        digest = hashlib.sha256(f"{counter}:{text}".encode("utf-8")).digest()
        for i in range(0, len(digest), 4):
            values.append(struct.unpack("I", digest[i:i + 4])[0] / 2**32 - 0.5)
            if len(values) >= dim:
                break
        counter += 1
    return values


class Command(BaseCommand):
    help = "Create EmbeddingChunk rows (pending by default) from loaded exam data."

    def add_arguments(self, parser):
        parser.add_argument("--mock", action="store_true",
                            help="Fill deterministic placeholder vectors (status 'mock').")

    def handle(self, *args, **options):
        use_mock = options["mock"]
        self.created = self.updated = self.skipped = 0
        self.warnings: list[str] = []

        exercises = (ExamExercise.objects
                     .select_related("exam", "exam__subject", "exam__section", "exam__era")
                     .prefetch_related("questions", "chapters", "common_mistakes")
                     .all())
        if not exercises:
            self.stdout.write(self.style.WARNING(
                "No exercises found — run load_example_exercises first. Nothing to do."))
            return

        for exercise in exercises:
            self._process_exercise(exercise, use_mock)

        self.stdout.write(self.style.SUCCESS(
            f"Done. created={self.created} updated={self.updated} "
            f"skipped={self.skipped} warnings={len(self.warnings)} "
            f"mode={'mock' if use_mock else 'pending'}"
        ))
        for w in self.warnings:
            self.stdout.write(self.style.WARNING(f"  ! {w}"))

    # ------------------------------------------------------------------ #

    @transaction.atomic
    def _process_exercise(self, exercise: ExamExercise, use_mock: bool):
        exam = exercise.exam
        chapter = exercise.chapters.first()
        meta = {
            "subject": exam.subject,
            "section": exam.section,
            "chapter": chapter,
            "era": exam.era,
            "year": exam.year,
            "relevance_weight": exercise.effective_relevance_weight,
            "relevance_status": exercise.relevance_status,
            "language": DEFAULT_LANGUAGE,
        }

        statement = (exercise.intro_text or exercise.title or "").strip()
        if statement:
            self._upsert("exercise", "ExamExercise", exercise.id, statement, meta, use_mock)

        combined_parts = [statement] if statement else []

        for question in exercise.questions.all():
            qtext = (question.text or "").strip()
            if qtext:
                self._upsert("question", "ExamQuestion", question.id, qtext, meta, use_mock)
                combined_parts.append(f"Q{question.number}: {qtext}")

        # corrections attached either to a question of this exercise or to the exercise
        corrections = list(Correction.objects.filter(question__exercise=exercise)) + \
            list(Correction.objects.filter(exercise=exercise))
        for correction in corrections:
            ctext = (correction.text or "").strip()
            if ctext:
                self._upsert("correction", "Correction", correction.id, ctext, meta, use_mock)
            for item in correction.rubric_items.all():
                rtext = item.description
                if item.keywords:
                    rtext += " | mots-clés: " + ", ".join(item.keywords)
                self._upsert("rubric", "RubricItem", item.id, rtext, meta, use_mock)

        for mistake in exercise.common_mistakes.all():
            mtext = mistake.description_fr
            if mistake.correct_approach:
                mtext += " | bonne approche: " + mistake.correct_approach
            self._upsert("mistake", "CommonMistake", mistake.id, mtext, meta, use_mock)

        if combined_parts:
            combined = "\n".join(combined_parts)
            self._upsert("combined", "ExamExercise", exercise.id, combined, meta, use_mock)

    def _upsert(self, content_type: str, src_type: str, src_id: int,
                content: str, meta: dict, use_mock: bool):
        defaults = dict(meta)
        defaults["content"] = content
        if use_mock:
            # A chunk with a usable vector is READY for vector search. The vectors
            # are deterministic placeholders (not semantic) — that fact is recorded
            # transparently in model_name, NOT hidden behind the status.
            defaults["embedding"] = mock_embedding(content)
            defaults["embedding_status"] = EmbeddingStatus.READY
            defaults["model_name"] = MOCK_MODEL_NAME
        else:
            # Leave existing real/mock embeddings untouched on re-run; only (re)set
            # text+metadata. New rows default to pending with no vector.
            defaults["embedding_status"] = EmbeddingStatus.PENDING

        obj, created = EmbeddingChunk.objects.update_or_create(
            source_object_type=src_type, source_object_id=src_id, content_type=content_type,
            defaults=defaults,
        )
        if created:
            self.created += 1
        else:
            self.updated += 1

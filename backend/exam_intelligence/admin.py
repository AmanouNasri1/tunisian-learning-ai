"""
Django admin for the exam-intelligence layer.

Priority: make manual review practical (source-document status, OCR confidence,
curriculum relevance, validation) and make the curriculum/exam tree browsable.
Not exhaustive — useful.
"""

from django.contrib import admin

from .models import (
    AIInteraction, BacSection, Chapter, CommonMistake, Concept, Correction,
    CurriculumEra, Exam, ExamExercise, ExamQuestion, EmbeddingChunk,
    ExerciseTag, KnowledgeState, MistakeNotebookEntry, RubricItem,
    SectionSubject, SourceDocument, StudentAttempt, Subject,
)


# --- Reference / curriculum ------------------------------------------------- #

@admin.register(BacSection)
class BacSectionAdmin(admin.ModelAdmin):
    list_display = ("code", "name_fr", "name_ar")
    search_fields = ("code", "name_fr", "name_ar")
    ordering = ("code",)


@admin.register(Subject)
class SubjectAdmin(admin.ModelAdmin):
    list_display = ("code", "name_fr", "name_ar")
    search_fields = ("code", "name_fr", "name_ar")
    ordering = ("code",)


@admin.register(SectionSubject)
class SectionSubjectAdmin(admin.ModelAdmin):
    list_display = ("section", "subject", "coefficient")
    list_filter = ("section", "subject")
    search_fields = ("section__code", "subject__code")


@admin.register(CurriculumEra)
class CurriculumEraAdmin(admin.ModelAdmin):
    list_display = ("code", "label", "start_year", "end_year", "relevance_weight")
    search_fields = ("code", "label")
    ordering = ("-start_year",)


@admin.register(Chapter)
class ChapterAdmin(admin.ModelAdmin):
    list_display = ("code", "name_fr", "subject", "era", "order")
    list_filter = ("subject", "era")
    search_fields = ("code", "name_fr", "name_ar")
    ordering = ("subject", "order")


@admin.register(Concept)
class ConceptAdmin(admin.ModelAdmin):
    list_display = ("code", "name_fr", "chapter", "parent")
    list_filter = ("chapter__subject", "chapter")
    search_fields = ("code", "name_fr")
    autocomplete_fields = ("chapter", "parent")


# --- Source documents (review workflow) ------------------------------------- #

@admin.register(SourceDocument)
class SourceDocumentAdmin(admin.ModelAdmin):
    list_display = ("original_filename", "doc_type", "detected_year", "detected_subject",
                    "detected_section", "is_scanned", "confidence_score", "review_status",
                    "created_at")
    list_filter = ("review_status", "doc_type", "is_scanned", "detected_subject",
                   "detected_section", "detected_year")
    search_fields = ("original_filename", "file", "raw_text")
    readonly_fields = ("created_at", "raw_text")
    list_editable = ("review_status",)
    ordering = ("review_status", "-created_at")
    date_hierarchy = "created_at"


# --- Exams / exercises / questions ------------------------------------------ #

class ExamQuestionInline(admin.TabularInline):
    model = ExamQuestion
    extra = 0
    fields = ("number", "text", "points", "order", "parent")
    autocomplete_fields = ("parent",)


class ExamExerciseInline(admin.TabularInline):
    model = ExamExercise
    extra = 0
    fields = ("number", "title", "total_points", "difficulty",
              "relevance_status", "validated_by_teacher")
    show_change_link = True


@admin.register(Exam)
class ExamAdmin(admin.ModelAdmin):
    list_display = ("__str__", "section", "subject", "year", "session", "era")
    list_filter = ("section", "subject", "session", "era", "year")
    search_fields = ("section__code", "subject__code")
    autocomplete_fields = ("source_document",)
    inlines = (ExamExerciseInline,)
    ordering = ("-year", "section", "subject")


@admin.register(ExamExercise)
class ExamExerciseAdmin(admin.ModelAdmin):
    list_display = ("__str__", "exam", "number", "difficulty", "estimated_minutes",
                    "relevance_status", "validated_by_teacher")
    list_filter = ("difficulty", "relevance_status", "validated_by_teacher",
                   "exam__section", "exam__subject", "exam__year")
    search_fields = ("title", "intro_text", "exam__section__code", "exam__subject__code")
    list_editable = ("relevance_status", "validated_by_teacher")
    filter_horizontal = ("chapters", "concepts")
    inlines = (ExamQuestionInline,)
    ordering = ("-exam__year", "number")


@admin.register(ExamQuestion)
class ExamQuestionAdmin(admin.ModelAdmin):
    list_display = ("__str__", "exercise", "number", "points", "order")
    list_filter = ("exercise__exam__subject", "exercise__exam__year")
    search_fields = ("number", "text")
    autocomplete_fields = ("exercise", "parent")
    filter_horizontal = ("concepts",)


# --- Corrections / rubric / mistakes ---------------------------------------- #

class RubricItemInline(admin.TabularInline):
    model = RubricItem
    extra = 0
    fields = ("order", "description", "points", "keywords")


@admin.register(Correction)
class CorrectionAdmin(admin.ModelAdmin):
    list_display = ("__str__", "author_type", "is_official", "reliability")
    list_filter = ("author_type", "is_official", "reliability")
    search_fields = ("text",)
    autocomplete_fields = ("question", "exercise", "source_document")
    inlines = (RubricItemInline,)


@admin.register(RubricItem)
class RubricItemAdmin(admin.ModelAdmin):
    list_display = ("description", "points", "order", "correction")
    search_fields = ("description",)


@admin.register(CommonMistake)
class CommonMistakeAdmin(admin.ModelAdmin):
    list_display = ("description_fr", "frequency", "concept", "exercise")
    list_filter = ("frequency",)
    search_fields = ("description_fr", "description_ar", "why_it_happens")
    autocomplete_fields = ("concept", "exercise")


@admin.register(ExerciseTag)
class ExerciseTagAdmin(admin.ModelAdmin):
    list_display = ("exercise", "tag_type", "value", "source")
    list_filter = ("tag_type", "source")
    search_fields = ("value",)
    autocomplete_fields = ("exercise",)


# --- Embeddings ------------------------------------------------------------- #

@admin.register(EmbeddingChunk)
class EmbeddingChunkAdmin(admin.ModelAdmin):
    list_display = ("id", "content_type", "subject", "section", "year",
                    "relevance_status", "model_name", "created_at")
    list_filter = ("content_type", "subject", "section", "relevance_status", "model_name")
    search_fields = ("content", "source_object_type")
    readonly_fields = ("created_at",)
    # `embedding` is a high-dimensional vector — never render it in a form.
    exclude = ("embedding",)


# --- Student state ---------------------------------------------------------- #

@admin.register(StudentAttempt)
class StudentAttemptAdmin(admin.ModelAdmin):
    list_display = ("student", "input_type", "estimated_score", "max_score",
                    "is_correct", "is_diagnostic", "is_mock", "created_at")
    list_filter = ("input_type", "is_correct", "is_diagnostic", "is_mock")
    search_fields = ("student__username", "raw_answer")
    readonly_fields = ("created_at",)
    date_hierarchy = "created_at"


@admin.register(AIInteraction)
class AIInteractionAdmin(admin.ModelAdmin):
    list_display = ("mode", "language", "student", "provider", "used_sources",
                    "refused", "model_name", "confidence", "created_at")
    list_filter = ("mode", "language", "provider", "used_sources", "refused", "model_name")
    search_fields = ("query", "response")
    readonly_fields = ("created_at",)
    date_hierarchy = "created_at"


@admin.register(KnowledgeState)
class KnowledgeStateAdmin(admin.ModelAdmin):
    list_display = ("student", "concept", "mastery", "attempts", "correct", "last_practiced")
    list_filter = ("concept__chapter__subject",)
    search_fields = ("student__username", "concept__code")
    autocomplete_fields = ("concept",)


@admin.register(MistakeNotebookEntry)
class MistakeNotebookEntryAdmin(admin.ModelAdmin):
    list_display = ("student", "concept", "common_mistake", "resolved", "created_at")
    list_filter = ("resolved",)
    search_fields = ("student__username", "note")
    autocomplete_fields = ("concept", "common_mistake", "attempt")

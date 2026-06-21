from django.apps import AppConfig


class ExamIntelligenceConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "backend.exam_intelligence"
    # Explicit label so fixtures/migrations use "exam_intelligence" regardless of package path.
    label = "exam_intelligence"
    verbose_name = "Exam Intelligence"

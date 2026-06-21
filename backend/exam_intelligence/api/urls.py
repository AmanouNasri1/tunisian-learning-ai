"""Read-only API routes under /api/."""

from rest_framework.routers import DefaultRouter

from .views import (
    BacSectionViewSet, ChapterViewSet, ConceptViewSet, ExamViewSet,
    ExamExerciseViewSet, SubjectViewSet,
)

router = DefaultRouter()
router.register(r"sections", BacSectionViewSet, basename="section")
router.register(r"subjects", SubjectViewSet, basename="subject")
router.register(r"chapters", ChapterViewSet, basename="chapter")
router.register(r"concepts", ConceptViewSet, basename="concept")
router.register(r"exams", ExamViewSet, basename="exam")
router.register(r"exercises", ExamExerciseViewSet, basename="exercise")

urlpatterns = router.urls

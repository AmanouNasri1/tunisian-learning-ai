"""
Read-only API viewsets for reference + exam browsing.

Filtering uses DRF's built-in SearchFilter/OrderingFilter (no extra dependency)
plus a few explicit query-param filters where they're obviously useful. All
endpoints are read-only; no internal/raw fields are exposed.
"""

from rest_framework import filters, status, viewsets
from rest_framework.response import Response
from rest_framework.views import APIView

from ai.llm_client import LLMConfigurationError
from backend.exam_intelligence.models import (
    BacSection, Chapter, Concept, Exam, ExamExercise, Subject,
)
from rag.tutor import answer_student_question
from rag.context_builder import RAGContextBuilder
from .serializers import (
    BacSectionSerializer, ChapterSerializer, ConceptSerializer, ExamSerializer,
    ExamExerciseSerializer, SubjectSerializer,
)


class BacSectionViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = BacSection.objects.all().order_by("code")
    serializer_class = BacSectionSerializer
    filter_backends = [filters.SearchFilter, filters.OrderingFilter]
    search_fields = ["code", "name_fr", "name_ar"]
    ordering_fields = ["code", "name_fr"]


class SubjectViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Subject.objects.all().order_by("code")
    serializer_class = SubjectSerializer
    filter_backends = [filters.SearchFilter, filters.OrderingFilter]
    search_fields = ["code", "name_fr", "name_ar"]
    ordering_fields = ["code", "name_fr"]


class ChapterViewSet(viewsets.ReadOnlyModelViewSet):
    serializer_class = ChapterSerializer
    filter_backends = [filters.SearchFilter, filters.OrderingFilter]
    search_fields = ["code", "name_fr", "name_ar"]
    ordering_fields = ["order", "code"]

    def get_queryset(self):
        qs = Chapter.objects.select_related("subject", "era").all()
        subject = self.request.query_params.get("subject")
        if subject:
            qs = qs.filter(subject__code=subject) if not subject.isdigit() \
                else qs.filter(subject_id=subject)
        return qs.order_by("subject", "order")


class ConceptViewSet(viewsets.ReadOnlyModelViewSet):
    serializer_class = ConceptSerializer
    filter_backends = [filters.SearchFilter, filters.OrderingFilter]
    search_fields = ["code", "name_fr"]
    ordering_fields = ["code"]

    def get_queryset(self):
        qs = Concept.objects.select_related("chapter").all()
        chapter = self.request.query_params.get("chapter")
        if chapter:
            qs = qs.filter(chapter__code=chapter) if not chapter.isdigit() \
                else qs.filter(chapter_id=chapter)
        subject = self.request.query_params.get("subject")
        if subject:
            qs = qs.filter(chapter__subject__code=subject)
        return qs.order_by("code")


class ExamViewSet(viewsets.ReadOnlyModelViewSet):
    serializer_class = ExamSerializer
    filter_backends = [filters.OrderingFilter]
    ordering_fields = ["year", "session"]

    def get_queryset(self):
        qs = Exam.objects.select_related("section", "subject", "era").all()
        params = self.request.query_params
        if params.get("subject"):
            qs = qs.filter(subject__code=params["subject"])
        if params.get("section"):
            qs = qs.filter(section__code=params["section"])
        if params.get("year"):
            qs = qs.filter(year=params["year"])
        if params.get("session"):
            qs = qs.filter(session=params["session"])
        return qs.order_by("-year", "section", "subject")


class ExamExerciseViewSet(viewsets.ReadOnlyModelViewSet):
    serializer_class = ExamExerciseSerializer
    filter_backends = [filters.SearchFilter, filters.OrderingFilter]
    search_fields = ["title", "intro_text"]
    ordering_fields = ["number", "difficulty"]

    def get_queryset(self):
        qs = ExamExercise.objects.select_related(
            "exam", "exam__subject", "exam__section").all()
        params = self.request.query_params
        if params.get("subject"):
            qs = qs.filter(exam__subject__code=params["subject"])
        if params.get("section"):
            qs = qs.filter(exam__section__code=params["section"])
        if params.get("year"):
            qs = qs.filter(exam__year=params["year"])
        if params.get("difficulty"):
            qs = qs.filter(difficulty=params["difficulty"])
        if params.get("relevance_status"):
            qs = qs.filter(relevance_status=params["relevance_status"])
        return qs.order_by("-exam__year", "number")


class RAGContextView(APIView):
    """Return structured retrieval context only. No LLM calls."""

    def get(self, request):
        query = (request.query_params.get("q") or "").strip()
        if not query:
            return Response({"detail": "Missing required query parameter: q"},
                            status=status.HTTP_400_BAD_REQUEST)

        top_k = request.query_params.get("top_k")
        package = RAGContextBuilder().build(
            query=query,
            section=request.query_params.get("section"),
            subject=request.query_params.get("subject"),
            chapter=request.query_params.get("chapter"),
            top_k=top_k,
        )
        return Response(package)


class TutorAskView(APIView):
    """Return a source-grounded tutor answer. Mock provider is the default."""

    def post(self, request):
        data = request.data if isinstance(request.data, dict) else {}
        query = (data.get("query") or "").strip()
        if not query:
            return Response({"detail": "Missing required field: query"},
                            status=status.HTTP_400_BAD_REQUEST)

        try:
            package = answer_student_question(
                query=query,
                section=data.get("section"),
                subject=data.get("subject"),
                chapter=data.get("chapter"),
                provider=data.get("provider") or "mock",
                top_k=data.get("top_k") or 6,
            )
        except LLMConfigurationError as exc:
            return Response({"detail": str(exc), "provider": data.get("provider")},
                            status=status.HTTP_400_BAD_REQUEST)
        except Exception as exc:
            return Response({"detail": f"Tutor request failed: {type(exc).__name__}: {exc}"},
                            status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response(package)

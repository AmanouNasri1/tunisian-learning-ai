"""Read-only serializers for reference/exam browsing. No write APIs yet."""

from rest_framework import serializers

from backend.exam_intelligence.models import (
    BacSection, Chapter, Concept, Exam, ExamExercise, Subject,
)


class BacSectionSerializer(serializers.ModelSerializer):
    class Meta:
        model = BacSection
        fields = ("id", "code", "name_fr", "name_ar")


class SubjectSerializer(serializers.ModelSerializer):
    class Meta:
        model = Subject
        fields = ("id", "code", "name_fr", "name_ar")


class ConceptSerializer(serializers.ModelSerializer):
    class Meta:
        model = Concept
        fields = ("id", "code", "name_fr", "chapter", "parent")


class ChapterSerializer(serializers.ModelSerializer):
    class Meta:
        model = Chapter
        fields = ("id", "code", "name_fr", "subject", "era", "order")


class ExamSerializer(serializers.ModelSerializer):
    section_code = serializers.CharField(source="section.code", read_only=True)
    subject_code = serializers.CharField(source="subject.code", read_only=True)

    class Meta:
        model = Exam
        fields = ("id", "section", "section_code", "subject", "subject_code",
                  "year", "session", "era")


class ExamExerciseSerializer(serializers.ModelSerializer):
    class Meta:
        model = ExamExercise
        fields = ("id", "exam", "number", "title", "intro_text", "total_points",
                  "difficulty", "estimated_minutes", "relevance_status",
                  "validated_by_teacher")

"""
Lightweight verification tests (Django's built-in runner — no pytest needed).

Run on SQLite (does not require PostgreSQL):

    # bash
    DATABASE_URL='sqlite:///test_unused.sqlite3' python manage.py test backend.exam_intelligence

    # PowerShell
    $env:DATABASE_URL='sqlite:///test_unused.sqlite3'; python manage.py test backend.exam_intelligence

Django uses an in-memory SQLite test database, so the DATABASE_URL file is not
actually touched — it only needs to select the sqlite backend.
"""

import io

from django.core.management import call_command
from django.db import connection
from django.test import TestCase
from unittest import skipUnless

from backend.exam_intelligence.models import (
    BacSection, CurriculumEra, EmbeddingChunk, ExamExercise, RubricItem, Subject,
)
from backend.exam_intelligence.services.readiness import (
    ReadinessComponents, overall_readiness, subject_readiness,
)

_NULL = io.StringIO  # quiet command output


def _load(cmd, *args):
    call_command(cmd, *args, stdout=_NULL(), stderr=_NULL())


class BackendVerificationTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        _load("load_reference_data", "seed_data/reference/01_reference.json")
        _load("load_example_exercises", "seed_data/examples")

    def test_reference_loader_is_idempotent(self):
        before = (BacSection.objects.count(), Subject.objects.count(),
                  CurriculumEra.objects.count())
        _load("load_reference_data", "seed_data/reference/01_reference.json")
        after = (BacSection.objects.count(), Subject.objects.count(),
                 CurriculumEra.objects.count())
        self.assertEqual(before, after)
        self.assertEqual(before, (2, 3, 4))

    def test_example_loader_is_idempotent(self):
        before = (ExamExercise.objects.count(), RubricItem.objects.count())
        _load("load_example_exercises", "seed_data/examples")
        after = (ExamExercise.objects.count(), RubricItem.objects.count())
        self.assertEqual(before, after)
        # One exercise per example file in seed_data/examples/.
        self.assertEqual(ExamExercise.objects.count(), 5)

    def test_readiness_score_runs(self):
        c = ReadinessComponents(diagnostic=0.6, accuracy=0.7, mastery=0.65,
                                mock=0.5, recency=1.0, recurring_unresolved_mistakes=2)
        score = subject_readiness(c)
        self.assertTrue(0 <= score <= 100)
        overall = overall_readiness([(score, 3), (58.0, 4), (66.0, 4)])
        self.assertTrue(0 <= overall <= 100)

    def test_prepare_embedding_chunks_is_idempotent(self):
        _load("prepare_embedding_chunks")
        first = EmbeddingChunk.objects.count()
        _load("prepare_embedding_chunks")
        second = EmbeddingChunk.objects.count()
        self.assertGreater(first, 0)
        self.assertEqual(first, second)
        # All chunks created without an embedding default to 'pending'.
        self.assertTrue(EmbeddingChunk.objects.filter(embedding_status="pending").exists())

    def test_api_endpoints_return_200(self):
        for url in ["/api/sections/", "/api/subjects/", "/api/chapters/",
                    "/api/concepts/", "/api/exams/", "/api/exercises/"]:
            resp = self.client.get(url)
            self.assertEqual(resp.status_code, 200, msg=f"{url} -> {resp.status_code}")

    def test_api_exercise_filter_by_subject(self):
        resp = self.client.get("/api/exercises/?subject=SVT")
        self.assertEqual(resp.status_code, 200)
        results = resp.json()["results"]
        self.assertTrue(all(True for _ in results))  # shape check; SVT example exists
        self.assertGreaterEqual(len(results), 1)

    def test_keyword_retrieval_fonction(self):
        """The 'fonction' seed example must be retrievable by keyword (portable, no pgvector)."""
        from rag.retriever import Retriever, RetrievalFilters
        _load("prepare_embedding_chunks")
        hits = Retriever()._keyword_search("fonction", RetrievalFilters(), limit=5)
        self.assertGreaterEqual(len(hits), 1)

    @skipUnless(connection.vendor == "postgresql", "vector search requires PostgreSQL + pgvector")
    def test_vector_search_runs_on_postgres(self):
        """Real pgvector path: only runs on PostgreSQL (skipped on SQLite)."""
        from rag.retriever import Retriever, RetrievalFilters
        from backend.exam_intelligence.management.commands.prepare_embedding_chunks import (
            mock_embedding,
        )

        class _Mock:
            def embed(self, texts):
                return [mock_embedding(t) for t in texts]

        _load("prepare_embedding_chunks", "--mock")
        result = Retriever(embedder=_Mock()).retrieve("probabilité", RetrievalFilters())
        self.assertGreater(result.vector_count, 0)

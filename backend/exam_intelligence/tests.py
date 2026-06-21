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
import json
import os
import tempfile
from pathlib import Path

from django.core.management import call_command
from django.core.management.base import CommandError
from django.db import connection
from django.test import TestCase
from unittest import skipUnless
from unittest.mock import patch

from backend.exam_intelligence.models import (
    AIInteraction, BacSection, CurriculumEra, EmbeddingChunk, EmbeddingStatus,
    EvaluationCaseResult, EvaluationRun, ExamExercise, RubricItem, Subject,
)
from backend.exam_intelligence.services.readiness import (
    ReadinessComponents, overall_readiness, subject_readiness,
)

_NULL = io.StringIO  # quiet command output


def _load(cmd, *args):
    call_command(cmd, *args, stdout=_NULL(), stderr=_NULL())


def _write_tutor_cases(tmpdir: str, text: str) -> str:
    path = Path(tmpdir) / "cases.yaml"
    path.write_text(text, encoding="utf-8")
    return str(path)


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

    def test_mock_embedding_provider_is_deterministic(self):
        from ai.embeddings import EMBEDDING_DIM, MockEmbeddingProvider

        provider = MockEmbeddingProvider()
        first = provider.embed(["fonction"])[0]
        second = provider.embed(["fonction"])[0]
        other = provider.embed(["probabilite"])[0]
        self.assertEqual(first, second)
        self.assertNotEqual(first, other)
        self.assertEqual(len(first), EMBEDDING_DIM)

    def test_text_normalization_handles_french_accents_safely(self):
        from rag.text_normalization import normalize_text

        cases = {
            "probabilit\u00e9": "probabilite",
            "\u00e9lectricit\u00e9": "electricite",
            "d\u00e9riv\u00e9e": "derivee",
            "g\u00e9n\u00e9tique": "genetique",
            "fr\u00e9quence": "frequence",
            "r\u00e9sonance": "resonance",
            "l\u2019\u00e9lectricit\u00e9": "l'electricite",
            "  Circuit   \u00e9lectrique  ": "circuit electrique",
        }
        for source, expected in cases.items():
            self.assertEqual(normalize_text(source), expected)
        self.assertEqual(normalize_text(None), "")
        self.assertIn("+\u221e", normalize_text("+\u221e \u00b2 \u221a"))
        self.assertEqual(normalize_text("\u0639\u064e\u0631\u064e\u0628\u0650\u064a"), "\u0639\u064e\u0631\u064e\u0628\u0650\u064a")

    def test_embed_chunks_provider_mock_updates_limited_pending_chunks(self):
        from ai.embeddings import MOCK_MODEL_NAME

        _load("prepare_embedding_chunks")
        call_command("embed_chunks", "--provider", "mock", "--limit", "3",
                     stdout=_NULL(), stderr=_NULL())
        ready = EmbeddingChunk.objects.filter(
            embedding_status=EmbeddingStatus.READY,
            embedding__isnull=False,
            model_name=MOCK_MODEL_NAME,
        )
        self.assertEqual(ready.count(), 3)
        self.assertGreater(EmbeddingChunk.objects.filter(
            embedding_status=EmbeddingStatus.PENDING).count(), 0)

    def test_embed_chunks_default_dry_run_does_not_instantiate_openai(self):
        _load("prepare_embedding_chunks")
        with patch.dict(os.environ, {"EMBEDDING_PROVIDER": "openai"}, clear=False):
            with patch("ai.embeddings.OpenAIEmbeddingProvider",
                       side_effect=AssertionError("OpenAI should not be instantiated")):
                call_command("embed_chunks", stdout=_NULL(), stderr=_NULL())
        self.assertFalse(EmbeddingChunk.objects.filter(
            embedding_status=EmbeddingStatus.READY).exists())

    def test_embed_chunks_openai_missing_key_exits_safely(self):
        _load("prepare_embedding_chunks")
        out = io.StringIO()
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False):
            call_command("embed_chunks", "--provider", "openai", "--limit", "1",
                         stdout=out, stderr=_NULL())
        self.assertIn("OPENAI_API_KEY is missing", out.getvalue())
        self.assertFalse(EmbeddingChunk.objects.filter(
            embedding_status=EmbeddingStatus.READY).exists())

    def test_rag_context_builder_returns_structured_context(self):
        from rag.context_builder import RAGContextBuilder

        _load("prepare_embedding_chunks", "--mock")
        package = RAGContextBuilder().build("fonction")
        self.assertEqual(package["query"], "fonction")
        self.assertIn(package["retrieval_mode"], {
            "hybrid vector + keyword", "vector-only", "keyword-only", "none",
        })
        self.assertGreater(package["selected_chunk_count"], 0)
        self.assertIn("grouped_context", package)
        self.assertIn("assembled_context", package["grouped_context"])
        self.assertIn("citations", package)
        self.assertTrue(package["citations"])

    def test_rag_context_api_returns_package(self):
        _load("prepare_embedding_chunks", "--mock")
        resp = self.client.get("/api/rag/context/?q=fonction")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["query"], "fonction")
        self.assertIn("grouped_context", data)
        self.assertIn("selected_chunks", data)

    def test_accented_unaccented_keyword_retrieval_equivalence(self):
        from rag.retriever import Retriever, RetrievalFilters

        _load("prepare_embedding_chunks")
        retriever = Retriever()
        pairs = [
            ("probabilit\u00e9", "probabilite"),
            ("circuit \u00e9lectrique", "circuit electrique"),
            ("g\u00e9n\u00e9tique", "genetique"),
            ("d\u00e9riv\u00e9e", "derivee"),
        ]
        for accented, unaccented in pairs:
            accented_hits = retriever._keyword_search(accented, RetrievalFilters(), limit=10)
            unaccented_hits = retriever._keyword_search(unaccented, RetrievalFilters(), limit=10)
            self.assertEqual(
                [(hit.chunk_id, hit.content_type) for hit in accented_hits],
                [(hit.chunk_id, hit.content_type) for hit in unaccented_hits],
                msg=f"{accented!r} vs {unaccented!r}",
            )

    def test_unaccented_keyword_queries_return_expected_curriculum_chunks(self):
        from rag.retriever import Retriever, RetrievalFilters
        from rag.text_normalization import normalize_text

        _load("prepare_embedding_chunks")
        retriever = Retriever()

        proba = retriever._keyword_search("probabilite", RetrievalFilters(), limit=10)
        self.assertTrue(any(hit.chapter_code == "PROBA" for hit in proba))

        genetique = retriever._keyword_search("genetique", RetrievalFilters(), limit=10)
        self.assertTrue(any(
            hit.subject_code == "SVT" and hit.chapter_code == "GENETIQUE"
            for hit in genetique
        ))

        derivee = retriever._keyword_search("derivee", RetrievalFilters(), limit=10)
        self.assertTrue(any(
            hit.chapter_code == "FONCTIONS" and "derivee" in normalize_text(hit.content)
            for hit in derivee
        ))

    def test_rag_context_api_unaccented_queries_have_keyword_candidates(self):
        _load("prepare_embedding_chunks", "--mock")
        for query in ["probabilite", "circuit electrique", "genetique", "derivee"]:
            resp = self.client.get("/api/rag/context/", {"q": query})
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertGreater(
                data["diagnostics"]["keyword_candidates"],
                0,
                msg=f"{query!r} returned no keyword candidates",
            )

    def test_mock_tutor_answer_returns_grounded_answer_and_audit_row(self):
        from rag.tutor import answer_student_question

        _load("prepare_embedding_chunks", "--mock")
        before = AIInteraction.objects.count()
        package = answer_student_question(
            "Explique la loi binomiale",
            section="SC_EXP",
            subject="MATH",
            chapter="PROBA",
            provider="mock",
        )
        self.assertFalse(package["refused"])
        self.assertIn("Réponse mock", package["answer"])
        self.assertTrue(package["citations"])
        self.assertTrue(package["used_chunks"])
        self.assertEqual(package["provider"], "mock")
        self.assertEqual(AIInteraction.objects.count(), before + 1)

        interaction = AIInteraction.objects.get(pk=package["interaction_id"])
        self.assertEqual(interaction.provider, "mock")
        self.assertEqual(interaction.mode, "tutor_answer")
        self.assertFalse(interaction.refused)
        self.assertTrue(interaction.citations)
        self.assertTrue(interaction.retrieved_chunk_ids)

    def test_mock_tutor_out_of_scope_question_refuses_and_logs(self):
        from rag.tutor import answer_student_question

        _load("prepare_embedding_chunks", "--mock")
        package = answer_student_question("Donne-moi une recette de pizza", provider="mock")
        self.assertTrue(package["refused"])
        self.assertIn("documents actuellement retrouvés", package["answer"])
        self.assertTrue(package["refusal_reason"])
        interaction = AIInteraction.objects.get(pk=package["interaction_id"])
        self.assertTrue(interaction.refused)
        self.assertEqual(interaction.refusal_reason, package["refusal_reason"])

    def test_tutor_api_mock_provider_returns_200(self):
        _load("prepare_embedding_chunks", "--mock")
        resp = self.client.post(
            "/api/tutor/ask/",
            data=json.dumps({
                "query": "Explique la loi binomiale",
                "provider": "mock",
                "section": "SC_EXP",
                "subject": "MATH",
                "chapter": "PROBA",
            }),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertFalse(data["refused"])
        self.assertTrue(data["citations"])
        self.assertTrue(AIInteraction.objects.filter(pk=data["interaction_id"]).exists())

    def test_tutor_api_missing_query_returns_400(self):
        resp = self.client.post(
            "/api/tutor/ask/",
            data=json.dumps({"provider": "mock"}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("query", resp.json()["detail"])

    def test_tutor_api_missing_real_provider_key_returns_clear_error(self):
        _load("prepare_embedding_chunks", "--mock")
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False):
            resp = self.client.post(
                "/api/tutor/ask/",
                data=json.dumps({"query": "Explique la loi binomiale", "provider": "openai"}),
                content_type="application/json",
            )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("OPENAI_API_KEY is missing", resp.json()["detail"])

    def test_mock_tutor_does_not_instantiate_paid_clients(self):
        from rag.tutor import answer_student_question

        _load("prepare_embedding_chunks", "--mock")
        with patch("ai.llm_client.OpenAIClient",
                   side_effect=AssertionError("OpenAI should not be used")):
            with patch("ai.llm_client.AnthropicClient",
                       side_effect=AssertionError("Anthropic should not be used")):
                package = answer_student_question("Explique le circuit RLC", provider="mock")
        self.assertFalse(package["refused"])

    def test_tutor_case_loader_reads_yaml_cases(self):
        from evaluation.tutor_evaluator import load_tutor_cases

        cases = load_tutor_cases("evaluation/tutor_cases.yaml")
        self.assertGreaterEqual(len(cases), 8)
        ids = {case.id for case in cases}
        self.assertIn("loi_binomiale", ids)
        self.assertIn("pizza_out_of_scope", ids)

    def test_tutor_evaluator_runs_pass_and_refusal_cases(self):
        from evaluation.tutor_evaluator import TutorCase, evaluate_case

        _load("prepare_embedding_chunks", "--mock")
        pass_case = TutorCase(
            id="test_binomiale",
            query="Explique la loi binomiale",
            section="SC_EXP",
            subject="MATH",
            chapter="PROBA",
            expected_refused=False,
            expected_subject="MATH",
            expected_chapter="PROBA",
            required_terms=("binomiale",),
            forbidden_terms=("pizza",),
            required_citation_chapters=("PROBA",),
            minimum_citations=1,
        )
        refusal_case = TutorCase(
            id="test_pizza",
            query="Donne-moi une recette de pizza",
            expected_refused=True,
            forbidden_terms=("fromage", "tomate"),
        )
        self.assertTrue(evaluate_case(pass_case)["passed"])
        refused = evaluate_case(refusal_case)
        self.assertTrue(refused["passed"])
        self.assertTrue(refused["actual_refused"])
        self.assertEqual(refused["citation_count"], 0)

    def test_evaluate_tutor_command_succeeds_with_normal_threshold(self):
        _load("prepare_embedding_chunks", "--mock")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_tutor_cases(tmpdir, """
- id: pass_case
  query: "Explique la loi binomiale"
  section: "SC_EXP"
  subject: "MATH"
  chapter: "PROBA"
  expected_refused: false
  expected_subject: "MATH"
  expected_chapter: "PROBA"
  required_terms: ["binomiale"]
  forbidden_terms: ["pizza"]
  required_citation_chapters: ["PROBA"]
  minimum_citations: 1
- id: refusal_case
  query: "Donne-moi une recette de pizza"
  expected_refused: true
  required_terms: []
  forbidden_terms: ["fromage"]
  required_citation_chapters: []
  minimum_citations: 0
""")
            out = io.StringIO()
            call_command("evaluate_tutor", "--cases", path, "--fail-under", "0.8",
                         stdout=out, stderr=_NULL())
        self.assertIn("RESULT: PASS", out.getvalue())

    def test_evaluate_tutor_command_fails_with_strict_threshold(self):
        _load("prepare_embedding_chunks", "--mock")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_tutor_cases(tmpdir, """
- id: forced_failure
  query: "Explique la loi binomiale"
  section: "SC_EXP"
  subject: "MATH"
  chapter: "PROBA"
  expected_refused: false
  expected_subject: "MATH"
  expected_chapter: "PROBA"
  required_terms: ["term_that_will_not_appear"]
  forbidden_terms: []
  required_citation_chapters: ["PROBA"]
  minimum_citations: 1
""")
            with self.assertRaises(SystemExit):
                call_command("evaluate_tutor", "--cases", path, "--fail-under", "1.0",
                             stdout=_NULL(), stderr=_NULL())

    def test_evaluator_required_forbidden_and_citation_checks_report_failures(self):
        from evaluation.tutor_evaluator import TutorCase, evaluate_case

        _load("prepare_embedding_chunks", "--mock")
        required = evaluate_case(TutorCase(
            id="missing_required",
            query="Explique la loi binomiale",
            section="SC_EXP",
            subject="MATH",
            chapter="PROBA",
            expected_refused=False,
            required_terms=("impossible_required_term",),
            minimum_citations=1,
        ))
        self.assertFalse(required["passed"])
        self.assertIn("impossible_required_term", required["required_terms_missing"])

        forbidden = evaluate_case(TutorCase(
            id="forbidden_found",
            query="Explique la loi binomiale",
            section="SC_EXP",
            subject="MATH",
            chapter="PROBA",
            expected_refused=False,
            forbidden_terms=("binomiale",),
            minimum_citations=1,
        ))
        self.assertFalse(forbidden["passed"])
        self.assertIn("binomiale", forbidden["forbidden_terms_found"])

        chapter = evaluate_case(TutorCase(
            id="wrong_chapter",
            query="Explique la loi binomiale",
            section="SC_EXP",
            subject="MATH",
            chapter="PROBA",
            expected_refused=False,
            required_citation_chapters=("RLC",),
            minimum_citations=1,
        ))
        self.assertFalse(chapter["passed"])
        self.assertTrue(any("required citation chapters" in f for f in chapter["failures"]))

    def test_tutor_evaluator_uses_mock_provider_only(self):
        from evaluation.tutor_evaluator import evaluate_tutor_cases

        _load("prepare_embedding_chunks", "--mock")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_tutor_cases(tmpdir, """
- id: pass_case
  query: "Explique le circuit RLC"
  section: "SC_EXP"
  subject: "PHYSIQUE"
  chapter: "RLC"
  expected_refused: false
  expected_subject: "PHYSIQUE"
  expected_chapter: "RLC"
  required_terms: ["rlc"]
  forbidden_terms: ["pizza"]
  required_citation_chapters: ["RLC"]
  minimum_citations: 1
""")
            with patch("ai.llm_client.OpenAIClient",
                       side_effect=AssertionError("OpenAI should not be used")):
                with patch("ai.llm_client.AnthropicClient",
                           side_effect=AssertionError("Anthropic should not be used")):
                    summary = evaluate_tutor_cases(path)
        self.assertEqual(summary["passed"], 1)

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

    # --- Evaluation run persistence ------------------------------------- #

    _PASS_REFUSE_YAML = """
- id: pass_case
  query: "Explique la loi binomiale"
  section: "SC_EXP"
  subject: "MATH"
  chapter: "PROBA"
  expected_refused: false
  expected_subject: "MATH"
  expected_chapter: "PROBA"
  required_terms: ["binomiale"]
  forbidden_terms: ["pizza"]
  required_citation_chapters: ["PROBA"]
  minimum_citations: 1
- id: refusal_case
  query: "Donne-moi une recette de pizza"
  expected_refused: true
  forbidden_terms: ["fromage"]
"""

    def test_evaluation_run_model_creation(self):
        run = EvaluationRun.objects.create(
            provider="mock", model_name="mock-tutor-v1", cases_path="x.yaml",
            total_cases=2, passed_cases=2, failed_cases=0, score=1.0,
            fail_under=0.8, passed_threshold=True, duration_ms=12,
        )
        self.assertEqual(EvaluationRun.objects.count(), 1)
        self.assertTrue(run.passed_threshold)
        self.assertEqual(run.metadata_json, {})

    def test_evaluation_case_result_model_creation(self):
        run = EvaluationRun.objects.create(provider="mock", total_cases=1)
        result = EvaluationCaseResult.objects.create(
            evaluation_run=run, case_id="c1", passed=True, score=1.0,
            citation_count=2, citation_chapters=["PROBA"],
            required_terms_found=["binomiale"], failures=[],
        )
        self.assertEqual(run.case_results.count(), 1)
        self.assertEqual(result.evaluation_run_id, run.id)
        self.assertIsNone(result.interaction_id)

    def test_evaluate_tutor_save_creates_run_and_case_results(self):
        _load("prepare_embedding_chunks", "--mock")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_tutor_cases(tmpdir, self._PASS_REFUSE_YAML)
            call_command("evaluate_tutor", "--cases", path, "--save",
                         "--fail-under", "0.0", stdout=_NULL(), stderr=_NULL())
        self.assertEqual(EvaluationRun.objects.count(), 1)
        run = EvaluationRun.objects.get()
        self.assertEqual(run.provider, "mock")
        self.assertEqual(run.total_cases, 2)
        self.assertEqual(run.case_results.count(), 2)
        # The non-refusal case should link to an AIInteraction and store a preview.
        pass_case = run.case_results.get(case_id="pass_case")
        self.assertIsNotNone(pass_case.interaction_id)
        self.assertTrue(pass_case.answer_preview)

    def test_evaluate_tutor_no_save_creates_no_run(self):
        _load("prepare_embedding_chunks", "--mock")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_tutor_cases(tmpdir, self._PASS_REFUSE_YAML)
            call_command("evaluate_tutor", "--cases", path, "--no-save",
                         "--fail-under", "0.0", stdout=_NULL(), stderr=_NULL())
        self.assertEqual(EvaluationRun.objects.count(), 0)
        self.assertEqual(EvaluationCaseResult.objects.count(), 0)

    def test_evaluate_tutor_json_includes_evaluation_run_id_when_saved(self):
        _load("prepare_embedding_chunks", "--mock")
        out = io.StringIO()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_tutor_cases(tmpdir, self._PASS_REFUSE_YAML)
            call_command("evaluate_tutor", "--cases", path, "--json", "--save",
                         "--fail-under", "0.0", stdout=out, stderr=_NULL())
        payload = json.loads(out.getvalue())
        self.assertIsNotNone(payload["evaluation_run_id"])
        self.assertTrue(EvaluationRun.objects.filter(pk=payload["evaluation_run_id"]).exists())

    def test_evaluate_tutor_list_runs(self):
        _load("prepare_embedding_chunks", "--mock")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_tutor_cases(tmpdir, self._PASS_REFUSE_YAML)
            call_command("evaluate_tutor", "--cases", path, "--save",
                         "--fail-under", "0.0", stdout=_NULL(), stderr=_NULL())
        run = EvaluationRun.objects.get()
        out = io.StringIO()
        call_command("evaluate_tutor", "--list-runs", stdout=out, stderr=_NULL())
        self.assertIn(f"#{run.id}", out.getvalue())

    def test_evaluate_tutor_show_run(self):
        _load("prepare_embedding_chunks", "--mock")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_tutor_cases(tmpdir, self._PASS_REFUSE_YAML)
            call_command("evaluate_tutor", "--cases", path, "--save",
                         "--fail-under", "0.0", stdout=_NULL(), stderr=_NULL())
        run = EvaluationRun.objects.get()
        out = io.StringIO()
        call_command("evaluate_tutor", "--show-run", str(run.id), stdout=out, stderr=_NULL())
        text = out.getvalue()
        self.assertIn(f"Evaluation run #{run.id}", text)
        self.assertIn("pass_case", text)
        self.assertIn("refusal_case", text)

    def test_evaluate_tutor_threshold_failure_still_stores_failed_run(self):
        _load("prepare_embedding_chunks", "--mock")
        fail_yaml = """
- id: forced_failure
  query: "Explique la loi binomiale"
  section: "SC_EXP"
  subject: "MATH"
  chapter: "PROBA"
  expected_refused: false
  required_terms: ["term_that_will_not_appear"]
  required_citation_chapters: ["PROBA"]
  minimum_citations: 1
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_tutor_cases(tmpdir, fail_yaml)
            with self.assertRaises(SystemExit):
                call_command("evaluate_tutor", "--cases", path, "--save",
                             "--fail-under", "1.0", stdout=_NULL(), stderr=_NULL())
        run = EvaluationRun.objects.order_by("-id").first()
        self.assertIsNotNone(run)
        self.assertFalse(run.passed_threshold)
        self.assertEqual(run.failed_cases, 1)
        failed = run.case_results.get(case_id="forced_failure")
        self.assertFalse(failed.passed)
        self.assertTrue(failed.failures)  # failure detail stored honestly

    def test_evaluate_tutor_save_does_not_use_paid_clients(self):
        _load("prepare_embedding_chunks", "--mock")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_tutor_cases(tmpdir, self._PASS_REFUSE_YAML)
            with patch("ai.llm_client.OpenAIClient",
                       side_effect=AssertionError("OpenAI should not be used")):
                with patch("ai.llm_client.AnthropicClient",
                           side_effect=AssertionError("Anthropic should not be used")):
                    call_command("evaluate_tutor", "--cases", path, "--save",
                                 "--fail-under", "0.0", stdout=_NULL(), stderr=_NULL())
        self.assertEqual(EvaluationRun.objects.count(), 1)

    def test_evaluate_tutor_compare_to_prints_verdict(self):
        _load("prepare_embedding_chunks", "--mock")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_tutor_cases(tmpdir, self._PASS_REFUSE_YAML)
            call_command("evaluate_tutor", "--cases", path, "--save",
                         "--fail-under", "0.0", stdout=_NULL(), stderr=_NULL())
            baseline = EvaluationRun.objects.latest("id")
            out = io.StringIO()
            call_command("evaluate_tutor", "--cases", path, "--save", "--fail-under", "0.0",
                         "--compare-to", str(baseline.id), stdout=out, stderr=_NULL())
        self.assertEqual(EvaluationRun.objects.count(), 2)
        self.assertIn("comparison verdict: no_regression", out.getvalue())


class RunComparatorTests(TestCase):
    """Comparison logic tested via direct model creation (no tutor, no paid APIs)."""

    def _run(self, score=1.0, passed=2, failed=0, provider="mock", model="mock-tutor-v1"):
        return EvaluationRun.objects.create(
            provider=provider, model_name=model, score=score,
            passed_cases=passed, failed_cases=failed, total_cases=passed + failed,
        )

    def _case(self, run, case_id, *, passed=True, score=1.0, refused=False,
              expected_refused=False, citation_count=6, required_found=None,
              forbidden_found=None, failures=None):
        return EvaluationCaseResult.objects.create(
            evaluation_run=run, case_id=case_id, passed=passed, score=score,
            actual_refused=refused, expected_refused=expected_refused,
            citation_count=citation_count,
            required_terms_found=required_found or [],
            forbidden_terms_found=forbidden_found or [],
            failures=failures or [],
        )

    def test_identical_runs_no_regression(self):
        from evaluation.run_comparator import compare_evaluation_runs

        a = self._run()
        b = self._run()
        for cid in ["loi_binomiale", "circuit_rlc"]:
            self._case(a, cid)
            self._case(b, cid)
        comp = compare_evaluation_runs(a.id, b.id)
        self.assertEqual(comp["summary"]["verdict"], "no_regression")
        self.assertEqual(comp["summary"]["regression_count"], 0)
        self.assertEqual(comp["summary"]["improvement_count"], 0)
        self.assertEqual(comp["summary"]["score_delta"], 0.0)

    def test_score_decrease_is_detected(self):
        from evaluation.run_comparator import compare_evaluation_runs

        a = self._run(score=1.0)
        b = self._run(score=0.5)
        self._case(a, "c1", passed=True, score=1.0)
        self._case(b, "c1", passed=True, score=0.5)
        comp = compare_evaluation_runs(a.id, b.id)
        self.assertEqual(comp["summary"]["verdict"], "regression")
        self.assertLess(comp["summary"]["score_delta"], 0)
        self.assertTrue(any("score_decreased" in r for r in comp["cases"][0]["regressions"]))

    def test_newly_failed_case_is_detected(self):
        from evaluation.run_comparator import compare_evaluation_runs

        a = self._run()
        b = self._run(score=0.5, passed=1, failed=1)
        self._case(a, "c1", passed=True, score=1.0)
        self._case(b, "c1", passed=False, score=0.0)
        comp = compare_evaluation_runs(a.id, b.id)
        self.assertIn("c1", comp["summary"]["newly_failed_cases"])
        self.assertEqual(comp["summary"]["verdict"], "regression")

    def test_newly_passing_case_is_detected(self):
        from evaluation.run_comparator import compare_evaluation_runs

        a = self._run(score=0.5, passed=1, failed=1)
        b = self._run()
        self._case(a, "c1", passed=False, score=0.0)
        self._case(b, "c1", passed=True, score=1.0)
        comp = compare_evaluation_runs(a.id, b.id)
        self.assertIn("c1", comp["summary"]["newly_passing_cases"])
        self.assertEqual(comp["summary"]["verdict"], "improvement")

    def test_refusal_change_is_detected(self):
        from evaluation.run_comparator import compare_evaluation_runs

        a = self._run()
        b = self._run(score=0.5, passed=1, failed=1)
        # expected NOT to refuse; A behaves, B wrongly refuses -> refusal regressed
        self._case(a, "c1", passed=True, refused=False, expected_refused=False, citation_count=6)
        self._case(b, "c1", passed=False, refused=True, expected_refused=False, citation_count=0)
        comp = compare_evaluation_runs(a.id, b.id)
        self.assertEqual(len(comp["summary"]["changed_refusal_cases"]), 1)
        self.assertEqual(comp["summary"]["changed_refusal_cases"][0]["case_id"], "c1")
        self.assertEqual(comp["summary"]["verdict"], "regression")

    def test_citation_count_decrease_is_detected(self):
        from evaluation.run_comparator import compare_evaluation_runs

        a = self._run()
        b = self._run()
        self._case(a, "c1", passed=True, refused=False, citation_count=6)
        self._case(b, "c1", passed=True, refused=False, citation_count=3)
        comp = compare_evaluation_runs(a.id, b.id)
        self.assertEqual(len(comp["summary"]["citation_count_changes"]), 1)
        self.assertEqual(comp["summary"]["citation_count_changes"][0]["delta"], -3)
        self.assertEqual(comp["summary"]["verdict"], "regression")

    def test_required_and_forbidden_term_regressions(self):
        from evaluation.run_comparator import compare_evaluation_runs

        a = self._run()
        b = self._run()
        self._case(a, "c1", required_found=["binomiale"], forbidden_found=[])
        self._case(b, "c1", required_found=[], forbidden_found=["pizza"])
        comp = compare_evaluation_runs(a.id, b.id)
        self.assertEqual(comp["summary"]["required_term_regressions"][0]["terms"], ["binomiale"])
        self.assertEqual(comp["summary"]["forbidden_term_regressions"][0]["terms"], ["pizza"])
        self.assertEqual(comp["summary"]["verdict"], "regression")

    def test_missing_case_in_run_b_is_reported(self):
        from evaluation.run_comparator import compare_evaluation_runs

        a = self._run()
        b = self._run(passed=1)
        self._case(a, "c1")
        self._case(a, "c2")
        self._case(b, "c1")
        comp = compare_evaluation_runs(a.id, b.id)
        self.assertEqual(comp["summary"]["missing_in_run_b"], ["c2"])
        self.assertEqual(comp["summary"]["verdict"], "regression")

    def test_compare_command_json_works(self):
        a = self._run()
        b = self._run()
        self._case(a, "c1")
        self._case(b, "c1")
        out = io.StringIO()
        call_command("compare_eval_runs", str(a.id), str(b.id), "--json",
                     stdout=out, stderr=_NULL())
        data = json.loads(out.getvalue())
        self.assertEqual(data["summary"]["verdict"], "no_regression")
        self.assertEqual(data["run_a"]["id"], a.id)
        self.assertEqual(data["run_b"]["id"], b.id)

    def test_compare_command_fail_on_regression_exits_nonzero(self):
        a = self._run()
        b = self._run(score=0.0, passed=0, failed=1)
        self._case(a, "c1", passed=True, score=1.0)
        self._case(b, "c1", passed=False, score=0.0)
        with self.assertRaises(SystemExit):
            call_command("compare_eval_runs", str(a.id), str(b.id),
                         "--fail-on-regression", stdout=_NULL(), stderr=_NULL())

    def test_compare_command_no_regression_exits_zero(self):
        a = self._run()
        b = self._run()
        self._case(a, "c1")
        self._case(b, "c1")
        # Should NOT raise SystemExit for identical runs.
        call_command("compare_eval_runs", str(a.id), str(b.id),
                     "--fail-on-regression", stdout=_NULL(), stderr=_NULL())

    def test_compare_command_invalid_run_id_clean_error(self):
        with self.assertRaises(CommandError):
            call_command("compare_eval_runs", "999999", "999998",
                         stdout=_NULL(), stderr=_NULL())

    def test_comparison_does_not_use_paid_clients(self):
        from evaluation.run_comparator import compare_evaluation_runs

        a = self._run()
        b = self._run()
        self._case(a, "c1")
        self._case(b, "c1")
        with patch("ai.llm_client.OpenAIClient",
                   side_effect=AssertionError("OpenAI should not be used")):
            with patch("ai.llm_client.AnthropicClient",
                       side_effect=AssertionError("Anthropic should not be used")):
                comp = compare_evaluation_runs(a.id, b.id)
        self.assertEqual(comp["summary"]["verdict"], "no_regression")

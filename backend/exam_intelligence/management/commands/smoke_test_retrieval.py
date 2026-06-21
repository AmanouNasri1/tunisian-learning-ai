"""
Smoke-test the retrieval pipeline. NO paid APIs.

Runs REAL pgvector vector search (hybrid with keyword) when all of these hold:
  - database vendor is PostgreSQL
  - the pgvector extension is installed
  - there is at least one chunk with embedding_status='ready' AND embedding IS NOT NULL

Otherwise it falls back to keyword/metadata-only retrieval and prints the precise
reason. The query is embedded with the SAME deterministic mock function used by
`prepare_embedding_chunks --mock`, so the vector path is exercised end-to-end
without any API call. (Mock vectors are not semantic; the keyword half carries
the meaning. This verifies the code path, not embedding quality.)

Usage:
    python manage.py smoke_test_retrieval
"""

from collections import Counter

from django.core.management.base import BaseCommand
from django.db import connection

from backend.exam_intelligence.models import EmbeddingChunk, EmbeddingStatus
from backend.exam_intelligence.management.commands.prepare_embedding_chunks import (
    mock_embedding,
)
from rag.retriever import Retriever, RetrievalFilters

QUERIES = [
    "probabilit\u00e9",
    "probabilite",
    "circuit \u00e9lectrique",
    "circuit electrique",
    "g\u00e9n\u00e9tique",
    "genetique",
    "d\u00e9riv\u00e9e",
    "derivee",
]


class _MockEmbedder:
    """Deterministic, no-API embedder matching prepare_embedding_chunks --mock."""
    model_name = "mock-deterministic-v1"

    def embed(self, texts):
        return [mock_embedding(t) for t in texts]


def pgvector_present(conn) -> bool:
    if conn.vendor != "postgresql":
        return False
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
        return cur.fetchone() is not None


class Command(BaseCommand):
    help = "Retrieval smoke test: real pgvector hybrid search when available, else keyword."

    def handle(self, *args, **options):
        # Exam content contains math symbols (∞, ², …). On Windows the console
        # codec (cp1252) can't encode them and write() would crash. Degrade
        # un-encodable characters instead of failing.
        try:
            self.stdout._out.reconfigure(errors="replace")
        except Exception:
            pass

        vendor = connection.vendor
        has_pgvector = pgvector_present(connection)

        total = EmbeddingChunk.objects.count()
        non_null = EmbeddingChunk.objects.filter(embedding__isnull=False).count()
        ready = EmbeddingChunk.objects.filter(
            embedding_status=EmbeddingStatus.READY, embedding__isnull=False).count()
        status_breakdown = dict(Counter(
            EmbeddingChunk.objects.values_list("embedding_status", flat=True)))
        model_breakdown = dict(Counter(
            EmbeddingChunk.objects.exclude(model_name="").values_list("model_name", flat=True)))

        vector_enabled, reason = self._vector_availability(vendor, has_pgvector, ready)

        # --- diagnostics ---
        self.stdout.write("Retrieval smoke test")
        self.stdout.write("=" * 60)
        self.stdout.write(f"database vendor       : {vendor}")
        self.stdout.write(f"pgvector extension    : {'yes' if has_pgvector else 'no'}")
        self.stdout.write(f"total chunks          : {total}")
        self.stdout.write(f"non-null embeddings   : {non_null}")
        self.stdout.write(f"status breakdown      : {status_breakdown}")
        self.stdout.write(f"model_name breakdown  : {model_breakdown or '{}'}")
        self.stdout.write(f"ready embeddings      : {ready}")
        if vector_enabled:
            self.stdout.write(self.style.SUCCESS("vector search enabled : yes (hybrid: vector + keyword)"))
        else:
            self.stdout.write(self.style.WARNING(f"vector search enabled : no"))
            self.stdout.write(self.style.WARNING(f"fallback reason       : {reason}"))
        self.stdout.write("=" * 60)

        if total == 0:
            self.stdout.write(self.style.WARNING(
                "No chunks — run prepare_embedding_chunks first. Nothing to query."))
            return

        embedder = _MockEmbedder() if vector_enabled else None
        retriever = Retriever(embedder=embedder)
        filters = RetrievalFilters()

        for query in QUERIES:
            self._run_query(retriever, query, filters, vector_enabled)

        self.stdout.write("=" * 60)
        self.stdout.write("Done.")

    # ------------------------------------------------------------------ #

    @staticmethod
    def _vector_availability(vendor, has_pgvector, ready) -> tuple[bool, str]:
        if vendor != "postgresql":
            return False, f"PostgreSQL required (vendor='{vendor}')"
        if not has_pgvector:
            return False, "pgvector extension missing (run: CREATE EXTENSION vector;)"
        if ready == 0:
            return False, ("no ready embeddings (run: prepare_embedding_chunks --mock, "
                           "or a real embedding job)")
        return True, ""

    def _run_query(self, retriever, query, filters, vector_enabled):
        self.stdout.write("-" * 60)
        self.stdout.write(f"Query: {query!r}")
        try:
            result = retriever.retrieve(query, filters)
        except Exception as exc:  # precise vector-query error, then degrade for this query
            self.stdout.write(self.style.ERROR(
                f"  [FAIL] vector query error: {type(exc).__name__}: {exc}"))
            self.stdout.write("  retrying keyword-only for this query...")
            kw = retriever._keyword_search(query, filters, retriever.top_k)
            self._report(query, 0, len(kw), kw, vector_enabled=False)
            return

        self._report(query, result.vector_count, result.keyword_count, result.chunks, vector_enabled)

    def _report(self, query, vec_count, kw_count, chunks, vector_enabled):
        if vector_enabled and vec_count and kw_count:
            mode = "hybrid (vector + keyword)"
        elif vec_count:
            mode = "vector-only"
        elif kw_count:
            mode = "keyword-only"
        else:
            mode = "none"
        self.stdout.write(f"  mode: {mode} | vector_candidates={vec_count} keyword_candidates={kw_count}")
        if not chunks:
            self.stdout.write(self.style.WARNING("  [WARN] no results"))
            return
        top = chunks[0]
        title = (top.content or "").strip().replace("\n", " ")
        self.stdout.write(
            f"  top: [{top.content_type}] chunk#{top.chunk_id} "
            f"section={top.section_code or '?'} subject={top.subject_code or '?'} "
            f"year={top.year or '?'} score={top.score:.4f}")
        self.stdout.write(f"       {title[:75]}")

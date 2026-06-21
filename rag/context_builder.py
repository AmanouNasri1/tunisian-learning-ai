"""
RAG context assembly for the future tutor.

This module retrieves and packages context only. It does not call an LLM, and it
does not call paid embedding APIs. Vector retrieval is used only when the ready
chunk vectors are known mock vectors, so smoke tests can exercise pgvector
without hidden costs. Real query embedding can be injected later behind an
explicit caller-controlled provider.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from django.db import connection

from ai.embeddings import MOCK_MODEL_NAME, MockEmbeddingProvider
from backend.exam_intelligence.models import EmbeddingChunk, EmbeddingStatus
from rag.retriever import RetrievedChunk, Retriever, RetrievalFilters


GROUPS = {
    "exercise": "exercise_statements",
    "question": "questions",
    "correction": "corrections",
    "rubric": "rubric_items",
    "mistake": "common_mistakes",
    "combined": "combined_context",
}


@dataclass
class ContextRequest:
    query: str
    section: str | None = None
    subject: str | None = None
    chapter: str | None = None
    top_k: int | None = None


class RAGContextBuilder:
    """Build structured, cited context for a student query."""

    def build(
        self,
        query: str,
        section: str | None = None,
        subject: str | None = None,
        chapter: str | None = None,
        top_k: int | None = None,
    ) -> dict:
        request = ContextRequest(
            query=(query or "").strip(),
            section=self._normalize_code(section),
            subject=self._normalize_code(subject),
            chapter=self._normalize_code(chapter),
            top_k=self._normalize_top_k(top_k),
        )

        warnings: list[str] = []
        vector_enabled, vector_reason, uses_mock = self._vector_availability()
        if uses_mock:
            warnings.append("mock embeddings in use")
        if not vector_enabled and vector_reason:
            warnings.append(vector_reason)

        embedder = MockEmbeddingProvider() if vector_enabled else None
        retriever = Retriever(embedder=embedder)
        if request.top_k:
            retriever.top_k = request.top_k

        filters = RetrievalFilters(
            section_code=request.section,
            subject_code=request.subject,
            chapter_code=request.chapter,
        )
        result = retriever.retrieve(request.query, filters)
        mode = self._mode(result.vector_count, result.keyword_count)

        selected = [self._chunk_payload(chunk) for chunk in result.chunks]
        grouped = self._group_chunks(result.chunks)
        citations = [self._citation(chunk) for chunk in result.chunks]

        if result.is_weak or not result.chunks:
            warnings.append("weak retrieval")
        if not grouped["corrections"]:
            warnings.append("no correction found")
        if not grouped["rubric_items"]:
            warnings.append("no rubric found")

        return {
            "query": request.query,
            "filters": {
                "section": request.section,
                "subject": request.subject,
                "chapter": request.chapter,
                "top_k": retriever.top_k,
            },
            "retrieval_mode": mode,
            "confidence": result.confidence,
            "selected_chunks": selected,
            "selected_chunk_count": len(selected),
            "grouped_context": grouped,
            "citations": citations,
            "warnings": self._dedupe(warnings),
            "uses_mock_embeddings": uses_mock,
            "diagnostics": {
                "vector_candidates": result.vector_count,
                "keyword_candidates": result.keyword_count,
                "ready_embedding_models": self._ready_model_breakdown(),
            },
        }

    @staticmethod
    def _normalize_code(value: str | None) -> str | None:
        if not value:
            return None
        return value.strip().upper() or None

    @staticmethod
    def _normalize_top_k(value: int | None) -> int | None:
        if value is None:
            return None
        try:
            top_k = int(value)
        except (TypeError, ValueError):
            return None
        return min(max(top_k, 1), 20)

    @staticmethod
    def _mode(vector_count: int, keyword_count: int) -> str:
        if vector_count and keyword_count:
            return "hybrid vector + keyword"
        if vector_count:
            return "vector-only"
        if keyword_count:
            return "keyword-only"
        return "none"

    @staticmethod
    def _chunk_payload(chunk: RetrievedChunk) -> dict:
        return {
            "chunk_id": chunk.chunk_id,
            "content_type": chunk.content_type,
            "score": chunk.score,
            "content": chunk.content,
            "citation": chunk.citation,
        }

    def _group_chunks(self, chunks: list[RetrievedChunk]) -> dict:
        grouped = {
            "exercise_statements": [],
            "questions": [],
            "corrections": [],
            "rubric_items": [],
            "common_mistakes": [],
            "combined_context": [],
            "assembled_context": "",
        }
        for chunk in chunks:
            key = GROUPS.get(chunk.content_type)
            if key:
                grouped[key].append(self._chunk_payload(chunk))
        grouped["assembled_context"] = self._assemble_context(chunks)
        return grouped

    @staticmethod
    def _assemble_context(chunks: list[RetrievedChunk]) -> str:
        order = {
            "exercise": 0,
            "question": 1,
            "correction": 2,
            "rubric": 3,
            "mistake": 4,
            "combined": 5,
        }
        ordered = sorted(chunks, key=lambda c: (order.get(c.content_type, 99), -c.score))
        parts = []
        for chunk in ordered:
            label = GROUPS.get(chunk.content_type, chunk.content_type).replace("_", " ")
            parts.append(f"[chunk#{chunk.chunk_id} {label}]\n{chunk.content}")
        return "\n\n".join(parts)

    @staticmethod
    def _citation(chunk: RetrievedChunk) -> dict:
        return {
            "chunk_id": chunk.chunk_id,
            "source_object_type": chunk.source_object_type,
            "source_object_id": chunk.source_object_id,
            "section": chunk.section_code,
            "subject": chunk.subject_code,
            "year": chunk.year,
            "chapter": chunk.chapter_code,
        }

    @staticmethod
    def _dedupe(values: list[str]) -> list[str]:
        seen = set()
        out = []
        for value in values:
            if value not in seen:
                out.append(value)
                seen.add(value)
        return out

    def _vector_availability(self) -> tuple[bool, str, bool]:
        ready = EmbeddingChunk.objects.filter(
            embedding_status=EmbeddingStatus.READY,
            embedding__isnull=False,
        )
        ready_count = ready.count()
        if ready_count == 0:
            return False, "vector search unavailable: no ready embeddings", False

        breakdown = self._ready_model_breakdown()
        mock_count = sum(
            count for model, count in breakdown.items()
            if model == MOCK_MODEL_NAME or model.startswith("mock/")
        )
        uses_mock = mock_count > 0
        if mock_count != ready_count:
            return (
                False,
                "vector search skipped: ready embeddings are not exclusively mock vectors",
                uses_mock,
            )
        if connection.vendor != "postgresql":
            return False, f"vector search unavailable: database vendor is {connection.vendor}", True
        if not self._pgvector_present():
            return False, "vector search unavailable: pgvector extension missing", True
        return True, "", True

    @staticmethod
    def _pgvector_present() -> bool:
        if connection.vendor != "postgresql":
            return False
        with connection.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
            return cur.fetchone() is not None

    @staticmethod
    def _ready_model_breakdown() -> dict[str, int]:
        return dict(Counter(
            EmbeddingChunk.objects
            .filter(embedding_status=EmbeddingStatus.READY, embedding__isnull=False)
            .exclude(model_name="")
            .values_list("model_name", flat=True)
        ))

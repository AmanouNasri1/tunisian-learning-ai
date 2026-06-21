"""
RAG retrieval service.

Pipeline: filter -> hybrid search (vector + keyword) -> RRF fuse -> rerank ->
relevance/recency weighting -> threshold check -> grounded result.

Vector search uses pgvector (PostgreSQL). Keyword search uses portable ORM
icontains matching over chunk content + denormalized metadata (works on any
backend, including SQLite for local checks). Django models are imported lazily
inside the DB methods so this module stays importable without Django configured.

Nothing here fabricates content. If retrieval is weak it says so (is_weak).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field


@dataclass
class RetrievalFilters:
    """Applied BEFORE search (the columns are denormalized on the chunk for this)."""
    section_code: str | None = None
    subject_code: str | None = None
    chapter_code: str | None = None
    min_relevance_weight: float = 0.0
    exclude_outdated: bool = True
    content_types: list[str] | None = None     # ["correction", "exercise", ...]


@dataclass
class RetrievedChunk:
    chunk_id: int
    content: str
    content_type: str
    score: float
    citation: str
    relevance_weight: float = 1.0
    year: int | None = None
    subject_code: str = ""
    section_code: str = ""


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk] = field(default_factory=list)
    confidence: float = 0.0                     # top score after weighting
    is_weak: bool = False
    vector_count: int = 0                        # candidates from vector search
    keyword_count: int = 0                       # candidates from keyword search

    def by_type(self, content_type: str) -> list[RetrievedChunk]:
        return [c for c in self.chunks if c.content_type == content_type]


class Retriever:
    def __init__(self, embedder=None, reranker=None):
        # Injected so this stays testable and provider-agnostic.
        self.embedder = embedder
        self.reranker = reranker
        self.top_k = int(os.environ.get("RETRIEVAL_TOP_K", "6"))
        self.candidates = int(os.environ.get("RETRIEVAL_CANDIDATES", "30"))
        self.threshold = float(os.environ.get("RETRIEVAL_THRESHOLD", "0.35"))

    # --- public API --------------------------------------------------------- #

    def retrieve(self, query: str, filters: RetrievalFilters | None = None) -> RetrievalResult:
        filters = filters or RetrievalFilters()
        query_vec = self.embedder.embed([query])[0] if self.embedder else None

        vector_hits = self._vector_search(query_vec, filters, limit=self.candidates)
        keyword_hits = self._keyword_search(query, filters, limit=self.candidates)
        fused = self._reciprocal_rank_fusion(vector_hits, keyword_hits)

        reranked = self._rerank(query, fused) if self.reranker else fused[: self.candidates]
        weighted = self._apply_relevance_and_recency(reranked)
        weighted.sort(key=lambda c: c.score, reverse=True)
        top = weighted[: self.top_k]

        confidence = top[0].score if top else 0.0
        return RetrievalResult(
            chunks=top, confidence=confidence, is_weak=confidence < self.threshold,
            vector_count=len(vector_hits), keyword_count=len(keyword_hits),
        )

    # --- backends ----------------------------------------------------------- #

    def _vector_search(self, query_vec, filters: RetrievalFilters, limit: int) -> list[RetrievedChunk]:
        """pgvector cosine search over ready, embedded chunks. Returns [] if no query vector."""
        if query_vec is None:
            return []
        from pgvector.django import CosineDistance
        from backend.exam_intelligence.models import EmbeddingChunk, EmbeddingStatus

        qs = EmbeddingChunk.objects.filter(
            embedding_status=EmbeddingStatus.READY, embedding__isnull=False)
        qs = self._apply_filters(qs, filters)
        qs = (qs.select_related("subject", "section")
              .annotate(distance=CosineDistance("embedding", query_vec))
              .order_by("distance")[:limit])
        # score = cosine similarity = 1 - cosine distance
        return [self._to_chunk(c, 1.0 - float(c.distance)) for c in qs]

    def _keyword_search(self, query: str, filters: RetrievalFilters, limit: int) -> list[RetrievedChunk]:
        """Portable keyword/metadata search (icontains over content + names)."""
        from django.db.models import Q
        from backend.exam_intelligence.models import EmbeddingChunk

        terms = [t for t in re.split(r"\s+", query.strip().lower()) if t]
        if not terms:
            return []
        match = Q()
        for t in terms:
            match |= (Q(content__icontains=t) | Q(subject__name_fr__icontains=t)
                      | Q(section__name_fr__icontains=t) | Q(chapter__name_fr__icontains=t))
        qs = self._apply_filters(EmbeddingChunk.objects.filter(match), filters)
        qs = qs.select_related("subject", "section", "chapter")[:max(limit * 5, 100)]

        scored: list[tuple[int, object]] = []
        for c in qs:
            haystack = " ".join(filter(None, [
                c.content,
                c.subject.name_fr if c.subject else "",
                c.section.name_fr if c.section else "",
                c.chapter.name_fr if c.chapter else "",
            ])).lower()
            hits = sum(1 for t in terms if t in haystack)
            if hits:
                scored.append((hits, c))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [self._to_chunk(c, float(hits)) for hits, c in scored[:limit]]

    # --- helpers ------------------------------------------------------------ #

    @staticmethod
    def _apply_filters(qs, filters: RetrievalFilters):
        if filters.section_code:
            qs = qs.filter(section__code=filters.section_code)
        if filters.subject_code:
            qs = qs.filter(subject__code=filters.subject_code)
        if filters.chapter_code:
            qs = qs.filter(chapter__code=filters.chapter_code)
        if filters.min_relevance_weight:
            qs = qs.filter(relevance_weight__gte=filters.min_relevance_weight)
        if filters.exclude_outdated:
            qs = qs.exclude(relevance_status="outdated")
        if filters.content_types:
            qs = qs.filter(content_type__in=filters.content_types)
        return qs

    @staticmethod
    def _to_chunk(c, score: float) -> RetrievedChunk:
        citation = (f"{c.content_type} chunk#{c.id} "
                    f"({c.source_object_type}#{c.source_object_id}, {c.year or '?'})")
        return RetrievedChunk(
            chunk_id=c.id, content=c.content, content_type=c.content_type,
            score=score, citation=citation,
            relevance_weight=float(c.relevance_weight or 1.0), year=c.year,
            subject_code=c.subject.code if c.subject else "",
            section_code=c.section.code if c.section else "",
        )

    # --- fusion / rerank / weighting (provider-agnostic) ------------------- #

    @staticmethod
    def _reciprocal_rank_fusion(*ranked_lists, k: int = 60) -> list[RetrievedChunk]:
        scores: dict[int, float] = {}
        objs: dict[int, RetrievedChunk] = {}
        for ranked in ranked_lists:
            for rank, chunk in enumerate(ranked):
                scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank + 1)
                objs[chunk.chunk_id] = chunk
        out = []
        for cid, s in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
            chunk = objs[cid]
            chunk.score = s
            out.append(chunk)
        return out

    def _rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Cross-encoder rerank (Cohere/LLM). Returns reranked chunks with normalized scores."""
        return self.reranker.rerank(query, chunks)

    @staticmethod
    def _apply_relevance_and_recency(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """score *= era relevance_weight * mild recency boost."""
        for c in chunks:
            recency = 1.0
            if c.year:
                recency = 1.0 + max(0.0, (c.year - 2008)) * 0.005
            c.score = c.score * float(c.relevance_weight) * recency
        return chunks

"""
Embed pending/missing EmbeddingChunk rows.

Default invocation is a dry run and makes no provider/API calls:

    python manage.py embed_chunks

Actual embedding requires an explicit provider:

    python manage.py embed_chunks --provider mock
    python manage.py embed_chunks --provider openai --limit 20

Use --dry-run with any provider to preview without writes or API calls.
"""

from __future__ import annotations

import os
from itertools import islice

from django.core.management.base import BaseCommand
from django.db.models import Q

from ai.embeddings import (
    DEFAULT_OPENAI_EMBEDDING_MODEL,
    EMBEDDING_DIM,
    MOCK_MODEL_NAME,
    EmbeddingConfigurationError,
    get_embedding_provider,
)
from backend.exam_intelligence.models import EmbeddingChunk, EmbeddingStatus


def _batched(items, size: int):
    iterator = iter(items)
    while True:
        batch = list(islice(iterator, size))
        if not batch:
            return
        yield batch


class Command(BaseCommand):
    help = "Embed chunks with an explicit provider; defaults to dry-run."

    def add_arguments(self, parser):
        parser.add_argument(
            "--provider",
            choices=["mock", "openai"],
            default=None,
            help="Provider to run. Omit for dry-run only.",
        )
        parser.add_argument(
            "--limit",
            type=int,
            default=None,
            help="Maximum number of eligible chunks to process.",
        )
        parser.add_argument(
            "--force",
            action="store_true",
            help="Re-embed all chunks, including ready chunks with existing vectors.",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Preview selected chunks without writes or API calls.",
        )

    def handle(self, *args, **options):
        try:
            self.stdout._out.reconfigure(errors="replace")
        except Exception:
            pass

        provider_arg = options["provider"]
        provider_name = provider_arg or os.environ.get("EMBEDDING_PROVIDER", "mock").lower()
        dry_run = bool(options["dry_run"] or provider_arg is None)
        limit = options["limit"]
        force = options["force"]
        warnings: list[str] = []

        if limit is not None and limit < 1:
            self.stdout.write(self.style.ERROR("--limit must be a positive integer."))
            raise SystemExit(1)

        if provider_name not in {"mock", "openai"}:
            self.stdout.write(self.style.ERROR(
                f"Unknown embedding provider '{provider_name}'. Use mock or openai."))
            raise SystemExit(1)

        candidates = self._eligible_queryset(force)
        chunks_found = candidates.count()
        selected = list(candidates[:limit] if limit else candidates)

        if provider_arg is None:
            warnings.append(
                "No --provider supplied; dry run only. Pass --provider mock or "
                "--provider openai to write embeddings.")
        if dry_run:
            warnings.append("Dry run: no embeddings will be written and no API calls will be made.")
        elif provider_name == "mock":
            warnings.append("Mock embeddings are deterministic placeholders, not semantic embeddings.")
        elif provider_name == "openai":
            warnings.append("OpenAI provider selected; this can call a paid API.")
            if limit is None:
                warnings.append("No --limit set for OpenAI provider; all eligible chunks may be processed.")

        self.stdout.write("Embedding chunk job")
        self.stdout.write("=" * 60)
        self.stdout.write(f"selected provider : {provider_name}")
        self.stdout.write(f"provider model    : {self._provider_model_label(provider_name)}")
        self.stdout.write(f"dry run           : {'yes' if dry_run else 'no'}")
        self.stdout.write(f"force             : {'yes' if force else 'no'}")
        self.stdout.write(f"limit             : {limit if limit is not None else 'none'}")
        self.stdout.write(f"chunks found      : {chunks_found}")
        self.stdout.write(f"chunks selected   : {len(selected)}")
        self.stdout.write(f"estimated batches : {self._estimated_batches(len(selected))}")

        embedded = 0
        failed = 0

        if not dry_run and selected:
            try:
                provider = get_embedding_provider(provider_name)
            except EmbeddingConfigurationError as exc:
                warnings.append(str(exc))
                self._print_summary(embedded, chunks_found, failed, warnings)
                return

            batch_size = self._batch_size()
            for batch in _batched(selected, batch_size):
                texts = [chunk.content for chunk in batch]
                try:
                    vectors = provider.embed(texts)
                    self._validate_vectors(vectors)
                    model_name = self._stored_model_name(provider)
                    for chunk, vector in zip(batch, vectors, strict=True):
                        chunk.embedding = vector
                        chunk.embedding_status = EmbeddingStatus.READY
                        chunk.model_name = model_name
                        chunk.save(update_fields=["embedding", "embedding_status", "model_name"])
                        embedded += 1
                except Exception as exc:
                    failed += len(batch)
                    self.stdout.write(self.style.ERROR(
                        f"[FAIL] batch starting chunk#{batch[0].id}: {type(exc).__name__}: {exc}"))

        self._print_summary(embedded, chunks_found, failed, warnings)

    @staticmethod
    def _eligible_queryset(force: bool):
        qs = EmbeddingChunk.objects.order_by("id")
        if force:
            return qs
        return qs.filter(Q(embedding__isnull=True) | Q(embedding_status=EmbeddingStatus.PENDING))

    @staticmethod
    def _provider_model_label(provider_name: str) -> str:
        if provider_name == "mock":
            return MOCK_MODEL_NAME
        return os.environ.get("EMBEDDING_MODEL", DEFAULT_OPENAI_EMBEDDING_MODEL)

    @staticmethod
    def _stored_model_name(provider) -> str:
        if provider.provider_name == "openai":
            return f"openai/{provider.model_name}"
        return provider.model_name

    @staticmethod
    def _batch_size() -> int:
        try:
            return max(1, int(os.environ.get("EMBEDDING_BATCH_SIZE", "32")))
        except ValueError:
            return 32

    def _estimated_batches(self, selected_count: int) -> int:
        if selected_count == 0:
            return 0
        batch_size = self._batch_size()
        return (selected_count + batch_size - 1) // batch_size

    @staticmethod
    def _validate_vectors(vectors: list[list[float]]):
        for vector in vectors:
            if len(vector) != EMBEDDING_DIM:
                raise ValueError(
                    f"Embedding dimension mismatch: expected {EMBEDDING_DIM}, got {len(vector)}.")

    def _print_summary(self, embedded: int, chunks_found: int, failed: int, warnings: list[str]):
        skipped = max(0, chunks_found - embedded - failed)
        self.stdout.write("-" * 60)
        self.stdout.write(f"embedded count    : {embedded}")
        self.stdout.write(f"skipped count     : {skipped}")
        self.stdout.write(f"failed count      : {failed}")
        self.stdout.write(f"warnings          : {len(warnings)}")
        for warning in warnings:
            self.stdout.write(self.style.WARNING(f"  ! {warning}"))
        if failed:
            self.stdout.write(self.style.ERROR("Result: completed with failures"))
        elif embedded:
            self.stdout.write(self.style.SUCCESS("Result: embeddings updated"))
        else:
            self.stdout.write(self.style.WARNING("Result: no embeddings updated"))

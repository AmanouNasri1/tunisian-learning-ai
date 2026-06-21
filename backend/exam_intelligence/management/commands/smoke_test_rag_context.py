"""
Smoke-test RAG context assembly. NO LLM calls and NO paid API calls.

Usage:
    python manage.py smoke_test_rag_context
"""

from django.core.management.base import BaseCommand

from rag.context_builder import RAGContextBuilder


QUERIES = [
    "probabilit\u00e9",
    "fonction",
    "SVT g\u00e9n\u00e9tique",
    "circuit \u00e9lectrique",
    "physique",
]


class Command(BaseCommand):
    help = "Smoke-test structured RAG context assembly without LLM/API calls."

    def handle(self, *args, **options):
        try:
            self.stdout._out.reconfigure(errors="replace")
        except Exception:
            pass

        builder = RAGContextBuilder()
        self.stdout.write("RAG context smoke test")
        self.stdout.write("=" * 60)

        for query in QUERIES:
            package = builder.build(query)
            self._print_package(query, package)

        self.stdout.write("=" * 60)
        self.stdout.write("Done.")

    def _print_package(self, query: str, package: dict):
        grouped = package["grouped_context"]
        counts = {
            "exercise_statements": len(grouped["exercise_statements"]),
            "questions": len(grouped["questions"]),
            "corrections": len(grouped["corrections"]),
            "rubric_items": len(grouped["rubric_items"]),
            "common_mistakes": len(grouped["common_mistakes"]),
            "combined_context": len(grouped["combined_context"]),
        }

        self.stdout.write("-" * 60)
        self.stdout.write(f"Query: {query!r}")
        self.stdout.write(f"  retrieval mode       : {package['retrieval_mode']}")
        self.stdout.write(f"  selected chunk count : {package['selected_chunk_count']}")
        self.stdout.write(f"  grouped counts       : {counts}")
        self.stdout.write(
            f"  mock embeddings      : {'yes' if package['uses_mock_embeddings'] else 'no'}")

        sources = package["citations"][:3]
        if sources:
            self.stdout.write("  top sources:")
            for source in sources:
                self.stdout.write(
                    "    - chunk#{chunk_id} {source_object_type}#{source_object_id} "
                    "section={section} subject={subject} year={year} chapter={chapter}".format(
                        **source
                    ))
        else:
            self.stdout.write(self.style.WARNING("  top sources          : none"))

        warnings = package["warnings"]
        if warnings:
            self.stdout.write(f"  warnings            : {warnings}")
        else:
            self.stdout.write("  warnings            : []")

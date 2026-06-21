"""Ask the source-grounded tutor from the command line."""

from __future__ import annotations

import json

from django.core.management.base import BaseCommand

from ai.llm_client import LLMConfigurationError
from rag.tutor import answer_student_question


class Command(BaseCommand):
    help = "Ask the source-grounded tutor. Defaults to the safe mock provider."

    def add_arguments(self, parser):
        parser.add_argument("query", help="Student question to answer.")
        parser.add_argument("--provider", choices=["mock", "openai", "anthropic"],
                            default="mock")
        parser.add_argument("--section")
        parser.add_argument("--subject")
        parser.add_argument("--chapter")
        parser.add_argument("--top-k", dest="top_k", type=int, default=6)
        parser.add_argument("--json", action="store_true",
                            help="Print the full structured package as JSON.")

    def handle(self, *args, **options):
        try:
            self.stdout._out.reconfigure(errors="replace")
        except Exception:
            pass

        try:
            package = answer_student_question(
                query=options["query"],
                section=options.get("section"),
                subject=options.get("subject"),
                chapter=options.get("chapter"),
                provider=options["provider"],
                top_k=options["top_k"],
            )
        except LLMConfigurationError as exc:
            self.stdout.write(self.style.ERROR(f"[ERROR] {exc}"))
            raise SystemExit(1)

        if options["json"]:
            self.stdout.write(json.dumps(package, ensure_ascii=False, indent=2))
            return

        self.stdout.write("Tutor answer")
        self.stdout.write("=" * 60)
        self.stdout.write(f"query              : {package['query']}")
        self.stdout.write(f"provider           : {package['provider']} / {package['model_name']}")
        self.stdout.write(f"refused            : {'yes' if package['refused'] else 'no'}")
        if package["refusal_reason"]:
            self.stdout.write(f"refusal reason     : {package['refusal_reason']}")
        self.stdout.write(f"retrieval mode     : {package['retrieval_mode']}")
        self.stdout.write(f"confidence         : {package['confidence']:.4f}")
        self.stdout.write(f"interaction id     : {package['interaction_id']}")
        self.stdout.write("-" * 60)
        self.stdout.write(package["answer"])
        self.stdout.write("-" * 60)
        self.stdout.write(f"warnings           : {package['warnings']}")
        self.stdout.write("citations:")
        for citation in package["citations"][:6]:
            self.stdout.write(
                "  - chunk#{chunk_id} {source_object_type}#{source_object_id} "
                "section={section} subject={subject} year={year} chapter={chapter}".format(
                    **citation
                ))
        self.stdout.write(f"diagnostics        : {package['diagnostics']}")

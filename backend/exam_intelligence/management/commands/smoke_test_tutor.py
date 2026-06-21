"""Smoke-test the source-grounded mock tutor. NO paid APIs."""

from __future__ import annotations

from django.core.management.base import BaseCommand

from backend.exam_intelligence.models import AIInteraction
from rag.tutor import answer_student_question


CASES = [
    ("Explique la loi binomiale", False),
    ("Comment \u00e9tudier une fonction avec la d\u00e9riv\u00e9e ?", False),
    ("Explique le circuit RLC", False),
    ("Explique la g\u00e9n\u00e9tique r\u00e9cessive", False),
    ("Question hors programme: donne-moi une recette de pizza", True),
]


class Command(BaseCommand):
    help = "Smoke-test tutor answers/refusals with the mock provider only."

    def handle(self, *args, **options):
        try:
            self.stdout._out.reconfigure(errors="replace")
        except Exception:
            pass

        before = AIInteraction.objects.count()
        failed = False

        self.stdout.write("Tutor smoke test")
        self.stdout.write("=" * 60)
        for query, should_refuse in CASES:
            package = answer_student_question(query=query, provider="mock")
            ok = self._is_ok(package, should_refuse)
            failed = failed or not ok
            self._print_case(query, package, should_refuse, ok)

        after = AIInteraction.objects.count()
        created = after - before
        self.stdout.write("=" * 60)
        self.stdout.write(f"AIInteraction rows created: {created}")
        if created != len(CASES):
            failed = True
            self.stdout.write(self.style.ERROR(
                f"[FAIL] expected {len(CASES)} interaction rows, got {created}"))

        if failed:
            self.stdout.write(self.style.ERROR("SMOKE TEST TUTOR: FAIL"))
            raise SystemExit(1)
        self.stdout.write(self.style.SUCCESS("SMOKE TEST TUTOR: PASS"))

    @staticmethod
    def _is_ok(package: dict, should_refuse: bool) -> bool:
        if should_refuse:
            return package["refused"] is True and bool(package["refusal_reason"])
        return (
            package["refused"] is False
            and bool(package["answer"])
            and bool(package["citations"])
            and package["provider"] == "mock"
        )

    def _print_case(self, query: str, package: dict, should_refuse: bool, ok: bool):
        status = "[OK]" if ok else "[FAIL]"
        style = self.style.SUCCESS if ok else self.style.ERROR
        self.stdout.write("-" * 60)
        self.stdout.write(style(f"{status} {query!r}"))
        self.stdout.write(f"  expected refused : {'yes' if should_refuse else 'no'}")
        self.stdout.write(f"  actual refused   : {'yes' if package['refused'] else 'no'}")
        self.stdout.write(f"  retrieval mode   : {package['retrieval_mode']}")
        self.stdout.write(
            f"  keyword candidates: {package['diagnostics'].get('keyword_candidates')}")
        self.stdout.write(f"  citations        : {len(package['citations'])}")
        self.stdout.write(f"  warnings         : {package['warnings']}")
        self.stdout.write(f"  interaction id   : {package['interaction_id']}")
        preview = " ".join(package["answer"].split())[:180]
        self.stdout.write(f"  answer preview   : {preview}")

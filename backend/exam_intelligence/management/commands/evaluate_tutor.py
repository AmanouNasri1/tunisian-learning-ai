"""Evaluate source-grounded tutor behavior against golden cases."""

from __future__ import annotations

import json

from django.core.management.base import BaseCommand

from evaluation.tutor_evaluator import DEFAULT_CASES_PATH, evaluate_tutor_cases


class Command(BaseCommand):
    help = "Evaluate the mock tutor against deterministic golden cases."

    def add_arguments(self, parser):
        parser.add_argument("--cases", default=str(DEFAULT_CASES_PATH),
                            help="Path to tutor cases YAML.")
        parser.add_argument("--json", action="store_true",
                            help="Print machine-readable JSON.")
        parser.add_argument("--fail-under", type=float, default=0.8,
                            help="Exit nonzero if overall score is below this value.")
        parser.add_argument("--verbose", action="store_true",
                            help="Print per-case details and answer previews.")

    def handle(self, *args, **options):
        try:
            self.stdout._out.reconfigure(errors="replace")
        except Exception:
            pass

        summary = evaluate_tutor_cases(
            path=options["cases"],
            verbose=options["verbose"],
        )
        summary["fail_under"] = options["fail_under"]
        summary["passed_threshold"] = summary["score"] >= options["fail_under"]

        if options["json"]:
            self.stdout.write(json.dumps(summary, ensure_ascii=False, indent=2))
        else:
            self._print_text(summary, verbose=options["verbose"])

        if not summary["passed_threshold"]:
            raise SystemExit(1)

    def _print_text(self, summary: dict, verbose: bool):
        self.stdout.write("Tutor evaluation")
        self.stdout.write("=" * 60)
        self.stdout.write(f"cases      : {summary['cases']}")
        self.stdout.write(f"passed     : {summary['passed']}")
        self.stdout.write(f"failed     : {summary['failed']}")
        self.stdout.write(f"score      : {summary['score']:.3f}")
        self.stdout.write(f"fail-under : {summary['fail_under']}")
        self.stdout.write("-" * 60)

        for result in summary["results"]:
            if result["passed"]:
                line = f"[PASS] {result['case_id']} score={result['score']:.3f}"
                self.stdout.write(self.style.SUCCESS(line))
            else:
                line = f"[FAIL] {result['case_id']} score={result['score']:.3f}"
                self.stdout.write(self.style.ERROR(line))
                for failure in result["failures"]:
                    self.stdout.write(self.style.ERROR(f"  - {failure}"))
            if verbose:
                self.stdout.write(
                    f"  refused={result['actual_refused']} "
                    f"citations={result['citation_count']} "
                    f"chapters={result['citation_chapters']} "
                    f"interaction_id={result['interaction_id']}"
                )
                if result.get("required_terms_missing"):
                    self.stdout.write(f"  missing terms={result['required_terms_missing']}")
                if result.get("answer_preview"):
                    self.stdout.write(f"  answer={result['answer_preview']}")

        self.stdout.write("=" * 60)
        if summary["passed_threshold"]:
            self.stdout.write(self.style.SUCCESS("RESULT: PASS"))
        else:
            self.stdout.write(self.style.ERROR("RESULT: FAIL"))

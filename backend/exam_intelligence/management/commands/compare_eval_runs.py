"""Compare two persisted tutor evaluation runs case-by-case.

    python manage.py compare_eval_runs <run_a_id> <run_b_id> [--json] [--verbose] [--fail-on-regression]

Run A is the baseline, run B the candidate. No paid APIs.
"""

from __future__ import annotations

import json

from django.core.management.base import BaseCommand, CommandError

from evaluation.run_comparator import (
    EvaluationRunNotFound, compare_evaluation_runs, render_comparison_lines,
    verdict_label,
)

_REGRESSION_VERDICTS = {"regression", "mixed"}


class Command(BaseCommand):
    help = "Compare two saved evaluation runs (baseline vs candidate)."

    def add_arguments(self, parser):
        parser.add_argument("run_a_id", type=int, help="Baseline run id.")
        parser.add_argument("run_b_id", type=int, help="Candidate run id.")
        parser.add_argument("--json", action="store_true", help="Machine-readable output.")
        parser.add_argument("--verbose", action="store_true", help="Show every case.")
        parser.add_argument("--fail-on-regression", action="store_true",
                            help="Exit nonzero if verdict is regression or mixed.")

    def handle(self, *args, **options):
        try:
            self.stdout._out.reconfigure(errors="replace")
        except Exception:
            pass

        try:
            comparison = compare_evaluation_runs(options["run_a_id"], options["run_b_id"])
        except EvaluationRunNotFound as exc:
            raise CommandError(str(exc))

        verdict = comparison["summary"]["verdict"]

        if options["json"]:
            self.stdout.write(json.dumps(comparison, ensure_ascii=False, indent=2, default=str))
        else:
            for line in render_comparison_lines(comparison, verbose=options["verbose"]):
                self.stdout.write(line)
            self.stdout.write("=" * 60)
            label = f"RESULT: {verdict_label(verdict)}"
            if verdict in _REGRESSION_VERDICTS:
                self.stdout.write(self.style.ERROR(label))
            else:
                self.stdout.write(self.style.SUCCESS(label))

        if options["fail_on_regression"] and verdict in _REGRESSION_VERDICTS:
            raise SystemExit(1)

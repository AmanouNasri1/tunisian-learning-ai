"""Evaluate source-grounded tutor behavior against golden cases.

By default the run is PERSISTED (EvaluationRun + EvaluationCaseResult) so results
can be compared over time. Use --no-save for a dry check. No paid APIs are called
(the evaluator always uses the mock provider).
"""

from __future__ import annotations

import argparse
import json

from django.core.management.base import BaseCommand

from evaluation.tutor_evaluator import DEFAULT_CASES_PATH, evaluate_tutor_cases


class Command(BaseCommand):
    help = "Evaluate the mock tutor against deterministic golden cases (persists runs)."

    def add_arguments(self, parser):
        parser.add_argument("--cases", default=str(DEFAULT_CASES_PATH),
                            help="Path to tutor cases YAML.")
        parser.add_argument("--json", action="store_true",
                            help="Print machine-readable JSON.")
        parser.add_argument("--fail-under", type=float, default=0.8,
                            help="Exit nonzero if overall score is below this value.")
        parser.add_argument("--verbose", action="store_true",
                            help="Print per-case details and answer previews.")
        parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=True,
                            help="Persist the run to the database (default: --save). "
                                 "Use --no-save for a dry check.")
        parser.add_argument("--notes", default="",
                            help="Optional note stored on the EvaluationRun.")
        parser.add_argument("--list-runs", action="store_true",
                            help="List recent persisted evaluation runs and exit.")
        parser.add_argument("--show-run", type=int, default=None,
                            help="Show one persisted run (with per-case results) and exit.")

    def handle(self, *args, **options):
        try:
            self.stdout._out.reconfigure(errors="replace")
        except Exception:
            pass

        if options["list_runs"]:
            self._list_runs(as_json=options["json"])
            return
        if options["show_run"] is not None:
            self._show_run(options["show_run"], as_json=options["json"])
            return

        summary = evaluate_tutor_cases(
            path=options["cases"],
            verbose=options["verbose"],
            save=options["save"],
            notes=options["notes"],
            fail_under=options["fail_under"],
        )

        if options["json"]:
            self.stdout.write(json.dumps(summary, ensure_ascii=False, indent=2))
        else:
            self._print_text(summary, verbose=options["verbose"])

        if not summary["passed_threshold"]:
            # The run is already persisted (if --save); we still exit nonzero so CI fails.
            raise SystemExit(1)

    # --- printers ----------------------------------------------------------- #

    def _print_text(self, summary: dict, verbose: bool):
        self.stdout.write("Tutor evaluation")
        self.stdout.write("=" * 60)
        self.stdout.write(f"cases      : {summary['cases']}")
        self.stdout.write(f"passed     : {summary['passed']}")
        self.stdout.write(f"failed     : {summary['failed']}")
        self.stdout.write(f"score      : {summary['score']:.3f}")
        self.stdout.write(f"fail-under : {summary['fail_under']}")
        self.stdout.write(f"provider   : {summary['provider']}/{summary['model_name'] or '?'}")
        self.stdout.write(f"duration   : {summary['duration_ms']} ms")
        if summary.get("evaluation_run_id") is not None:
            self.stdout.write(f"evaluation_run_id: {summary['evaluation_run_id']}")
        else:
            self.stdout.write("evaluation_run_id: (not saved)")
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

    def _list_runs(self, as_json: bool, limit: int = 20):
        from backend.exam_intelligence.models import EvaluationRun

        runs = list(EvaluationRun.objects.all()[:limit])  # ordering = -created_at
        if as_json:
            payload = [self._run_to_dict(run) for run in runs]
            self.stdout.write(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
            return

        self.stdout.write(f"Evaluation runs (most recent {len(runs)})")
        self.stdout.write("=" * 60)
        if not runs:
            self.stdout.write("(none yet — run: evaluate_tutor --save)")
            return
        for run in runs:
            flag = "PASS" if run.passed_threshold else "FAIL"
            self.stdout.write(
                f"#{run.id} {run.created_at:%Y-%m-%d %H:%M} "
                f"{run.provider}/{run.model_name or '?'} "
                f"score={run.score:.3f} [{flag}] "
                f"({run.passed_cases}/{run.total_cases} passed)"
            )

    def _show_run(self, run_id: int, as_json: bool):
        from backend.exam_intelligence.models import EvaluationRun

        run = EvaluationRun.objects.filter(pk=run_id).first()
        if not run:
            self.stderr.write(self.style.ERROR(f"No EvaluationRun with id={run_id}"))
            raise SystemExit(1)

        cases = list(run.case_results.all())
        if as_json:
            payload = self._run_to_dict(run)
            payload["case_results"] = [self._case_to_dict(c) for c in cases]
            self.stdout.write(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
            return

        self.stdout.write(f"Evaluation run #{run.id}")
        self.stdout.write("=" * 60)
        self.stdout.write(f"created_at       : {run.created_at:%Y-%m-%d %H:%M:%S}")
        self.stdout.write(f"provider/model   : {run.provider}/{run.model_name or '?'}")
        self.stdout.write(f"cases_path       : {run.cases_path}")
        self.stdout.write(f"score            : {run.score:.3f} "
                          f"(fail_under={run.fail_under}, "
                          f"{'PASS' if run.passed_threshold else 'FAIL'})")
        self.stdout.write(f"passed/total     : {run.passed_cases}/{run.total_cases}")
        self.stdout.write(f"duration_ms      : {run.duration_ms}")
        if run.git_commit_sha:
            self.stdout.write(f"git_commit_sha   : {run.git_commit_sha}")
        if run.notes:
            self.stdout.write(f"notes            : {run.notes}")
        self.stdout.write("-" * 60)
        for c in cases:
            status = "PASS" if c.passed else "FAIL"
            self.stdout.write(
                f"[{status}] {c.case_id} score={c.score:.3f} "
                f"refused(exp/act)={c.expected_refused}/{c.actual_refused} "
                f"citations={c.citation_count}"
            )
            for failure in (c.failures or []):
                self.stdout.write(self.style.ERROR(f"    - {failure}"))

    @staticmethod
    def _run_to_dict(run) -> dict:
        return {
            "id": run.id,
            "created_at": run.created_at.isoformat(),
            "provider": run.provider,
            "model_name": run.model_name,
            "cases_path": run.cases_path,
            "total_cases": run.total_cases,
            "passed_cases": run.passed_cases,
            "failed_cases": run.failed_cases,
            "score": run.score,
            "fail_under": run.fail_under,
            "passed_threshold": run.passed_threshold,
            "duration_ms": run.duration_ms,
            "git_commit_sha": run.git_commit_sha,
            "notes": run.notes,
        }

    @staticmethod
    def _case_to_dict(case) -> dict:
        return {
            "case_id": case.case_id,
            "passed": case.passed,
            "score": case.score,
            "expected_refused": case.expected_refused,
            "actual_refused": case.actual_refused,
            "citation_count": case.citation_count,
            "citation_chapters": case.citation_chapters,
            "required_terms_missing": case.required_terms_missing,
            "forbidden_terms_found": case.forbidden_terms_found,
            "failures": case.failures,
            "interaction_id": case.interaction_id,
            "answer_preview": case.answer_preview,
        }

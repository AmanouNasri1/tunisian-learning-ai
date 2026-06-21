"""Compare two persisted tutor evaluation runs case-by-case.

Used before enabling a real (paid) LLM provider: run a mock baseline, later run a
gated provider evaluation, then diff them to catch regressions (score drops,
newly-failing cases, refusal/citation changes, term regressions) before trusting
the new provider.

Pure read-only logic over EvaluationRun / EvaluationCaseResult. No paid APIs.
"""

from __future__ import annotations


class EvaluationRunNotFound(LookupError):
    """Raised when a requested evaluation run id does not exist."""


VERDICT_NO_REGRESSION = "no_regression"
VERDICT_REGRESSION = "regression"
VERDICT_IMPROVEMENT = "improvement"
VERDICT_MIXED = "mixed"


def compare_evaluation_runs(run_a_id: int, run_b_id: int) -> dict:
    """Compare run A (baseline) against run B (candidate). Returns structured diff.

    Raises EvaluationRunNotFound if either id is missing.
    """
    from backend.exam_intelligence.models import EvaluationCaseResult, EvaluationRun

    run_a = EvaluationRun.objects.filter(pk=run_a_id).first()
    if run_a is None:
        raise EvaluationRunNotFound(f"EvaluationRun id={run_a_id} does not exist")
    run_b = EvaluationRun.objects.filter(pk=run_b_id).first()
    if run_b is None:
        raise EvaluationRunNotFound(f"EvaluationRun id={run_b_id} does not exist")

    a_cases = {c.case_id: c for c in EvaluationCaseResult.objects.filter(evaluation_run=run_a)}
    b_cases = {c.case_id: c for c in EvaluationCaseResult.objects.filter(evaluation_run=run_b)}

    missing_in_run_b = sorted(set(a_cases) - set(b_cases))   # dropped from B
    missing_in_run_a = sorted(set(b_cases) - set(a_cases))   # new in B
    shared = sorted(set(a_cases) & set(b_cases))

    cases: list[dict] = []
    newly_failed: list[str] = []
    newly_passing: list[str] = []
    changed_refusal: list[dict] = []
    citation_changes: list[dict] = []
    required_term_regressions: list[dict] = []
    forbidden_term_regressions: list[dict] = []
    regressed_cases: list[str] = []
    improved_cases: list[str] = []

    for case_id in shared:
        a = a_cases[case_id]
        b = b_cases[case_id]
        a_refused = bool(a.actual_refused)
        b_refused = bool(b.actual_refused)
        refusal_changed = a_refused != b_refused
        score_delta = round(b.score - a.score, 6)
        citation_delta = b.citation_count - a.citation_count

        regressions: list[str] = []
        improvements: list[str] = []

        # pass/fail transitions
        if a.passed and not b.passed:
            regressions.append("passed_in_a_failed_in_b")
            newly_failed.append(case_id)
        elif not a.passed and b.passed:
            improvements.append("failed_in_a_passed_in_b")
            newly_passing.append(case_id)

        # score movement
        if score_delta < 0:
            regressions.append(f"score_decreased ({a.score:.3f} -> {b.score:.3f})")
        elif score_delta > 0:
            improvements.append(f"score_increased ({a.score:.3f} -> {b.score:.3f})")

        # refusal movement (classified against the case's expected behavior)
        if refusal_changed:
            changed_refusal.append({"case_id": case_id, "a": a_refused, "b": b_refused})
            expected = bool(b.expected_refused)
            if a_refused == expected and b_refused != expected:
                regressions.append("refusal_regressed")
            else:  # b now matches expected, a did not
                improvements.append("refusal_corrected")

        # citation count change (only meaningful for in-scope, non-refused cases)
        if citation_delta != 0:
            citation_changes.append({
                "case_id": case_id, "a": a.citation_count, "b": b.citation_count,
                "delta": citation_delta,
            })
            if not a_refused and not b_refused:
                if citation_delta < 0:
                    regressions.append(f"citation_count_decreased ({a.citation_count} -> {b.citation_count})")
                else:
                    improvements.append(f"citation_count_increased ({a.citation_count} -> {b.citation_count})")

        # required terms that A found but B no longer finds
        lost_required = sorted(set(a.required_terms_found or []) - set(b.required_terms_found or []))
        if lost_required:
            regressions.append(f"required_terms_missing: {lost_required}")
            required_term_regressions.append({"case_id": case_id, "terms": lost_required})

        # forbidden terms that appeared in B but were absent in A
        new_forbidden = sorted(set(b.forbidden_terms_found or []) - set(a.forbidden_terms_found or []))
        if new_forbidden:
            regressions.append(f"forbidden_terms_appeared: {new_forbidden}")
            forbidden_term_regressions.append({"case_id": case_id, "terms": new_forbidden})

        if regressions:
            regressed_cases.append(case_id)
        if improvements:
            improved_cases.append(case_id)

        cases.append({
            "case_id": case_id,
            "a_passed": a.passed,
            "b_passed": b.passed,
            "a_score": a.score,
            "b_score": b.score,
            "score_delta": score_delta,
            "a_refused": a_refused,
            "b_refused": b_refused,
            "refusal_changed": refusal_changed,
            "a_citation_count": a.citation_count,
            "b_citation_count": b.citation_count,
            "citation_delta": citation_delta,
            "a_failures": a.failures or [],
            "b_failures": b.failures or [],
            "regressions": regressions,
            "improvements": improvements,
        })

    # Structural change: a case dropped from B is a regression in coverage.
    regression_count = len(regressed_cases) + len(missing_in_run_b)
    improvement_count = len(improved_cases)
    verdict = _verdict(regression_count, improvement_count)

    return {
        "run_a": _run_summary(run_a),
        "run_b": _run_summary(run_b),
        "summary": {
            "score_delta": round(run_b.score - run_a.score, 6),
            "newly_failed_cases": newly_failed,
            "newly_passing_cases": newly_passing,
            "changed_refusal_cases": changed_refusal,
            "citation_count_changes": citation_changes,
            "required_term_regressions": required_term_regressions,
            "forbidden_term_regressions": forbidden_term_regressions,
            "missing_in_run_a": missing_in_run_a,
            "missing_in_run_b": missing_in_run_b,
            "regression_count": regression_count,
            "improvement_count": improvement_count,
            "verdict": verdict,
        },
        "cases": cases,
    }


def _verdict(regression_count: int, improvement_count: int) -> str:
    if regression_count > 0 and improvement_count > 0:
        return VERDICT_MIXED
    if regression_count > 0:
        return VERDICT_REGRESSION
    if improvement_count > 0:
        return VERDICT_IMPROVEMENT
    return VERDICT_NO_REGRESSION


def _run_summary(run) -> dict:
    return {
        "id": run.id,
        "provider": run.provider,
        "model_name": run.model_name,
        "score": run.score,
        "passed_cases": run.passed_cases,
        "failed_cases": run.failed_cases,
        "total_cases": run.total_cases,
        "created_at": run.created_at.isoformat(),
    }


def render_comparison_lines(comparison: dict, verbose: bool = False) -> list[str]:
    """Human-readable lines for the comparison (shared by commands)."""
    a = comparison["run_a"]
    b = comparison["run_b"]
    s = comparison["summary"]
    lines = [
        "Evaluation run comparison",
        "=" * 60,
        f"Run A: #{a['id']} {a['provider']}/{a['model_name'] or '?'} score={a['score']:.3f}",
        f"Run B: #{b['id']} {b['provider']}/{b['model_name'] or '?'} score={b['score']:.3f}",
        "",
        f"score delta: {s['score_delta']:+.3f}",
        f"newly failed cases: {len(s['newly_failed_cases'])} {s['newly_failed_cases'] or ''}".strip(),
        f"newly passing cases: {len(s['newly_passing_cases'])} {s['newly_passing_cases'] or ''}".strip(),
        f"refusal changes: {len(s['changed_refusal_cases'])}",
        f"citation changes: {len(s['citation_count_changes'])}",
        f"required-term regressions: {len(s['required_term_regressions'])}",
        f"forbidden-term regressions: {len(s['forbidden_term_regressions'])}",
        f"missing in run A: {s['missing_in_run_a'] or 0}",
        f"missing in run B: {s['missing_in_run_b'] or 0}",
        f"regression_count: {s['regression_count']}  improvement_count: {s['improvement_count']}",
        f"verdict: {s['verdict']}",
    ]

    shown = comparison["cases"] if verbose else [
        c for c in comparison["cases"] if c["regressions"] or c["improvements"]
    ]
    lines.append("")
    lines.append("Cases:")
    if not shown:
        lines.append("  (no per-case changes)" if not verbose else "  (no cases)")
    for c in shown:
        marker = "OK" if not c["regressions"] else ("CHG" if c["improvements"] else "REG")
        lines.append(
            f"[{marker}] {c['case_id']} "
            f"score {c['a_score']:.3f} -> {c['b_score']:.3f} "
            f"citations {c['a_citation_count']} -> {c['b_citation_count']}"
        )
        for reg in c["regressions"]:
            lines.append(f"     - regression: {reg}")
        for imp in c["improvements"]:
            lines.append(f"     + improvement: {imp}")

    for case_id in s["missing_in_run_b"]:
        lines.append(f"[MISSING] {case_id} present in run A, absent in run B")
    for case_id in s["missing_in_run_a"]:
        lines.append(f"[NEW] {case_id} present in run B, absent in run A")

    return lines


def verdict_label(verdict: str) -> str:
    return {
        VERDICT_NO_REGRESSION: "NO REGRESSION",
        VERDICT_REGRESSION: "REGRESSION",
        VERDICT_IMPROVEMENT: "IMPROVEMENT",
        VERDICT_MIXED: "MIXED",
    }.get(verdict, verdict.upper())

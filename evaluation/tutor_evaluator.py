"""Deterministic evaluator for source-grounded tutor answers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from rag.text_normalization import normalize_text
from rag.tutor import answer_student_question


DEFAULT_CASES_PATH = Path("evaluation/tutor_cases.yaml")


@dataclass(frozen=True)
class TutorCase:
    id: str
    query: str
    section: str | None = None
    subject: str | None = None
    chapter: str | None = None
    expected_refused: bool = False
    expected_subject: str | None = None
    expected_chapter: str | None = None
    required_terms: tuple[str, ...] = ()
    forbidden_terms: tuple[str, ...] = ()
    required_citation_chapters: tuple[str, ...] = ()
    minimum_citations: int = 0


def load_tutor_cases(path: str | Path = DEFAULT_CASES_PATH) -> list[TutorCase]:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or []
    if not isinstance(raw, list):
        raise ValueError("Tutor cases YAML must contain a list of cases.")
    return [_case_from_dict(item) for item in raw]


def evaluate_tutor_cases(path: str | Path = DEFAULT_CASES_PATH, verbose: bool = False) -> dict:
    cases = load_tutor_cases(path)
    results = [evaluate_case(case, verbose=verbose) for case in cases]
    passed = sum(1 for result in results if result["passed"])
    total = len(results)
    score = passed / total if total else 0.0
    return {
        "cases": total,
        "passed": passed,
        "failed": total - passed,
        "score": score,
        "results": results,
    }


def evaluate_case(case: TutorCase, verbose: bool = False) -> dict:
    tutor_result = answer_student_question(
        query=case.query,
        section=case.section,
        subject=case.subject,
        chapter=case.chapter,
        provider="mock",
    )

    answer_text = tutor_result.get("answer", "")
    answer_norm = normalize_text(answer_text)
    citations = tutor_result.get("citations", [])
    citation_chapters = _unique_nonempty(c.get("chapter") for c in citations)
    citation_subjects = _unique_nonempty(c.get("subject") for c in citations)

    failures: list[str] = []
    checks: list[bool] = []

    checks.append(_expect(
        tutor_result.get("refused") is case.expected_refused,
        failures,
        f"refused mismatch: expected {case.expected_refused}, got {tutor_result.get('refused')}",
    ))

    checks.append(_expect(
        bool(tutor_result.get("diagnostics")),
        failures,
        "missing diagnostics",
    ))
    checks.append(_expect(
        tutor_result.get("provider") == "mock" and bool(tutor_result.get("model_name")),
        failures,
        "missing provider/model metadata or non-mock provider used",
    ))

    required_found = [term for term in case.required_terms if _contains_term(answer_norm, term)]
    required_missing = [term for term in case.required_terms if term not in required_found]
    forbidden_found = [term for term in case.forbidden_terms if _contains_term(answer_norm, term)]

    if case.expected_refused:
        checks.append(_expect(
            len(citations) == 0,
            failures,
            f"refusal should not expose used citations, got {len(citations)}",
        ))
    else:
        checks.append(_expect(
            len(citations) >= case.minimum_citations,
            failures,
            f"citation count {len(citations)} below minimum {case.minimum_citations}",
        ))
        checks.append(_expect(
            not case.expected_subject or case.expected_subject in citation_subjects,
            failures,
            f"expected subject {case.expected_subject!r} not found in citations {citation_subjects}",
        ))
        checks.append(_expect(
            not case.expected_chapter or case.expected_chapter in citation_chapters,
            failures,
            f"expected chapter {case.expected_chapter!r} not found in citations {citation_chapters}",
        ))

    checks.append(_expect(
        not required_missing,
        failures,
        f"required terms missing: {required_missing}",
    ))
    checks.append(_expect(
        not forbidden_found,
        failures,
        f"forbidden terms found: {forbidden_found}",
    ))
    checks.append(_expect(
        all(chapter in citation_chapters for chapter in case.required_citation_chapters),
        failures,
        f"required citation chapters {list(case.required_citation_chapters)} "
        f"not satisfied by {citation_chapters}",
    ))

    passed_checks = sum(1 for check in checks if check)
    score = passed_checks / len(checks) if checks else 0.0
    passed = not failures

    result = {
        "case_id": case.id,
        "passed": passed,
        "score": 1.0 if passed else score,
        "failures": failures,
        "query": case.query,
        "expected_refused": case.expected_refused,
        "actual_refused": tutor_result.get("refused"),
        "citation_count": len(citations),
        "citation_chapters": citation_chapters,
        "required_terms_found": required_found,
        "required_terms_missing": required_missing,
        "forbidden_terms_found": forbidden_found,
        "interaction_id": tutor_result.get("interaction_id"),
    }
    if verbose:
        result["answer_preview"] = " ".join(answer_text.split())[:300]
        result["diagnostics"] = tutor_result.get("diagnostics", {})
    return result


def _case_from_dict(item: dict[str, Any]) -> TutorCase:
    if not isinstance(item, dict):
        raise ValueError(f"Invalid tutor case entry: {item!r}")
    for required in ["id", "query", "expected_refused"]:
        if required not in item:
            raise ValueError(f"Tutor case missing required field {required!r}: {item!r}")
    return TutorCase(
        id=str(item["id"]),
        query=str(item["query"]),
        section=_optional_str(item.get("section")),
        subject=_optional_str(item.get("subject")),
        chapter=_optional_str(item.get("chapter")),
        expected_refused=bool(item["expected_refused"]),
        expected_subject=_optional_str(item.get("expected_subject")),
        expected_chapter=_optional_str(item.get("expected_chapter")),
        required_terms=tuple(str(x) for x in item.get("required_terms") or []),
        forbidden_terms=tuple(str(x) for x in item.get("forbidden_terms") or []),
        required_citation_chapters=tuple(
            str(x) for x in item.get("required_citation_chapters") or []
        ),
        minimum_citations=int(item.get("minimum_citations") or 0),
    )


def _expect(condition: bool, failures: list[str], message: str) -> bool:
    if not condition:
        failures.append(message)
    return condition


def _contains_term(normalized_text: str, term: str) -> bool:
    return normalize_text(term) in normalized_text


def _optional_str(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _unique_nonempty(values) -> list[str]:
    seen = set()
    out = []
    for value in values:
        if value and value not in seen:
            out.append(value)
            seen.add(value)
    return out

"""
Bac Readiness Score v1 (Part 11).

Deliberately simple and explainable. Each component normalized to [0,1]; the
per-subject score is a weighted blend minus a repeated-mistake penalty; the
overall score weights subjects by Bac coefficient.

Missing-component handling: if a component (e.g. no mock taken) has no data, its
weight is redistributed across the present components so the student isn't unfairly
penalized for not having done a mock yet.
"""

from __future__ import annotations

from dataclasses import dataclass

WEIGHTS = {
    "diagnostic": 0.15,
    "accuracy": 0.25,
    "mastery": 0.30,
    "mock": 0.20,
    "recency": 0.10,
}
MAX_MISTAKE_PENALTY = 0.10
PER_MISTAKE_PENALTY = 0.02


@dataclass
class ReadinessComponents:
    diagnostic: float | None      # 0..1 or None if not taken
    accuracy: float | None
    mastery: float | None
    mock: float | None
    recency: float                # always available (defaults to a decay)
    recurring_unresolved_mistakes: int = 0


def subject_readiness(c: ReadinessComponents) -> float:
    present = {k: getattr(c, k) for k in WEIGHTS if getattr(c, k) is not None}
    if not present:
        return 0.0

    total_weight = sum(WEIGHTS[k] for k in present)
    raw = sum(WEIGHTS[k] * present[k] for k in present) / total_weight  # renormalized

    penalty = min(MAX_MISTAKE_PENALTY, PER_MISTAKE_PENALTY * c.recurring_unresolved_mistakes)
    score = max(0.0, min(1.0, raw - penalty))
    return round(100 * score, 1)


def overall_readiness(per_subject: list[tuple[float, float]]) -> float:
    """per_subject = [(subject_score_0_100, coefficient), ...]."""
    pairs = [(s, w) for s, w in per_subject if w > 0]
    if not pairs:
        return 0.0
    return round(sum(s * w for s, w in pairs) / sum(w for _, w in pairs), 1)


# Worked example (matches Part 11 of the design):
if __name__ == "__main__":
    maths = ReadinessComponents(diagnostic=0.6, accuracy=0.7, mastery=0.65,
                                mock=0.5, recency=1.0, recurring_unresolved_mistakes=2)
    print("Maths readiness:", subject_readiness(maths))   # ~63.0
    print("Overall:", overall_readiness([(63.0, 3), (58.0, 4), (66.0, 4)]))

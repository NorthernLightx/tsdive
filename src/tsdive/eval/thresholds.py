"""The worst-baseline alarm rule and the false-alarm floor it carries.

A window exchangeable with k baseline windows exceeds their maximum with
probability 1/(k+1), so that is the false-alarm rate a tool that reads
nothing scores, and every published false-alarm rate is read against it.
"""

from __future__ import annotations

from collections.abc import Iterable


def worst_baseline_threshold(baseline_scores: Iterable[float]) -> float:
    """The alarm threshold: the largest score the tool gave a baseline window.

    Raises ``ValueError`` when ``baseline_scores`` is empty.
    """
    scores = [float(s) for s in baseline_scores]
    if not scores:
        raise ValueError("worst_baseline_threshold needs at least one baseline score")
    return max(scores)


def fires(score: float, threshold: float) -> bool:
    """``True`` when ``score`` is strictly above ``threshold``."""
    return bool(score > threshold)


def far_floor(k_baseline: int) -> float:
    """False-alarm rate of a score that reads nothing, at k baseline windows.

    Returns ``1 / (k_baseline + 1)``. Raises ``ValueError`` when
    ``k_baseline`` is below 1.
    """
    if k_baseline < 1:
        raise ValueError("far_floor needs at least one baseline window")
    return 1.0 / (k_baseline + 1)


def over_floor(false_alarm_rate: float, k_baseline: int) -> float:
    """How far a measured false-alarm rate sits above ``far_floor(k_baseline)``."""
    return float(false_alarm_rate) - far_floor(k_baseline)

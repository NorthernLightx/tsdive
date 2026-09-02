"""Evaluation protocol for detector studies.

The package holds the pieces a published detector number is built from:
group holdout that keeps every instance of a group on one side of the
split (``group_holdout``, ``GroupSplit``, ``GroupLeakage``), ranking
metrics that report a one-class fold as a refusal (``ranking_metrics``),
the clock control that is published beside every detector
(``clock_control``), and the worst-baseline alarm rule with the
false-alarm floor it carries (``worst_baseline_threshold``, ``fires``,
``far_floor``, ``over_floor``).
"""

from __future__ import annotations

from tsdive.data.splits import GroupSplit, group_holdout
from tsdive.errors import GroupLeakage
from tsdive.eval.controls import clock_control
from tsdive.eval.metrics import RankingMetrics, ranking_metrics
from tsdive.eval.thresholds import far_floor, fires, over_floor, worst_baseline_threshold

__all__ = [
    "GroupLeakage",
    "GroupSplit",
    "RankingMetrics",
    "clock_control",
    "far_floor",
    "fires",
    "group_holdout",
    "over_floor",
    "ranking_metrics",
    "worst_baseline_threshold",
]

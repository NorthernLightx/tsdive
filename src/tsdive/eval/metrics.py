"""Ranking metrics for a scored fold, with a refusal for a one-class fold."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ONE_CLASS_REFUSAL = "fold under test carries one class only; no ranking metric exists"


@dataclass(frozen=True)
class RankingMetrics:
    """Threshold-free metrics of a detector's scores against binary labels.

    ``roc_auc``, ``pr_auc`` and ``precision_at_recall`` are ``None`` when
    ``refusal`` is set. ``to_dict()`` rounds every metric to four decimals
    and is the row a study publishes.
    """

    roc_auc: float | None
    pr_auc: float | None
    precision_at_recall: float | None
    n_pos: int
    n_neg: int
    refusal: str | None

    def to_dict(self) -> dict:
        def rounded(value: float | None) -> float | None:
            return None if value is None else round(float(value), 4)

        return {
            "roc_auc": rounded(self.roc_auc),
            "pr_auc": rounded(self.pr_auc),
            "precision_at_recall_50": rounded(self.precision_at_recall),
            "n_pos": self.n_pos,
            "n_neg": self.n_neg,
            "refusal": self.refusal,
        }


def _average_ranks(scores: np.ndarray) -> np.ndarray:
    """Rank from 1, tied scores sharing the mean of the ranks they span."""
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    top = np.cumsum(counts)
    return (top - (counts - 1) / 2.0)[inverse]


def _pr_curve(y: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Precision and recall at each distinct score, highest score first.

    The point at a score counts every window scoring at or above it as
    flagged, which is the curve ``sklearn.metrics.precision_recall_curve``
    builds before it appends its final (precision 1, recall 0) point.
    """
    order = np.argsort(-scores, kind="mergesort")
    y_sorted = y[order]
    s_sorted = scores[order]
    last_of_run = np.r_[np.where(np.diff(s_sorted))[0], y.size - 1]
    tps = np.cumsum(y_sorted)[last_of_run]
    fps = 1 + last_of_run - tps
    precision = tps / (tps + fps)
    recall = tps / tps[-1]
    return precision, recall


def ranking_metrics(y, scores, *, recall_floor: float = 0.5) -> RankingMetrics:
    """Score a fold: ROC-AUC, PR-AUC and precision at a recall floor.

    ROC-AUC is the Mann-Whitney statistic, a tie counting one half, which
    equals ``sklearn.metrics.roc_auc_score``. PR-AUC is the step-wise
    average precision ``sklearn.metrics.average_precision_score`` returns.
    ``precision_at_recall`` is the largest precision on the curve at any
    recall of at least ``recall_floor``. A fold with one class present
    returns the refusal and ``None`` for every metric.
    """
    y = np.asarray(y).astype(bool)
    scores = np.asarray(scores, dtype=float)
    if y.shape != scores.shape or y.ndim != 1:
        raise ValueError("y and scores must be one-dimensional and the same length")
    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return RankingMetrics(None, None, None, n_pos, n_neg, ONE_CLASS_REFUSAL)

    ranks = _average_ranks(scores)
    roc_auc = (ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

    precision, recall = _pr_curve(y.astype(int), scores)
    pr_auc = float(np.sum(np.diff(np.r_[0.0, recall]) * precision))
    reachable = precision[recall >= recall_floor]
    at_recall = float(reachable.max()) if reachable.size else None
    return RankingMetrics(float(roc_auc), pr_auc, at_recall, n_pos, n_neg, None)

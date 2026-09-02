"""tsdive.eval: ranking metrics against scikit-learn, controls, thresholds."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

import tsdive.eval as ev
from tsdive.eval import (
    RankingMetrics,
    clock_control,
    far_floor,
    fires,
    over_floor,
    ranking_metrics,
    worst_baseline_threshold,
)
from tsdive.eval.metrics import ONE_CLASS_REFUSAL

TOL = 1e-9


def _cases(n_cases: int = 200, seed: int = 7) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    out: list[tuple[np.ndarray, np.ndarray]] = []
    degenerate = [
        (np.array([0, 1]), np.array([0.0, 1.0])),
        (np.array([1, 0]), np.array([0.0, 1.0])),
        (np.array([0, 1, 0, 1]), np.array([0.5, 0.5, 0.5, 0.5])),
        (np.array([0, 0, 0, 1]), np.array([3.0, 2.0, 1.0, 0.0])),
        (np.array([1, 1, 1, 0]), np.array([1.0, 1.0, 1.0, 1.0])),
        (np.array([0, 1, 1, 0, 1]), np.array([-1.0, -1.0, 2.0, 2.0, 2.0])),
    ]
    out.extend(degenerate)
    while len(out) < n_cases:
        n = int(rng.integers(2, 60))
        y = rng.integers(0, 2, size=n)
        if y.sum() == 0 or y.sum() == n:
            continue
        kind = rng.integers(0, 4)
        if kind == 0:
            scores = rng.normal(size=n) + y * rng.uniform(0.0, 2.0)
        elif kind == 1:
            scores = rng.integers(0, 3, size=n).astype(float)
        elif kind == 2:
            scores = np.round(rng.normal(size=n), 1)
        else:
            scores = np.where(rng.random(n) < 0.7, 0.0, rng.normal(size=n))
        out.append((y, scores))
    return out


def test_ranking_metrics_match_scikit_learn_before_rounding():
    worst = {"roc_auc": 0.0, "pr_auc": 0.0, "precision_at_recall": 0.0}
    for y, scores in _cases():
        got = ranking_metrics(y, scores)
        assert got.refusal is None
        precision, recall, _ = precision_recall_curve(y, scores)
        want = {
            "roc_auc": float(roc_auc_score(y, scores)),
            "pr_auc": float(average_precision_score(y, scores)),
            "precision_at_recall": float(precision[recall >= 0.5].max()),
        }
        for name, expected in want.items():
            deviation = abs(getattr(got, name) - expected)
            worst[name] = max(worst[name], deviation)
            assert deviation <= TOL, (name, y.tolist(), scores.tolist())
        assert got.n_pos == int(y.sum()) and got.n_neg == int(len(y) - y.sum())
    assert max(worst.values()) <= TOL


def test_to_dict_rounds_to_four_decimals_with_the_study_keys():
    y = np.array([0, 0, 1, 1, 0, 1])
    scores = np.array([0.1, 0.4, 0.35, 0.8, 0.7, 0.2])
    row = ranking_metrics(y, scores).to_dict()
    assert list(row) == [
        "roc_auc",
        "pr_auc",
        "precision_at_recall_50",
        "n_pos",
        "n_neg",
        "refusal",
    ]
    assert row["roc_auc"] == round(float(roc_auc_score(y, scores)), 4)
    assert row["pr_auc"] == round(float(average_precision_score(y, scores)), 4)
    assert row["refusal"] is None
    assert isinstance(ranking_metrics(y, scores), RankingMetrics)


@pytest.mark.parametrize("y", [np.zeros(5, dtype=int), np.ones(5, dtype=int), np.array([])])
def test_one_class_fold_is_a_refusal_not_a_number(y):
    row = ranking_metrics(y, np.arange(len(y), dtype=float)).to_dict()
    assert row == {
        "roc_auc": None,
        "pr_auc": None,
        "precision_at_recall_50": None,
        "n_pos": int(y.sum()) if len(y) else 0,
        "n_neg": int(len(y) - y.sum()) if len(y) else 0,
        "refusal": "fold under test carries one class only; no ranking metric exists",
    }
    assert row["refusal"] == ONE_CLASS_REFUSAL


def test_ranking_metrics_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        ranking_metrics([0, 1, 1], [0.1, 0.2])


def test_far_floor_and_over_floor():
    assert far_floor(3) == 0.25
    assert far_floor(1) == 0.5
    with pytest.raises(ValueError):
        far_floor(0)
    with pytest.raises(ValueError):
        far_floor(-2)
    assert over_floor(0.45, 3) == pytest.approx(0.20)
    assert over_floor(0.25, 3) == pytest.approx(0.0)
    assert over_floor(0.1, 3) == pytest.approx(-0.15)


def test_fires_is_strict():
    assert fires(1.01, 1.0)
    assert not fires(1.0, 1.0)
    assert not fires(0.99, 1.0)
    assert fires(np.float64(2.0), 1) is True


def test_worst_baseline_threshold_is_the_max_and_rejects_empty():
    assert worst_baseline_threshold([0.2, 1.5, -3.0]) == 1.5
    assert worst_baseline_threshold(x for x in [4, 2]) == 4.0
    with pytest.raises(ValueError):
        worst_baseline_threshold([])


def test_clock_control_on_a_dict_and_on_a_series():
    from_dict = clock_control({7: 0, 3: 1, 9: 2})
    assert from_dict == {7: 0.0, 3: 1.0, 9: 2.0}
    assert list(from_dict) == [7, 3, 9]
    assert all(isinstance(v, float) for v in from_dict.values())
    series = pd.Series([2, -1, 5], index=[10, 11, 12], name="window_index")
    from_series = clock_control(series)
    assert from_series == {10: 2.0, 11: -1.0, 12: 5.0}
    assert list(from_series) == [10, 11, 12]


def test_reexports_resolve():
    from tsdive.data.splits import GroupSplit, group_holdout
    from tsdive.errors import GroupLeakage

    assert ev.group_holdout is group_holdout
    assert ev.GroupSplit is GroupSplit
    assert ev.GroupLeakage is GroupLeakage
    for name in ev.__all__:
        assert getattr(ev, name) is not None

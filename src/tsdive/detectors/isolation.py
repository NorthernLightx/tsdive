"""ML detector, IsolationForest over physics-gated feature rows.

Rules of engagement:

- Trained only on feature rows whose windows passed the physics layer;
  rows containing None features are EXCLUDED and the exclusion is
  reported, never silently dropped.
- Deterministic: random_state pinned; training data order is fixed by
  sorting on (tag, window_start).
- Scores are oriented so HIGHER = more anomalous (negated decision
  function), because sklearn's default orientation reads backwards.

scikit-learn is an optional extra (``tsdive[ml]``): the IsolationForest
detector is the only thing that needs it, and importing it would cost
every other caller. It is imported when a forest is actually built.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from sklearn.ensemble import IsolationForest


def _isolation_forest_class() -> type[IsolationForest]:
    try:
        from sklearn.ensemble import IsolationForest
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "the IsolationForest detector needs scikit-learn, which tsdive ships "
            "as the optional 'ml' extra: install tsdive[ml] (uv: "
            "`uv sync --extra ml`). Everything else works without it."
        ) from e

    return IsolationForest


FEATURE_COLUMNS = (
    "weighted_mean",
    "std",
    "min",
    "max",
    "median",
    "mad",
    "distinct_count",
    "stall_s",
    "changes_per_hour",
)


@dataclass(frozen=True)
class IsolationResult:
    scores: pd.Series  # index aligned with kept rows; higher = more anomalous
    n_excluded_incomplete: int


def prepare_rows(features: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Select finite feature columns; count excluded incomplete rows."""
    missing = [c for c in FEATURE_COLUMNS if c not in features.columns]
    if missing:
        raise KeyError(f"feature frame missing columns: {missing}")
    sub = features[["tag", "window_start", *FEATURE_COLUMNS]].sort_values(
        ["tag", "window_start"]
    )
    complete = sub.dropna(subset=list(FEATURE_COLUMNS))
    return complete.reset_index(drop=True), len(sub) - len(complete)


def train_isolation(rows: pd.DataFrame, *, seed: int = 42) -> IsolationForest:
    model = _isolation_forest_class()(contamination="auto", random_state=seed)
    model.fit(rows[list(FEATURE_COLUMNS)].to_numpy(dtype=float))
    return model


def score_isolation(
    model: IsolationForest,
    rows: pd.DataFrame,
    *,
    n_excluded_incomplete: int = 0,
) -> IsolationResult:
    """Score prepared rows. ``n_excluded_incomplete`` is the count that
    ``prepare_rows`` dropped for these rows; it travels with the result so
    the exclusion is reported rather than lost."""
    raw = model.decision_function(rows[list(FEATURE_COLUMNS)].to_numpy(dtype=float))
    return IsolationResult(
        scores=pd.Series(-raw, index=rows.index, name="anomaly_score"),
        n_excluded_incomplete=n_excluded_incomplete,
    )

"""Cross-instance population baselines per (asset, variable).

A tag can hold one value for its whole archive. Every threshold at
stages 3-5 is computed from the tag's own history, so a tag that never
moved supplies a reference that never moved either: its p99 change
interval is empty and its reference distinct counts are all 1. A
self-referencing detector therefore cannot flag it.

The reference here is the *population* instead: every other window of the
same variable on the same asset, taken from training folds only. A window
stuck on one value is indicted when that pair normally moves, and the
evidence names how many windows the claim rests on. A pair with too few
training windows is refused with :class:`PopulationTooSparse` rather than
screened against a baseline of three.

Asset-scoped on purpose. A transmitter that is flat on one asset may be
flat because that asset is idle, so pooling assets would let a busy
asset's movement indict a quiet one's sensor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tsdive.errors import PopulationTooSparse

# Below this many training windows the percentiles below are noise: a p05
# over 8 windows is the smallest of 8 numbers.
MIN_BASELINE_WINDOWS = 20


@dataclass(frozen=True)
class PopulationBaseline:
    """What one (asset, variable) pair normally does, over training windows."""

    asset: str
    variable: str
    n_windows: int
    min_windows: int
    p05_distinct: float
    median_distinct: float
    p95_stall_s: float
    median_stall_s: float
    n_frozen_windows: int
    method: str = "per-(asset, variable) window percentiles over training folds"

    @property
    def sufficient(self) -> bool:
        return self.n_windows >= self.min_windows

    @property
    def frozen_share(self) -> float:
        return self.n_frozen_windows / self.n_windows if self.n_windows else 0.0


@dataclass(frozen=True)
class PopulationScreen:
    fired: bool
    observed_distinct: int
    baseline_p05_distinct: float
    baseline_median_distinct: float
    n_baseline_windows: int
    evidence: str


def population_baseline(
    train_rows: pd.DataFrame,
    *,
    asset_col: str = "asset",
    variable_col: str = "variable",
    distinct_col: str = "distinct_count",
    stall_col: str = "stall_s",
    min_windows: int = MIN_BASELINE_WINDOWS,
) -> dict[tuple[str, str], PopulationBaseline]:
    """Summarise each (asset, variable) pair's training windows.

    ``train_rows`` is a feature table - one row per (window, variable) -
    restricted to the training folds by the caller. Passing rows from the
    fold under test would make every number below a memory of the answer,
    so this function does not accept a fold argument at all: the
    restriction is the caller's, visibly.

    Pairs with fewer than ``min_windows`` rows are still returned, carrying
    their true ``n_windows``; :func:`screen_population` is what refuses
    them. A baseline that quietly vanished would look the same as a pair
    nobody asked about.
    """
    for col in (asset_col, variable_col, distinct_col, stall_col):
        if col not in train_rows.columns:
            raise KeyError(f"column {col!r} missing from the feature table")

    out: dict[tuple[str, str], PopulationBaseline] = {}
    usable = train_rows[train_rows[distinct_col].notna()]
    for (asset, variable), group in usable.groupby(
        [asset_col, variable_col], sort=True, observed=True
    ):
        distinct = group[distinct_col].to_numpy(dtype=float)
        stall = pd.to_numeric(group[stall_col], errors="coerce").to_numpy(dtype=float)
        stall = stall[np.isfinite(stall)]
        out[(str(asset), str(variable))] = PopulationBaseline(
            asset=str(asset),
            variable=str(variable),
            n_windows=len(distinct),
            min_windows=min_windows,
            p05_distinct=float(np.quantile(distinct, 0.05)),
            median_distinct=float(np.median(distinct)),
            p95_stall_s=float(np.quantile(stall, 0.95)) if len(stall) else float("nan"),
            median_stall_s=float(np.median(stall)) if len(stall) else float("nan"),
            n_frozen_windows=int((distinct <= 1).sum()),
        )
    return out


def match_population(
    baselines: dict[tuple[str, str], PopulationBaseline], asset: str, variable: str
) -> PopulationBaseline:
    """The baseline for one pair, or a refusal naming the pair."""
    try:
        return baselines[(str(asset), str(variable))]
    except KeyError as e:
        raise PopulationTooSparse(
            f"no training window for ({asset!r}, {variable!r}); "
            "this pair appears only in the fold under test"
        ) from e


def screen_population(row: pd.Series, baseline: PopulationBaseline) -> PopulationScreen:
    """Indict a frozen window when its pair normally moves.

    Fires only on the case a self-referencing threshold cannot reach: the
    window holds one distinct value, and the same (asset, variable) held
    more than one in at least 95% of its training windows. Both halves are
    required - a pair that is usually frozen says nothing about a frozen
    window, and this returns ``fired=False`` saying exactly that.
    """
    if not baseline.sufficient:
        raise PopulationTooSparse(
            f"({baseline.asset!r}, {baseline.variable!r}) has {baseline.n_windows} "
            f"training window(s); {baseline.min_windows} required before a "
            "percentile of them may indict anything"
        )
    observed = row.get("distinct_count")
    if observed is None or pd.isna(observed):
        raise PopulationTooSparse(
            f"({baseline.asset!r}, {baseline.variable!r}): the window under test has "
            "no distinct-value count, so there is nothing to compare"
        )
    observed_int = int(observed)
    pair = f"({baseline.asset}, {baseline.variable})"
    moves = baseline.p05_distinct > 1
    fired = observed_int <= 1 and moves
    if fired:
        evidence = (
            f"window holds {observed_int} distinct value; {pair} holds more than one "
            f"in at least 95% of its {baseline.n_windows} training windows "
            f"(p05 {baseline.p05_distinct:.4g}, median {baseline.median_distinct:.4g})"
        )
    elif observed_int > 1:
        evidence = (
            f"window holds {observed_int} distinct values, so it is not frozen; "
            f"{pair} baseline over {baseline.n_windows} training windows not consulted"
        )
    else:
        evidence = (
            f"window holds {observed_int} distinct value, but {pair} is frozen in "
            f"{baseline.n_frozen_windows} of its {baseline.n_windows} training windows "
            f"(p05 {baseline.p05_distinct:.4g}); a normally-still sensor is not "
            "indicted for being still"
        )
    return PopulationScreen(
        fired=fired,
        observed_distinct=observed_int,
        baseline_p05_distinct=baseline.p05_distinct,
        baseline_median_distinct=baseline.median_distinct,
        n_baseline_windows=baseline.n_windows,
        evidence=evidence,
    )

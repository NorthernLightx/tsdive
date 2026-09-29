"""PROVISIONAL baselines, the MAD screen and the moving-range screen.

Provisional by design: these ignore regime structure (regime baselines
refine them). Every object carries its caveat; the benchmark table
states it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

from tsdive.errors import InsufficientQuality, ZeroSpreadBaseline
from tsdive.store.quality import usable_mask

MAD_TO_SIGMA = 1.4826  # consistency constant for normal data
MR_D2 = 1.128  # d2 for subgroup size 2: sigma ~ MRbar / d2

MIN_HISTORY_GOOD = 30


@dataclass(frozen=True)
class ProvisionalBaseline:
    kind: str  # "MAD" | "MOVING_RANGE"
    center: float
    scale: float
    n_history_good: int
    caveat: str = "provisional: one baseline for every regime in the window"

    def limits(self, k: float = 3.0) -> tuple[float, float]:
        return self.center - k * self.scale, self.center + k * self.scale


def _good_values(history: pd.DataFrame) -> np.ndarray:
    if "quality" not in history.columns or "value" not in history.columns:
        raise InsufficientQuality("history frame needs value+quality columns")
    good = history[usable_mask(history)]
    vals = (
        cast(pd.Series, pd.to_numeric(good["value"], errors="coerce"))
        .dropna()
        .to_numpy(dtype=float)
    )
    if len(vals) < MIN_HISTORY_GOOD:
        raise InsufficientQuality(
            f"only {len(vals)} GOOD history samples; {MIN_HISTORY_GOOD} required "
            "for a baseline"
        )
    return vals


def zero_spread(values: np.ndarray, *, scale: str, label: str | None) -> ZeroSpreadBaseline:
    """The refusal for a baseline whose ``scale`` is 0 over the GOOD ``values``.

    The message names ``label`` (the tag and the baseline window, when the
    caller knows them), the count of distinct values, and the most common
    value with its share of the samples.
    """
    distinct, counts = np.unique(values, return_counts=True)
    top = int(np.argmax(counts))
    where = f"{label}: " if label else ""
    kinds = "value" if len(distinct) == 1 else "values"
    return ZeroSpreadBaseline(
        f"{where}the {scale} is 0, because the {len(values)} GOOD baseline samples hold "
        f"{len(distinct)} distinct {kinds} and {distinct[top]:g} makes up "
        f"{counts[top] / len(values):.0%} of them; choose a baseline window where the "
        "tag moves"
    )


def mad_baseline(history: pd.DataFrame, *, label: str | None = None) -> ProvisionalBaseline:
    """Median/MAD center+scale. Robust to spikes; still regime-blind.

    Raises ``ZeroSpreadBaseline`` when the MAD is 0, that is when more
    than half the GOOD samples share one value. ``label`` names the tag
    and window in that message.
    """
    vals = _good_values(history)
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    if mad == 0:
        raise zero_spread(vals, scale="MAD scale", label=label)
    return ProvisionalBaseline(
        kind="MAD", center=med, scale=mad * MAD_TO_SIGMA, n_history_good=len(vals)
    )


def moving_range_baseline(
    history: pd.DataFrame, *, label: str | None = None
) -> ProvisionalBaseline:
    """Individuals-style scale from mean absolute successive range.

    Sensitive to dynamics: on a stepping process it measures step size,
    not noise, hence provisional status and the MAD companion. Raises
    ``ZeroSpreadBaseline`` when every GOOD sample holds one value.
    """
    vals = _good_values(history)
    mr = float(np.mean(np.abs(np.diff(vals))))
    if mr == 0:
        raise zero_spread(vals, scale="moving-range scale", label=label)
    med = float(np.median(vals))
    return ProvisionalBaseline(
        kind="MOVING_RANGE",
        center=med,
        scale=mr / MR_D2,
        n_history_good=len(vals),
        caveat="provisional: moving-range scale includes process motion as well as noise",
    )


@dataclass(frozen=True)
class ScreenResult:
    kind: str
    n_screened: int
    n_flagged: int
    flagged_timestamps: tuple[pd.Timestamp, ...]
    limits: tuple[float, float]


def screen(baseline: ProvisionalBaseline, window: pd.DataFrame, k: float = 3.0) -> ScreenResult:
    """Flag GOOD samples outside k*scale of center. Counts only; no blend."""
    lo, hi = baseline.limits(k)
    good = window[usable_mask(window)]
    vals = cast(pd.Series, pd.to_numeric(good["value"], errors="coerce"))
    mask = ((vals < lo) | (vals > hi)) & vals.notna()
    mask = cast(pd.Series, mask.fillna(False))
    ts = tuple(cast(pd.Timestamp, t) for t in good.loc[mask, "timestamp"])
    return ScreenResult(
        kind=baseline.kind,
        n_screened=len(good),
        n_flagged=len(ts),
        flagged_timestamps=ts,
        limits=(lo, hi),
    )

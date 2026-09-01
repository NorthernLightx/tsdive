"""Regime-aware refined baselines keyed by MODE tags.

Per-regime center/scale with a sparsity refusal. When no MODE series is
supplied, an explicit conditioning column may be given instead; both
paths raise rather than derive regimes from raw value clustering alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

from tsdive.baselines.provisional import (
    MAD_TO_SIGMA,
    ScreenResult,
)
from tsdive.errors import InsufficientQuality, RegimeTooSparse
from tsdive.store.quality import usable_mask


def _require_alignment(frame: pd.DataFrame, modes: pd.Series, what: str) -> None:
    """Refuse a mode series that does not sit on the frame's own index.

    Every regime mask below is a pandas boolean operation, which aligns on
    the index rather than on position. A ``modes`` series of the right
    length carrying a different index therefore produces mostly-empty
    regimes and a shower of :class:`RegimeTooSparse` refusals that name
    the wrong cause. Length alone was the old check, and a caller that
    reset one index and not the other slipped through it.
    """
    if len(modes) != len(frame):
        raise ValueError(f"modes must align with {what} rows")
    if not modes.index.equals(frame.index):
        raise ValueError(
            f"modes carries a different index from {what}; regime masks align on "
            "the index, so a same-length series on another index would silently "
            "empty every regime - reindex one to match the other"
        )


@dataclass(frozen=True)
class RegimeBaseline:
    regime: str
    center: float
    scale: float
    n_good: int
    method: str = "per-regime median/MAD"


def regime_baselines(
    history: pd.DataFrame,
    modes: pd.Series,
    *,
    min_samples: int = 20,
) -> dict[str, RegimeBaseline]:
    """One baseline per distinct mode value.

    ``modes`` must align row-for-row with ``history``. A regime with fewer
    than ``min_samples`` GOOD rows raises :class:`RegimeTooSparse` rather
    than producing a quietly wide interval.
    """
    _require_alignment(history, modes, "history")
    usable = usable_mask(history) & cast(
        pd.Series, pd.to_numeric(history["value"], errors="coerce")
    ).notna()
    out: dict[str, RegimeBaseline] = {}
    for regime in sorted(set(str(m) for m in modes)):
        mask = usable & (modes.astype(str) == regime)
        vals = (
            cast(pd.Series, pd.to_numeric(history.loc[mask, "value"], errors="coerce"))
            .to_numpy(dtype=float)
        )
        if len(vals) < min_samples:
            raise RegimeTooSparse(
                f"regime {regime!r} has {len(vals)} GOOD samples; {min_samples} required"
            )
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med))) * MAD_TO_SIGMA
        out[regime] = RegimeBaseline(regime=regime, center=med, scale=mad, n_good=len(vals))
    if not out:
        raise InsufficientQuality("no GOOD samples in history at all")
    return out


def match_regime(baselines: dict[str, RegimeBaseline], mode_value: str) -> RegimeBaseline:
    try:
        return baselines[str(mode_value)]
    except KeyError as e:
        raise RegimeTooSparse(f"no baseline for unseen regime {mode_value!r}") from e


def screen_regime(
    baselines: dict[str, RegimeBaseline],
    window: pd.DataFrame,
    modes: pd.Series,
    k: float = 3.0,
) -> ScreenResult:
    """Screen a window whose rows carry their regime labels."""
    _require_alignment(window, modes, "window")
    mode_str = cast(pd.Series, modes.astype(str))
    flagged: list[pd.Timestamp] = []
    n_screened = 0
    for regime in sorted(set(mode_str)):
        base = match_regime(baselines, regime)
        mask = mode_str == regime
        sub = cast(pd.DataFrame, window.loc[mask])
        good = sub[usable_mask(sub)]
        vals = cast(pd.Series, pd.to_numeric(good["value"], errors="coerce"))
        hits = (
            (vals < base.center - k * base.scale) | (vals > base.center + k * base.scale)
        ) & vals.notna()
        hit_mask = cast(pd.Series, hits.fillna(False))
        flagged.extend(
            cast(pd.Timestamp, t) for t in good.loc[hit_mask, "timestamp"]
        )
        n_screened += len(good)
    return ScreenResult(
        kind="REGIME_MAD",
        n_screened=n_screened,
        n_flagged=len(flagged),
        flagged_timestamps=tuple(flagged),
        limits=(float("nan"), float("nan")),
    )

"""Contract-aware averaging.

The same recorded samples produce different averages under different
calculation bases. This module computes exactly what the declared
contract says and raises when results from different contracts are mixed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tsdive.errors import InsufficientQuality, NonMonotonicIndex
from tsdive.store.quality import Severity
from tsdive.store.sampling_contract import CalculationBasis, SamplingContract


def _good_values(frame: pd.DataFrame) -> pd.DataFrame:
    mask = (frame["severity"] == Severity.GOOD) & frame["value"].notna()
    good = frame.loc[mask.fillna(False).astype(bool)]
    if not isinstance(good, pd.DataFrame):  # pragma: no cover - defensive
        raise TypeError("expected a DataFrame after masking")
    if len(good) == 0:
        raise InsufficientQuality(
            "no GOOD samples remain after masking; refusing to invent a statistic"
        )
    return good


def time_weighted_mean(
    frame: pd.DataFrame,
    *,
    window_end: pd.Timestamp | None = None,
) -> float:
    """Stepped-interpolation time-weighted mean over GOOD samples.

    Each sample holds until the next sample (or window end). Requires a
    monotonic index and raises :class:`NonMonotonicIndex` rather than
    sorting silently.
    """
    good = _good_values(frame)
    ts = good["timestamp"].reset_index(drop=True)
    vals = good["value"].astype(float).reset_index(drop=True)
    # Whole-array arithmetic, not a row loop: a 1 Hz historian hands this
    # function 3,600 samples per hour and a per-row .iloc walk costs more
    # than reading the archive it came from.
    epoch = ts.to_numpy(dtype="datetime64[ns]").astype("int64")
    backwards = [int(i) for i in np.flatnonzero(np.diff(epoch) < 0) + 1]
    if backwards:
        # Positions index the GOOD subset, not the caller's frame. Saying
        # "positions" alone reads the same as ensure_monotonic's frame-row
        # message and sends the reader to the wrong rows.
        raise NonMonotonicIndex(
            f"non-monotonic timestamps at GOOD-sample positions {backwards} "
            "(counted over GOOD samples, not frame rows); refusing to sort silently",
            offending_positions=backwards,
        )
    end = window_end if window_end is not None else ts.iloc[-1]
    end_ns = np.int64(pd.Timestamp(end).to_datetime64().astype("int64"))
    seg_end = np.append(epoch[1:], end_ns)
    weights = np.maximum(0.0, (seg_end - epoch) / 1e9)
    total_weight = float(weights.sum())
    accumulator = float(np.dot(vals.to_numpy(dtype=float), weights))
    if total_weight <= 0:
        raise InsufficientQuality("window has zero duration; time-weighted mean undefined")
    return accumulator / total_weight


def event_weighted_mean(frame: pd.DataFrame) -> float:
    """Plain mean over recorded events (each stored sample counts once).

    On compressed archives this is dominated by whatever the exception
    reporting happened to store, so it must never be silently swapped for
    the time-weighted figure.
    """
    good = _good_values(frame)
    return float(good["value"].astype(float).mean())


def weighted_mean(
    frame: pd.DataFrame,
    contract: SamplingContract,
    *,
    window_end: pd.Timestamp | None = None,
) -> float:
    """Compute the mean dictated by the contract's calculation basis."""
    if contract.calculation_basis == CalculationBasis.TIME_WEIGHTED:
        return time_weighted_mean(frame, window_end=window_end)
    return event_weighted_mean(frame)

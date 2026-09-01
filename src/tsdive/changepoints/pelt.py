"""Offline changepoint segmentation with PELT and the L2 cost.

Refined baselines key regimes by a MODE tag. Most tags have none, and
before choosing a baseline a reader needs to see where the tag's own
behaviour changed. That is all this produces: a table of segments with
their timestamps and their level.

What it does not do: name the regimes, claim a cause, or decide which
segment is normal operation. A segment boundary at a saturation edge
looks exactly like one at a genuine process step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise
from typing import cast

import numpy as np
import pandas as pd

from tsdive.baselines.provisional import MAD_TO_SIGMA
from tsdive.errors import InsufficientQuality
from tsdive.store.identity import Role
from tsdive.store.quality import usable_mask
from tsdive.store.tagstore import Window

METHOD = "PELT L2 on MAD-scaled values"

# BIC-shaped default: log n grows the penalty with the sample count, so a
# longer window does not buy more segments by being longer. The
# multiplier was chosen against the synthetic backbone - at 3 the
# segmenter recovers the injected regime steps and adds none of its own
# on a clean day. It is a knob, not a law: a tag whose steps are smaller
# than its noise needs a smaller one, which is why --penalty exists.
DEFAULT_PENALTY_MULTIPLIER = 3.0


def default_penalty(n: int) -> float:
    """``3 * log(n)``, the default penalty on the MAD-scaled series."""
    return DEFAULT_PENALTY_MULTIPLIER * math.log(n)


def pelt_l2(values: np.ndarray, *, penalty: float, min_size: int) -> list[int]:
    """Breakpoints of the penalised least-squares partition (Killick 2012).

    A segment costs the sum of squared deviations from its own mean, read
    off prefix sums of x and x^2 so every candidate is O(1); ``penalty``
    is what one more breakpoint must buy in cost to be worth taking. The
    exact dynamic programme, with Killick's pruning (the L2 cost never
    increases when a segment is split, so the pruning constant is 0).

    Returns the position of the first sample of each new segment, sorted,
    excluding 0 and ``len(values)``. Positions, not timestamps: the cost
    knows nothing about when the samples were taken. ``values`` must be
    finite, and fewer than ``2 * min_size`` of them can hold no break.
    """
    if min_size < 1:
        raise ValueError(f"min_size must be at least 1, not {min_size}")
    x = np.asarray(values, dtype=float)
    n = len(x)
    if n < 2 * min_size:
        return []
    s1 = np.concatenate(([0.0], np.cumsum(x)))
    s2 = np.concatenate(([0.0], np.cumsum(x * x)))
    # f[t] is the best penalised cost of the first t samples; f[0] carries
    # -penalty so that the m-segment optimum is charged m-1 breakpoints.
    f = np.full(n + 1, np.inf)
    f[0] = -penalty
    prev = np.zeros(n + 1, dtype=np.int64)
    # Segment starts pruning has not ruled out. Vectorised over the
    # survivors rather than looped: a series with few breaks prunes to a
    # handful, one with many keeps hundreds, and only numpy sees them.
    keep = np.zeros(1, dtype=np.int64)
    for t in range(min_size, n + 1):
        if t >= 2 * min_size:
            keep = np.append(keep, t - min_size)
        lengths = t - keep
        totals = s1[t] - s1[keep]
        with_penalty = f[keep] + (s2[t] - s2[keep]) - totals * totals / lengths + penalty
        best = int(np.argmin(with_penalty))
        f[t] = with_penalty[best]
        prev[t] = keep[best]
        keep = keep[with_penalty - penalty <= f[t]]

    found: list[int] = []
    at = n
    while at > 0:
        start = int(prev[at])
        if start > 0:
            found.append(start)
        at = start
    return sorted(found)


@dataclass(frozen=True)
class Segment:
    """One piecewise-constant stretch, in the tag's own units."""

    start: pd.Timestamp
    end: pd.Timestamp
    n: int
    median: float
    mad: float


@dataclass(frozen=True)
class Segmentation:
    penalty: float
    min_size: int
    n_used: int
    # Timestamp of the first sample of each new segment.
    breakpoints: tuple[pd.Timestamp, ...]
    segments: tuple[Segment, ...]
    method: str = METHOD


def segment_window(
    window: Window, *, penalty: float | None = None, min_size: int = 10
) -> Segmentation:
    """Segment a window's usable samples into piecewise-constant regimes.

    The cost runs over sample POSITIONS, not time. Irregular spacing and
    the holes a historian leaves are invisible to it, so a breakpoint
    means "the samples changed here", never "the process changed at this
    rate"; the segment table reports each segment's real first and last
    timestamp, which is where the reader sees the gap.

    Values are divided by their MAD-sigma before the cost, so one
    ``penalty`` means the same thing on a flow and on a temperature.

    Raises:
        ValueError: the tag is state-valued (``role=MODE``); its states
            are the regimes already.
        InsufficientQuality: fewer than ``2 * min_size`` usable samples,
            or no spread at all to segment.
    """
    if window.meta.role is Role.MODE:
        raise ValueError(
            f"{window.identity} is state-valued (role=MODE); its states are already "
            "the regimes, so segmenting its labels would invent boundaries inside them"
        )
    frame = window.frame[usable_mask(window.frame)]
    values = cast(
        pd.Series, pd.to_numeric(frame["value"], errors="coerce")
    ).to_numpy(dtype=float)
    stamps = cast(pd.Series, frame["timestamp"].reset_index(drop=True))
    n = len(values)
    if n < 2 * min_size:
        raise InsufficientQuality(
            f"{n} usable samples; {2 * min_size} required to place a break with "
            f"min-size {min_size} on each side of it"
        )
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad == 0.0:
        raise InsufficientQuality(
            f"no spread to segment: the MAD of {n} usable samples is 0, so every "
            "split costs exactly what every other one costs"
        )
    # Centred as well as scaled: the cost sums squares off prefix sums, and
    # subtracting the median keeps that sum small. It moves no cost, only
    # the numerical headroom.
    scaled = (values - median) / (mad * MAD_TO_SIGMA)
    chosen = default_penalty(n) if penalty is None else float(penalty)
    positions = pelt_l2(scaled, penalty=chosen, min_size=min_size)

    bounds = [0, *positions, n]
    segments: list[Segment] = []
    for lo, hi in pairwise(bounds):
        chunk = values[lo:hi]
        centre = float(np.median(chunk))
        segments.append(
            Segment(
                start=cast(pd.Timestamp, stamps.iloc[lo]),
                end=cast(pd.Timestamp, stamps.iloc[hi - 1]),
                n=hi - lo,
                median=centre,
                mad=float(np.median(np.abs(chunk - centre))),
            )
        )
    return Segmentation(
        penalty=chosen,
        min_size=min_size,
        n_used=n,
        breakpoints=tuple(cast(pd.Timestamp, stamps.iloc[p]) for p in positions),
        segments=tuple(segments),
    )

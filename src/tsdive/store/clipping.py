"""Clipping detection against the tag's engineering range.

A value sitting at (or beyond) zero/span, or carrying an Over/Under Range
digital state, is censored: we know only that it exceeded the range, not
what it was. Censored windows are flagged and may never serve as a
baseline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tsdive.errors import InsufficientQuality
from tsdive.store.identity import EngRange, TagMeta
from tsdive.store.quality import (
    Severity,
    digital_state_label,
    per_distinct_code,
)

_OVER_UNDER_STATES = {"Over Range", "Under Range"}


@dataclass(frozen=True)
class ClippingReport:
    """``fraction`` is ``None`` when the engineering range is unknown."""

    fraction: float | None
    n_clipped: int
    n_samples: int
    censored: bool


def clipped_fraction(
    values, qualities, eng_range: EngRange | None
) -> ClippingReport:
    """Fraction of samples at/beyond range or carrying a range digital state.

    Tolerance is 1e-9 of span so float round-trips at exactly full scale
    still count. Unknown eng range yields ``fraction=None``, never a
    placeholder number.
    """
    vals = pd.Series(values).reset_index(drop=True)
    quals = pd.Series(qualities).reset_index(drop=True)
    n = len(vals)
    if n != len(quals):
        raise ValueError(
            f"clipping needs one quality per value; got {n} values and {len(quals)} qualities"
        )
    flags = per_distinct_code(
        quals, lambda q: digital_state_label(q) in _OVER_UNDER_STATES
    ).astype(bool)
    if eng_range is not None and eng_range.span > 0:
        v = pd.to_numeric(vals, errors="coerce").to_numpy(dtype="float64", na_value=np.nan)
        tol = 1e-9 * eng_range.span
        # NaN compares false on both sides: a value that is not a number
        # is not evidence of saturation.
        flags = flags | (v >= eng_range.high - tol) | (v <= eng_range.zero + tol)
    n_clipped = int(flags.sum())
    if eng_range is None or eng_range.span <= 0:
        fraction = None
    else:
        fraction = n_clipped / n if n > 0 else None
    return ClippingReport(
        fraction=fraction,
        n_clipped=n_clipped,
        n_samples=n,
        censored=n_clipped > 0,
    )


def assert_usable_baseline(meta: TagMeta, report: ClippingReport) -> None:
    """Refuse censored windows as baselines.

    A window where the sensor sat at full scale carries no information
    about normal operation; using one as a baseline poisons every
    downstream comparison.
    """
    if report.censored:
        # A digital range state censors a window with no eng range to
        # divide by, so the fraction can be absent even here.
        fraction = (
            "null (eng range unknown)"
            if report.fraction is None
            else f"{report.fraction:.3f}"
        )
        raise InsufficientQuality(
            f"tag {meta.identity}: window is censored "
            f"(clipped fraction {fraction}); a clipped window may never "
            "serve as a baseline"
        )


def severity_floor_ok(counts: dict[Severity, int], min_good: int) -> bool:
    return counts.get(Severity.GOOD, 0) >= min_good

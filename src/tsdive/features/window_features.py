"""Contract-aware feature extraction over physics-validated windows.

A feature is ``None`` when the window does not support it; the notes list
says why. Features never silently drop censoring, gaps, or quality.

:class:`WindowStats` sits beside the feature row for readers rather than
models: the same numbers plus distribution tails, the sample interval the
archive actually carries, and a MODE tag's state counts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from tsdive.store.averaging import weighted_mean
from tsdive.store.identity import Role
from tsdive.store.quality import Severity
from tsdive.store.tagstore import Window


@dataclass(frozen=True)
class WindowFeatures:
    """One row of the feature ledger. ``None`` = not computable, with a note."""

    tag: str
    window_start: str
    window_end: str
    n_samples: int
    n_good: int
    coverage: float | None
    weighted_mean: float | None
    std: float | None
    min: float | None
    max: float | None
    median: float | None
    mad: float | None
    distinct_count: int | None
    stall_s: float | None
    changes_per_hour: float | None
    clipped_fraction: float | None
    censored: bool
    unmapped_quality_codes: int
    notes: tuple[str, ...]

    def to_row(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "notes"}
        d["notes"] = "; ".join(self.notes) if self.notes else ""
        return d


def _good_values(window: Window) -> pd.DataFrame:
    return window.frame[window.frame["valid"]]


def _stall_and_changes(good: pd.DataFrame) -> tuple[float | None, float | None]:
    if len(good) < 2:
        return None, None
    ts = good["timestamp"].reset_index(drop=True)
    # Compared as stored, not as floats: a MODE tag's states ("R0" -> "R1")
    # are changes in exactly the same sense a measurement's are. Compared
    # whole-array, not row by row: an hour of 1 Hz data is 3,600 rows and
    # a .iloc walk over them costs more than every other feature together.
    arr = good["value"].to_numpy(dtype=object)
    change_idx = np.flatnonzero(arr[1:] != arr[:-1]) + 1
    span_s = (ts.iloc[-1] - ts.iloc[0]).total_seconds()
    last_change_s = (
        (ts.iloc[-1] - ts.iloc[int(change_idx[-1])]).total_seconds()
        if len(change_idx)
        else span_s
    )
    per_hour = len(change_idx) / (span_s / 3600.0) if span_s > 0 else None
    return last_change_s, per_hour


def extract(window: Window, *, window_end: pd.Timestamp | None = None) -> WindowFeatures:
    """Extract one feature row from a store-read window."""
    p = window.physics
    good = _good_values(window)
    notes: list[str] = []

    # A MODE tag carries state labels; averaging or spreading them would
    # be arithmetic on names. Refuse the numeric features, keep the counts.
    state_valued = window.meta.role is Role.MODE

    mean_f: float | None = None
    if state_valued:
        notes.append("mean refused: state-valued tag (role=MODE)")
    else:
        try:
            mean_f = weighted_mean(window.frame, window.contract, window_end=window_end)
        except Exception as e:
            notes.append(f"mean refused: {type(e).__name__}")

    values = (
        pd.Series([], dtype="float64")
        if state_valued
        else good["value"].dropna().astype(float)
    )
    std_f = float(values.std()) if len(values) > 1 else None
    min_f = float(values.min()) if len(values) > 0 else None
    max_f = float(values.max()) if len(values) > 0 else None
    median_f = float(values.median()) if len(values) > 0 else None
    mad_f = (
        float(np.median(np.abs(values.to_numpy() - float(median_f))))
        if len(values) > 0 and median_f is not None
        else None
    )
    distinct = int(values.nunique()) if len(values) > 0 else None
    stall_s, cph = _stall_and_changes(good)

    if p.clipping.censored:
        notes.append("censored window: excluded from baseline use upstream")
    coverage = p.coverage.coverage
    if coverage is not None and coverage < 1.0:
        notes.append(f"data-loss gaps present (coverage {coverage:.3f})")

    return WindowFeatures(
        tag=str(window.identity),
        window_start=window.start.isoformat(),
        window_end=window.end.isoformat(),
        n_samples=len(window.frame),
        n_good=len(good),
        coverage=coverage,
        weighted_mean=_round(mean_f),
        std=_round(std_f),
        min=_round(min_f),
        max=_round(max_f),
        median=_round(median_f),
        mad=_round(mad_f),
        distinct_count=distinct,
        stall_s=_round(stall_s, 0),
        changes_per_hour=_round(cph, 4),
        clipped_fraction=p.clipping.fraction,
        censored=p.clipping.censored,
        unmapped_quality_codes=len(p.unmapped_quality_codes),
        notes=tuple(notes),
    )


def _round(v: float | None, nd: int = 6) -> float | None:
    return None if v is None else round(float(v), nd)


# A stored interval half or twice the declared rate describes a different
# clock from the tag metadata. Anything inside that band is what jitter, a
# single long gap, or a rounded declaration already produce.
INTERVAL_DECLARED_FACTOR = 2.0

# Past this many states the list stops being readable; the rest is counted.
MAX_STATES_LISTED = 8


@dataclass(frozen=True)
class WindowStats:
    """Descriptive statistics for one window, wrapped around its feature row.

    Holds a :class:`WindowFeatures` rather than restating it, so the
    feature ledger stays the single definition of every number it already
    computes.
    """

    features: WindowFeatures
    p05: float | None
    p95: float | None
    # None for a single sample: WindowFeatures reports 0.0 there, the
    # median deviation of one value from itself, which reads as a measured
    # spread. std is already None in that case.
    mad: float | None
    # (state, count) for a MODE tag, count-descending then label; empty
    # for every other role. Ties break on the label so the order is stable.
    state_counts: tuple[tuple[str, int], ...]
    interval_median_s: float | None
    interval_p05_s: float | None
    interval_p95_s: float | None
    declared_rate_s: float | None
    interval_differs_from_declared: bool
    state_valued: bool

    @property
    def distinct_count(self) -> int | None:
        """Distinct GOOD values; distinct state labels for a MODE tag."""
        if self.state_valued:
            return len(self.state_counts)
        return self.features.distinct_count

    @property
    def mean_note(self) -> str | None:
        """The feature row's refusal note for the mean, if it refused."""
        for note in self.features.notes:
            if note.startswith("mean refused:"):
                return note
        return None


def compute_stats(window: Window, *, window_end: pd.Timestamp | None = None) -> WindowStats:
    """Descriptive statistics over a window's GOOD rows, plus its clock.

    The sample interval is measured over every row in the window, GOOD or
    not: a timestamp is present whether or not the value beside it is
    usable.
    """
    features = extract(window, window_end=window_end)
    good = _good_values(window)
    state_valued = window.meta.role is Role.MODE

    p05: float | None = None
    p95: float | None = None
    state_counts: tuple[tuple[str, int], ...] = ()
    if state_valued:
        counts = good["value"].astype(str).value_counts()
        state_counts = tuple(
            sorted(
                ((str(k), int(n)) for k, n in counts.items()),
                key=lambda kv: (-kv[1], kv[0]),
            )
        )
    else:
        values = good["value"].dropna().astype(float)
        if len(values) > 0:
            p05 = _round(float(values.quantile(0.05)))
            p95 = _round(float(values.quantile(0.95)))

    intervals = window.frame["timestamp"].diff().dt.total_seconds().dropna()
    has_intervals = len(intervals) > 0
    median_s = float(intervals.median()) if has_intervals else None
    declared = window.meta.sample_rate_s
    differs = (
        median_s is not None
        and declared is not None
        and declared > 0
        and not (
            declared / INTERVAL_DECLARED_FACTOR
            <= median_s
            <= declared * INTERVAL_DECLARED_FACTOR
        )
    )

    return WindowStats(
        features=features,
        p05=p05,
        p95=p95,
        mad=features.mad if len(good) > 1 else None,
        state_counts=state_counts,
        interval_median_s=_round(median_s),
        interval_p05_s=_round(float(intervals.quantile(0.05))) if has_intervals else None,
        interval_p95_s=_round(float(intervals.quantile(0.95))) if has_intervals else None,
        declared_rate_s=declared,
        interval_differs_from_declared=differs,
        state_valued=state_valued,
    )


def severity_floor_note(counts: dict[Severity, int]) -> str:
    return f"GOOD={counts.get(Severity.GOOD, 0)}"

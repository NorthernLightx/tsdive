"""Gap detection and gap classification.

Every gap gets a class with its stated rule attached. ``unknown`` is the
default and is a real answer, not a failure. Benign classes
(compression_steady, sparse_by_design) do not reduce coverage; data-loss
classes (comm_outage_multitag, scan_off, unknown) do.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
import pandas as pd

from tsdive.store.timebase import ensure_monotonic


class GapClass(StrEnum):
    COMPRESSION_STEADY = "compression_steady"
    COMM_OUTAGE_MULTITAG = "comm_outage_multitag"
    SCAN_OFF = "scan_off"
    SPARSE_BY_DESIGN = "sparse_by_design"
    UNKNOWN = "unknown"


DATA_LOSS_CLASSES = frozenset(
    {GapClass.COMM_OUTAGE_MULTITAG, GapClass.SCAN_OFF, GapClass.UNKNOWN}
)


@dataclass(frozen=True)
class GapContext:
    """Everything the classifier is allowed to know about the tag."""

    contract_stepped: bool = False
    observed_median_interval_s: float | None = None
    declared_sample_rate_s: float | None = None
    peer_gap_overlap: bool = False
    scan_off_periods: tuple[tuple[pd.Timestamp, pd.Timestamp], ...] = field(
        default_factory=tuple
    )


@dataclass(frozen=True)
class GapClassification:
    cls: GapClass
    rule: str


@dataclass(frozen=True)
class GapFinding:
    start: pd.Timestamp
    end: pd.Timestamp
    duration_s: float
    classification: GapClassification


def classify_gap(duration_s: float, context: GapContext) -> GapClassification:
    """Classify one gap. Total function; ``unknown`` is the default.

    Precedence (first match wins):

    1. scan_off: gap lies inside a declared scan-off period.
    2. comm_outage_multitag: the same hole appears in a peer tag of the
       same source; collection failed, the process did not stop.
    3. compression_steady: stepped contract and the hole is within 50x
       the observed median interval, so the value did not change and the
       historian stored nothing.
    4. sparse_by_design: hole within 3x the tag's declared sample rate,
       the expected spacing for a slow tag.
    5. unknown: none of the above could be asserted.
    """
    if context.scan_off_periods:
        return GapClassification(
            GapClass.SCAN_OFF,
            "gap inside a declared scan-off period; scanner was off by design",
        )
    if context.peer_gap_overlap:
        return GapClassification(
            GapClass.COMM_OUTAGE_MULTITAG,
            "coincident gap in a peer tag of the same source; collection outage",
        )
    if (
        context.contract_stepped
        and context.observed_median_interval_s is not None
        and context.observed_median_interval_s > 0
        and duration_s <= 50 * context.observed_median_interval_s
    ):
        return GapClassification(
            GapClass.COMPRESSION_STEADY,
            "stepped contract with gap <= 50x median interval; value unchanged, "
            "compression stored nothing",
        )
    if (
        context.declared_sample_rate_s is not None
        and context.declared_sample_rate_s > 0
        and duration_s <= 3 * context.declared_sample_rate_s
    ):
        return GapClassification(
            GapClass.SPARSE_BY_DESIGN,
            "gap <= 3x declared sample rate; expected spacing for this tag",
        )
    return GapClassification(GapClass.UNKNOWN, "no rule matched")


def detect_gaps(
    timestamps: pd.Series, threshold_s: float
) -> list[tuple[pd.Timestamp, pd.Timestamp, float]]:
    """Return (start, end, duration_s) for every inter-arrival > threshold.

    ``timestamps`` must already be non-decreasing;
    :class:`~tsdive.errors.NonMonotonicIndex` otherwise. Sorting here
    would put samples next to neighbours they never had and invent the
    gaps between them, which is the opposite of an account of what the
    historian recorded.
    """
    ts = timestamps.reset_index(drop=True)
    ensure_monotonic(ts)
    if len(ts) < 2:
        return []
    deltas = ts.diff().dt.total_seconds().to_numpy()
    # Only the holes are walked in Python: a clean 1 s archive of a
    # million rows produces an empty index here, not a million compares.
    over = np.flatnonzero(deltas[1:] > threshold_s) + 1
    return [(ts.iloc[i - 1], ts.iloc[i], float(deltas[i])) for i in over]


@dataclass(frozen=True)
class CoverageSummary:
    coverage: float | None
    n_gaps: int
    longest_gap_s: float | None
    gaps: tuple[GapFinding, ...]


def summarize_coverage(
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    gaps: tuple[GapFinding, ...],
) -> CoverageSummary:
    """Coverage = window time minus data-loss-class gap time, over window time.

    A zero-length or reversed span is unpopulatable and yields ``None``,
    never a fake 1.0. Callers are responsible for supplying the window-edge
    gaps (start to first sample, last sample to end); a window with no
    samples at all is expressed as one gap spanning the window, which makes
    coverage 0.0 here.
    """
    duration = (window_end - window_start).total_seconds()
    longest = max((g.duration_s for g in gaps), default=None)
    if duration <= 0:
        return CoverageSummary(coverage=None, n_gaps=len(gaps), longest_gap_s=longest, gaps=gaps)
    lost = sum(g.duration_s for g in gaps if g.classification.cls in DATA_LOSS_CLASSES)
    coverage = max(0.0, min(1.0, (duration - lost) / duration))
    return CoverageSummary(
        coverage=coverage, n_gaps=len(gaps), longest_gap_s=longest, gaps=gaps
    )

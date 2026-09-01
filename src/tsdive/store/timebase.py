"""UTC-only timebase rules and timestamp audits.

Invariants:

- Timestamps are UTC-aware at ingest. Naive timestamps are a schema
  error, never auto-localized.
- A non-monotonic index is reported as a typed refusal
  (:class:`~tsdive.errors.NonMonotonicIndex`) instead of being silently
  sorted. Sorting would fabricate an ordering the historian never had.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from tsdive.errors import NonMonotonicIndex, SchemaError
from tsdive.store.identity import TagIdentity


def backwards_positions(ts: pd.Series) -> list[int]:
    """Positions where a timestamp precedes the one before it.

    Positional, like ``iloc``: the returned indices count rows from the
    start of the window, never the frame's index labels.
    """
    if len(ts) < 2:
        return []
    deltas = ts.diff().to_numpy()[1:]
    return [int(i) + 1 for i in np.flatnonzero(deltas < np.timedelta64(0, "ns"))]


def require_utc(ts: pd.Series) -> None:
    """Raise :class:`SchemaError` on naive or non-datetime timestamps."""
    if not pd.api.types.is_datetime64_any_dtype(ts):
        raise SchemaError("timestamp column must be datetime64")
    if len(ts) > 0 and ts.dt.tz is None:
        raise SchemaError(
            "naive timestamps rejected at ingest: tsdive stores UTC only"
        )


def ensure_monotonic(ts: pd.Series) -> None:
    """Raise :class:`NonMonotonicIndex` if the index ever moves backwards.

    Call this wherever an operation would otherwise be tempted to sort.
    """
    positions = backwards_positions(ts)
    if positions:
        raise NonMonotonicIndex(
            f"non-monotonic timestamps at positions {positions}; "
            "refusing to sort silently",
            offending_positions=positions,
        )


def require_unique(ts: pd.Series) -> None:
    """Raise :class:`SchemaError` on duplicated timestamps.

    Duplicates double-count under summation-style aggregates; they are a
    finding first-class enough to refuse on when uniqueness is required.
    """
    dupes = ts[ts.duplicated()]
    if len(dupes) > 0:
        sample = ", ".join(str(t) for t in dupes.unique()[:5])
        raise SchemaError(f"duplicate timestamps present ({sample}); refusing to double-count")


@dataclass(frozen=True)
class DSTTransition:
    tz: str
    instant_utc: pd.Timestamp
    offset_before_s: int
    offset_after_s: int


@dataclass
class TimestampAuditReport:
    tag: TagIdentity
    n_samples: int = 0
    duplicate_timestamps: list[pd.Timestamp] = field(default_factory=list)
    non_monotonic_positions: list[int] = field(default_factory=list)
    dst_transitions: list[DSTTransition] = field(default_factory=list)

    @property
    def has_findings(self) -> bool:
        return bool(self.duplicate_timestamps or self.non_monotonic_positions)


def _offset_at(instant: datetime, tz: ZoneInfo) -> int:
    return int(instant.astimezone(tz).utcoffset().total_seconds())


def _binary_search_transition(lo: datetime, hi: datetime, tz: ZoneInfo) -> datetime:
    lo_off = _offset_at(lo, tz)
    while (hi - lo).total_seconds() > 1:
        mid = lo + (hi - lo) / 2
        mid = mid.replace(microsecond=0)
        if _offset_at(mid, tz) == lo_off:
            lo = mid
        else:
            hi = mid
    return hi


def find_dst_transitions(
    start_utc: pd.Timestamp, end_utc: pd.Timestamp, tz_name: str
) -> list[DSTTransition]:
    """Locate DST transitions of ``tz_name`` inside ``[start_utc, end_utc]``.

    The archive itself is UTC; this audit exists so reports can state that
    a wall-clock shift occurred inside the window (ambiguity for humans,
    none for the stored data).
    """
    tz = ZoneInfo(tz_name)
    transitions: list[DSTTransition] = []
    cursor = start_utc.to_pydatetime().astimezone(UTC)
    end = end_utc.to_pydatetime().astimezone(UTC)
    step = timedelta(hours=1)
    prev_offset = _offset_at(cursor, tz)
    while cursor < end:
        nxt = min(cursor + step, end)
        new_offset = _offset_at(nxt, tz)
        if new_offset != prev_offset:
            instant = _binary_search_transition(cursor, nxt, tz)
            transitions.append(
                DSTTransition(
                    tz=tz_name,
                    instant_utc=pd.Timestamp(instant),
                    offset_before_s=prev_offset,
                    offset_after_s=_offset_at(instant, tz),
                )
            )
        prev_offset = new_offset
        cursor = nxt
    return transitions


def timestamp_audit(
    tag: TagIdentity,
    df: pd.DataFrame,
    *,
    tz_names: tuple[str, ...] = (),
) -> TimestampAuditReport:
    """Audit a window's timestamps: duplicates, backwards runs, DST in-window."""
    ts = df["timestamp"]
    report = TimestampAuditReport(tag=tag, n_samples=len(ts))
    duped = ts[ts.duplicated()]
    report.duplicate_timestamps = sorted(duped.unique().tolist())
    report.non_monotonic_positions = backwards_positions(ts)
    if len(ts) > 0:
        for name in tz_names:
            report.dst_transitions.extend(
                find_dst_transitions(ts.min(), ts.max(), name)
            )
    return report

"""Fixtures (c) and (d): DST transitions, duplicates, naive timestamps."""

from __future__ import annotations

import pandas as pd
import pytest

from conftest import TagIdentity, annotate, good_frame, make_meta, stepped_contract, write_archive
from tsdive.errors import NonMonotonicIndex, SchemaError
from tsdive.store.averaging import time_weighted_mean
from tsdive.store.tagstore import SingleFileStore
from tsdive.store.timebase import (
    ensure_monotonic,
    require_unique,
    require_utc,
    timestamp_audit,
)

# DST transition instants (2024), verified against zoneinfo:
SPRING_CHICAGO = pd.Timestamp("2024-03-10 08:00:00+00:00")  # 02:00 CST -> 03:00 CDT
FALL_CHICAGO = pd.Timestamp("2024-11-03 07:00:00+00:00")  # 02:00 CDT -> 01:00 CST
SPRING_LONDON = pd.Timestamp("2024-03-31 01:00:00+00:00")  # 01:00 GMT -> 02:00 BST
FALL_LONDON = pd.Timestamp("2024-10-27 01:00:00+00:00")  # 02:00 BST -> 01:00 GMT


def _hourly_around(instant: pd.Timestamp, hours_before: int = 4, hours_after: int = 4):
    start = instant - pd.Timedelta(hours_before, unit="h")
    n = hours_before + hours_after + 1
    return [start + pd.Timedelta(i, unit="h") for i in range(n)]


@pytest.mark.parametrize(
    ("instant", "tz", "before_off_h", "after_off_h"),
    [
        (SPRING_CHICAGO, "America/Chicago", -6, -5),
        (FALL_CHICAGO, "America/Chicago", -5, -6),
        (SPRING_LONDON, "Europe/London", 0, 1),
        (FALL_LONDON, "Europe/London", 1, 0),
    ],
)
def test_silent_paths_across_dst_transitions(instant, tz, before_off_h, after_off_h):
    """diff / shift / integer-window rolling on the UTC index are immune."""
    stamps = _hourly_around(instant)
    values = list(range(len(stamps)))
    idx = pd.DatetimeIndex(stamps)
    s = pd.Series(values, index=idx)

    # diff: every delta exactly one hour, including across the transition.
    deltas = s.index.to_series().diff().dropna()
    assert (deltas == pd.Timedelta(3600, unit="s")).all()

    # shift: adjacent pairing stays aligned, no phantom sample appears.
    shifted = s.shift(1)
    assert len(shifted) == len(s)
    assert (s - shifted).dropna().eq(pd.Series([1] * (len(s) - 1), index=idx[1:])).all()

    # integer-window rolling sums adjacent samples; DST changes nothing.
    rolled = s.rolling(2).sum().dropna()
    assert len(rolled) == len(s) - 1
    assert rolled.iloc[-1] == values[-1] + values[-2]

    # The audit states the wall-clock fact without touching stored data.
    df = good_frame(stamps, [float(v) for v in values])
    report = timestamp_audit(TagIdentity("p", "t"), df, tz_names=(tz,))
    assert len(report.dst_transitions) == 1
    t = report.dst_transitions[0]
    assert t.instant_utc == instant
    assert t.offset_before_s == before_off_h * 3600
    assert t.offset_after_s == after_off_h * 3600


def test_duplicate_timestamp_double_count_guard():
    """Rolling sums silently double-count duplicated stamps; we refuse first."""
    stamps = [
        pd.Timestamp("2024-11-03 06:30:00+00:00"),
        pd.Timestamp("2024-11-03 07:00:00+00:00"),  # fall-back hour begins
        pd.Timestamp("2024-11-03 07:00:00+00:00"),  # duplicate: local 01:00 twice
        pd.Timestamp("2024-11-03 07:30:00+00:00"),
    ]
    s = pd.Series([10.0, 20.0, 30.0, 40.0])
    assert s.rolling(2).sum().iloc[-1] == 70.0  # pandas happily counts overlaps
    with pytest.raises(SchemaError) as err:
        require_unique(pd.Series(pd.DatetimeIndex(stamps)))
    assert "duplicate" in str(err.value).lower()

    df = good_frame(stamps, [10.0, 20.0, 30.0, 40.0])
    report = timestamp_audit(TagIdentity("p", "t"), df)
    assert len(report.duplicate_timestamps) == 1
    assert report.has_findings


def test_naive_timestamp_rejected_at_ingest(tmp_path):
    naive_stamps = [pd.Timestamp("2024-03-01 00:00:00"), pd.Timestamp("2024-03-01 00:00:03")]
    df = pd.DataFrame(
        {
            "timestamp": naive_stamps,
            "value": [1.0, 2.0],
            "quality": ["GOOD", "GOOD"],
        }
    )
    with pytest.raises(SchemaError) as err:
        annotate(df)
    assert "utc only" in str(err.value).lower()
    with pytest.raises(SchemaError):
        require_utc(pd.Series(naive_stamps))

    path = write_archive(
        tmp_path / "src" / "naive.parquet", df, make_meta(source_id="src"), validate=False
    )
    store = SingleFileStore(path)
    import datetime as dt

    with pytest.raises(SchemaError):
        store.read_window(
            make_meta(source_id="src").identity,
            dt.datetime(2024, 3, 1, tzinfo=dt.UTC),
            dt.datetime(2024, 3, 1, 1, tzinfo=dt.UTC),
            stepped_contract(),
        )


def test_non_monotonic_raises_instead_of_silent_sort():
    stamps = [
        pd.Timestamp("2024-03-01 00:00:00+00:00"),
        pd.Timestamp("2024-03-01 00:00:03+00:00"),
        pd.Timestamp("2024-03-01 00:00:01+00:00"),  # goes backwards
    ]
    ts = pd.Series(pd.DatetimeIndex(stamps))
    with pytest.raises(NonMonotonicIndex) as err:
        ensure_monotonic(ts)
    assert err.value.offending_positions == [2]

    df = annotate(good_frame(stamps, [1.0, 2.0, 3.0]))
    with pytest.raises(NonMonotonicIndex):
        time_weighted_mean(df)


def test_audit_reports_backwards_positions():
    stamps = [
        pd.Timestamp("2024-03-01 00:00:00+00:00"),
        pd.Timestamp("2024-03-01 00:00:05+00:00"),
        pd.Timestamp("2024-03-01 00:00:02+00:00"),
    ]
    df = good_frame(stamps, [1.0, 2.0, 3.0])
    report = timestamp_audit(TagIdentity("p", "t"), df)
    assert report.non_monotonic_positions == [2]

"""Fixture (f): flatline fires on stuck tags, spares compression heartbeats."""

from __future__ import annotations

import datetime as dt
from collections import Counter

import numpy as np
import pandas as pd
import pytest

from conftest import (
    Role,
    TagIdentity,
    good_frame,
    make_meta,
    stepped_contract,
    write_archive,
)
from tsdive.api import profile, reference_history
from tsdive.detectors.flatline import NotAssessed, assess_flatline
from tsdive.store.tagstore import SingleFileStore

UTC = dt.UTC


def _window_for(tmp_path, point_id, stamps, values, qualities=None):
    meta = make_meta(source_id="p", point_id=point_id, sample_rate_s=3.0)
    df = (
        good_frame(stamps, values)
        if qualities is None
        else pd.DataFrame(
            {"timestamp": stamps, "value": values, "quality": qualities}
        )
    )
    path = write_archive(tmp_path / "p" / f"{point_id}.parquet", df, meta)
    store = SingleFileStore(path)
    return store.read_window(
        meta.identity,
        dt.datetime(2024, 3, 1, tzinfo=UTC),
        dt.datetime(2024, 3, 1, 2, tzinfo=UTC),
        stepped_contract(),
    )


def _stamps(start_min: int, end_min: int, step_s: int = 30):
    base = pd.Timestamp("2024-03-01 00:00:00+00:00")
    out = []
    t = 0
    while t <= (end_min - start_min) * 60:
        out.append(base + pd.Timedelta(start_min, unit="min") + pd.Timedelta(t, unit="s"))
        t += step_s
    return out


def test_stuck_tag_with_moving_coupled_tag_fires(tmp_path):
    # Frozen throughout the window at 42.0; history shows ~1-min changes.
    stamps = _stamps(0, 120)
    values = [42.0] * len(stamps)
    stuck = _window_for(tmp_path, "STUCK.PV", stamps, values)

    coupled_vals = [10.0 + i * 0.5 for i in range(len(stamps))]
    moving = _window_for(tmp_path / "x", "TIC101.OP", stamps, coupled_vals)

    verdict = assess_flatline(
        stuck,
        reference_change_intervals_s=[55.0, 61.0, 58.0, 62.0, 57.0],
        reference_distinct_counts=[8, 9, 7, 10],
        couplings=[(TagIdentity("p", "STUCK.PV"), TagIdentity("x", "TIC101.OP"))],
        coupled_windows={TagIdentity("x", "TIC101.OP"): moving},
    )
    assert not verdict.saturated_not_frozen
    assert verdict.fired
    fired_signals = {s.signal for s in verdict.signals if s.fired}
    assert "time_since_last_actual_change" in fired_signals
    assert "distinct_value_count_vs_p05" in fired_signals
    assert "zero_std_while_coupled_tag_moves" in fired_signals

    # Every signal states its two numbers, so a caller never parses prose.
    numbers = {s.signal: (s.observed, s.reference) for s in verdict.signals}
    assert numbers["time_since_last_actual_change"] == (7200.0, pytest.approx(61.96, abs=0.01))
    assert numbers["distinct_value_count_vs_p05"] == (1.0, pytest.approx(7.15, abs=0.01))
    observed_std, coupled_std = numbers["zero_std_while_coupled_tag_moves"]
    assert observed_std == 0.0
    assert coupled_std is not None and coupled_std > 0.0


def test_healthy_compression_heartbeat_does_not_fire(tmp_path):
    """Steady process + periodic re-writes: normal compressed signature."""
    stamps = _stamps(0, 120)
    values = [7.2] * len(stamps)  # constant; historian rewrites it every scan
    steady = _window_for(tmp_path, "STEADY.PV", stamps, values)

    # Honest references from this tag's own history: changes are rare
    # (~4h apart) and low distinctness is normal.
    verdict = assess_flatline(
        steady,
        reference_change_intervals_s=[14000.0, 15100.0, 14600.0, 16200.0],
        reference_distinct_counts=[1, 1, 2, 1],
    )
    assert not verdict.saturated_not_frozen
    assert not verdict.fired


def test_saturated_window_reported_not_frozen(archive_factory, store):
    stamps = [
        pd.Timestamp("2024-03-01 00:00:00+00:00"),
        pd.Timestamp("2024-03-01 00:00:03+00:00"),
        pd.Timestamp("2024-03-01 00:00:06+00:00"),
    ]
    from conftest import EngRange

    df = pd.DataFrame(
        {
            "timestamp": stamps,
            "value": [100.0, 100.0, 100.0],
            "quality": ["GOOD", "GOOD", "GOOD"],
        }
    )
    meta = make_meta(source_id="p", point_id="SAT.PV", eng_range=EngRange(zero=0.0, span=100.0))
    archive_factory(df, meta)
    window = store.read_window(
        meta.identity,
        dt.datetime(2024, 3, 1, tzinfo=UTC),
        dt.datetime(2024, 3, 1, 1, tzinfo=UTC),
        stepped_contract(),
    )
    verdict = assess_flatline(
        window,
        reference_change_intervals_s=[60.0, 60.0],
        reference_distinct_counts=[10, 12],
    )
    assert verdict.saturated_not_frozen
    assert not verdict.fired


def _mode_archive(tmp_path, point_id, stamps, values):
    meta = make_meta(source_id="p", point_id=point_id, role=Role.MODE, sample_rate_s=60.0)
    df = pd.DataFrame(
        {"timestamp": stamps, "value": values, "quality": ["GOOD"] * len(stamps)}
    )
    return write_archive(tmp_path / "p" / f"{point_id}.parquet", df, meta)


def test_mode_tag_is_not_assessed_for_flatline(tmp_path):
    """A settled unit holds one state for hours; that is not a stuck sensor."""
    base = pd.Timestamp("2024-03-01 00:00:00+00:00")
    stamps = [base + pd.Timedelta(60 * i, unit="s") for i in range(240)]
    values = [("R0" if (i // 5) % 2 == 0 else "R1") if i < 180 else "R0" for i in range(240)]
    path = _mode_archive(tmp_path, "MODE.STUCK", stamps, values)

    verdict = profile(
        path,
        "2024-03-01T03:00:00+00:00/2024-03-01T04:00:00+00:00",
        flatline=True,
    ).flatline

    assert verdict is not None
    assert not verdict.fired
    assert verdict.not_assessed is NotAssessed.STATE_VALUED_TAG
    assert verdict.summary_lines() == [
        "flatline verdict: NOT ASSESSED (state-valued tag (role=MODE); "
        "stillness is not a flatline signal)"
    ]


def test_mode_tag_reference_history_does_not_coerce_states_to_float(tmp_path):
    """State labels are compared as stored; float() would raise on "R0"."""
    base = pd.Timestamp("2024-03-01 00:00:00+00:00")
    stamps = [base + pd.Timedelta(60 * i, unit="s") for i in range(240)]
    values = [("R0" if (i // 5) % 2 == 0 else "R1") if i < 180 else "R0" for i in range(240)]
    path = _mode_archive(tmp_path, "MODE.STATE", stamps, values)

    intervals, counts = reference_history(
        path,
        pd.Timestamp("2024-03-01T03:00:00+00:00"),
        pd.Timestamp("2024-03-01T04:00:00+00:00"),
    )

    assert counts == [2, 2]  # R0 and R1 in each of the two earlier hours
    # The state is held for five 60 s samples before it flips, so the time
    # between change events is 300 s. 60 s would be the sample spacing.
    assert intervals and set(intervals) == {300.0}


def test_a_window_with_nothing_to_compare_against_is_not_assessed(tmp_path):
    """No reference history, no reference windows, no coupling: no verdict."""
    stamps = _stamps(0, 120)
    window = _window_for(tmp_path, "LONELY.PV", stamps, [42.0] * len(stamps))

    verdict = assess_flatline(
        window, reference_change_intervals_s=[], reference_distinct_counts=[]
    )

    assert not verdict.fired
    assert verdict.not_assessed is NotAssessed.NO_EVALUABLE_SIGNAL
    assert verdict.summary_lines() == [
        "flatline verdict: NOT ASSESSED (no evaluable signal "
        "(no reference history / no coupled tag))"
    ]
    assert not any(s.evaluable for s in verdict.signals)


def test_one_evaluable_signal_is_enough_to_return_a_verdict(tmp_path):
    """Reference windows but no change history still answers the question."""
    stamps = _stamps(0, 120)
    window = _window_for(tmp_path, "SOME.PV", stamps, [42.0] * len(stamps))

    verdict = assess_flatline(
        window, reference_change_intervals_s=[], reference_distinct_counts=[8, 9, 7]
    )

    assert verdict.not_assessed is None
    assert verdict.fired  # 1 distinct value against a reference p05 of 7.1


def test_window_with_no_valid_samples_is_not_assessed(tmp_path):
    """Nothing observed is not evidence of a frozen tag."""
    base = pd.Timestamp("2024-03-01 00:00:00+00:00")
    stamps = [base + pd.Timedelta(30 * i, unit="s") for i in range(20)]
    meta = make_meta(source_id="p", point_id="EMPTY.PV", sample_rate_s=30.0)
    path = write_archive(
        tmp_path / "p" / "EMPTY.PV.parquet",
        good_frame(stamps, [float(i) for i in range(20)]),
        meta,
    )

    verdict = profile(
        path, "2030-01-01T00:00:00Z/2030-01-02T00:00:00Z", flatline=True
    ).flatline

    assert verdict is not None
    assert not verdict.fired
    assert verdict.not_assessed == "no valid samples in window"
    assert verdict.summary_lines() == [
        "flatline verdict: NOT ASSESSED (no valid samples in window)"
    ]


def test_empty_window_does_not_fire_on_a_distinct_count_of_zero(tmp_path):
    """Reference p05 above zero would otherwise make an empty window a flatline."""
    base = pd.Timestamp("2024-03-01 00:00:00+00:00")
    stamps = [base + pd.Timedelta(30 * i, unit="s") for i in range(20)]
    meta = make_meta(source_id="p", point_id="GONE.PV", sample_rate_s=30.0)
    path = write_archive(
        tmp_path / "p" / "GONE.PV.parquet",
        good_frame(stamps, [float(i) for i in range(20)]),
        meta,
    )
    window = SingleFileStore(path).read_window(
        meta.identity,
        dt.datetime(2030, 1, 1, tzinfo=UTC),
        dt.datetime(2030, 1, 2, tzinfo=UTC),
        stepped_contract(),
    )

    verdict = assess_flatline(
        window,
        reference_change_intervals_s=[30.0] * 10,
        reference_distinct_counts=[5, 6, 7],
    )

    assert not verdict.fired
    assert verdict.not_assessed == "no valid samples in window"


def _one_second_archive(tmp_path, point_id, values):
    """400 samples at 1 s from 2024-01-01, so reference windows are 100 s wide."""
    base = pd.Timestamp("2024-01-01 00:00:00+00:00")
    stamps = [base + pd.Timedelta(i, unit="s") for i in range(len(values))]
    meta = make_meta(source_id="p", point_id=point_id)
    return write_archive(tmp_path / point_id / f"{point_id}.parquet",
                         good_frame(stamps, values), meta)


# Reference windows for these three: span 100 s, assessed window at 300 s,
# so the walk lays down [0,100) and [100,200).
_REF_START = pd.Timestamp("2024-01-01 00:05:00+00:00")
_REF_END = pd.Timestamp("2024-01-01 00:06:40+00:00")


def test_change_intervals_are_between_events_not_sample_spacing(tmp_path):
    """A tag that moves at every 1 s sample has a 1 s interval."""
    path = _one_second_archive(tmp_path, "EVERY", [float(i) for i in range(400)])

    intervals, _ = reference_history(path, _REF_START, _REF_END)

    assert intervals
    assert set(intervals) == {1.0}


def test_a_tag_moving_every_tenth_sample_reports_ten_seconds(tmp_path):
    """The spacing is still 1 s; the time between changes is 10 s.

    This is the pair that separates the two quantities: reading the
    spacing would call this tag a 1 s tag and let signal 1 fire on any
    two-sample hold.
    """
    path = _one_second_archive(tmp_path, "TENTH", [float(i // 10) for i in range(400)])

    intervals, _ = reference_history(path, _REF_START, _REF_END)

    assert intervals
    assert set(intervals) == {10.0}


def test_mixed_change_rates_pool_into_the_percentile(tmp_path):
    """Two regimes in the reference span give a percentile, not one value."""
    values = [float(i) if i < 100 else float(i // 20) for i in range(400)]
    path = _one_second_archive(tmp_path, "MIXED", values)

    intervals, _ = reference_history(path, _REF_START, _REF_END)

    # Window [0,100) moves every sample: 98 intervals of 1 s. Window
    # [100,200) moves every 20th: 3 intervals of 20 s.
    assert sorted(Counter(intervals).items()) == [(1.0, 98), (20.0, 3)]
    assert float(np.quantile(intervals, 0.99)) == 20.0

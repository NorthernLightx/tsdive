"""Changepoint segmentation: the PELT-L2 solver and the window wrapper."""

from __future__ import annotations

import math
import time
from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from conftest import EngRange, Role, TagIdentity, make_meta, stepped_contract
from tsdive import analyses
from tsdive.changepoints import (
    METHOD,
    default_penalty,
    pelt_l2,
    segment_window,
)
from tsdive.errors import InsufficientQuality, RegimeTooSparse
from tsdive.store.tagstore import SingleFileStore, Window, meta_from_parquet

BASE = pd.Timestamp("2024-05-01 00:00:00+00:00")


def _frame(values, *, minutes=1) -> pd.DataFrame:
    """``values`` on a regular grid from ``BASE``, every row GOOD."""
    stamps = [BASE + pd.Timedelta(minutes * i, unit="min") for i in range(len(values))]
    return pd.DataFrame(
        {"timestamp": stamps, "value": values, "quality": ["GOOD"] * len(values)}
    )


def _window(archive_factory, values, *, meta=None, minutes=1) -> Window:
    """Write ``values`` on a regular grid and read the whole span back."""
    meta = meta or make_meta(sample_rate_s=60.0 * minutes)
    frame = _frame(values, minutes=minutes)
    store = SingleFileStore(archive_factory(frame, meta))
    return store.read_window(
        meta.identity,
        frame["timestamp"].iloc[0].to_pydatetime(),
        frame["timestamp"].iloc[-1].to_pydatetime(),
        stepped_contract(),
    )


def _two_levels(n_first: int, n_second: int) -> list[float]:
    """Level 50 for ``n_first`` samples, then level 60, each with a 0.1 ripple."""
    return [50.0 + 0.1 * (i % 3) for i in range(n_first)] + [
        60.0 + 0.1 * (i % 3) for i in range(n_second)
    ]


def test_pelt_finds_a_single_step_where_it_happens():
    rng = np.random.default_rng(5)
    values = np.concatenate([rng.normal(0.0, 1.0, 200), rng.normal(6.0, 1.0, 200)])
    found = pelt_l2(values, penalty=default_penalty(len(values)), min_size=10)
    assert len(found) == 1
    assert abs(found[0] - 200) <= 2


def test_pelt_finds_nothing_in_a_series_that_never_steps():
    rng = np.random.default_rng(5)
    assert pelt_l2(rng.normal(0.0, 1.0, 400), penalty=default_penalty(400), min_size=10) == []


def test_pelt_never_cuts_a_segment_shorter_than_min_size():
    rng = np.random.default_rng(7)
    values = np.concatenate([rng.normal(m, 0.5, 25) for m in (0.0, 8.0) * 3])
    n = len(values)
    found = pelt_l2(values, penalty=2.0, min_size=40)
    assert found  # a penalty this small does cut somewhere
    edges = [0, *found, n]
    assert all(hi - lo >= 40 for lo, hi in pairwise(edges))


def test_pelt_refuses_a_min_size_below_one():
    with pytest.raises(ValueError, match="min_size"):
        pelt_l2(np.arange(50, dtype=float), penalty=10.0, min_size=0)


def test_pelt_stays_inside_its_time_budget():
    rng = np.random.default_rng(1)
    values = np.concatenate([rng.normal(0.0, 1.0, 1800), rng.normal(2.0, 1.0, 1800)])
    started = time.perf_counter()
    found = pelt_l2(values, penalty=default_penalty(len(values)), min_size=10)
    elapsed = time.perf_counter() - started
    assert len(found) == 1
    assert elapsed < 5.0, f"n=3600 took {elapsed:.2f}s"


def test_segment_window_reports_real_timestamps_over_a_gap(archive_factory):
    # A step at position 60, with the samples either side an hour apart
    # and no sample at all for the ten hours before the step: the cost
    # counts positions, the table states when they happened.
    rng = np.random.default_rng(3)
    values = [50.0 + float(rng.normal(0, 0.2)) for _ in range(60)]
    values += [70.0 + float(rng.normal(0, 0.2)) for _ in range(60)]
    window = _window(archive_factory, values, minutes=60)
    found = segment_window(window)
    assert found.method == METHOD
    assert found.n_used == 120
    assert found.penalty == pytest.approx(default_penalty(120))
    assert len(found.breakpoints) == 1
    assert found.breakpoints[0] == BASE + pd.Timedelta(60, unit="h")
    first, second = found.segments
    assert (first.n, second.n) == (60, 60)
    assert first.median == pytest.approx(50.0, abs=0.2)
    assert second.median == pytest.approx(70.0, abs=0.2)
    assert first.start == BASE
    assert second.end == BASE + pd.Timedelta(119, unit="h")


def test_segment_window_refuses_a_frozen_tag(archive_factory):
    window = _window(archive_factory, [50.0] * 100)
    with pytest.raises(InsufficientQuality, match="no spread to segment"):
        segment_window(window)


def test_segment_window_refuses_a_window_too_short_for_one_break(archive_factory):
    rng = np.random.default_rng(4)
    window = _window(archive_factory, [50.0 + float(rng.normal(0, 1)) for _ in range(19)])
    with pytest.raises(InsufficientQuality, match="19 usable samples"):
        segment_window(window, min_size=10)
    assert segment_window(window, min_size=5).n_used == 19


def test_segment_window_refuses_a_state_valued_tag(archive_factory):
    window = _window(
        archive_factory,
        ["R0"] * 50 + ["R1"] * 50,
        meta=make_meta(point_id="FIC101.MODE", role=Role.MODE, sample_rate_s=60.0),
    )
    with pytest.raises(ValueError, match="state-valued"):
        segment_window(window)


def test_segment_window_counts_only_usable_rows(archive_factory):
    """A BAD row is not a segment boundary, and not a sample either."""
    rng = np.random.default_rng(9)
    values = [50.0 + float(rng.normal(0, 0.2)) for _ in range(40)]
    values += [70.0 + float(rng.normal(0, 0.2)) for _ in range(40)]
    stamps = [BASE + pd.Timedelta(i, unit="min") for i in range(80)]
    qualities = ["GOOD"] * 80
    qualities[39] = "BAD"
    meta = make_meta(sample_rate_s=60.0)
    frame = pd.DataFrame({"timestamp": stamps, "value": values, "quality": qualities})
    store = SingleFileStore(archive_factory(frame, meta))
    window = store.read_window(
        meta.identity,
        stamps[0].to_pydatetime(),
        stamps[-1].to_pydatetime(),
        stepped_contract(),
    )
    found = segment_window(window, min_size=10)
    assert found.n_used == 79
    assert len(found.breakpoints) == 1
    assert found.breakpoints[0] == stamps[40]


def test_segment_window_splits_a_saturation_into_its_own_segment(archive_factory):
    rng = np.random.default_rng(11)
    values = [62.0 + float(rng.normal(0, 0.2)) for _ in range(60)]
    values += [100.0] * 30
    values += [62.0 + float(rng.normal(0, 0.2)) for _ in range(60)]
    window = _window(
        archive_factory,
        values,
        meta=make_meta(sample_rate_s=60.0, eng_range=EngRange(zero=0.0, span=100.0)),
    )
    found = segment_window(window)
    assert [s.n for s in found.segments] == [60, 30, 60]
    assert found.segments[1].median == pytest.approx(100.0)
    assert found.segments[1].mad == 0.0
    # Segments are not baselines, so this is reported, never refused.
    assert window.physics.clipping.censored is True


def test_default_penalty_is_bic_shaped():
    assert default_penalty(561) == pytest.approx(3.0 * math.log(561))


# --- write_mode_archive ------------------------------------------------------

DEMO_BASELINE = "2024-03-30T20:00:00Z/2024-03-31T01:00:00Z"
DEMO_WINDOW = "2024-03-31T01:00:00Z/2024-03-31T06:00:00Z"


def test_write_mode_archive_labels_every_row_of_the_demo_window(demo_archives, tmp_path):
    flow, _ = demo_archives
    analysis = analyses.segment(flow)
    out = analysis.write_mode_archive(tmp_path / "fic101_seg.parquet")
    written = pd.read_parquet(out)
    assert len(written) == len(analysis.window.frame)
    assert list(dict.fromkeys(written["value"])) == ["S1", "S2", "S3", "S4", "S5", "S6"]
    assert set(written["quality"]) == {"GOOD"}
    # A segment's first and last sample carry its own number, so the label
    # changes exactly where the segment table says it does.
    by_stamp = written.set_index("timestamp")["value"]
    for i, seg in enumerate(analysis.found.segments, start=1):
        assert by_stamp[seg.start] == f"S{i}"
        assert by_stamp[seg.end] == f"S{i}"
    meta = meta_from_parquet(out)
    assert meta.identity == TagIdentity("demo", "FIC101.PV.SEG")
    assert meta.name == f"{analysis.window.meta.name} segments"
    assert meta.role is Role.MODE
    assert meta.quality_assumed is True
    assert meta.sample_rate_s == analysis.window.meta.sample_rate_s
    assert (meta.unit_raw, meta.eng_range, meta.asset, meta.loop_id) == (None,) * 4


def test_screen_mode_refuses_a_segment_the_baseline_never_saw(demo_archives, tmp_path):
    flow, _ = demo_archives
    modes = analyses.segment(flow).write_mode_archive(tmp_path / "fic101_seg.parquet")
    # Segment 4, the half hour at full scale from 02:01, lies inside the
    # monitored window only, so no baseline exists for its label.
    with pytest.raises(RegimeTooSparse, match="no baseline for unseen regime 'S4'"):
        analyses.screen(flow, DEMO_BASELINE, DEMO_WINDOW, mode=modes)


def test_screen_mode_grades_each_segment_against_its_own_baseline(archive_factory, tmp_path):
    # The step comes 90 minutes in and the baseline ends 30 minutes after
    # it, so both segments get a baseline and the monitored window, all
    # second level, is graded against S2 alone.
    values = _two_levels(90, 150)
    values[200] = 61.0
    path = archive_factory(_frame(values), make_meta(sample_rate_s=60.0))
    modes = analyses.segment(path).write_mode_archive(tmp_path / "seg.parquet")
    result = analyses.screen(
        path,
        "2024-05-01T00:00:00Z/2024-05-01T02:00:00Z",
        "2024-05-01T02:00:00Z/2024-05-01T03:59:00Z",
        mode=modes,
    )
    assert result.regimes is not None and sorted(result.regimes) == ["S1", "S2"]
    assert result.regimes["S1"].center == pytest.approx(50.1)
    assert result.regimes["S2"].center == pytest.approx(60.1)
    assert result.result.n_screened == 120
    assert result.frame["timestamp"].tolist() == [BASE + pd.Timedelta(200, unit="min")]


def test_write_mode_archive_refuses_an_existing_path(archive_factory, tmp_path):
    path = archive_factory(_frame(_two_levels(60, 60)), make_meta(sample_rate_s=60.0))
    analysis = analyses.segment(path)
    out = analysis.write_mode_archive(tmp_path / "seg.parquet")
    with pytest.raises(FileExistsError):
        analysis.write_mode_archive(out)
    assert analysis.write_mode_archive(out, overwrite=True) == out

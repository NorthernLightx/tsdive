"""Changepoint segmentation: the PELT-L2 solver and the window wrapper."""

from __future__ import annotations

import math
import time
from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from conftest import EngRange, Role, make_meta, stepped_contract
from tsdive.changepoints import (
    METHOD,
    default_penalty,
    pelt_l2,
    segment_window,
)
from tsdive.errors import InsufficientQuality
from tsdive.store.tagstore import SingleFileStore, Window

BASE = pd.Timestamp("2024-05-01 00:00:00+00:00")


def _window(archive_factory, values, *, meta=None, minutes=1) -> Window:
    """Write ``values`` on a regular grid and read the whole span back."""
    meta = meta or make_meta(sample_rate_s=60.0 * minutes)
    stamps = [BASE + pd.Timedelta(minutes * i, unit="min") for i in range(len(values))]
    frame = pd.DataFrame(
        {"timestamp": stamps, "value": values, "quality": ["GOOD"] * len(values)}
    )
    store = SingleFileStore(archive_factory(frame, meta))
    return store.read_window(
        meta.identity,
        stamps[0].to_pydatetime(),
        stamps[-1].to_pydatetime(),
        stepped_contract(),
    )


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

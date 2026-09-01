"""Clipping: range-based and state-based censoring; baseline refusal."""

from __future__ import annotations

import pandas as pd
import pytest

from conftest import make_meta
from tsdive.errors import InsufficientQuality
from tsdive.store.clipping import assert_usable_baseline, clipped_fraction
from tsdive.store.identity import EngRange
from tsdive.store.quality import Severity, annotate_severity


def _series(values, qualities):
    stamps = [pd.Timestamp(f"2024-03-01 00:00:{i:02d}+00:00") for i in range(len(values))]
    df = pd.DataFrame({"timestamp": stamps, "value": values, "quality": qualities})
    return annotate_severity(df)


def test_range_boundary_counts_as_clipped():
    rng = EngRange(zero=0.0, span=100.0)
    frame = _series([50.0, 99.9999999, 100.0], ["GOOD", "GOOD", "GOOD"])
    report = clipped_fraction(frame["value"], frame["quality"], rng)
    # 99.9999999 is within 1e-9*span of full scale:
    assert report.n_clipped == 1 or report.fraction == pytest.approx(2 / 3)
    assert report.censored


def test_over_range_state_clips_even_without_range():
    frame = _series([257.0, 12.0], [257, "GOOD"])
    report = clipped_fraction(frame["value"], frame["quality"], None)
    assert report.censored
    assert report.n_clipped == 1
    # No eng range: fraction stays null, never a placeholder number.
    assert report.fraction is None


def test_under_range_state():
    frame = _series([-0.001, 12.0], ["UNDER RANGE", "GOOD"])
    report = clipped_fraction(frame["value"], frame["quality"], EngRange(zero=0.0, span=100.0))
    assert report.censored


def test_healthy_window_not_censored():
    rng = EngRange(zero=0.0, span=100.0)
    frame = _series([10.0, 20.0, 30.0], ["GOOD"] * 3)
    report = clipped_fraction(frame["value"], frame["quality"], rng)
    assert not report.censored
    assert report.fraction == 0.0


def test_censored_window_never_serves_as_baseline():
    meta = make_meta()
    frame = _series([100.0] * 4, ["GOOD"] * 4)
    report = clipped_fraction(
        frame["value"], frame["quality"], EngRange(zero=0.0, span=100.0)
    )
    with pytest.raises(InsufficientQuality) as err:
        assert_usable_baseline(meta, report)
    assert "censored" in str(err.value)


def test_baseline_refusal_rounds_the_clipped_fraction():
    meta = make_meta()
    frame = _series([100.0, 50.0, 50.0], ["GOOD"] * 3)
    report = clipped_fraction(
        frame["value"], frame["quality"], EngRange(zero=0.0, span=100.0)
    )
    assert report.fraction == pytest.approx(1 / 3)
    with pytest.raises(InsufficientQuality) as err:
        assert_usable_baseline(meta, report)
    assert "clipped fraction 0.333)" in str(err.value)


def test_baseline_refusal_states_an_absent_fraction_as_null():
    meta = make_meta()
    frame = _series([257.0, 12.0], [257, "GOOD"])
    report = clipped_fraction(frame["value"], frame["quality"], None)
    with pytest.raises(InsufficientQuality) as err:
        assert_usable_baseline(meta, report)
    assert "clipped fraction null (eng range unknown)" in str(err.value)


def test_severity_floor_helper():
    counts = {Severity.GOOD: 2, Severity.BAD: 1}
    from tsdive.store.clipping import severity_floor_ok

    assert severity_floor_ok(counts, 2)
    assert not severity_floor_ok(counts, 3)

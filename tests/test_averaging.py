"""Fixture (a): 59-min-at-50 / 1-min-at-100 under two calculation bases."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from conftest import (
    annotate,
    event_contract,
    fixture_a_frame,
    make_meta,
    stepped_contract,
    write_archive,
)
from tsdive.errors import IncomparableSamplingError
from tsdive.store.averaging import weighted_mean
from tsdive.store.tagstore import TagStore, assert_windows_comparable

START = "2024-03-01 00:00:00+00:00"
END = "2024-03-01 01:00:00+00:00"


def _windows(tmp_path: Path):
    meta_a = make_meta(source_id="plant1", point_id="A.PV")
    meta_b = make_meta(source_id="plant1", point_id="B.PV")
    path_a = write_archive(tmp_path / "plant1" / "A.PV.parquet", fixture_a_frame(), meta_a)
    path_b = write_archive(tmp_path / "plant1" / "B.PV.parquet", fixture_a_frame(), meta_b)
    store = TagStore(tmp_path)
    start = START if isinstance(START, str) else str(START)
    w_time = store.read_window(
        meta_a.identity,
        pd_ts(start),
        pd_ts(END),
        stepped_contract(),
    )
    w_event = store.read_window(
        meta_b.identity,
        pd_ts(start),
        pd_ts(END),
        event_contract(),
    )
    del path_a, path_b
    return w_time, w_event


def pd_ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def test_time_weighted_is_50_83():
    frame = annotate(fixture_a_frame())
    tw = weighted_mean(frame, stepped_contract())
    assert tw == pytest.approx(50.8333, abs=0.001)


def test_event_weighted_is_95_45():
    frame = annotate(fixture_a_frame())
    ew = weighted_mean(frame, event_contract())
    assert ew == pytest.approx(95.4545, abs=0.001)


def test_store_read_honors_contract(tmp_path: Path):
    w_time, w_event = _windows(tmp_path)
    tw = weighted_mean(w_time.frame, stepped_contract())
    ew = weighted_mean(w_event.frame, event_contract())
    assert tw == pytest.approx(50.8333, abs=0.001)
    assert ew == pytest.approx(95.4545, abs=0.001)


def test_mixing_contracts_refused(tmp_path: Path):
    w_time, w_event = _windows(tmp_path)
    with pytest.raises(IncomparableSamplingError) as err:
        assert_windows_comparable(w_time, w_event)
    msg = str(err.value)
    assert "TIME_WEIGHTED" in msg and "EVENT_WEIGHTED" in msg


def test_same_basis_windows_comparable(tmp_path: Path):
    w_time, _ = _windows(tmp_path)
    assert_windows_comparable(w_time, w_time)

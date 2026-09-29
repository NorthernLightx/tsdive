"""An archive stored at micro- or millisecond resolution reads as the nanosecond one.

pandas keeps the parquet column's unit, so an archive written from a
``datetime64[us, UTC]`` frame reads back in microseconds. Parquet has
no second unit. Each command below runs on the demo archives at every
resolution and must print the same document as on the nanosecond
archives.
"""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import make_meta
from tsdive import write_tag
from tsdive.cli import main
from tsdive.compare import compare
from tsdive.demo import PROFILE_DIR, demo_archives
from tsdive.store.averaging import time_weighted_mean

BASELINE = "2024-03-30T20:00:00Z/2024-03-31T01:00:00Z"
WINDOW = "2024-03-31T01:00:00Z/2024-03-31T06:00:00Z"
BEFORE = "2024-03-30T20:00:00Z/2024-03-30T23:00:00Z"
AFTER = "2024-03-31T04:00:00Z/2024-03-31T06:00:00Z"

UNITS = ("ns", "us", "ms")


@pytest.fixture(scope="module")
def archives(tmp_path_factory) -> dict[str, tuple[str, str]]:
    """The two demo profile archives at each resolution: unit -> (FIC-101, TIC-101)."""
    root = tmp_path_factory.mktemp("units")
    out: dict[str, tuple[str, str]] = {}
    for unit in UNITS:
        paths = []
        for name in ("fic101_demo", "tic101_demo"):
            frame, meta = demo_archives()[f"{PROFILE_DIR}/{name}.parquet"]
            frame = frame.assign(timestamp=frame["timestamp"].dt.as_unit(unit))
            path = write_tag(root / unit / f"{name}.parquet", frame, meta)
            assert pd.read_parquet(path)["timestamp"].dt.unit == unit
            paths.append(path.as_posix())
        out[unit] = (paths[0], paths[1])
    return out


def _commands(fic: str, tic: str) -> list[list[str]]:
    return [
        ["profile", fic, "--flatline", "--tz", "Europe/London"],
        ["profile", tic, "--window", AFTER, "--flatline"],
        ["segment", fic],
        ["screen", fic, "--baseline", BASELINE, "--window", WINDOW],
        ["spc", fic, "--baseline", BASELINE, "--window", WINDOW],
        ["mspc", fic, tic, "--baseline", BEFORE, "--window", AFTER],
        ["compare", fic, tic, "--before", BEFORE, "--after", AFTER],
    ]


def _run(argv: list[str], directory: str) -> dict:
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
        assert main(["--json", *argv]) == 0
    return json.loads(out.getvalue().replace(directory, "<dir>"))


@pytest.mark.parametrize("unit", [u for u in UNITS if u != "ns"])
def test_every_command_prints_the_nanosecond_document(archives, unit):
    ns_dir = str(Path(archives["ns"][0]).parent.as_posix())
    unit_dir = str(Path(archives[unit][0]).parent.as_posix())
    for ns_argv, unit_argv in zip(
        _commands(*archives["ns"]), _commands(*archives[unit]), strict=True
    ):
        assert _run(unit_argv, unit_dir) == _run(ns_argv, ns_dir), ns_argv[0]


def test_flatline_reads_the_reference_history_at_every_resolution(archives):
    for unit in UNITS:
        argv = ["profile", archives[unit][1], "--window", AFTER, "--flatline"]
        doc = _run(argv, str(Path(archives[unit][1]).parent.as_posix()))
        signals = {s["signal"]: s for s in doc["flatline"]["signals"]}
        stall = signals["time_since_last_actual_change"]
        assert stall["evaluable"], unit
        assert stall["reference"] == pytest.approx(60.0), unit
        assert signals["distinct_value_count_vs_p05"]["reference"] == pytest.approx(84.9), unit


@pytest.mark.parametrize("unit", [u for u in UNITS if u != "ns"])
def test_report_html_draws_the_nanosecond_figure(archives, unit, tmp_path):
    pages = {}
    for u in ("ns", unit):
        out = tmp_path / f"{u}.html"
        with contextlib.redirect_stdout(io.StringIO()):
            assert main(["report-html", *archives[u], "--window", WINDOW, "-o", str(out)]) == 0
        directory = str(Path(archives[u][0]).parent.as_posix())
        pages[u] = out.read_text(encoding="utf-8").replace(directory, "<dir>")
    assert pages[unit] == pages["ns"]


@pytest.mark.parametrize("unit", ["ns", "us", "ms", "s"])
def test_time_weighted_mean_holds_the_last_sample_to_the_window_end(unit):
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-03-01", periods=2, freq="min", tz="UTC"),
            "value": [1.0, 3.0],
            "quality": ["GOOD", "GOOD"],
            "severity": ["GOOD", "GOOD"],
            "valid": [True, True],
        }
    )
    frame["timestamp"] = frame["timestamp"].dt.as_unit(unit)
    end = pd.Timestamp("2024-03-01 00:03", tz="UTC").as_unit(unit)
    # 1.0 holds for 60 s, 3.0 for 120 s.
    assert time_weighted_mean(frame, window_end=end) == pytest.approx(7.0 / 3.0)


@pytest.mark.parametrize("unit", UNITS)
def test_compare_reads_the_before_change_intervals_in_seconds(unit, tmp_path):
    # Both tags change every 60 s sample. The last after-period sample
    # repeats, a 60 s stall that the before period's own 60 s p99 allows.
    rng = np.random.default_rng(5)
    stamps = pd.date_range("2024-03-01", periods=241, freq="min", tz="UTC").as_unit(unit)
    paths = []
    for point_id in ("FIC101.PV", "TIC101.PV"):
        values = 50.0 + rng.normal(0.0, 1.0, size=len(stamps))
        values[-1] = values[-2]
        frame = pd.DataFrame({"timestamp": stamps, "value": values, "quality": "GOOD"})
        meta = make_meta(point_id=point_id, sample_rate_s=60.0)
        paths.append(str(write_tag(tmp_path / f"{point_id}.parquet", frame, meta)))
    result = compare(
        paths,
        "2024-03-01T00:00:00Z/2024-03-01T02:00:00Z",
        "2024-03-01T02:00:00Z/2024-03-01T04:00:00Z",
    )
    assert [t.quality for t in result.tags] == ["ok", "ok"]

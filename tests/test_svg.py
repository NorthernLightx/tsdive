"""The report figures: one deterministic SVG per window."""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd

from conftest import (
    AggregateType,
    CalculationBasis,
    RetrievalMode,
    SamplingContract,
    TagIdentity,
    TagMeta,
    make_meta,
    write_archive,
)
from tsdive.api import profile
from tsdive.store.tagstore import SingleFileStore, meta_from_parquet, write_tag
from tsdive.ui.svg import window_figure

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
README_PROFILE_WINDOW = "2024-03-30T20:00:00Z/2024-03-31T06:00:00Z"
BASE = pd.Timestamp("2024-03-01 00:00:00+00:00")


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _window(path: Path, start: str, end: str):
    store = SingleFileStore(path)
    contract = SamplingContract(
        calculation_basis=CalculationBasis.TIME_WEIGHTED,
        retrieval_mode=RetrievalMode.RECORDED,
        aggregate_type=AggregateType.NONE,
        stepped=False,
    )
    return store.read_window(
        meta_from_parquet(store.path).identity,
        pd.Timestamp(start).to_pydatetime(),
        pd.Timestamp(end).to_pydatetime(),
        contract,
    )


def _small_archive(tmp_path: Path, meta: TagMeta | None = None) -> Path:
    """Sixty seconds at 1 s, one UNCERTAIN row, one BAD row."""
    values = [50.0 + (i % 7) for i in range(60)]
    qualities = ["GOOD"] * 60
    qualities[20] = "UNCERTAIN"
    qualities[40] = "BAD"
    frame = pd.DataFrame(
        {
            "timestamp": [BASE + pd.Timedelta(i, unit="s") for i in range(60)],
            "value": values,
            "quality": qualities,
        }
    )
    return write_archive(tmp_path / "small.parquet", frame, meta or make_meta())


def _polyline_points(svg: str) -> list[str]:
    match = re.search(r'<polyline class="line" points="([^"]*)"', svg)
    assert match is not None
    return match.group(1).split()


def test_the_same_window_draws_the_same_bytes(tmp_path):
    window = _window(_small_archive(tmp_path), "2024-03-01T00:00:00Z", "2024-03-01T00:01:00Z")
    first = window_figure(window)
    assert first == window_figure(window)
    assert first.startswith("<svg ") and first.endswith("</svg>")
    assert "<script" not in first
    assert 'role="img"' in first
    assert "<title>plant1:FIC101.PV  FIC101.PV display name</title>" in first


def test_demo_fic101_window_matches_the_golden_figure(tmp_path, file_regression):
    make_demo_archive = _load("make_demo_archive")
    flow, meta = make_demo_archive.build_fic101()
    path = write_tag(tmp_path / "fic101_demo.parquet", flow, meta)
    window = profile(path, README_PROFILE_WINDOW).window
    file_regression.check(window_figure(window), extension=".svg")


def test_a_long_window_is_thinned_and_keeps_every_flag(tmp_path):
    n = 100_000
    rng = np.random.default_rng(3)
    values = 50.0 + 5.0 * np.sin(np.arange(n) / 900.0) + rng.normal(0.0, 0.3, n)
    qualities = np.array(["GOOD"] * n, dtype=object)
    qualities[::5000] = "BAD"
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range(BASE, periods=n, freq="1s"),
            "value": values,
            "quality": list(qualities),
        }
    )
    path = write_archive(tmp_path / "long.parquet", frame, make_meta(sample_rate_s=1.0))
    window = _window(path, "2024-03-01T00:00:00Z", "2024-03-02T03:46:40Z")
    assert len(window.frame) == n
    flagged = [BASE + pd.Timedelta(1999 * i + 1, unit="s") for i in range(50)]

    svg = window_figure(window, flagged=flagged)

    assert len(_polyline_points(svg)) <= 2 * 720
    assert len(svg.encode("utf-8")) < 150_000
    assert svg.count('class="flag"') == 50
    assert svg.count('class="below"') >= 1


def test_markers_and_limits_are_counted_by_class(tmp_path):
    window = _window(_small_archive(tmp_path), "2024-03-01T00:00:00Z", "2024-03-01T00:01:00Z")
    flagged = [BASE + pd.Timedelta(i, unit="s") for i in (3, 10, 17)]
    hits = [BASE + pd.Timedelta(i, unit="s") for i in (10, 31)]
    svg = window_figure(window, limits=(53.0, 49.0, 57.0), flagged=flagged, hits=hits)
    assert svg.count('class="flag"') == 3
    assert svg.count('class="hit"') == 2
    assert svg.count('class="below"') == 2
    assert svg.count('class="center"') == 1
    assert svg.count('class="limit"') == 2
    assert "ucl 57</text>" in svg
    assert "lcl 49</text>" in svg
    # a timestamp the window does not hold draws no marker
    assert window_figure(window, hits=[BASE - pd.Timedelta(1, unit="h")]).count('class="hit"') == 0


def test_tag_name_and_unit_are_escaped(tmp_path):
    meta = TagMeta(
        identity=TagIdentity(source_id="plant1", point_id="X<1>"),
        name="A <b> & c",
        unit_raw="m3/h & <u>",
    )
    window = _window(
        _small_archive(tmp_path, meta), "2024-03-01T00:00:00Z", "2024-03-01T00:01:00Z"
    )
    svg = window_figure(window)
    assert "<title>plant1:X&lt;1&gt;  A &lt;b&gt; &amp; c</title>" in svg
    assert "m3/h &amp; &lt;u&gt;</text>" in svg
    assert "<b>" not in svg and "<u>" not in svg

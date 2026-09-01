"""The library entry point and what the package exports."""

from __future__ import annotations

import pandas as pd
import pytest

import tsdive
from conftest import EngRange, Role, make_meta, write_archive
from tsdive.cli import cmd_profile, cmd_report_html
from tsdive.errors import SchemaError
from tsdive.report import fmt_span

WINDOW = "2024-03-01T00:00:00Z/2024-03-01T01:00:00Z"


def _demo_archive(tmp_path):
    base = pd.Timestamp("2024-03-01 00:00:00+00:00")
    stamps = [base + pd.Timedelta(3 * i, unit="s") for i in range(60)]
    stamps += [
        base + pd.Timedelta(20 * 60 + 3 * i, unit="s") for i in range(60)
    ]
    values = [50.0 + (i % 5) for i in range(120)]
    qualities = ["GOOD"] * 118 + [257, "GOOD"]
    df = pd.DataFrame({"timestamp": stamps, "value": values, "quality": qualities})
    meta = make_meta(
        loop_id="FIC101",
        role=Role.PV,
        eng_range=EngRange(zero=0.0, span=100.0),
    )
    return write_archive(tmp_path / "plant1" / "FIC101.PV.parquet", df, meta), meta


def test_profile_returns_objects_not_text(tmp_path):
    path, meta = _demo_archive(tmp_path)
    result = tsdive.profile(path, WINDOW)
    assert result.identity == meta.identity
    assert result.physics.coverage.coverage < 1.0
    assert set(result.frame.columns) >= {"timestamp", "value", "quality", "severity", "valid"}
    assert result.flatline is None


def test_profile_carries_statistics_as_numbers(tmp_path):
    path, _ = _demo_archive(tmp_path)
    stats = tsdive.profile(path, WINDOW).stats
    assert stats.features.n_good == 119
    assert stats.distinct_count == 5
    assert (stats.interval_median_s, stats.declared_rate_s) == (3.0, 3.0)
    assert stats.interval_differs_from_declared is False
    assert stats.state_counts == ()


def test_rendered_profile_equals_cli_output(tmp_path, capsys):
    path, _ = _demo_archive(tmp_path)
    rc = cmd_profile([str(path), "--window", WINDOW])
    cli_out = capsys.readouterr().out
    assert rc == 0
    assert tsdive.profile(path, WINDOW).render() + "\n" == cli_out


def test_profile_accepts_a_single_tz_string(tmp_path):
    path, _ = _demo_archive(tmp_path)
    result = tsdive.profile(path, WINDOW, tz="Europe/London")
    assert result.physics.timestamp_audit.n_samples > 0


def test_profile_refuses_naive_window(tmp_path):
    path, _ = _demo_archive(tmp_path)
    with pytest.raises(ValueError, match="naive"):
        tsdive.profile(path, "2024-03-01T00:00:00/2024-03-01T01:00:00")


def test_profile_flatline_opt_in_carries_a_verdict(tmp_path):
    path, _ = _demo_archive(tmp_path)
    result = tsdive.profile(
        path, "2024-03-01T00:02:00Z/2024-03-01T01:00:00Z", flatline=True
    )
    assert result.flatline is not None
    assert result.render().splitlines()[-2] == "Flatline"


def test_omitted_window_profiles_the_whole_archive(tmp_path, capsys):
    path, _ = _demo_archive(tmp_path)
    first = pd.Timestamp("2024-03-01 00:00:00+00:00")
    last = pd.Timestamp("2024-03-01 00:20:00+00:00") + pd.Timedelta(3 * 59, unit="s")

    result = tsdive.profile(path)
    assert (result.window.start, result.window.end) == (first, last)

    rc = cmd_profile([str(path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert f"window    {fmt_span(first, last)}" in out


def test_report_html_without_window_uses_the_archive_extent(tmp_path, capsys):
    path, _ = _demo_archive(tmp_path)
    out_file = tmp_path / "report.html"
    rc = cmd_report_html([str(path), "-o", str(out_file)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "window    2024-03-01 00:00:00Z -> 00:22:57Z" in out
    assert "<td>profiles rendered</td><td>1</td>" in out_file.read_text(encoding="utf-8")


def test_report_html_without_window_refuses_when_nothing_is_readable(tmp_path, capsys):
    rc = cmd_report_html([str(tmp_path / "missing.parquet"), "-o", str(tmp_path / "r.html")])
    err = capsys.readouterr().err
    assert rc == 2
    assert "pass --window explicitly" in err


def test_empty_archive_has_no_extent(tmp_path):
    empty = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(pd.Series([], dtype="object"), utc=True),
            "value": pd.Series([], dtype="float64"),
            "quality": pd.Series([], dtype="object"),
        }
    )
    path = write_archive(tmp_path / "plant1" / "EMPTY.PV.parquet", empty, make_meta())
    with pytest.raises(SchemaError, match="no samples"):
        tsdive.profile(path)


def test_public_names_are_all_importable():
    for name in tsdive.__all__:
        assert getattr(tsdive, name, None) is not None, name

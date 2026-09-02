"""The library entry point and what the package exports."""

from __future__ import annotations

import json

import pandas as pd
import pytest

import tsdive
from conftest import EngRange, Role, make_meta, write_archive
from tsdive.api import parse_window
from tsdive.cli import cmd_profile, cmd_report_html, main
from tsdive.errors import SchemaError
from tsdive.report import fmt_span

WINDOW = "2024-03-01T00:00:00Z/2024-03-01T01:00:00Z"

# The windows the README transcripts use on the two demo archives.
DEMO_BASELINE = "2024-03-30T20:00:00Z/2024-03-31T01:00:00Z"
DEMO_WINDOW = "2024-03-31T01:00:00Z/2024-03-31T06:00:00Z"
DEMO_BEFORE = "2024-03-30T20:00:00Z/2024-03-30T23:00:00Z"
DEMO_AFTER = "2024-03-31T04:00:00Z/2024-03-31T06:00:00Z"


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


def _cli(capsys, argv: list[str]) -> str:
    rc = main(argv)
    out = capsys.readouterr().out
    assert rc == 0
    return out


def _agrees_with_cli(analysis, cls, argv: list[str], capsys) -> None:
    """render() is the CLI's stdout and to_dict() is its --json payload."""
    assert isinstance(analysis, cls)
    assert analysis.render() + "\n" == _cli(capsys, argv)
    payload = json.loads(json.dumps(analysis.to_dict()))
    assert payload == json.loads(_cli(capsys, [*argv, "--json"]))


def test_segment_agrees_with_the_cli(demo_archives, capsys):
    flow, _ = demo_archives
    a = tsdive.segment(flow)
    _agrees_with_cli(a, tsdive.SegmentAnalysis, ["segment", str(flow)], capsys)
    assert a.to_dict()["tag"] == "demo:FIC101.PV"
    assert len(a.frame) == len(a.found.segments) == 6


def test_screen_agrees_with_the_cli(demo_archives, capsys):
    flow, _ = demo_archives
    a = tsdive.screen(flow, DEMO_BASELINE, DEMO_WINDOW)
    argv = ["screen", str(flow), "--baseline", DEMO_BASELINE, "--window", DEMO_WINDOW]
    _agrees_with_cli(a, tsdive.ScreenAnalysis, argv, capsys)
    assert a.to_dict()["n_flagged"] == 29
    assert list(a.frame["timestamp"]) == sorted(a.result.flagged_timestamps)


def test_spc_agrees_with_the_cli(demo_archives, capsys):
    flow, _ = demo_archives
    a = tsdive.spc(flow, DEMO_BASELINE, DEMO_WINDOW)
    argv = ["spc", str(flow), "--baseline", DEMO_BASELINE, "--window", DEMO_WINDOW]
    _agrees_with_cli(a, tsdive.SpcAnalysis, argv, capsys)
    assert len(a.frame) == len(a.hits) == a.to_dict()["n_hits"] == 37


def test_mspc_agrees_with_the_cli(demo_archives, capsys):
    flow, temp = demo_archives
    a = tsdive.mspc([flow, temp], DEMO_BEFORE, DEMO_AFTER)
    argv = [
        "mspc", str(flow), str(temp), "--baseline", DEMO_BEFORE, "--window", DEMO_AFTER
    ]
    _agrees_with_cli(a, tsdive.MspcAnalysis, argv, capsys)
    assert a.tags == ["demo:FIC101.PV", "demo:TIC101.PV"]
    assert int(a.frame["spe_breach"].sum()) == len(a.found.spe_breaches) == 108
    assert len(a.frame) == len(a.test.index) == 121


def test_compare_agrees_with_the_cli(demo_archives, capsys):
    flow, temp = demo_archives
    a = tsdive.compare([flow, temp], DEMO_BEFORE, DEMO_AFTER)
    argv = [
        "compare", str(flow), str(temp), "--before", DEMO_BEFORE, "--after", DEMO_AFTER
    ]
    _agrees_with_cli(a, tsdive.CompareAnalysis, argv, capsys)
    assert a.n_refused == 0
    assert list(a.frame["tag"]) == [c.tag for c in a.tags]


def test_analysis_refusals_raise_what_the_cli_reports(tmp_path, capsys):
    path, _ = _demo_archive(tmp_path)
    with pytest.raises(ValueError, match="baseline and window overlap"):
        tsdive.spc(path, WINDOW, WINDOW)
    rc = main(["spc", str(path), "--baseline", WINDOW, "--window", WINDOW])
    assert rc == 2
    assert capsys.readouterr().err == "error: baseline and window overlap\n"

    # A baseline pinned at the top of the range is censored and refused
    # with the same typed error on both paths.
    base = pd.Timestamp("2024-03-01 00:00:00+00:00")
    stamps = [base + pd.Timedelta(60 * i, unit="s") for i in range(120)]
    values = [100.0] * 60 + [50.0 + (i % 5) for i in range(60)]
    frame = pd.DataFrame({"timestamp": stamps, "value": values, "quality": ["GOOD"] * 120})
    meta = make_meta(loop_id="FIC102", role=Role.PV, eng_range=EngRange(zero=0.0, span=100.0))
    pinned = write_archive(tmp_path / "plant1" / "FIC102.PV.parquet", frame, meta)
    baseline = "2024-03-01T00:00:00Z/2024-03-01T01:00:00Z"
    window = "2024-03-01T01:00:00Z/2024-03-01T02:00:00Z"
    with pytest.raises(tsdive.InsufficientQuality):
        tsdive.screen(pinned, baseline, window)
    rc = main(["screen", str(pinned), "--baseline", baseline, "--window", window])
    assert rc == 2
    assert capsys.readouterr().err.startswith("[InsufficientQuality]")


def test_parse_window_reads_every_iso_8601_interval_form():
    explicit = parse_window("2024-03-31T01:00:00Z/2024-03-31T06:00:00Z")
    assert explicit == (
        pd.Timestamp("2024-03-31T01:00:00Z"),
        pd.Timestamp("2024-03-31T06:00:00Z"),
    )
    assert parse_window("2024-03-31T01:00:00Z/PT5H") == explicit
    assert parse_window("PT5H/2024-03-31T06:00:00Z") == explicit
    # A bare date is the whole UTC day, bounds read the same way.
    assert parse_window("2024-03-31") == parse_window(
        "2024-03-31T00:00:00Z/2024-04-01T00:00:00Z"
    )
    assert parse_window("2024-03-31/PT5H") == (
        pd.Timestamp("2024-03-31T00:00:00Z"),
        pd.Timestamp("2024-03-31T05:00:00Z"),
    )
    assert parse_window("P1DT12H/2024-03-31") == (
        pd.Timestamp("2024-03-29T12:00:00Z"),
        pd.Timestamp("2024-03-31T00:00:00Z"),
    )


def test_parse_window_still_refuses_a_naive_bound():
    with pytest.raises(ValueError) as caught:
        parse_window("2024-03-31T01:00:00/2024-03-31T06:00:00Z")
    assert str(caught.value) == (
        "window START is naive (2024-03-31 01:00:00); append 'Z' or '+00:00' - "
        "tsdive stores UTC only"
    )


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        # pandas reads P1M as one minute, so a calendar duration is refused
        # rather than silently sized.
        ("2024-03-31T01:00:00Z/P1M", "not days, hours, minutes and seconds"),
        ("2024-03-31T01:00:00Z/P1Y", "not days, hours, minutes and seconds"),
        ("2024-03-31T01:00:00Z/PT5X", "not days, hours, minutes and seconds"),
        ("PT5H/PT1H", "window must be <START>/<END>"),
        ("2024-03-31T06:00:00Z/2024-03-31T01:00:00Z", "window END is not after START"),
        ("2024-03-31T01:00:00Z/2024-03-31T01:00:00Z", "window END is not after START"),
        ("last tuesday", "window must be <START>/<END>"),
    ],
)
def test_parse_window_refuses_what_it_cannot_size(spec, message):
    with pytest.raises(ValueError, match=message):
        parse_window(spec)

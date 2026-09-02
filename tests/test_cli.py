"""CLI end-to-end on a synthetic archive."""

from __future__ import annotations

import datetime as dt
import json
import math

import numpy as np
import pandas as pd
import pytest

import tsdive
from conftest import EngRange, Role, make_meta, write_archive
from tsdive.cli import (
    _parser_compare,
    _parser_mspc,
    _parser_profile,
    _parser_screen,
    _parser_segment,
    _parser_spc,
    cmd_compare,
    cmd_mspc,
    cmd_profile,
    cmd_report_html,
    cmd_screen,
    cmd_segment,
    cmd_spc,
    main,
    run_compare,
    run_mspc,
    run_profile,
    run_screen,
    run_segment,
    run_spc,
)

UTC = dt.UTC

WINDOW = "2024-03-01T00:00:00Z/2024-03-01T01:00:00Z"
BASELINE_SPAN = "2024-03-01T00:00:00Z/2024-03-01T00:59:00Z"
TOUCHING_BASELINE = "2024-03-01T00:00:00Z/2024-03-01T01:00:00Z"
MONITOR_SPAN = "2024-03-01T01:00:00Z/2024-03-01T02:00:00Z"


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


def _every_minute(tmp_path, values, qualities, *, meta, name="STATS.PV"):
    base = pd.Timestamp("2024-03-01 00:00:00+00:00")
    df = pd.DataFrame(
        {
            "timestamp": [base + pd.Timedelta(60 * i, unit="s") for i in range(len(values))],
            "value": values,
            "quality": qualities,
        }
    )
    return write_archive(tmp_path / "plant1" / f"{name}.parquet", df, meta)


def _stats_block(out: str) -> list[str]:
    """The Values section, from its header to the interval line."""
    lines = out.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("Values"))
    end = next(i for i, ln in enumerate(lines) if ln.startswith("  interval "))
    return lines[start : end + 1]


def test_version_flag_reports_package_version(capsys):
    rc = main(["--version"])
    assert rc == 0
    assert capsys.readouterr().out.strip() == f"tsdive {tsdive.__version__}"


def test_help_and_unknown_command_list_every_command(capsys):
    commands = (
        "ingest",
        "profile",
        "segment",
        "screen",
        "spc",
        "mspc",
        "compare",
        "report-html",
    )
    assert main(["--help"]) == 0
    listed = capsys.readouterr().err
    assert all(f"  {name} " in listed for name in commands)

    assert main(["screeen"]) == 2
    suggested = capsys.readouterr().err
    assert all(f"'{name}'" in suggested for name in commands)


def test_global_flags_are_accepted_before_the_command_name(tmp_path, capsys):
    path, meta = _demo_archive(tmp_path)
    assert main(["--no-color", "segment", str(path)]) == 0
    assert capsys.readouterr().out.startswith(f"{meta.identity}  ")

    assert main(["--json", "segment", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["tag"] == str(meta.identity)


def test_help_names_the_installed_command(capsys):
    """The package ships a console script, so the help spells commands that way."""
    assert main(["--help"]) == 0
    assert "python -m" not in capsys.readouterr().err


def test_profile_prints_physics_report(tmp_path, capsys):
    path, meta = _demo_archive(tmp_path)
    rc = cmd_profile([str(path), "--window", WINDOW])
    out = capsys.readouterr().out
    assert rc == 0
    assert out.startswith(f"{meta.identity}  {meta.name}\n")
    for marker in (
        "coverage 0.098",
        "gaps 2",
        "longest 2223 s",
        "unknown (no rule matched)",
        "GOOD 119   UNCERTAIN 0   BAD 1",
        "clipped 0.0083",
        "audited 120",
        "contract  TIME_WEIGHTED",
    ):
        assert marker in out, f"missing {marker!r}"


def test_profile_lines_fit_in_eighty_columns(tmp_path, capsys):
    path, _ = _demo_archive(tmp_path)
    assert cmd_profile([str(path), "--window", WINDOW, "--flatline"]) == 0
    over = [ln for ln in capsys.readouterr().out.splitlines() if len(ln) >= 80]
    assert over == []


def test_profile_json_carries_the_numbers_the_report_states(tmp_path, capsys):
    path, meta = _demo_archive(tmp_path)
    assert cmd_profile([str(path), "--window", WINDOW, "--json"]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    payload = json.loads(captured.out)  # one object, nothing else
    assert payload["tag"] == str(meta.identity)
    assert payload["window"] == {
        "start": "2024-03-01T00:00:00+00:00",
        "end": "2024-03-01T01:00:00+00:00",
        "duration_s": 3600.0,
    }
    assert payload["coverage"]["n_gaps"] == 2
    assert payload["coverage"]["gaps"][0]["class"] == "unknown"
    assert payload["quality"]["counts"] == {"GOOD": 119, "UNCERTAIN": 0, "BAD": 1}
    assert payload["range"]["censored"] is True
    assert payload["values"]["n_good"] == 119
    assert payload["values"]["median"] == 52.0
    assert payload["flatline"] is None


def test_profile_json_reports_the_flatline_verdict(tmp_path, capsys):
    path, _ = _demo_archive(tmp_path)
    rc = cmd_profile(
        [
            str(path),
            "--window",
            "2024-03-01T00:02:00Z/2024-03-01T01:00:00Z",
            "--flatline",
            "--json",
        ]
    )
    assert rc == 0
    flatline = json.loads(capsys.readouterr().out)["flatline"]
    assert flatline["verdict"] == "NOT ASSESSED"
    assert flatline["saturated_not_frozen"] is True


def test_profile_json_still_refuses_on_stderr(tmp_path, capsys):
    path, _ = _demo_archive(tmp_path)
    rc = cmd_profile([str(path), "--window", "2024-03-01T00:00:00/2024", "--json"])
    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    assert captured.err.startswith("error: ")


def test_profile_refuses_naive_window(tmp_path, capsys):
    path, _ = _demo_archive(tmp_path)
    rc = cmd_profile(
        [str(path), "--window", "2024-03-01T00:00:00/2024-03-01T01:00:00"]
    )
    err = capsys.readouterr().err
    assert rc == 2
    assert "naive" in err
    assert "UTC" in err


def test_profile_dst_audit_flag(tmp_path, capsys):
    base = pd.Timestamp("2024-03-10 06:00:00+00:00")  # spans 08:00Z spring-forward
    stamps = [base + pd.Timedelta(10 * i, unit="min") for i in range(25)]
    df = pd.DataFrame(
        {
            "timestamp": stamps,
            "value": [50.0 + i for i in range(len(stamps))],
            "quality": ["GOOD"] * len(stamps),
        }
    )
    meta = make_meta()
    path = write_archive(tmp_path / "plant1" / "DST.PV.parquet", df, meta)
    rc = cmd_profile(
        [
            str(path),
            "--window",
            "2024-03-10T06:00:00Z/2024-03-10T10:30:00Z",
            "--tz",
            "America/Chicago",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "  DST (America/Chicago) 2024-03-10 08:00:00Z  -6h -> -5h" in out


def test_profile_rejects_unknown_tz(tmp_path, capsys):
    path, _ = _demo_archive(tmp_path)
    rc = cmd_profile([str(path), "--window", WINDOW, "--tz", "Mars/Olympus_Mons"])
    captured = capsys.readouterr()
    assert rc == 2
    assert captured.err.strip().count("\n") == 0
    assert "Mars/Olympus_Mons" in captured.err
    assert "IANA" in captured.err
    assert "Traceback" not in captured.err


def test_profile_accepts_known_tz(tmp_path, capsys):
    path, _ = _demo_archive(tmp_path)
    rc = cmd_profile([str(path), "--window", WINDOW, "--tz", "Europe/London"])
    assert rc == 0
    assert capsys.readouterr().err == ""


def test_report_html_writes_snapshot(tmp_path, capsys):
    path, _ = _demo_archive(tmp_path)
    out = tmp_path / "report.html"
    rc = cmd_report_html([str(path), "--window", WINDOW, "-o", str(out)])
    assert rc == 0
    assert capsys.readouterr().out.splitlines() == [
        f"wrote     {out.as_posix()}",
        "window    2024-03-01 00:00:00Z -> 01:00:00Z",
        "profiles  1   refusals 0",
    ]
    text = out.read_text(encoding="utf-8")
    # the ledger sees the profiles it rendered, so its count matches the table
    assert "<td>profiles rendered</td><td>1</td>" in text
    assert "profiles=1" in text


def test_report_html_draws_one_figure_per_archive(tmp_path, capsys):
    path, _ = _demo_archive(tmp_path)
    other = _every_minute(
        tmp_path,
        [20.0 + (i % 3) for i in range(61)],
        ["GOOD"] * 61,
        meta=make_meta(point_id="TIC101.PV"),
        name="TIC101.PV",
    )
    out = tmp_path / "report.html"
    rc = cmd_report_html([str(path), str(other), "--window", WINDOW, "-o", str(out)])
    assert rc == 0
    assert "profiles  2   refusals 0" in capsys.readouterr().out
    text = out.read_text(encoding="utf-8")
    assert text.count("<svg") == 2
    assert text.count("<h2>Plots</h2>") == 1
    assert text.index("<h2>Plots</h2>") < text.index("<h2>Window profiles</h2>")


def test_report_html_reports_bad_window_without_traceback(tmp_path, capsys):
    path, _ = _demo_archive(tmp_path)
    naive = "2024-03-01T00:00:00/2024-03-01T01:00:00"
    rc = cmd_report_html([str(path), "--window", naive, "-o", str(tmp_path / "r.html")])
    err = capsys.readouterr().err
    assert rc == 2
    assert err.startswith("error: ")
    assert "naive" in err


def test_profile_prints_descriptive_statistics(tmp_path, capsys):
    path = _every_minute(
        tmp_path,
        [50.0 + i for i in range(21)],
        ["GOOD"] * 21,
        meta=make_meta(sample_rate_s=60.0),
    )
    rc = cmd_profile([str(path), "--window", "2024-03-01T00:00:00Z/2024-03-01T00:20:00Z"])
    assert rc == 0
    assert _stats_block(capsys.readouterr().out) == [
        "Values  GOOD n=21",
        "  min 50.00   p05 51.00   median 60.00   p95 69.00   max 70.00",
        "  mean 59.50 (time-weighted)   std 6.205   mad 5.000",
        "  distinct 21   stall 0 s   changes/h 60.00",
        "  interval 60 s (p05 60 s, p95 60 s)   declared 60 s",
    ]


def test_profile_counts_states_for_a_mode_tag(tmp_path, capsys):
    path = _every_minute(
        tmp_path,
        ["R0"] * 6 + ["R1"] * 7,
        ["GOOD"] * 13,
        meta=make_meta(point_id="FIC101.MODE", role=Role.MODE, sample_rate_s=60.0),
        name="STATS.MODE",
    )
    rc = cmd_profile([str(path), "--window", "2024-03-01T00:00:00Z/2024-03-01T00:12:00Z"])
    assert rc == 0
    assert _stats_block(capsys.readouterr().out) == [
        "Values  states  GOOD n=13",
        "  R1 7   R0 6",
        "  distinct 2   stall 360 s   changes/h 5.000",
        "  interval 60 s (p05 60 s, p95 60 s)   declared 60 s",
    ]


def test_profile_refuses_statistics_with_no_good_samples(tmp_path, capsys):
    path = _every_minute(
        tmp_path,
        [50.0] * 5,
        ["BAD"] * 5,
        meta=make_meta(sample_rate_s=None),
    )
    rc = cmd_profile([str(path), "--window", "2024-03-01T00:00:00Z/2024-03-01T00:04:00Z"])
    assert rc == 0
    assert _stats_block(capsys.readouterr().out) == [
        "Values  none GOOD",
        "  interval 60 s (p05 60 s, p95 60 s)   declared none",
    ]


def test_profile_statistics_over_a_single_sample(tmp_path, capsys):
    path = _every_minute(
        tmp_path,
        [62.5, 63.0, 63.5],
        ["GOOD"] * 3,
        meta=make_meta(sample_rate_s=60.0),
    )
    rc = cmd_profile([str(path), "--window", "2024-03-01T00:00:00Z/2024-03-01T00:00:30Z"])
    assert rc == 0
    assert _stats_block(capsys.readouterr().out) == [
        "Values  GOOD n=1",
        "  min 62.50   p05 62.50   median 62.50   p95 62.50   max 62.50",
        "  mean refused: InsufficientQuality   std n/a   mad n/a",
        "  distinct 1   stall n/a   changes/h n/a",
        "  interval n/a (p05 n/a, p95 n/a)   declared 60 s",
    ]


def test_profile_flags_an_interval_unlike_the_declared_rate(tmp_path, capsys):
    path = _every_minute(
        tmp_path,
        [50.0 + i for i in range(21)],
        ["GOOD"] * 21,
        meta=make_meta(sample_rate_s=3.0),
    )
    rc = cmd_profile([str(path), "--window", "2024-03-01T00:00:00Z/2024-03-01T00:20:00Z"])
    assert rc == 0
    assert _stats_block(capsys.readouterr().out)[-1] == (
        "  interval 60 s (p05 60 s, p95 60 s)   declared 3 s   "
        "(differs from declared)"
    )


def _minute_frame(indices, values, qualities=None) -> pd.DataFrame:
    base = pd.Timestamp("2024-03-01 00:00:00+00:00")
    idx = list(indices)
    return pd.DataFrame(
        {
            "timestamp": [base + pd.Timedelta(60 * i, unit="s") for i in idx],
            "value": values,
            "quality": qualities or ["GOOD"] * len(idx),
        }
    )


def _screen_archive(archive_factory, spikes=(), *, eng_range=None):
    """121 minute-spaced samples cycling 50..54, with ``spikes`` substituted in."""
    values = [50.0 + (i % 5) for i in range(121)]
    for position, value in spikes:
        values[position] = value
    return archive_factory(
        _minute_frame(range(121), values),
        make_meta(sample_rate_s=60.0, eng_range=eng_range),
    )


def _regime_archives(archive_factory, spikes=()):
    """A PV stepping between two levels, plus the MODE tag that explains it."""
    values = [(50.0 if i % 60 < 30 else 80.0) + (i % 5) for i in range(120)]
    for position, value in spikes:
        values[position] = value
    pv = archive_factory(
        _minute_frame(range(120), values), make_meta(sample_rate_s=60.0)
    )
    modes = archive_factory(
        _minute_frame(range(120), ["R0" if i % 60 < 30 else "R1" for i in range(120)]),
        make_meta(point_id="FIC101.MODE", role=Role.MODE, sample_rate_s=60.0),
    )
    return pv, modes


def _stepped_archive(archive_factory, **meta_kwargs):
    """120 minute-spaced samples that step from 50 to 60 halfway through."""
    values = [50.0 + 0.1 * (i % 3) for i in range(60)]
    values += [60.0 + 0.1 * (i % 3) for i in range(60)]
    return archive_factory(
        _minute_frame(range(120), values),
        make_meta(sample_rate_s=60.0, **meta_kwargs),
    )


def test_segment_finds_the_step_without_a_mode_tag(archive_factory, capsys):
    path = _stepped_archive(archive_factory)
    rc = cmd_segment([str(path)])
    assert rc == 0
    assert capsys.readouterr().out.splitlines() == [
        "plant1:FIC101.PV  2 segments   1 breakpoint   censored no",
        "",
        "window    2024-03-01 00:00:00Z -> 01:59:00Z   usable 120",
        "method    PELT L2 on MAD-scaled values   penalty 14.36 (3*log n)   "
        "min-size 10",
        "",
        "  #  start                 end                     n   median     mad",
        "  1  2024-03-01 00:00:00Z  2024-03-01 00:59:00Z   60    50.10  0.1000",
        "  2  2024-03-01 01:00:00Z  2024-03-01 01:59:00Z   60    60.10  0.1000",
    ]


def test_segment_states_a_penalty_the_caller_chose(archive_factory, capsys):
    path = _stepped_archive(archive_factory)
    rc = cmd_segment([str(path), "--penalty", "1000"])
    out = capsys.readouterr().out.splitlines()
    assert rc == 0
    assert out[0] == "plant1:FIC101.PV  1 segment   0 breakpoints   censored no"
    assert out[3] == (
        "method    PELT L2 on MAD-scaled values   penalty 1000   min-size 10"
    )


def test_segment_reports_a_censored_window_and_still_segments(archive_factory, capsys):
    path = _screen_archive(
        archive_factory, [(i, 100.0) for i in range(60, 90)],
        eng_range=EngRange(zero=0.0, span=100.0),
    )
    rc = cmd_segment([str(path)])
    out = capsys.readouterr().out.splitlines()
    assert rc == 0
    assert out[0] == "plant1:FIC101.PV  3 segments   2 breakpoints   censored yes"
    assert out[2].endswith("usable 121")


def test_segment_refuses_a_state_valued_tag(archive_factory, capsys):
    _, modes = _regime_archives(archive_factory)
    rc = cmd_segment([str(modes)])
    err = capsys.readouterr().err
    assert rc == 2
    assert err.startswith("error: ")
    assert "state-valued" in err


@pytest.mark.parametrize(("flag", "value"), [("--penalty", "0"), ("--min-size", "-1")])
def test_segment_rejects_an_out_of_range_option(archive_factory, capsys, flag, value):
    path = _stepped_archive(archive_factory)
    with pytest.raises(SystemExit) as exit_info:
        cmd_segment([str(path), flag, value])
    err = capsys.readouterr().err
    assert exit_info.value.code == 2
    assert f"argument {flag}" in err
    assert "Traceback" not in err


def test_screen_flags_a_spike_against_a_mad_baseline(archive_factory, capsys):
    path = _screen_archive(archive_factory, [(70, 100.0), (80, 100.0), (90, 100.0)])
    rc = cmd_screen(
        [str(path), "--baseline", BASELINE_SPAN, "--window", MONITOR_SPAN]
    )
    assert rc == 0
    assert capsys.readouterr().out.splitlines() == [
        "plant1:FIC101.PV  flagged 3 of 61 (4.9%)",
        "",
        "baseline  2024-03-01 00:00:00Z -> 00:59:00Z   GOOD 60   censored no",
        "window    2024-03-01 01:00:00Z -> 02:00:00Z",
        "method    MAD   center 52.00   scale 1.483   k 3.0   "
        "limits [47.55, 56.45]",
        "caveat    provisional: one baseline for every regime in the window",
        "",
        "Flagged",
        "  2024-03-01 01:10:00Z",
        "  2024-03-01 01:20:00Z",
        "  2024-03-01 01:30:00Z",
    ]


def test_screen_keys_baselines_by_regime_with_a_mode_archive(archive_factory, capsys):
    pv, modes = _regime_archives(archive_factory, [(95, 52.0), (100, 52.0)])
    rc = cmd_screen(
        [
            str(pv),
            "--baseline",
            BASELINE_SPAN,
            "--window",
            "2024-03-01T01:00:00Z/2024-03-01T01:59:00Z",
            "--mode",
            str(modes),
        ]
    )
    assert rc == 0
    assert capsys.readouterr().out.splitlines() == [
        "plant1:FIC101.PV  flagged 2 of 60 (3.3%)",
        "",
        "baseline  2024-03-01 00:00:00Z -> 00:59:00Z   GOOD 60   censored no",
        "window    2024-03-01 01:00:00Z -> 01:59:00Z",
        f"mode      {modes}",
        "alignment  baseline 60/60   window 60/60",
        "method    REGIME_MAD   k 3.0",
        "",
        "Regimes",
        "  regime R0   center 52.00   scale 1.483   n 30",
        "  regime R1   center 82.00   scale 1.483   n 30",
        "",
        "Flagged",
        "  2024-03-01 01:35:00Z",
        "  2024-03-01 01:40:00Z",
    ]


@pytest.mark.parametrize(
    "baseline",
    [BASELINE_SPAN, TOUCHING_BASELINE],
    ids=["a minute apart", "bounds touching"],
)
def test_screen_counts_every_sample_once(archive_factory, capsys, baseline):
    """Adjacent baseline and window share no sample.

    The archive holds 121 minute-spaced samples over 00:00..02:00 and
    reads are inclusive at both ends, so a baseline ending at 01:00 reads
    the same sample the window starts on.
    """
    path = _screen_archive(archive_factory)
    rc = cmd_screen([str(path), "--baseline", baseline, "--window", MONITOR_SPAN])
    out = capsys.readouterr().out.splitlines()
    assert rc == 0
    assert out[2].endswith("GOOD 60   censored no")
    # 60 + 61 = every sample, once
    assert out[0] == "plant1:FIC101.PV  flagged 0 of 61 (0.0%)"


def test_screen_refuses_a_censored_baseline(archive_factory, capsys):
    path = _screen_archive(
        archive_factory, [(10, 100.0)], eng_range=EngRange(zero=0.0, span=100.0)
    )
    rc = cmd_screen([str(path), "--baseline", BASELINE_SPAN, "--window", MONITOR_SPAN])
    err = capsys.readouterr().err
    assert rc == 2
    assert err.startswith("[InsufficientQuality]")
    assert "may never serve as a baseline" in err


@pytest.mark.parametrize("value", ["-1", "0"])
def test_screen_rejects_a_non_positive_k(archive_factory, capsys, value):
    path = _screen_archive(archive_factory)
    with pytest.raises(SystemExit) as exit_info:
        cmd_screen(
            [
                str(path),
                "--baseline",
                BASELINE_SPAN,
                "--window",
                MONITOR_SPAN,
                "--k",
                value,
            ]
        )
    captured = capsys.readouterr()
    assert exit_info.value.code == 2
    assert "argument --k" in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""


def test_screen_refuses_a_baseline_overlapping_the_window(archive_factory, capsys):
    path = _screen_archive(archive_factory)
    rc = cmd_screen(
        [
            str(path),
            "--baseline",
            "2024-03-01T00:00:00Z/2024-03-01T01:30:00Z",
            "--window",
            MONITOR_SPAN,
        ]
    )
    captured = capsys.readouterr()
    assert rc == 2
    assert captured.err.strip() == "error: baseline and window overlap"
    assert captured.out == ""


def test_screen_refuses_a_baseline_with_too_few_good_samples(archive_factory, capsys):
    path = _screen_archive(archive_factory)
    rc = cmd_screen(
        [
            str(path),
            "--baseline",
            "2024-03-01T00:00:00Z/2024-03-01T00:10:00Z",
            "--window",
            MONITOR_SPAN,
        ]
    )
    err = capsys.readouterr().err
    assert rc == 2
    assert err.strip() == (
        "[InsufficientQuality] only 11 GOOD history samples; 30 required for a baseline"
    )


def test_screen_refuses_when_most_rows_have_no_mode_sample(archive_factory, capsys):
    pv, _ = _regime_archives(archive_factory)
    base = pd.Timestamp("2024-03-01 00:00:30+00:00")  # half a sample out of step
    offset = archive_factory(
        pd.DataFrame(
            {
                "timestamp": [base + pd.Timedelta(60 * i, unit="s") for i in range(120)],
                "value": ["R0"] * 120,
                "quality": ["GOOD"] * 120,
            }
        ),
        make_meta(point_id="FIC101.OFFSET_MODE", role=Role.MODE, sample_rate_s=60.0),
    )
    rc = cmd_screen(
        [
            str(pv),
            "--baseline",
            BASELINE_SPAN,
            "--window",
            "2024-03-01T01:00:00Z/2024-03-01T01:59:00Z",
            "--mode",
            str(offset),
        ]
    )
    err = capsys.readouterr().err
    assert rc == 2
    assert err.strip() == (
        "[InsufficientQuality] 60 of 60 baseline rows carry no mode sample at their "
        "own timestamp; more than half the window would be screened blind"
    )


def test_screen_refuses_a_mode_archive_with_duplicate_timestamps(
    archive_factory, capsys
):
    pv, _ = _regime_archives(archive_factory)
    # The repeat sits in order, so the read passes the monotonicity audit
    # and the join is what has to refuse.
    indices = [*range(6), 5, *range(6, 120)]
    doubled = archive_factory(
        _minute_frame(indices, ["R0" if i % 60 < 30 else "R1" for i in indices]),
        make_meta(point_id="FIC101.DUP_MODE", role=Role.MODE, sample_rate_s=60.0),
    )
    rc = cmd_screen(
        [
            str(pv),
            "--baseline",
            BASELINE_SPAN,
            "--window",
            "2024-03-01T01:00:00Z/2024-03-01T01:59:00Z",
            "--mode",
            str(doubled),
        ]
    )
    err = capsys.readouterr().err
    assert rc == 2
    assert err.startswith("[SchemaError]")
    assert "duplicate timestamps in the mode archive" in err


def test_spc_reports_each_rule_separately(archive_factory, capsys):
    path = _screen_archive(
        archive_factory, [*((i, 53.0) for i in range(60, 72)), (80, 100.0)]
    )
    rc = cmd_spc([str(path), "--baseline", BASELINE_SPAN, "--window", MONITOR_SPAN])
    assert rc == 0
    assert capsys.readouterr().out.splitlines() == [
        "plant1:FIC101.PV  3 rule hits in 61 samples",
        "",
        "baseline  2024-03-01 00:00:00Z -> 00:59:00Z   GOOD 60   censored no",
        "window    2024-03-01 01:00:00Z -> 02:00:00Z",
        "limits    center 52.00   sigma 1.483   lcl 47.55   ucl 56.45",
        "basis     individuals 3-sigma",
        "",
        "BEYOND_3SIGMA  1",
        "  2024-03-01 01:20:00Z   value 100 outside [47.55, 56.45]",
        "",
        "RUN_9_SAMESIDE  1",
        "  2024-03-01 01:08:00Z   9 consecutive points on one side of center 52",
        "",
        "TREND_6  1",
        "  2024-03-01 01:20:00Z   6 consecutively increasing/decreasing points",
    ]


def test_spc_prints_a_rule_that_never_fired(archive_factory, capsys):
    path = _screen_archive(archive_factory)
    rc = cmd_spc([str(path), "--baseline", BASELINE_SPAN, "--window", MONITOR_SPAN])
    assert rc == 0
    assert capsys.readouterr().out.splitlines()[-5:] == [
        "BEYOND_3SIGMA  0",
        "",
        "RUN_9_SAMESIDE  0",
        "",
        "TREND_6  0",
    ]


def test_segment_json_carries_every_segment(archive_factory, capsys):
    path = _stepped_archive(archive_factory)
    assert cmd_segment([str(path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["tag"] == "plant1:FIC101.PV"
    assert payload["usable"] == 120
    assert payload["penalty_is_default"] is True
    assert payload["breakpoints"] == ["2024-03-01T01:00:00+00:00"]
    assert [s["n"] for s in payload["segments"]] == [60, 60]
    assert payload["segments"][1]["median"] == 60.1


def test_screen_json_lists_every_flagged_timestamp(archive_factory, capsys):
    path = _screen_archive(archive_factory, [(70, 100.0), (80, 100.0), (90, 100.0)])
    rc = cmd_screen(
        [str(path), "--baseline", BASELINE_SPAN, "--window", MONITOR_SPAN, "--json"]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["method"] == "MAD"
    assert payload["mode"] is None
    assert payload["regimes"] is None
    assert payload["baseline"]["n_good"] == 60
    assert payload["n_screened"] == 61
    # Three flagged, and the text output shows exactly FLAGGED_SHOWN of them.
    assert payload["flagged"] == [
        "2024-03-01T01:10:00+00:00",
        "2024-03-01T01:20:00+00:00",
        "2024-03-01T01:30:00+00:00",
    ]


def test_screen_json_reports_one_baseline_per_regime(archive_factory, capsys):
    pv, modes = _regime_archives(archive_factory, [(95, 52.0), (100, 52.0)])
    rc = cmd_screen(
        [
            str(pv),
            "--baseline",
            BASELINE_SPAN,
            "--window",
            "2024-03-01T01:00:00Z/2024-03-01T01:59:00Z",
            "--mode",
            str(modes),
            "--json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["method"] == "REGIME_MAD"
    assert payload["center"] is None
    assert [r["regime"] for r in payload["regimes"]] == ["R0", "R1"]
    assert payload["alignment"] == {
        "baseline": {"kept": 60, "total": 60},
        "window": {"kept": 60, "total": 60},
    }


def test_spc_json_reports_every_rule_including_the_quiet_ones(
    archive_factory, capsys
):
    path = _screen_archive(
        archive_factory, [*((i, 53.0) for i in range(60, 72)), (80, 100.0)]
    )
    rc = cmd_spc(
        [str(path), "--baseline", BASELINE_SPAN, "--window", MONITOR_SPAN, "--json"]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["limits"]["basis"] == "individuals 3-sigma"
    assert payload["n_monitored"] == 61
    assert payload["n_hits"] == 3
    assert [(r["rule"], r["n"]) for r in payload["rules"]] == [
        ("BEYOND_3SIGMA", 1),
        ("RUN_9_SAMESIDE", 1),
        ("TREND_6", 1),
    ]
    assert payload["rules"][0]["hits"][0]["detail"].startswith("value 100 outside")


def test_every_step_fits_in_eighty_columns(archive_factory, capsys):
    path = _screen_archive(
        archive_factory, [*((i, 53.0) for i in range(60, 72)), (80, 100.0)]
    )
    spans = ["--baseline", BASELINE_SPAN, "--window", MONITOR_SPAN]
    assert cmd_segment([str(path)]) == 0
    assert cmd_screen([str(path), *spans]) == 0
    assert cmd_spc([str(path), *spans]) == 0
    over = [ln for ln in capsys.readouterr().out.splitlines() if len(ln) >= 80]
    assert over == []


def test_spc_refuses_a_baseline_overlapping_the_window(archive_factory, capsys):
    path = _screen_archive(archive_factory)
    rc = cmd_spc(
        [
            str(path),
            "--baseline",
            "2024-03-01T00:00:00Z/2024-03-01T01:30:00Z",
            "--window",
            MONITOR_SPAN,
        ]
    )
    captured = capsys.readouterr()
    assert rc == 2
    assert captured.err.strip() == "error: baseline and window overlap"
    assert captured.out == ""


MSPC_BASELINE = "2024-03-01T00:00:00Z/2024-03-01T01:59:00Z"
MSPC_WINDOW = "2024-03-01T02:00:00Z/2024-03-01T02:59:00Z"


def _mspc_archives(archive_factory, *, burst=0.0, second_rate=60.0):
    """Two tags moving together on a 60 s grid, the second one bursting late.

    Each carries its own small wiggle so the fitted residual is not
    numerically zero, which would make every monitored row a breach.
    """
    n = 180
    level = [50.0 + 5.0 * math.sin(2 * math.pi * i / 24) for i in range(n)]
    flow = [level[i] + 0.1 * math.sin(2 * math.pi * i / 7) for i in range(n)]
    temp = [2 * level[i] + 0.1 * math.cos(2 * math.pi * i / 11) for i in range(n)]
    for i in range(150, n):
        temp[i] += burst
    return (
        archive_factory(
            _minute_frame(range(n), flow), make_meta(sample_rate_s=60.0)
        ),
        archive_factory(
            _minute_frame(range(n), temp),
            make_meta(point_id="TIC101.PV", sample_rate_s=second_rate),
        ),
    )


def _mspc_four_archives(archive_factory, *, burst=0.0):
    """Four tags on one grid, all driven by the same level, one bursting."""
    n = 180
    level = [50.0 + 5.0 * math.sin(2 * math.pi * i / 24) for i in range(n)]
    series = {
        "FIC101.PV": [level[i] + 0.1 * math.sin(2 * math.pi * i / 7) for i in range(n)],
        "TIC101.PV": [2 * level[i] + 0.1 * math.cos(2 * math.pi * i / 11) for i in range(n)],
        "PIC101.PV": [0.5 * level[i] + 0.1 * math.sin(2 * math.pi * i / 13) for i in range(n)],
        "LIC101.PV": [1.5 * level[i] + 0.1 * math.cos(2 * math.pi * i / 17) for i in range(n)],
    }
    for i in range(150, n):
        series["TIC101.PV"][i] += burst
    return [
        archive_factory(
            _minute_frame(range(n), values),
            make_meta(point_id=point_id, sample_rate_s=60.0),
        )
        for point_id, values in series.items()
    ]


def test_mspc_surfaces_a_residual_burst_in_spe(archive_factory, capsys):
    flow, temp = _mspc_archives(archive_factory, burst=5.0)
    rc = cmd_mspc(
        [
            str(flow),
            str(temp),
            "--baseline",
            MSPC_BASELINE,
            "--window",
            MSPC_WINDOW,
        ]
    )
    out = capsys.readouterr().out.splitlines()
    assert rc == 0
    # Two long tag names push the list off the headline, onto a tags line.
    assert out[0] == "2 tags  T2 breaches 12   SPE breaches 30   of 60 rows"
    assert out[2] == "tags      plant1:FIC101.PV, plant1:TIC101.PV"
    assert out[5].startswith("model     rate 60 s (declared)")
    assert "SPE breaches  30" in out
    assert out[out.index("SPE breaches  30") + 1] == "  2024-03-01 02:30:00Z"


def test_mspc_does_not_rank_contributors_over_three_tags(archive_factory, capsys):
    flow, temp = _mspc_archives(archive_factory, burst=5.0)
    rc = cmd_mspc(
        [str(flow), str(temp), "--baseline", MSPC_BASELINE, "--window", MSPC_WINDOW]
    )
    out = capsys.readouterr().out.splitlines()
    assert rc == 0
    # The note closes the label block, once, and no breach line names tags.
    limits = next(i for i, line in enumerate(out) if line.startswith("limits    "))
    assert out[limits + 1] == (
        "contributors  not ranked (2 tags; top-3 would list every one)"
    )
    assert out[limits + 2] == ""
    assert not any("contributors " in line for line in out[limits + 2 :])


def test_mspc_ranks_contributors_past_three_tags(archive_factory, capsys):
    paths = _mspc_four_archives(archive_factory, burst=5.0)
    rc = cmd_mspc(
        [*[str(p) for p in paths], "--baseline", MSPC_BASELINE, "--window", MSPC_WINDOW]
    )
    out = capsys.readouterr().out.splitlines()
    assert rc == 0
    assert not any(line.startswith("contributors  not ranked") for line in out)
    spe_count = next(line for line in out if line.startswith("SPE breaches  "))
    assert spe_count != "SPE breaches  0"
    at = out.index(spe_count)
    first_spe = out[at + 1]
    assert first_spe.startswith("  2024-03-01 02:30:00Z   contributors ")
    # Three of the four tags are named, so the ranking is a choice. The
    # third one falls onto the continuation line the wrap produced.
    named = first_spe.split("contributors ")[1] + " " + out[at + 2].strip()
    assert len(named.split(", ")) == 3
    assert "plant1:TIC101.PV" in first_spe


def test_mspc_reports_t2_and_spe_separately(archive_factory, capsys):
    # Nothing breaks the correlation here, so SPE stays clean while T2
    # still records the one exceedance an empirical q0.99 limit implies.
    flow, temp = _mspc_archives(archive_factory)
    rc = cmd_mspc(
        [str(flow), str(temp), "--baseline", MSPC_BASELINE, "--window", MSPC_WINDOW]
    )
    out = capsys.readouterr().out.splitlines()
    assert rc == 0
    assert "T2 breaches  1" in out
    assert out[-1] == "SPE breaches  0"


def test_mspc_json_carries_every_breach(archive_factory, capsys):
    flow, temp = _mspc_archives(archive_factory, burst=5.0)
    rc = cmd_mspc(
        [
            str(flow),
            str(temp),
            "--baseline",
            MSPC_BASELINE,
            "--window",
            MSPC_WINDOW,
            "--json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["tags"] == ["plant1:FIC101.PV", "plant1:TIC101.PV"]
    assert payload["rate_s"] == 60
    assert payload["rate_source"] == "declared"
    assert payload["model"] == {
        "components": 1,
        "n_columns": 2,
        "explained_variance": payload["model"]["explained_variance"],
    }
    assert payload["contributors_ranked"] is False
    assert payload["contributors"] == {}
    # The text output shows five of these; the JSON carries all of them.
    assert len(payload["spe_breaches"]) == 30
    assert payload["spe_breaches"][0] == "2024-03-01T02:30:00+00:00"


def test_mspc_json_ranks_contributors_past_three_tags(archive_factory, capsys):
    paths = _mspc_four_archives(archive_factory, burst=5.0)
    rc = cmd_mspc(
        [
            *[str(p) for p in paths],
            "--baseline",
            MSPC_BASELINE,
            "--window",
            MSPC_WINDOW,
            "--json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["contributors_ranked"] is True
    named = payload["contributors"]["2024-03-01T02:30:00+00:00"]
    assert len(named) == 3
    assert "plant1:TIC101.PV" in named


def test_mspc_lines_fit_in_eighty_columns(archive_factory, capsys):
    paths = _mspc_four_archives(archive_factory, burst=5.0)
    rc = cmd_mspc(
        [*[str(p) for p in paths], "--baseline", MSPC_BASELINE, "--window", MSPC_WINDOW]
    )
    assert rc == 0
    over = [ln for ln in capsys.readouterr().out.splitlines() if len(ln) >= 80]
    assert over == []


def test_mspc_refuses_archives_declaring_different_rates(archive_factory, capsys):
    flow, temp = _mspc_archives(archive_factory, second_rate=30.0)
    rc = cmd_mspc(
        [str(flow), str(temp), "--baseline", MSPC_BASELINE, "--window", MSPC_WINDOW]
    )
    err = capsys.readouterr().err
    assert rc == 2
    assert err.strip() == (
        "error: archives declare different sample rates (plant1:FIC101.PV=60.0, "
        "plant1:TIC101.PV=30.0); pass --rate-s to state the grid"
    )


def test_mspc_refuses_an_archive_that_declares_no_rate(archive_factory, capsys):
    flow, temp = _mspc_archives(archive_factory, second_rate=None)
    rc = cmd_mspc(
        [str(flow), str(temp), "--baseline", MSPC_BASELINE, "--window", MSPC_WINDOW]
    )
    err = capsys.readouterr().err
    assert rc == 2
    assert err.strip() == (
        "error: no sample_rate_s declared by plant1:TIC101.PV; pass --rate-s to "
        "state the grid"
    )


def test_mspc_refuses_a_single_archive(archive_factory, capsys):
    flow, _ = _mspc_archives(archive_factory)
    rc = cmd_mspc([str(flow), "--baseline", MSPC_BASELINE, "--window", MSPC_WINDOW])
    err = capsys.readouterr().err
    assert rc == 2
    assert err.strip() == (
        "[MspcAlignmentError] need at least two windows to align"
    )


@pytest.mark.parametrize(
    ("flag", "value"),
    [("--rate-s", "0"), ("--rate-s", "-5"), ("--quantile", "1.5")],
)
def test_mspc_rejects_an_out_of_range_option(archive_factory, capsys, flag, value):
    flow, temp = _mspc_archives(archive_factory)
    with pytest.raises(SystemExit) as exit_info:
        cmd_mspc(
            [
                str(flow),
                str(temp),
                "--baseline",
                MSPC_BASELINE,
                "--window",
                MSPC_WINDOW,
                flag,
                value,
            ]
        )
    err = capsys.readouterr().err
    assert exit_info.value.code == 2
    assert f"argument {flag}" in err
    assert "Traceback" not in err


COMPARE_BEFORE = "2024-03-01T00:00:00Z/2024-03-01T02:00:00Z"
COMPARE_AFTER = "2024-03-01T02:00:00Z/2024-03-01T04:00:00Z"
COMPARE_N = 241  # 60 s samples over both periods
COMPARE_SPLIT = 120  # first sample of the after period


def _compare_archives(archive_factory, *, point_ids=("FIC101.PV", "TIC101.PV")):
    """Two tags on one random walk; the second takes its own after 02:00Z."""
    rng = np.random.default_rng(5)
    shared = 50.0 + np.cumsum(rng.normal(0.0, 1.0, size=COMPARE_N))
    other = shared + rng.normal(0.0, 0.02, size=COMPARE_N)
    own = other[COMPARE_SPLIT] + np.cumsum(
        rng.normal(0.0, 1.0, size=COMPARE_N - COMPARE_SPLIT)
    )
    other[COMPARE_SPLIT:] = own
    return [
        archive_factory(
            _minute_frame(range(COMPARE_N), [float(v) for v in values]),
            make_meta(point_id=point_id, sample_rate_s=60.0),
        )
        for point_id, values in zip(point_ids, (shared, other), strict=True)
    ]


def _compare_out(capsys, paths, *extra):
    rc = cmd_compare(
        [*[str(p) for p in paths], "--before", COMPARE_BEFORE, "--after",
         COMPARE_AFTER, *extra]
    )
    return rc, capsys.readouterr()


def test_compare_prints_the_three_tables(archive_factory, capsys):
    rc, captured = _compare_out(capsys, _compare_archives(archive_factory))
    out = captured.out.splitlines()
    assert rc == 0
    assert out[:8] == [
        "plant1  2 tags   refused 0   pairs 1 of 1 clearing",
        "",
        "before    2024-03-01 00:00:00Z -> 02:00:00Z   (2 h)",
        "after     2024-03-01 02:00:00Z -> 04:00:00Z   (2 h)",
        "grid      rate 60 s (declared)   coverage 0.9917 -> 1.000",
        "",
        "Tags that changed",
        "  tag        quality   sigma  spread  flagged  changed at",
    ]
    # Both tags keep every column: the walk gives a spread to divide by,
    # and the tag that moved furthest is ranked first.
    assert [ln.split()[0] for ln in out[8:10]] == ["TIC101.PV", "FIC101.PV"]
    assert all("n/a" not in ln for ln in out[8:10])

    pairs = out.index("Pairs that decoupled  (Pearson on first differences, "
                      "95% block bootstrap)")
    assert out[pairs + 1] == "  pair                   before   after   delta  interval"
    row = out[pairs + 2].split()
    assert row[:3] == ["FIC101.PV", "~", "TIC101.PV"]
    assert float(row[3]) > 0.9  # one walk before
    assert float(row[5]) < -0.5  # and the interval below says so
    assert out[pairs + 2].endswith("]")

    joint = out.index(
        "Joint structure  (PCA fitted on before, 2 of 2 tags aligned)"
    )
    assert out[joint + 1].startswith("  components 1 of 2   explained ")
    assert out[joint + 1].endswith(" on after")
    assert out[joint + 2].startswith("  rows 121   T2 breaches ")
    assert out[joint + 3].startswith("  SPE contributors   ")


def test_compare_lines_fit_in_eighty_columns(archive_factory, capsys):
    """Long identities and a wide quality value together, over four tags."""
    paths = _compare_archives(
        archive_factory,
        point_ids=("FIC101.PROCESS_VARIABLE", "TIC101.PROCESS_VARIABLE"),
    )
    unmapped = archive_factory(
        _minute_frame(
            range(COMPARE_N),
            [50.0 + (i % 7) for i in range(COMPARE_N)],
            ["GOOD"] * COMPARE_SPLIT + ["SENSOR DRIFT"] * (COMPARE_N - COMPARE_SPLIT),
        ),
        make_meta(point_id="PIC101.PROCESS_VARIABLE", sample_rate_s=60.0),
    )
    rc, captured = _compare_out(capsys, [*paths, unmapped])
    assert rc == 0
    assert "UNCERTAIN 100%" in captured.out
    over = [ln for ln in captured.out.splitlines() if len(ln) >= 80]
    assert over == []


def test_compare_truncates_a_tag_the_column_cannot_hold(archive_factory, capsys):
    paths = _compare_archives(
        archive_factory,
        point_ids=("FIC101.PROCESS_VARIABLE", "TIC101.PROCESS_VARIABLE"),
    )
    _, captured = _compare_out(capsys, paths)
    assert "  FIC101.PROCESS..  " in captured.out
    assert "  FIC101.PROCESS.. ~ TIC101.PROCESS..  " in captured.out
    # The full identity is still one line down, in the SPE contributors.
    assert "FIC101.PROCESS_VARIABLE" in captured.out


def test_compare_top_counts_the_rows_it_left_out(archive_factory, capsys):
    paths = _compare_archives(archive_factory)
    rc, captured = _compare_out(capsys, paths, "--top", "1")
    out = captured.out.splitlines()
    assert rc == 0
    assert out[0] == "plant1  2 tags   refused 0   pairs 1 of 1 clearing"
    more = next(ln for ln in out if ln.startswith("  (+1 more within "))
    assert more.endswith(" sigma)")


def test_compare_refuses_overlapping_periods(archive_factory, capsys):
    paths = _compare_archives(archive_factory)
    rc = cmd_compare(
        [
            *[str(p) for p in paths],
            "--before",
            COMPARE_BEFORE,
            "--after",
            "2024-03-01T01:00:00Z/2024-03-01T04:00:00Z",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    assert captured.err.strip() == "error: --before and --after overlap"


def test_compare_names_the_error_class_of_an_unreadable_archive(
    archive_factory, capsys, tmp_path
):
    paths = _compare_archives(archive_factory)
    missing = tmp_path / "plant1" / "ZIC101.PV.parquet"
    rc, captured = _compare_out(capsys, [*paths, missing])
    out = captured.out.splitlines()
    assert rc == 0
    assert out[0].endswith("3 tags   refused 1   pairs 1 of 1 clearing")
    refused = next(ln for ln in out if "ZIC101.PV" in ln)
    assert refused.split()[1:] == ["[FileNotFoundError]", "n/a", "n/a", "n/a", "n/a"]


def test_compare_json_carries_every_column(archive_factory, capsys):
    paths = _compare_archives(archive_factory)
    rc, captured = _compare_out(capsys, paths, "--json")
    assert rc == 0
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["before"] == {
        "start": "2024-03-01T00:00:00+00:00",
        "end": "2024-03-01T02:00:00+00:00",
        "duration_s": 7200.0,
    }
    assert payload["source_id"] == "plant1"
    assert payload["grid"]["rate_s"] == 60
    assert payload["grid"]["rate_source"] == "declared"
    # The table truncates the label; the JSON carries the identity.
    assert [t["tag"] for t in payload["tags"]] == [
        "plant1:TIC101.PV",
        "plant1:FIC101.PV",
    ]
    assert all(t["refused"] is False for t in payload["tags"])
    assert payload["pairs"][0]["left"] == "FIC101.PV"
    assert payload["pairs"][0]["clears"] is True
    assert len(payload["pairs"][0]["interval"]) == 2
    assert payload["pairs"][0]["spearman_delta"] < -0.5
    assert payload["pairs_reason"] is None
    assert payload["joint"]["components"] == 1
    assert payload["joint"]["tags_aligned"] == 2
    shares = payload["joint"]["spe_contributors"]
    assert {c["tag"] for c in shares} == {"FIC101.PV", "TIC101.PV"}
    assert sum(c["share"] for c in shares) == pytest.approx(1.0)
    assert payload["joint"]["spe_other_share"] is None
    assert payload["joint_reason"] is None


def test_compare_json_puts_a_reason_beside_every_null(archive_factory, capsys, tmp_path):
    flat = archive_factory(
        _minute_frame(range(COMPARE_N), [50.0] * COMPARE_N),
        make_meta(point_id="LIC101.PV", sample_rate_s=60.0),
    )
    missing = tmp_path / "plant1" / "ZIC101.PV.parquet"
    rc, captured = _compare_out(
        capsys, [*_compare_archives(archive_factory), flat, missing], "--json"
    )
    assert rc == 0
    payload = json.loads(captured.out)
    rows = {t["tag"]: t for t in payload["tags"]}
    still = rows["plant1:LIC101.PV"]
    assert still["level_shift_sigma"] is None
    assert still["level_shift_reason"] == "no spread before"
    assert still["spread_reason"] == "no spread before"
    # A read that never reached the metadata has only the file name to
    # call the tag by.
    gone = rows["ZIC101.PV"]
    assert gone["refused"] is True
    assert gone["changed_at"] is None
    assert gone["changed_at_reason"] == "the read raised FileNotFoundError"
    assert payload["pairs_undefined"] == 2


def test_compare_json_reports_a_refused_pair_grid(archive_factory, capsys):
    shared, other = _compare_archives(archive_factory)
    slower = archive_factory(
        _minute_frame(range(COMPARE_N), [50.0 + (i % 7) for i in range(COMPARE_N)]),
        make_meta(point_id="PIC101.PV", sample_rate_s=30.0),
    )
    rc, captured = _compare_out(capsys, [shared, other, slower], "--json")
    assert rc == 0
    payload = json.loads(captured.out)
    assert payload["pairs"] is None
    assert payload["pairs_reason"].startswith("archives declare different sample rates")
    assert payload["joint"] is None
    assert payload["joint_reason"] == payload["pairs_reason"]


def test_compare_prints_a_refused_grid_as_one_reason_line(archive_factory, capsys):
    shared, other = _compare_archives(archive_factory)
    slower = archive_factory(
        _minute_frame(range(COMPARE_N), [50.0 + (i % 7) for i in range(COMPARE_N)]),
        make_meta(point_id="PIC101.PV", sample_rate_s=30.0),
    )
    rc, captured = _compare_out(capsys, [shared, other, slower])
    out = captured.out.splitlines()
    assert rc == 0
    assert out[0] == "plant1  3 tags   refused 0   pairs refused"
    assert "Pairs that decoupled" in out
    assert out[out.index("Pairs that decoupled") + 1].startswith(
        "  archives declare different sample rates"
    )
    assert out[out.index("Joint structure") + 1].startswith(
        "  archives declare different sample rates"
    )


def test_run_compare_returns_the_same_lines_the_command_prints(
    archive_factory, capsys
):
    paths = _compare_archives(archive_factory)
    args = _parser_compare().parse_args(
        [*[str(p) for p in paths], "--before", COMPARE_BEFORE, "--after", COMPARE_AFTER]
    )
    lines = run_compare(args)
    _, captured = _compare_out(capsys, paths)
    assert lines == captured.out.splitlines()


def test_main_dispatches_compare(archive_factory, capsys):
    paths = _compare_archives(archive_factory)
    rc = main(
        [
            "compare",
            *[str(p) for p in paths],
            "--before",
            COMPARE_BEFORE,
            "--after",
            COMPARE_AFTER,
        ]
    )
    assert rc == 0
    assert capsys.readouterr().out.startswith("plant1  2 tags")


def test_run_profile_returns_the_report_lines(tmp_path):
    path, meta = _demo_archive(tmp_path)
    lines = run_profile(_parser_profile().parse_args([str(path), "--window", WINDOW]))
    assert lines[0] == f"{meta.identity}  {meta.name}"
    for marker in (
        "coverage 0.098   GOOD 119/120",
        "contract  TIME_WEIGHTED",
        "Coverage",
        "Quality",
        "Range",
        "Values  GOOD n=119",
    ):
        assert any(ln.startswith(marker) for ln in lines), f"missing {marker!r}"


def test_run_segment_returns_the_segment_table(archive_factory):
    path = _stepped_archive(archive_factory)
    assert run_segment(_parser_segment().parse_args([str(path)])) == [
        "plant1:FIC101.PV  2 segments   1 breakpoint   censored no",
        "",
        "window    2024-03-01 00:00:00Z -> 01:59:00Z   usable 120",
        "method    PELT L2 on MAD-scaled values   penalty 14.36 (3*log n)   "
        "min-size 10",
        "",
        "  #  start                 end                     n   median     mad",
        "  1  2024-03-01 00:00:00Z  2024-03-01 00:59:00Z   60    50.10  0.1000",
        "  2  2024-03-01 01:00:00Z  2024-03-01 01:59:00Z   60    60.10  0.1000",
    ]


def test_run_screen_returns_the_screen_lines(archive_factory):
    path = _screen_archive(archive_factory, [(70, 100.0), (80, 100.0), (90, 100.0)])
    args = _parser_screen().parse_args(
        [str(path), "--baseline", BASELINE_SPAN, "--window", MONITOR_SPAN]
    )
    assert run_screen(args) == [
        "plant1:FIC101.PV  flagged 3 of 61 (4.9%)",
        "",
        "baseline  2024-03-01 00:00:00Z -> 00:59:00Z   GOOD 60   censored no",
        "window    2024-03-01 01:00:00Z -> 02:00:00Z",
        "method    MAD   center 52.00   scale 1.483   k 3.0   "
        "limits [47.55, 56.45]",
        "caveat    provisional: one baseline for every regime in the window",
        "",
        "Flagged",
        "  2024-03-01 01:10:00Z",
        "  2024-03-01 01:20:00Z",
        "  2024-03-01 01:30:00Z",
    ]


def test_run_spc_returns_the_chart_lines(archive_factory):
    path = _screen_archive(
        archive_factory, [*((i, 53.0) for i in range(60, 72)), (80, 100.0)]
    )
    args = _parser_spc().parse_args(
        [str(path), "--baseline", BASELINE_SPAN, "--window", MONITOR_SPAN]
    )
    assert run_spc(args) == [
        "plant1:FIC101.PV  3 rule hits in 61 samples",
        "",
        "baseline  2024-03-01 00:00:00Z -> 00:59:00Z   GOOD 60   censored no",
        "window    2024-03-01 01:00:00Z -> 02:00:00Z",
        "limits    center 52.00   sigma 1.483   lcl 47.55   ucl 56.45",
        "basis     individuals 3-sigma",
        "",
        "BEYOND_3SIGMA  1",
        "  2024-03-01 01:20:00Z   value 100 outside [47.55, 56.45]",
        "",
        "RUN_9_SAMESIDE  1",
        "  2024-03-01 01:08:00Z   9 consecutive points on one side of center 52",
        "",
        "TREND_6  1",
        "  2024-03-01 01:20:00Z   6 consecutively increasing/decreasing points",
    ]


def test_run_mspc_returns_the_detection_lines(archive_factory):
    flow, temp = _mspc_archives(archive_factory, burst=5.0)
    args = _parser_mspc().parse_args(
        [str(flow), str(temp), "--baseline", MSPC_BASELINE, "--window", MSPC_WINDOW]
    )
    lines = run_mspc(args)
    assert lines[0] == "2 tags  T2 breaches 12   SPE breaches 30   of 60 rows"
    assert lines[2] == "tags      plant1:FIC101.PV, plant1:TIC101.PV"
    assert lines[5].startswith("model     rate 60 s (declared)")
    assert "SPE breaches  30" in lines
    assert lines[lines.index("SPE breaches  30") + 1] == "  2024-03-01 02:30:00Z"


def test_profile_flatline_optin(tmp_path, capsys):
    path, _ = _demo_archive(tmp_path)
    # Archive spans ~1h; window is the last 20 min so prior windows exist.
    rc = cmd_profile(
        [
            str(path),
            "--window",
            "2024-03-01T00:02:00Z/2024-03-01T01:00:00Z",
            "--flatline",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert out.splitlines()[-2] == "Flatline"


def _wide_fixture(tmp_path):
    """A 3-tag wide CSV with ``_q`` quality columns and a meta dir for it."""
    tags = ("FIC101.PV", "TIC201.PV", "PIC301.PV")
    stamps = pd.date_range("2024-03-01T00:00:00Z", periods=4, freq="60s")
    rows: dict[str, list] = {"ts": [s.isoformat() for s in stamps]}
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir()
    for i, tag in enumerate(tags):
        rows[tag] = [10.0 * i + k for k in range(len(stamps))]
        rows[f"{tag}_q"] = ["Good"] * len(stamps)
        payload = {
            "identity": {"source_id": "plant1", "point_id": tag},
            "name": tag,
            "sample_rate_s": 60.0,
            "quality_codes": {"Good": "GOOD"},
        }
        (meta_dir / f"{tag}.json").write_text(json.dumps(payload), encoding="utf-8")
    src = tmp_path / "wide.csv"
    pd.DataFrame(rows).to_csv(src, index=False)
    return src, meta_dir, tags


def test_ingest_wide_writes_one_archive_per_tag_and_prints_each(tmp_path, capsys):
    src, meta_dir, tags = _wide_fixture(tmp_path)
    out_dir = tmp_path / "archive"
    rc = main(
        [
            "ingest",
            str(src),
            "--wide",
            "--out",
            str(out_dir),
            "--meta-dir",
            str(meta_dir),
            "--timestamp-col",
            "ts",
            "--quality-suffix",
            "_q",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 0
    assert captured.err == ""
    assert captured.out.splitlines() == [
        f"wrote     {(out_dir / f'{tag}.parquet').as_posix()}" for tag in tags
    ]
    assert all((out_dir / f"{tag}.parquet").exists() for tag in tags)
    assert cmd_profile([str(out_dir / "TIC201.PV.parquet")]) == 0
    assert "GOOD 4   UNCERTAIN 0   BAD 0" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("extra", "named"),
    [
        (["--wide", "--meta", "m.json"], "--meta"),
        (["--wide"], "--meta-dir"),
        (["--meta", "m.json", "--meta-dir", "meta"], "--meta-dir"),
        (["--meta", "m.json", "--tags", "A,B"], "--tags"),
        (["--meta", "m.json", "--quality-suffix", "_q"], "--quality-suffix"),
    ],
)
def test_ingest_rejects_flags_of_the_other_form(tmp_path, capsys, extra, named):
    with pytest.raises(SystemExit) as exit_info:
        main(["ingest", "export.csv", "--out", str(tmp_path / "x"), *extra])
    err = capsys.readouterr().err
    assert exit_info.value.code == 2
    assert named in err
    assert "Traceback" not in err


def test_ingest_init_meta_writes_templates_and_stops(tmp_path, capsys):
    src, _, tags = _wide_fixture(tmp_path)
    out_dir = tmp_path / "templates"
    rc = main(
        [
            "ingest",
            str(src),
            "--wide",
            "--init-meta",
            str(out_dir),
            "--source-id",
            "plant1",
            "--timestamp-col",
            "ts",
            "--quality-suffix",
            "_q",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 0
    assert captured.err == ""
    assert captured.out.splitlines() == [
        f"wrote     {(out_dir / f'{tag}.json').as_posix()}" for tag in tags
    ]
    payload = json.loads((out_dir / "PIC301.PV.json").read_text(encoding="utf-8"))
    assert payload["identity"] == {"source_id": "plant1", "point_id": "PIC301.PV"}
    assert payload["quality_codes"] == {"Good": "GOOD"}
    assert not any(tmp_path.glob("**/*.parquet"))


@pytest.mark.parametrize(
    ("extra", "named"),
    [
        (["--wide", "--init-meta", "t"], "--source-id"),
        (["--wide", "--init-meta", "t", "--source-id", "p", "--out", "a"], "--out"),
        (["--wide", "--init-meta", "t", "--source-id", "p", "--meta-dir", "m"], "--meta-dir"),
        (["--init-meta", "t", "--source-id", "p"], "--init-meta"),
    ],
)
def test_ingest_init_meta_rejects_conflicting_flags(capsys, extra, named):
    with pytest.raises(SystemExit) as exit_info:
        main(["ingest", "export.csv", *extra])
    err = capsys.readouterr().err
    assert exit_info.value.code == 2
    assert named in err
    assert "Traceback" not in err


def test_profile_reads_a_bare_date_as_one_utc_day(tmp_path, capsys):
    path, meta = _demo_archive(tmp_path)
    assert cmd_profile([str(path), "--window", "2024-03-01"]) == 0
    by_date = capsys.readouterr().out
    explicit = "2024-03-01T00:00:00Z/2024-03-02T00:00:00Z"
    assert cmd_profile([str(path), "--window", explicit]) == 0
    assert capsys.readouterr().out == by_date
    assert by_date.splitlines()[0] == f"{meta.identity}  {meta.name}"


def test_segment_mode_out_writes_the_archive_and_names_it(archive_factory, tmp_path, capsys):
    path = _stepped_archive(archive_factory)
    out = tmp_path / "seg.parquet"
    assert cmd_segment([str(path)]) == 0
    plain = capsys.readouterr().out
    assert cmd_segment([str(path), "--mode-out", str(out)]) == 0
    assert capsys.readouterr().out == f"{plain}\nwrote     {out.as_posix()}\n"
    assert pd.read_parquet(out)["value"].tolist() == ["S1"] * 60 + ["S2"] * 60
    assert cmd_segment([str(path), "--mode-out", str(out)]) == 2
    err = capsys.readouterr().err
    assert err.startswith("error:") and "already exists" in err
    assert cmd_segment([str(path), "--mode-out", str(out), "--overwrite"]) == 0


def test_segment_json_mode_out_stays_one_object(archive_factory, tmp_path, capsys):
    path = _stepped_archive(archive_factory)
    out = tmp_path / "seg.parquet"
    assert cmd_segment([str(path), "--json", "--mode-out", str(out)]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["mode_out"] == out.as_posix()
    assert len(doc["segments"]) == 2
    assert out.exists()

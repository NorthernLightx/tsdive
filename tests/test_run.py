"""``tsdive run``: one plan, several archives, one evidence ledger."""

from __future__ import annotations

import json
import math
import re
from itertools import takewhile

import pandas as pd
import pytest

from conftest import EngRange, make_meta, write_archive
from tsdive.cli import cmd_run

ALL_FIVE = """\
archives = ["plant1/*.parquet"]
window   = "2024-03-01T01:00:00Z/2024-03-01T02:00:00Z"
baseline = "2024-03-01T00:00:00Z/2024-03-01T00:59:00Z"
steps    = ["mspc", "profile", "segment", "screen", "spc"]

[options.profile]
flatline = true
tz = ["Europe/London"]
"""


def _archive(tmp_path, point_id, values, *, eng_range=None):
    """One minute-spaced archive under ``tmp_path/plant1``."""
    base = pd.Timestamp("2024-03-01 00:00:00+00:00")
    frame = pd.DataFrame(
        {
            "timestamp": [
                base + pd.Timedelta(60 * i, unit="s") for i in range(len(values))
            ],
            "value": values,
            "quality": ["GOOD"] * len(values),
        }
    )
    meta = make_meta(point_id=point_id, sample_rate_s=60.0, eng_range=eng_range)
    return write_archive(tmp_path / "plant1" / f"{point_id}.parquet", frame, meta)


def _two_archives(tmp_path):
    """Two tags moving together on a 60 s grid, each with its own wiggle."""
    n = 121
    level = [50.0 + 5.0 * math.sin(2 * math.pi * i / 24) for i in range(n)]
    flow = [level[i] + 0.1 * math.sin(2 * math.pi * i / 7) for i in range(n)]
    temp = [2 * level[i] + 0.1 * math.cos(2 * math.pi * i / 11) for i in range(n)]
    return _archive(tmp_path, "FIC101.PV", flow), _archive(tmp_path, "TIC101.PV", temp)


def _plan(tmp_path, body: str, name: str = "plan.toml"):
    path = tmp_path / name
    path.write_text(body, encoding="utf-8", newline="\n")
    return path


def _ledger(out_dir) -> dict:
    return json.loads((out_dir / "ledger.json").read_text(encoding="utf-8"))


def test_plan_runs_every_step_over_every_archive(tmp_path, capsys):
    _two_archives(tmp_path)
    plan = _plan(tmp_path, ALL_FIVE)
    out = tmp_path / "out"
    assert cmd_run([str(plan), "-o", str(out)]) == 0
    ledger = _ledger(out)
    assert len(ledger["profiles"]) == 2
    assert ledger["refusals"] == []
    # Listed mspc-first, run in pipeline order, per-tag steps once per archive.
    assert [(f["step"], f["tags"]) for f in ledger["findings"]] == [
        ("segment", "plant1:FIC101.PV"),
        ("segment", "plant1:TIC101.PV"),
        ("screen", "plant1:FIC101.PV"),
        ("screen", "plant1:TIC101.PV"),
        ("spc", "plant1:FIC101.PV"),
        ("spc", "plant1:TIC101.PV"),
        ("mspc", "plant1:FIC101.PV, plant1:TIC101.PV"),
    ]
    printed = capsys.readouterr().out.splitlines()
    assert printed[0] == (
        "run plan.toml   2 archives   5 steps   profiles 2   findings 7   "
        "refusals 0"
    )
    assert printed[-4:] == [
        "",
        f"wrote     {(out / 'ledger.json').as_posix()}",
        f"          {(out / 'ledger.txt').as_posix()}",
        f"          {(out / 'report.html').as_posix()}",
    ]
    # ledger.txt opens with the same text, then every finding in full.
    text = (out / "ledger.txt").read_text(encoding="utf-8").splitlines()
    assert text[: len(printed)] == printed
    assert sum(1 for ln in text if ln.startswith("FINDING  ")) == 7


def test_a_refusal_names_the_step_then_wraps_its_message(tmp_path, capsys):
    _archive(tmp_path, "FIC101.PV", [50.0 + (i % 5) for i in range(121)])
    plan = _plan(
        tmp_path,
        """\
archives = ["plant1/FIC101.PV.parquet"]
window   = "2024-03-01T01:00:00Z/2024-03-01T02:00:00Z"
baseline = "2024-03-01T00:00:00Z/2024-03-01T00:59:00Z"
steps    = ["mspc"]
""",
    )
    assert cmd_run([str(plan), "-o", str(tmp_path / "out")]) == 2
    printed = capsys.readouterr().out.splitlines()
    assert printed[0].endswith("profiles 0   findings 0   refusals 1")
    assert printed[2] == "REFUSAL  mspc   plant1:FIC101.PV"
    assert printed[3] == "  [MspcAlignmentError] need at least two windows to align"
    # The ledger keeps the one-line form the refusal log has always had.
    assert _ledger(tmp_path / "out")["refusals"] == [
        "[MspcAlignmentError] mspc plant1:FIC101.PV: need at least two windows to align"
    ]


def test_a_long_refusal_message_wraps_under_eighty_columns(tmp_path, capsys):
    censored = [50.0 + (i % 5) for i in range(121)]
    censored[10] = 100.0  # inside the baseline, pinned at full scale
    _archive(tmp_path, "TIC101.PV", censored, eng_range=EngRange(zero=0.0, span=100.0))
    plan = _plan(
        tmp_path,
        """\
archives = ["plant1/TIC101.PV.parquet"]
window   = "2024-03-01T01:00:00Z/2024-03-01T02:00:00Z"
baseline = "2024-03-01T00:00:00Z/2024-03-01T00:59:00Z"
steps    = ["screen"]
""",
    )
    assert cmd_run([str(plan), "-o", str(tmp_path / "out")]) == 2
    printed = capsys.readouterr().out.splitlines()
    at = printed.index("REFUSAL  screen   plant1:TIC101.PV")
    folded = list(takewhile(bool, printed[at + 1 :]))
    assert len(folded) > 1  # the message did not fit on one line
    assert all(len(ln) < 80 for ln in folded)
    assert " ".join(ln.strip() for ln in folded) == (
        "[InsufficientQuality] tag plant1:TIC101.PV: window is censored (clipped "
        "fraction 0.017); a clipped window may never serve as a baseline"
    )


def test_a_boolean_option_reaches_the_step_as_a_bare_flag(tmp_path):
    _two_archives(tmp_path)
    plan = _plan(tmp_path, ALL_FIVE)
    out = tmp_path / "out"
    assert cmd_run([str(plan), "-o", str(out)]) == 0
    # flatline = true is a plan option; the default profile never says this.
    assert all(p.splitlines()[-2] == "Flatline" for p in _ledger(out)["profiles"])


def test_a_list_option_repeats_its_flag_and_hits_the_same_validator(tmp_path):
    _two_archives(tmp_path)
    plan = _plan(
        tmp_path,
        """\
archives = ["plant1/FIC101.PV.parquet"]
steps    = ["profile"]

[options.profile]
tz = ["Europe/London", "Mars/Olympus_Mons"]
""",
    )
    out = tmp_path / "out"
    assert cmd_run([str(plan), "-o", str(out)]) == 2
    row = _ledger(out)["refusals"][0]
    assert row.startswith("[ValueError] profile plant1:FIC101.PV: ")
    assert "Mars/Olympus_Mons" in row
    assert "IANA" in row


def test_ledger_json_is_byte_identical_run_to_run(tmp_path):
    _two_archives(tmp_path)
    plan = _plan(tmp_path, ALL_FIVE)
    first, second = tmp_path / "a", tmp_path / "b"
    assert cmd_run([str(plan), "-o", str(first)]) == 0
    assert cmd_run([str(plan), "-o", str(second)]) == 0
    assert (first / "ledger.json").read_bytes() == (second / "ledger.json").read_bytes()


def test_report_html_carries_the_findings(tmp_path):
    _two_archives(tmp_path)
    plan = _plan(tmp_path, ALL_FIVE)
    out = tmp_path / "out"
    assert cmd_run([str(plan), "-o", str(out)]) == 0
    html_text = (out / "report.html").read_text(encoding="utf-8")
    assert "<h2>Findings</h2>" in html_text
    assert "<h3>mspc - plant1:FIC101.PV, plant1:TIC101.PV</h3>" in html_text


def test_report_html_draws_the_screen_and_spc_results_over_each_plot(tmp_path):
    _two_archives(tmp_path)
    plan = _plan(tmp_path, ALL_FIVE)
    out = tmp_path / "out"
    assert cmd_run([str(plan), "-o", str(out)]) == 0
    html_text = (out / "report.html").read_text(encoding="utf-8")
    figures = re.findall(r"<svg .*?</svg>", html_text)
    assert len(figures) == 2
    assert html_text.index("<h2>Plots</h2>") < html_text.index("<h2>Window profiles</h2>")
    findings = _ledger(out)["findings"]
    for figure, tag in zip(figures, ["plant1:FIC101.PV", "plant1:TIC101.PV"], strict=True):
        assert f"<title>{tag}  " in figure
        by_step = {f["step"]: f["text"] for f in findings if f["tags"] == tag}
        flagged = re.search(r"flagged (\d+) of", by_step["screen"])
        hits = re.search(r"(\d+) rule hits? in", by_step["spc"])
        assert flagged is not None and hits is not None
        assert figure.count('class="flag"') == int(flagged.group(1))
        assert figure.count('class="hit"') == int(hits.group(1))
        assert figure.count('class="center"') == 1
        assert figure.count('class="limit"') == 2
    headline = (out / "ledger.txt").read_text(encoding="utf-8").splitlines()[0]
    assert headline.startswith("run plan.toml   2 archives   5 steps   profiles 2")


def test_a_censored_baseline_refuses_that_tag_and_the_run_continues(tmp_path):
    _archive(tmp_path, "FIC101.PV", [50.0 + (i % 5) for i in range(121)])
    censored = [50.0 + (i % 5) for i in range(121)]
    censored[10] = 100.0  # inside the baseline, pinned at full scale
    _archive(
        tmp_path, "TIC101.PV", censored, eng_range=EngRange(zero=0.0, span=100.0)
    )
    plan = _plan(
        tmp_path,
        """\
archives = ["plant1/*.parquet"]
window   = "2024-03-01T01:00:00Z/2024-03-01T02:00:00Z"
baseline = "2024-03-01T00:00:00Z/2024-03-01T00:59:00Z"
steps    = ["profile", "screen", "spc"]
""",
    )
    out = tmp_path / "out"
    assert cmd_run([str(plan), "-o", str(out)]) == 0
    ledger = _ledger(out)
    assert len(ledger["profiles"]) == 2
    assert [f["tags"] for f in ledger["findings"]] == [
        "plant1:FIC101.PV",
        "plant1:FIC101.PV",
    ]
    # the refused tag keeps its plot from the profile step, with nothing drawn over it
    figures = re.findall(r"<svg .*?</svg>", (out / "report.html").read_text(encoding="utf-8"))
    assert len(figures) == 2
    assert figures[0].count('class="center"') == 1
    assert figures[1].count('class="center"') == 0
    assert figures[1].count('class="flag"') == 0
    assert len(ledger["refusals"]) == 2
    for row, step in zip(ledger["refusals"], ("screen", "spc"), strict=True):
        assert row.startswith(f"[InsufficientQuality] {step} plant1:TIC101.PV: ")
        assert "may never serve as a baseline" in row


def test_unknown_step_refuses_before_anything_runs(tmp_path, capsys):
    _two_archives(tmp_path)
    plan = _plan(
        tmp_path,
        'archives = ["plant1/*.parquet"]\nsteps = ["profile", "narrate"]\n',
    )
    out = tmp_path / "out"
    assert cmd_run([str(plan), "-o", str(out)]) == 2
    err = capsys.readouterr().err
    assert err.strip() == (
        "error: unknown step(s): narrate; known: profile, segment, screen, spc, mspc"
    )
    assert not out.exists()


def test_a_step_listed_twice_refuses_the_plan(tmp_path, capsys):
    _two_archives(tmp_path)
    plan = _plan(
        tmp_path, 'archives = ["plant1/*.parquet"]\nsteps = ["profile", "profile"]\n'
    )
    assert cmd_run([str(plan), "-o", str(tmp_path / "out")]) == 2
    assert capsys.readouterr().err.strip() == "error: step(s) listed twice: profile"


def test_an_unquoted_window_refuses_the_plan(tmp_path, capsys):
    """TOML reads a bare timestamp as one instant, which is not a range."""
    _two_archives(tmp_path)
    plan = _plan(
        tmp_path,
        'archives = ["plant1/*.parquet"]\nsteps = ["profile"]\n'
        "window = 2024-03-31T01:00:00Z\n",
    )
    out = tmp_path / "out"
    assert cmd_run([str(plan), "-o", str(out)]) == 2
    assert capsys.readouterr().err.strip() == (
        "error: 'window' must be a quoted START/END string, not a datetime"
    )
    assert not out.exists()


def test_a_glob_matching_nothing_refuses_the_plan(tmp_path, capsys):
    _two_archives(tmp_path)
    plan = _plan(tmp_path, 'archives = ["unit-9/*.parquet"]\nsteps = ["profile"]\n')
    out = tmp_path / "out"
    assert cmd_run([str(plan), "-o", str(out)]) == 2
    assert capsys.readouterr().err.strip() == (
        "error: no archive matches 'unit-9/*.parquet'"
    )
    assert not out.exists()


@pytest.mark.parametrize(
    ("option", "expected"),
    [
        ("k = -1", "argument --k: must be greater than 0, not -1.0"),
        ("nope = 1", "unrecognized arguments: --nope 1"),
    ],
    ids=["bad value", "bad name"],
)
def test_a_bad_option_refuses_only_that_step(tmp_path, option, expected):
    _two_archives(tmp_path)
    plan = _plan(
        tmp_path,
        f"""\
archives = ["plant1/*.parquet"]
window   = "2024-03-01T01:00:00Z/2024-03-01T02:00:00Z"
baseline = "2024-03-01T00:00:00Z/2024-03-01T00:59:00Z"
steps    = ["profile", "screen"]

[options.screen]
{option}
""",
    )
    out = tmp_path / "out"
    assert cmd_run([str(plan), "-o", str(out)]) == 0  # profile still produced output
    ledger = _ledger(out)
    assert ledger["findings"] == []
    assert ledger["refusals"] == [
        f"[ValueError] screen plant1:FIC101.PV: {expected}",
        f"[ValueError] screen plant1:TIC101.PV: {expected}",
    ]


def test_mspc_over_one_archive_is_a_refusal_row(tmp_path):
    _two_archives(tmp_path)
    plan = _plan(
        tmp_path,
        """\
archives = ["plant1/FIC101.PV.parquet"]
window   = "2024-03-01T01:00:00Z/2024-03-01T02:00:00Z"
baseline = "2024-03-01T00:00:00Z/2024-03-01T00:59:00Z"
steps    = ["mspc"]
""",
    )
    out = tmp_path / "out"
    assert cmd_run([str(plan), "-o", str(out)]) == 2  # nothing else produced output
    assert _ledger(out)["refusals"] == [
        "[MspcAlignmentError] mspc plant1:FIC101.PV: need at least two windows to align"
    ]


def test_a_missing_window_refuses_the_step_that_needs_one(tmp_path):
    _two_archives(tmp_path)
    plan = _plan(
        tmp_path,
        """\
archives = ["plant1/FIC101.PV.parquet"]
baseline = "2024-03-01T00:00:00Z/2024-03-01T00:59:00Z"
steps    = ["profile", "screen"]
""",
    )
    out = tmp_path / "out"
    assert cmd_run([str(plan), "-o", str(out)]) == 0
    ledger = _ledger(out)
    # profile defaults an omitted window to the archive's own extent.
    assert len(ledger["profiles"]) == 1
    assert ledger["refusals"] == [
        "[ValueError] screen plant1:FIC101.PV: the plan sets no 'window', which "
        "screen requires"
    ]


def test_a_missing_baseline_refuses_the_step_that_needs_one(tmp_path):
    _two_archives(tmp_path)
    plan = _plan(
        tmp_path,
        """\
archives = ["plant1/FIC101.PV.parquet"]
window   = "2024-03-01T01:00:00Z/2024-03-01T02:00:00Z"
steps    = ["spc"]
""",
    )
    out = tmp_path / "out"
    assert cmd_run([str(plan), "-o", str(out)]) == 2
    assert _ledger(out)["refusals"] == [
        "[ValueError] spc plant1:FIC101.PV: the plan sets no 'baseline', which "
        "spc requires"
    ]


def test_the_default_output_directory_sits_beside_the_plan(tmp_path):
    _two_archives(tmp_path)
    plan = _plan(tmp_path, 'archives = ["plant1/*.parquet"]\nsteps = ["profile"]\n')
    assert cmd_run([str(plan)]) == 0
    assert (tmp_path / "tsdive-run" / "ledger.json").exists()

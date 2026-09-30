"""``tsdive run``: one plan, several archives, one evidence ledger."""

from __future__ import annotations

import json
import math
import re
from itertools import takewhile

import pandas as pd
import pytest

from conftest import EngRange, make_meta, without_contract, write_archive
from tsdive.api import profile
from tsdive.cli import cmd_compare, cmd_run, cmd_screen

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
    # ledger.txt opens with the same text, then the tag table, every
    # profile and every finding in full.
    text = (out / "ledger.txt").read_text(encoding="utf-8").splitlines()
    assert text[: len(printed)] == printed
    assert text[len(printed) : len(printed) + 2] == ["", "TAGS"]
    assert text[len(printed) + 2].split() == [
        "tag", "coverage", "GOOD", "censored", "gaps", "longest", "constant", "flatline",
        "refused", "errors",
    ]
    assert [ln for ln in text if ln.startswith("PROFILE  ")] == [
        "PROFILE  plant1:FIC101.PV",
        "PROFILE  plant1:TIC101.PV",
    ]
    assert sum(1 for ln in text if ln.startswith("FINDING  ")) == 7
    first_finding = text.index("FINDING  segment   plant1:FIC101.PV")
    assert text.index("PROFILE  plant1:TIC101.PV") < first_finding
    for profile_text in ledger["profiles"]:
        assert profile_text in "\n".join(text)


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
    assert _ledger(tmp_path / "out")["refusals"] == [
        {
            "step": "mspc",
            "tags": "plant1:FIC101.PV",
            "error_type": "MspcAlignmentError",
            "cause": "need at least two windows to align",
        }
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
    ledger = _ledger(out)
    assert ledger["refusals"] == []
    (row,) = ledger["errors"]
    assert (row["step"], row["tags"], row["error_type"]) == (
        "profile",
        "plant1:FIC101.PV",
        "ValueError",
    )
    assert "Mars/Olympus_Mons" in row["cause"]
    assert "IANA" in row["cause"]


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
        assert (row["step"], row["tags"], row["error_type"]) == (
            step,
            "plant1:TIC101.PV",
            "InsufficientQuality",
        )
        assert "may never serve as a baseline" in row["cause"]
    assert [(t["tag"], t["refused"]) for t in ledger["tags"]] == [
        ("plant1:FIC101.PV", []),
        ("plant1:TIC101.PV", ["screen", "spc"]),
    ]


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
        "error: unknown step(s): narrate; known: profile, segment, screen, spc, mspc, compare, "
        "switchback"
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
    assert ledger["refusals"] == []
    assert ledger["errors"] == [
        {"step": "screen", "tags": tag, "error_type": "ValueError", "cause": expected}
        for tag in ("plant1:FIC101.PV", "plant1:TIC101.PV")
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
        {
            "step": "mspc",
            "tags": "plant1:FIC101.PV",
            "error_type": "MspcAlignmentError",
            "cause": "need at least two windows to align",
        }
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
    assert ledger["errors"] == [
        {
            "step": "screen",
            "tags": "plant1:FIC101.PV",
            "error_type": "ValueError",
            "cause": "the plan sets no 'window', which screen requires",
        }
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
    assert _ledger(out)["errors"] == [
        {
            "step": "spc",
            "tags": "plant1:FIC101.PV",
            "error_type": "ValueError",
            "cause": "the plan sets no 'baseline', which spc requires",
        }
    ]


BEFORE = "2024-03-01T00:00:00Z/2024-03-01T00:59:00Z"
AFTER = "2024-03-01T01:00:00Z/2024-03-01T02:00:00Z"


def test_before_and_after_run_compare_over_every_archive_at_once(tmp_path, capsys):
    flow, temp = _two_archives(tmp_path)
    plan = _plan(
        tmp_path,
        f"""\
archives = ["plant1/*.parquet"]
window   = "2024-03-01T01:00:00Z/2024-03-01T02:00:00Z"
baseline = "2024-03-01T00:00:00Z/2024-03-01T00:59:00Z"
before   = "{BEFORE}"
after    = "{AFTER}"
steps    = ["compare", "profile", "screen"]

[options.compare]
top = 1
""",
    )
    out = tmp_path / "out"
    assert cmd_run([str(plan), "-o", str(out)]) == 0
    ledger = _ledger(out)
    assert ledger["refusals"] == []
    # compare runs once over both archives, after the per-tag steps.
    assert [(f["step"], f["tags"]) for f in ledger["findings"]] == [
        ("screen", "plant1:FIC101.PV"),
        ("screen", "plant1:TIC101.PV"),
        ("compare", "plant1:FIC101.PV, plant1:TIC101.PV"),
    ]
    capsys.readouterr()
    argv = [str(flow), str(temp), "--before", BEFORE, "--after", AFTER, "--top", "1"]
    assert cmd_compare(argv) == 0
    assert ledger["findings"][-1]["text"] == capsys.readouterr().out.rstrip("\n")
    # A finding's data is the document the command prints under --json.
    assert cmd_compare([*argv, "--json"]) == 0
    printed = without_contract(json.loads(capsys.readouterr().out))
    assert ledger["findings"][-1]["data"] == printed


def test_compare_without_before_and_after_is_a_refusal_row(tmp_path):
    _two_archives(tmp_path)
    plan = _plan(
        tmp_path,
        """\
archives = ["plant1/*.parquet"]
window   = "2024-03-01T01:00:00Z/2024-03-01T02:00:00Z"
baseline = "2024-03-01T00:00:00Z/2024-03-01T00:59:00Z"
steps    = ["compare"]
""",
    )
    out = tmp_path / "out"
    assert cmd_run([str(plan), "-o", str(out)]) == 2
    assert _ledger(out)["errors"] == [
        {
            "step": "compare",
            "tags": "plant1:FIC101.PV, plant1:TIC101.PV",
            "error_type": "ValueError",
            "cause": "the plan sets no 'before' and 'after', which compare requires",
        }
    ]


@pytest.mark.parametrize(
    ("given", "missing"), [("before", "after"), ("after", "before")]
)
def test_one_period_without_the_other_refuses_the_plan(tmp_path, capsys, given, missing):
    _two_archives(tmp_path)
    plan = _plan(
        tmp_path,
        f"""\
archives = ["plant1/*.parquet"]
{given} = "{BEFORE}"
steps = ["profile"]
""",
    )
    out = tmp_path / "out"
    assert cmd_run([str(plan), "-o", str(out)]) == 2
    err = capsys.readouterr().err
    assert err.startswith(
        f"error: the plan sets '{given}' without '{missing}'; compare needs both"
    )
    assert not out.exists()


def test_the_default_output_directory_sits_beside_the_plan(tmp_path):
    _two_archives(tmp_path)
    plan = _plan(tmp_path, 'archives = ["plant1/*.parquet"]\nsteps = ["profile"]\n')
    assert cmd_run([str(plan)]) == 0
    assert (tmp_path / "tsdive-run" / "ledger.json").exists()


OVERLAPPING = """\
archives = ["plant1/*.parquet"]
window   = "2024-03-01T01:00:00Z/2024-03-01T02:00:00Z"
baseline = "2024-03-01T00:30:00Z/2024-03-01T01:30:00Z"
steps    = ["profile", "screen"]
"""

# mspc over one archive raises MspcAlignmentError, a typed refusal.
ONE_ARCHIVE_MSPC = """\
archives = ["plant1/FIC101.PV.parquet"]
window   = "2024-03-01T01:00:00Z/2024-03-01T02:00:00Z"
baseline = "2024-03-01T00:00:00Z/2024-03-01T00:59:00Z"
steps    = ["profile", "mspc"]
"""


def test_an_overlapping_baseline_is_an_error_row_not_a_refusal(tmp_path, capsys):
    _two_archives(tmp_path)
    plan = _plan(tmp_path, OVERLAPPING)
    out = tmp_path / "out"
    assert cmd_run([str(plan), "-o", str(out)]) == 0  # the default rule
    ledger = _ledger(out)
    assert ledger["refusals"] == []
    assert [(e["step"], e["error_type"], e["cause"]) for e in ledger["errors"]] == [
        ("screen", "ValueError", "baseline and window overlap"),
        ("screen", "ValueError", "baseline and window overlap"),
    ]
    printed = capsys.readouterr().out.splitlines()
    assert printed[0].endswith("refusals 0   errors 2")
    assert printed[2:4] == [
        "ERROR    screen   plant1:FIC101.PV",
        "  [ValueError] baseline and window overlap",
    ]


@pytest.mark.parametrize(
    ("body", "status"),
    [
        (ALL_FIVE, 0),
        (OVERLAPPING, 2),
        (ONE_ARCHIVE_MSPC, 3),
        (ONE_ARCHIVE_MSPC.replace('"mspc"', '"screen", "mspc"') + "[options.screen]\nk = -1\n", 2),
    ],
    ids=["clean", "error", "refusal", "error and refusal"],
)
def test_strict_exits_on_errors_then_refusals(tmp_path, body, status):
    _two_archives(tmp_path)
    plan = _plan(tmp_path, body)
    out = tmp_path / "out"
    assert cmd_run([str(plan), "-o", str(out), "--strict"]) == status
    ledger = _ledger(out)
    assert bool(ledger["errors"]) == (status == 2)
    assert bool(ledger["refusals"]) == (status == 3 or "k = -1" in body)


def test_a_finding_carries_its_json_document(tmp_path, capsys):
    flow, _ = _two_archives(tmp_path)
    out = tmp_path / "out"
    assert cmd_run([str(_plan(tmp_path, ALL_FIVE)), "-o", str(out)]) == 0
    capsys.readouterr()
    by_step = {(f["step"], f["tags"]): f["data"] for f in _ledger(out)["findings"]}
    argv = [str(flow), "--window", AFTER, "--baseline", BEFORE, "--json"]
    assert cmd_screen(argv) == 0
    printed = without_contract(json.loads(capsys.readouterr().out))
    assert by_step[("screen", "plant1:FIC101.PV")] == printed


def test_the_tag_rows_read_the_profile_document(tmp_path):
    flow, _ = _two_archives(tmp_path)
    out = tmp_path / "out"
    assert cmd_run([str(_plan(tmp_path, ALL_FIVE)), "-o", str(out)]) == 0
    ledger = _ledger(out)
    doc = profile(flow, AFTER, tz="Europe/London", flatline=True).to_dict()
    assert ledger["tags"][0] == {
        "tag": "plant1:FIC101.PV",
        "coverage": doc["coverage"]["coverage"],
        "good_share": doc["values"]["n_good"] / doc["values"]["n_samples"],
        "censored": doc["range"]["censored"],
        "gaps": doc["coverage"]["n_gaps"],
        "longest_gap_s": doc["coverage"]["longest_gap_s"],
        "constant_run_s": doc["values"]["constant_run"]["duration_s"],
        "constant_run_samples": doc["values"]["constant_run"]["samples"],
        "flatline": doc["flatline"]["verdict"],
        "refused": [],
        "errors": [],
    }
    html_text = (out / "report.html").read_text(encoding="utf-8")
    assert html_text.index("<h2>Tags</h2>") < html_text.index("<h2>Plots</h2>")
    assert "<td>plant1:TIC101.PV</td>" in html_text


def test_a_tag_without_a_profile_reads_n_a(tmp_path):
    _two_archives(tmp_path)
    body = ALL_FIVE.replace('"profile", ', "")
    out = tmp_path / "out"
    assert cmd_run([str(_plan(tmp_path, body)), "-o", str(out)]) == 0
    row = _ledger(out)["tags"][0]
    assert row["gaps"] is None and row["coverage"] is None and row["refused"] == []
    assert row["constant_run_s"] is None and row["constant_run_samples"] is None
    text = (out / "ledger.txt").read_text(encoding="utf-8").splitlines()
    at = text.index("TAGS")
    assert text[at + 2].split() == ["plant1:FIC101.PV", *["n/a"] * 7, "none", "none"]


def test_a_step_that_errors_is_named_under_errors_not_refused(tmp_path):
    _two_archives(tmp_path)
    body = ONE_ARCHIVE_MSPC.replace('"mspc"', '"screen", "mspc"') + "[options.screen]\nk = -1\n"
    out = tmp_path / "out"
    assert cmd_run([str(_plan(tmp_path, body)), "-o", str(out)]) == 0
    (row,) = _ledger(out)["tags"]
    assert (row["refused"], row["errors"]) == (["mspc"], ["screen"])
    text = (out / "ledger.txt").read_text(encoding="utf-8").splitlines()
    at = text.index("TAGS")
    assert text[at + 2].split()[-2:] == ["mspc", "screen"]


def test_the_tag_table_carries_the_longest_constant_run(tmp_path):
    values = [50.0 + (i % 5) for i in range(121)]
    values[70:81] = [58.0] * 11
    _archive(tmp_path, "FIC101.PV", values)
    body = 'archives = ["plant1/*.parquet"]\nsteps = ["profile"]\n'
    out = tmp_path / "out"
    assert cmd_run([str(_plan(tmp_path, body)), "-o", str(out)]) == 0
    (row,) = _ledger(out)["tags"]
    assert (row["constant_run_s"], row["constant_run_samples"]) == (600.0, 11)
    text = (out / "ledger.txt").read_text(encoding="utf-8").splitlines()
    header, cells = text[text.index("TAGS") + 1 : text.index("TAGS") + 3]
    at = header.index("constant")
    assert cells[at : at + len("10 min n=11")] == "10 min n=11"
    html_text = (out / "report.html").read_text(encoding="utf-8")
    assert "<td>10 min n=11</td>" in html_text

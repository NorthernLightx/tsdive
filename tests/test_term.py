"""Terminal styling, the shared formatters, and the JSON encoder."""

from __future__ import annotations

import argparse
import io
import math
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from tsdive.cli import _report_and_exit
from tsdive.errors import InsufficientQuality
from tsdive.report import fmt_duration, fmt_span, fmt_ts
from tsdive.ui.jsonout import to_jsonable
from tsdive.ui.term import colour_enabled, colourise

ESC = "\x1b"


class _Tty(io.TextIOBase):
    """A stream that claims to be a terminal and writes through."""

    def __init__(self, target) -> None:
        self.target = target

    def isatty(self) -> bool:
        return True

    def write(self, text: str) -> int:
        return self.target.write(text)


def _args(**overrides) -> argparse.Namespace:
    return argparse.Namespace(**{"json": False, "no_color": False, **overrides})


def _lines(args) -> list[str]:
    return ["Coverage", "  coverage 0.933   censored yes"]


def _as_tty(monkeypatch, name: str = "stdout") -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setattr(sys, name, _Tty(getattr(sys, name)))


def test_a_pipe_gets_no_escape_codes(capsys):
    assert _report_and_exit(_lines, _args()) == 0
    assert ESC not in capsys.readouterr().out


def test_a_terminal_gets_escape_codes(capsys, monkeypatch):
    _as_tty(monkeypatch)
    assert _report_and_exit(_lines, _args()) == 0
    out = capsys.readouterr().out
    assert f"{ESC}[1mCoverage{ESC}[0m" in out
    assert f"{ESC}[31mcensored yes{ESC}[0m" in out


def test_no_color_env_beats_a_terminal(capsys, monkeypatch):
    _as_tty(monkeypatch)
    monkeypatch.setenv("NO_COLOR", "1")
    assert _report_and_exit(_lines, _args()) == 0
    assert ESC not in capsys.readouterr().out


def test_no_color_flag_beats_a_terminal(capsys, monkeypatch):
    _as_tty(monkeypatch)
    assert _report_and_exit(_lines, _args(no_color=True)) == 0
    assert ESC not in capsys.readouterr().out


def test_a_dumb_terminal_gets_no_colour(capsys, monkeypatch):
    _as_tty(monkeypatch)
    monkeypatch.setenv("TERM", "dumb")
    assert _report_and_exit(_lines, _args()) == 0
    assert ESC not in capsys.readouterr().out


def test_a_refusal_prefix_is_red_on_a_terminal(capsys, monkeypatch):
    _as_tty(monkeypatch, "stderr")

    def refuse(args):
        raise InsufficientQuality("no GOOD samples")

    assert _report_and_exit(refuse, _args()) == 2
    captured = capsys.readouterr()
    assert captured.err == f"{ESC}[31m[InsufficientQuality]{ESC}[0m no GOOD samples\n"
    assert captured.out == ""


def test_a_piped_refusal_keeps_its_plain_prefix(capsys):
    def refuse(args):
        raise ValueError("baseline and window overlap")

    assert _report_and_exit(refuse, _args()) == 2
    assert capsys.readouterr().err == "error: baseline and window overlap\n"


def test_colour_enabled_needs_a_terminal(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm")
    assert colour_enabled(_Tty(io.StringIO())) is True
    assert colour_enabled(io.StringIO()) is False


def test_only_counts_above_zero_are_marked():
    quiet = colourise(["  GOOD 561   UNCERTAIN 0   BAD 0"], enabled=True)[0]
    assert quiet == "  GOOD 561   UNCERTAIN 0   BAD 0"
    loud = colourise(["  GOOD 561   UNCERTAIN 1   BAD 2"], enabled=True)[0]
    assert f"{ESC}[33mUNCERTAIN 1{ESC}[0m" in loud
    assert f"{ESC}[31mBAD 2{ESC}[0m" in loud


def test_a_quality_share_is_marked_whole():
    """compare states a percentage where profile states a count."""
    styled = colourise(["  FIC101.PV  BAD 100%", "  TIC101.PV  UNCERTAIN 40%"])
    assert styled[0] == f"  FIC101.PV  {ESC}[31mBAD 100%{ESC}[0m"
    assert styled[1] == f"  TIC101.PV  {ESC}[33mUNCERTAIN 40%{ESC}[0m"


def test_a_summary_line_that_names_what_it_left_out_is_dim():
    styled = colourise(["  (+8 more within 0.5 sigma)", "  (+3 more)"])
    assert styled[0] == f"{ESC}[2m  (+8 more within 0.5 sigma){ESC}[0m"
    assert styled[1] == f"{ESC}[2m  (+3 more){ESC}[0m"


def test_a_headline_is_not_a_section_header():
    styled = colourise(["demo:FIC101.PV  FIC-101 flow", "Values  GOOD n=561"])
    assert styled[0] == "demo:FIC101.PV  FIC-101 flow"
    assert styled[1] == f"{ESC}[1mValues  GOOD n=561{ESC}[0m"


def test_disabled_colour_returns_the_lines_unchanged():
    lines = ["Coverage", "  censored yes"]
    assert colourise(lines, enabled=False) == lines


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2024-03-30T20:00:00+00:00", "2024-03-30 20:00:00Z"),
        ("2024-03-30T21:00:00+01:00", "2024-03-30 20:00:00Z"),
        ("2024-03-30 20:00:00", "2024-03-30 20:00:00Z"),
        ("2024-03-30T20:00:00.250+00:00", "2024-03-30 20:00:00.25Z"),
    ],
)
def test_fmt_ts_prints_utc_with_a_trailing_z(value, expected):
    assert fmt_ts(pd.Timestamp(value)) == expected


def test_fmt_span_drops_a_repeated_date():
    same = fmt_span(
        pd.Timestamp("2024-03-30T23:00:00Z"), pd.Timestamp("2024-03-30T23:40:00Z")
    )
    assert same == "2024-03-30 23:00:00Z -> 23:40:00Z"
    across = fmt_span(
        pd.Timestamp("2024-03-30T20:00:00Z"), pd.Timestamp("2024-03-31T06:00:00Z")
    )
    assert across == "2024-03-30 20:00:00Z -> 2024-03-31 06:00:00Z"


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (None, "n/a"),
        (0.0, "0 s"),
        (45.0, "45 s"),
        (90.0, "90 s"),
        (120.0, "2 min"),
        (2400.0, "40 min"),
        (36000.0, "10 h"),
        (172800.0, "2 d"),
        (1.25, "1.2 s"),
        (float("inf"), "inf s"),
    ],
)
def test_fmt_duration_picks_the_largest_whole_unit(seconds, expected):
    assert fmt_duration(seconds) == expected


@dataclass(frozen=True)
class _Row:
    name: str
    value: float | None


def test_to_jsonable_walks_dataclasses_and_numpy():
    payload = to_jsonable(
        {
            "row": _Row(name="a", value=None),
            "counts": (np.int64(3), np.float64(1.5), np.True_),
            "when": pd.Timestamp("2024-03-30T20:00:00Z"),
            "array": np.array([1.0, 2.0]),
        }
    )
    assert payload == {
        "row": {"name": "a", "value": None},
        "counts": [3, 1.5, True],
        "when": "2024-03-30T20:00:00+00:00",
        "array": [1.0, 2.0],
    }


def test_to_jsonable_turns_non_finite_floats_into_null():
    assert to_jsonable([math.nan, math.inf, 1.0]) == [None, None, 1.0]
    assert to_jsonable(pd.NaT) is None

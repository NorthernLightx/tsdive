"""ANSI styling for command output.

Renderers return plain text. This module is the only place an escape
sequence is added, and it adds one only when the destination is a
terminal that takes colour: ``NO_COLOR`` (no-color.org), ``TERM=dumb``
and ``--no-color`` each switch it off, and a pipe or a file is never
coloured.

Colour marks meaning: a censored window, a refusal, a count above zero.
:func:`colourise` holds the whole recogniser, so no renderer has to know
that colour exists.

No width detection: every line the renderers produce is written to fit
in 80 columns.
"""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from typing import TextIO

_SGR = {"bold": "1", "dim": "2", "red": "31", "green": "32", "yellow": "33"}
_RESET = "\x1b[0m"

# Section header lines, printed bold. Listed rather than matched by shape:
# a headline and a ``label     value`` line also start in column 0.
SECTION_HEADERS = frozenset(
    {
        "Coverage",
        "Quality",
        "Range",
        "Timestamps",
        "Values",
        "Flatline",
        "Regimes",
        "Flagged",
        "BEYOND_3SIGMA",
        "RUN_9_SAMESIDE",
        "TREND_6",
        "T2 breaches",
        "SPE breaches",
        "Tags that changed",
        "Pairs that decoupled",
        "Joint structure",
    }
)

# (pattern, style) for the tokens that carry a verdict. Counts match only
# above zero, so a quiet report stays uncoloured.
_TOKENS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bREFUSAL\b"), "red"),
    (re.compile(r"\bcensored yes\b"), "red"),
    # A count, or the share compare's quality column states.
    (re.compile(r"\bBAD [1-9]\d*%?"), "red"),
    (re.compile(r"\bcensored no\b"), "green"),
    (re.compile(r"\bNOT ASSESSED\b"), "yellow"),
    (re.compile(r"\bUNCERTAIN(?: [1-9]\d*%?)?(?! [0-9])"), "yellow"),
    (re.compile(r"\bflagged [1-9]\d*\b"), "bold"),
    (re.compile(r"\bbreaches [1-9]\d*\b"), "bold"),
    (re.compile(r"\b[1-9]\d* rule hits\b"), "bold"),
    (re.compile(r"\brefusals [1-9]\d*\b"), "bold"),
    # "(+3 more)", and the forms that name what was left out.
    (re.compile(r"^ *\(\+\d+ more\b.*\)$"), "dim"),
)


def _sgr(style: str, text: str, on: bool) -> str:
    return f"\x1b[{_SGR[style]}m{text}{_RESET}" if on else text


def bold(text: str, on: bool = True) -> str:
    return _sgr("bold", text, on)


def dim(text: str, on: bool = True) -> str:
    return _sgr("dim", text, on)


def red(text: str, on: bool = True) -> str:
    return _sgr("red", text, on)


def green(text: str, on: bool = True) -> str:
    return _sgr("green", text, on)


def yellow(text: str, on: bool = True) -> str:
    return _sgr("yellow", text, on)


def colour_enabled(stream: TextIO, *, no_color: bool = False) -> bool:
    """True when ``stream`` is a terminal that takes colour."""
    if no_color or os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    try:
        return bool(stream.isatty())
    except (AttributeError, ValueError):
        return False


def rule(label: str, detail: str = "") -> list[str]:
    """A blank line and the section header for ``label``."""
    header = f"{label}  {detail}" if detail else label
    return ["", header]


def _style(line: str) -> str:
    if line[:1] not in {"", " "} and line.split("  ", 1)[0] in SECTION_HEADERS:
        return bold(line)
    for pattern, style in _TOKENS:
        # A match that already carries an escape was styled by an earlier
        # pattern; nesting two SGR runs would leave the reset in the wrong
        # place.
        line = pattern.sub(
            lambda m, style=style: _sgr(style, m.group(0), "\x1b" not in m.group(0)),
            line,
        )
    return line


def colourise(lines: Sequence[str], *, enabled: bool = True) -> list[str]:
    """The lines with SGR codes around the tokens that carry meaning."""
    if not enabled:
        return list(lines)
    return [_style(line) for line in lines]

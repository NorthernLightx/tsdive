"""Banned phrases, checked over the user-facing text.

The list is deliberately short: each entry is a phrase this repository
has actually used, not a general vocabulary filter. A hit names the
file, the line and the pattern.

The README shape is fixed too. The section list is closed, so new
material goes under an existing section or into a ``docs/`` page reached
by one link line, and the line and word budgets force a deletion for
every addition.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

PROSE_FILES = [
    ROOT / "README.md",
    ROOT / "BENCHMARKS.md",
    ROOT / "pyproject.toml",
    *sorted((ROOT / "docs").glob("*.md")),
    *sorted((ROOT / "examples" / "studies").glob("*/REPORT.md")),
    *sorted((ROOT / "scripts").glob("*.py")),
    ROOT / "src" / "tsdive" / "cli.py",
]

# (pattern, reason). Case-insensitive. Word boundaries keep "honest" from
# matching identifiers such as "dishonest_rows" that do not exist yet.
BANNED: list[tuple[str, str]] = [
    (r"\bhonest(?:ly)?\b", "claims virtue; state the fact"),
    (r"\brefus\w+ rather than guess\w*", "slogan; name the error"),
    (r"\brather than guess\w*", "slogan; name the error"),
    (r"\bnever guess\w*", "slogan; name the error"),
    (r"\bdoctrine\b", "philosophy; state the rule"),
    (r"\btrust ladder\b", "slogan; say 'the stages' or the order"),
    (r"\bclassics first\b", "slogan"),
    (r"\btheatre\b", "metaphor"),
    (r"\bon display\b", "metaphor"),
    (r"\bearns? (?:its|their) (?:place|keep)\b", "metaphor"),
    (r"\bload-bearing\b", "metaphor"),
    (r"\bsilence is not evidence\b", "aphorism"),
    (r"\bit(?:'s| is) (?:worth noting|important to note)\b", "filler"),
    (r"\bcrucially\b|\bnotably\b|\bimportantly\b", "inflated significance"),
    (r"\bin other words\b|\bput simply\b|\bthat is to say\b", "restating"),
    (r"\bsimply\b|\beasily\b", "Google style: drop"),
    (r"\bplease\b", "Google style: drop"),
    (r"\bnote that\b", "Google style: drop"),
    (r"\be\.g\.|\bi\.e\.|\betc\.", "Latin abbreviation; write it out"),
    # The en dash here is the character being matched, not a typo for a hyphen.
    (r"—|–", "dash in prose; use a comma, period or parentheses"),  # noqa: RUF001
    (r"\bnot (?:just|only|merely) \w+ but\b", "rhetorical contrast; state the claim"),
]

def _hits(path: Path) -> list[str]:
    out: list[str] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        for pattern, reason in BANNED:
            if re.search(pattern, line, flags=re.IGNORECASE):
                out.append(f"{path.relative_to(ROOT)}:{lineno}: {reason}: {line.strip()[:90]}")
    return out


@pytest.mark.parametrize("path", PROSE_FILES, ids=str)
def test_prose_has_no_banned_phrases(path: Path) -> None:
    hits = _hits(path)
    assert not hits, "\n".join(hits)


README = ROOT / "README.md"

README_SECTIONS = [
    "# tsdive",
    "## Install",
    "## Use",
    "## What it checks",
    "## How it compares",
    "## Results on real data",
    "## Limits",
    "## Development",
    "## License",
]

README_USE_SUBSECTIONS = [
    "profile",
    "segment",
    "screen",
    "spc",
    "mspc",
    "compare",
    "run",
    "ingest",
    "Python",
    "MCP",
]

README_MAX_LINES = 370
README_MAX_WORDS = 1950


def _headings(text: str, prefix: str) -> list[str]:
    """Return the lines starting with ``prefix``, in file order.

    Fenced blocks are skipped, so a ``#`` written inside a console
    transcript is not read as a heading.
    """
    out: list[str] = []
    fenced = False
    for line in text.splitlines():
        if line.startswith("```"):
            fenced = not fenced
            continue
        if not fenced and line.startswith(prefix):
            out.append(line.rstrip())
    return out


def test_readme_sections_match_the_fixed_list() -> None:
    text = README.read_text(encoding="utf-8")
    top = [h for h in _headings(text, "#") if h.startswith(("# ", "## "))]
    assert top == README_SECTIONS


def test_readme_use_subsections_match_the_fixed_list() -> None:
    text = README.read_text(encoding="utf-8")
    expected = ["### " + name for name in README_USE_SUBSECTIONS]
    assert _headings(text, "### ") == expected


def test_readme_stays_within_budget() -> None:
    text = README.read_text(encoding="utf-8")
    lines = len(text.splitlines())
    words = len(text.split())
    assert lines <= README_MAX_LINES, f"{lines} lines, budget {README_MAX_LINES}"
    assert words <= README_MAX_WORDS, f"{words} words, budget {README_MAX_WORDS}"


def test_readme_states_no_test_count() -> None:
    hit = re.search(r"\b\d+ tests\b", README.read_text(encoding="utf-8"))
    assert hit is None, f"README states a test count: {hit.group(0) if hit else ''}"

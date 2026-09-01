"""Banned phrases, checked over the user-facing text.

The list is deliberately short: each entry is a phrase this repository
has actually used, not a general vocabulary filter. A hit names the
file, the line and the pattern.
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

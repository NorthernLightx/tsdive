"""The evidence ledger.

A deterministic, serializable bundle of everything downstream narration
is allowed to see: window physics summaries, benchmark rows, refusal
log. The LLM layer (if configured at all) reads this ledger, never raw
historian write paths (which do not exist in this library anyway).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field

import tsdive
from tsdive.report import SEP, fmt_duration, yes_no_unknown

# The per-tag table of ledger.txt and report.html: (header, key of a
# ``tags`` row, right-aligned). The tag column comes first and is as
# wide as the longest tag.
TAG_COLUMNS = (
    ("coverage", "coverage", True),
    ("GOOD", "good_share", True),
    ("censored", "censored", False),
    ("gaps", "gaps", True),
    ("longest", "longest_gap_s", False),
    ("flatline", "flatline", False),
    ("refused", "refused", False),
)


def row_text(row: Mapping[str, str]) -> str:
    """A refusal or error row on one line: ``[ErrorType] step tags: cause``."""
    return f"[{row['error_type']}] {row['step']} {row['tags']}: {row['cause']}"


def _cell(row: Mapping[str, object], key: str) -> str:
    """One cell of the tag table.

    ``gaps`` is an integer on every profile, so a row whose ``gaps`` is
    None belongs to a tag the profile step did not read, and each of its
    profile cells is ``n/a``.
    """
    value = row[key]
    if key == "refused":
        return ", ".join(str(step) for step in value) or "none"  # type: ignore[attr-defined]
    if row["gaps"] is None:
        return "n/a"
    if key == "censored":
        return yes_no_unknown(value)  # type: ignore[arg-type]
    if key == "flatline":
        return "not run" if value is None else str(value)
    if value is None:
        return "n/a"
    if key in {"coverage", "good_share"}:
        return f"{float(value):.3f}"  # type: ignore[arg-type]
    if key == "longest_gap_s":
        return fmt_duration(float(value))  # type: ignore[arg-type]
    return str(value)


def tag_table(tags: Sequence[Mapping[str, object]]) -> list[tuple[str, ...]]:
    """The header and one row of cells per ``tags`` row, as strings."""
    header = ("tag", *(head for head, _, _ in TAG_COLUMNS))
    rows = [(str(t["tag"]), *(_cell(t, key) for _, key, _ in TAG_COLUMNS)) for t in tags]
    return [header, *rows]


def tag_table_lines(tags: Sequence[Mapping[str, object]]) -> list[str]:
    """The ``TAGS`` section of ledger.txt: one aligned row per tag."""
    table = tag_table(tags)
    widths = [max(len(r[i]) for r in table) for i in range(len(table[0]))]
    right = (False, *(r for _, _, r in TAG_COLUMNS))

    def line(cells: tuple[str, ...]) -> str:
        return (
            "  "
            + "  ".join(
                c.rjust(w) if r else c.ljust(w)
                for c, w, r in zip(cells, widths, right, strict=True)
            )
        ).rstrip()

    return ["TAGS", *(line(r) for r in table)]


@dataclass
class EvidenceLedger:
    title: str
    tsdive_version: str = field(default_factory=lambda: tsdive.__version__)
    profiles: list[str] = field(default_factory=list)
    # What a step other than ``profile`` said, keyed by step and by the
    # tags it read: ``step``, ``tags``, ``text`` and ``data``, the step
    # result's ``to_dict()`` document. Kept apart from ``profiles``
    # because a profile is a description of a window and a finding is a
    # verdict about one.
    findings: list[dict[str, object]] = field(default_factory=list)
    benchmark_rows: list[dict[str, str]] = field(default_factory=list)
    # A typed tsdive error of one step: ``step``, ``tags``,
    # ``error_type``, ``cause``.
    refusals: list[dict[str, str]] = field(default_factory=list)
    # Any other error of one step (a rejected option, overlapping windows,
    # a file the OS cannot read), in the same four keys.
    errors: list[dict[str, str]] = field(default_factory=list)
    # One row per archive: the profile's headline numbers and the steps
    # refused for it. A tag the profile step did not read carries None.
    tags: list[dict[str, object]] = field(default_factory=list)
    # The contract ledger.json follows, named as in every tsdive JSON document.
    result_kind: str = "ledger"

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    def to_text(self, header: Sequence[str] = ()) -> str:
        """``header``, the tag table, then one section per profile and finding.

        ``tsdive run`` passes the lines it printed, so ledger.txt opens
        with the same text the terminal showed. Without them the header is
        the ledger's own counts. Profiles precede findings because the
        profile step runs first.
        """
        lines = list(header) or [
            f"{self.title}{SEP}profiles {len(self.profiles)}{SEP}"
            f"findings {len(self.findings)}{SEP}"
            f"benchmark rows {len(self.benchmark_rows)}{SEP}"
            f"refusals {len(self.refusals)}{SEP}errors {len(self.errors)}",
            *(f"REFUSAL  {row_text(r)}" for r in self.refusals),
            *(f"ERROR    {row_text(r)}" for r in self.errors),
        ]
        if self.tags:
            lines.extend(["", *tag_table_lines(self.tags)])
        for text in self.profiles:
            lines.extend(["", f"PROFILE  {text.split('  ', 1)[0]}", text])
        for f in self.findings:
            lines.extend(["", f"FINDING  {f['step']}{SEP}{f['tags']}", str(f["text"])])
        return "\n".join(lines)

    def summary_stats(self) -> dict[str, int]:
        return {
            "profiles": len(self.profiles),
            "benchmark_rows": len(self.benchmark_rows),
            "refusals": len(self.refusals),
        }

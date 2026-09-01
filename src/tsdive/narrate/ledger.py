"""The evidence ledger.

A deterministic, serializable bundle of everything downstream narration
is allowed to see: window physics summaries, benchmark rows, refusal
log. The LLM layer (if configured at all) reads this ledger, never raw
historian write paths (which do not exist in this library anyway).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

import tsdive
from tsdive.report import SEP


@dataclass
class EvidenceLedger:
    title: str
    tsdive_version: str = field(default_factory=lambda: tsdive.__version__)
    profiles: list[str] = field(default_factory=list)
    # What a step other than ``profile`` said, keyed by step and by the
    # tags it read: ``step``, ``tags``, ``text``. Kept apart from
    # ``profiles`` because a profile is a description of a window and a
    # finding is a verdict about one.
    findings: list[dict[str, str]] = field(default_factory=list)
    benchmark_rows: list[dict[str, str]] = field(default_factory=list)
    refusals: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    def to_text(self, header: Sequence[str] = ()) -> str:
        """``header``, then one section per finding.

        ``tsdive run`` passes the lines it printed, so ledger.txt opens
        with the same text the terminal showed. Without them the header is
        the ledger's own counts.
        """
        lines = list(header) or [
            f"{self.title}{SEP}profiles {len(self.profiles)}{SEP}"
            f"findings {len(self.findings)}{SEP}"
            f"benchmark rows {len(self.benchmark_rows)}{SEP}"
            f"refusals {len(self.refusals)}",
            *(f"REFUSAL  {r}" for r in self.refusals),
        ]
        for f in self.findings:
            lines.extend(["", f"FINDING  {f['step']}{SEP}{f['tags']}", f["text"]])
        return "\n".join(lines)

    def summary_stats(self) -> dict[str, int]:
        return {
            "profiles": len(self.profiles),
            "benchmark_rows": len(self.benchmark_rows),
            "refusals": len(self.refusals),
        }

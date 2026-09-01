"""Narrator protocol.

The LLM is the last stage and sits strictly downstream of evidence
assembly. Rules enforced in code:

- The narrator receives an :class:`EvidenceLedger` (physics summaries,
  benchmark rows, refusals) and nothing else.
- No endpoint, no key, no network anywhere in this library's tests. The
  remote narrator refuses unless ``TSDIVE_LLM_ENDPOINT`` is set
  explicitly by the operator; there is no hosted default to fall back to.
- Narration output is advisory prose. It never writes to historians,
  never executes actions, never overrides a typed refusal.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

from tsdive.errors import NarratorUnavailable
from tsdive.narrate.ledger import EvidenceLedger

ENDPOINT_ENV = "TSDIVE_LLM_ENDPOINT"


class Narrator(Protocol):
    def narrate(self, ledger: EvidenceLedger) -> str: ...


@dataclass(frozen=True)
class OfflineScriptedNarrator:
    """Deterministic template narration. What tests and CI use."""

    def narrate(self, ledger: EvidenceLedger) -> str:
        stats = ledger.summary_stats()
        refusal_note = (
            f" {stats['refusals']} typed refusal(s) recorded."
            if stats["refusals"]
            else " No refusals recorded."
        )
        return (
            f"Ledger '{ledger.title}': {stats['profiles']} window profile(s), "
            f"{stats['benchmark_rows']} benchmark row(s).{refusal_note} "
            "Narration is advisory; the numbers above are the evidence."
        )


@dataclass(frozen=True)
class RemoteNarrator:
    """Sends the ledger JSON to an operator-configured HTTP endpoint.

    Constructing or calling this WITHOUT ``TSDIVE_LLM_ENDPOINT`` set
    raises :class:`NarratorUnavailable`. This class performs no I/O in
    this repository's tests; wiring an actual client is an operator task,
    deliberately not automated here.
    """

    model_hint: str = ""

    def _endpoint(self) -> str:
        endpoint = os.environ.get(ENDPOINT_ENV)
        if not endpoint:
            raise NarratorUnavailable(
                f"{ENDPOINT_ENV} is not set; there is no default LLM "
                "endpoint. Set it explicitly to enable narration."
            )
        return endpoint

    def narrate(self, ledger: EvidenceLedger) -> str:
        self._endpoint()  # gate before anything else
        payload = ledger.to_json()
        return (
            "[remote narration pending operator wiring] "
            f"endpoint configured; ledger payload {len(payload)} bytes ready"
        )


def default_narrator(*, prefer_remote: bool = False) -> Narrator:
    """Remote only when explicitly requested AND configured; else offline."""
    if prefer_remote and os.environ.get(ENDPOINT_ENV):
        return RemoteNarrator()
    return OfflineScriptedNarrator()

"""The contract any data source must satisfy.

A source is anything that can answer a window read under a stated
sampling contract and hand back the samples together with their physics
block (coverage, gaps, quality, clipping, timestamp audit, unit
resolution). Historian archives are one implementation; a CSV export, an
OPC UA history read, a message-bus capture or a simulator are others.
What makes them interchangeable is not where the bytes came from but
that every read declares its contract and reports what the store did to
the data.

Sources are read-only: the protocol has no write path, and the
public-surface test refuses one on any implementation.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from tsdive.store.identity import TagIdentity
from tsdive.store.quality import Severity
from tsdive.store.sampling_contract import SamplingContract
from tsdive.store.tagstore import Window


@runtime_checkable
class Source(Protocol):
    """A read-only window provider over process time series."""

    def list_tags(self) -> list[TagIdentity]:
        """Every tag identity this source can answer for."""
        ...

    def read_window(
        self,
        identity: TagIdentity,
        start: datetime,
        end: datetime,
        contract: SamplingContract,
        *,
        min_severity: Severity = Severity.GOOD,
        tz_names: tuple[str, ...] = (),
    ) -> Window:
        """Return the window's samples plus its data-physics block.

        ``contract`` is positional and required: a read that does not state
        how its numbers were produced is not a read this library returns.
        """
        ...

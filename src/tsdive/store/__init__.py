"""Store + data-physics layer.

Identity, sampling contracts, quality codes, timebase rules, unit
resolution, gap classification, clipping, the parquet read path, and
``write_tag`` for creating archives.
"""

from __future__ import annotations

from tsdive.store.clipping import ClippingReport
from tsdive.store.gaps import CoverageSummary, GapClass, GapContext, GapFinding
from tsdive.store.identity import EngRange, Role, TagIdentity, TagMeta
from tsdive.store.sampling_contract import (
    AggregateType,
    CalculationBasis,
    RetrievalMode,
    SamplingContract,
)
from tsdive.store.tagstore import (
    DataPhysics,
    TagStore,
    Window,
    archive_extent,
    write_tag,
)

__all__ = [
    "AggregateType",
    "CalculationBasis",
    "ClippingReport",
    "CoverageSummary",
    "DataPhysics",
    "EngRange",
    "GapClass",
    "GapContext",
    "GapFinding",
    "RetrievalMode",
    "Role",
    "SamplingContract",
    "TagIdentity",
    "TagMeta",
    "TagStore",
    "Window",
    "archive_extent",
    "write_tag",
]

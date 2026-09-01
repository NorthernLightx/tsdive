"""tsdive: data-quality profiling for process time series from any source."""

from __future__ import annotations

__version__ = "0.1.0"

from tsdive.api import Profile, ingest, profile, read_meta_json
from tsdive.errors import (
    IncomparableSamplingError,
    IncomparableUnitsError,
    InsufficientQuality,
    MspcAlignmentError,
    NarratorUnavailable,
    NonMonotonicIndex,
    RegimeTooSparse,
    SchemaError,
    TSDiveError,
    UnresolvedUnitError,
)
from tsdive.store.identity import EngRange, Role, TagIdentity, TagMeta
from tsdive.store.sampling_contract import (
    AggregateType,
    CalculationBasis,
    RetrievalMode,
    SamplingContract,
)
from tsdive.store.source import Source
from tsdive.store.tagstore import (
    SingleFileStore,
    TagStore,
    Window,
    write_tag,
)

__all__ = [
    "AggregateType",
    "CalculationBasis",
    "EngRange",
    "IncomparableSamplingError",
    "IncomparableUnitsError",
    "InsufficientQuality",
    "MspcAlignmentError",
    "NarratorUnavailable",
    "NonMonotonicIndex",
    "Profile",
    "RegimeTooSparse",
    "RetrievalMode",
    "Role",
    "SamplingContract",
    "SchemaError",
    "SingleFileStore",
    "Source",
    "TSDiveError",
    "TagIdentity",
    "TagMeta",
    "TagStore",
    "UnresolvedUnitError",
    "Window",
    "__version__",
    "ingest",
    "profile",
    "read_meta_json",
    "write_tag",
]

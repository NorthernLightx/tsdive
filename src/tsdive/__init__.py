"""tsdive: data-quality profiling for process time series from any source."""

from __future__ import annotations

__version__ = "0.2.0"

from tsdive.analyses import (
    CompareAnalysis,
    MspcAnalysis,
    ScreenAnalysis,
    SegmentAnalysis,
    SpcAnalysis,
    compare,
    mspc,
    screen,
    segment,
    spc,
)
from tsdive.api import Profile, ingest, ingest_wide, init_meta, profile, read_meta_json
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
    "CompareAnalysis",
    "EngRange",
    "IncomparableSamplingError",
    "IncomparableUnitsError",
    "InsufficientQuality",
    "MspcAlignmentError",
    "MspcAnalysis",
    "NarratorUnavailable",
    "NonMonotonicIndex",
    "Profile",
    "RegimeTooSparse",
    "RetrievalMode",
    "Role",
    "SamplingContract",
    "SchemaError",
    "ScreenAnalysis",
    "SegmentAnalysis",
    "SingleFileStore",
    "Source",
    "SpcAnalysis",
    "TSDiveError",
    "TagIdentity",
    "TagMeta",
    "TagStore",
    "UnresolvedUnitError",
    "Window",
    "__version__",
    "compare",
    "ingest",
    "ingest_wide",
    "init_meta",
    "mspc",
    "profile",
    "read_meta_json",
    "screen",
    "segment",
    "spc",
    "write_tag",
]

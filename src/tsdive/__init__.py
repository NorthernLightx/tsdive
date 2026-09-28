"""tsdive: data-quality profiling for process time series from any source."""

from __future__ import annotations

__version__ = "0.4.0"

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
from tsdive.api import (
    Profile,
    ingest,
    ingest_wide,
    init_meta,
    init_tag_meta,
    profile,
    read_meta_json,
    switchback_analyze,
    switchback_plan,
)
from tsdive.errors import (
    DesignTooSmall,
    IncomparableSamplingError,
    IncomparableUnitsError,
    InsufficientQuality,
    MspcAlignmentError,
    NarratorUnavailable,
    NonMonotonicIndex,
    RegimeTooSparse,
    ScheduleMismatch,
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
from tsdive.switchback.archive import SwitchbackAnalysis, SwitchbackEstimate
from tsdive.switchback.plan import SwitchbackPlan

__all__ = [
    "AggregateType",
    "CalculationBasis",
    "CompareAnalysis",
    "DesignTooSmall",
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
    "ScheduleMismatch",
    "SchemaError",
    "ScreenAnalysis",
    "SegmentAnalysis",
    "SingleFileStore",
    "Source",
    "SpcAnalysis",
    "SwitchbackAnalysis",
    "SwitchbackEstimate",
    "SwitchbackPlan",
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
    "init_tag_meta",
    "mspc",
    "profile",
    "read_meta_json",
    "screen",
    "segment",
    "spc",
    "switchback_analyze",
    "switchback_plan",
    "write_tag",
]

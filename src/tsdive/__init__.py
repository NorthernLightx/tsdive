"""tsdive: data-quality profiling for process time series from any source."""

from __future__ import annotations

__version__ = "0.9.0"

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
    ingest_long,
    ingest_wide,
    init_long_meta,
    init_meta,
    init_tag_meta,
    profile,
    read_meta_json,
    switchback_analyze,
    switchback_plan,
)
from tsdive.demo import write_demo_data
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
    ZeroSpreadBaseline,
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
    "ZeroSpreadBaseline",
    "__version__",
    "compare",
    "ingest",
    "ingest_long",
    "ingest_wide",
    "init_long_meta",
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
    "write_demo_data",
    "write_tag",
]

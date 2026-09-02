"""Shared fixtures. All data synthetic; zero network."""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import pandas as pd
import pytest

from tsdive.store.identity import EngRange, Role, TagIdentity, TagMeta
from tsdive.store.quality import annotate_severity, null_digital_state_values
from tsdive.store.sampling_contract import (
    AggregateType,
    CalculationBasis,
    RetrievalMode,
    SamplingContract,
)
from tsdive.store.tagstore import (
    TagStore,
    _write_parquet,
    meta_from_parquet,
    write_tag,
)

UTC = pd.Timestamp.now(tz="UTC").tz


def stepped_contract() -> SamplingContract:
    return SamplingContract(
        calculation_basis=CalculationBasis.TIME_WEIGHTED,
        retrieval_mode=RetrievalMode.RECORDED,
        aggregate_type=AggregateType.NONE,
        stepped=True,
    )


def event_contract() -> SamplingContract:
    return SamplingContract(
        calculation_basis=CalculationBasis.EVENT_WEIGHTED,
        retrieval_mode=RetrievalMode.RECORDED,
        aggregate_type=AggregateType.NONE,
        stepped=False,
    )


def good_frame(
    timestamps: Sequence[object], values: list[float]
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": [pd.Timestamp(str(t)) for t in timestamps],
            "value": values,
            "quality": ["GOOD"] * len(timestamps),
        }
    )


def annotate(df: pd.DataFrame) -> pd.DataFrame:
    return null_digital_state_values(annotate_severity(df))


def fixture_a_frame() -> pd.DataFrame:
    """59 min at 50 / 1 min at 100 with exception-reporting burst.

    Events: 50@00:00, twenty 100-samples filling [00:59, 01:00) every 3s,
    trailing 50@01:00. Time-weighted = (59*50 + 1*100)/60 = 50.83;
    event-weighted = (2*50 + 20*100)/22 = 95.45.
    """
    start = pd.Timestamp("2024-03-01 00:00:00+00:00")
    ts = [start]
    ts.extend(start + pd.Timedelta(59 * 60 + 3 * k, unit="s") for k in range(20))
    ts.append(start + pd.Timedelta(60, unit="min"))
    return good_frame(ts, [50.0] + [100.0] * 20 + [50.0])


def make_meta(
    source_id: str = "plant1",
    point_id: str = "FIC101.PV",
    *,
    unit_raw: str | None = "m3/h",
    eng_range: EngRange | None = None,
    loop_id: str | None = None,
    role: Role | None = None,
    sample_rate_s: float | None = 3.0,
    quality_codes: dict[str, str] | None = None,
) -> TagMeta:
    return TagMeta(
        identity=TagIdentity(source_id=source_id, point_id=point_id),
        name=f"{point_id} display name",
        unit_raw=unit_raw,
        eng_range=eng_range,
        sample_rate_s=sample_rate_s,
        asset="unit-100",
        loop_id=loop_id,
        role=role,
        quality_codes=quality_codes,
    )


def write_archive(
    path: Path, df: pd.DataFrame, meta: TagMeta, *, validate: bool = True
) -> Path:
    """Write a single-tag test archive through the store's public writer.

    Quality is stored as string so numeric digital-state codes survive
    verbatim ("257"), never coerced. ``validate=False`` bypasses
    ``write_tag``'s schema check, which is the only way to build the
    deliberately broken archives that provoke read-side refusals.
    """
    out = df.reset_index(drop=True).copy()
    if "quality" in out.columns:
        out["quality"] = out["quality"].map(lambda q: q if isinstance(q, str) else str(q))
    if not validate:
        path.parent.mkdir(parents=True, exist_ok=True)
        return _write_parquet(path, out, meta)
    return write_tag(path, out, meta, overwrite=True)


@pytest.fixture()
def archive_factory(tmp_path: Path) -> Callable[..., Path]:
    def _write(
        df: pd.DataFrame,
        meta: TagMeta | None = None,
        *,
        subdir: bool = True,
        filename: str | None = None,
    ) -> Path:
        meta = meta or make_meta()
        fname = filename or f"{meta.identity.point_id}.parquet"
        path = (
            tmp_path / meta.identity.source_id / fname
            if subdir
            else tmp_path / fname
        )
        return write_archive(path, df, meta)

    return _write


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture(scope="module")
def demo_archives(tmp_path_factory) -> tuple[Path, Path]:
    """The two README demo archives, built by the script that ships them."""
    spec = importlib.util.spec_from_file_location(
        "make_demo_archive", SCRIPTS / "make_demo_archive.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["make_demo_archive"] = module
    spec.loader.exec_module(module)
    out = tmp_path_factory.mktemp("demo")
    flow, flow_meta = module.build_fic101()
    temp, temp_meta = module.build_tic101(flow)
    return (
        write_tag(out / "fic101_demo.parquet", flow, flow_meta, overwrite=True),
        write_tag(out / "tic101_demo.parquet", temp, temp_meta, overwrite=True),
    )


@pytest.fixture()
def store(tmp_path: Path) -> TagStore:
    return TagStore(tmp_path)


__all__ = [
    "UTC",
    "AggregateType",
    "CalculationBasis",
    "EngRange",
    "RetrievalMode",
    "Role",
    "SamplingContract",
    "TagIdentity",
    "TagMeta",
    "annotate",
    "event_contract",
    "fixture_a_frame",
    "good_frame",
    "make_meta",
    "meta_from_parquet",
    "stepped_contract",
    "write_archive",
]

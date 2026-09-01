"""The no-theatre rule: every refusal type must have a provoking test here."""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

import tsdive.errors as errors_mod
from tsdive.errors import (
    GroupLeakage,
    IncomparableSamplingError,
    IncomparableUnitsError,
    InsufficientQuality,
    MspcAlignmentError,
    NarratorUnavailable,
    NonMonotonicIndex,
    PopulationTooSparse,
    RegimeTooSparse,
    SchemaError,
    TSDiveError,
    UnresolvedUnitError,
)
from tsdive.store.sampling_contract import CalculationBasis, RetrievalMode, SamplingContract
from tsdive.store.tagstore import SingleFileStore, assert_windows_comparable
from tsdive.store.timebase import ensure_monotonic
from tsdive.store.units import check_comparable

UTC = dt.UTC


def _provoke_schema_error():
    from tsdive.store.quality import annotate_severity

    df = pd.DataFrame({"timestamp": [pd.Timestamp("2024-03-01 00:00:00+00:00")], "value": [1.0]})
    annotate_severity(df)


def _provoke_insufficient_quality():
    from tsdive.store.averaging import event_weighted_mean
    from tsdive.store.quality import annotate_severity, null_digital_state_values

    stamps = [
        pd.Timestamp("2024-03-01 00:00:00+00:00"),
        pd.Timestamp("2024-03-01 00:00:03+00:00"),
    ]
    df = pd.DataFrame(
        {
            "timestamp": stamps,
            "value": [10.0, 11.0],
            "quality": ["COMM FAILURE", "OVER RANGE"],
        }
    )
    event_weighted_mean(null_digital_state_values(annotate_severity(df)))


def _window(tmp_path, point_id, contract):
    from conftest import make_meta, write_archive

    stamps = [
        pd.Timestamp("2024-03-01 00:00:00+00:00"),
        pd.Timestamp("2024-03-01 00:00:03+00:00"),
    ]
    meta = make_meta(source_id="p", point_id=point_id)
    path = write_archive(
        tmp_path / "p" / f"{point_id}.parquet",
        pd.DataFrame(
            {"timestamp": stamps, "value": [1.0, 2.0], "quality": ["GOOD", "GOOD"]}
        ),
        meta,
    )
    return SingleFileStore(path).read_window(
        meta.identity,
        dt.datetime(2024, 3, 1, tzinfo=UTC),
        dt.datetime(2024, 3, 1, 1, tzinfo=UTC),
        contract,
    )


def _provoke_incomparable_sampling(tmp_path):
    stepped = SamplingContract(CalculationBasis.TIME_WEIGHTED, RetrievalMode.RECORDED, stepped=True)
    event = SamplingContract(CalculationBasis.EVENT_WEIGHTED, RetrievalMode.RECORDED)
    a = _window(tmp_path, "A.PV", stepped)
    b = _window(tmp_path, "B.PV", event)
    assert_windows_comparable(a, b)


def _provoke_non_monotonic():
    stamps = [
        pd.Timestamp("2024-03-01 00:00:03+00:00"),
        pd.Timestamp("2024-03-01 00:00:01+00:00"),
    ]
    ensure_monotonic(pd.Series(pd.DatetimeIndex(stamps)))


def _provoke_unresolved_unit():
    check_comparable("???/h", "m3/h")


def _provoke_incomparable_units():
    check_comparable("Nm3/h", "m3/h")


def _provoke_regime_too_sparse():
    from tsdive.baselines.regime import regime_baselines

    stamps = [pd.Timestamp(f"2024-03-01 00:00:{i:02d}+00:00") for i in range(5)]
    values = pd.DataFrame(
        {
            "timestamp": stamps,
            "value": [1.0, 2.0, 3.0, 4.0, 5.0],
            "quality": ["GOOD"] * 5,
        }
    )
    modes = pd.Series(["R1"] * len(stamps))
    regime_baselines(values, modes, min_samples=20)


def _provoke_mspc_alignment():
    from typing import cast

    from tsdive.mspc.pca import align_windows
    from tsdive.store.identity import TagIdentity
    from tsdive.store.tagstore import Window

    def shell(point: str, offset_s: int):
        stamps = [
            pd.Timestamp("2024-03-01 00:00:00+00:00") + pd.Timedelta(offset_s + 3 * i, unit="s")
            for i in range(5)
        ]
        frame = pd.DataFrame(
            {
                "timestamp": stamps,
                "value": [float(i) for i in range(5)],
                "quality": ["GOOD"] * 5,
                "severity": ["GOOD"] * 5,
                "valid": [True] * 5,
            }
        )
        return Window(
            identity=TagIdentity("p", point),
            meta=None,  # type: ignore[arg-type]
            contract=SamplingContract(CalculationBasis.TIME_WEIGHTED, RetrievalMode.RECORDED),
            start=cast(pd.Timestamp, frame["timestamp"].min()),
            end=cast(pd.Timestamp, frame["timestamp"].max()),
            frame=frame,
            physics=None,  # type: ignore[arg-type]
        )

    # 2 s offset against a 3 s grid: almost nothing lands on the grid.
    align_windows([shell("A.PV", 0), shell("B.PV", 2)], rate_s=3, min_coverage=0.99)


def _provoke_population_too_sparse():
    from tsdive.baselines.population import population_baseline, screen_population

    rows = pd.DataFrame(
        {
            "asset": ["ASSET-A"] * 3,
            "variable": ["P-TPT"] * 3,
            "distinct_count": [900.0, 880.0, 910.0],
            "stall_s": [1.0, 1.0, 2.0],
        }
    )
    baselines = population_baseline(rows, min_windows=20)
    screen_population(
        pd.Series({"distinct_count": 1.0}), baselines[("ASSET-A", "P-TPT")]
    )


def _provoke_group_leakage():
    from tsdive.data.splits import GroupSplit

    GroupSplit(
        group_col="well",
        stratum_col="folder_label",
        n_folds=2,
        seed=42,
        fold_of_group={"WELL-A": 0},
        folds=(("WELL-A",), ("WELL-A",)),
        fold_sizes=(1, 1),
        strata_counts=({"0": 1}, {"0": 1}),
        group_sizes={"WELL-A": 1},
    ).leakage_check()


def _provoke_narrator_unavailable(monkeypatch=None):
    import os

    from tsdive.narrate.base import RemoteNarrator
    from tsdive.narrate.ledger import EvidenceLedger

    saved = os.environ.pop("TSDIVE_LLM_ENDPOINT", None)
    try:
        RemoteNarrator().narrate(EvidenceLedger(title="provoker"))
    finally:
        if saved is not None:
            os.environ["TSDIVE_LLM_ENDPOINT"] = saved


PROVOKERS = {
    SchemaError: _provoke_schema_error,
    InsufficientQuality: _provoke_insufficient_quality,
    NonMonotonicIndex: _provoke_non_monotonic,
    UnresolvedUnitError: _provoke_unresolved_unit,
    IncomparableUnitsError: _provoke_incomparable_units,
    RegimeTooSparse: _provoke_regime_too_sparse,
    MspcAlignmentError: _provoke_mspc_alignment,
    GroupLeakage: _provoke_group_leakage,
    PopulationTooSparse: _provoke_population_too_sparse,
}


@pytest.mark.parametrize(
    "refusal",
    [
        SchemaError,
        InsufficientQuality,
        NonMonotonicIndex,
        UnresolvedUnitError,
        IncomparableUnitsError,
        RegimeTooSparse,
        MspcAlignmentError,
        GroupLeakage,
        PopulationTooSparse,
    ],
)
def test_refusal_is_provoked(refusal):
    with pytest.raises(refusal):
        PROVOKERS[refusal]()


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_narrator_unavailable_is_provoked():
    with pytest.raises(NarratorUnavailable):
        _provoke_narrator_unavailable()


def test_incomparable_sampling_error_is_provoked(tmp_path):
    with pytest.raises(IncomparableSamplingError):
        _provoke_incomparable_sampling(tmp_path)


def test_inventory_matches_module():
    """A refusal type without a provoker fails this suite — add one or delete it."""
    declared = {
        cls
        for cls in vars(errors_mod).values()
        if isinstance(cls, type) and issubclass(cls, TSDiveError) and cls is not TSDiveError
    }
    provoked = set(PROVOKERS) | {IncomparableSamplingError, NarratorUnavailable}
    assert declared == provoked, (
        f"unprovoked refusal types: {declared - provoked}; "
        f"phantom provokers: {provoked - declared}"
    )

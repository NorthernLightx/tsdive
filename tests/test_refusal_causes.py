"""Fault injection: every refusal names the defect that provoked it.

``tests/test_errors.py`` proves each refusal type is reachable. This file
proves each one says WHICH defect it fired on. A refusal carrying a
plausible but wrong cause reads as correct, so it costs a reader more
than a wrong number does.

Each case builds one known defect into a synthetic frame and asserts the
class and a phrase from the message. Within a class, a cause phrase must
reject every sibling case's message: see
``test_causes_within_a_class_are_distinguishable``.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import tsdive.errors as errors_mod
from conftest import annotate, make_meta, write_archive
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
from tsdive.store.tagstore import SingleFileStore

T0 = pd.Timestamp("2024-03-01 00:00:00+00:00")
TIME_WEIGHTED = SamplingContract(
    CalculationBasis.TIME_WEIGHTED, RetrievalMode.RECORDED, stepped=True
)
EVENT_WEIGHTED = SamplingContract(CalculationBasis.EVENT_WEIGHTED, RetrievalMode.RECORDED)


def _stamps(n: int, *, step_s: int = 3, start: pd.Timestamp = T0) -> list[pd.Timestamp]:
    return [start + pd.Timedelta(step_s * i, unit="s") for i in range(n)]


def _frame(timestamps, values, quality=None) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "value": values,
            "quality": quality if quality is not None else ["GOOD"] * len(timestamps),
        }
    )


def _window(
    tmp_path: Path,
    point_id: str,
    frame: pd.DataFrame,
    *,
    contract: SamplingContract = TIME_WEIGHTED,
    start: pd.Timestamp = T0,
    end: pd.Timestamp | None = None,
    validate: bool = True,
):
    """Read one synthetic archive back as a Window.

    ``validate=False`` bypasses the write-side schema check, which is the
    only way to store a defect the reader is supposed to catch.
    """
    meta = make_meta(source_id="p", point_id=point_id)
    path = write_archive(
        tmp_path / "p" / f"{point_id}.parquet", frame, meta, validate=validate
    )
    return SingleFileStore(path).read_window(
        meta.identity,
        start,
        end if end is not None else start + pd.Timedelta(1, unit="h"),
        contract,
    )


# --------------------------------------------------------------------------
# SchemaError
# --------------------------------------------------------------------------
def s_missing_quality_column(tmp_path: Path) -> None:
    from tsdive.store.quality import validate_schema

    validate_schema(pd.DataFrame({"timestamp": _stamps(2), "value": [1.0, 2.0]}))


def s_naive_timestamps(tmp_path: Path) -> None:
    from tsdive.store.quality import validate_schema

    validate_schema(_frame([pd.Timestamp("2024-03-01 00:00:00")], [1.0]))


def s_duplicate_timestamps(tmp_path: Path) -> None:
    from tsdive.store.timebase import require_unique

    require_unique(pd.Series([T0, T0, T0 + pd.Timedelta(3, unit="s")]))


def s_archive_without_metadata(tmp_path: Path) -> None:
    from tsdive.store.tagstore import meta_from_parquet

    path = tmp_path / "bare.parquet"
    _frame(_stamps(2), [1.0, 2.0]).to_parquet(path)
    meta_from_parquet(path)


def s_non_numeric_value(tmp_path: Path) -> None:
    from tsdive.store.quality import annotate_severity, null_digital_state_values

    null_digital_state_values(annotate_severity(_frame(_stamps(2), ["RUN", "STOP"])))


def s_quality_code_maps_to_no_severity(tmp_path: Path) -> None:
    from tsdive.store.quality import annotate_severity

    annotate_severity(_frame(_stamps(1), [1.0], ["SUB"]), codes={"SUB": "MAYBE"})


def s_archive_without_samples(tmp_path: Path) -> None:
    from tsdive.store.tagstore import archive_extent

    empty = pd.DataFrame(
        {
            "timestamp": pd.Series([], dtype="datetime64[ns, UTC]"),
            "value": pd.Series([], dtype=float),
            "quality": pd.Series([], dtype=str),
        }
    )
    path = write_archive(
        tmp_path / "p" / "EMPTY.PV.parquet",
        empty,
        make_meta(source_id="p", point_id="EMPTY.PV"),
    )
    archive_extent(path)


# --------------------------------------------------------------------------
# InsufficientQuality
# --------------------------------------------------------------------------
def q_every_sample_bad(tmp_path: Path) -> None:
    from tsdive.store.averaging import event_weighted_mean

    event_weighted_mean(annotate(_frame(_stamps(2), [1.0, 2.0], ["BAD", "BAD"])))


def q_window_of_zero_duration(tmp_path: Path) -> None:
    from tsdive.store.averaging import time_weighted_mean

    time_weighted_mean(annotate(_frame([T0], [1.0])), window_end=T0)


def q_window_pinned_at_full_scale(tmp_path: Path) -> None:
    from tsdive.store.clipping import assert_usable_baseline, clipped_fraction
    from tsdive.store.identity import EngRange

    meta = make_meta(eng_range=EngRange(zero=0.0, span=100.0))
    frame = _frame(_stamps(3), [100.0, 100.0, 100.0])
    assert_usable_baseline(
        meta, clipped_fraction(frame["value"], frame["quality"], meta.eng_range)
    )


def q_history_shorter_than_the_floor(tmp_path: Path) -> None:
    from tsdive.baselines.provisional import mad_baseline

    mad_baseline(annotate(_frame(_stamps(5), [1.0, 2.0, 3.0, 4.0, 5.0])))


def q_history_without_quality_column(tmp_path: Path) -> None:
    from tsdive.baselines.provisional import mad_baseline

    mad_baseline(pd.DataFrame({"timestamp": _stamps(40), "value": [1.0] * 40}))


def q_too_few_samples_to_place_a_break(tmp_path: Path) -> None:
    from tsdive.changepoints.pelt import segment_window

    window = _window(tmp_path, "SHORT.PV", _frame(_stamps(12), [float(i) for i in range(12)]))
    segment_window(window, min_size=10)


def q_flat_values_have_no_spread(tmp_path: Path) -> None:
    from tsdive.changepoints.pelt import segment_window

    window = _window(tmp_path, "FLAT.PV", _frame(_stamps(40), [7.0] * 40))
    segment_window(window, min_size=10)


def q_history_with_no_rows(tmp_path: Path) -> None:
    from tsdive.baselines.regime import regime_baselines

    history = annotate(_frame([], []))
    regime_baselines(history, pd.Series([], dtype=object, index=history.index))


# --------------------------------------------------------------------------
# IncomparableSamplingError
# --------------------------------------------------------------------------
def c_calculation_bases_differ(tmp_path: Path) -> None:
    from tsdive.store.tagstore import assert_windows_comparable

    values = [1.0, 2.0, 3.0]
    a = _window(tmp_path, "A.PV", _frame(_stamps(3), values), contract=TIME_WEIGHTED)
    b = _window(tmp_path, "B.PV", _frame(_stamps(3), values), contract=EVENT_WEIGHTED)
    assert_windows_comparable(a, b)


# --------------------------------------------------------------------------
# NonMonotonicIndex
# --------------------------------------------------------------------------
_BACKWARDS = [T0, T0 + pd.Timedelta(10, unit="s"), T0 + pd.Timedelta(5, unit="s")]


def n_raw_column_steps_backwards(tmp_path: Path) -> None:
    from tsdive.store.timebase import ensure_monotonic

    ensure_monotonic(pd.Series(_BACKWARDS))


def n_archive_window_steps_backwards(tmp_path: Path) -> None:
    _window(
        tmp_path,
        "BACK.PV",
        _frame(_BACKWARDS, [1.0, 2.0, 3.0]),
        validate=False,
    )


def n_good_samples_step_backwards(tmp_path: Path) -> None:
    """A bad sample between two good ones offsets the two position bases.

    The frame is backwards at row 2; the GOOD samples the mean uses are
    backwards at their own position 1.
    """
    from tsdive.store.averaging import time_weighted_mean

    stamps = [
        T0 + pd.Timedelta(10, unit="s"),
        T0 + pd.Timedelta(20, unit="s"),
        T0 + pd.Timedelta(5, unit="s"),
    ]
    frame = annotate(_frame(stamps, [1.0, 2.0, 3.0], ["GOOD", "BAD", "GOOD"]))
    time_weighted_mean(frame)


# --------------------------------------------------------------------------
# UnresolvedUnitError
# --------------------------------------------------------------------------
def u_unit_spelling_unknown(tmp_path: Path) -> None:
    from tsdive.store.units import check_comparable

    check_comparable("gpm-ish", "m3/h")


def u_no_unit_declared(tmp_path: Path) -> None:
    from tsdive.store.units import check_comparable

    check_comparable(None, "m3/h")


# --------------------------------------------------------------------------
# IncomparableUnitsError
# --------------------------------------------------------------------------
def v_reference_state_undeclared(tmp_path: Path) -> None:
    from tsdive.store.units import check_comparable

    check_comparable("Nm3/h", "m3/h")


def v_dimensions_differ(tmp_path: Path) -> None:
    from tsdive.store.units import check_comparable

    check_comparable("m3/h", "degC")


# --------------------------------------------------------------------------
# RegimeTooSparse
# --------------------------------------------------------------------------
def _regime_history(counts: dict[str, int]) -> tuple[pd.DataFrame, pd.Series]:
    n = sum(counts.values())
    history = annotate(_frame(_stamps(n), [float(i) for i in range(n)]))
    labels: list[str] = []
    for regime, count in counts.items():
        labels.extend([regime] * count)
    return history, pd.Series(labels, index=history.index)


def r_regime_below_the_sample_floor(tmp_path: Path) -> None:
    from tsdive.baselines.regime import regime_baselines

    history, modes = _regime_history({"R1": 20, "R2": 3})
    regime_baselines(history, modes, min_samples=20)


def r_regime_never_seen_in_training(tmp_path: Path) -> None:
    from tsdive.baselines.regime import match_regime, regime_baselines

    history, modes = _regime_history({"R1": 20})
    match_regime(regime_baselines(history, modes, min_samples=20), "R9")


# --------------------------------------------------------------------------
# PopulationTooSparse
# --------------------------------------------------------------------------
def _population_rows(n: int, *, asset: str = "ASSET-A", variable: str = "P-TPT"):
    return pd.DataFrame(
        {
            "asset": [asset] * n,
            "variable": [variable] * n,
            "distinct_count": [900.0 + i for i in range(n)],
            "stall_s": [1.0] * n,
        }
    )


def p_pair_absent_from_training_folds(tmp_path: Path) -> None:
    from tsdive.baselines.population import match_population, population_baseline

    match_population(population_baseline(_population_rows(25)), "ASSET-B", "P-TPT")


def p_pair_below_the_window_floor(tmp_path: Path) -> None:
    from tsdive.baselines.population import population_baseline, screen_population

    baselines = population_baseline(_population_rows(3), min_windows=20)
    screen_population(pd.Series({"distinct_count": 1.0}), baselines[("ASSET-A", "P-TPT")])


def p_window_under_test_has_no_count(tmp_path: Path) -> None:
    from tsdive.baselines.population import population_baseline, screen_population

    baselines = population_baseline(_population_rows(25), min_windows=20)
    screen_population(pd.Series({"distinct_count": None}), baselines[("ASSET-A", "P-TPT")])


# --------------------------------------------------------------------------
# GroupLeakage
# --------------------------------------------------------------------------
def _split(fold_of_group: dict[str, int], folds: tuple[tuple[str, ...], ...]):
    from tsdive.data.splits import GroupSplit

    groups = sorted({g for members in folds for g in members} | set(fold_of_group))
    return GroupSplit(
        group_col="well",
        stratum_col="folder_label",
        n_folds=len(folds),
        seed=42,
        fold_of_group=fold_of_group,
        folds=folds,
        fold_sizes=tuple(len(members) for members in folds),
        strata_counts=tuple({"0": len(members)} for members in folds),
        group_sizes=dict.fromkeys(groups, 1),
    )


def g_group_reaches_two_folds(tmp_path: Path) -> None:
    _split({"WELL-A": 0}, (("WELL-A",), ("WELL-A",))).leakage_check()


def g_fold_list_disagrees_with_the_map(tmp_path: Path) -> None:
    _split({"WELL-A": 1}, (("WELL-A",), ())).leakage_check()


def g_fold_holds_a_group_outside_the_map(tmp_path: Path) -> None:
    _split({"WELL-A": 0}, (("WELL-A", "WELL-B"), ())).leakage_check()


# --------------------------------------------------------------------------
# MspcAlignmentError
# --------------------------------------------------------------------------
def m_only_one_window(tmp_path: Path) -> None:
    from tsdive.mspc.pca import align_windows

    align_windows([_window(tmp_path, "SOLO.PV", _frame(_stamps(5), [1.0] * 5))], rate_s=3)


def m_spans_do_not_overlap(tmp_path: Path) -> None:
    from tsdive.mspc.pca import align_windows

    later = T0 + pd.Timedelta(1, unit="h")
    a = _window(
        tmp_path, "EARLY.PV", _frame(_stamps(5), [1.0] * 5), end=T0 + pd.Timedelta(1, unit="min")
    )
    b = _window(
        tmp_path,
        "LATE.PV",
        _frame(_stamps(5, start=later), [1.0] * 5),
        start=later,
        end=later + pd.Timedelta(1, unit="min"),
    )
    align_windows([a, b], rate_s=3)


def m_coverage_below_the_floor(tmp_path: Path) -> None:
    from tsdive.mspc.pca import align_windows

    end = T0 + pd.Timedelta(12, unit="s")
    a = _window(tmp_path, "ONGRID.PV", _frame(_stamps(5), [1.0] * 5), end=end)
    off_grid = _frame(_stamps(5, start=T0 + pd.Timedelta(2, unit="s")), [1.0] * 5)
    b = _window(tmp_path, "OFFGRID.PV", off_grid, end=end)
    align_windows([a, b], rate_s=3, min_coverage=0.99)


def m_no_row_is_fully_observed(tmp_path: Path) -> None:
    from tsdive.mspc.pca import align_windows

    end = T0 + pd.Timedelta(10, unit="s")
    a = _window(tmp_path, "FIRST.PV", _frame([T0], [1.0]), end=end)
    b = _window(tmp_path, "SECOND.PV", _frame([end], [1.0]), end=end)
    align_windows([a, b], rate_s=10, min_coverage=0.5)


def m_monitor_columns_differ_from_the_model(tmp_path: Path) -> None:
    from tsdive.mspc.pca import AlignedMatrix, detect, fit_pca

    index = pd.DatetimeIndex(_stamps(30))
    rng = np.random.default_rng(0)
    train = AlignedMatrix(
        index=index,
        columns=["p:A.PV", "p:B.PV"],
        matrix=rng.normal(size=(30, 2)),
        coverage=1.0,
    )
    monitor = AlignedMatrix(
        index=index,
        columns=["p:A.PV", "p:C.PV"],
        matrix=rng.normal(size=(30, 2)),
        coverage=1.0,
    )
    detect(fit_pca(train), monitor)


# --------------------------------------------------------------------------
# NarratorUnavailable
# --------------------------------------------------------------------------
def x_endpoint_variable_unset(tmp_path: Path) -> None:
    from tsdive.narrate.base import RemoteNarrator
    from tsdive.narrate.ledger import EvidenceLedger

    saved = os.environ.pop("TSDIVE_LLM_ENDPOINT", None)
    try:
        RemoteNarrator().narrate(EvidenceLedger(title="fault injection"))
    finally:
        if saved is not None:
            os.environ["TSDIVE_LLM_ENDPOINT"] = saved


@dataclass(frozen=True)
class Case:
    """One known defect, the refusal it must raise, and the cause it must name."""

    defect: str
    error: type[TSDiveError]
    cause: str  # regex the message must match, and no sibling may match
    provoke: Callable[[Path], None]


CORPUS: list[Case] = [
    Case(
        "value column with no quality column",
        SchemaError,
        r"missing required column\(s\): quality",
        s_missing_quality_column,
    ),
    Case(
        "timestamps with no UTC offset",
        SchemaError,
        r"naive timestamps rejected at ingest",
        s_naive_timestamps,
    ),
    Case(
        "the same timestamp stored twice",
        SchemaError,
        r"duplicate timestamps present",
        s_duplicate_timestamps,
    ),
    Case(
        "parquet file carrying no tag metadata",
        SchemaError,
        r"no tsdive\.meta metadata",
        s_archive_without_metadata,
    ),
    Case(
        "string value on a tag not declared role=MODE",
        SchemaError,
        r"'RUN' is not numeric and no digital state",
        s_non_numeric_value,
    ),
    Case(
        "quality_codes naming a severity that does not exist",
        SchemaError,
        r"quality_codes maps 'SUB' to 'MAYBE'",
        s_quality_code_maps_to_no_severity,
    ),
    Case(
        "archive with zero rows asked for its extent",
        SchemaError,
        r"archive has no samples",
        s_archive_without_samples,
    ),
    Case(
        "every sample carries a BAD quality code",
        InsufficientQuality,
        r"no GOOD samples remain after masking",
        q_every_sample_bad,
    ),
    Case(
        "one sample, window end at that same instant",
        InsufficientQuality,
        r"window has zero duration",
        q_window_of_zero_duration,
    ),
    Case(
        "every sample sits at the top of the engineering range",
        InsufficientQuality,
        r"window is censored \(clipped fraction",
        q_window_pinned_at_full_scale,
    ),
    Case(
        "5 GOOD history samples against a floor of 30",
        InsufficientQuality,
        r"only 5 GOOD history samples",
        q_history_shorter_than_the_floor,
    ),
    Case(
        "history frame with no quality column",
        InsufficientQuality,
        r"history frame needs value\+quality columns",
        q_history_without_quality_column,
    ),
    Case(
        "12 usable samples against a min-size of 10 per side",
        InsufficientQuality,
        r"12 usable samples; 20 required to place a break",
        q_too_few_samples_to_place_a_break,
    ),
    Case(
        "40 usable samples all holding one value",
        InsufficientQuality,
        r"no spread to segment",
        q_flat_values_have_no_spread,
    ),
    Case(
        "history frame with no rows at all",
        InsufficientQuality,
        r"history has no rows",
        q_history_with_no_rows,
    ),
    # One cause: comparable_with() compares the calculation basis and
    # nothing else, so a mixed basis is the only way to reach this class.
    Case(
        "a time-weighted window compared with an event-weighted one",
        IncomparableSamplingError,
        r"p:A\.PV is TIME_WEIGHTED, p:B\.PV is EVENT_WEIGHTED",
        c_calculation_bases_differ,
    ),
    # The two position-based messages must differ in wording, not only in
    # the digits they print: a reader who cannot tell which basis the
    # positions count reads a correct-looking refusal against wrong rows.
    Case(
        "raw timestamp column stepping back at row 2",
        NonMonotonicIndex,
        r"non-monotonic timestamps at positions \[\d+\]; refusing to sort silently",
        n_raw_column_steps_backwards,
    ),
    Case(
        "stored archive stepping back at window position 2",
        NonMonotonicIndex,
        r"timestamp at window position 2 .* precedes the sample before it",
        n_archive_window_steps_backwards,
    ),
    Case(
        "GOOD samples stepping back around a BAD one",
        NonMonotonicIndex,
        r"GOOD-sample positions \[\d+\] \(counted over GOOD samples, not frame rows\)",
        n_good_samples_step_backwards,
    ),
    Case(
        "unit spelling absent from the alias table",
        UnresolvedUnitError,
        r"unit 'gpm-ish' is not in the alias table",
        u_unit_spelling_unknown,
    ),
    Case(
        "tag declaring no unit at all",
        UnresolvedUnitError,
        r"a_raw names no unit",
        u_no_unit_declared,
    ),
    Case(
        "Nm3/h against m3/h with no reference state declared",
        IncomparableUnitsError,
        r"reference-condition unit\(s\) \['Nm3/h'\]",
        v_reference_state_undeclared,
    ),
    Case(
        "volumetric flow against a temperature",
        IncomparableUnitsError,
        r"units 'm3/h' and 'degC' have different dimensions",
        v_dimensions_differ,
    ),
    Case(
        "regime R2 holding 3 GOOD samples against a floor of 20",
        RegimeTooSparse,
        r"regime 'R2' has 3 GOOD samples; 20 required",
        r_regime_below_the_sample_floor,
    ),
    Case(
        "screening a mode value that training never held",
        RegimeTooSparse,
        r"no baseline for unseen regime 'R9'",
        r_regime_never_seen_in_training,
    ),
    Case(
        "(asset, variable) pair present only in the fold under test",
        PopulationTooSparse,
        r"no training window for \('ASSET-B', 'P-TPT'\)",
        p_pair_absent_from_training_folds,
    ),
    Case(
        "3 training windows against a floor of 20",
        PopulationTooSparse,
        r"has 3 training window\(s\); 20 required",
        p_pair_below_the_window_floor,
    ),
    Case(
        "window under test carrying no distinct_count",
        PopulationTooSparse,
        r"the window under test has no distinct-value count",
        p_window_under_test_has_no_count,
    ),
    Case(
        "one well listed in both folds",
        GroupLeakage,
        r"appears in fold 0 and fold 1",
        g_group_reaches_two_folds,
    ),
    Case(
        "fold list placing a well the map assigns elsewhere",
        GroupLeakage,
        r"fold membership disagrees with fold_of_group",
        g_fold_list_disagrees_with_the_map,
    ),
    Case(
        "fold holding a well the map never mentions",
        GroupLeakage,
        r"folds contain groups absent from fold_of_group",
        g_fold_holds_a_group_outside_the_map,
    ),
    Case(
        "one window handed to a multivariate alignment",
        MspcAlignmentError,
        r"need at least two windows to align",
        m_only_one_window,
    ),
    Case(
        "two windows an hour apart",
        MspcAlignmentError,
        r"window time spans do not overlap",
        m_spans_do_not_overlap,
    ),
    Case(
        "second tag sampled 2 s off a 3 s grid",
        MspcAlignmentError,
        r"aligned coverage 0\.\d+ below required 0\.99",
        m_coverage_below_the_floor,
    ),
    Case(
        "two tags whose samples land on alternating grid points",
        MspcAlignmentError,
        r"no fully-observed aligned rows remain",
        m_no_row_is_fully_observed,
    ),
    Case(
        "monitoring matrix holding a tag the model never saw",
        MspcAlignmentError,
        r"monitor matrix columns differ from trained model",
        m_monitor_columns_differ_from_the_model,
    ),
    # One cause: base.py reads one environment variable, which is either
    # set or absent.
    Case(
        "narration requested with TSDIVE_LLM_ENDPOINT absent",
        NarratorUnavailable,
        r"TSDIVE_LLM_ENDPOINT is not set",
        x_endpoint_variable_unset,
    ),
]

# Classes the corpus covers with one case, because the source holds one
# cause for them. Both are checked against the raise sites above.
SINGLE_CAUSE = {IncomparableSamplingError, NarratorUnavailable}

MULTI_CAUSE = sorted(
    {c.error for c in CORPUS} - SINGLE_CAUSE, key=lambda e: e.__name__
)


def _message(case: Case, tmp_path: Path) -> str:
    with pytest.raises(case.error) as excinfo:
        case.provoke(tmp_path)
    return str(excinfo.value)


@pytest.mark.parametrize("case", CORPUS, ids=[c.provoke.__name__ for c in CORPUS])
def test_refusal_names_its_cause(case: Case, tmp_path: Path) -> None:
    with pytest.raises(case.error, match=case.cause):
        case.provoke(tmp_path)


@pytest.mark.parametrize("refusal", MULTI_CAUSE, ids=[e.__name__ for e in MULTI_CAUSE])
def test_causes_within_a_class_are_distinguishable(
    refusal: type[TSDiveError], tmp_path: Path
) -> None:
    """A cause phrase matches its own message and rejects its siblings'.

    Two defects that share a class and a message leave the reader with a
    refusal that looks right and points at the wrong data.
    """
    cases = [c for c in CORPUS if c.error is refusal]
    assert len(cases) >= 2, f"{refusal.__name__} is listed as multi-cause"
    messages = {c.provoke.__name__: _message(c, tmp_path) for c in cases}
    for case in cases:
        for other in cases:
            if other is case:
                continue
            assert not re.search(case.cause, messages[other.provoke.__name__]), (
                f"{case.provoke.__name__} and {other.provoke.__name__} raise "
                f"{refusal.__name__} with messages a reader cannot tell apart: "
                f"{messages[other.provoke.__name__]}"
            )


def test_every_refusal_type_has_a_cause_case() -> None:
    """A refusal type with no case here fails this suite; add one.

    ``TSDiveError`` is the base of every class below and has no raise
    site of its own, so the corpus covers its subclasses.
    """
    declared = {
        cls
        for cls in vars(errors_mod).values()
        if isinstance(cls, type) and issubclass(cls, TSDiveError) and cls is not TSDiveError
    }
    covered = {c.error for c in CORPUS}
    assert declared == covered, (
        f"refusal types with no cause case: {declared - covered}; "
        f"phantom cases: {covered - declared}"
    )
    thin = {e for e in covered if sum(c.error is e for c in CORPUS) < 2}
    assert thin == SINGLE_CAUSE, (
        f"classes carrying one case but not listed in SINGLE_CAUSE: {thin - SINGLE_CAUSE}"
    )


# --------------------------------------------------------------------------
# Anti-conflation: data a reader could plausibly blame on the neighbour
# --------------------------------------------------------------------------
def test_all_bad_regime_names_the_regime_not_the_whole_history(tmp_path: Path) -> None:
    """History whose every row is BAD, under one mode label.

    Plausible wrong blame: InsufficientQuality, "no GOOD samples". The
    refusal names the regime and its GOOD count instead, which is what
    tells the reader the floor is per regime.
    """
    history = annotate(_frame(_stamps(25), [float(i) for i in range(25)], ["BAD"] * 25))
    modes = pd.Series(["R1"] * 25, index=history.index)

    from tsdive.baselines.regime import regime_baselines

    with pytest.raises(RegimeTooSparse, match=r"regime 'R1' has 0 GOOD samples; 20 required"):
        regime_baselines(history, modes, min_samples=20)

    # The empty history is the case that belongs to InsufficientQuality,
    # and it says so in its own words.
    with pytest.raises(InsufficientQuality, match=r"history has no rows"):
        q_history_with_no_rows(tmp_path)


def test_sparse_population_and_sparse_regime_name_different_counts(tmp_path: Path) -> None:
    """Both refusals report 3 against a floor of 20 on the same variable.

    One counts GOOD samples inside a regime, the other counts training
    windows for an (asset, variable) pair. Each message states which.
    """
    with pytest.raises(RegimeTooSparse) as regime:
        r_regime_below_the_sample_floor(tmp_path)
    with pytest.raises(PopulationTooSparse) as population:
        p_pair_below_the_window_floor(tmp_path)

    assert re.search(r"regime 'R2' has 3 GOOD samples", str(regime.value))
    assert "training window" not in str(regime.value)
    assert re.search(r"has 3 training window\(s\)", str(population.value))
    assert "GOOD samples" not in str(population.value)


def test_repeated_timestamps_are_a_schema_defect_not_a_backwards_index(
    tmp_path: Path,
) -> None:
    """A repeated timestamp does not move backwards, so ordering is intact.

    ``backwards_positions`` compares strictly, so the duplicate reaches
    ``require_unique`` and gets the double-counting refusal.
    """
    from tsdive.store.timebase import ensure_monotonic, require_unique

    repeated = pd.Series([T0, T0, T0 + pd.Timedelta(3, unit="s")])
    ensure_monotonic(repeated)  # no refusal: equal is not backwards
    with pytest.raises(SchemaError, match=r"duplicate timestamps present"):
        require_unique(repeated)

    backwards = pd.Series(_BACKWARDS)
    require_unique(backwards)  # no refusal: every timestamp is distinct
    with pytest.raises(NonMonotonicIndex, match=r"non-monotonic timestamps at positions \[2\]"):
        ensure_monotonic(backwards)


def test_the_two_position_bases_are_named_where_they_differ(tmp_path: Path) -> None:
    """One backwards run, counted twice: frame row 2, GOOD sample 1.

    ``offending_positions`` carries the two bases and the messages say
    which one they count, so a reader sent to a position lands on the row
    that moved.
    """
    from tsdive.store.averaging import time_weighted_mean
    from tsdive.store.timebase import ensure_monotonic

    stamps = [
        T0 + pd.Timedelta(10, unit="s"),
        T0 + pd.Timedelta(20, unit="s"),
        T0 + pd.Timedelta(5, unit="s"),
    ]
    frame = annotate(_frame(stamps, [1.0, 2.0, 3.0], ["GOOD", "BAD", "GOOD"]))

    with pytest.raises(NonMonotonicIndex) as over_frame:
        ensure_monotonic(frame["timestamp"])
    with pytest.raises(NonMonotonicIndex) as over_good:
        time_weighted_mean(frame)

    assert over_frame.value.offending_positions == [2]
    assert over_good.value.offending_positions == [1]
    assert "GOOD-sample positions" not in str(over_frame.value)
    assert "GOOD-sample positions" in str(over_good.value)


def test_naive_and_backwards_archive_names_the_schema_defect(tmp_path: Path) -> None:
    """The archive carries both defects; the schema gate runs first.

    Naive timestamps place no instant, so the ordering claim cannot be
    checked until the offsets are fixed.
    """
    naive = [
        pd.Timestamp("2024-03-01 00:00:00"),
        pd.Timestamp("2024-03-01 00:00:10"),
        pd.Timestamp("2024-03-01 00:00:05"),
    ]
    with pytest.raises(SchemaError, match=r"naive timestamps rejected at ingest"):
        _window(
            tmp_path,
            "NAIVE.PV",
            _frame(naive, [1.0, 2.0, 3.0]),
            validate=False,
        )

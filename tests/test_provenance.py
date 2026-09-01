"""Fixture (e): replay verdicts MATCH / CODE_DRIFT / DATA_CHANGED / BOTH."""

from __future__ import annotations

import pandas as pd

from conftest import good_frame
from tsdive.provenance.record import (
    make_record,
    replay,
    save_record,
)
from tsdive.store.sampling_contract import CalculationBasis, RetrievalMode, SamplingContract

CONTRACT = SamplingContract(CalculationBasis.TIME_WEIGHTED, RetrievalMode.RECORDED, stepped=True)


def _frame():
    stamps = [
        "2024-03-01 00:00:00+00:00",
        "2024-03-01 00:00:03+00:00",
        "2024-03-01 00:00:06+00:00",
    ]
    return good_frame(stamps, [10.0, 11.0, 12.0])


def test_match_on_identical_inputs(tmp_path):
    record = make_record("r1", "p:T.PV", _frame(), CONTRACT, as_of="2024-03-02T00:00:00+00:00")
    save_record(record, tmp_path)
    result = replay("r1", _frame(), CONTRACT, tmp_path)
    assert result.verdict.value == "MATCH"
    assert result.moved_samples == []


def test_backfill_into_completed_window_is_data_changed(tmp_path):
    record = make_record("r2", "p:T.PV", _frame(), CONTRACT)
    save_record(record, tmp_path)

    late = _frame()
    backfilled = pd.concat(
        [
            late,
            good_frame(
                ["2024-03-01 00:00:04+00:00"],
                [11.5],
            ),
        ],
        ignore_index=True,
    )
    result = replay("r2", backfilled, CONTRACT, tmp_path)
    assert result.verdict.value == "DATA_CHANGED"
    moved = dict(result.moved_samples)
    assert "2024-03-01T00:00:04+00:00" in moved
    assert "added" in moved["2024-03-01T00:00:04+00:00"]


def test_value_rewrite_names_the_timestamp(tmp_path):
    record = make_record("r3", "p:T.PV", _frame(), CONTRACT)
    save_record(record, tmp_path)

    edited = _frame().copy()
    edited.loc[edited.index[1], "value"] = 99.0
    result = replay("r3", edited, CONTRACT, tmp_path)
    assert result.verdict.value == "DATA_CHANGED"
    moved = dict(result.moved_samples)
    assert "value/quality changed" in moved["2024-03-01T00:00:03+00:00"]


def test_version_bump_is_code_drift(tmp_path):
    record = make_record("r4", "p:T.PV", _frame(), CONTRACT, module_version="0.1.0")
    save_record(record, tmp_path)
    result = replay(
        "r4", _frame(), CONTRACT, tmp_path, module_version="0.2.0"
    )
    assert result.verdict.value == "CODE_DRIFT"


def test_contract_change_counts_as_drift(tmp_path):
    record = make_record("r5", "p:T.PV", _frame(), CONTRACT)
    save_record(record, tmp_path)
    other = SamplingContract(CalculationBasis.EVENT_WEIGHTED, RetrievalMode.RECORDED)
    result = replay("r5", _frame(), other, tmp_path)
    assert result.verdict.value == "CODE_DRIFT"


def test_backfill_and_bump_together_are_both(tmp_path):
    record = make_record("r6", "p:T.PV", _frame(), CONTRACT, module_version="0.1.0")
    save_record(record, tmp_path)
    backfilled = pd.concat(
        [_frame(), good_frame(["2024-03-01 00:00:09+00:00"], [13.0])],
        ignore_index=True,
    )
    result = replay(
        "r6", backfilled, CONTRACT, tmp_path, module_version="0.2.0"
    )
    assert result.verdict.value == "BOTH"
    assert result.moved_samples


def test_row_order_does_not_change_fingerprint():
    """Historians reorder rows; content is what we fingerprint."""
    a = make_record("r7", "p:T.PV", _frame(), CONTRACT)
    shuffled = _frame().iloc[::-1].reset_index(drop=True)
    b = make_record("r7b", "p:T.PV", shuffled, CONTRACT)
    assert a.data_fingerprint == b.data_fingerprint

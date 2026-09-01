from __future__ import annotations

import dataclasses

import pytest

from conftest import event_contract, stepped_contract
from tsdive.store.sampling_contract import (
    AggregateType,
    CalculationBasis,
    RetrievalMode,
    SamplingContract,
)

# Pinned constant: the digest scheme must not change silently, because
# provenance records store it across versions.
EXPECTED_STEPPED_DIGEST = "d6227cd3b5289d7d76c568b439c54597983db7a8a7ac47c0133ada1d75941461"


def test_digest_stable_across_processes():
    assert stepped_contract().digest() == EXPECTED_STEPPED_DIGEST


def test_digest_is_deterministic_rebuild():
    again = SamplingContract(
        calculation_basis=CalculationBasis.TIME_WEIGHTED,
        retrieval_mode=RetrievalMode.RECORDED,
        aggregate_type=AggregateType.NONE,
        stepped=True,
    )
    assert stepped_contract().digest() == again.digest()


def test_digest_changes_when_semantics_change():
    base = stepped_contract()
    assert base.digest() != event_contract().digest()
    assert base.digest() != dataclasses.replace(base, stepped=False).digest()
    assert base.digest() != dataclasses.replace(
        base, aggregate_type=AggregateType.OPC_UA_TIME_AVERAGE
    ).digest()


def test_frozen():
    c = stepped_contract()
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.stepped = False  # type: ignore[misc]


def test_comparable_only_on_calculation_basis():
    a = SamplingContract(CalculationBasis.TIME_WEIGHTED, RetrievalMode.RECORDED)
    b = SamplingContract(
        CalculationBasis.TIME_WEIGHTED, RetrievalMode.INTERPOLATED, AggregateType.OPC_UA_AVERAGE
    )
    c = SamplingContract(CalculationBasis.EVENT_WEIGHTED, RetrievalMode.RECORDED)
    assert a.comparable_with(b)
    assert not a.comparable_with(c)

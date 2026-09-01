"""Sampling contracts.

A window read must state *how* the historian was asked to produce the
numbers it returns. The field names mirror vendor enums on purpose:

- AVEVA PI Asset Framework ``CalculationBasis`` (TimeWeighted / EventWeighted)
- AVEVA PI AF ``RetrievalMode`` (Recorded / Interpolated)
- OPC UA Part 13 aggregate distinction (Average vs TimeAverage)
- PI ``Stepped`` interpolation flag

Two windows whose contracts differ describe different quantities. Mixing
them is refused with :class:`~tsdive.errors.IncomparableSamplingError`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum


class CalculationBasis(StrEnum):
    """PI AF CalculationBasis values relevant to a window read."""

    TIME_WEIGHTED = "TIME_WEIGHTED"
    EVENT_WEIGHTED = "EVENT_WEIGHTED"


class RetrievalMode(StrEnum):
    """How raw samples were retrieved from the archive."""

    RECORDED = "RECORDED"
    INTERPOLATED = "INTERPOLATED"


class AggregateType(StrEnum):
    """OPC UA Part 13 aggregate flavour, when the source is an aggregate."""

    NONE = "NONE"
    OPC_UA_AVERAGE = "OPC_UA_AVERAGE"
    OPC_UA_TIME_AVERAGE = "OPC_UA_TIME_AVERAGE"


@dataclass(frozen=True)
class SamplingContract:
    """Declares how a window's samples were produced.

    Passed as a required positional argument on every window read so a
    caller cannot accidentally omit the statement of provenance.
    """

    calculation_basis: CalculationBasis
    retrieval_mode: RetrievalMode
    aggregate_type: AggregateType = AggregateType.NONE
    stepped: bool = False

    def digest(self) -> str:
        """Stable hash across processes and platforms.

        Dataclass hashes are stable for StrEnum members, but we commit to
        an explicit SHA-256 over canonical text so the value can be stored
        in provenance records and compared years later.
        """
        payload = "|".join(
            [
                f"calculation_basis={self.calculation_basis.value}",
                f"retrieval_mode={self.retrieval_mode.value}",
                f"aggregate_type={self.aggregate_type.value}",
                f"stepped={self.stepped}",
            ]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def comparable_with(self, other: SamplingContract) -> bool:
        """True when two windows may be compared or combined.

        Only the calculation basis changes the *meaning* of aggregated
        numbers; retrieval mode and stepped-ness change sample placement.
        The store refuses cross-basis mixing outright.
        """
        return self.calculation_basis == other.calculation_basis

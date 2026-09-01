"""Typed refusals for tsdive.

Rule (enforced by tests/test_errors.py): every exception class in this
module has a provoking test. A refusal type without a test that raises it
is deleted, not kept as decoration.
"""

from __future__ import annotations


class TSDiveError(Exception):
    """Base class for every typed refusal raised by tsdive."""


class SchemaError(TSDiveError):
    """Input violates the mandatory schema.

    Raised for: missing quality column, naive timestamps at ingest,
    duplicate timestamps where uniqueness is required.
    """


class InsufficientQuality(TSDiveError):
    """A requested statistic has too few GOOD samples to compute.

    Raised when too few GOOD samples remain after masking (including the
    all-clipped case: a clipped window may never serve as a baseline).
    """


class IncomparableSamplingError(TSDiveError):
    """Windows produced under different sampling contracts were mixed.

    A time-weighted mean and an event-weighted mean are different
    quantities; comparing or combining them is refused.
    """


class NonMonotonicIndex(TSDiveError):
    """Timestamp index goes backwards.

    Raised instead of silently sorting. Carries the offending positions.
    """

    def __init__(self, message: str, offending_positions: list[int]) -> None:
        super().__init__(message)
        self.offending_positions = offending_positions


class UnresolvedUnitError(TSDiveError):
    """A unit string could not be resolved and comparison was attempted.

    Unknown units resolve to ``unit_canonical=None`` and are surfaced in
    reports, and no spelling is resolved by inference. This is raised only
    when an operation needs the canonical unit and finds none.
    """


class IncomparableUnitsError(TSDiveError):
    """Two tags' units cannot be compared.

    Raised on dimension mismatch, and on reference-condition units
    (Nm3/h, Sm3/h, scfm) whose declared reference state is missing.
    """


class RegimeTooSparse(TSDiveError):
    """A regime has too few GOOD samples to support a baseline.

    Raised by the refined baseliner; a sparse regime gets a refusal, not
    a quietly wide interval.
    """


class PopulationTooSparse(TSDiveError):
    """A cross-instance (asset, variable) baseline has too few windows.

    Raised by the population baseliner when the training folds hold fewer
    windows for that pair than the caller declared sufficient, or none at
    all. A tag frozen for its whole life cannot indict itself, and a
    population of three windows cannot indict it either.
    """


class GroupLeakage(TSDiveError):
    """A holdout group was found on both sides of a split.

    Raised by :meth:`~tsdive.data.splits.GroupSplit.leakage_check`. A
    number computed across a leaked group measures memorisation of that
    group's signature, so the split is refused rather than reported.
    """


class MspcAlignmentError(TSDiveError):
    """Multivariate alignment could not be assembled.

    Raised when tags cannot share a common UTC grid at the declared rate
    with sufficient coverage, or when contracts/units are incomparable.
    """


class NarratorUnavailable(TSDiveError):
    """LLM narration was requested with no endpoint configured.

    The narrator refuses unless TSDIVE_LLM_ENDPOINT is explicitly set;
    it never falls back to a hosted default.
    """

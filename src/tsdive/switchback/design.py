"""Balanced switchback schedules and the assignments a randomization test reads.

Settings A and B alternate in K time blocks of one unit. Exactly K/2 of
the blocks run setting B, chosen by balanced complete randomization.
``make_design`` returns the observed assignment and the reference set of
assignments its p-value is computed over: every balanced assignment when
there are at most 1000 of them, otherwise the observed one plus 1000
fresh seeded draws.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations

import numpy as np

from tsdive.errors import DesignTooSmall

ALPHA = 0.05
"""Level of the two-sided randomization test.

The interval holds the shifts the test accepts at this level. The power
readout counts the p-values at or below it. A design whose smallest
attainable p is above it is too small (``DesignTooSmall``).
"""
PERMUTATIONS = 1000
"""Largest reference set enumerated in full.

A design with more balanced assignments reads its p-value over the
observed assignment plus this many seeded draws.
"""
MIN_ASSIGNMENTS = 20
"""Fewest balanced assignments a design may have (``DesignTooSmall`` below it)."""

DESIGN_TOO_SMALL = "design_too_small"


def schedule_blocks(span: float, length: float) -> int:
    """K = floor(span / length), less one when it is odd."""
    k = math.floor(span / length)
    return k - (k % 2)


def design_size(k: int) -> tuple[int, bool, float]:
    """(balanced assignments, enumerated, smallest attainable two-sided p) of K blocks.

    An assignment and its complement give the same |T|, so an enumerated
    design reaches 2 / C(K, K/2) at best; a sampled one reaches 1 / 1001.
    """
    if k < 2:
        return 0, True, 1.0
    n = math.comb(k, k // 2)
    if n <= PERMUTATIONS:
        return n, True, 2.0 / n
    return n, False, 1.0 / (PERMUTATIONS + 1)


def order_of_magnitude(n: int) -> int:
    """The largest E with 10**E <= n, from the bit length, for counts too long to print."""
    return math.floor((n.bit_length() - 1) * math.log10(2.0))


def design_refusal(k: int) -> str:
    """``design_too_small`` when the design has too few assignments or too coarse a p."""
    n, _, min_p = design_size(k)
    return DESIGN_TOO_SMALL if n < MIN_ASSIGNMENTS or min_p > ALPHA else ""


@lru_cache(maxsize=16)
def all_assignments(k: int) -> np.ndarray:
    """(C(K, K/2), K) int8 balanced assignments, B positions in lexicographic order."""
    rows = np.zeros((math.comb(k, k // 2), k), dtype=np.int8)
    for i, chosen in enumerate(combinations(range(k), k // 2)):
        rows[i, list(chosen)] = 1
    rows.setflags(write=False)
    return rows


def balanced_draws(rng: np.random.Generator, k: int, count: int) -> np.ndarray:
    """(count, K) int8 assignments, each uniform over the balanced ones."""
    order = np.argsort(rng.random((count, k)), axis=1, kind="stable")
    out = np.zeros((count, k), dtype=np.int8)
    np.put_along_axis(out, order[:, : k // 2], 1, axis=1)
    return out


@dataclass(frozen=True)
class Design:
    """The observed assignment of K blocks and the assignments its p-value reads."""

    k: int
    observed: np.ndarray  # (K,) int8, 1 marks a B block
    assignments: np.ndarray  # (R, K) int8
    matrix: np.ndarray  # (R, K) float64 copy of ``assignments``
    enumerated: bool
    n_assignments: int
    min_p: float


def make_design(k: int, seed: int | Sequence[int]) -> Design:
    """Observed assignment and reference set, deterministic in ``seed``.

    Enumerated designs list every balanced assignment, the observed one
    included; larger designs draw the observed one and then 1000 fresh
    ones. ``seed`` is anything ``numpy.random.default_rng`` accepts as a
    seed, and the same seed gives the same design on every platform.

    Raises:
        DesignTooSmall: fewer than 20 balanced assignments, or a smallest
            attainable two-sided p above 0.05.
    """
    if design_refusal(k):
        n, _, min_p = design_size(k)
        defect = (
            f"{k} blocks give {n} balanced assignments, fewer than {MIN_ASSIGNMENTS}"
            if n < MIN_ASSIGNMENTS
            else f"{k} blocks reach a smallest two-sided p of {min_p:.4f}, above {ALPHA}"
        )
        raise DesignTooSmall(f"{defect}; use more, shorter blocks")
    n, enumerated, min_p = design_size(k)
    rng = np.random.default_rng(seed)
    if enumerated:
        rows = all_assignments(k)
        observed = rows[int(rng.integers(n))].copy()
    else:
        observed = balanced_draws(rng, k, 1)[0]
        rows = balanced_draws(rng, k, PERMUTATIONS)
    matrix = rows.astype(float)
    matrix.setflags(write=False)
    return Design(k, observed, rows, matrix, enumerated, n, min_p)


@dataclass(frozen=True)
class Blocks:
    """Block of each sample (-1 outside the schedule) and its offset from the block start.

    ``offset`` and ``length`` share one time unit, the unit ``washout`` is
    stated in wherever a frame is cut from these blocks.
    """

    k: int
    length: float
    block: np.ndarray
    offset: np.ndarray


def cut_blocks(times: np.ndarray, span: float, length: float) -> Blocks:
    """Blocks of ``length`` from time 0, K = ``schedule_blocks(span, length)``.

    ``times`` are measured from the schedule start in the unit of
    ``length``. A time before 0 or at or after K * ``length`` is outside
    the schedule.
    """
    k = schedule_blocks(span, length)
    t = np.asarray(times, dtype=float)
    index = np.floor(t / length).astype(np.int64)
    offset = t - index * length
    block = np.where((index >= 0) & (index < k), index, -1)
    return Blocks(k, float(length), block, offset)


def setting(blocks: Blocks, observed: np.ndarray) -> np.ndarray:
    """Per-sample setting: 1 in B blocks, 0 in A blocks and outside the schedule."""
    inside = blocks.block >= 0
    return np.where(inside, observed[np.where(inside, blocks.block, 0)], 0).astype(float)

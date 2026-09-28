"""Randomized switchback schedules on one record: schedule, injected response, inference arms.

Settings A and B alternate in time blocks of one record. ``cut_blocks``
cuts K = floor(span / L) blocks from the first sample (the last block is
dropped when K is odd), and ``make_design`` assigns exactly K/2 of them to
B by balanced complete randomization. A design with fewer than 20
balanced assignments, or whose smallest attainable two-sided p exceeds
0.05, is refused (``design_too_small``). Times, L, tau and the washout w
are in steps of the record's nominal sampling interval.

The injected response is u = delta * sigma in B blocks and 0 in A blocks,
passed through a first-order lag with time constant tau (``lag_response``).
The setting of a sample applies from the previous sample to it: on a
regular record s_t = s_{t-1} + (u_t - s_{t-1}) (1 - exp(-1 / tau)), and a
gap of dt steps moves the response 1 - exp(-dt / tau) of the way. Washout
drops the samples in the first w steps of every block.

Inference arms on the kept samples, 95% two-sided:

- ``randomization``: the coefficient of z in OLS of y on [1, z] (the
  difference in means, B minus A), or on [1, z, X] with X centred, and its
  randomization p-value over balanced assignments: all of them when there
  are at most 1000, else 1000 fresh draws with p = (1 + #{|T_j| >= |T|}) /
  1001. Frisch-Waugh keeps every assignment in block space: with Q an
  orthonormal basis of [1, X] over the kept samples and Q_B its block
  sums, z'Mz = z.n - |Q_B'z|^2 and z'My = z.S with S the block sums of My.
  The interval inverts the test under a constant additive shift: for each
  assignment the condition |a_j - theta c_j| >= |T - theta| is a
  quadratic in theta, and the interval is the hull of the thetas where
  enough of them hold.
- ``block_t``: Welch t on the kept block means of the fitted setting term
  plus the OLS residual (the block means of y on the raw arm), with the
  Welch-Satterthwaite degrees of freedom.
- ``hac``, ``ewc``, ``naive``: the same OLS coefficient, with the standard
  error from the long-run variance of the score (M z) e: Newey-West with
  the Andrews bandwidth and a normal critical value, the equal-weighted
  cosine estimator with t on nu degrees of freedom, or independent samples.
- ``prepost_hac``, ``prepost_ewc``: the first half of the schedule span as
  A and the second half as B, analysed by ``shift.level_interval`` as the
  shift interval study does, raw or adjusted by a before-period fit, with
  that study's hard refusals.

The long-run variance estimators, the before/after interval, the
adjustment and the refusal rules come from
``examples/studies/shift_intervals/shift.py``, imported rather than
copied. ``ewc_long_run_variance`` evaluates the same cosine sum as
``shift.ewc_long_run_variance`` through a type II DCT, so a 50,000-sample
record needs no (nu, n) basis in memory.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import fft as sfft
from scipy import stats

HERE = Path(__file__).resolve().parent
SHIFT_DIR = HERE.parent / "shift_intervals"
if str(SHIFT_DIR) not in sys.path:
    sys.path.insert(0, str(SHIFT_DIR))

import shift as sh  # noqa: E402

ALPHA = 0.05
PERMUTATIONS = 1000
MIN_ASSIGNMENTS = 20
MAD_TO_SD = 1.4826
MIN_SAMPLES = sh.MIN_SAMPLES
SAMPLES_PER_COVARIATE = sh.SAMPLES_PER_COVARIATE
MIN_BLOCKS_PER_SETTING = 2
# Singular values below this fraction of the largest mark a covariate that
# repeats a combination of the others; its direction is dropped.
RANK_RTOL = 1e-10
# |T_j| within this fraction of |T| + sigma counts as a tie: the complement of
# the observed assignment gives -T up to float rounding.
TIE_RTOL = 1e-9
# z'Mz below this fraction of n: the assignment lies in the span of [1, X].
DENOM_RTOL = 1e-9
QUADRATIC_ATOL = 1e-12

RANDOMIZATION = "randomization"
BLOCK_T = "block_t"
HAC = "hac"
EWC = "ewc"
NAIVE = "naive"
PREPOST_HAC = "prepost_hac"
PREPOST_EWC = "prepost_ewc"
SCHEDULE_ARMS = (RANDOMIZATION, BLOCK_T, HAC, EWC, NAIVE)
PREPOST_ARMS = (PREPOST_HAC, PREPOST_EWC)
ARMS = SCHEDULE_ARMS + PREPOST_ARMS
PREPOST_METHOD = {PREPOST_HAC: sh.HAC, PREPOST_EWC: sh.EWC}
RAW = sh.RAW
ADJUSTED = sh.ADJUSTED
ADJUSTMENTS = (RAW, ADJUSTED)

DESIGN_TOO_SMALL = "design_too_small"
NO_SPREAD = sh.NO_SPREAD
TOO_FEW = sh.TOO_FEW
NO_COVARIATE = sh.NO_COVARIATE
TOO_MANY_COVARIATES = sh.TOO_MANY_COVARIATES
COVARIATE_OUTSIDE = sh.COVARIATE_OUTSIDE
TOO_FEW_BLOCKS = "too_few_blocks"
COLLINEAR = "collinear"
REASONS = (
    "",
    DESIGN_TOO_SMALL,
    NO_SPREAD,
    TOO_FEW,
    NO_COVARIATE,
    TOO_MANY_COVARIATES,
    COVARIATE_OUTSIDE,
    TOO_FEW_BLOCKS,
    COLLINEAR,
)


# ---------------------------------------------------------------- schedule


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


def make_design(k: int, seed: int) -> Design:
    """Observed assignment and reference set, deterministic in ``seed``.

    Enumerated designs list every balanced assignment, the observed one
    included; larger designs draw the observed one and then 1000 fresh ones.
    Raises ``ValueError`` on a refused design.
    """
    if design_refusal(k):
        raise ValueError(f"{DESIGN_TOO_SMALL}: K = {k}")
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
    """Block of each sample (-1 outside the schedule) and its offset from the block start."""

    k: int
    length: float
    block: np.ndarray
    offset: np.ndarray


def cut_blocks(times: np.ndarray, span: float, length: float) -> Blocks:
    """Blocks of ``length`` steps from time 0; ``times`` are steps from the first sample."""
    k = schedule_blocks(span, length)
    index = np.floor(np.asarray(times, dtype=float) / length).astype(np.int64)
    offset = np.asarray(times, dtype=float) - index * length
    block = np.where(index < k, index, -1)
    return Blocks(k, float(length), block, offset)


def setting(blocks: Blocks, observed: np.ndarray) -> np.ndarray:
    """Per-sample setting: 1 in B blocks, 0 in A blocks and outside the schedule."""
    inside = blocks.block >= 0
    return np.where(inside, observed[np.where(inside, blocks.block, 0)], 0).astype(float)


def washout_steps(tau: float) -> int:
    """ceil(3 tau), the registered washout: the lag settles to within exp(-3)."""
    return math.ceil(3.0 * tau - 1e-9) if tau > 0 else 0


# ---------------------------------------------------------------- response


def lag_response(u: np.ndarray, times: np.ndarray, tau: float) -> np.ndarray:
    """First-order lag response to the per-sample setting ``u``, starting from 0.

    Over the gap dt from the previous sample the response moves
    1 - exp(-dt / tau) of the way to the sample's setting; the first sample
    follows a gap of one step. ``tau`` 0 returns ``u``. Runs of equal
    setting are solved in closed form, so the loop runs once per switch.
    """
    u = np.asarray(u, dtype=float)
    if tau == 0 or u.size == 0:
        return u.copy()
    t = np.asarray(times, dtype=float)
    change = np.flatnonzero(np.diff(u)) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [u.size]])
    s = np.empty_like(u)
    prev_s, prev_t = 0.0, t[0] - 1.0
    for i0, i1 in zip(starts, ends, strict=True):
        level = u[i0]
        s[i0:i1] = level + (prev_s - level) * np.exp(-(t[i0:i1] - prev_t) / tau)
        prev_s, prev_t = s[i1 - 1], t[i1 - 1]
    return s


# ---------------------------------------------------------------- frames


def orthonormal_basis(x: np.ndarray | None, n: int) -> np.ndarray:
    """(n, p) orthonormal basis of [1, X centred], dropping repeated directions."""
    if x is None or x.shape[1] == 0:
        return np.full((n, 1), 1.0 / math.sqrt(n))
    centred = x - x.mean(axis=0)
    sd = centred.std(axis=0)
    centred = centred / np.where(sd > 0, sd, 1.0)
    design = np.column_stack([np.ones(n), centred])
    u, sv, _ = np.linalg.svd(design, full_matrices=False)
    return u[:, sv > RANK_RTOL * sv[0]]


def _block_sums(values: np.ndarray, starts: np.ndarray, nonempty: np.ndarray) -> np.ndarray:
    out = np.zeros((len(nonempty), *values.shape[1:]))
    if starts.size:
        out[nonempty] = np.add.reduceat(values, starts, axis=0)
    return out


@dataclass(frozen=True)
class Frame:
    """Kept samples of one target under one washout and one adjustment, in block space."""

    positions: np.ndarray  # kept sample positions in the record, time order
    block: np.ndarray  # block of each kept sample
    counts: np.ndarray  # (K,) kept samples per block
    starts: np.ndarray  # first kept position of each non-empty block
    nonempty: np.ndarray  # (K,) bool
    basis: np.ndarray  # (n, p) orthonormal basis of [1, X centred]
    basis_sums: np.ndarray  # (K, p) block sums of ``basis``
    y: np.ndarray  # kept target values
    resid: np.ndarray  # M y
    resid_sums: np.ndarray  # (K,) block sums of M y
    n_covariates: int

    @property
    def n(self) -> int:
        return len(self.positions)

    def sums(self, values: np.ndarray) -> np.ndarray:
        """(K,) or (K, m) block sums of per-sample values over the kept samples."""
        return _block_sums(values, self.starts, self.nonempty)

    def project_out(self, values: np.ndarray) -> np.ndarray:
        """M values: the residual of ``values`` on [1, X centred]."""
        return values - self.basis @ (self.basis.T @ values)


def make_frame(y: np.ndarray, x: np.ndarray | None, blocks: Blocks, washout: float) -> Frame | str:
    """The kept samples and block-space pieces, or the refusal reason.

    Kept: inside the schedule, at least ``washout`` steps into the block,
    target and every covariate finite. Refusals: fewer than 30 kept
    samples (``too_few``), a kept target with MAD 0 or a residual whose MAD
    is float noise (``no_spread``), more than one covariate per 10 kept
    samples (``too_many_covariates``).
    """
    mask = (blocks.block >= 0) & (blocks.offset >= washout) & np.isfinite(y)
    k_cov = 0
    if x is not None:
        mask &= np.all(np.isfinite(x), axis=1)
        k_cov = x.shape[1]
    positions = np.flatnonzero(mask)
    n = len(positions)
    if n < MIN_SAMPLES:
        return TOO_FEW
    kept_y = np.asarray(y, dtype=float)[positions]
    spread = sh.mad(kept_y)
    if spread == 0:
        return NO_SPREAD
    if k_cov > n / SAMPLES_PER_COVARIATE:
        return TOO_MANY_COVARIATES
    basis = orthonormal_basis(None if x is None else x[positions], n)
    resid = kept_y - basis @ (basis.T @ kept_y)
    if x is not None and sh.mad(resid) <= sh.RESIDUAL_MAD_RTOL * spread:
        return NO_SPREAD
    block = blocks.block[positions]
    counts = np.bincount(block, minlength=blocks.k).astype(float)
    nonempty = counts > 0
    starts = np.flatnonzero(np.diff(block, prepend=-1))
    return Frame(
        positions=positions,
        block=block,
        counts=counts,
        starts=starts,
        nonempty=nonempty,
        basis=basis,
        basis_sums=_block_sums(basis, starts, nonempty),
        y=kept_y,
        resid=resid,
        resid_sums=_block_sums(resid, starts, nonempty),
        n_covariates=k_cov,
    )


# ---------------------------------------------------------------- randomization


@dataclass(frozen=True)
class Terms:
    """The pieces of one observed assignment that every shift and response reuse."""

    denom: np.ndarray  # (R,) z'Mz of each reference assignment
    valid: np.ndarray  # (R,) bool, z'Mz > 0
    unit: np.ndarray  # (R,) bool, the observed assignment or its complement on kept blocks
    c: np.ndarray  # (R,) z'M z_obs / z'Mz
    denom_obs: float
    z_tilde: np.ndarray  # (n,) M z_obs over the kept samples
    z_kept: np.ndarray  # (n,) z_obs over the kept samples


def assignment_terms(frame: Frame, design: Design) -> Terms | str:
    """Frisch-Waugh terms of the observed assignment, or ``collinear``."""
    obs = design.observed.astype(float)
    zq = design.matrix @ frame.basis_sums
    oq = obs @ frame.basis_sums
    denom = design.matrix @ frame.counts - np.einsum("ij,ij->i", zq, zq)
    denom_obs = float(obs @ frame.counts - oq @ oq)
    floor = DENOM_RTOL * frame.n
    if denom_obs <= floor:
        return COLLINEAR
    valid = denom > floor
    cross = design.matrix @ (frame.counts * obs) - zq @ oq
    c = np.divide(cross, denom, out=np.zeros_like(cross), where=valid)
    kept = frame.nonempty
    rows = design.assignments[:, kept]
    same = np.all(rows == design.observed[kept], axis=1)
    complement = np.all(rows == 1 - design.observed[kept], axis=1)
    z_kept = obs[frame.block]
    z_tilde = z_kept - frame.basis @ oq
    return Terms(denom, valid, same | complement, c, denom_obs, z_tilde, z_kept)


def _reference(design: Design) -> tuple[int, int]:
    """(offset, denominator) of the p-value: the observed assignment is added to fresh draws."""
    if design.enumerated:
        return 0, design.n_assignments
    return 1, PERMUTATIONS + 1


def accept_count(design: Design) -> int:
    """Smallest count of reference assignments at least as extreme that gives p > alpha."""
    offset, total = _reference(design)
    return math.floor(ALPHA * total - offset + 1e-12) + 1


def _row_sets(
    a: np.ndarray, c: np.ndarray, t: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per (assignment, shift) the thetas where |a - theta c| >= |t - theta|.

    The set is [l1, h1] united with [l2, h2]; ``second`` marks where the
    second piece exists. (a - theta c)^2 - (t - theta)^2 is quadratic in
    theta with leading coefficient c^2 - 1 and is >= 0 at theta = t.
    """
    lead = np.broadcast_to((c * c - 1.0)[:, None], a.shape)
    lin = -2.0 * (a * c[:, None] - t[None, :])
    const = a * a - t[None, :] ** 2
    disc = np.maximum(lin * lin - 4.0 * lead * const, 0.0)
    root = np.sqrt(disc)
    inf = np.inf
    l1 = np.full(a.shape, -inf)
    h1 = np.full(a.shape, inf)
    l2 = np.full(a.shape, inf)
    h2 = np.full(a.shape, inf)
    second = np.zeros(a.shape, dtype=bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        neg = lead < -QUADRATIC_ATOL
        l1 = np.where(neg, (-lin + root) / (2.0 * lead), l1)
        h1 = np.where(neg, (-lin - root) / (2.0 * lead), h1)
        pos = (lead > QUADRATIC_ATOL) & (disc > 0)
        h1 = np.where(pos, (-lin - root) / (2.0 * lead), h1)
        l2 = np.where(pos, (-lin + root) / (2.0 * lead), l2)
        second = pos
        flat = np.abs(lead) <= QUADRATIC_ATOL
        up = flat & (lin > 0)
        down = flat & (lin < 0)
        l1 = np.where(up, -const / lin, l1)
        h1 = np.where(down, -const / lin, h1)
    return l1, h1, l2, h2, second


def invert(
    a: np.ndarray, c: np.ndarray, t: np.ndarray, weight: np.ndarray, always: int, needed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Hull of the thetas where at least ``needed`` reference assignments are as extreme.

    ``weight`` (R,) is 1 for an assignment whose set is computed, 0 for one
    left out; ``always`` counts the assignments extreme at every theta.
    """
    m = a.shape[1]
    if always >= needed:
        return np.full(m, -np.inf), np.full(m, np.inf)
    l1, h1, l2, h2, second = _row_sets(a, c, t)
    w = weight[:, None].astype(np.int64)
    w2 = w * second
    pos = np.concatenate([l1, h1, l2, h2], axis=0).T  # (m, 4R)
    step = np.concatenate([np.broadcast_to(w, a.shape), -np.broadcast_to(w, a.shape), w2, -w2])
    step = step.T
    kind = np.concatenate(
        [np.zeros(a.shape), np.ones(a.shape), np.zeros(a.shape), np.ones(a.shape)]
    ).T
    order = np.lexsort((kind, pos), axis=-1)
    pos_s = np.take_along_axis(pos, order, axis=-1)
    step_s = np.take_along_axis(step, order, axis=-1)
    running = np.cumsum(step_s, axis=-1) + always
    rising = (step_s > 0) & (running >= needed)
    falling = (step_s < 0) & (running - step_s >= needed)
    lo = np.where(rising.any(axis=-1), pos_s[np.arange(m), rising.argmax(axis=-1)], np.nan)
    last = pos.shape[1] - 1 - falling[:, ::-1].argmax(axis=-1)
    hi = np.where(falling.any(axis=-1), pos_s[np.arange(m), last], np.nan)
    return lo, hi


@dataclass(frozen=True)
class Result:
    """Estimates, 95% intervals and p-values of one arm for m shifts or responses."""

    estimate: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    p_value: np.ndarray
    reason: str = ""

    @property
    def claims(self) -> np.ndarray:
        return (self.lo > 0) | (self.hi < 0)


def randomization(design: Design, terms: Terms, sums: np.ndarray, scale: float) -> Result:
    """Randomization estimate, p-value and inverted interval for each column of ``sums``.

    ``sums`` (K, m) holds block sums of M y for m versions of the target.
    """
    obs = design.observed.astype(float)
    t = (obs @ sums) / terms.denom_obs
    raw = design.matrix @ sums
    a = np.divide(
        raw,
        terms.denom[:, None],
        out=np.zeros_like(raw),
        where=terms.valid[:, None],
    )
    tol = TIE_RTOL * (np.abs(t) + scale)
    extreme = (np.abs(a) >= (np.abs(t) - tol)[None, :]) & terms.valid[:, None]
    extreme |= terms.unit[:, None]
    offset, total = _reference(design)
    p = (offset + extreme.sum(axis=0)) / total
    counted = terms.valid & ~terms.unit
    lo, hi = invert(
        a, terms.c, t, counted.astype(np.int64), int(terms.unit.sum()), accept_count(design)
    )
    return Result(t, lo, hi, p)


def exact_rejection(frame: Frame, k: int, scale: float) -> float:
    """Share of all balanced assignments the test rejects when each one is the observed one.

    Every assignment's p-value is the enumerated one of ``randomization``,
    #{j: |T_j| >= |T_i|} / C, with its tie tolerance; the assignment and its
    complement on the kept blocks count as ties. An assignment in the span
    of [1, X] is refused by the test, so it does not count as a rejection.
    """
    rows = all_assignments(k)
    matrix = rows.astype(float)
    zq = matrix @ frame.basis_sums
    denom = matrix @ frame.counts - np.einsum("ij,ij->i", zq, zq)
    valid = denom > DENOM_RTOL * frame.n
    raw = matrix @ frame.resid_sums
    t = np.abs(np.divide(raw, denom, out=np.zeros_like(raw), where=valid))
    tol = TIE_RTOL * (t + scale)
    extreme = (t[None, :] >= (t - tol)[:, None]) & valid[None, :]
    kept = rows[:, frame.nonempty]
    extreme |= np.all(kept[:, None, :] == kept[None, :, :], axis=-1)
    extreme |= np.all(kept[:, None, :] == 1 - kept[None, :, :], axis=-1)
    p = extreme.sum(axis=1) / len(rows)
    return float(np.mean((p <= ALPHA) & valid))


def randomization_p(frame: Frame, design: Design, y: np.ndarray, scale: float) -> float:
    """Randomization p-value of the kept samples ``y`` refitted from scratch (a reference path)."""
    resid = frame.project_out(y)
    terms = assignment_terms(frame, design)
    if isinstance(terms, str):
        raise ValueError(terms)
    return float(randomization(design, terms, frame.sums(resid)[:, None], scale).p_value[0])


# ---------------------------------------------------------------- sample-level arms


def ewc_long_run_variance(u: np.ndarray) -> tuple[np.ndarray, int]:
    """``shift.ewc_long_run_variance`` of each column of u, through a type II DCT."""
    u = np.asarray(u, dtype=float)
    n = u.shape[0]
    nu = sh.ewc_terms(n)
    centred = u - u.mean(axis=0)
    projections = sfft.dct(centred, type=2, axis=0)[1 : nu + 1] * (math.sqrt(2.0 / n) / 2.0)
    return (projections**2).sum(axis=0) / nu, nu


@lru_cache(maxsize=4096)
def t_critical(df: int) -> float:
    return float(stats.t.ppf(1 - ALPHA / 2, df))


def _normal(estimate: np.ndarray, se: np.ndarray) -> Result:
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(se > 0, np.abs(estimate) / se, np.inf)
    p = np.array([sh.normal_p(float(v)) for v in z])
    return Result(estimate, estimate - sh.Z95 * se, estimate + sh.Z95 * se, p)


def sample_arms(frame: Frame, terms: Terms, resid: np.ndarray) -> dict[str, Result]:
    """``naive``, ``hac``, ``ewc`` and ``block_t`` for each column of ``resid`` (M y, (n, m))."""
    n, p_basis = frame.basis.shape
    m = resid.shape[1]
    estimate = (terms.z_tilde @ resid) / terms.denom_obs
    e = resid - terms.z_tilde[:, None] * estimate[None, :]
    out: dict[str, Result] = {}

    dof = n - p_basis - 1
    s2 = (e * e).sum(axis=0) / dof
    out[NAIVE] = _normal(estimate, np.sqrt(s2 / terms.denom_obs))

    score = terms.z_tilde[:, None] * e
    lrv = np.array([sh.long_run_variance(score[:, j]) for j in range(m)])
    out[HAC] = _normal(estimate, np.sqrt(n * np.maximum(lrv, 0.0)) / terms.denom_obs)

    omega, nu = ewc_long_run_variance(score)
    se = np.sqrt(n * omega) / terms.denom_obs
    crit = t_critical(nu)
    with np.errstate(divide="ignore", invalid="ignore"):
        tstat = np.where(se > 0, np.abs(estimate) / se, np.inf)
    out[EWC] = Result(
        estimate, estimate - crit * se, estimate + crit * se, 2.0 * stats.t.sf(tstat, nu)
    )

    out[BLOCK_T] = _block_t(frame, terms, e, estimate)
    return out


def _block_t(frame: Frame, terms: Terms, e: np.ndarray, estimate: np.ndarray) -> Result:
    m = e.shape[1]
    counts = frame.counts[frame.nonempty]
    z_block = terms.z_kept[frame.starts]
    means = frame.sums(e)[frame.nonempty] / counts[:, None] + z_block[:, None] * estimate
    in_b, in_a = z_block == 1, z_block == 0
    nb, na = int(in_b.sum()), int(in_a.sum())
    if nb < MIN_BLOCKS_PER_SETTING or na < MIN_BLOCKS_PER_SETTING:
        nan = np.full(m, np.nan)
        return Result(nan, nan, nan, nan, TOO_FEW_BLOCKS)
    mb, ma = means[in_b], means[in_a]
    diff = mb.mean(axis=0) - ma.mean(axis=0)
    vb = mb.var(axis=0, ddof=1) / nb
    va = ma.var(axis=0, ddof=1) / na
    se = np.sqrt(vb + va)
    with np.errstate(divide="ignore", invalid="ignore"):
        df = (vb + va) ** 2 / (vb**2 / (nb - 1) + va**2 / (na - 1))
        df = np.where(np.isfinite(df), df, float(nb + na - 2))
        tstat = np.where(se > 0, np.abs(diff) / se, np.inf)
    crit = stats.t.ppf(1 - ALPHA / 2, df)
    return Result(diff, diff - crit * se, diff + crit * se, 2.0 * stats.t.sf(tstat, df))


# ---------------------------------------------------------------- before/after reference


def prepost(
    before: np.ndarray,
    after: np.ndarray,
    method: str,
    x_before: np.ndarray | None = None,
    x_after: np.ndarray | None = None,
) -> tuple[float, float, float, float] | str:
    """(estimate, lo, hi, p) of ``shift.level_interval`` on the two periods, or the refusal.

    Without covariates the target is read raw after the shift study's R0.
    With covariates the rows with every value finite are kept, R0, the
    covariate count rule and R2 apply, and the interval reads the residuals
    of a before-period fit (``shift.adjust``).
    """
    if x_before is None:
        reason = sh.usable(before, after)
        if reason:
            return reason
        interval = sh.level_interval(before[np.isfinite(before)], after[np.isfinite(after)], method)
        return interval.estimate, interval.lo, interval.hi, interval.p_value
    k = x_before.shape[1]
    if k == 0:
        return NO_COVARIATE
    fb = np.isfinite(before) & np.all(np.isfinite(x_before), axis=1)
    fa = np.isfinite(after) & np.all(np.isfinite(x_after), axis=1)
    yb, ya, xb, xa = before[fb], after[fa], x_before[fb], x_after[fa]
    if len(yb) < MIN_SAMPLES or len(ya) < MIN_SAMPLES:
        return TOO_FEW
    spread = sh.mad(yb)
    if spread == 0:
        return NO_SPREAD
    if k > len(yb) / SAMPLES_PER_COVARIATE:
        return TOO_MANY_COVARIATES
    if sh.covariate_outside(xb, xa):
        return COVARIATE_OUTSIDE
    resid_b, resid_a, _ = sh.adjust(yb, ya, xb, xa)
    if sh.mad(resid_b) <= sh.RESIDUAL_MAD_RTOL * spread:
        return NO_SPREAD
    interval = sh.level_interval(resid_b, resid_a, method)
    return interval.estimate, interval.lo, interval.hi, interval.p_value

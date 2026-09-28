"""Randomization inference on the kept samples of one switchback schedule.

The statistic is the coefficient of z (1 in B blocks, 0 in A blocks) in
OLS of the target on [1, z], the difference in means B minus A, or on
[1, z, X] with X the centred covariates. Its p-value is the share of
reference assignments whose statistic is at least as far from 0 as the
observed one: #/C over all C balanced assignments when the design is
enumerated, (1 + #) / 1001 over 1000 fresh draws when it is sampled.

Frisch-Waugh keeps every assignment in block space. With Q an
orthonormal basis of [1, X] over the kept samples and Q_B its block
sums, z'Mz = z.n - |Q_B'z|^2 and z'My = z.S, with n the kept samples per
block and S the block sums of My. Every reference assignment is refitted
without touching the samples again.

The 95% interval inverts the test under a constant additive shift: for
each assignment the condition |a_j - theta c_j| >= |T - theta| is a
quadratic in theta, and the interval is the hull of the thetas where
enough of them hold. A side on which the accepted set does not close is
reported as unbounded (an infinite bound), never as a number.

Empty blocks: a block with no kept sample has block sums of 0, so the
statistic of an assignment reads only the non-empty blocks. An
assignment that equals the observed one, or its complement, on the
non-empty blocks gives the same |T| up to rounding and counts as at
least as extreme (``Terms.unit``). ``analyze`` refuses a frame with an
empty block (``empty_block``), so the design size a plan states is the
size its p-value is read over.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from tsdive.switchback.design import ALPHA, PERMUTATIONS, Blocks, Design, all_assignments

MAD_TO_SD = 1.4826
MIN_SAMPLES = 30
SAMPLES_PER_COVARIATE = 10
# A residual MAD at or below this fraction of the target's MAD is float
# noise: the covariates reproduce the target.
RESIDUAL_MAD_RTOL = 1e-9
# Singular values below this fraction of the largest mark a covariate that
# repeats a combination of the others; its direction is dropped.
RANK_RTOL = 1e-10
# |T_j| within this fraction of |T| + scale counts as a tie: the complement
# of the observed assignment gives -T up to float rounding.
TIE_RTOL = 1e-9
# z'Mz below this fraction of n: the assignment lies in the span of [1, X].
DENOM_RTOL = 1e-9
QUADRATIC_ATOL = 1e-12

TOO_FEW = "too_few"
"""Refusal reason: fewer than 30 kept samples."""
NO_SPREAD = "no_spread"
"""Refusal reason: the kept target has MAD 0, or the covariates reproduce it."""
TOO_MANY_COVARIATES = "too_many_covariates"
"""Refusal reason: more than one covariate per 10 kept samples."""
EMPTY_BLOCK = "empty_block"
"""Refusal reason: a block with no kept sample."""
COLLINEAR = "collinear"
"""Refusal reason: the observed assignment lies in the span of the covariates."""
REASONS = (TOO_FEW, NO_SPREAD, TOO_MANY_COVARIATES, EMPTY_BLOCK, COLLINEAR)
"""Every refusal reason ``analyze`` can put in ``Analysis.reason``."""


def mad(values: np.ndarray) -> float:
    """Median absolute deviation from the median, unscaled."""
    return float(np.median(np.abs(values - np.median(values))))


def kept_mask(
    y: np.ndarray, x: np.ndarray | None, blocks: Blocks, washout: float
) -> np.ndarray:
    """Samples inside the schedule, ``washout`` or more into their block, all values finite."""
    mask = (blocks.block >= 0) & (blocks.offset >= washout) & np.isfinite(y)
    if x is not None:
        mask &= np.all(np.isfinite(x), axis=1)
    return mask


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
    """Kept samples of one target under one washout and one covariate set, in block space."""

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

    Samples must be in time order, so the kept samples of a block are
    contiguous. Kept: see :func:`kept_mask`. Refusals: fewer than 30 kept
    samples (``too_few``), a kept target with MAD 0 or a residual whose
    MAD is float noise (``no_spread``), more than one covariate per 10
    kept samples (``too_many_covariates``).
    """
    positions = np.flatnonzero(kept_mask(y, x, blocks, washout))
    k_cov = 0 if x is None else x.shape[1]
    n = len(positions)
    if n < MIN_SAMPLES:
        return TOO_FEW
    kept_y = np.asarray(y, dtype=float)[positions]
    spread = mad(kept_y)
    if spread == 0:
        return NO_SPREAD
    if k_cov > n / SAMPLES_PER_COVARIATE:
        return TOO_MANY_COVARIATES
    basis = orthonormal_basis(None if x is None else x[positions], n)
    resid = kept_y - basis @ (basis.T @ kept_y)
    if x is not None and mad(resid) <= RESIDUAL_MAD_RTOL * spread:
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
    """The pieces of one observed assignment that every shift reuses."""

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
    A side the accepted set does not close on is -inf or +inf.
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
    """Estimates, 95% intervals and p-values for m versions of the target."""

    estimate: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    p_value: np.ndarray

    @property
    def claims(self) -> np.ndarray:
        """True where the interval excludes 0."""
        return (self.lo > 0) | (self.hi < 0)


def randomization(design: Design, terms: Terms, sums: np.ndarray, scale: float) -> Result:
    """Randomization estimate, p-value and inverted interval for each column of ``sums``.

    ``sums`` (K, m) holds block sums of M y for m versions of the target.
    ``scale`` sets the tie tolerance in the target's unit.
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


# ---------------------------------------------------------------- one analysis


@dataclass(frozen=True)
class Analysis:
    """One randomization analysis of a target, or the reason it has none.

    ``counts`` holds the kept samples per block, also on a refusal. The
    bounds are -inf or +inf on a side the interval does not close on. On
    a refusal ``reason`` is one of ``REASONS`` and the numbers are NaN.
    """

    counts: np.ndarray
    estimate: float = math.nan
    lo: float = math.nan
    hi: float = math.nan
    p_value: float = math.nan
    reason: str = ""

    @property
    def n_kept(self) -> int:
        return int(self.counts.sum())


def analyze(
    y: np.ndarray, x: np.ndarray | None, blocks: Blocks, washout: float, design: Design
) -> Analysis:
    """The B - A coefficient of ``y`` on [1, z] or [1, z, X], its p-value and interval.

    Refusals, in the order they are checked: those of ``make_frame``,
    then a block with no kept sample (``empty_block``), then an observed
    assignment in the span of the covariates (``collinear``). The tie
    tolerance is scaled by 1.4826 MAD of the kept target samples.

    Examples:
        Setting B adds 0.5 to a noisy target sampled once a minute:

        >>> import numpy as np
        >>> from tsdive.switchback import analyze, cut_blocks, make_design, setting
        >>> design = make_design(24, seed=7)
        >>> times = np.arange(24 * 60, dtype=float)
        >>> blocks = cut_blocks(times, span=times.size, length=60.0)
        >>> noise = np.random.default_rng(0).normal(0.0, 0.2, times.size)
        >>> y = 0.5 * setting(blocks, design.observed) + noise
        >>> result = analyze(y, None, blocks, washout=10.0, design=design)
        >>> round(result.estimate, 2), round(result.lo, 2), round(result.hi, 2)
        (0.49, 0.46, 0.52)
        >>> round(result.p_value, 4), result.n_kept, result.reason
        (0.001, 1200, '')
    """
    counts = np.bincount(
        blocks.block[kept_mask(y, x, blocks, washout)], minlength=blocks.k
    ).astype(np.int64)
    frame = make_frame(y, x, blocks, washout)
    if isinstance(frame, str):
        return Analysis(counts, reason=frame)
    if not frame.nonempty.all():
        return Analysis(counts, reason=EMPTY_BLOCK)
    terms = assignment_terms(frame, design)
    if isinstance(terms, str):
        return Analysis(counts, reason=terms)
    scale = MAD_TO_SD * mad(frame.y)
    result = randomization(design, terms, frame.resid_sums[:, None], scale)
    lo, hi = float(result.lo[0]), float(result.hi[0])
    # The accepted set always holds the estimate, so a NaN bound cannot
    # arise; were one to, it is reported as the open side it would be.
    lo = -math.inf if math.isnan(lo) else lo
    hi = math.inf if math.isnan(hi) else hi
    return Analysis(
        counts,
        estimate=float(result.estimate[0]),
        lo=lo,
        hi=hi,
        p_value=float(result.p_value[0]),
    )


# ---------------------------------------------------------------- response


def lag_response(u: np.ndarray, times: np.ndarray, tau: float) -> np.ndarray:
    """First-order lag response to the per-sample setting ``u``, starting from 0.

    Over the gap dt from the previous sample the response moves
    1 - exp(-dt / tau) of the way to the sample's setting; the first sample
    follows a gap of one time unit. ``tau`` 0 returns ``u``. Runs of equal
    setting are solved in closed form, so the loop runs once per switch.

    Examples:
        >>> import numpy as np
        >>> from tsdive.switchback import lag_response
        >>> u = np.array([0.0, 1.0, 1.0, 1.0])
        >>> lag_response(u, np.array([0.0, 1.0, 2.0, 3.0]), tau=1.0).round(3).tolist()
        [0.0, 0.632, 0.865, 0.95]
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

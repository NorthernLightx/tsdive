"""Two periods of one unit, compared tag by tag and pair by pair.

The question this answers is "something broke on this unit two days ago,
what changed?". The caller states every tag of the unit, a period before
and a period after. The answer is three tables, each a statement about
the samples:

1. Tags that changed. Per tag: the after period's quality read against
   the before period's, the level shift in before-period MAD-sigmas, the
   spread ratio, the share of after samples a before-period MAD screen
   flags, and the first changepoint at or after the after period starts.
2. Pairs that decoupled. Per pair: the Pearson correlation of first
   differences in each period, the difference between them, and a moving
   block bootstrap interval on that difference.
3. Joint structure. A PCA fitted on the before period, read on the after
   period.

A tag "changed" and a pair "decoupled". Neither word names a cause. The
same numbers come out of a recalibrated instrument, a rate change and a
stuck valve.

Every column refuses on its own: a tag whose MAD is 0 before still
reports a flagged share, and a tag whose read raises still occupies a row
carrying the error class.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from itertools import combinations
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from tsdive.api import parse_window
from tsdive.baselines.provisional import MAD_TO_SIGMA, mad_baseline, screen
from tsdive.changepoints.pelt import segment_window
from tsdive.detectors.flatline import assess_flatline
from tsdive.errors import InsufficientQuality, TSDiveError
from tsdive.mspc.pca import (
    AlignedMatrix,
    PcaModel,
    align_windows,
    common_rate,
    detect,
    fit_pca,
)
from tsdive.store.clipping import assert_usable_baseline
from tsdive.store.quality import Severity, usable_mask
from tsdive.store.sampling_contract import (
    AggregateType,
    CalculationBasis,
    RetrievalMode,
    SamplingContract,
)
from tsdive.store.tagstore import SingleFileStore, Window, meta_from_parquet

# How far the after period's GOOD share may fall below the before
# period's before the quality column stops saying "ok". Below this the
# two periods are still the same instrument reporting the same way.
GOOD_FRACTION_SLACK = 0.05

# Coverage the pair grid demands, against 0.95 for the mspc grid. A
# correlation needs the two tags observed at the same instants, not the
# whole grid filled: a period missing a fifth of its cells still holds
# thousands of paired observations, and dropping the pair table over that
# would answer nothing.
PAIR_MIN_COVERAGE = 0.8

# Moving block bootstrap over the differenced after period. Blocks keep
# what is left of the serial dependence after differencing; 10 samples is
# the floor, and n**(1/3) is the standard growth rate with the sample
# count.
BOOTSTRAP_REPLICATES = 200
BOOTSTRAP_SEED = 42
BOOTSTRAP_MIN_BLOCK = 10
BOOTSTRAP_LOW = 2.5
BOOTSTRAP_HIGH = 97.5

# Tags the joint table names before it summarises the rest as "other".
CONTRIBUTORS_KEPT = 3


@dataclass(frozen=True)
class TagChange:
    """One archive's row of table 1. ``None`` columns carry their reason."""

    tag: str
    label: str
    quality: str
    quality_ok: bool
    refused: bool = False
    level_shift_sigma: float | None = None
    level_shift_reason: str | None = None
    spread_ratio: float | None = None
    spread_reason: str | None = None
    flagged_fraction: float | None = None
    flagged_reason: str | None = None
    changed_at: pd.Timestamp | None = None
    changed_at_reason: str | None = None

    @property
    def rank_key(self) -> float:
        """What the table ranks on: the level shift, else the flagged share."""
        if self.level_shift_sigma is not None:
            return abs(self.level_shift_sigma)
        if self.flagged_fraction is not None:
            return self.flagged_fraction
        return 0.0


# Why a pair keeps its delta and loses its interval.
NO_INTERVAL = "a block resample of the after period left one of the two tags flat"


@dataclass(frozen=True)
class PairChange:
    """One pair's row of table 2. ``lo``/``hi`` bound the delta, not rho."""

    left: str
    right: str
    pearson_before: float
    pearson_after: float
    delta: float
    lo: float | None
    hi: float | None
    spearman_before: float
    spearman_after: float
    spearman_delta: float

    @property
    def clears(self) -> bool:
        """True when the bootstrap interval excludes a delta of 0."""
        if self.lo is None or self.hi is None:
            return False
        return bool(self.lo > 0.0 or self.hi < 0.0)


@dataclass(frozen=True)
class PairTable:
    """Table 2, ranked by ``|delta|``, or the reason it holds no rows."""

    pairs: tuple[PairChange, ...] = ()
    n_pairs: int = 0
    # Pairs left out: no correlation at all in one period, against pairs
    # that kept their delta and lost only the interval around it.
    n_undefined: int = 0
    n_no_interval: int = 0
    rate_s: int | None = None
    rate_source: str | None = None
    coverage_before: float | None = None
    coverage_after: float | None = None
    reason: str | None = None

    @property
    def clearing(self) -> tuple[PairChange, ...]:
        return tuple(p for p in self.pairs if p.clears)


@dataclass(frozen=True)
class JointStructure:
    """Table 3: a PCA fitted on the before period, read on the after one."""

    n_offered: int = 0
    n_aligned: int = 0
    n_components: int = 0
    rows: int = 0
    explained_before: float | None = None
    explained_after: float | None = None
    t2_breaches: int = 0
    spe_breaches: int = 0
    contributors: tuple[tuple[str, float], ...] = ()
    other_share: float | None = None
    reason: str | None = None


@dataclass(frozen=True)
class TagPeriods:
    """One archive read over both periods, or the refusal that stopped it."""

    path: str
    tag: str
    point_id: str
    source_id: str | None
    before: Window | None = None
    after: Window | None = None
    refusal: str | None = None


@dataclass(frozen=True)
class CompareResult:
    """The tables both periods support, ranked and ready to render."""

    before: tuple[pd.Timestamp, pd.Timestamp]
    after: tuple[pd.Timestamp, pd.Timestamp]
    source_id: str | None
    tags: tuple[TagChange, ...]
    pairs: PairTable
    joint: JointStructure

    @property
    def n_refused(self) -> int:
        """Archives whose read raised for one of the two periods."""
        return sum(1 for t in self.tags if t.refused)


def parse_periods(
    before: str, after: str
) -> tuple[tuple[pd.Timestamp, pd.Timestamp], tuple[pd.Timestamp, pd.Timestamp]]:
    """Parse both ``START/END`` periods, refusing an overlap or a swap.

    Overlapping periods grade samples against themselves. Periods stated
    the other way round would rank a level shift by its own negation and
    read the changepoint at the wrong end of the record.
    """
    b_start, b_end = parse_window(before)
    a_start, a_end = parse_window(after)
    if b_start < a_end and a_start < b_end:
        raise ValueError("--before and --after overlap")
    if a_start < b_end:
        raise ValueError(
            "--after ends before --before starts; state the earlier period as --before"
        )
    return (b_start, b_end), (a_start, a_end)


def _contract() -> SamplingContract:
    return SamplingContract(
        calculation_basis=CalculationBasis.TIME_WEIGHTED,
        retrieval_mode=RetrievalMode.RECORDED,
        aggregate_type=AggregateType.NONE,
        stepped=False,
    )


def read_periods(
    paths: Sequence[str],
    before: tuple[pd.Timestamp, pd.Timestamp],
    after: tuple[pd.Timestamp, pd.Timestamp],
) -> list[TagPeriods]:
    """Read every archive over both periods, keeping each refusal as a row.

    A read that raises for either period leaves the tag in the table with
    the error class instead of numbers, so a unit with one unreadable
    archive still gets an answer about the rest.

    Reads are inclusive at both ends, so adjacent periods would share the
    sample on the boundary. It is given to the after period and dropped
    from the before frame; the before period's physics still covers it,
    which is the conservative side of a censoring verdict.
    """
    b_start, b_end = before
    a_start, a_end = after
    out: list[TagPeriods] = []
    for path in paths:
        label = Path(path).stem
        source_id: str | None = None
        try:
            store = SingleFileStore(Path(path))
            meta = meta_from_parquet(store.path)
            label = str(meta.identity)
            source_id = meta.identity.source_id
            reads = [
                store.read_window(
                    meta.identity, s.to_pydatetime(), e.to_pydatetime(), _contract()
                )
                for s, e in ((b_start, b_end), (a_start, a_end))
            ]
        except (TSDiveError, OSError, ValueError) as e:
            out.append(
                TagPeriods(
                    path=str(path),
                    tag=label,
                    point_id=label.split(":")[-1],
                    source_id=source_id,
                    refusal=type(e).__name__,
                )
            )
            continue
        base, monitor = reads
        if b_end == a_start:
            base = replace(base, frame=base.frame[base.frame["timestamp"] < a_start])
        out.append(
            TagPeriods(
                path=str(path),
                tag=str(meta.identity),
                point_id=meta.identity.point_id,
                source_id=source_id,
                before=base,
                after=monitor,
            )
        )
    return out


def _numeric_good(frame: pd.DataFrame) -> np.ndarray:
    """Finite GOOD values of a window frame, as floats."""
    good = frame[usable_mask(frame)]
    values = cast(pd.Series, pd.to_numeric(good["value"], errors="coerce"))
    arr = values.dropna().to_numpy(dtype=float)
    return arr[np.isfinite(arr)]


def _change_intervals_s(frame: pd.DataFrame) -> list[float]:
    """Seconds between successive value changes among a frame's usable rows.

    The same quantity :func:`tsdive.api.reference_history` pools over
    prior windows, measured here over the before period alone: the time
    between changes, not the spacing at which a change was noticed.
    """
    good = frame[usable_mask(frame)]
    if len(good) < 2:
        return []
    values = cast(pd.Series, good["value"].reset_index(drop=True))
    stamps = cast(pd.Series, good["timestamp"].reset_index(drop=True))
    changed = values.ne(values.shift()).to_numpy(dtype=bool, na_value=False)
    # The first row compares against nothing, so it is not a change.
    changed[0] = False
    at = np.flatnonzero(changed)
    if at.size < 2:
        return []
    seconds = stamps.iloc[at].astype("int64").to_numpy() / 1e9
    return [float(v) for v in np.diff(seconds)]


def _frozen(before: Window, after: Window) -> bool:
    """True when the after period fires the flatline stall signal.

    The reference is the before period's own change intervals. Reference
    distinct counts are withheld: that signal compares a count against a
    count, and counts scale with how long a period is, so two periods of
    different lengths would fire it on their lengths alone.
    """
    verdict = assess_flatline(
        after,
        reference_change_intervals_s=_change_intervals_s(before.frame),
        reference_distinct_counts=(),
    )
    return verdict.fired


def _quality(before: Window, after: Window) -> str:
    """The after period's quality verdict, read against the before period."""
    counts = after.physics.severity_counts
    n = sum(counts.values())
    if n == 0:
        return "no rows"
    b_counts = before.physics.severity_counts
    b_n = sum(b_counts.values())
    good_before = b_counts[Severity.GOOD] / b_n if b_n else 1.0
    if counts[Severity.GOOD] / n < good_before - GOOD_FRACTION_SLACK:
        worst = Severity.BAD if counts[Severity.BAD] else Severity.UNCERTAIN
        return f"{worst.value} {100.0 * counts[worst] / n:.0f}%"
    if after.physics.clipping.censored:
        return "censored"
    if _frozen(before, after):
        return "frozen"
    return "ok"


def _level_and_spread(
    before: Window, after: Window
) -> tuple[float | None, float | None, str | None]:
    """Level shift in before-period MAD-sigmas, spread ratio, and the reason.

    Both divide by the before period's MAD, so both refuse together on a
    period that never moved: a shift of any size is infinitely many
    sigmas when the sigma is 0.
    """
    b_vals = _numeric_good(before.frame)
    a_vals = _numeric_good(after.frame)
    if len(b_vals) == 0:
        return None, None, "no GOOD rows before"
    if len(a_vals) == 0:
        return None, None, "no GOOD rows after"
    b_median = float(np.median(b_vals))
    b_mad = float(np.median(np.abs(b_vals - b_median)))
    if b_mad == 0.0:
        return None, None, "no spread before"
    a_median = float(np.median(a_vals))
    a_mad = float(np.median(np.abs(a_vals - a_median)))
    return (a_median - b_median) / (b_mad * MAD_TO_SIGMA), a_mad / b_mad, None


def _flagged(before: Window, after: Window) -> tuple[float | None, str | None]:
    """Share of after samples a before-period MAD screen flags, or the reason."""
    try:
        assert_usable_baseline(before.meta, before.physics.clipping)
        result = screen(mad_baseline(before.frame), after.frame)
    except TSDiveError as e:
        return None, f"{type(e).__name__}: {e}"
    if result.n_screened == 0:
        return None, "no GOOD rows after"
    return result.n_flagged / result.n_screened, None


def _changed_at(
    before: Window, after: Window
) -> tuple[pd.Timestamp | None, str | None]:
    """First changepoint at or after the after period starts, or the reason.

    Both periods are segmented as one series, because a step is only a
    step against what came before it. The window handed to the segmenter
    carries the after period's physics block, which nothing here reads or
    prints; only its frame is replaced.
    """
    joined = pd.concat([before.frame, after.frame], ignore_index=True)
    try:
        found = segment_window(replace(after, frame=joined))
    except (TSDiveError, ValueError) as e:
        return None, f"{type(e).__name__}: {e}"
    at = next((b for b in found.breakpoints if b >= after.start), None)
    return at, None


def tag_changes(periods: Sequence[TagPeriods], *, shared_source: bool) -> list[TagChange]:
    """Table 1, ranked: refused and quality-failed tags first, then by shift."""
    rows: list[TagChange] = []
    for p in periods:
        label = p.point_id if shared_source else p.tag
        if p.before is None or p.after is None:
            # One reason on every column: the read that would have fed
            # them never happened.
            why = f"the read raised {p.refusal}"
            rows.append(
                TagChange(
                    tag=p.tag,
                    label=label,
                    quality=f"[{p.refusal}]",
                    quality_ok=False,
                    refused=True,
                    level_shift_reason=why,
                    spread_reason=why,
                    flagged_reason=why,
                    changed_at_reason=why,
                )
            )
            continue
        shift, spread, spread_reason = _level_and_spread(p.before, p.after)
        flagged, flagged_reason = _flagged(p.before, p.after)
        at, at_reason = _changed_at(p.before, p.after)
        quality = _quality(p.before, p.after)
        rows.append(
            TagChange(
                tag=p.tag,
                label=label,
                quality=quality,
                quality_ok=quality == "ok",
                level_shift_sigma=shift,
                level_shift_reason=spread_reason,
                spread_ratio=spread,
                spread_reason=spread_reason,
                flagged_fraction=flagged,
                flagged_reason=flagged_reason,
                changed_at=at,
                changed_at_reason=at_reason,
            )
        )
    return sorted(rows, key=lambda r: (r.quality_ok, -r.rank_key, r.tag))


def _corr(x: np.ndarray) -> np.ndarray:
    """Pearson correlation matrix over the columns of ``x``.

    A column that never moves has no correlation with anything; it comes
    back NaN here rather than as a 0 that reads like independence.
    """
    xc = x - x.mean(axis=0)
    norm = np.sqrt((xc**2).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        z = xc / norm
    return cast(np.ndarray, z.T @ z)


def _ranked(x: np.ndarray) -> np.ndarray:
    """Column-wise ranks, ties averaged, for the Spearman correlation."""
    return pd.DataFrame(x).rank().to_numpy(dtype=float)


def _block_bootstrap(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Correlation matrices of ``BOOTSTRAP_REPLICATES`` block resamples of ``x``.

    One row index is drawn per replicate and reused for every column, so
    a resample costs one correlation matrix however many pairs read it.
    """
    n = len(x)
    block = min(max(BOOTSTRAP_MIN_BLOCK, int(n ** (1 / 3))), n)
    n_blocks = -(-n // block)
    offsets = np.arange(block)
    out = np.empty((BOOTSTRAP_REPLICATES, x.shape[1], x.shape[1]), dtype=float)
    for r in range(BOOTSTRAP_REPLICATES):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        idx = (starts[:, None] + offsets[None, :]).ravel()[:n]
        out[r] = _corr(x[idx])
    return out


def _gridable(periods: Sequence[TagPeriods]) -> list[TagPeriods]:
    """Tags holding more than one GOOD numeric sample in both periods.

    The alignment reads every value as a float, so a state-valued tag and
    a tag with nothing usable are held back from the grid rather than
    raising inside it.
    """
    return [
        p
        for p in periods
        if p.before is not None
        and p.after is not None
        and len(_numeric_good(p.before.frame)) > 1
        and len(_numeric_good(p.after.frame)) > 1
    ]


def _too_few(usable: Sequence[TagPeriods], periods: Sequence[TagPeriods]) -> str:
    return (
        f"{len(usable)} of {len(periods)} tags carry GOOD numeric rows in both "
        "periods; a grid needs two"
    )


def _grids(
    usable: Sequence[TagPeriods], *, rate_s: int, min_coverage: float
) -> tuple[AlignedMatrix, AlignedMatrix]:
    return (
        align_windows(
            [cast(Window, p.before) for p in usable],
            rate_s=rate_s,
            min_coverage=min_coverage,
        ),
        align_windows(
            [cast(Window, p.after) for p in usable],
            rate_s=rate_s,
            min_coverage=min_coverage,
        ),
    )


def pair_changes(
    periods: Sequence[TagPeriods],
    *,
    shared_source: bool,
    rate_s: int | None = None,
) -> PairTable:
    """Table 2: how each pair's differenced correlation moved, with an interval.

    Correlations are taken on first differences of each aligned column.
    Raw process values share a level and an autocorrelation, which puts
    two unrelated tags near 1.0 and leaves the number describing the day
    rather than the coupling. Differencing removes both.

    The interval is a moving block bootstrap of the after period's
    correlation, shifted by the before period's, so it bounds the delta.
    A pair clears when that interval excludes 0.
    """
    usable = _gridable(periods)
    if len(usable) < 2:
        return PairTable(reason=_too_few(usable, periods))
    try:
        rate = rate_s if rate_s is not None else common_rate(
            [cast(Window, p.before) for p in usable]
        )
    except ValueError as e:
        return PairTable(reason=str(e))
    source = "declared" if rate_s is None else "--rate-s"
    try:
        train, test = _grids(usable, rate_s=rate, min_coverage=PAIR_MIN_COVERAGE)
    except TSDiveError as e:
        return PairTable(rate_s=rate, rate_source=source, reason=f"{type(e).__name__}: {e}")

    d_before = np.diff(train.matrix, axis=0)
    d_after = np.diff(test.matrix, axis=0)
    if len(d_before) < 2 or len(d_after) < 2:
        return PairTable(
            rate_s=rate,
            rate_source=source,
            reason=f"{len(d_before)} differenced rows before and {len(d_after)} after; "
            "a correlation needs two",
        )
    rho_before = _corr(d_before)
    rho_after = _corr(d_after)
    s_before = _corr(_ranked(d_before))
    s_after = _corr(_ranked(d_after))
    replicates = _block_bootstrap(d_after, np.random.default_rng(BOOTSTRAP_SEED))
    lo = np.percentile(replicates, BOOTSTRAP_LOW, axis=0)
    hi = np.percentile(replicates, BOOTSTRAP_HIGH, axis=0)

    columns = train.columns
    labels = {p.tag: (p.point_id if shared_source else p.tag) for p in usable}
    rows: list[PairChange] = []
    undefined = 0
    no_interval = 0
    for i, j in combinations(range(len(columns)), 2):
        delta = rho_after[i, j] - rho_before[i, j]
        if not np.isfinite(delta):
            undefined += 1
            continue
        bounds: tuple[float | None, float | None] = (
            float(lo[i, j] - rho_before[i, j]),
            float(hi[i, j] - rho_before[i, j]),
        )
        # A tag that moves at a handful of samples - a stepped setpoint -
        # has no spread in the resamples that miss them, and no
        # correlation there either. The delta still stands; only the
        # interval around it does not.
        if not all(v is not None and np.isfinite(v) for v in bounds):
            no_interval += 1
            bounds = (None, None)
        rows.append(
            PairChange(
                left=labels.get(columns[i], columns[i]),
                right=labels.get(columns[j], columns[j]),
                pearson_before=float(rho_before[i, j]),
                pearson_after=float(rho_after[i, j]),
                delta=float(delta),
                lo=bounds[0],
                hi=bounds[1],
                spearman_before=float(s_before[i, j]),
                spearman_after=float(s_after[i, j]),
                spearman_delta=float(s_after[i, j] - s_before[i, j]),
            )
        )
    rows.sort(key=lambda p: (-abs(p.delta), p.left, p.right))
    return PairTable(
        pairs=tuple(rows),
        n_pairs=len(rows) + undefined,
        n_undefined=undefined,
        n_no_interval=no_interval,
        rate_s=rate,
        rate_source=source,
        coverage_before=train.coverage,
        coverage_after=test.coverage,
    )


def _residuals(model: PcaModel, aligned: AlignedMatrix) -> tuple[np.ndarray, float]:
    """Residual matrix, and the share of ``aligned``'s variance the model keeps.

    :func:`~tsdive.mspc.pca.detect` computes the same projection but
    hands back only its two statistics; the per-column residuals are what
    the contribution shares divide up.
    """
    centred = aligned.matrix - model.mean
    resid = centred - (centred @ model.components.T) @ model.components
    total = float((centred**2).sum())
    kept = 1.0 - float((resid**2).sum()) / total if total > 0 else 0.0
    return resid, kept


def joint_structure(
    periods: Sequence[TagPeriods],
    *,
    shared_source: bool,
    rate_s: int | None = None,
    min_coverage: float = 0.95,
    variance: float = 0.95,
    quantile: float = 0.99,
) -> JointStructure:
    """Table 3: the before period's PCA, and what the after period does to it.

    ``explained_after`` projects the after period's rows onto the
    components fitted on the before period and reports the share of their
    variance those components keep. A drop is the collinearity change:
    the tags still move, and they no longer move along the same axes.

    Contributions divide the squared residual of the SPE breach rows over
    the columns, so a share says which tag the residual sits in, never
    which tag moved first.
    """
    usable = _gridable(periods)
    if len(usable) < 2:
        return JointStructure(n_offered=len(periods), reason=_too_few(usable, periods))
    try:
        rate = rate_s if rate_s is not None else common_rate(
            [cast(Window, p.before) for p in usable]
        )
    except ValueError as e:
        return JointStructure(n_offered=len(periods), reason=str(e))
    try:
        train, test = _grids(usable, rate_s=rate, min_coverage=min_coverage)
    except TSDiveError as e:
        return JointStructure(
            n_offered=len(periods), reason=f"{type(e).__name__}: {e}"
        )
    model = fit_pca(train, variance_threshold=variance, limit_quantile=quantile)
    found = detect(model, test)
    resid, explained_after = _residuals(model, test)

    labels = {p.tag: (p.point_id if shared_source else p.tag) for p in usable}
    breached = np.zeros(len(test.index), dtype=bool)
    if found.spe_breaches:
        breached = np.isin(test.index, pd.DatetimeIndex(found.spe_breaches))
    per_column = (resid[breached] ** 2).sum(axis=0)
    total = float(per_column.sum())
    shares: list[tuple[str, float]] = []
    other: float | None = None
    if total > 0:
        ranked = sorted(
            (
                (labels.get(name, name), float(v) / total)
                for name, v in zip(model.columns, per_column, strict=True)
            ),
            key=lambda kv: (-kv[1], kv[0]),
        )
        shares = ranked[:CONTRIBUTORS_KEPT]
        rest = ranked[CONTRIBUTORS_KEPT:]
        other = sum(share for _, share in rest) if rest else None
    return JointStructure(
        n_offered=len(periods),
        n_aligned=len(model.columns),
        n_components=len(model.components),
        rows=len(test.index),
        explained_before=float(sum(model.explained_variance)),
        explained_after=explained_after,
        t2_breaches=len(found.t2_breaches),
        spe_breaches=len(found.spe_breaches),
        contributors=tuple(shares),
        other_share=other,
    )


def compare(
    paths: Sequence[str],
    before: str,
    after: str,
    *,
    rate_s: int | None = None,
) -> CompareResult:
    """Compare two periods over every archive of one unit.

    Raises:
        ValueError: the periods overlap, or they are stated in the wrong
            order.
        InsufficientQuality: not one archive could be read over both
            periods.
    """
    b_span, a_span = parse_periods(before, after)
    periods = read_periods(paths, b_span, a_span)
    if all(p.refusal is not None for p in periods):
        first = periods[0]
        raise InsufficientQuality(
            f"{len(periods)} archives, none readable over both periods; "
            f"{first.tag} raised {first.refusal}"
        )
    sources = {p.source_id for p in periods if p.source_id is not None}
    shared = sources.pop() if len(sources) == 1 else None
    return CompareResult(
        before=b_span,
        after=a_span,
        source_id=shared,
        tags=tuple(tag_changes(periods, shared_source=shared is not None)),
        pairs=pair_changes(periods, shared_source=shared is not None, rate_s=rate_s),
        joint=joint_structure(periods, shared_source=shared is not None, rate_s=rate_s),
    )

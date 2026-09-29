"""Switchback plans and analyses over parquet archives.

Every archive is read through :meth:`SingleFileStore.read_window` under
the sampling contract ``compare`` reads with (time-weighted, recorded),
over the plan's schedule. A sample counts when the read marks it valid
(GOOD quality and a numeric value) and its value is finite, as in every
other analysis. Covariates join the target on the exact timestamp: a
target sample with no valid sample of every covariate at the same instant
is dropped from the adjusted analysis and kept in the direct one.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import KW_ONLY, dataclass, replace
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from tsdive.errors import SchemaError, TSDiveError
from tsdive.store.quality import Severity, usable_mask
from tsdive.store.sampling_contract import (
    AggregateType,
    CalculationBasis,
    RetrievalMode,
    SamplingContract,
)
from tsdive.store.tagstore import (
    SingleFileStore,
    Window,
    assert_windows_comparable,
    meta_from_parquet,
)
from tsdive.switchback.design import Design
from tsdive.switchback.inference import (
    COLLINEAR,
    EMPTY_BLOCK,
    MIN_SAMPLES,
    NO_SPREAD,
    SAMPLES_PER_COVARIATE,
    TOO_FEW,
    TOO_MANY_COVARIATES,
    Analysis,
    analyze,
    kept_mask,
    mad,
)
from tsdive.switchback.plan import (
    HISTORY_SHORT,
    POWER_DELTAS,
    POWER_DRAWS,
    PowerReadout,
    SwitchbackPlan,
    iso,
    power_rates,
    schedule_offsets,
    smallest_detected,
    verify_plan,
)

ASSUMPTIONS = (
    "difference between settings A and B under the declared random schedule, by "
    "intended assignment; holds if the schedule was followed and carryover ended "
    "within the washout"
)

_NS = 1_000_000_000

# A covariate whose own B - A difference has a randomization p-value
# below this moves with the setting.
COVARIATE_ALPHA = 0.05


def _contract() -> SamplingContract:
    return SamplingContract(
        calculation_basis=CalculationBasis.TIME_WEIGHTED,
        retrieval_mode=RetrievalMode.RECORDED,
        aggregate_type=AggregateType.NONE,
        stepped=False,
    )


def read_span(path: str | Path, start: pd.Timestamp, end: pd.Timestamp) -> Window:
    """One archive read from ``start`` to ``end`` under ``compare``'s sampling contract."""
    store = SingleFileStore(Path(path))
    meta = meta_from_parquet(store.path)
    return store.read_window(
        meta.identity, start.to_pydatetime(), end.to_pydatetime(), _contract()
    )


def _samples(window: Window) -> tuple[pd.Series, np.ndarray]:
    """Timestamps and float values of the valid rows, in archive order."""
    good = window.frame[usable_mask(window.frame)]
    values = cast(pd.Series, pd.to_numeric(good["value"], errors="coerce"))
    return (
        cast(pd.Series, good["timestamp"].reset_index(drop=True)),
        values.to_numpy(dtype=float),
    )


def unit_name(window: Window) -> str | None:
    """The canonical unit when the tag's unit resolves, else the raw spelling, else ``None``."""
    unit = window.physics.unit
    if unit.canonical is not None:
        return unit.canonical
    return unit.raw or None


def _good_share(window: Window) -> float | None:
    counts = window.physics.severity_counts
    n = sum(counts.values())
    return counts.get(Severity.GOOD, 0) / n if n else None


# ---------------------------------------------------------------- power


def power_readout(
    history: str | Path, window: tuple[pd.Timestamp, pd.Timestamp], plan: SwitchbackPlan
) -> PowerReadout:
    """The plan's power over a history window of one archive, or the readout's refusal.

    The plan's blocks are laid from the history window's start, so the
    readout reads the first K blocks of the window. A history window
    shorter than the schedule refuses the readout (``history_short``), and
    so does a typed refusal of the read or a history the analysis would
    refuse.
    """
    h_start, h_end = window
    span_ns = plan.k * plan.block_s * _NS
    used_end = cast(pd.Timestamp, h_start + pd.Timedelta(span_ns, unit="ns"))
    # A refused readout states the history window that was passed; a
    # readout that runs states the first K blocks of it that it read.
    base = {
        "tag": Path(history).stem,
        "start": h_start,
        "end": h_end,
        "unit": None,
        "draws": POWER_DRAWS,
        "deltas": POWER_DELTAS,
    }
    try:
        base["tag"] = _identity_of(history)
    except TSDiveError as e:
        return PowerReadout(**base, reason=type(e).__name__, detail=str(e))
    if (h_end - h_start).value < span_ns:
        return PowerReadout(
            **base,
            reason=HISTORY_SHORT,
            detail=(
                f"the history window spans {_hours(h_end - h_start)} and the schedule "
                f"{_hours(plan.schedule_end - plan.start)}; pass a history window at "
                "least as long as the schedule"
            ),
        )
    try:
        read = read_span(history, h_start, used_end)
    except TSDiveError as e:
        return PowerReadout(**base, reason=type(e).__name__, detail=str(e))
    base["unit"] = unit_name(read)
    base["end"] = used_end
    times, y = _samples(read)
    blocks = schedule_offsets(times, h_start, plan.block_s, plan.k)
    out = power_rates(y, blocks, plan.washout_s * _NS, plan.seed)
    if isinstance(out, str):
        kept = kept_mask(y, None, blocks, plan.washout_s * _NS)
        counts = np.bincount(blocks.block[kept], minlength=plan.k)
        return PowerReadout(
            **{**base, "end": h_end}, reason=out, detail=_power_detail(out, counts)
        )
    sigma, n_kept, rates = out
    return PowerReadout(
        **base,
        sigma=sigma,
        n_kept=n_kept,
        rates=rates,
        smallest=smallest_detected(POWER_DELTAS, rates),
    )


def _hours(delta: pd.Timedelta) -> str:
    return f"{delta.total_seconds() / 3600:g} h"


def _power_detail(reason: str, counts: np.ndarray) -> str:
    if reason == NO_SPREAD:
        return "the history's valid samples have MAD 0"
    if reason == EMPTY_BLOCK:
        empty = np.flatnonzero(counts == 0)
        return (
            f"{len(empty)} of {len(counts)} blocks laid over the history hold no sample "
            "past the washout"
        )
    if reason == TOO_FEW:
        return f"the history holds fewer than {MIN_SAMPLES} samples past the washout"
    return reason


# ---------------------------------------------------------------- analysis


@dataclass(frozen=True)
class SwitchbackEstimate:
    """One analysis of the target: direct (no covariates) or adjusted.

    ``estimate`` is B minus A in the target's unit. ``lo`` and ``hi`` bound
    the 95% interval and are -inf or +inf on a side it does not close on.
    On a refusal ``reason`` is one of ``too_few``, ``no_spread``,
    ``too_many_covariates``, ``empty_block`` or ``collinear``, ``detail``
    says what the data showed, and the numbers are ``None``.
    """

    covariates: tuple[str, ...]
    kept_per_block: tuple[int, ...]
    # Optional fields are keyword-only, so adding one never moves another.
    _: KW_ONLY
    estimate: float | None = None
    p_value: float | None = None
    lo: float | None = None
    hi: float | None = None
    reason: str | None = None
    detail: str | None = None

    @property
    def n_kept(self) -> int:
        return sum(self.kept_per_block)

    @property
    def refused(self) -> bool:
        return self.reason is not None

    @property
    def lo_unbounded(self) -> bool:
        return self.lo is not None and math.isinf(self.lo)

    @property
    def hi_unbounded(self) -> bool:
        return self.hi is not None and math.isinf(self.hi)


@dataclass(frozen=True)
class CovariateCheck:
    """One covariate's own difference between settings A and B under the plan.

    ``difference`` is the covariate analysed as the target would be, with
    the same design and washout. A covariate the setting moves, such as a
    controller output, carries part of the effect, and the adjusted
    estimate can absorb it. ``moves`` is True when the randomization test
    rejects a zero difference at p < 0.05; a refused test leaves it False.
    """

    tag: str
    unit: str | None
    difference: SwitchbackEstimate

    @property
    def moves(self) -> bool:
        p = self.difference.p_value
        return p is not None and p < COVARIATE_ALPHA


@dataclass(frozen=True)
class SwitchbackAnalysis:
    """The difference between settings A and B on one target, under a verified plan.

    ``target`` and ``covariates`` are the reads over the schedule.
    ``unused`` names the archives passed in that are neither. ``adjusted``
    is ``None`` when no covariate was declared. ``covariate_checks`` holds
    one ``CovariateCheck`` per
    covariate, in the order they were named; the analysis does not refuse
    on one that moves, it reports it.
    """

    plan: SwitchbackPlan
    target: Window
    covariates: tuple[Window, ...]
    unused: tuple[str, ...]
    direct: SwitchbackEstimate
    adjusted: SwitchbackEstimate | None
    # Optional fields are keyword-only, so adding one never moves another.
    _: KW_ONLY
    assumptions: str = ASSUMPTIONS
    covariate_checks: tuple[CovariateCheck, ...] = ()

    @property
    def moving_covariates(self) -> tuple[CovariateCheck, ...]:
        """The covariates whose own B - A difference has p below 0.05."""
        return tuple(check for check in self.covariate_checks if check.moves)

    @property
    def tag(self) -> str:
        return str(self.target.identity)

    @property
    def unit(self) -> str | None:
        return unit_name(self.target)

    @property
    def good_share(self) -> float | None:
        """GOOD rows over all rows of the target read."""
        return _good_share(self.target)

    @property
    def censored(self) -> bool | None:
        """Whether a target sample sits at its engineering range limit.

        ``None`` when the target declares no engineering range and no
        digital range state flags a sample.
        """
        return self.target.physics.clipping.censored_verdict

    @property
    def frame(self) -> pd.DataFrame:
        """One row per block: index, start, setting, kept samples of each analysis."""
        rows: dict[str, list[object]] = {
            "block": [b.index for b in self.plan.blocks],
            "start": [b.start for b in self.plan.blocks],
            "setting": [b.setting for b in self.plan.blocks],
            "kept": list(self.direct.kept_per_block),
        }
        if self.adjusted is not None:
            rows["kept_adjusted"] = list(self.adjusted.kept_per_block)
        return pd.DataFrame(rows)

    def render(self) -> str:
        """The text ``tsdive switchback analyze`` prints.

        Examples:
            >>> import tsdive
            >>> plan = tsdive.switchback_plan("2024-06-03T00:00:00Z", "2024-06-04T00:00:00Z",
            ...                               block="PT1H", washout="PT15M", seed=7)
            >>> archives = ["data/switchback_demo/ti201.parquet",
            ...             "data/switchback_demo/fi200.parquet",
            ...             "data/switchback_demo/tt001.parquet"]
            >>> result = tsdive.switchback_analyze(archives, plan, target="TI201.PV",
            ...                                    covariates=["FI200.PV", "TT001.PV"])
            >>> print(result.render().splitlines()[0])
            demo:TI201.PV   B - A +0.6024 degrees Celsius   p 0.154
        """
        from tsdive.switchback.render import analysis_lines

        return "\n".join(analysis_lines(self))

    def to_dict(self) -> dict[str, object]:
        """The document ``tsdive switchback analyze --json`` prints, ready for ``json.dumps``.

        The command adds ``result_kind`` and ``tsdive_version`` in front.
        """
        from tsdive.switchback.render import analysis_json

        return analysis_json(self)


def _identity_of(path: str | Path) -> str:
    return str(meta_from_parquet(Path(path)).identity)


def _resolve(name: str, labels: dict[str, str], role: str) -> str:
    """The archive path whose tag is ``name``, as ``source:point`` or a unique point id."""
    exact = [path for path, label in labels.items() if label == name]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        raise ValueError(f"{len(exact)} archives carry the tag {name}; pass each tag once")
    by_point = [path for path, label in labels.items() if label.split(":", 1)[-1] == name]
    if len(by_point) == 1:
        return by_point[0]
    known = ", ".join(sorted(labels.values()))
    if by_point:
        raise ValueError(
            f"{role} {name} matches {len(by_point)} archives; name it as source:point "
            f"from {known}"
        )
    raise ValueError(f"no archive carries the {role} {name}; the archives carry {known}")


def _detail(
    reason: str, result: Analysis, y: np.ndarray, k_cov: int, tag: str, plan: SwitchbackPlan
) -> str:
    n = result.n_kept
    if reason == TOO_FEW:
        return f"{n} samples kept past the washout; the analysis needs {MIN_SAMPLES}"
    if reason == NO_SPREAD:
        if k_cov == 0 or mad(y[np.isfinite(y)]) == 0:
            return f"the kept samples of {tag} have MAD 0"
        return f"the covariates reproduce {tag}: its residual MAD is float noise"
    if reason == TOO_MANY_COVARIATES:
        return (
            f"{k_cov} covariates over {n} kept samples; the analysis takes one "
            f"covariate per {SAMPLES_PER_COVARIATE} kept samples"
        )
    if reason == EMPTY_BLOCK:
        empty = np.flatnonzero(result.counts == 0)
        first = plan.blocks[int(empty[0])]
        return (
            f"{len(empty)} of {len(result.counts)} blocks hold no sample past the "
            f"washout, first block {first.index} from {iso(first.start)}"
        )
    if reason == COLLINEAR:
        return "the covariates follow the schedule: the observed assignment lies in their span"
    return reason


def _estimate(
    y: np.ndarray,
    x: np.ndarray | None,
    names: tuple[str, ...],
    times: pd.Series,
    plan: SwitchbackPlan,
    design: Design,
    tag: str,
) -> SwitchbackEstimate:
    blocks = schedule_offsets(times, plan.start, plan.block_s, plan.k)
    result = analyze(y, x, blocks, plan.washout_s * _NS, design)
    counts = tuple(int(c) for c in result.counts)
    if result.reason:
        return SwitchbackEstimate(
            covariates=names,
            kept_per_block=counts,
            reason=result.reason,
            detail=_detail(result.reason, result, y, len(names), tag, plan),
        )
    return SwitchbackEstimate(
        covariates=names,
        kept_per_block=counts,
        estimate=result.estimate,
        p_value=result.p_value,
        lo=result.lo,
        hi=result.hi,
    )


def _joined(
    times: pd.Series, y: np.ndarray, covariates: Sequence[Window]
) -> tuple[np.ndarray, np.ndarray]:
    """The target values and the covariate matrix on the target's own timestamps."""
    frame = pd.DataFrame({"timestamp": times, "y": y})
    names = []
    for i, window in enumerate(covariates):
        c_times, c_values = _samples(window)
        if bool(c_times.duplicated().any()):
            raise SchemaError(
                f"{window.identity}: duplicate timestamps among its valid samples; joining "
                "them to the target would repeat target samples"
            )
        name = f"x{i}"
        names.append(name)
        frame = frame.merge(
            pd.DataFrame({"timestamp": c_times, name: c_values}), on="timestamp", how="left"
        )
    return frame["y"].to_numpy(dtype=float), frame[names].to_numpy(dtype=float)


def analyze_archives(
    archives: Sequence[str | Path],
    plan: SwitchbackPlan,
    *,
    target: str,
    covariates: Sequence[str] = (),
) -> SwitchbackAnalysis:
    """Verify ``plan``, read the target and covariates over it, and analyse them.

    Raises:
        ScheduleMismatch: the plan's schedule differs from its fields.
        DesignTooSmall: the plan's block count is too small.
        ValueError: a tag named twice or matching no archive.
        IncomparableSamplingError: two reads under different contracts.
        SchemaError: a covariate repeats a timestamp among its valid samples.
        TSDiveError: any typed refusal of the read path.
    """
    design = verify_plan(plan)
    paths = [str(p) for p in archives]
    labels = {path: _identity_of(path) for path in paths}
    named = [target, *covariates]
    twice = sorted({n for n in named if named.count(n) > 1})
    if twice:
        raise ValueError(f"tag(s) named twice among the target and covariates: {', '.join(twice)}")
    target_path = _resolve(target, labels, "target")
    covariate_paths = [_resolve(name, labels, "covariate") for name in covariates]
    if target_path in covariate_paths:
        raise ValueError(f"the target {labels[target_path]} is also named as a covariate")
    t_read = read_span(target_path, plan.start, plan.schedule_end)
    c_reads = tuple(read_span(p, plan.start, plan.schedule_end) for p in covariate_paths)
    for read in c_reads:
        assert_windows_comparable(t_read, read)
    tag = str(t_read.identity)
    times, y = _samples(t_read)
    direct = _estimate(y, None, (), times, plan, design, tag)
    adjusted = None
    if c_reads:
        y_adj, x = _joined(times, y, c_reads)
        names = tuple(str(r.identity) for r in c_reads)
        adjusted = _estimate(y_adj, x, names, times, plan, design, tag)
    checks = []
    for read in c_reads:
        c_times, c_values = _samples(read)
        c_tag = str(read.identity)
        checks.append(
            CovariateCheck(
                tag=c_tag,
                unit=unit_name(read),
                difference=_estimate(c_values, None, (), c_times, plan, design, c_tag),
            )
        )
    used = {target_path, *covariate_paths}
    return SwitchbackAnalysis(
        plan=plan,
        target=t_read,
        covariates=c_reads,
        unused=tuple(labels[p] for p in paths if p not in used),
        direct=direct,
        adjusted=adjusted,
        covariate_checks=tuple(checks),
    )


def with_power(
    plan: SwitchbackPlan, history: str | Path, window: tuple[pd.Timestamp, pd.Timestamp]
) -> SwitchbackPlan:
    """``plan`` carrying the power readout of ``history`` over ``window``."""
    return replace(plan, power=power_readout(history, window, plan))


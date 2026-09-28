"""Randomized switchback schedules scored on real records with a known injected shift.

Each record is one where nothing was changed. A draw cuts it into blocks
of L steps, assigns half of them to setting B at random, adds the lagged
response to an injected shift of delta * sigma in the B blocks
(sigma = 1.4826 * MAD of the target over the whole record) and runs every
inference arm of ``switchback.py``. The injected shift is the truth, so a
claim at delta 0 is a false claim and a claim at delta > 0 is a detection.

Beds, loaded with the shift interval study's loaders (``run_shift.py``):

- ``3w``: ``placebo_3w`` on ``data/3w_windows``, the 538 instances with no
  fault window and 4 consecutive one-hour windows, 240 minute medians of
  the first such run. Every usable tag is the target in turn. L 15, 30
  and 60 min, 4 draws per record.
- ``tep``: runs 251-350 of the fault-free testing file, all 960 samples at
  180 s, from the cache ``build_tep_cache.py`` writes. Every variable is
  the target in turn. L 40 and 80 samples, 10 draws per run.
- ``turbine``: ``load_turbine`` on ``data/turbine_upgrade``, the rows of
  each pair before its upgrade, target ``y_test``, calendar blocks from
  the first timestamp in steps of 10 min. L 1 and 3 days, 250 draws per
  pair.
- ``skab``: ``load_skab`` on ``data/skab_archives``, the anomaly-free
  record, 1 s rows as recorded, calendar blocks. Every usable tag is the
  target in turn. L 5 and 10 min, 500 draws.

A tag is usable when it has 30 finite samples and MAD > 0 over the record.
The adjusted arm regresses on every other usable tag of the record (3W,
SKAB, TEP) or on the usable tags of V, VcosD, VsinD, rho, S, I and y_ctrl
(turbine). Grid: delta 0, 0.1, 0.25 and 0.5; tau 0, L/10 and L/4 steps;
washout 0 and ceil(3 tau) steps; raw and adjusted. The seed of a draw
comes from (bed, record, L, draw) alone, so every target of a record
shares the draw's schedule.

Outputs in ``--out``, merged per bed so one bed can be rerun alone:

- ``summary.csv``: one row per (bed, L, tau, washout, delta, arm,
  adjustment) with the claim rate at delta 0 or the detection rate
  otherwise, pooled over (record, target, draw) with a Monte Carlo SE
  clustered by (record, draw), the rate averaged per record, coverage of
  the exact truth and its SE, the median width and median bias in sigma,
  the median width ratio adjusted over raw, the answered share and the
  most frequent refusal.
- ``per_well_3w.csv``: the 3W claim rates at delta 0 per well.
- ``turbine_pairs.csv``: the turbine rows per pair.
- ``designs.csv``: K, the number of balanced assignments, enumeration,
  the smallest attainable p and the refusal per (bed, L, K).
- ``refusals.csv``: refused (record, target, draw) units per (bed, L,
  washout, adjustment, arm, reason).
- ``run.json``: provenance, parameters, seeds, counts and wall seconds.

Per-row results go to ``--rows`` (untracked), one parquet per bed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE.parent / "shift_intervals"))
sys.path.insert(0, str(HERE))

import run_shift as rs  # noqa: E402
import switchback as sb  # noqa: E402

sh = sb.sh

DEFAULT_WINDOWS = "data/3w_windows"
DEFAULT_3W_SOURCE = "data/3w"
DEFAULT_SKAB_SOURCE = "data/skab"
DEFAULT_TEP_CACHE = "data/tep_switchback_cache"
DEFAULT_TURBINE = "data/turbine_upgrade"
DEFAULT_ARCHIVES = "data/skab_archives"
DEFAULT_ROWS = "data/switchback_rows"
DEFAULT_WORKERS = 8
BUDGET_SECONDS = 1800

BED_3W = "3w"
BED_TEP = "tep"
BED_TURBINE = "turbine"
BED_SKAB = "skab"
BEDS = (BED_3W, BED_TEP, BED_TURBINE, BED_SKAB)


@dataclass(frozen=True)
class BedSpec:
    blocks: tuple[int, ...]  # L in steps
    draws: int
    draw_chunk: int  # draws per task
    step_seconds: float


BED_SPECS = {
    BED_3W: BedSpec((15, 30, 60), 4, 4, 60.0),
    BED_TEP: BedSpec((40, 80), 10, 10, 180.0),
    BED_TURBINE: BedSpec((144, 432), 250, 25, 600.0),
    BED_SKAB: BedSpec((300, 600), 500, 25, 1.0),
}
DELTAS = (0.0, 0.1, 0.25, 0.5)
TAU_DIVISORS = (None, 10, 4)  # tau = 0, L/10, L/4
TEP_VARIABLES = tuple(rs.TEP_VARIABLES)
SUMMARY_CSV = "summary.csv"
PER_WELL_CSV = "per_well_3w.csv"
PAIRS_CSV = "turbine_pairs.csv"
DESIGNS_CSV = "designs.csv"
REFUSALS_CSV = "refusals.csv"
REASON_CODE = {reason: code for code, reason in enumerate(sb.REASONS)}


# ---------------------------------------------------------------- helpers


def r4(value) -> float | None:
    """Four decimals; NaN and None become None, infinities stay."""
    if value is None:
        return None
    v = float(value)
    if math.isnan(v):
        return None
    return v if math.isinf(v) else round(v, 4) + 0.0  # + 0.0 turns -0.0 into 0.0


def round_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        out[column] = [r4(v) for v in out[column]]
    return out


def draw_seed(bed: str, record: str, length: int, draw: int) -> int:
    return sh.cell_seed({"bed": bed, "record": record, "block": length, "draw": draw})


def tau_washouts(length: int) -> list[tuple[float, int]]:
    """The (tau, washout) pairs of one L: washout 0 and ceil(3 tau), once each."""
    out = []
    for divisor in TAU_DIVISORS:
        tau = 0.0 if divisor is None else length / divisor
        for washout in dict.fromkeys((0, sb.washout_steps(tau))):
            out.append((tau, washout))
    return out


def block_label(bed: str, length: int) -> str:
    seconds = length * BED_SPECS[bed].step_seconds
    for unit, size in (("d", 86400), ("h", 3600), ("min", 60)):
        if seconds >= size and seconds % size == 0:
            return f"{int(seconds // size)} {unit}"
    return f"{seconds:g} s"


def bed_cells(bed: str) -> list[tuple]:
    """Every (L, tau, washout, delta, arm, adjustment) of a bed, in output order."""
    return [
        (length, tau, washout, delta, arm, adjustment)
        for length in BED_SPECS[bed].blocks
        for tau, washout in tau_washouts(length)
        for delta in DELTAS
        for arm in sb.ARMS
        for adjustment in sb.ADJUSTMENTS
    ]


def usable_reason(values: np.ndarray) -> str:
    finite = values[np.isfinite(values)]
    if finite.size < sb.MIN_SAMPLES:
        return sb.TOO_FEW
    return sb.NO_SPREAD if sh.mad(finite) == 0 else ""


def sigma_of(values: np.ndarray) -> float:
    return sb.MAD_TO_SD * sh.mad(values[np.isfinite(values)])


# ---------------------------------------------------------------- records


@dataclass
class Record:
    """One record where nothing was changed; ``times`` are steps from its first sample."""

    record: str
    group: str
    times: np.ndarray
    span: float
    values: dict[str, np.ndarray]
    targets: tuple[str, ...]
    pool: tuple[str, ...]  # the tags a target's covariates are drawn from

    def covariates(self, target: str) -> list[str]:
        return [t for t in self.pool if t != target and usable_reason(self.values[t]) == ""]


def records_3w(cache: Path) -> tuple[list[Record], dict]:
    splits, counts = rs.placebo_3w(cache)
    records = []
    for split in splits:
        tags = tuple(sorted(split.before))
        values = {t: np.concatenate([split.before[t], split.after[t]]) for t in tags}
        n = len(values[tags[0]])
        records.append(
            Record(
                split.record, split.group, np.arange(n, dtype=float), float(n), values, tags, tags
            )
        )
    return records, counts


def records_tep(cache: Path) -> tuple[list[Record], dict]:
    manifest = json.loads((cache / "MANIFEST.json").read_text("utf-8"))
    frame = pd.read_parquet(cache / manifest["cache"])
    records = []
    for run, part in frame.groupby("run", sort=True):
        part = part.sort_values("sample", kind="stable")
        n = len(part)
        values = {v: part[v].to_numpy(dtype=float) for v in TEP_VARIABLES}
        name = f"run{int(run):03d}"
        records.append(
            Record(
                name,
                name,
                np.arange(n, dtype=float),
                float(n),
                values,
                TEP_VARIABLES,
                TEP_VARIABLES,
            )
        )
    counts = {
        "cache": rs.rel(cache),
        "cache_sha256": manifest["cache_sha256"],
        "source_manifest_sha256": manifest["source_manifest_sha256"],
        "runs": manifest["runs"],
        "n_runs": len(records),
    }
    return records, counts


def records_turbine(directory: Path) -> tuple[list[Record], dict]:
    pairs, _, counts = rs.load_turbine(directory)
    records = []
    for name in sorted(pairs):
        frame = pairs[name].frame
        frame = frame[frame["status"] == 0]
        times = ((frame["time"] - frame["time"].iloc[0]) / rs.TURBINE_STEP).to_numpy(dtype=float)
        values = {
            c: frame[c].to_numpy(dtype=float) for c in (rs.TURBINE_TARGET, *rs.TURBINE_COVARIATES)
        }
        records.append(
            Record(
                name,
                name,
                times,
                float(times[-1] + 1),
                values,
                (rs.TURBINE_TARGET,),
                tuple(rs.TURBINE_COVARIATES),
            )
        )
    return records, counts


def records_skab(archives: Path) -> tuple[list[Record], dict]:
    loaded, counts = rs.load_skab(archives)
    records = []
    for rec in loaded:
        if rec.anomaly is not None:
            continue
        stamps = rec.frame["timestamp"]
        times = (stamps - stamps.iloc[0]).dt.total_seconds().to_numpy(dtype=float)
        tags = tuple(sorted(rec.tags))
        values = {t: rec.frame[t].to_numpy(dtype=float) for t in tags}
        records.append(
            Record(rec.record, rec.folder, times, float(times[-1] + 1), values, tags, tags)
        )
    counts = {
        **counts,
        "n_anomaly_free": len(records),
        "duplicate_timestamps": int(sum(np.sum(np.diff(r.times) == 0) for r in records)),
    }
    return records, counts


# ---------------------------------------------------------------- scoring


@dataclass
class Task:
    bed: str
    record_id: int
    record: Record
    length: int
    draws: tuple[int, ...]
    prepost: bool


FIELDS = (
    "cell",
    "target",
    "draw",
    "reason",
    "estimate",
    "lo",
    "hi",
    "p_value",
    "truth",
    "claim",
    "covered",
)


class RowBuffer:
    """Rows of one task, collected as arrays and joined once."""

    def __init__(self) -> None:
        self.parts: dict[str, list[np.ndarray]] = {f: [] for f in FIELDS}

    def add(
        self,
        cell,
        target,
        draw,
        reason,
        estimate=np.nan,
        lo=np.nan,
        hi=np.nan,
        p_value=np.nan,
        truth=np.nan,
        claim=False,
        covered=False,
    ) -> None:
        cell = np.atleast_1d(np.asarray(cell, dtype=np.int32))
        n = cell.size
        values = (cell, target, draw, reason, estimate, lo, hi, p_value, truth, claim, covered)
        for name, value in zip(FIELDS, values, strict=True):
            self.parts[name].append(np.broadcast_to(np.asarray(value), (n,)))

    def arrays(self) -> dict[str, np.ndarray]:
        dtypes = {
            "cell": np.int16,
            "target": np.int16,
            "draw": np.int16,
            "reason": np.int8,
            "claim": bool,
            "covered": bool,
        }
        out = {}
        for name in FIELDS:
            joined = np.concatenate(self.parts[name]) if self.parts[name] else np.zeros(0)
            out[name] = joined.astype(dtypes.get(name, np.float32))
        return out


@dataclass(frozen=True)
class Plan:
    """Which cell each output row fills: config index, arm index, cell id."""

    config: np.ndarray
    arm: np.ndarray
    cell: np.ndarray


def _configs(tws: list[tuple[float, int]], washout: int) -> list[tuple[list[float], float, float]]:
    """(taus the row fills, tau of the response, delta) for one washout.

    At delta 0 the response is 0 whatever tau, so one computation fills the
    cell of every tau paired with this washout.
    """
    taus = [tau for tau, w in tws if w == washout]
    out = [(taus, 0.0, 0.0)]
    out += [([tau], tau, delta) for tau in taus for delta in DELTAS if delta > 0]
    return out


def _plan(cells: dict, length, washout, configs, arms, adjustment) -> Plan:
    config, arm, cell = [], [], []
    for j, (taus, _, delta) in enumerate(configs):
        for tau in taus:
            for a, name in enumerate(arms):
                config.append(j)
                arm.append(a)
                cell.append(cells[(length, tau, washout, delta, name, adjustment)])
    return Plan(np.array(config), np.array(arm), np.array(cell))


def _claims(arm_names: tuple[str, ...], plan: Plan, lo, hi, p) -> np.ndarray:
    randomization = np.array([n == sb.RANDOMIZATION for n in arm_names])[plan.arm]
    by_interval = (lo > 0) | (hi < 0)
    return np.where(randomization, p <= sb.ALPHA, by_interval)


def _emit(
    rows: RowBuffer, plan: Plan, arm_names, target, draw, reasons, est, lo, hi, p, truth
) -> None:
    """Rows of one (target, draw) from (configs, arms) matrices of results in sigma units."""
    j, a = plan.config, plan.arm
    reason = reasons[j, a]
    answered = reason == 0
    e, lo_, hi_, p_, t = est[j, a], lo[j, a], hi[j, a], p[j, a], truth[j]
    claim = answered & _claims(arm_names, plan, lo_, hi_, p_)
    covered = answered & (lo_ <= t) & (t <= hi_)
    rows.add(plan.cell, target, draw, reason, e, lo_, hi_, p_, t, claim, covered)


def _refuse(rows: RowBuffer, plan: Plan, target: int, draws, reason: str) -> None:
    draws = np.asarray(draws, dtype=np.int64)
    rows.add(
        np.tile(plan.cell, draws.size),
        target,
        np.repeat(draws, plan.cell.size),
        REASON_CODE[reason],
    )


def score_task(task: Task) -> dict[str, np.ndarray]:
    """Rows of every target, draw and cell of one (record, L, draw chunk)."""
    rec = task.record
    length = task.length
    cells = {key: i for i, key in enumerate(bed_cells(task.bed))}
    blocks = sb.cut_blocks(rec.times, rec.span, length)
    tws = tau_washouts(length)
    washouts = sorted({w for _, w in tws})
    refusal = sb.design_refusal(blocks.k)
    rows = RowBuffer()
    designs = (
        {}
        if refusal
        else {
            d: sb.make_design(blocks.k, draw_seed(task.bed, rec.record, length, d))
            for d in task.draws
        }
    )
    responses: dict[tuple[int, float], np.ndarray] = {}

    def response(draw: int, tau: float) -> np.ndarray:
        key = (draw, tau)
        if key not in responses:
            u = sb.setting(blocks, designs[draw].observed)
            responses[key] = sb.lag_response(u, rec.times, tau)
        return responses[key]

    for ti, target in enumerate(rec.targets):
        y = rec.values[target]
        target_reason = refusal or usable_reason(y)
        sigma = sigma_of(y) if not target_reason else math.nan
        covariates = rec.covariates(target)
        for adjustment in sb.ADJUSTMENTS:
            x = None
            adj_reason = target_reason
            if adjustment == sb.ADJUSTED:
                if covariates:
                    x = np.column_stack([rec.values[c] for c in covariates])
                elif not adj_reason:
                    adj_reason = sb.NO_COVARIATE
            for washout in washouts:
                configs = _configs(tws, washout)
                plan = _plan(cells, length, washout, configs, sb.SCHEDULE_ARMS, adjustment)
                if adj_reason:
                    _refuse(rows, plan, ti, task.draws, adj_reason)
                    continue
                frame = sb.make_frame(y, x, blocks, washout)
                if isinstance(frame, str):
                    _refuse(rows, plan, ti, task.draws, frame)
                    continue
                for draw in task.draws:
                    _score_draw(
                        rows, plan, frame, designs[draw], configs, response, draw, ti, sigma
                    )
        if task.prepost:
            _score_prepost(
                rows, cells, rec, blocks, tws, length, ti, target, target_reason, covariates
            )
    if task.prepost:
        sh._cosine_basis.cache_clear()  # a 20,000-sample basis holds tens of MB
    return rows.arrays()


def _score_draw(rows, plan, frame, design, configs, response, draw, ti, sigma) -> None:
    terms = sb.assignment_terms(frame, design)
    if isinstance(terms, str):
        _refuse(rows, plan, ti, [draw], terms)
        return
    m = len(configs)
    sums = np.repeat(frame.resid_sums[:, None], m, axis=1)
    resid = np.repeat(frame.resid[:, None], m, axis=1)
    truth = np.zeros(m)
    z = terms.z_kept == 1
    unit: dict[float, tuple[np.ndarray, np.ndarray, float]] = {}
    for j, (_, tau, delta) in enumerate(configs):
        if delta == 0:
            continue
        if tau not in unit:
            g = response(draw, tau)[frame.positions]
            rg = frame.project_out(g)
            unit[tau] = (frame.sums(rg), rg, float(g[z].mean() - g[~z].mean()))
        g_sums, g_resid, g_truth = unit[tau]
        sums[:, j] += delta * sigma * g_sums
        resid[:, j] += delta * sigma * g_resid
        truth[j] = delta * g_truth
    results = {sb.RANDOMIZATION: sb.randomization(design, terms, sums, sigma)}
    results.update(sb.sample_arms(frame, terms, resid))
    arms = sb.SCHEDULE_ARMS
    shape = (m, len(arms))
    est, lo, hi, p = (np.empty(shape) for _ in range(4))
    reasons = np.zeros(shape, dtype=np.int64)
    for a, name in enumerate(arms):
        res = results[name]
        est[:, a], lo[:, a], hi[:, a], p[:, a] = res.estimate, res.lo, res.hi, res.p_value
        reasons[:, a] = REASON_CODE[res.reason]
    _emit(rows, plan, arms, ti, draw, reasons, est / sigma, lo / sigma, hi / sigma, p, truth)


def _score_prepost(
    rows, cells, rec, blocks, tws, length, ti, target, target_reason, covariates
) -> None:
    """The before/after reference: first half of the schedule span as A, second half as B."""
    y = rec.values[target]
    span = blocks.k * length
    halves = sb.cut_blocks(rec.times, span, span / 2) if blocks.k else None
    sigma = sigma_of(y) if not target_reason else math.nan
    arms = sb.PREPOST_ARMS
    for adjustment in sb.ADJUSTMENTS:
        x = None
        reason = target_reason
        if adjustment == sb.ADJUSTED:
            if covariates:
                x = np.column_stack([rec.values[c] for c in covariates])
            elif not reason:
                reason = sb.NO_COVARIATE
        for washout in sorted({w for _, w in tws}):
            configs = _configs(tws, washout)
            plan = _plan(cells, length, washout, configs, arms, adjustment)
            if reason:
                _refuse(rows, plan, ti, [-1], reason)
                continue
            kept = (halves.block >= 0) & (halves.offset >= washout) & np.isfinite(y)
            if x is not None:
                kept &= np.all(np.isfinite(x), axis=1)
            in_b = halves.block == 1
            u = in_b.astype(float)
            m = len(configs)
            shape = (m, len(arms))
            est, lo, hi, p = (np.full(shape, np.nan) for _ in range(4))
            reasons = np.zeros(shape, dtype=np.int64)
            truth = np.zeros(m)
            for j, (_, tau, delta) in enumerate(configs):
                g = sb.lag_response(u, rec.times, tau)
                shifted = y + delta * sigma * g
                before, after = kept & ~in_b, kept & in_b
                truth[j] = delta * float(g[after].mean() - g[before].mean())
                for a, name in enumerate(arms):
                    got = sb.prepost(
                        shifted[before],
                        shifted[after],
                        sb.PREPOST_METHOD[name],
                        None if x is None else x[before],
                        None if x is None else x[after],
                    )
                    if isinstance(got, str):
                        reasons[j, a] = REASON_CODE[got]
                    else:
                        est[j, a], lo[j, a], hi[j, a], p[j, a] = got
            _emit(rows, plan, arms, ti, -1, reasons, est / sigma, lo / sigma, hi / sigma, p, truth)


def make_tasks(bed: str, records: list[Record], draws: int) -> list[Task]:
    spec = BED_SPECS[bed]
    tasks = []
    for record_id, rec in enumerate(records):
        for length in spec.blocks:
            for start in range(0, draws, spec.draw_chunk):
                chunk = tuple(range(start, min(start + spec.draw_chunk, draws)))
                tasks.append(Task(bed, record_id, rec, length, chunk, start == 0))
    return tasks


# ---------------------------------------------------------------- aggregation


def cluster_mcse(x: np.ndarray, cluster: np.ndarray) -> float:
    """Monte Carlo SE of mean(x) with clusters, G / (G - 1) small-sample factor."""
    n = x.size
    if n == 0:
        return math.nan
    ids, inverse = np.unique(cluster, return_inverse=True)
    if ids.size < 2:
        return math.nan
    sums = np.bincount(inverse, weights=x - x.mean())
    return math.sqrt(ids.size / (ids.size - 1) * float(np.dot(sums, sums))) / n


def _top_reason(codes: np.ndarray) -> str:
    refused = codes[codes != 0]
    if refused.size == 0:
        return ""
    counts = np.bincount(refused, minlength=len(sb.REASONS))
    return sb.REASONS[int(counts.argmax())]


def _cell_stats(rows: dict[str, np.ndarray], index: np.ndarray, delta: float) -> dict:
    reason = rows["reason"][index]
    answered = index[reason == 0]
    n = answered.size
    out = {
        "n_rows": int(index.size),
        "n_answered": int(n),
        "answered_share": n / index.size if index.size else math.nan,
        "top_reason": _top_reason(reason),
    }
    if n == 0:
        return out | {"n_records": 0}
    record = rows["record"][answered].astype(np.int64)
    cluster = record * 100_000 + rows["draw"][answered].astype(np.int64) + 1
    claim = rows["claim"][answered].astype(float)
    covered = rows["covered"][answered].astype(float)
    per_record_n = np.bincount(record)
    per_record_claim = np.bincount(record, weights=claim)
    has = per_record_n > 0
    width = rows["width"][answered].astype(float)
    bias = rows["estimate"][answered].astype(float) - delta
    return out | {
        "n_records": int(has.sum()),
        "rate": float(claim.mean()),
        "rate_mcse": cluster_mcse(claim, cluster),
        "rate_record_mean": float(np.mean(per_record_claim[has] / per_record_n[has])),
        "coverage": float(covered.mean()),
        "coverage_mcse": cluster_mcse(covered, cluster),
        "median_width": float(np.median(width)),
        "unbounded_share": float(np.mean(~np.isfinite(width))),
        "median_bias": float(np.median(bias)),
    }


def _sorted_rows(rows, index) -> np.ndarray:
    keys = (rows["draw"][index], rows["target"][index], rows["record"][index])
    return index[np.lexsort(keys)]


def _width_ratio(rows, raw_index, adj_index) -> float:
    a, b = _sorted_rows(rows, raw_index), _sorted_rows(rows, adj_index)
    if a.size != b.size:
        raise ValueError("raw and adjusted cells hold different units")
    ok = (rows["reason"][a] == 0) & (rows["reason"][b] == 0)
    wa, wb = rows["width"][a][ok].astype(float), rows["width"][b][ok].astype(float)
    good = np.isfinite(wa) & np.isfinite(wb) & (wa > 0)
    return float(np.median(wb[good] / wa[good])) if good.any() else math.nan


SUMMARY_FLOATS = [
    "tau",
    "delta",
    "answered_share",
    "rate",
    "rate_mcse",
    "rate_record_mean",
    "coverage",
    "coverage_mcse",
    "median_width",
    "unbounded_share",
    "median_bias",
    "median_width_ratio",
]


def aggregate(bed: str, rows: dict[str, np.ndarray], records: list[Record]) -> dict:
    """Summary, per-well, per-pair and refusal frames of one bed."""
    cells = bed_cells(bed)
    order = np.argsort(rows["cell"], kind="stable")
    bounds = np.searchsorted(rows["cell"][order], np.arange(len(cells) + 1))
    index_of = {i: order[bounds[i] : bounds[i + 1]] for i in range(len(cells))}
    lookup = {key: i for i, key in enumerate(cells)}
    summary, pairs, per_well, refusals = [], [], [], []
    group_of = np.array([r.group for r in records])
    for i, (length, tau, washout, delta, arm, adjustment) in enumerate(cells):
        index = index_of[i]
        head = {
            "bed": bed,
            "block": length,
            "block_label": block_label(bed, length),
            "tau": tau,
            "washout": washout,
            "washout_rule": "ceil(3 tau)" if washout == sb.washout_steps(tau) else "0",
            "delta": delta,
            "arm": arm,
            "adjustment": adjustment,
        }
        stats = _cell_stats(rows, index, delta)
        ratio = math.nan
        if adjustment == sb.ADJUSTED:
            raw = index_of[lookup[(length, tau, washout, delta, arm, sb.RAW)]]
            ratio = _width_ratio(rows, raw, index)
        summary.append(head | stats | {"median_width_ratio": ratio})
        if bed == BED_TURBINE:
            for rid, rec in enumerate(records):
                sub = index[rows["record"][index] == rid]
                pairs.append(head | {"pair": rec.record} | _cell_stats(rows, sub, delta))
        if bed == BED_3W and delta == 0:
            groups = group_of[rows["record"][index]]
            for well in sorted(set(group_of)):
                sub = index[groups == well]
                got = _cell_stats(rows, sub, delta)
                per_well.append(
                    head
                    | {
                        "well": well,
                        "n_answered": got["n_answered"],
                        "rate": got.get("rate", math.nan),
                    }
                )
        first_tau = min(t for t, w in tau_washouts(length) if w == washout)
        if delta == 0 and tau == first_tau:
            codes = rows["reason"][index]
            for code, count in sorted(Counter(codes[codes != 0].tolist()).items()):
                refusals.append(
                    {
                        "bed": bed,
                        "block": length,
                        "washout": washout,
                        "adjustment": adjustment,
                        "arm": arm,
                        "reason": sb.REASONS[code],
                        "n_units": int(count),
                        "n_units_total": int(index.size),
                    }
                )
    return {"summary": summary, "pairs": pairs, "per_well": per_well, "refusals": refusals}


def design_rows(bed: str, records: list[Record]) -> list[dict]:
    seen: dict[tuple[int, int], int] = {}
    for rec in records:
        for length in BED_SPECS[bed].blocks:
            k = sb.schedule_blocks(rec.span, length)
            seen[(length, k)] = seen.get((length, k), 0) + 1
    out = []
    for (length, k), n_records in sorted(seen.items()):
        n, enumerated, min_p = sb.design_size(k)
        out.append(
            {
                "bed": bed,
                "block": length,
                "block_label": block_label(bed, length),
                "k": k,
                "n_records": n_records,
                "n_assignments": n if n < 10**6 else f"{n:.3e}",
                "enumerated": enumerated,
                "min_p": r4(min_p),
                "refusal": sb.design_refusal(k),
            }
        )
    return out


# ---------------------------------------------------------------- run


def _load(bed: str, paths: dict) -> tuple[list[Record], dict]:
    if bed == BED_3W:
        records, counts = records_3w(paths["windows"])
        return records, counts | {"source_manifest_sha256": rs.source_sha(paths["source_3w"])}
    if bed == BED_TEP:
        return records_tep(paths["tep_cache"])
    if bed == BED_TURBINE:
        return records_turbine(paths["turbine"])
    records, counts = records_skab(paths["archives"])
    return records, counts | {"source_manifest_sha256": rs.source_sha(paths["source_skab"])}


def _write_rows(path: Path, bed: str, frame: pd.DataFrame, records, cells) -> None:
    path.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path / f"{bed}.parquet", index=False)
    codes = {
        "records": [r.record for r in records],
        "targets": sorted({t for r in records for t in r.targets}),
        "targets_per_record": "index into the record's own target order",
        "cells": [list(c) for c in cells],
        "reasons": list(sb.REASONS),
    }
    (path / f"{bed}_codes.json").write_text(json.dumps(codes) + "\n", encoding="utf-8")


def score_bed(bed: str, records: list[Record], draws: int, workers: int) -> dict[str, np.ndarray]:
    tasks = make_tasks(bed, records, draws)
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(score_task, tasks, chunksize=1))
    else:
        results = [score_task(t) for t in tasks]
    joined = {}
    for name in FIELDS:
        joined[name] = np.concatenate([r[name] for r in results])
    joined["record"] = np.concatenate(
        [
            np.full(r["cell"].size, t.record_id, dtype=np.int16)
            for r, t in zip(results, tasks, strict=True)
        ]
    )
    return joined


def _merge_csv(path: Path, frame: pd.DataFrame, beds: tuple[str, ...], sort: list[str]) -> None:
    if path.exists():
        old = pd.read_csv(path, keep_default_na=False, na_values=[""], dtype={"bed": str})
        old = old[~old["bed"].isin(beds)]
        frame = pd.concat([old, frame], ignore_index=True) if len(old) else frame
    keys = frame.copy()
    keys["_bed"] = keys["bed"].map({b: i for i, b in enumerate(BEDS)})
    if "arm" in keys:
        keys["_arm"] = keys["arm"].map({a: i for i, a in enumerate(sb.ARMS)})
    if "adjustment" in keys:
        keys["_adj"] = keys["adjustment"].map({a: i for i, a in enumerate(sb.ADJUSTMENTS)})
    order = keys.sort_values(["_bed", *sort], kind="stable").index
    rs.write_csv(frame.loc[order].reset_index(drop=True), path)


def run(
    out: Path,
    beds: tuple[str, ...],
    *,
    windows: Path,
    tep_cache: Path,
    turbine: Path,
    archives: Path,
    rows_dir: Path | None,
    source_3w: Path | None = None,
    source_skab: Path | None = None,
    workers: int = 1,
    draws: dict[str, int] | None = None,
) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "windows": windows,
        "tep_cache": tep_cache,
        "turbine": turbine,
        "archives": archives,
        "source_3w": source_3w,
        "source_skab": source_skab,
    }
    run_path = out / "run.json"
    info = json.loads(run_path.read_text("utf-8")) if run_path.exists() else {}
    info.setdefault("beds", {})
    frames = {k: [] for k in ("summary", "pairs", "per_well", "refusals", "designs")}
    for bed in beds:
        started = time.perf_counter()
        records, counts = _load(bed, paths)
        n_draws = (draws or {}).get(bed, BED_SPECS[bed].draws)
        rows = score_bed(bed, records, n_draws, workers)
        scored = time.perf_counter() - started
        rows["width"] = rows["hi"] - rows["lo"]
        agg = aggregate(bed, rows, records)
        for key in ("summary", "pairs", "per_well", "refusals"):
            frames[key] += agg[key]
        frames["designs"] += design_rows(bed, records)
        if rows_dir is not None:
            keep = {k: rows[k] for k in (*FIELDS, "record")}
            _write_rows(rows_dir, bed, pd.DataFrame(keep), records, bed_cells(bed))
        n_targets = sum(len(r.targets) for r in records)
        n_usable = sum(usable_reason(r.values[t]) == "" for r in records for t in r.targets)
        info["beds"][bed] = {
            "code": rs.code_version(),
            "dataset": counts,
            "blocks": list(BED_SPECS[bed].blocks),
            "block_labels": [block_label(bed, b) for b in BED_SPECS[bed].blocks],
            "draws": n_draws,
            "draws_registered": BED_SPECS[bed].draws,
            "n_records": len(records),
            "n_groups": len({r.group for r in records}),
            "n_targets": n_targets,
            "n_usable_targets": int(n_usable),
            "n_rows": int(rows["cell"].size),
            "seed": "shift.cell_seed of (bed, record, block, draw)",
            "wall_seconds": round(time.perf_counter() - started, 1),
            "scoring_seconds": round(scored, 1),
        }
        del rows
    sort = ["block", "tau", "washout", "delta", "_arm", "_adj"]
    _merge_csv(
        out / SUMMARY_CSV, round_frame(pd.DataFrame(frames["summary"]), SUMMARY_FLOATS), beds, sort
    )
    if BED_3W in beds:
        per_well = round_frame(pd.DataFrame(frames["per_well"]), ["tau", "delta", "rate"])
        per_well = per_well[
            ["block", "tau", "washout", "delta", "arm", "adjustment", "well", "n_answered", "rate"]
        ]
        rs.write_csv(per_well, out / PER_WELL_CSV)
    if BED_TURBINE in beds:
        pairs = round_frame(
            pd.DataFrame(frames["pairs"]), [c for c in SUMMARY_FLOATS if c != "median_width_ratio"]
        )
        rs.write_csv(pairs, out / PAIRS_CSV)
    _merge_csv(out / DESIGNS_CSV, pd.DataFrame(frames["designs"]), beds, ["block", "k"])
    _merge_csv(
        out / REFUSALS_CSV,
        pd.DataFrame(frames["refusals"]),
        beds,
        ["block", "washout", "_adj", "_arm", "reason"],
    )
    info["code"] = rs.code_version()
    info["parameters"] = {
        "alpha": sb.ALPHA,
        "permutations": sb.PERMUTATIONS,
        "min_assignments": sb.MIN_ASSIGNMENTS,
        "deltas": list(DELTAS),
        "tau": "0, L/10, L/4 steps",
        "washout": "0 and ceil(3 tau) steps",
        "sigma": "1.4826 * MAD of the target over the whole record",
        "samples_per_covariate": sb.SAMPLES_PER_COVARIATE,
        "min_samples": sb.MIN_SAMPLES,
        "blocks": {b: list(BED_SPECS[b].blocks) for b in BEDS},
        "draws": {b: BED_SPECS[b].draws for b in BEDS},
    }
    info["workers"] = workers
    info["budget_seconds"] = BUDGET_SECONDS
    info["wall_seconds_all_beds"] = round(
        sum(section["wall_seconds"] for section in info["beds"].values()), 1
    )
    reduced = {
        b: s["draws"] for b, s in info["beds"].items() if s["draws"] != s["draws_registered"]
    }
    info["draws_reduced"] = reduced or None
    run_path.write_text(
        json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    return info


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--beds", nargs="+", choices=BEDS, default=list(BEDS))
    parser.add_argument("--out", default=str(HERE / "results"))
    parser.add_argument("--windows", default=str(ROOT / DEFAULT_WINDOWS))
    parser.add_argument("--tep-cache", default=str(ROOT / DEFAULT_TEP_CACHE))
    parser.add_argument("--turbine", default=str(ROOT / DEFAULT_TURBINE))
    parser.add_argument("--archives", default=str(ROOT / DEFAULT_ARCHIVES))
    parser.add_argument("--source-3w", default=str(ROOT / DEFAULT_3W_SOURCE))
    parser.add_argument("--source-skab", default=str(ROOT / DEFAULT_SKAB_SOURCE))
    parser.add_argument("--rows", default=str(ROOT / DEFAULT_ROWS))
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--draws",
        nargs="*",
        default=[],
        metavar="BED=N",
        help="override the registered draws of a bed",
    )
    args = parser.parse_args(argv)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    draws = {}
    for item in args.draws:
        bed, _, n = item.partition("=")
        draws[bed] = int(n)
    info = run(
        Path(args.out),
        tuple(b for b in BEDS if b in args.beds),
        windows=Path(args.windows),
        tep_cache=Path(args.tep_cache),
        turbine=Path(args.turbine),
        archives=Path(args.archives),
        rows_dir=Path(args.rows),
        source_3w=Path(args.source_3w),
        source_skab=Path(args.source_skab),
        workers=args.workers,
        draws=draws,
    )
    for bed in args.beds:
        section = info["beds"][bed]
        print(f"{bed}: {section['n_rows']} rows, {section['wall_seconds']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

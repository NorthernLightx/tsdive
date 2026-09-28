"""Before/after shift intervals on synthetic series, placebo dates and known change dates.

The study asks which interval method for a before/after level or spread
shift keeps its coverage on autocorrelated, drifting process data, and
whether covariate adjustment narrows the interval without biasing it.
Estimators and rules live in ``shift.py``.

``synthetic`` draws ``--replicates`` series per cell from the generator in
``shift.py``; each cell's seed comes from its parameters alone. Grids:

- ``level``: phi x rho x drift x delta x n per period.
- ``affected_covariate``: rho 0.9, the recorded covariate steps by half the
  target's shift after the change date, drift 0.
- ``spread``: phi x n x SD ratio, delta 0, rho 0, drift 0.

``placebo`` moves the date where nothing was changed:

- ``3w_placebo``: every 3W instance with no fault window and 4 consecutive
  windows; 2 windows of minute medians before, the next 2 after.
- ``skab_free_placebo``: the SKAB anomaly-free record, non-overlapping
  10 min before and 10 min after pairs from its first timestamp.
- ``skab_preonset_placebo``: each labelled SKAB record's rows before its
  first anomaly row, first half against second half.

``known_change`` puts the date at a fault onset:

- ``3w_aligned_onset``: the 3 onset-aligned windows nearest before the onset
  against the windows after it.
- ``skab_labelled_onset``: rows before the first anomaly row against the
  first to the last anomaly row.

Every tag of a split is the target in turn; its covariates are every other
tag that passes R0 on the split. Nothing is fitted across records.

Outputs in ``--out``:

- ``synthetic.csv``: one row per (cell, quantity, method, adjustment,
  refusal set) with coverage of the true shift and its Monte Carlo SE, the
  claim rate at a zero shift, the detection rate otherwise, the median width
  in before-period SDs, the bias, the median variance reduction and the
  firing rate of each rule.
- ``placebo.csv`` and ``known_change.csv``: one row per (split, tag,
  quantity, method, adjustment) with the estimate and interval (level in
  before-period SDs of the target, spread as a log SD ratio), the p-value,
  whether the interval excludes 0, the hard refusal and its reason, the R1
  to R4 flags, the lag-1 autocorrelation of both periods and the HAC
  bandwidth of the before period of the series the arm reads, and for the
  adjusted arm the variance reduction over the before period and
  var(residual) / var(target) over the after period.
- ``summary.csv``: one row per (design, quantity, method, adjustment,
  refusal set) over both CSVs: clear rate pooled and averaged per well or
  folder, the record-level rate at alpha 0.05 / k over the k answered tags,
  widths, width ratio adjusted over raw, refusal rates, the median lag-1
  autocorrelation and the share of HAC bandwidths at their n - 1 cap.
- ``run.json``: provenance, parameters, seeds, counts and wall seconds per bed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

import shift as sh  # noqa: E402

import tsdive  # noqa: E402

DEFAULT_WINDOWS = "data/3w_windows"
DEFAULT_ALIGNED = "data/3w_windows_aligned"
DEFAULT_3W_SOURCE = "data/3w"
DEFAULT_ARCHIVES = "data/skab_archives"
DEFAULT_SKAB_SOURCE = "data/skab"
DEFAULT_REPLICATES = 200
SYNTHETIC_BUDGET_S = 1200

BED_SYNTHETIC = "synthetic"
BED_PLACEBO = "placebo"
BED_KNOWN = "known_change"
BEDS = (BED_SYNTHETIC, BED_PLACEBO, BED_KNOWN)

DESIGN_3W_PLACEBO = "3w_placebo"
DESIGN_SKAB_FREE = "skab_free_placebo"
DESIGN_SKAB_PRE = "skab_preonset_placebo"
DESIGN_3W_ONSET = "3w_aligned_onset"
DESIGN_SKAB_ONSET = "skab_labelled_onset"

PLACEBO_CSV = "placebo.csv"
KNOWN_CSV = "known_change.csv"
SUMMARY_CSV = "summary.csv"

SUBGROUP_COLS = [f"s{i:02d}" for i in range(60)]
PLACEBO_WINDOWS = 2  # per period
ONSET_BEFORE_WINDOWS = 3
SKAB_PAIR = pd.Timedelta(600, unit="s")
SKAB_GAP_S = 60.0

GRID_LEVEL = "level"
GRID_AFFECTED = "affected_covariate"
GRID_SPREAD = "spread"

PHIS = (0.0, 0.5, 0.9, 0.98)
RHOS = (0.0, 0.5, 0.9)
DRIFTS = (0.0, 1.0)
DELTAS = (0.0, 0.25, 0.5, 1.0)
NS = (120, 480)
SD_RATIOS = (1.0, 0.7, 0.5)
AFFECTED_RHO = 0.9
AFFECTED_DELTAS = (0.25, 0.5, 1.0)
COVARIATE_STEP_FRACTION = 0.5

CELL_KEYS = ("grid", "phi", "rho", "drift", "delta", "n", "sd_ratio", "covariate_step")
FLAG_RULES = (sh.BEFORE_TREND, sh.COVARIATE_OUTSIDE, sh.COVARIATE_SHIFTED, sh.TOO_PERSISTENT)


# ---------------------------------------------------------------- helpers


def git_head() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover
        return "unknown"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def r4(value) -> float | None:
    if value is None:
        return None
    v = float(value)
    return None if not math.isfinite(v) else round(v, 4)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n")


def code_version() -> dict:
    return {"git_head": git_head(), "tsdive_version": tsdive.__version__}


def round_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        out[column] = [r4(v) for v in out[column]]
    return out


# ---------------------------------------------------------------- synthetic


def synthetic_cells() -> list[dict]:
    """Every synthetic cell, in a fixed order that the seeds do not depend on."""
    cells = []
    for phi, rho, drift, delta, n in product(PHIS, RHOS, DRIFTS, DELTAS, NS):
        cells.append(
            {
                "grid": GRID_LEVEL,
                "phi": phi,
                "rho": rho,
                "drift": drift,
                "delta": delta,
                "n": n,
                "sd_ratio": 1.0,
                "covariate_step": 0.0,
            }
        )
    for phi, delta, n in product(PHIS, AFFECTED_DELTAS, NS):
        cells.append(
            {
                "grid": GRID_AFFECTED,
                "phi": phi,
                "rho": AFFECTED_RHO,
                "drift": 0.0,
                "delta": delta,
                "n": n,
                "sd_ratio": 1.0,
                "covariate_step": COVARIATE_STEP_FRACTION * delta,
            }
        )
    for phi, n, ratio in product(PHIS, NS, SD_RATIOS):
        cells.append(
            {
                "grid": GRID_SPREAD,
                "phi": phi,
                "rho": 0.0,
                "drift": 0.0,
                "delta": 0.0,
                "n": n,
                "sd_ratio": ratio,
                "covariate_step": 0.0,
            }
        )
    return cells


def cell_quantity(cell: dict) -> str:
    return sh.SPREAD if cell["grid"] == GRID_SPREAD else sh.LEVEL


def true_value(cell: dict) -> float:
    if cell["grid"] == GRID_SPREAD:
        return math.log(cell["sd_ratio"])
    return float(cell["delta"])


def score_cell(cell: dict, replicates: int) -> tuple[list[dict], int]:
    """Aggregated rows of one cell and the seed it drew from."""
    seed = sh.cell_seed(cell)
    n = cell["n"]
    y, x = sh.simulate(
        phi=cell["phi"],
        rho=cell["rho"],
        drift=cell["drift"],
        delta=cell["delta"],
        n=n,
        replicates=replicates,
        sd_ratio=cell["sd_ratio"],
        covariate_step=cell["covariate_step"],
        seed=seed,
    )
    quantity = cell_quantity(cell)
    rows: list[dict] = []
    for r in range(replicates):
        rows += sh.score_target(
            {"y": y[r, :n], "x": x[r, :n]},
            {"y": y[r, n:], "x": x[r, n:]},
            "y",
            ["x"],
            quantities=(quantity,),
        )
    return aggregate_cell(cell, seed, pd.DataFrame(rows), replicates), seed


def _rate(mask: pd.Series) -> float | None:
    return float(mask.mean()) if len(mask) else None


def aggregate_cell(cell: dict, seed: int, frame: pd.DataFrame, replicates: int) -> list[dict]:
    truth = true_value(cell)
    out = []
    for (method, adjustment), group in frame.groupby(["method", "adjustment"], sort=False):
        sets = sh.RAW_REFUSAL_SETS if adjustment == sh.RAW else tuple(sh.REFUSAL_SETS)
        hard = group["refused"].astype(bool)
        flag_rates = {
            f"rate_{rule}": (
                _rate(group[rule].eq(True))
                if rule in (sh.BEFORE_TREND, sh.TOO_PERSISTENT) or adjustment == sh.ADJUSTED
                else None
            )
            for rule in FLAG_RULES
        }
        for refusal_set in sets:
            refused = hard.copy()
            for rule in sh.REFUSAL_SETS[refusal_set]:
                refused |= group[rule].eq(True)
            answered = group[~refused]
            covered = (answered["lo"] <= truth) & (answered["hi"] >= truth)
            coverage = _rate(covered)
            n_answered = len(answered)
            excludes_zero = _rate(answered["clears"].astype(bool))
            if cell_quantity(cell) == sh.LEVEL:
                width = (answered["hi"] - answered["lo"]) / answered["scale"]
            else:
                width = answered["hi"] - answered["lo"]
            out.append(
                {
                    **{key: cell[key] for key in CELL_KEYS},
                    "seed": seed,
                    "quantity": cell_quantity(cell),
                    "method": method,
                    "adjustment": adjustment,
                    "refusal_set": refusal_set,
                    "true_value": truth,
                    "n_replicates": replicates,
                    "n_answered": n_answered,
                    "coverage": coverage,
                    "coverage_mcse": (
                        math.sqrt(coverage * (1 - coverage) / n_answered)
                        if n_answered
                        else None
                    ),
                    "claim_rate": excludes_zero if truth == 0 else None,
                    "detect_rate": excludes_zero if truth != 0 else None,
                    "bias": (
                        float(answered["estimate"].mean()) - truth if n_answered else None
                    ),
                    "median_width": float(width.median()) if n_answered else None,
                    "median_variance_reduction": (
                        float(group["variance_reduction"].median())
                        if adjustment == sh.ADJUSTED
                        else None
                    ),
                    "rate_refused": _rate(refused),
                    **flag_rates,
                }
            )
    return out


SYNTHETIC_FLOATS = [
    "phi",
    "rho",
    "drift",
    "delta",
    "sd_ratio",
    "covariate_step",
    "true_value",
    "coverage",
    "coverage_mcse",
    "claim_rate",
    "detect_rate",
    "bias",
    "median_width",
    "median_variance_reduction",
    "rate_refused",
    *[f"rate_{rule}" for rule in FLAG_RULES],
]


def run_synthetic(out: Path, replicates: int) -> dict:
    started = time.perf_counter()
    rows: list[dict] = []
    cells = synthetic_cells()
    for cell in cells:
        rows += score_cell(cell, replicates)[0]
    frame = round_frame(pd.DataFrame(rows), SYNTHETIC_FLOATS)
    write_csv(frame, out / "synthetic.csv")
    wall = round(time.perf_counter() - started, 1)
    return {
        "code": code_version(),
        "replicates": replicates,
        "budget_seconds": SYNTHETIC_BUDGET_S,
        "over_budget": wall > SYNTHETIC_BUDGET_S,
        "grid": {
            "phi": list(PHIS),
            "rho": list(RHOS),
            "drift": list(DRIFTS),
            "delta": list(DELTAS),
            "n_per_period": list(NS),
            "sd_ratio": list(SD_RATIOS),
            "affected_rho": AFFECTED_RHO,
            "affected_delta": list(AFFECTED_DELTAS),
            "covariate_step_fraction": COVARIATE_STEP_FRACTION,
        },
        "n_cells": {
            grid: sum(1 for c in cells if c["grid"] == grid)
            for grid in (GRID_LEVEL, GRID_AFFECTED, GRID_SPREAD)
        },
        "seed_rule": (
            "first 8 bytes, little endian, of sha256 over the cell's sorted "
            "name=repr(value) pairs joined by '|'; the seed column of synthetic.csv"
        ),
        "n_rows": len(frame),
        "wall_seconds": wall,
    }


def cell_label(cell: dict) -> str:
    return ",".join(f"{key}={cell[key]}" for key in CELL_KEYS)


# ---------------------------------------------------------------- real records


@dataclass
class Split:
    """One before/after split of one record; each tag's samples are position-aligned."""

    design: str
    record: str
    group: str
    segment: int
    before: dict[str, np.ndarray]
    after: dict[str, np.ndarray]


def rel(path: Path) -> str:
    """The path relative to the repository root with forward slashes, else its name."""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def source_sha(source: Path | None) -> str | None:
    if source is None:
        return None
    path = source / "MANIFEST.sha256.json"
    return sha256_file(path)[:12] if path.exists() else None


def _matrices(
    medians: pd.DataFrame | None, index: list[int], variables: list[str]
) -> dict[str, np.ndarray]:
    """(len(index), 60) one-minute medians per variable, NaN where absent."""
    out = {v: np.full((len(index), len(SUBGROUP_COLS)), np.nan) for v in variables}
    if medians is None or medians.empty:
        return out
    position_of = {window: position for position, window in enumerate(index)}
    values = medians[SUBGROUP_COLS].to_numpy(dtype=float)
    window_of = medians["window_index"].to_numpy()
    variable_of = medians["variable"].to_numpy()
    for row in range(len(medians)):
        position = position_of.get(int(window_of[row]))
        target = out.get(str(variable_of[row]))
        if position is None or target is None:
            continue
        target[position] = values[row]
    return out


def first_run(index: list[int], length: int) -> int | None:
    """Position where the first run of ``length`` consecutive window indices starts."""
    start = 0
    for p in range(1, len(index) + 1):
        if p == len(index) or index[p] != index[p - 1] + 1:
            if p - start >= length:
                return start
            start = p
    return None


def _consecutive(index: list[int]) -> bool:
    return all(index[p] == index[p - 1] + 1 for p in range(1, len(index)))


def _read_3w(cache: Path) -> tuple[list[str], pd.DataFrame, dict[str, pd.DataFrame]]:
    manifest = json.loads((cache / "MANIFEST.json").read_text("utf-8"))
    windows = pd.read_parquet(cache / "windows.parquet")
    sub = pd.read_parquet(cache / "subgroup_medians.parquet")
    grouped = {str(key): frame for key, frame in sub.groupby("instance", sort=False)}
    return list(manifest["common_variables"]), windows, grouped


def placebo_3w(cache: Path) -> tuple[list[Split], dict]:
    """One label-blind placebo per 3W instance with no fault window.

    The instance needs 4 consecutive windows. Before is the first 2 of the
    first such run, after is the next 2: 120 minute medians each.
    """
    variables, windows, grouped = _read_3w(cache)
    splits: list[Split] = []
    n_normal = 0
    for instance, frame in windows.groupby("instance", sort=True):
        frame = frame.sort_values("window_index", kind="stable")
        if frame["label"].any():
            continue
        n_normal += 1
        index = [int(v) for v in frame["window_index"]]
        start = first_run(index, 2 * PLACEBO_WINDOWS)
        if start is None:
            continue
        chosen = index[start : start + 2 * PLACEBO_WINDOWS]
        matrices = _matrices(grouped.get(str(instance)), chosen, variables)
        splits.append(
            Split(
                design=DESIGN_3W_PLACEBO,
                record=str(instance),
                group=str(frame["well"].iloc[0]),
                segment=0,
                before={v: m[:PLACEBO_WINDOWS].ravel() for v, m in matrices.items()},
                after={v: m[PLACEBO_WINDOWS:].ravel() for v, m in matrices.items()},
            )
        )
    counts = {
        "cache": rel(cache),
        "manifest_sha256": sha256_file(cache / "MANIFEST.json")[:12],
        "n_instances_no_fault_window": n_normal,
        "n_instances_scored": len(splits),
        "n_instances_too_short": n_normal - len(splits),
        "n_wells_scored": len({s.group for s in splits}),
    }
    return splits, counts


def onset_3w(cache: Path) -> tuple[list[Split], dict]:
    """Onset-aligned 3W instances: the 3 windows nearest before the onset against
    the windows after it that the cache holds."""
    variables, windows, grouped = _read_3w(cache)
    splits: list[Split] = []
    offsets_before: set[int] = set()
    offsets_after: set[int] = set()
    n_not_consecutive = 0
    for instance, frame in windows.groupby("instance", sort=True):
        frame = frame.sort_values("onset_offset", kind="stable")
        pre = frame[frame["design_label"] == 0].tail(ONSET_BEFORE_WINDOWS)
        post = frame[frame["design_label"] == 1]
        if pre.empty or post.empty:
            continue
        offsets_before |= {int(v) for v in pre["onset_offset"]}
        offsets_after |= {int(v) for v in post["onset_offset"]}
        index_pre = [int(v) for v in pre["window_index"]]
        index_post = [int(v) for v in post["window_index"]]
        if not (_consecutive(index_pre) and _consecutive(index_post)):
            n_not_consecutive += 1
        medians = grouped.get(str(instance))
        before = _matrices(medians, index_pre, variables)
        after = _matrices(medians, index_post, variables)
        splits.append(
            Split(
                design=DESIGN_3W_ONSET,
                record=str(instance),
                group=str(frame["well"].iloc[0]),
                segment=0,
                before={v: m.ravel() for v, m in before.items()},
                after={v: m.ravel() for v, m in after.items()},
            )
        )
    counts = {
        "cache": rel(cache),
        "manifest_sha256": sha256_file(cache / "MANIFEST.json")[:12],
        "n_instances_scored": len(splits),
        "n_wells_scored": len({s.group for s in splits}),
        "before_onset_offsets": sorted(offsets_before),
        "after_onset_offsets": sorted(offsets_after),
        "n_instances_with_a_period_not_consecutive": n_not_consecutive,
    }
    return splits, counts


@dataclass
class SkabRecord:
    """One SKAB archive directory as a wide frame, 1 s rows as recorded."""

    record: str
    folder: str
    frame: pd.DataFrame
    anomaly: np.ndarray | None

    @property
    def tags(self) -> list[str]:
        return [c for c in self.frame.columns if c != "timestamp"]

    def span(self) -> tuple[int, int] | None:
        """First and last anomaly row, or None when the record has none."""
        if self.anomaly is None:
            return None
        rows = np.flatnonzero(self.anomaly == 1)
        return (int(rows[0]), int(rows[-1])) if rows.size else None


def load_skab(archives: Path) -> tuple[list[SkabRecord], dict]:
    """Every archive directory; values are kept where quality is GOOD."""
    records: list[SkabRecord] = []
    for directory in sorted(p for p in archives.iterdir() if p.is_dir()):
        paths = sorted(p for p in directory.glob("*.parquet") if p.name != "labels.parquet")
        if not paths:
            continue
        wide: pd.DataFrame | None = None
        for path in paths:
            frame = pd.read_parquet(path)
            values = pd.to_numeric(frame["value"], errors="coerce").where(
                frame["quality"] == "GOOD"
            )
            column = pd.DataFrame({"timestamp": frame["timestamp"], path.stem: values})
            wide = column if wide is None else wide.merge(column, on="timestamp", how="outer")
        wide = wide.sort_values("timestamp", kind="stable").reset_index(drop=True)
        anomaly = None
        labels_path = directory / "labels.parquet"
        if labels_path.exists():
            labels = pd.read_parquet(labels_path)[["timestamp", "anomaly"]]
            merged = wide[["timestamp"]].merge(labels, on="timestamp", how="left")
            anomaly = merged["anomaly"].fillna(0).astype(int).to_numpy()
        folder = directory.name.partition("__")[0]
        records.append(SkabRecord(directory.name, folder, wide, anomaly))
    counts = {
        "archives": rel(archives),
        "n_records": len(records),
        "n_labelled": sum(1 for r in records if r.anomaly is not None),
        "n_with_an_anomaly_row": sum(1 for r in records if r.span() is not None),
    }
    return records, counts


def skab_gaps(records: list[SkabRecord]) -> list[dict]:
    """Records with a hole longer than ``SKAB_GAP_S`` between consecutive rows."""
    out = []
    for rec in records:
        diffs = rec.frame["timestamp"].diff().dt.total_seconds().to_numpy()[1:]
        big = diffs[diffs > SKAB_GAP_S]
        if big.size:
            out.append(
                {
                    "record": rec.record,
                    "rows": len(rec.frame),
                    "n_gaps": int(big.size),
                    "longest_gap_s": float(big.max()),
                }
            )
    return out


def _skab_split(
    rec: SkabRecord, design: str, segment: int, before: np.ndarray, after: np.ndarray
) -> Split:
    b, a = rec.frame.loc[before], rec.frame.loc[after]
    return Split(
        design=design,
        record=rec.record,
        group=rec.folder,
        segment=segment,
        before={t: b[t].to_numpy(dtype=float) for t in rec.tags},
        after={t: a[t].to_numpy(dtype=float) for t in rec.tags},
    )


def placebo_skab_free(records: list[SkabRecord]) -> list[Split]:
    """Non-overlapping 10 min before / 10 min after pairs from the first timestamp."""
    splits = []
    for rec in records:
        if rec.anomaly is not None:
            continue
        stamps = rec.frame["timestamp"]
        t0, t_end = stamps.iloc[0], stamps.iloc[-1] + pd.Timedelta(1, unit="s")
        pair = 0
        while t0 + (2 * pair + 2) * SKAB_PAIR <= t_end:
            edges = [t0 + (2 * pair + k) * SKAB_PAIR for k in range(3)]
            before = ((stamps >= edges[0]) & (stamps < edges[1])).to_numpy()
            after = ((stamps >= edges[1]) & (stamps < edges[2])).to_numpy()
            splits.append(_skab_split(rec, DESIGN_SKAB_FREE, pair, before, after))
            pair += 1
    return splits


def placebo_skab_pre(records: list[SkabRecord]) -> list[Split]:
    """The rows before the first anomaly row, first half against second half."""
    splits = []
    for rec in records:
        span = rec.span()
        if span is None:
            continue
        rows = np.arange(len(rec.frame))
        half = span[0] // 2
        before, after = rows < half, (rows >= half) & (rows < span[0])
        splits.append(_skab_split(rec, DESIGN_SKAB_PRE, 0, before, after))
    return splits


def onset_skab(records: list[SkabRecord]) -> list[Split]:
    """Rows before the first anomaly row against the first to the last anomaly row."""
    splits = []
    for rec in records:
        span = rec.span()
        if span is None:
            continue
        rows = np.arange(len(rec.frame))
        before, after = rows < span[0], (rows >= span[0]) & (rows <= span[1])
        splits.append(_skab_split(rec, DESIGN_SKAB_ONSET, 0, before, after))
    return splits


def score_split(split: Split) -> list[dict]:
    """Rows for every tag of the split as the target.

    The covariates of a target are every other tag that passes R0 on this
    split, declared before any interval is computed.
    """
    tags = sorted(split.before)
    usable_tags = [t for t in tags if sh.usable(split.before[t], split.after[t]) == ""]
    head = {
        "design": split.design,
        "record": split.record,
        "group": split.group,
        "segment": split.segment,
    }
    rows = []
    for tag in tags:
        covariates = [t for t in usable_tags if t != tag]
        for row in sh.score_target(split.before, split.after, tag, covariates):
            rows.append({**head, "tag": tag, **row})
    return rows


ROW_COLUMNS = [
    "design",
    "record",
    "group",
    "segment",
    "tag",
    "quantity",
    "method",
    "adjustment",
    "n_before",
    "n_after",
    "n_covariates",
    "estimate",
    "lo",
    "hi",
    "width",
    "p_value",
    "clears",
    "refused",
    "reason",
    "trend_t",
    "lag1",
    "lag1_after",
    "hac_bandwidth",
    "before_trend",
    "covariate_outside",
    "covariate_shifted",
    "too_persistent",
    "variance_reduction",
    "variance_ratio_after",
]
ROW_FLOATS = [
    "estimate",
    "lo",
    "hi",
    "width",
    "p_value",
    "trend_t",
    "lag1",
    "lag1_after",
    "hac_bandwidth",
    "variance_reduction",
    "variance_ratio_after",
]
ROW_FLAGS = ["clears", "before_trend", "covariate_outside", "covariate_shifted", "too_persistent"]
ROW_ORDER = ["design", "record", "segment", "tag", "quantity", "method", "adjustment"]


def rows_frame(rows: list[dict]) -> pd.DataFrame:
    """Rows as written: level values in before-period SDs of the target, rounded."""
    frame = pd.DataFrame(rows)
    level = frame["quantity"] == sh.LEVEL
    for column in ("estimate", "lo", "hi"):
        frame.loc[level, column] = frame.loc[level, column] / frame.loc[level, "scale"]
    frame["width"] = frame["hi"] - frame["lo"]
    for column in ROW_FLAGS:
        frame[column] = frame[column].map({True: 1, False: 0}).astype("Int8")
    frame["refused"] = frame["refused"].astype(int)
    frame = round_frame(frame, ROW_FLOATS)
    frame = frame.sort_values(ROW_ORDER, kind="stable").reset_index(drop=True)
    return frame[ROW_COLUMNS]


def read_rows(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False, na_values=[""])


def _flag(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame[column].eq(1)


def refused_mask(frame: pd.DataFrame, refusal_set: str) -> pd.Series:
    mask = frame["refused"].eq(1)
    for rule in sh.REFUSAL_SETS[refusal_set]:
        mask |= _flag(frame, rule)
    return mask


def raw_counterpart(refusal_set: str) -> str:
    """The raw-arm set an adjusted set is compared with: the same set without R3."""
    return {
        "base": "base",
        "base+R1": "base+R1",
        "base+R3": "base",
        "base+R1+R3": "base+R1",
        "base+R4": "base+R4",
        "base+R1+R4": "base+R1+R4",
    }[refusal_set]


def _share(mask: pd.Series) -> float | None:
    return float(mask.mean()) if len(mask) else None


def summarise(frame: pd.DataFrame) -> list[dict]:
    """One row per (design, quantity, method, adjustment, refusal set)."""
    out = []
    keys = ["design", "quantity", "method", "adjustment"]
    pair_keys = ["record", "segment", "tag"]
    for (design, quantity, method, adjustment), group in frame.groupby(keys, sort=True):
        sets = sh.RAW_REFUSAL_SETS if adjustment == sh.RAW else tuple(sh.REFUSAL_SETS)
        raw = frame[
            (frame["design"] == design)
            & (frame["quantity"] == quantity)
            & (frame["method"] == method)
            & (frame["adjustment"] == sh.RAW)
        ]
        hard = group.loc[group["refused"].eq(1), "reason"].value_counts().sort_index()
        for refusal_set in sets:
            refused = refused_mask(group, refusal_set)
            answered = group[~refused]
            clears = answered["clears"].eq(1)
            per_group = clears.groupby(answered["group"]).mean()
            per_record = answered.groupby(["record", "segment"])["p_value"].agg(["min", "size"])
            record_clears = per_record["min"] < sh.ALPHA / per_record["size"]
            ratio = None
            n_pairs = 0
            if adjustment == sh.ADJUSTED:
                raw_answered = raw[~refused_mask(raw, raw_counterpart(refusal_set))]
                pairs = answered[[*pair_keys, "width"]].merge(
                    raw_answered[[*pair_keys, "width"]], on=pair_keys, suffixes=("", "_raw")
                )
                n_pairs = len(pairs)
                if n_pairs:
                    ratio = float((pairs["width"] / pairs["width_raw"]).median())
            out.append(
                {
                    "design": design,
                    "quantity": quantity,
                    "method": method,
                    "adjustment": adjustment,
                    "refusal_set": refusal_set,
                    "n_rows": len(group),
                    "n_records": int(group.groupby(["record", "segment"]).ngroups),
                    "n_groups": int(group["group"].nunique()),
                    "n_answered": len(answered),
                    "answered_share": _share(~refused),
                    "hard_refusals": ";".join(f"{k}:{v}" for k, v in hard.items()),
                    "rate_before_trend": _share(_flag(group, sh.BEFORE_TREND)),
                    "rate_too_persistent": _share(_flag(group, sh.TOO_PERSISTENT)),
                    "rate_covariate_outside": (
                        _share(_flag(group, sh.COVARIATE_OUTSIDE))
                        if adjustment == sh.ADJUSTED
                        else None
                    ),
                    "rate_covariate_shifted": (
                        _share(_flag(group, sh.COVARIATE_SHIFTED))
                        if adjustment == sh.ADJUSTED
                        else None
                    ),
                    "clear_rate_pooled": _share(clears),
                    "clear_rate_group_avg": float(per_group.mean()) if len(per_group) else None,
                    "n_groups_answered": len(per_group),
                    "record_clear_rate": _share(record_clears),
                    "n_records_answered": len(per_record),
                    "median_width": (
                        float(answered["width"].median()) if len(answered) else None
                    ),
                    "median_variance_reduction": (
                        float(answered["variance_reduction"].median())
                        if adjustment == sh.ADJUSTED and len(answered)
                        else None
                    ),
                    "median_variance_ratio_after": (
                        float(answered["variance_ratio_after"].median())
                        if adjustment == sh.ADJUSTED and len(answered)
                        else None
                    ),
                    "median_width_ratio": ratio,
                    "n_width_pairs": n_pairs if adjustment == sh.ADJUSTED else None,
                    "median_lag1": (
                        float(answered["lag1"].median()) if len(answered) else None
                    ),
                    "share_bandwidth_capped": _share(
                        answered["hac_bandwidth"] >= answered["n_before"] - 1
                    ),
                }
            )
    return out


SUMMARY_FLOATS = [
    "answered_share",
    "rate_before_trend",
    "rate_too_persistent",
    "rate_covariate_outside",
    "rate_covariate_shifted",
    "clear_rate_pooled",
    "clear_rate_group_avg",
    "record_clear_rate",
    "median_width",
    "median_variance_reduction",
    "median_variance_ratio_after",
    "median_width_ratio",
    "median_lag1",
    "share_bandwidth_capped",
]


def write_summary(out: Path) -> int:
    """summary.csv from the placebo and known-change CSVs on disk."""
    frames = [read_rows(out / name) for name in (PLACEBO_CSV, KNOWN_CSV) if (out / name).exists()]
    if not frames:
        return 0
    summary = pd.DataFrame(summarise(pd.concat(frames, ignore_index=True)))
    summary["n_width_pairs"] = summary["n_width_pairs"].astype("Int64")
    summary = round_frame(summary, SUMMARY_FLOATS)
    write_csv(summary, out / SUMMARY_CSV)
    return len(summary)


def _design_counts(splits: list[Split]) -> dict:
    out: dict[str, dict] = {}
    for design in sorted({s.design for s in splits}):
        chosen = [s for s in splits if s.design == design]
        out[design] = {
            "n_splits": len(chosen),
            "n_records": len({s.record for s in chosen}),
            "n_groups": len({s.group for s in chosen}),
        }
    return out


def _score_bed(splits: list[Split], path: Path) -> int:
    rows = [row for split in splits for row in score_split(split)]
    frame = rows_frame(rows)
    write_csv(frame, path)
    return len(frame)


def run_placebo(windows: Path, archives: Path, sources: dict, out: Path) -> dict:
    started = time.perf_counter()
    splits_3w, counts_3w = placebo_3w(windows)
    records, counts_skab = load_skab(archives)
    splits = splits_3w + placebo_skab_free(records) + placebo_skab_pre(records)
    n_rows = _score_bed(splits, out / PLACEBO_CSV)
    return {
        "code": code_version(),
        "dataset_manifest_sha256": sources,
        "3w_cache": counts_3w,
        "skab": counts_skab,
        "designs": _design_counts(splits),
        "n_rows": n_rows,
        "wall_seconds": round(time.perf_counter() - started, 1),
    }


def run_known(aligned: Path, archives: Path, sources: dict, out: Path) -> dict:
    started = time.perf_counter()
    splits_3w, counts_3w = onset_3w(aligned)
    records, counts_skab = load_skab(archives)
    splits = splits_3w + onset_skab(records)
    n_rows = _score_bed(splits, out / KNOWN_CSV)
    return {
        "code": code_version(),
        "dataset_manifest_sha256": sources,
        "3w_cache": counts_3w,
        "skab": counts_skab,
        "skab_gaps_over_60_s": skab_gaps(records),
        "designs": _design_counts(splits),
        "n_rows": n_rows,
        "wall_seconds": round(time.perf_counter() - started, 1),
    }


# ---------------------------------------------------------------- main


def parameters() -> dict:
    return {
        "z95": sh.Z95,
        "alpha": sh.ALPHA,
        "min_samples": sh.MIN_SAMPLES,
        "trend_t": sh.TREND_T,
        "bootstrap_replicates": sh.BOOT_REPLICATES,
        "bootstrap_seed": sh.BOOT_SEED,
        "bootstrap_min_block": sh.BOOT_MIN_BLOCK,
        "andrews_bartlett_constant": sh.ANDREWS_BARTLETT,
        "covariate_outside_percentiles": [sh.OUTSIDE_LO, sh.OUTSIDE_HI],
        "refusal_sets": {name: list(rules) for name, rules in sh.REFUSAL_SETS.items()},
    }


def update_run_json(out: Path, sections: dict) -> dict:
    """Merge the beds this call ran into ``run.json``; beds not run keep their entry."""
    path = out / "run.json"
    info = json.loads(path.read_text("utf-8")) if path.exists() else {}
    info["study"] = "shift_intervals"
    info["parameters"] = parameters()
    beds = info.get("beds", {})
    beds.update(sections)
    info["beds"] = beds
    text = json.dumps(info, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")
    return info


def run(
    out: Path,
    beds: tuple[str, ...] = BEDS,
    replicates: int = DEFAULT_REPLICATES,
    *,
    windows: Path = ROOT / DEFAULT_WINDOWS,
    aligned: Path = ROOT / DEFAULT_ALIGNED,
    archives: Path = ROOT / DEFAULT_ARCHIVES,
    source_3w: Path | None = ROOT / DEFAULT_3W_SOURCE,
    source_skab: Path | None = ROOT / DEFAULT_SKAB_SOURCE,
) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    sources = {"3w": source_sha(source_3w), "skab": source_sha(source_skab)}
    sections = {}
    if BED_SYNTHETIC in beds:
        sections[BED_SYNTHETIC] = run_synthetic(out, replicates)
    if BED_PLACEBO in beds:
        sections[BED_PLACEBO] = run_placebo(windows, archives, sources, out)
    if BED_KNOWN in beds:
        sections[BED_KNOWN] = run_known(aligned, archives, sources, out)
    if BED_PLACEBO in beds or BED_KNOWN in beds:
        write_summary(out)
    return update_run_json(out, sections)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--windows", default=DEFAULT_WINDOWS)
    parser.add_argument("--aligned", default=DEFAULT_ALIGNED)
    parser.add_argument("--archives", default=DEFAULT_ARCHIVES)
    parser.add_argument("--source", default=DEFAULT_3W_SOURCE)
    parser.add_argument("--skab-source", default=DEFAULT_SKAB_SOURCE)
    parser.add_argument("--out", default=str(HERE / "results"))
    parser.add_argument("--beds", nargs="+", choices=BEDS, default=list(BEDS))
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    args = parser.parse_args(argv)
    info = run(
        Path(args.out),
        tuple(args.beds),
        args.replicates,
        windows=Path(args.windows),
        aligned=Path(args.aligned),
        archives=Path(args.archives),
        source_3w=Path(args.source),
        source_skab=Path(args.skab_source),
    )
    for bed in args.beds:
        print(f"{bed}: {info['beds'][bed]['wall_seconds']} s, {info['beds'][bed]['n_rows']} rows")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

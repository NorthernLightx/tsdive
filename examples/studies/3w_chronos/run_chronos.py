"""Chronos-Bolt zero-shot forecast surprise on the onset-aligned 3W windows.

The study puts a time-series foundation model, Chronos-Bolt (small), on
the same 288 onset-aligned windows the detector study scored, under the
same protocol, as a benchmark subject. Nothing is fitted on 3W. For each
window and each of the six common variables the model forecasts the
window hour from the four hours before it, at one-minute resolution, and
the surprise is the mean absolute forecast error in units of the
forecast band. The window score is the mean surprise over its scored
variables. The clock control from the detector study is scored beside
it, and ``tsdive.eval`` carries the protocol: pooled ranking metrics,
the worst-baseline alarm rule and the false-alarm floor.

A variable whose context holds fewer than 60 minutes with data, or
whose target hour holds fewer than 30, is a refusal with a cause string.
A window with fewer than four scored variables is a refusal. An
instance with a refused baseline window has no threshold and is a
refusal. Every refusal is a row.

Outputs (``--out``, default ``results/`` beside this file):

- ``window_scores.csv``: one row per window and variable.
- ``windows.csv``: one row per window with its score or refusal.
- ``per_instance.csv``: one row per instance and tool, the detector
  study's ``aligned_per_instance.csv`` columns.
- ``per_fold.csv``: pooled ranking per fold and tool.
- ``summary.csv``: one row per tool, the detector study's
  ``aligned_summary.csv`` columns plus the false-alarm floor.
- ``run.json``: provenance, parameters and refusal counts.

Run: ``uv sync --extra tsfm`` then
``uv run python examples/studies/3w_chronos/run_chronos.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))

import tsdive  # noqa: E402
from tsdive.eval import (  # noqa: E402
    clock_control,
    far_floor,
    fires,
    over_floor,
    ranking_metrics,
    worst_baseline_threshold,
)

MODEL_ID = "amazon/chronos-bolt-small"
TOOL = "chronos-bolt-small surprise"
CLOCK = "clock"
CLOCK_SAME = "clock (same instances)"
DESIGN = "onset-aligned"
VARIANT = "all"

MINUTE_S = 60.0
CONTEXT_MINUTES = 240
TARGET_MINUTES = 60
MIN_CONTEXT_MINUTES = 60
MIN_TARGET_MINUTES = 30
MIN_SCORED_VARIABLES = 4
QUANTILE_LEVELS = (0.1, 0.5, 0.9)
# The forecast band is floored at this share of the context spread so a
# band that collapses to zero width cannot make one minute's error
# arbitrarily large.
WIDTH_FLOOR_SHARE = 0.05
BASELINE_WINDOWS = 3
GOOD = "GOOD_ASSUMED"
SEED = 0
THREADS = 4

# ``Forecaster``: contexts of shape (n, CONTEXT_MINUTES) in, quantiles
# of shape (n, TARGET_MINUTES, len(QUANTILE_LEVELS)) out, in the order of
# QUANTILE_LEVELS. The model call sits behind this so tests run a stub.
Forecaster = Callable[[np.ndarray], np.ndarray]

DEFAULT_WINDOWS = "data/3w_windows_aligned"
DEFAULT_ARCHIVES = "data/3w_archives"
DEFAULT_SOURCE = "data/3w"
DEFAULT_FOLDS = "examples/studies/3w_detectors/results/aligned_per_instance.csv"


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


def repo_relative(path: str | Path) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def r4(value) -> float | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return round(float(value), 4)


# ------------------------------------------------------- minute aggregation


def minute_means(frame: pd.DataFrame, start: pd.Timestamp, minutes: int) -> np.ndarray:
    """Mean of the ``GOOD`` rows in each whole minute from ``start``.

    Minute ``i`` covers ``[start + i min, start + (i + 1) min)``. A minute
    with no ``GOOD`` row is ``NaN``. Rows outside the span are ignored.
    """
    out = np.full(minutes, np.nan)
    if not len(frame):
        return out
    good = frame["quality"].to_numpy() == GOOD
    stamps = pd.to_datetime(frame["timestamp"], utc=True)
    offset = (stamps - start).dt.total_seconds().to_numpy()
    values = frame["value"].to_numpy(dtype=float)
    keep = good & (offset >= 0) & (offset < minutes * MINUTE_S) & np.isfinite(values)
    if not keep.any():
        return out
    bucket = np.floor(offset[keep] / MINUTE_S).astype(int)
    sums = np.bincount(bucket, weights=values[keep], minlength=minutes)
    counts = np.bincount(bucket, minlength=minutes)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = sums / counts
    out[counts > 0] = means[counts > 0]
    return out


def forward_fill(context: np.ndarray) -> tuple[np.ndarray, int]:
    """Fill a missing context minute with the last observed one.

    Minutes before the first observation take the first observed value;
    they are counted in the returned fill count as well. Raises
    ``ValueError`` on a context with no observation at all.
    """
    if not np.isfinite(context).any():
        raise ValueError("forward_fill needs at least one observed minute")
    missing = ~np.isfinite(context)
    idx = np.where(missing, -1, np.arange(len(context)))
    idx = np.maximum.accumulate(idx)
    filled = context[np.where(idx < 0, int(np.argmax(~missing)), idx)]
    return filled, int(missing.sum())


# A minute mean of identical float64 values can differ from its neighbours
# by a few units in the last place, so a constant context is one whose
# spread is under this share of its magnitude, not one whose std is zero.
CONSTANT_RELATIVE_SPREAD = 1e-9


def is_constant(context: np.ndarray) -> bool:
    """``True`` when the observed context minutes hold one value up to float noise."""
    observed = context[np.isfinite(context)]
    if not len(observed):
        return True
    spread = float(observed.max() - observed.min())
    return spread <= CONSTANT_RELATIVE_SPREAD * max(1.0, float(np.abs(observed).max()))


@dataclass
class VariableSeries:
    """One (window, variable) pair, ready for the forecaster or refused."""

    window_key: int
    variable: str
    context: np.ndarray
    target: np.ndarray
    n_context: int
    n_target: int
    refusal: str | None
    n_context_filled: int = 0


def prepare_variable(
    frame: pd.DataFrame, window_start: pd.Timestamp, window_key: int, variable: str
) -> VariableSeries:
    """Aggregate one archive to minute means and apply the two thresholds."""
    context_start = window_start - pd.Timedelta(CONTEXT_MINUTES, "min")
    context = minute_means(frame, context_start, CONTEXT_MINUTES)
    target = minute_means(frame, window_start, TARGET_MINUTES)
    n_context = int(np.isfinite(context).sum())
    n_target = int(np.isfinite(target).sum())
    refusal = None
    if n_context < MIN_CONTEXT_MINUTES:
        refusal = (
            f"context: {n_context} of {CONTEXT_MINUTES} minutes with data, "
            f"fewer than {MIN_CONTEXT_MINUTES}"
        )
    elif n_target < MIN_TARGET_MINUTES:
        refusal = (
            f"target: {n_target} of {TARGET_MINUTES} minutes with data, "
            f"fewer than {MIN_TARGET_MINUTES}"
        )
    elif is_constant(context):
        refusal = "context is constant: the surprise has no scale"
    series = VariableSeries(window_key, variable, context, target, n_context, n_target, refusal)
    if refusal is None:
        series.context, series.n_context_filled = forward_fill(context)
    return series


# ------------------------------------------------------------- the score


def surprise(
    target: np.ndarray, quantiles: np.ndarray, context: np.ndarray
) -> tuple[float, int]:
    """Mean over observed target minutes of ``|actual - q50| / band``.

    ``band = max(q90 - q10, WIDTH_FLOOR_SHARE * std(context))`` per minute.
    Returns the mean and the number of target minutes skipped for want of
    data. ``quantiles`` is ``(TARGET_MINUTES, 3)`` in QUANTILE_LEVELS order.
    """
    q10, q50, q90 = quantiles[:, 0], quantiles[:, 1], quantiles[:, 2]
    floor = WIDTH_FLOOR_SHARE * float(np.std(context))
    band = np.maximum(q90 - q10, floor)
    observed = np.isfinite(target)
    ratio = np.abs(target[observed] - q50[observed]) / band[observed]
    return float(ratio.mean()), int((~observed).sum())


def chronos_forecaster(model_id: str = MODEL_ID, batch: int = 128) -> tuple[Forecaster, dict]:
    """Load Chronos-Bolt on the CPU and return (forecaster, provenance).

    ``chronos`` and ``torch`` are imported here and nowhere else, so the
    tests and the library never need them.
    """
    import torch
    from chronos import BaseChronosPipeline
    from huggingface_hub import snapshot_download

    torch.manual_seed(SEED)
    torch.set_num_threads(THREADS)
    pipeline = BaseChronosPipeline.from_pretrained(
        model_id, device_map="cpu", torch_dtype=torch.float32
    )
    snapshot = Path(snapshot_download(model_id, local_files_only=True))
    provenance = {
        "model_id": model_id,
        "model_revision": snapshot.name,
        "pipeline_class": type(pipeline).__name__,
        "chronos_forecasting_version": importlib.metadata.version("chronos-forecasting"),
        "torch_version": torch.__version__,
        "device": "cpu",
        "dtype": "float32",
        "seed": SEED,
        "threads": THREADS,
        "batch": batch,
    }

    def forecast(contexts: np.ndarray) -> np.ndarray:
        out = []
        for lo in range(0, len(contexts), batch):
            chunk = torch.tensor(contexts[lo : lo + batch], dtype=torch.float32)
            with torch.no_grad():
                quantiles, _ = pipeline.predict_quantiles(
                    chunk,
                    prediction_length=TARGET_MINUTES,
                    quantile_levels=list(QUANTILE_LEVELS),
                )
            out.append(quantiles.to(torch.float32).cpu().numpy())
        return np.concatenate(out, axis=0)

    return forecast, provenance


# ------------------------------------------------------------- the study


@dataclass
class StudyResult:
    window_scores: pd.DataFrame
    windows: pd.DataFrame
    per_instance: pd.DataFrame
    per_fold: pd.DataFrame
    summary: pd.DataFrame
    counts: dict = field(default_factory=dict)


def load_archive(archives: Path, instance: str, variable: str) -> pd.DataFrame:
    path = archives / instance / f"{variable}.parquet"
    if not path.exists():
        return pd.DataFrame({"timestamp": [], "value": [], "quality": []})
    return pd.read_parquet(path, columns=["timestamp", "value", "quality"])


def score_windows(
    windows: pd.DataFrame,
    variables: list[str],
    archives: Path,
    forecaster: Forecaster,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score every (window, variable) and every window. Returns both tables.

    ``windows`` carries ``window_key``, ``instance``, ``window_start`` and
    the layout columns. Archives are read once per (instance, variable).
    """
    series: list[VariableSeries] = []
    for instance, group in windows.groupby("instance", sort=True):
        for variable in variables:
            frame = load_archive(archives, str(instance), variable)
            for _, row in group.iterrows():
                start = pd.Timestamp(row["window_start"])
                if start.tzinfo is None:
                    start = start.tz_localize("UTC")
                series.append(
                    prepare_variable(frame, start, int(row["window_key"]), variable)
                )
    scored = [s for s in series if s.refusal is None]
    if scored:
        contexts = np.stack([s.context for s in scored]).astype(np.float32)
        quantiles = np.asarray(forecaster(contexts), dtype=float)
        expected = (len(scored), TARGET_MINUTES, len(QUANTILE_LEVELS))
        if quantiles.shape != expected:
            raise ValueError(f"forecaster returned {quantiles.shape}, expected {expected}")
    results: dict[tuple[int, str], tuple[float, int]] = {}
    for i, s in enumerate(scored):
        results[(s.window_key, s.variable)] = surprise(s.target, quantiles[i], s.context)

    layout = windows.set_index("window_key")
    rows = []
    for s in series:
        w = layout.loc[s.window_key]
        value, skipped = results.get((s.window_key, s.variable), (None, None))
        rows.append(
            {
                "instance": w["instance"],
                "well": w["well"],
                "window_key": s.window_key,
                "window_index": int(w["window_index"]),
                "onset_offset": int(w["onset_offset"]),
                "role": w["role"],
                "variable": s.variable,
                "n_context_minutes": s.n_context,
                "n_context_filled": s.n_context_filled if s.refusal is None else None,
                "n_target_minutes": s.n_target,
                "n_target_skipped": skipped,
                "surprise": r4(value),
                "refusal": s.refusal,
            }
        )
    window_scores = pd.DataFrame(rows).sort_values(["instance", "window_index", "variable"])
    window_scores = window_scores.reset_index(drop=True)

    wrows = []
    for key, part in window_scores.groupby("window_key", sort=True):
        w = layout.loc[key]
        ok = part[part["surprise"].notna()]
        refusal = None
        if len(ok) < MIN_SCORED_VARIABLES:
            causes = sorted(
                f"{v}: {c}" for v, c in zip(part["variable"], part["refusal"], strict=True) if c
            )
            refusal = (
                f"{len(ok)} of {len(part)} variables scored, fewer than "
                f"{MIN_SCORED_VARIABLES}; " + "; ".join(causes)
            )
        wrows.append(
            {
                "instance": w["instance"],
                "well": w["well"],
                "window_key": int(key),
                "window_index": int(w["window_index"]),
                "onset_offset": int(w["onset_offset"]),
                "role": w["role"],
                "n_variables_scored": len(ok),
                "score": r4(ok["surprise"].mean()) if refusal is None else None,
                "refusal": refusal,
            }
        )
    window_table = pd.DataFrame(wrows).sort_values(["instance", "window_index"])
    return window_scores, window_table.reset_index(drop=True)


def per_instance_table(
    windows: pd.DataFrame,
    scores: dict[int, float],
    refusals: dict[int, str],
    tool: str,
    folds: pd.DataFrame,
) -> pd.DataFrame:
    """One row per instance, the detector study's ``aligned_per_instance`` columns.

    The threshold is the worst (largest) score over the instance's
    baseline windows. An instance with fewer than ``BASELINE_WINDOWS``
    scored baseline windows has no threshold and carries the cause.
    """
    fold_of = folds.drop_duplicates("instance").set_index("instance")
    rows = []
    for instance, group in windows.groupby("instance", sort=True):
        by_role = {
            str(role): [int(k) for k in sub.sort_values("window_index")["window_key"]]
            for role, sub in group.groupby("role")
        }
        first = group.iloc[0]
        base_keys = by_role.get("baseline", [])
        base = [scores[k] for k in base_keys if k in scores]
        reasons = [refusals.get(k, "no score and no refusal") for k in base_keys if k not in scores]
        info = fold_of.loc[str(instance)] if str(instance) in fold_of.index else None
        row = {
            "tool": tool,
            "instance": str(instance),
            "well": str(first["well"]),
            "folder_label": str(first["folder_label"]),
            "onset_kind": (str(info["onset_kind"]) if info is not None else None),
            "fold": (int(info["fold"]) if info is not None else None),
            "n_baseline_scored": len(base),
            "baseline_max": (
                r4(worst_baseline_threshold(base)) if len(base) == BASELINE_WINDOWS else None
            ),
        }
        if len(base) < BASELINE_WINDOWS:
            reasons.append(
                f"baseline: {len(base)} of {BASELINE_WINDOWS} windows scored, no threshold"
            )
        for role in ("pre", "post0", "post1"):
            keys = by_role.get(role, [])
            key = keys[0] if keys else None
            row[role] = r4(scores.get(key)) if key is not None else None
            if key is not None and key not in scores:
                reasons.append(f"{role}: {refusals.get(key, 'no score and no refusal')}")
        row["refusal"] = "; ".join(sorted(set(reasons))) or None
        rows.append(row)
    return pd.DataFrame(rows)


def _ranking(usable: pd.DataFrame) -> dict:
    y, s = [], []
    for _, row in usable.iterrows():
        if pd.notna(row["pre"]):
            y.append(0)
            s.append(float(row["pre"]))
        for role in ("post0", "post1"):
            if pd.notna(row[role]):
                y.append(1)
                s.append(float(row[role]))
    if not y:
        return {
            "roc_auc": None,
            "pr_auc": None,
            "n_pos": 0,
            "n_neg": 0,
            "refusal": "no window scored",
        }
    return ranking_metrics(np.array(y, dtype=int), np.array(s, dtype=float)).to_dict()


def instance_metrics(frame: pd.DataFrame) -> dict:
    """The detector study's ``aligned_metrics`` row, plus the false-alarm floor."""
    usable = frame[frame["baseline_max"].notna()]
    ranking = _ranking(usable)

    def paired(role: str) -> tuple[int, float | None]:
        sub = usable[usable["pre"].notna() & usable[role].notna()]
        if not len(sub):
            return 0, None
        wins = int((sub[role].to_numpy(dtype=float) > sub["pre"].to_numpy(dtype=float)).sum())
        return len(sub), round(wins / len(sub), 4)

    def fired(role: str) -> tuple[int, float | None]:
        sub = usable[usable[role].notna()]
        if not len(sub):
            return 0, None
        hits = sum(
            fires(v, t)
            for v, t in zip(
                sub[role].to_numpy(dtype=float),
                sub["baseline_max"].to_numpy(dtype=float),
                strict=True,
            )
        )
        return len(sub), round(hits / len(sub), 4)

    delays: list[int] = []
    n_never = 0
    for _, row in usable.iterrows():
        limit = float(row["baseline_max"])
        delay = None
        for offset, role in enumerate(("post0", "post1")):
            if pd.notna(row[role]) and fires(float(row[role]), limit):
                delay = offset
                break
        if delay is None:
            n_never += 1
        else:
            delays.append(delay)

    n_paired, hit0 = paired("post0")
    _, hit1 = paired("post1")
    n_pre, far = fired("pre")
    n_post0, det0 = fired("post0")
    n_post1, det1 = fired("post1")
    per_fold = [
        _ranking(part)["roc_auc"] for _, part in usable.groupby("fold", sort=True)
    ]
    per_fold = [v for v in per_fold if v is not None]
    return {
        "n_instances": len(frame),
        "n_instances_with_threshold": len(usable),
        "n_instances_refused": int(len(frame) - len(usable)),
        "roc_auc": ranking["roc_auc"],
        "pr_auc": ranking["pr_auc"],
        "n_pre_windows": ranking["n_neg"],
        "n_post_windows": ranking["n_pos"],
        "ranking_refusal": ranking["refusal"],
        "n_paired": n_paired,
        "paired_hit_post0": hit0,
        "paired_hit_post1": hit1,
        "n_pre_at_threshold": n_pre,
        "false_alarm_rate_pre": far,
        "n_post0_at_threshold": n_post0,
        "detection_rate_post0": det0,
        "n_post1_at_threshold": n_post1,
        "detection_rate_post1": det1,
        "n_detected": len(delays),
        "n_never_detected": n_never,
        "median_delay_windows": (float(np.median(delays)) if delays else None),
        "n_folds_scored": len(per_fold),
        "auc_fold_min": min(per_fold) if per_fold else None,
        "auc_fold_max": max(per_fold) if per_fold else None,
        "far_floor": r4(far_floor(BASELINE_WINDOWS)),
        "over_floor": (r4(over_floor(far, BASELINE_WINDOWS)) if far is not None else None),
    }


def per_fold_table(per_instance: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tool, frame in per_instance.groupby("tool", sort=True):
        usable = frame[frame["baseline_max"].notna()]
        for fold, part in usable.groupby("fold", sort=True):
            m = instance_metrics(part)
            rows.append(
                {
                    "tool": tool,
                    "fold": int(fold),
                    "n_instances": len(part),
                    "n_pre_windows": m["n_pre_windows"],
                    "n_post_windows": m["n_post_windows"],
                    "roc_auc": m["roc_auc"],
                    "paired_hit_post0": m["paired_hit_post0"],
                    "paired_hit_post1": m["paired_hit_post1"],
                    "false_alarm_rate_pre": m["false_alarm_rate_pre"],
                    "detection_rate_post1": m["detection_rate_post1"],
                    "ranking_refusal": m["ranking_refusal"],
                }
            )
    return pd.DataFrame(rows)


def run_study(
    windows: pd.DataFrame,
    variables: list[str],
    archives: Path,
    forecaster: Forecaster,
    folds: pd.DataFrame,
) -> StudyResult:
    """Score the windows, then build the instance, fold and summary tables."""
    windows = windows.reset_index(drop=True).copy()
    windows["window_key"] = np.arange(len(windows))
    window_scores, window_table = score_windows(windows, variables, archives, forecaster)

    scored = window_table[window_table["score"].notna()]
    scores = dict(zip(scored["window_key"].astype(int), scored["score"].astype(float), strict=True))
    refused = window_table[window_table["refusal"].notna()]
    refusals = dict(zip(refused["window_key"].astype(int), refused["refusal"], strict=True))

    index = windows.set_index("window_key")["window_index"]
    clock = clock_control(index)
    tables = [
        per_instance_table(windows, scores, refusals, TOOL, folds),
        per_instance_table(windows, clock, {}, CLOCK, folds),
    ]
    chronos_instances = set(tables[0].loc[tables[0]["baseline_max"].notna(), "instance"])
    if len(chronos_instances) < windows["instance"].nunique():
        same = windows[windows["instance"].isin(chronos_instances)]
        tables.append(per_instance_table(same, clock, {}, CLOCK_SAME, folds))
    per_instance = pd.concat(tables, ignore_index=True)

    summary_rows = []
    for tool in per_instance["tool"].drop_duplicates():
        frame = per_instance[per_instance["tool"] == tool]
        row = {"tool": tool, "design": DESIGN, "variant": VARIANT}
        row.update(instance_metrics(frame))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    counts = {
        "n_windows": len(window_table),
        "n_windows_scored": int(window_table["score"].notna().sum()),
        "n_windows_refused": int(window_table["refusal"].notna().sum()),
        "n_variable_rows": len(window_scores),
        "n_variable_rows_scored": int(window_scores["surprise"].notna().sum()),
        "n_variable_rows_refused": int(window_scores["refusal"].notna().sum()),
        "variable_refusals_by_cause": dict(
            sorted(Counter(c.split(":")[0] for c in window_scores["refusal"].dropna()).items())
        ),
        "variable_refusals_by_variable": {
            str(k): int(v)
            for k, v in window_scores.loc[window_scores["refusal"].notna(), "variable"]
            .value_counts()
            .sort_index()
            .items()
        },
        "n_context_minutes_filled": int(window_scores["n_context_filled"].fillna(0).sum()),
        "n_target_minutes_skipped": int(window_scores["n_target_skipped"].fillna(0).sum()),
        "n_instances": int(windows["instance"].nunique()),
        "n_instances_scored": len(chronos_instances),
        "n_instances_refused": int(windows["instance"].nunique() - len(chronos_instances)),
    }
    return StudyResult(
        window_scores, window_table, per_instance, per_fold_table(per_instance), summary, counts
    )


INT_COLUMNS = frozenset(
    {
        "fold",
        "window_key",
        "window_index",
        "onset_offset",
        "n_context_minutes",
        "n_context_filled",
        "n_target_minutes",
        "n_target_skipped",
        "n_variables_scored",
        "n_baseline_scored",
    }
)


def write_result(result: StudyResult, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in (
        ("window_scores.csv", result.window_scores),
        ("windows.csv", result.windows),
        ("per_instance.csv", result.per_instance),
        ("per_fold.csv", result.per_fold),
        ("summary.csv", result.summary),
    ):
        frame = frame.copy()
        for col in frame.columns:
            if col in INT_COLUMNS:
                frame[col] = frame[col].astype("Int64")
        frame.to_csv(out_dir / name, index=False, lineterminator="\n")


# --------------------------------------------------------------- the driver


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--windows", default=DEFAULT_WINDOWS)
    parser.add_argument("--archives", default=DEFAULT_ARCHIVES)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--folds", default=DEFAULT_FOLDS)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--out", default=str(HERE / "results"))
    args = parser.parse_args(argv)

    started = time.perf_counter()
    cache = Path(args.windows)
    manifest = json.loads((cache / "MANIFEST.json").read_text("utf-8"))
    windows = pd.read_parquet(cache / "windows.parquet")
    variables = list(manifest["common_variables"])
    folds = pd.read_csv(args.folds, dtype={"folder_label": str})
    forecaster, model_provenance = chronos_forecaster(args.model)
    loaded = time.perf_counter()
    result = run_study(windows, variables, Path(args.archives), forecaster, folds)
    elapsed = time.perf_counter() - started

    out_dir = Path(args.out)
    write_result(result, out_dir)
    source_manifest = Path(args.source) / "MANIFEST.sha256.json"
    run = {
        "design": DESIGN,
        "tool": TOOL,
        "model": model_provenance,
        "fitted_on_3w": False,
        "dataset_manifest_sha256": (
            sha256_file(source_manifest) if source_manifest.exists() else None
        ),
        "aligned_windows_manifest_sha256": sha256_file(cache / "MANIFEST.json"),
        "aligned_windows": repo_relative(cache / "windows.parquet"),
        "archives": repo_relative(args.archives),
        "folds_from": repo_relative(args.folds),
        "code": {"git_head": git_head(), "tsdive_version": tsdive.__version__},
        "parameters": {
            "variables": variables,
            "context_minutes": CONTEXT_MINUTES,
            "target_minutes": TARGET_MINUTES,
            "min_context_minutes": MIN_CONTEXT_MINUTES,
            "min_target_minutes": MIN_TARGET_MINUTES,
            "min_scored_variables": MIN_SCORED_VARIABLES,
            "quantile_levels": list(QUANTILE_LEVELS),
            "width_floor_share": WIDTH_FLOOR_SHARE,
            "baseline_windows": BASELINE_WINDOWS,
            "quality_used": GOOD,
        },
        "threshold_rule": (
            "each instance's own alarm limit is the largest score over its three "
            "baseline windows; a test window fires when it scores strictly above it"
        ),
        "far_floor": r4(far_floor(BASELINE_WINDOWS)),
        "refusals": result.counts,
        "wall_seconds": round(elapsed, 1),
        "model_load_seconds": round(loaded - started, 1),
    }
    (out_dir / "run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    shown = ["tool", "n_instances_with_threshold", "roc_auc", "paired_hit_post0",
             "paired_hit_post1", "false_alarm_rate_pre", "over_floor"]
    print(result.summary[shown].to_string(index=False))
    print(f"wall {elapsed:.1f} s; wrote {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

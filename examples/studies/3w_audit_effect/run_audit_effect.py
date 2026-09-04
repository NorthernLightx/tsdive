"""What the zero-scale and quality checks change in the 3W own-history numbers.

`own_history_baselines` in the detector study drops an (instance,
variable) pair when `mad_baseline` raises `InsufficientQuality` or when
the scale it returns is zero. This study runs the own-history design
twice over the same cache and one code path, and publishes the
difference:

- ``checked``: the shipped behaviour. A pair that fails either check is
  dropped, and a window with no usable variable is refused.
- ``unchecked``: every pair with at least one finite baseline value is
  kept. A zero scale is used as naive code uses it, so the MAD screen
  scores 0 where the value equals the centre and ``+inf`` everywhere
  else, and the SPC chart, whose limits collapse onto the centre, counts
  every sub-group median that differs from it as a rule hit.

Both variants read the same cache, take each instance's first ``--k``
windows as its baseline and score the rest. The alarm threshold is the
largest score the detector gives that instance's own baseline windows,
which is in sample by construction, and ``fires`` is strict. The clock
control sits beside both.

Outputs in ``--out``:

- ``pairs.csv``: one row per (instance, variable, variant).
- ``per_record.csv``: the alarm rule and its metrics per instance.
- ``summary.csv``: one row per (detector, variant, group).
- ``run.json``: provenance, parameters, counts, the frozen-tag population.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))

import tsdive  # noqa: E402
from tsdive.baselines import mad_baseline  # noqa: E402
from tsdive.errors import InsufficientQuality  # noqa: E402
from tsdive.eval import (  # noqa: E402
    clock_control,
    far_floor,
    fires,
    ranking_metrics,
    worst_baseline_threshold,
)
from tsdive.spc import apply_rules, individuals_limits  # noqa: E402

DEFAULT_WINDOWS = "data/3w_windows"
DEFAULT_SOURCE = "data/3w"
DEFAULT_PER_TAG = "examples/studies/3w_profile/results/per_tag.csv"
DEFAULT_K = 3

MAD_TO_SIGMA = 1.4826  # the constant tsdive.baselines.mad_baseline uses

VARIANT_CHECKED = "checked"
VARIANT_UNCHECKED = "unchecked"
VARIANTS = (VARIANT_CHECKED, VARIANT_UNCHECKED)

TOOL_SCREEN = "screen"
TOOL_SPC = "spc"
TOOL_CLOCK = "clock"
TOOLS = (TOOL_SCREEN, TOOL_SPC, TOOL_CLOCK)

GROUP_NORMAL = "normal"
GROUP_MIXED = "mixed"
GROUP_POSITIVE_BASELINE = "positive_baseline"
GROUP_SHORT = "short"
GROUPS = (GROUP_NORMAL, GROUP_MIXED, GROUP_POSITIVE_BASELINE, GROUP_SHORT)

STATUS_KEPT = "kept"
STATUS_ZERO_SCALE = "dropped: zero scale"
STATUS_QUALITY = "dropped: InsufficientQuality"
STATUS_NO_FINITE = "no finite baseline value"
STATUS_ZERO_SCALE_KEPT = "kept with a zero scale"

REFUSAL_SHORT = "design: instance has {n} window(s); the first {k} are its baseline"
REFUSAL_NO_VARIABLE = "no usable variable in this window"
REFUSAL_ONE_CLASS = "fold under test carries one class only; no ranking metric exists"
COMMON_ROLE_SKIP = "MODE"


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


def _rate(num, den) -> float | None:
    return r4(num / den) if den else None


def rank_ready(values: np.ndarray) -> np.ndarray:
    """Replace ``+inf`` with one step above the largest finite score.

    The ranking metrics need an order, and every infinite score shares the
    top rank. Leaving ``inf`` in place makes ``numpy.diff`` produce NaN on
    the tied run, which the precision-recall curve reads as a new point.
    """
    out = np.asarray(values, dtype=float).copy()
    infinite = np.isposinf(out)
    if infinite.any():
        finite = out[np.isfinite(out)]
        out[infinite] = (float(finite.max()) + 1.0) if finite.size else 1.0
    return out


def mask_beyond_float32(frame: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, dict]:
    """Blank the cells no float32 can hold, and count them.

    3W ships a historian null sentinel as a number, so three
    ``WELL-00006`` instances hold ``P-PDG`` at -1.180116e+42 for their
    whole life. Both variants get the same mask, so it is not part of the
    difference this study measures.
    """
    block = frame[columns].to_numpy(dtype=float)
    bad = np.abs(block) > float(np.finfo(np.float32).max)
    detail = {"n_cells": int(bad.sum()), "n_rows": int(bad.any(axis=1).sum())}
    if not bad.any():
        return frame, detail
    block[bad] = np.nan
    out = frame.copy()
    out[columns] = block
    return out, detail


def synthetic_history(values: np.ndarray) -> pd.DataFrame:
    """A value+quality frame ``mad_baseline`` accepts; the stamps are placeholders."""
    stamps = pd.date_range("2000-01-01", periods=len(values), freq="min", tz="UTC")
    return pd.DataFrame(
        {"timestamp": stamps, "value": np.asarray(values, dtype=float), "quality": "GOOD"}
    )


# ---------------------------------------------------------------- cache


@dataclass
class Instance:
    """One instance's windows, sub-group medians and window medians."""

    instance: str
    well: str
    folder_label: str
    window_index: list[int]
    window_start: list[str]
    labels: list[int]
    group: str
    grid: dict[str, np.ndarray] = field(default_factory=dict)  # (n_windows, 60)
    medians: dict[str, np.ndarray] = field(default_factory=dict)  # (n_windows,)

    @property
    def n_windows(self) -> int:
        return len(self.window_index)


def assign_group(labels: list[int], k: int) -> str:
    if len(labels) <= k:
        return GROUP_SHORT
    if any(labels[:k]):
        return GROUP_POSITIVE_BASELINE
    if not any(labels):
        return GROUP_NORMAL
    return GROUP_MIXED


def load_cache(cache: Path, k: int) -> tuple[list[Instance], list[str], dict]:
    manifest = json.loads((cache / "MANIFEST.json").read_text("utf-8"))
    variables = list(manifest["common_variables"])
    windows = pd.read_parquet(cache / "windows.parquet")
    features = pd.read_parquet(cache / "features.parquet")
    grid = pd.read_parquet(cache / "subgroup_medians.parquet")
    subgroup_cols = [c for c in grid.columns if c.startswith("s") and c[1:].isdigit()]
    features, feature_beyond = mask_beyond_float32(features, ["median"])
    grid, grid_beyond = mask_beyond_float32(grid, subgroup_cols)
    grid_by = {key: frame for key, frame in grid.groupby("instance", sort=False)}
    med_by = {key: frame for key, frame in features.groupby("instance", sort=False)}
    out: list[Instance] = []
    for name, frame in windows.groupby("instance", sort=True):
        frame = frame.sort_values("window_index", kind="stable")
        index = [int(v) for v in frame["window_index"]]
        labels = [int(v) for v in frame["label"]]
        item = Instance(
            instance=str(name),
            well=str(frame["well"].iloc[0]),
            folder_label=str(frame["folder_label"].iloc[0]),
            window_index=index,
            window_start=[str(v) for v in frame["window_start"]],
            labels=labels,
            group=assign_group(labels, k),
        )
        if item.group != GROUP_SHORT:
            item.grid = _grid_matrix(grid_by.get(str(name)), index, variables, subgroup_cols)
            item.medians = _median_vector(med_by.get(str(name)), index, variables)
        out.append(item)
    counts = {
        "cache": str(cache).replace("\\", "/"),
        "manifest_sha256": sha256_file(cache / "MANIFEST.json")[:12],
        "n_instances": int(windows["instance"].nunique()),
        "n_windows": len(windows),
        "window_s": float(manifest["window_s"]),
        "subgroup_cols": len(subgroup_cols),
        "masked_beyond_float32": {"features": feature_beyond, "subgroups": grid_beyond},
    }
    return out, variables, counts


def _grid_matrix(
    frame: pd.DataFrame | None, index: list[int], variables: list[str], cols: list[str]
) -> dict[str, np.ndarray]:
    out = {v: np.full((len(index), len(cols)), np.nan) for v in variables}
    if frame is None or frame.empty:
        return out
    position_of = {window: position for position, window in enumerate(index)}
    values = frame[cols].to_numpy(dtype=float)
    window_of = frame["window_index"].to_numpy()
    variable_of = frame["variable"].to_numpy()
    for row in range(len(frame)):
        position = position_of.get(int(window_of[row]))
        target = out.get(str(variable_of[row]))
        if position is None or target is None:
            continue
        target[position] = values[row]
    return out


def _median_vector(
    frame: pd.DataFrame | None, index: list[int], variables: list[str]
) -> dict[str, np.ndarray]:
    out = {v: np.full(len(index), np.nan) for v in variables}
    if frame is None or frame.empty:
        return out
    position_of = {window: position for position, window in enumerate(index)}
    values = frame["median"].to_numpy(dtype=float)
    window_of = frame["window_index"].to_numpy()
    variable_of = frame["variable"].to_numpy()
    for row in range(len(frame)):
        position = position_of.get(int(window_of[row]))
        target = out.get(str(variable_of[row]))
        if position is None or target is None:
            continue
        target[position] = values[row]
    return out


# ---------------------------------------------------------------- baselines


@dataclass
class Baseline:
    centre: float
    scale: float
    status: str
    n_finite: int
    distinct: int


def fit_pair(values: np.ndarray, variant: str) -> Baseline:
    """The centre and scale for one (instance, variable), or the drop reason."""
    finite = values[np.isfinite(values)]
    distinct = int(np.unique(finite).size)
    if finite.size == 0:
        return Baseline(float("nan"), float("nan"), STATUS_NO_FINITE, 0, 0)
    try:
        baseline = mad_baseline(synthetic_history(finite))
        centre, scale = float(baseline.center), float(baseline.scale)
        quality_ok = True
    except InsufficientQuality:
        centre = float(np.median(finite))
        scale = MAD_TO_SIGMA * float(np.median(np.abs(finite - centre)))
        quality_ok = False
    if variant == VARIANT_CHECKED:
        if not quality_ok:
            return Baseline(centre, scale, STATUS_QUALITY, finite.size, distinct)
        if scale <= 0:
            return Baseline(centre, scale, STATUS_ZERO_SCALE, finite.size, distinct)
        return Baseline(centre, scale, STATUS_KEPT, finite.size, distinct)
    status = STATUS_ZERO_SCALE_KEPT if scale <= 0 else STATUS_KEPT
    return Baseline(centre, scale, status, finite.size, distinct)


def fit_instance(item: Instance, variables: list[str], k: int, variant: str) -> dict[str, Baseline]:
    return {
        variable: fit_pair(item.grid[variable][:k].ravel(), variant) for variable in variables
    }


# ---------------------------------------------------------------- scoring


@dataclass
class Scored:
    scores: dict[int, float] = field(default_factory=dict)
    refusals: dict[int, str] = field(default_factory=dict)


def screen_scores(item: Instance, fits: dict[str, Baseline], variables: list[str]) -> Scored:
    """Largest robust z of the window median over the usable variables."""
    out = Scored()
    for position in range(item.n_windows):
        best = None
        for variable in variables:
            fit = fits[variable]
            if fit.status in (STATUS_NO_FINITE, STATUS_ZERO_SCALE, STATUS_QUALITY):
                continue
            value = item.medians[variable][position]
            if not np.isfinite(value):
                continue
            if fit.scale <= 0:
                z = 0.0 if value == fit.centre else float("inf")
            else:
                z = abs(float(value) - fit.centre) / fit.scale
            best = z if best is None else max(best, z)
        if best is None:
            out.refusals[position] = REFUSAL_NO_VARIABLE
        else:
            out.scores[position] = best
    return out


def spc_scores(
    item: Instance, fits: dict[str, Baseline], variables: list[str], window_s: float
) -> Scored:
    """Rule hits summed over the usable variables, on the sub-group medians."""
    out = Scored()
    n_cols = next(iter(item.grid.values())).shape[1]
    offsets = np.arange(n_cols) * (window_s / n_cols)
    limits = {
        variable: individuals_limits(fit.centre, fit.scale)
        for variable, fit in fits.items()
        if fit.status not in (STATUS_NO_FINITE, STATUS_ZERO_SCALE, STATUS_QUALITY)
        and fit.scale > 0
    }
    for position in range(item.n_windows):
        hits, scored = 0, 0
        stamps_all = pd.Timestamp(item.window_start[position]) + pd.to_timedelta(
            offsets, unit="s"
        )
        for variable in variables:
            fit = fits[variable]
            if fit.status in (STATUS_NO_FINITE, STATUS_ZERO_SCALE, STATUS_QUALITY):
                continue
            series = item.grid[variable][position]
            finite = np.isfinite(series)
            if finite.sum() < 2:
                continue
            if fit.scale <= 0:
                hits += int(np.count_nonzero(series[finite] != fit.centre))
                scored += 1
                continue
            stamps = pd.Series(stamps_all[finite])
            hits += len(apply_rules(stamps, pd.Series(series[finite]), limits[variable]))
            scored += 1
        if scored:
            out.scores[position] = float(hits)
        else:
            out.refusals[position] = REFUSAL_NO_VARIABLE
    return out


def clock_scores(item: Instance) -> Scored:
    out = Scored()
    for position, value in clock_control({p: p for p in range(item.n_windows)}).items():
        out.scores[int(position)] = float(value)
    return out


# ---------------------------------------------------------------- alarm rule


def per_record_row(item: Instance, tool: str, variant: str, scored: Scored, k: int) -> dict:
    stream = list(range(k, item.n_windows))
    first = next((p for p, label in enumerate(item.labels) if label == 1), None)
    row: dict = {
        "variant": variant,
        "detector": tool,
        "instance": item.instance,
        "well": item.well,
        "folder_label": item.folder_label,
        "group": item.group,
        "n_windows": item.n_windows,
        "n_stream": len(stream),
        "n_scored": 0,
        "n_refused": 0,
        "n_inf_windows": 0,
        "threshold": None,
        "first_fault_position": first,
        "n_pre": None,
        "n_pre_fired": None,
        "detected": None,
        "delay_windows": None,
        "refusal": "",
    }
    if item.group == GROUP_SHORT:
        row["refusal"] = REFUSAL_SHORT.format(n=item.n_windows, k=k)
        return row
    scored_stream = [p for p in stream if p in scored.scores]
    row["n_scored"] = len(scored_stream)
    row["n_refused"] = len(stream) - len(scored_stream)
    row["n_inf_windows"] = int(
        sum(1 for p in scored_stream if not math.isfinite(scored.scores[p]))
    )
    baseline_scores = [scored.scores[p] for p in range(k) if p in scored.scores]
    if not baseline_scores:
        row["refusal"] = "no baseline window scored: " + "; ".join(
            sorted({scored.refusals.get(p, "not scored") for p in range(k)})
        )
        return row
    threshold = worst_baseline_threshold(baseline_scores)
    row["threshold"] = threshold
    fired = {p: fires(scored.scores[p], threshold) for p in scored_stream}
    pre = [p for p in fired if first is None or p < first]
    row["n_pre"] = len(pre)
    row["n_pre_fired"] = int(sum(fired[p] for p in pre))
    if first is not None:
        span = sorted(p for p in fired if p >= first and fired[p])
        row["detected"] = bool(span)
        row["delay_windows"] = (span[0] - first) if span else None
    return row


# ---------------------------------------------------------------- tables


def pair_rows(item: Instance, variant: str, fits: dict[str, Baseline]) -> list[dict]:
    return [
        {
            "variant": variant,
            "instance": item.instance,
            "well": item.well,
            "variable": variable,
            "group": item.group,
            "status": fit.status,
            "n_finite_baseline": fit.n_finite,
            "baseline_distinct": fit.distinct,
            "centre": r4(fit.centre),
            "scale": r4(fit.scale),
        }
        for variable, fit in sorted(fits.items())
    ]


def score_rows(item: Instance, tool: str, variant: str, scored: Scored, k: int) -> list[dict]:
    rows = []
    for position in range(item.n_windows):
        value = scored.scores.get(position)
        rows.append(
            {
                "variant": variant,
                "detector": tool,
                "instance": item.instance,
                "group": item.group,
                "position": position,
                "label": item.labels[position],
                "role": "baseline" if position < k else "stream",
                "score": value,
                "refusal": scored.refusals.get(position, ""),
            }
        )
    return rows


def summary_rows(
    scores: pd.DataFrame, per_record: pd.DataFrame, pairs: pd.DataFrame, k: int
) -> list[dict]:
    floor = far_floor(k)
    rows = []
    for variant in VARIANTS:
        for tool in TOOLS:
            for group in (*GROUPS, "pooled"):
                pr = per_record[
                    (per_record["variant"] == variant) & (per_record["detector"] == tool)
                ]
                sc = scores[
                    (scores["variant"] == variant)
                    & (scores["detector"] == tool)
                    & (scores["role"] == "stream")
                ]
                if group != "pooled":
                    pr = pr[pr["group"] == group]
                    sc = sc[sc["group"] == group]
                else:
                    pr = pr[pr["group"] != GROUP_SHORT]
                    sc = sc[sc["group"] != GROUP_SHORT]
                if pr.empty:
                    continue
                usable = sc[sc["score"].notna()]
                if usable.empty or usable["label"].nunique() < 2:
                    metrics = {
                        "roc_auc": None,
                        "pr_auc": None,
                        "refusal": REFUSAL_ONE_CLASS if not usable.empty else "no scored window",
                    }
                else:
                    m = ranking_metrics(
                        usable["label"].to_numpy(),
                        rank_ready(usable["score"].to_numpy(dtype=float)),
                    )
                    metrics = {
                        "roc_auc": r4(m.roc_auc),
                        "pr_auc": r4(m.pr_auc),
                        "refusal": m.refusal or "",
                    }
                ok = pr[pr["threshold"].notna()]
                thresholds = pd.to_numeric(ok["threshold"], errors="coerce").to_numpy(
                    dtype=float
                )
                n_pre = int(ok["n_pre"].fillna(0).sum())
                n_pre_fired = int(ok["n_pre_fired"].fillna(0).sum())
                far = _rate(n_pre_fired, n_pre)
                detected = ok["detected"].eq(True)
                delays = ok.loc[detected, "delay_windows"].dropna()
                pair_part = pairs[pairs["variant"] == variant]
                if group != "pooled":
                    pair_part = pair_part[pair_part["group"] == group]
                rows.append(
                    {
                        "variant": variant,
                        "detector": tool,
                        "group": group,
                        "n_instances": len(pr),
                        "n_instances_with_threshold": len(ok),
                        "n_instances_infinite_threshold": int(np.isinf(thresholds).sum()),
                        "n_stream_windows": len(sc),
                        "n_windows_scored": len(usable),
                        "n_windows_refused": int(len(sc) - len(usable)),
                        "n_windows_infinite": int(
                            (~np.isfinite(usable["score"].to_numpy(dtype=float))).sum()
                        ),
                        "roc_auc": metrics["roc_auc"],
                        "pr_auc": metrics["pr_auc"],
                        "auc_refusal": metrics["refusal"],
                        "n_pairs": len(pair_part),
                        "n_pairs_kept": int(
                            pair_part["status"].isin([STATUS_KEPT, STATUS_ZERO_SCALE_KEPT]).sum()
                        ),
                        "n_pairs_zero_scale": int(
                            pair_part["status"]
                            .isin([STATUS_ZERO_SCALE, STATUS_ZERO_SCALE_KEPT])
                            .sum()
                        ),
                        "n_pairs_insufficient_quality": int(
                            (pair_part["status"] == STATUS_QUALITY).sum()
                        ),
                        "n_pre_windows": n_pre,
                        "far_window": far,
                        "far_record": _rate(
                            int((ok["n_pre_fired"].fillna(0) > 0).sum()), len(ok)
                        )
                        if n_pre
                        else None,
                        "far_floor": r4(floor),
                        "over_floor": r4(far - floor) if far is not None else None,
                        "n_detected": int(detected.sum()) if detected.any() else 0,
                        "detect_rate": _rate(
                            int(detected.sum()), int(ok["detected"].notna().sum())
                        ),
                        "median_delay_windows": r4(delays.median()) if len(delays) else None,
                    }
                )
    return rows


def frozen_population(per_tag: Path, instances: list[Instance], variables: list[str]) -> dict:
    """How many whole-archive frozen measurement tags the cache can reach."""
    if not per_tag.exists():
        return {"available": False}
    frame = pd.read_csv(per_tag)
    measurement = frame[frame["role"] != COMMON_ROLE_SKIP]
    frozen = measurement[measurement["whole_life_distinct"] == 1]
    on_common = frozen[frozen["variable"].isin(variables)]
    cached = {item.instance for item in instances}
    scored = {item.instance for item in instances if item.group != GROUP_SHORT}
    return {
        "available": True,
        "per_tag": str(per_tag).replace("\\", "/"),
        "n_measurement_tags": len(measurement),
        "n_frozen_whole_archive": len(frozen),
        "n_frozen_on_common_variables": len(on_common),
        "n_frozen_in_cached_instances": int(on_common["instance"].isin(cached).sum()),
        "n_frozen_in_scored_instances": int(on_common["instance"].isin(scored).sum()),
        "per_variable": {
            str(variable): int(count)
            for variable, count in sorted(on_common["variable"].value_counts().items())
        },
    }


# ---------------------------------------------------------------- run


@dataclass
class StudyResult:
    pairs: pd.DataFrame
    scores: pd.DataFrame
    per_record: pd.DataFrame
    summary: pd.DataFrame
    run_info: dict


def run(
    windows: Path,
    out_dir: Path,
    *,
    k: int = DEFAULT_K,
    source: Path | None = None,
    per_tag: Path | None = None,
) -> StudyResult:
    started = time.perf_counter()
    instances, variables, counts = load_cache(windows, k)
    window_s = float(counts["window_s"])
    pair_parts: list[dict] = []
    score_parts: list[dict] = []
    record_parts: list[dict] = []
    for item in instances:
        if item.group == GROUP_SHORT:
            for variant in VARIANTS:
                for tool in TOOLS:
                    record_parts.append(per_record_row(item, tool, variant, Scored(), k))
            continue
        for variant in VARIANTS:
            fits = fit_instance(item, variables, k, variant)
            pair_parts.extend(pair_rows(item, variant, fits))
            answers = {
                TOOL_SCREEN: screen_scores(item, fits, variables),
                TOOL_SPC: spc_scores(item, fits, variables, window_s),
                TOOL_CLOCK: clock_scores(item),
            }
            for tool in TOOLS:
                score_parts.extend(score_rows(item, tool, variant, answers[tool], k))
                record_parts.append(per_record_row(item, tool, variant, answers[tool], k))

    pairs = pd.DataFrame(pair_parts).sort_values(
        ["variant", "instance", "variable"], kind="stable"
    )
    scores = pd.DataFrame(score_parts).sort_values(
        ["variant", "detector", "instance", "position"], kind="stable"
    )
    per_record = pd.DataFrame(record_parts).sort_values(
        ["variant", "detector", "group", "instance"], kind="stable"
    )
    summary = pd.DataFrame(summary_rows(scores, per_record, pairs, k))
    source_manifest = None if source is None else source / "MANIFEST.sha256.json"
    run_info = {
        "study": "3w_audit_effect",
        "code": {"git_head": git_head(), "tsdive_version": tsdive.__version__},
        "dataset_manifest_sha256": (
            sha256_file(source_manifest)[:12]
            if source_manifest is not None and source_manifest.exists()
            else None
        ),
        "cache": counts,
        "parameters": {
            "k_baseline_windows": k,
            "variables": variables,
            "far_floor": r4(far_floor(k)),
        },
        "group_counts": {
            group: int(sum(1 for item in instances if item.group == group)) for group in GROUPS
        },
        "frozen_population": frozen_population(
            per_tag or Path(DEFAULT_PER_TAG), instances, variables
        ),
        "n_rows": {
            "pairs": len(pairs),
            "window_scores": len(scores),
            "per_record": len(per_record),
            "summary": len(summary),
        },
        "wall_seconds": round(time.perf_counter() - started, 1),
    }
    result = StudyResult(pairs, scores, per_record, summary, run_info)
    write_result(result, out_dir)
    return result


def write_result(result: StudyResult, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    write_csv(result.pairs, out / "pairs.csv")
    write_csv(result.per_record, out / "per_record.csv")
    write_csv(result.summary, out / "summary.csv")
    (out / "run.json").write_text(
        json.dumps(result.run_info, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--windows", default=DEFAULT_WINDOWS)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--per-tag", default=DEFAULT_PER_TAG)
    parser.add_argument("--out", default=str(HERE / "results"))
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    args = parser.parse_args(argv)

    result = run(
        Path(args.windows),
        Path(args.out),
        k=args.k,
        source=Path(args.source),
        per_tag=Path(args.per_tag),
    )
    print(json.dumps(result.run_info["n_rows"], sort_keys=True))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

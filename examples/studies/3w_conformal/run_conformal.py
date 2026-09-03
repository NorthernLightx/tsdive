"""Conformal test martingale alarm on the 3W window caches.

The study reads the detector study's two window caches, the own-history
cache under ``data/3w_windows`` and the onset-aligned cache under
``data/3w_windows_aligned``, and scores every instance with four alarm
rules that share one nonconformity score:

- ``conformal``: ``conformal_p_values`` over the calibration window,
  ``mixture_martingale``, ``martingale_alarm`` at each ``--deltas`` value.
- ``worst_baseline``: the detector study's rule, ``fires`` above
  ``worst_baseline_threshold`` of the three baseline windows.
- ``clock``: the martingale on the sub-group's position in the record,
  reading no sensor value. Published as the control.
- ``permutation``: the conformal rule on the instance's own scores after
  a shuffle across the calibration and stream, ``--seeds`` times. This
  makes exchangeability true by construction and checks the bound.

Per instance the first two windows are the fit set (median and MAD per
variable through ``mad_baseline``), the third is the calibration window
and the rest are the stream. The nonconformity of a one-minute sub-group
is the largest robust z over the usable variables.

Outputs in ``--out``:

- ``per_instance.csv``: one row per (bed, instance, method, delta).
- ``summary.csv``: one row per (bed, group, method, delta).
- ``permutation.csv``: alarm fraction per (bed, group, delta, seed).
- ``run.json``: provenance, parameters, group and refusal counts.
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
    conformal_p_values,
    fires,
    martingale_alarm,
    mixture_martingale,
    worst_baseline_threshold,
)

DEFAULT_WINDOWS = "data/3w_windows"
DEFAULT_ALIGNED = "data/3w_windows_aligned"
DEFAULT_SOURCE = "data/3w"
DEFAULT_DELTAS = (0.05, 0.01)
DEFAULT_SEEDS = 10
DEFAULT_K = 3
N_FIT = 2
MIN_CALIBRATION = 30
EPSILONS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
SUBGROUP_COLS = [f"s{i:02d}" for i in range(60)]
LN10 = math.log(10.0)

BED_OWN = "own_history"
BED_ALIGNED = "aligned"
METHOD_CONFORMAL = "conformal"
METHOD_WORST = "worst_baseline"
METHOD_CLOCK = "clock"
METHOD_PERMUTATION = "permutation"
MARTINGALE_METHODS = (METHOD_CONFORMAL, METHOD_CLOCK)
PER_INSTANCE_METHODS = (METHOD_CONFORMAL, METHOD_WORST, METHOD_CLOCK)
OUTCOMES = ("false_alarm", "detection", "none", "refused")

GROUP_NORMAL = "normal"
GROUP_MIXED = "mixed"
GROUP_POSITIVE_BASELINE = "positive_baseline"
GROUP_ALIGNED = "aligned"
GROUP_SHORT = "short"
DESIGN_REFUSAL = "design"

PER_INSTANCE_COLUMNS = [
    "bed",
    "group",
    "instance",
    "well",
    "folder_label",
    "method",
    "delta",
    "n_stream_windows",
    "n_usable_variables",
    "first_fault_window",
    "alarm_window",
    "alarm_subgroup",
    "alarm_label",
    "outcome",
    "delay_windows",
    "max_log10_martingale",
    "refusal",
]
SUMMARY_COLUMNS = [
    "bed",
    "group",
    "method",
    "delta",
    "n_instances",
    "n_refused",
    "n_scored",
    "n_alarm",
    "n_false_alarm",
    "n_detection",
    "alarm_rate",
    "far",
    "detect",
    "detect_post0",
    "detect_post1",
    "median_delay_windows",
    "median_max_log10_martingale",
]
PERMUTATION_COLUMNS = ["bed", "group", "delta", "seed", "n_instances", "alarm_fraction"]


# ------------------------------------------------------------ provenance


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


# ------------------------------------------------------------------ data


@dataclass(frozen=True)
class Cache:
    path: Path
    manifest: dict
    windows: pd.DataFrame
    subgroups: pd.DataFrame
    aligned: bool

    @property
    def bed(self) -> str:
        return BED_ALIGNED if self.aligned else BED_OWN


def load_cache(path: Path, aligned: bool) -> Cache:
    manifest = json.loads((path / "MANIFEST.json").read_text("utf-8"))
    windows = pd.read_parquet(path / "windows.parquet")
    subgroups = pd.read_parquet(path / "subgroup_medians.parquet")
    order = ["instance", "onset_offset"] if aligned else ["instance", "window_index"]
    windows = windows.sort_values(order).reset_index(drop=True)
    return Cache(path, manifest, windows, subgroups, aligned)


def quality_frame(values: np.ndarray) -> pd.DataFrame:
    """A value+quality history frame for ``mad_baseline``; stamps are placeholders.

    ``mad_baseline`` reads value and quality only. Every sub-group median
    comes from GOOD samples by construction of the cache.
    """
    stamps = pd.date_range("2000-01-01", periods=len(values), freq="min", tz="UTC")
    return pd.DataFrame(
        {"timestamp": stamps, "value": np.asarray(values, dtype=float), "quality": "GOOD"}
    )


# ------------------------------------------------------------- scoring


@dataclass
class Scored:
    """An instance's windows, its calibration and stream scores, and the fits."""

    instance: str
    well: str
    folder_label: str
    window_index: list[int]
    labels: list[int]
    k: int
    calibration: np.ndarray
    stream: np.ndarray
    stream_window: np.ndarray  # position in window_index for each stream score
    stream_subgroup: np.ndarray
    window_scores: list[float]  # max nonconformity per window, nan when no finite sub-group
    usable: list[str] = field(default_factory=list)
    unusable: dict[str, str] = field(default_factory=dict)

    @property
    def baseline_labels(self) -> list[int]:
        return self.labels[: self.k]

    @property
    def stream_labels(self) -> list[int]:
        return self.labels[self.k :]

    @property
    def n_stream_windows(self) -> int:
        return len(self.labels) - self.k


@dataclass(frozen=True)
class Refused:
    instance: str
    well: str
    folder_label: str
    labels: list[int]
    k: int
    reason: str

    @property
    def baseline_labels(self) -> list[int]:
        return self.labels[: self.k]

    @property
    def stream_labels(self) -> list[int]:
        return self.labels[self.k :]

    @property
    def n_stream_windows(self) -> int:
        return max(len(self.labels) - self.k, 0)


def window_matrix(sub: pd.DataFrame, window_index: int, variables: list[str]) -> np.ndarray:
    """Sub-group medians of one window as a (60, n_variables) array; NaN where absent."""
    rows = sub[sub["window_index"] == window_index].set_index("variable")
    out = np.full((len(SUBGROUP_COLS), len(variables)), np.nan)
    for j, variable in enumerate(variables):
        if variable in rows.index:
            out[:, j] = rows.loc[variable, SUBGROUP_COLS].to_numpy(dtype=float)
    return out


def fit_variables(
    fit: list[np.ndarray], variables: list[str]
) -> tuple[dict[str, tuple[float, float]], dict[str, str]]:
    """median and MAD scale per variable over the fit windows' finite sub-group medians."""
    fits: dict[str, tuple[float, float]] = {}
    unusable: dict[str, str] = {}
    for j, variable in enumerate(variables):
        values = np.concatenate([m[:, j] for m in fit])
        values = values[np.isfinite(values)]
        try:
            baseline = mad_baseline(quality_frame(values))
        except InsufficientQuality as e:
            unusable[variable] = f"InsufficientQuality: {e}"
            continue
        if baseline.scale <= 0:
            unusable[variable] = "MAD over the fit windows is zero, so no robust z exists"
            continue
        fits[variable] = (float(baseline.center), float(baseline.scale))
    return fits, unusable


def nonconformity(matrix: np.ndarray, fits: list[tuple[float, float]]) -> np.ndarray:
    """Per sub-group: max over usable variables of |x - centre| / scale; NaN when none finite."""
    centre = np.array([c for c, _ in fits])
    scale = np.array([s for _, s in fits])
    z = np.abs(matrix - centre) / scale
    out = np.full(z.shape[0], np.nan)
    valid = np.isfinite(z).any(axis=1)
    if valid.any():
        out[valid] = np.nanmax(z[valid], axis=1)
    return out


def score_instance(
    windows: pd.DataFrame, sub: pd.DataFrame, variables: list[str], k: int, aligned: bool
) -> Scored | Refused:
    """Fit, calibration and stream scores for one instance, or the refusal."""
    instance = str(windows["instance"].iloc[0])
    well = str(windows["well"].iloc[0])
    folder = str(windows["folder_label"].iloc[0])
    label_col = "design_label" if aligned else "label"
    labels = [int(v) for v in windows[label_col]]
    index = [int(v) for v in windows["window_index"]]
    if len(index) < k + 1:
        return Refused(
            instance,
            well,
            folder,
            labels,
            k,
            f"{DESIGN_REFUSAL}: {len(index)} window(s); {k} form the baseline, "
            "leaving none to score",
        )
    matrices = [window_matrix(sub, w, variables) for w in index]
    fits, unusable = fit_variables(matrices[:N_FIT], variables)
    if not fits:
        detail = "; ".join(f"{v}: {r}" for v, r in sorted(unusable.items()))
        return Refused(instance, well, folder, labels, k, f"no usable variable: {detail}")
    usable = [v for v in variables if v in fits]
    cols = [variables.index(v) for v in usable]
    fit_list = [fits[v] for v in usable]
    scores = [nonconformity(m[:, cols], fit_list) for m in matrices]
    cal = scores[k - 1]
    cal = cal[np.isfinite(cal)]
    if cal.size < MIN_CALIBRATION:
        return Refused(
            instance,
            well,
            folder,
            labels,
            k,
            f"calibration: window has {cal.size} finite sub-group scores; "
            f"{MIN_CALIBRATION} required",
        )
    stream_parts, pos_parts, sub_parts = [], [], []
    for position in range(k, len(index)):
        s = scores[position]
        finite = np.flatnonzero(np.isfinite(s))
        stream_parts.append(s[finite])
        pos_parts.append(np.full(finite.size, position, dtype=int))
        sub_parts.append(finite)
    window_scores = [float(np.nanmax(s)) if np.isfinite(s).any() else float("nan") for s in scores]
    return Scored(
        instance=instance,
        well=well,
        folder_label=folder,
        window_index=index,
        labels=labels,
        k=k,
        calibration=cal,
        stream=np.concatenate(stream_parts),
        stream_window=np.concatenate(pos_parts),
        stream_subgroup=np.concatenate(sub_parts),
        window_scores=window_scores,
        usable=usable,
        unusable=unusable,
    )


# ------------------------------------------------------------ evaluation


def group_of(item: Scored | Refused, aligned: bool) -> str:
    """The instance's group by its baseline and stream labels; ``short`` when refused by design."""
    if aligned:
        return GROUP_ALIGNED
    if isinstance(item, Refused) and item.reason.startswith(DESIGN_REFUSAL):
        return GROUP_SHORT
    if any(item.baseline_labels):
        return GROUP_POSITIVE_BASELINE
    if any(item.stream_labels):
        return GROUP_MIXED
    return GROUP_NORMAL


def first_fault_position(labels: list[int], k: int) -> int | None:
    """Position (in the full window list) of the first stream window labelled 1."""
    for position in range(k, len(labels)):
        if labels[position] == 1:
            return position
    return None


def martingale_log10(calibration: np.ndarray, stream: np.ndarray) -> np.ndarray:
    p = conformal_p_values(calibration, stream, online=True)
    return mixture_martingale(p, EPSILONS) / LN10


def alarm_at(log10_values: np.ndarray, delta: float) -> int | None:
    return martingale_alarm(log10_values * LN10, delta)


def outcome_row(scored: Scored, alarm_position: int | None, alarm_subgroup: int | None) -> dict:
    """alarm window, its label, the outcome and the delay for one alarm rule.

    An alarm in a window labelled 1 is a detection; in a window labelled 0
    it is a false alarm. The 3W caches carry no label-0 window after an
    instance's first label-1 window, so a false alarm always precedes the
    fault.
    """
    first = first_fault_position(scored.labels, scored.k)
    row = {
        "n_stream_windows": scored.n_stream_windows,
        "n_usable_variables": len(scored.usable),
        "first_fault_window": scored.window_index[first] if first is not None else None,
        "alarm_window": None,
        "alarm_subgroup": None,
        "alarm_label": None,
        "outcome": "none",
        "delay_windows": None,
        "refusal": "",
    }
    if alarm_position is None:
        return row
    label = scored.labels[alarm_position]
    row["alarm_window"] = scored.window_index[alarm_position]
    row["alarm_subgroup"] = alarm_subgroup
    row["alarm_label"] = label
    if label == 1:
        row["outcome"] = "detection"
        if first is not None:
            row["delay_windows"] = scored.window_index[alarm_position] - scored.window_index[first]
    else:
        row["outcome"] = "false_alarm"
    return row


def evaluate_instance(
    scored: Scored, deltas: tuple[float, ...], seeds: int
) -> tuple[list[dict], list[dict]]:
    """per_instance rows for the three methods and permutation alarm flags."""
    head = {
        "instance": scored.instance,
        "well": scored.well,
        "folder_label": scored.folder_label,
    }
    rows: list[dict] = []

    def locate(idx: int | None) -> tuple[int | None, int | None]:
        if idx is None:
            return None, None
        return int(scored.stream_window[idx]), int(scored.stream_subgroup[idx])

    n_cal = scored.calibration.size
    conformal = martingale_log10(scored.calibration, scored.stream)
    clock = martingale_log10(
        np.arange(n_cal, dtype=float), n_cal + np.arange(scored.stream.size, dtype=float)
    )
    for method, values in ((METHOD_CONFORMAL, conformal), (METHOD_CLOCK, clock)):
        peak = round(float(values.max()), 4) if values.size else None
        for delta in deltas:
            position, subgroup = locate(alarm_at(values, delta))
            rows.append(
                {
                    **head,
                    "method": method,
                    "delta": delta,
                    **outcome_row(scored, position, subgroup),
                    "max_log10_martingale": peak,
                }
            )

    baseline_scores = [s for s in scored.window_scores[: scored.k] if math.isfinite(s)]
    threshold = worst_baseline_threshold(baseline_scores)
    position = next(
        (
            p
            for p in range(scored.k, len(scored.window_index))
            if math.isfinite(scored.window_scores[p]) and fires(scored.window_scores[p], threshold)
        ),
        None,
    )
    rows.append(
        {
            **head,
            "method": METHOD_WORST,
            "delta": None,
            **outcome_row(scored, position, None),
            "max_log10_martingale": None,
        }
    )

    permutation: list[dict] = []
    pooled = np.concatenate([scored.calibration, scored.stream])
    for seed in range(seeds):
        shuffled = np.random.default_rng(seed).permutation(pooled)
        values = martingale_log10(shuffled[:n_cal], shuffled[n_cal:])
        for delta in deltas:
            permutation.append(
                {
                    "instance": scored.instance,
                    "delta": delta,
                    "seed": seed,
                    "alarm": alarm_at(values, delta) is not None,
                }
            )
    return rows, permutation


def refused_rows(item: Refused, deltas: tuple[float, ...]) -> list[dict]:
    head = {
        "instance": item.instance,
        "well": item.well,
        "folder_label": item.folder_label,
        "n_stream_windows": item.n_stream_windows,
        "n_usable_variables": None,
        "first_fault_window": None,
        "alarm_window": None,
        "alarm_subgroup": None,
        "alarm_label": None,
        "outcome": "refused",
        "delay_windows": None,
        "max_log10_martingale": None,
        "refusal": item.reason,
    }
    rows = [{**head, "method": METHOD_WORST, "delta": None}]
    for method in MARTINGALE_METHODS:
        rows += [{**head, "method": method, "delta": delta} for delta in deltas]
    return rows


def run_bed(
    cache: Cache, k: int, deltas: tuple[float, ...], seeds: int
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """per_instance and permutation frames for one cache, plus its counts."""
    variables = sorted(cache.subgroups["variable"].unique())
    sub_by_instance = dict(tuple(cache.subgroups.groupby("instance", sort=True)))
    empty_sub = cache.subgroups.iloc[0:0]
    rows: list[dict] = []
    perm: list[dict] = []
    groups: dict[str, int] = {}
    refusals: dict[str, int] = {}
    for instance, windows in cache.windows.groupby("instance", sort=True):
        item = score_instance(
            windows, sub_by_instance.get(instance, empty_sub), variables, k, cache.aligned
        )
        group = group_of(item, cache.aligned)
        groups[group] = groups.get(group, 0) + 1
        if isinstance(item, Refused):
            kind = item.reason.split(":")[0]
            refusals[kind] = refusals.get(kind, 0) + 1
            new_rows = refused_rows(item, deltas)
            new_perm: list[dict] = []
        else:
            new_rows, new_perm = evaluate_instance(item, deltas, seeds)
        rows += [{"bed": cache.bed, "group": group, **r} for r in new_rows]
        perm += [{"bed": cache.bed, "group": group, **r} for r in new_perm]
    per_instance = pd.DataFrame(rows, columns=PER_INSTANCE_COLUMNS)
    permutation = pd.DataFrame(perm, columns=["bed", "group", "instance", "delta", "seed", "alarm"])
    counts = {
        "n_windows": len(cache.windows),
        "n_instances": int(cache.windows["instance"].nunique()),
        "manifest_n_windows": cache.manifest.get("n_windows"),
        "manifest_n_instances": cache.manifest.get("n_instances"),
        "variables": variables,
        "groups": dict(sorted(groups.items())),
        "refusals": dict(sorted(refusals.items())),
    }
    return per_instance, permutation, counts


# --------------------------------------------------------------- summary


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def summarise(per_instance: pd.DataFrame, permutation: pd.DataFrame) -> pd.DataFrame:
    """One row per (bed, group, method, delta); rates over the scored instances.

    ``far`` is empty for ``positive_baseline`` (a baseline window carries a
    fault, so no alarm there is a false one by the labels); ``detect`` is
    filled for ``mixed`` and ``aligned``; ``detect_post0`` and
    ``detect_post1`` for ``aligned`` only. ``permutation`` rows carry the
    mean alarm fraction over seeds in ``alarm_rate``.
    """
    rows: list[dict] = []
    keyed = per_instance.assign(delta=per_instance["delta"].fillna(-1.0))
    for (bed, group, method, delta), g in keyed.groupby(
        ["bed", "group", "method", "delta"], sort=True
    ):
        scored = g[g["outcome"] != "refused"]
        n_scored = len(scored)
        alarms = scored[scored["outcome"] != "none"]
        false_alarms = scored[scored["outcome"] == "false_alarm"]
        detections = scored[scored["outcome"] == "detection"]
        peak = None
        if method in MARTINGALE_METHODS and n_scored:
            peak = round(float(scored["max_log10_martingale"].median()), 4)
        row = {
            "bed": bed,
            "group": group,
            "method": method,
            "delta": None if delta < 0 else delta,
            "n_instances": len(g),
            "n_refused": int(len(g) - n_scored),
            "n_scored": n_scored,
            "n_alarm": len(alarms),
            "n_false_alarm": len(false_alarms),
            "n_detection": len(detections),
            "alarm_rate": _rate(len(alarms), n_scored),
            "far": None,
            "detect": None,
            "detect_post0": None,
            "detect_post1": None,
            "median_delay_windows": None,
            "median_max_log10_martingale": peak,
        }
        if group != GROUP_POSITIVE_BASELINE:
            row["far"] = _rate(len(false_alarms), n_scored)
        if group in (GROUP_MIXED, GROUP_ALIGNED):
            row["detect"] = _rate(len(detections), n_scored)
            if len(detections):
                row["median_delay_windows"] = float(detections["delay_windows"].median())
        if group == GROUP_ALIGNED:
            row["detect_post0"] = _rate(int((detections["delay_windows"] == 0).sum()), n_scored)
            row["detect_post1"] = _rate(int((detections["delay_windows"] <= 1).sum()), n_scored)
        rows.append(row)
    for (bed, group, delta), g in permutation_table(permutation).groupby(
        ["bed", "group", "delta"], sort=True
    ):
        n_scored = int(g["n_instances"].iloc[0])
        rows.append(
            {
                "bed": bed,
                "group": group,
                "method": METHOD_PERMUTATION,
                "delta": delta,
                "n_instances": n_scored,
                "n_refused": 0,
                "n_scored": n_scored,
                "n_alarm": None,
                "n_false_alarm": None,
                "n_detection": None,
                "alarm_rate": round(float(g["alarm_fraction"].mean()), 4),
                "far": None,
                "detect": None,
                "detect_post0": None,
                "detect_post1": None,
                "median_delay_windows": None,
                "median_max_log10_martingale": None,
            }
        )
    out = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    return out.sort_values(
        ["bed", "group", "method", "delta"], na_position="first", kind="stable"
    ).reset_index(drop=True)


def permutation_table(permutation: pd.DataFrame) -> pd.DataFrame:
    """Alarm fraction over the group's scored instances per (bed, group, delta, seed)."""
    if not len(permutation):
        return pd.DataFrame(columns=PERMUTATION_COLUMNS)
    out = (
        permutation.groupby(["bed", "group", "delta", "seed"], sort=True)
        .agg(n_instances=("instance", "nunique"), alarm_fraction=("alarm", "mean"))
        .reset_index()
    )
    out["alarm_fraction"] = out["alarm_fraction"].astype(float).round(4)
    return out[PERMUTATION_COLUMNS]


# ------------------------------------------------------------------ main


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n")


def run(
    windows: Path,
    aligned: Path,
    out_dir: Path,
    *,
    deltas: tuple[float, ...] = DEFAULT_DELTAS,
    seeds: int = DEFAULT_SEEDS,
    k: int = DEFAULT_K,
    source: Path | None = None,
) -> dict:
    """Score both caches, write the four outputs and return the run.json content."""
    started = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    per_instance_parts, permutation_parts, beds = [], [], {}
    for path, is_aligned in ((windows, False), (aligned, True)):
        cache = load_cache(path, is_aligned)
        per_instance_bed, permutation_bed, counts = run_bed(cache, k, deltas, seeds)
        per_instance_parts.append(per_instance_bed)
        permutation_parts.append(permutation_bed)
        beds[cache.bed] = {
            "cache": str(path).replace("\\", "/"),
            "manifest_sha256": sha256_file(path / "MANIFEST.json"),
            **counts,
        }
    per_instance = pd.concat(per_instance_parts, ignore_index=True).sort_values(
        ["bed", "group", "instance", "method", "delta"], na_position="first", kind="stable"
    )
    permutation = pd.concat(permutation_parts, ignore_index=True)
    write_csv(per_instance, out_dir / "per_instance.csv")
    write_csv(summarise(per_instance, permutation), out_dir / "summary.csv")
    write_csv(permutation_table(permutation), out_dir / "permutation.csv")
    source_manifest = None if source is None else source / "MANIFEST.sha256.json"
    run_info = {
        "study": "3w_conformal",
        "code": {"git_head": git_head(), "tsdive_version": tsdive.__version__},
        "dataset_manifest_sha256": (
            sha256_file(source_manifest)
            if source_manifest is not None and source_manifest.exists()
            else None
        ),
        "beds": beds,
        "parameters": {
            "k": k,
            "n_fit_windows": N_FIT,
            "min_calibration_scores": MIN_CALIBRATION,
            "deltas": list(deltas),
            "permutation_seeds": seeds,
            "epsilons": list(EPSILONS),
        },
        "wall_seconds": round(time.perf_counter() - started, 1),
    }
    (out_dir / "run.json").write_text(
        json.dumps(run_info, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    return run_info


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--windows", default=DEFAULT_WINDOWS)
    parser.add_argument("--aligned-windows", default=DEFAULT_ALIGNED)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--out", default=str(HERE / "results"))
    parser.add_argument("--deltas", type=float, nargs="+", default=list(DEFAULT_DELTAS))
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    args = parser.parse_args(argv)
    info = run(
        Path(args.windows),
        Path(args.aligned_windows),
        Path(args.out),
        deltas=tuple(args.deltas),
        seeds=args.seeds,
        k=args.k,
        source=Path(args.source),
    )
    print(f"wall {info['wall_seconds']} s; wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

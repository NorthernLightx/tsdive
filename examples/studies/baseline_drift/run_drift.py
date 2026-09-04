"""Static, differenced and rolling baselines on the 3W and SKAB minute medians.

The study asks whether a differenced score or a moving baseline removes
the false alarms a static own-history baseline produces on records that
carry no fault, and what that costs in detection and delay. Four beds
run the same alarm rule:

- ``3w_own_history``: the unaligned 3W window cache, one hour per window
  and 60 one-minute sub-group medians inside it.
- ``3w_aligned``: the onset-aligned 3W cache, six windows per instance.
- ``skab``: the 34 labelled SKAB records, 60 s windows from each
  record's first timestamp, one minute median per window.
- ``skab_anomaly_free``: the single SKAB record with no labelled anomaly.

Three scorings share one code path. Each gives one score per window: the
largest robust z over the usable variables and over the window's minutes.

- ``static``: centre and scale are the median and 1.4826 x MAD of the
  minute medians over the first ``--k`` windows.
- ``differenced``: the same over the minute-to-minute differences.
- ``rolling``: centre and scale come from the ``--w`` windows immediately
  before the scored window, recomputed for every window, label-blind.

``clock`` sits beside them: the window's position in its own record,
reading no sensor value.

A variable is unusable when its scale is zero, and for ``rolling`` that
verdict is per window. A window with no usable variable and no finite
minute median is refused with the reason. The alarm threshold is
``worst_baseline_threshold`` over the design's threshold windows and
``fires`` is strict, so ``far_floor(--k)`` is the rate a score that reads
nothing already reaches.

Outputs in ``--out``:

- ``windows.csv``: one row per scored record's window.
- ``scores.csv``: one row per (bed, record, scoring, window).
- ``per_record.csv``: the alarm rule and its metrics per record.
- ``summary.csv``: one row per (bed, group, scoring).
- ``usability.csv``: one row per variable a scoring could not use.
- ``run.json``: provenance, parameters and counts.
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
from tsdive.eval import (  # noqa: E402
    clock_control,
    far_floor,
    fires,
    ranking_metrics,
    worst_baseline_threshold,
)

DEFAULT_WINDOWS = "data/3w_windows"
DEFAULT_ALIGNED = "data/3w_windows_aligned"
DEFAULT_3W_SOURCE = "data/3w"
DEFAULT_ARCHIVES = "data/skab_archives"
DEFAULT_SKAB_SOURCE = "data/skab"
DEFAULT_K = 3
DEFAULT_W = 3
DEFAULT_WINDOW_S = 60
RECOVERY_WINDOWS = 2

MAD_TO_SIGMA = 1.4826  # the constant tsdive.baselines.mad_baseline uses
SUBGROUP_COLS = [f"s{i:02d}" for i in range(60)]

BED_3W_OWN = "3w_own_history"
BED_3W_ALIGNED = "3w_aligned"
BED_SKAB = "skab"
BED_SKAB_FREE = "skab_anomaly_free"
BEDS = (BED_3W_OWN, BED_3W_ALIGNED, BED_SKAB, BED_SKAB_FREE)

SCORING_STATIC = "static"
SCORING_DIFF = "differenced"
SCORING_ROLLING = "rolling"
SCORING_CLOCK = "clock"
SCORINGS = (SCORING_STATIC, SCORING_DIFF, SCORING_ROLLING, SCORING_CLOCK)

GROUP_NORMAL = "normal"
GROUP_MIXED = "mixed"
GROUP_POSITIVE_BASELINE = "positive_baseline"
GROUP_SHORT = "short"
GROUP_ALIGNED = "aligned"
GROUP_LABELLED = "labelled"
GROUP_FREE = "anomaly_free"
COUNT_ONLY_GROUPS = (GROUP_POSITIVE_BASELINE, GROUP_SHORT)

USABILITY_COLUMNS = [
    "bed",
    "record",
    "variable",
    "scoring",
    "reason",
    "n_windows_unusable",
    "n_windows_total",
]

ROLE_FIT = "fit"
ROLE_THRESHOLD = "threshold"
ROLE_STREAM = "stream"
ROLE_OTHER = "other"

REFUSAL_ZERO_SCALE = "the scale over the baseline minutes is zero"
REFUSAL_NO_FINITE = "no finite minute median over the baseline windows"
REFUSAL_NO_MINUTE = "no usable variable has a finite minute median in this window"
REFUSAL_NO_HISTORY = "fewer than {w} windows precede this window"
REFUSAL_ONE_CLASS = "fold under test carries one class only; no ranking metric exists"
REFUSAL_ROLLING_ALIGNED = (
    "a rolling baseline needs {w} windows before the scored window, and the "
    "threshold windows of this design are the instance's first three, which "
    "have no predecessors in the cache"
)


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


def mad_centre_scale(values: np.ndarray) -> tuple[tuple[float, float] | None, str]:
    """median and 1.4826 x MAD of the finite entries, or the reason there is none."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None, REFUSAL_NO_FINITE
    centre = float(np.median(finite))
    scale = MAD_TO_SIGMA * float(np.median(np.abs(finite - centre)))
    if scale <= 0:
        return None, REFUSAL_ZERO_SCALE
    return (centre, scale), ""


# ---------------------------------------------------------------- records


@dataclass(frozen=True)
class Design:
    """Which window positions fit the baseline, set the threshold and stream."""

    fit: tuple[int, ...]
    threshold: tuple[int, ...]
    stream: tuple[int, ...] | None  # None: every position after the threshold ones

    @property
    def min_windows(self) -> int:
        last = max((*self.fit, *self.threshold, *(self.stream or ())))
        return last + 2 if self.stream is None else last + 1

    def stream_positions(self, n: int) -> list[int]:
        if self.stream is not None:
            return [p for p in self.stream if p < n]
        return list(range(max(self.threshold) + 1, n))

    def role(self, position: int, n: int) -> str:
        if position in self.threshold:
            return ROLE_THRESHOLD
        if position in self.stream_positions(n):
            return ROLE_STREAM
        if position in self.fit:
            return ROLE_FIT
        return ROLE_OTHER


DESIGN_OWN = Design(fit=(0, 1, 2), threshold=(3, 4, 5), stream=None)
DESIGN_ALIGNED = Design(fit=(0, 1, 2), threshold=(0, 1, 2), stream=(3, 4, 5))
DESIGNS = {
    BED_3W_OWN: DESIGN_OWN,
    BED_3W_ALIGNED: DESIGN_ALIGNED,
    BED_SKAB: DESIGN_OWN,
    BED_SKAB_FREE: DESIGN_OWN,
}


@dataclass
class RecordSeries:
    """One record's minute medians per variable, one row of the matrix per window."""

    bed: str
    record: str
    fold: str
    group: str
    window_index: list[int]
    window_start: list[str]
    labels: list[int]
    contiguous: list[bool]
    values: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def n_windows(self) -> int:
        return len(self.window_index)

    @property
    def variables(self) -> list[str]:
        return sorted(self.values)

    def differences(self) -> dict[str, np.ndarray]:
        """Minute-to-minute differences, NaN where the previous minute is missing."""
        out: dict[str, np.ndarray] = {}
        for variable, matrix in self.values.items():
            diff = np.full(matrix.shape, np.nan)
            if matrix.shape[1] > 1:
                diff[:, 1:] = np.diff(matrix, axis=1)
            for position in range(1, matrix.shape[0]):
                if self.contiguous[position]:
                    diff[position, 0] = matrix[position, 0] - matrix[position - 1, -1]
            out[variable] = diff
        return out


def assign_group(labels: list[int], k: int) -> str:
    if len(labels) <= k:
        return GROUP_SHORT
    if any(labels[:k]):
        return GROUP_POSITIVE_BASELINE
    if not any(labels):
        return GROUP_NORMAL
    return GROUP_MIXED


def load_3w(cache: Path, bed: str, k: int) -> tuple[list[RecordSeries], dict]:
    """Records of one 3W window cache; the sub-group medians are read per group."""
    manifest = json.loads((cache / "MANIFEST.json").read_text("utf-8"))
    variables = list(manifest["common_variables"])
    windows = pd.read_parquet(cache / "windows.parquet")
    label_col = "design_label" if bed == BED_3W_ALIGNED else "label"
    sub = pd.read_parquet(cache / "subgroup_medians.parquet")
    grouped = {key: frame for key, frame in sub.groupby("instance", sort=False)}
    records: list[RecordSeries] = []
    for instance, frame in windows.groupby("instance", sort=True):
        frame = frame.sort_values("window_index", kind="stable")
        index = [int(v) for v in frame["window_index"]]
        labels = [int(v) for v in frame[label_col]]
        group = GROUP_ALIGNED if bed == BED_3W_ALIGNED else assign_group(labels, k)
        record = RecordSeries(
            bed=bed,
            record=str(instance),
            fold=str(frame["well"].iloc[0]),
            group=group,
            window_index=index,
            window_start=[str(v) for v in frame["window_start"]],
            labels=labels,
            contiguous=[False] + [index[p] == index[p - 1] + 1 for p in range(1, len(index))],
        )
        if group not in COUNT_ONLY_GROUPS:
            medians = grouped.get(str(instance))
            record.values = _matrices(medians, index, variables)
        records.append(record)
    counts = {
        "cache": str(cache).replace("\\", "/"),
        "manifest_sha256": sha256_file(cache / "MANIFEST.json")[:12],
        "n_instances": int(windows["instance"].nunique()),
        "n_windows": len(windows),
    }
    return records, counts


def _matrices(
    medians: pd.DataFrame | None, index: list[int], variables: list[str]
) -> dict[str, np.ndarray]:
    """(n_windows, 60) of one-minute medians per variable, NaN where absent."""
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


def load_skab(archives: Path, window_s: int, k: int) -> tuple[list[RecordSeries], dict]:
    """One record per archive directory, one minute median per 60 s window.

    The windows are the ones ``examples/studies/skab/run_skab.py`` builds:
    fixed steps from the record's first timestamp, a window labelled 1 when
    it holds at least one anomaly row.
    """
    step = pd.Timedelta(window_s, unit="s")
    records: list[RecordSeries] = []
    n_windows = 0
    for directory in sorted(p for p in archives.iterdir() if p.is_dir()):
        paths = sorted(p for p in directory.glob("*.parquet") if p.name != "labels.parquet")
        if not paths:
            continue
        folder, _, _ = directory.name.partition("__")
        frames = {
            p.stem: pd.read_parquet(p).sort_values("timestamp").reset_index(drop=True)
            for p in paths
        }
        labels_path = directory / "labels.parquet"
        labels = pd.read_parquet(labels_path) if labels_path.exists() else None
        t0 = min(f["timestamp"].iloc[0] for f in frames.values())
        t_last = max(f["timestamp"].iloc[-1] for f in frames.values())
        n = int((t_last - t0) // step) + 1
        window_labels = [0] * n
        if labels is not None:
            position = ((labels["timestamp"] - t0) // step).astype(int)
            for p in sorted(set(position[labels["anomaly"] == 1])):
                if 0 <= p < n:
                    window_labels[int(p)] = 1
        bed = BED_SKAB_FREE if labels is None else BED_SKAB
        group = GROUP_FREE if labels is None else assign_group(window_labels, k)
        if group == GROUP_MIXED:
            # A SKAB record returns to normal after its fault, so the group the
            # 3W beds call mixed is the one the SKAB study calls labelled.
            group = GROUP_LABELLED
        record = RecordSeries(
            bed=bed,
            record=directory.name,
            fold=folder,
            group=group,
            window_index=list(range(n)),
            window_start=[(t0 + i * step).isoformat() for i in range(n)],
            labels=window_labels,
            contiguous=[False] + [True] * (n - 1),
        )
        if group not in COUNT_ONLY_GROUPS:
            for tag, frame in frames.items():
                position = ((frame["timestamp"] - t0) // step).astype(int)
                values = pd.to_numeric(frame["value"], errors="coerce")
                good = np.isfinite(values) & (frame["quality"] == "GOOD")
                medians = values[good].groupby(position[good]).median()
                column = np.full((n, 1), np.nan)
                for p, value in medians.items():
                    if 0 <= int(p) < n:
                        column[int(p), 0] = float(value)
                record.values[tag] = column
        records.append(record)
        n_windows += n
    counts = {
        "archives": str(archives).replace("\\", "/"),
        "n_records": len(records),
        "n_windows": n_windows,
    }
    return records, counts


# ---------------------------------------------------------------- scoring


@dataclass
class Scored:
    """One scoring's answer for one record: a score or a refusal per position."""

    scores: dict[int, float] = field(default_factory=dict)
    refusals: dict[int, str] = field(default_factory=dict)
    usability: list[dict] = field(default_factory=list)

    def record(self, position: int, value: float | None, reason: str) -> None:
        if value is None or not math.isfinite(value):
            self.refusals[position] = reason
        else:
            self.scores[position] = float(value)


def _combine(per_variable: dict[str, np.ndarray], n: int) -> Scored:
    out = Scored()
    for position in range(n):
        candidates = [
            per_variable[v][position]
            for v in sorted(per_variable)
            if np.isfinite(per_variable[v][position])
        ]
        out.record(position, max(candidates) if candidates else None, REFUSAL_NO_MINUTE)
    return out


def _z_per_window(matrix: np.ndarray, centre: float, scale: float) -> np.ndarray:
    z = np.abs(matrix - centre) / scale
    out = np.full(matrix.shape[0], np.nan)
    finite = np.isfinite(z)
    rows = finite.any(axis=1)
    if rows.any():
        out[rows] = np.nanmax(np.where(finite, z, -np.inf)[rows], axis=1)
    return out


def score_fixed(
    record: RecordSeries, matrices: dict[str, np.ndarray], design: Design, scoring: str
) -> Scored:
    """One centre and scale per variable over the design's fit windows."""
    per_variable: dict[str, np.ndarray] = {}
    usability: list[dict] = []
    for variable in record.variables:
        matrix = matrices[variable]
        fit = matrix[list(design.fit)].ravel()
        stat, reason = mad_centre_scale(fit)
        if stat is None:
            usability.append(
                {
                    "bed": record.bed,
                    "record": record.record,
                    "variable": variable,
                    "scoring": scoring,
                    "reason": reason,
                    "n_windows_unusable": record.n_windows,
                    "n_windows_total": record.n_windows,
                }
            )
            continue
        per_variable[variable] = _z_per_window(matrix, stat[0], stat[1])
    out = _combine(per_variable, record.n_windows)
    out.usability = usability
    return out


def score_rolling(record: RecordSeries, w: int) -> Scored:
    """Centre and scale from the ``w`` windows before each scored window."""
    n = record.n_windows
    per_variable = {v: np.full(n, np.nan) for v in record.variables}
    unusable = {v: 0 for v in record.variables}
    reasons: dict[str, str] = {}
    for position in range(w, n):
        for variable in record.variables:
            matrix = record.values[variable]
            stat, reason = mad_centre_scale(matrix[position - w : position].ravel())
            if stat is None:
                unusable[variable] += 1
                reasons.setdefault(variable, reason)
                continue
            z = np.abs(matrix[position] - stat[0]) / stat[1]
            if np.isfinite(z).any():
                per_variable[variable][position] = float(np.nanmax(z))
    out = _combine(per_variable, n)
    for position in range(min(w, n)):
        out.scores.pop(position, None)
        out.refusals[position] = REFUSAL_NO_HISTORY.format(w=w)
    out.usability = [
        {
            "bed": record.bed,
            "record": record.record,
            "variable": variable,
            "scoring": SCORING_ROLLING,
            "reason": reasons[variable],
            "n_windows_unusable": unusable[variable],
            "n_windows_total": max(n - w, 0),
        }
        for variable in sorted(unusable)
        if unusable[variable]
    ]
    return out


def score_clock(record: RecordSeries) -> Scored:
    out = Scored()
    control = clock_control({p: p for p in range(record.n_windows)})
    for position, value in control.items():
        out.scores[int(position)] = float(value)
    return out


def score_record(record: RecordSeries, design: Design, w: int) -> dict[str, Scored]:
    """Every scoring for one record, keyed by scoring name."""
    out = {
        SCORING_STATIC: score_fixed(record, record.values, design, SCORING_STATIC),
        SCORING_DIFF: score_fixed(record, record.differences(), design, SCORING_DIFF),
        SCORING_CLOCK: score_clock(record),
    }
    if record.bed == BED_3W_ALIGNED:
        refused = Scored()
        for position in range(record.n_windows):
            refused.refusals[position] = REFUSAL_ROLLING_ALIGNED.format(w=w)
        out[SCORING_ROLLING] = refused
    else:
        out[SCORING_ROLLING] = score_rolling(record, w)
    return out


# ---------------------------------------------------------------- alarm rule


def fault_span(labels: list[int]) -> tuple[int | None, int | None]:
    positive = [p for p, label in enumerate(labels) if label == 1]
    return (positive[0], positive[-1]) if positive else (None, None)


def fault_fire_run(fired: dict[int, bool], labels: list[int]) -> int | None:
    """Consecutive fault windows that fire, counted from the first one that does.

    On a moving baseline this is how long a persistent fault stays above
    the threshold before the baseline has absorbed it. ``None`` when no
    fault window fires.
    """
    faults = [p for p, label in enumerate(labels) if label == 1]
    firing = [p for p in faults if fired.get(p)]
    if not firing:
        return None
    run = 0
    for position in faults[faults.index(firing[0]) :]:
        if not fired.get(position):
            break
        run += 1
    return run


def per_record_row(
    record: RecordSeries, scoring: str, scored: Scored, design: Design
) -> dict:
    n = record.n_windows
    stream = design.stream_positions(n)
    first, last = fault_span(record.labels)
    row: dict = {
        "bed": record.bed,
        "record": record.record,
        "fold": record.fold,
        "group": record.group,
        "scoring": scoring,
        "n_windows": n,
        "n_stream": len(stream),
        "n_scored": 0,
        "n_refused": 0,
        "threshold": None,
        "roc_auc": None,
        "auc_refusal": "",
        "first_fault_position": first,
        "last_fault_position": last,
        "n_pre": None,
        "n_pre_fired": None,
        "detected": None,
        "delay_windows": None,
        "n_consecutive_fault_fires": None,
        "n_post": None,
        "n_post_fired": None,
        "recovered": None,
        "refusal": "",
    }
    if n < design.min_windows:
        row["refusal"] = (
            f"record has {n} window(s); the rule needs {design.min_windows} "
            "so three threshold windows sit after the baseline"
        )
        row["auc_refusal"] = row["refusal"]
        return row
    scored_stream = [p for p in stream if p in scored.scores]
    row["n_scored"] = len(scored_stream)
    row["n_refused"] = len(stream) - len(scored_stream)
    labels = np.array([record.labels[p] for p in scored_stream])
    values = np.array([scored.scores[p] for p in scored_stream], dtype=float)
    if labels.size == 0:
        row["auc_refusal"] = "no scored stream window"
    else:
        metrics = ranking_metrics(labels, values)
        row["roc_auc"] = r4(metrics.roc_auc)
        row["auc_refusal"] = metrics.refusal or ""
    threshold_scores = [scored.scores[p] for p in design.threshold if p in scored.scores]
    if not threshold_scores:
        reasons = sorted(
            {scored.refusals.get(p, "not scored") for p in design.threshold}
        )
        row["refusal"] = "no threshold window scored: " + "; ".join(reasons)
        return row
    threshold = worst_baseline_threshold(threshold_scores)
    row["threshold"] = r4(threshold)
    fired = {p: fires(scored.scores[p], threshold) for p in scored_stream}
    if first is None:
        row["n_pre"] = len(fired)
        row["n_pre_fired"] = int(sum(fired.values()))
        return row
    pre = [p for p in fired if p < first]
    span = sorted(p for p in fired if first <= p <= last and fired[p])
    post = [p for p in fired if p > last]
    row["n_pre"] = len(pre)
    row["n_pre_fired"] = int(sum(fired[p] for p in pre))
    row["detected"] = bool(span)
    row["delay_windows"] = (span[0] - first) if span else None
    row["n_consecutive_fault_fires"] = fault_fire_run(fired, record.labels)
    row["n_post"] = len(post)
    row["n_post_fired"] = int(sum(fired[p] for p in post))
    after = [p for p in stream if p > last]
    if span and after:
        quiet = [p for p in after if p in fired and not fired[p]]
        row["recovered"] = bool(quiet) and (quiet[0] - last) <= RECOVERY_WINDOWS
    return row


# ---------------------------------------------------------------- tables


def window_rows(record: RecordSeries, design: Design) -> list[dict]:
    n = record.n_windows
    return [
        {
            "bed": record.bed,
            "record": record.record,
            "fold": record.fold,
            "group": record.group,
            "position": position,
            "window_index": record.window_index[position],
            "window_start": record.window_start[position],
            "label": record.labels[position],
            "role": design.role(position, n),
        }
        for position in range(n)
    ]


def score_rows(record: RecordSeries, scoring: str, scored: Scored, design: Design) -> list[dict]:
    n = record.n_windows
    rows = []
    for position in range(n):
        rows.append(
            {
                "bed": record.bed,
                "record": record.record,
                "scoring": scoring,
                "position": position,
                "window_index": record.window_index[position],
                "label": record.labels[position],
                "role": design.role(position, n),
                "score": r4(scored.scores.get(position)),
                "refusal": scored.refusals.get(position, ""),
            }
        )
    return rows


def _rate(num, den) -> float | None:
    return r4(num / den) if den else None


def _ranking(frame: pd.DataFrame) -> dict:
    scored = frame[frame["refusal"] == ""]
    if scored.empty:
        return {"roc_auc": None, "pr_auc": None, "refusal": "no scored stream window"}
    metrics = ranking_metrics(
        scored["label"].to_numpy(), scored["score"].to_numpy(dtype=float)
    )
    return {
        "roc_auc": r4(metrics.roc_auc),
        "pr_auc": r4(metrics.pr_auc),
        "refusal": metrics.refusal or "",
    }


def summary_rows(
    scores: pd.DataFrame, per_record: pd.DataFrame, counts: dict[tuple[str, str], int], k: int
) -> list[dict]:
    floor = far_floor(k)
    rows = []
    for bed in BEDS:
        groups = sorted(per_record.loc[per_record["bed"] == bed, "group"].unique())
        for group in groups:
            for scoring in SCORINGS:
                pr = per_record[
                    (per_record["bed"] == bed)
                    & (per_record["group"] == group)
                    & (per_record["scoring"] == scoring)
                ]
                if pr.empty:
                    continue
                sc = scores[
                    (scores["bed"] == bed)
                    & (scores["scoring"] == scoring)
                    & (scores["role"] == ROLE_STREAM)
                    & (scores["record"].isin(set(pr["record"])))
                ]
                ok = pr[pr["threshold"].notna()]
                n_pre = int(ok["n_pre"].fillna(0).sum())
                n_pre_fired = int(ok["n_pre_fired"].fillna(0).sum())
                far = _rate(n_pre_fired, n_pre)
                fired_records = int((ok["n_pre_fired"].fillna(0) > 0).sum())
                ranking = _ranking(sc)
                refusals = pr.loc[pr["refusal"] != "", "refusal"]
                row = {
                    "bed": bed,
                    "group": group,
                    "scoring": scoring,
                    "n_records": len(pr),
                    "n_records_scored": len(ok),
                    "n_records_refused": int(len(pr) - len(ok)),
                    "record_refusal": str(refusals.mode().iloc[0]) if len(refusals) else "",
                    "n_stream_windows": len(sc),
                    "n_windows_scored": int((sc["refusal"] == "").sum()),
                    "n_windows_refused": int((sc["refusal"] != "").sum()),
                    "roc_auc": ranking["roc_auc"],
                    "pr_auc": ranking["pr_auc"],
                    "auc_refusal": ranking["refusal"],
                    "n_pre_windows": n_pre,
                    "far_window": far,
                    "far_record": _rate(fired_records, len(ok)),
                    "far_floor": r4(floor),
                    "over_floor": r4(far - floor) if far is not None else None,
                    "n_detected": None,
                    "detect_rate": None,
                    "median_delay_windows": None,
                    "median_consecutive_fault_fires": None,
                    "n_post_windows": None,
                    "far_post": None,
                    "n_recovery_eligible": None,
                    "recovered_rate": None,
                }
                if ok["detected"].notna().any():
                    detected = ok["detected"].eq(True)
                    delays = ok.loc[detected, "delay_windows"].dropna()
                    runs = ok["n_consecutive_fault_fires"].dropna()
                    n_post = int(ok["n_post"].fillna(0).sum())
                    eligible = ok["recovered"].notna()
                    row.update(
                        {
                            "n_detected": int(detected.sum()),
                            "detect_rate": _rate(int(detected.sum()), len(ok)),
                            "median_delay_windows": r4(delays.median())
                            if len(delays)
                            else None,
                            "median_consecutive_fault_fires": r4(runs.median())
                            if len(runs)
                            else None,
                            "n_post_windows": n_post,
                            "far_post": _rate(
                                int(ok["n_post_fired"].fillna(0).sum()), n_post
                            ),
                            "n_recovery_eligible": int(eligible.sum()),
                            "recovered_rate": _rate(
                                int(ok.loc[eligible, "recovered"].sum()), int(eligible.sum())
                            ),
                        }
                    )
                rows.append(row)
        for group in COUNT_ONLY_GROUPS:
            n_records = counts.get((bed, group), 0)
            if not n_records:
                continue
            rows.append(
                {
                    "bed": bed,
                    "group": group,
                    "scoring": "none",
                    "n_records": n_records,
                    "n_records_scored": 0,
                    "n_records_refused": n_records,
                    "record_refusal": "a fault row lies inside the baseline windows"
                    if group == GROUP_POSITIVE_BASELINE
                    else f"record has {k} window(s) or fewer, all of them baseline",
                    "n_stream_windows": 0,
                    "n_windows_scored": 0,
                    "n_windows_refused": 0,
                    "roc_auc": None,
                    "pr_auc": None,
                    "auc_refusal": REFUSAL_ONE_CLASS
                    if group == GROUP_SHORT
                    else "a fault row lies inside the baseline windows",
                    "n_pre_windows": 0,
                    "far_window": None,
                    "far_record": None,
                    "far_floor": r4(floor),
                    "over_floor": None,
                    "n_detected": None,
                    "detect_rate": None,
                    "median_delay_windows": None,
                    "median_consecutive_fault_fires": None,
                    "n_post_windows": None,
                    "far_post": None,
                    "n_recovery_eligible": None,
                    "recovered_rate": None,
                }
            )
    return rows


# ---------------------------------------------------------------- run


@dataclass
class StudyResult:
    windows: pd.DataFrame
    scores: pd.DataFrame
    per_record: pd.DataFrame
    summary: pd.DataFrame
    usability: pd.DataFrame
    run_info: dict


def run(
    windows: Path,
    aligned: Path,
    archives: Path,
    out_dir: Path,
    *,
    k: int = DEFAULT_K,
    w: int = DEFAULT_W,
    window_s: int = DEFAULT_WINDOW_S,
    source_3w: Path | None = None,
    source_skab: Path | None = None,
) -> StudyResult:
    started = time.perf_counter()
    own, own_counts = load_3w(windows, BED_3W_OWN, k)
    align, align_counts = load_3w(aligned, BED_3W_ALIGNED, k)
    skab, skab_counts = load_skab(archives, window_s, k)
    records = own + align + skab

    window_parts: list[dict] = []
    score_parts: list[dict] = []
    record_parts: list[dict] = []
    usability_parts: list[dict] = []
    counts: dict[tuple[str, str], int] = {}
    for record in records:
        counts[(record.bed, record.group)] = counts.get((record.bed, record.group), 0) + 1
        if record.group in COUNT_ONLY_GROUPS:
            continue
        design = DESIGNS[record.bed]
        scored = score_record(record, design, w)
        keep = record.n_windows >= design.min_windows
        if keep:
            window_parts.extend(window_rows(record, design))
        for scoring in SCORINGS:
            record_parts.append(per_record_row(record, scoring, scored[scoring], design))
            if keep:
                score_parts.extend(score_rows(record, scoring, scored[scoring], design))
                usability_parts.extend(scored[scoring].usability)

    windows_out = pd.DataFrame(window_parts).sort_values(
        ["bed", "record", "position"], kind="stable"
    )
    scores_out = pd.DataFrame(score_parts).sort_values(
        ["bed", "record", "scoring", "position"], kind="stable"
    )
    per_record_out = pd.DataFrame(record_parts).sort_values(
        ["bed", "group", "record", "scoring"], kind="stable"
    )
    usability_out = pd.DataFrame(usability_parts, columns=USABILITY_COLUMNS)
    usability_out = usability_out.sort_values(
        ["bed", "record", "scoring", "variable"], kind="stable"
    )
    summary_out = pd.DataFrame(summary_rows(scores_out, per_record_out, counts, k))
    run_info = {
        "study": "baseline_drift",
        "code": {"git_head": git_head(), "tsdive_version": tsdive.__version__},
        "dataset_manifest_sha256": {
            "3w": _source_sha(source_3w),
            "skab": _source_sha(source_skab),
        },
        "beds": {
            BED_3W_OWN: own_counts,
            BED_3W_ALIGNED: align_counts,
            BED_SKAB: skab_counts,
        },
        "parameters": {
            "k_fit_windows": k,
            "w_rolling_windows": w,
            "skab_window_s": window_s,
            "far_floor": r4(far_floor(k)),
            "recovery_windows": RECOVERY_WINDOWS,
        },
        "group_counts": {
            f"{bed}:{group}": n for (bed, group), n in sorted(counts.items())
        },
        "n_rows": {
            "windows": len(windows_out),
            "scores": len(scores_out),
            "per_record": len(per_record_out),
            "summary": len(summary_out),
            "usability": len(usability_out),
        },
        "wall_seconds": round(time.perf_counter() - started, 1),
    }
    result = StudyResult(
        windows_out, scores_out, per_record_out, summary_out, usability_out, run_info
    )
    write_result(result, out_dir)
    return result


def _source_sha(source: Path | None) -> str | None:
    if source is None:
        return None
    path = source / "MANIFEST.sha256.json"
    return sha256_file(path)[:12] if path.exists() else None


def write_result(result: StudyResult, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    write_csv(result.windows, out / "windows.csv")
    write_csv(result.scores, out / "scores.csv")
    write_csv(result.per_record, out / "per_record.csv")
    write_csv(result.summary, out / "summary.csv")
    write_csv(result.usability, out / "usability.csv")
    (out / "run.json").write_text(
        json.dumps(result.run_info, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--windows", default=DEFAULT_WINDOWS)
    parser.add_argument("--aligned", default=DEFAULT_ALIGNED)
    parser.add_argument("--archives", default=DEFAULT_ARCHIVES)
    parser.add_argument("--source", default=DEFAULT_3W_SOURCE)
    parser.add_argument("--skab-source", default=DEFAULT_SKAB_SOURCE)
    parser.add_argument("--out", default=str(HERE / "results"))
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--w", type=int, default=DEFAULT_W)
    parser.add_argument("--window-s", type=int, default=DEFAULT_WINDOW_S)
    args = parser.parse_args(argv)

    result = run(
        Path(args.windows),
        Path(args.aligned),
        Path(args.archives),
        Path(args.out),
        k=args.k,
        w=args.w,
        window_s=args.window_s,
        source_3w=Path(args.source),
        source_skab=Path(args.skab_source),
    )
    print(json.dumps(result.run_info["n_rows"], sort_keys=True))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Score the detectors and the clock control on the SKAB archives.

Every record is read on its own: the first ``k`` fixed windows are its
baseline (label-blind, ``own history``), the rest is the stream, and
nothing is fitted across records. Per window and tool the script writes
a score or a refusal reason, never neither. The tools are the package's
own ``screen``, ``spc`` and ``mspc`` calls with the baseline window
string covering the first ``k`` minutes and the monitored window string
the scored minute; the clock control scores a window by its position.

Outputs under ``--out``:

- ``profile_per_tag.csv``: one row per archive from ``api.profile``.
- ``profile_summary.csv``: the profile rows summarised per tag.
- ``windows.csv``: the fixed windows with their labels.
- ``scores.csv``: one row per (tool, window), score or refusal.
- ``tag_usability.csv``: why a tag is left out of a record's score.
- ``tag_scores.csv``: the per-tag ``screen`` and ``spc`` scores behind ``scores.csv``.
- ``per_record.csv``: the alarm rule per (tool, record).
- ``summary.csv``: the pooled numbers per tool and group.
- ``conformal.csv``, ``conformal_summary.csv``, ``permutation.csv``: the
  conformal martingale bed.
- ``run.json``: provenance, parameters, counts and wall seconds.
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

import tsdive
from tsdive import analyses, api
from tsdive.baselines.provisional import mad_baseline
from tsdive.errors import TSDiveError
from tsdive.eval import (
    clock_control,
    conformal_p_values,
    far_floor,
    fires,
    martingale_alarm,
    mixture_martingale,
    ranking_metrics,
    worst_baseline_threshold,
)
from tsdive.mspc.pca import detect
from tsdive.spc.charts import apply_rules

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

UPSTREAM_COMMIT = "b2c0d46c2971dcbfe71e26087b6d231998bb91c2"
ANOMALY_FREE_FOLDER = "anomaly-free"
FOLDERS = ("valve1", "valve2", "other")

TOOL_SCREEN = "screen"
TOOL_SPC = "spc"
TOOL_T2 = "mspc_t2"
TOOL_SPE = "mspc_spe"
TOOL_CLOCK = "clock"
TOOLS = (TOOL_SCREEN, TOOL_SPC, TOOL_T2, TOOL_SPE, TOOL_CLOCK)

GROUP_LABELLED = "labelled"
GROUP_POSITIVE_BASELINE = "positive_baseline"
GROUP_ANOMALY_FREE = "anomaly_free"
GROUP_SHORT = "short"

ROLE_BASELINE = "baseline"
ROLE_STREAM = "stream"

EPSILONS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
RECOVERY_WINDOWS = 2

REFUSAL_ONE_CLASS = "record carries one class only in its scored stream windows"
REFUSAL_POSITIVE_BASELINE = "an anomaly row lies inside the baseline windows"
REFUSAL_NO_ROWS = "no rows in window"
REFUSAL_NO_TAG = "no usable tag scored this window"


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


def window_spec(start: pd.Timestamp, end: pd.Timestamp) -> str:
    return f"{start.isoformat()}/{end.isoformat()}"


def read_spec(start: pd.Timestamp, end: pd.Timestamp) -> str:
    """The ``START/END`` string for the rows in ``[start, end)``.

    Archive reads are inclusive at both ends, so the read ends one second
    before ``end``; at the 1 s step the rows are exactly the window's.
    """
    return window_spec(start, end - pd.Timedelta(1, unit="s"))


def quality_frame(values: np.ndarray) -> pd.DataFrame:
    """A value+quality frame ``mad_baseline`` accepts; the stamps are placeholders."""
    stamps = pd.date_range("2000-01-01", periods=len(values), freq="s", tz="UTC")
    return pd.DataFrame(
        {"timestamp": stamps, "value": np.asarray(values, dtype=float), "quality": "GOOD"}
    )


# ---------------------------------------------------------------- records


@dataclass
class Record:
    experiment: str
    folder: str
    name: str
    directory: Path
    tags: list[str]
    frames: dict[str, pd.DataFrame]
    labels: pd.DataFrame | None
    windows: pd.DataFrame = field(default_factory=pd.DataFrame)
    group: str = ""

    @property
    def is_anomaly_free(self) -> bool:
        return self.labels is None

    def archive(self, tag: str) -> Path:
        return self.directory / f"{tag}.parquet"

    @property
    def first_stamp(self) -> pd.Timestamp:
        return min(f["timestamp"].iloc[0] for f in self.frames.values())

    @property
    def last_stamp(self) -> pd.Timestamp:
        return max(f["timestamp"].iloc[-1] for f in self.frames.values())


def load_records(archives: Path) -> list[Record]:
    records: list[Record] = []
    for directory in sorted(p for p in archives.iterdir() if p.is_dir()):
        paths = sorted(p for p in directory.glob("*.parquet") if p.name != "labels.parquet")
        if not paths:
            continue
        folder, _, name = directory.name.partition("__")
        frames = {
            p.stem: pd.read_parquet(p).sort_values("timestamp").reset_index(drop=True)
            for p in paths
        }
        labels_path = directory / "labels.parquet"
        labels = (
            pd.read_parquet(labels_path).sort_values("timestamp").reset_index(drop=True)
            if labels_path.exists()
            else None
        )
        records.append(
            Record(
                experiment=directory.name,
                folder=folder,
                name=name,
                directory=directory,
                tags=[p.stem for p in paths],
                frames=frames,
                labels=labels,
            )
        )
    return records


# ---------------------------------------------------------------- profile


def profile_rows(record: Record, window_s: int) -> list[dict]:
    """One row per archive: the whole-record profile and the windowed flatline pass."""
    rows = []
    step = pd.Timedelta(window_s, unit="s")
    for tag in record.tags:
        path = record.archive(tag)
        prof = api.profile(path, flatline=True)
        physics = prof.physics
        stats = prof.stats
        contract = prof.window.contract
        first, last = prof.window.start, prof.window.end
        assessed = fired = 0
        cursor = first + step
        while cursor + step <= last + pd.Timedelta(1, unit="s"):
            verdict = api.profile(path, read_spec(cursor, cursor + step), flatline=True).flatline
            if verdict is not None and verdict.not_assessed is None:
                assessed += 1
                fired += int(verdict.fired)
            cursor = cursor + step
        rows.append(
            {
                "experiment": record.experiment,
                "folder": record.folder,
                "tag": tag,
                "n_samples": int(stats.features.n_samples),
                "n_good": int(stats.features.n_good),
                "first_timestamp": first.isoformat(),
                "last_timestamp": last.isoformat(),
                "calculation_basis": contract.calculation_basis.value,
                "retrieval_mode": contract.retrieval_mode.value,
                "aggregate_type": contract.aggregate_type.value,
                "stepped": bool(contract.stepped),
                "quality_assumed": bool(prof.meta.quality_assumed),
                "unit_raw": prof.meta.unit_raw,
                "declared_rate_s": stats.declared_rate_s,
                "interval_median_s": r4(stats.interval_median_s),
                "interval_p05_s": r4(stats.interval_p05_s),
                "interval_p95_s": r4(stats.interval_p95_s),
                "coverage": r4(physics.coverage.coverage),
                "n_gaps": int(physics.coverage.n_gaps),
                "longest_gap_s": r4(physics.coverage.longest_gap_s),
                "n_duplicate_timestamps": len(physics.timestamp_audit.duplicate_timestamps),
                "distinct_count": stats.distinct_count,
                "mad": r4(stats.mad),
                "min": r4(stats.features.min),
                "max": r4(stats.features.max),
                "frozen_whole_record": bool(stats.distinct_count == 1),
                "zero_mad_whole_record": bool(stats.mad is not None and stats.mad == 0.0),
                "flatline_whole_record": prof.flatline.summary_lines()[0]
                if prof.flatline is not None
                else None,
                "flatline_windows_assessed": assessed,
                "flatline_windows_fired": fired,
            }
        )
    return rows


def profile_summary(per_tag: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tag, sub in per_tag.groupby("tag", sort=True):
        rows.append(
            {
                "tag": tag,
                "n_records": len(sub),
                "n_frozen_whole_record": int(sub["frozen_whole_record"].sum()),
                "n_zero_mad_whole_record": int(sub["zero_mad_whole_record"].sum()),
                "distinct_count_min": int(sub["distinct_count"].min()),
                "distinct_count_median": r4(sub["distinct_count"].median()),
                "n_records_with_gaps": int((sub["n_gaps"] > 0).sum()),
                "longest_gap_s_max": r4(sub["longest_gap_s"].max()),
                "flatline_windows_assessed": int(sub["flatline_windows_assessed"].sum()),
                "flatline_windows_fired": int(sub["flatline_windows_fired"].sum()),
                "n_records_flatline_fired": int((sub["flatline_windows_fired"] > 0).sum()),
                "n_duplicate_timestamps": int(sub["n_duplicate_timestamps"].sum()),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- windows


def build_windows(record: Record, window_s: int, k: int) -> pd.DataFrame:
    """Fixed windows from the record's first timestamp, with the row labels."""
    step = pd.Timedelta(window_s, unit="s")
    t0 = record.first_stamp
    t_last = record.last_stamp
    n_windows = int((t_last - t0) // step) + 1
    stamps = next(iter(record.frames.values()))["timestamp"]
    labels = record.labels
    rows = []
    for index in range(n_windows):
        start = t0 + index * step
        end = start + step
        inside = (stamps >= start) & (stamps < end)
        n_rows = int(inside.sum())
        if labels is not None:
            lab_inside = (labels["timestamp"] >= start) & (labels["timestamp"] < end)
            n_anomaly = int(labels.loc[lab_inside, "anomaly"].sum())
            n_label_rows = int(lab_inside.sum())
            changepoint = bool(labels.loc[lab_inside, "changepoint"].any())
        else:
            n_anomaly = n_label_rows = 0
            changepoint = False
        rows.append(
            {
                "window_key": f"{record.experiment}:{index}",
                "experiment": record.experiment,
                "folder": record.folder,
                "window_index": index,
                "role": ROLE_BASELINE if index < k else ROLE_STREAM,
                "window_start": start.isoformat(),
                "window_end": end.isoformat(),
                "n_rows": n_rows,
                "n_anomaly_rows": n_anomaly,
                "label": int(n_anomaly > 0),
                "label_majority": int(n_label_rows > 0 and n_anomaly * 2 > n_label_rows),
                "changepoint_in_window": changepoint,
            }
        )
    return pd.DataFrame(rows)


def assign_group(record: Record, k: int) -> str:
    if record.is_anomaly_free:
        return GROUP_ANOMALY_FREE
    if len(record.windows) <= k:
        return GROUP_SHORT
    if record.windows["label"].iloc[:k].any():
        return GROUP_POSITIVE_BASELINE
    return GROUP_LABELLED


# ---------------------------------------------------------------- scoring


@dataclass
class ToolScores:
    """One tool's answer for one record: a score or a refusal per window index."""

    scores: dict[int, float] = field(default_factory=dict)
    refusals: dict[int, str] = field(default_factory=dict)

    def record(self, index: int, value: float) -> None:
        if value is not None and math.isfinite(float(value)):
            self.scores[index] = float(value)
        else:
            self.refusals[index] = "score not finite"

    def refuse(self, index: int, reason: str) -> None:
        self.refusals[index] = reason


def _bounds(windows: pd.DataFrame) -> list[tuple[int, pd.Timestamp, pd.Timestamp]]:
    return [
        (int(r.window_index), pd.Timestamp(r.window_start), pd.Timestamp(r.window_end))
        for r in windows.itertuples()
    ]


def _slice(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return frame[(frame["timestamp"] >= start) & (frame["timestamp"] < end)]


def _max_z(frame: pd.DataFrame, center: float, scale: float) -> float | None:
    good = frame[frame["valid"]] if "valid" in frame.columns else frame
    if good.empty:
        return None
    values = pd.to_numeric(good["value"], errors="coerce").dropna().to_numpy(dtype=float)
    if values.size == 0:
        return None
    return float(np.max(np.abs(values - center)) / scale)


def score_univariate(
    record: Record, k: int, k_mad: float, frozen: set[str]
) -> tuple[ToolScores, ToolScores, list[dict], list[str], list[dict]]:
    """``screen`` and ``spc`` per window.

    Returns the two tools' scores, the usability rows, the usable tags and
    one row per (tool, tag, window) with the tag's own score.
    """
    bounds = _bounds(record.windows)
    baseline = read_spec(bounds[0][1], bounds[k - 1][2])
    screen_out, spc_out = ToolScores(), ToolScores()
    per_tag_screen: dict[str, dict[int, float]] = {}
    per_tag_spc: dict[str, dict[int, int]] = {}
    per_tag_refusals: dict[int, list[str]] = {}
    usability: list[dict] = []
    usable: list[str] = []

    def unusable(tag: str, reason: str) -> None:
        usability.append(
            {"experiment": record.experiment, "tag": tag, "usable": False, "reason": reason}
        )

    for tag in record.tags:
        if tag in frozen:
            unusable(tag, "frozen for the whole record (one distinct value)")
            continue
        path = record.archive(tag)
        first_index, first_start, first_end = bounds[k]
        try:
            first = analyses.screen(path, baseline, read_spec(first_start, first_end), k=k_mad)
            first_spc = analyses.spc(path, baseline, read_spec(first_start, first_end))
        except (TSDiveError, ValueError) as e:
            unusable(tag, f"{type(e).__name__}: {e}")
            continue
        provisional = first.provisional
        if provisional is None or provisional.scale <= 0:
            unusable(tag, "baseline MAD scale is zero")
            continue
        usable.append(tag)
        usability.append(
            {"experiment": record.experiment, "tag": tag, "usable": True, "reason": ""}
        )
        z_scores: dict[int, float] = {}
        hit_counts: dict[int, int] = {}
        base_frame = first.baseline.frame
        for index, start, end in bounds[:k]:
            part = _slice(base_frame, start, end)
            z = _max_z(part, provisional.center, provisional.scale)
            if z is None:
                per_tag_refusals.setdefault(index, []).append(f"{tag}: {REFUSAL_NO_ROWS}")
                continue
            z_scores[index] = z
            good = part[part["valid"]]
            hit_counts[index] = len(
                apply_rules(good["timestamp"], good["value"].astype(float), first_spc.limits)
            )
        for index, start, end in bounds[k:]:
            spec = read_spec(start, end)
            try:
                if index == first_index:
                    sc, sp = first, first_spc
                else:
                    sc = analyses.screen(path, baseline, spec, k=k_mad)
                    sp = analyses.spc(path, baseline, spec)
            except (TSDiveError, ValueError) as e:
                per_tag_refusals.setdefault(index, []).append(f"{tag}: {type(e).__name__}: {e}")
                continue
            z = _max_z(sc.monitor.frame, provisional.center, provisional.scale)
            if z is None or sp.n_monitored == 0:
                per_tag_refusals.setdefault(index, []).append(f"{tag}: {REFUSAL_NO_ROWS}")
                continue
            z_scores[index] = z
            hit_counts[index] = len(sp.hits)
        per_tag_screen[tag] = z_scores
        per_tag_spc[tag] = hit_counts

    tag_rows = [
        {
            "tool": tool,
            "experiment": record.experiment,
            "tag": tag,
            "window_index": index,
            "score": r4(value),
        }
        for tool, per_tag in ((TOOL_SCREEN, per_tag_screen), (TOOL_SPC, per_tag_spc))
        for tag in usable
        for index, value in sorted(per_tag[tag].items())
    ]
    for index, _, _ in bounds:
        zs = [per_tag_screen[t][index] for t in usable if index in per_tag_screen[t]]
        hits = [per_tag_spc[t][index] for t in usable if index in per_tag_spc[t]]
        reasons = "; ".join(sorted(set(per_tag_refusals.get(index, [])))) or REFUSAL_NO_TAG
        if zs:
            screen_out.record(index, max(zs))
            spc_out.record(index, float(sum(hits)))
        else:
            screen_out.refuse(index, reasons)
            spc_out.refuse(index, reasons)
    return screen_out, spc_out, usability, usable, tag_rows


def score_multivariate(
    record: Record, k: int, usable: list[str], rate_s: int
) -> tuple[ToolScores, ToolScores]:
    """``mspc`` per stream window over the usable tags; T2 and SPE window means."""
    bounds = _bounds(record.windows)
    baseline = read_spec(bounds[0][1], bounds[k - 1][2])
    t2_out, spe_out = ToolScores(), ToolScores()
    if len(usable) < 2:
        reason = f"{len(usable)} usable tag(s); mspc needs at least two"
        for index, _, _ in bounds:
            t2_out.refuse(index, reason)
            spe_out.refuse(index, reason)
        return t2_out, spe_out
    paths = [record.archive(t) for t in usable]
    train_found = None
    for index, start, end in bounds[k:]:
        try:
            m = analyses.mspc(paths, baseline, read_spec(start, end), rate_s=rate_s)
        except (TSDiveError, ValueError) as e:
            t2_out.refuse(index, f"{type(e).__name__}: {e}")
            spe_out.refuse(index, f"{type(e).__name__}: {e}")
            continue
        if train_found is None:
            train_found = detect(m.model, m.train)
        if m.found.t2.size == 0:
            t2_out.refuse(index, REFUSAL_NO_ROWS)
            spe_out.refuse(index, REFUSAL_NO_ROWS)
            continue
        t2_out.record(index, float(np.mean(m.found.t2)))
        spe_out.record(index, float(np.mean(m.found.spe)))
    if train_found is None:
        reason = "every stream window refused, so no fitted model scores the baseline"
        for index, _, _ in bounds[:k]:
            t2_out.refuse(index, reason)
            spe_out.refuse(index, reason)
        return t2_out, spe_out
    stamps = train_found.timestamps
    for index, start, end in bounds[:k]:
        inside = (stamps >= start) & (stamps < end)
        if not inside.any():
            t2_out.refuse(index, REFUSAL_NO_ROWS)
            spe_out.refuse(index, REFUSAL_NO_ROWS)
            continue
        t2_out.record(index, float(np.mean(train_found.t2[inside])))
        spe_out.record(index, float(np.mean(train_found.spe[inside])))
    return t2_out, spe_out


def score_clock(record: Record) -> ToolScores:
    """The clock control: the window's position in its own record, nothing read."""
    out = ToolScores()
    positions = {int(i): int(i) for i in record.windows["window_index"]}
    for index, value in clock_control(positions).items():
        out.record(index, value)
    return out


def score_rows(record: Record, tool: str, scores: ToolScores) -> list[dict]:
    rows = []
    for r in record.windows.itertuples():
        index = int(r.window_index)
        rows.append(
            {
                "tool": tool,
                "experiment": record.experiment,
                "folder": record.folder,
                "group": record.group,
                "window_key": r.window_key,
                "window_index": index,
                "role": r.role,
                "label": int(r.label),
                "label_majority": int(r.label_majority),
                "score": r4(scores.scores.get(index)),
                "refusal": scores.refusals.get(index, ""),
            }
        )
    return rows


# ---------------------------------------------------------------- alarm rule


def anomaly_span(windows: pd.DataFrame) -> tuple[int | None, int | None]:
    positives = windows.loc[windows["label"] == 1, "window_index"]
    if positives.empty:
        return None, None
    return int(positives.min()), int(positives.max())


def per_record_row(record: Record, tool: str, scores: ToolScores, k: int) -> dict:
    """The alarm rule for one (tool, record); refusals carry a reason."""
    w = record.windows
    stream = w[w["role"] == ROLE_STREAM]
    scored = stream[stream["window_index"].isin(scores.scores)]
    row: dict = {
        "tool": tool,
        "experiment": record.experiment,
        "folder": record.folder,
        "group": record.group,
        "n_windows": len(w),
        "n_stream": len(stream),
        "n_scored": len(scored),
        "n_refused": int(len(stream) - len(scored)),
        "threshold": None,
        "roc_auc": None,
        "auc_refusal": "",
        "first_anomaly_window": None,
        "last_anomaly_window": None,
        "n_pre": None,
        "n_pre_fired": None,
        "detected": None,
        "delay_windows": None,
        "n_post": None,
        "n_post_fired": None,
        "recovered": None,
        "refusal": "",
    }
    first, last = anomaly_span(w)
    row["first_anomaly_window"], row["last_anomaly_window"] = first, last
    if record.group in (GROUP_SHORT, GROUP_POSITIVE_BASELINE):
        row["refusal"] = (
            REFUSAL_POSITIVE_BASELINE
            if record.group == GROUP_POSITIVE_BASELINE
            else f"record has {len(w)} window(s), none left after the {k} baseline windows"
        )
        row["auc_refusal"] = row["refusal"]
        return row
    if record.group == GROUP_LABELLED:
        y = scored["label"].to_numpy()
        s = np.array([scores.scores[int(i)] for i in scored["window_index"]])
        if len(y) == 0:
            row["auc_refusal"] = "no scored stream window"
        else:
            metrics = ranking_metrics(y, s)
            row["roc_auc"] = r4(metrics.roc_auc)
            row["auc_refusal"] = metrics.refusal or ""
    else:
        row["auc_refusal"] = REFUSAL_ONE_CLASS
    baseline_refused = [
        f"window {i}: {scores.refusals[i]}" for i in range(k) if i in scores.refusals
    ]
    if baseline_refused:
        row["refusal"] = "baseline window refused: " + "; ".join(baseline_refused)
        return row
    threshold = worst_baseline_threshold([scores.scores[i] for i in range(k)])
    row["threshold"] = r4(threshold)
    fired = {int(i): fires(scores.scores[int(i)], threshold) for i in scored["window_index"]}
    if record.group == GROUP_ANOMALY_FREE or first is None:
        row["n_pre"] = len(fired)
        row["n_pre_fired"] = int(sum(fired.values()))
        return row
    pre = [i for i in fired if i < first]
    span = [i for i in fired if first <= i <= last]
    post = [i for i in fired if i > last]
    row["n_pre"], row["n_pre_fired"] = len(pre), int(sum(fired[i] for i in pre))
    span_fired = sorted(i for i in span if fired[i])
    row["detected"] = bool(span_fired)
    row["delay_windows"] = (span_fired[0] - first) if span_fired else None
    row["n_post"], row["n_post_fired"] = len(post), int(sum(fired[i] for i in post))
    after = sorted(int(i) for i in stream["window_index"] if int(i) > last)
    if span_fired and after:
        quiet = [i for i in after if i in fired and not fired[i]]
        row["recovered"] = bool(quiet) and (quiet[0] - last) <= RECOVERY_WINDOWS
    return row


# ---------------------------------------------------------------- summary


def _ranking(frame: pd.DataFrame, label_col: str) -> dict:
    if frame.empty:
        return {"roc_auc": None, "pr_auc": None, "refusal": "no scored window"}
    m = ranking_metrics(frame[label_col].to_numpy(), frame["score"].to_numpy(dtype=float))
    return {"roc_auc": r4(m.roc_auc), "pr_auc": r4(m.pr_auc), "refusal": m.refusal or ""}


def _rate(num: int, den: int) -> float | None:
    return r4(num / den) if den else None


def summary_rows(scores: pd.DataFrame, per_record: pd.DataFrame, k: int) -> list[dict]:
    rows = []
    floor = far_floor(k)
    for tool in TOOLS:
        sc = scores[scores["tool"] == tool]
        pr = per_record[per_record["tool"] == tool]
        for group in (GROUP_LABELLED, GROUP_ANOMALY_FREE, GROUP_POSITIVE_BASELINE):
            g_sc = sc[(sc["group"] == group) & (sc["role"] == ROLE_STREAM)]
            g_pr = pr[pr["group"] == group]
            if g_pr.empty:
                continue
            scored = g_sc[g_sc["refusal"] == ""]
            row: dict = {
                "tool": tool,
                "group": group,
                "n_records": len(g_pr),
                "n_stream_windows": len(g_sc),
                "n_windows_scored": len(scored),
                "n_windows_refused": int(len(g_sc) - len(scored)),
                "roc_auc": None,
                "pr_auc": None,
                "roc_auc_majority": None,
                "auc_refusal": "",
                "n_records_auc": int(g_pr["roc_auc"].notna().sum()),
                "record_auc_median": r4(g_pr["roc_auc"].median())
                if g_pr["roc_auc"].notna().any()
                else None,
                "n_records_auc_refused": int((g_pr["auc_refusal"] != "").sum()),
                "far_floor": r4(floor),
            }
            for folder in FOLDERS:
                fold = _ranking(scored[scored["folder"] == folder], "label")
                row[f"roc_auc_{folder}"] = fold["roc_auc"]
                row[f"roc_auc_{folder}_refusal"] = fold["refusal"]
            if group == GROUP_LABELLED:
                pooled = _ranking(scored, "label")
                row["roc_auc"], row["pr_auc"] = pooled["roc_auc"], pooled["pr_auc"]
                row["auc_refusal"] = pooled["refusal"]
                row["roc_auc_majority"] = _ranking(scored, "label_majority")["roc_auc"]
            else:
                row["auc_refusal"] = (
                    REFUSAL_ONE_CLASS if group == GROUP_ANOMALY_FREE else REFUSAL_POSITIVE_BASELINE
                )
            with_threshold = g_pr[g_pr["threshold"].notna()]
            n_pre = int(with_threshold["n_pre"].fillna(0).sum())
            n_pre_fired = int(with_threshold["n_pre_fired"].fillna(0).sum())
            far = _rate(n_pre_fired, n_pre)
            row.update(
                {
                    "n_records_with_threshold": len(with_threshold),
                    "n_records_no_threshold": int(len(g_pr) - len(with_threshold)),
                    "n_pre_windows": n_pre,
                    "far_pre": far,
                    "over_floor": r4(far - floor) if far is not None else None,
                }
            )
            if group == GROUP_LABELLED:
                detected = with_threshold["detected"].eq(True)
                delays = with_threshold.loc[detected, "delay_windows"].dropna()
                n_post = int(with_threshold["n_post"].fillna(0).sum())
                n_post_fired = int(with_threshold["n_post_fired"].fillna(0).sum())
                eligible = with_threshold["recovered"].notna()
                row.update(
                    {
                        "n_detected": int(detected.sum()),
                        "detect_rate": _rate(int(detected.sum()), len(with_threshold)),
                        "median_delay_windows": r4(delays.median()) if len(delays) else None,
                        "n_post_windows": n_post,
                        "far_post": _rate(n_post_fired, n_post),
                        "n_recovery_eligible": int(eligible.sum()),
                        "n_recovered": int(with_threshold.loc[eligible, "recovered"].sum()),
                        "recovered_rate": _rate(
                            int(with_threshold.loc[eligible, "recovered"].sum()),
                            int(eligible.sum()),
                        ),
                    }
                )
            else:
                row.update(
                    {
                        "n_detected": None,
                        "detect_rate": None,
                        "median_delay_windows": None,
                        "n_post_windows": None,
                        "far_post": None,
                        "n_recovery_eligible": None,
                        "n_recovered": None,
                        "recovered_rate": None,
                    }
                )
            rows.append(row)
    return rows


# ---------------------------------------------------------------- conformal


OUTCOME_NONE = "no alarm"
OUTCOME_FALSE_PRE = "false alarm before onset"
OUTCOME_DETECT = "alarm inside the anomaly span"
OUTCOME_AFTER = "alarm after recovery"
OUTCOME_FALSE_FREE = "false alarm (anomaly-free record)"


def martingale_log(calibration: np.ndarray, stream: np.ndarray) -> np.ndarray:
    return mixture_martingale(conformal_p_values(calibration, stream, online=True), EPSILONS)


def conformal_record(
    record: Record, k: int, deltas: tuple[float, ...], seeds: int
) -> tuple[list[dict], list[dict]]:
    """The conformal bed on one record: rows per delta, and permutation alarm flags."""
    bounds = _bounds(record.windows)
    if len(bounds) <= k:
        return [], []
    fit_start, fit_end = bounds[0][1], bounds[k - 2][2]
    cal_start, cal_end = bounds[k - 1][1], bounds[k - 1][2]
    stream_start = bounds[k][1]
    fits: dict[str, tuple[float, float]] = {}
    unusable: dict[str, str] = {}
    for tag in record.tags:
        frame = record.frames[tag]
        values = _slice(frame, fit_start, fit_end)["value"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        try:
            base = mad_baseline(quality_frame(values))
        except TSDiveError as e:
            unusable[tag] = f"{type(e).__name__}: {e}"
            continue
        if base.scale <= 0:
            unusable[tag] = "MAD over the fit windows is zero"
            continue
        fits[tag] = (float(base.center), float(base.scale))
    base_row = {
        "experiment": record.experiment,
        "folder": record.folder,
        "group": record.group,
        "n_usable_tags": len(fits),
        "unusable_tags": "; ".join(f"{t}: {r}" for t, r in sorted(unusable.items())),
    }
    if not fits:
        return [
            {**base_row, "delta": d, "refusal": "no usable tag over the fit windows"}
            for d in deltas
        ], []
    usable = sorted(fits)
    stamps = record.frames[usable[0]]["timestamp"]

    def nonconformity(start: pd.Timestamp, end: pd.Timestamp) -> tuple[np.ndarray, pd.Series]:
        inside = (stamps >= start) & (stamps < end)
        columns = []
        for tag in usable:
            v = record.frames[tag].loc[inside, "value"].to_numpy(dtype=float)
            c, s = fits[tag]
            columns.append(np.abs(v - c) / s)
        z = np.stack(columns, axis=1) if columns else np.empty((0, 0))
        return np.nanmax(z, axis=1) if z.size else np.empty(0), stamps[inside]

    cal, _ = nonconformity(cal_start, cal_end)
    stream, stream_stamps = nonconformity(stream_start, record.last_stamp + pd.Timedelta(1, "s"))
    finite = np.isfinite(stream)
    stream, stream_stamps = stream[finite], stream_stamps[finite].reset_index(drop=True)
    if cal.size == 0 or not np.isfinite(cal).all() or stream.size == 0:
        return [
            {**base_row, "delta": d, "refusal": "empty or non-finite calibration or stream"}
            for d in deltas
        ], []
    first_anomaly = last_anomaly = None
    if record.labels is not None:
        lab = record.labels.set_index("timestamp")["anomaly"].reindex(stream_stamps).fillna(0)
        anomalous = np.flatnonzero(lab.to_numpy() > 0)
        if anomalous.size:
            first_anomaly, last_anomaly = int(anomalous[0]), int(anomalous[-1])
    logs = martingale_log(cal, stream)
    rows = []
    for delta in deltas:
        alarm = martingale_alarm(logs, delta)
        row = {
            **base_row,
            "delta": delta,
            "refusal": "",
            "n_fit_rows": len(_slice(record.frames[usable[0]], fit_start, fit_end)),
            "n_calibration": int(cal.size),
            "n_stream": int(stream.size),
            "first_anomaly_stream_index": first_anomaly,
            "last_anomaly_stream_index": last_anomaly,
            "alarm_index": alarm,
            "alarm_time": stream_stamps.iloc[alarm].isoformat() if alarm is not None else None,
            "delay_s": None,
        }
        if alarm is None:
            row["outcome"] = OUTCOME_NONE
        elif record.group == GROUP_ANOMALY_FREE or first_anomaly is None:
            row["outcome"] = OUTCOME_FALSE_FREE if record.is_anomaly_free else OUTCOME_FALSE_PRE
        elif alarm < first_anomaly:
            row["outcome"] = OUTCOME_FALSE_PRE
        elif alarm <= last_anomaly:
            row["outcome"] = OUTCOME_DETECT
            row["delay_s"] = r4(
                (stream_stamps.iloc[alarm] - stream_stamps.iloc[first_anomaly]).total_seconds()
            )
        else:
            row["outcome"] = OUTCOME_AFTER
        rows.append(row)
    pooled = np.concatenate([cal, stream])
    permutation = []
    for seed in range(seeds):
        shuffled = np.random.default_rng(seed).permutation(pooled)
        p_logs = martingale_log(shuffled[: cal.size], shuffled[cal.size :])
        for delta in deltas:
            permutation.append(
                {
                    "experiment": record.experiment,
                    "group": record.group,
                    "delta": delta,
                    "seed": seed,
                    "alarm": martingale_alarm(p_logs, delta) is not None,
                }
            )
    return rows, permutation


def conformal_summary(conformal: pd.DataFrame, permutation: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if conformal.empty:
        return pd.DataFrame()
    for (group, delta), sub in conformal.groupby(["group", "delta"], sort=True):
        scored = sub[sub["refusal"] == ""]
        n = len(scored)
        counts = scored["outcome"].value_counts() if n else pd.Series(dtype=int)
        perm = permutation[(permutation["group"] == group) & (permutation["delta"] == delta)]
        per_seed = perm.groupby("seed")["alarm"].mean() if len(perm) else pd.Series(dtype=float)
        rows.append(
            {
                "group": group,
                "delta": delta,
                "n_records": len(sub),
                "n_scored": n,
                "n_refused": int(len(sub) - n),
                "n_no_alarm": int(counts.get(OUTCOME_NONE, 0)),
                "n_false_before_onset": int(counts.get(OUTCOME_FALSE_PRE, 0)),
                "n_detect": int(counts.get(OUTCOME_DETECT, 0)),
                "n_after_recovery": int(counts.get(OUTCOME_AFTER, 0)),
                "n_false_anomaly_free": int(counts.get(OUTCOME_FALSE_FREE, 0)),
                "far_before_onset": _rate(int(counts.get(OUTCOME_FALSE_PRE, 0)), n),
                "detect_rate": _rate(int(counts.get(OUTCOME_DETECT, 0)), n),
                "after_recovery_rate": _rate(int(counts.get(OUTCOME_AFTER, 0)), n),
                "false_anomaly_free_rate": _rate(int(counts.get(OUTCOME_FALSE_FREE, 0)), n),
                "median_delay_s": r4(scored["delay_s"].dropna().median())
                if scored["delay_s"].notna().any()
                else None,
                "permutation_alarm_fraction": r4(per_seed.mean()) if len(per_seed) else None,
                "permutation_seeds": len(per_seed),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- study


@dataclass
class StudyResult:
    profile_per_tag: pd.DataFrame
    profile_summary: pd.DataFrame
    windows: pd.DataFrame
    scores: pd.DataFrame
    tag_usability: pd.DataFrame
    tag_scores: pd.DataFrame
    per_record: pd.DataFrame
    summary: pd.DataFrame
    conformal: pd.DataFrame
    conformal_summary: pd.DataFrame
    permutation: pd.DataFrame
    counts: dict


def run_study(
    archives: Path,
    *,
    window_s: int = 60,
    k: int = 3,
    k_mad: float = 3.0,
    rate_s: int = 1,
    deltas: tuple[float, ...] = (0.05, 0.01),
    seeds: int = 10,
) -> StudyResult:
    records = load_records(archives)
    profile_frame = pd.DataFrame(
        [row for record in records for row in profile_rows(record, window_s)]
    )
    frozen_by_record: dict[str, set[str]] = {}
    for r in profile_frame.itertuples():
        if r.frozen_whole_record:
            frozen_by_record.setdefault(r.experiment, set()).add(r.tag)

    all_windows, all_scores, usability, per_record, tag_scores = [], [], [], [], []
    conformal_rows, permutation_rows = [], []
    for record in records:
        record.windows = build_windows(record, window_s, k)
        record.group = assign_group(record, k)
        all_windows.append(record.windows.assign(group=record.group))
        if record.group == GROUP_SHORT:
            reason = f"record has {len(record.windows)} window(s); the first {k} are the baseline"
            tools = {tool: ToolScores() for tool in TOOLS}
            for tool_scores in tools.values():
                for index in record.windows["window_index"]:
                    tool_scores.refuse(int(index), reason)
        else:
            frozen = frozen_by_record.get(record.experiment, set())
            screen_out, spc_out, rows, usable, tag_rows = score_univariate(record, k, k_mad, frozen)
            usability.extend(rows)
            tag_scores.extend(tag_rows)
            t2_out, spe_out = score_multivariate(record, k, usable, rate_s)
            tools = {
                TOOL_SCREEN: screen_out,
                TOOL_SPC: spc_out,
                TOOL_T2: t2_out,
                TOOL_SPE: spe_out,
                TOOL_CLOCK: score_clock(record),
            }
            c_rows, p_rows = conformal_record(record, k, deltas, seeds)
            conformal_rows.extend(c_rows)
            permutation_rows.extend(p_rows)
        for tool in TOOLS:
            all_scores.extend(score_rows(record, tool, tools[tool]))
            per_record.append(per_record_row(record, tool, tools[tool], k))

    windows = pd.concat(all_windows, ignore_index=True)
    scores = pd.DataFrame(all_scores).sort_values(["tool", "experiment", "window_index"])
    scores = scores.reset_index(drop=True)
    per_record_frame = pd.DataFrame(per_record).sort_values(["tool", "experiment"])
    per_record_frame = per_record_frame.reset_index(drop=True)
    summary = pd.DataFrame(summary_rows(scores, per_record_frame, k))
    conformal = pd.DataFrame(conformal_rows)
    if not conformal.empty:
        conformal = conformal.sort_values(["experiment", "delta"]).reset_index(drop=True)
    permutation = pd.DataFrame(permutation_rows)
    if not permutation.empty:
        permutation = permutation.sort_values(["experiment", "delta", "seed"]).reset_index(
            drop=True
        )
    counts = {
        "n_records": len(records),
        "n_windows": len(windows),
        "groups": {
            g: int(n)
            for g, n in windows.drop_duplicates("experiment")["group"].value_counts().items()
        },
        "windows_refused_per_tool": {
            tool: int((scores[(scores["tool"] == tool)]["refusal"] != "").sum()) for tool in TOOLS
        },
        "unusable_tags": int(sum(1 for r in usability if not r["usable"])),
    }
    return StudyResult(
        profile_per_tag=profile_frame,
        profile_summary=profile_summary(profile_frame),
        windows=windows,
        scores=scores,
        tag_usability=pd.DataFrame(usability),
        tag_scores=pd.DataFrame(tag_scores),
        per_record=per_record_frame,
        summary=summary,
        conformal=conformal,
        conformal_summary=conformal_summary(conformal, permutation),
        permutation=permutation,
        counts=counts,
    )


def write_result(result: StudyResult, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    files = {
        "profile_per_tag.csv": result.profile_per_tag,
        "profile_summary.csv": result.profile_summary,
        "windows.csv": result.windows,
        "scores.csv": result.scores,
        "tag_usability.csv": result.tag_usability,
        "tag_scores.csv": result.tag_scores,
        "per_record.csv": result.per_record,
        "summary.csv": result.summary,
        "conformal.csv": result.conformal,
        "conformal_summary.csv": result.conformal_summary,
        "permutation.csv": result.permutation,
    }
    for name, frame in files.items():
        frame.to_csv(out / name, index=False, lineterminator="\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--archives", default="data/skab_archives")
    parser.add_argument("--source", default="data/skab")
    parser.add_argument("--out", default=str(HERE / "results"))
    parser.add_argument("--window-s", type=int, default=60)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--k-mad", type=float, default=3.0)
    parser.add_argument("--rate-s", type=int, default=1)
    parser.add_argument("--deltas", type=float, nargs="+", default=[0.05, 0.01])
    parser.add_argument("--seeds", type=int, default=10)
    args = parser.parse_args(argv)

    started = time.perf_counter()
    result = run_study(
        Path(args.archives),
        window_s=args.window_s,
        k=args.k,
        k_mad=args.k_mad,
        rate_s=args.rate_s,
        deltas=tuple(args.deltas),
        seeds=args.seeds,
    )
    out = Path(args.out)
    write_result(result, out)
    manifest = Path(args.source) / "MANIFEST.sha256.json"
    fetch = Path(args.source) / "FETCH.json"
    upstream = (
        json.loads(fetch.read_text("utf-8")).get("commit") if fetch.exists() else UPSTREAM_COMMIT
    )
    run = {
        "dataset": "SKAB",
        "upstream_repo": "waico/SKAB",
        "upstream_commit": upstream,
        "dataset_manifest_sha256": sha256_file(manifest) if manifest.exists() else None,
        "code": {"git_head": git_head(), "tsdive_version": tsdive.__version__},
        "parameters": {
            "window_s": args.window_s,
            "k_baseline_windows": args.k,
            "k_mad": args.k_mad,
            "rate_s": args.rate_s,
            "deltas": list(args.deltas),
            "permutation_seeds": args.seeds,
            "epsilons": list(EPSILONS),
            "recovery_windows": RECOVERY_WINDOWS,
        },
        "design": "own history: first k windows per record, label-blind; nothing fitted "
        "across records; folds are the upstream folders",
        "counts": result.counts,
        "wall_seconds": round(time.perf_counter() - started, 1),
    }
    (out / "run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"wrote {out} in {run['wall_seconds']} s; counts {json.dumps(result.counts)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

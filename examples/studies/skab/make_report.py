"""Report, figure and BENCHMARKS.md section for the SKAB study.

Reads ``results/`` as ``run_skab.py`` wrote it and renders ``REPORT.md``,
``out/01_score_trace.png`` and ``results/benchmarks_section.md``. With
``--update-benchmarks`` the section replaces the ``## REAL: SKAB``
section of the repository's ``BENCHMARKS.md``; the SYNTHETIC table and
the other REAL sections are left as they are. Every number in the prose
is read from the same frames as the tables.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))

import run_skab as rs  # noqa: E402

BENCH_MD = ROOT / "BENCHMARKS.md"
SECTION_HEADING = "## REAL: SKAB"
FIGURE_RECORD = "valve1__0"
FIGURE_NAME = "01_score_trace.png"

TOOL_LABELS = {
    rs.TOOL_SCREEN: "MAD screen, largest robust z over the usable tags",
    rs.TOOL_SPC: "SPC individuals rules, hit count over the usable tags",
    rs.TOOL_T2: "MSPC Hotelling T2, window mean",
    rs.TOOL_SPE: "MSPC SPE, window mean",
    rs.TOOL_CLOCK: "clock control, window position in the record, no sensor read",
}
SHORT_LABELS = {
    rs.TOOL_SCREEN: "MAD screen",
    rs.TOOL_SPC: "SPC rules",
    rs.TOOL_T2: "MSPC T2",
    rs.TOOL_SPE: "MSPC SPE",
    rs.TOOL_CLOCK: "clock control",
}
STAGES = {
    rs.TOOL_SCREEN: "3",
    rs.TOOL_SPC: "5",
    rs.TOOL_T2: "6",
    rs.TOOL_SPE: "6",
    rs.TOOL_CLOCK: "none",
}
COLORS = {
    rs.TOOL_SCREEN: "#1f77b4",
    rs.TOOL_SPC: "#2ca02c",
    rs.TOOL_T2: "#d62728",
    rs.TOOL_SPE: "#9467bd",
}


# ---------------------------------------------------------------- loading


def load(results: Path) -> dict:
    def csv(name: str) -> pd.DataFrame:
        path = results / name
        if path.stat().st_size <= 1:
            return pd.DataFrame()
        return pd.read_csv(path, keep_default_na=True)

    data = {name[:-4]: csv(name) for name in (
        "profile_per_tag.csv", "profile_summary.csv", "windows.csv", "scores.csv",
        "tag_usability.csv", "tag_scores.csv", "per_record.csv", "summary.csv",
        "conformal.csv", "conformal_summary.csv", "permutation.csv",
        "drift.csv", "window_max_tag.csv", "stream_layout.csv", "score_by_phase.csv",
    )}
    for name in ("scores", "per_record", "tag_usability"):
        frame = data[name]
        if "refusal" in frame.columns:
            frame["refusal"] = frame["refusal"].fillna("")
        if "auc_refusal" in frame.columns:
            frame["auc_refusal"] = frame["auc_refusal"].fillna("")
        if "reason" in frame.columns:
            frame["reason"] = frame["reason"].fillna("")
    data["run"] = json.loads((results / "run.json").read_text("utf-8"))
    return data


# ---------------------------------------------------------------- helpers


def table(header: list[str], rows: list[tuple[str, ...]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def fmt(value, nd: int = 3) -> str:
    if value is None or pd.isna(value):
        return "refused"
    return f"{float(value):.{nd}f}"


def pct(value) -> str:
    if value is None or pd.isna(value):
        return "refused"
    return f"{100 * float(value):.1f}%"


def signed(value, nd: int = 3) -> str:
    if value is None or pd.isna(value):
        return "refused"
    return f"{float(value):+.{nd}f}"


def num(value) -> str:
    if value is None or pd.isna(value):
        return "refused"
    return f"{int(value):,d}"


def code_cite(run: dict) -> str:
    code = run["code"]
    return f"tsdive {code['tsdive_version']} @ `{code['git_head'][:8]}`"


def dataset_cite(run: dict) -> str:
    return f"SKAB `{run['dataset_manifest_sha256'][:12]}`"


def summary_row(summary: pd.DataFrame, tool: str, group: str) -> pd.Series | None:
    sub = summary[(summary["tool"] == tool) & (summary["group"] == group)]
    return None if sub.empty else sub.iloc[0]


def refusal_class(reason: str) -> str:
    """The error class or the first clause of a window refusal string."""
    first = reason.split(";")[0]
    if ": " in first:
        head, _, rest = first.partition(": ")
        if head in rs.TOOLS or (head[0].isupper() and "Error" in head):
            return head if "Error" in head else refusal_class(rest)
        if "Error" in rest.split(":")[0]:
            return rest.split(":")[0]
        # "<tag>: <reason>" names one tag of many; the reason is the cause.
        return rest.strip()
    return first.strip()


# ---------------------------------------------------------------- tables


def profile_table(summary: pd.DataFrame) -> str:
    rows = [
        (
            f"`{r.tag}`",
            num(r.n_records),
            num(r.n_frozen_whole_record),
            num(r.n_zero_mad_whole_record),
            f"{num(r.distinct_count_min)} / {fmt(r.distinct_count_median, 0)}",
            num(r.n_records_with_gaps),
            fmt(r.longest_gap_s_max, 0),
            f"{num(r.flatline_windows_fired)} / {num(r.flatline_windows_assessed)}",
        )
        for r in summary.itertuples()
    ]
    return table(
        [
            "tag", "records", "frozen whole record", "MAD 0 whole record",
            "distinct values, min / median", "records with a gap", "longest gap (s)",
            "flatline windows fired / assessed",
        ],
        rows,
    )


def gap_table(per_tag: pd.DataFrame) -> str:
    one = per_tag[per_tag["tag"] == per_tag["tag"].iloc[0]]
    gaps = one[one["longest_gap_s"].fillna(0) > 0].sort_values("longest_gap_s", ascending=False)
    rows = [
        (
            f"`{r.experiment}`",
            num(r.n_samples),
            num(r.n_gaps),
            fmt(r.longest_gap_s, 0),
            fmt(r.coverage),
        )
        for r in gaps.itertuples()
    ]
    return table(["record", "rows", "gaps", "longest gap (s)", "coverage"], rows)


def usability_table(usability: pd.DataFrame) -> str:
    unusable = usability[~usability["usable"].astype(bool)]
    rows = []
    for tag, sub in unusable.groupby("tag", sort=True):
        reasons = "; ".join(
            f"{n} {reason}" for reason, n in sub["reason"].value_counts().sort_index().items()
        )
        rows.append((f"`{tag}`", num(len(sub)), reasons))
    return table(["tag", "records where unusable", "reason (records)"], rows)


def ranking_table(summary: pd.DataFrame) -> str:
    rows = []
    for tool in rs.TOOLS:
        r = summary_row(summary, tool, rs.GROUP_LABELLED)
        if r is None:
            continue
        rows.append(
            (
                SHORT_LABELS[tool],
                f"{num(r.n_windows_scored)} / {num(r.n_windows_refused)}",
                fmt(r.roc_auc),
                fmt(r.pr_auc),
                fmt(r.roc_auc_majority),
                fmt(r.roc_auc_valve1),
                fmt(r.roc_auc_valve2),
                fmt(r.roc_auc_other),
                f"{fmt(r.record_auc_median)} ({num(r.n_records_auc)})",
            )
        )
    return table(
        [
            "tool", "stream windows scored / refused", "ROC-AUC", "PR-AUC",
            "ROC-AUC, majority label", "AUC valve1", "AUC valve2", "AUC other",
            "per-record AUC median (records)",
        ],
        rows,
    )


def alarm_table(summary: pd.DataFrame) -> str:
    rows = []
    for tool in rs.TOOLS:
        r = summary_row(summary, tool, rs.GROUP_LABELLED)
        if r is None:
            continue
        rows.append(
            (
                SHORT_LABELS[tool],
                f"{num(r.n_records_with_threshold)} / {num(r.n_records_no_threshold)}",
                f"{pct(r.far_pre)} ({signed(r.over_floor)} over the {pct(r.far_floor)} floor, "
                f"{num(r.n_pre_windows)} windows)",
                f"{pct(r.detect_rate)} ({num(r.n_detected)})",
                fmt(r.median_delay_windows, 0),
                f"{pct(r.far_post)} ({num(r.n_post_windows)} windows)",
                f"{pct(r.recovered_rate)} ({num(r.n_recovered)} of {num(r.n_recovery_eligible)})",
            )
        )
    return table(
        [
            "tool", "records with / without a threshold", "FAR before onset",
            "detected", "median delay (windows)", "FAR after recovery",
            "recovered within 2 windows",
        ],
        rows,
    )


def anomaly_free_table(summary: pd.DataFrame, per_record: pd.DataFrame) -> str:
    free = per_record[per_record["group"] == rs.GROUP_ANOMALY_FREE]
    rows = []
    for tool in rs.TOOLS:
        r = summary_row(summary, tool, rs.GROUP_ANOMALY_FREE)
        if r is None:
            continue
        own = free[free["tool"] == tool]
        threshold = own["threshold"].iloc[0] if len(own) else None
        far = (
            "refused"
            if pd.isna(r.far_pre)
            else f"{pct(r.far_pre)} ({signed(r.over_floor)} over the {pct(r.far_floor)} floor, "
            f"{num(r.n_pre_windows)} windows)"
        )
        rows.append(
            (
                SHORT_LABELS[tool],
                f"{num(r.n_windows_scored)} / {num(r.n_windows_refused)}",
                fmt(threshold),
                far,
            )
        )
    return table(
        [
            "tool",
            "stream windows scored / refused",
            "threshold",
            "false-alarm rate over the stream",
        ],
        rows,
    )


def refusal_table(scores: pd.DataFrame) -> str:
    rows = []
    refused = scores[(scores["refusal"] != "") & (scores["role"] == rs.ROLE_STREAM)]
    for tool in rs.TOOLS:
        sub = refused[refused["tool"] == tool]
        if sub.empty:
            rows.append((SHORT_LABELS[tool], "0", ""))
            continue
        classes = sub["refusal"].map(refusal_class).value_counts().sort_index()
        rows.append(
            (
                SHORT_LABELS[tool],
                num(len(sub)),
                "; ".join(f"{n} {cls}" for cls, n in classes.items()),
            )
        )
    return table(["tool", "stream windows refused", "cause (windows)"], rows)


def conformal_table(summary: pd.DataFrame) -> str:
    rows = []
    for r in summary.sort_values(["group", "delta"], ascending=[True, False]).itertuples():
        rows.append(
            (
                r.group,
                fmt(r.delta, 2),
                f"{num(r.n_scored)} / {num(r.n_refused)}",
                num(r.n_no_alarm),
                f"{num(r.n_false_before_onset)} ({pct(r.far_before_onset)})"
                if r.group != rs.GROUP_ANOMALY_FREE
                else f"{num(r.n_false_anomaly_free)} ({pct(r.false_anomaly_free_rate)})",
                f"{num(r.n_detect)} ({pct(r.detect_rate)})",
                f"{num(r.n_after_recovery)} ({pct(r.after_recovery_rate)})",
                fmt(r.median_delay_s, 0) if r.n_detect else "",
                pct(r.permutation_alarm_fraction),
            )
        )
    return table(
        [
            "group", "delta", "records scored / refused", "no alarm",
            "false alarm", "alarm inside the span", "alarm after recovery",
            "median delay (s)", "permutation check",
        ],
        rows,
    )


# ---------------------------------------------------------------- figure


def drift_table(drift: pd.DataFrame) -> str:
    def level(value, nd: int) -> str:
        return "" if value is None or pd.isna(value) else fmt(value, nd)

    rows = []
    for r in drift.itertuples():
        name = r.tag if r.tag == rs.DRIFT_COMBINED else f"`{r.tag}`"
        rows.append(
            (
                name,
                level(r.median_first_10, 3),
                level(r.median_last_10, 3),
                "" if pd.isna(r.change) else signed(r.change),
                level(r.spearman_rho, 2),
                pct(r.share_flagged_min_0_3),
                pct(r.share_flagged_min_3_6),
                pct(r.share_flagged_min_60_63),
            )
        )
    later = drift.iloc[0]
    return table(
        [
            "tag",
            "median, first 10 min",
            "median, last 10 min",
            "change",
            "Spearman rho",
            f"flagged after a baseline at min 0-3 ({num(later.n_later_min_0_3)} min)",
            f"min 3-6 ({num(later.n_later_min_3_6)} min)",
            f"min 60-63 ({num(later.n_later_min_60_63)} min)",
        ],
        rows,
    )


def window_max_table(window_max: pd.DataFrame) -> str:
    by_tool = {tool: sub.set_index("tag") for tool, sub in window_max.groupby("tool")}
    screen = by_tool.get(rs.TOOL_SCREEN, pd.DataFrame())
    spc = by_tool.get(rs.TOOL_SPC, pd.DataFrame())
    tags = sorted(
        set(screen.index) | set(spc.index),
        key=lambda t: -float(screen["share"].get(t, 0.0)),
    )

    def cell(frame: pd.DataFrame, tag: str) -> str:
        if tag not in frame.index:
            return "0 windows"
        r = frame.loc[tag]
        return f"{pct(r.share)} ({num(r.n_windows)})"

    return table(
        [
            "tag",
            f"{SHORT_LABELS[rs.TOOL_SCREEN]}: largest robust z",
            f"{SHORT_LABELS[rs.TOOL_SPC]}: largest rule-hit count",
        ],
        [(f"`{t}`", cell(screen, t), cell(spc, t)) for t in tags],
    )


def stream_layout_table(layout: pd.DataFrame) -> str:
    rows = []
    for column, label in (
        ("n_stream", "stream windows"),
        ("n_pre", "pre-onset windows"),
        ("n_span", "in-span windows"),
        ("n_post", "post-span windows"),
    ):
        values = layout[column].astype(float)
        rows.append(
            (
                label,
                fmt(values.mean(), 1),
                fmt(values.median(), 1),
                num(values.min()),
                num(values.max()),
            )
        )
    return table(["per labelled record", "mean", "median", "min", "max"], rows)


def phase_table(phase: pd.DataFrame) -> str:
    rows = []
    for tool in rs.TOOLS:
        sub = phase[phase["tool"] == tool].set_index("phase")
        if sub.empty:
            continue

        def cell(name: str, frame: pd.DataFrame = sub) -> str:
            r = frame.loc[name]
            return f"{fmt(r.median_score, 2)} ({num(r.n_windows)})"

        span = float(sub.loc[rs.PHASE_SPAN, "median_score"])
        post = float(sub.loc[rs.PHASE_POST, "median_score"])
        rows.append(
            (
                SHORT_LABELS[tool],
                cell(rs.PHASE_PRE),
                cell(rs.PHASE_SPAN),
                cell(rs.PHASE_POST),
                fmt(post / span, 3) if span else "refused",
            )
        )
    return table(
        [
            "tool",
            "pre-onset median (windows)",
            "in-span median (windows)",
            "post-span median (windows)",
            "post-span / in-span",
        ],
        rows,
    )


def plot_trace(data: dict, out: Path) -> str | None:
    scores = data["scores"]
    per_record = data["per_record"]
    windows = data["windows"]
    record = FIGURE_RECORD
    if record not in set(windows["experiment"]):
        labelled = windows[windows["group"] == rs.GROUP_LABELLED]
        if labelled.empty:
            return None
        record = str(labelled["experiment"].iloc[0])
    w = windows[windows["experiment"] == record].sort_values("window_index")
    first, last = rs.anomaly_span(w)
    k = int((w["role"] == rs.ROLE_BASELINE).sum())
    tools = [rs.TOOL_SCREEN, rs.TOOL_SPC, rs.TOOL_T2, rs.TOOL_SPE]
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 6.0), dpi=160, sharex=True)
    for ax, tool in zip(axes.ravel(), tools, strict=True):
        sc = scores[(scores["tool"] == tool) & (scores["experiment"] == record)]
        sc = sc.sort_values("window_index")
        pr = per_record[(per_record["tool"] == tool) & (per_record["experiment"] == record)]
        threshold = float(pr["threshold"].iloc[0]) if pr["threshold"].notna().any() else None
        ax.axvspan(-0.5, k - 0.5, color="#dddddd", alpha=0.6, lw=0, label="baseline windows")
        if first is not None:
            ax.axvspan(first - 0.5, last + 0.5, color="#f4c7a1", alpha=0.6, lw=0,
                       label="anomaly rows in window")
        ax.plot(sc["window_index"], sc["score"], marker="o", ms=3.5, lw=1.2,
                color=COLORS[tool], label="window score")
        refused = sc[sc["refusal"] != ""]
        if not refused.empty:
            ax.scatter(refused["window_index"], [0] * len(refused), marker="x", color="#444444",
                       s=24, label="refused window")
        if threshold is not None:
            ax.axhline(threshold, color="#444444", ls="--", lw=1.0, label="threshold")
        ax.set_title(SHORT_LABELS[tool], fontsize=10)
        ax.grid(True, color="#eeeeee", lw=0.8)
    for ax in axes[1]:
        ax.set_xlabel("window index (60 s windows)")
    axes[0][0].set_ylabel("score")
    axes[1][0].set_ylabel("score")
    axes[0][0].legend(loc="upper left", fontsize=7, frameon=False)
    fig.suptitle(
        f"SKAB {record.replace('__', '/')}: window scores across the anomaly span; "
        "the threshold is the worst baseline window",
        fontsize=10,
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor="white", metadata={"Software": None})
    plt.close(fig)
    return record


# ---------------------------------------------------------------- report


def render_report(data: dict, figure_record: str | None) -> str:
    run = data["run"]
    summary = data["summary"]
    windows = data["windows"]
    per_record = data["per_record"]
    profile = data["profile_summary"]
    per_tag = data["profile_per_tag"]
    counts = run["counts"]
    params = run["parameters"]
    k = int(params["k_baseline_windows"])
    groups = counts["groups"]
    n_labelled = int(groups.get(rs.GROUP_LABELLED, 0))
    n_positive = int(groups.get(rs.GROUP_POSITIVE_BASELINE, 0))
    n_free = int(groups.get(rs.GROUP_ANOMALY_FREE, 0))
    labelled = windows[windows["group"] == rs.GROUP_LABELLED]
    stream = labelled[labelled["role"] == rs.ROLE_STREAM]
    n_stream = len(stream)
    n_pos = int(stream["label"].sum())
    free = windows[windows["group"] == rs.GROUP_ANOMALY_FREE]
    n_free_stream = int((free["role"] == rs.ROLE_STREAM).sum())

    def s(tool: str, group: str = rs.GROUP_LABELLED) -> pd.Series:
        row = summary_row(summary, tool, group)
        assert row is not None, (tool, group)
        return row

    screen, _, t2, _, clock = (s(t) for t in rs.TOOLS)
    frozen_tags = profile[profile["n_zero_mad_whole_record"] > 0].sort_values(
        "n_zero_mad_whole_record", ascending=False
    )
    frozen_text = ", ".join(
        f"`{r.tag}` in {num(r.n_zero_mad_whole_record)} of {num(r.n_records)} records"
        for r in frozen_tags.itertuples()
    )
    unusable = data["tag_usability"]
    unusable = unusable[~unusable["usable"].astype(bool)]
    unusable_counts = unusable["tag"].value_counts()
    unusable_text = ", ".join(
        f"`{tag}` in {num(n)} records"
        for tag, n in unusable_counts.sort_values(ascending=False).items()
    )
    one_tag = per_tag[per_tag["tag"] == per_tag["tag"].iloc[0]]
    n_gap_records = int((one_tag["longest_gap_s"].fillna(0) > 0).sum())
    longest_gap = one_tag["longest_gap_s"].max()
    # tsdive.store.tagstore reports a gap over max(2.5 x median interval, 60 s).
    gap_threshold = max(2.5 * float(per_tag["interval_median_s"].median()), 60.0)
    flat_fired = int(profile["flatline_windows_fired"].sum())
    flat_assessed = int(profile["flatline_windows_assessed"].sum())
    conformal = data["conformal"]
    cal = (
        conformal["n_calibration"].dropna()
        if "n_calibration" in conformal
        else pd.Series(dtype=float)
    )
    cal_rows = (
        f"{num(cal.min())} to {num(cal.max())}"
        if len(cal) and cal.min() != cal.max()
        else num(cal.iloc[0])
        if len(cal)
        else "0"
    )
    missing_folder: dict[str, list[str]] = {}
    for tool in rs.TOOLS:
        row = summary_row(summary, tool, rs.GROUP_LABELLED)
        for folder in rs.FOLDERS:
            if row is not None and pd.isna(row[f"roc_auc_{folder}"]):
                missing_folder.setdefault(folder, []).append(SHORT_LABELS[tool])
    folder_refusal_text = " ".join(
        f"No window of a {folder} record is scored by "
        f"{' or '.join(tools)}, so those cells carry no AUC "
        f"(`roc_auc_{folder}_refusal` in `summary.csv`)."
        for folder, tools in missing_folder.items()
    )
    cs = data["conformal_summary"]
    conf = cs[(cs["group"] == rs.GROUP_LABELLED) & (cs["delta"] == max(params["deltas"]))]
    conf = conf.iloc[0] if not conf.empty else None
    conf_free = cs[(cs["group"] == rs.GROUP_ANOMALY_FREE) & (cs["delta"] == max(params["deltas"]))]
    conf_free = conf_free.iloc[0] if not conf_free.empty else None
    free_alarm = (
        "raised"
        if conf_free is not None and conf_free["n_false_anomaly_free"] > 0
        else "not raised"
    )
    best_auc = max(rs.TOOLS[:-1], key=lambda t: float(s(t)["roc_auc"] or 0))
    lowest_far = min(
        rs.TOOLS[:-1],
        key=lambda t: float("inf") if pd.isna(s(t)["far_pre"]) else float(s(t)["far_pre"]),
    )
    no_threshold = per_record[
        (per_record["group"] == rs.GROUP_LABELLED) & per_record["threshold"].isna()
    ]
    by_records: dict[tuple[str, ...], list[str]] = {}
    for tool in rs.TOOLS:
        recs = tuple(sorted(set(no_threshold[no_threshold["tool"] == tool]["experiment"])))
        if recs:
            by_records.setdefault(recs, []).append(SHORT_LABELS[tool])
    no_threshold_text = "; ".join(
        f"{' and '.join(tools)} on {num(len(recs))} "
        f"({', '.join(f'`{r}`' for r in recs)})"
        for recs, tools in by_records.items()
    )
    screen_free = s(rs.TOOL_SCREEN, rs.GROUP_ANOMALY_FREE)
    drift = data["drift"]
    drift_tags = drift[drift["tag"] != rs.DRIFT_COMBINED]
    combined = drift[drift["tag"] == rs.DRIFT_COMBINED].iloc[0]
    rising = drift_tags.loc[drift_tags["spearman_rho"].idxmax()]
    falling = drift_tags.loc[drift_tags["spearman_rho"].idxmin()]
    zero_mad_tags = ", ".join(
        f"`{t}`" for t in drift_tags[drift_tags["refusal"].fillna("") != ""]["tag"]
    )
    window_max = data["window_max_tag"]
    screen_max = window_max[window_max["tool"] == rs.TOOL_SCREEN].sort_values(
        "share", ascending=False
    )
    top_max = [f"`{r.tag}` on {pct(r.share)}" for r in screen_max.head(3).itertuples()]
    top_max_text = ", ".join(top_max[:-1]) + " and " + top_max[-1]
    layout = data["stream_layout"]
    layout = layout[layout["group"] == rs.GROUP_LABELLED]
    n_no_post = int((layout["n_post"] == 0).sum())
    multi_span = layout[layout["n_spans"] > 1]
    multi_span_text = ", ".join(
        f"`{r.experiment.replace('__', '/')}`" for r in multi_span.itertuples()
    )
    phase = data["score_by_phase"]
    ratio_parts = []
    for tool in rs.TOOLS:
        sub = phase[phase["tool"] == tool].set_index("phase")
        if sub.empty:
            continue
        span_median = float(sub.loc[rs.PHASE_SPAN, "median_score"])
        post_median = float(sub.loc[rs.PHASE_POST, "median_score"])
        ratio_parts.append(f"{fmt(post_median / span_median, 3)} for {SHORT_LABELS[tool]}")
    ratio_text = ", ".join(ratio_parts[:-1]) + " and " + ratio_parts[-1]

    lines = [
        "# The detectors and the clock control on SKAB",
        "",
        "## Summary",
        "",
        f"The study scores {n_labelled} labelled SKAB records (manifest "
        f"`{run['dataset_manifest_sha256'][:12]}`, {n_stream} stream windows of "
        f"{params['window_s']} s, {n_pos} of them holding an anomaly row) and the "
        f"anomaly-free record ({n_free_stream} stream windows) with `screen`, `spc` and "
        f"`mspc` through the package API, each record against its own first {k} windows, "
        "and publishes the clock control beside every number. The profile finds two "
        f"sensors with a zero MAD over the whole record ({frozen_text}) and {n_gap_records} "
        f"records with a gap over its {fmt(gap_threshold, 0)} s threshold, the longest "
        f"{fmt(longest_gap, 0)} s, before any detector runs. "
        f"Pooled over the stream windows the best ranking is {SHORT_LABELS[best_auc]} at "
        f"ROC-AUC {fmt(s(best_auc)['roc_auc'])} against {fmt(clock['roc_auc'])} for the "
        "clock control, so window position alone accounts for most of the ranking. "
        "Under the worst-baseline threshold the lowest false-alarm rate before "
        f"onset is {pct(s(lowest_far)['far_pre'])} for {SHORT_LABELS[lowest_far]} against a "
        f"{pct(screen['far_floor'])} floor, and after the anomaly span the MAD screen fires "
        f"on {pct(screen['far_post'])} of the normal windows and comes back under its "
        f"threshold within {params['recovery_windows']} windows on "
        f"{pct(screen['recovered_rate'])} of the records that fired. The conformal "
        f"martingale alarm at delta {fmt(conf['delta'], 2) if conf is not None else 'n/a'} "
        f"fires before onset on {pct(conf['far_before_onset']) if conf is not None else 'n/a'} "
        "of the labelled records; the permutation check gives "
        f"{pct(conf['permutation_alarm_fraction']) if conf is not None else 'n/a'}.",
        "",
        "## Data",
        "",
        f"SKAB (`waico/SKAB` @ `{run['upstream_commit'][:8]}`, GPL-3.0), manifest sha256 "
        f"`{run['dataset_manifest_sha256']}`, fetched and converted as "
        "[docs/DATA.md](../../../docs/DATA.md) describes: 34 labelled experiments on a "
        "water-circulation testbed, each about 20 minutes with one anomaly run that ends "
        "before the record does, and one anomaly-free run. Eight sensors per record at a "
        "1 s step, no quality column (`quality_assumed`), no units, timestamps localised as "
        f"UTC. The study reads {counts['n_records']} records and {counts['n_windows']} fixed "
        f"{params['window_s']} s windows from each record's first timestamp. A window is "
        "positive when any of its rows is an anomaly row (`label`); `label_majority` "
        "is also scored and reported beside it.",
        "",
        f"Groups by position, label-blind: {n_labelled} records are `labelled` (no anomaly "
        f"row in the first {k} windows), {n_positive} are `positive_baseline` (an anomaly "
        f"row inside the baseline, reported without a claim) and {n_free} is "
        "`anomaly_free`, its own bed. Folds are the three upstream folders (valve1, "
        "valve2, other); nothing is fitted across records, so group holdout holds by "
        "construction and the per-folder AUCs are the only cross-record cut.",
        "",
        "### What the profile found before any detector ran",
        "",
        f"`api.profile(path, flatline=True)` on every archive, then again on every "
        f"{params['window_s']} s window with the earlier windows as its reference history:",
        "",
        profile_table(profile),
        "",
        f"Two tags carry a zero MAD over the whole record ({frozen_text}). The score reads "
        f"the MAD over the first {k} windows instead, and that scale is zero for "
        f"{unusable_text} (`tag_usability.csv`). A tag with a zero baseline scale gives no "
        f"robust z, so it is left out of that record's score. The flatline check fires on "
        f"{num(flat_fired)} of {num(flat_assessed)} assessed windows over all tags. The "
        f"profile reports a gap when a hole runs longer than {fmt(gap_threshold, 0)} s, "
        "the larger of 2.5 times the median interval and 60 s. Shorter holes are common "
        "on this bed and [docs/DATA.md](../../../docs/DATA.md) lists them. The records "
        "with a reported gap:",
        "",
        gap_table(per_tag),
        "",
        "A window inside one of those gaps holds no rows and is a refusal for every tool. "
        "The unusable tags per record:",
        "",
        usability_table(data["tag_usability"]),
        "",
        "## Method",
        "",
        f"1. Windows: fixed {params['window_s']} s windows from the first timestamp of each "
        "record. Archive reads are inclusive at both ends, so every API window string ends "
        "one second before the next window starts.",
        f"2. Own history: the first {k} windows of each record are its baseline, the rest "
        f"its stream. The baseline string covers the first {k} minutes; the monitored "
        "string is the scored minute.",
        "3. `screen(archive, baseline, window, k=3)` per usable tag and window; the tag "
        "score is the largest |value - centre| / scale over the window's valid rows, with "
        "centre and scale from the returned MAD baseline; the window score is the maximum "
        "over the usable tags. The baseline windows are scored from the same baseline read.",
        "4. `spc(archive, baseline, window)` per usable tag and window; the window score is "
        "the rule-hit count summed over the usable tags. Baseline windows apply the same "
        "`apply_rules` with the returned limits to the baseline read.",
        f"5. `mspc(archives, baseline, window, rate_s={params['rate_s']})` over the usable "
        "tags; the T2 and SPE scores are the window means. Baseline windows are scored by "
        "`detect` on the fitted model's own training matrix. A record with fewer than two "
        "usable tags, or a baseline whose aligned grid coverage is under the package's "
        "0.95, is a refusal with the error's message.",
        "6. `clock_control` over the window index. A tag is unusable for a record when the "
        "profile says it holds one distinct value or its baseline MAD scale is zero.",
        "7. Alarm rule: `worst_baseline_threshold` over the record's own baseline scores, "
        "`fires` strictly above it. FAR before onset is pooled over the stream windows "
        "before the first anomaly window; detect is any anomaly window firing; delay is "
        "the first firing anomaly window minus the first anomaly window; FAR after "
        "recovery is pooled over the normal windows after the last anomaly window; "
        f"recovered means the score is back under the threshold within "
        f"{params['recovery_windows']} windows after the span, among records that fired "
        f"in the span and have a window after it. `far_floor({k})` = "
        f"{fmt(far_floor_value(k))} sits beside every FAR.",
        "8. Ranking: `ranking_metrics` pooled over the labelled stream windows, per folder "
        "and per record, with the one-class refusal.",
        f"9. Conformal bed: per record, median and MAD per usable tag on windows 1 to "
        f"{k - 1}, calibration on window {k} ({cal_rows} one-second scores, one per row "
        "of that window; nonconformity = largest robust z over the usable tags), "
        "stream from window "
        f"{k + 1}; `conformal_p_values`, `mixture_martingale` (epsilons "
        f"{', '.join(str(e) for e in params['epsilons'])}) and `martingale_alarm` at "
        f"delta {', '.join(str(d) for d in params['deltas'])}. The permutation check "
        f"shuffles calibration and stream together, {params['permutation_seeds']} seeds.",
        "",
        "```",
        "uv run python examples/studies/skab/fetch_skab.py --dest data/skab",
        "uv run python examples/studies/skab/build_archives.py",
        "uv run python examples/studies/skab/run_skab.py",
        "uv run python examples/studies/skab/make_report.py --update-benchmarks",
        "```",
        "",
        f"The scoring run takes {run['wall_seconds']} s. Two runs write byte-identical "
        "CSVs.",
        "",
        "## Results",
        "",
        "### Ranking on the labelled records",
        "",
        ranking_table(summary),
        "",
        f"The clock control ranks at {fmt(clock['roc_auc'])} pooled, which is what window "
        "position alone is worth on this bed. The anomaly run sits in the later part of "
        "most records, so position ranks above chance, and it ends before the record does, "
        "so position does not separate the classes. "
        f"{SHORT_LABELS[best_auc]} ranks highest at {fmt(s(best_auc)['roc_auc'])}, "
        f"{signed(float(s(best_auc)['roc_auc']) - float(clock['roc_auc']))} over the clock. "
        f"The MSPC tools refuse {num(t2['n_windows_refused'])} stream windows, so their AUCs "
        f"read over fewer windows than the univariate ones. {folder_refusal_text}",
        "",
        "### The alarm rule on the labelled records",
        "",
        alarm_table(summary),
        "",
        f"The clock control fires on every stream window (FAR {pct(clock['far_pre'])}, "
        f"recovered {pct(clock['recovered_rate'])}), which is what a score that grows with "
        "position does under a threshold set on the first windows. "
        f"{SHORT_LABELS[lowest_far]} has the lowest FAR before onset, "
        f"{pct(s(lowest_far)['far_pre'])} ({signed(s(lowest_far)['over_floor'])} over the "
        "floor). A record has no threshold when one of its baseline windows is refused: "
        f"{no_threshold_text or 'no record'}. Those rows are in `per_record.csv`.",
        "",
        "### The anomaly-free record",
        "",
        anomaly_free_table(summary, per_record),
        "",
        "### Level movement on the anomaly-free record",
        "",
        drift_table(drift),
        "",
        "Each row is one tag on the anomaly-free record. The medians are of the "
        "per-minute medians over the first and last ten minutes. Spearman rho is the "
        "per-minute median against the minute index. A flagged share is the share of the "
        "minutes after the baseline whose largest |value - centre| / scale is above "
        f"{fmt(params['k_mad'], 0)}, with centre and scale from a three-minute MAD baseline "
        f"at that position. `{rising.tag}` {'rises' if rising.change > 0 else 'falls'} "
        f"from {fmt(rising.median_first_10, 1)} to {fmt(rising.median_last_10, 1)} over "
        f"{num(rising.n_windows)} minutes (rho {fmt(rising.spearman_rho, 2)}) and "
        f"`{falling.tag}` {'falls' if falling.change < 0 else 'rises'} from "
        f"{fmt(falling.median_first_10, 1)} to {fmt(falling.median_last_10, 1)} "
        f"(rho {fmt(falling.spearman_rho, 2)}). The last row takes the maximum over the "
        "usable tags, the quantity `screen` scores, and flags "
        f"{pct(combined.share_flagged_min_0_3)}, {pct(combined.share_flagged_min_3_6)} and "
        f"{pct(combined.share_flagged_min_60_63)} of the later minutes for the three "
        f"baseline positions. {zero_mad_tags} has a zero MAD in every baseline, so its "
        "shares are refusals (`refusal` in `drift.csv`).",
        "",
        "### Which tag sets the window score",
        "",
        window_max_table(window_max),
        "",
        f"Share of the {num(screen_max['n_windows_total'].iloc[0])} scored windows on which "
        "the tag holds the per-tag maximum, over all records. `screen` takes the maximum "
        "over the usable tags, so that tag sets the window score. `spc` sums the rule hits, "
        "so that tag contributes the most of them.",
        "",
        "### Where the anomaly span sits in the stream",
        "",
        stream_layout_table(layout),
        "",
        f"Over the {num(len(layout))} labelled records, {num(n_no_post)} have no window "
        f"after the anomaly span and {num(len(multi_span))} "
        f"({multi_span_text or 'none'}) {'holds' if len(multi_span) == 1 else 'hold'} more "
        "than one span.",
        "",
        "### Score by phase",
        "",
        phase_table(phase),
        "",
        "Median window score over the labelled records' scored stream windows before the "
        "first anomaly window, inside the span and after the last anomaly window. A score "
        "is only comparable with itself, so read along a row, never down a column.",
        "",
        "### Refusals",
        "",
        refusal_table(data["scores"]),
        "",
        "### Conformal martingale bed",
        "",
        conformal_table(cs),
        "",
    ]
    if conf is not None:
        lines += [
            f"At delta {fmt(conf['delta'], 2)} the alarm fires before onset on "
            f"{pct(conf['far_before_onset'])} of the labelled records and inside the span on "
            f"{pct(conf['detect_rate'])}; the permutation check, which makes the scores "
            f"exchangeable, gives {pct(conf['permutation_alarm_fraction'])}. On the "
            "anomaly-free record the alarm is "
            f"{free_alarm}.",
            "",
        ]
    if figure_record is not None:
        lines += [
            f"![window scores across the anomaly span]({'out/' + FIGURE_NAME})",
            "",
            f"Figure 1: `{figure_record.replace('__', '/')}`, one window score per tool "
            "with its own threshold; the baseline windows are grey and the windows holding "
            "anomaly rows are orange.",
            "",
        ]
    lines += [
        "## Discussion",
        "",
        f"The clock control scores ROC-AUC {fmt(clock['roc_auc'])} and fires on "
        f"{pct(clock['far_pre'])} of the pre-onset windows. On 3W the same control outscored "
        "every detector because the fault, once present, stayed to the end of the "
        "instance. On SKAB the records return to normal, so position stops at "
        f"{fmt(clock['roc_auc'])} and a detector has to beat that number to have read "
        "anything from the sensors. The gap is "
        f"{signed(float(s(best_auc)['roc_auc']) - float(clock['roc_auc']))} for "
        f"{SHORT_LABELS[best_auc]}. The anomaly span starts after a mean "
        f"{fmt(layout['n_pre'].mean(), 1)} pre-onset windows, covers "
        f"{fmt(layout['n_span'].mean(), 1)} and leaves {fmt(layout['n_post'].mean(), 1)} "
        f"post-span windows of a mean {fmt(layout['n_stream'].mean(), 1)}-window stream "
        "(`stream_layout.csv`), so the post-span stretch is short and position still "
        "orders most anomaly windows above the normal ones.",
        "",
        f"The MAD screen fires on {pct(screen['far_pre'])} of the pre-onset windows, "
        f"{pct(screen['far_post'])} of the post-recovery windows and "
        f"{pct(screen_free['far_pre'])} of the anomaly-free stream. Two tags change level "
        f"over the anomaly-free record. The `{rising.tag}` minute median "
        f"{'rises' if rising.change > 0 else 'falls'} from {fmt(rising.median_first_10, 1)} "
        f"to {fmt(rising.median_last_10, 1)} over {num(rising.n_windows)} minutes, Spearman "
        f"{fmt(rising.spearman_rho, 2)} against the minute index, and `{falling.tag}` "
        f"{'falls' if falling.change < 0 else 'rises'} from "
        f"{fmt(falling.median_first_10, 1)} to {fmt(falling.median_last_10, 1)} "
        f"({fmt(falling.spearman_rho, 2)}). A three-minute MAD baseline at minutes 0-3, 3-6 "
        f"or 60-63 leaves {pct(combined.share_flagged_min_0_3)}, "
        f"{pct(combined.share_flagged_min_3_6)} and {pct(combined.share_flagged_min_60_63)} "
        "of the later minutes above 3 MAD on the maximum over the usable tags "
        "(`drift.csv`). A baseline an hour into the record gives the same share as one at "
        "the start, so the false-alarm rates on the anomaly-free record and before onset "
        "measure that level movement wherever the baseline sits. The tag holding the "
        f"screen's window maximum is {top_max_text} of the "
        f"{num(screen_max['n_windows_total'].iloc[0])} scored windows (`window_max_tag.csv`). "
        f"MSPC T2 fires on {pct(t2['far_pre'])} before onset and detects "
        f"{pct(t2['detect_rate'])} of the records, with {pct(t2['far_post'])} after "
        "recovery.",
        "",
        f"The profile found the two low-information sensors ({frozen_text}) and the "
        f"{n_gap_records} gapped records before any detector ran. The refusal rows carry "
        f"them through the tables. A zero baseline scale leaves {unusable_text} out of a "
        "score, and a window inside a gap is refused by name.",
        "",
        "## Limits",
        "",
        "- SKAB is a testbed with 34 short records; a 60 s window and a three-window "
        "baseline leave 7 to 9 pre-onset windows per record, so every FAR is read against "
        f"a {pct(screen['far_floor'])} floor.",
        "- The baseline is the first three minutes of each record by position and its "
        f"centre and scale are static. `{rising.tag}` moves {signed(rising.change, 2)} and "
        f"`{falling.tag}` {signed(falling.change, 2)} over the anomaly-free record, and the "
        "maximum over the usable tags stays above 3 MAD on "
        f"{pct(combined.share_flagged_min_60_63)} of the later minutes for a baseline placed "
        "at minutes 60-63, so the level movement sets the `screen` and `spc` false-alarm "
        "rates.",
        "- The label end may precede the physical recovery of the loop, so a recovered rate "
        "read against the labels understates recovery to the pre-fault state. The post-span "
        f"median score over the in-span median is {ratio_text} (`score_by_phase.csv`).",
        "- `mspc` refuses a baseline whose aligned coverage is under 0.95, so the gapped "
        "records and the anomaly-free record carry no MSPC number.",
        "- The anomaly label is per row and the window label is any anomaly row; the "
        "majority label is reported beside it and moves the AUC little.",
        "- Timestamps are naive and localised as UTC; nothing here depends on the offset.",
        "",
        "## Files",
        "",
        table(
            ["file", "holds"],
            [
                ("`results/profile_per_tag.csv`", "one row per archive from `api.profile`"),
                ("`results/profile_summary.csv`", "the profile per tag over the records"),
                ("`results/windows.csv`", "the fixed windows with their labels and group"),
                ("`results/scores.csv`", "one row per (tool, window), score or refusal"),
                ("`results/tag_scores.csv`", "the per-tag `screen` and `spc` scores"),
                ("`results/tag_usability.csv`", "why a tag is left out of a record's score"),
                ("`results/per_record.csv`", "the alarm rule per (tool, record)"),
                ("`results/summary.csv`", "the pooled numbers per tool and group"),
                ("`results/conformal.csv`", "the conformal bed per record and delta"),
                ("`results/conformal_summary.csv`", "the conformal bed per group and delta"),
                ("`results/permutation.csv`", "the permutation check per record, delta, seed"),
                ("`results/drift.csv`", "level movement per tag on the anomaly-free record"),
                ("`results/window_max_tag.csv`", "the tag holding the per-tag maximum per window"),
                ("`results/stream_layout.csv`", "pre, span and post window counts per record"),
                ("`results/score_by_phase.csv`", "median score per tool and phase"),
                ("`results/run.json`", "provenance, parameters, counts, wall seconds"),
                ("`results/benchmarks_section.md`", "the BENCHMARKS.md REAL section"),
                (f"`out/{FIGURE_NAME}`", "Figure 1"),
            ],
        ),
        "",
    ]
    return "\n".join(lines)


def far_floor_value(k: int) -> float:
    return rs.far_floor(k)


# ---------------------------------------------------------------- benchmarks


def render_benchmarks_section(data: dict) -> str:
    run = data["run"]
    summary = data["summary"]
    profile = data["profile_summary"]
    params = run["parameters"]
    k = int(params["k_baseline_windows"])
    groups = run["counts"]["groups"]
    dataset = dataset_cite(run)
    code = code_cite(run)
    design = "own-history"
    split = (
        f"own history: first {k} windows of each record, label-blind; nothing fitted across "
        "records, so no group holdout applies; folds are the upstream folders; "
        f"{groups.get(rs.GROUP_LABELLED, 0)} labelled records, "
        f"{groups.get(rs.GROUP_ANOMALY_FREE, 0)} anomaly-free"
    )
    rows: list[tuple[str, str, str]] = []
    frozen = profile[profile["n_zero_mad_whole_record"] > 0]
    rows.append(
        (
            "1",
            "sensors with a zero MAD over the whole record (profile, before any detector)",
            "; ".join(
                f"`{r.tag}` {num(r.n_zero_mad_whole_record)} of {num(r.n_records)} records"
                for r in frozen.itertuples()
            )
            or "none",
        )
    )
    usability = data["tag_usability"]
    n_assessed = usability["experiment"].nunique()
    unusable = usability[~usability["usable"].astype(bool)]
    rows.append(
        (
            "1",
            "sensors left out of a record's score (zero MAD over the baseline windows)",
            "; ".join(
                f"`{tag}` {num(n)} of {num(n_assessed)} records"
                for tag, n in unusable["tag"].value_counts().sort_values(ascending=False).items()
            )
            or "none",
        )
    )
    for tool in rs.TOOLS:
        r = summary_row(summary, tool, rs.GROUP_LABELLED)
        if r is None:
            continue
        label = TOOL_LABELS[tool]
        if tool == rs.TOOL_CLOCK:
            label = "**clock control** (window position in its own record, no sensor read)"
        rows.append(
            (
                STAGES[tool],
                f"{label}: ranking, labelled records",
                f"AUC {fmt(r.roc_auc)} (valve1 {fmt(r.roc_auc_valve1)}, valve2 "
                f"{fmt(r.roc_auc_valve2)}, other {fmt(r.roc_auc_other)}), "
                f"{num(r.n_windows_scored)} windows scored, {num(r.n_windows_refused)} refused",
            )
        )
        rows.append(
            (
                STAGES[tool],
                f"{label}: worst-baseline alarm, labelled records",
                f"FAR before onset {pct(r.far_pre)} ({signed(r.over_floor)} over the "
                f"{pct(r.far_floor)} floor), detect {pct(r.detect_rate)}, median delay "
                f"{fmt(r.median_delay_windows, 0)} windows, FAR after recovery "
                f"{pct(r.far_post)}, recovered within {params['recovery_windows']} windows "
                f"{pct(r.recovered_rate)}; {num(r.n_records_with_threshold)} records with a "
                "threshold",
            )
        )
        free = summary_row(summary, tool, rs.GROUP_ANOMALY_FREE)
        if free is not None:
            value = (
                f"refused, {num(free.n_windows_refused)} of "
                f"{num(free.n_stream_windows)} windows refused"
                if pd.isna(free.far_pre)
                else f"FAR {pct(free.far_pre)} ({signed(free.over_floor)} over the "
                f"{pct(free.far_floor)} floor) over {num(free.n_pre_windows)} windows, "
                f"{num(free.n_windows_refused)} refused"
            )
            rows.append(
                (
                    STAGES[tool],
                    f"{label}: false-alarm rate over the anomaly-free record",
                    value,
                )
            )
    cs = data["conformal_summary"]
    for r in cs.sort_values(["group", "delta"], ascending=[True, False]).itertuples():
        if r.group == rs.GROUP_ANOMALY_FREE:
            value = (
                f"false alarm on {num(r.n_false_anomaly_free)} of {num(r.n_scored)} "
                f"records (bound {pct(r.delta)}); permutation check "
                f"{pct(r.permutation_alarm_fraction)} over {num(r.permutation_seeds)} seeds"
            )
        else:
            value = (
                f"FAR before onset {pct(r.far_before_onset)} (bound {pct(r.delta)}), alarm "
                f"inside the span {pct(r.detect_rate)}, after recovery "
                f"{pct(r.after_recovery_rate)}, median delay {fmt(r.median_delay_s, 0)} s; "
                f"{num(r.n_scored)} scored, {num(r.n_refused)} refused; permutation check "
                f"{pct(r.permutation_alarm_fraction)} over {num(r.permutation_seeds)} seeds"
            )
        rows.append(
            (
                "none",
                f"conformal martingale alarm, delta {fmt(r.delta, 2)}, {r.group} records",
                value,
            )
        )
    lines = [
        SECTION_HEADING,
        "",
        "The package's `screen`, `spc` and `mspc` calls and the clock control on SKAB, the "
        "Skoltech Anomaly Benchmark, where every labelled record returns to normal after "
        "its fault. Each record is scored against its own first three 60 s windows; the "
        "profile runs first and its findings (zero-MAD sensors, gapped records) become "
        "refusal rows. Method, tables and the conformal bed are in "
        "[examples/studies/skab/REPORT.md](examples/studies/skab/REPORT.md).",
        "",
        "This section is not regenerated by `make bench`; it is produced by "
        "`examples/studies/skab/run_skab.py` and `make_report.py`, and carried through the "
        "generator unchanged.",
        "",
        "| stage | metric | value | dataset (manifest sha256) | code | design | split |",
        "|---|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {stage} | {metric} | {value} | {dataset} | {code} | {design} | {split} |"
        for stage, metric, value in rows
    ]
    return "\n".join(lines) + "\n"


def splice_section(text: str, section: str) -> str:
    """Replace this study's section in BENCHMARKS.md, or append it."""
    start = text.find(SECTION_HEADING)
    if start < 0:
        return text.rstrip("\n") + "\n\n" + section
    nxt = text.find("\n## ", start + len(SECTION_HEADING))
    end = len(text) if nxt < 0 else nxt + 1
    return text[:start] + section + text[end:]


# ---------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results", default=str(HERE / "results"))
    parser.add_argument("--out", default=str(HERE / "out"))
    parser.add_argument("--report", default=str(HERE / "REPORT.md"))
    parser.add_argument("--update-benchmarks", action="store_true")
    parser.add_argument("--benchmarks", default=str(BENCH_MD))
    args = parser.parse_args(argv)

    results = Path(args.results)
    data = load(results)
    figure_record = plot_trace(data, Path(args.out) / FIGURE_NAME)
    Path(args.report).write_text(
        render_report(data, figure_record), encoding="utf-8", newline="\n"
    )
    section = render_benchmarks_section(data)
    (results / "benchmarks_section.md").write_text(section, encoding="utf-8", newline="\n")
    if args.update_benchmarks:
        bench = Path(args.benchmarks)
        bench.write_text(
            splice_section(bench.read_text(encoding="utf-8"), section),
            encoding="utf-8",
            newline="\n",
        )
        print(f"updated {bench}")
    print(f"wrote {args.report}, {Path(args.out) / FIGURE_NAME} and "
          f"{results / 'benchmarks_section.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

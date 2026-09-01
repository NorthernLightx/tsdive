"""Turn the detector run into REPORT.md, summary.json and four PNGs.

    uv run python examples/studies/3w_detectors/make_report.py

Reads only what ``run_detectors.py`` wrote. Nothing here recomputes a
score: the ROC curves and the per-folder table are drawn from
``results/scores.csv``, so the pictures and the tables cannot drift apart.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
ROOT = Path(__file__).parents[3]
sys.path.insert(0, str(ROOT / "src"))

POOLED = "pooled-cross-well"
MATCHED = "pooled-cross-well, own-history windows"
OWN = "own-history"
STANDARDISED = "per-instance-standardised"

TOOL_LABELS = {
    "mad": ("3", "MAD screen (regime-blind)"),
    "regime": ("4", "regime screen (LABEL_state)"),
    "population": ("4", "population screen (frozen tags)"),
    "spc": ("5", "SPC individuals rules"),
    "mspc_t2": ("6", "MSPC Hotelling T2"),
    "mspc_spe": ("6", "MSPC SPE"),
    "iforest": ("7", "IsolationForest"),
}
# Which design replaces the pooled one for each tool. The regime and
# population screens keep their pooled design: neither fits a per-tag
# limit, so neither is what this rewrite is about.
NEW_DESIGN = {
    "mad": OWN,
    "spc": OWN,
    "mspc_t2": STANDARDISED,
    "mspc_spe": STANDARDISED,
    "iforest": STANDARDISED,
}
POOLED_SPLIT = {"population": "instance"}
VARIANT_LABELS = {
    "all": "all windows",
    "no_folder9": "folder 9 excluded",
    "no_transients": "transient-only windows excluded",
    "no_folder9_no_transients": "folder 9 and transient-only excluded",
}
COLORS = {
    "mad": "#1f77b4",
    "regime": "#ff7f0e",
    "population": "#8c564b",
    "spc": "#2ca02c",
    "mspc_t2": "#9467bd",
    "mspc_spe": "#d62728",
    "iforest": "#17becf",
    "clock": "#7f7f7f",
    "clock_flat": "#bbbbbb",
}

ALIGNED = "onset-aligned"
ALIGNED_TOOLS = ("mad", "spc", "mspc_t2", "mspc_spe", "iforest", "clock", "clock_flat")
ALIGNED_LABELS = {
    "mad": "MAD screen (regime-blind), own history",
    "spc": "SPC individuals rules, own history",
    "mspc_t2": "MSPC Hotelling T2, per-instance-standardised",
    "mspc_spe": "MSPC SPE, per-instance-standardised",
    "iforest": "IsolationForest, per-instance-standardised",
    "clock": "**clock control** (position in the record)",
    "clock_flat": "clock control, design offset removed",
}


def fmt(value, nd: int = 3, dash: str = "—") -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return dash
    return f"{value:.{nd}f}"


def pick(summary: pd.DataFrame, tool: str, design: str, variant: str) -> pd.Series | None:
    """One summary row. The population screen exists under two splits."""
    sub = summary[
        (summary["tool"] == tool)
        & (summary["design"] == design)
        & (summary["variant"] == variant)
    ]
    if tool in POOLED_SPLIT and design == POOLED:
        sub = sub[sub["split"] == POOLED_SPLIT[tool]]
    return None if not len(sub) else sub.iloc[0]


def auc_cell(row: pd.Series | None) -> str:
    if row is None or row["n_folds_scored"] == 0 or pd.isna(row["auc_mean"]):
        return "refused"
    return (
        f"{row['auc_mean']:.3f} ({row['auc_min']:.3f}-{row['auc_max']:.3f})"
        if row["n_folds_scored"] > 1
        else f"{row['auc_mean']:.3f}"
    )


def auc(labels: pd.Series, frame: pd.DataFrame) -> float | None:
    from sklearn.metrics import roc_auc_score

    y = labels.loc[frame["window_key"]].to_numpy(dtype=int)
    if len(y) == 0 or y.min() == y.max():
        return None
    return float(roc_auc_score(y, frame["score"].to_numpy(dtype=float)))


# ------------------------------------------------------------------ figures


def plot_roc(scores: pd.DataFrame, windows: pd.DataFrame, out: Path) -> None:
    """One ROC per fold per tool, pooled design.

    Curves are drawn per fold and never pooled. Each fold fits its own
    baselines, so a score from fold 0 and a score from fold 3 are numbers
    on different scales; a pooled curve over them would be an artifact of
    that mismatch rather than a detector's behaviour.
    """
    from sklearn.metrics import roc_curve

    tools = list(TOOL_LABELS)
    fig, axes = plt.subplots(2, 4, figsize=(13.0, 6.6), dpi=160, sharex=True, sharey=True)
    labels = windows.set_index("window_key")["label"]
    for ax, tool in zip(axes.ravel(), tools, strict=False):
        split = POOLED_SPLIT.get(tool, "well")
        sub_all = scores[
            (scores["tool"] == tool)
            & (scores["design"] == POOLED)
            & (scores["split"] == split)
        ]
        ax.plot([0, 1], [0, 1], color="#888888", lw=1.0, ls="--")
        for fold in sorted(sub_all["fold"].unique()) if len(sub_all) else []:
            sub = sub_all[sub_all["fold"] == fold]
            y = labels.loc[sub["window_key"]].to_numpy(dtype=int)
            if y.min() == y.max():
                continue
            fpr, tpr, _ = roc_curve(y, sub["score"].to_numpy(dtype=float))
            ax.plot(fpr, tpr, color=COLORS[tool], lw=1.3, alpha=0.85, label=f"fold {fold}")
        if not len(sub_all):
            ax.text(0.5, 0.5, "refused\non every fold", ha="center", va="center", fontsize=9)
        title = (
            "population screen (frozen tags, instance holdout)"
            if tool == "population"
            else TOOL_LABELS[tool][1]
        )
        ax.set_title(title, fontsize=8.5)
        ax.grid(alpha=0.25)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes.ravel()[-1].axis("off")
    axes.ravel()[-1].text(
        0.0,
        0.5,
        "Pooled-cross-well design.\nOne curve per held-out fold.\nCurves are never pooled:\n"
        "each fold fits its own\nbaselines, so scores from\ntwo folds are numbers on\n"
        "different scales.\n\nThe dashed diagonal is\nchance.",
        fontsize=9,
        va="center",
    )
    fig.supxlabel("false positive rate", fontsize=10)
    fig.supylabel("true positive rate", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    plt.close(fig)


def plot_same_population(summary: pd.DataFrame, out: Path, n_own: int, k: int) -> None:
    """The design change, isolated: identical windows, two designs."""
    tools = [t for t in TOOL_LABELS if t in NEW_DESIGN]
    fig, ax = plt.subplots(figsize=(9.0, 5.0), dpi=160)
    width = 0.36
    for i, tool in enumerate(tools):
        old = pick(summary, tool, MATCHED, "all")
        new = pick(summary, tool, NEW_DESIGN[tool], "all")
        ax.bar(i - width / 2, old["auc_mean"], width, color="#bbbbbb", zorder=3)
        ax.bar(i + width / 2, new["auc_mean"], width, color=COLORS[tool], zorder=3)
        ax.text(i - width / 2, old["auc_mean"] + 0.012, f"{old['auc_mean']:.3f}",
                ha="center", fontsize=7.5)
        ax.text(i + width / 2, new["auc_mean"] + 0.012, f"{new['auc_mean']:.3f}",
                ha="center", fontsize=7.5)
    clock = pick(summary, "clock", OWN, "all")
    ax.axhline(clock["auc_mean"], color="#000000", lw=1.2, ls=":", zorder=4)
    ax.text(
        len(tools) - 0.4,
        clock["auc_mean"] + 0.012,
        f"clock control {clock['auc_mean']:.3f}",
        ha="right",
        fontsize=8,
    )
    ax.axhline(0.5, color="#888888", lw=1.0, ls="--")
    ax.set_xticks(range(len(tools)))
    ax.set_xticklabels(
        [TOOL_LABELS[t][1].replace(" (", "\n(") for t in tools],
        fontsize=7.5,
        rotation=18,
        ha="right",
    )
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("ROC-AUC")
    ax.set_title(
        f"Same {n_own:,d} windows, two designs. Grey: limits pooled across wells.\n"
        f"Coloured: limits from the tag's own first {k} windows.\n"
        "The dotted line reads no sensor value at all.",
        fontsize=10,
    )
    ax.grid(alpha=0.25, axis="y", zorder=0)
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    plt.close(fig)


def plot_fold_spread(fold_df: pd.DataFrame, out: Path, n_all: int, n_own: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0), dpi=160, sharey=True)
    panels = [
        (axes[0], POOLED, f"pooled-cross-well, all {n_all:,d} windows", list(TOOL_LABELS)),
        (
            axes[1],
            None,
            f"own-history / per-instance-standardised, {n_own:,d} windows",
            [t for t in TOOL_LABELS if t in NEW_DESIGN],
        ),
    ]
    for ax, design, title, tools in panels:
        for i, tool in enumerate(tools):
            want = design or NEW_DESIGN[tool]
            sub = fold_df[
                (fold_df["tool"] == tool)
                & (fold_df["variant"] == "all")
                & (fold_df["design"] == want)
            ]
            aucs = sub["roc_auc"].dropna().to_numpy(dtype=float)
            if not len(aucs):
                ax.text(i, 0.5, "refused", ha="center", va="center", fontsize=8, rotation=90)
                continue
            ax.scatter([i] * len(aucs), aucs, color=COLORS[tool], s=42, zorder=3)
            ax.plot([i, i], [aucs.min(), aucs.max()], color=COLORS[tool], lw=1.4, zorder=2)
            ax.scatter([i], [aucs.mean()], marker="_", s=460, color="black", zorder=4)
        ax.axhline(0.5, color="#888888", lw=1.0, ls="--")
        ax.set_xticks(range(len(tools)))
        ax.set_xticklabels(
            [TOOL_LABELS[t][1].replace(" (", "\n(") for t in tools],
            fontsize=7.0,
            rotation=18,
            ha="right",
        )
        ax.set_ylim(0.0, 1.0)
        ax.set_title(title, fontsize=9.5)
        ax.grid(alpha=0.25, axis="y")
    axes[0].set_ylabel("ROC-AUC on the held-out fold")
    fig.suptitle(
        "One point per fold; the black bar is the mean, the dashed line is chance. "
        "The own-history tools have no folds - nothing is fitted across instances - "
        "so they show one point.",
        fontsize=9.5,
    )
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    plt.close(fig)


def plot_refusals(summary: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=160)
    rows = []
    for tool in TOOL_LABELS:
        row = pick(summary, tool, POOLED, "all")
        if row is not None:
            rows.append((f"{TOOL_LABELS[tool][1]}\npooled", tool, row))
        if tool in NEW_DESIGN:
            row = pick(summary, tool, NEW_DESIGN[tool], "all")
            if row is not None:
                rows.append((f"{TOOL_LABELS[tool][1]}\n{NEW_DESIGN[tool]}", tool, row))
    labels = [label for label, _, _ in rows]
    design = [r["n_refused_by_design"] / r["n_windows"] for _, _, r in rows]
    other = [
        (r["n_refused"] - r["n_refused_by_design"]) / r["n_windows"] for _, _, r in rows
    ]
    ax.bar(labels, design, color="#cccccc", label="imposed by the design")
    ax.bar(labels, other, bottom=design, color="#d62728", label="the data could not be scored")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    for i, (d, o) in enumerate(zip(design, other, strict=True)):
        ax.text(i, d + o + 0.015, f"{d + o:.0%}", ha="center", fontsize=7.5)
    ax.set_ylabel("fraction of windows refused")
    ax.set_ylim(0, 1.12)
    ax.tick_params(axis="x", labelsize=6.5)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title(
        "Refusals are results. Grey is the price of the design - baseline windows "
        "and instances\ntoo short to have one; red is a window the data could not "
        "answer for.",
        fontsize=9.5,
    )
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    plt.close(fig)


def plot_paired_deltas(per_instance: pd.DataFrame, out: Path) -> None:
    """post minus pre, per instance, one panel per tool with its own scale.

    The tools' scores are a robust z, a count of rule hits, a Hotelling
    T2, a squared residual and a forest's decision function; putting them
    on one axis would compare a rule hit to a pascal. Each panel keeps its
    own y-axis and the only thing read across panels is the sign.
    """
    tools = [t for t in ALIGNED_TOOLS if t != "clock_flat"]
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 6.4), dpi=160)
    rng = np.random.default_rng(0)
    for ax, tool in zip(axes.ravel(), tools, strict=True):
        frame = per_instance[
            (per_instance["tool"] == tool) & per_instance["pre"].notna()
        ]
        for i, role in enumerate(("post0", "post1")):
            sub = frame[frame[role].notna()]
            delta = sub[role].to_numpy(dtype=float) - sub["pre"].to_numpy(dtype=float)
            jitter = rng.uniform(-0.13, 0.13, len(delta))
            ax.scatter(
                i + jitter,
                delta,
                s=22,
                color=COLORS[tool],
                alpha=0.75,
                edgecolors="none",
                zorder=3,
            )
            share = float((delta > 0).mean()) if len(delta) else float("nan")
            ax.text(
                i,
                1.01,
                f"{share:.0%} above zero",
                transform=ax.get_xaxis_transform(),
                ha="center",
                fontsize=7.5,
            )
        ax.axhline(0.0, color="#333333", lw=1.0)
        ax.set_yscale("symlog", linthresh=1e-3)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["post0 - pre", "post1 - pre"], fontsize=8)
        ax.set_xlim(-0.5, 1.5)
        ax.set_title(ALIGNED_LABELS[tool].replace("**", ""), fontsize=8.5, pad=16)
        ax.tick_params(axis="y", labelsize=6.5)
        ax.grid(alpha=0.25, axis="y")
    fig.suptitle(
        "One point per instance: how much more anomalous the hour after onset scored "
        "than the hour before it.\nSymmetric log scale; the clock control is above zero "
        "for every instance because a later window is always later.",
        fontsize=9.5,
    )
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    plt.close(fig)


def plot_detection_delay(summary: pd.DataFrame, out: Path) -> None:
    """Where the alarm first fires, and how often it fires before onset."""
    tools = [t for t in ALIGNED_TOOLS if t != "clock_flat"]
    fig, ax = plt.subplots(figsize=(9.5, 5.0), dpi=160)
    width = 0.38
    for i, tool in enumerate(tools):
        row = summary[
            (summary["tool"] == tool) & (summary["variant"] == "all")
        ].iloc[0]
        n = float(row["n_instances_with_threshold"])
        at0 = float(row["detection_rate_post0"] or 0.0)
        by1 = float(row["detection_rate_post1"] or 0.0)
        # An instance counts once, at the first post window that fires.
        first0 = float(row["n_detected"]) / n if n else 0.0
        later = max(first0 - at0, 0.0)
        ax.bar(i - width / 2, at0, width, color=COLORS[tool], zorder=3)
        ax.bar(
            i - width / 2, later, width, bottom=at0, color=COLORS[tool], alpha=0.45,
            zorder=3,
        )
        ax.bar(
            i - width / 2, 1.0 - first0, width, bottom=first0, color="#e8e8e8", zorder=3
        )
        ax.bar(i + width / 2, float(row["false_alarm_rate_pre"] or 0.0), width,
               color="#d62728", hatch="///", edgecolor="white", zorder=3)
        ax.text(i - width / 2, first0 + 0.015, f"{by1:.0%} by post1", ha="center",
                fontsize=7)
    ax.set_xticks(range(len(tools)))
    ax.set_xticklabels(
        [ALIGNED_LABELS[t].replace("**", "").replace(", ", ",\n") for t in tools],
        fontsize=7.0,
        rotation=18,
        ha="right",
    )
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("fraction of instances with a threshold")
    ax.set_title(
        "Left bar: the alarm fires in the first hour after onset (solid), the second "
        "(pale), never (grey).\nRight bar, hatched: it already fired on the hour "
        "*before* onset. A tool whose two bars match has no threshold at all.",
        fontsize=9.0,
    )
    ax.grid(alpha=0.25, axis="y", zorder=0)
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------ aligned section


def pct(value, dash: str = "—") -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return dash
    return f"{float(value):.1%}"


def aligned_reading(add, res: Path, clock_row: pd.Series) -> None:
    """The fourth reading: what is left once the clock is taken away."""
    path = res / "aligned_summary.csv"
    if not path.exists():
        return
    summary = pd.read_csv(path)

    def row(tool: str) -> pd.Series:
        return summary[
            (summary["tool"] == tool) & (summary["variant"] == "all")
        ].iloc[0]

    mad, spc, clock, flat = row("mad"), row("spc"), row("clock"), row("clock_flat")
    add(
        "**4. Take the clock away entirely and the ranking holds, the alarm does not.** "
        "Under the onset-aligned design every instance contributes the same six windows "
        "relative to its own fault onset, the clock control falls from "
        f"{clock_row['auc_mean']:.3f} to {clock['roc_auc']:.3f} - and to exactly "
        f"{flat['roc_auc']:.4f} once the fixed spacing the design itself imposes is "
        f"removed. Against that the MAD screen ranks at {fmt(mad['roc_auc'])} and the "
        f"SPC rules at {fmt(spc['roc_auc'])} over "
        f"{int(mad['n_instances_with_threshold'])} instances. But asked for an alarm "
        "rather than a ranking - fire above the worst score of the instance's own three "
        f"baseline hours - the SPC rules catch "
        f"{pct(spc['detection_rate_post1'])} of onsets within two hours at a "
        f"{pct(spc['false_alarm_rate_pre'])} false-alarm rate on the hour before. Three "
        "hours of own history ranks; it does not threshold."
    )
    add("")


def aligned_section(add, res: Path, out_dir: Path, clock_row: pd.Series) -> None:
    """The `## Onset-aligned design` section, or nothing if it was not run."""
    run_path = res / "aligned_run.json"
    if not run_path.exists():
        return
    run = json.loads(run_path.read_text("utf-8"))
    summary = pd.read_csv(res / "aligned_summary.csv")
    per_instance = pd.read_csv(res / "aligned_per_instance.csv")
    prov = run["provenance"]
    manifest = prov["aligned_manifest"]
    onsets = prov["onsets"]
    refusals = prov["refusals"]
    roles = manifest["row_label_by_role"]
    k = int(manifest["baseline_windows"])

    plot_paired_deltas(per_instance, out_dir / "05_paired_deltas.png")
    plot_detection_delay(summary, out_dir / "06_detection_delay.png")

    def row(tool: str, variant: str = "all") -> pd.Series | None:
        sub = summary[(summary["tool"] == tool) & (summary["variant"] == variant)]
        return None if not len(sub) else sub.iloc[0]

    clock_aligned = row("clock")
    flat = row("clock_flat")
    mad, spc = row("mad"), row("spc")

    add("## Onset-aligned design")
    add("")
    add(
        "Everything above scores a window against a target whose positives arrive "
        f"late in the record, and the clock control reads {clock_row['auc_mean']:.3f} "
        "on it. That is a property of the *design*, not of the corpus: the windows "
        "each instance contributes are cut from its first sample, so instances whose "
        "fault arrives at hour 3 and instances whose fault arrives at hour 40 are "
        "pooled and position becomes a predictor. This section removes that. Every "
        "instance is cut relative to *its own onset*, so each contributes the same "
        "relative positions and no instance's fault is earlier in the evaluation "
        "than another's."
    )
    add("")
    add("```console")
    add("uv run python examples/studies/3w_detectors/build_onsets.py --jobs 14")
    add("uv run python examples/studies/3w_detectors/build_windows.py --aligned --jobs 14")
    add("uv run python examples/studies/3w_detectors/run_detectors.py --aligned")
    add("```")
    add("")
    add("### The six windows")
    add("")
    add(
        "`t_onset` is the timestamp of an instance's first row carrying a fault class "
        "- 1-9 steady or 101-109 transient. Relative to it, each evaluable instance "
        "contributes:"
    )
    add("")
    add("| offset from onset | role | the design's label |")
    add("|---|---|---|")
    built = {int(s["offset"]): s for s in manifest["offsets"]}
    lowest = min(built)
    for hours in range(lowest, max(built) + 1):
        spec = built.get(hours)
        if spec is None:
            add(f"| [{hours:+d} h, {hours + 1:+d} h) | *guard, not built* | none |")
            continue
        add(
            f"| [{hours:+d} h, {hours + 1:+d} h) | {spec['role']} | "
            + ("positive" if spec["design_label"] else "negative")
            + " |"
        )
    add("")
    add(
        f"The guard hour is dropped on purpose. A fault that shows in the tags a few "
        "minutes before its first labelled row would otherwise land inside the window "
        "the design calls negative, and the negative would be doing the detector's "
        f"work. The {k} baseline windows are the own-history baseline every per-tag "
        "limit below is fitted on; they are chosen by position relative to onset and "
        "never by label."
    )
    add("")
    add(
        "The design asserts *pre* is normal and *post0*/*post1* are faulted. The rows "
        "were read anyway and they agree: "
        f"{roles['pre']['row_positive']} of {roles['pre']['windows']} pre windows and "
        f"{roles['baseline']['row_positive']} of {roles['baseline']['windows']} "
        f"baseline windows carry a fault row, "
        f"{roles['post0']['row_positive']} of {roles['post0']['windows']} post0 and "
        f"{roles['post1']['row_positive']} of {roles['post1']['windows']} post1 do. "
        f"{manifest['n_test_windows_whose_rows_disagree_with_the_design_label']} test "
        "windows disagree with the label the design gives them."
    )
    add("")
    add(
        f"**The positives are the run-up, not the fault.** "
        f"{roles['post1']['row_transient_only']} of the {roles['post1']['windows']} "
        "post1 windows contain only transient rows. Two hours after onset the fault "
        "has almost never settled into its steady class, so this design asks whether "
        "a detector sees the *beginning* of a fault within one to two hours - which is "
        "a different and harder question than the one the unaligned tables above ask."
    )
    add("")

    add("### The evaluable set")
    add("")
    add(
        f"Of {onsets['n_instances']:,d} real instances, {onsets['n_with_onset']:,d} "
        f"carry a fault row at all; the other {onsets['n_without_onset']:,d} are the "
        "negatives-only pool - all of folder 0 and the folder-9 instances the profile study found "
        "carrying no row labelled 9 or 109. An instance is *evaluable* when it has "
        f"{onsets['pre_onset_windows_required']} whole windows of record before onset "
        f"and {onsets['post_onset_windows_required']} after it:"
    )
    add("")
    add("| event folder | instances | with an onset | ≥3 h pre and ≥2 h post | evaluable |")
    add("|---|---:|---:|---:|---:|")
    for folder, counts in sorted(onsets["per_folder"].items(), key=lambda kv: int(kv[0])):
        add(
            f"| {folder} | {counts['instances']:,d} | {counts['with_onset']:,d} | "
            f"{counts['k_pre_and_2_post']:,d} | {counts['evaluable']:,d} |"
        )
    add(
        f"| **all** | {onsets['n_instances']:,d} | {onsets['n_with_onset']:,d} | "
        f"{onsets['n_with_k_pre_and_2_post']:,d} | {onsets['n_evaluable']:,d} |"
    )
    add("")
    add(
        f"{onsets['n_evaluable']} instances over {onsets['n_evaluable_wells']} wells, "
        f"{manifest['n_windows']} windows. The gap between the two right-hand columns "
        f"is the two extra windows the design needs beyond the {k} of baseline - one "
        "scored pre window and one guard hour."
    )
    add("")
    add(
        f"**Every evaluable instance has a transient onset.** "
        f"{onsets['n_evaluable_transient_onset']} of {onsets['n_evaluable']}, against "
        f"{onsets['n_evaluable_steady_onset']} with a steady one, out of "
        f"{onsets['n_onset_transient']} transient and {onsets['n_onset_steady']} steady "
        "onsets in the corpus. That is not a sampling accident: an instance whose first "
        "fault row is a steady class is one whose fault was already there when labelling "
        f"started, and {onsets['n_with_onset'] - onsets['n_with_k_pre_windows']:,d} of "
        f"the {onsets['n_with_onset']:,d} instances with an onset have less than "
        f"{k} h of record before it. The transient-versus-steady sensitivity asked for "
        "below therefore has one arm and no other, and that is the finding."
    )
    add("")

    add("### What the clock is worth now")
    add("")
    add(
        f"The clock control reads the same quantity it read above - how many windows "
        f"into its own record the window sits, no sensor value - and scores "
        f"**{clock_aligned['roc_auc']:.3f}** here against "
        f"{clock_row['auc_mean']:.3f} under the own-history design. The residual is "
        "not noise and it is not signal: it is the fixed two-to-three window spacing "
        "the design itself puts between the pre window and the post windows. Take that "
        "spacing back out - score every window by `window_index - onset_offset`, which "
        "is where the instance's onset sits in its own record and is the same number "
        f"for all six of its windows - and the AUC is exactly "
        f"**{flat['roc_auc']:.4f}**, every comparison a tie. The clock is uninformative "
        "under this design up to a constant the design hands it, and the row is in the "
        "table below so that claim can be checked rather than believed."
    )
    add("")

    add("### What each tool scored")
    add("")
    add(
        "Same tools, same thresholds, nothing retuned. **AUC** pools every scored pre "
        "window as a negative and every scored post window as a positive. **Paired hit** "
        "is the fraction of instances whose post window outscores that instance's own "
        "pre window. **FAR** and the detection rates use each instance's own threshold: "
        "the largest score the tool gave that instance's baseline windows, fired on "
        "strictly above."
    )
    add("")
    k_base = int(manifest["baseline_windows"])
    far_floor = 1.0 / (k_base + 1)
    add(
        f"That threshold rule carries a false-alarm floor. With {k_base} baseline "
        "windows, a pre window that is exchangeable with them - drawn from the same "
        "distribution, the tool reading nothing real - is equally likely to land in "
        f"any of the {k_base + 1} rank positions, so it exceeds all {k_base} of them "
        f"with probability 1/{k_base + 1} = **{pct(far_floor)}**. That is the number "
        "every FAR below has to be read against: it is what a tool that has learned "
        "nothing scores, not zero. The **over floor** column is the gap."
    )
    add("")
    add(
        "| tool | instances | AUC | paired hit post0 | post1 | FAR on pre | over floor | "
        "detect post0 | detect post1 | median delay |"
    )
    add("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for tool in ALIGNED_TOOLS:
        r = row(tool)
        if r is None:
            continue
        delay = r["median_delay_windows"]
        far = r["false_alarm_rate_pre"]
        over = (
            "n/a"
            if pd.isna(far)
            else f"{100.0 * (far - far_floor):+.1f} pt"
        )
        add(
            f"| {ALIGNED_LABELS[tool]} | {int(r['n_instances_with_threshold'])} | "
            f"{fmt(r['roc_auc'])} | {fmt(r['paired_hit_post0'])} | "
            f"{fmt(r['paired_hit_post1'])} | {pct(far)} | {over} | "
            f"{pct(r['detection_rate_post0'])} | {pct(r['detection_rate_post1'])} | "
            + ("never fires" if pd.isna(delay) else f"{delay:.0f} window(s)")
            + " |"
        )
    add("")
    spc_far = row("spc")["false_alarm_rate_pre"]
    add(
        f"SPC has the lowest FAR of the five detectors at {pct(spc_far)}, and that is "
        f"still {100.0 * (spc_far - far_floor):.1f} points above the "
        f"{pct(far_floor)} floor: on {pct(spc_far)} of instances a window taken "
        "before the fault beat the worst of that instance's own three quiet hours. A "
        "detector whose FAR sat *at* the floor would be firing exactly as often as "
        "chance; none of these do."
    )
    add("")
    uneven = []
    for tool in ALIGNED_TOOLS:
        r = row(tool)
        n = int(r["n_instances_with_threshold"])
        for label, column in (
            ("pre", "n_pre_at_threshold"),
            ("post0", "n_post0_at_threshold"),
            ("post1", "n_post1_at_threshold"),
        ):
            if int(r[column]) != n:
                uneven.append(
                    f"{ALIGNED_LABELS[tool].replace('**', '')} has a threshold for {n} "
                    f"instances but a scored {label} window for {int(r[column])}"
                )
    if uneven:
        add(
            "Each rate is over the instances that have both a threshold and that "
            "window, and those counts are not always the same number: "
            + "; ".join(uneven)
            + ". The rest of the rates are over the instance count in the second column."
        )
        add("")
    add(
        "The three per-instance-standardised tools fit one model per fold under 5-fold "
        f"group holdout by well over the {onsets['n_evaluable_wells']} evaluable wells, "
        "on the training instances' baseline and pre windows only - no post window "
        "reaches any fit. Their pooled AUC therefore ranks scores five separate models "
        "produced; the per-fold range is "
        + ", ".join(
            f"{ALIGNED_LABELS[t].split(',')[0]} {fmt(row(t)['auc_fold_min'])}-"
            f"{fmt(row(t)['auc_fold_max'])}"
            for t in ("mspc_t2", "mspc_spe", "iforest")
        )
        + ". The own-history tools fit nothing across instances and have no folds."
    )
    add("")
    std_far = ", ".join(
        f"{ALIGNED_LABELS[t].split(',')[0].replace('**', '')} "
        f"{pct(row(t)['false_alarm_rate_pre'])}"
        for t in ("mspc_t2", "mspc_spe", "iforest")
    )
    add(
        f"**Those three FARs ({std_far}) are a design limit, not a measurement of the "
        "tools.** `_instance_zscore` takes each column's centre and scale from the "
        f"same {k_base} baseline windows the threshold is then read off. After that "
        "standardisation those windows are, by construction, the most ordinary rows "
        "the model will ever see for that instance - centred at zero with unit spread "
        "- while every pre and post window is scored without having contributed to "
        "the scaling. Setting the alarm at the maximum of the three in-fit rows and "
        "testing out-of-fit rows against it is an unfair comparison in a fixed "
        "direction, and it pushes the FAR toward 100% independently of whether the "
        "tool sees anything. The correction is a leave-one-out threshold: "
        "standardise each baseline window against the other "
        f"{k_base - 1} and take the alarm from those held-out scores. It is not done "
        "here because `score_mspc` and `score_iforest` fit and score in a single call "
        "and do not hand back a fitted model, so every left-out window needs its own "
        "refit; that is a change to the scoring interface rather than a few lines in "
        "this report. Until it exists, read the standardised tools' AUC and paired hit "
        "rate - which never compare in-fit rows against out-of-fit ones - and treat "
        "their FAR as unmeasured."
    )
    add("")
    add("![paired score deltas](out/05_paired_deltas.png)")
    add("")
    add(
        "**Read the paired hit rate against the clock's row.** The clock wins it "
        f"{clock_aligned['paired_hit_post0']:.0%} of the time, because a later window "
        "is always later. Any score that rises with time takes that column outright, so "
        "it says a tool is not *worse* than the clock rather than that it is better."
    )
    add("")

    add("### Detection at each instance's own threshold")
    add("")
    add("![detection delay](out/06_detection_delay.png)")
    add("")
    add(
        f"This is the deployable question and it is where the result is thin. With the "
        f"alarm set at the worst of the {k} baseline hours, the SPC rules fire on "
        f"{pct(spc['false_alarm_rate_pre'])} of the hours *before* onset while catching "
        f"{pct(spc['detection_rate_post1'])} of instances by the second hour after it; "
        f"the MAD screen catches more ({pct(mad['detection_rate_post1'])}) and false-"
        f"alarms more ({pct(mad['false_alarm_rate_pre'])}). Neither is a usable alarm "
        "rate. Every detected instance is detected in the first hour: the median delay "
        "is 0 windows for every tool that detects anything, so the second post window "
        "adds coverage rather than speed."
    )
    add("")
    add(
        "**The three standardised tools have no threshold at all.** IsolationForest "
        f"fires on {pct(row('iforest')['false_alarm_rate_pre'])} of pre windows - "
        f"exactly what the clock control does - and MSPC SPE on "
        f"{pct(row('mspc_spe')['false_alarm_rate_pre'])}. "
        "The mechanism is in the design: each column is z-scored against "
        f"that instance's own {k} baseline windows, so the baseline rows are the very "
        "rows the scale was fitted to and land in a band nothing else can enter. Across "
        "all 48 instances IsolationForest scores every baseline window between -0.190 "
        "and -0.143 and every pre window above its own instance's baseline maximum. A "
        f"threshold taken from {k} in-sample points is not a threshold, and the "
        "false-alarm column is what says so."
    )
    add("")

    add("### Per event folder, and folder 9")
    add("")
    folders = sorted(manifest["per_folder_instances"], key=int)
    add(
        "| tool | " + " | ".join(f"folder {f}" for f in folders)
        + " | folder 9 excluded |"
    )
    add("|---|" + "---:|" * (len(folders) + 1))
    for tool in ALIGNED_TOOLS:
        cells = []
        for folder in folders:
            r = row(tool, f"folder_{folder}")
            cells.append(
                "refused"
                if r is None or pd.isna(r["roc_auc"])
                else f"{r['roc_auc']:.3f}"
            )
        r9 = row(tool, "no_folder9")
        cells.append("refused" if r9 is None or pd.isna(r9["roc_auc"]) else f"{r9['roc_auc']:.3f}")
        add(f"| {ALIGNED_LABELS[tool]} | " + " | ".join(cells) + " |")
    add("")
    add(
        "Folder 9's 7 evaluable instances are the ones that *do* carry a fault row, so "
        "they are not the label-noise case the profile study found; dropping them moves nothing by "
        f"more than {abs(float(mad['roc_auc']) - float(row('mad', 'no_folder9')['roc_auc'])):.3f} "
        "AUC. The single folder-6 instance carries no ranking metric worth reading and "
        "is shown for completeness."
    )
    add("")

    add("### Refusals under this design")
    add("")
    add(
        f"**{refusals['n_instances_refused_by_design']:,d} of "
        f"{refusals['n_instances_in_corpus']:,d} instances are refused before any "
        f"detector runs**: {refusals['design_refusal_reason']}. That is the price of "
        "the design and it is large - larger than the unaligned design's, and for the "
        "same underlying reason, that 3W ships hour-scale excerpts rather than a "
        "historian. On a live archive an instance has months of its own past and the "
        "refusal disappears; on this corpus it leaves "
        f"{refusals['n_instances_evaluable']} instances."
    )
    add("")
    add("| tool | instances scored | refused from the data | why |")
    add("|---|---:|---:|---|")
    why = {
        "mspc_t2": "the instance has a window missing one of the six common variables, "
        "so no matrix can be aligned without a hole in it (`MspcAlignmentError`)",
        "mspc_spe": "same alignment refusal as T2",
    }
    for tool in ALIGNED_TOOLS:
        r = row(tool)
        if r is None:
            continue
        n_ref = int(r["n_instances_refused"])
        add(
            f"| {ALIGNED_LABELS[tool]} | {int(r['n_instances_with_threshold'])} | "
            f"{n_ref} | "
            + (why.get(tool, "nothing left to refuse") if n_ref else "nothing left to refuse")
            + " |"
        )
    add("")

    add("### What the aligned numbers say")
    add("")
    add(
        "**It does not beat the 0.923 the unaligned tables report for the SPC rules on "
        "steady faults - it replaces the question.** That number was measured with "
        "transient-only windows *excluded*, so its positives are settled faults. Here "
        f"{roles['post1']['row_transient_only']} of {roles['post1']['windows']} "
        "positives are transient-only, because two hours after onset is where the "
        "run-up lives. The two numbers are not two answers to one question."
    )
    add("")
    add(
        "What this design does settle is how much of an own-history AUC was the clock. "
        f"Under it the clock falls from {clock_row['auc_mean']:.3f} to "
        f"{clock_aligned['roc_auc']:.3f}, and to exactly {flat['roc_auc']:.4f} once the "
        "design's own fixed spacing is removed. Against that floor the MAD screen ranks "
        f"at {fmt(mad['roc_auc'])} and the SPC rules at {fmt(spc['roc_auc'])}, both on "
        f"all {int(mad['n_instances_with_threshold'])} evaluable instances with nothing "
        "refused. Those are the first numbers in this study that are about the tags "
        "rather than the timetable."
    )
    add("")
    add(
        "And the deployable reading is negative: at a threshold set from "
        f"{k} hours of the instance's own baseline, the best false-alarm rate any tool "
        f"reaches is {pct(spc['false_alarm_rate_pre'])}, on the hour immediately before "
        "onset. Three hours of history is enough to rank windows and not enough to set "
        "a limit."
    )
    add("")


# ------------------------------------------------------------------- report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=str(HERE / "results"))
    parser.add_argument("--out", default=str(HERE / "out"))
    args = parser.parse_args(argv)

    res = Path(args.results)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    run = json.loads((res / "run.json").read_text("utf-8"))
    fold_df = pd.read_csv(res / "per_fold.csv")
    summary = pd.read_csv(res / "summary.csv")
    scores = pd.read_csv(res / "scores.csv")
    windows = pd.read_csv(res / "windows.csv")

    prov = run["provenance"]
    wm = prov["windows_manifest"]
    well_split = prov["splits"]["well"]
    own = prov["own_history"]
    variables = prov["common_variables"]
    pop_well = run["population"]["well"]
    pop_inst = run["population"]["instance"]
    labels = windows.set_index("window_key")["label"]

    plot_roc(scores, windows, out_dir / "01_roc_per_fold.png")
    plot_same_population(
        summary, out_dir / "02_same_population.png", own["n_scored_windows"],
        own["baseline_windows"],
    )
    plot_fold_spread(
        fold_df, out_dir / "03_fold_spread.png", prov["n_windows"], own["n_scored_windows"]
    )
    plot_refusals(summary, out_dir / "04_refusals.png")

    lines: list[str] = []
    add = lines.append
    add("# Stages 3-7 on real data: 3W v2.0.0, three study designs")
    add("")
    add(
        "The first version of this study fitted every per-tag limit on training-fold "
        "windows pooled across wells and scored held-out wells with it. For the two "
        "tools whose object is a limit *for one tag* that is a study-design error, not "
        "a detector result, and it is corrected here. Both designs are run, both are "
        "reported, and the pooled rows are kept so the difference is visible."
    )
    add("")
    add(
        "Correcting it exposed a second problem, which is what the third design is for. "
        "A control that reads no sensor value at all - only how many windows into its "
        "own record a window sits - scores above every detector on the corrected design, "
        "because a 3W instance opens normal and its fault arrives late. "
        "[The onset-aligned design](#onset-aligned-design) cuts every instance relative "
        "to its own fault onset instead, so that control has nothing left to read."
    )
    add("")

    add("## Provenance")
    add("")
    add("| field | value |")
    add("|---|---|")
    add("| dataset | `petrobras/3W` v2.0.0 (CC BY 4.0), real `WELL-*` instances |")
    add(f"| manifest sha256 | `{prov['dataset_manifest_sha256']}` |")
    add(
        f"| code | tsdive {prov['code']['tsdive_version']} @ "
        f"`{prov['code']['git_head'][:8]}` |"
    )
    add(
        f"| pooled split | 5-fold group holdout by well, seed {well_split['seed']}, "
        f"stratified on folder label |"
    )
    add(
        f"| own-history split | first {own['baseline_windows']} windows of each "
        f"instance, label-blind; all {own['n_instances_with_windows']:,d} instances "
        "carrying a window, no group holdout |"
    )
    add(f"| windows | {wm['n_windows']:,d} of {wm['window_s']:.0f} s over "
        f"{wm['n_instances']:,d} instances |")
    add(f"| positive rate | {prov['positive_rate']:.1%} "
        f"({prov['n_positive']:,d} of {prov['n_windows']:,d}) |")
    add(f"| common variables | {', '.join(f'`{v}`' for v in variables)} |")
    add(f"| runtime | {wm['wall_seconds']:.0f} s to build windows on {wm['jobs']} "
        f"processes, {prov['wall_seconds']:.0f} s to score |")
    add("")
    add("```console")
    add("uv run python examples/studies/3w_detectors/build_windows.py --jobs 14")
    add("uv run python examples/studies/3w_detectors/run_detectors.py")
    add("uv run python examples/studies/3w_detectors/make_report.py")
    add("```")
    add("")
    add("### The folds")
    add("")
    add(
        "Wells are assigned whole, largest first. `WELL-00002` alone carries "
        f"{max(wm['per_well_windows'].values()):,d} windows, so the fold holding it is "
        "several times the size of its siblings; that lopsidedness is the shape of 3W, "
        "not a defect of the split. `GroupSplit.leakage_check()` runs before any model "
        "is fitted."
    )
    add("")
    add("| fold | windows | wells held out |")
    add("|---:|---:|---|")
    for i, wells in enumerate(well_split["folds"]):
        add(f"| {i} | {well_split['fold_sizes'][i]:,d} | {', '.join(f'`{w}`' for w in wells)} |")
    add("")

    add("## Two designs")
    add("")
    add(
        "**pooled-cross-well.** Five folds of `group_holdout` over the 40 wells; every "
        "model is fitted on the training folds' windows, pooled across wells, and "
        "scored on the fold under test."
    )
    add("")
    add(
        "**own-history.** For each instance and variable the baseline is that "
        f"instance's first {own['baseline_windows']} windows, taken by position and "
        "never by label, and every later window of the same instance is scored against "
        "it. Group holdout is irrelevant here and is not run: nothing is fitted across "
        "instances, so there is no cross-instance parameter for a holdout to protect. "
        "The split is stated as what it is - own history, first "
        f"{own['baseline_windows']} windows, label-blind, all "
        f"{own['n_instances_with_windows']:,d} instances."
    )
    add("")
    mad_pooled = pick(summary, "mad", POOLED, "all")
    worst = fold_df[
        (fold_df["tool"] == "mad")
        & (fold_df["design"] == POOLED)
        & (fold_df["variant"] == "all")
    ].sort_values("roc_auc").iloc[0]
    add(
        "### Why the pooled numbers for stages 3 and 5 are a design artefact"
    )
    add("")
    add(
        "`mad_baseline` returns a median and a MAD for one tag. Pooling the training "
        "folds' window medians across wells and calling the result a baseline for "
        "`P-PDG` assumes `P-PDG` means the same number on every well. It does not: the "
        "six common variables are absolute pressures and a temperature, and their "
        "levels differ between wells by orders of magnitude. The robust z that comes "
        "out therefore answers *how far is this well from the other wells*, which has "
        "nothing to do with whether this hour is abnormal for this tag."
    )
    add("")
    add(
        f"That is what the fold spread was: the pooled MAD screen scores "
        f"{mad_pooled['auc_mean']:.3f} on average and "
        f"**{worst['roc_auc']:.3f} on fold {int(worst['fold'])}** - well below chance, "
        "which is not noise but a sign flip. Hold out a group of wells whose normal "
        "level sits further from the pooled median than the faulted windows of the "
        "training wells and the ranking inverts. The earlier version of this report "
        "read that as the classics failing on 3W. It was the study asking them the "
        "wrong question."
    )
    add("")
    add(
        "Stages 6 and 7 fit a model across instances either way, so they keep the "
        "5-fold well holdout in both designs and differ only in what a column means: "
        "pooled autoscaling on training statistics, or a z-score against the same "
        "instance's own first-k-window mean and standard deviation."
    )
    add("")
    add(
        f"The own-history design scores {own['n_scored_windows']:,d} of "
        f"{prov['n_windows']:,d} windows. The rest are the price of it: "
        f"{own['n_baseline_windows']:,d} baseline windows and "
        f"{prov['n_windows'] - own['n_scored_windows'] - own['n_baseline_windows']:,d} "
        f"windows in the {own['n_instances_too_short']:,d} instances of "
        f"{own['baseline_windows']} windows or fewer, which get a typed refusal and no "
        "score. Those windows are not comparable to the pooled design's, so every "
        "pooled tool is also read back over exactly the "
        f"{own['n_scored_windows']:,d} own-history windows and that row is labelled "
        "`pooled-cross-well, own-history windows`. Only those two rows differ in method "
        "alone."
    )
    add("")

    add("## Windows and labels")
    add("")
    add(
        f"Each instance is cut into non-overlapping {wm['window_s']:.0f} s windows aligned "
        f"to its own first sample. The {wm['n_windows_dropped_unlabelled_prefix']:,d} windows "
        "overlapping the unlabelled 3,600-row prefix are dropped - exactly one per "
        "instance, which is what a 3,600-row prefix and a 3,600 s window come to. A "
        "window is positive when any row inside it carries a fault class (1-9 steady, "
        "101-109 transient)."
    )
    add("")
    add("| event folder | windows | positive | own-history scored | of those, positive |")
    add("|---|---:|---:|---:|---:|")
    for folder, counts in sorted(wm["per_folder"].items(), key=lambda kv: int(kv[0])):
        scored = own["scored_per_folder"].get(folder)
        add(
            f"| {folder} | {counts['windows']:,d} | {counts['positive']:,d} | "
            + (f"{scored['windows']:,d} | {scored['positive']:,d} |" if scored else "0 | 0 |")
        )
    add("")
    add(
        f"Folder 4 disappears from the own-history column entirely: every one of its "
        "instances is four windows or shorter, so none has a baseline and a window "
        "left over. That is a real limit of the design on this corpus, not a rounding "
        "detail - the own-history metric is computed over a corpus with a "
        f"{own['scored_positive_rate']:.1%} positive rate against the pooled design's "
        f"{prov['positive_rate']:.1%}."
    )
    add("")
    add(
        f"{wm['n_transient_only']:,d} of the {wm['n_positive']:,d} positive windows "
        f"({wm['n_transient_only'] / max(wm['n_positive'], 1):.0%}) contain only transient "
        "rows: the run-up to a fault, not the fault. Folder 9 is the label-noise case "
        "the profile study found - its real instances carry no row labelled 9 or 109, so "
        f"{wm['per_folder']['9']['windows'] - wm['per_folder']['9']['positive']:,d} of its "
        f"{wm['per_folder']['9']['windows']:,d} windows are negatives sitting in a folder "
        "named for an event. Both are held out of the metric in the variants below."
    )
    add("")
    add("### Baselines that are themselves fault windows")
    add("")
    add(
        "Folders 3 and 4 are all-positive, so an instance's first "
        f"{own['baseline_windows']} windows can be fault windows. "
        f"{own['n_instances_with_fully_positive_baseline']} of the "
        f"{len(own['scored_per_folder']) and own['n_instances_scored']:,d} scored "
        "instances have a baseline every window of which is labelled positive, "
        f"covering {own['n_scored_windows_in_those_instances']} scored windows. The "
        "`no_positive_baseline` row of every table below drops those instances; on this "
        "corpus it moves nothing by more than 0.01 AUC, which is what "
        f"{own['n_scored_windows_in_those_instances']} windows in "
        f"{own['n_scored_windows']:,d} can do."
    )
    add("")
    add("### The common variable set")
    add("")
    add(
        "Six measurement variables are present in at least 60% of real instances, read "
        "off each instance's own `present_variables` list: "
        + ", ".join(f"`{v}`" for v in variables)
        + ". `T-PDG` (50.3%) and `T-JUS-CKP` (50.2%) miss the threshold, so the set is "
        "six, not eight. `ESTADO-*` and the two `LABEL_*` tags are `role=MODE` and carry "
        "state names rather than measurements, so no numeric feature is computed for "
        "them at all."
    )
    add("")

    add("## What each tool scored")
    add("")
    add(
        "ROC-AUC with the fold-to-fold range in brackets where there are folds. The "
        "own-history tools have none - nothing is fitted across instances - so they "
        "carry a single number over all "
        f"{own['n_scored_windows']:,d} scored windows."
    )
    add("")
    add(
        "| stage | tool | design | windows scored | ROC-AUC | PR-AUC | "
        "precision@recall 0.5 | refused |"
    )
    add("|---|---|---|---:|---|---:|---:|---:|")
    for tool, (stage, label) in TOOL_LABELS.items():
        designs = [POOLED]
        if tool in NEW_DESIGN:
            designs += [MATCHED, NEW_DESIGN[tool]]
        for design in designs:
            row = pick(summary, tool, design, "all")
            if row is None:
                continue
            note = " *(instance holdout)*" if tool == "population" else ""
            add(
                f"| {stage} | {label}{note} | {design} | {int(row['n_scored']):,d} | "
                f"{auc_cell(row)} | {fmt(row['pr_auc_mean'])} | "
                f"{fmt(row['precision_at_recall_50_mean'])} | "
                f"{row['refused_frac_mean']:.1%} |"
            )
    clock = pick(summary, "clock", OWN, "all")
    add(
        f"| none | **clock control** (window position, no sensor read) | {OWN} | "
        f"{int(clock['n_scored']):,d} | {auc_cell(clock)} | "
        f"{fmt(clock['pr_auc_mean'])} | {fmt(clock['precision_at_recall_50_mean'])} | "
        f"{clock['refused_frac_mean']:.1%} |"
    )
    add("")
    add("![same population, two designs](out/02_same_population.png)")
    add("")
    add(
        "**The design change is worth more than any of the detectors.** On the "
        f"identical {own['n_scored_windows']:,d} windows the MAD screen goes from "
        f"{pick(summary, 'mad', MATCHED, 'all')['auc_mean']:.3f} to "
        f"{pick(summary, 'mad', OWN, 'all')['auc_mean']:.3f}, the SPC rules from "
        f"{pick(summary, 'spc', MATCHED, 'all')['auc_mean']:.3f} to "
        f"{pick(summary, 'spc', OWN, 'all')['auc_mean']:.3f}, IsolationForest from "
        f"{pick(summary, 'iforest', MATCHED, 'all')['auc_mean']:.3f} to "
        f"{pick(summary, 'iforest', STANDARDISED, 'all')['auc_mean']:.3f}, T2 from "
        f"{pick(summary, 'mspc_t2', MATCHED, 'all')['auc_mean']:.3f} to "
        f"{pick(summary, 'mspc_t2', STANDARDISED, 'all')['auc_mean']:.3f}. Nothing "
        "about the detectors changed between those two columns; only the question did."
    )
    add("")

    add("### The clock control, and what is left of the gain")
    add("")
    add(
        f"The clock reads no sensor value. Its score is how many windows into its own "
        f"instance the window sits, and on the same population it scores "
        f"{clock['auc_mean']:.3f} - above every detector in the table. A 3W instance "
        "opens normal and the fault, once it arrives, stays, so position alone "
        "separates the classes; and the own-history design removes each instance's "
        "opening windows, which are the normal ones. Any own-history number below "
        f"{clock['auc_mean']:.3f} on the all-windows target is measuring when the "
        "window happened at least as much as what the tag did."
    )
    add("")
    add(
        "The cut that breaks the clock is the transient one. Drop the transient-only "
        "windows - the run-up, where the fault has not arrived and time is all there is "
        f"- and the clock falls to "
        f"{pick(summary, 'clock', OWN, 'no_transients')['auc_mean']:.3f}, while the "
        "own-history SPC rules rise to "
        f"{pick(summary, 'spc', OWN, 'no_transients')['auc_mean']:.3f} "
        f"({pick(summary, 'spc', OWN, 'no_folder9_no_transients')['auc_mean']:.3f} with "
        "folder 9 also out). That gap is the part of the result that is about the "
        "signal:"
    )
    add("")
    add("| tool | design | all windows | transient-only excluded | both cuts |")
    add("|---|---|---:|---:|---:|")
    for tool in ("clock", *NEW_DESIGN):
        design = OWN if tool == "clock" else NEW_DESIGN[tool]
        label = "clock control" if tool == "clock" else TOOL_LABELS[tool][1]
        cells = [
            fmt(pick(summary, tool, design, v)["auc_mean"])
            for v in ("all", "no_transients", "no_folder9_no_transients")
        ]
        add(f"| {label} | {design} | " + " | ".join(cells) + " |")
    add("")

    add("### Per fold, and per event folder")
    add("")
    add("| tool | design | " + " | ".join(f"fold {i}" for i in range(5)) + " |")
    add("|---|---|" + "---:|" * 5)
    for tool, (_, label) in TOOL_LABELS.items():
        for design in [POOLED] + ([NEW_DESIGN[tool]] if tool in NEW_DESIGN else []):
            if design == OWN:
                continue
            sub = fold_df[
                (fold_df["tool"] == tool)
                & (fold_df["variant"] == "all")
                & (fold_df["design"] == design)
                & (fold_df["split"] == POOLED_SPLIT.get(tool, "well"))
            ].set_index("fold")
            cells = []
            for i in range(5):
                if i not in sub.index:
                    cells.append("—")
                    continue
                value = sub.loc[i, "roc_auc"]
                cells.append("refused" if pd.isna(value) else f"{value:.3f}")
            add(f"| {label} | {design} | " + " | ".join(cells) + " |")
    add("")
    add(
        "The own-history tools have no folds. Their equivalent breakdown is by event "
        "folder, over the same scored windows - a folder that is all-positive or "
        "all-negative among the scored windows has no ranking metric and says so:"
    )
    add("")
    folders = sorted(own["scored_per_folder"], key=int)
    add("| tool | overall | " + " | ".join(f"folder {f}" for f in folders) + " |")
    add("|---|---:|" + "---:|" * len(folders))
    folder_of = windows.set_index("window_key")["folder_label"].astype(str)
    for tool in ("mad", "spc", "clock"):
        sub = scores[(scores["tool"] == tool) & (scores["design"] == OWN)]
        label = "clock control" if tool == "clock" else TOOL_LABELS[tool][1]
        cells = [fmt(auc(labels, sub))]
        for folder in folders:
            part = sub[folder_of.loc[sub["window_key"]].to_numpy() == folder]
            value = auc(labels, part)
            cells.append("one class" if value is None else f"{value:.3f}")
        add(f"| {label} | " + " | ".join(cells) + " |")
    add("")
    add("![ROC per fold](out/01_roc_per_fold.png)")
    add("")
    add("![per-fold spread](out/03_fold_spread.png)")
    add("")

    aligned_section(add, res, out_dir, clock)

    add("## Refusals")
    add("")
    add(
        "Two kinds, counted separately. A refusal *imposed by the design* is a baseline "
        "window or a window in an instance too short to have a baseline; the own-history "
        "design cannot score those by construction. A refusal *from the data* is a "
        "window the tool was asked about and could not answer for."
    )
    add("")
    add("| tool | design | refused | imposed by the design | why the rest |")
    add("|---|---|---:|---:|---|")
    reasons = {
        ("mspc_t2", POOLED): "the window is missing one of the six common variables, so "
        "no matrix can be aligned without a hole in it (`MspcAlignmentError`)",
        ("mspc_spe", POOLED): "same alignment refusal as T2",
        ("mspc_t2", STANDARDISED): "the same alignment refusal, plus a sub-group cell "
        "whose (instance, variable) had no usable baseline statistics",
        ("mspc_spe", STANDARDISED): "same as T2",
        ("regime", POOLED): "the window's modal `LABEL_state` never reached 20 training "
        "windows for that variable, or never appeared in training at all "
        "(`RegimeTooSparse`)",
        ("population", POOLED): "the (well, variable) pair has fewer than 20 training "
        "windows; under well holdout this is every window (`PopulationTooSparse`)",
    }
    for tool, (_, label) in TOOL_LABELS.items():
        for design in [POOLED] + ([NEW_DESIGN[tool]] if tool in NEW_DESIGN else []):
            row = pick(summary, tool, design, "all")
            if row is None:
                continue
            rest = row["n_refused"] - row["n_refused_by_design"]
            why = reasons.get((tool, design), "nothing: every scored window carried at "
                              "least one scorable variable")
            add(
                f"| {label} | {design} | {row['refused_frac_mean']:.1%} | "
                f"{row['n_refused_by_design'] / row['n_windows']:.1%} | "
                + (why if rest else "nothing left to refuse")
                + " |"
            )
    add("")
    add("![refusal rates](out/04_refusals.png)")
    add("")
    add(
        "The per-instance-standardised MSPC refuses "
        f"{pick(summary, 'mspc_t2', STANDARDISED, 'all')['refused_frac_mean']:.1%} of "
        "windows against the pooled design's "
        f"{pick(summary, 'mspc_t2', POOLED, 'all')['refused_frac_mean']:.1%}, and the "
        "gap is not all design cost: standardising needs baseline statistics for the "
        "same (instance, variable), and "
        f"{own['grid_standardisation']['n_scored_rows_without_baseline_stats']:,d} "
        "scored grid rows have none because every sub-group median in their baseline "
        "windows was missing. Its AUC is therefore computed over a smaller and easier "
        "population than the other tools', which is why the refused fraction sits in "
        "the same table as the AUC."
    )
    add("")

    add("## The population screen, and the split it does not have")
    add("")
    add(
        f"The profile study found "
        f"{pop_inst['whole_life_frozen']['n_whole_life_frozen_measurement_tags']:,d} "
        "measurement tags holding one distinct value across their "
        "whole archive. A self-referencing threshold cannot indict them, which is what "
        "`baselines/population.py` was built for: the reference becomes the same "
        "(well, variable) pair's other windows. This screen is not a per-tag limit, so "
        "the own-history rewrite does not touch it."
    )
    add("")
    add(
        "**Under holdout by well it refuses everything.** The baseline is that well's "
        "other windows and well holdout removes them, so all "
        f"{pop_well['frozen_windows']['n_frozen_feature_rows']:,d} frozen feature rows in "
        "the held-out folds come back `PopulationTooSparse` and not one is flagged. That "
        "is the answer to the question as posed, and it is the result: the tool "
        "asks whether a sensor moves *on this well*, and a split that holds the well out "
        "holds out the question."
    )
    add("")
    add(
        "So it is also run under a second split - 5-fold group holdout by **instance**, "
        "same seed - which holds out the window's own instance while leaving the well's "
        "other instances in training. Every number in this section is labelled with which "
        "split produced it, and the instance-holdout numbers are not held-out-well "
        "numbers."
    )
    add("")
    add("| | well holdout | instance holdout |")
    add("|---|---:|---:|")
    add(
        f"| frozen feature rows on the test side | "
        f"{pop_well['frozen_windows']['n_frozen_feature_rows']:,d} | "
        f"{pop_inst['frozen_windows']['n_frozen_feature_rows']:,d} |"
    )
    add(
        f"| flagged | {pop_well['frozen_windows']['flagged_frac']:.1%} | "
        f"{pop_inst['frozen_windows']['flagged_frac']:.1%} "
        f"({pop_inst['frozen_windows']['n_frozen_flagged']:,d}) |"
    )
    add(
        f"| refused | {pop_well['frozen_windows']['refused_frac']:.1%} | "
        f"{pop_inst['frozen_windows']['refused_frac']:.1%} "
        f"({pop_inst['frozen_windows']['n_frozen_refused']:,d}) |"
    )
    wl = pop_inst["whole_life_frozen"]
    add(
        f"| whole-life-frozen tags indicted | 0 of "
        f"{wl['n_reachable_in_common_variable_set']:,d} reachable | "
        f"{wl['n_indicted']} of {wl['n_reachable_in_common_variable_set']:,d} reachable |"
    )
    add("")
    add(
        f"Of the profile study's {wl['n_whole_life_frozen_measurement_tags']:,d} whole-life-frozen "
        f"measurement tags, {wl['n_reachable_in_common_variable_set']:,d} sit on one of the "
        f"six common variables and are therefore reachable at all. Under instance holdout "
        f"the screen indicts {wl['n_indicted']} of them:"
    )
    add("")
    add("| variable | whole-life-frozen tags reachable | indicted |")
    add("|---|---:|---:|")
    for variable in variables:
        add(
            f"| `{variable}` | {wl['per_variable_reachable'][variable]:,d} | "
            f"{wl['per_variable_indicted'][variable]} |"
        )
    add("")
    biggest = max(wl["per_variable_reachable"], key=lambda v: wl["per_variable_reachable"][v])
    add(
        f"**{wl['per_variable_reachable'][biggest]:,d} of the "
        f"{wl['n_reachable_in_common_variable_set']:,d} reachable tags are `{biggest}`, and "
        f"{wl['per_variable_indicted'][biggest]} is indicted.** That is the "
        "screen working, not failing. A `P-PDG` that is flat on a well where `P-PDG` is "
        "usually flat gets `fired=False` with the reason stated - a normally-still sensor "
        "is not indicted for being still. The pairs it does indict are wells where the "
        "same tag moves in hundreds of other windows:"
    )
    add("")
    for evidence in pop_inst["example_evidence"][:3]:
        add(f"- {evidence}")
    add("")

    add("## What these numbers mean")
    add("")
    spc_cut = pick(summary, "spc", OWN, "no_folder9_no_transients")
    clock_cut = pick(summary, "clock", OWN, "no_folder9_no_transients")
    add("Four readings, in order of how much weight they carry.")
    add("")
    add(
        "**1. The pooled numbers for stages 3 and 5 were measuring the wrong thing, and "
        "the correction is large.** A per-tag limit fitted across wells is not a per-tag "
        f"limit. On identical windows the MAD screen moves "
        f"{pick(summary, 'mad', MATCHED, 'all')['auc_mean']:.3f} → "
        f"{pick(summary, 'mad', OWN, 'all')['auc_mean']:.3f} and the below-chance folds "
        "go away, because there are no folds: the baseline no longer depends on which "
        "wells happened to be in training."
    )
    add("")
    add(
        "**2. Most of that number is still the clock.** On the all-windows target the "
        f"clock control scores {clock['auc_mean']:.3f}, above every detector. The "
        "own-history design removes each instance's opening windows, and in 3W those are "
        "the normal ones, so the scored population is "
        f"{own['scored_positive_rate']:.1%} positive and skewed late. Read the "
        "all-windows column as a lower bound on how easy the target is, not as detector "
        "performance."
    )
    add("")
    add(
        f"**3. What survives the clock is the SPC rules on steady faults.** With "
        "transient-only windows and folder 9 out, the clock is "
        f"{clock_cut['auc_mean']:.3f} and the own-history SPC rules are "
        f"{spc_cut['auc_mean']:.3f} over "
        f"{int(spc_cut['n_scored']):,d} windows. That is the one place in this study "
        "where a detector clearly beats knowing nothing but the time, and it is the "
        "tool whose object - a rule hit against limits from this tag's own history - "
        "matches the question the corpus can actually answer."
    )
    add("")
    aligned_reading(add, Path(args.results), clock)
    add(
        "Three specific reasons the target stays hard, all visible in the numbers above "
        "rather than inferred:"
    )
    add("")
    add(
        f"1. **The label is not a window property.** "
        f"{wm['n_transient_only'] / max(wm['n_positive'], 1):.0%} of positive windows "
        "contain only transient rows. A window is called positive because one second in "
        "3,600 carried class 107; a detector reading hour-scale statistics is being "
        "scored against a second-scale label. Every number in this study moves when "
        "those windows are dropped, in both directions."
    )
    state_share = (windows["state_mode"].astype(str) == "0").mean()
    complete = int((windows["n_variables_present"] == len(variables)).sum())
    regime_pooled = pick(summary, "regime", POOLED, "all")
    add(
        f"2. **The regime key is nearly constant.** `LABEL_state` is `0` in "
        f"{state_share:.1%} of windows, so conditioning on it refines almost nothing - "
        f"which is why the stage-4 regime screen ({fmt(regime_pooled['auc_mean'])}) does "
        f"not beat the regime-blind stage-3 MAD screen ({fmt(mad_pooled['auc_mean'])}) "
        "under the same design."
    )
    add(
        f"3. **Half the windows cannot be aligned at all.** Only {complete:,d} "
        f"of {len(windows):,d} windows carry all six common variables, so MSPC scores "
        "fewer than half of them under either design and its AUC is computed over a "
        "different, easier population than the other tools'."
    )
    add("")
    add(
        f"And a number that is not a performance figure: the own-history design refuses "
        f"{pick(summary, 'mad', OWN, 'all')['refused_frac_mean']:.1%} of the corpus "
        f"before it starts, because {own['n_instances_too_short']:,d} of "
        f"{own['n_instances_with_windows']:,d} instances are "
        f"{own['baseline_windows']} windows or shorter. On a corpus of hour-long "
        "excerpts that is the design's cost; on a live historian, where a tag has "
        "months of its own past, it is not."
    )
    add("")

    add("## Label-noise sensitivity")
    add("")
    add(
        "Excluding folder 9, excluding transient-only windows, and excluding instances "
        "whose baseline windows are themselves fault windows. Each changes the target, "
        "so each changes the number."
    )
    add("")
    for design_name, tools in (
        (POOLED, list(TOOL_LABELS)),
        (MATCHED, [t for t in TOOL_LABELS if t in NEW_DESIGN]),
        ("own-history and per-instance-standardised", ["clock", *NEW_DESIGN]),
    ):
        add(f"### {design_name}")
        add("")
        add(
            "| tool | all | no folder 9 | no transients | neither | no positive baseline |"
        )
        add("|---|---:|---:|---:|---:|---:|")
        for tool in tools:
            if tool == "clock":
                design, label = OWN, "clock control"
            elif design_name.startswith("own-history"):
                design, label = NEW_DESIGN[tool], TOOL_LABELS[tool][1]
            else:
                design, label = design_name, TOOL_LABELS[tool][1]
            cells = []
            for variant in (*VARIANT_LABELS, "no_positive_baseline"):
                row = pick(summary, tool, design, variant)
                cells.append(
                    "refused"
                    if row is None or pd.isna(row["auc_mean"])
                    else f"{row['auc_mean']:.3f}"
                )
            add(f"| {label} | " + " | ".join(cells) + " |")
        add("")

    add("## What the real data broke")
    add("")
    beyond = prov["beyond_float32"]
    add(
        "1. **A window feature cost eight times its own archive read.** "
        "`store/averaging.py` walked its samples with `.iloc` to accumulate the "
        "time-weighted mean, and `features/window_features.py` did the same to count "
        "value changes. An hour of 1 Hz data is 3,600 rows, which put `extract` at 127 ms "
        "against a 16 ms parquet read and the whole study out of reach. Vectorised, "
        "`extract` is 1.9 ms; 120 real windows were compared before and after and the "
        "time-weighted means agree to the last bit."
    )
    add(
        "2. **`regime_baselines` accepted a mode series on the wrong index.** It checked "
        "length, but every regime mask inside it is a pandas boolean operation, which "
        "aligns on the index. A same-length series carrying a different index emptied "
        "every regime and came back `RegimeTooSparse` - a refusal naming the wrong cause, "
        "which is worse than a wrong number because it reads as a correct refusal. It "
        "now refuses the misaligned series by name. Before the fix this study reported "
        "the regime screen as 48% refused; after it, 4%."
    )
    add(
        "3. **3W ships a null sentinel as a number, and per-tag scaling turns it into "
        "the whole model.** Three `WELL-00006` instances hold `P-PDG` pinned at "
        "`-1.180116e+42` for their entire life. 3W publishes no engineering ranges, so "
        "`eng_range` is `None` and nothing can call it out of range: the window reports "
        "coverage 1.000, valid fraction 1.000 and GOOD quality. `IsolationForest` casts "
        "to float32 internally, where that value is `-inf`. Worse, dividing by that "
        "tag's own near-zero baseline spread gave a z-score of 1.5e26, and the "
        "standardised PCA became a model of that one row - Hotelling T2 came back "
        "constant to every window on four folds of five, an AUC of exactly 0.500 that "
        "looked like a result. The study now blanks every cell beyond float32 range "
        f"once, for every design: {beyond['features']['n_cells']:,d} feature cells and "
        f"{beyond['subgroup_grid']['n_cells']:,d} sub-group cells, over "
        f"{beyond['subgroup_grid']['n_rows']:,d} rows. `DataPhysics` now carries "
        "`implausible_magnitude_count`, so a profile states the count for any window that "
        "holds one: *values beyond float32 range: N (not censored: eng range unknown)*. "
        "It is a count, not a verdict - without an engineering range nothing in the "
        "library can say the value is wrong."
    )
    add("")

    add("## Files")
    add("")
    add("| path | what |")
    add("|---|---|")
    add("| `build_windows.py` | windows, labels, features, sub-group grid (cached under `data/`) |")
    add(
        "| `build_onsets.py` | one row per instance: where its fault starts and whether "
        "the aligned design can score it |"
    )
    add("| `run_detectors.py` | fits and scores every tool under every design |")
    add("| `make_report.py` | this report, `out/*.png` |")
    add("| `results/per_fold.csv` | one row per (tool, design, split, fold, variant) |")
    add("| `results/summary.csv` | the same aggregated across folds |")
    add("| `results/scores.csv` | every score the run produced, per window |")
    add("| `results/windows.csv` | one row per window: well, folder, label, transient flag |")
    add(
        "| `results/run.json` | provenance, fold well lists, own-history layout, "
        "population and MSPC detail |"
    )
    add("| `results/onsets.json` | onset distribution and the evaluable-set counts |")
    add(
        "| `results/aligned_per_instance.csv` | one row per (tool, instance): baseline "
        "threshold and the three test scores |"
    )
    add("| `results/aligned_summary.csv` | one row per (tool, variant) under the aligned design |")
    add("| `results/aligned_scores.csv` | every aligned score, per window |")
    add(
        "| `results/aligned_windows.csv` | one row per aligned window: role, offset from "
        "onset, design label, row label |"
    )
    add("| `results/aligned_run.json` | aligned provenance, onset summary, split, refusals |")

    text = "\n".join(lines) + "\n"
    (HERE / "REPORT.md").write_text(text, encoding="utf-8", newline="\n")
    (res / "summary.json").write_text(
        json.dumps(
            {
                "provenance": prov,
                "summary": summary.replace({np.nan: None}).to_dict("records"),
                "population": run["population"],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    n_png = len(sorted(out_dir.glob("*.png")))
    print(f"wrote {HERE / 'REPORT.md'} ({len(text):,d} bytes) and {n_png} PNGs")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Report, figure and BENCHMARKS.md section for the Chronos-Bolt 3W study.

Reads ``results/`` written by ``run_chronos.py`` and the detector study's
frozen ``aligned_summary.csv`` and ``aligned_run.json`` (read, never
recomputed), then writes ``REPORT.md``, ``out/01_paired_surprise.png``
and ``results/benchmarks_section.md``. With ``--update-benchmarks`` the
section replaces or appends its own ``## REAL`` heading in the
repository's ``BENCHMARKS.md``; the SYNTHETIC table and the other REAL
sections are left as they are.

Run: ``uv run python examples/studies/3w_chronos/make_report.py``.
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

import run_chronos as rc  # noqa: E402

DETECTOR_RESULTS = HERE.parent / "3w_detectors" / "results"
BENCH_MD = ROOT / "BENCHMARKS.md"
SECTION_HEADING = "## REAL: 3W Chronos-Bolt zero-shot"
CHRONOS_COLOR = "#e377c2"
CLOCK_COLOR = "#7f7f7f"

# The detector study's aligned rows the comparison table carries, in the
# order of that study's report, with the stage column BENCHMARKS.md uses.
DETECTOR_ROWS = (
    ("mad", "MAD screen (regime-blind), own history", "3"),
    ("spc", "SPC individuals rules, own history", "5"),
    ("mspc_t2", "MSPC Hotelling T2, per-instance-standardised", "6"),
    ("mspc_spe", "MSPC SPE, per-instance-standardised", "6"),
    ("iforest", "IsolationForest, per-instance-standardised", "7"),
    ("clock", "clock control (position in the record)", "none"),
)
LABELS = {
    rc.TOOL: "Chronos-Bolt small, zero-shot forecast surprise",
    rc.CLOCK: "clock control (position in the record), all 48 instances",
    rc.CLOCK_SAME: "clock control, the 33 instances Chronos-Bolt scores",
}


# ---------------------------------------------------------------- loading


def load(results: Path, detector: Path) -> dict:
    summary = pd.read_csv(results / "summary.csv")
    per_instance = pd.read_csv(results / "per_instance.csv", dtype={"folder_label": str})
    det_summary = pd.read_csv(detector / "aligned_summary.csv")
    det_summary = det_summary[det_summary["variant"] == "all"].set_index("tool")
    return {
        "summary": summary.set_index("tool"),
        "per_instance": per_instance,
        "per_fold": pd.read_csv(results / "per_fold.csv"),
        "windows": pd.read_csv(results / "windows.csv"),
        "window_scores": pd.read_csv(results / "window_scores.csv"),
        "run": json.loads((results / "run.json").read_text("utf-8")),
        "det_summary": det_summary,
        "det_run": json.loads((detector / "aligned_run.json").read_text("utf-8")),
    }


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


def auc_cell(row: pd.Series) -> str:
    auc = fmt(row["roc_auc"])
    lo, hi = row.get("auc_fold_min"), row.get("auc_fold_max")
    if pd.notna(lo) and pd.notna(hi) and row.get("n_folds_scored", 0) > 1:
        return f"{auc} ({fmt(lo)}-{fmt(hi)})"
    return auc


def code_cite(run_code: dict) -> str:
    version = run_code.get("tsdive_version") or run_code.get("tagledger_version")
    name = "tsdive" if "tsdive_version" in run_code else "tagledger"
    return f"{name} {version} @ `{run_code['git_head'][:8]}`"


# ----------------------------------------------------------------- tables


def comparison_rows(data: dict) -> list[tuple[str, ...]]:
    det_code = code_cite(data["det_run"]["provenance"]["code"])
    this_code = code_cite(data["run"]["code"])
    rows = []
    for tool, label, _ in DETECTOR_ROWS:
        r = data["det_summary"].loc[tool]
        rows.append(_metric_cells(label, r, f"detector study, {det_code}"))
    for tool in (rc.TOOL, rc.CLOCK_SAME, rc.CLOCK):
        if tool in data["summary"].index:
            r = data["summary"].loc[tool]
            rows.append(_metric_cells(LABELS[tool], r, f"this study, {this_code}"))
    return rows


def _metric_cells(label: str, r: pd.Series, source: str) -> tuple[str, ...]:
    far = r["false_alarm_rate_pre"]
    over = None if pd.isna(far) else float(far) - rc.far_floor(rc.BASELINE_WINDOWS)
    return (
        label,
        f"{int(r['n_instances_with_threshold'])} of {int(r['n_instances'])}",
        auc_cell(r),
        f"{fmt(r['paired_hit_post0'])} / {fmt(r['paired_hit_post1'])}",
        pct(far),
        signed(over),
        pct(r["detection_rate_post1"]),
        source,
    )


def table(header: list[str], rows: list[tuple[str, ...]], align: str | None = None) -> str:
    align = align or "|".join(["---"] * len(header))
    lines = ["| " + " | ".join(header) + " |", "|" + align + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def per_fold_rows(per_fold: pd.DataFrame) -> list[tuple[str, ...]]:
    rows = []
    frame = per_fold[per_fold["tool"].isin([rc.TOOL, rc.CLOCK_SAME])]
    for _, r in frame.sort_values(["fold", "tool"]).iterrows():
        rows.append(
            (
                str(int(r["fold"])),
                "Chronos-Bolt surprise" if r["tool"] == rc.TOOL else "clock, same instances",
                str(int(r["n_instances"])),
                f"{int(r['n_pre_windows'])} / {int(r['n_post_windows'])}",
                fmt(r["roc_auc"]),
                f"{fmt(r['paired_hit_post0'])} / {fmt(r['paired_hit_post1'])}",
                pct(r["false_alarm_rate_pre"]),
            )
        )
    return rows


def refusal_tables(data: dict) -> str:
    run = data["run"]["refusals"]
    scores = data["window_scores"]
    windows = data["windows"]
    per_instance = data["per_instance"]
    refused_vars = scores[scores["refusal"].notna()].copy()
    refused_vars["cause"] = refused_vars["refusal"].str.split(":").str[0]
    cause_rows = [
        (cause, str(n), pct(n / len(scores)))
        for cause, n in sorted(run["variable_refusals_by_cause"].items())
    ]
    var_rows = []
    for variable in data["run"]["parameters"]["variables"]:
        part = refused_vars[refused_vars["variable"] == variable]
        by_cause = part["cause"].value_counts()
        detail = ", ".join(f"{c} {int(n)}" for c, n in sorted(by_cause.items())) or "none"
        var_rows.append((f"`{variable}`", str(len(part)), detail))
    role_rows = []
    for offset, part in windows.groupby("onset_offset", sort=True):
        n_ref = int(part["refusal"].notna().sum())
        role_rows.append(
            (f"{int(offset):+d} h ({part['role'].iloc[0]})", str(len(part)), str(n_ref))
        )
    chronos = per_instance[per_instance["tool"] == rc.TOOL]
    refused_inst = chronos[chronos["baseline_max"].isna()]
    inst_rows = []
    for _, r in refused_inst.sort_values("instance").iterrows():
        w = windows[(windows["instance"] == r["instance"]) & windows["refusal"].notna()]
        roles = ", ".join(f"{int(o):+d} h" for o in sorted(w["onset_offset"]))
        v = refused_vars[refused_vars["instance"] == r["instance"]]
        causes = ", ".join(f"{c} {int(n)}" for c, n in sorted(v["cause"].value_counts().items()))
        inst_rows.append((f"`{r['instance']}`", r["well"], roles, causes))
    parts = [
        f"Variable rows: {run['n_variable_rows_refused']} of {run['n_variable_rows']} "
        f"refused ({pct(run['n_variable_rows_refused'] / run['n_variable_rows'])}). "
        "The cause is the first word of the row's `refusal` string in "
        "`results/window_scores.csv`.",
        "",
        table(["cause", "variable rows", "share of 1,728"], cause_rows, "---|---:|---:"),
        "",
        table(["variable", "rows refused", "by cause"], var_rows, "---|---:|---"),
        "",
        f"Windows: {run['n_windows_refused']} of {run['n_windows']} refused, each with "
        f"fewer than {rc.MIN_SCORED_VARIABLES} scored variables.",
        "",
        table(["window (role)", "windows", "refused"], role_rows, "---|---:|---:"),
        "",
        f"Instances: {run['n_instances_refused']} of {run['n_instances']} have no threshold "
        "because at least one of their three baseline windows is refused. Their four "
        "test-window scores, where they exist, are in `results/per_instance.csv` and "
        "enter no metric.",
        "",
        table(
            ["instance", "well", "windows refused", "variable refusals by cause"],
            inst_rows,
            "---|---|---|---",
        ),
    ]
    return "\n".join(parts)


# ----------------------------------------------------------------- figure


def plot_paired(per_instance: pd.DataFrame, out: Path) -> None:
    chronos = per_instance[(per_instance["tool"] == rc.TOOL) & per_instance["baseline_max"].notna()]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.8), dpi=160, sharex=True, sharey=True)
    lo = float(chronos[["pre", "post0", "post1"]].min().min()) * 0.8
    hi = float(chronos[["pre", "post0", "post1"]].max().max()) * 1.25
    for ax, role, offset in zip(axes, ("post0", "post1"), ("0 h", "+1 h"), strict=True):
        pre = chronos["pre"].to_numpy(dtype=float)
        post = chronos[role].to_numpy(dtype=float)
        fired = post > chronos["baseline_max"].to_numpy(dtype=float)
        ax.plot([lo, hi], [lo, hi], color="#888888", lw=1.0, ls="--", zorder=1)
        ax.scatter(
            pre[~fired], post[~fired], s=34, facecolors="none", edgecolors=CHRONOS_COLOR,
            lw=1.2, zorder=3, label="below the instance's own threshold",
        )
        ax.scatter(
            pre[fired], post[fired], s=34, color=CHRONOS_COLOR, zorder=3,
            label="above the instance's own threshold",
        )
        hits = int((post > pre).sum())
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(f"post window at {offset}: {hits} of {len(pre)} above the pre window",
                     fontsize=10)
        ax.set_xlabel("surprise, pre window (-2 h)")
        ax.grid(True, which="major", color="#eeeeee", lw=0.8)
    axes[0].set_ylabel("surprise, post window")
    axes[0].legend(loc="lower right", fontsize=8, frameon=False)
    fig.suptitle(
        "Chronos-Bolt small, zero-shot: forecast surprise of each instance's post window "
        "against its pre window (33 instances, 3W onset-aligned)",
        fontsize=10,
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor="white", metadata={"Software": None})
    plt.close(fig)


# ----------------------------------------------------------------- report


def render_report(data: dict) -> str:
    run = data["run"]
    s = data["summary"]
    ch = s.loc[rc.TOOL]
    clock = s.loc[rc.CLOCK]
    same = s.loc[rc.CLOCK_SAME] if rc.CLOCK_SAME in s.index else clock
    det = data["det_summary"]
    ref = run["refusals"]
    n_scored = int(ch["n_instances_with_threshold"])
    n_all = int(ch["n_instances"])
    floor = rc.far_floor(rc.BASELINE_WINDOWS)
    model = run["model"]
    fold_rows = per_fold_rows(data["per_fold"])
    chronos_folds = data["per_fold"][data["per_fold"]["tool"] == rc.TOOL]
    n_folds_under_half = int((chronos_folds["roc_auc"] < 0.5).sum())
    det_code = code_cite(data["det_run"]["provenance"]["code"])
    this_code = code_cite(run["code"])
    manifest = run["dataset_manifest_sha256"][:12]
    aligned_sha = run["aligned_windows_manifest_sha256"][:12]
    best_det = max(
        (float(det.loc[t, "roc_auc"]), t) for t, _, _ in DETECTOR_ROWS if t != "clock"
    )
    windows = data["windows"]
    n_open_record = int(
        ((windows["onset_offset"] == -5) & (windows["window_index"] == 0)).sum()
    )
    ws = data["window_scores"]
    n_pdg_constant = int(
        ((ws["variable"] == "P-PDG") & ws["refusal"].fillna("").str.startswith("context is")).sum()
    )
    n_hits0 = round(float(ch["paired_hit_post0"]) * float(ch["n_paired"]))
    auc_gap = float(ch["roc_auc"]) - float(same["roc_auc"])
    det_far_min = float(det.loc[[t for t, _, _ in DETECTOR_ROWS], "false_alarm_rate_pre"].min())
    under_floor = float(ch["false_alarm_rate_pre"]) <= floor and det_far_min > floor
    floor_sentence = (
        "so the forecast surprise is the only row in the table whose false-alarm rate sits "
        "at or under the floor."
        if under_floor
        else "and the forecast surprise's rate is read against the same floor."
    )
    population_sentence = (
        f" The detector rows cover 48 instances and this row {n_scored}; the "
        f"{ref['n_instances_refused']} missing instances are the ones with a refused baseline "
        "window, so the two false-alarm rates are not over the same instances."
    )

    add = []
    p = add.append
    p("# Chronos-Bolt zero-shot on the onset-aligned 3W windows")
    p("")
    p("## Summary")
    p("")
    p(
        f"The study scores the {n_all} evaluable instances of 3W v2.0.0 (manifest "
        f"`{manifest}`, {ref['n_windows']} onset-aligned one-hour windows) with "
        f"`{model['model_id']}`, a pretrained time-series forecaster, with nothing fitted on "
        "3W. For each window and each of the six common variables the model forecasts the "
        "window hour from the four hours before it at one-minute resolution, and the window "
        "score is the mean absolute error of the median forecast in units of the 10-90% "
        f"forecast band. Under each instance's own worst-baseline threshold the score fires on "
        f"{pct(ch['false_alarm_rate_pre'])} of the pre windows (floor {pct(floor)}) and on "
        f"{pct(ch['detection_rate_post0'])} of the post0 windows; pooled, it ranks post above "
        f"pre at ROC-AUC {fmt(ch['roc_auc'])} against {fmt(same['roc_auc'])} for the clock "
        f"control on the same instances, and the paired hit rate is "
        f"{fmt(ch['paired_hit_post0'])} at onset and {fmt(ch['paired_hit_post1'])} an hour "
        "later. "
        f"{ref['n_instances_refused']} of {n_all} instances are refusals because at least one "
        "baseline window lacks 60 minutes of context or holds a constant variable. The 48 "
        "positives are the first two hours of a transient run-up, so the numbers describe "
        f"{n_scored} instances and the beginning of a fault. The per-fold table shows the "
        "spread."
    )
    p("")
    p("## Data")
    p("")
    p(
        f"The windows are the detector study's onset-aligned cache "
        f"(`{run['aligned_windows']}`, manifest sha256 `{aligned_sha}`): {n_all} instances "
        f"over 21 wells, six windows each at -5, -4, -3 (baseline), -2 (pre), 0 (post0) and "
        f"+1 h (post1) from the instance's own fault onset, {ref['n_windows']} windows in all. "
        "The design, the evaluable-set rule and the onset definition are in "
        "[the detector report](../3w_detectors/REPORT.md); nothing about them changes here. "
        f"The raw rows come from the 1 s archives under `{run['archives']}`; only rows with "
        f"quality `{run['parameters']['quality_used']}` are read. The six variables are "
        + ", ".join(f"`{v}`" for v in run["parameters"]["variables"])
        + ". Fold membership per instance is read from the detector study's "
        f"`{run['folds_from']}`, so the per-fold table below cuts the same wells the detector "
        "study cut."
    )
    p("")
    p(
        "The positives are the run-up. All 48 evaluable onsets are transient classes "
        "(101-109), and 47 of the 48 post1 windows contain only transient rows (detector "
        "report, onset-aligned design). A post window here is the first or second hour "
        "after a fault begins, so the question is whether the forecast error rises at the "
        "beginning of a fault; the steady fault state is never scored."
    )
    p("")
    p("## Method")
    p("")
    p(
        f"Model: `{model['model_id']}`, Hugging Face revision `{model['model_revision']}`, "
        f"loaded with `chronos.BaseChronosPipeline.from_pretrained` (class "
        f"`{model['pipeline_class']}`) from `chronos-forecasting` "
        f"{model['chronos_forecasting_version']} on `torch` {model['torch_version']}, "
        f"{model['device']}, {model['dtype']}, `torch.manual_seed({model['seed']})`, "
        f"{model['threads']} threads. Nothing is fitted or fine-tuned on 3W in this study; "
        "whether the model's public pretraining corpus contains any 3W series is not "
        "checked."
    )
    p("")
    p("Per window and variable:")
    p("")
    p(
        f"1. Context is the {rc.CONTEXT_MINUTES} minutes before `window_start`, target is the "
        f"{rc.TARGET_MINUTES} minutes of the window, both as one-minute means of the good "
        f"rows. Fewer than {rc.MIN_CONTEXT_MINUTES} context minutes with data, fewer than "
        f"{rc.MIN_TARGET_MINUTES} target minutes with data, or a constant context is a "
        "refusal with a cause string; the row stays in `results/window_scores.csv` unscored."
    )
    p(
        "2. A missing context minute takes the last observed minute (the first observed "
        f"minute where none precedes it); {ref['n_context_minutes_filled']:,} minutes are "
        "filled this way and each row records its count in `n_context_filled`."
    )
    p(
        f"3. `predict_quantiles(context, prediction_length={rc.TARGET_MINUTES}, "
        f"quantile_levels={list(rc.QUANTILE_LEVELS)})`, batched {model['batch']} series "
        "at a time."
    )
    p(
        "4. Surprise = mean over the observed target minutes of `|actual - q50| / "
        f"max(q90 - q10, {rc.WIDTH_FLOOR_SHARE} * std(context))`. A target minute without "
        f"data is skipped and counted ({ref['n_target_minutes_skipped']} in this run)."
    )
    p("")
    p("Per window and instance:")
    p("")
    p(
        f"5. The window score is the mean surprise over its scored variables; fewer than "
        f"{rc.MIN_SCORED_VARIABLES} scored variables is a window refusal."
    )
    p(
        "6. Each instance's threshold is `worst_baseline_threshold` over its three baseline "
        "window scores (`tsdive.eval`); an instance with a refused baseline window has no "
        "threshold and is a refusal. `fires` decides pre, post0 and post1 at that threshold; "
        "the false-alarm floor is `far_floor(3)` = 0.250 and `over_floor` is the measured "
        "rate minus it."
    )
    p(
        "7. Pooled ranking uses `ranking_metrics` with the pre windows as negatives and the "
        "post0 and post1 windows as positives, over all instances with a threshold and per "
        "fold. The paired hit rate is the share of instances whose post window outscores "
        "its own pre window. The detection delay is 0 when post0 fires, 1 when only post1 "
        "fires."
    )
    p(
        "8. The clock control is `clock_control` on `window_index`, the window's position in "
        "its own record, exactly as the detector study's aligned clock; it is reported over "
        f"all {n_all} instances and again over the {n_scored} instances the model scores."
    )
    p("")
    p("Commands:")
    p("")
    p("```")
    p("uv sync --extra tsfm")
    p("uv run python examples/studies/3w_chronos/run_chronos.py")
    p("uv run python examples/studies/3w_chronos/make_report.py")
    p("```")
    p("")
    p(
        f"The run takes {run['wall_seconds']:.0f} s on the CPU, {run['model_load_seconds']:.0f} "
        "of them loading the model. Two runs write byte-identical CSVs; the study checked "
        "this with a second run into a scratch directory and `cmp` on every file."
    )
    p("")
    p("## Results")
    p("")
    p("### Against the detector study")
    p("")
    p(
        "The detector rows are read from the detector study's frozen "
        "`results/aligned_summary.csv` (variant `all`, onset-aligned design) and are not "
        "recomputed. AUC pools pre windows as negatives and post windows as positives; the "
        "range in parentheses is the per-fold minimum and maximum where more than one fold "
        "scores. Paired hit is post0 / post1. FAR is the share of pre windows firing at "
        "each instance's own threshold, and over floor is that share minus 0.250. The two "
        "clock rows differ only in the instance set."
    )
    p("")
    p(
        table(
            [
                "tool",
                "instances with threshold",
                "ROC-AUC (fold range)",
                "paired hit post0 / post1",
                "FAR on pre",
                "over floor",
                "detect post1",
                "source",
            ],
            comparison_rows(data),
            "---|---:|---|---|---:|---:|---:|---",
        )
    )
    p("")
    p("### Per fold")
    p("")
    p(
        "Folds are the detector study's 5-fold group holdout by well. The model fits "
        "nothing, so a fold here is a subset of instances and nothing else; the spread is "
        "how much the pooled number depends on which wells are in it."
    )
    p("")
    p(
        table(
            ["fold", "tool", "instances", "pre / post windows", "ROC-AUC",
             "paired hit post0 / post1", "FAR on pre"],
            fold_rows,
            "---:|---|---:|---|---:|---|---:",
        )
    )
    p("")
    p("### Refusals")
    p("")
    p(refusal_tables(data))
    p("")
    p("![post against pre surprise per instance](out/01_paired_surprise.png)")
    p("")
    p(
        f"Figure 1. Each point is one of the {n_scored} instances with a threshold; the "
        "diagonal is post equal to pre, and a filled point fires at the instance's own "
        "worst-baseline threshold. Source: `results/per_instance.csv`."
    )
    p("")
    p("## Discussion")
    p("")
    p(
        f"At onset the surprise rises for {n_hits0} "
        f"of {int(ch['n_paired'])} instances (paired hit {fmt(ch['paired_hit_post0'])}), and "
        f"{pct(ch['detection_rate_post0'])} of the post0 windows fire at a threshold that "
        f"fires on {pct(ch['false_alarm_rate_pre'])} of the pre windows, "
        f"{fmt(abs(float(ch['false_alarm_rate_pre']) - floor))} "
        f"{'under' if float(ch['false_alarm_rate_pre']) <= floor else 'over'} the "
        f"{pct(floor)} floor, which is the false-alarm rate of a score that reads nothing. "
        "Under the same threshold rule the detector "
        f"study's tools fire on {pct(det.loc['spc', 'false_alarm_rate_pre'])} (SPC) to "
        f"{pct(det.loc['iforest', 'false_alarm_rate_pre'])} (IsolationForest) of the pre "
        f"windows, {floor_sentence}{population_sentence}"
    )
    p("")
    p(
        f"One hour later the paired hit rate falls to {fmt(ch['paired_hit_post1'])} and the "
        f"post1 detection rate to {pct(ch['detection_rate_post1'])}. The score is a one-hour "
        "forecast error, and for post1 the context already holds the first fault hour. "
        "Whether the median forecast tracks the move or the 10-90% band widens is not "
        "measured here; either lowers the surprise. The clock control takes "
        "both paired hit rates at 1.000 and fires on every window, because under this "
        "design every post window sits later in its record than every baseline window."
    )
    p("")
    p(
        f"Pooled, ROC-AUC {fmt(ch['roc_auc'])} sits {signed(auc_gap)} "
        f"above the clock on the same {n_scored} instances ({fmt(same['roc_auc'])}) and below "
        f"the best detector-study tool ({fmt(best_det[0])}, `{best_det[1]}`, on 48 instances). "
        f"The per-fold range {fmt(ch['auc_fold_min'])}-{fmt(ch['auc_fold_max'])} spans one "
        f"half, and in {n_folds_under_half} of {int(ch['n_folds_scored'])} folds the AUC is "
        "below 0.5, so the pooled AUC depends on which wells are in the pool as much as on "
        "the model. The AUC and the paired hit rate answer different questions: the "
        "paired rate compares each instance with itself and is the number the clock control "
        "is read against, and the AUC compares surprise levels across instances whose "
        "variables sit on different scales."
    )
    p("")
    p(
        "What the numbers do not support: a claim about the steady fault state (never "
        "scored), about the 15 refused instances, about any 3W instance outside the "
        "evaluable 48, or about Chronos-Bolt as a detector with a tuned threshold. The "
        "band floor, the four-hour context and the one-minute aggregation are choices made "
        "once before the run and not varied."
    )
    p("")
    p("## Limits and further work")
    p("")
    p(
        f"- {ref['n_instances_refused']} instances are refused because a baseline window at "
        f"-5 h opens the record with under 60 minutes of context ({n_open_record} instances "
        "have `window_index` 0 there) or because a variable is constant over the context "
        f"(`P-PDG` in {n_pdg_constant} variable rows). A shorter context for the first "
        "baseline window would score more instances "
        "and change the score's scale between windows; the study does not do this."
    )
    p(
        "- Every scored instance is one transient onset; no steady-state fault is in the "
        "positives, and 21 wells contribute unevenly to the five folds."
    )
    p(
        "- The context is 240 minutes at one-minute means. Chronos-Bolt accepts up to 2048 "
        "points, so a 1 s or 10 s context over a shorter span, or a longer span at one "
        "minute, are untried settings."
    )
    p(
        "- The surprise is one statistic of the forecast error. The quantile loss, the "
        "share of target minutes outside the 10-90% band, and a per-variable threshold are "
        "untried."
    )
    p(
        "- The model runs zero-shot. Fine-tuning on other wells' baseline windows under the "
        "group holdout is a different study with a fitted model and its own leakage check."
    )
    p("")
    p("## Files")
    p("")
    p(
        table(
            ["path", "what"],
            [
                ("`run_chronos.py`", "scores every window and variable, writes `results/`"),
                ("`make_report.py`", "this report, `out/01_paired_surprise.png`, "
                                     "`results/benchmarks_section.md`"),
                ("`results/window_scores.csv`",
                 "one row per (window, variable): minutes with data, filled, skipped, "
                 "surprise or refusal"),
                ("`results/windows.csv`", "one row per window: scored variables, score or refusal"),
                ("`results/per_instance.csv`",
                 "one row per (tool, instance): the detector study's `aligned_per_instance` "
                 "columns"),
                ("`results/per_fold.csv`", "pooled ranking per (tool, fold)"),
                ("`results/summary.csv`",
                 "one row per tool: the detector study's `aligned_summary` columns plus "
                 "`far_floor` and `over_floor`"),
                ("`results/benchmarks_section.md`", "the BENCHMARKS.md REAL section"),
                ("`results/run.json`",
                 "model id and revision, library versions, code head, manifest hashes, "
                 "parameters, refusal counts, wall seconds"),
            ],
        )
    )
    p("")
    p(f"Detector-study numbers: {det_code}. This study: {this_code}.")
    p("")
    return "\n".join(add)


# ------------------------------------------------------------- benchmarks


def render_benchmarks_section(data: dict) -> str:
    run = data["run"]
    s = data["summary"]
    ch = s.loc[rc.TOOL]
    same = s.loc[rc.CLOCK_SAME] if rc.CLOCK_SAME in s.index else s.loc[rc.CLOCK]
    n_all = int(ch["n_instances"])
    n_scored = int(ch["n_instances_with_threshold"])
    dataset = f"3W v2.0.0 real `{run['dataset_manifest_sha256'][:12]}`"
    code = code_cite(run["code"])
    design = "onset-aligned"
    split = (
        "no fit on 3W; own history, 3 baseline windows before onset, label-blind; no group "
        f"holdout; {n_scored} of {n_all} evaluable instances scored"
    )
    model = run["model"]
    over = float(ch["false_alarm_rate_pre"]) - rc.far_floor(rc.BASELINE_WINDOWS)
    same_over = float(same["false_alarm_rate_pre"]) - rc.far_floor(rc.BASELINE_WINDOWS)

    def metric_row(r: pd.Series, over_floor: float) -> str:
        return (
            f"AUC {auc_cell(r)}, paired hit {fmt(r['paired_hit_post0'])}/"
            f"{fmt(r['paired_hit_post1'])}, FAR {pct(r['false_alarm_rate_pre'])} "
            f"({signed(over_floor)} over the 25.0% floor), detect post1 "
            f"{pct(r['detection_rate_post1'])}"
        )

    rows = [
        ("none", "Chronos-Bolt small zero-shot: instances scored",
         f"{n_scored} of {n_all} ({pct(n_scored / n_all)})"),
        ("none", "Chronos-Bolt small zero-shot: instances refused",
         f"{n_all - n_scored} of {n_all}; a baseline window with under 60 context minutes "
         "or a constant variable"),
        ("none", "Chronos-Bolt small zero-shot forecast surprise (no fit on 3W)",
         metric_row(ch, over)),
        ("none", f"**clock control** on the same {n_scored} instances (window position in "
                 "its own instance, no sensor read)",
         metric_row(same, same_over)),
    ]
    lines = [
        SECTION_HEADING,
        "",
        f"A pretrained forecaster, `{model['model_id']}` (revision "
        f"`{model['model_revision'][:8]}`, `chronos-forecasting` "
        f"{model['chronos_forecasting_version']}), scores the detector study's "
        f"{run['refusals']['n_windows']} onset-aligned windows with nothing fitted on 3W. "
        "The score is the mean absolute error of the one-hour median forecast in units of "
        "the 10-90% band, over six variables at one-minute means from a four-hour context. "
        "Method, refusals and the per-fold spread are in "
        "[examples/studies/3w_chronos/REPORT.md](examples/studies/3w_chronos/REPORT.md).",
        "",
        "This section is not regenerated by `make bench`; it is produced by "
        "`examples/studies/3w_chronos/run_chronos.py` and `make_report.py`, and carried "
        "through the generator unchanged.",
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


# ------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results", default=str(HERE / "results"))
    parser.add_argument("--detector-results", default=str(DETECTOR_RESULTS))
    parser.add_argument("--out", default=str(HERE / "out"))
    parser.add_argument("--report", default=str(HERE / "REPORT.md"))
    parser.add_argument("--update-benchmarks", action="store_true")
    parser.add_argument("--benchmarks", default=str(BENCH_MD))
    args = parser.parse_args(argv)

    results = Path(args.results)
    data = load(results, Path(args.detector_results))
    plot_paired(data["per_instance"], Path(args.out) / "01_paired_surprise.png")
    Path(args.report).write_text(render_report(data), encoding="utf-8", newline="\n")
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
    print(f"wrote {args.report}, {Path(args.out) / '01_paired_surprise.png'} and "
          f"{results / 'benchmarks_section.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

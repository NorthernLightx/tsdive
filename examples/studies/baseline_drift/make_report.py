"""Report, figure and BENCHMARKS.md section for the baseline drift study.

Reads ``results/`` as ``run_drift.py`` wrote it and renders ``REPORT.md``,
``out/01_score_over_threshold.png`` and ``results/benchmarks_section.md``.
With ``--update-benchmarks`` the section replaces the
``## REAL: baseline drift (3W and SKAB)`` section of the repository's
``BENCHMARKS.md``; the SYNTHETIC table and the other REAL sections are
left as they are. Every number in the prose is read from the frames the
tables are built from.
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

import run_drift as rd  # noqa: E402

BENCH_MD = ROOT / "BENCHMARKS.md"
SECTION_HEADING = "## REAL: baseline drift (3W and SKAB)"
FIGURE_NAME = "01_score_over_threshold.png"
SKAB_FIGURE_RECORD = "anomaly-free__anomaly-free"

LABELS = {
    rd.SCORING_STATIC: "static",
    rd.SCORING_DIFF: "differenced",
    rd.SCORING_ROLLING: "rolling",
    rd.SCORING_CLOCK: "clock control",
}
COLORS = {
    rd.SCORING_STATIC: "#1f77b4",
    rd.SCORING_DIFF: "#2ca02c",
    rd.SCORING_ROLLING: "#9467bd",
    rd.SCORING_CLOCK: "#7f7f7f",
}
DATASET_3W = "3W v2.0.0 real `14bc1397d3ee`"
DATASET_SKAB = "SKAB `c0d612939333`"


# ---------------------------------------------------------------- formatting


def pct(value) -> str:
    return "refused" if pd.isna(value) else f"{float(value) * 100:.1f}%"


def signed(value) -> str:
    return "" if pd.isna(value) else f"{float(value):+.3f}"


def num(value, digits: int = 3) -> str:
    return "refused" if pd.isna(value) else f"{float(value):.{digits}f}"


def count(value) -> str:
    return "0" if pd.isna(value) else f"{int(value)}"


def unwrap(text: str) -> str:
    """Join each prose paragraph onto one line, as the other reports carry it.

    Fenced blocks, headings, table rows and image lines pass through. A list
    item and its indented continuations become one line.
    """
    out: list[str] = []
    buffer: list[str] = []
    fenced = False

    def flush() -> None:
        if buffer:
            out.append(" ".join(part.strip() for part in buffer))
            buffer.clear()

    for line in text.splitlines():
        if line.startswith("```"):
            flush()
            fenced = not fenced
            out.append(line)
            continue
        if fenced:
            out.append(line)
            continue
        stripped = line.strip()
        if not stripped:
            flush()
            out.append("")
            continue
        if stripped.startswith(("#", "|", "![")):
            flush()
            out.append(stripped)
            continue
        if stripped.startswith("- "):
            flush()
        buffer.append(stripped)
    flush()
    return "\n".join(out).rstrip("\n") + "\n"


def table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(out)


# ---------------------------------------------------------------- loading


def load(results: Path) -> dict:
    data = {
        "summary": pd.read_csv(results / "summary.csv"),
        "per_record": pd.read_csv(results / "per_record.csv"),
        "scores": pd.read_csv(results / "scores.csv"),
        "windows": pd.read_csv(results / "windows.csv"),
        "usability": pd.read_csv(results / "usability.csv"),
        "run": json.loads((results / "run.json").read_text("utf-8")),
    }
    return data


def row(data: dict, bed: str, group: str, scoring: str) -> pd.Series:
    frame = data["summary"]
    hit = frame[
        (frame["bed"] == bed) & (frame["group"] == group) & (frame["scoring"] == scoring)
    ]
    return hit.iloc[0]


def records(data: dict, bed: str, group: str, scoring: str) -> pd.DataFrame:
    frame = data["per_record"]
    return frame[
        (frame["bed"] == bed) & (frame["group"] == group) & (frame["scoring"] == scoring)
    ]


def post0_rate(data: dict, scoring: str) -> float | None:
    """Share of aligned instances whose post0 window fires: delay 0 among scored."""
    frame = records(data, rd.BED_3W_ALIGNED, rd.GROUP_ALIGNED, scoring)
    scored = frame[frame["threshold"].notna()]
    if scored.empty:
        return None
    return float((scored["delay_windows"] == 0).sum()) / len(scored)


# ---------------------------------------------------------------- tables


def normal_table(data: dict) -> str:
    rows = []
    for scoring in rd.SCORINGS:
        r = row(data, rd.BED_3W_OWN, rd.GROUP_NORMAL, scoring)
        rows.append(
            [
                LABELS[scoring],
                count(r["n_pre_windows"]),
                pct(r["far_window"]),
                signed(r["over_floor"]),
                pct(r["far_record"]),
            ]
        )
    return table(
        [
            "scoring",
            "stream windows",
            "false-alarm rate per window",
            "over the 25.0% floor",
            "records with an alarm",
        ],
        rows,
    )


def mixed_table(data: dict) -> str:
    rows = []
    for scoring in rd.SCORINGS:
        r = row(data, rd.BED_3W_OWN, rd.GROUP_MIXED, scoring)
        rows.append(
            [
                LABELS[scoring],
                num(r["roc_auc"], 3),
                pct(r["far_window"]),
                signed(r["over_floor"]),
                pct(r["detect_rate"]),
                num(r["median_delay_windows"], 1),
                num(r["median_consecutive_fault_fires"], 1),
            ]
        )
    return table(
        [
            "scoring",
            "AUC",
            "false-alarm rate before the first fault window",
            "over the 25.0% floor",
            "detect",
            "median delay (windows)",
            "median fault windows firing in a row",
        ],
        rows,
    )


def aligned_table(data: dict) -> str:
    rows = []
    for scoring in rd.SCORINGS:
        r = row(data, rd.BED_3W_ALIGNED, rd.GROUP_ALIGNED, scoring)
        post0 = post0_rate(data, scoring)
        rows.append(
            [
                LABELS[scoring],
                count(r["n_records_scored"]),
                pct(r["far_window"]),
                signed(r["over_floor"]),
                "refused" if post0 is None else pct(post0),
                pct(r["detect_rate"]),
            ]
        )
    return table(
        [
            "scoring",
            "instances scored",
            "false-alarm rate on pre",
            "over the 25.0% floor",
            "first alarm in post0",
            "detect by the end of post1",
        ],
        rows,
    )


def skab_table(data: dict) -> str:
    rows = []
    for scoring in rd.SCORINGS:
        r = row(data, rd.BED_SKAB, rd.GROUP_LABELLED, scoring)
        rows.append(
            [
                LABELS[scoring],
                num(r["roc_auc"], 3),
                pct(r["far_window"]),
                signed(r["over_floor"]),
                pct(r["detect_rate"]),
                num(r["median_delay_windows"], 1),
                pct(r["far_post"]),
                pct(r["recovered_rate"]),
            ]
        )
    return table(
        [
            "scoring",
            "AUC",
            "false-alarm rate before onset",
            "over the 25.0% floor",
            "detect",
            "median delay (windows)",
            "false-alarm rate after the span",
            "recovered within 2 windows",
        ],
        rows,
    )


def free_table(data: dict) -> str:
    rows = []
    for scoring in rd.SCORINGS:
        r = row(data, rd.BED_SKAB_FREE, rd.GROUP_FREE, scoring)
        rows.append(
            [
                LABELS[scoring],
                count(r["n_pre_windows"]),
                pct(r["far_window"]),
                signed(r["over_floor"]),
            ]
        )
    return table(
        ["scoring", "stream windows", "false-alarm rate per window", "over the 25.0% floor"],
        rows,
    )


def usability_table(data: dict) -> str:
    frame = data["usability"]
    rows = []
    for bed in (rd.BED_3W_OWN, rd.BED_3W_ALIGNED, rd.BED_SKAB, rd.BED_SKAB_FREE):
        for scoring in (rd.SCORING_STATIC, rd.SCORING_DIFF, rd.SCORING_ROLLING):
            part = frame[(frame["bed"] == bed) & (frame["scoring"] == scoring)]
            if part.empty:
                continue
            zero = part[part["reason"] == rd.REFUSAL_ZERO_SCALE]
            absent = part[part["reason"] == rd.REFUSAL_NO_FINITE]
            top = part["variable"].value_counts()
            rows.append(
                [
                    bed,
                    LABELS[scoring],
                    str(len(zero)),
                    str(len(absent)),
                    f"`{top.index[0]}` ({int(top.iloc[0])})",
                ]
            )
    return table(
        [
            "bed",
            "scoring",
            "pairs with a zero scale",
            "pairs with no finite minute median",
            "variable dropped most often",
        ],
        rows,
    )


def group_table(data: dict) -> str:
    counts = data["run"]["group_counts"]
    rows = [[key, str(value)] for key, value in sorted(counts.items())]
    return table(["bed and group", "records"], rows)


# ---------------------------------------------------------------- figure


def _trace(data: dict, bed: str, record: str) -> dict[str, pd.DataFrame]:
    scores = data["scores"]
    part = scores[(scores["bed"] == bed) & (scores["record"] == record)]
    out = {}
    for scoring in (rd.SCORING_STATIC, rd.SCORING_DIFF, rd.SCORING_ROLLING):
        rows = records(data, bed, part_group(data, bed, record), scoring)
        hit = rows[rows["record"] == record]
        if hit.empty or pd.isna(hit.iloc[0]["threshold"]):
            continue
        threshold = float(hit.iloc[0]["threshold"])
        trace = part[(part["scoring"] == scoring) & part["score"].notna()].copy()
        trace["ratio"] = trace["score"] / threshold
        out[scoring] = trace
    return out


def part_group(data: dict, bed: str, record: str) -> str:
    frame = data["per_record"]
    hit = frame[(frame["bed"] == bed) & (frame["record"] == record)]
    return str(hit.iloc[0]["group"])


def pick_3w_record(data: dict) -> str:
    """The normal 3W record whose static score fires on the most stream windows."""
    frame = records(data, rd.BED_3W_OWN, rd.GROUP_NORMAL, rd.SCORING_STATIC)
    scored = frame[frame["threshold"].notna()].copy()
    scored["share"] = scored["n_pre_fired"] / scored["n_pre"]
    scored = scored.sort_values(["share", "record"], ascending=[False, True], kind="stable")
    return str(scored.iloc[0]["record"])


def plot_ratio(data: dict, path: Path) -> str:
    record_3w = pick_3w_record(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.4))
    panels = [
        (axes[0], rd.BED_3W_OWN, record_3w, "3W instance with no fault window"),
        (
            axes[1],
            rd.BED_SKAB_FREE,
            SKAB_FIGURE_RECORD,
            "SKAB record with no labelled anomaly",
        ),
    ]
    for axis, bed, record, title in panels:
        for scoring, trace in _trace(data, bed, record).items():
            axis.plot(
                trace["position"],
                trace["ratio"],
                color=COLORS[scoring],
                linewidth=1.2,
                label=LABELS[scoring],
            )
        axis.axhline(1.0, color="#d62728", linewidth=1.0, linestyle="--")
        axis.set_yscale("log")
        axis.set_title(f"{title}: {record}", fontsize=10)
        axis.set_xlabel("window position in the record")
        axis.set_ylabel("score / alarm threshold")
        axis.legend(fontsize=8, loc="upper left")
        axis.grid(alpha=0.25)
    fig.suptitle(
        "A window fires where its score sits above the dashed line", fontsize=11
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return record_3w


# ---------------------------------------------------------------- report


def render_report(data: dict, figure_record: str) -> str:
    run = data["run"]
    parameters = run["parameters"]
    k = parameters["k_fit_windows"]
    w = parameters["w_rolling_windows"]
    normal = {s: row(data, rd.BED_3W_OWN, rd.GROUP_NORMAL, s) for s in rd.SCORINGS}
    mixed = {s: row(data, rd.BED_3W_OWN, rd.GROUP_MIXED, s) for s in rd.SCORINGS}
    aligned = {s: row(data, rd.BED_3W_ALIGNED, rd.GROUP_ALIGNED, s) for s in rd.SCORINGS}
    skab = {s: row(data, rd.BED_SKAB, rd.GROUP_LABELLED, s) for s in rd.SCORINGS}
    free = {s: row(data, rd.BED_SKAB_FREE, rd.GROUP_FREE, s) for s in rd.SCORINGS}
    normal_scored = int(normal[rd.SCORING_STATIC]["n_records_scored"])
    normal_all = int(normal[rd.SCORING_STATIC]["n_records"])
    mixed_scored = int(mixed[rd.SCORING_STATIC]["n_records_scored"])
    mixed_all = int(mixed[rd.SCORING_STATIC]["n_records"])
    skab_scored = int(skab[rd.SCORING_STATIC]["n_records"])
    rolling_refusal = str(aligned[rd.SCORING_ROLLING]["record_refusal"])
    mixed_rows = records(data, rd.BED_3W_OWN, rd.GROUP_MIXED, rd.SCORING_STATIC)
    mixed_ok = mixed_rows[mixed_rows["threshold"].notna()]
    fault_len = int(
        (mixed_ok["last_fault_position"] - mixed_ok["first_fault_position"] + 1).median()
    )
    mixed_len = int(mixed_ok["n_windows"].median())
    normal_rows = records(data, rd.BED_3W_OWN, rd.GROUP_NORMAL, rd.SCORING_STATIC)
    normal_ok = normal_rows[normal_rows["threshold"].notna()]
    stream_min = int(normal_ok["n_pre"].min())
    stream_max = int(normal_ok["n_pre"].max())

    return f"""# Baseline drift: static, differenced and rolling baselines

## Summary

A static own-history baseline fires on {pct(normal[rd.SCORING_STATIC]["far_window"])} of the
stream windows of 3W records that carry no fault and on
{pct(free[rd.SCORING_STATIC]["far_window"])} of the SKAB anomaly-free record, against a
{pct(normal[rd.SCORING_STATIC]["far_floor"])} floor. Scoring the minute-to-minute
difference takes those rates to {pct(normal[rd.SCORING_DIFF]["far_window"])} and
{pct(free[rd.SCORING_DIFF]["far_window"])}, and moving the baseline to the {w} windows
before each scored window takes them to {pct(normal[rd.SCORING_ROLLING]["far_window"])} and
{pct(free[rd.SCORING_ROLLING]["far_window"])}. Detection on the 3W records whose fault
arrives later falls from {pct(mixed[rd.SCORING_STATIC]["detect_rate"])} to
{pct(mixed[rd.SCORING_DIFF]["detect_rate"])} for the differenced score and
{pct(mixed[rd.SCORING_ROLLING]["detect_rate"])} for the rolling baseline, and the median
delay goes from {num(mixed[rd.SCORING_STATIC]["median_delay_windows"], 1)} windows to
{num(mixed[rd.SCORING_ROLLING]["median_delay_windows"], 1)}. A 3W fault that lasts a median
of {fault_len} windows stays above a rolling threshold for a median of
{num(mixed[rd.SCORING_ROLLING]["median_consecutive_fault_fires"], 1)} consecutive fault
windows before the baseline has absorbed it, against
{num(mixed[rd.SCORING_STATIC]["median_consecutive_fault_fires"], 1)} for the static
baseline. The alarm rule needs seven windows, so it scores {normal_scored} of the
{normal_all} 3W records with no fault window.

## Data

Two beds come from the 3W window caches the detector study builds, and two
from the SKAB archives.

- 3W v2.0.0, dataset manifest sha256 `{run["dataset_manifest_sha256"]["3w"]}`.
  The own-history cache is `{run["beds"][rd.BED_3W_OWN]["cache"]}`, manifest
  `{run["beds"][rd.BED_3W_OWN]["manifest_sha256"]}`,
  {run["beds"][rd.BED_3W_OWN]["n_instances"]} instances and
  {run["beds"][rd.BED_3W_OWN]["n_windows"]} one-hour windows, each window holding 60
  one-minute sub-group medians for the six variables every instance shares. The
  onset-aligned cache is `{run["beds"][rd.BED_3W_ALIGNED]["cache"]}`, manifest
  `{run["beds"][rd.BED_3W_ALIGNED]["manifest_sha256"]}`,
  {run["beds"][rd.BED_3W_ALIGNED]["n_instances"]} instances of six windows.
- SKAB, dataset manifest sha256 `{run["dataset_manifest_sha256"]["skab"]}`,
  {run["beds"][rd.BED_SKAB]["n_records"]} records converted to tsdive archives under
  `{run["beds"][rd.BED_SKAB]["archives"]}`. The windows are 60 s from each record's
  first timestamp, the same ones `examples/studies/skab/results/windows.csv`
  carries, and the minute median of a window is the median of its GOOD values.

Nothing is fitted across records, so no group holdout applies. The 3W folds
are wells and the SKAB folds are the upstream folders; both are recorded in
the result frames so a later reader can group by them.

{group_table(data)}

Records whose baseline windows already carry a fault row, and records with
{k} windows or fewer, are counted and not scored.

## Method

Run the study and render this report:

```
uv run python examples/studies/baseline_drift/run_drift.py
uv run python examples/studies/baseline_drift/make_report.py --update-benchmarks
```

Every scoring gives one score per window: the largest robust z over the
usable variables and over the window's minutes. The centre and scale are the
median and 1.4826 x MAD of a set of minute medians, the constants
`tsdive.baselines.mad_baseline` uses, computed with numpy so that a set
smaller than the 30 samples that function requires still yields a number.

- `static`: the set is the minute medians of the first {k} windows.
- `differenced`: the set is the minute-to-minute differences inside those
  windows, and a window's score reads the differences inside it. The first
  minute of a window whose predecessor is missing from the cache carries no
  difference.
- `rolling`: the set is the minute medians of the {w} windows immediately
  before the scored window, recomputed for every window and label-blind.
- `clock`: the window's position in its own record, through
  `tsdive.eval.clock_control`. It reads no sensor value.

A variable is unusable when the scale over its baseline set is zero or when
that set holds no finite minute median. For `rolling` the verdict is per
window. A window with no usable variable is refused with the reason.

The alarm rule is the same for all four. On the own-history beds the
threshold is `worst_baseline_threshold` over the scores of window positions
3, 4 and 5, which are the first three windows a rolling baseline can score,
and the stream is position 6 onward. On the onset-aligned bed the threshold
comes from the three baseline windows, as in the detector study, and the
stream is pre, post0 and post1. `fires` is strict, so
`far_floor({k})` = {pct(normal[rd.SCORING_STATIC]["far_floor"])} is the rate a
score that reads nothing already reaches.

## Results

### 3W records with no fault window, own history

{normal_scored} of {normal_all} records carry the seven windows the rule needs.
Their stream lengths run from {stream_min} to {stream_max} windows.

{normal_table(data)}

### 3W records whose fault arrives later, own history

{mixed_scored} of {mixed_all} records carry the seven windows the rule needs.
The median record holds {mixed_len} windows and its fault runs for {fault_len} of them.

{mixed_table(data)}

### 3W onset-aligned instances

`rolling` is refused on this bed: {rolling_refusal}.

{aligned_table(data)}

### SKAB labelled records

{skab_scored} records, each about 20 minutes long, every one returning to
normal after its fault.

{skab_table(data)}

### SKAB anomaly-free record

{free_table(data)}

### Variables a scoring could not use

One row per (record, variable) pair dropped, over the records the rule
scores.

{usability_table(data)}

![score over threshold](out/{FIGURE_NAME})

The figure plots each scoring's score divided by its own alarm threshold, so
a window fires where its curve sits above 1. The upper panel is
`{figure_record}`, the 3W record with no fault window whose static score
fires on the largest share of its stream. The lower panel is the SKAB
record with no labelled anomaly.

## Discussion

On both beds the static baseline reaches a false-alarm rate that the floor
does not explain: {pct(normal[rd.SCORING_STATIC]["far_window"])} per window on the 3W
records with no fault ({signed(normal[rd.SCORING_STATIC]["over_floor"])} over the floor)
and {pct(free[rd.SCORING_STATIC]["far_window"])} on the SKAB anomaly-free record
({signed(free[rd.SCORING_STATIC]["over_floor"])}). The clock control reaches
{pct(normal[rd.SCORING_CLOCK]["far_window"])} on both, because a score that rises with
position crosses the worst of three early windows at every later window. The
static score sits between the two, which is what a level that keeps moving
away from the fit windows produces.

Differencing removes most of that. It takes the 3W rate to
{pct(normal[rd.SCORING_DIFF]["far_window"])} ({signed(normal[rd.SCORING_DIFF]["over_floor"])})
and the SKAB anomaly-free rate to {pct(free[rd.SCORING_DIFF]["far_window"])}
({signed(free[rd.SCORING_DIFF]["over_floor"])}). The cost is detection: on the 3W
records whose fault arrives later it detects
{pct(mixed[rd.SCORING_DIFF]["detect_rate"])} against
{pct(mixed[rd.SCORING_STATIC]["detect_rate"])}, and on the onset-aligned instances it
raises a first alarm in post0 on {pct(post0_rate(data, rd.SCORING_DIFF))} of them
against {pct(post0_rate(data, rd.SCORING_STATIC))}. A differenced score reads the
edge of a fault, so a fault that arrives inside a window it does not cover is
gone from the score by the next window.

The rolling baseline takes the rates lowest:
{pct(normal[rd.SCORING_ROLLING]["far_window"])} on the 3W records with no fault
({signed(normal[rd.SCORING_ROLLING]["over_floor"])}) and
{pct(free[rd.SCORING_ROLLING]["far_window"])} on the SKAB anomaly-free record
({signed(free[rd.SCORING_ROLLING]["over_floor"])}), the lowest of the three on that
record. It keeps
{pct(mixed[rd.SCORING_ROLLING]["detect_rate"])} detection on the 3W records whose fault
arrives later, above the differenced score, and pays for it in delay: the
median first alarm arrives {num(mixed[rd.SCORING_ROLLING]["median_delay_windows"], 1)}
windows after the first fault window against
{num(mixed[rd.SCORING_STATIC]["median_delay_windows"], 1)} for the static baseline.

A moving baseline also stops seeing a fault that stays. On the 3W records
whose fault arrives later the median fault runs for {fault_len} windows; the rolling
score keeps a median of
{num(mixed[rd.SCORING_ROLLING]["median_consecutive_fault_fires"], 1)} consecutive fault
windows above its threshold, against
{num(mixed[rd.SCORING_STATIC]["median_consecutive_fault_fires"], 1)} for the static
baseline and {num(mixed[rd.SCORING_CLOCK]["median_consecutive_fault_fires"], 1)} for the
clock, which fires on every window after the threshold ones. Three windows of
history absorb an hour-scale step in about two windows. On SKAB, where the
fault ends and the record returns to normal, that same property cuts the
false alarms after the span: the rolling baseline's rate there is
{pct(skab[rd.SCORING_ROLLING]["far_post"])} against
{pct(skab[rd.SCORING_STATIC]["far_post"])} for the static baseline, and it goes quiet
within two windows on {pct(skab[rd.SCORING_ROLLING]["recovered_rate"])} of the records
against {pct(skab[rd.SCORING_STATIC]["recovered_rate"])}.

The record-level rates move the other way. On the 3W records with no fault
the share of records raising at least one alarm is
{pct(normal[rd.SCORING_STATIC]["far_record"])} for the static baseline and
{pct(normal[rd.SCORING_ROLLING]["far_record"])} for the rolling one, while the per-window
rates are {pct(normal[rd.SCORING_STATIC]["far_window"])} and
{pct(normal[rd.SCORING_ROLLING]["far_window"])}. A local threshold is a low threshold, so
any single window that steps out of its own three-window history crosses it.
A rolling baseline shortens an alarm. It does not stop one from starting.

The ranking metrics run the same way. On the 3W records whose fault arrives later
the static AUC is {num(mixed[rd.SCORING_STATIC]["roc_auc"], 3)} and the clock control
takes {num(mixed[rd.SCORING_CLOCK]["roc_auc"], 3)}, so the static score ranks below a
timer; the differenced and rolling AUCs are
{num(mixed[rd.SCORING_DIFF]["roc_auc"], 3)} and
{num(mixed[rd.SCORING_ROLLING]["roc_auc"], 3)}, which is what a score that reads only
the edge of a fault gives on a record where most fault windows are steady.
On SKAB the order is different: the clock takes
{num(skab[rd.SCORING_CLOCK]["roc_auc"], 3)} and the static score
{num(skab[rd.SCORING_STATIC]["roc_auc"], 3)}, because a SKAB fault ends and position
alone no longer separates the labels.

## Limits and further work

- The rule needs seven windows, so it scores {normal_scored} of the
  {normal_all} 3W records with no fault window and {mixed_scored} of the
  {mixed_all} whose fault arrives later. The static rate here is not
  comparable with the 76.3% the conformal study reports over 537 records
  under a rule that needs four windows
  (`examples/studies/3w_conformal/REPORT.md`).
- A SKAB window is 60 s, so its minute median is one number and a baseline
  of three windows is three numbers. That is why `Pressure` is unusable in
  every SKAB record here while the SKAB study, whose baseline reads 180
  one-second samples, keeps it in five of them.
- `rolling` is refused on the onset-aligned bed. The design keeps six
  windows per instance and the threshold windows are the first three, which
  have no predecessors in the cache.
- The study scores the one-minute medians only. It says nothing about a
  scoring's behaviour on the raw one-second samples, which is what `screen`
  and `spc` read when you point them at an archive.
- No scoring here is wired into `screen`, `spc` or `mspc`. The package still
  fits one baseline over a fixed history.

## Files

| file | holds |
|---|---|
| `run_drift.py` | the study runner |
| `make_report.py` | this report, the figure and the BENCHMARKS section |
| `results/windows.csv` | one row per window of every record the rule scores |
| `results/scores.csv` | one row per (bed, record, scoring, window): the score or the refusal |
| `results/per_record.csv` | the alarm rule and its metrics per record |
| `results/summary.csv` | one row per (bed, group, scoring) |
| `results/usability.csv` | one row per variable a scoring could not use |
| `results/run.json` | manifests, code version, parameters, counts |
| `results/benchmarks_section.md` | the BENCHMARKS.md REAL section |
| `out/{FIGURE_NAME}` | the figure above |
"""


# ---------------------------------------------------------------- benchmarks


def render_benchmarks_section(data: dict) -> str:
    run = data["run"]
    code = f"tsdive {run['code']['tsdive_version']} @ `{run['code']['git_head'][:8]}`"
    k = run["parameters"]["k_fit_windows"]
    w = run["parameters"]["w_rolling_windows"]
    own_manifest = run["beds"][rd.BED_3W_OWN]["manifest_sha256"]
    aligned_manifest = run["beds"][rd.BED_3W_ALIGNED]["manifest_sha256"]
    normal_scored = int(row(data, rd.BED_3W_OWN, rd.GROUP_NORMAL, rd.SCORING_STATIC)[
        "n_records_scored"
    ])
    mixed_scored = int(row(data, rd.BED_3W_OWN, rd.GROUP_MIXED, rd.SCORING_STATIC)[
        "n_records_scored"
    ])
    split_own = (
        f"own history: {k} fit windows, threshold from window positions 3 to 5, "
        f"label-blind; nothing fitted across records, so no group holdout applies; "
        f"folds are wells; {normal_scored} records with no fault window and "
        f"{mixed_scored} whose fault arrives later"
    )
    split_aligned = (
        f"own history: {k} baseline windows before the onset, label-blind; "
        f"no group holdout; 48 instances; cache manifest `{aligned_manifest}`"
    )
    split_skab = (
        f"own history: first {k} windows of each record, label-blind; nothing "
        "fitted across records, so no group holdout applies; folds are the "
        "upstream folders; 34 labelled records, 1 anomaly-free"
    )
    split_own_full = f"{split_own}; cache manifest `{own_manifest}`"

    rows: list[tuple[str, str, str, str, str]] = []
    for scoring in rd.SCORINGS:
        label = LABELS[scoring]
        mark = "**clock control** " if scoring == rd.SCORING_CLOCK else ""
        normal = row(data, rd.BED_3W_OWN, rd.GROUP_NORMAL, scoring)
        mixed = row(data, rd.BED_3W_OWN, rd.GROUP_MIXED, scoring)
        rows.append(
            (
                f"{mark}{label} baseline: false-alarm rate on 3W records with no "
                "fault window",
                f"FAR {pct(normal['far_window'])} per window "
                f"({signed(normal['over_floor'])} over the "
                f"{pct(normal['far_floor'])} floor), "
                f"{pct(normal['far_record'])} of records raise one",
                DATASET_3W,
                "own-history",
                split_own_full,
            )
        )
        rows.append(
            (
                f"{mark}{label} baseline: 3W records whose fault arrives later",
                f"AUC {num(mixed['roc_auc'], 3)}, FAR before the first fault window "
                f"{pct(mixed['far_window'])} ({signed(mixed['over_floor'])} over the "
                f"{pct(mixed['far_floor'])} floor), detect "
                f"{pct(mixed['detect_rate'])}, median delay "
                f"{num(mixed['median_delay_windows'], 1)} windows, median "
                f"{num(mixed['median_consecutive_fault_fires'], 1)} fault windows "
                "firing in a row",
                DATASET_3W,
                "own-history",
                split_own_full,
            )
        )
    for scoring in rd.SCORINGS:
        label = LABELS[scoring]
        mark = "**clock control** " if scoring == rd.SCORING_CLOCK else ""
        aligned = row(data, rd.BED_3W_ALIGNED, rd.GROUP_ALIGNED, scoring)
        post0 = post0_rate(data, scoring)
        if pd.isna(aligned["far_window"]):
            value = f"refused: {aligned['record_refusal']}"
        else:
            value = (
                f"FAR on pre {pct(aligned['far_window'])} "
                f"({signed(aligned['over_floor'])} over the "
                f"{pct(aligned['far_floor'])} floor), first alarm in post0 "
                f"{pct(post0)}, detect post1 {pct(aligned['detect_rate'])}"
            )
        rows.append(
            (
                f"{mark}{label} baseline, onset-aligned instances",
                value,
                DATASET_3W,
                "onset-aligned",
                split_aligned,
            )
        )
    for scoring in rd.SCORINGS:
        label = LABELS[scoring]
        mark = "**clock control** " if scoring == rd.SCORING_CLOCK else ""
        skab = row(data, rd.BED_SKAB, rd.GROUP_LABELLED, scoring)
        free = row(data, rd.BED_SKAB_FREE, rd.GROUP_FREE, scoring)
        rows.append(
            (
                f"{mark}{label} baseline: SKAB labelled records",
                f"AUC {num(skab['roc_auc'], 3)}, FAR before onset "
                f"{pct(skab['far_window'])} ({signed(skab['over_floor'])} over the "
                f"{pct(skab['far_floor'])} floor), detect {pct(skab['detect_rate'])}, "
                f"median delay {num(skab['median_delay_windows'], 1)} windows, FAR "
                f"after the span {pct(skab['far_post'])}, recovered within 2 windows "
                f"{pct(skab['recovered_rate'])}",
                DATASET_SKAB,
                "own-history",
                split_skab,
            )
        )
        rows.append(
            (
                f"{mark}{label} baseline: SKAB anomaly-free record",
                f"FAR {pct(free['far_window'])} ({signed(free['over_floor'])} over the "
                f"{pct(free['far_floor'])} floor) over "
                f"{count(free['n_pre_windows'])} windows",
                DATASET_SKAB,
                "own-history",
                split_skab,
            )
        )

    lines = [
        SECTION_HEADING,
        "",
        "Three own-history baselines scored by one code path on the 3W window "
        "caches and the SKAB archives, with the clock control beside them. "
        "`static` fits the median and MAD of the first three windows' minute "
        f"medians, `differenced` fits the same over the minute-to-minute "
        f"differences, and `rolling` refits over the {w} windows before every "
        "scored window. Method, groups and refusals are in "
        "[examples/studies/baseline_drift/REPORT.md]"
        "(examples/studies/baseline_drift/REPORT.md).",
        "",
        "This section is not regenerated by `make bench`; it is produced by "
        "`examples/studies/baseline_drift/run_drift.py` and `make_report.py`, "
        "and carried through the generator unchanged.",
        "",
        "| stage | metric | value | dataset (manifest sha256) | code | design | split |",
        "|---|---|---|---|---|---|---|",
    ]
    lines += [
        f"| none | {metric} | {value} | {dataset} | {code} | {design} | {split} |"
        for metric, value, dataset, design, split in rows
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
    figure_record = plot_ratio(data, Path(args.out) / FIGURE_NAME)
    Path(args.report).write_text(
        unwrap(render_report(data, figure_record)), encoding="utf-8", newline="\n"
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
    print(f"wrote {args.report} and {results / 'benchmarks_section.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

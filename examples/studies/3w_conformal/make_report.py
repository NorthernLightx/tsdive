"""REPORT.md and the BENCHMARKS.md section for the 3W conformal martingale study.

Reads ``results/`` written by ``run_conformal.py`` and, for the aligned
comparison, the detector study's frozen ``results/aligned_summary.csv``.
Writes ``REPORT.md`` and ``results/benchmarks_section.md``;
``--update-benchmarks`` replaces this study's ``## REAL`` section in
``BENCHMARKS.md`` or appends it.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))

import run_conformal as rc  # noqa: E402

from tsdive.eval import far_floor  # noqa: E402

DETECTOR_RESULTS = HERE.parent / "3w_detectors" / "results"
BENCH_MD = ROOT / "BENCHMARKS.md"
SECTION_HEADING = "## REAL: 3W conformal test martingale"
FLOOR = far_floor(rc.DEFAULT_K)

RULE_LABELS = {
    rc.METHOD_CONFORMAL: "conformal martingale",
    rc.METHOD_WORST: "worst-baseline threshold",
    rc.METHOD_CLOCK: "clock control (position, no sensor read)",
    rc.METHOD_PERMUTATION: "permutation check (scores shuffled)",
}
GROUP_LABELS = {
    rc.GROUP_NORMAL: "normal",
    rc.GROUP_MIXED: "mixed",
    rc.GROUP_POSITIVE_BASELINE: "positive baseline",
    rc.GROUP_SHORT: "short",
    rc.GROUP_ALIGNED: "aligned",
}
DETECTOR_ROWS = (("mad", "MAD screen, own history"), ("spc", "SPC individuals rules, own history"))
LENGTH_BUCKETS = ((1, 1, "1"), (2, 5, "2 to 5"), (6, 10_000, "6 or more"))


# ---------------------------------------------------------------- loading


def load(results: Path, detector: Path) -> dict:
    per_instance = pd.read_csv(
        results / "per_instance.csv", dtype={"folder_label": str, "refusal": str}
    )
    per_instance["refusal"] = per_instance["refusal"].fillna("")
    data = {
        "summary": pd.read_csv(results / "summary.csv"),
        "per_instance": per_instance,
        "permutation": pd.read_csv(results / "permutation.csv"),
        "run": json.loads((results / "run.json").read_text("utf-8")),
        "det_summary": None,
        "det_code": None,
    }
    aligned_summary = detector / "aligned_summary.csv"
    if aligned_summary.exists():
        det = pd.read_csv(aligned_summary)
        data["det_summary"] = det[det["variant"] == "all"].set_index("tool")
        det_run = json.loads((detector / "aligned_run.json").read_text("utf-8"))
        data["det_code"] = code_cite(det_run["provenance"]["code"])
    return data


def fmt(value, nd: int = 3) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.{nd}f}"


def pct(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{100 * float(value):.1f}%"


def pct2(value) -> str:
    """Two decimals, for the permutation fractions that sit near zero."""
    if value is None or pd.isna(value):
        return ""
    return f"{100 * float(value):.2f}%"


def signed(value, nd: int = 3) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):+.{nd}f}"


def code_cite(run_code: dict) -> str:
    version = run_code.get("tsdive_version") or run_code.get("tagledger_version")
    name = "tsdive" if "tsdive_version" in run_code else "tagledger"
    return f"{name} {version} @ `{run_code['git_head'][:8]}`"


def delta_cell(value) -> str:
    return "" if value is None or pd.isna(value) else f"{float(value):g}"


def summary_row(summary: pd.DataFrame, bed: str, group: str, method: str, delta=None) -> pd.Series:
    g = summary[
        (summary["bed"] == bed) & (summary["group"] == group) & (summary["method"] == method)
    ]
    g = g[g["delta"].isna()] if delta is None else g[g["delta"] == delta]
    assert len(g) == 1, (bed, group, method, delta)
    return g.iloc[0]


def rule_rows(deltas: list[float]) -> list[tuple[str, float | None]]:
    rows: list[tuple[str, float | None]] = [(rc.METHOD_CONFORMAL, d) for d in deltas]
    rows.append((rc.METHOD_WORST, None))
    rows += [(rc.METHOD_CLOCK, d) for d in deltas]
    rows += [(rc.METHOD_PERMUTATION, d) for d in deltas]
    return rows


def table(header: list[str], align: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(align) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return out


# ----------------------------------------------------------------- tables


def own_history_table(summary: pd.DataFrame, deltas: list[float]) -> list[str]:
    rows = []
    for group in (rc.GROUP_NORMAL, rc.GROUP_MIXED, rc.GROUP_POSITIVE_BASELINE):
        for method, delta in rule_rows(deltas):
            r = summary_row(summary, rc.BED_OWN, group, method, delta)
            rate = pct2 if method == rc.METHOD_PERMUTATION else pct
            rows.append(
                [
                    GROUP_LABELS[group],
                    RULE_LABELS[method],
                    delta_cell(r["delta"]),
                    f"{int(r['n_scored'])} of {int(r['n_instances'])}",
                    rate(r["alarm_rate"]),
                    pct(r["far"]),
                    pct(r["detect"]),
                    fmt(r["median_delay_windows"], 0),
                    fmt(r["median_max_log10_martingale"], 1),
                ]
            )
    return table(
        [
            "group",
            "rule",
            "delta",
            "scored",
            "alarm rate",
            "FAR",
            "detect",
            "median delay (windows)",
            "median max log10 M",
        ],
        ["---", "---", "---:", "---:", "---:", "---:", "---:", "---:", "---:"],
        rows,
    )


def aligned_table(data: dict, deltas: list[float]) -> list[str]:
    summary = data["summary"]
    this_code = code_cite(data["run"]["code"])
    rows = []
    for method, delta in rule_rows(deltas):
        r = summary_row(summary, rc.BED_ALIGNED, rc.GROUP_ALIGNED, method, delta)
        far = r["far"]
        rate = pct2 if method == rc.METHOD_PERMUTATION else pct
        rows.append(
            [
                RULE_LABELS[method],
                delta_cell(r["delta"]),
                f"{int(r['n_scored'])} of {int(r['n_instances'])}",
                rate(r["alarm_rate"]),
                pct(far),
                signed(far - FLOOR) if pd.notna(far) else "",
                pct(r["detect_post0"]),
                pct(r["detect_post1"]),
                fmt(r["median_max_log10_martingale"], 1),
                f"this study, {this_code}",
            ]
        )
    det = data["det_summary"]
    if det is not None:
        for tool, label in DETECTOR_ROWS:
            d = det.loc[tool]
            rows.append(
                [
                    label,
                    "",
                    f"{int(d['n_instances_with_threshold'])} of {int(d['n_instances'])}",
                    "",
                    pct(d["false_alarm_rate_pre"]),
                    signed(float(d["false_alarm_rate_pre"]) - FLOOR),
                    pct(d["detection_rate_post0"]),
                    pct(d["detection_rate_post1"]),
                    "",
                    f"detector study, {data['det_code']}",
                ]
            )
    return table(
        [
            "rule",
            "delta",
            "scored",
            "crossed by end of post1",
            "FAR on pre",
            "over floor",
            "detect post0",
            "detect post1",
            "median max log10 M",
            "source",
        ],
        ["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---"],
        rows,
    )


def permutation_table(data: dict) -> list[str]:
    perm = data["permutation"]
    rows = []
    for (bed, group, delta), g in perm.groupby(["bed", "group", "delta"], sort=True):
        rows.append(
            [
                bed.replace("_", " "),
                GROUP_LABELS[group],
                delta_cell(delta),
                str(int(g["n_instances"].iloc[0])),
                str(len(g)),
                pct2(g["alarm_fraction"].mean()),
                pct2(g["alarm_fraction"].min()),
                pct2(g["alarm_fraction"].max()),
            ]
        )
    return table(
        ["bed", "group", "delta", "instances", "seeds", "mean alarm fraction", "min", "max"],
        ["---", "---", "---:", "---:", "---:", "---:", "---:", "---:"],
        rows,
    )


def length_table(per_instance: pd.DataFrame, delta: float) -> list[str]:
    """FAR of the normal group by the number of stream windows, per rule."""
    normal = per_instance[
        (per_instance["bed"] == rc.BED_OWN)
        & (per_instance["group"] == rc.GROUP_NORMAL)
        & (per_instance["outcome"] != "refused")
    ]
    rows = []
    for lo, hi, label in LENGTH_BUCKETS:
        bucket = normal[normal["n_stream_windows"].between(lo, hi)]
        cells = [label]
        n = None
        for method in (rc.METHOD_CONFORMAL, rc.METHOD_WORST, rc.METHOD_CLOCK):
            g = bucket[bucket["method"] == method]
            g = g[g["delta"].isna()] if method == rc.METHOD_WORST else g[g["delta"] == delta]
            n = len(g)
            cells.append(pct((g["outcome"] == "false_alarm").mean()) if n else "")
        rows.append([cells[0], str(n or 0), *cells[1:]])
    return table(
        [
            "stream windows",
            "instances",
            f"conformal FAR (delta {delta:g})",
            "worst-baseline FAR",
            f"clock FAR (delta {delta:g})",
        ],
        ["---", "---:", "---:", "---:", "---:"],
        rows,
    )


def refusal_table(data: dict) -> list[str]:
    rows = []
    for bed in (rc.BED_OWN, rc.BED_ALIGNED):
        info = data["run"]["beds"][bed]
        refusals = info["refusals"] or {"none": 0}
        for reason, n in refusals.items():
            rows.append([bed.replace("_", " "), reason, str(n), str(info["n_instances"])])
    return table(
        ["bed", "refusal", "instances", "instances in cache"],
        ["---", "---", "---:", "---:"],
        rows,
    )


def usable_table(per_instance: pd.DataFrame) -> list[str]:
    scored = per_instance[
        (per_instance["method"] == rc.METHOD_WORST) & (per_instance["outcome"] != "refused")
    ]
    counts = (
        scored.groupby(["bed", "n_usable_variables"]).size().unstack(0, fill_value=0).sort_index()
    )
    rows = [
        [str(int(k)), *(str(int(counts.loc[k, bed])) if bed in counts else "0" for bed in (
            rc.BED_OWN, rc.BED_ALIGNED))]
        for k in counts.index
    ]
    return table(
        ["usable variables", "own history instances", "aligned instances"],
        ["---:", "---:", "---:"],
        rows,
    )


# ----------------------------------------------------------------- report


def render_report(data: dict) -> str:
    run = data["run"]
    summary = data["summary"]
    pi = data["per_instance"]
    deltas = [float(d) for d in run["parameters"]["deltas"]]
    d0 = deltas[0]
    seeds = int(run["parameters"]["permutation_seeds"])
    own, ali = run["beds"][rc.BED_OWN], run["beds"][rc.BED_ALIGNED]
    this_code = code_cite(run["code"])
    dataset_sha = (run.get("dataset_manifest_sha256") or "unknown")[:12]

    def own_row(group: str, method: str, delta=None) -> pd.Series:
        return summary_row(summary, rc.BED_OWN, group, method, delta)

    def ali_row(method: str, delta=None) -> pd.Series:
        return summary_row(summary, rc.BED_ALIGNED, rc.GROUP_ALIGNED, method, delta)

    n_normal = own_row(rc.GROUP_NORMAL, rc.METHOD_WORST)
    c_norm = {d: own_row(rc.GROUP_NORMAL, rc.METHOD_CONFORMAL, d) for d in deltas}
    w_norm = own_row(rc.GROUP_NORMAL, rc.METHOD_WORST)
    k_norm = own_row(rc.GROUP_NORMAL, rc.METHOD_CLOCK, d0)
    p_norm = {d: own_row(rc.GROUP_NORMAL, rc.METHOD_PERMUTATION, d) for d in deltas}
    c_mix = own_row(rc.GROUP_MIXED, rc.METHOD_CONFORMAL, d0)
    w_mix = own_row(rc.GROUP_MIXED, rc.METHOD_WORST)
    k_mix = own_row(rc.GROUP_MIXED, rc.METHOD_CLOCK, d0)
    c_ali = {d: ali_row(rc.METHOD_CONFORMAL, d) for d in deltas}
    w_ali = ali_row(rc.METHOD_WORST)
    k_ali = ali_row(rc.METHOD_CLOCK, d0)
    p_ali = {d: ali_row(rc.METHOD_PERMUTATION, d) for d in deltas}
    perm_max = float(data["permutation"]["alarm_fraction"].max())
    normal_streams = pi[
        (pi["bed"] == rc.BED_OWN) & (pi["group"] == rc.GROUP_NORMAL)
        & (pi["method"] == rc.METHOD_WORST) & (pi["outcome"] != "refused")
    ]["n_stream_windows"]
    mixed_streams = pi[
        (pi["bed"] == rc.BED_OWN) & (pi["group"] == rc.GROUP_MIXED)
        & (pi["method"] == rc.METHOD_WORST)
    ]["n_stream_windows"]
    usable = pi[(pi["method"] == rc.METHOD_WORST) & (pi["outcome"] != "refused")]
    usable_own = usable[usable["bed"] == rc.BED_OWN]["n_usable_variables"]
    mixed_first_fault = pi[
        (pi["bed"] == rc.BED_OWN) & (pi["group"] == rc.GROUP_MIXED)
        & (pi["method"] == rc.METHOD_WORST) & (pi["outcome"] != "refused")
    ]["first_fault_window"]
    det = data["det_summary"]

    add: list[str] = []
    p = add.append
    p("# Conformal test martingale on the 3W window caches")
    p("")
    p("## Summary")
    p("")
    p(
        f"The study runs a conformal test martingale alarm over the detector study's two "
        f"3W v2.0.0 window caches (dataset manifest `{dataset_sha}`): "
        f"{own['n_instances']} own-history instances ({own['n_windows']} one-hour windows) and "
        f"{ali['n_instances']} onset-aligned instances ({ali['n_windows']} windows). "
        f"Per instance the first two windows fit a median and MAD per variable, the third is the "
        f"calibration window and the rest are the stream; the martingale fires when it reaches "
        f"1/delta. On the {int(n_normal['n_scored'])} normal own-history records the alarm fires "
        f"on {pct(c_norm[d0]['far'])} at delta {d0:g} against a bound of {pct(d0)}, and the "
        f"worst-baseline rule fires on {pct(w_norm['far'])}; when the same scores are shuffled "
        f"across the calibration and stream the alarm fraction is {pct2(p_norm[d0]['alarm_rate'])} "
        f"(mean over {seeds} seeds). On the aligned bed the alarm fires in the pre window on "
        f"{pct(c_ali[d0]['far'])} of {int(c_ali[d0]['n_scored'])} instances (floor "
        f"{pct(FLOOR)}) and first fires in a post window on {pct(c_ali[d0]['detect_post1'])}. "
        f"The bound holds when exchangeability is made true by construction and fails on the "
        f"records as recorded, so a one-hour calibration window is not exchangeable with the "
        f"hours that follow it on these wells."
    )
    p("")
    p("## Data")
    p("")
    p(
        f"Own-history bed: `data/3w_windows` (cache manifest sha256 "
        f"`{own['manifest_sha256'][:12]}`), {own['n_windows']} windows over "
        f"{own['n_instances']} instances in `windows.parquet` (the manifest counts "
        f"{own['manifest_n_instances']} instances; "
        f"{own['manifest_n_instances'] - own['n_instances']} "
        f"have no window). Onset-aligned bed: `data/3w_windows_aligned` (cache manifest sha256 "
        f"`{ali['manifest_sha256'][:12]}`), {ali['n_windows']} windows over {ali['n_instances']} "
        f"instances at offsets -5, -4, -3 (baseline), -2 (pre), 0 (post0) and 1 (post1) from the "
        f"fault onset. Both caches hold, per window and variable, 60 one-minute sub-group medians "
        f"of "
        f"the GOOD rows for the six common variables `P-ANULAR`, `P-JUS-CKGL`, `P-MON-CKP`, "
        f"`P-PDG`, `P-TPT`, `T-TPT`. How the caches are built is in "
        f"[the detector report](../3w_detectors/REPORT.md)."
    )
    p("")
    p("Groups on the own-history bed, by the window labels:")
    p("")
    p(
        f"- normal ({own['groups'].get(rc.GROUP_NORMAL, 0)} instances): every window labelled 0; "
        f"any alarm is a false alarm. Stream length median {int(normal_streams.median())} window, "
        f"maximum {int(normal_streams.max())}."
    )
    p(
        f"- mixed ({own['groups'].get(rc.GROUP_MIXED, 0)}): the three baseline windows labelled 0 "
        f"and at least one later window labelled 1. Stream length median "
        f"{int(mixed_streams.median())} windows, maximum {int(mixed_streams.max())}."
    )
    p(
        f"- positive baseline ({own['groups'].get(rc.GROUP_POSITIVE_BASELINE, 0)}): a baseline "
        f"window labelled 1, so the fit or the calibration already holds fault rows. The alarm "
        f"rate is reported and no false-alarm or detection rate."
    )
    p(
        f"- short ({own['groups'].get(rc.GROUP_SHORT, 0)}): fewer than {rc.DEFAULT_K + 1} windows, "
        f"a design refusal."
    )
    p("")
    p("The aligned bed is one group of 48 instances; its pre window carries the false-alarm rate "
      "and its post windows the detection rate.")
    p("")
    p("## Method")
    p("")
    p("Per instance and bed:")
    p("")
    p(
        "1. Windows are ordered by `window_index` (own history) or `onset_offset` (aligned). "
        f"Fewer than {rc.DEFAULT_K + 1} windows is a design refusal."
    )
    p(
        f"2. Fit: `mad_baseline` on the finite sub-group medians of the first {rc.N_FIT} windows, "
        "per variable (at most 120 values). A variable with under 30 values "
        "(`InsufficientQuality`) or a zero MAD is unusable; an instance with no usable variable is "
        "a refusal."
    )
    p(
        "3. Nonconformity of a sub-group: the largest |x - centre| / scale over the usable "
        "variables with a finite value at that minute. A minute with no finite variable is dropped."
    )
    p(
        f"4. Calibration scores: the third window's sub-groups; fewer than {rc.MIN_CALIBRATION} "
        "finite scores is a refusal. Stream scores: windows four onward, in order."
    )
    p(
        "5. `conformal`: `conformal_p_values(calibration, stream, online=True)`, "
        f"`mixture_martingale` over epsilons {run['parameters']['epsilons']}, "
        "`martingale_alarm` at each delta. The alarm minute maps to its window."
    )
    p(
        "6. `worst_baseline`: the window score is its largest sub-group nonconformity; the "
        "threshold is `worst_baseline_threshold` over the three baseline windows (in sample, as "
        "in the detector study); the alarm is the first stream window where `fires`. Floor "
        f"`far_floor({rc.DEFAULT_K})` = {FLOOR:.3f}."
    )
    p(
        "7. `clock`: the same martingale with nonconformity = position in the record "
        "(calibration 0 to 59, stream counting on). Reads no sensor value."
    )
    p(
        f"8. `permutation`: calibration and stream scores concatenated, shuffled with "
        f"`numpy.random.default_rng(seed)` for seeds 0 to {seeds - 1}, split back at the "
        "calibration length, then rule 5. Exchangeability then holds by construction."
    )
    p("")
    p(
        "Outcome per instance and rule: the first alarm only. An alarm in a window labelled 1 is "
        "a detection (aligned: post0 or post1), in a window labelled 0 a false alarm (aligned: "
        "pre), and no crossing is `none`. Rates are over the scored instances of the group. On the "
        "aligned bed an instance whose alarm falls in the pre window is a false alarm and is not "
        "also a detection; the column `crossed by end of post1` counts an alarm in any of the "
        "three stream windows."
    )
    p("")
    p("Commands:")
    p("")
    p("```")
    p("uv run python examples/studies/3w_conformal/run_conformal.py")
    p("uv run python examples/studies/3w_conformal/make_report.py --update-benchmarks")
    p("```")
    p("")
    p(
        f"The run takes {run['wall_seconds']} s. Two runs write byte-identical CSVs; the study "
        "checked this with a second run into a scratch directory and `cmp` on every file."
    )
    p("")
    p("## Results")
    p("")
    p("### Own-history bed")
    p("")
    p(
        "`alarm rate` is the share of scored instances with any alarm. FAR is the share whose "
        "alarm falls in a window labelled 0, detect the share whose alarm falls in a window "
        "labelled 1, delay the alarm window minus the first fault window. `median max log10 M` "
        "is the median over instances of the largest log10 martingale value on the stream; the "
        f"alarm lines are {math.log10(1 / d0):.2f} at delta {d0:g}"
        + (
            f" and {math.log10(1 / deltas[1]):.2f} at delta {deltas[1]:g}."
            if len(deltas) > 1
            else "."
        )
    )
    p("")
    add += own_history_table(summary, deltas)
    p("")
    p(f"FAR of the normal group by stream length, at delta {d0:g}:")
    p("")
    add += length_table(pi, d0)
    p("")
    p("### Onset-aligned bed")
    p("")
    p(
        "The detector-study rows come from its frozen `results/aligned_summary.csv` (variant "
        "`all`) and are not recomputed. Their `detect post0` and `detect post1` are the share of "
        "post windows above the instance's threshold, whatever the pre window did; this study's "
        "rows count the first alarm only, so a pre alarm removes the instance from `detect`. "
        f"`over floor` is FAR on pre minus {FLOOR:.3f}."
    )
    p("")
    add += aligned_table(data, deltas)
    p("")
    p("### Permutation check")
    p("")
    p(
        "Each seed shuffles every scored instance's calibration and stream scores together and "
        "reruns the conformal rule; the alarm fraction is over the group's instances."
    )
    p("")
    add += permutation_table(data)
    p("")
    p("### Refusals")
    p("")
    add += refusal_table(data)
    p("")
    p(
        f"Usable variables per scored instance (median {int(usable_own.median())} of 6 on the "
        "own-history bed). A variable becomes unusable when its MAD over the two fit windows is "
        "zero, which happens when the rounded sub-group medians repeat one value."
    )
    p("")
    add += usable_table(pi)
    p("")
    p("## Discussion")
    p("")
    p(
        f"On the {int(n_normal['n_scored'])} normal records the martingale reaches 1/delta on "
        f"{pct(c_norm[d0]['far'])} at delta {d0:g}"
        + (
            f" and {pct(c_norm[deltas[1]]['far'])} at delta {deltas[1]:g}"
            if len(deltas) > 1
            else ""
        )
        + f". The bound says at most {pct(d0)}"
        + (f" and {pct(deltas[1])}" if len(deltas) > 1 else "")
        + f". The permutation check on the same scores gives {pct2(p_norm[d0]['alarm_rate'])}"
        + (
            f" and {pct2(p_norm[deltas[1]]['alarm_rate'])}"
            if len(deltas) > 1
            else ""
        )
        + f", under the bound, and no (bed, group, delta, seed) cell exceeds {pct2(perm_max)}. "
        f"The code keeps the bound when the scores are exchangeable; the records are not. The "
        f"clock control, which reads position alone, fires on {pct(k_norm['far'])} of the same "
        f"records with a median peak of 10^{fmt(k_norm['median_max_log10_martingale'], 1)}: "
        f"every new minute is the largest position so far, so its p-value is the smallest "
        f"available and the martingale grows at every step. A record whose level drifts between "
        f"hours does the same to the sensor scores, at a slower rate "
        f"(median peak 10^{fmt(c_norm[d0]['median_max_log10_martingale'], 1)})."
    )
    p("")
    p(
        f"The worst-baseline rule fires on {pct(w_norm['far'])} of the normal records, "
        f"{pct(c_norm[d0]['far'])} for the martingale at delta {d0:g}. Most normal records are one "
        f"stream window long (median {int(normal_streams.median())}), so both rates are mostly "
        f"the chance that the fourth hour exceeds the first three; the length table shows how "
        f"the rates move with longer streams. The worst-baseline floor {pct(FLOOR)} is a "
        f"per-window statement and the martingale bound {pct(d0)} a per-record one; neither rule "
        f"reaches its own reference on this bed."
    )
    p("")
    p(
        f"On the {int(c_mix['n_scored'])} mixed records the martingale fires before the fault on "
        f"{pct(c_mix['far'])} and first fires inside a fault window on {pct(c_mix['detect'])} "
        f"(median delay {fmt(c_mix['median_delay_windows'], 0)} windows); the worst-baseline rule "
        f"gives {pct(w_mix['far'])} and {pct(w_mix['detect'])}, the clock {pct(k_mix['far'])} and "
        f"{pct(k_mix['detect'])}. With a median of {int(mixed_streams.median())} stream windows "
        f"and the first fault at a median of window {int(mixed_first_fault.median())} in the "
        f"record, an alarm that fires "
        f"eventually fires before the fault more often than in it."
    )
    p("")
    p(
        f"On the aligned bed the martingale fires in the pre window on {pct(c_ali[d0]['far'])} of "
        f"48 instances at delta {d0:g} ({signed(float(c_ali[d0]['far']) - FLOOR)} over the "
        f"{pct(FLOOR)} floor, {pct(w_ali['far'])} for the worst-baseline rule on the same scores) "
        f"and first fires in post0 on {pct(c_ali[d0]['detect_post0'])}, by the end of post1 on "
        f"{pct(c_ali[d0]['detect_post1'])}; by the end of post1 it has crossed at some point on "
        f"{pct(c_ali[d0]['alarm_rate'])}. The permutation check gives "
        f"{pct2(p_ali[d0]['alarm_rate'])}. "
        + (
            f"The detector study's MAD screen fires on "
            f"{pct(det.loc['mad', 'false_alarm_rate_pre'])} of the pre windows and the SPC rules "
            f"on {pct(det.loc['spc', 'false_alarm_rate_pre'])} "
            f"under the worst-baseline rule. "
            if det is not None
            else ""
        )
        + f"The clock fires in the pre window on every instance ({pct(k_ali['far'])})."
    )
    p("")
    p(
        "What the numbers support: `conformal_p_values`, `mixture_martingale` and "
        "`martingale_alarm` hold their false-alarm bound when the calibration and stream scores "
        "are exchangeable, and a 3W own-history record scored this way is not exchangeable at the "
        "one-hour scale, so the bound does not transfer to it. What they do not support: a "
        "detection claim for the steady fault state (the aligned positives are the first two "
        "hours of a transient), a claim about the short instances, or a comparison of the bound "
        "with the floor as if they measured the same thing."
    )
    p("")
    p("## Limits and further work")
    p("")
    p(
        "- Exchangeability is the assumption and the records break it. Detrending the scores "
        "(a residual against a slow fit, or a difference between consecutive minutes) before "
        "the conformal step, or a longer calibration window, are the untried settings that "
        "could restore it. Each is a different study."
    )
    p(
        "- The nonconformity is the maximum robust z over the usable variables; with a median of "
        f"{int(usable_own.median())} usable variables the score often reads one or two tags."
    )
    p(
        "- The fit and the calibration are one hour each, and `mad_baseline` on 120 sub-group "
        "medians is a small fit. A rounded tag gives a zero MAD and drops out; on this bed that "
        "is the only refusal of the sensor path."
    )
    p(
        "- The aligned bed is 48 instances on 21 wells and the positives are transient onsets; "
        "no group holdout applies because nothing is fitted across instances."
    )
    p("")
    p("## Files")
    p("")
    add += table(
        ["path", "what"],
        ["---", "---"],
        [
            ["`run_conformal.py`", "scores both caches, writes `results/`"],
            ["`make_report.py`", "this report and `results/benchmarks_section.md`"],
            [
                "`results/per_instance.csv`",
                "one row per (bed, instance, rule, delta): alarm window, sub-group, label, "
                "outcome, delay, peak log10 martingale, refusal",
            ],
            ["`results/summary.csv`", "one row per (bed, group, rule, delta)"],
            ["`results/permutation.csv`", "alarm fraction per (bed, group, delta, seed)"],
            ["`results/benchmarks_section.md`", "the BENCHMARKS.md REAL section"],
            [
                "`results/run.json`",
                "code head, tsdive version, cache and dataset manifest hashes, parameters, group "
                "and refusal counts, wall seconds",
            ],
        ],
    )
    p("")
    det_cite = f" Detector-study rows: {data['det_code']}." if data["det_code"] else ""
    p(f"This study: {this_code}.{det_cite}")
    p("")
    return "\n".join(add)


# ------------------------------------------------------------- benchmarks


def render_benchmarks_section(data: dict) -> str:
    run = data["run"]
    summary = data["summary"]
    deltas = [float(d) for d in run["parameters"]["deltas"]]
    seeds = int(run["parameters"]["permutation_seeds"])
    own, ali = run["beds"][rc.BED_OWN], run["beds"][rc.BED_ALIGNED]
    dataset = f"3W v2.0.0 real `{(run.get('dataset_manifest_sha256') or 'unknown')[:12]}`"
    code = code_cite(run["code"])

    def row(bed, group, method, delta=None) -> pd.Series:
        return summary_row(summary, bed, group, method, delta)

    n_norm = row(rc.BED_OWN, rc.GROUP_NORMAL, rc.METHOD_WORST)
    n_mix = row(rc.BED_OWN, rc.GROUP_MIXED, rc.METHOD_WORST)
    own_split = (
        f"own history: 2 fit windows, 1 calibration window, label-blind; no group holdout; "
        f"{int(n_norm['n_scored'])} of {int(n_norm['n_instances'])} normal and "
        f"{int(n_mix['n_scored'])} of {int(n_mix['n_instances'])} mixed instances scored; "
        f"cache manifest `{own['manifest_sha256'][:12]}`"
    )
    ali_split = (
        f"own history: 2 fit windows, 1 calibration window before onset, label-blind; no group "
        f"holdout; {ali['n_instances']} evaluable instances; cache manifest "
        f"`{ali['manifest_sha256'][:12]}`"
    )
    rows: list[tuple[str, str, str, str, str]] = []
    for d in deltas:
        c = row(rc.BED_OWN, rc.GROUP_NORMAL, rc.METHOD_CONFORMAL, d)
        pm = row(rc.BED_OWN, rc.GROUP_NORMAL, rc.METHOD_PERMUTATION, d)
        rows.append((
            "none",
            f"conformal martingale alarm, delta {d:g}: FAR over the whole normal record",
            f"FAR {pct(c['far'])} (bound {pct(d)}); permutation check {pct2(pm['alarm_rate'])} "
            f"over {seeds} seeds",
            "own-history", own_split,
        ))
    w = row(rc.BED_OWN, rc.GROUP_NORMAL, rc.METHOD_WORST)
    rows.append((
        "none", "worst-baseline threshold on the same scores: FAR over the whole normal record",
        f"FAR {pct(w['far'])} ({signed(float(w['far']) - FLOOR)} over the {pct(FLOOR)} floor)",
        "own-history", own_split,
    ))
    k = row(rc.BED_OWN, rc.GROUP_NORMAL, rc.METHOD_CLOCK, deltas[0])
    rows.append((
        "none",
        f"**clock control** martingale (position in the record, no sensor read), delta "
        f"{deltas[0]:g}: FAR over the whole normal record",
        f"FAR {pct(k['far'])}", "own-history", own_split,
    ))
    c = row(rc.BED_OWN, rc.GROUP_MIXED, rc.METHOD_CONFORMAL, deltas[0])
    w = row(rc.BED_OWN, rc.GROUP_MIXED, rc.METHOD_WORST)
    rows.append((
        "none", f"conformal martingale alarm, delta {deltas[0]:g}, mixed records",
        f"FAR {pct(c['far'])}, detect {pct(c['detect'])}, median delay "
        f"{fmt(c['median_delay_windows'], 0)} windows; worst-baseline FAR {pct(w['far'])}, "
        f"detect {pct(w['detect'])}",
        "own-history", own_split,
    ))
    for d in deltas:
        c = row(rc.BED_ALIGNED, rc.GROUP_ALIGNED, rc.METHOD_CONFORMAL, d)
        pm = row(rc.BED_ALIGNED, rc.GROUP_ALIGNED, rc.METHOD_PERMUTATION, d)
        rows.append((
            "none", f"conformal martingale alarm, delta {d:g}",
            f"FAR on pre {pct(c['far'])} ({signed(float(c['far']) - FLOOR)} over the "
            f"{pct(FLOOR)} floor), first alarm in post0 {pct(c['detect_post0'])}, by end of post1 "
            f"{pct(c['detect_post1'])}, crossed by end of post1 {pct(c['alarm_rate'])}; "
            f"permutation check {pct2(pm['alarm_rate'])} over {seeds} seeds",
            "onset-aligned", ali_split,
        ))
    w = row(rc.BED_ALIGNED, rc.GROUP_ALIGNED, rc.METHOD_WORST)
    rows.append((
        "none", "worst-baseline threshold on the same scores",
        f"FAR on pre {pct(w['far'])} ({signed(float(w['far']) - FLOOR)} over the {pct(FLOOR)} "
        f"floor), first alarm in post0 {pct(w['detect_post0'])}, by end of post1 "
        f"{pct(w['detect_post1'])}",
        "onset-aligned", ali_split,
    ))
    k = row(rc.BED_ALIGNED, rc.GROUP_ALIGNED, rc.METHOD_CLOCK, deltas[0])
    rows.append((
        "none",
        f"**clock control** martingale (position in the record, no sensor read), delta "
        f"{deltas[0]:g}",
        f"FAR on pre {pct(k['far'])}, first alarm in post0 {pct(k['detect_post0'])}",
        "onset-aligned", ali_split,
    ))
    lines = [
        SECTION_HEADING,
        "",
        "A conformal test martingale (`conformal_p_values`, `mixture_martingale`, "
        "`martingale_alarm` in `tsdive.eval`) over the largest robust z per minute, calibrated "
        "on each instance's third window and run over the rest of its record. Under "
        "exchangeability of the calibration and stream scores the alarm fires with probability "
        "at most delta over the whole record (Ville's inequality); the permutation check makes "
        "exchangeability true by shuffling the scores and measures the same alarm. Method, "
        "groups and refusals are in "
        "[examples/studies/3w_conformal/REPORT.md](examples/studies/3w_conformal/REPORT.md).",
        "",
        "This section is not regenerated by `make bench`; it is produced by "
        "`examples/studies/3w_conformal/run_conformal.py` and `make_report.py`, and carried "
        "through the generator unchanged.",
        "",
        "| stage | metric | value | dataset (manifest sha256) | code | design | split |",
        "|---|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {stage} | {metric} | {value} | {dataset} | {code} | {design} | {split} |"
        for stage, metric, value, design, split in rows
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
    parser.add_argument("--report", default=str(HERE / "REPORT.md"))
    parser.add_argument("--update-benchmarks", action="store_true")
    parser.add_argument("--benchmarks", default=str(BENCH_MD))
    args = parser.parse_args(argv)

    results = Path(args.results)
    data = load(results, Path(args.detector_results))
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
    print(f"wrote {args.report} and {results / 'benchmarks_section.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

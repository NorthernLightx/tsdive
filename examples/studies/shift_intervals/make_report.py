"""Report, figure and BENCHMARKS.md section for the shift interval study.

Reads ``results/`` as ``run_shift.py`` wrote it and renders ``REPORT.md``,
``out/01_coverage_at_zero_shift.png`` and ``results/benchmarks_section.md``.
With ``--update-benchmarks`` the section replaces the
``## REAL: shift intervals (3W, SKAB, TEP, Turbine Upgrade, synthetic)`` section of the
repository's ``BENCHMARKS.md``, or is appended when the file has none. Every
number in the prose is read from the frames the tables are built from.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))

import run_shift as rs  # noqa: E402
import shift as sh  # noqa: E402

BENCH_MD = ROOT / "BENCHMARKS.md"
SECTION_HEADING = "## REAL: shift intervals (3W, SKAB, TEP, Turbine Upgrade, synthetic)"
FIGURE_NAME = "01_coverage_at_zero_shift.png"

METHOD_LABELS = {
    sh.NAIVE: "naive",
    sh.HAC: "HAC",
    sh.EWC: "EWC",
    sh.BOOTSTRAP: "block bootstrap",
}
COLORS = {sh.NAIVE: "#d95f02", sh.HAC: "#1f77b4", sh.EWC: "#1b9e77", sh.BOOTSTRAP: "#7570b3"}
MARKERS = {sh.NAIVE: "o", sh.HAC: "s", sh.EWC: "D", sh.BOOTSTRAP: "^"}
DESIGN_LABELS = {
    rs.DESIGN_3W_PLACEBO: "3W placebo",
    rs.DESIGN_SKAB_FREE: "SKAB anomaly-free placebo",
    rs.DESIGN_SKAB_PRE: "SKAB pre-onset placebo",
    rs.DESIGN_3W_ONSET: "3W onset-aligned",
    rs.DESIGN_SKAB_ONSET: "SKAB labelled onset",
}
PAIR_LABELS = {rs.PAIR_VG: "VG pair", rs.PAIR_PITCH: "pitch pair"}
USABLE_MAX = 0.10
WIDTH_RATIO_MAX = 0.8
R3_MIN = 0.8
ANSWERED_MIN = 0.5
N_HOLD = 480
PHI_HOLD = 0.9
SHORT_SETS = ("base", "base+R1", "base+R4")
REGISTERED_RAW = ("base", "base+R1")
REGISTERED_ADJUSTED = ("base", "base+R1", "base+R3", "base+R1+R3")
POST_HOC_OF = {"base": "base+R4", "base+R1": "base+R1+R4"}


# ---------------------------------------------------------------- formatting


def pct(value) -> str:
    return "none answered" if value is None or pd.isna(value) else f"{float(value) * 100:.1f}%"


def num(value, digits: int = 3) -> str:
    return "none answered" if value is None or pd.isna(value) else f"{float(value):.{digits}f}"


def phi_label(phi: float) -> str:
    return f"phi {phi:g}"


def set_label(refusal_set: str) -> str:
    return f"{refusal_set} (post hoc)" if refusal_set in sh.POST_HOC_SETS else refusal_set


def registered(refusal_set: str, adjustment: str) -> bool:
    allowed = REGISTERED_RAW if adjustment == sh.RAW else REGISTERED_ADJUSTED
    return refusal_set in allowed


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
        if stripped.startswith(("- ", "1. ", "2. ", "3. ", "4. ", "5. ", "6. ", "7. ")):
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
    return {
        "synthetic": pd.read_csv(results / "synthetic.csv"),
        "summary": pd.read_csv(results / rs.SUMMARY_CSV, keep_default_na=False, na_values=[""]),
        "placebo": rs.read_rows(results / rs.PLACEBO_CSV),
        "known": rs.read_rows(results / rs.KNOWN_CSV),
        "turbine": rs.read_rows(results / rs.TURBINE_CSV),
        "tep": pd.read_csv(results / rs.TEP_CSV),
        "run": json.loads((results / "run.json").read_text("utf-8")),
    }


def code_label(section: dict) -> str:
    return f"tsdive {section['code']['tsdive_version']} @ `{section['code']['git_head'][:8]}`"


# ---------------------------------------------------------------- synthetic


def cells(syn: pd.DataFrame, **match) -> pd.DataFrame:
    mask = pd.Series(True, index=syn.index)
    for key, value in match.items():
        mask &= syn[key] == value
    return syn[mask]


def pooled(frame: pd.DataFrame) -> tuple[float | None, float | None, int]:
    """Coverage over the answered replicates of several cells, its MCSE and count."""
    n = int(frame["n_answered"].sum())
    if n == 0:
        return None, None, 0
    covered = round(float((frame["coverage"] * frame["n_answered"]).sum()))
    c = covered / n
    return c, math.sqrt(c * (1 - c) / n), n


def cov_text(c, mcse) -> str:
    return "none answered" if c is None else f"{c:.3f} ({mcse:.3f})"


def zero_shift(syn: pd.DataFrame, **match) -> pd.DataFrame:
    return cells(syn, grid=rs.GRID_LEVEL, delta=0.0, **match)


def coverage_rows(syn: pd.DataFrame, n: int) -> list[list[str]]:
    rows = []
    for method in sh.LEVEL_METHODS:
        for adjustment in sh.ADJUSTMENTS:
            row = [METHOD_LABELS[method], adjustment]
            for phi in rs.PHIS:
                chosen = zero_shift(
                    syn, drift=0.0, n=n, phi=phi, method=method, adjustment=adjustment,
                    refusal_set="base",
                )
                row.append(cov_text(*pooled(chosen)[:2]))
            rows.append(row)
    return rows


def hold_verdicts(syn: pd.DataFrame) -> pd.DataFrame:
    """Per (quantity, method, adjustment, set): the cells criterion 1 reads."""
    level = syn[
        (syn["grid"] == rs.GRID_LEVEL)
        & (syn["delta"] == 0)
        & (syn["drift"] == 0)
        & (syn["phi"] <= PHI_HOLD)
        & (syn["n"] == N_HOLD)
    ]
    spread = syn[
        (syn["grid"] == rs.GRID_SPREAD)
        & (syn["sd_ratio"] == 1.0)
        & (syn["phi"] <= PHI_HOLD)
        & (syn["n"] == N_HOLD)
    ]
    frame = pd.concat([level, spread], ignore_index=True)
    frame = frame.assign(passes=frame["coverage"] >= 0.95 - 2 * frame["coverage_mcse"])
    out = frame.groupby(["quantity", "method", "adjustment", "refusal_set"], sort=True).agg(
        n_cells=("passes", "size"),
        n_pass=("passes", "sum"),
        worst=("coverage", "min"),
        max_refused=("rate_refused", "max"),
    )
    out["holds"] = out["n_pass"] == out["n_cells"]
    out = out.reset_index()
    out["registered"] = [
        registered(s, a) for s, a in zip(out["refusal_set"], out["adjustment"], strict=True)
    ]
    return out


def plot_coverage(syn: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.9), sharey=True)
    for ax, n in zip(axes, rs.NS, strict=True):
        ax.axhline(0.95, color="#555555", linewidth=1.0, linestyle=":")
        for method in sh.LEVEL_METHODS:
            for adjustment, style in ((sh.RAW, "-"), (sh.ADJUSTED, "--")):
                values = [
                    pooled(
                        zero_shift(
                            syn, drift=0.0, n=n, phi=phi, method=method,
                            adjustment=adjustment, refusal_set="base",
                        )
                    )[0]
                    for phi in rs.PHIS
                ]
                ax.plot(
                    range(len(rs.PHIS)),
                    values,
                    linestyle=style,
                    linewidth=2.0,
                    marker=MARKERS[method],
                    markersize=6,
                    color=COLORS[method],
                    label=f"{METHOD_LABELS[method]}, {adjustment}",
                )
        ax.set_xticks(range(len(rs.PHIS)), [f"{phi:g}" for phi in rs.PHIS])
        ax.set_xlabel("AR(1) coefficient phi")
        ax.set_title(f"{n} samples per period", fontsize=10)
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", color="#e5e5e5", linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("coverage of the true shift")
    axes[1].legend(fontsize=7, frameon=False, loc="lower left", ncol=2)
    fig.suptitle(
        "SYNTHETIC: 95% interval coverage at a zero level shift, no drift, "
        "pooled over rho (dotted line 0.95)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------- real beds


def srow(summary: pd.DataFrame, design, quantity, method, adjustment, refusal_set) -> pd.Series:
    chosen = summary[
        (summary["design"] == design)
        & (summary["quantity"] == quantity)
        & (summary["method"] == method)
        & (summary["adjustment"] == adjustment)
        & (summary["refusal_set"] == refusal_set)
    ]
    assert len(chosen) == 1, (design, quantity, method, adjustment, refusal_set)
    return chosen.iloc[0]


def combos(quantity: str, sets: tuple[str, ...] | None = None) -> list[tuple[str, str, str]]:
    """(method, adjustment, set) in table order; ``sets`` limits the refusal sets."""
    methods = sh.LEVEL_METHODS if quantity == sh.LEVEL else sh.SPREAD_METHODS
    out = []
    for method in methods:
        for adjustment in sh.ADJUSTMENTS:
            allowed = sh.RAW_REFUSAL_SETS if adjustment == sh.RAW else tuple(sh.REFUSAL_SETS)
            for refusal_set in allowed:
                if sets is None or refusal_set in sets:
                    out.append((method, adjustment, refusal_set))
    return out


def placebo_3w_rows(summary: pd.DataFrame, quantity: str) -> list[list[str]]:
    rows = []
    for method, adjustment, refusal_set in combos(quantity):
        r = srow(summary, rs.DESIGN_3W_PLACEBO, quantity, method, adjustment, refusal_set)
        rows.append(
            [
                METHOD_LABELS[method],
                adjustment,
                set_label(refusal_set),
                f"{int(r['n_answered'])} ({pct(r['answered_share'])})",
                pct(r["clear_rate_pooled"]),
                pct(r["clear_rate_group_avg"]),
                f"{pct(r['record_clear_rate'])} of {int(r['n_records_answered'])}",
            ]
        )
    return rows


def per_well_rows(placebo: pd.DataFrame) -> list[list[str]]:
    chosen = placebo[
        (placebo["design"] == rs.DESIGN_3W_PLACEBO)
        & (placebo["quantity"] == sh.LEVEL)
        & (placebo["method"] == sh.HAC)
        & (placebo["adjustment"] == sh.RAW)
    ]
    rows = []
    for well, group in chosen.groupby("group", sort=True):
        answered = group[group["refused"] == 0]
        trend_free = answered[answered["before_trend"] == 0]
        rows.append(
            [
                well,
                str(group["record"].nunique()),
                str(len(answered)),
                pct(answered["clears"].eq(1).mean()) if len(answered) else "none answered",
                str(len(trend_free)),
                pct(trend_free["clears"].eq(1).mean()) if len(trend_free) else "none answered",
            ]
        )
    return rows


def skab_placebo_rows(summary: pd.DataFrame) -> list[list[str]]:
    rows = []
    for method, adjustment, refusal_set in combos(sh.LEVEL, SHORT_SETS):
        free = srow(summary, rs.DESIGN_SKAB_FREE, sh.LEVEL, method, adjustment, refusal_set)
        pre = srow(summary, rs.DESIGN_SKAB_PRE, sh.LEVEL, method, adjustment, refusal_set)
        rows.append(
            [
                METHOD_LABELS[method],
                adjustment,
                set_label(refusal_set),
                f"{pct(free['clear_rate_pooled'])} of {int(free['n_answered'])}",
                f"{pct(pre['clear_rate_pooled'])} of {int(pre['n_answered'])}",
                pct(pre["clear_rate_group_avg"]),
                pct(pre["record_clear_rate"]),
            ]
        )
    return rows


def detection_rows(summary: pd.DataFrame, quantity: str) -> list[list[str]]:
    rows = []
    for method, adjustment, refusal_set in combos(quantity, SHORT_SETS):
        cells_ = []
        for design in (
            rs.DESIGN_3W_ONSET,
            rs.DESIGN_3W_PLACEBO,
            rs.DESIGN_SKAB_ONSET,
            rs.DESIGN_SKAB_PRE,
        ):
            r = srow(summary, design, quantity, method, adjustment, refusal_set)
            cells_.append(f"{pct(r['clear_rate_pooled'])} of {int(r['n_answered'])}")
        rows.append([METHOD_LABELS[method], adjustment, set_label(refusal_set), *cells_])
    return rows


def adjusted_rows(summary: pd.DataFrame) -> list[list[str]]:
    rows = []
    for design in (
        rs.DESIGN_3W_PLACEBO,
        rs.DESIGN_SKAB_PRE,
        rs.DESIGN_3W_ONSET,
        rs.DESIGN_TURBINE_PLACEBO,
    ):
        for method in sh.LEVEL_METHODS:
            r = srow(summary, design, sh.LEVEL, method, sh.ADJUSTED, "base")
            raw = srow(summary, design, sh.LEVEL, method, sh.RAW, "base")
            rows.append(
                [
                    DESIGN_LABELS.get(design, "turbine placebo"),
                    METHOD_LABELS[method],
                    num(r["median_variance_reduction"]),
                    num(r["median_variance_ratio_after"]),
                    num(r["median_width_ratio"]),
                    str(int(r["n_width_pairs"])),
                    pct(r["clear_rate_pooled"]),
                    pct(raw["clear_rate_pooled"]),
                    pct(r["rate_covariate_outside"]),
                    pct(r["rate_covariate_shifted"]),
                ]
            )
    return rows


def refusal_rows(summary: pd.DataFrame) -> list[list[str]]:
    rows = []
    for design in DESIGN_LABELS:
        for adjustment in sh.ADJUSTMENTS:
            r = srow(summary, design, sh.LEVEL, sh.HAC, adjustment, "base")
            rows.append(
                [
                    DESIGN_LABELS[design],
                    adjustment,
                    str(int(r["n_rows"])),
                    r["hard_refusals"] if isinstance(r["hard_refusals"], str) else "none",
                    pct(r["rate_before_trend"]),
                    pct(r["rate_too_persistent"]),
                    num(r["median_lag1"]),
                    pct(r["share_bandwidth_capped"]),
                ]
            )
    return rows


# ---------------------------------------------------------------- turbine


def turbine_pair_rows(dataset: dict) -> list[list[str]]:
    rows = []
    for name, counts in dataset["pairs"].items():
        steps = counts["steps_minutes"]
        gaps = steps["20_to_60"] + steps["70_to_1440"] + steps["over_1440"]
        rows.append(
            [
                PAIR_LABELS[name],
                f"{counts['n_rows']:,}",
                f"{counts['first'][:10]} to {counts['last'][:10]}",
                f"{counts['n_upgraded_rows']:,} from {counts['upgrade_starts'][:16]}",
                f"{steps['10']:,}",
                f"{gaps:,} ({steps['over_1440']} over a day, longest "
                f"{steps['longest'] / 1440:.1f} days)",
            ]
        )
    return rows


def turbine_week_rows(summary: pd.DataFrame) -> list[list[str]]:
    rows = []
    for method, adjustment, refusal_set in combos(sh.LEVEL, SHORT_SETS):
        placebo = srow(
            summary, rs.DESIGN_TURBINE_PLACEBO, sh.LEVEL, method, adjustment, refusal_set
        )
        row = [
            METHOD_LABELS[method],
            adjustment,
            set_label(refusal_set),
            f"{pct(placebo['clear_rate_pooled'])} of {int(placebo['n_answered'])}",
        ]
        for r in rs.TURBINE_INJECT_R:
            inj = srow(
                summary,
                rs.DESIGN_TURBINE_INJECTED.format(r=r),
                sh.LEVEL,
                method,
                adjustment,
                refusal_set,
            )
            row.append(
                f"{cov_text(inj['coverage'], inj['coverage_mcse'])}, "
                f"clears {pct(inj['clear_rate_pooled'])}"
            )
        rows.append(row)
    return rows


def turbine_level(turbine: pd.DataFrame, design: str) -> pd.DataFrame:
    return turbine[(turbine["design"] == design) & (turbine["quantity"] == sh.LEVEL)]


def interval_text(row: pd.Series) -> str:
    if row["refused"] == 1:
        return f"refused: `{row['reason']}`"
    return f"{row['estimate']:+.3f} [{row['lo']:+.3f}, {row['hi']:+.3f}]"


def pitch_rows(turbine: pd.DataFrame) -> list[list[str]]:
    pitch = turbine_level(turbine, rs.DESIGN_TURBINE_PITCH)
    rows = []
    for record, group in pitch.groupby("record", sort=True):
        raw = group[group["adjustment"] == sh.RAW].set_index("method")
        adjusted = group[group["adjustment"] == sh.ADJUSTED].iloc[0]
        truth = float(raw["truth"].iloc[0])
        row = [record.rsplit("_r", 1)[1], f"{truth:+.4f}"]
        for method in (sh.HAC, sh.EWC):
            r = raw.loc[method]
            covers = r["lo"] <= truth <= r["hi"]
            row.append(f"{interval_text(r)}, {'covers' if covers else 'misses'}")
        row.append(interval_text(adjusted))
        rows.append(row)
    return rows


def vg_rows(turbine: pd.DataFrame, summary: pd.DataFrame) -> list[list[str]]:
    vg = turbine_level(turbine, rs.DESIGN_TURBINE_VG)
    rows = []
    for method in sh.LEVEL_METHODS:
        for adjustment in sh.ADJUSTMENTS:
            r = vg[(vg["method"] == method) & (vg["adjustment"] == adjustment)].iloc[0]
            placebo = srow(
                summary, rs.DESIGN_TURBINE_PLACEBO, sh.LEVEL, method, adjustment, "base"
            )
            rows.append(
                [
                    METHOD_LABELS[method],
                    adjustment,
                    interval_text(r),
                    "yes" if r["clears"] == 1 else "no",
                    pct(placebo["clear_rate_pooled"]),
                ]
            )
    return rows


# ---------------------------------------------------------------- TEP


def tep_rows(tep: pd.DataFrame, fault: int, quantity, method, refusal_set) -> pd.DataFrame:
    return tep[
        (tep["fault"] == fault)
        & (tep["quantity"] == quantity)
        & (tep["method"] == method)
        & (tep["refusal_set"] == refusal_set)
    ]


def tep_holds(tep: pd.DataFrame) -> pd.DataFrame:
    """Criterion 5 per (quantity, method, set): fault-free coverage in each group."""
    free = tep[tep["fault"] == 0]
    free = free.assign(passes=free["coverage"] >= 0.95 - 2 * free["coverage_mcse"])
    out = free.groupby(["quantity", "method", "refusal_set"], sort=True).agg(
        n_groups=("passes", "size"),
        n_pass=("passes", "sum"),
        worst=("coverage", "min"),
        n_answered=("n_answered", "sum"),
        n_rows=("n_rows", "sum"),
    )
    out["answered_share"] = out["n_answered"] / out["n_rows"]
    out["holds"] = out["n_pass"] == out["n_groups"]
    return out.reset_index()


def tep_free_rows(tep: pd.DataFrame) -> list[list[str]]:
    rows = []
    for quantity, methods in sh.QUANTITY_METHODS:
        for method in methods:
            for refusal_set in sh.RAW_REFUSAL_SETS:
                chosen = tep_rows(tep, 0, quantity, method, refusal_set).set_index(
                    "variable_group"
                )
                row = [quantity, METHOD_LABELS[method], set_label(refusal_set)]
                for group in rs.TEP_GROUPS:
                    r = chosen.loc[group]
                    row.append(
                        f"{cov_text(r['coverage'], r['coverage_mcse'])}, "
                        f"answered {pct(r['answered_share'])}"
                    )
                rows.append(row)
    return rows


def tep_fault_rows(tep: pd.DataFrame) -> list[list[str]]:
    rows = []
    faults = tep[tep["fault"] > 0]
    for method in sh.LEVEL_METHODS:
        chosen = faults[
            (faults["quantity"] == sh.LEVEL)
            & (faults["method"] == method)
            & (faults["refusal_set"] == "base")
        ]
        row = [METHOD_LABELS[method]]
        for group in rs.TEP_GROUPS:
            g = chosen[chosen["variable_group"] == group]
            row.append(
                f"{pct(g['detect_rate'].min())} to {pct(g['detect_rate'].max())}; coverage "
                f"{num(g['coverage'].min())} to {num(g['coverage'].max())}"
            )
        rows.append(row)
    return rows


# ---------------------------------------------------------------- verdict


def verdicts(data: dict) -> dict:
    syn, summary, tep = data["synthetic"], data["summary"], data["tep"]
    holds = hold_verdicts(syn)
    usable = []
    for quantity in (sh.LEVEL, sh.SPREAD):
        for method, adjustment, refusal_set in combos(quantity):
            r = srow(summary, rs.DESIGN_3W_PLACEBO, quantity, method, adjustment, refusal_set)
            ok = (
                not pd.isna(r["clear_rate_pooled"])
                and r["clear_rate_pooled"] <= USABLE_MAX
                and r["clear_rate_group_avg"] <= USABLE_MAX
            )
            usable.append((quantity, method, adjustment, refusal_set, ok, r))
    affected = syn[(syn["grid"] == rs.GRID_AFFECTED) & (syn["adjustment"] == sh.ADJUSTED)]
    affected = affected.drop_duplicates(["phi", "delta", "n"])
    r3_rate = float(affected["rate_covariate_shifted"].mean())
    adjust = []
    for quantity in (sh.LEVEL, sh.SPREAD):
        methods = sh.LEVEL_METHODS if quantity == sh.LEVEL else sh.SPREAD_METHODS
        for method in methods:
            for refusal_set in REGISTERED_ADJUSTED:
                adj = srow(
                    summary, rs.DESIGN_3W_PLACEBO, quantity, method, sh.ADJUSTED, refusal_set
                )
                raw = srow(
                    summary,
                    rs.DESIGN_3W_PLACEBO,
                    quantity,
                    method,
                    sh.RAW,
                    rs.raw_counterpart(refusal_set),
                )
                ratio_ok = not pd.isna(adj["median_width_ratio"]) and (
                    adj["median_width_ratio"] <= WIDTH_RATIO_MAX
                )
                claim_ok = (
                    not pd.isna(adj["clear_rate_pooled"])
                    and adj["clear_rate_pooled"] <= raw["clear_rate_pooled"]
                    and adj["clear_rate_group_avg"] <= raw["clear_rate_group_avg"]
                )
                adjust.append((quantity, method, refusal_set, ratio_ok, claim_ok, adj, raw))
    turbine = []
    for method, adjustment, refusal_set in combos(sh.LEVEL):
        placebo = srow(
            summary, rs.DESIGN_TURBINE_PLACEBO, sh.LEVEL, method, adjustment, refusal_set
        )
        injected = [
            srow(
                summary,
                rs.DESIGN_TURBINE_INJECTED.format(r=r),
                sh.LEVEL,
                method,
                adjustment,
                refusal_set,
            )
            for r in rs.TURBINE_INJECT_R
        ]
        claim_ok = not pd.isna(placebo["clear_rate_pooled"]) and (
            placebo["clear_rate_pooled"] <= USABLE_MAX
        )
        cover_ok = all(
            not pd.isna(i["coverage"]) and i["coverage"] >= 0.95 - 2 * i["coverage_mcse"]
            for i in injected
        )
        answered = min(
            float(placebo["answered_share"]), *(float(i["answered_share"]) for i in injected)
        )
        turbine.append(
            {
                "method": method,
                "adjustment": adjustment,
                "refusal_set": refusal_set,
                "passes": claim_ok and cover_ok,
                "placebo": placebo,
                "injected": injected,
                "answered": answered,
            }
        )
    tep_h = tep_holds(tep)
    confirm = []
    for _, row in tep_h.iterrows():
        if row["refusal_set"] not in POST_HOC_OF:
            continue
        after = tep_h[
            (tep_h["quantity"] == row["quantity"])
            & (tep_h["method"] == row["method"])
            & (tep_h["refusal_set"] == POST_HOC_OF[row["refusal_set"]])
        ].iloc[0]
        if not row["holds"] and after["holds"] and after["answered_share"] >= ANSWERED_MIN:
            confirm.append(
                ("TEP (criterion 5)", row["quantity"], row["method"], sh.RAW,
                 row["refusal_set"], after["refusal_set"], float(after["answered_share"]))
            )
    by_key = {(t["method"], t["adjustment"], t["refusal_set"]): t for t in turbine}
    for t in turbine:
        if t["refusal_set"] not in POST_HOC_OF:
            continue
        after = by_key.get((t["method"], t["adjustment"], POST_HOC_OF[t["refusal_set"]]))
        if after and not t["passes"] and after["passes"] and after["answered"] >= ANSWERED_MIN:
            confirm.append(
                ("turbine (criterion 6)", sh.LEVEL, t["method"], t["adjustment"],
                 t["refusal_set"], after["refusal_set"], after["answered"])
            )
    return {
        "holds": holds,
        "usable": usable,
        "r3_rate": r3_rate,
        "adjust": adjust,
        "turbine": turbine,
        "tep": tep_h,
        "confirm": confirm,
    }


# ---------------------------------------------------------------- report


def render_report(data: dict) -> str:
    syn, summary, placebo = data["synthetic"], data["summary"], data["placebo"]
    turbine, tep, run = data["turbine"], data["tep"], data["run"]
    beds = run["beds"]
    synthetic, pbed, kbed = beds[rs.BED_SYNTHETIC], beds[rs.BED_PLACEBO], beds[rs.BED_KNOWN]
    tbed, ebed = beds[rs.BED_TURBINE], beds[rs.BED_TEP]
    sources = pbed["dataset_manifest_sha256"]
    v = verdicts(data)
    holds = v["holds"]
    replicates = synthetic["replicates"]
    cache_3w, cache_on = pbed["3w_cache"], kbed["3w_cache"]
    designs = pbed["designs"] | kbed["designs"]

    def s3w(method, adjustment, refusal_set, quantity=sh.LEVEL, design=rs.DESIGN_3W_PLACEBO):
        return srow(summary, design, quantity, method, adjustment, refusal_set)

    hac_raw, hac_raw_r1 = s3w(sh.HAC, sh.RAW, "base"), s3w(sh.HAC, sh.RAW, "base+R1")
    ewc_raw, ewc_raw_r1 = s3w(sh.EWC, sh.RAW, "base"), s3w(sh.EWC, sh.RAW, "base+R1")
    hac_adj = s3w(sh.HAC, sh.ADJUSTED, "base")
    hac_onset = s3w(sh.HAC, sh.RAW, "base", design=rs.DESIGN_3W_ONSET)
    hac_skab_onset = s3w(sh.HAC, sh.RAW, "base", design=rs.DESIGN_SKAB_ONSET)
    hac_skab_pre = s3w(sh.HAC, sh.RAW, "base", design=rs.DESIGN_SKAB_PRE)
    hac_free = s3w(sh.HAC, sh.RAW, "base", design=rs.DESIGN_SKAB_FREE)
    lag_3w = hac_raw

    registered_usable = [r for r in v["usable"] if registered(r[3], r[2])]
    lowest = min(
        (r for r in registered_usable if not pd.isna(r[5]["clear_rate_pooled"])),
        key=lambda r: r[5]["clear_rate_pooled"],
    )
    post_hoc_usable = [r for r in v["usable"] if not registered(r[3], r[2])]
    lowest_ph = min(
        (r for r in post_hoc_usable if not pd.isna(r[5]["clear_rate_pooled"])),
        key=lambda r: r[5]["clear_rate_pooled"],
    )
    level_holds = holds[(holds["quantity"] == sh.LEVEL) & holds["registered"]]
    best = level_holds.sort_values(["n_pass", "worst"], ascending=False).iloc[0]
    r4_in_hold = holds[holds["refusal_set"].isin(sh.POST_HOC_SETS)]["max_refused"].max()
    verdict_of = {
        (h["quantity"], h["method"], h["adjustment"], h["refusal_set"]): h["holds"]
        for _, h in holds.iterrows()
    }
    moved = [
        key for key, held in verdict_of.items()
        if key[3] in POST_HOC_OF and verdict_of.get((*key[:3], POST_HOC_OF[key[3]])) != held
    ]
    r4_hold_moves = (
        "moves no verdict" if not moved else f"moves {len(moved)} verdicts"
    )
    any_hold = bool(holds.loc[holds["registered"], "holds"].any())
    any_usable = any(r[4] for r in registered_usable)
    any_adjust = any(r[3] and r[4] for r in v["adjust"]) and v["r3_rate"] >= R3_MIN
    tep_h = v["tep"]
    tep_reg = tep_h[tep_h["refusal_set"].isin(REGISTERED_RAW)]
    any_tep = bool(tep_reg["holds"].any())
    tep_best = tep_reg.sort_values(["n_pass", "worst"], ascending=False).iloc[0]
    turbine_reg = [t for t in v["turbine"] if registered(t["refusal_set"], t["adjustment"])]
    any_turbine = any(t["passes"] for t in turbine_reg)
    turbine_best = min(
        (t for t in turbine_reg if not pd.isna(t["placebo"]["clear_rate_pooled"])),
        key=lambda t: t["placebo"]["clear_rate_pooled"],
    )
    confirmed = bool(v["confirm"])

    naive09 = cov_text(
        *pooled(zero_shift(syn, drift=0.0, n=480, phi=0.9, method=sh.NAIVE,
                           adjustment=sh.RAW, refusal_set="base"))[:2]
    )

    # ---- synthetic tables
    sanity, sanity_ok = [], True
    for method in sh.LEVEL_METHODS:
        for adjustment in sh.ADJUSTMENTS:
            row = [METHOD_LABELS[method], adjustment]
            for n in rs.NS:
                c, mcse, _ = pooled(
                    zero_shift(syn, phi=0.0, rho=0.0, drift=0.0, n=n, method=method,
                               adjustment=adjustment, refusal_set="base")
                )
                sanity_ok &= abs(c - 0.95) <= 2 * mcse
                row.append(cov_text(c, mcse))
            sanity.append(row)
    sanity_verdict = (
        "Every cell lies within 2 MCSE of 0.95, so the estimators pass the check."
        if sanity_ok
        else "At least one cell lies more than 2 MCSE from 0.95."
    )
    phi_cols = [phi_label(p) for p in rs.PHIS]
    t_cov480 = table(["method", "adjustment", *phi_cols], coverage_rows(syn, 480))
    t_cov120 = table(["method", "adjustment", *phi_cols], coverage_rows(syn, 120))
    hold_rows = [
        [
            h["quantity"],
            METHOD_LABELS[h["method"]],
            h["adjustment"],
            h["refusal_set"],
            f"{int(h['n_pass'])} of {int(h['n_cells'])}",
            num(h["worst"]),
            "holds" if h["holds"] else "does not hold",
        ]
        for _, h in holds[holds["registered"]].iterrows()
    ]
    t_hold = table(
        ["quantity", "method", "adjustment", "refusal set", "cells passing", "worst", "verdict"],
        hold_rows,
    )

    drift_rows = []
    drift_r1, drift_cov, nodrift_r1 = {}, {}, {}
    for method in (sh.HAC, sh.EWC):
        for drift in rs.DRIFTS:
            for refusal_set in REGISTERED_RAW:
                row = [METHOD_LABELS[method], f"{drift:g}", refusal_set]
                for phi in rs.PHIS:
                    chosen = zero_shift(syn, drift=drift, n=N_HOLD, phi=phi, method=method,
                                        adjustment=sh.RAW, refusal_set=refusal_set)
                    c, mcse, _ = pooled(chosen)
                    refused = float(chosen["rate_refused"].mean())
                    row.append(f"{cov_text(c, mcse)}, refused {pct(refused)}")
                    if method == sh.HAC and refusal_set == "base+R1":
                        (drift_r1 if drift == 1.0 else nodrift_r1)[phi] = pct(refused)
                        if drift == 1.0:
                            drift_cov[phi] = num(c)
                drift_rows.append(row)
    t_drift = table(["method", "drift", "refusal set", *phi_cols], drift_rows)

    affected_rows = []
    for delta in rs.AFFECTED_DELTAS:
        for phi in rs.PHIS:
            def get(adjustment, refusal_set, phi=phi, delta=delta):
                chosen = cells(
                    syn, grid=rs.GRID_AFFECTED, delta=delta, n=N_HOLD, phi=phi,
                    method=sh.HAC, adjustment=adjustment, refusal_set=refusal_set,
                )
                assert len(chosen) == 1
                return chosen.iloc[0]

            raw, adj = get(sh.RAW, "base"), get(sh.ADJUSTED, "base")
            adj_r3 = get(sh.ADJUSTED, "base+R3")
            affected_rows.append(
                [
                    f"{delta:g}",
                    f"{phi:g}",
                    f"{num(raw['coverage'])}, bias {num(raw['bias'])}",
                    f"{num(adj['coverage'])}, bias {num(adj['bias'])}",
                    pct(adj["rate_covariate_shifted"]),
                    f"{num(adj_r3['coverage'])} of {int(adj_r3['n_answered'])}",
                ]
            )
    t_affected = table(
        ["delta", "phi", "raw", "adjusted", "R3 flags", "adjusted, base+R3"], affected_rows
    )
    unaffected = syn[
        (syn["grid"] == rs.GRID_LEVEL)
        & (syn["rho"] == rs.AFFECTED_RHO)
        & (syn["drift"] == 0)
        & (syn["adjustment"] == sh.ADJUSTED)
    ].drop_duplicates(["phi", "delta", "n"])
    r3_false = float(unaffected["rate_covariate_shifted"].mean())
    r3_false_phi0 = float(unaffected.loc[unaffected["phi"] == 0, "rate_covariate_shifted"].mean())
    r3_false_top = float(
        unaffected.loc[unaffected["phi"] == rs.PHIS[-1], "rate_covariate_shifted"].mean()
    )

    def width_ratio(method, rho):
        widths = {}
        for adjustment in sh.ADJUSTMENTS:
            widths[adjustment] = cells(
                syn, grid=rs.GRID_LEVEL, rho=rho, phi=0.5, drift=0.0, delta=0.0, n=N_HOLD,
                method=method, adjustment=adjustment, refusal_set="base",
            ).iloc[0]
        return widths[sh.ADJUSTED]["median_width"] / widths[sh.RAW]["median_width"], widths

    width_rows = []
    for rho in rs.RHOS:
        row = [f"{rho:g}"]
        for method in sh.LEVEL_METHODS:
            ratio, widths = width_ratio(method, rho)
            row.append(
                f"{num(ratio)} (reduction "
                f"{num(widths[sh.ADJUSTED]['median_variance_reduction'])})"
            )
        width_rows.append(row)
    t_width = table(["rho", *[METHOD_LABELS[m] for m in sh.LEVEL_METHODS]], width_rows)
    hac_ratio_09 = num(width_ratio(sh.HAC, 0.9)[0])
    hac_power = {
        adjustment: pct(
            cells(
                syn, grid=rs.GRID_LEVEL, rho=0.9, phi=0.5, drift=0.0, delta=0.25, n=N_HOLD,
                method=sh.HAC, adjustment=adjustment, refusal_set="base",
            ).iloc[0]["detect_rate"]
        )
        for adjustment in sh.ADJUSTMENTS
    }
    power_rows = []
    for delta in rs.DELTAS[1:]:
        row = [f"{delta:g}"]
        for method in sh.LEVEL_METHODS:
            for adjustment in sh.ADJUSTMENTS:
                chosen = cells(
                    syn, grid=rs.GRID_LEVEL, rho=0.9, phi=0.5, drift=0.0, delta=delta,
                    n=N_HOLD, method=method, adjustment=adjustment, refusal_set="base",
                ).iloc[0]
                row.append(pct(chosen["detect_rate"]))
        power_rows.append(row)
    t_power = table(
        ["delta", *[f"{METHOD_LABELS[m]} {a}" for m in sh.LEVEL_METHODS for a in sh.ADJUSTMENTS]],
        power_rows,
    )
    spread_rows = []
    for method in sh.SPREAD_METHODS:
        for n in rs.NS:
            for ratio in rs.SD_RATIOS:
                row = [METHOD_LABELS[method], str(n), f"{ratio:g}"]
                for phi in rs.PHIS:
                    chosen = cells(
                        syn, grid=rs.GRID_SPREAD, sd_ratio=ratio, n=n, phi=phi,
                        method=method, adjustment=sh.RAW, refusal_set="base",
                    ).iloc[0]
                    rate = chosen["claim_rate"] if ratio == 1.0 else chosen["detect_rate"]
                    row.append(f"{num(chosen['coverage'])}, clears {pct(rate)}")
                spread_rows.append(row)
    t_spread = table(["method", "n", "SD ratio", *phi_cols], spread_rows)

    # ---- real-bed tables
    placebo_head = [
        "method", "adjustment", "refusal set", "answered", "pooled", "well-averaged",
        "record-level",
    ]
    t_p3w_level = table(placebo_head, placebo_3w_rows(summary, sh.LEVEL))
    t_p3w_spread = table(placebo_head, placebo_3w_rows(summary, sh.SPREAD))
    t_wells = table(
        ["well", "instances", "answered", "claim rate", "answered without R1 flag",
         "claim rate without R1 flag"],
        per_well_rows(placebo),
    )
    t_refusals = table(
        ["design", "adjustment", "rows", "hard refusals", "R1 flags", "R4 flags",
         "median lag-1", "bandwidth capped"],
        refusal_rows(summary),
    )
    t_skab = table(
        ["method", "adjustment", "refusal set", "anomaly-free", "pre-onset",
         "pre-onset folder-averaged", "pre-onset record-level"],
        skab_placebo_rows(summary),
    )
    detection_head = [
        "method", "adjustment", "refusal set", "3W onset detect", "3W placebo claim",
        "SKAB onset detect", "SKAB pre-onset claim",
    ]
    t_det_level = table(detection_head, detection_rows(summary, sh.LEVEL))
    t_det_spread = table(detection_head, detection_rows(summary, sh.SPREAD))
    t_adjusted = table(
        ["design", "method", "variance reduction", "after-period variance ratio",
         "width ratio", "pairs", "adjusted rate", "raw rate", "R2 refuses", "R3 flags"],
        adjusted_rows(summary),
    )
    gap_rows = [
        [g["record"], str(g["rows"]), str(g["n_gaps"]), f"{g['longest_gap_s']:g}"]
        for g in kbed["skab_gaps_over_60_s"]
    ]
    gap_3w = f"{(hac_onset['clear_rate_pooled'] - hac_raw['clear_rate_pooled']) * 100:.1f}"
    gap_skab = (
        f"{(hac_skab_onset['clear_rate_pooled'] - hac_skab_pre['clear_rate_pooled']) * 100:.1f}"
    )

    # ---- turbine tables
    tdata = tbed["dataset"]
    tdesigns = tbed["designs"]
    t_pairs = table(
        ["pair", "rows", "span", "upgraded rows", "10 min steps", "longer steps"],
        turbine_pair_rows(tdata),
    )
    t_weeks = table(
        ["method", "adjustment", "refusal set", "placebo claim",
         *[f"injected r {r:g}: coverage (MCSE)" for r in rs.TURBINE_INJECT_R]],
        turbine_week_rows(summary),
    )
    t_pitch = table(
        ["r", "truth", "HAC raw", "EWC raw", "adjusted (every method)"], pitch_rows(turbine)
    )
    t_vg = table(
        ["method", "adjustment", "estimate [95% interval]", "excludes 0",
         "turbine placebo claim rate"],
        vg_rows(turbine, summary),
    )
    t_placebo_hac = srow(summary, rs.DESIGN_TURBINE_PLACEBO, sh.LEVEL, sh.HAC, sh.RAW, "base")
    t_placebo_ewc = srow(summary, rs.DESIGN_TURBINE_PLACEBO, sh.LEVEL, sh.EWC, sh.RAW, "base")
    t_placebo_adj = srow(
        summary, rs.DESIGN_TURBINE_PLACEBO, sh.LEVEL, sh.HAC, sh.ADJUSTED, "base"
    )
    vg_hac_raw = turbine_level(turbine, rs.DESIGN_TURBINE_VG)
    vg_hac_raw = vg_hac_raw[
        (vg_hac_raw["method"] == sh.HAC) & (vg_hac_raw["adjustment"] == sh.RAW)
    ].iloc[0]
    vg_hac_adj = turbine_level(turbine, rs.DESIGN_TURBINE_VG)
    vg_hac_adj = vg_hac_adj[
        (vg_hac_adj["method"] == sh.HAC) & (vg_hac_adj["adjustment"] == sh.ADJUSTED)
    ].iloc[0]
    pitch_pairs = tdesigns[rs.DESIGN_TURBINE_PLACEBO]
    turbine_ratios = [
        srow(summary, rs.DESIGN_TURBINE_PLACEBO, sh.LEVEL, m, sh.ADJUSTED, "base")[
            "median_width_ratio"
        ]
        for m in sh.LEVEL_METHODS
    ]
    turbine_ratio_lo, turbine_ratio_hi = num(min(turbine_ratios)), num(max(turbine_ratios))
    week_counts = (
        turbine[(turbine["design"] == rs.DESIGN_TURBINE_PLACEBO)]
        .groupby("record")["segment"]
        .nunique()
        .to_dict()
    )

    # ---- TEP tables
    t_tep_free = table(
        ["quantity", "method", "refusal set", *[f"{g}: coverage (MCSE)" for g in rs.TEP_GROUPS]],
        tep_free_rows(tep),
    )
    t_tep_faults = table(
        ["method", *[f"{g}: detection, coverage" for g in rs.TEP_GROUPS]], tep_fault_rows(tep)
    )
    tep_ratio = tep[
        (tep["quantity"] == sh.LEVEL) & (tep["method"] == sh.HAC) & (tep["refusal_set"] == "base")
    ]["median_truth_se_over_width"]
    check = ebed["fault_free_ensemble_check"]
    level_z = max(c["max_abs_level_z"] for c in check.values())
    spread_z = max(c["max_abs_spread_z"] for c in check.values())
    tep_hold_rows = [
        [
            h["quantity"],
            METHOD_LABELS[h["method"]],
            set_label(h["refusal_set"]),
            f"{int(h['n_pass'])} of {int(h['n_groups'])}",
            num(h["worst"]),
            pct(h["answered_share"]),
            "holds" if h["holds"] else "does not hold",
        ]
        for _, h in tep_h.iterrows()
    ]
    t_tep_hold = table(
        ["quantity", "method", "refusal set", "groups passing", "worst", "answered",
         "verdict"],
        tep_hold_rows,
    )
    ewc_free = {
        s: tep_rows(tep, 0, sh.LEVEL, sh.EWC, s).set_index("variable_group")
        for s in ("base", "base+R4")
    }
    cache = ebed["cache"]
    ewc_tep = tep_h[(tep_h["method"] == sh.EWC) & (tep_h["quantity"] == sh.LEVEL)]
    ewc_tep = ewc_tep.set_index("refusal_set")
    ewc_r4_answered = ewc_tep.loc["base+R4", "answered_share"]
    ewc_base_pass = int(ewc_tep.loc["base", "n_pass"])
    ewc_r4_pass = int(ewc_tep.loc["base+R4", "n_pass"])
    peak_gib = (cache.get("peak_memory_bytes") or 0) / 2**30
    onset = cache["onset_check"]
    onset_idx = onset["samples"].index(rs.TEP_BEFORE + 1)

    # ---- verdict rows
    adjust_reg = [r for r in v["adjust"] if not pd.isna(r[5]["median_width_ratio"])]
    adjust_best = min(adjust_reg, key=lambda r: r[5]["median_width_ratio"])
    verdict_rows = [
        [
            "1. holds on synthetic series (first pass)",
            f"best level: `{best['method']}` {best['adjustment']} {best['refusal_set']}, "
            f"{int(best['n_pass'])} of {int(best['n_cells'])} cells, worst {num(best['worst'])}",
            "passes" if any_hold else "fails for every configuration",
        ],
        [
            "2. usable on real records (first pass)",
            f"lowest 3W placebo claim rate: `{lowest[1]}` {lowest[2]} {lowest[3]} "
            f"({lowest[0]}), pooled {pct(lowest[5]['clear_rate_pooled'])}, well-averaged "
            f"{pct(lowest[5]['clear_rate_group_avg'])}",
            "passes" if any_usable else "fails for every configuration",
        ],
        [
            "3. adjustment kept (first pass)",
            f"smallest 3W placebo width ratio {num(adjust_best[5]['median_width_ratio'])} "
            f"(`{adjust_best[1]}` {adjust_best[2]}, {adjust_best[0]}); R3 flags "
            f"{pct(v['r3_rate'])} of affected-covariate replicates",
            "passes" if any_adjust else "fails",
        ],
        [
            "5. TEP fault-free coverage (this pass)",
            f"best: `{tep_best['method']}` {tep_best['refusal_set']} ({tep_best['quantity']}), "
            f"{int(tep_best['n_pass'])} of {int(tep_best['n_groups'])} groups, worst "
            f"{num(tep_best['worst'])}",
            "passes" if any_tep else "fails for every registered configuration",
        ],
        [
            "6. usable on the turbine pairs (this pass)",
            f"lowest placebo claim rate: `{turbine_best['method']}` "
            f"{turbine_best['adjustment']} {turbine_best['refusal_set']}, "
            f"{pct(turbine_best['placebo']['clear_rate_pooled'])}; its injected coverage "
            + ", ".join(
                f"{num(i['coverage'])}" for i in turbine_best["injected"]
            )
            + " at r " + ", ".join(f"{r:g}" for r in rs.TURBINE_INJECT_R),
            "passes" if any_turbine else "fails for every registered configuration",
        ],
    ]
    t_verdict = table(["criterion", "measured", "verdict"], verdict_rows)
    margins, flipped = [], set()
    for bed, quantity, method, _adjustment, before, after, _share in v["confirm"]:
        if not bed.startswith("TEP"):
            continue
        prior = tep_rows(tep, 0, quantity, method, before).set_index("variable_group")
        for _, r in tep_rows(tep, 0, quantity, method, after).iterrows():
            group = r["variable_group"]
            margin = r["coverage"] - (0.95 - 2 * r["coverage_mcse"])
            old = prior.loc[group]
            margins.append((margin, group, after, old["coverage"], r["coverage"]))
            if old["coverage"] < 0.95 - 2 * old["coverage_mcse"] and margin >= 0:
                flipped.add(group)
    if margins:
        low = min(margins)
        per_run = rs.TEP_SCORED_RUNS[1] - rs.TEP_SCORED_RUNS[0] + 1
        confirm_margin = (
            f"The pass is narrow: the smallest margin over 0.95 - 2 MCSE is {low[0]:.4f}, in "
            f"`{low[1]}` under {low[2]}, where R4 takes the coverage from {low[3]:.3f} to "
            f"{low[4]:.3f}. The MCSE treats the {len(rs.TEP_GROUPS[low[1]])} variables of each "
            f"of the {per_run} runs as independent replicates although the variables of one "
            "run move together, so the Monte Carlo error is larger than stated and the "
            f"{low[0]:.4f} margin lies inside it. "
            + (
                "The group that moves from failing to passing is "
                if len(flipped) == 1
                else "The groups that move from failing to passing are "
            )
            + ", ".join(f"`{g}`" for g in sorted(flipped))
            + "."
        )
    else:
        confirm_margin = ""
    confirm_rows = [
        [bed, quantity, f"`{method}`", adjustment, before, after, pct(share)]
        for bed, quantity, method, adjustment, before, after, share in v["confirm"]
    ]
    t_confirm = (
        table(
            ["bed", "quantity", "method", "adjustment", "fails under", "passes under",
             "answered"],
            confirm_rows,
        )
        if confirm_rows
        else "No (method, adjustment) moves from failing to passing when R4 is added."
    )

    # ---- document
    text = f"""
# Shift intervals: before/after level and spread on autocorrelated records

## Summary

The study asks which 95% interval for a before/after level or spread shift keeps its
coverage on autocorrelated process records, and whether regressing the target on the
other tags of the record narrows it, on SYNTHETIC AR(1) series, placebo dates and fault
onsets on 3W and SKAB, injected and published changes on the Turbine Upgrade pairs, and
TEP fault onsets. No method keeps its coverage at 0.95 minus 2 MCSE or above in every
synthetic cell with phi up to 0.9 and 480 samples per period: the best,
`{best['method']}` {best['adjustment']} {best['refusal_set']}, holds in
{int(best['n_pass'])} of {int(best['n_cells'])} cells. On the
{cache_3w['n_instances_scored']} 3W instances with no fault window a placebo date gets
an interval that excludes 0 on {pct(hac_raw['clear_rate_pooled'])} of answered
(instance, tag) pairs with `hac` and {pct(ewc_raw['clear_rate_pooled'])} with `ewc`,
and on the turbine placebo weeks on {pct(t_placebo_hac['clear_rate_pooled'])} and
{pct(t_placebo_ewc['clear_rate_pooled'])}. On the TEP fault-free runs `ewc` covers the
level at {num(ewc_free['base']['coverage'].min())} to
{num(ewc_free['base']['coverage'].max())} over the three variable groups and passes
criterion 5 in {ewc_base_pass} of 3; with the post-hoc persistence refusal R4 it passes
in {ewc_r4_pass} of 3 while answering {pct(ewc_r4_answered)} of the rows. Criteria 1, 2,
3, 5 and 6 fail for every registered configuration, so the roadmap parks the interval.

## Data

- 3W v2.0.0, dataset manifest sha256 `{sources['3w']}`. The placebo bed reads the
  own-history cache `{cache_3w['cache']}` (manifest `{cache_3w['manifest_sha256']}`):
  {cache_3w['n_instances_no_fault_window']} instances have no fault window, and
  {cache_3w['n_instances_scored']} of them from {cache_3w['n_wells_scored']} wells hold
  4 consecutive one-hour windows; {cache_3w['n_instances_too_short']} do not and are
  counted. The known-change bed reads the onset-aligned cache `{cache_on['cache']}`
  (manifest `{cache_on['manifest_sha256']}`), {cache_on['n_instances_scored']} instances
  from {cache_on['n_wells_scored']} wells. Each window carries 60 one-minute sub-group
  medians for the 6 variables every instance shares.
- SKAB, dataset manifest sha256 `{sources['skab']}`, {pbed['skab']['n_records']} records
  under `{pbed['skab']['archives']}`: {pbed['skab']['n_labelled']} labelled records and
  the anomaly-free record, 8 tags at 1 s, quality assumed GOOD, read as recorded. Records
  with a hole over 60 s between rows, the same 4 records `examples/studies/skab/REPORT.md`
  names:

{table(["record", "rows", "gaps over 60 s", "longest gap (s)"], gap_rows)}

- Turbine Upgrade Dataset (Zenodo record 5516556, CC BY 4.0), manifest sha256
  `{tdata['manifest_sha256']}` under `{tdata['directory']}`. Two turbine pairs from one
  wind farm, 10 min rows, `y_test` the normalised power of the turbine that is upgraded
  and `y_ctrl` its unchanged sister. The pitch pair's upgrade is simulated by the dataset:
  `y_test` is multiplied by 1.05 on upgraded rows with V over 9 m/s, and the Table 7.3
  file repeats the pair at r 0.02 to 0.09. The study finds
  {tdata['pitch_table']['n_rows_modified']:,} modified rows, exactly the upgraded rows
  with V > 9 ({tdata['pitch_table']['n_upgraded_rows_at_v_9_unmodified']} upgraded rows at
  V = 9.00 are unmodified), and reconstructs the unmodified `y_test` to within
  {num(tdata['pitch_table']['base_max_spread_across_r'], 4)} across r. Rows with V under
  3.5 are absent, so the sequence is irregular:

{t_pairs}

- TEP (Rieth and others, Harvard Dataverse `doi:10.7910/DVN/6C3JR1`), dataset manifest
  sha256 `{ebed['dataset_manifest_sha256']}`, the testing split: 500 runs per condition
  of 960 samples at 180 s. `build_tep_cache.py` keeps runs {cache['runs'][0]} to
  {cache['runs'][1]} and samples {cache['samples'][0]} to {cache['samples'][1]} of
  fault-free and faults 1 to 20 (cache sha256 `{cache['cache_sha256']}`, peak process
  memory {peak_gib:.1f} GiB while reading the testing files). Fault 6 zeroes `xmeas_1` at
  sample {rs.TEP_BEFORE + 1} (ensemble mean {onset['fault_mean'][onset_idx]:.4f} against
  {onset['fault_free_mean'][onset_idx]:.4f} fault-free), so samples 1 to
  {rs.TEP_BEFORE} precede the fault.

Nothing is fitted across records: every interval, coefficient and refusal reads one
record's before and after periods, and each TEP truth reads runs the scored runs do not.
Group holdout holds by construction. The 3W tables give placebo rates averaged per well
as well as pooled, because wells contribute unequal instance counts.

| design | splits | records | wells, folders or pairs |
|---|---|---|---|
""".lstrip("\n")
    for design, label in DESIGN_LABELS.items():
        d = designs[design]
        text += f"| {label} (`{design}`) | {d['n_splits']} | {d['n_records']} | {d['n_groups']} |\n"
    for design, d in tdesigns.items():
        text += f"| `{design}` | {d['n_splits']} | {d['n_records']} | {d['n_groups']} |\n"
    per_condition = rs.TEP_SCORED_RUNS[1] - rs.TEP_SCORED_RUNS[0] + 1
    text += (
        f"| `{rs.DESIGN_TEP}` | {ebed['conditions'] * per_condition} "
        f"| {ebed['conditions']} conditions x {per_condition} runs | {ebed['conditions']} |\n"
    )

    text += f"""
## Method

Run the study and render this report:

```
uv run python examples/studies/shift_intervals/fetch_turbine.py --dest data/turbine_upgrade
uv run python examples/studies/shift_intervals/build_tep_cache.py
uv run python examples/studies/shift_intervals/run_shift.py
uv run python examples/studies/shift_intervals/make_report.py --update-benchmarks
```

`run_shift.py --beds synthetic` runs one bed. Wall seconds: synthetic
{synthetic['wall_seconds']:g} with {replicates} replicates per cell, placebo
{pbed['wall_seconds']:g}, known change {kbed['wall_seconds']:g}, turbine
{tbed['wall_seconds']:g}, TEP {ebed['wall_seconds']:g}. Estimators and rules are in
`shift.py`.

### Estimators

Every period is an ordered sequence read as evenly spaced: 3W minute medians, SKAB 1 s
rows, turbine 10 min rows, TEP samples. The level estimate is mean(after) -
mean(before), reported in before-period SDs of the target. The spread estimate is
log(SD after / SD before). Intervals are 95%.

- `naive`: Welch-style standard error that treats the samples as independent, normal
  quantile 1.959964.
- `hac`: per-period long-run variance with the Newey-West Bartlett kernel and the
  Andrews (1991) AR(1) plug-in bandwidth, capped at n - 1, normal quantile. Level only.
- `ewc`: the rule of Lazarus, Lewis, Stock and Watson (2018, Journal of Business and
  Economic Statistics, "HAR Inference: Recommendations for Practice"). Per period, the
  long-run variance is (1/nu) sum of L_j^2 over nu = max(1, floor(0.4 n^(2/3))) type II
  cosine projections of the demeaned series. The critical value is Student t on the
  Welch-Satterthwaite degrees of freedom of the two periods. Level only.
- `block_bootstrap`: moving block bootstrap inside each period, block
  min(max(10, n^(1/3)), n) as `tsdive compare` uses, 200 resamples, seed 42, percentile
  interval on the difference of resampled means or on the resampled log SD ratio.
- Spread `naive`: standard error 0.5 sqrt(2/(n_a - 1) + 2/(n_b - 1)) on the log SD ratio.

`raw` runs the method on the target. `adjusted` fits OLS of the target on the
covariates over the before period only, applies the coefficients to both periods and
runs the same method on the residuals. The covariates are declared by rule. On 3W, SKAB
and TEP they are every other tag of the record that passes R0 on the split. On the
turbine pairs they are V, VcosD, VsinD, rho, S, I and `y_ctrl` where they pass R0 (D is
covered by its cosine and sine). The variance reduction is 1 - var(residual before) /
var(target before).

### Rules

A refused interval is a row with its reason.

- R0 `too_few`: fewer than 30 samples in either period. `no_spread`: the before-period
  MAD of the target is 0, or the before-period residual MAD is 0 (below 1e-9 of the
  target's MAD, the float noise of an exact linear copy). `no_covariate`: no declared
  covariate passes R0.
- `too_many_covariates`: more than one covariate per 10 before-period samples. Refuses
  the adjusted arm. It refuses every adjusted TEP row (51 covariates against 160
  samples), and on the earlier beds it refuses the 7 adjusted targets of the SKAB
  pre-onset placebo of `other__2`, whose halves are too short for 6 covariates.
- R1 `before_trend`: the HAC t statistic of an OLS slope over the before period exceeds
  1.96 in absolute value, on the series the interval reads.
- R2 `covariate_outside`: a covariate's after-period median lies outside its
  before-period 1st to 99th percentile. Refuses the adjusted arm.
- R3 `covariate_shifted`: a covariate's own `hac` level interval excludes 0. A flag on
  the adjusted arm.
- R4 `too_persistent`, post hoc: in either period n (1 - r) / (1 + r) < 10, with r the
  lag-1 autocorrelation of the series the interval reads, floored at 0. It was chosen
  after the 3W placebo result below, so it does not count toward criterion 2; the
  turbine and TEP beds, which ran after it was fixed, are where it is tested.

R0, R2 and `too_many_covariates` refuse in every refusal set. The sets add the flags:
`base`, `base+R1`, `base+R3` and `base+R1+R3` (R3 on the adjusted arm only), and the
post-hoc `base+R4` and `base+R1+R4`.

### Beds

- SYNTHETIC: y = delta 1[after] + rho x + sqrt(1 - rho^2) e + drift, with x and e
  independent unit-variance AR(1) series sharing phi. The level grid crosses phi
  {{0, 0.5, 0.9, 0.98}}, rho {{0, 0.5, 0.9}}, drift {{0, 1 SD of linear rise over both
  periods}}, delta {{0, 0.25, 0.5, 1}} and n {{120, 480}} per period. The
  affected-covariate cells set rho 0.9, delta {{0.25, 0.5, 1}} and drift 0, and the
  recorded covariate steps by 0.5 delta after the change date while y reads the
  unstepped x, so the target's true shift stays delta. The spread grid crosses phi, n and
  SD ratio {{1, 0.7, 0.5}}. Each cell draws {replicates} replicates from a seed derived from
  its parameters alone. Coverage has Monte Carlo SE sqrt(c(1 - c)/m) over the m answered
  replicates.
- Placebo dates, the control: the date moves and nothing was changed. `3w_placebo`
  takes, per instance, the first 2 of the first 4 consecutive windows (120 minute
  medians) against the next 2, label-blind. `skab_free_placebo` takes non-overlapping
  10 min before and 10 min after pairs of the anomaly-free record. `skab_preonset_placebo`
  splits each labelled record's rows before its first anomaly row in half.
  `turbine_placebo` takes non-overlapping 1 week before and 1 week after pairs over each
  pair's rows before its upgrade, from the first timestamp, label-blind.
- Injected shifts with a known truth: `turbine_injected_r<r>` repeats each placebo week
  pair with the after week's `y_test` times (1 + r) where V > 9, for r 0.02, 0.05 and
  0.09, the rule the dataset applies to its pitch pair. The truth is the mean injected
  increment over the after rows, for both arms, because the covariates are untouched.
  Adjacent week pairs share weather, so they are not independent replicates.
- Known change dates: `3w_aligned_onset` takes the 3 windows nearest before the onset
  (offsets {', '.join(str(o) for o in cache_on['before_onset_offsets'])}) against the
  {len(cache_on['after_onset_offsets'])} after it (offsets
  {', '.join(str(o) for o in cache_on['after_onset_offsets'])}); `skab_labelled_onset`
  takes the rows before the first anomaly row against the first to the last anomaly row;
  both are fault onsets, not planned changes. `turbine_pitch_change` takes the 7,000
  upgraded rows of the pitch pair against the 7,000 before them, once per published r
  and once on the reconstructed unmodified `y_test` (r 0), with the truth the mean of
  y_r minus the unmodified `y_test` over the after rows. `turbine_vg_change` takes the
  5,000 rows after the vortex generator retrofit against the 5,000 before them; no truth
  is known.
- TEP: before is samples 1-{rs.TEP_BEFORE}, after is {rs.TEP_BEFORE + 1}-{rs.TEP_LAST}.
  The truth per (condition, variable) is the mean over runs 1-{rs.TEP_TRUTH_RUNS} of the
  run's own estimate, and runs {rs.TEP_SCORED_RUNS[0]}-{rs.TEP_SCORED_RUNS[1]} are scored,
  100 per condition, every variable as the target. The fault-free level truth is 0. The
  fault-free spread truth is the ensemble estimate: the fault-free runs start with a lower
  spread, and the ensemble log SD ratio reaches |z| {spread_z:.1f} against 0, while the
  ensemble level estimate stays within |z| {level_z:.2f}. Variables fall in three groups:
  `xmeas_1_22` continuous measurements, `xmeas_23_41` sampled analysers held between
  samples, `xmv_1_11` manipulated variables.

A claim rate is the share of answered rows whose interval excludes 0 on a placebo date;
a detection rate is the same share on a known change. The record-level rate counts a
record when any of its k answered tags has a p-value under 0.05 / k (Bonferroni). For
`block_bootstrap` the p-value is twice the smaller share of resampled differences on
either side of 0.

### Pre-registered criteria

Criteria 1 to 4 were fixed in the first pass, before any result was computed. Criteria 5
to 7 were fixed in this pass, before the turbine and TEP beds ran. Criteria 1 to 3 are
re-evaluated with `ewc` added and are otherwise unchanged. The registered sets are
`base` and `base+R1` for the raw arm and `base`, `base+R1`, `base+R3` and `base+R1+R3`
for the adjusted arm; the R4 sets are post hoc.

1. An interval method holds when its synthetic coverage at a zero shift is at least
   0.95 - 2 MCSE in every drift-0 cell with phi up to 0.9 and n = 480: the 9 level cells
   of phi {{0, 0.5, 0.9}} and rho {{0, 0.5, 0.9}}, and for spread the 3 cells of phi
   {{0, 0.5, 0.9}} at SD ratio 1. The MCSE is the cell's own.
2. It is usable on real records when its 3W placebo claim rate per (instance, tag) is at
   most {USABLE_MAX:.2f}, pooled and averaged per well, after the refusals of its set.
3. Adjustment is kept when, on the 3W placebo, the median adjusted/raw width ratio is at
   most {WIDTH_RATIO_MAX} and the adjusted claim rate (pooled and averaged per well) is
   not above raw, and R3 flags at least {R3_MIN:.0%} of the replicates of the
   affected-covariate cells, pooled over its 24 cells.
4. If no (method, adjustment, refusal set) passes, the roadmap parks the study with the
   measured reason.
5. TEP: a method holds when its coverage over the fault-free runs is at least
   0.95 - 2 MCSE in each variable group.
6. Turbine: a method is usable when its `turbine_placebo` claim rate is at most
   {USABLE_MAX:.2f} and its `turbine_injected` coverage is at least 0.95 - 2 MCSE at
   every r.
7. R4 is confirmed when, for some method, adding R4 moves a failing criterion 5 or 6 to
   passing while answering at least half the rows.

## Results

### Synthetic coverage

The dumbest check first: with phi 0, rho 0 and no drift every method should cover near
0.95. Coverage (MCSE):

{table(["method", "adjustment", "n 120", "n 480"], sanity)}

{sanity_verdict} `block_bootstrap` sits lowest: its percentile endpoints carry the noise
of 200 resamples, and at n 120 a period holds 12 blocks of 10.

Coverage at a zero shift, no drift, n 480, pooled over the three rho cells (MCSE):

{t_cov480}

The same at n 120:

{t_cov120}

![coverage at a zero shift](out/{FIGURE_NAME})

`naive` covers {naive09} at phi 0.9, where neighbouring samples correlate at 0.9. `hac`
and `ewc` are the best of the four and still fall under the line at phi 0.9. The
adjusted arm tracks the raw arm, because the covariate carries the same phi.

Criterion 1 over the registered sets. In these cells R4 refuses at most
{pct(r4_in_hold)} of the replicates, and adding it {r4_hold_moves}:

{t_hold}

Drift of 1 SD over both periods, raw, zero shift, n 480, coverage (MCSE) and the share
refused:

{t_drift}

The drift moves mean(after) - mean(before) by about 0.5 SD, so the intervals centre off
the true shift. For `hac`, R1 refuses {drift_r1[0.0]} of the drifting replicates at phi 0
and {drift_r1[0.5]} at phi 0.5, and the replicates it answers still cover
{drift_cov[0.0]} and {drift_cov[0.5]}. At phi 0.9 and 0.98 it refuses {drift_r1[0.9]}
and {drift_r1[0.98]}, against {nodrift_r1[0.9]} and {nodrift_r1[0.98]} without drift:
the before-period slope is lost in the noise of a persistent series.

Affected covariate, `hac`, n 480: coverage of the target's true shift and the mean bias,
raw and adjusted, the share of replicates R3 flags, and the adjusted coverage over the
replicates `base+R3` answers:

{t_affected}

Adjusting on a covariate that moved with the change date subtracts rho x its step, 0.45
delta, from the estimate. R3 flags {pct(v['r3_rate'])} of the replicates over the 24
affected cells (both n). It flags the large steps at low phi and misses the small ones
and the persistent series, and the replicates it misses carry the full bias. On the
unaffected rho 0.9 cells R3 flags {pct(r3_false)} of replicates, from {pct(r3_false_phi0)}
at phi 0 to {pct(r3_false_top)} at phi 0.98.

Width of the adjusted interval over the raw one at a zero shift, phi 0.5, n 480, with
the median variance reduction:

{t_width}

Detection rate at rho 0.9, phi 0.5, n 480 (raw, then adjusted):

{t_power}

With an untouched covariate at rho 0.9 the adjusted `hac` interval is {hac_ratio_09} of
the raw width, against sqrt(1 - 0.81) = 0.436, and the 0.25 SD shift that raw `hac`
detects in {hac_power[sh.RAW]} of replicates is detected in {hac_power[sh.ADJUSTED]}.

Spread, raw arm: coverage of the true log SD ratio and the share of intervals excluding
0 (a claim at ratio 1, a detection below it):

{t_spread}

### Placebo dates on 3W

{cache_3w['n_instances_scored']} instances from {cache_3w['n_wells_scored']} wells, 6
tags each. Level, answered rows, claim rate pooled and averaged per well, and the
record-level rate. The R4 rows are post hoc and do not count toward criterion 2:

{t_p3w_level}

Spread:

{t_p3w_spread}

With R4 (post hoc) the lowest 3W claim rate is {pct(lowest_ph[5]['clear_rate_pooled'])}
pooled and {pct(lowest_ph[5]['clear_rate_group_avg'])} well-averaged (`{lowest_ph[1]}`
{lowest_ph[2]} {lowest_ph[3]}, {lowest_ph[0]}), over
{int(lowest_ph[5]['n_answered'])} answered rows.

Per-well cut, level, `hac` raw, with and without the rows R1 flags:

{t_wells}

Refusals and diagnostics per design, level, `hac`, `base`: hard refusals by reason, the
shares R1 and R4 flag, the median lag-1 autocorrelation of the before period of the
series the arm reads, and the share of HAC bandwidths at their n - 1 cap:

{t_refusals}

### Placebo dates on SKAB

Level, claim rate over answered rows for the anomaly-free pairs and the pre-onset
halves, the pre-onset rate averaged per folder and its record-level rate:

{t_skab}

### Known change dates beside their placebo

Level, pooled rate over answered rows. A detection rate reads only against the placebo
rate of the same bed:

{t_det_level}

Spread:

{t_det_spread}

### Adjustment on real records

Level, `base`: median variance reduction over the before period, median var(residual)
/ var(target) over the after period, median adjusted/raw width ratio over the rows both
arms answer, adjusted and raw claim or detection rate, and the share of rows R2 refuses
and R3 flags:

{t_adjusted}

### Turbine Upgrade pairs

{pitch_pairs['n_splits']} placebo week pairs ({week_counts.get(rs.PAIR_VG, 0)} on the VG
pair, {week_counts.get(rs.PAIR_PITCH, 0)} on the pitch pair; the long holes in the
record leave fewer weeks than the span holds). Level, placebo claim rate over answered
week pairs, and for each injected r the coverage of the injected truth (MCSE) and the
share of intervals excluding 0:

{t_weeks}

On the placebo weeks `hac` raw claims {pct(t_placebo_hac['clear_rate_pooled'])} and the
adjusted `hac` interval {pct(t_placebo_adj['clear_rate_pooled'])}, at a median
{num(t_placebo_adj['median_width_ratio'])} of the raw width: `y_ctrl` removes a median
{pct(t_placebo_adj['median_variance_reduction'])} of the before-week variance. The
before-week lag-1 autocorrelation of `y_test` is a median
{num(t_placebo_hac['median_lag1'])}, and R4 flags {pct(t_placebo_hac['rate_too_persistent'])}
of the raw rows.

The published pitch change, one record per r: truth and estimate in before-period SDs of
`y_test`, 95% interval, and whether it covers the truth:

{t_pitch}

The vortex generator retrofit has no known truth. Estimate and interval in before-period
SDs of `y_test`, beside the turbine placebo claim rate of the same method and arm:

{t_vg}

The raw arm reads the retrofit as {vg_hac_raw['estimate']:+.3f} SD for `hac` and
compares different weeks of wind. The adjusted arm reads `y_test` against `y_ctrl` and
the weather over the same weeks and gives {vg_hac_adj['estimate']:+.4f} SD.

### TEP fault onsets

Fault-free runs, raw arm (the adjusted arm is refused by `too_many_covariates` on every
row): coverage of the truth (MCSE) and the answered share per variable group:

{t_tep_free}

Criterion 5 per configuration:

{t_tep_hold}

Faults 1 to 20, level, raw, `base`: the range over faults of the detection rate and of
the coverage of the ensemble truth, per group:

{t_tep_faults}

The truth's own standard error is a median {num(tep_ratio.median())} of the median `hac`
interval width (range {num(tep_ratio.min())} to {num(tep_ratio.max())} over conditions
and groups), so the error of the truth is small against the intervals it scores.

### Verdict

Registered criteria:

{t_verdict}

Post hoc, not counted above: R4 on 3W brings the lowest placebo claim rate to
{pct(lowest_ph[5]['clear_rate_pooled'])}, still over the {USABLE_MAX:.0%} bar. Criterion
7 reads R4 on the beds that ran after it was fixed:

{t_confirm}

{"R4 is confirmed under criterion 7." if confirmed else "R4 is not confirmed under criterion 7."}
{confirm_margin}
No configuration passes criteria 1 and 2 together, and adjustment fails criterion 3.
Under criterion 4 the roadmap parks the interval, and `compare` keeps reporting the level
shift and spread ratio without one.

## Discussion

The synthetic bed separates the methods: an independent-samples interval covers
{naive09} at phi 0.9, and `hac` and `ewc` recover most of the gap without closing it.
TEP, a closed loop simulated 500 times per condition, is the bed where a long-run
variance estimated inside each period comes closest: with `ewc` the fault-free level
coverage is {num(ewc_free['base']['coverage'].min())} to
{num(ewc_free['base']['coverage'].max())}, and {num(ewc_free['base+R4']['coverage'].min())}
to {num(ewc_free['base+R4']['coverage'].max())} once R4 refuses the persistent rows.

The 3W and turbine records sit at the persistent end of the grid and move between
periods on their own. The median lag-1 autocorrelation of a 3W before period is
{num(lag_3w['median_lag1'])}, and {pct(lag_3w['share_bandwidth_capped'])} of the Andrews
bandwidths hit their n - 1 cap. The placebo claim rate says how often a method would
report a shift where nothing was changed: with `hac` it is
{pct(hac_raw['clear_rate_pooled'])} on 3W, {pct(hac_free['clear_rate_pooled'])} on the
SKAB anomaly-free pairs, {pct(hac_skab_pre['clear_rate_pooled'])} on the SKAB pre-onset
halves and {pct(t_placebo_hac['clear_rate_pooled'])} on the turbine weeks. The fault
onsets clear at {pct(hac_onset['clear_rate_pooled'])} on 3W and
{pct(hac_skab_onset['clear_rate_pooled'])} on SKAB, {gap_3w} and {gap_skab} percentage
points above the 3W placebo rate and the SKAB pre-onset placebo rate.

R1 lowers the 3W placebo claim rate for `hac` from {pct(hac_raw['clear_rate_pooled'])} to
{pct(hac_raw_r1['clear_rate_pooled'])} and for `ewc` from
{pct(ewc_raw['clear_rate_pooled'])} to {pct(ewc_raw_r1['clear_rate_pooled'])}. The rows it
keeps show no trend inside the before period, and their level still moves between the
periods.

Adjustment narrows the interval on synthetic series by sqrt(1 - rho^2) when the
covariate is untouched and biases it by rho times the covariate's step when the
covariate moved with the change date. On 3W the covariates remove a median
{pct(hac_adj['median_variance_reduction'])} of the before-period variance, but R2
refuses {pct(hac_adj['rate_covariate_outside'])} of the adjusted rows, and in the after
period the residual variance is a median {num(hac_adj['median_variance_ratio_after'])}
times the target's own. On the turbine pairs, where `y_ctrl` is a sister turbine in the
same wind, the adjusted interval is {turbine_ratio_lo} to {turbine_ratio_hi} of the raw
width, and adjusted `hac` still claims a shift on
{pct(t_placebo_adj['clear_rate_pooled'])} of the placebo weeks.

## Limits and further work

- The 3W, SKAB and turbine change dates are fault onsets, a simulated upgrade and one
  retrofit. No bed carries a documented setpoint or tuning change on a plant.
- The 3W periods are 2 hours (placebo) and 3 and 2 hours (onset), the SKAB periods are
  minutes and the turbine weeks are one week. A before period taken from several other
  days, to estimate the between-period variance, is the next design to measure.
- The onset-aligned 3W cache holds 2 windows after the onset, so that bed compares 180
  minute medians against 120.
- The turbine placebo and injected beds have {pitch_pairs['n_splits']} week pairs, and
  adjacent pairs share weather, so their coverage MCSE understates the uncertainty.
- R4 was chosen after the 3W result. Its confirmation rests on TEP, a simulation.
- `block_bootstrap` uses {run['parameters']['bootstrap_replicates']} resamples, as
  `compare` did at {code_label(synthetic)}, the code this study ran. Its percentile
  endpoints carry that resampling noise.

## Files

| file | holds |
|---|---|
| `shift.py` | estimators, rules and the synthetic generator |
| `run_shift.py` | the beds and their result frames |
| `fetch_turbine.py` | the Turbine Upgrade fetch, pinned to its size and sha256 |
| `build_tep_cache.py` | the local TEP cache of samples 1-320 |
| `make_report.py` | this report, the figure and the BENCHMARKS section |
| `results/synthetic.csv` | one row per (cell, quantity, method, adjustment, refusal set) |
| `results/placebo.csv` | one row per (split, tag, quantity, method, adjustment) on placebo dates |
| `results/known_change.csv` | the same on known change dates |
| `results/turbine.csv` | the same on the turbine pairs, with the truth where one is known |
| `results/tep.csv` | one row per (condition, variable group, quantity, method, refusal set) |
| `results/summary.csv` | one row per (design, quantity, method, adjustment, refusal set) |
| `results/run.json` | provenance, parameters, counts and wall seconds per bed |
| `results/benchmarks_section.md` | the BENCHMARKS.md REAL section |
| `out/{FIGURE_NAME}` | synthetic coverage at a zero shift against phi |

The per-row TEP results stay in the local cache (`{ebed['per_row_file']}`,
{ebed['n_per_row']:,} rows).
"""
    return text


# ---------------------------------------------------------------- benchmarks


def render_benchmarks_section(data: dict) -> str:
    syn, summary, run, tep = data["synthetic"], data["summary"], data["run"], data["tep"]
    turbine = data["turbine"]
    beds = run["beds"]
    synthetic, pbed, kbed = beds[rs.BED_SYNTHETIC], beds[rs.BED_PLACEBO], beds[rs.BED_KNOWN]
    tbed, ebed = beds[rs.BED_TURBINE], beds[rs.BED_TEP]
    sources = pbed["dataset_manifest_sha256"]
    dataset_3w = f"3W v2.0.0 real `{sources['3w']}`"
    dataset_skab = f"SKAB `{sources['skab']}`"
    dataset_turbine = f"Turbine Upgrade `{tbed['dataset']['manifest_sha256']}`"
    dataset_tep = f"TEP (Rieth 2017 simulation, testing split) `{ebed['dataset_manifest_sha256']}`"
    code_syn, code_p, code_k = code_label(synthetic), code_label(pbed), code_label(kbed)
    code_t, code_e = code_label(tbed), code_label(ebed)
    cache_3w, cache_on = pbed["3w_cache"], kbed["3w_cache"]
    split_syn = (
        f"{synthetic['replicates']} replicates per cell, seed from the cell's parameters; "
        "nothing fitted across series"
    )
    split_3w = (
        "nothing fitted across records, so group holdout by well holds by construction; "
        f"{cache_3w['n_instances_scored']} instances with no fault window, "
        f"{cache_3w['n_wells_scored']} wells, 2 windows before and 2 after, label-blind; "
        f"cache manifest `{cache_3w['manifest_sha256']}`"
    )
    split_on = (
        "nothing fitted across records, so group holdout by well holds by construction; "
        f"{cache_on['n_instances_scored']} instances, {cache_on['n_wells_scored']} wells, "
        f"{len(cache_on['before_onset_offsets'])} windows before the onset and "
        f"{len(cache_on['after_onset_offsets'])} after; cache manifest "
        f"`{cache_on['manifest_sha256']}`"
    )
    split_skab = (
        "nothing fitted across records, so group holdout by folder holds by construction; "
        f"{pbed['skab']['n_labelled']} labelled records, "
        f"{pbed['skab']['n_records'] - pbed['skab']['n_labelled']} anomaly-free, "
        "1 s rows as recorded"
    )
    n_weeks = tbed["designs"][rs.DESIGN_TURBINE_PLACEBO]["n_splits"]
    split_turbine = (
        "nothing fitted across records, so group holdout by pair holds by construction; "
        f"{n_weeks} week pairs on 2 turbine pairs, before the upgrade, label-blind; "
        "adjacent weeks share weather"
    )
    scored = rs.TEP_SCORED_RUNS
    split_tep = (
        f"truth from runs 1-{rs.TEP_TRUTH_RUNS}, scored runs {scored[0]}-{scored[1]}, "
        f"samples 1-{rs.TEP_BEFORE} against {rs.TEP_BEFORE + 1}-{rs.TEP_LAST}; nothing "
        f"fitted across runs; cache sha256 `{ebed['cache']['cache_sha256']}`"
    )

    rows: list[tuple[str, str, str, str, str, str]] = []
    for method in sh.LEVEL_METHODS:
        parts = []
        for phi in rs.PHIS:
            c, mcse, _ = pooled(
                zero_shift(syn, drift=0.0, n=480, phi=phi, method=method,
                           adjustment=sh.RAW, refusal_set="base")
            )
            parts.append(f"phi {phi:g}: {c:.3f} ({mcse:.3f})")
        rows.append(
            (
                f"`{method}` level interval, raw: coverage at a zero shift, no drift, "
                "n 480 per period, pooled over rho (MCSE)",
                ", ".join(parts),
                "SYNTHETIC",
                code_syn,
                "synthetic AR(1)",
                split_syn,
            )
        )
    drift = zero_shift(syn, drift=1.0, n=480, phi=0.5, method=sh.HAC, adjustment=sh.RAW,
                       refusal_set="base+R1")
    c, mcse, _ = pooled(drift)
    rows.append(
        (
            "`hac` level interval, raw, R1 refusing: coverage under a 1 SD drift, phi 0.5, "
            "n 480",
            f"{c:.3f} ({mcse:.3f}) over the answered replicates, R1 refuses "
            f"{pct(float(drift['rate_refused'].mean()))}",
            "SYNTHETIC",
            code_syn,
            "synthetic AR(1)",
            split_syn,
        )
    )
    v = verdicts(data)
    affected = syn[
        (syn["grid"] == rs.GRID_AFFECTED)
        & (syn["method"] == sh.HAC)
        & (syn["adjustment"] == sh.ADJUSTED)
        & (syn["refusal_set"] == "base")
    ]
    bias_ratio = float((affected["bias"] / affected["delta"]).mean())
    rows.append(
        (
            "`hac` level interval, adjusted, covariate stepping by half the target's shift",
            f"mean bias {bias_ratio:+.3f} delta; R3 flags {pct(v['r3_rate'])} of replicates "
            "over the 24 cells",
            "SYNTHETIC",
            code_syn,
            "synthetic AR(1)",
            split_syn,
        )
    )
    for method, adjustment, refusal_set in (
        (sh.NAIVE, sh.RAW, "base"),
        (sh.HAC, sh.RAW, "base"),
        (sh.HAC, sh.RAW, "base+R1"),
        (sh.EWC, sh.RAW, "base"),
        (sh.EWC, sh.RAW, "base+R1"),
        (sh.EWC, sh.RAW, "base+R4"),
        (sh.BOOTSTRAP, sh.RAW, "base"),
        (sh.HAC, sh.ADJUSTED, "base"),
        (sh.HAC, sh.ADJUSTED, "base+R1+R3"),
    ):
        r = srow(summary, rs.DESIGN_3W_PLACEBO, sh.LEVEL, method, adjustment, refusal_set)
        value = (
            f"claim rate pooled {pct(r['clear_rate_pooled'])}, well-averaged "
            f"{pct(r['clear_rate_group_avg'])}, record-level {pct(r['record_clear_rate'])}; "
            f"{int(r['n_answered'])} of {int(r['n_rows'])} (instance, tag) rows answered"
        )
        if adjustment == sh.ADJUSTED:
            value += f"; width ratio adjusted/raw {num(r['median_width_ratio'])}"
        rows.append(
            (
                f"`{method}` level interval, {adjustment}, {set_label(refusal_set)}: "
                "3W placebo date",
                value,
                dataset_3w,
                code_p,
                "placebo date",
                split_3w,
            )
        )
    for method in (sh.HAC, sh.EWC):
        onset = srow(summary, rs.DESIGN_3W_ONSET, sh.LEVEL, method, sh.RAW, "base")
        placebo = srow(summary, rs.DESIGN_3W_PLACEBO, sh.LEVEL, method, sh.RAW, "base")
        rows.append(
            (
                f"`{method}` level interval, raw, base: 3W fault onset",
                f"detection {pct(onset['clear_rate_pooled'])} of {int(onset['n_answered'])} "
                f"answered (instance, tag) rows, beside the placebo claim rate "
                f"{pct(placebo['clear_rate_pooled'])}",
                dataset_3w,
                code_k,
                "known change date (fault onset)",
                split_on,
            )
        )
    for method in (sh.HAC, sh.EWC):
        free = srow(summary, rs.DESIGN_SKAB_FREE, sh.LEVEL, method, sh.RAW, "base")
        pre = srow(summary, rs.DESIGN_SKAB_PRE, sh.LEVEL, method, sh.RAW, "base")
        onset = srow(summary, rs.DESIGN_SKAB_ONSET, sh.LEVEL, method, sh.RAW, "base")
        rows.append(
            (
                f"`{method}` level interval, raw, base: SKAB placebo dates and fault onset",
                f"claim rate {pct(free['clear_rate_pooled'])} on the anomaly-free pairs, "
                f"{pct(pre['clear_rate_pooled'])} on the pre-onset halves; detection "
                f"{pct(onset['clear_rate_pooled'])} at the onset",
                dataset_skab,
                code_k,
                "placebo date; known change date (fault onset)",
                split_skab,
            )
        )
    for method, adjustment in ((sh.HAC, sh.RAW), (sh.EWC, sh.RAW), (sh.HAC, sh.ADJUSTED)):
        placebo = srow(
            summary, rs.DESIGN_TURBINE_PLACEBO, sh.LEVEL, method, adjustment, "base"
        )
        parts = []
        for r in rs.TURBINE_INJECT_R:
            inj = srow(
                summary, rs.DESIGN_TURBINE_INJECTED.format(r=r), sh.LEVEL, method,
                adjustment, "base",
            )
            parts.append(f"r {r:g}: {num(inj['coverage'])} ({num(inj['coverage_mcse'])})")
        rows.append(
            (
                f"`{method}` level interval, {adjustment}, base: turbine placebo weeks and "
                "injected shifts",
                f"placebo claim rate {pct(placebo['clear_rate_pooled'])} of "
                f"{int(placebo['n_answered'])} week pairs; injected coverage (MCSE) "
                + ", ".join(parts),
                dataset_turbine,
                code_t,
                "placebo date; injected shift",
                split_turbine,
            )
        )
    vg = turbine[
        (turbine["design"] == rs.DESIGN_TURBINE_VG)
        & (turbine["quantity"] == sh.LEVEL)
        & (turbine["method"] == sh.HAC)
    ].set_index("adjustment")
    rows.append(
        (
            "`hac` level interval, base: vortex generator retrofit, 5,000 rows each side",
            f"raw {vg.loc[sh.RAW, 'estimate']:+.3f} [{vg.loc[sh.RAW, 'lo']:+.3f}, "
            f"{vg.loc[sh.RAW, 'hi']:+.3f}] SD, adjusted {vg.loc[sh.ADJUSTED, 'estimate']:+.4f} "
            f"[{vg.loc[sh.ADJUSTED, 'lo']:+.4f}, {vg.loc[sh.ADJUSTED, 'hi']:+.4f}] SD; "
            "no truth known",
            dataset_turbine,
            code_t,
            "known change date (retrofit)",
            "one record; nothing fitted across records",
        )
    )
    for method, refusal_set in ((sh.HAC, "base"), (sh.EWC, "base"), (sh.EWC, "base+R4")):
        free = tep[
            (tep["fault"] == 0)
            & (tep["quantity"] == sh.LEVEL)
            & (tep["method"] == method)
            & (tep["refusal_set"] == refusal_set)
        ].set_index("variable_group")
        faults = tep[
            (tep["fault"] > 0)
            & (tep["quantity"] == sh.LEVEL)
            & (tep["method"] == method)
            & (tep["refusal_set"] == refusal_set)
        ]
        rows.append(
            (
                f"`{method}` level interval, raw, {set_label(refusal_set)}: TEP fault-free "
                "coverage per variable group, detection over faults 1-20",
                "coverage (MCSE) "
                + ", ".join(
                    f"{g} {num(free.loc[g, 'coverage'])} ({num(free.loc[g, 'coverage_mcse'])})"
                    for g in rs.TEP_GROUPS
                )
                + f"; answered {pct(free['n_answered'].sum() / free['n_rows'].sum())}; "
                f"detection {pct(faults['detect_rate'].min())} to "
                f"{pct(faults['detect_rate'].max())} over (fault, group)",
                dataset_tep,
                code_e,
                "fault onset with an ensemble truth",
                split_tep,
            )
        )

    lines = [
        SECTION_HEADING,
        "",
        "Before/after level intervals (`naive`, `hac`, `ewc`, `block_bootstrap`), raw and "
        "adjusted on the other tags of the record, scored for coverage on synthetic AR(1) "
        "series, TEP fault onsets and injected turbine shifts, and for claim rates on "
        "placebo dates where nothing was changed. Rows marked post hoc use a refusal chosen "
        "after the 3W result. Method, refusal rules and the pre-registered criteria are in "
        "[examples/studies/shift_intervals/REPORT.md]"
        "(examples/studies/shift_intervals/REPORT.md).",
        "",
        "This section is not regenerated by `make bench`; it is produced by "
        "`examples/studies/shift_intervals/run_shift.py` and `make_report.py`, and "
        "carried through the generator unchanged.",
        "",
        "| stage | metric | value | dataset (manifest sha256) | code | design | split |",
        "|---|---|---|---|---|---|---|",
    ]
    lines += [
        f"| none | {metric} | {value} | {dataset} | {code} | {design} | {split} |"
        for metric, value, dataset, code, design, split in rows
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
    plot_coverage(data["synthetic"], Path(args.out) / FIGURE_NAME)
    Path(args.report).write_text(unwrap(render_report(data)), encoding="utf-8", newline="\n")
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

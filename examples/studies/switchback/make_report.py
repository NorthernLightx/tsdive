"""Report, figures and BENCHMARKS.md section for the switchback schedule study.

Reads ``results/`` as ``run_switchback.py`` wrote it and renders
``REPORT.md``, ``out/01_claim_rate_at_zero_shift.png``,
``out/02_detection_by_shift.png`` and ``results/benchmarks_section.md``.
With ``--update-benchmarks`` the section replaces the
``## REAL: switchback schedules (3W, TEP, Turbine Upgrade, SKAB)`` section
of the repository's ``BENCHMARKS.md``, or is appended when the file has
none. Every number in the prose is read from the frames the tables are
built from, and every criterion is evaluated on the rounded values the
CSVs carry.
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
sys.path.insert(0, str(HERE.parent / "shift_intervals"))
sys.path.insert(0, str(HERE))

import run_switchback as rsw  # noqa: E402
import switchback as sb  # noqa: E402

BENCH_MD = ROOT / "BENCHMARKS.md"
SECTION_HEADING = "## REAL: switchback schedules (3W, TEP, Turbine Upgrade, SKAB)"
FIGURE_CLAIMS = "01_claim_rate_at_zero_shift.png"
FIGURE_DETECTION = "02_detection_by_shift.png"

BED_LABELS = {
    rsw.BED_3W: "3W",
    rsw.BED_TEP: "TEP",
    rsw.BED_TURBINE: "Turbine Upgrade",
    rsw.BED_SKAB: "SKAB",
}
UNIT_LABELS = {
    rsw.BED_3W: "(instance, tag, draw)",
    rsw.BED_TEP: "(run, variable, draw)",
    rsw.BED_TURBINE: "(pair, draw)",
    rsw.BED_SKAB: "(tag, draw)",
}
PAIR_LABELS = {"vg_pair": "VG pair", "pitch_pair": "pitch pair"}
SCHEDULE_ARMS = sb.SCHEDULE_ARMS
W4_DELTA = 0.25
W4_RATE = 0.8
W5_RATIO = 0.8
W3_FRACTION = 0.10
COVERAGE = 0.95
SERIES = ("#2a78d6", "#eb6834")  # categorical slots 1 and 2, validated for colour vision
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dc"


# ---------------------------------------------------------------- formatting


def num(value, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    if isinstance(value, float) and math.isinf(value):
        return "inf"
    return f"{value:.{digits}f}"


def pct(value, digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{100 * value:.{digits}f}%"


def rate_text(row: pd.Series, digits: int = 3) -> str:
    if row is None or row["n_answered"] == 0:
        return f"refused: {row['top_reason']}" if row is not None else "n/a"
    mcse = row["rate_mcse"]
    return (
        f"{num(row['rate'], digits)} ({num(mcse, digits)})"
        if not math.isnan(mcse)
        else num(row["rate"], digits)
    )


def unwrap(text: str) -> str:
    """Join each prose paragraph onto one line; headings, tables, fences and images pass."""
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
        if stripped.startswith("- ") or (stripped[:1].isdigit() and stripped[1:3] == ". "):
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
    read = {"keep_default_na": False, "na_values": [""]}
    return {
        "summary": pd.read_csv(results / rsw.SUMMARY_CSV, **read),
        "designs": pd.read_csv(
            results / rsw.DESIGNS_CSV, keep_default_na=False, dtype={"n_assignments": str}
        ),
        "refusals": pd.read_csv(results / rsw.REFUSALS_CSV, keep_default_na=False),
        "per_well": pd.read_csv(results / rsw.PER_WELL_CSV, **read),
        "pairs": pd.read_csv(results / rsw.PAIRS_CSV, **read),
        "pooled": pd.read_csv(results / rsw.POOLED_CSV, **read),
        "mean_bias": pd.read_csv(results / rsw.MEAN_BIAS_CSV, **read),
        "exact": pd.read_csv(results / rsw.EXACT_CSV, keep_default_na=False),
        "run": json.loads((results / "run.json").read_text("utf-8")),
    }


def code_label(section: dict) -> str:
    return f"tsdive {section['code']['tsdive_version']} @ `{section['code']['git_head'][:8]}`"


def cell(summary: pd.DataFrame, bed, block, tau, washout, delta, arm, adjustment) -> pd.Series:
    mask = (
        (summary["bed"] == bed)
        & (summary["block"] == block)
        & (summary["tau"] == tau)
        & (summary["washout"] == washout)
        & (summary["delta"] == delta)
        & (summary["arm"] == arm)
        & (summary["adjustment"] == adjustment)
    )
    rows = summary[mask]
    if len(rows) != 1:
        raise KeyError((bed, block, tau, washout, delta, arm, adjustment))
    return rows.iloc[0]


def open_blocks(designs: pd.DataFrame, bed: str) -> list[int]:
    """Block lengths of a bed with no refused design on any record."""
    rows = designs[designs["bed"] == bed]
    return [int(b) for b, g in rows.groupby("block", sort=True) if (g["refusal"] == "").all()]


def taus(block: int) -> list[float]:
    return sorted({tau for tau, _ in rsw.tau_washouts(block)})


# ---------------------------------------------------------------- criteria


def w1_rows(data: dict, arms=SCHEDULE_ARMS) -> pd.DataFrame:
    """Every delta-0 cell of the open blocks: rate against 0.05 + 2 MCSE."""
    summary, designs = data["summary"], data["designs"]
    out = []
    for bed in rsw.BEDS:
        for block in open_blocks(designs, bed):
            for tau, washout in rsw.tau_washouts(block):
                for arm in arms:
                    for adjustment in sb.ADJUSTMENTS:
                        row = cell(summary, bed, block, tau, washout, 0.0, arm, adjustment)
                        if row["n_answered"] == 0:
                            continue
                        mcse = 0.0 if math.isnan(row["rate_mcse"]) else row["rate_mcse"]
                        bar = round(sb.ALPHA + 2 * mcse, 4)
                        out.append(
                            {
                                "bed": bed,
                                "block": block,
                                "tau": tau,
                                "washout": washout,
                                "arm": arm,
                                "adjustment": adjustment,
                                "rate": row["rate"],
                                "mcse": mcse,
                                "bar": bar,
                                "excess": round(row["rate"] - bar, 4),
                                "passes": bool(row["rate"] <= bar),
                            }
                        )
    return pd.DataFrame(out)


def w2_rows(data: dict) -> pd.DataFrame:
    summary, designs = data["summary"], data["designs"]
    out = []
    for bed in rsw.BEDS:
        for block in open_blocks(designs, bed):
            for tau in taus(block):
                washout = sb.washout_steps(tau)
                for delta in rsw.DELTAS[1:]:
                    for adjustment in sb.ADJUSTMENTS:
                        row = cell(
                            summary, bed, block, tau, washout, delta, sb.RANDOMIZATION, adjustment
                        )
                        if row["n_answered"] == 0:
                            continue
                        bar = round(COVERAGE - 2 * row["coverage_mcse"], 4)
                        out.append(
                            {
                                "bed": bed,
                                "block": block,
                                "tau": tau,
                                "washout": washout,
                                "delta": delta,
                                "adjustment": adjustment,
                                "coverage": row["coverage"],
                                "mcse": row["coverage_mcse"],
                                "bar": bar,
                                "passes": bool(row["coverage"] >= bar),
                            }
                        )
    return pd.DataFrame(out)


def w3_rows(data: dict) -> pd.DataFrame:
    summary, designs = data["summary"], data["designs"]
    out = []
    for bed in rsw.BEDS:
        for block in open_blocks(designs, bed):
            for tau in taus(block):
                for delta in rsw.DELTAS[1:]:
                    for adjustment in sb.ADJUSTMENTS:
                        registered = cell(
                            summary,
                            bed,
                            block,
                            tau,
                            sb.washout_steps(tau),
                            delta,
                            sb.RANDOMIZATION,
                            adjustment,
                        )
                        none = cell(
                            summary, bed, block, tau, 0, delta, sb.RANDOMIZATION, adjustment
                        )
                        null = cell(
                            summary,
                            bed,
                            block,
                            tau,
                            sb.washout_steps(tau),
                            0.0,
                            sb.RANDOMIZATION,
                            adjustment,
                        )
                        if registered["n_answered"] == 0:
                            continue
                        bar = round(W3_FRACTION * delta, 4)
                        out.append(
                            {
                                "bed": bed,
                                "block": block,
                                "tau": tau,
                                "delta": delta,
                                "adjustment": adjustment,
                                "bias_registered": registered["median_bias"],
                                "bias_none": none["median_bias"],
                                "null_median": null["median_bias"],
                                "carryover": round(
                                    registered["median_bias"] - null["median_bias"], 4
                                ),
                                "bar": bar,
                                "passes": bool(abs(registered["median_bias"]) <= bar),
                            }
                        )
    return pd.DataFrame(out)


def detection(data: dict, bed: str, block: int, adjustment: str) -> dict[float, float]:
    return {
        delta: cell(data["summary"], bed, block, 0.0, 0, delta, sb.RANDOMIZATION, adjustment)[
            "rate"
        ]
        for delta in rsw.DELTAS
    }


def smallest_delta(rates: dict[float, float]) -> str:
    for delta in rsw.DELTAS[1:]:
        if rates[delta] >= W4_RATE:
            return f"{delta:g}"
    return "none"


def w4_rows(data: dict) -> pd.DataFrame:
    out = []
    for bed in rsw.BEDS:
        for block in open_blocks(data["designs"], bed):
            for adjustment in sb.ADJUSTMENTS:
                rates = detection(data, bed, block, adjustment)
                out.append(
                    {
                        "bed": bed,
                        "block": block,
                        "adjustment": adjustment,
                        **{f"d{d:g}": rates[d] for d in rsw.DELTAS},
                        "smallest": smallest_delta(rates),
                        "passes": bool(rates[W4_DELTA] >= W4_RATE),
                    }
                )
    return pd.DataFrame(out)


def w5_rows(data: dict) -> pd.DataFrame:
    out = []
    bed = rsw.BED_TURBINE
    for block in open_blocks(data["designs"], bed):
        row = cell(data["summary"], bed, block, 0.0, 0, 0.0, sb.RANDOMIZATION, sb.ADJUSTED)
        out.append(
            {
                "block": block,
                "ratio": row["median_width_ratio"],
                "rate": row["rate"],
                "mcse": row["rate_mcse"],
                "passes": bool(row["median_width_ratio"] <= W5_RATIO),
            }
        )
    return pd.DataFrame(out)


def verdicts(data: dict) -> dict:
    w1 = w1_rows(data)
    rand = w1[w1["arm"] == sb.RANDOMIZATION]
    w2 = w2_rows(data)
    w2_raw = w2[w2["adjustment"] == sb.RAW]
    w3 = w3_rows(data)
    w3_raw = w3[w3["adjustment"] == sb.RAW]
    w4 = w4_rows(data)
    w4_raw = w4[w4["adjustment"] == sb.RAW]
    w4_beds = {bed: bool(g["passes"].any()) for bed, g in w4_raw.groupby("bed", sort=False)}
    w5 = w5_rows(data)
    w1_turbine_adj = rand[(rand["bed"] == rsw.BED_TURBINE) & (rand["adjustment"] == sb.ADJUSTED)]
    out = {
        "w1": bool(rand["passes"].all()),
        "w2": bool(w2_raw["passes"].all()),
        "w3": bool(w3_raw["passes"].all()),
        "w4": any(w4_beds.values()),
        "w4_beds": w4_beds,
        "w5": bool(w5["passes"].all() and w1_turbine_adj["passes"].all()),
        "arms_hold": {
            key: bool(g["passes"].all()) for key, g in w1.groupby(["arm", "adjustment"], sort=False)
        },
        "tables": {"w1": w1, "w2": w2, "w3": w3, "w4": w4, "w5": w5},
    }
    out["supported"] = out["w1"] and out["w2"] and out["w3"] and out["w4"]
    return out


# ---------------------------------------------------------------- figures


def _style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_claims(data: dict, path: Path) -> None:
    """Claim rate at delta 0 per arm, raw, tau 0, washout 0: one panel per bed, one colour per L."""
    summary, designs = data["summary"], data["designs"]
    arms = list(sb.ARMS)
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.6), sharey=True)
    for ax, bed in zip(axes, rsw.BEDS, strict=True):
        _style(ax)
        blocks = open_blocks(designs, bed)
        for k, block in enumerate(blocks):
            offset = (k - (len(blocks) - 1) / 2) * 0.22
            for i, arm in enumerate(arms):
                row = cell(summary, bed, block, 0.0, 0, 0.0, arm, sb.RAW)
                if row["n_answered"] == 0:
                    continue
                ax.plot(
                    max(row["rate"], 0.005),
                    i + offset,
                    "o",
                    ms=6,
                    color=SERIES[k],
                    mec="white",
                    mew=1.0,
                    label=rsw.block_label(bed, block) if i == 0 else None,
                )
        ax.axvline(sb.ALPHA, color=INK, linewidth=1.0, linestyle=(0, (3, 2)))
        ax.set_xscale("log")
        ax.set_xlim(0.005, 1.2)
        ax.set_xticks([0.01, 0.05, 0.2, 1.0])
        ax.set_xticklabels(["0.01", "0.05", "0.2", "1"])
        ax.set_title(BED_LABELS[bed], fontsize=10, color=INK, loc="left")
        ax.legend(frameon=False, fontsize=8, loc="upper right", title="block", title_fontsize=8)
        ax.set_xlabel("claim rate at a zero injected shift", fontsize=8, color=MUTED)
    axes[0].set_yticks(range(len(arms)))
    axes[0].set_yticklabels(arms, fontsize=8, color=INK)
    axes[0].invert_yaxis()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, metadata={"Software": None})
    plt.close(fig)


def plot_detection(data: dict, path: Path) -> None:
    """Randomization detection rate against the injected shift, tau 0, washout 0."""
    designs = data["designs"]
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.4), sharey=True)
    for ax, bed in zip(axes, rsw.BEDS, strict=True):
        _style(ax)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        for k, block in enumerate(open_blocks(designs, bed)):
            for adjustment, style in ((sb.RAW, "-"), (sb.ADJUSTED, (0, (4, 2)))):
                rates = detection(data, bed, block, adjustment)
                ax.plot(
                    list(rates),
                    list(rates.values()),
                    linestyle=style,
                    marker="o",
                    ms=4.5,
                    linewidth=2.0,
                    color=SERIES[k],
                    label=f"{rsw.block_label(bed, block)}, {adjustment}",
                )
        ax.axhline(W4_RATE, color=INK, linewidth=1.0, linestyle=(0, (3, 2)))
        ax.set_ylim(0, 1.02)
        ax.set_xticks(list(rsw.DELTAS))
        ax.set_title(BED_LABELS[bed], fontsize=10, color=INK, loc="left")
        ax.set_xlabel("injected shift (sigma)", fontsize=8, color=MUTED)
        ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    axes[0].set_ylabel("rate with p <= 0.05", fontsize=8, color=MUTED)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, metadata={"Software": None}, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- report tables


def data_rows(data: dict) -> list[list[str]]:
    run, designs = data["run"]["beds"], data["designs"]
    rows = []
    for bed in rsw.BEDS:
        section = run[bed]
        ks = designs[designs["bed"] == bed]
        blocks = []
        for block, group in ks.groupby("block", sort=True):
            k_text = "/".join(str(int(k)) for k in sorted(group["k"]))
            refused = " (refused)" if (group["refusal"] != "").any() else ""
            blocks.append(f"{rsw.block_label(bed, int(block))}: K {k_text}{refused}")
        rows.append(
            [
                BED_LABELS[bed],
                f"`{dataset_sha(data, bed)}`",
                str(section["n_records"]),
                f"{section['n_usable_targets']} of {section['n_targets']}",
                "; ".join(blocks),
                str(section["draws"]),
            ]
        )
    return rows


def dataset_sha(data: dict, bed: str) -> str:
    dataset = data["run"]["beds"][bed]["dataset"]
    if bed == rsw.BED_TURBINE:
        return dataset["manifest_sha256"]
    return dataset["source_manifest_sha256"]


def w1_table(data: dict, adjustment: str) -> list[list[str]]:
    summary = data["summary"]
    rows = []
    for bed in rsw.BEDS:
        for block in sorted(summary.loc[summary["bed"] == bed, "block"].unique()):
            block = int(block)
            entries = [BED_LABELS[bed], rsw.block_label(bed, block)]
            for arm in sb.ARMS:
                row = cell(summary, bed, block, 0.0, 0, 0.0, arm, adjustment)
                entries.append(rate_text(row) if row["n_answered"] else row["top_reason"])
            rows.append(entries)
    return rows


def worst_rows(verdict: dict) -> list[list[str]]:
    w1 = verdict["tables"]["w1"]
    rows = []
    for arm in SCHEDULE_ARMS:
        for adjustment in sb.ADJUSTMENTS:
            sub = w1[(w1["arm"] == arm) & (w1["adjustment"] == adjustment)]
            worst = sub.sort_values(
                ["excess", "bed", "block"], ascending=[False, True, True], kind="stable"
            ).iloc[0]
            rows.append(
                [
                    f"`{arm}`",
                    adjustment,
                    f"{BED_LABELS[worst['bed']]}, "
                    f"{rsw.block_label(worst['bed'], int(worst['block']))}, "
                    f"tau {worst['tau']:g}, washout {int(worst['washout'])}",
                    num(worst["rate"], 4),
                    num(worst["bar"], 4),
                    f"{int((~sub['passes']).sum())} of {len(sub)}",
                ]
            )
    return rows


def w2_table(verdict: dict) -> list[list[str]]:
    w2 = verdict["tables"]["w2"]
    rows = []
    for (bed, block, adjustment), group in w2.groupby(["bed", "block", "adjustment"], sort=False):
        worst = group.sort_values(["coverage", "tau", "delta"], kind="stable").iloc[0]
        by_delta = {d: group.loc[group["delta"] == d, "coverage"].min() for d in (0.25, 0.5)}
        rows.append(
            [
                BED_LABELS[bed],
                rsw.block_label(bed, int(block)),
                adjustment,
                num(by_delta[0.25]),
                num(by_delta[0.5]),
                f"{num(worst['coverage'], 4)} against {num(worst['bar'], 4)} "
                f"(tau {worst['tau']:g}, delta {worst['delta']:g})",
                f"{int((~group['passes']).sum())} of {len(group)}",
            ]
        )
    return rows


W3_HEADER = [
    "bed",
    "L",
    "tau",
    "w = ceil(3 tau)",
    "delta",
    "bias at w 0",
    "bias at w = ceil(3 tau)",
    "bar",
    "delta 0 median",
    "net of it",
]


def _w3_row(row: pd.Series) -> list[str]:
    return [
        BED_LABELS[row["bed"]],
        rsw.block_label(row["bed"], int(row["block"])),
        f"{row['tau']:g}",
        f"{sb.washout_steps(row['tau'])}",
        f"{row['delta']:g}",
        num(row["bias_none"], 4),
        num(row["bias_registered"], 4),
        num(row["bar"], 4),
        num(row["null_median"], 4),
        num(row["carryover"], 4),
    ]


def w3_table(verdict: dict) -> list[list[str]]:
    """Raw arm at tau = L/4, delta 0.25 and 0.5."""
    w3 = verdict["tables"]["w3"]
    raw = w3[w3["adjustment"] == sb.RAW]
    rows = []
    for (_, _), group in raw.groupby(["bed", "block"], sort=False):
        tau = max(group["tau"])
        for delta in (0.25, 0.5):
            rows.append(_w3_row(group[(group["tau"] == tau) & (group["delta"] == delta)].iloc[0]))
    return rows


def w3_failing(verdict: dict) -> list[list[str]]:
    w3 = verdict["tables"]["w3"]
    return [_w3_row(r) for _, r in w3[(w3["adjustment"] == sb.RAW) & ~w3["passes"]].iterrows()]


def w4_table(verdict: dict) -> list[list[str]]:
    w4 = verdict["tables"]["w4"]
    rows = []
    for _, row in w4.iterrows():
        rows.append(
            [
                BED_LABELS[row["bed"]],
                rsw.block_label(row["bed"], int(row["block"])),
                row["adjustment"],
                num(row["d0.1"]),
                num(row["d0.25"]),
                num(row["d0.5"]),
                row["smallest"],
            ]
        )
    return rows


def refusal_rows(data: dict) -> list[list[str]]:
    """Refused ``randomization`` units per (bed, L, adjustment, reason), over the washouts."""
    ref = data["refusals"]
    rand = ref[ref["arm"] == sb.RANDOMIZATION]
    rows = []
    keys = ["bed", "block", "adjustment", "reason"]
    for (bed, block, adjustment, reason), group in rand.groupby(keys, sort=False):
        washouts = ", ".join(str(int(w)) for w in group["washout"])
        lo, hi = int(group["n_units"].min()), int(group["n_units"].max())
        units = f"{lo}" if lo == hi else f"{lo} to {hi}"
        rows.append(
            [
                BED_LABELS[bed],
                rsw.block_label(bed, int(block)),
                adjustment,
                f"`{reason}`",
                washouts,
                f"{units} of {int(group['n_units_total'].iloc[0])}",
            ]
        )
    return rows


def well_rows(data: dict) -> list[list[str]]:
    wells = data["per_well"]
    base = wells[(wells["tau"] == 0) & (wells["washout"] == 0) & (wells["adjustment"] == sb.RAW)]
    rows = []
    for well, group in base.groupby("well", sort=True):

        def get(arm, block, group=group):
            row = group[(group["arm"] == arm) & (group["block"] == block)].iloc[0]
            return row

        r15, r30 = get(sb.RANDOMIZATION, 15), get(sb.RANDOMIZATION, 30)
        prepost = get(sb.PREPOST_HAC, 15)
        rows.append(
            [
                well,
                str(int(r15["n_answered"])),
                num(r15["rate"]),
                num(r30["rate"]),
                f"{num(prepost['rate'])} of {int(prepost['n_answered'])}",
            ]
        )
    return rows


def pair_rows(data: dict) -> list[list[str]]:
    pairs = data["pairs"]
    rows = []
    for pair in sorted(pairs["pair"].unique()):
        for block in open_blocks(data["designs"], rsw.BED_TURBINE):
            sub = pairs[
                (pairs["pair"] == pair)
                & (pairs["block"] == block)
                & (pairs["tau"] == 0)
                & (pairs["washout"] == 0)
                & (pairs["adjustment"] == sb.RAW)
            ]

            def get(arm, delta, sub=sub):
                return sub[(sub["arm"] == arm) & (sub["delta"] == delta)].iloc[0]

            prepost = get(sb.PREPOST_HAC, 0.0)
            rows.append(
                [
                    PAIR_LABELS.get(pair, pair),
                    rsw.block_label(rsw.BED_TURBINE, block),
                    num(get(sb.RANDOMIZATION, 0.0)["rate"]),
                    num(get(sb.RANDOMIZATION, 0.25)["rate"]),
                    num(get(sb.RANDOMIZATION, 0.0)["median_width"]),
                    "claims" if prepost["rate"] == 1 else "no claim",
                ]
            )
    return rows


# ---------------------------------------------------------------- corrected criteria


def w1_corrected(conf: dict) -> pd.DataFrame:
    """W1' per (bed, adjustment): the pooled claim rate and the worst cell against 3.5 MCSE."""
    pooled = conf["pooled"]
    cells = w1_rows(conf, arms=(sb.RANDOMIZATION,))
    out = []
    for bed in rsw.BEDS:
        for adjustment in sb.ADJUSTMENTS:
            sub = cells[(cells["bed"] == bed) & (cells["adjustment"] == adjustment)].copy()
            if sub.empty:
                continue
            p = pooled[
                (pooled["bed"] == bed)
                & (pooled["adjustment"] == adjustment)
                & (pooled["quantity"] == "claim_rate_delta_0")
            ].iloc[0]
            sub["bar35"] = [round(sb.ALPHA + 3.5 * m, 4) for m in sub["mcse"]]
            sub["over"] = sub["rate"] - sub["bar35"]
            worst = sub.sort_values(
                ["over", "block", "tau"], ascending=[False, True, True], kind="stable"
            ).iloc[0]
            bar = round(sb.ALPHA + 2 * p["mcse"], 4)
            out.append(
                {
                    "bed": bed,
                    "adjustment": adjustment,
                    "pooled": p["value"],
                    "pooled_mcse": p["mcse"],
                    "pooled_bar": bar,
                    "n_units": int(p["n_units"]),
                    "pooled_passes": bool(p["value"] <= bar),
                    "worst": worst.to_dict(),
                    "cells_pass": bool((sub["rate"] <= sub["bar35"]).all()),
                    "n_cells": len(sub),
                }
            )
    return pd.DataFrame(out)


def w2_corrected(conf: dict) -> pd.DataFrame:
    """W2' per (bed, adjustment): pooled coverage and the worst cell against 3.5 MCSE."""
    pooled = conf["pooled"]
    cells = w2_rows(conf)
    out = []
    for bed in rsw.BEDS:
        for adjustment in sb.ADJUSTMENTS:
            sub = cells[(cells["bed"] == bed) & (cells["adjustment"] == adjustment)].copy()
            if sub.empty:
                continue
            p = pooled[
                (pooled["bed"] == bed)
                & (pooled["adjustment"] == adjustment)
                & (pooled["quantity"] == "coverage_registered_washout")
            ].iloc[0]
            sub["bar35"] = [round(COVERAGE - 3.5 * m, 4) for m in sub["mcse"]]
            sub["under"] = sub["bar35"] - sub["coverage"]
            worst = sub.sort_values(
                ["under", "block", "tau"], ascending=[False, True, True], kind="stable"
            ).iloc[0]
            bar = round(COVERAGE - 2 * p["mcse"], 4)
            out.append(
                {
                    "bed": bed,
                    "adjustment": adjustment,
                    "pooled": p["value"],
                    "pooled_mcse": p["mcse"],
                    "pooled_bar": bar,
                    "pooled_passes": bool(p["value"] >= bar),
                    "worst": worst.to_dict(),
                    "cells_pass": bool((sub["coverage"] >= sub["bar35"]).all()),
                    "n_cells": len(sub),
                }
            )
    return pd.DataFrame(out)


def w3_corrected(conf: dict) -> pd.DataFrame:
    """W3' per cell at washout ceil(3 tau): |mean bias| against 0.10 delta + 2 MCSE."""
    mb = conf["mean_bias"]
    out = []
    for bed in rsw.BEDS:
        for block in open_blocks(conf["designs"], bed):
            for tau in taus(block):
                for delta in rsw.DELTAS[1:]:
                    for adjustment in sb.ADJUSTMENTS:
                        pick = (
                            (mb["bed"] == bed)
                            & (mb["block"] == block)
                            & (mb["tau"] == tau)
                            & (mb["delta"] == delta)
                            & (mb["adjustment"] == adjustment)
                        )
                        registered = mb[pick & (mb["washout"] == sb.washout_steps(tau))].iloc[0]
                        none = mb[pick & (mb["washout"] == 0)].iloc[0]
                        if registered["n_units"] == 0:
                            continue
                        bar = round(W3_FRACTION * delta + 2 * registered["mcse"], 4)
                        out.append(
                            {
                                "bed": bed,
                                "block": block,
                                "tau": tau,
                                "delta": delta,
                                "adjustment": adjustment,
                                "mean": registered["mean_bias"],
                                "mcse": registered["mcse"],
                                "mean_none": none["mean_bias"],
                                "bar": bar,
                                "passes": bool(abs(registered["mean_bias"]) <= bar),
                            }
                        )
    return pd.DataFrame(out)


def corrected_verdicts(conf: dict) -> dict:
    w1 = w1_corrected(conf)
    w2 = w2_corrected(conf)
    w3 = w3_corrected(conf)
    exact = conf["exact"]
    exact_3w = exact[exact["bed"] == rsw.BED_3W]
    base = verdicts(conf)
    w2_raw = w2[w2["adjustment"] == sb.RAW]
    w3_raw = w3[w3["adjustment"] == sb.RAW]
    turbine_adj = w1[(w1["bed"] == rsw.BED_TURBINE) & (w1["adjustment"] == sb.ADJUSTED)]
    out = {
        "w1": bool(
            w1["pooled_passes"].all()
            and w1["cells_pass"].all()
            and (exact_3w["max"] <= sb.ALPHA).all()
        ),
        "w1_exact_max": float(exact_3w["max"].max()),
        "w2": bool(w2_raw["pooled_passes"].all() and w2_raw["cells_pass"].all()),
        "w3": bool(w3_raw["passes"].all()),
        "w4": base["w4"],
        "w4_beds": base["w4_beds"],
        "w5": base["w5"],
        "w5_prime": bool(
            base["tables"]["w5"]["passes"].all()
            and turbine_adj["pooled_passes"].all()
            and turbine_adj["cells_pass"].all()
        ),
        "tables": {**base["tables"], "w1": w1, "w2": w2, "w3": w3, "exact": exact},
        "base": base,
    }
    out["supported"] = out["w1"] and out["w2"] and out["w3"] and out["w4"]
    return out


def _cell_where(row: pd.Series) -> str:
    return (
        f"{rsw.block_label(row['bed'], int(row['block']))}, tau {row['tau']:g}, "
        f"washout {int(row['washout'])}"
    )


def w1c_table(verdict: dict) -> list[list[str]]:
    rows = []
    for _, r in verdict["tables"]["w1"].iterrows():
        worst = r["worst"]
        rows.append(
            [
                BED_LABELS[r["bed"]],
                r["adjustment"],
                f"{num(r['pooled'], 4)} ({num(r['pooled_mcse'], 4)})",
                num(r["pooled_bar"], 4),
                _cell_where(worst),
                num(worst["rate"], 4),
                num(worst["bar35"], 4),
                "passes" if r["pooled_passes"] and r["cells_pass"] else "fails",
            ]
        )
    return rows


def w2c_table(verdict: dict) -> list[list[str]]:
    rows = []
    for _, r in verdict["tables"]["w2"].iterrows():
        worst = r["worst"]
        rows.append(
            [
                BED_LABELS[r["bed"]],
                r["adjustment"],
                f"{num(r['pooled'], 4)} ({num(r['pooled_mcse'], 4)})",
                num(r["pooled_bar"], 4),
                f"{_cell_where(worst)}, delta {worst['delta']:g}",
                num(worst["coverage"], 4),
                num(worst["bar35"], 4),
                "passes" if r["pooled_passes"] and r["cells_pass"] else "fails",
            ]
        )
    return rows


W3C_HEADER = [
    "bed",
    "L",
    "tau",
    "delta",
    "mean bias at w = ceil(3 tau) (MCSE)",
    "bar 0.10 delta + 2 MCSE",
    "mean bias at w 0",
]


def w3c_rows(frame: pd.DataFrame) -> list[list[str]]:
    return [
        [
            BED_LABELS[r["bed"]],
            rsw.block_label(r["bed"], int(r["block"])),
            f"{r['tau']:g}",
            f"{r['delta']:g}",
            f"{num(r['mean'], 4)} ({num(r['mcse'], 4)})",
            num(r["bar"], 4),
            num(r["mean_none"], 4),
        ]
        for _, r in frame.iterrows()
    ]


def w3c_table(verdict: dict) -> list[list[str]]:
    """Raw arm at tau = L/4, every delta > 0."""
    w3 = verdict["tables"]["w3"]
    raw = w3[w3["adjustment"] == sb.RAW]
    quarter = raw[raw["tau"] == raw.groupby(["bed", "block"])["tau"].transform("max")]
    return w3c_rows(quarter)


def exact_table(verdict: dict) -> list[list[str]]:
    return [
        [
            BED_LABELS[r["bed"]],
            rsw.block_label(r["bed"], int(r["block"])),
            str(int(r["k"])),
            str(int(r["n_assignments"])),
            r["adjustment"],
            str(int(r["n_units"])),
            num(r["max"], 4),
            num(r["mean"], 4),
            num(r["bound"], 4),
        ]
        for _, r in verdict["tables"]["exact"].iterrows()
    ]


def w4_design_guidance(verdict: dict, conf: dict) -> list[str]:
    """(bed, L) pairs whose raw detection at 0.25 sigma reached 0.8, with record span and K."""
    w4 = verdict["tables"]["w4"]
    designs = conf["designs"]
    out = []
    for _, r in w4[(w4["adjustment"] == sb.RAW) & w4["passes"]].iterrows():
        ks = designs[(designs["bed"] == r["bed"]) & (designs["block"] == r["block"])]["k"]
        k_text = " and ".join(str(int(k)) for k in sorted(ks))
        label = rsw.block_label(r["bed"], int(r["block"]))
        out.append(
            f"{BED_LABELS[r['bed']]} records scheduled as {k_text} blocks of {label} "
            f"(detection {num(r['d0.25'])}, raw)"
        )
    return out


# ---------------------------------------------------------------- report


def render_report(data: dict, conf: dict) -> str:
    summary = data["summary"]
    run = data["run"]
    verdict = verdicts(data)
    corrected = corrected_verdicts(conf)
    tables = verdict["tables"]
    w1 = tables["w1"]
    rand = w1[w1["arm"] == sb.RANDOMIZATION]

    def c(bed, block, arm, adjustment=sb.RAW, delta=0.0, tau=0.0, washout=0):
        return cell(summary, bed, block, tau, washout, delta, arm, adjustment)

    pp_3w = c(rsw.BED_3W, 15, sb.PREPOST_HAC)
    pp_tep = c(rsw.BED_TEP, 40, sb.PREPOST_HAC)
    failing_w1 = rand[~rand["passes"]]
    w2_raw = tables["w2"][tables["w2"]["adjustment"] == sb.RAW]
    failing_w2 = w2_raw[~w2_raw["passes"]]
    w3_raw = tables["w3"][tables["w3"]["adjustment"] == sb.RAW]
    failing_w3 = w3_raw[~w3_raw["passes"]]
    w5 = tables["w5"]
    arms_hold = verdict["arms_hold"]
    others = [arm for arm in SCHEDULE_ARMS if arm != sb.RANDOMIZATION]
    held = [f"`{a}` {adj}" for a in others for adj in sb.ADJUSTMENTS if arms_hold[(a, adj)]]
    hold_text = ("only " + " and ".join(held) + " holds") if held else "none holds"
    ctab = corrected["tables"]
    pooled_lo, pooled_hi = ctab["w1"]["pooled"].min(), ctab["w1"]["pooled"].max()
    conf_turbine = cell(
        conf["summary"], rsw.BED_TURBINE, 144, 0.0, 0, W4_DELTA, sb.RANDOMIZATION, sb.RAW
    )
    guidance = w4_design_guidance(corrected, conf)
    offset = conf["run"]["draw_offset"]

    def verdict_word(ok: bool) -> str:
        return "passes" if ok else "fails"

    first = (
        f"W1 in {len(failing_w1)} of {len(rand)} cells, W2 in {len(failing_w2)} of "
        f"{len(w2_raw)} and W3 in {len(failing_w3)} of {len(w3_raw)}"
    )
    second = (
        "W1', W2' and W3' pass and W4 passes on "
        + ", ".join(BED_LABELS[b] for b, ok in corrected["w4_beds"].items() if ok)
        + ", so the numbers support a switchback plan-and-analyze command"
        if corrected["supported"]
        else "the corrected criteria "
        + ", ".join(
            name
            for name, ok in (
                ("W1'", corrected["w1"]),
                ("W2'", corrected["w2"]),
                ("W3'", corrected["w3"]),
                ("W4", corrected["w4"]),
            )
            if not ok
        )
        + " fail, so the command stays parked"
    )

    def before_rows(pair: str) -> int:
        counts = run["beds"][rsw.BED_TURBINE]["dataset"]["pairs"][pair]
        return counts["n_rows"] - counts["n_upgraded_rows"]

    tau_word = {
        rsw.BED_3W: "min",
        rsw.BED_TEP: "samples",
        rsw.BED_TURBINE: "steps of 10 min",
        rsw.BED_SKAB: "s",
    }
    exact_tep = data["exact"][
        (data["exact"]["bed"] == rsw.BED_TEP) & (data["exact"]["adjustment"] == sb.RAW)
    ]

    lines = [
        "# Switchback schedules on real records",
        "",
        "## Summary",
        "",
        f"""The study adds a known shift to randomly assigned blocks of records where nothing
was changed: {run["beds"]["3w"]["n_records"]} 3W instances, {run["beds"]["tep"]["n_records"]}
TEP fault-free testing runs, the two Turbine Upgrade pairs before their upgrades and the SKAB
anomaly-free record. Under the first registration the `randomization` arm misses {first}, and
the command is parked; that registration demanded every cell within 2 MCSE with no
multiplicity control and read the median bias where randomization makes the mean unbiased. On
fresh assignments (seed offset {offset}) the pooled claim rate at a zero injected shift is
{pct(pooled_lo)} to {pct(pooled_hi)} per bed and adjustment, where the first-half against
second-half split of the same 3W records claims a shift on {pct(pp_3w["rate"])} of tags.
Detection of a 0.25 sigma shift reaches 0.8 only on the turbine pairs with 1-day blocks
({num(conf_turbine["rate"])} on the fresh draws). Under the corrected criteria, registered
before the fresh draws, {second}.""",
        "",
        "## Data",
        "",
        table(
            [
                "bed",
                "dataset manifest sha256",
                "records",
                "usable targets",
                "block (K)",
                "draws per record",
            ],
            data_rows(data),
        ),
        "",
        f"""- 3W v2.0.0 through `run_shift.placebo_3w` on the window cache (manifest
`{run["beds"]["3w"]["dataset"]["manifest_sha256"]}`): the
{run["beds"]["3w"]["n_records"]} instances with no fault window and 4 consecutive one-hour
windows, 240 minute medians of the first such run, {run["beds"]["3w"]["n_groups"]} wells.
Every tag with 30 finite samples and MAD > 0 is the target in turn.""",
        f"""- TEP fault-free testing runs {run["beds"]["tep"]["dataset"]["runs"][0]} to
{run["beds"]["tep"]["dataset"]["runs"][1]}, 960 samples at 180 s, from
`build_tep_cache.py` (cache sha256 `{run["beds"]["tep"]["dataset"]["cache_sha256"]}`). Every
variable is the target in turn.""",
        f"""- Turbine Upgrade, the rows of each pair before its upgrade ({before_rows("vg_pair")}
VG pair rows and {before_rows("pitch_pair")} pitch pair rows), target `y_test`. Calendar
blocks start at each pair's first timestamp, so the holes of up to 57 days leave some blocks
empty.""",
        """- SKAB, the anomaly-free record, 1 s rows as recorded, calendar blocks. Every tag with
MAD > 0 is the target in turn.""",
        "",
        "## Method",
        "",
        "### Schedule and injected shift",
        "",
        """- Schedule: K = floor(T / L) blocks of L steps from the first sample (the last block is
dropped when K is odd), exactly K/2 of them assigned to B by balanced complete randomization.
The seed comes from (bed, record, L, draw), so every tag of a record shares a draw's
schedule.""",
        """- Refusal `design_too_small`: fewer than 20 balanced assignments, or a smallest
attainable two-sided p above 0.05. A refused design gives a row with its reason for every arm.""",
        """- Injected response: u = delta sigma in B blocks and 0 in A blocks, sigma = 1.4826 MAD
of the target over the whole record, through a first-order lag with time constant tau. The
setting of a sample applies from the previous sample to it; over a hole of dt steps the
response moves 1 - exp(-dt / tau) of the way.""",
        """- Washout: the samples in the first w steps of every block are dropped. The grid is
w = 0 and the registered w = ceil(3 tau).""",
        """- Truth: the mean of the response over kept B samples minus its mean over kept A
samples. Bias is also read against the steady state delta sigma.""",
        f"""- Grid: delta 0, 0.1, 0.25 and 0.5 sigma; tau 0, L/10 and L/4 steps (the steps are
{tau_word["3w"]} on 3W, {tau_word["tep"]} on TEP, {tau_word["turbine"]} on the turbine pairs
and {tau_word["skab"]} on SKAB); raw and adjusted.""",
        "",
        "### Inference arms",
        "",
        """All arms read the kept samples and report 95% two-sided intervals.""",
        "",
        """- `randomization`: the coefficient of z in OLS of y on [1, z] (B mean minus A mean) or on
[1, z, X], and its randomization p-value over the balanced assignments. Designs with at most
1000 assignments are enumerated; larger ones draw 1000 fresh assignments and use
(1 + #{|T_j| >= |T|}) / 1001. The adjusted statistic refits for every assignment through
Frisch-Waugh in block space. The interval inverts the test under a constant additive shift and
is the hull of the accepted shifts; an adjusted interval is unbounded when enough assignments
lie closer to the covariates than the observed one.""",
        """- `block_t`: Welch t on the kept block means, with the Welch-Satterthwaite degrees of
freedom.""",
        """- `hac`, `ewc`: the same OLS coefficient with the long-run variance of its score from
the shift interval study's estimators: Newey-West with the Andrews bandwidth and a normal
critical value, and equal-weighted cosine with nu = 0.4 n^(2/3) and t on nu degrees of
freedom.""",
        """- `naive`: the OLS standard error for independent samples.""",
        """- `prepost_hac`, `prepost_ewc`: the first half of the schedule span as A and the
second half as B, analysed by `shift.level_interval` with the shift interval study's hard
refusals and its before-period adjustment.""",
        """- Adjustment: the covariates are every other usable tag of the record (3W, SKAB, TEP)
or V, VcosD, VsinD, rho, S, I and y_ctrl (turbine), declared before any interval is computed.
The adjusted arm is refused (`too_many_covariates`) when the covariates outnumber one per 10
kept samples.""",
        """- Exact size: for every enumerated design, each balanced assignment is taken as the
observed one in turn and the share the test rejects is counted per (record, target), at delta
0, tau 0 and washout 0. It reads no draw.""",
        "",
        "Commands:",
        "",
        "```",
        "uv run python examples/studies/switchback/build_tep_cache.py",
        "uv run python examples/studies/switchback/run_switchback.py",
        f"uv run python examples/studies/switchback/run_switchback.py --draw-offset {offset} \\",
        "    --out examples/studies/switchback/results/confirm --rows data/switchback_rows/confirm",
        "uv run python examples/studies/switchback/make_report.py --update-benchmarks",
        "```",
        "",
        f"""`run_switchback.py` runs the beds on `--workers` processes (default
{rsw.DEFAULT_WORKERS}); the results do not depend on the worker count.""",
        "",
        "### First registration",
        "",
        """These criteria were fixed before the first run. Rates pool over the units of the table
headers below; the Monte Carlo SE (MCSE) of a rate is clustered by (record, draw), the unit the
assignment is drawn for. Every comparison reads the four-decimal values of
`results/summary.csv`.""",
        "",
        """- W1 validity: the `randomization` claim rate at delta 0 is at most 0.05 + 2 MCSE on
every bed, every block length that is not refused, every tau and washout, raw and adjusted.
Another arm holds only if it meets the same bar everywhere.""",
        """- W2 coverage: the `randomization` interval covers the exact truth at 0.95 - 2 MCSE or
more for every delta > 0 with w = ceil(3 tau), on every bed and block length (raw arm).""",
        """- W3 washout: with w = ceil(3 tau), |median bias against delta sigma| <= 0.10 delta
sigma (raw arm, every delta > 0 and tau).""",
        """- W4 practical power: on each bed, the `randomization` detection rate at delta 0.25
(raw, tau 0, washout 0) reaches 0.8 for at least one block length that is not refused.""",
        """- W5 adjustment: on the turbine bed, the median width ratio adjusted over raw of the
`randomization` interval (tau 0, washout 0, delta 0) is at most 0.8 at every block length, and
W1 holds for the adjusted arm.""",
        """- Verdict: if W1, W2 and W3 pass and W4 passes on at least one bed, the numbers support
a switchback plan-and-analyze command. Otherwise the command is parked with the measured
reason.""",
        "",
        "### Corrected criteria",
        "",
        f"""These criteria were registered after the first run and before the confirmatory run,
and apply to the confirmatory run only. That run draws every observed and reference assignment
from a new seed (`--draw-offset {offset}`) on the same records, targets, draw counts and grid.
A pooled MCSE is clustered by (record, L, draw). At delta 0 the cells of one washout are one
computation whatever tau, so the claim pool reads each (L, washout) once.""",
        "",
        """- W1': per bed, raw and adjusted separately, the `randomization` claim rate at delta 0
pooled over the open (L, washout) cells is at most 0.05 + 2 MCSE, no cell exceeds
0.05 + 3.5 MCSE of the cell, and the largest exact rejection share over all 3W (record, target)
is at most 0.05.""",
        """- W2': per bed, the coverage pooled over the delta > 0 cells at w = ceil(3 tau) is at
least 0.95 - 2 MCSE, and no cell is below 0.95 - 3.5 MCSE of the cell (raw arm).""",
        """- W3': per bed, L, tau and delta > 0 at w = ceil(3 tau), |mean(estimate - delta sigma)|
<= 0.10 delta sigma + 2 MCSE, clustered by (record, draw) (raw arm).""",
        """- W4 and W5: unchanged, read on the confirmatory run.""",
        """- Verdict: if W1', W2' and W3' pass and W4 passes on at least one bed, the numbers
support a switchback plan-and-analyze command, which needs a SCOPE amendment for statements
under a randomized schedule. Otherwise the command stays parked.""",
        "",
        "## Results on the first draws",
        "",
        "### Claims at a zero injected shift",
        "",
        """Claim rate at delta 0 with its MCSE, tau 0 and washout 0. Units are (instance, tag,
draw) on 3W, (run, variable, draw) on TEP, (pair, draw) on the turbine pairs and (tag, draw) on
SKAB; the `prepost` arms have one unit per (record, target).""",
        "",
        "Raw:",
        "",
        table(["bed", "L", *[f"`{a}`" for a in sb.ARMS]], w1_table(data, sb.RAW)),
        "",
        "Adjusted:",
        "",
        table(["bed", "L", *[f"`{a}`" for a in sb.ARMS]], w1_table(data, sb.ADJUSTED)),
        "",
        f"""On the same 3W records the `prepost_hac` arm claims a shift on {pct(pp_3w["rate"])}
of {int(pp_3w["n_answered"])} answered tags, the placebo failure of the shift interval study.
On the TEP runs it claims {pct(pp_tep["rate"])}. Both turbine pairs get a claim from both
`prepost` arms at every block length. The `prepost` arms read no draw, so the confirmatory run
repeats these numbers.""",
        "",
        "Worst cell over every tau and washout of the open block lengths:",
        "",
        table(
            ["arm", "adjustment", "worst cell", "claim rate", "bar 0.05 + 2 MCSE", "cells over"],
            worst_rows(verdict),
        ),
        "",
        f"""The `randomization` arm holds in {int(rand["passes"].sum())} of {len(rand)} cells.
Of the other arms, {hold_text}.""",
        "",
        "### Coverage of the exact truth",
        "",
        table(
            [
                "bed",
                "L",
                "adjustment",
                "lowest at delta 0.25",
                "lowest at delta 0.5",
                "lowest against its bar",
                "cells under",
            ],
            w2_table(verdict),
        ),
        "",
        "### Bias and washout",
        "",
        """Median bias of the `randomization` estimate against delta sigma, in sigma, raw arm,
at tau = L/4. The last two columns are descriptive: the median of the estimate at delta 0 under
the same washout, and the bias net of it.""",
        "",
        table(W3_HEADER, w3_table(verdict)),
        "",
        f"""{len(w3_raw) - len(failing_w3)} of {len(w3_raw)} cells meet the W3 bar. The cells
over it:""",
        "",
        table(W3_HEADER, w3_failing(verdict)) if len(failing_w3) else "none",
        "",
        "### Detection",
        "",
        """`randomization` detection rate by injected shift, tau 0 and washout 0, and the
smallest shift on the grid with a rate of 0.8:""",
        "",
        table(
            [
                "bed",
                "L",
                "adjustment",
                "delta 0.1",
                "delta 0.25",
                "delta 0.5",
                "smallest delta at 0.8",
            ],
            w4_table(verdict),
        ),
        "",
        "### Adjustment on the turbine pairs",
        "",
        table(
            ["L", "median width ratio adjusted/raw", "adjusted claim rate at delta 0 (MCSE)"],
            [
                [
                    rsw.block_label(rsw.BED_TURBINE, int(r["block"])),
                    num(r["ratio"]),
                    f"{num(r['rate'])} ({num(r['mcse'])})",
                ]
                for _, r in w5.iterrows()
            ],
        ),
        "",
        "Per pair, raw, tau 0, washout 0:",
        "",
        table(
            [
                "pair",
                "L",
                "claim rate at delta 0",
                "detection at delta 0.25",
                "median width (sigma)",
                "`prepost_hac` at delta 0",
            ],
            pair_rows(data),
        ),
        "",
        "### Refused designs and units",
        "",
        table(
            ["bed", "L", "K", "assignments", "smallest p", "refusal"],
            [
                [
                    BED_LABELS[r["bed"]],
                    r["block_label"],
                    str(int(r["k"])),
                    str(r["n_assignments"]),
                    num(r["min_p"], 4),
                    r["refusal"] or "none",
                ]
                for _, r in data["designs"].iterrows()
            ],
        ),
        "",
        """Refused `randomization` units, delta 0 (a unit is a record, target and draw):""",
        "",
        table(["bed", "L", "adjustment", "reason", "washouts", "units"], refusal_rows(data)),
        "",
        "### 3W per well",
        "",
        """Claim rate at delta 0, raw, tau 0, washout 0. The `prepost_hac` column counts answered
tags:""",
        "",
        table(
            ["well", "units", "`randomization` 15 min", "`randomization` 30 min", "`prepost_hac`"],
            well_rows(data),
        ),
        "",
        "### Verdict under the first registration",
        "",
        table(
            ["criterion", "result"],
            [
                ["W1 validity", verdict_word(verdict["w1"])],
                ["W2 coverage", verdict_word(verdict["w2"])],
                ["W3 washout", verdict_word(verdict["w3"])],
                [
                    "W4 practical power",
                    ", ".join(
                        f"{BED_LABELS[b]} {verdict_word(ok)}"
                        for b, ok in verdict["w4_beds"].items()
                    ),
                ],
                ["W5 adjustment", verdict_word(verdict["w5"])],
                ["verdict", "supported" if verdict["supported"] else "parked"],
            ],
        ),
        "",
    ]
    lines += flaw_paragraphs(data, verdict, exact_tep)
    lines += [
        "## Confirmatory run on fresh draws",
        "",
        f"""Every observed and reference assignment comes from the seed of (bed, record, L,
draw + {offset}). Records, targets, draws per record and grid are those of the first run.
Outputs are in `results/confirm/`.""",
        "",
        "### Validity (W1')",
        "",
        """`randomization` claim rate at delta 0, pooled over the open (L, washout) cells, and
the cell furthest over 0.05 + 3.5 MCSE:""",
        "",
        table(
            [
                "bed",
                "adjustment",
                "pooled rate (MCSE)",
                "bar 0.05 + 2 MCSE",
                "worst cell",
                "its rate",
                "bar 0.05 + 3.5 MCSE",
                "result",
            ],
            w1c_table(corrected),
        ),
        "",
        """Exact rejection share over every balanced assignment of the enumerated designs, per
(record, target), delta 0, tau 0, washout 0. The bound floor(0.05 C) / C is the largest share a
test of size 0.05 can reach on C assignments:""",
        "",
        table(
            [
                "bed",
                "L",
                "K",
                "assignments",
                "adjustment",
                "(record, target) units",
                "largest share",
                "mean share",
                "bound",
            ],
            exact_table(corrected),
        ),
        "",
        f"![Claim rate at a zero injected shift, fresh draws](out/{FIGURE_CLAIMS})",
        "",
        "Claim rates of every arm on the fresh draws, raw, tau 0, washout 0:",
        "",
        table(["bed", "L", *[f"`{a}`" for a in sb.ARMS]], w1_table(conf, sb.RAW)),
        "",
        "### Coverage (W2')",
        "",
        table(
            [
                "bed",
                "adjustment",
                "pooled coverage (MCSE)",
                "bar 0.95 - 2 MCSE",
                "worst cell",
                "its coverage",
                "bar 0.95 - 3.5 MCSE",
                "result",
            ],
            w2c_table(corrected),
        ),
        "",
        "### Mean bias and washout (W3')",
        "",
        """Mean of the `randomization` estimate minus delta sigma, in sigma, raw arm, at
tau = L/4, with the mean at washout 0 beside it:""",
        "",
        table(W3C_HEADER, w3c_table(corrected)),
        "",
    ]
    w3c = ctab["w3"]
    w3c_raw = w3c[w3c["adjustment"] == sb.RAW]
    failing = w3c_raw[~w3c_raw["passes"]]
    lines += [
        f"""{len(w3c_raw) - len(failing)} of {len(w3c_raw)} raw cells meet the W3' bar"""
        + (" and none is over it." if failing.empty else ". The cells over it:"),
        "",
    ]
    if not failing.empty:
        lines += [table(W3C_HEADER, w3c_rows(failing)), ""]
    lines += [
        "### Detection (W4)",
        "",
        table(
            [
                "bed",
                "L",
                "adjustment",
                "delta 0.1",
                "delta 0.25",
                "delta 0.5",
                "smallest delta at 0.8",
            ],
            w4_table(corrected["base"]),
        ),
        "",
        f"![Detection by injected shift, fresh draws](out/{FIGURE_DETECTION})",
        "",
        "### Adjustment on the turbine pairs (W5)",
        "",
        table(
            ["L", "median width ratio adjusted/raw", "adjusted claim rate at delta 0 (MCSE)"],
            [
                [
                    rsw.block_label(rsw.BED_TURBINE, int(r["block"])),
                    num(r["ratio"]),
                    f"{num(r['rate'])} ({num(r['mcse'])})",
                ]
                for _, r in ctab["w5"].iterrows()
            ],
        ),
        "",
        "### Verdict under the corrected criteria",
        "",
        table(
            ["criterion", "result"],
            [
                ["W1' validity", verdict_word(corrected["w1"])],
                ["W2' coverage", verdict_word(corrected["w2"])],
                ["W3' mean bias", verdict_word(corrected["w3"])],
                [
                    "W4 practical power",
                    ", ".join(
                        f"{BED_LABELS[b]} {verdict_word(ok)}"
                        for b, ok in corrected["w4_beds"].items()
                    ),
                ],
                ["W5 adjustment", verdict_word(corrected["w5"])],
                ["verdict", "supported" if corrected["supported"] else "parked"],
            ],
        ),
        "",
    ]
    if corrected["supported"] and guidance:
        lines += [
            "Design guidance: 0.8 detection at 0.25 sigma was reached by "
            + "; ".join(guidance)
            + ".",
            "",
        ]
    lines += [
        "Wall time:",
        "",
        table(["bed", "rows", "first draws (s)", "fresh draws (s)"], wall_rows(data, conf)),
        "",
    ]
    lines += discussion(data, conf, verdict, corrected)
    return "\n".join(lines)


def wall_rows(data: dict, conf: dict) -> list[list[str]]:
    first, fresh = data["run"], conf["run"]
    rows = [
        [
            BED_LABELS[bed],
            str(first["beds"][bed]["n_rows"]),
            f"{first['beds'][bed]['wall_seconds']}",
            f"{fresh['beds'][bed]['wall_seconds']}",
        ]
        for bed in rsw.BEDS
    ]
    rows.append(
        ["all beds", "", f"{first['wall_seconds_all_beds']}", f"{fresh['wall_seconds_all_beds']}"]
    )
    return rows


def flaw_paragraphs(data: dict, verdict: dict, exact_tep: pd.DataFrame) -> list[str]:
    """Why the first registration fails an exact test, with the post hoc TEP enumeration."""
    tables = verdict["tables"]
    rand = tables["w1"][tables["w1"]["arm"] == sb.RANDOMIZATION]
    failing = rand[~rand["passes"]].sort_values("excess", ascending=False, kind="stable")
    w2 = tables["w2"]
    w2_raw = w2[w2["adjustment"] == sb.RAW]
    w3 = tables["w3"]
    w3_raw = w3[w3["adjustment"] == sb.RAW]
    w3_fail = w3_raw[~w3_raw["passes"]]
    lines = ["### Why the first registration was flawed", ""]
    worst_text = ""
    if len(failing):
        worst = failing.iloc[0]
        worst_text = (
            f" The one W1 miss is {num(worst['rate'], 4)} against {num(worst['bar'], 4)} "
            f"({BED_LABELS[worst['bed']]}, L {rsw.block_label(worst['bed'], int(worst['block']))}"
            f", tau {worst['tau']:g}, washout {int(worst['washout'])}), "
            f"{num((worst['rate'] - sb.ALPHA) / worst['mcse'], 1)} MCSE over 0.05, and the W2 "
            "misses sit in the same cells."
        )
    lines += [
        f"""W1 and W2 demanded every one of {len(rand)} and {len(w2_raw)} cells within 2 MCSE,
with no multiplicity control, so a test of exact size fails some cell by chance.{worst_text}""",
        "",
    ]
    if len(w3_fail):
        null = w3_fail["null_median"]
        lines += [
            f"""W3 read the median of the per-unit estimates. Randomization makes the mean
unbiased, and the median of a skewed estimate distribution is not 0 at delta 0: on 3W with
15 min blocks it sits at {num(null.max(), 4)} to {num(null.min(), 4)} sigma, and net of it the
bias left after the registered washout is at most {num(w3_fail["carryover"].abs().max(), 4)}
sigma.""",
            "",
        ]
    if not exact_tep.empty:
        r = exact_tep.iloc[0]
        lines += [
            f"""Post hoc: on the enumerated TEP design (L 4 h, K {int(r["k"])},
{int(r["n_assignments"])} assignments), no one of {int(r["n_units"])} (run, variable) units
rejects more than {num(r["max"], 4)} of its assignments on the raw arm, and the bound for a
test of size 0.05 on {int(r["n_assignments"])} assignments is {num(r["bound"], 4)}. The
misses above are draw fluctuations of a test whose size the design fixes. The first verdict
stands as registered.""",
            "",
        ]
    return lines


def discussion(data: dict, conf: dict, verdict: dict, corrected: dict) -> list[str]:
    summary = conf["summary"]
    ctab = corrected["tables"]
    k8 = cell(summary, rsw.BED_3W, 30, 0.0, 0, 0.0, sb.RANDOMIZATION, sb.ADJUSTED)
    k16 = cell(summary, rsw.BED_3W, 15, 0.0, 0, 0.0, sb.RANDOMIZATION, sb.ADJUSTED)
    tep_adj = cell(summary, rsw.BED_TEP, 40, 0.0, 0, W4_DELTA, sb.RANDOMIZATION, sb.ADJUSTED)
    tep_raw = cell(summary, rsw.BED_TEP, 40, 0.0, 0, W4_DELTA, sb.RANDOMIZATION, sb.RAW)
    w3c = ctab["w3"]
    quarter = w3c[
        (w3c["adjustment"] == sb.RAW)
        & (w3c["delta"] == rsw.DELTAS[-1])
        & (w3c["tau"] == w3c.groupby(["bed", "block"])["tau"].transform("max"))
    ]
    share = quarter["mean_none"].abs() / quarter["delta"]
    base = corrected["base"]
    held_raw = base["arms_hold"][(sb.BLOCK_T, sb.RAW)]
    held_adj = base["arms_hold"][(sb.BLOCK_T, sb.ADJUSTED)]
    block_t_text = {
        (True, True): "`block_t` meets the same per-cell bar raw and adjusted",
        (True, False): "`block_t` meets it on the raw arm and misses on the adjusted one",
        (False, True): "`block_t` misses on the raw arm and meets it on the adjusted one",
        (False, False): "`block_t` misses on both arms",
    }[(held_raw, held_adj)]
    pooled_lo, pooled_hi = ctab["w1"]["pooled"].min(), ctab["w1"]["pooled"].max()
    lines = [
        "## Discussion",
        "",
        f"""On fresh assignments the randomization test keeps its size on every bed: the
pooled claim rate at a zero injected shift is {pct(pooled_lo)} to {pct(pooled_hi)}, and on
the enumerated 3W design no (record, target) rejects more than
{num(corrected["w1_exact_max"], 4)} of its assignments. The first-half against second-half
split of the same 3W records claims a shift on
{pct(cell(summary, rsw.BED_3W, 15, 0.0, 0, 0.0, sb.PREPOST_HAC, sb.RAW)["rate"])} of tags.
The `hac`, `ewc` and `naive` intervals on the same schedules claim above the per-cell bar on
at least one bed, because their variance estimates miss the slow wander of the records;
{block_t_text}.""",
        "",
        f"""Without washout the lag moves the mean estimate by {pct(share.min(), 0)} to
{pct(share.max(), 0)} of delta at tau = L/4 and delta {rsw.DELTAS[-1]:g}, and the registered
washout removes most of it.""",
        "",
        f"""The price is length. With 16 blocks of 15 min on a 240-minute 3W record the raw
detection rate at 0.25 sigma is
{num(cell(summary, rsw.BED_3W, 15, 0.0, 0, W4_DELTA, sb.RANDOMIZATION, sb.RAW)["rate"])}, and
TEP runs of 48 hours reach {num(tep_raw["rate"])} raw and {num(tep_adj["rate"])} adjusted with
2-hour blocks. Only the turbine pairs, with more than 200 days of rows, reach 0.8 on the raw
arm. A lag of L/4 costs a further share of the detection rate, because its washout drops 3/4 of
every block.""",
        "",
        f"""Adjustment narrows the interval where the covariates carry the target: the turbine
ratio is {num(ctab["w5"]["ratio"].min())} to {num(ctab["w5"]["ratio"].max())}. With 8 blocks
the adjusted interval is unbounded on {pct(k8["unbounded_share"])} of 3W units, against
{pct(k16["unbounded_share"])} with 16 blocks. The inverted statistic is the unstudentized
coefficient, and with few assignments enough of them lie close to the covariates to accept any
shift.""",
        "",
        "## Limits and further work",
        "",
        """- The injected response is additive and follows one first-order lag; a real setting
change can alter the spread or interact with the operating point.""",
        """- The records carry no real switch. The study measures the analysis on the noise of
real records; the plant's response to a real schedule is outside it.""",
        """- The `hac` and `ewc` kernels treat the kept samples as one contiguous series, so they
skip the washout gaps and the turbine holes.""",
        """- MCSE clusters by (record, draw), or by (record, L, draw) when cells of several block
lengths pool. Rows of different tags in one draw share an assignment and are correlated.""",
        """- The corrected criteria were written after the first run. The confirmatory run reads
fresh assignments, so its numbers are new data for them.""",
        """- A plan-and-analyze command needs a SCOPE amendment for statements under a randomized
schedule before any code lands in `src/`.""",
        "",
        "## References",
        "",
        """- Bojinov, I., Simchi-Levi, D. and Zhao, J. (2023). Design and Analysis of Switchback
Experiments. Management Science 69(7), 3759-3777. doi:10.1287/mnsc.2022.4583""",
        """- Lin, W. (2013). Agnostic notes on regression adjustments to experimental data:
Reexamining Freedman's critique. The Annals of Applied Statistics 7(1), 295-318.
doi:10.1214/12-AOAS583""",
        "",
        "## Files",
        "",
        table(
            ["file", "what it holds"],
            [
                ["`switchback.py`", "schedule, response, washout, the inference arms"],
                ["`run_switchback.py`", "beds, grid, aggregation; `--beds`, `--draw-offset`"],
                ["`build_tep_cache.py`", "the TEP fault-free cache under `data/`"],
                ["`make_report.py`", "this report, the figures and the BENCHMARKS section"],
                [
                    "`results/summary.csv`",
                    "one row per (bed, L, tau, washout, delta, arm, adjustment)",
                ],
                ["`results/pooled.csv`", "claim rate and coverage pooled per bed"],
                ["`results/mean_bias.csv`", "mean bias of `randomization` per cell"],
                ["`results/exact_size.csv`", "exact rejection shares of enumerated designs"],
                ["`results/per_well_3w.csv`", "3W claim rates at delta 0 per well"],
                ["`results/turbine_pairs.csv`", "the turbine rows per pair"],
                ["`results/designs.csv`", "K, assignments, smallest p and refusal per block"],
                ["`results/refusals.csv`", "refused units per reason"],
                ["`results/run.json`", "provenance, parameters, counts, wall seconds"],
                ["`results/confirm/`", "the same files for the fresh draws"],
                ["`results/benchmarks_section.md`", "the BENCHMARKS.md REAL section"],
                ["`out/`", "the two figures, fresh draws"],
                ["`data/switchback_rows/` (untracked)", "one row per (record, target, draw, cell)"],
            ],
        ),
        "",
    ]
    return lines


# ---------------------------------------------------------------- benchmarks


def render_benchmarks_section(data: dict, conf: dict) -> str:
    summary, run = conf["summary"], conf["run"]["beds"]
    first = verdicts(data)
    corrected = corrected_verdicts(conf)
    tables = corrected["tables"]
    offset = conf["run"]["draw_offset"]

    def dataset(bed: str) -> str:
        sha = dataset_sha(conf, bed)
        names = {
            rsw.BED_3W: "3W v2.0.0 real",
            rsw.BED_TEP: "TEP (Rieth 2017 simulation, testing split)",
            rsw.BED_TURBINE: "Turbine Upgrade",
            rsw.BED_SKAB: "SKAB",
        }
        return f"{names[bed]} `{sha}`"

    def split(bed: str) -> str:
        section = run[bed]
        if bed == rsw.BED_3W:
            text = (
                "nothing fitted across records, so group holdout by well holds by "
                f"construction; {section['n_records']} instances with no fault window, "
                f"{section['n_groups']} wells, {section['draws']} draws each, label-blind; "
                f"cache manifest `{section['dataset']['manifest_sha256']}`"
            )
        elif bed == rsw.BED_TEP:
            text = (
                f"nothing fitted across runs; fault-free runs "
                f"{section['dataset']['runs'][0]}-{section['dataset']['runs'][1]}, "
                f"{section['draws']} draws each; cache sha256 "
                f"`{section['dataset']['cache_sha256']}`"
            )
        elif bed == rsw.BED_TURBINE:
            text = (
                "nothing fitted across records, so group holdout by pair holds by "
                f"construction; 2 pairs before the upgrade, {section['draws']} draws each"
            )
        else:
            text = f"one record, nothing fitted across records; {section['draws']} draws"
        return text + f"; fresh draws, seed from (bed, record, L, draw + {offset})"

    design = "switchback schedule, injected shift"
    rows = []
    w1c = tables["w1"]
    for bed in rsw.BEDS:
        code = code_label(run[bed])
        blocks = open_blocks(conf["designs"], bed)
        parts = []
        for adjustment in sb.ADJUSTMENTS:
            r = w1c[(w1c["bed"] == bed) & (w1c["adjustment"] == adjustment)].iloc[0]
            parts.append(
                f"{adjustment} pooled {num(r['pooled'])} ({num(r['pooled_mcse'])}), worst cell "
                f"{num(r['worst']['rate'])}"
            )
        p_hac = cell(summary, bed, blocks[0], 0.0, 0, 0.0, sb.PREPOST_HAC, sb.RAW)
        p_ewc = cell(summary, bed, blocks[0], 0.0, 0, 0.0, sb.PREPOST_EWC, sb.RAW)
        pp = (
            f"; first-half against second-half split of the same records: `hac` "
            f"{pct(p_hac['rate'])}, `ewc` {pct(p_ewc['rate'])} of "
            f"{int(p_hac['n_answered'])} (record, target) units"
        )
        rows.append(
            (
                "`randomization` claim rate at a zero injected shift over every open (L, "
                "washout) (MCSE)",
                "; ".join(parts) + pp,
                dataset(bed),
                code,
                design,
                split(bed),
            )
        )
        worst = []
        cells = w1_rows(conf)
        for arm in (sb.BLOCK_T, sb.HAC, sb.EWC, sb.NAIVE):
            sub = cells[
                (cells["bed"] == bed) & (cells["arm"] == arm) & (cells["adjustment"] == sb.RAW)
            ]
            worst.append(f"`{arm}` {num(sub['rate'].max())}")
        rows.append(
            (
                "highest claim rate at a zero injected shift over tau and washout, raw, other "
                "arms on the same schedules",
                ", ".join(worst),
                dataset(bed),
                code,
                design,
                split(bed),
            )
        )
        w2c = tables["w2"]
        cov = w2c[(w2c["bed"] == bed) & (w2c["adjustment"] == sb.RAW)].iloc[0]
        det = []
        for block in blocks:
            rates = detection(conf, bed, block, sb.RAW)
            det.append(
                f"L {rsw.block_label(bed, block)}: {num(rates[W4_DELTA])}, smallest delta at "
                f"0.8 {smallest_delta(rates)}"
            )
        rows.append(
            (
                "`randomization` raw: coverage of the exact truth pooled over delta > 0 at "
                "washout ceil(3 tau); detection at delta 0.25, tau 0",
                f"coverage {num(cov['pooled'], 4)} ({num(cov['pooled_mcse'], 4)}); "
                + "; ".join(det),
                dataset(bed),
                code,
                design,
                split(bed),
            )
        )
    w5 = tables["w5"]
    rows.append(
        (
            "`randomization` adjusted on the turbine covariates: median width ratio "
            "adjusted/raw, tau 0, washout 0",
            ", ".join(
                f"L {rsw.block_label(rsw.BED_TURBINE, int(r['block']))}: {num(r['ratio'])}"
                for _, r in w5.iterrows()
            ),
            dataset(rsw.BED_TURBINE),
            code_label(run[rsw.BED_TURBINE]),
            design,
            split(rsw.BED_TURBINE),
        )
    )
    word = {True: "pass", False: "fail"}

    def w4_text(verdict: dict) -> str:
        passing = [BED_LABELS[b] for b, ok in verdict["w4_beds"].items() if ok]
        return "passes on " + ", ".join(passing) if passing else "no bed passes"

    rows.append(
        (
            "registered criteria",
            f"first registration on the first draws: W1 {word[first['w1']]}, W2 "
            f"{word[first['w2']]}, W3 {word[first['w3']]}, W4 {word[first['w4']]} "
            f"({w4_text(first)}), W5 {word[first['w5']]}, "
            f"{'supported' if first['supported'] else 'parked'}; corrected criteria on fresh "
            f"draws: W1' {word[corrected['w1']]}, W2' {word[corrected['w2']]}, W3' "
            f"{word[corrected['w3']]}, W4 {word[corrected['w4']]} ({w4_text(corrected)}), W5 "
            f"{word[corrected['w5']]}, "
            f"{'supported' if corrected['supported'] else 'parked'}",
            "3W, TEP, Turbine Upgrade, SKAB",
            code_label(run[rsw.BED_3W]),
            design,
            "see the rows above; the corrected criteria were registered after the first run",
        )
    )
    lines = [
        SECTION_HEADING,
        "",
        "Randomized switchback schedules on records where nothing was changed, with a known "
        "shift injected into the B blocks through a first-order lag. The `randomization` arm "
        "is scored beside `block_t`, `hac`, `ewc` and `naive` intervals on the same schedules "
        "and beside the first-half against second-half split of the shift interval study. "
        "The rows read the confirmatory run on fresh assignments; the first registration and "
        "its verdict are in "
        "[examples/studies/switchback/REPORT.md](examples/studies/switchback/REPORT.md), with "
        "the method and refusals.",
        "",
        "This section is not regenerated by `make bench`; it is produced by "
        "`examples/studies/switchback/run_switchback.py` and `make_report.py`, and carried "
        "through the generator unchanged.",
        "",
        "| stage | metric | value | dataset (manifest sha256) | code | design | split |",
        "|---|---|---|---|---|---|---|",
    ]
    lines += [
        f"| none | {metric} | {value} | {ds} | {code} | {des} | {spl} |"
        for metric, value, ds, code, des, spl in rows
    ]
    return "\n".join(lines) + "\n"


def splice_section(text: str, section: str) -> str:
    """Replace this study's section in BENCHMARKS.md, or append it."""
    start = text.find(SECTION_HEADING)
    if start < 0:
        return text.rstrip("\n") + "\n\n" + section
    nxt = text.find("\n## ", start + len(SECTION_HEADING))
    if nxt < 0:
        return text[:start] + section
    # The blank line before the next heading belongs to the replaced text.
    return text[:start] + section + "\n" + text[nxt + 1 :]


# ---------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results", default=str(HERE / "results"))
    parser.add_argument("--confirm", default=str(HERE / "results" / "confirm"))
    parser.add_argument("--out", default=str(HERE / "out"))
    parser.add_argument("--report", default=str(HERE / "REPORT.md"))
    parser.add_argument("--update-benchmarks", action="store_true")
    parser.add_argument("--benchmarks", default=str(BENCH_MD))
    args = parser.parse_args(argv)

    results = Path(args.results)
    data = load(results)
    conf = load(Path(args.confirm))
    plot_claims(conf, Path(args.out) / FIGURE_CLAIMS)
    plot_detection(conf, Path(args.out) / FIGURE_DETECTION)
    Path(args.report).write_text(unwrap(render_report(data, conf)), encoding="utf-8", newline="\n")
    section = render_benchmarks_section(data, conf)
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

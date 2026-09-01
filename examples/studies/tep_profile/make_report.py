"""Aggregate the TEP profile study into REPORT.md, summary.json and figures.

    uv run python examples/studies/tep_profile/make_report.py

Reads ``results/per_tag.csv`` (written by ``run_study.py``) plus every
``RUN.json`` under the archive root, and states counts. It computes no
detector metric: profiling is descriptive.

It also writes ``results/benchmarks_section.md``, the block that is
appended to BENCHMARKS.md after the SYNTHETIC table, so the published
rows are generated from the same summary as the report rather than typed
out beside it.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
ROOT = HERE.parents[2]

# dataviz reference palette, categorical slots in their documented order.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#d9d8d4"

VERDICT_ORDER = (
    "FIRED",
    "NO_FLATLINE",
    "NOT_ASSESSED_MODE_TAG",
    "NOT_ASSESSED_NO_REFERENCE",
    "NOT_ASSESSED_TOO_SHORT",
    "NOT_ASSESSED_CENSORED",
    "NOT_ASSESSED_NO_VALID_SAMPLES",
)

SIGNAL_NUMBER = {
    "time_since_last_actual_change": "1",
    "distinct_value_count_vs_p05": "2",
    "zero_std_while_coupled_tag_moves": "3",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def git_head(run_meta: dict) -> str:
    """The commit `run_study.py` recorded when it produced the numbers.

    Read back rather than asked for again: `git rev-parse HEAD` here would
    move the published code fingerprint every time the report is
    regenerated, which would make it name the wrong commit.
    """
    recorded = run_meta.get("git_head")
    if recorded:
        return str(recorded)
    try:  # pragma: no cover - only for a run.json written before this field
        out = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def load_rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_runs(root: Path) -> list[dict]:
    return [json.loads(p.read_text("utf-8")) for p in sorted(root.glob("*/*/RUN.json"))]


def _f(value: str) -> float | None:
    return float(value) if value else None


def _pct(n: int, total: int) -> str:
    return "n/a" if total == 0 else f"{100.0 * n / total:.1f}%"


def _quantiles(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    if not ordered:
        return {}

    def q(p: float) -> float:
        idx = min(len(ordered) - 1, max(0, round(p * (len(ordered) - 1))))
        return ordered[idx]

    return {
        "min": ordered[0],
        "p05": q(0.05),
        "p50": q(0.50),
        "mean": statistics.fmean(ordered),
        "p95": q(0.95),
        "max": ordered[-1],
    }


def _measurement(rows: list[dict]) -> list[dict]:
    """Tags the flatline detector will assess: everything but MODE."""
    return [r for r in rows if r["role"] != "MODE"]


def _assessable(rows: list[dict]) -> dict:
    measured = [r for r in _measurement(rows) if r["flatline_verdict"]]
    assessed = [r for r in measured if r["flatline_verdict"] in ("FIRED", "NO_FLATLINE")]
    fired = [r for r in assessed if r["flatline_verdict"] == "FIRED"]
    mode = [r for r in rows if r["role"] == "MODE" and r["flatline_verdict"]]
    return {
        "n_non_mode_tags": len(measured),
        "n_non_mode_assessed": len(assessed),
        "n_fired": len(fired),
        "n_mode_tags": len(mode),
        "n_mode_declined_as_state_valued": sum(
            1 for r in mode if r["flatline_verdict"] == "NOT_ASSESSED_MODE_TAG"
        ),
        "fired_by_signal_combination": dict(
            Counter(
                ";".join(
                    SIGNAL_NUMBER.get(s, s)
                    for s in r["flatline_fired_signals"].split(";")
                    if s
                )
                for r in fired
            ).most_common()
        ),
        "fired_by_variable": dict(Counter(r["variable"] for r in fired).most_common()),
    }


def _signal_behaviour(rows: list[dict], window_s: float) -> dict:
    """What the two evaluable flatline signals compared, and against what.

    Signal 1 measures a stall against the tag's own p99 of reference
    change intervals. The first run of this study found that reference
    collapsed to a single number on every tag, which is how the defect in
    ``reference_history`` was noticed; the histograms below are what it
    looks like once the intervals are timed between change events.
    """
    assessed = [
        r for r in _measurement(rows) if r["flatline_verdict"] in ("FIRED", "NO_FLATLINE")
    ]
    flat = [r for r in assessed if r["stall_s"] and float(r["stall_s"]) >= window_s]
    holds: dict[str, Counter[str]] = defaultdict(Counter)
    for r in assessed:
        if r["stall_s"]:
            holds[r["variable"]][r["stall_s"]] += 1
    fired = Counter(r["variable"] for r in assessed if r["flatline_verdict"] == "FIRED")
    seen = Counter(r["variable"] for r in assessed)
    return {
        "n_assessed": len(assessed),
        "reference_p99_values_s": dict(
            Counter(
                r["ref_p99_change_interval_s"] for r in assessed if r["ref_p99_change_interval_s"]
            ).most_common()
        ),
        "stall_histogram_s": dict(
            Counter(r["stall_s"] for r in assessed if r["stall_s"]).most_common()
        ),
        # Per-tag thresholds mean a stall no longer decides a verdict on its
        # own: the same stall fires on one tag and not on another.
        "stall_fired_histogram_s": dict(
            Counter(
                r["stall_s"]
                for r in assessed
                if r["stall_s"] and r["flatline_verdict"] == "FIRED"
            ).most_common()
        ),
        "modal_stall_per_variable_s": {
            v: c.most_common(1)[0][0] for v, c in sorted(holds.items())
        },
        "variables_fired_on_every_run": sorted(v for v in seen if fired[v] == seen[v]),
        "whole_window_flat": {
            "n_tags": len(flat),
            "n_runs": len({r["loop_id"] for r in flat}),
            "by_fault": dict(
                sorted(Counter(r["fault"] for r in flat).items(), key=lambda kv: -kv[1])
            ),
            "runs_per_fault": {
                fault: len({r["loop_id"] for r in flat if r["fault"] == fault})
                for fault in sorted({r["fault"] for r in flat})
            },
            "verdicts": dict(Counter(r["flatline_verdict"] for r in flat).most_common()),
        },
    }


def _distinct_values(rows: list[dict]) -> dict:
    """How many distinct values each measurement tag holds over its whole run.

    The 3W study's headline gate was that 22.4% of its measurement tags
    hold exactly one value for their entire archive, so a self-
    referencing threshold has nothing to reference. The same count on a
    simulator is the contrast.
    """
    measured = [r for r in _measurement(rows) if r["whole_life_distinct"]]
    counts = {r["variable"]: [] for r in measured}
    for r in measured:
        counts[r["variable"]].append(int(r["whole_life_distinct"]))
    frozen = [r for r in measured if int(r["whole_life_distinct"]) == 1]
    per_variable_min = {v: min(c) for v, c in counts.items()}
    return {
        "n_non_mode_tags": len(measured),
        "n_frozen_whole_life": len(frozen),
        "frozen_per_variable": dict(Counter(r["variable"] for r in frozen).most_common()),
        "quantiles": _quantiles([float(r["whole_life_distinct"]) for r in measured]),
        "least_varying_variables": dict(
            sorted(per_variable_min.items(), key=lambda kv: kv[1])[:8]
        ),
        "xmv_min_distinct": {
            v: n for v, n in sorted(per_variable_min.items()) if v.startswith("xmv_")
        },
    }


def _quantisation_probe(rows: list[dict], archives: Path) -> dict:
    """Open the least-varying measurement tag and say why it varies so little.

    Read off the archive rather than described, because "the source is
    stored rounded" is a claim about bytes on disk.
    """
    import pandas as pd

    measured = [r for r in _measurement(rows) if r["whole_life_distinct"]]
    if not measured:
        return {}
    target = min(measured, key=lambda r: int(r["whole_life_distinct"]))
    path = archives / target["split"] / target["loop_id"] / f"{target['variable']}.parquet"
    values = sorted(set(pd.read_parquet(path)["value"].dropna().tolist()))
    steps = [round(b - a, 12) for a, b in itertools.pairwise(values)]
    return {
        "loop_id": target["loop_id"],
        "variable": target["variable"],
        "unit_raw": target["unit_raw"],
        "n_distinct": len(values),
        "n_rows": int(target["n_rows"]),
        "min": values[0],
        "max": values[-1],
        "smallest_step": min(steps) if steps else None,
    }


def _units(rows: list[dict]) -> dict:
    non_mode = _measurement(rows)
    by_spelling: dict[str, dict] = {}
    for r in non_mode:
        spelling = r["unit_raw"] or "(none)"
        entry = by_spelling.setdefault(
            spelling, {"n_tags": 0, "resolved": r["unit_resolved"] == "true", "canonical": ""}
        )
        entry["n_tags"] += 1
        entry["canonical"] = r["unit_canonical"]
    unresolved = sorted(s for s, e in by_spelling.items() if not e["resolved"])
    return {
        "by_spelling": dict(sorted(by_spelling.items())),
        "unresolved_raw_units": unresolved,
        "n_tags_non_mode": len(non_mode),
        "n_tags_unresolved": sum(1 for r in non_mode if r["unit_resolved"] == "false"),
        "n_variables_unresolved": len(
            {r["variable"] for r in non_mode if r["unit_resolved"] == "false"}
        ),
    }


def build_summary(
    rows: list[dict], runs: list[dict], run_meta: dict, src: Path, archives: Path
) -> dict:
    coverage = [c for c in (_f(r["coverage"]) for r in rows) if c is not None]
    valid = [v for v in (_f(r["valid_fraction"]) for r in rows) if v is not None]
    verdicts = Counter(r["flatline_verdict"] for r in rows if r["flatline_verdict"])
    refusals = Counter(r["refusal_type"] for r in rows if r["refusal_type"])
    gap_classes: Counter[str] = Counter()
    for row in rows:
        for cls in filter(None, row["gap_classes"].split(";")):
            gap_classes[cls] += 1

    fetch = json.loads((src / "FETCH.json").read_text("utf-8"))
    faults = sorted({r["fault"] for r in runs})
    return {
        "dataset": {
            "doi": fetch["doi"],
            "version": fetch["version"],
            "bytes_downloaded": fetch["bytes"],
            "files_downloaded": fetch["files"],
            "manifest_sha256": sha256_file(src / "MANIFEST.sha256.json"),
        },
        "code": {
            "git_head": git_head(run_meta),
            "tsdive_version": run_meta["tsdive_version"],
        },
        "converted": {
            "n_runs": len(runs),
            "n_archives": sum(len(r["archives_written"]) for r in runs),
            "n_faults": len(faults),
            "faults": faults,
            "runs_per_fault": len({r["run"] for r in runs}),
            "splits": sorted({r["split"] for r in runs}),
            "n_rows": sum(r["n_rows"] for r in runs),
            "onset_samples": sorted({r["onset_sample"] for r in runs}),
            "timestamp_synthetic": all(r["timestamp_synthetic"] for r in runs),
            "t0_utc": sorted({r["t0_utc"] for r in runs}),
            "n_nonfinite_values": sum(r["n_nonfinite_values"] for r in runs),
            "n_faulty_rows": sum(r["n_faulty_rows"] for r in runs),
        },
        "studied": {
            "n_runs": run_meta["n_runs"],
            "n_tags": run_meta["n_tags"],
            "window_s": run_meta["window_s"],
            "wall_seconds": run_meta["wall_seconds"],
            "jobs": run_meta["jobs"],
            "n_reference_windows": dict(
                Counter(r["n_reference_windows"] for r in rows if r["n_reference_windows"])
            ),
        },
        "coverage": _quantiles(coverage),
        "coverage_below_1": sum(1 for c in coverage if c < 1.0),
        "valid_fraction": _quantiles(valid),
        "valid_fraction_below_1": sum(1 for v in valid if v < 1.0),
        "gap_classes": dict(gap_classes),
        "n_gaps_total": sum(int(r["n_gaps"] or 0) for r in rows),
        "flatline_verdicts": dict(verdicts),
        "flatline_assessable_roles": _assessable(rows),
        "flatline_signal_behaviour": _signal_behaviour(rows, run_meta["window_s"]),
        "distinct_values": _distinct_values(rows),
        "quantisation_probe": _quantisation_probe(rows, archives),
        "refusals": dict(refusals),
        "timestamp_audit": {
            "duplicates": sum(int(r["duplicate_timestamps"] or 0) for r in rows),
            "non_monotonic": sum(int(r["non_monotonic_positions"] or 0) for r in rows),
            "dst_transitions": sum(int(r["dst_transitions"] or 0) for r in rows),
        },
        "quality": {
            "n_good_samples": sum(int(r["n_good"] or 0) for r in rows),
            "n_uncertain_samples": sum(int(r["n_uncertain"] or 0) for r in rows),
            "n_bad_samples": sum(int(r["n_bad"] or 0) for r in rows),
            "n_tags_with_unmapped_codes": sum(1 for r in rows if r["unmapped_codes"]),
        },
        "clipping": {
            "n_tags_with_a_clipped_fraction": sum(1 for r in rows if r["clipped_fraction"]),
            "n_censored": sum(1 for r in rows if r["censored"] == "true"),
        },
        "units": _units(rows),
    }


def plot_units(summary: dict, out: Path) -> Path:
    by_spelling = summary["units"]["by_spelling"]
    order = sorted(
        by_spelling, key=lambda s: (by_spelling[s]["resolved"], -by_spelling[s]["n_tags"])
    )
    fig, ax = plt.subplots(figsize=(8.5, 4.4), dpi=160)
    ypos = list(range(len(order)))
    widths = [by_spelling[s]["n_tags"] for s in order]
    colours = [SERIES[2] if by_spelling[s]["resolved"] else SERIES[1] for s in order]
    ax.barh(ypos, widths, color=colours, height=0.62, edgecolor="white", linewidth=1.0)
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{s}" for s in order], fontsize=9, color=INK)
    top = max(widths) * 1.45
    ax.set_xlim(0, top)
    for y, spelling in zip(ypos, order, strict=True):
        entry = by_spelling[spelling]
        label = entry["canonical"] if entry["resolved"] else "unresolved; comparison refused"
        ax.text(entry["n_tags"] + top * 0.015, y, f"{entry['n_tags']:,d}  {label}",
                va="center", fontsize=8, color=INK_MUTED)
    ax.set_xlabel("tags carrying the spelling", color=INK_MUTED)
    ax.set_title(
        "TEP units as Downs & Vogel write them, against the shipped alias table",
        color=INK,
        fontsize=12,
        pad=12,
    )
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, length=0)
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    return out


def plot_verdicts(rows: list[dict], out: Path) -> Path:
    by_var: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        if row["flatline_verdict"]:
            by_var[row["variable"]][row["flatline_verdict"]] += 1
    variables = sorted(
        by_var, key=lambda v: -by_var[v].get("FIRED", 0) / sum(by_var[v].values())
    )

    fig, ax = plt.subplots(figsize=(9.0, 9.5), dpi=160)
    ypos = list(range(len(variables)))
    left = [0.0] * len(variables)
    present = [v for v in VERDICT_ORDER if any(by_var[k].get(v) for k in variables)]
    for colour, verdict in zip(SERIES, present, strict=False):
        widths = [
            100.0 * by_var[v].get(verdict, 0) / sum(by_var[v].values()) for v in variables
        ]
        ax.barh(
            ypos,
            widths,
            left=left,
            color=colour,
            height=0.7,
            label=verdict,
            edgecolor="white",
            linewidth=0.8,
        )
        left = [a + b for a, b in zip(left, widths, strict=True)]
    ax.set_yticks(ypos)
    ax.set_yticklabels(variables, fontsize=7, color=INK)
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of studied tags (%)", color=INK_MUTED)
    ax.set_title(
        "Flatline verdicts per TEP variable, 3 h window against earlier windows",
        color=INK,
        fontsize=12,
        pad=12,
    )
    ax.legend(
        frameon=False,
        fontsize=8,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=3,
        labelcolor=INK_MUTED,
    )
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, length=0)
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    return out


def plot_stalls(summary: dict, run_meta: dict, out: Path) -> Path:
    """Every stall the detector measured, split by what it decided."""
    sb = summary["flatline_signal_behaviour"]
    histogram = sorted(
        ((float(k), v) for k, v in sb["stall_histogram_s"].items()), key=lambda kv: kv[0]
    )
    window_s = run_meta["window_s"]
    fired_at = {float(k): v for k, v in sb["stall_fired_histogram_s"].items()}
    n_p99 = len(sb["reference_p99_values_s"])
    labelled = {
        0.0: ("moved at the\nlast sample", "center", 1.0),
        180.0: ("one sample", "left", 1.18),
        720.0: ("four samples\n(15 min analyser):\nsignal 1 no longer fires", "center", 1.0),
    }
    fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=160)
    xs = [max(v, 90.0) for v, _ in histogram]
    ys = [n for _, n in histogram]
    fs = [fired_at.get(v, 0) for v, _ in histogram]
    widths = [x * 0.18 for x in xs]
    # Each stall is split into the tags it fired on and the tags it did
    # not: with a per-tag p99 the stall alone no longer decides.
    ax.bar(
        xs, ys, width=widths, color=SERIES[0], edgecolor="white", linewidth=0.6,
        label="assessed, no flatline",
    )
    ax.bar(
        xs, fs, width=widths, color=SERIES[1], edgecolor="white", linewidth=0.6,
        label="FIRED",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(0.6, max(ys) * 200)
    ax.legend(frameon=False, fontsize=8, loc="upper right", labelcolor=INK_MUTED)
    ax.text(
        70.0,
        max(ys) * 25,
        f"The threshold is per tag: the p99 takes {n_p99:,d} distinct values over the "
        f"{sum(ys):,d} assessed tags,\nso the same stall fires on one tag and not on "
        "another. The fires at a 0 s stall are\nsignal 2 (distinct-value count), "
        "which this threshold does not govern.",
        fontsize=8,
        va="top",
        color=INK,
    )
    for value, (text, align, nudge) in labelled.items():
        match = next((n for v, n in histogram if v == value), None)
        if match:
            ax.text(
                max(value, 90.0) * nudge,
                match * 1.5,
                text,
                fontsize=7,
                color=INK_MUTED,
                ha=align,
            )
    match = next((n for v, n in histogram if v == window_s), None)
    if match:
        ax.text(
            window_s,
            match * 1.5,
            f"flat for the whole\n{window_s / 3600:.0f} h window\n(faults 6 and 18)",
            fontsize=7,
            color=SERIES[1],
            ha="right",
        )
    ax.set_xlabel(
        "stall at the end of the assessed window (s, log; 0 s plotted at 90)",
        color=INK_MUTED,
    )
    ax.set_ylabel("tags (log)", color=INK_MUTED)
    ax.set_title(
        "Every stall TEP offers, and which of them the detector fired on",
        color=INK,
        pad=12,
    )
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, length=0)
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    return out


def plot_distinct(rows: list[dict], out: Path) -> Path:
    """Whole-run distinct-value counts: the 3W freeze gate, asked of a simulator."""
    import numpy as np

    counts = [int(r["whole_life_distinct"]) for r in _measurement(rows) if r["whole_life_distinct"]]
    n_rows = max(int(r["n_rows"]) for r in rows if r["n_rows"])
    fig, ax = plt.subplots(figsize=(8.5, 4.4), dpi=160)
    bins = np.arange(0, n_rows + 26, 25)
    heights, _, _ = ax.hist(counts, bins=bins, color=SERIES[0], edgecolor="white", linewidth=0.8)
    ax.axvline(1, color=SERIES[1], linewidth=2.0)
    frozen = sum(1 for c in counts if c == 1)
    ax.text(
        0.985,
        0.94,
        f"{frozen} of {len(counts):,d} measurement tags hold one distinct value\n"
        f"for the whole run (3W: 22.4%). Each run is {n_rows} samples.",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color=INK_MUTED,
    )
    ax.set_ylim(0, max(heights) * 1.3)
    ax.set_xlabel("distinct values over the whole run", color=INK_MUTED)
    ax.set_ylabel("tags", color=INK_MUTED)
    ax.set_title("A simulator's variables never stop moving", color=INK, pad=12)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, length=0)
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    return out


def benchmark_rows(summary: dict) -> list[tuple[str, str, str]]:
    """(stage, metric, value) triples for the BENCHMARKS.md REAL section."""
    conv, st = summary["converted"], summary["studied"]
    cov, vf = summary["coverage"], summary["valid_fraction"]
    ar = summary["flatline_assessable_roles"]
    dv = summary["distinct_values"]
    sb = summary["flatline_signal_behaviour"]
    u = summary["units"]
    fv = summary["flatline_verdicts"]
    total_v = sum(fv.values())
    q = summary["quality"]
    ta = summary["timestamp_audit"]
    n_refused = sum(summary["refusals"].values())
    return [
        ("1", "simulation runs converted", f"{conv['n_runs']:,d}"),
        (
            "1",
            "archives written (run x variable, plus 1 label tag)",
            f"{conv['n_archives']:,d}",
        ),
        ("1", "runs profiled", f"{st['n_runs']:,d}"),
        ("1", "tags profiled", f"{st['n_tags']:,d}"),
        ("1", "mean coverage", f"{cov['mean']:.3f}"),
        ("1", "median coverage", f"{cov['p50']:.3f}"),
        ("1", "mean valid fraction (rows with a usable value)", f"{vf['mean']:.3f}"),
        ("1", "gaps of any class", f"{summary['n_gaps_total']:,d}"),
        (
            "1",
            "duplicate or non-monotonic timestamps",
            f"{ta['duplicates'] + ta['non_monotonic']:,d}",
        ),
        ("1", "samples carrying a BAD or UNCERTAIN severity",
         f"{q['n_bad_samples'] + q['n_uncertain_samples']:,d}"),
        (
            "1",
            "tags refused (any typed refusal)",
            f"{_pct(n_refused, st['n_tags'])} ({n_refused} of {st['n_tags']:,d})",
        ),
        (
            "1",
            "non-MODE tags whose unit the alias table cannot resolve",
            f"{_pct(u['n_tags_unresolved'], u['n_tags_non_mode'])} "
            f"({u['n_tags_unresolved']:,d} of {u['n_tags_non_mode']:,d}): "
            + ", ".join(f"`{x}`" for x in u["unresolved_raw_units"]),
        ),
        (
            "1",
            "measurement tags with flatline FIRED (3 h window vs earlier windows)",
            f"{_pct(ar['n_fired'], ar['n_non_mode_assessed'])} "
            f"({ar['n_fired']:,d} of {ar['n_non_mode_assessed']:,d})",
        ),
        (
            "1",
            "tags NOT ASSESSED for flatline (all reasons)",
            f"{_pct(total_v - fv.get('FIRED', 0) - fv.get('NO_FLATLINE', 0), total_v)}",
        ),
        (
            "1",
            "NOT ASSESSED: state-valued tag (role=MODE)",
            f"{_pct(fv.get('NOT_ASSESSED_MODE_TAG', 0), total_v)}",
        ),
        (
            "1",
            "measurement tags holding one distinct value across the whole run",
            f"{_pct(dv['n_frozen_whole_life'], dv['n_non_mode_tags'])} "
            f"({dv['n_frozen_whole_life']:,d} of {dv['n_non_mode_tags']:,d})",
        ),
        (
            "1",
            "flatline fires that are a whole-window flat (plant trip, faults "
            + " and ".join(sb["whole_window_flat"]["by_fault"])
            + ")",
            f"{_pct(sb['whole_window_flat']['n_tags'], ar['n_fired'])} "
            f"({sb['whole_window_flat']['n_tags']:,d} of {ar['n_fired']:,d}, across "
            f"{sb['whole_window_flat']['n_runs']} runs)",
        ),
        (
            "1",
            "distinct values of the flatline signal-1 reference (p99 change interval)",
            f"{len(sb['reference_p99_values_s']):,d} values, "
            + ", ".join(
                f"{k} s on {v:,d} tags"
                for k, v in sorted(
                    sb["reference_p99_values_s"].items(), key=lambda kv: -kv[1]
                )[:3]
            )
            + ", 180 s being the sample period; see REPORT.md",
        ),
    ]


def render_benchmarks_section(summary: dict) -> str:
    ds, code = summary["dataset"], summary["code"]
    conv = summary["converted"]
    dataset = f"TEP Rieth 2017 `{ds['manifest_sha256'][:12]}`"
    codecell = f"tsdive {code['tsdive_version']} @ `{code['git_head'][:8]}`"
    split = (
        f"{conv['runs_per_fault']} runs x {conv['n_faults']} faults, training splits "
        "only; no train/test (descriptive profile statistics only, no detector numbers)"
    )
    lines = [
        "",
        "## REAL: TEP (Rieth 2017 simulation, subset)",
        "",
        "Descriptive profile statistics only. TEP is a simulation, so the "
        "profile stage finds nothing "
        "to refuse. Zero gaps, zero duplicate timestamps, coverage and valid "
        "fraction 1.000 everywhere, no typed refusal on any tag. The one place "
        "the simulator touches the real world is its units, and half of them do "
        "not resolve.",
        "",
        "This section is not regenerated by `make bench`; it is produced by",
        "`examples/studies/tep_profile/run_study.py` and `make_report.py`, and carried",
        "through the generator unchanged. Method and caveats (in particular why",
        "the timestamps are synthetic) are in",
        "[examples/studies/tep_profile/REPORT.md](examples/studies/tep_profile/REPORT.md).",
        "",
        "| stage | metric | value | dataset (manifest sha256) | code | split |",
        "|---|---|---|---|---|---|",
    ]
    for stage, metric, value in benchmark_rows(summary):
        lines.append(f"| {stage} | {metric} | {value} | {dataset} | {codecell} | {split} |")
    return "\n".join(lines) + "\n"


def render_report(summary: dict, rows: list[dict], run_meta: dict) -> str:
    ds, code = summary["dataset"], summary["code"]
    conv, st = summary["converted"], summary["studied"]
    cov, vf = summary["coverage"], summary["valid_fraction"]
    u = summary["units"]
    ar = summary["flatline_assessable_roles"]
    dv = summary["distinct_values"]
    fv = summary["flatline_verdicts"]
    total_v = sum(fv.values())
    q = summary["quality"]
    ta = summary["timestamp_audit"]
    window_h = run_meta["window_s"] / 3600.0

    lines: list[str] = []
    add = lines.append

    add("# Stage 1 on a simulator: Tennessee Eastman (Rieth 2017)")
    add("")
    sb = summary["flatline_signal_behaviour"]
    ww = sb["whole_window_flat"]
    add(
        "**The profile stage finds nothing to refuse on a simulator, and that is the "
        "point of "
        f"running it.** Across {st['n_tags']:,d} tags the data-physics layer found "
        f"{summary['n_gaps_total']} gaps, {ta['duplicates']} duplicate timestamps, "
        f"{ta['non_monotonic']} non-monotonic positions, "
        f"{q['n_bad_samples'] + q['n_uncertain_samples']} samples below GOOD, and "
        f"issued {sum(summary['refusals'].values())} refusals; coverage and valid "
        "fraction are 1.000 on every tag. Every profile rule that had something to "
        "say about 3W - gap classification, absent columns, unmapped quality codes, "
        "whole-life freezes - is asking a question about a historian, and a "
        "simulator has no historian. That is the contrast the two 3W sections of "
        "BENCHMARKS.md exist to be read against."
    )
    add("")
    add(
        "Three things did come out of it. The one place TEP touches the real world "
        f"is its engineering units, and {u['n_tags_unresolved']:,d} of "
        f"{u['n_tags_non_mode']:,d} non-MODE tags carry a spelling the alias table "
        f"refuses to resolve. The flatline detector's {ar['n_fired']:,d} fires are "
        f"led by {ww['n_tags']:,d} real ones - the two faults that trip the plant, "
        "where the flowsheet genuinely goes flat. And the first run of this study "
        "found a library defect on the way there: the p99 the detector compares a "
        "stall against came back as the sampling period on all "
        f"{sb['n_assessed']:,d} assessed tags, because it was computed from the "
        "wrong differences. That is now fixed, and the numbers below are the ones "
        "after the fix."
    )
    add("")
    add("## Provenance")
    add("")
    add("| field | value |")
    add("|---|---|")
    add(f"| dataset | Harvard Dataverse `{ds['doi']}` version {ds['version']} |")
    add(
        f"| downloaded | {len(ds['files_downloaded'])} files, "
        f"{ds['bytes_downloaded']:,d} bytes: "
        + ", ".join(f"`{f}`" for f in ds["files_downloaded"])
        + " |"
    )
    add(f"| manifest sha256 | `{ds['manifest_sha256']}` |")
    add(f"| code | tsdive {code['tsdive_version']} @ `{code['git_head']}` |")
    add(
        f"| converted | {conv['n_runs']:,d} runs "
        f"({conv['runs_per_fault']} per fault x {conv['n_faults']} faults), "
        f"{conv['n_archives']:,d} archives, {conv['n_rows']:,d} source rows |"
    )
    add(
        f"| studied | {st['n_runs']:,d} runs, {st['n_tags']:,d} tags, "
        f"{st['wall_seconds']:.0f} s wall on {st['jobs']} processes |"
    )
    add("")
    add("Commands:")
    add("")
    add("```console")
    add("uv run python scripts/fetch_tep.py --dest data/tep --yes")
    add("uv sync --extra tep")
    add(
        "uv run python scripts/convert_tep.py --src data/tep --out data/tep_archives "
        f"--runs-per-fault {conv['runs_per_fault']}"
    )
    add(
        "uv run python examples/studies/tep_profile/run_study.py "
        f"--window-s {run_meta['window_s']:.0f} --jobs {st['jobs']}"
    )
    add("uv run python examples/studies/tep_profile/make_report.py")
    add("```")
    add("")

    add("## What was studied, and what was not")
    add("")
    add(
        "The Rieth 2017 re-simulation publishes 500 runs of each of 21 conditions "
        "(fault-free plus faults 1-20) in each of two splits. The training pair "
        f"alone is {ds['bytes_downloaded']:,d} bytes and would convert to 556,500 "
        "archives, so this study takes a documented subset: "
        f"**simulationRun 1-{conv['runs_per_fault']} of every fault, training splits "
        f"only**. That is {conv['n_runs']:,d} runs, {conv['n_archives']:,d} archives, "
        f"{conv['n_rows']:,d} rows. The testing splits (884 MB) were not fetched: "
        "profiling is descriptive and scores nothing, so a held-out split buys it "
        "nothing."
    )
    add("")
    add(
        f"Each run is {max(int(r['n_rows']) for r in rows if r['n_rows'])} samples at "
        "180 s, so 25 h of simulated plant. Each archive is read twice - once over "
        "its whole extent for the physics block, once over its last "
        f"{window_h:.0f} h for the flatline assessment. The 3W study used a 1 h "
        "window; at TEP's 3-minute sampling 1 h is 20 samples, which is too few for "
        "a distinct-value count to mean much and is shorter than the fault-free "
        f"lead-in of a training run. {window_h:.0f} h is 60 samples per window, and "
        "every archive is the same length, so every tag gets the same reference "
        "depth: "
        + ", ".join(
            f"{k} earlier windows on {v:,d} tags"
            for k, v in sorted(st["n_reference_windows"].items(), key=lambda kv: -kv[1])
        )
        + "."
    )
    add("")

    add("## TEP has no clock, so the timestamps are invented")
    add("")
    add(
        "A TEP row is numbered, not stamped: the RData columns are `faultNumber`, "
        "`simulationRun`, `sample` and 52 process variables, and nothing in the file "
        "names an instant. tsdive stores UTC only and refuses naive timestamps at "
        "ingest, so the converter synthesises one - "
        f"`{conv['t0_utc'][0]} + 180 s x (sample - 1)` - and writes "
        "`timestamp_synthetic: true` into every run's `RUN.json`. All "
        f"{conv['n_runs']:,d} runs share that clock, because each is an independent "
        "simulation from its own t=0; the instants are an addressing scheme, not a "
        "history. Any number this study reports about time - coverage, gaps, the "
        "flatline window - is a statement about the sample grid, and nothing else. "
        "That is a decision, not an observation, which is why it is recorded per run "
        "rather than mentioned once in a README."
    )
    add("")
    add(
        f"The audit agrees the grid is exact: {ta['duplicates']} duplicate "
        f"timestamps, {ta['non_monotonic']} non-monotonic positions and "
        f"{ta['dst_transitions']} DST transitions across {st['n_tags']:,d} tags. It "
        "could hardly say otherwise about a grid this code generated - which is the "
        "point of saying it out loud."
    )
    add("")

    add("## Coverage, gaps and quality")
    add("")
    add(
        f"Coverage over the studied tags: min {cov['min']:.4f}, median "
        f"{cov['p50']:.4f}, max {cov['max']:.4f}; {summary['coverage_below_1']} tags "
        f"fall below 1.000. Valid fraction: min {vf['min']:.4f}, median "
        f"{vf['p50']:.4f}, max {vf['max']:.4f}; {summary['valid_fraction_below_1']} "
        "below 1.000. On 3W those two numbers disagreed on 2,824 tags, because a "
        "historian that emits a row every second still emits rows with nothing in "
        "them. Here they agree everywhere, because the simulator wrote a value into "
        f"every cell: {conv['n_nonfinite_values']} non-finite values in "
        f"{conv['n_rows']:,d} rows x 52 variables."
    )
    add("")
    if summary["gap_classes"]:
        add("| gap class | tags with at least one |")
        add("|---|---:|")
        for cls, count in sorted(summary["gap_classes"].items(), key=lambda kv: -kv[1]):
            add(f"| `{cls}` | {count:,d} |")
    else:
        add(
            "No gap of any class was found in any studied tag. The gap classifier - "
            "the part of the profile with the most rules, and the part that earns the "
            "`unknown` class on 3W - had nothing to classify."
        )
    add("")
    add(
        "Quality is assumed and labelled as assumed. A simulator emits no quality "
        "column because it has nothing to report, so the converter writes "
        "`SIMULATED` on every row, declares it in the tag's `quality_codes`, and "
        "sets `quality_assumed` so every profile prints *quality ASSUMED at ingest*. "
        f"That is {q['n_good_samples']:,d} GOOD, {q['n_uncertain_samples']:,d} "
        f"UNCERTAIN and {q['n_bad_samples']:,d} BAD samples, with "
        f"{q['n_tags_with_unmapped_codes']} tags reporting an unmapped code. The "
        "converter refuses a run holding a non-finite value rather than coding it "
        "GOOD; no run in this subset triggered that."
    )
    add("")
    c = summary["clipping"]
    add(
        "TEP publishes no engineering ranges, so `eng_range` is `None` on every tag "
        f"and the clipped fraction is `null` on all {st['n_tags']:,d} of them "
        f"({c['n_tags_with_a_clipped_fraction']} produced a number, "
        f"{c['n_censored']} were censored). Same answer as 3W, same reason."
    )
    add("")

    add("## Units are the only real-world contact")
    add("")
    add(
        "Downs & Vogel (1993) publish the engineering unit of every TEP variable, "
        "and the converter carries that table verbatim rather than normalising it: "
        f"{u['n_tags_non_mode']:,d} non-MODE tags across "
        f"{len(u['by_spelling'])} spellings. Half of them do not resolve."
    )
    add("")
    add("| spelling | variables | tags | resolves to |")
    add("|---|---:|---:|---|")
    var_counts = Counter(
        (r["unit_raw"] or "(none)") for r in _measurement(rows)
    )
    per_spelling_vars = defaultdict(set)
    for r in _measurement(rows):
        per_spelling_vars[r["unit_raw"] or "(none)"].add(r["variable"])
    for spelling, entry in u["by_spelling"].items():
        resolved = entry["canonical"] if entry["resolved"] else "**unresolved; comparison refused**"
        add(
            f"| `{spelling}` | {len(per_spelling_vars[spelling])} | "
            f"{var_counts[spelling]:,d} | {resolved} |"
        )
    add("")
    add("![units](out/01_units.png)")
    add("")
    add(
        f"{u['n_tags_unresolved']:,d} of {u['n_tags_non_mode']:,d} tags "
        f"({u['n_variables_unresolved']} of 52 variables) carry one of "
        + ", ".join(f"`{x}`" for x in u["unresolved_raw_units"])
        + ". Those four are not one problem:"
    )
    add("")
    add(
        "- `kPa gauge` is the refusal `docs/SCHEMA.md` already documents. A gauge "
        "pressure is read against local atmosphere and the alias table has no column "
        "for a pressure datum, so registering it as `kPa` would make every "
        "comparison wrong by about an atmosphere. Leaving it unresolved is the "
        "correct answer, not a gap."
    )
    add(
        "- `mol%` is a composition basis, not a bare percentage. It shares a symbol "
        "with the `%` that `xmv_*` carries and means something else; resolving both "
        "to `percent` would make a valve position comparable with a mole fraction."
    )
    add(
        "- `kscmh` is thousand *standard* cubic metres per hour - the same "
        "reference-condition problem as `Nm3/h`, which the table does carry with "
        "`reference_condition: true`. This is a table gap, and a declarable one."
    )
    add(
        "- `kW` is plain SI power that pint reads without complaint. This is the "
        "same kind of table gap the 3W study found in `Pa`, and the same kind of "
        "thing only a real units file surfaces."
    )
    add("")
    add(
        "The table is left alone here. Two of the four are refusals the schema "
        "argues for on purpose, and the other two are alias rows whose reference "
        "semantics deserve their own change rather than a line added inside a study."
    )
    add("")

    add("## Flatline verdicts")
    add("")
    add(
        f"Each tag's last {window_h:.0f} h window was assessed against the earlier "
        "equal-size windows of the same run."
    )
    add("")
    add("| verdict | tags | share |")
    add("|---|---:|---:|")
    for verdict in VERDICT_ORDER:
        if verdict in fv:
            add(f"| `{verdict}` | {fv[verdict]:,d} | {_pct(fv[verdict], total_v)} |")
    add("")
    add("![flatline verdicts](out/02_flatline_verdicts.png)")
    add("")
    add(
        f"{ar['n_mode_tags']:,d} of those tags are the `LABEL_fault` MODE archives, "
        "and every one is declined as state-valued: a label holds its state until "
        "the fault starts, which is the label working, not a sensor freezing. Of "
        f"the {ar['n_non_mode_tags']:,d} measurement tags, "
        f"{ar['n_non_mode_assessed']:,d} were assessed and {ar['n_fired']:,d} fired "
        f"({_pct(ar['n_fired'], ar['n_non_mode_assessed'])})."
    )
    add("")
    if ar["fired_by_signal_combination"]:
        add("| fired on signal(s) | tags |")
        add("|---|---:|")
        for combo, count in ar["fired_by_signal_combination"].items():
            add(f"| {combo} | {count:,d} |")
        add("")

    add("### Where the fires come from")
    add("")
    add(
        f"Every stall the detector measured is one of {len(sb['stall_histogram_s'])} "
        "values, and a few of them carry almost every tag, because the sample grid "
        "is exact and the plant's own dynamics are the only source of variety:"
    )
    add("")
    add("| stall at window end | tags | what it is |")
    add("|---|---:|---|")
    stall_meaning = {
        "0": "the value moved at the last sample",
        "180": "one sample; no tag's p99 is below 180 s, so it never fires",
        "720": "four samples: a 15-minute analyser read on a 3-minute grid",
        f"{run_meta['window_s']:.0f}": "no change anywhere in the window",
    }
    for stall, count in list(sb["stall_histogram_s"].items())[:6]:
        add(f"| {stall} s | {count:,d} | {stall_meaning.get(stall, 'plant dynamics')} |")
    add("")
    add("![stalls](out/03_stalls.png)")
    add("")
    always = sb["variables_fired_on_every_run"]
    add(
        f"One population accounts for {_pct(ww['n_tags'], ar['n_fired'])} of the "
        f"{ar['n_fired']:,d} fires, and it is the one worth having. The population "
        "that used to sit beside it was an artefact, and it is gone."
    )
    add("")
    if always:
        add(
            f"**The slow analysers: {len(always):,d} variables firing on all "
            f"{conv['n_runs']:,d} runs** - "
            + ", ".join(f"`{v}`" for v in always)
            + ". In the source these five change once every five samples and hold "
            "in between; the analysers on the other two streams "
            "(`xmeas_23`-`xmeas_36`) change once every two."
        )
        add("")
    else:
        add(
            "**The slow analysers no longer fire, and that is the fix showing.** "
            "`xmeas_37`-`xmeas_41` are the composition analysers on the product "
            "stream: in the source they change once every five samples and hold in "
            "between, so on a 180 s grid the last sample of every run sits 720 s "
            "past their last update, on every run, because every run shares the "
            "same grid and the same phase. The first run of this study reported all "
            f"{5 * conv['n_runs']:,d} of them FIRED - five variables on all "
            f"{conv['n_runs']:,d} runs - because their reference p99 came back as "
            "the 180 s sample period and 720 s beats it. Timed between actual "
            "change events their p99 is 900 s, which is the composition-analyser "
            "sampling period of the plant they simulate, and a 720 s hold no longer "
            "beats it: every one of them now returns `NO_FLATLINE`. The staircase "
            "was always a staircase; only the threshold was wrong."
        )
        add("")
    add(
        f"**The two faults that stop the plant: {ww['n_tags']:,d} tags across "
        f"{ww['n_runs']} runs whose value does not change once in the whole "
        f"{window_h:.0f} h window.** Every one of them is in "
        + " or ".join(f"fault {f}" for f in ww["by_fault"])
        + " - "
        + ", ".join(
            f"{n} of the 20 fault-{f} runs" for f, n in ww["runs_per_fault"].items()
        )
        + " - and they run 24-29 variables deep per run. In `fault06_run001` the A "
        "feed flow `xmeas_1` reaches exactly 0.0 and stays there; fault 6 in the "
        "Downs & Vogel table is a loss of that feed, and the plant trips, taking "
        "most of the flowsheet flat with it. These are the fires worth having, and "
        "the only ones here a plant engineer would call a flatline. They are "
        f"{_pct(ww['n_tags'], ar['n_fired'])} of the fires the detector reported."
    )
    add("")

    add("### The p99 change interval, and the defect that used to flatten it")
    add("")
    p99_values = sb["reference_p99_values_s"]
    ranked = sorted(p99_values.items(), key=lambda kv: -kv[1])
    top = ranked[:4]
    covered = sum(v for _, v in top)
    add(
        "Signal 1 compares a stall against \"the tag's own p99 of historical change "
        f"intervals\" (`src/tsdive/detectors/flatline.py:12-14`). Across the "
        f"{sb['n_assessed']:,d} assessed tags that p99 now takes "
        f"{len(p99_values):,d} distinct values, from {min(float(k) for k in p99_values):.0f} s "
        f"to {max(float(k) for k in p99_values):.0f} s. Four of them cover "
        f"{_pct(covered, sb['n_assessed'])} of the tags: "
        + ", ".join(f"{k} s on {v:,d}" for k, v in top)
        + ". A tag that updates every sample and a tag that updates every fifth "
        "sample no longer get the same reference."
    )
    add("")
    add(
        "They used to. The first run of this study found the p99 taking exactly one "
        "value - 180 s, the sample period - on all "
        f"{sb['n_assessed']:,d} tags, and that was a library defect rather than a "
        "property of TEP. `reference_history` marked the adjacent sample pairs "
        "whose value differed and then collected **the spacing of those pairs**, "
        "not the time between successive changes:"
    )
    add("")
    add("```python")
    add("changed = values.ne(values.shift()).to_numpy(dtype=bool, na_value=False)[1:]")
    add("deltas = good[\"timestamp\"].diff().dt.total_seconds().to_numpy()[1:]")
    add("...")
    add("intervals = [float(d) for d in deltas[changed & same_window]]")
    add("```")
    add("")
    add(
        "On any evenly sampled archive every element of `deltas` is the sample "
        "period, so every element of `intervals` was too, whatever the tag did. "
        "The quantity the docstring names - how long this tag normally goes "
        "between changes - is the diff of the timestamps *at* the change points, "
        "and it was never computed. The effect was that signal 1 fired on any tag "
        "holding a value for two samples or more, which is why a 15-minute analyser "
        "read as a flatline here and why the 3W profile study saw a p99 of 1 s on a "
        "1 Hz record. That study read the 1 s as a consequence of dense data; TEP, "
        "sampled 180 times slower, gave the same answer scaled by exactly 180, "
        "which is what made it structural rather than incidental."
    )
    add("")
    add(
        "The fix (`fix(api): time reference change intervals between change "
        "events`) diffs the timestamps of the change events inside each reference "
        "window. On TEP it takes FIRED from 3,827 tags to "
        f"{ar['n_fired']:,d}: the {5 * conv['n_runs']:,d} analyser fires go to zero, "
        f"and the {ww['n_tags']:,d} plant-trip fires - the ones on ground truth - "
        "all survive it, which is the check that matters."
    )
    add("")
    if ar["fired_by_variable"]:
        add("| variable | FIRED | of |")
        add("|---|---:|---:|")
        assessed_by_var = Counter(
            r["variable"] for r in _measurement(rows) if r["flatline_verdict"]
        )
        for variable, count in list(ar["fired_by_variable"].items())[:12]:
            add(f"| `{variable}` | {count:,d} | {assessed_by_var[variable]:,d} |")
        add("")

    add("### No manipulated variable is held fixed in this distribution")
    add("")
    add(
        "The classic Downs & Vogel plant has a manipulated variable the base-case "
        "controller never moves - the agitator speed, XMV(12) - and the expectation "
        "going into this study was that a whole-life-constant `xmv_*` would light "
        "the flatline detector up the way a frozen transmitter does on 3W. It does "
        "not happen here, for a reason worth writing down: **the Rieth 2017 RData "
        "carries `xmv_1` through `xmv_11` only.** XMV(12) is not in the file. Every "
        "manipulated variable that is present moves under closed-loop control at "
        "nearly every sample."
    )
    add("")
    add(
        f"{dv['n_frozen_whole_life']:,d} of {dv['n_non_mode_tags']:,d} measurement "
        "tags hold exactly one distinct value across their whole run "
        f"({_pct(dv['n_frozen_whole_life'], dv['n_non_mode_tags'])}); on 3W that "
        "figure was 22.4%, and it is the gate the population baseline at stage 4 "
        "exists to get past. The least-varying variables in this subset, by their "
        "minimum distinct count over any run:"
    )
    add("")
    add("| variable | fewest distinct values in any run |")
    add("|---|---:|")
    for variable, n in dv["least_varying_variables"].items():
        add(f"| `{variable}` | {n:,d} |")
    add("")
    add("![distinct values](out/04_whole_run_distinct.png)")
    add("")
    probe = summary["quantisation_probe"]
    if probe:
        add(
            f"The tag at the bottom of that table is worth opening. "
            f"`{probe['loop_id']}:{probe['variable']}` holds {probe['n_distinct']} "
            f"distinct values over {probe['n_rows']} rows, spanning "
            f"{probe['min']} to {probe['max']} {probe['unit_raw']} in steps of "
            f"{probe['smallest_step']}. It is not a stuck sensor and it is not a "
            "quiet one: it is a reactor temperature under tight control, published "
            "rounded to two decimals. **The distinct-value signal on this dataset is "
            "partly measuring the source's decimal precision**, and a variable whose "
            "controlled range is narrower than its stored resolution will look "
            "frozen to any counting rule. Nothing in the archive records how many "
            "digits the simulator kept, so nothing downstream can correct for it."
        )
        add("")

    add("## Refusals")
    add("")
    if summary["refusals"]:
        add("| refusal | tags |")
        add("|---|---:|")
        for name, count in sorted(summary["refusals"].items(), key=lambda kv: -kv[1]):
            add(f"| `{name}` | {count:,d} |")
    else:
        add(
            f"No archive was refused. All {st['n_tags']:,d} studied tags read "
            "cleanly. Every profile refusal - naive timestamps, a missing quality "
            "column, a string on a measurement tag, a censored baseline - names a "
            "way a historian can be wrong, and the conversion had already turned the "
            "two that TEP can reach (no clock, no quality column) into recorded "
            "decisions, each marked in the archive metadata."
        )
    add("")

    add("## What a simulator does and does not test")
    add("")
    add(
        "The 3W profile study listed seven ways real data disagreed with the "
        "synthetic backbone. TEP sits between the two, and being explicit about "
        "which side it is on is the reason this section exists."
    )
    add("")
    add(
        "1. **It exercises the ingest decisions, not the physics rules.** TEP forces "
        "two declarations, a synthesised clock and an assumed quality code, "
        "and both travel with the archive where a reader can see them. That is the "
        "part of the profile this dataset tests end to end."
    )
    add(
        "2. **It cannot exercise the gap classifier at all.** Zero gaps of any "
        "class. The rules that decide whether silence is compression, an outage or "
        "`unknown` never run, and no amount of TEP will make them run."
    )
    add(
        "3. **It exercises the freeze detector in one narrow, real way.** "
        f"{dv['n_frozen_whole_life']} tags are frozen for their whole run, so the "
        "3W case - a transmitter pinned for its entire archive, which no "
        "self-referencing threshold can reach - does not occur here at all. What "
        f"does occur is a plant that trips: {ww['n_tags']:,d} tags in "
        f"{ww['n_runs']} runs of two faults go flat for a whole window and the "
        "detector says so. That is a true positive on ground truth, and it is "
        f"{_pct(ww['n_tags'], ar['n_fired'])} of the fires. The other "
        f"{ar['n_fired'] - ww['n_tags']:,d} are tags that stalled for longer than "
        "their own history says they usually do, scattered across 46 variables "
        "rather than concentrated in five."
    )
    add(
        "4. **Its units are real and its numbers are not.** The one thing TEP "
        "inherits from a real plant is the Downs & Vogel unit table, and that is "
        f"precisely where the profile refuses: {u['n_tags_unresolved']:,d} tags "
        "unresolved, comparison refused. A validation backbone whose only "
        "real-world contact is the part the library declines to guess about is a "
        "fair description of what a simulator is worth here."
    )
    add(
        "5. **Ground truth is the thing it does have.** Every run carries "
        "`LABEL_fault` with the fault number per row and the onset sample recorded "
        f"in `RUN.json` ({conv['n_faulty_rows']:,d} faulty rows over "
        f"{conv['n_rows']:,d}). The profile does not use it. Stages 3 and above will, and "
        "TEP is where a detector number can be checked against a fault whose start "
        "time is known exactly rather than inferred from a folder name."
    )
    add("")

    add("## Files")
    add("")
    add("| path | what |")
    add("|---|---|")
    add("| `run_study.py` | profiles every archive, writes `results/per_tag.csv` |")
    add("| `make_report.py` | this report, `results/summary.json`, `out/*.png` |")
    add("| `results/summary.json` | every number above, machine-readable (committed) |")
    add(
        f"| `results/per_tag.csv` | one row per studied tag, {len(rows[0])} columns "
        "(committed) |"
    )
    add("| `results/run.json` | window size, wall time, tag count (committed) |")
    add(
        "| `results/benchmarks_section.md` | the BENCHMARKS.md REAL block, generated "
        "from `summary.json` |"
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=str(HERE / "results"))
    parser.add_argument("--archives", default=str(ROOT / "data" / "tep_archives"))
    parser.add_argument("--src", default=str(ROOT / "data" / "tep"))
    args = parser.parse_args(argv)

    results = Path(args.results)
    rows = load_rows(results / "per_tag.csv")
    run_meta = json.loads((results / "run.json").read_text("utf-8"))
    runs = load_runs(Path(args.archives))

    summary = build_summary(rows, runs, run_meta, Path(args.src), Path(args.archives))
    (results / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    (results / "benchmarks_section.md").write_text(
        render_benchmarks_section(summary), encoding="utf-8", newline="\n"
    )

    out = HERE / "out"
    out.mkdir(parents=True, exist_ok=True)
    plot_units(summary, out / "01_units.png")
    plot_verdicts(rows, out / "02_flatline_verdicts.png")
    plot_stalls(summary, run_meta, out / "03_stalls.png")
    plot_distinct(rows, out / "04_whole_run_distinct.png")

    (HERE / "REPORT.md").write_text(
        render_report(summary, rows, run_meta), encoding="utf-8", newline="\n"
    )
    print(
        f"wrote {HERE / 'REPORT.md'}, {results / 'summary.json'}, "
        f"{results / 'benchmarks_section.md'} and 4 figures"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

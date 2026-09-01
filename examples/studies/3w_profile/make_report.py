"""Aggregate the 3W profile study into REPORT.md, summary.json and figures.

    uv run python examples/studies/3w_profile/make_report.py

Reads ``results/per_tag.csv`` (written by ``run_study.py``) plus every
``INSTANCE.json`` under the archive root, and states counts. It computes
no detector metric: profiling is descriptive, and this repo does not publish
detector numbers on real data until the split machinery has group
holdout (see REPORT.md).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def git_head() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover - not a test path
        return "unknown"


def load_rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_instances(root: Path) -> list[dict]:
    return [
        json.loads(p.read_text("utf-8")) for p in sorted(root.glob("*/INSTANCE.json"))
    ]


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
    """FIRED share among the tags a flatline verdict can be reached on.

    MODE tags are NOT ASSESSED by construction now, so counting them in
    the denominator would understate the fire rate on the tags the
    detector actually judges.
    """
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
    }


def _whole_life_frozen(rows: list[dict]) -> dict:
    """Non-MODE tags holding one value across the entire archive.

    Their own history contains no movement, so every self-referencing
    threshold is empty for them: the stage-4 cross-instance-baseline case.
    """
    measured = [r for r in _measurement(rows) if r["whole_life_distinct"]]
    frozen = [r for r in measured if int(r["whole_life_distinct"]) == 1]
    return {
        "n_non_mode_tags": len(measured),
        "n_frozen_whole_life": len(frozen),
        "n_instances_affected": len({r["instance"] for r in frozen}),
        "per_variable": dict(Counter(r["variable"] for r in frozen).most_common()),
        "verdicts": dict(Counter(r["flatline_verdict"] for r in frozen).most_common()),
    }


SIGNAL_NUMBER = {
    "time_since_last_actual_change": "1",
    "distinct_value_count_vs_p05": "2",
    "zero_std_while_coupled_tag_moves": "3",
}


def _signal_behaviour(rows: list[dict]) -> dict:
    """What the two evaluable signals did, and how tight their thresholds are.

    Signal 1 compares a stall against the tag's own p99 change interval.
    On a 1 Hz archive whose value moves at nearly every sample that p99
    is 1 s, so the threshold any stall has to beat is one sample. Whether
    that is a useful bar is a question this study can only answer by
    stating the distribution.
    """
    assessed = [
        r
        for r in _measurement(rows)
        if r["flatline_verdict"] in ("FIRED", "NO_FLATLINE")
    ]
    p99 = [
        float(r["ref_p99_change_interval_s"])
        for r in assessed
        if r["ref_p99_change_interval_s"]
    ]
    fired = [r for r in assessed if r["flatline_verdict"] == "FIRED"]
    combos = Counter(
        ";".join(SIGNAL_NUMBER.get(s, s) for s in r["flatline_fired_signals"].split(";") if s)
        for r in fired
    )
    stall_only = [
        float(r["stall_s"])
        for r in fired
        if r["flatline_fired_signals"] == "time_since_last_actual_change" and r["stall_s"]
    ]
    # Signal 2 fires on distinct < p05. Within a tenth of the threshold is
    # a fire the reference distribution barely separates from normal.
    sig2 = [
        (float(r["final_distinct"]), float(r["ref_p05_distinct"]))
        for r in fired
        if "distinct_value_count_vs_p05" in r["flatline_fired_signals"]
        and r["final_distinct"]
        and r["ref_p05_distinct"]
    ]
    near = [(d, p) for d, p in sig2 if p > 0 and d >= 0.9 * p]
    return {
        "n_signal2_fires": len(sig2),
        "n_signal2_fires_within_10pc_of_p05": len(near),
        "n_assessed": len(assessed),
        "n_with_a_p99": len(p99),
        "n_p99_of_one_second": sum(1 for v in p99 if v == 1.0),
        "fired_by_signal_combination": dict(combos.most_common()),
        "n_signal1_only_fires": len(stall_only),
        "n_signal1_only_fires_under_5s": sum(1 for s in stall_only if s <= 5.0),
        "signal1_only_stall_median_s": (
            statistics.median(stall_only) if stall_only else None
        ),
    }


def spot_check(rows: list[dict], n: int = 10) -> list[dict]:
    """FIRED non-MODE tags, spread across variables, with their numbers.

    Round-robin over variables in descending fire count, first instance
    first, so ten rows cannot all come from one variable or one well.
    """
    by_variable: dict[str, list[dict]] = defaultdict(list)
    for row in _measurement(rows):
        if row["flatline_verdict"] == "FIRED":
            by_variable[row["variable"]].append(row)
    order = sorted(by_variable, key=lambda v: (-len(by_variable[v]), v))
    for variable in order:
        by_variable[variable].sort(key=lambda r: (r["instance"], r["variable"]))
    picked: list[dict] = []
    depth = 0
    while len(picked) < n and order:
        added = False
        for variable in order:
            if depth < len(by_variable[variable]) and len(picked) < n:
                picked.append(by_variable[variable][depth])
                added = True
        if not added:
            break
        depth += 1
    return picked


def plot_absent(instances: list[dict], out: Path) -> Path:
    counts: Counter[str] = Counter()
    for record in instances:
        counts.update(record["absent_variables"])
    variables = sorted(
        {v for r in instances for v in r["absent_variables"] + r["present_variables"]}
    )
    total = len(instances)
    shares = [(v, 100.0 * counts.get(v, 0) / total) for v in variables]
    shares.sort(key=lambda t: t[1])

    fig, ax = plt.subplots(figsize=(8.5, 8.0), dpi=160)
    ypos = range(len(shares))
    ax.barh(
        list(ypos),
        [s for _, s in shares],
        color=SERIES[0],
        height=0.66,
        edgecolor="white",
        linewidth=1.0,
    )
    ax.set_yticks(list(ypos))
    ax.set_yticklabels([v for v, _ in shares], fontsize=9, color=INK)
    ax.set_xlim(0, 112)
    ax.set_xlabel("instances where the column is entirely NaN (%)", color=INK_MUTED)
    ax.set_title(
        f"3W: an absent variable is an all-NaN column ({total:,d} real instances)",
        color=INK,
        fontsize=12,
        pad=12,
    )
    for y, (_, share) in zip(ypos, shares, strict=True):
        ax.text(share + 1.5, y, f"{share:.1f}%", va="center", fontsize=8, color=INK_MUTED)
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
    variables = sorted(by_var, key=lambda v: -by_var[v].get("FIRED", 0) / sum(by_var[v].values()))

    fig, ax = plt.subplots(figsize=(9.0, 8.0), dpi=160)
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
            height=0.66,
            label=verdict,
            edgecolor="white",
            linewidth=1.0,
        )
        left = [a + b for a, b in zip(left, widths, strict=True)]
    ax.set_yticks(ypos)
    ax.set_yticklabels(variables, fontsize=9, color=INK)
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of studied tags (%)", color=INK_MUTED)
    ax.set_title(
        "Flatline verdicts per variable, 1 h window against earlier windows",
        color=INK,
        fontsize=12,
        pad=12,
    )
    ax.legend(
        frameon=False,
        fontsize=8,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.13),
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


def plot_extents(instances: list[dict], window_s: float, out: Path) -> Path:
    """Log-x: instance lengths span 1.5 h to 213 h, so a linear axis hides the bulk."""
    import numpy as np

    hours = sorted((r["n_rows"] - 1) / 3600.0 for r in instances)
    threshold = 2 * window_s / 3600.0
    bins = np.logspace(np.log10(min(hours) * 0.9), np.log10(max(hours) * 1.1), 36)
    fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=160)
    counts, _, _ = ax.hist(hours, bins=bins, color=SERIES[0], edgecolor="white", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_ylim(0, max(counts) * 1.28)
    top = ax.get_ylim()[1]
    ax.axvline(threshold, color=SERIES[1], linewidth=2.0, ymax=0.78)
    below = sum(1 for h in hours if h < threshold)
    ax.text(
        threshold * 1.06,
        top * 0.77,
        f"two {window_s / 3600:.0f} h windows",
        rotation=90,
        va="top",
        fontsize=8,
        color=SERIES[1],
    )
    ax.text(
        0.985,
        0.94,
        f"{below} of {len(hours):,d} instances fall left of the line:\n"
        "no earlier window exists to reference",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color=INK_MUTED,
    )
    median = hours[len(hours) // 2]
    ax.text(
        median,
        max(counts) * 1.05,
        f"median {median:.1f} h",
        fontsize=9,
        color=INK_MUTED,
        ha="center",
    )
    ax.set_xlabel("instance extent (hours, log scale)", color=INK_MUTED)
    ax.set_ylabel("instances", color=INK_MUTED)
    ax.set_xticks([2, 5, 10, 25, 50, 100, 200])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_title("Real 3W instance lengths against the flatline reference need", color=INK, pad=12)
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


def build_summary(rows: list[dict], instances: list[dict], run: dict, src: Path) -> dict:
    studied = {r["instance"] for r in rows}
    coverage = [c for c in (_f(r["coverage"]) for r in rows) if c is not None]
    valid = [v for v in (_f(r["valid_fraction"]) for r in rows) if v is not None]
    verdicts = Counter(r["flatline_verdict"] for r in rows if r["flatline_verdict"])
    refusals = Counter(r["refusal_type"] for r in rows if r["refusal_type"])
    gap_classes: Counter[str] = Counter()
    for row in rows:
        for cls in filter(None, row["gap_classes"].split(";")):
            gap_classes[cls] += 1
    units = Counter(
        (r["unit_raw"] or "(none)", r["unit_resolved"]) for r in rows if r["role"] != "MODE"
    )
    unresolved = sorted({u for (u, ok) in units if ok == "false"})

    absent = Counter()
    present = Counter()
    for record in instances:
        absent.update(record["absent_variables"])
        present.update(record["present_variables"])
    n_variables = len(set(absent) | set(present))

    prefix = [r["unlabelled_prefix_rows"] for r in instances]
    folder_mismatch = []
    for record in instances:
        folder = record["folder_label"]
        labels = {k for k in record["row_label_histogram"]["class"] if k != "NA"}
        expected = {folder, str(int(folder) + 100)}
        if not (labels & expected):
            folder_mismatch.append(
                {"instance": record["instance"], "folder": folder, "row_labels": sorted(labels)}
            )

    fetch = json.loads((src / "FETCH.json").read_text("utf-8"))
    return {
        "dataset": {
            "repo": fetch["repo"],
            "commit": fetch["commit"],
            "bytes_downloaded": fetch["bytes"],
            "n_files_downloaded": fetch["n_files"],
            "filters": fetch["filters"],
            "manifest_sha256": sha256_file(src / "MANIFEST.sha256.json"),
        },
        "code": {"git_head": git_head(), "tsdive_version": run["tsdive_version"]},
        "converted": {
            "n_instances": len(instances),
            "n_archives": sum(len(r["archives_written"]) for r in instances),
            "n_wells": len({r["well"] for r in instances}),
            "n_rows": sum(r["n_rows"] for r in instances),
        },
        "studied": {
            "n_instances": len(studied),
            "n_tags": len(rows),
            "cap_per_folder": run.get("cap_per_folder"),
            "window_s": run["window_s"],
            "wall_seconds": run["wall_seconds"],
            "jobs": run["jobs"],
        },
        "coverage": _quantiles(coverage),
        "coverage_below_1": sum(1 for c in coverage if c < 1.0),
        "valid_fraction": _quantiles(valid),
        "valid_fraction_below_1": sum(1 for v in valid if v < 1.0),
        "gap_classes": dict(gap_classes),
        "flatline_verdicts": dict(verdicts),
        "flatline_assessable_roles": _assessable(rows),
        "flatline_signal_behaviour": _signal_behaviour(rows),
        "whole_life_frozen": _whole_life_frozen(rows),
        "refusals": dict(refusals),
        "timestamp_audit": {
            "duplicates": sum(int(r["duplicate_timestamps"] or 0) for r in rows),
            "non_monotonic": sum(int(r["non_monotonic_positions"] or 0) for r in rows),
        },
        "quality": {
            "n_bad_samples": sum(int(r["n_bad"] or 0) for r in rows),
            "n_good_samples": sum(int(r["n_good"] or 0) for r in rows),
            "n_uncertain_samples": sum(int(r["n_uncertain"] or 0) for r in rows),
            "n_tags_with_unmapped_codes": sum(1 for r in rows if r["unmapped_codes"]),
        },
        "clipping": {
            "n_tags_with_a_clipped_fraction": sum(1 for r in rows if r["clipped_fraction"]),
            "n_censored": sum(1 for r in rows if r["censored"] == "true"),
        },
        "units": {
            "unresolved_raw_units": unresolved,
            "n_tags_unresolved": sum(
                1 for r in rows if r["role"] != "MODE" and r["unit_resolved"] == "false"
            ),
            "n_tags_non_mode": sum(1 for r in rows if r["role"] != "MODE"),
        },
        "absent_variables": {
            "n_variables": n_variables,
            "per_variable_absent_instances": dict(sorted(absent.items())),
            "mean_absent_per_instance": statistics.fmean(
                [len(r["absent_variables"]) for r in instances]
            ),
            "min_absent_per_instance": min(len(r["absent_variables"]) for r in instances),
            "max_absent_per_instance": max(len(r["absent_variables"]) for r in instances),
        },
        "labels": {
            "unlabelled_prefix_rows_distinct": dict(Counter(prefix)),
            "n_instances_with_3600_prefix": sum(1 for p in prefix if p == 3600),
            "n_folder_row_label_mismatch": len(folder_mismatch),
            "folder_mismatch_by_folder": dict(
                Counter(m["folder"] for m in folder_mismatch)
            ),
        },
    }


def _mismatch_label_sets(instances: list[dict]) -> str:
    """Which class labels the folder-mismatched instances actually carry."""
    seen: Counter[tuple[str, ...]] = Counter()
    for record in instances:
        folder = record["folder_label"]
        labels = {k for k in record["row_label_histogram"]["class"] if k != "NA"}
        if not (labels & {folder, str(int(folder) + 100)}):
            seen[tuple(sorted(labels))] += 1
    return "; ".join(
        f"{', '.join('`' + c + '`' for c in k) or 'none at all'} in {v} of them"
        for k, v in seen.most_common()
    )


def render_report(summary: dict, rows: list[dict], instances: list[dict], run: dict) -> str:
    ds, code = summary["dataset"], summary["code"]
    conv, st = summary["converted"], summary["studied"]
    cov = summary["coverage"]
    lines: list[str] = []
    add = lines.append

    add("# Stage 1 on real data: 3W v2.0.0")
    add("")
    add(
        "What tsdive's data-physics layer says about real offshore well data, "
        "and what it refuses to say. Descriptive only: no detector metric is "
        "published here (see *Why no detector numbers* below)."
    )
    add("")
    add("## Provenance")
    add("")
    add("| field | value |")
    add("|---|---|")
    add(f"| dataset | `{ds['repo']}` @ `{ds['commit']}` (v2.0.0, CC BY 4.0) |")
    add(f"| downloaded | {ds['n_files_downloaded']:,d} files, {ds['bytes_downloaded']:,d} bytes |")
    add(f"| fetch filters | `only_real={ds['filters']['only_real']}`, classes=all |")
    add(f"| manifest sha256 | `{ds['manifest_sha256']}` |")
    add(f"| code | tsdive {code['tsdive_version']} @ `{code['git_head']}` |")
    add(f"| converted | {conv['n_instances']:,d} instances, {conv['n_archives']:,d} archives, "
        f"{conv['n_wells']} wells, {conv['n_rows']:,d} source rows |")
    add(
        f"| studied | {st['n_instances']:,d} instances, {st['n_tags']:,d} tags, "
        f"{st['wall_seconds']:.0f} s wall on {st['jobs']} processes |"
    )
    add("")
    add("Commands:")
    add("")
    cap = st["cap_per_folder"]
    add("```console")
    add("uv run python scripts/fetch_3w.py --dest data/3w --only-real --yes")
    add("uv run python scripts/convert_3w.py --src data/3w --out data/3w_archives --only-real")
    add(
        "uv run python examples/studies/3w_profile/run_study.py"
        + (f" --cap-per-folder {cap}" if cap else "")
        + f" --jobs {st['jobs']}"
    )
    add("uv run python examples/studies/3w_profile/make_report.py")
    add("```")
    add("")
    add("## What was studied, and what was not")
    add("")
    if cap:
        add(
            f"All {conv['n_instances']:,d} real `WELL-*` instances were fetched and "
            f"converted. This run profiled a **stratified subset**: at most {cap} "
            "instances per event folder, taken round-robin across the wells in that "
            "folder, plus a guarantee pass so every well appears. That is "
            f"{st['n_instances']:,d} instances / {st['n_tags']:,d} tags."
        )
    else:
        add(
            f"Every one of the {conv['n_instances']:,d} real `WELL-*` instances was "
            f"fetched, converted and profiled: {st['n_tags']:,d} tags across "
            f"{conv['n_wells']} wells and all ten event folders, "
            f"{st['wall_seconds']:.0f} s of wall time on {st['jobs']} processes. Each "
            "archive is read twice - once over its whole extent for the physics block, "
            f"once over its last {run['window_s'] / 3600:.0f} h for the flatline "
            "assessment - so every number below is over the whole real set, with no "
            "subset to argue about."
        )
    add("")

    add("## Absent variables")
    add("")
    av = summary["absent_variables"]
    add(
        f"3W has {av['n_variables']} process variables. An instance that never "
        "instrumented one still carries the column - filled entirely with NaN. "
        f"Across {conv['n_instances']:,d} real instances the count of absent variables per "
        f"instance runs {av['min_absent_per_instance']}-{av['max_absent_per_instance']} "
        f"(mean {av['mean_absent_per_instance']:.1f}). The converter writes no archive for "
        "an absent variable and lists it in `INSTANCE.json`: an archive of nothing but "
        "`NO_DATA` would assert the tag existed and failed, which the source never says."
    )
    add("")
    always = sorted(
        v
        for v, n in av["per_variable_absent_instances"].items()
        if n == conv["n_instances"]
    )
    add(
        f"{len(always)} of the {av['n_variables']} variables are absent from **every** real "
        "instance - " + ", ".join(f"`{v}`" for v in always) + " never carry a single "
        "value on a real well. A schema read off the column list alone would offer them "
        "as tags."
    )
    add("")
    add("| variable | instances where absent | share |")
    add("|---|---:|---:|")
    for variable, count in sorted(
        av["per_variable_absent_instances"].items(), key=lambda kv: (-kv[1], kv[0])
    ):
        add(f"| `{variable}` | {count:,d} | {_pct(count, conv['n_instances'])} |")
    add("")
    add("![absent variables](out/01_absent_variables.png)")
    add("")

    add("## Coverage, gaps and the timestamp audit")
    add("")
    add(
        f"Coverage over the studied tags: min {cov['min']:.4f}, p05 {cov['p05']:.4f}, "
        f"median {cov['p50']:.4f}, mean {cov['mean']:.4f}, max {cov['max']:.4f}; "
        f"{summary['coverage_below_1']:,d} of {st['n_tags']:,d} tags fall below 1.000."
    )
    add("")
    vf = summary["valid_fraction"]
    add(
        f"Valid fraction over the same tags - rows carrying a usable value, over rows "
        f"in the window: min {vf['min']:.4f}, p05 {vf['p05']:.4f}, median "
        f"{vf['p50']:.4f}, mean {vf['mean']:.4f}, max {vf['max']:.4f}; "
        f"{summary['valid_fraction_below_1']:,d} of {st['n_tags']:,d} tags fall below "
        "1.000. That is the same tag population and a different question, and the two "
        "answers disagree on every one of those tags."
    )
    add("")
    if summary["gap_classes"]:
        add("| gap class | tags with at least one |")
        add("|---|---:|")
        for cls, count in sorted(summary["gap_classes"].items(), key=lambda kv: -kv[1]):
            add(f"| `{cls}` | {count:,d} |")
    else:
        add(
            "No gap of any class was found in any studied tag. 3W is written at a "
            "strict 1 s cadence with no interior holes, so the gap classifier - the "
            "part of the profile with the most rules - had nothing to classify."
        )
    add("")
    ta = summary["timestamp_audit"]
    add(
        f"Timestamp audit: {ta['duplicates']:,d} duplicate timestamps and "
        f"{ta['non_monotonic']:,d} non-monotonic positions across the studied tags. "
        "The 3W index is clean; the audit's value here is that it says so with a "
        "number rather than by silence."
    )
    add("")
    nan_tags = sum(1 for r in rows if int(r["n_bad"] or 0) > 0)
    q = summary["quality"]
    nan_share = 100.0 * q["n_bad_samples"] / max(1, q["n_bad_samples"] + q["n_good_samples"])
    add(
        f"**Coverage 1.000 does not mean every value is there.** {nan_tags:,d} of "
        f"{st['n_tags']:,d} studied tags contain interior NaNs - {nan_share:.2f}% of all "
        "samples - and every one of those tags still reports coverage 1.000. Coverage "
        "measures whether the historian emitted a row, and 3W emits one every second "
        "whether or not the sensor had a value to put in it. The NaN rows are visible "
        "in the severity counts as BAD, and in `valid fraction`, which this study is "
        "the reason the layer now reports."
    )
    add("")

    add("## Quality: assumed, and labelled as assumed")
    add("")
    add(
        "3W carries no quality column. The converter writes `GOOD_ASSUMED` on finite "
        "values and `NO_DATA` on NaNs, declares both in the tag's `quality_codes`, and "
        "sets `quality_assumed`, so every profile prints *quality ASSUMED at ingest*. "
        f"Over the studied tags that is {q['n_good_samples']:,d} GOOD and "
        f"{q['n_bad_samples']:,d} BAD samples, with {q['n_uncertain_samples']:,d} "
        f"UNCERTAIN and {q['n_tags_with_unmapped_codes']} tag(s) reporting an unmapped "
        "code. The BAD samples are not sensor failures: they are the NaN holes inside "
        "otherwise-present columns, which no 3W column distinguishes from a failed read."
    )
    add("")

    add("## Units")
    add("")
    u = summary["units"]
    add(
        "The first run of this study left every one of the "
        f"{u['n_tags_non_mode']:,d} non-MODE tags reporting `unresolved (null); "
        "comparison refused`. `dataset.ini` spells its four units `Pa`, `oC`, `%` and "
        "`m3/s`; the shipped alias table carried `kPa`, `degC` and `percent` and none "
        "of those four. That is a gap in the table, not a property of 3W - a general "
        "unit table that cannot resolve pascals is missing the SI base spelling of "
        "pressure, and only a real units file was going to say so. The four aliases "
        "are now registered, along with the rest of the SI spellings the table lacked "
        "(`m³/h`, `MPa`, `mbar`, `t/h`, `l/min`, `m/s`, `rpm`, `mA`, `V`), and a test "
        "walks every row of the table to check pint can still read it."
    )
    add("")
    if u["unresolved_raw_units"]:
        add(
            f"{u['n_tags_unresolved']:,d} of {u['n_tags_non_mode']:,d} non-MODE tags "
            "still carry an unresolved unit: "
            + ", ".join(f"`{x}`" for x in u["unresolved_raw_units"])
            + "."
        )
    else:
        add(
            f"All {u['n_tags_non_mode']:,d} non-MODE tags now resolve. What stays "
            "unresolved by design is documented in `docs/SCHEMA.md`: a bare `C` is "
            "coulombs as readily as Celsius, and `barg`/`psig` name a datum the table "
            "has no column for, so registering them as `bar`/`psi` would make every "
            "comparison wrong by about an atmosphere."
        )
    add("")

    add("## Clipping")
    add("")
    c = summary["clipping"]
    add(
        "3W publishes no engineering ranges, so `eng_range` is `None` on every tag and "
        f"the clipped fraction is `null` on all {st['n_tags']:,d} of them "
        f"({c['n_tags_with_a_clipped_fraction']} tags produced a number, "
        f"{c['n_censored']} were censored). A layer that defaulted the range would have "
        "reported a clipping percentage for 3W; this one reports that it cannot."
    )
    add("")

    add("## Flatline verdicts")
    add("")
    fv = summary["flatline_verdicts"]
    total_v = sum(fv.values())
    add(
        f"Each tag's last {run['window_s'] / 3600:.0f} h window was assessed against the "
        "earlier equal-size windows of the same archive."
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
    add("![instance extents](out/03_instance_extents.png)")
    add("")
    ar = summary["flatline_assessable_roles"]
    add(
        f"{ar['n_mode_tags']:,d} of those tags are `role=MODE` - the `ESTADO-*` valve "
        "states and the two label columns - and not one of them carries a verdict: "
        f"{ar['n_mode_declined_as_state_valued']:,d} are declined as state-valued and "
        f"the remaining {ar['n_mode_tags'] - ar['n_mode_declined_as_state_valued']:,d} "
        "sit in archives too short to reference themselves. A state tag holds one "
        "state until the plant changes it, so stillness is what a settled unit looks "
        "like rather than evidence of a stuck sensor. The first run of this study "
        "reported them as FIRED, which is why the detector now declines the question."
    )
    add("")
    add(
        f"Of the {ar['n_non_mode_tags']:,d} measurement tags, "
        f"{ar['n_non_mode_assessed']:,d} could be assessed and {ar['n_fired']:,d} of "
        f"those fired ({_pct(ar['n_fired'], ar['n_non_mode_assessed'])}). Read the "
        "spot check below before taking that number at face value."
    )
    add("")
    fired = Counter(r["variable"] for r in _measurement(rows) if r["flatline_verdict"] == "FIRED")
    if fired:
        add("Measurement variables whose tags fired most often:")
        add("")
        add("| variable | FIRED | assessed |")
        add("|---|---:|---:|")
        assessed = Counter(r["variable"] for r in _measurement(rows) if r["flatline_verdict"])
        for variable, count in fired.most_common(12):
            add(f"| `{variable}` | {count:,d} | {assessed[variable]:,d} |")
        add("")

    picked = spot_check(rows)
    if picked:
        add("### Spot check: what ten of those fires actually look like")
        add("")
        add(
            "Ten FIRED tags, taken round-robin across variables so no one variable "
            "carries the table. Each row states what the last "
            f"{run['window_s'] / 3600:.0f} h window did and what the tag's own earlier "
            "windows said it usually does. A fire is signal 1 when the stall beats the "
            "p99 change interval, signal 2 when the distinct count falls under the p05, "
            "or both."
        )
        add("")
        add(
            "| instance | variable | n_distinct | ref p05 distinct | std | stall (s) "
            "| ref p99 interval (s) | fired |"
        )
        add("|---|---|---:|---:|---:|---:|---:|---|")
        for row in picked:
            signals = ";".join(
                SIGNAL_NUMBER.get(s, s)
                for s in row["flatline_fired_signals"].split(";")
                if s
            )
            add(
                f"| `{row['instance']}` | `{row['variable']}` | {row['final_distinct']} "
                f"| {row['ref_p05_distinct'] or 'null'} | {row['final_std']} "
                f"| {row['stall_s']} | {row['ref_p99_change_interval_s'] or 'null'} "
                f"| {signals} |"
            )
        add("")

    sb = summary["flatline_signal_behaviour"]
    add(
        f"**Read that table before believing the {_pct(ar['n_fired'], ar['n_non_mode_assessed'])} "
        "headline.** The p99 change interval is 1 s on "
        f"{sb['n_p99_of_one_second']:,d} of the {sb['n_with_a_p99']:,d} assessed tags "
        f"that have one - {_pct(sb['n_p99_of_one_second'], sb['n_with_a_p99'])} of "
        "them, which really do move at every 1 Hz sample. The first run of this "
        "study reported 5,584 of 5,586, and read that as a fact about 3W: dense "
        "sampling, sensors moving constantly, a tag's own history saying it never "
        "holds a value for two seconds. It was a fact about the library. "
        "`reference_history` collected the sample spacing preceding each changed "
        "sample instead of the time between successive change events, so on an "
        "evenly sampled archive the reference came back as the sampling period "
        "whatever the tag did (`fix(api): time reference change intervals between "
        "change events`; the TEP profile study has the full diagnosis). With the "
        "intervals timed between change events, 2,002 of those tags turn out to "
        "hold values for longer than a second."
    )
    add("")
    add(
        f"The fix moves FIRED from 2,580 tags (35.5% of the {ar['n_non_mode_assessed']:,d} "
        f"assessed) to {ar['n_fired']:,d} "
        f"({_pct(ar['n_fired'], ar['n_non_mode_assessed'])}). Signal 1 stopped firing "
        "on 425 tags; 139 of those still fire on signal 2, which is why the verdict "
        "count falls by 286 rather than 425. Signal 2 is untouched by the fix and "
        f"fires on the same {sb['n_signal2_fires']:,d} tags it did before. Signal 1 is "
        "now held against a threshold that means something: "
        f"{sb['n_signal1_only_fires_under_5s']:,d} of the "
        f"{sb['n_signal1_only_fires']:,d} tags that fired on signal 1 alone did so on "
        "a stall of 5 s or less, against 142 of 801 before, and the median such stall "
        f"moved from 123 s to {sb['signal1_only_stall_median_s']:.0f} s. Signal 2 "
        f"remains the steadier of the two, and still "
        f"{sb['n_signal2_fires_within_10pc_of_p05']:,d} of its "
        f"{sb['n_signal2_fires']:,d} fires land within a tenth of the p05 they had to "
        "beat - visible in the rows above where the distinct count sits a percent or "
        "two under the threshold."
    )
    add("")
    add("| fired on signal(s) | tags |")
    add("|---|---:|")
    for combo, count in sb["fired_by_signal_combination"].items():
        add(f"| {combo} | {count:,d} |")
    add("")
    add(
        "A per-tag p99 is now the quantity the detector promises, but on the "
        f"{sb['n_p99_of_one_second']:,d} tags that genuinely move at every sample it "
        "is still a 2 s bar, and a stall of 2 s on a 1 Hz record is not a frozen "
        "transmitter. A per-(well, variable) baseline over other instances is the "
        "reference that would settle it, and it is stage-4 work. Until then these "
        "counts are a description of what the shipped detector does on 3W, not a "
        f"claim that 3W has {ar['n_fired']:,d} frozen sensors."
    )
    add("")

    wl = summary["whole_life_frozen"]
    add("### Frozen for the whole archive")
    add("")
    add(
        f"{wl['n_frozen_whole_life']:,d} of {wl['n_non_mode_tags']:,d} measurement tags "
        f"({_pct(wl['n_frozen_whole_life'], wl['n_non_mode_tags'])}, across "
        f"{wl['n_instances_affected']:,d} instances) hold exactly **one** distinct value "
        "across their entire archive, not merely across the final window. Such a tag "
        "establishes no history of movement, so its own p99 of change intervals is "
        "empty and its reference distinct counts are all 1 - the detector's two "
        "self-referencing thresholds are computed from a record that never moved. Their "
        "verdicts come out "
        + ", ".join(f"{v} on {n:,d}" for v, n in wl["verdicts"].items())
        + ". A self-referencing threshold cannot reach such a tag; "
        "catching it needs a baseline from other instances of the same (well, "
        "variable), which is stage-4 work. This count is the gate on that: it is how "
        "much of the real set a self-referencing threshold cannot reach."
    )
    add("")
    if wl["per_variable"]:
        add("| variable | tags frozen for the whole archive |")
        add("|---|---:|")
        for variable, count in list(wl["per_variable"].items())[:12]:
            add(f"| `{variable}` | {count:,d} |")
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
            f"No archive was refused. All {st['n_tags']:,d} studied tags read cleanly: "
            "the conversion had already turned every condition the profile refuses "
            "(naive timestamps, a missing quality column, string values on a "
            "measurement tag) into an explicit, recorded decision."
        )
    add("")

    add("## Labels")
    add("")
    lb = summary["labels"]
    lengths = sorted(lb["unlabelled_prefix_rows_distinct"].items(), key=lambda kv: -kv[1])
    add(
        f"Every one of the {conv['n_instances']:,d} real instances opens with exactly "
        "3,600 rows whose `class` and `state` are `<NA>` - one unlabelled hour before "
        "the labelled record starts. Distinct prefix lengths across the whole real set: "
        + ", ".join(f"{k} rows in {v:,d} instances" for k, v in lengths)
        + ". Those rows are not normal operation and they are not the event either; a "
        "loader that filled them with the folder's label would invent an hour of "
        "ground truth per instance."
    )
    add("")
    by_folder = Counter(r["folder_label"] for r in instances)
    mismatch_detail = ", ".join(
        f"folder {k}: {v} of {by_folder[k]}"
        for k, v in sorted(lb["folder_mismatch_by_folder"].items())
    )
    add(
        f"**The folder name is not the row label.** {lb['n_folder_row_label_mismatch']:,d} "
        f"of {conv['n_instances']:,d} instances contain no row labelled with their own "
        f"folder's event (neither `F` nor the transient `F+100`) - {mismatch_detail}. "
        f"The labels those instances do carry: {_mismatch_label_sets(instances)}. Any "
        "pipeline that took the directory name as the target would train on labels the "
        "rows do not carry."
    )
    add("")

    add("## What the profile study found that a synthetic backbone could not")
    add("")
    add(
        "The synthetic backbone is generated by the same code that reads it, so every "
        "field it needs exists. Real 3W disagrees on seven points, and each one is a "
        "path the backbone never exercises."
    )
    add("")
    add(
        f"1. **Absent is not missing.** {av['min_absent_per_instance']}-"
        f"{av['max_absent_per_instance']} of {av['n_variables']} variables per instance "
        "are present as entirely-NaN columns. The backbone has no concept of a column "
        "that exists but was never instrumented, so nothing before this study forced a "
        "decision about whether such a column becomes an archive. It does not."
    )
    add(
        "2. **A dataset can ship no quality at all.** The backbone writes quality codes "
        "because its generator does. 3W has none, which is what `quality_assumed` and "
        "the per-tag `quality_codes` map were built for - and it took real data to run "
        "them together end to end."
    )
    add(
        "3. **The alias table does not survive contact with a real units file.** "
        f"All {u['n_tags_non_mode']:,d} non-MODE tags carry `Pa`, `oC`, `%` or `m3/s`, "
        "and the shipped table knew none of the four. The backbone only ever emits "
        "units the table already has, so unit resolution had never failed at scale "
        "before - and a table with `kPa` but no `Pa` is a bug nobody was going to spot "
        "from synthetic data."
    )
    short_instances = len(
        {r["instance"] for r in rows if r["flatline_verdict"] == "NOT_ASSESSED_TOO_SHORT"}
    )
    add(
        "4. **Coverage answers a narrower question than it looks.** Every studied tag "
        f"reports coverage 1.000, and {nan_tags:,d} of them still contain interior "
        "NaNs. On a historian that emits a row per second regardless, coverage is a "
        "statement about rows, not about values. The backbone's gaps are always missing "
        "rows, so the two questions never came apart there - which is why the physics "
        "block had no second number to report until this study forced one."
    )
    add(
        "5. **Instances too short to reference themselves.** Flatline thresholds come "
        "from earlier windows of the same archive, so an instance shorter than two "
        f"{run['window_s'] / 3600:.0f} h windows can never be assessed: "
        f"{fv.get('NOT_ASSESSED_TOO_SHORT', 0):,d} tags across {short_instances:,d} "
        f"instances ({_pct(fv.get('NOT_ASSESSED_TOO_SHORT', 0), total_v)} of studied "
        "tags) get NOT ASSESSED for that reason alone. A further "
        f"{fv.get('NOT_ASSESSED_NO_VALID_SAMPLES', 0):,d} tags are present in the "
        "instance but entirely NaN across their final window - a sensor that stopped "
        "mid-record, which is neither an absent variable nor an assessable one. The "
        "backbone is 48 h long and never hits either case."
    )
    add(
        "6. **A tag frozen for its whole life carries no usable history.** "
        f"{wl['n_frozen_whole_life']:,d} measurement tags "
        f"({_pct(wl['n_frozen_whole_life'], wl['n_non_mode_tags'])}) hold one value "
        "across the entire archive, so their own p99 of change intervals is empty and "
        "their reference distinct counts are all 1. Self-referencing thresholds "
        "cannot reach them; catching these needs a cross-instance "
        "baseline per (well, variable), which is stage-4 work this study did not do."
    )
    add(
        "7. **Stillness means something else on a state tag.** 3W ships "
        f"{ar['n_mode_tags']:,d} `role=MODE` tags - valve states and the two label "
        "columns. The first run of this study reported many of them as FIRED, which "
        "read as a fault and was really a settled unit holding a state. The backbone "
        "has MODE tags too, but they cycle on a schedule the generator wrote, so they "
        "never sit still long enough for the question to arise."
    )
    add("")
    add(
        "The study also surfaced five library defects. `reference_history` built its "
        "flatline thresholds with `good_mask(chunk)` and never passed the tag's "
        "declared `quality_codes`, so every 3W reference window contained zero GOOD "
        "samples and every verdict came back `no flatline` from an empty reference. "
        "The alias table shipped without `Pa`. The physics block reported coverage but "
        "no valid fraction, so a column of NaNs read as fully covered with nothing "
        "beside it to say otherwise. And the read path walked its windows a row at a "
        "time in Python, which put the full real set out of reach: vectorising it took "
        "one 21,474-row archive from 1.18 s to 0.027 s and is why this run covers all "
        f"{conv['n_instances']:,d} instances rather than a subset. And "
        "`reference_history` timed its change intervals from the wrong pair of "
        "samples, which is the defect the flatline section above is about. All five "
        "are fixed; the reference-case and benchmark outputs are byte-identical "
        "across the quality, performance and interval fixes, because the synthetic "
        "archives declare no `quality_codes`, the vectorised path computes the same "
        "numbers, and the synthetic backbone changes at every 60 s sample so its "
        "change interval and its sample spacing are the same number."
    )
    add("")

    add("## Why no detector numbers")
    add("")
    add(
        "3W instances are grouped by well, and the same well appears in many instances "
        "and many folders. Any detector number computed without holding whole wells out "
        "would be measuring memorisation of a well's own signature. "
        "`src/tsdive/data/splits.py` "
        "slices rows inside each stratum, so a well lands on both sides of the cut; it "
        "has no group-holdout mode, so **group "
        "holdout is the gate on any stage-3-and-above number over 3W**. Until it exists, "
        "this study publishes counts only."
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
    add("| `results/run.json` | subset selection, window size, wall time (committed) |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=str(HERE / "results"))
    parser.add_argument("--archives", default=str(ROOT / "data" / "3w_archives"))
    parser.add_argument("--src", default=str(ROOT / "data" / "3w"))
    args = parser.parse_args(argv)

    results = Path(args.results)
    rows = load_rows(results / "per_tag.csv")
    run = json.loads((results / "run.json").read_text("utf-8"))
    instances = load_instances(Path(args.archives))

    summary = build_summary(rows, instances, run, Path(args.src))
    (results / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    out = HERE / "out"
    out.mkdir(parents=True, exist_ok=True)
    plot_absent(instances, out / "01_absent_variables.png")
    plot_verdicts(rows, out / "02_flatline_verdicts.png")
    plot_extents(instances, run["window_s"], out / "03_instance_extents.png")

    (HERE / "REPORT.md").write_text(
        render_report(summary, rows, instances, run), encoding="utf-8", newline="\n"
    )
    print(f"wrote {HERE / 'REPORT.md'}, {results / 'summary.json'} and 3 figures")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Report and BENCHMARKS.md section for the 3W audit-effect study.

Reads ``results/`` as ``run_audit_effect.py`` wrote it and renders
``REPORT.md`` and ``results/benchmarks_section.md``. With
``--update-benchmarks`` the section replaces the
``## REAL: 3W audit effect (the checks the detectors run)`` section of the
repository's ``BENCHMARKS.md``; the SYNTHETIC table and the other REAL
sections are left as they are. Every number in the prose is read from the
frames the tables are built from.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))

import run_audit_effect as ra  # noqa: E402

BENCH_MD = ROOT / "BENCHMARKS.md"
SECTION_HEADING = "## REAL: 3W audit effect (the checks the detectors run)"
DATASET = "3W v2.0.0 real `14bc1397d3ee`"
BENCH_LABELS = {
    ra.TOOL_SCREEN: "MAD screen, max robust z",
    ra.TOOL_SPC: "SPC individuals rule hits",
    ra.TOOL_CLOCK: "**clock control** (window position in its own instance, no sensor read)",
}
STAGES = {ra.TOOL_SCREEN: "3", ra.TOOL_SPC: "5", ra.TOOL_CLOCK: "none"}


# ---------------------------------------------------------------- formatting


def pct(value) -> str:
    return "refused" if pd.isna(value) else f"{float(value) * 100:.1f}%"


def signed(value) -> str:
    return "" if pd.isna(value) else f"{float(value):+.3f}"


def num(value, digits: int = 3) -> str:
    return "refused" if pd.isna(value) else f"{float(value):.{digits}f}"


def count(value) -> str:
    return "0" if pd.isna(value) else f"{int(value)}"


def table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(out)


def unwrap(text: str) -> str:
    """Join each prose paragraph onto one line, as the other reports carry it."""
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


# ---------------------------------------------------------------- loading


def load(results: Path) -> dict:
    return {
        "summary": pd.read_csv(results / "summary.csv"),
        "pairs": pd.read_csv(results / "pairs.csv"),
        "per_record": pd.read_csv(results / "per_record.csv"),
        "run": json.loads((results / "run.json").read_text("utf-8")),
    }


def row(data: dict, variant: str, detector: str, group: str) -> pd.Series:
    frame = data["summary"]
    hit = frame[
        (frame["variant"] == variant)
        & (frame["detector"] == detector)
        & (frame["group"] == group)
    ]
    return hit.iloc[0]


# ---------------------------------------------------------------- tables


def pairs_table(data: dict) -> str:
    rows = []
    for variant in ra.VARIANTS:
        part = data["pairs"][data["pairs"]["variant"] == variant]
        status = part["status"].value_counts()
        rows.append(
            [
                variant,
                str(len(part)),
                str(int(status.get(ra.STATUS_KEPT, 0) + status.get(ra.STATUS_ZERO_SCALE_KEPT, 0))),
                str(int(status.get(ra.STATUS_ZERO_SCALE, 0))),
                str(int(status.get(ra.STATUS_QUALITY, 0))),
                str(int(status.get(ra.STATUS_NO_FINITE, 0))),
            ]
        )
    return table(
        [
            "variant",
            "pairs",
            "scored",
            "dropped, zero scale",
            "dropped, InsufficientQuality",
            "no finite baseline value",
        ],
        rows,
    )


def zero_scale_table(data: dict) -> str:
    part = data["pairs"]
    zero = part[
        (part["variant"] == ra.VARIANT_CHECKED) & (part["status"] == ra.STATUS_ZERO_SCALE)
    ]
    rows = []
    for variable, group in zero.groupby("variable", sort=True):
        frozen = int((group["baseline_distinct"] == 1).sum())
        rows.append(
            [
                f"`{variable}`",
                str(len(group)),
                str(frozen),
                str(len(group) - frozen),
            ]
        )
    rows.append(
        [
            "all",
            str(len(zero)),
            str(int((zero["baseline_distinct"] == 1).sum())),
            str(int((zero["baseline_distinct"] > 1).sum())),
        ]
    )
    return table(
        [
            "variable",
            "pairs with a zero scale",
            "one distinct baseline value",
            "more than one",
        ],
        rows,
    )


def ranking_table(data: dict) -> str:
    rows = []
    for detector in ra.TOOLS:
        for variant in ra.VARIANTS:
            r = row(data, variant, detector, "pooled")
            rows.append(
                [
                    BENCH_LABELS[detector].replace("**", ""),
                    variant,
                    num(r["roc_auc"], 4),
                    count(r["n_windows_scored"]),
                    count(r["n_windows_infinite"]),
                ]
            )
    return table(
        ["detector", "variant", "AUC", "stream windows scored", "windows scoring +inf"],
        rows,
    )


def normal_table(data: dict) -> str:
    rows = []
    for detector in ra.TOOLS:
        for variant in ra.VARIANTS:
            r = row(data, variant, detector, ra.GROUP_NORMAL)
            rows.append(
                [
                    BENCH_LABELS[detector].replace("**", ""),
                    variant,
                    pct(r["far_window"]),
                    signed(r["over_floor"]),
                    pct(r["far_record"]),
                    count(r["n_instances_infinite_threshold"]),
                ]
            )
    return table(
        [
            "detector",
            "variant",
            "false-alarm rate per window",
            "over the 25.0% floor",
            "instances raising an alarm",
            "instances with an infinite threshold",
        ],
        rows,
    )


def mixed_table(data: dict) -> str:
    rows = []
    for detector in ra.TOOLS:
        for variant in ra.VARIANTS:
            r = row(data, variant, detector, ra.GROUP_MIXED)
            rows.append(
                [
                    BENCH_LABELS[detector].replace("**", ""),
                    variant,
                    pct(r["detect_rate"]),
                    num(r["median_delay_windows"], 1),
                    pct(r["far_window"]),
                    signed(r["over_floor"]),
                ]
            )
    return table(
        [
            "detector",
            "variant",
            "detect",
            "median delay (windows)",
            "false-alarm rate before the first fault window",
            "over the 25.0% floor",
        ],
        rows,
    )


def population_table(data: dict) -> str:
    population = data["run"]["frozen_population"]
    rows = [
        ["measurement tags in the profile", str(population["n_measurement_tags"])],
        [
            "tags holding one distinct value over the whole archive",
            str(population["n_frozen_whole_archive"]),
        ],
        [
            "of those, on the six common variables",
            str(population["n_frozen_on_common_variables"]),
        ],
        [
            "of those, in an instance the window cache holds",
            str(population["n_frozen_in_cached_instances"]),
        ],
        [
            "of those, in an instance this design scores",
            str(population["n_frozen_in_scored_instances"]),
        ],
    ]
    return table(["population", "tags"], rows)


# ---------------------------------------------------------------- report


def render_report(data: dict) -> str:
    run = data["run"]
    k = run["parameters"]["k_baseline_windows"]
    cache = run["cache"]
    groups = run["group_counts"]
    population = run["frozen_population"]
    pairs = data["pairs"][data["pairs"]["variant"] == ra.VARIANT_CHECKED]
    n_pairs = len(pairs)
    n_zero = int((pairs["status"] == ra.STATUS_ZERO_SCALE).sum())
    n_quality = int((pairs["status"] == ra.STATUS_QUALITY).sum())
    n_frozen = int(
        ((pairs["status"] == ra.STATUS_ZERO_SCALE) & (pairs["baseline_distinct"] == 1)).sum()
    )
    n_lattice = n_zero - n_frozen
    partial = pairs[
        (pairs["n_finite_baseline"] > 0)
        & (pairs["n_finite_baseline"] < cache["subgroup_cols"] * k)
    ]
    n_partial = len(partial)
    smallest_partial = int(partial["n_finite_baseline"].min()) if n_partial else 0
    screen_c = row(data, ra.VARIANT_CHECKED, ra.TOOL_SCREEN, "pooled")
    screen_u = row(data, ra.VARIANT_UNCHECKED, ra.TOOL_SCREEN, "pooled")
    spc_c = row(data, ra.VARIANT_CHECKED, ra.TOOL_SPC, "pooled")
    spc_u = row(data, ra.VARIANT_UNCHECKED, ra.TOOL_SPC, "pooled")
    clock_c = row(data, ra.VARIANT_CHECKED, ra.TOOL_CLOCK, "pooled")
    screen_nc = row(data, ra.VARIANT_CHECKED, ra.TOOL_SCREEN, ra.GROUP_NORMAL)
    screen_nu = row(data, ra.VARIANT_UNCHECKED, ra.TOOL_SCREEN, ra.GROUP_NORMAL)
    spc_nc = row(data, ra.VARIANT_CHECKED, ra.TOOL_SPC, ra.GROUP_NORMAL)
    spc_nu = row(data, ra.VARIANT_UNCHECKED, ra.TOOL_SPC, ra.GROUP_NORMAL)
    screen_mc = row(data, ra.VARIANT_CHECKED, ra.TOOL_SCREEN, ra.GROUP_MIXED)
    screen_mu = row(data, ra.VARIANT_UNCHECKED, ra.TOOL_SCREEN, ra.GROUP_MIXED)
    spc_mc = row(data, ra.VARIANT_CHECKED, ra.TOOL_SPC, ra.GROUP_MIXED)
    spc_mu = row(data, ra.VARIANT_UNCHECKED, ra.TOOL_SPC, ra.GROUP_MIXED)

    return f"""# What the zero-scale and quality checks change on 3W

## Summary

`own_history_baselines` drops an (instance, variable) pair when
`mad_baseline` raises `InsufficientQuality` or when the scale it returns is
zero. On the 3W own-history cache the zero-scale check drops {n_zero} of the
{n_pairs} pairs and the quality check drops {n_quality}. Keeping those pairs
and scoring a zero scale the way naive code does moves the pooled MAD screen
AUC from {num(screen_c["roc_auc"], 4)} to {num(screen_u["roc_auc"], 4)} and
the SPC AUC from {num(spc_c["roc_auc"], 4)} to {num(spc_u["roc_auc"], 4)},
against a clock control at {num(clock_c["roc_auc"], 4)}. It also gives
{count(screen_u["n_instances_infinite_threshold"])} of the
{count(screen_u["n_instances_with_threshold"])} instances an infinite screen
threshold, which is why the false-alarm rate on instances with no fault
window falls from {pct(screen_nc["far_window"])} to
{pct(screen_nu["far_window"])} while detection on the instances whose fault
arrives later falls from {pct(screen_mc["detect_rate"])} to
{pct(screen_mu["detect_rate"])}. {population["n_frozen_on_common_variables"]}
of the {population["n_frozen_whole_archive"]} tags that hold one value across
their whole archive sit on the six variables this cache carries.

## Data

- 3W v2.0.0, dataset manifest sha256 `{run["dataset_manifest_sha256"]}`.
- Window cache `{cache["cache"]}`, manifest `{cache["manifest_sha256"]}`,
  {cache["n_instances"]} instances and {cache["n_windows"]} one-hour windows,
  each holding {cache["subgroup_cols"]} one-minute sub-group medians for the
  six variables every instance shares.
- Groups by label, the same split the conformal study uses:
  {groups[ra.GROUP_NORMAL]} instances with every window labelled 0,
  {groups[ra.GROUP_MIXED]} with a label-0 prefix and a fault later,
  {groups[ra.GROUP_POSITIVE_BASELINE]} with a fault row inside the baseline
  windows and {groups[ra.GROUP_SHORT]} with {k} windows or fewer, which the
  design refuses.
- The frozen-tag population is read from
  `{population["per_tag"]}`, the profile study's per-tag table.

Nothing is fitted across instances, so no group holdout applies. The well is
carried in every result row, so a later reader can group by it.

## Method

Run the study and render this report:

```
uv run python examples/studies/3w_audit_effect/run_audit_effect.py
uv run python examples/studies/3w_audit_effect/make_report.py --update-benchmarks
```

Each instance's first {k} windows are its baseline, chosen by position and
never by label, and the rest are its stream. A pair's centre and scale come
from `mad_baseline` over that instance's baseline sub-group medians, the
history `own_history_baselines` builds. The MAD screen scores a window by the
largest robust z of the window medians over the usable variables; the SPC
chart scores it by the rule hits `apply_rules` returns over the sub-group
medians, summed over the usable variables. The clock control scores the
window's position in its own instance.

Two variants share that code path.

- `checked`: the shipped behaviour. A pair whose baseline raises
  `InsufficientQuality`, or whose scale is zero, is dropped, and a window
  with no usable variable is refused.
- `unchecked`: every pair with at least one finite baseline value is kept.
  Where the scale is zero the MAD screen scores 0 for a window median equal
  to the centre and `+inf` for any other, and the SPC limits collapse onto
  the centre, so every sub-group median that differs from it counts as a rule
  hit. A pair that raises `InsufficientQuality` keeps the median and MAD its
  finite values give.

The alarm threshold is the largest score the detector gives that instance's
own baseline windows, which is in sample by construction, and `fires` is
strict, so `far_floor({k})` = {pct(screen_nc["far_floor"])} is the rate a
score that reads nothing already reaches. For the ranking metrics a `+inf`
score is replaced by one step above the largest finite score, so every
infinite window shares the top rank.

The cache holds a historian null sentinel that no float32 can carry, and
the study blanks it before either variant runs:
{cache["masked_beyond_float32"]["subgroups"]["n_cells"]} sub-group cells over
{cache["masked_beyond_float32"]["subgroups"]["n_rows"]} rows, and
{cache["masked_beyond_float32"]["features"]["n_cells"]} window medians. Both
variants get the same mask, so it is outside the difference the study
measures.

## Results

### What the checks drop

{pairs_table(data)}

The quality check drops {n_quality} pairs. A pair on this cache either
holds all {cache["subgroup_cols"] * k} baseline minutes, or none at all, or
one of the {n_partial} partial baselines, and the smallest of those holds
{smallest_partial} finite minutes against the 30 `mad_baseline` requires.

{zero_scale_table(data)}

{n_frozen} of the {n_zero} zero-scale pairs hold one distinct value across
their baseline windows and {n_lattice} hold more than one, which is a tag
rounded to a lattice coarse enough that half its baseline minutes land on the
median.

### Ranking, pooled over every scored stream window

{ranking_table(data)}

### Instances with no fault window

{normal_table(data)}

### Instances whose fault arrives later

{mixed_table(data)}

### The frozen-tag population

{population_table(data)}

## Discussion

The MAD screen loses {num(float(screen_c["roc_auc"]) - float(screen_u["roc_auc"]), 4)}
of AUC when the checks come off, {num(screen_c["roc_auc"], 4)} to
{num(screen_u["roc_auc"], 4)}, and its alarm rule changes shape.
{count(screen_u["n_instances_infinite_threshold"])} of the
{count(screen_u["n_instances_with_threshold"])} scored instances take an
infinite threshold from their own baseline, so nothing they score afterwards
can exceed it. Detection on the instances whose fault arrives later drops
from {pct(screen_mc["detect_rate"])} to {pct(screen_mu["detect_rate"])}, and
the false-alarm rate on instances with no fault window drops from
{pct(screen_nc["far_window"])} to {pct(screen_nu["far_window"])} because
{count(screen_nu["n_instances_infinite_threshold"])} of those
{count(screen_nu["n_instances_with_threshold"])} instances can no longer
raise an alarm at all.

The SPC chart moves less: AUC {num(spc_c["roc_auc"], 4)} to
{num(spc_u["roc_auc"], 4)}, detection {pct(spc_mc["detect_rate"])} to
{pct(spc_mu["detect_rate"])}, and the false-alarm rate on instances with no
fault window rises from {pct(spc_nc["far_window"])} to
{pct(spc_nu["far_window"])}. Collapsed limits produce a finite hit count
rather than an infinity, so a zero-scale pair adds hits to the baseline
windows and the stream windows alike and the threshold moves with them.

The clock control does not move at all: {num(clock_c["roc_auc"], 4)} under
both variants, above the MAD screen's {num(screen_c["roc_auc"], 4)} either
way. The checks change what the detectors read. They leave the position
signal this design carries where it was.

The population row bounds how far this reaches. Of the
{population["n_frozen_whole_archive"]} measurement tags that hold one value
across their whole archive,
{population["n_frozen_on_common_variables"]} sit on the six common variables,
{population["n_frozen_in_cached_instances"]} in an instance the cache holds
and {population["n_frozen_in_scored_instances"]} in an instance this design
scores. `P-PDG` carries {population["per_variable"]["P-PDG"]} of them.

## Limits and further work

- The study measures two checks. It says nothing about the float32 mask, the
  window builder's own refusals, or the profile's flatline verdict, which the
  detector study does not consult.
- The `unchecked` SPC rule count is an equivalent, not a call into
  `apply_rules` with zero limits: a zero sigma makes the three-sigma band a
  point, and the study counts the sub-group medians that differ from the
  centre.
- The alarm rule reads each instance's own baseline scores as its threshold,
  which is in sample. That is what an alarm limit set during a learning
  period is, and it is what makes the false-alarm rate readable, but it is
  not a held-out threshold.
- The frozen-tag population comes from the profile study's table, which
  covers {population["n_measurement_tags"]} measurement tags across all real
  instances. The overlap counted here is by instance and variable, not by the
  window range this cache cut.

## Files

| file | holds |
|---|---|
| `run_audit_effect.py` | the study runner |
| `make_report.py` | this report and the BENCHMARKS section |
| `results/pairs.csv` | one row per (instance, variable, variant) |
| `results/per_record.csv` | the alarm rule and its metrics per instance |
| `results/summary.csv` | one row per (detector, variant, group) |
| `results/run.json` | manifests, code version, parameters, counts, the frozen-tag population |
| `results/benchmarks_section.md` | the BENCHMARKS.md REAL section |
"""


# ---------------------------------------------------------------- benchmarks


def render_benchmarks_section(data: dict) -> str:
    run = data["run"]
    code = f"tsdive {run['code']['tsdive_version']} @ `{run['code']['git_head'][:8]}`"
    k = run["parameters"]["k_baseline_windows"]
    cache = run["cache"]
    population = run["frozen_population"]
    groups = run["group_counts"]
    split = (
        f"own history, first {k} windows per instance, label-blind; no group "
        f"holdout; {groups[ra.GROUP_NORMAL]} instances with no fault window, "
        f"{groups[ra.GROUP_MIXED]} whose fault arrives later, "
        f"{groups[ra.GROUP_SHORT]} refused as too short; cache manifest "
        f"`{cache['manifest_sha256']}`"
    )
    rows: list[tuple[str, str, str]] = []
    pairs = data["pairs"][data["pairs"]["variant"] == ra.VARIANT_CHECKED]
    n_zero = int((pairs["status"] == ra.STATUS_ZERO_SCALE).sum())
    n_frozen = int(
        ((pairs["status"] == ra.STATUS_ZERO_SCALE) & (pairs["baseline_distinct"] == 1)).sum()
    )
    n_quality = int((pairs["status"] == ra.STATUS_QUALITY).sum())
    rows.append(
        (
            "none",
            "(instance, variable) pairs the own-history checks drop",
            f"zero scale {n_zero} of {len(pairs)} ({n_frozen} hold one distinct "
            f"baseline value, {n_zero - n_frozen} are rounded to a lattice); "
            f"InsufficientQuality {n_quality}",
        )
    )
    rows.append(
        (
            "none",
            "tags holding one distinct value over the whole archive, reachable by this cache",
            f"{population['n_frozen_whole_archive']} of "
            f"{population['n_measurement_tags']} measurement tags; "
            f"{population['n_frozen_on_common_variables']} on the six common "
            f"variables, {population['n_frozen_in_scored_instances']} in an "
            "instance this design scores",
        )
    )
    for detector in ra.TOOLS:
        for variant in ra.VARIANTS:
            pooled = row(data, variant, detector, "pooled")
            normal = row(data, variant, detector, ra.GROUP_NORMAL)
            mixed = row(data, variant, detector, ra.GROUP_MIXED)
            rows.append(
                (
                    STAGES[detector],
                    f"{BENCH_LABELS[detector]}, {variant} baselines",
                    f"AUC {num(pooled['roc_auc'], 4)} over "
                    f"{count(pooled['n_windows_scored'])} stream windows "
                    f"({count(pooled['n_windows_infinite'])} scoring +inf); FAR on "
                    f"instances with no fault window {pct(normal['far_window'])} "
                    f"({signed(normal['over_floor'])} over the "
                    f"{pct(normal['far_floor'])} floor), "
                    f"{count(normal['n_instances_infinite_threshold'])} of them with "
                    f"an infinite threshold; detect {pct(mixed['detect_rate'])}, "
                    f"median delay {num(mixed['median_delay_windows'], 1)} windows",
                )
            )
    lines = [
        SECTION_HEADING,
        "",
        "The own-history design run twice over the same cache and one code path. "
        "`checked` is the shipped behaviour, where `own_history_baselines` drops "
        "an (instance, variable) pair whose baseline raises "
        "`InsufficientQuality` or whose MAD scale is zero. `unchecked` keeps "
        "every pair and uses a zero scale the way naive code does: the MAD "
        "screen returns `+inf` for any window median off the centre, and the SPC "
        "limits collapse onto the centre. Method, groups and refusals are in "
        "[examples/studies/3w_audit_effect/REPORT.md]"
        "(examples/studies/3w_audit_effect/REPORT.md).",
        "",
        "This section is not regenerated by `make bench`; it is produced by "
        "`examples/studies/3w_audit_effect/run_audit_effect.py` and "
        "`make_report.py`, and carried through the generator unchanged.",
        "",
        "| stage | metric | value | dataset (manifest sha256) | code | design | split |",
        "|---|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {stage} | {metric} | {value} | {DATASET} | {code} | own-history | {split} |"
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
    parser.add_argument("--report", default=str(HERE / "REPORT.md"))
    parser.add_argument("--update-benchmarks", action="store_true")
    parser.add_argument("--benchmarks", default=str(BENCH_MD))
    args = parser.parse_args(argv)

    results = Path(args.results)
    data = load(results)
    Path(args.report).write_text(
        unwrap(render_report(data)), encoding="utf-8", newline="\n"
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

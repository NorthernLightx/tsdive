"""Window + label builder for the 3W detector study.

Cuts every real 3W instance into non-overlapping fixed-length windows,
labels each window from the per-row `class` tag, and extracts one honest
feature row per (window, variable) through the shipped
:func:`tsdive.features.extract`.

Three decisions this file makes, and why:

- **The unlabelled prefix is dropped, not filled.** Every real instance
  opens with 3,600 rows whose `class` is null. A window overlapping any
  of them would carry a label invented by this script, so those windows
  are dropped and counted.
- **A window is positive when any row inside it carries a fault class.**
  Classes 1-9 are steady faults, 101-109 the transient run-up to the same
  fault. Both count positive; a window whose only fault rows are
  transient is flagged ``transient_only`` so the metric can be recomputed
  without them.
- **Windows are half-open in effect.** ``TagStore.read_window`` is
  inclusive at both ends, so the requested end is one sample short of the
  next window's start. At 3W's strict 1 Hz that is exactly ``window_s``
  samples per window and no sample counted twice.

Alongside the feature table it caches a sub-group median grid: each
window is cut into ``--subgroups`` equal slices and the median of the
GOOD values in each slice is stored. That grid is what the SPC and MSPC
tools read at scoring time, so the raw 1 Hz archives are walked once for
the whole study rather than once per fold.

``--aligned`` builds a second, much smaller cache in the same three-file
shape, cut relative to each instance's fault onset instead of its first
sample. Every evaluable instance contributes the identical six windows -
``k`` of baseline, one *pre*, a skipped guard hour, then ``post0`` and
``post1`` - so no instance's fault arrives earlier in the evaluation than
another's, and a control reading only the clock has nothing left to read.
It needs ``onsets.csv`` from ``build_onsets.py``.

    uv run python examples/studies/3w_detectors/build_windows.py --jobs 14
    uv run python examples/studies/3w_detectors/build_windows.py --aligned --jobs 14
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[3] / "src"))

import tsdive
from tsdive.features import extract
from tsdive.store.identity import TagIdentity
from tsdive.store.quality import good_mask
from tsdive.store.sampling_contract import (
    AggregateType,
    CalculationBasis,
    RetrievalMode,
    SamplingContract,
)
from tsdive.store.tagstore import TagStore

HERE = Path(__file__).parent

# 3W codes the transient run-up to fault F as F+100, so any class outside
# {0} and outside the null prefix is a fault of some kind.
STEADY_FAULT_CLASSES = frozenset(range(1, 10))
TRANSIENT_FAULT_CLASSES = frozenset(range(101, 110))

FEATURE_FIELDS = (
    "n_samples",
    "n_good",
    "coverage",
    "weighted_mean",
    "std",
    "min",
    "max",
    "median",
    "mad",
    "distinct_count",
    "stall_s",
    "changes_per_hour",
    "clipped_fraction",
    "censored",
)


def contract() -> SamplingContract:
    """3W is a recorded 1 Hz archive; nothing here is an aggregate."""
    return SamplingContract(
        calculation_basis=CalculationBasis.TIME_WEIGHTED,
        retrieval_mode=RetrievalMode.RECORDED,
        aggregate_type=AggregateType.NONE,
        stepped=True,
    )


def instance_records(root: Path) -> list[dict]:
    return [
        json.loads((d / "INSTANCE.json").read_text("utf-8"))
        for d in sorted(root.iterdir())
        if (d / "INSTANCE.json").exists()
    ]


def common_variables(records: list[dict], min_share: float) -> list[str]:
    """Measurement variables present in at least ``min_share`` of instances.

    Presence is read off each instance's own ``present_variables`` list -
    the converter's record of which columns carried a value - never off
    the 27-column schema, which lists variables no real well ever
    instrumented. ``ESTADO-*`` and the two ``LABEL_*`` tags are
    ``role=MODE``: their values are state names, so the numeric features
    below are refused for them and they are not candidates here.
    """
    n = len(records)
    counts: dict[str, int] = {}
    for record in records:
        for variable in record["present_variables"]:
            counts[variable] = counts.get(variable, 0) + 1
    return sorted(
        v
        for v, c in counts.items()
        if c / n >= min_share and not v.startswith(("ESTADO-", "LABEL_"))
    )


def _class_ints(frame: pd.DataFrame) -> pd.Series:
    """Per-row class as a nullable integer, nulls preserved."""
    return pd.to_numeric(frame["value"], errors="coerce").astype("Int64")


def _subgroup_medians(
    values: pd.Series, stamps: pd.Series, start: pd.Timestamp, span_s: float, k: int
) -> list[float]:
    """Median of the GOOD values in each of ``k`` equal slices of the window."""
    if len(values) == 0:
        return [float("nan")] * k
    offsets = (stamps - start).dt.total_seconds().to_numpy(dtype=float)
    slot = np.clip((offsets / (span_s / k)).astype(int), 0, k - 1)
    arr = values.to_numpy(dtype=float)
    out = [float("nan")] * k
    order = np.argsort(slot, kind="stable")
    slot_sorted, arr_sorted = slot[order], arr[order]
    bounds = np.searchsorted(slot_sorted, np.arange(k + 1))
    for i in range(k):
        chunk = arr_sorted[bounds[i] : bounds[i + 1]]
        chunk = chunk[np.isfinite(chunk)]
        if len(chunk):
            out[i] = float(np.median(chunk))
    return out


def _extract_window(
    store: TagStore,
    instance: str,
    variables: list[str],
    present: set[str],
    w_start: pd.Timestamp,
    w_end: pd.Timestamp,
    idx: int,
    window_s: float,
    k: int,
) -> tuple[list[dict], list[dict]]:
    """One feature row and one sub-group row per present variable."""
    feature_rows: list[dict] = []
    grid_rows: list[dict] = []
    for variable in variables:
        if variable not in present:
            continue
        window = store.read_window(
            TagIdentity(source_id=instance, point_id=variable),
            w_start.to_pydatetime(),
            w_end.to_pydatetime(),
            contract(),
        )
        features = extract(window)
        row = {"instance": instance, "window_index": idx, "variable": variable}
        full = features.to_row()
        row.update({field: full[field] for field in FEATURE_FIELDS})
        row["notes"] = full["notes"]
        feature_rows.append(row)

        good = window.frame[good_mask(window.frame, window.meta.quality_codes)]
        good = good[good["value"].notna()]
        medians = _subgroup_medians(good["value"], good["timestamp"], w_start, window_s, k)
        grid_rows.append(
            {
                "instance": instance,
                "window_index": idx,
                "variable": variable,
                **{f"s{i:02d}": medians[i] for i in range(k)},
            }
        )
    return feature_rows, grid_rows


def build_instance(payload: dict) -> dict:
    """Window one instance: label rows, extract features, grid sub-groups."""
    instance_dir = Path(payload["instance_dir"])
    window_s = float(payload["window_s"])
    k = int(payload["subgroups"])
    variables = list(payload["variables"])
    record = json.loads((instance_dir / "INSTANCE.json").read_text("utf-8"))
    store = TagStore(instance_dir.parent)
    instance = record["instance"]

    labels = pd.read_parquet(instance_dir / "LABEL_class.parquet")
    states = pd.read_parquet(instance_dir / "LABEL_state.parquet")
    classes = _class_ints(labels)
    labelled = labels.loc[classes.notna(), "timestamp"]
    first = pd.Timestamp(record["first_timestamp_utc"])
    last = pd.Timestamp(record["last_timestamp_utc"])
    # The label tag's own first non-null row, not a row count: the prefix
    # length is a property of the data, and reading it here means a file
    # whose prefix differs from INSTANCE.json is followed, not assumed.
    labelled_from = pd.Timestamp(labelled.iloc[0]) if len(labelled) else None

    present = set(record["present_variables"])
    span = pd.Timedelta(window_s, unit="s")
    one = pd.Timedelta(1, unit="s")

    feature_rows: list[dict] = []
    grid_rows: list[dict] = []
    windows: list[dict] = []
    n_dropped_prefix = 0
    n_dropped_short = 0
    index = 0
    cursor = first
    while cursor + span <= last + one:
        w_start, w_end = cursor, cursor + span - one
        cursor = cursor + span
        idx, index = index, index + 1
        if labelled_from is None or w_start < labelled_from:
            n_dropped_prefix += 1
            continue
        in_window = (labels["timestamp"] >= w_start) & (labels["timestamp"] <= w_end)
        window_classes = classes[in_window].dropna().astype(int)
        if len(window_classes) == 0:
            n_dropped_short += 1
            continue
        seen = set(window_classes.unique().tolist())
        steady = bool(seen & STEADY_FAULT_CLASSES)
        transient = bool(seen & TRANSIENT_FAULT_CLASSES)
        window_state = states.loc[in_window, "value"].dropna()
        windows.append(
            {
                "instance": instance,
                "well": record["well"] or record["instance_source"],
                "folder_label": record["folder_label"],
                "window_index": idx,
                "window_start": w_start.isoformat(),
                "window_end": w_end.isoformat(),
                "label": int(steady or transient),
                "transient_only": bool(transient and not steady),
                "classes": ";".join(str(c) for c in sorted(seen)),
                "state_mode": (
                    str(window_state.mode().iloc[0]) if len(window_state) else ""
                ),
                "n_state_values": int(window_state.nunique()),
                "n_variables_present": sum(1 for v in variables if v in present),
            }
        )
        f_rows, g_rows = _extract_window(
            store, instance, variables, present, w_start, w_end, idx, window_s, k
        )
        feature_rows.extend(f_rows)
        grid_rows.extend(g_rows)
    return {
        "windows": windows,
        "features": feature_rows,
        "grid": grid_rows,
        "n_dropped_prefix": n_dropped_prefix,
        "n_dropped_unlabelled": n_dropped_short,
    }


# ------------------------------------------------------ onset-aligned design


def aligned_offsets(k: int) -> list[tuple[int, str, int]]:
    """(offset in windows from onset, role, label the design asserts).

    ``k`` baseline windows, then one *pre* window, then a skipped guard
    window, then the two windows after onset. Offset -1 is absent on
    purpose: a fault that announces itself in the last minutes before its
    first labelled row would otherwise land inside the window the design
    calls negative, and the negative would be doing the detector's work.
    """
    baseline = [(-(k + 2) + i, "baseline", 0) for i in range(k)]
    return [*baseline, (-2, "pre", 0), (0, "post0", 1), (1, "post1", 1)]


def build_aligned_instance(payload: dict) -> dict:
    """Six windows at fixed offsets from this instance's fault onset.

    The label the design asserts is written as ``design_label``; the label
    the rows carry is extracted the same way as in the unaligned build and
    written alongside it, so the two can be compared rather than assumed
    equal.
    """
    instance_dir = Path(payload["instance_dir"])
    window_s = float(payload["window_s"])
    k_sub = int(payload["subgroups"])
    variables = list(payload["variables"])
    onset = pd.Timestamp(payload["t_onset"])
    offsets = aligned_offsets(int(payload["baseline_windows"]))
    record = json.loads((instance_dir / "INSTANCE.json").read_text("utf-8"))
    store = TagStore(instance_dir.parent)
    instance = record["instance"]

    labels = pd.read_parquet(instance_dir / "LABEL_class.parquet")
    states = pd.read_parquet(instance_dir / "LABEL_state.parquet")
    classes = _class_ints(labels)
    first = pd.Timestamp(record["first_timestamp_utc"])
    present = set(record["present_variables"])
    span = pd.Timedelta(window_s, unit="s")
    one = pd.Timedelta(1, unit="s")

    feature_rows: list[dict] = []
    grid_rows: list[dict] = []
    windows: list[dict] = []
    for offset, role, design_label in offsets:
        w_start = onset + pd.Timedelta(window_s * offset, unit="s")
        w_end = w_start + span - one
        in_window = (labels["timestamp"] >= w_start) & (labels["timestamp"] <= w_end)
        window_classes = classes[in_window].dropna().astype(int)
        seen = set(window_classes.unique().tolist())
        steady = bool(seen & STEADY_FAULT_CLASSES)
        transient = bool(seen & TRANSIENT_FAULT_CLASSES)
        window_state = states.loc[in_window, "value"].dropna()
        # The clock control's quantity: how many windows into its own
        # record this window sits, counted from the instance's first
        # sample exactly as the unaligned build counts it. Under the
        # aligned design it is the only time signal a deployed detector
        # could have, because the offset from onset is not knowable
        # until the onset has happened.
        record_index = int((w_start - first).total_seconds() // window_s)
        windows.append(
            {
                "instance": instance,
                "well": record["well"] or record["instance_source"],
                "folder_label": record["folder_label"],
                "window_index": record_index,
                "onset_offset": offset,
                "role": role,
                "design_label": design_label,
                "window_start": w_start.isoformat(),
                "window_end": w_end.isoformat(),
                "label": int(steady or transient),
                "transient_only": bool(transient and not steady),
                "n_labelled_rows": len(window_classes),
                "classes": ";".join(str(c) for c in sorted(seen)),
                "state_mode": str(window_state.mode().iloc[0]) if len(window_state) else "",
                "n_state_values": int(window_state.nunique()),
                "n_variables_present": sum(1 for v in variables if v in present),
            }
        )
        f_rows, g_rows = _extract_window(
            store, instance, variables, present, w_start, w_end, record_index, window_s, k_sub
        )
        feature_rows.extend(f_rows)
        grid_rows.extend(g_rows)
    return {"windows": windows, "features": feature_rows, "grid": grid_rows}


def build_aligned(args, dirs: list[Path], variables: list[str], out_dir: Path) -> int:
    """Cut the aligned cache for every instance the onset table calls evaluable."""
    onsets = pd.read_csv(args.onsets)
    evaluable = onsets[onsets["evaluable"].astype(bool)]
    by_instance = {str(p.name): p for p in dirs}
    payloads = [
        {
            "instance_dir": str(by_instance[str(row["instance"])]),
            "window_s": args.window_s,
            "subgroups": args.subgroups,
            "baseline_windows": args.baseline_windows,
            "variables": variables,
            "t_onset": str(row["t_onset"]),
        }
        for _, row in evaluable.iterrows()
        if str(row["instance"]) in by_instance
    ]
    if not payloads:
        print(f"no evaluable instance in {args.onsets}; run build_onsets.py first")
        return 1

    started = time.perf_counter()
    windows: list[dict] = []
    features: list[dict] = []
    grid: list[dict] = []
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = [pool.submit(build_aligned_instance, p) for p in payloads]
        for done, fut in enumerate(as_completed(futures), start=1):
            result = fut.result()
            windows.extend(result["windows"])
            features.extend(result["features"])
            grid.extend(result["grid"])
            if done % 20 == 0 or done == len(payloads):
                print(f"  {done}/{len(payloads)} instances, {len(windows)} windows", flush=True)
    elapsed = time.perf_counter() - started

    win_df = pd.DataFrame(windows).sort_values(["instance", "onset_offset"])
    feat_df = pd.DataFrame(features).sort_values(["instance", "window_index", "variable"])
    grid_df = pd.DataFrame(grid).sort_values(["instance", "window_index", "variable"])
    out_dir.mkdir(parents=True, exist_ok=True)
    win_df.to_parquet(out_dir / "windows.parquet", index=False)
    feat_df.to_parquet(out_dir / "features.parquet", index=False)
    grid_df.to_parquet(out_dir / "subgroup_medians.parquet", index=False)

    # The design asserts pre is normal and post is faulted. The rows are
    # read anyway and the disagreement counted, because a design label
    # nobody checked against the data is an assumption, not a label.
    test = win_df[win_df["role"] != "baseline"]
    disagree = test[test["design_label"] != test["label"]]
    manifest = {
        "design": "onset-aligned",
        "archives_root": str(args.archives),
        "onsets_csv": str(args.onsets),
        "n_instances": int(win_df["instance"].nunique()),
        "n_wells": int(win_df["well"].nunique()),
        "window_s": args.window_s,
        "subgroups": args.subgroups,
        "baseline_windows": args.baseline_windows,
        "offsets": [
            {"offset": o, "role": r, "design_label": lab}
            for o, r, lab in aligned_offsets(args.baseline_windows)
        ],
        "common_variables": variables,
        "n_windows": len(win_df),
        "n_feature_rows": len(feat_df),
        "n_windows_per_role": {
            str(k): int(v) for k, v in win_df["role"].value_counts().items()
        },
        "row_label_by_role": {
            str(role): {
                "windows": len(g),
                "row_positive": int(g["label"].sum()),
                "row_transient_only": int(g["transient_only"].sum()),
            }
            for role, g in win_df.groupby("role")
        },
        "n_test_windows_whose_rows_disagree_with_the_design_label": len(disagree),
        "disagreeing_windows": [
            {
                "instance": str(r["instance"]),
                "role": str(r["role"]),
                "design_label": int(r["design_label"]),
                "row_label": int(r["label"]),
                "classes": str(r["classes"]),
            }
            for _, r in disagree.iterrows()
        ][:20],
        "per_folder_instances": {
            str(k): int(v)
            for k, v in win_df.drop_duplicates("instance")["folder_label"]
            .astype(str)
            .value_counts()
            .items()
        },
        "wall_seconds": round(elapsed, 1),
        "jobs": args.jobs,
        "tsdive_version": tsdive.__version__,
    }
    (out_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        f"{len(win_df):,d} aligned windows over {manifest['n_instances']} instances "
        f"and {manifest['n_wells']} wells, {len(feat_df):,d} feature rows, {elapsed:.1f}s"
    )
    print(f"per role: {manifest['n_windows_per_role']}")
    print(f"rows disagreeing with the design label: {len(disagree)}")
    print(json.dumps(manifest["row_label_by_role"], indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", default="data/3w_archives")
    parser.add_argument("--out", default=None)
    parser.add_argument("--window-s", type=float, default=3600.0)
    parser.add_argument("--subgroups", type=int, default=60)
    parser.add_argument("--min-share", type=float, default=0.60)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument(
        "--aligned",
        action="store_true",
        help="cut windows at fixed offsets from each instance's fault onset "
        "instead of from its first sample",
    )
    parser.add_argument("--onsets", default="data/3w_windows/onsets.csv")
    parser.add_argument(
        "--baseline-windows",
        type=int,
        default=3,
        help="aligned design: k windows of own-history baseline before the "
        "pre window (default 3 = 3 h)",
    )
    args = parser.parse_args(argv)
    if args.out is None:
        args.out = "data/3w_windows_aligned" if args.aligned else "data/3w_windows"

    root = Path(args.archives)
    dirs = sorted(p for p in root.iterdir() if (p / "INSTANCE.json").exists())
    if args.limit is not None:
        dirs = dirs[: args.limit]
    if not dirs:
        print(f"no converted instances under {root}; run scripts/convert_3w.py first")
        return 1
    records = instance_records(root)
    variables = common_variables(records, args.min_share)
    print(f"common variables (>= {args.min_share:.0%} of {len(records)} instances): {variables}")

    out_dir = Path(args.out)
    if args.aligned:
        return build_aligned(args, dirs, variables, out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    windows: list[dict] = []
    features: list[dict] = []
    grid: list[dict] = []
    dropped_prefix = 0
    dropped_unlabelled = 0
    payloads = [
        {
            "instance_dir": str(d),
            "window_s": args.window_s,
            "subgroups": args.subgroups,
            "variables": variables,
        }
        for d in dirs
    ]
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = [pool.submit(build_instance, p) for p in payloads]
        for done, fut in enumerate(as_completed(futures), start=1):
            result = fut.result()
            windows.extend(result["windows"])
            features.extend(result["features"])
            grid.extend(result["grid"])
            dropped_prefix += result["n_dropped_prefix"]
            dropped_unlabelled += result["n_dropped_unlabelled"]
            if done % 50 == 0 or done == len(payloads):
                print(f"  {done}/{len(payloads)} instances, {len(windows)} windows", flush=True)
    elapsed = time.perf_counter() - started

    win_df = pd.DataFrame(windows).sort_values(["instance", "window_index"])
    feat_df = pd.DataFrame(features).sort_values(["instance", "window_index", "variable"])
    grid_df = pd.DataFrame(grid).sort_values(["instance", "window_index", "variable"])
    win_df.to_parquet(out_dir / "windows.parquet", index=False)
    feat_df.to_parquet(out_dir / "features.parquet", index=False)
    grid_df.to_parquet(out_dir / "subgroup_medians.parquet", index=False)

    positives = int(win_df["label"].sum())
    per_folder = (
        win_df.groupby("folder_label")["label"]
        .agg(["size", "sum"])
        .rename(columns={"size": "windows", "sum": "positive"})
    )
    manifest = {
        "archives_root": str(root),
        "n_instances": len(dirs),
        "window_s": args.window_s,
        "subgroups": args.subgroups,
        "min_share": args.min_share,
        "common_variables": variables,
        "n_windows": len(win_df),
        "n_positive": positives,
        "positive_rate": round(positives / max(len(win_df), 1), 6),
        "n_transient_only": int(win_df["transient_only"].sum()),
        "n_feature_rows": len(feat_df),
        "n_windows_dropped_unlabelled_prefix": dropped_prefix,
        "n_windows_dropped_no_labelled_rows": dropped_unlabelled,
        "per_folder": {
            str(k): {"windows": int(v["windows"]), "positive": int(v["positive"])}
            for k, v in per_folder.to_dict("index").items()
        },
        "per_well_windows": {
            str(k): int(v) for k, v in win_df["well"].value_counts().items()
        },
        "wall_seconds": round(elapsed, 1),
        "jobs": args.jobs,
        "tsdive_version": tsdive.__version__,
    }
    (out_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        f"{len(win_df):,d} windows ({positives:,d} positive, "
        f"{positives / max(len(win_df), 1):.1%}), {len(feat_df):,d} feature rows, "
        f"{elapsed:.1f}s"
    )
    print(per_folder.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())

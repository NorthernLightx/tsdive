"""When each 3W instance's fault starts, and whether it can be scored.

Phase 4 published a clock control - the window's position in its own
instance's record, reading no sensor value - at AUC 0.900. A 3W instance
opens normal and the fault, once it arrives, stays, so *when* a window
happened separates the classes almost as well as *what* the tags did.
The fix is to stop letting the design vary onset time across instances,
and that starts here: one row per instance saying where its fault begins.

``t_onset`` is the timestamp of the first row whose ``class`` is a fault
- 1-9 steady or 101-109 transient. The 3,600-row unlabelled prefix holds
no class at all, so it cannot contain an onset; the first labelled row is
recorded separately because the *pre* window of the aligned design is
asserted negative and may not sit on rows nobody labelled.

An instance with no fault row - folder 0, and the folder-9 instances the
profile study found carrying only class 0 - gets ``t_onset`` empty. It is not an
error and not dropped: it is a negatives-only instance, and the count is
the point.

    uv run python examples/studies/3w_detectors/build_onsets.py --jobs 14
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[3] / "src"))

HERE = Path(__file__).parent

STEADY_FAULT_CLASSES = frozenset(range(1, 10))
TRANSIENT_FAULT_CLASSES = frozenset(range(101, 110))
FAULT_CLASSES = STEADY_FAULT_CLASSES | TRANSIENT_FAULT_CLASSES

COLUMNS = (
    "instance",
    "well",
    "folder_label",
    "t_first",
    "t_labelled_from",
    "t_onset",
    "t_last",
    "onset_class",
    "onset_kind",
    "pre_onset_s",
    "pre_onset_labelled_s",
    "post_onset_s",
    "pre_onset_windows",
    "post_onset_windows",
    "has_k_pre_windows",
    "has_2_post_windows",
    "evaluable",
)


def onset_of(classes: pd.Series, stamps: pd.Series) -> tuple[pd.Timestamp | None, int | None]:
    """First fault row's timestamp and class, or ``(None, None)``.

    Reads the class column as a nullable integer so the unlabelled prefix
    stays null rather than becoming a zero, which would make every
    instance look like it opened in a labelled normal state.
    """
    values = pd.to_numeric(classes, errors="coerce").astype("Int64")
    hit = values.isin(sorted(FAULT_CLASSES)) & values.notna()
    if not bool(hit.any()):
        return None, None
    position = int(hit.to_numpy().argmax())
    return pd.Timestamp(stamps.iloc[position]), int(values.iloc[position])


def instance_onset(payload: dict) -> dict:
    """One row of the onset table for one converted instance."""
    instance_dir = Path(payload["instance_dir"])
    window_s = float(payload["window_s"])
    k = int(payload["baseline_windows"])
    record = json.loads((instance_dir / "INSTANCE.json").read_text("utf-8"))
    labels = pd.read_parquet(instance_dir / "LABEL_class.parquet")

    first = pd.Timestamp(record["first_timestamp_utc"])
    last = pd.Timestamp(record["last_timestamp_utc"])
    values = pd.to_numeric(labels["value"], errors="coerce").astype("Int64")
    labelled = labels.loc[values.notna(), "timestamp"]
    labelled_from = pd.Timestamp(labelled.iloc[0]) if len(labelled) else None
    t_onset, onset_class = onset_of(labels["value"], labels["timestamp"])

    row = {
        "instance": record["instance"],
        "well": record["well"] or record["instance_source"],
        "folder_label": record["folder_label"],
        "t_first": first.isoformat(),
        "t_labelled_from": None if labelled_from is None else labelled_from.isoformat(),
        "t_onset": None,
        "t_last": last.isoformat(),
        "onset_class": None,
        "onset_kind": "none",
        "pre_onset_s": None,
        "pre_onset_labelled_s": None,
        "post_onset_s": None,
        "pre_onset_windows": None,
        "post_onset_windows": None,
        "has_k_pre_windows": False,
        "has_2_post_windows": False,
        "evaluable": False,
    }
    if t_onset is None:
        return row

    pre_s = (t_onset - first).total_seconds()
    pre_labelled_s = (
        None if labelled_from is None else (t_onset - labelled_from).total_seconds()
    )
    post_s = (last - t_onset).total_seconds()
    row.update(
        t_onset=t_onset.isoformat(),
        onset_class=onset_class,
        onset_kind="transient" if onset_class in TRANSIENT_FAULT_CLASSES else "steady",
        pre_onset_s=pre_s,
        pre_onset_labelled_s=pre_labelled_s,
        post_onset_s=post_s,
        pre_onset_windows=int(pre_s // window_s),
        # The archive's last sample is inclusive, so a span one second
        # short of two whole windows still holds two whole windows.
        post_onset_windows=int((post_s + 1.0) // window_s),
    )
    row["has_k_pre_windows"] = bool(row["pre_onset_windows"] >= k)
    row["has_2_post_windows"] = bool(row["post_onset_windows"] >= 2)
    # The aligned design needs k+2 whole windows before onset: k of
    # baseline, one scored *pre* window, and a one-window guard so a
    # fault that announces itself a few minutes early does not land in
    # the window the design calls negative. The pre window must also sit
    # on labelled rows, because the design asserts it is class 0.
    row["evaluable"] = bool(
        row["pre_onset_windows"] >= k + 2
        and row["has_2_post_windows"]
        and pre_labelled_s is not None
        and pre_labelled_s >= 2 * window_s
    )
    return row


def summarise(frame: pd.DataFrame, window_s: float, k: int) -> dict:
    """Counts the report reads back: who has an onset, who can be scored."""
    with_onset = frame[frame["t_onset"].notna()]
    evaluable = frame[frame["evaluable"]]

    def per_folder(sub: pd.DataFrame) -> dict[str, int]:
        return {
            str(key): int(value)
            for key, value in sub["folder_label"].astype(str).value_counts().items()
        }

    return {
        "window_s": window_s,
        "baseline_windows": k,
        "pre_onset_windows_required": k + 2,
        "post_onset_windows_required": 2,
        "n_instances": len(frame),
        "n_with_onset": len(with_onset),
        "n_without_onset": int(len(frame) - len(with_onset)),
        "n_onset_transient": int((with_onset["onset_kind"] == "transient").sum()),
        "n_onset_steady": int((with_onset["onset_kind"] == "steady").sum()),
        "n_with_k_pre_windows": int(frame["has_k_pre_windows"].sum()),
        "n_with_2_post_windows": int(frame["has_2_post_windows"].sum()),
        "n_with_k_pre_and_2_post": int(
            (frame["has_k_pre_windows"] & frame["has_2_post_windows"]).sum()
        ),
        "n_evaluable": len(evaluable),
        "n_evaluable_transient_onset": int((evaluable["onset_kind"] == "transient").sum()),
        "n_evaluable_steady_onset": int((evaluable["onset_kind"] == "steady").sum()),
        "n_evaluable_wells": int(evaluable["well"].nunique()),
        "per_folder": {
            folder: {
                "instances": int(n),
                "with_onset": per_folder(with_onset).get(folder, 0),
                "k_pre_and_2_post": per_folder(
                    frame[frame["has_k_pre_windows"] & frame["has_2_post_windows"]]
                ).get(folder, 0),
                "evaluable": per_folder(evaluable).get(folder, 0),
            }
            for folder, n in sorted(
                per_folder(frame).items(), key=lambda kv: int(kv[0])
            )
        },
        "onset_class_counts": {
            str(int(c)): int(n)
            for c, n in sorted(with_onset["onset_class"].value_counts().items())
        },
        "seconds_to_onset": {
            "median": float(with_onset["pre_onset_s"].median()),
            "p05": float(with_onset["pre_onset_s"].quantile(0.05)),
            "p95": float(with_onset["pre_onset_s"].quantile(0.95)),
        },
        "seconds_after_onset": {
            "median": float(with_onset["post_onset_s"].median()),
            "p05": float(with_onset["post_onset_s"].quantile(0.05)),
            "p95": float(with_onset["post_onset_s"].quantile(0.95)),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", default="data/3w_archives")
    parser.add_argument("--out", default="data/3w_windows")
    parser.add_argument("--results", default=str(HERE / "results"))
    parser.add_argument("--window-s", type=float, default=3600.0)
    parser.add_argument(
        "--baseline-windows",
        type=int,
        default=3,
        help="k: how many windows of own-history baseline the aligned design "
        "takes before onset (default 3 = 3 h)",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args(argv)

    root = Path(args.archives)
    dirs = sorted(p for p in root.iterdir() if (p / "INSTANCE.json").exists())
    if args.limit is not None:
        dirs = dirs[: args.limit]
    if not dirs:
        print(f"no converted instances under {root}; run scripts/convert_3w.py first")
        return 1

    payloads = [
        {
            "instance_dir": str(d),
            "window_s": args.window_s,
            "baseline_windows": args.baseline_windows,
        }
        for d in dirs
    ]
    started = time.perf_counter()
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = [pool.submit(instance_onset, p) for p in payloads]
        for done, fut in enumerate(as_completed(futures), start=1):
            rows.append(fut.result())
            if done % 200 == 0 or done == len(payloads):
                print(f"  {done}/{len(payloads)} instances", flush=True)
    elapsed = time.perf_counter() - started

    frame = pd.DataFrame(rows, columns=list(COLUMNS)).sort_values("instance")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / "onsets.csv", index=False, lineterminator="\n")

    summary = summarise(frame, args.window_s, args.baseline_windows)
    summary["wall_seconds"] = round(elapsed, 1)
    results = Path(args.results)
    results.mkdir(parents=True, exist_ok=True)
    (results / "onsets.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        f"{summary['n_instances']:,d} instances, {summary['n_with_onset']:,d} with an "
        f"onset, {summary['n_evaluable']:,d} evaluable "
        f"({summary['n_evaluable_wells']} wells), {elapsed:.1f}s"
    )
    print(json.dumps(summary["per_folder"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())

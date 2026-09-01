"""Data-physics profile study over the real 3W instances.

Runs the shipped public API - :func:`tsdive.profile` - over every
archive produced by ``scripts/convert_3w.py`` and records what it says,
including what it refuses to say. Nothing here computes a detector
metric: profiling is descriptive.

Two profiles per archive:

- **physics** over the archive's full extent (the honest default window),
- **flatline** over the last ``--window-s`` seconds, whose references are
  the earlier equal-size windows of the same archive.

An archive shorter than two windows has no earlier window to reference,
so its flatline verdict is ``NOT_ASSESSED_TOO_SHORT`` and the assessment
is not run at all. That count is a result, not a gap in the study. Every
other verdict comes off the ``FlatlineVerdict`` the library returned; the
study classifies nothing of its own.

Every converted instance is studied by default. ``--cap-per-folder``
takes a stratified subset instead, for a quicker pass.

    uv run python examples/studies/3w_profile/run_study.py --jobs 14
    uv run python examples/studies/3w_profile/run_study.py \\
        --archives data/3w_archives --limit 20
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[3] / "src"))

import tsdive
from tsdive.detectors.flatline import NotAssessed
from tsdive.errors import TSDiveError
from tsdive.store.quality import Severity
from tsdive.store.tagstore import archive_extent, meta_from_parquet

HERE = Path(__file__).parent

COLUMNS = (
    "instance",
    "instance_source",
    "well",
    "folder_label",
    "variable",
    "role",
    "unit_raw",
    "unit_resolved",
    "unit_canonical",
    "n_rows",
    "extent_s",
    "coverage",
    "valid_fraction",
    "whole_life_distinct",
    "n_gaps",
    "gap_classes",
    "longest_gap_s",
    "n_good",
    "n_uncertain",
    "n_bad",
    "unmapped_codes",
    "clipped_fraction",
    "censored",
    "duplicate_timestamps",
    "non_monotonic_positions",
    "dst_transitions",
    "edge_slack_s",
    "flatline_verdict",
    "flatline_fired_signals",
    "flatline_signal_evidence",
    "final_distinct",
    "final_std",
    "stall_s",
    "ref_p99_change_interval_s",
    "ref_p05_distinct",
    "n_reference_windows",
    "refusal_type",
    "refusal_message",
)

# The library's own reasons for declining, named here only to give each
# one a column value. The study classifies nothing itself: every verdict
# below comes off the FlatlineVerdict it was handed.
NOT_ASSESSED_VERDICTS = {
    NotAssessed.NO_VALID_SAMPLES: "NOT_ASSESSED_NO_VALID_SAMPLES",
    NotAssessed.STATE_VALUED_TAG: "NOT_ASSESSED_MODE_TAG",
    NotAssessed.NO_EVALUABLE_SIGNAL: "NOT_ASSESSED_NO_REFERENCE",
}


def _reference_window_count(first: pd.Timestamp, start: pd.Timestamp, span_s: float) -> int:
    """How many earlier windows ``reference_history`` will walk.

    Arithmetic only - the same cursor walk the library does, without
    reading the archive a third time.
    """
    span = pd.Timedelta(span_s, unit="s")
    cursor = first
    n = 0
    while cursor + span < start:
        n += 1
        cursor = cursor + span
    return n


def assess_archive(path: Path, instance: dict, window_s: float) -> dict:
    """Profile one archive twice and flatten the answer into one row."""
    row = dict.fromkeys(COLUMNS, "")
    row.update(
        instance=instance["instance"],
        instance_source=instance["instance_source"],
        well=instance["well"] or "",
        folder_label=instance["folder_label"],
        variable=path.stem,
    )
    try:
        meta = meta_from_parquet(path)
        first, last = archive_extent(path)
        row["role"] = meta.role.value if meta.role else ""
        row["unit_raw"] = meta.unit_raw or ""
        extent_s = (last - first).total_seconds()
        row["extent_s"] = f"{extent_s:.0f}"

        p = tsdive.profile(path, flatline=False)
        phys = p.physics
        row["unit_resolved"] = str(phys.unit.resolved).lower()
        row["unit_canonical"] = phys.unit.canonical or ""
        row["n_rows"] = str(phys.timestamp_audit.n_samples)
        row["coverage"] = "" if phys.coverage.coverage is None else f"{phys.coverage.coverage:.6f}"
        row["valid_fraction"] = (
            "" if phys.valid_fraction is None else f"{phys.valid_fraction:.6f}"
        )
        # Distinct values over the whole archive, not just the final
        # window: a tag with one is frozen for its entire life and can
        # never indict itself from its own history.
        whole_life = p.frame[p.frame["valid"]]["value"]
        row["whole_life_distinct"] = str(int(whole_life.dropna().nunique()))
        row["n_gaps"] = str(phys.coverage.n_gaps)
        row["gap_classes"] = ";".join(
            sorted({g.classification.cls.value for g in phys.coverage.gaps})
        )
        row["longest_gap_s"] = (
            "" if phys.coverage.longest_gap_s is None else f"{phys.coverage.longest_gap_s:.0f}"
        )
        counts = phys.severity_counts
        row["n_good"] = str(counts.get(Severity.GOOD, 0))
        row["n_uncertain"] = str(counts.get(Severity.UNCERTAIN, 0))
        row["n_bad"] = str(counts.get(Severity.BAD, 0))
        row["unmapped_codes"] = ";".join(phys.unmapped_quality_codes)
        row["clipped_fraction"] = (
            "" if phys.clipping.fraction is None else f"{phys.clipping.fraction:.6f}"
        )
        row["censored"] = str(phys.clipping.censored).lower()
        row["duplicate_timestamps"] = str(len(phys.timestamp_audit.duplicate_timestamps))
        row["non_monotonic_positions"] = str(len(phys.timestamp_audit.non_monotonic_positions))
        row["dst_transitions"] = str(len(phys.timestamp_audit.dst_transitions))
        row["edge_slack_s"] = f"{phys.edge_slack_s:.0f}"

        if extent_s < 2 * window_s:
            row["flatline_verdict"] = "NOT_ASSESSED_TOO_SHORT"
            row["n_reference_windows"] = "0"
            return row

        start = last - pd.Timedelta(window_s, unit="s")
        row["n_reference_windows"] = str(_reference_window_count(first, start, window_s))
        spec = f"{start.isoformat()}/{last.isoformat()}"
        fl = tsdive.profile(path, spec, flatline=True).flatline
        assert fl is not None
        row["flatline_signal_evidence"] = " | ".join(f"{s.signal}={s.evidence}" for s in fl.signals)
        row["flatline_fired_signals"] = ";".join(s.signal for s in fl.signals if s.fired)
        # Read off the signals rather than parsed back out of their
        # evidence sentences, so the spot-check table cannot drift from
        # the numbers the detector actually compared.
        by_signal = {s.signal: s for s in fl.signals}
        for column, signal, field in (
            ("stall_s", "time_since_last_actual_change", "observed"),
            ("ref_p99_change_interval_s", "time_since_last_actual_change", "reference"),
            ("final_distinct", "distinct_value_count_vs_p05", "observed"),
            ("ref_p05_distinct", "distinct_value_count_vs_p05", "reference"),
            ("final_std", "zero_std_while_coupled_tag_moves", "observed"),
        ):
            found = by_signal.get(signal)
            value = getattr(found, field) if found is not None else None
            row[column] = "" if value is None else f"{value:.6g}"
        if fl.saturated_not_frozen:
            row["flatline_verdict"] = "NOT_ASSESSED_CENSORED"
        elif fl.not_assessed is not None:
            row["flatline_verdict"] = NOT_ASSESSED_VERDICTS[fl.not_assessed]
        elif fl.fired:
            row["flatline_verdict"] = "FIRED"
        else:
            row["flatline_verdict"] = "NO_FLATLINE"
    except TSDiveError as e:
        row["refusal_type"] = type(e).__name__
        row["refusal_message"] = str(e).replace("\n", " ")
    except Exception as e:  # untyped failure: still a finding, just a different one
        row["refusal_type"] = f"UNTYPED:{type(e).__name__}"
        row["refusal_message"] = str(e).replace("\n", " ")
    return row


def study_instance(payload: dict) -> list[dict]:
    instance_dir = Path(payload["instance_dir"])
    instance = json.loads((instance_dir / "INSTANCE.json").read_text("utf-8"))
    window_s = payload["window_s"]
    return [
        assess_archive(p, instance, window_s)
        for p in sorted(instance_dir.glob("*.parquet"))
    ]


def stratify(dirs: list[Path], cap: int) -> list[Path]:
    """Cap instances per event folder while keeping every well represented.

    Within a folder the wells are taken round-robin in sorted order, so
    the cap trims the over-represented wells first instead of truncating
    the folder alphabetically. Any well that the caps missed entirely
    then contributes its first instance: a study that silently dropped a
    well would misreport the absent-variable and freeze statistics, which
    are per-well facts.
    """
    meta = {
        d: json.loads((d / "INSTANCE.json").read_text("utf-8"))
        for d in dirs
    }
    by_folder: dict[str, dict[str, list[Path]]] = {}
    for d in dirs:
        record = meta[d]
        well = record["well"] or record["instance_source"]
        by_folder.setdefault(record["folder_label"], {}).setdefault(well, []).append(d)

    chosen: list[Path] = []
    for folder in sorted(by_folder):
        wells = sorted(by_folder[folder])
        pools = {w: sorted(by_folder[folder][w]) for w in wells}
        picked: list[Path] = []
        depth = 0
        while len(picked) < cap:
            added = False
            for well in wells:
                if depth < len(pools[well]) and len(picked) < cap:
                    picked.append(pools[well][depth])
                    added = True
            if not added:
                break
            depth += 1
        chosen.extend(picked)

    covered = {meta[d]["well"] or meta[d]["instance_source"] for d in chosen}
    for d in dirs:
        well = meta[d]["well"] or meta[d]["instance_source"]
        if well not in covered:
            chosen.append(d)
            covered.add(well)
    return sorted(set(chosen))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", default="data/3w_archives")
    parser.add_argument("--out", default=str(HERE / "results"))
    parser.add_argument("--window-s", type=float, default=3600.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--cap-per-folder",
        type=int,
        default=None,
        help="stratified subset: at most N instances per event folder, all wells kept",
    )
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args(argv)

    root = Path(args.archives)
    dirs = sorted(p for p in root.iterdir() if (p / "INSTANCE.json").exists())
    n_available = len(dirs)
    if args.cap_per_folder is not None:
        dirs = stratify(dirs, args.cap_per_folder)
    if args.limit is not None:
        dirs = dirs[: args.limit]
    if not dirs:
        print(f"no converted instances under {root}; run scripts/convert_3w.py first")
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    rows: list[dict] = []
    payloads = [{"instance_dir": str(d), "window_s": args.window_s} for d in dirs]
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = [pool.submit(study_instance, p) for p in payloads]
        for done, fut in enumerate(as_completed(futures), start=1):
            rows.extend(fut.result())
            if done % 50 == 0 or done == len(payloads):
                print(f"  {done}/{len(payloads)} instances, {len(rows)} tags", flush=True)
    elapsed = time.perf_counter() - started

    csv_path = out_dir / "per_tag.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(COLUMNS))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["instance"], r["variable"])))
    print(
        f"studied {len(dirs)} instance(s), {len(rows)} tag(s) in {elapsed:.1f}s; "
        f"{csv_path} is {csv_path.stat().st_size:,d} bytes"
    )
    (out_dir / "run.json").write_text(
        json.dumps(
            {
                "archives_root": str(root),
                "n_instances_available": n_available,
                "n_instances": len(dirs),
                "n_tags": len(rows),
                "window_s": args.window_s,
                "cap_per_folder": args.cap_per_folder,
                "wall_seconds": round(elapsed, 1),
                "jobs": args.jobs,
                "tsdive_version": tsdive.__version__,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

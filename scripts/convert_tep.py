"""Convert fetched TEP RData into single-tag tsdive archives.

One archive per (split, fault, run, variable) at
``<out>/<split>/fault<NN>_run<RRR>/<variable>.parquet``, written through
:func:`tsdive.store.write_tag`, plus a ``LABEL_fault`` MODE archive so
the ground truth lives in the same store as the measurements.

Three source facts drive every decision here, and each is recorded rather
than smoothed over (see docs/DATA.md):

- **TEP has no clock.** It is a simulation: a row is numbered
  ``sample``, not stamped. tsdive stores UTC only, so a timestamp is
  synthesised - ``2000-01-01T00:00:00Z + 180 s x (sample - 1)`` - and
  every run's ``RUN.json`` carries ``timestamp_synthetic: true``. All runs
  share that clock because each is an independent simulation from its own
  t=0; the instants are an addressing scheme, not a history.
- **TEP has no quality column,** because a simulator has nothing to
  report. Every row is written ``SIMULATED``, the tag declares what that
  code means, and ``quality_assumed`` is set so every profile says so out
  loud. A non-finite value would make that code a lie, so a run holding
  one is refused rather than written.
- **The fault is a step at a known sample,** not a property of the file:
  a faulty run is normal for its first 20 samples (training; 160 in the
  testing splits) and faulty afterwards. ``LABEL_fault`` carries "0"
  before the onset and the fault number after it, per row.

The full set is 21 faults x 500 runs x 53 tags = 556,500 archives per
split, which is more small files than the point of the exercise
justifies. ``--runs-per-fault`` (default 20) takes a documented subset;
``--faults`` narrows it further. The count actually written is printed.

Usage:
    uv run python scripts/convert_tep.py --src data/tep --out data/tep_archives \\
        [--runs-per-fault 20] [--faults 0,1,4] [--jobs 8]
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

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from tsdive.store.identity import Role, TagIdentity, TagMeta
from tsdive.store.tagstore import write_tag

# Which RData file holds which split, and which faults it can contain.
# The Rieth 2017 distribution splits fault-free from faulty, so fault 0
# never appears in a "Faulty" file and 1-20 never in a "FaultFree" one.
SOURCES = {
    "TEP_FaultFree_Training.RData": "train",
    "TEP_Faulty_Training.RData": "train",
    "TEP_FaultFree_Testing.RData": "test",
    "TEP_Faulty_Testing.RData": "test",
}

# Sample index at which the fault is introduced. Training runs are 500
# samples (25 h) with the fault stepped in after 1 h; testing runs are
# 960 samples (48 h) with the fault stepped in after 8 h.
ONSET_SAMPLE = {"train": 20, "test": 160}

T0 = pd.Timestamp("2000-01-01T00:00:00Z")
SAMPLE_RATE_S = 180.0
ASSET = "TEP"

QUALITY_SIMULATED = "SIMULATED"
QUALITY_CODES = {QUALITY_SIMULATED: "GOOD"}

LABEL_POINT = "LABEL_fault"
KEY_COLUMNS = ("faultNumber", "simulationRun", "sample")

# Engineering units from Downs & Vogel (1993), "A plant-wide industrial
# process control problem", Computers & Chemical Engineering 17(3),
# pp. 245-255: Table 4 (XMEAS 1-22, continuous process measurements),
# Table 5 (XMEAS 23-41, composition analyses, all mol %) and Table 6
# (XMV 1-11, manipulated variables, % of valve range). The RData spells
# them lower-case; the paper writes XMEAS(i) / XMV(i).
#
# Four spellings here do not resolve against the shipped alias table and
# are left exactly as the paper writes them:
#   kscmh      thousand *standard* cubic metres per hour - a
#              reference-condition flow, which the table may only carry
#              with a declared reference state (docs/SCHEMA.md).
#   kPa gauge  read against local atmosphere; the table has no column
#              for a pressure datum, so registering it as kPa would make
#              every comparison wrong by about an atmosphere.
#   mol%       a composition basis, not a bare percentage.
#   kW         plain SI power the table does not carry.
_XMEAS_UNITS = {
    1: "kscmh",
    2: "kg/h",
    3: "kg/h",
    4: "kg/h",
    5: "kg/h",
    6: "kscmh",
    7: "kPa gauge",
    8: "%",
    9: "degC",
    10: "kscmh",
    11: "degC",
    12: "%",
    13: "kPa gauge",
    14: "m3/h",
    15: "%",
    16: "kPa gauge",
    17: "m3/h",
    18: "degC",
    19: "kg/h",
    20: "kW",
    21: "degC",
    22: "degC",
}
UNITS: dict[str, str] = {f"xmeas_{i}": u for i, u in _XMEAS_UNITS.items()}
UNITS.update({f"xmeas_{i}": "mol%" for i in range(23, 42)})
UNITS.update({f"xmv_{i}": "%" for i in range(1, 12)})

VARIABLES: tuple[str, ...] = tuple(
    [f"xmeas_{i}" for i in range(1, 42)] + [f"xmv_{i}" for i in range(1, 12)]
)


def read_rdata(path: Path) -> pd.DataFrame:
    """Read one TEP ``.RData`` into its single data frame.

    ``pyreadr`` is imported here rather than at module scope: it is an
    optional dependency (``uv sync --extra tep``) and nothing else in
    this repository needs it.
    """
    try:
        import pyreadr
    except ImportError as e:  # pragma: no cover - exercised by hand, not in CI
        raise ImportError(
            "reading TEP .RData needs pyreadr, which is an optional dependency: "
            "run `uv sync --extra tep` (or `pip install 'tsdive[tep]'`) and retry"
        ) from e
    result = pyreadr.read_r(str(path))
    if len(result) != 1:
        raise ValueError(
            f"{path.name}: expected one R object, found {sorted(result)}"
        )
    return next(iter(result.values()))


def role_for(variable: str) -> Role:
    """``xmeas_*`` are measurements; ``xmv_*`` are what the controller moves."""
    return Role.OP if variable.startswith("xmv_") else Role.PV


def variable_name(variable: str) -> str:
    """A structural display name.

    Downs & Vogel name every stream, but this converter states only what
    the RData itself states plus the variable's class. An invented stream
    description would read like it came off the file.
    """
    if variable.startswith("xmv_"):
        return f"TEP {variable} (manipulated variable)"
    index = int(variable.split("_", maxsplit=1)[1])
    kind = "composition analysis" if index >= 23 else "process measurement"
    return f"TEP {variable} ({kind})"


def timestamps(samples: pd.Series) -> pd.Series:
    """Synthetic UTC instants: ``T0 + 180 s x (sample - 1)``."""
    offsets = (samples.astype("int64") - 1) * int(SAMPLE_RATE_S)
    return T0 + pd.to_timedelta(offsets, unit="s")


def fault_labels(samples: pd.Series, fault: int, onset: int) -> list[str]:
    """Per-row ground truth: "0" before the onset sample, the fault after.

    A fault-free run is "0" throughout, which this rule already gives:
    its fault number is 0.
    """
    return ["0" if s < onset else str(fault) for s in samples.astype("int64")]


def run_dir(out_root: Path, split: str, fault: int, run: int) -> Path:
    return out_root / split / f"fault{fault:02d}_run{run:03d}"


def convert_run(
    frame: pd.DataFrame,
    out_root: Path,
    *,
    split: str,
    fault: int,
    run: int,
    source: str,
    source_md5: str | None,
    force: bool = False,
) -> dict:
    """Convert one (fault, run) simulation into a directory of archives."""
    out_dir = run_dir(out_root, split, fault, run)
    sidecar = out_dir / "RUN.json"
    if sidecar.exists() and not force:
        return {"skipped": True, **json.loads(sidecar.read_text("utf-8"))}

    ordered = frame.sort_values("sample", kind="stable").reset_index(drop=True)
    samples = ordered["sample"]
    stamps = timestamps(samples).to_numpy()
    onset = ONSET_SAMPLE[split]
    loop_id = f"fault{fault:02d}_run{run:03d}"

    present = [v for v in VARIABLES if v in ordered.columns]
    values = {v: ordered[v].to_numpy(dtype=float) for v in present}
    bad = {
        v: int((~np.isfinite(column)).sum())
        for v, column in values.items()
        if not np.isfinite(column).all()
    }
    if bad:
        raise ValueError(
            f"{loop_id}: {sum(bad.values())} non-finite value(s) in {sorted(bad)}; "
            f"every row is written quality {QUALITY_SIMULATED!r}, which such a row "
            "would make a lie - refusing rather than coding it GOOD"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    quality = [QUALITY_SIMULATED] * len(ordered)
    written: list[str] = []
    for variable in present:
        meta = TagMeta(
            identity=TagIdentity(source_id=loop_id, point_id=variable),
            name=variable_name(variable),
            unit_raw=UNITS[variable],
            eng_range=None,
            sample_rate_s=SAMPLE_RATE_S,
            asset=ASSET,
            loop_id=loop_id,
            role=role_for(variable),
            quality_codes=dict(QUALITY_CODES),
            quality_assumed=True,
        )
        archive = pd.DataFrame(
            {
                "timestamp": stamps,
                "value": values[variable],
                "quality": quality,
            }
        )
        write_tag(out_dir / f"{variable}.parquet", archive, meta, overwrite=True)
        written.append(variable)

    labels = fault_labels(samples, fault, onset)
    label_meta = TagMeta(
        identity=TagIdentity(source_id=loop_id, point_id=LABEL_POINT),
        name="TEP per-row fault label",
        unit_raw=None,
        eng_range=None,
        sample_rate_s=SAMPLE_RATE_S,
        asset=ASSET,
        loop_id=loop_id,
        role=Role.MODE,
        quality_codes=dict(QUALITY_CODES),
        quality_assumed=True,
    )
    write_tag(
        out_dir / f"{LABEL_POINT}.parquet",
        pd.DataFrame({"timestamp": stamps, "value": labels, "quality": quality}),
        label_meta,
        overwrite=True,
    )
    written.append(LABEL_POINT)

    record = {
        "archives_written": written,
        "asset": ASSET,
        "fault": fault,
        "first_timestamp_utc": pd.Timestamp(stamps[0]).isoformat(),
        "last_timestamp_utc": pd.Timestamp(stamps[-1]).isoformat(),
        "loop_id": loop_id,
        "n_faulty_rows": sum(1 for label in labels if label != "0"),
        "n_nonfinite_values": 0,
        "n_rows": len(ordered),
        "onset_sample": onset,
        "quality_assumed": True,
        "quality_codes": dict(QUALITY_CODES),
        "run": run,
        "sample_rate_s": SAMPLE_RATE_S,
        "source_file": source,
        "source_md5": source_md5,
        "split": split,
        "t0_utc": T0.isoformat(),
        "timestamp_synthetic": True,
    }
    sidecar.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    return {"skipped": False, **record}


def _worker(payload: dict) -> dict:
    return convert_run(
        payload["frame"],
        Path(payload["out_root"]),
        split=payload["split"],
        fault=payload["fault"],
        run=payload["run"],
        source=payload["source"],
        source_md5=payload["source_md5"],
        force=payload["force"],
    )


def payloads_for(
    frame: pd.DataFrame,
    out_root: Path,
    *,
    split: str,
    source: str,
    source_md5: str | None,
    faults: set[int] | None,
    runs_per_fault: int | None,
    force: bool,
) -> list[dict]:
    """One payload per (fault, run) the filters keep, smallest frame possible."""
    keep = frame
    fault_col = keep["faultNumber"].astype("int64")
    if faults is not None:
        keep = keep[fault_col.isin(faults)]
    if runs_per_fault is not None:
        keep = keep[keep["simulationRun"].astype("int64") <= runs_per_fault]
    out: list[dict] = []
    columns = list(KEY_COLUMNS) + [v for v in VARIABLES if v in keep.columns]
    for (fault, run), group in keep.groupby(
        [keep["faultNumber"].astype("int64"), keep["simulationRun"].astype("int64")],
        sort=True,
    ):
        out.append(
            {
                "frame": group[columns].reset_index(drop=True),
                "out_root": str(out_root),
                "split": split,
                "fault": int(fault),
                "run": int(run),
                "source": source,
                "source_md5": source_md5,
                "force": force,
            }
        )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", default="data/tep", help="directory written by fetch_tep.py")
    parser.add_argument("--out", default="data/tep_archives")
    parser.add_argument(
        "--runs-per-fault",
        type=int,
        default=20,
        help="convert simulationRun 1..N of each fault (0 or less converts all 500)",
    )
    parser.add_argument(
        "--faults",
        default=None,
        help="comma-separated fault numbers to keep; default is all present (0-20)",
    )
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--force", action="store_true", help="reconvert runs already done")
    args = parser.parse_args(argv)

    src = Path(args.src)
    sources = sorted(p for p in src.glob("*.RData") if p.name in SOURCES)
    if not sources:
        print(f"no TEP .RData under {src}; run scripts/fetch_tep.py first")
        return 1

    manifest_path = src / "MANIFEST.sha256.json"
    manifest = json.loads(manifest_path.read_text("utf-8")) if manifest_path.exists() else {}
    faults = (
        {int(f) for f in args.faults.split(",") if f.strip()} if args.faults else None
    )
    runs_per_fault = args.runs_per_fault if args.runs_per_fault > 0 else None

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    done = 0
    n_archives = 0
    n_runs = 0
    for path in sources:
        split = SOURCES[path.name]
        read_started = time.perf_counter()
        frame = read_rdata(path)
        print(
            f"{path.name}: {len(frame):,d} rows, {len(frame.columns)} columns, "
            f"read in {time.perf_counter() - read_started:.1f}s",
            flush=True,
        )
        jobs = payloads_for(
            frame,
            out_root,
            split=split,
            source=path.name,
            source_md5=manifest.get(path.name, {}).get("md5"),
            faults=faults,
            runs_per_fault=runs_per_fault,
            force=args.force,
        )
        del frame
        n_runs += len(jobs)
        with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as pool:
            futures = [pool.submit(_worker, payload) for payload in jobs]
            for fut in as_completed(futures):
                record = fut.result()
                n_archives += len(record["archives_written"])
                done += 1
                if done % 50 == 0 or done == n_runs:
                    print(f"  {done}/{n_runs} runs", flush=True)

    elapsed = time.perf_counter() - started
    total_bytes = sum(p.stat().st_size for p in out_root.rglob("*.parquet"))
    print(
        f"converted {done} run(s) -> {n_archives} archive(s), "
        f"{total_bytes:,d} bytes on disk, {elapsed:.1f}s wall"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

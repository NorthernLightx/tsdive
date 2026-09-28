"""Runs 251-350 of the TEP fault-free testing file, all 960 samples, as one float32 parquet.

Reads ``TEP_FaultFree_Testing.RData`` from ``--source`` (``pyreadr``, a dev
dependency) and no faulty file. Writes
``--out/tep_faultfree_testing_251_350.parquet`` with ``MANIFEST.json``
beside it: the source manifest sha256, the cache sha256, row counts, wall
seconds and peak process memory. The checksum and memory helpers come from
the shift interval study's ``build_tep_cache.py``. The cache is local and
never committed.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
_spec = importlib.util.spec_from_file_location(
    "shift_build_tep_cache", HERE.parent / "shift_intervals" / "build_tep_cache.py"
)
shift_cache = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(shift_cache)

DEFAULT_SOURCE = "data/tep"
DEFAULT_OUT = "data/tep_switchback_cache"
FIRST_RUN, LAST_RUN = 251, 350
SAMPLES = 960
CACHE_NAME = "tep_faultfree_testing_251_350.parquet"
SOURCE_FILE = shift_cache.FILES["fault_free"]
VARIABLES = shift_cache.VARIABLES


def read_runs(path: Path, first: int, last: int) -> pd.DataFrame:
    """The fault-free frame cut to runs ``first``..``last``, float32 values."""
    import pyreadr

    frame = next(iter(pyreadr.read_r(str(path)).values()))
    keep = (frame["simulationRun"] >= first) & (frame["simulationRun"] <= last)
    out = pd.DataFrame(
        {
            "run": frame.loc[keep, "simulationRun"].astype("int16").to_numpy(),
            "sample": frame.loc[keep, "sample"].astype("int16").to_numpy(),
        }
    )
    faults = frame.loc[keep, "faultNumber"].unique()
    if len(faults) != 1 or int(faults[0]) != 0:
        raise ValueError(f"{path.name}: expected fault 0 only, found {sorted(faults)}")
    for variable in VARIABLES:
        out[variable] = frame.loc[keep, variable].astype("float32").to_numpy()
    del frame
    gc.collect()
    return out.sort_values(["run", "sample"], kind="stable").reset_index(drop=True)


def build(source: Path, out: Path, first: int = FIRST_RUN, last: int = LAST_RUN) -> dict:
    started = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    cache = read_runs(source / SOURCE_FILE, first, last)
    per_run = cache.groupby("run").size()
    if not (per_run == SAMPLES).all():
        raise ValueError(f"expected {SAMPLES} samples per run, found {sorted(set(per_run))}")
    path = out / CACHE_NAME
    cache.to_parquet(path, index=False)
    manifest = {
        "source_manifest_sha256": shift_cache.sha256_file(source / "MANIFEST.sha256.json")[:12],
        "source_file": SOURCE_FILE,
        "cache": CACHE_NAME,
        "cache_sha256": shift_cache.sha256_file(path)[:12],
        "runs": [first, last],
        "samples": [1, SAMPLES],
        "n_runs": int(per_run.size),
        "n_rows": len(cache),
        "non_finite_values": int((~np.isfinite(cache[VARIABLES].to_numpy())).sum()),
        "peak_memory_bytes": shift_cache.peak_memory_bytes(),
        "wall_seconds": round(time.perf_counter() - started, 1),
    }
    (out / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", default=str(ROOT / DEFAULT_SOURCE))
    parser.add_argument("--out", default=str(ROOT / DEFAULT_OUT))
    args = parser.parse_args(argv)
    manifest = build(Path(args.source), Path(args.out))
    print(json.dumps({k: manifest[k] for k in ("n_rows", "wall_seconds", "peak_memory_bytes")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Samples 1-320 of the TEP testing runs, as one float32 parquet the shift study reads.

Reads ``TEP_FaultFree_Testing.RData`` and ``TEP_Faulty_Testing.RData`` from
``--source`` (``pyreadr``, a dev dependency), keeps runs 1 to ``--runs`` and
samples 1 to 320 of fault-free plus faults 1-20, and writes
``--out/tep_testing_1_320.parquet`` with ``MANIFEST.json`` beside it: the
source manifest sha256, the cache sha256, row counts, wall seconds, peak
process memory and the onset check. The cache is local and never committed.

The onset check reads fault 6 (A feed loss), whose ``xmeas_1`` falls within
one sample: the ensemble mean of ``xmeas_1`` over the kept runs for samples
155 to 166, fault 6 against fault-free.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

DEFAULT_SOURCE = "data/tep"
DEFAULT_OUT = "data/tep_shift_cache"
DEFAULT_RUNS = 350
LAST_SAMPLE = 320
CACHE_NAME = "tep_testing_1_320.parquet"
FILES = {"fault_free": "TEP_FaultFree_Testing.RData", "faulty": "TEP_Faulty_Testing.RData"}
VARIABLES = [f"xmeas_{i}" for i in range(1, 42)] + [f"xmv_{i}" for i in range(1, 12)]
ONSET_FAULT = 6
ONSET_VARIABLE = "xmeas_1"
ONSET_SAMPLES = range(155, 167)


def peak_memory_bytes() -> int | None:
    """Peak working set of this process on Windows, peak RSS elsewhere."""
    if sys.platform == "win32":

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        current = ctypes.windll.kernel32.GetCurrentProcess
        current.restype = ctypes.c_void_p  # a pseudo handle is -1 at pointer width
        query = ctypes.windll.psapi.GetProcessMemoryInfo
        query.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
        ok = query(current(), ctypes.byref(counters), counters.cb)
        return int(counters.PeakWorkingSetSize) if ok else None
    try:
        import resource

        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    except (ImportError, OSError):  # pragma: no cover
        return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_subset(path: Path, runs: int) -> pd.DataFrame:
    """One RData frame cut to runs 1..``runs`` and samples 1..320, float32 values."""
    import pyreadr

    frame = next(iter(pyreadr.read_r(str(path)).values()))
    keep = (frame["simulationRun"] <= runs) & (frame["sample"] <= LAST_SAMPLE)
    out = pd.DataFrame(
        {
            "fault": frame.loc[keep, "faultNumber"].astype("int8").to_numpy(),
            "run": frame.loc[keep, "simulationRun"].astype("int16").to_numpy(),
            "sample": frame.loc[keep, "sample"].astype("int16").to_numpy(),
        }
    )
    for variable in VARIABLES:
        out[variable] = frame.loc[keep, variable].astype("float32").to_numpy()
    del frame
    gc.collect()
    return out


def onset_check(cache: pd.DataFrame) -> dict:
    rows = cache[cache["sample"].isin(list(ONSET_SAMPLES))]
    means = rows.groupby(["fault", "sample"])[ONSET_VARIABLE].mean()
    return {
        "variable": ONSET_VARIABLE,
        "fault": ONSET_FAULT,
        "samples": list(ONSET_SAMPLES),
        "fault_free_mean": [round(float(means[(0, s)]), 4) for s in ONSET_SAMPLES],
        "fault_mean": [round(float(means[(ONSET_FAULT, s)]), 4) for s in ONSET_SAMPLES],
    }


def build(source: Path, out: Path, runs: int) -> dict:
    started = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    parts = [read_subset(source / FILES[key], runs) for key in ("fault_free", "faulty")]
    cache = pd.concat(parts, ignore_index=True)
    del parts
    cache = cache.sort_values(["fault", "run", "sample"], kind="stable").reset_index(drop=True)
    path = out / CACHE_NAME
    cache.to_parquet(path, index=False)
    source_manifest = source / "MANIFEST.sha256.json"
    manifest = {
        "source_manifest_sha256": sha256_file(source_manifest)[:12],
        "source_files": [FILES["fault_free"], FILES["faulty"]],
        "cache": CACHE_NAME,
        "cache_sha256": sha256_file(path)[:12],
        "runs": [1, runs],
        "samples": [1, LAST_SAMPLE],
        "faults": sorted(int(f) for f in cache["fault"].unique()),
        "n_rows": len(cache),
        "rows_per_condition": {
            str(k): int(v) for k, v in cache.groupby("fault").size().items()
        },
        "non_finite_values": int((~np.isfinite(cache[VARIABLES].to_numpy())).sum()),
        "onset_check": onset_check(cache),
        "peak_memory_bytes": peak_memory_bytes(),
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
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    args = parser.parse_args(argv)
    manifest = build(Path(args.source), Path(args.out), args.runs)
    print(json.dumps({k: manifest[k] for k in ("n_rows", "wall_seconds", "peak_memory_bytes")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

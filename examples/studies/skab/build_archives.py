"""Convert the fetched SKAB files into tsdive archives, one per sensor.

For every experiment ``raw/<folder>/<name>.csv`` the script writes a
wide CSV with the eight sensor columns under ``<work>/<folder>__<name>/``,
one metadata template per sensor through ``tsdive.api.init_meta``, and
the archives ``<archives>/<folder>__<name>/<sensor>.parquet`` through
``tsdive.api.ingest_wide``. The ``anomaly`` and ``changepoint`` columns
go to ``<archives>/<folder>__<name>/labels.parquet`` and never into an
archive. A typed refusal on any step is recorded by experiment in
``<archives>/refusals.json`` and the script continues with the next file.

SKAB timestamps are naive and carry no offset; the study localises them
as UTC because a within-record analysis reads only differences between
stamps. The upstream files document no units, so ``unit_raw`` stays
null. There is no quality column, so every sample is ingested with
``assume_quality="GOOD"`` and the archives say ``quality_assumed``.

Usage:
    python examples/studies/skab/build_archives.py [--raw data/skab/raw]
        [--archives data/skab_archives] [--work data/skab_work]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from tsdive import api
from tsdive.errors import TSDiveError
from tsdive.store.timebase import require_unique

TIMESTAMP_COL = "datetime"
SENSORS = (
    "Accelerometer1RMS",
    "Accelerometer2RMS",
    "Current",
    "Pressure",
    "Temperature",
    "Thermocouple",
    "Voltage",
    "Volume Flow RateRMS",
)
LABELS = ("anomaly", "changepoint")
SEPARATOR = ";"
TZ = "UTC"
ASSUME_QUALITY = "GOOD"


def experiment_id(folder: str, stem: str) -> str:
    return f"{folder}__{stem}"


def list_experiments(raw: Path) -> list[tuple[str, str, Path]]:
    """``(folder, stem, path)`` for every CSV under ``raw``, sorted by path."""
    found = []
    for path in sorted(raw.glob("*/*.csv")):
        found.append((path.parent.name, path.stem, path))
    return found


def read_experiment(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep=SEPARATOR)
    missing = [c for c in (TIMESTAMP_COL, *SENSORS) if c not in frame.columns]
    if missing:
        raise TSDiveError(f"{path.name}: missing column(s) {missing}")
    return frame


def write_labels(frame: pd.DataFrame, stamps: pd.Series, target: Path) -> bool:
    """Write ``labels.parquet`` when the file carries both label columns."""
    if not all(c in frame.columns for c in LABELS):
        return False
    labels = pd.DataFrame({"timestamp": stamps})
    for column in LABELS:
        labels[column] = frame[column].astype(int).to_numpy()
    labels.to_parquet(target, index=False)
    return True


def convert_one(
    folder: str, stem: str, path: Path, *, archives: Path, work: Path
) -> dict[str, object]:
    """Convert one experiment; the result names the archives or the refusal."""
    exp = experiment_id(folder, stem)
    source_id = f"skab/{folder}/{stem}"
    exp_dir = archives / exp
    work_dir = work / exp
    record: dict[str, object] = {"experiment": exp, "folder": folder, "name": stem}
    try:
        frame = read_experiment(path)
        stamps = pd.to_datetime(frame[TIMESTAMP_COL])
        require_unique(stamps)
        wide = frame[[TIMESTAMP_COL, *SENSORS]].rename(columns={TIMESTAMP_COL: "timestamp"})
        work_dir.mkdir(parents=True, exist_ok=True)
        wide_csv = work_dir / "wide.csv"
        wide.to_csv(wide_csv, index=False, lineterminator="\n")
        meta_dir = work_dir / "meta"
        api.init_meta(
            wide_csv,
            out_dir=meta_dir,
            source_id=source_id,
            timestamp_col="timestamp",
            tags=list(SENSORS),
            overwrite=True,
        )
        written = api.ingest_wide(
            wide_csv,
            out_dir=exp_dir,
            meta_dir=meta_dir,
            timestamp_col="timestamp",
            tags=list(SENSORS),
            tz=TZ,
            assume_quality=ASSUME_QUALITY,
            overwrite=True,
        )
        has_labels = write_labels(frame, stamps.dt.tz_localize(TZ), exp_dir / "labels.parquet")
    except (TSDiveError, ValueError, FileExistsError) as e:
        record["refusal"] = type(e).__name__
        record["message"] = str(e)
        return record
    record["archives"] = [p.name for p in written]
    record["n_rows"] = len(frame)
    record["labels"] = has_labels
    return record


def build(raw: Path, archives: Path, work: Path) -> tuple[list[dict], list[dict]]:
    """Convert every experiment under ``raw``; return (converted, refusals)."""
    converted: list[dict] = []
    refusals: list[dict] = []
    for folder, stem, path in list_experiments(raw):
        record = convert_one(folder, stem, path, archives=archives, work=work)
        (refusals if "refusal" in record else converted).append(record)
    archives.mkdir(parents=True, exist_ok=True)
    (archives / "refusals.json").write_text(
        json.dumps(refusals, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return converted, refusals


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--raw", default="data/skab/raw")
    parser.add_argument("--archives", default="data/skab_archives")
    parser.add_argument("--work", default="data/skab_work")
    args = parser.parse_args(argv)
    converted, refusals = build(Path(args.raw), Path(args.archives), Path(args.work))
    n_archives = sum(len(r["archives"]) for r in converted)
    print(
        f"{len(converted)} experiment(s) converted, {n_archives} archives, "
        f"{len(refusals)} refusal(s); see {Path(args.archives) / 'refusals.json'}"
    )
    for r in refusals:
        print(f"  {r['experiment']}: {r['refusal']}: {r['message']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

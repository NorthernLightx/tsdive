"""Convert fetched 3W instances into single-tag tsdive archives.

One archive per (instance, present variable) at
``<out>/<instance_stem>/<variable>.parquet``, written through
:func:`tsdive.store.write_tag`, plus ``LABEL_class`` and
``LABEL_state`` MODE archives so the row labels live in the same store as
the measurements they describe.

Three source facts drive every decision here, and each is recorded rather
than smoothed over (see docs/DATA.md):

- The 3W ``timestamp`` index is tz-naive, so it names no instant.
  ``--tz`` states what zone the source is in; the assumption is written
  into every instance's ``INSTANCE.json`` as ``timestamp_tz_assumed``.
- There is no quality column anywhere in 3W. Quality is therefore
  assumed: finite values get ``GOOD_ASSUMED``, NaNs get ``NO_DATA``,
  the tag declares what those codes mean, and ``quality_assumed`` is set
  so every profile says so out loud.
- An absent variable is an entirely-NaN column, not a missing one. Such a
  variable gets no archive at all - an archive of nothing but NO_DATA
  would assert the tag existed and failed, which the source does not say.

Usage:
    uv run python scripts/convert_3w.py --src data/3w --out data/3w_archives \\
        --only-real [--limit 20] [--jobs 8]
"""

from __future__ import annotations

import argparse
import configparser
import json
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from tsdive.store.identity import Role, TagIdentity, TagMeta
from tsdive.store.tagstore import write_tag

LABEL_COLUMNS = ("class", "state")
QUALITY_GOOD = "GOOD_ASSUMED"
QUALITY_NODATA = "NO_DATA"
QUALITY_CODES = {QUALITY_GOOD: "GOOD", QUALITY_NODATA: "BAD"}
SAMPLE_RATE_S = 1.0

_UNIT_RE = re.compile(r"\[([^\]]*)\]\s*$")


def instance_source(stem: str) -> str:
    """Provenance from the filename prefix: real, simulated, or hand-drawn."""
    if stem.startswith("WELL-"):
        return "REAL"
    if stem.startswith("SIMULATED"):
        return "SIMULATED"
    if stem.startswith("DRAWN"):
        return "DRAWN"
    return "UNKNOWN"


def well_id(stem: str) -> str | None:
    return stem.split("_", maxsplit=1)[0] if stem.startswith("WELL-") else None


def role_for(variable: str) -> Role:
    """Control-loop role of a 3W variable.

    ``ABER-*`` are choke openings - a manipulated output, so OP.
    ``ESTADO-*`` are valve states, which are labels rather than
    measurements, so MODE. Everything else (``P-*``, ``PT-P``, ``T-*``,
    ``Q*``) is a measured process variable.
    """
    if variable.startswith("ABER-"):
        return Role.OP
    if variable.startswith("ESTADO-"):
        return Role.MODE
    return Role.PV


def read_variable_descriptions(ini_path: Path) -> dict[str, tuple[str, str | None]]:
    """``variable -> (description, unit_raw)`` from ``dataset.ini``.

    The unit is the trailing bracket of each description. ``ESTADO-*``
    brackets state a value domain (``[0, 0.5, or 1]``), not a unit, so
    those tags carry ``unit_raw=None``: a state has no engineering unit,
    and copying the domain in would put a non-unit through unit
    resolution.
    """
    parser = configparser.ConfigParser(inline_comment_prefixes=None)
    parser.optionxform = str  # keep P-PDG, not p-pdg
    parser.read(ini_path, encoding="utf-8")
    out: dict[str, tuple[str, str | None]] = {}
    for name, raw in parser["PARQUET_FILE_PROPERTIES"].items():
        description = raw.strip()
        match = _UNIT_RE.search(description)
        unit: str | None = None
        if match:
            candidate = match.group(1).strip()
            if not name.startswith("ESTADO-"):
                unit = candidate
        out[name] = (description, unit)
    return out


def _quality_for(values: pd.Series) -> pd.Series:
    finite = values.notna()
    return pd.Series(
        [QUALITY_GOOD if ok else QUALITY_NODATA for ok in finite],
        index=values.index,
        dtype=object,
    )


def _label_values(column: pd.Series) -> pd.Series:
    """Int16 labels rendered as state strings; ``<NA>`` stays missing."""
    return pd.Series(
        [None if pd.isna(v) else str(int(v)) for v in column],
        index=column.index,
        dtype=object,
    )


def convert_instance(
    src: Path,
    out_root: Path,
    *,
    tz: str,
    descriptions: dict[str, tuple[str, str | None]],
    source_sha256: str | None,
    force: bool,
) -> dict:
    """Convert one 3W parquet into a directory of single-tag archives."""
    stem = src.stem
    folder_label = src.parent.name
    out_dir = out_root / stem
    sidecar = out_dir / "INSTANCE.json"
    if sidecar.exists() and not force:
        return {"instance": stem, "skipped": True, **json.loads(sidecar.read_text("utf-8"))}

    frame = pd.read_parquet(src)
    index = pd.DatetimeIndex(frame.index)
    if index.tz is not None:
        raise ValueError(f"{src.name}: timestamp index is already tz-aware; --tz would be a lie")
    timestamps = pd.Series(index.tz_localize(tz).tz_convert("UTC"), index=frame.index)

    variables = [c for c in frame.columns if c not in LABEL_COLUMNS]
    absent = [c for c in variables if frame[c].isna().all()]
    present = [c for c in variables if c not in absent]

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for variable in present:
        values = frame[variable]
        description, unit_raw = descriptions.get(variable, (variable, None))
        meta = TagMeta(
            identity=TagIdentity(source_id=stem, point_id=variable),
            name=description,
            unit_raw=unit_raw,
            eng_range=None,
            sample_rate_s=SAMPLE_RATE_S,
            asset=well_id(stem) or instance_source(stem),
            loop_id=stem,
            role=role_for(variable),
            quality_codes=dict(QUALITY_CODES),
            quality_assumed=True,
        )
        archive = pd.DataFrame(
            {
                "timestamp": timestamps.to_numpy(),
                "value": values.to_numpy(),
                "quality": _quality_for(values).to_numpy(),
            }
        )
        write_tag(out_dir / f"{variable}.parquet", archive, meta, overwrite=True)
        written.append(variable)

    label_histogram: dict[str, dict[str, int]] = {}
    for column in LABEL_COLUMNS:
        raw = frame[column]
        values = _label_values(raw)
        label_histogram[column] = {
            ("NA" if pd.isna(k) else str(int(k))): int(v)
            for k, v in raw.value_counts(dropna=False).items()
        }
        meta = TagMeta(
            identity=TagIdentity(source_id=stem, point_id=f"LABEL_{column}"),
            name=f"3W per-row {column} label",
            unit_raw=None,
            eng_range=None,
            sample_rate_s=SAMPLE_RATE_S,
            asset=well_id(stem) or instance_source(stem),
            loop_id=stem,
            role=Role.MODE,
            quality_codes=dict(QUALITY_CODES),
            quality_assumed=True,
        )
        archive = pd.DataFrame(
            {
                "timestamp": timestamps.to_numpy(),
                "value": values.to_numpy(),
                "quality": _quality_for(raw).to_numpy(),
            }
        )
        write_tag(out_dir / f"LABEL_{column}.parquet", archive, meta, overwrite=True)
        written.append(f"LABEL_{column}")

    # Leading run of <NA> class labels: argmin finds the first labelled
    # row, which is that run's length. An all-<NA> instance has no such
    # row, so the whole file is the prefix.
    class_na = frame["class"].isna().to_numpy()
    leading_na = len(frame) if class_na.all() else int(class_na.argmin())

    record = {
        "instance": stem,
        "instance_source": instance_source(stem),
        "folder_label": folder_label,
        "well": well_id(stem),
        "n_rows": len(frame),
        "first_timestamp_utc": timestamps.iloc[0].isoformat() if len(frame) else None,
        "last_timestamp_utc": timestamps.iloc[-1].isoformat() if len(frame) else None,
        "timestamp_tz_assumed": tz,
        "absent_variables": absent,
        "present_variables": present,
        "unlabelled_prefix_rows": leading_na,
        "row_label_histogram": label_histogram,
        "source_relpath": str(src).replace("\\", "/"),
        "source_sha256": source_sha256,
        "quality_assumed": True,
        "quality_codes": dict(QUALITY_CODES),
        "archives_written": written,
    }
    sidecar.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    return {"skipped": False, **record}


def _worker(payload: dict) -> dict:
    return convert_instance(
        Path(payload["src"]),
        Path(payload["out_root"]),
        tz=payload["tz"],
        descriptions=payload["descriptions"],
        source_sha256=payload["source_sha256"],
        force=payload["force"],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", default="data/3w", help="directory written by fetch_3w.py")
    parser.add_argument("--out", default="data/3w_archives")
    parser.add_argument(
        "--tz",
        default="UTC",
        help="zone the naive 3W timestamps are read as; recorded on every instance",
    )
    parser.add_argument("--only-real", action="store_true")
    parser.add_argument("--classes", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--force", action="store_true", help="reconvert instances already done")
    args = parser.parse_args(argv)

    src_root = Path(args.src)
    dataset = src_root / "dataset"
    if not dataset.exists():
        print(f"no dataset at {dataset}; run scripts/fetch_3w.py first")
        return 1
    descriptions = read_variable_descriptions(dataset / "dataset.ini")

    manifest_path = src_root / "MANIFEST.sha256.json"
    manifest = json.loads(manifest_path.read_text("utf-8")) if manifest_path.exists() else {}

    classes = {c.strip() for c in args.classes.split(",") if c.strip()} if args.classes else None
    files = sorted(p for p in dataset.glob("*/*.parquet") if p.parent.name.isdigit())
    if args.only_real:
        files = [p for p in files if p.stem.startswith("WELL-")]
    if classes is not None:
        files = [p for p in files if p.parent.name in classes]
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        print("no instances selected")
        return 1

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    payloads = [
        {
            "src": str(p),
            "out_root": str(out_root),
            "tz": args.tz,
            "descriptions": descriptions,
            "source_sha256": (
                manifest.get(f"dataset/{p.parent.name}/{p.name}", {}).get("sha256")
            ),
            "force": args.force,
        }
        for p in files
    ]

    started = time.perf_counter()
    done = 0
    n_archives = 0
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = [pool.submit(_worker, payload) for payload in payloads]
        for fut in as_completed(futures):
            record = fut.result()
            n_archives += len(record.get("archives_written", []))
            done += 1
            if done % 50 == 0 or done == len(payloads):
                print(f"  {done}/{len(payloads)} instances", flush=True)
    elapsed = time.perf_counter() - started
    total_bytes = sum(p.stat().st_size for p in out_root.rglob("*.parquet"))
    print(
        f"converted {done} instance(s) -> {n_archives} archive(s), "
        f"{total_bytes:,d} bytes on disk, {elapsed:.1f}s wall"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

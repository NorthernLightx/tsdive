"""Provenance fingerprints and replay verdicts.

Guarantee, stated precisely: a computation is reproducible given
identical inputs, and this module *detects* when the inputs changed,
whether the data, the code, or both.

Fingerprints cover exact ``(timestamp, value, quality)`` triples plus the
sampling-contract digest, the module version, and an ``as_of`` watermark.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

import tsdive
from tsdive.store.sampling_contract import SamplingContract


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _value_repr(v: object) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "<null>"
    try:
        return repr(float(v))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return repr(v)


def _iso_ts(timestamp: object) -> str:
    return cast(pd.Timestamp, pd.Timestamp(str(timestamp))).isoformat()


def row_hash(timestamp: object, value: object, quality: object) -> str:
    """Stable hash of one exact (timestamp, value, quality) triple."""
    return _sha256(f"{_iso_ts(timestamp)}|{_value_repr(value)}|{quality!r}")


def data_fingerprint(frame: pd.DataFrame) -> str:
    """SHA-256 over the sorted set of exact sample rows.

    Sorted so that row order (which historians reorder freely) does not
    change the fingerprint; content does.
    """
    lines = sorted(
        row_hash(t, v, q)
        for t, v, q in zip(frame["timestamp"], frame["value"], frame["quality"], strict=True)
    )
    return _sha256("\n".join(lines))


def env_fingerprint() -> dict[str, str]:
    """Library versions, BLAS vendor and thread count for a run."""
    blas_info = getattr(np.__config__, "blas_opt_info", {}) or {}
    blasd = {k: ", ".join(map(str, v)) for k, v in blas_info.items()}
    return {
        "python": platform.python_version(),
        "tsdive": tsdive.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "platform": platform.platform(),
        "blas": json.dumps(blasd, sort_keys=True),
        "threads": str(os.cpu_count()),
    }


class ReplayVerdict(StrEnum):
    MATCH = "MATCH"
    CODE_DRIFT = "CODE_DRIFT"
    DATA_CHANGED = "DATA_CHANGED"
    BOTH = "BOTH"


@dataclass
class ReplayRecord:
    id: str
    tag: str
    data_fingerprint: str
    row_hashes: dict[str, str]  # iso timestamp -> row hash
    contract_digest: str
    module_version: str
    as_of: str


@dataclass
class ReplayResult:
    verdict: ReplayVerdict
    moved_samples: list[tuple[str, str]]  # (iso timestamp, what happened)

    def summary_lines(self) -> list[str]:
        lines = [f"replay verdict: {self.verdict.value}"]
        if self.moved_samples:
            lines.append("samples that moved:")
            lines.extend(f"  {ts}: {what}" for ts, what in self.moved_samples[:10])
        return lines


def _rows(frame: pd.DataFrame) -> dict[str, str]:
    return {
        _iso_ts(t): row_hash(t, v, q)
        for t, v, q in zip(frame["timestamp"], frame["value"], frame["quality"], strict=True)
    }


def make_record(
    record_id: str,
    tag: str,
    frame: pd.DataFrame,
    contract: SamplingContract,
    *,
    module_version: str | None = None,
    as_of: str | None = None,
) -> ReplayRecord:
    return ReplayRecord(
        id=record_id,
        tag=tag,
        data_fingerprint=data_fingerprint(frame),
        row_hashes=_rows(frame),
        contract_digest=contract.digest(),
        module_version=module_version if module_version is not None else tsdive.__version__,
        as_of=as_of if as_of is not None else pd.Timestamp.now(tz="UTC").isoformat(),
    )


def save_record(record: ReplayRecord, records_dir: Path) -> Path:
    records_dir.mkdir(parents=True, exist_ok=True)
    path = records_dir / f"{record.id}.json"
    path.write_text(json.dumps(asdict(record), indent=2), encoding="utf-8")
    return path


def load_record(record_id: str, records_dir: Path) -> ReplayRecord:
    path = records_dir / f"{record_id}.json"
    d = json.loads(path.read_text(encoding="utf-8"))
    return ReplayRecord(**d)


def diff_frames(old: ReplayRecord, new_frame: pd.DataFrame) -> list[tuple[str, str]]:
    """Name which sample timestamps moved between the record and now."""
    old_rows, new_rows = old.row_hashes, _rows(new_frame)
    moved: list[tuple[str, str]] = []
    for ts in sorted(set(old_rows) - set(new_rows)):
        moved.append((ts, "removed"))
    for ts in sorted(set(new_rows) - set(old_rows)):
        moved.append((ts, "added (backfill or late write)"))
    for ts in sorted(set(old_rows) & set(new_rows)):
        if old_rows[ts] != new_rows[ts]:
            moved.append((ts, "value/quality changed"))
    return moved


def replay(
    record_id: str,
    frame: pd.DataFrame,
    contract: SamplingContract,
    records_dir: Path,
    *,
    module_version: str | None = None,
) -> ReplayResult:
    """Compare fresh inputs against a stored record.

    MATCH         identical data, identical module version.
    CODE_DRIFT    identical data, module version changed.
    DATA_CHANGED  data changed, same module version; names which samples moved.
    BOTH          both changed.
    """
    old = load_record(record_id, records_dir)
    current_version = (
        module_version if module_version is not None else tsdive.__version__
    )
    data_same = data_fingerprint(frame) == old.data_fingerprint
    contract_same = contract.digest() == old.contract_digest
    code_same = current_version == old.module_version and contract_same

    if data_same and code_same:
        return ReplayResult(ReplayVerdict.MATCH, [])
    if data_same and not code_same:
        return ReplayResult(ReplayVerdict.CODE_DRIFT, [])
    moved = diff_frames(old, frame)
    if not code_same:
        return ReplayResult(ReplayVerdict.BOTH, moved)
    return ReplayResult(ReplayVerdict.DATA_CHANGED, moved)


__all__ = [
    "ReplayRecord",
    "ReplayResult",
    "ReplayVerdict",
    "data_fingerprint",
    "diff_frames",
    "env_fingerprint",
    "load_record",
    "make_record",
    "replay",
    "row_hash",
    "save_record",
]

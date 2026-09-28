"""Parquet-backed window retrieval with a data-physics account.

Read-only by design: this module exposes no way to modify an ingested
archive. A window comes back with the samples, the quality column, and
the physics of what the historian did to it (coverage, gaps, clipping,
timestamp audit, unit resolution).

:func:`write_tag` is the one write path, and it creates archives rather
than editing them: it refuses an existing file unless the caller says
``overwrite=True``, and no store class exposes it.
"""

from __future__ import annotations

import difflib
import json
import math
import re
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pyarrow as pa
from pyarrow import parquet

from tsdive.errors import IncomparableSamplingError, NonMonotonicIndex, SchemaError
from tsdive.store import gaps as gaps_mod
from tsdive.store import units as units_mod
from tsdive.store.clipping import ClippingReport, clipped_fraction
from tsdive.store.identity import Role, TagIdentity, TagMeta
from tsdive.store.quality import (
    Severity,
    annotate_severity,
    is_byte_sized_code,
    is_unmapped,
    null_digital_state_values,
    severity_counts,
    severity_meets,
    validate_schema,
)
from tsdive.store.sampling_contract import SamplingContract
from tsdive.store.timebase import TimestampAuditReport, require_utc, timestamp_audit

META_KEY = "tsdive.meta"

# Characters no common filesystem accepts in a name, plus control characters.
_UNSAFE_IN_FILENAME = re.compile(r'[\\/:*?"<>|\x00-\x1f\x7f]')


def safe_filename(text: str) -> str:
    """Return ``text`` with every character a filesystem rejects replaced by ``_``.

    Replaces ``\\ / : * ? " < > |`` and control characters, so a tag such as
    ``FIC101/PV:1`` lands as ``FIC101_PV_1``. Two names can collide after
    replacement; callers that write files check for that first.
    """
    return _UNSAFE_IN_FILENAME.sub("_", text)


@dataclass
class DataPhysics:
    coverage: gaps_mod.CoverageSummary
    severity_counts: dict[Severity, int]
    clipping: ClippingReport
    timestamp_audit: TimestampAuditReport
    unit: units_mod.UnitResolution
    # Rows carrying a usable value over rows in the window. Coverage asks
    # whether the historian emitted a row; this asks whether the row had
    # anything in it. On a historian that writes one row per scan
    # regardless, the two come apart and only this one moves. ``None``
    # when the window holds no rows at all - a fraction of nothing.
    valid_fraction: float | None = None
    unmapped_quality_codes: list[str] = field(default_factory=list)
    # Uncovered seconds at the window edges that fell under the gap
    # threshold, so coverage counts them as covered. Reported so a window
    # whose samples stop well before ``end`` states them beside its 1.0.
    edge_slack_s: float = 0.0
    edge_slack_start_s: float = 0.0
    edge_slack_end_s: float = 0.0
    # Every unmapped code is a byte-sized integer and the tag declares no
    # quality_codes: the signature of an OPC DA export read as OPC UA,
    # where 192 is Good and 0 is Bad. Reported, never acted on.
    unmapped_look_like_opc_da: bool = False
    # Rows whose magnitude exceeds float32's largest finite value. Any
    # consumer that casts to float32 - sklearn does, silently - turns them
    # into +-inf. A count of what is in the archive, not a judgement that
    # the value is wrong: without an eng_range nothing here can say so.
    implausible_magnitude_count: int = 0


@dataclass
class Window:
    """One tag's samples from ``start`` to ``end`` and the data physics of the read.

    ``TagStore.read_window`` returns it. ``frame`` holds the columns
    timestamp, value, quality (verbatim from the archive), severity and
    valid. ``physics`` covers the whole requested span, the stretches
    before the first sample and after the last included.
    """

    identity: TagIdentity
    meta: TagMeta
    contract: SamplingContract
    start: pd.Timestamp
    end: pd.Timestamp
    frame: pd.DataFrame  # timestamp, value, quality (verbatim), severity, valid
    physics: DataPhysics


def meta_from_parquet(path: Path) -> TagMeta:
    """Load TagMeta embedded in a parquet file's key-value metadata."""
    schema = parquet.read_schema(path)
    raw = schema.metadata.get(META_KEY.encode("ascii")) if schema.metadata else None
    if raw is None:
        raise SchemaError(f"{path.name}: no {META_KEY} metadata; refusing to invent tag metadata")
    return meta_from_dict(json.loads(raw), label=path.name)


def meta_to_dict(meta: TagMeta) -> dict:
    return {
        "identity": {"source_id": meta.identity.source_id, "point_id": meta.identity.point_id},
        "name": meta.name,
        "unit_raw": meta.unit_raw,
        "unit_canonical": meta.unit_canonical,
        "eng_range_zero": meta.eng_range.zero if meta.eng_range else None,
        "eng_range_span": meta.eng_range.span if meta.eng_range else None,
        "sample_rate_s": meta.sample_rate_s,
        "asset": meta.asset,
        "loop_id": meta.loop_id,
        "role": meta.role.value if meta.role else None,
        "quality_codes": dict(meta.quality_codes) if meta.quality_codes else None,
        "quality_assumed": meta.quality_assumed,
    }


# Every key a tsdive.meta object may carry, in the order docs/SCHEMA.md
# lists them, and the keys of its "identity" object.
META_KEYS = (
    "identity",
    "name",
    "unit_raw",
    "unit_canonical",
    "eng_range_zero",
    "eng_range_span",
    "sample_rate_s",
    "asset",
    "loop_id",
    "role",
    "quality_codes",
    "quality_assumed",
)
IDENTITY_KEYS = ("source_id", "point_id")
_TEXT_KEYS = ("name", "unit_raw", "unit_canonical", "asset", "loop_id")


def _unknown_key(label: str, key: str, known: tuple[str, ...], prefix: str = "") -> SchemaError:
    """The refusal for a key tsdive does not read, naming the closest known key."""
    shown = f"{prefix}{key}"
    if not prefix and key == "eng_range":
        return SchemaError(
            f"{label}: unknown key 'eng_range'; write the range as eng_range_zero "
            "and eng_range_span, two numbers at the top level"
        )
    close = difflib.get_close_matches(key, known, n=1)
    if close:
        return SchemaError(
            f"{label}: unknown key {shown!r}; did you mean {prefix + close[0]!r}?"
        )
    return SchemaError(
        f"{label}: unknown key {shown!r}; the known keys are {', '.join(known)}"
    )


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _check_meta_dict(d: object, label: str) -> None:
    """Raise ``SchemaError`` unless ``d`` is a well-formed ``tsdive.meta`` object.

    Every key is checked at every level, so a misspelt key raises instead
    of dropping the fact it carried. Values are checked for type, and the
    engineering range for both ends being present and a positive span.
    """
    if not isinstance(d, dict):
        raise SchemaError(f"{label}: expected a JSON object of tag metadata")
    for key in d:
        if key not in META_KEYS:
            raise _unknown_key(label, str(key), META_KEYS)
    if "identity" not in d:
        raise SchemaError(f"{label}: missing required key 'identity'")
    ident = d["identity"]
    if not isinstance(ident, dict):
        raise SchemaError(
            f"{label}: identity must be an object with source_id and point_id"
        )
    for key in ident:
        if key not in IDENTITY_KEYS:
            raise _unknown_key(label, str(key), IDENTITY_KEYS, prefix="identity.")
    for key in IDENTITY_KEYS:
        if key not in ident:
            raise SchemaError(f"{label}: missing required key 'identity.{key}'")
        if not isinstance(ident[key], str) or not ident[key]:
            raise SchemaError(
                f"{label}: identity.{key} must be a non-empty string, not {ident[key]!r}"
            )
    if "name" not in d:
        raise SchemaError(f"{label}: missing required key 'name'")
    for key in _TEXT_KEYS:
        value = d.get(key)
        if value is not None and not isinstance(value, str):
            raise SchemaError(f"{label}: {key} must be a string or null, not {value!r}")
    if not isinstance(d["name"], str):
        raise SchemaError(f"{label}: name must be a string, not {d['name']!r}")
    zero, span = d.get("eng_range_zero"), d.get("eng_range_span")
    if (zero is None) != (span is None):
        given, missing = (
            ("eng_range_zero", "eng_range_span") if span is None
            else ("eng_range_span", "eng_range_zero")
        )
        raise SchemaError(
            f"{label}: {given} is set and {missing} is not; state both ends of the "
            "range or neither"
        )
    for key, value in (("eng_range_zero", zero), ("eng_range_span", span)):
        if value is not None and not (_is_number(value) and math.isfinite(value)):
            raise SchemaError(f"{label}: {key} must be a finite number, not {value!r}")
    if span is not None and span <= 0:
        raise SchemaError(
            f"{label}: eng_range_span must be greater than 0, not {span!r}; the range "
            "runs from eng_range_zero to eng_range_zero + eng_range_span"
        )
    rate = d.get("sample_rate_s")
    if rate is not None and not (_is_number(rate) and math.isfinite(rate) and rate > 0):
        raise SchemaError(
            f"{label}: sample_rate_s must be a number of seconds greater than 0, "
            f"not {rate!r}"
        )
    role = d.get("role")
    if role is not None and role not in {r.value for r in Role}:
        raise SchemaError(
            f"{label}: role {role!r} is not one of {', '.join(r.value for r in Role)}"
        )
    codes = d.get("quality_codes")
    if codes is not None and not isinstance(codes, dict):
        raise SchemaError(
            f"{label}: quality_codes must be an object mapping raw codes to severities"
        )
    severities = {s.value for s in Severity}
    for code, declared in (codes or {}).items():
        if not isinstance(declared, str) or declared.strip().upper() not in severities:
            raise SchemaError(
                f"{label}: quality_codes maps {code!r} to {declared!r}; "
                f"expected one of {', '.join(sorted(severities))}"
            )
    assumed = d.get("quality_assumed")
    if assumed is not None and not isinstance(assumed, bool):
        raise SchemaError(f"{label}: quality_assumed must be true or false, not {assumed!r}")


def meta_from_dict(d: dict, *, label: str = META_KEY) -> TagMeta:
    """Build a [`TagMeta`][tsdive.TagMeta] from a ``tsdive.meta`` object.

    ``label`` names the object's origin (a file name) in every refusal.

    Raises:
        SchemaError: a key tsdive does not read, at the top level or under
            ``identity`` (the message names the closest known key); a
            missing required key; a value of the wrong type; one end of the
            engineering range without the other; a span or sample rate of
            0 or less; or a ``quality_codes`` entry that names no severity.

    Examples:
        >>> from tsdive.store.tagstore import meta_from_dict
        >>> meta_from_dict({"identity": {"source_id": "plant1", "point_id": "FIC101.PV"},
        ...                 "name": "FIC-101 flow", "unit": "m3/h"}, label="meta.json")
        Traceback (most recent call last):
        ...
        tsdive.errors.SchemaError: meta.json: unknown key 'unit'; did you mean 'unit_raw'?
    """
    from tsdive.store.identity import EngRange

    _check_meta_dict(d, label)
    eng_range = None
    if d.get("eng_range_zero") is not None and d.get("eng_range_span") is not None:
        eng_range = EngRange(zero=float(d["eng_range_zero"]), span=float(d["eng_range_span"]))
    ident = d["identity"]
    identity = TagIdentity(source_id=ident["source_id"], point_id=ident["point_id"])
    name = d["name"]
    return TagMeta(
        identity=identity,
        name=name,
        unit_raw=d.get("unit_raw"),
        unit_canonical=d.get("unit_canonical"),
        eng_range=eng_range,
        sample_rate_s=d.get("sample_rate_s"),
        asset=d.get("asset"),
        loop_id=d.get("loop_id"),
        role=Role(d["role"]) if d.get("role") else None,
        quality_codes=dict(d["quality_codes"]) if d.get("quality_codes") else None,
        quality_assumed=bool(d.get("quality_assumed", False)),
    )


def _write_parquet(path: Path, frame: pd.DataFrame, meta: TagMeta) -> Path:
    """Serialize one tag's frame with ``tsdive.meta`` attached, unchecked."""
    table = pa.Table.from_pandas(frame.reset_index(drop=True), preserve_index=False)
    table = table.replace_schema_metadata({META_KEY: json.dumps(meta_to_dict(meta))})
    path.parent.mkdir(parents=True, exist_ok=True)
    parquet.write_table(table, path)
    return path


def write_tag(
    path: str | Path,
    frame: pd.DataFrame,
    meta: TagMeta,
    *,
    overwrite: bool = False,
) -> Path:
    """Create a single-tag parquet archive at ``path`` and return it.

    This is an archive *creation* call, not an edit. Ingested archives stay
    read-only: nothing here rewrites, appends to, or patches an existing
    file, and an existing ``path`` raises ``FileExistsError`` unless
    ``overwrite=True``, which replaces the whole archive rather than
    changing part of it.

    ``frame`` must carry ``timestamp`` (UTC-aware), ``value`` and
    ``quality``; ``validate_schema`` refuses anything else, so a
    quality-free archive cannot be created by accident. ``meta`` is
    embedded under the ``tsdive.meta`` key, which is what makes the
    file readable by [`TagStore`][tsdive.TagStore] at all.

    Examples:
        >>> import pandas as pd
        >>> import tsdive
        >>> frame = pd.DataFrame({
        ...     "timestamp": pd.date_range("2024-03-30 20:00", periods=3, freq="min", tz="UTC"),
        ...     "value": [62.0, 62.1, 62.3],
        ...     "quality": ["GOOD", "GOOD", "GOOD"],
        ... })
        >>> meta = tsdive.TagMeta(identity=tsdive.TagIdentity("plant1", "FI102.PV"),
        ...                       name="FI-102 flow", sample_rate_s=60.0)
        >>> path = tsdive.write_tag("archive/plant1/FI102.PV.parquet", frame, meta)
        >>> path.as_posix()
        'archive/plant1/FI102.PV.parquet'
        >>> tsdive.write_tag(path, frame, meta)
        Traceback (most recent call last):
        ...
        FileExistsError: ...
    """
    validate_schema(frame)
    out = Path(path)
    if out.exists() and not overwrite:
        raise FileExistsError(
            f"{out} already exists; tsdive does not mutate archives. "
            "Pass overwrite=True to replace it, or write to a new path."
        )
    return _write_parquet(out, frame, meta)


def archive_extent(path: str | Path) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return ``(first, last)`` timestamp of an archive, UTC-aware.

    This is the default window when the caller states none, because the
    extent is already in the file whereas any narrower default is not. An
    archive with no rows has no extent and is refused rather than reported
    as an instant.
    """
    file = Path(path)
    schema = parquet.read_schema(file)
    if "timestamp" not in schema.names:
        raise SchemaError(f"{file.name}: missing required column 'timestamp'")
    ts = cast(pd.Series, parquet.read_table(file, columns=["timestamp"])["timestamp"].to_pandas())
    if len(ts) == 0:
        raise SchemaError(
            f"{file.name}: archive has no samples, so it has no extent; "
            "state a window explicitly"
        )
    require_utc(ts)
    return cast(pd.Timestamp, ts.min()), cast(pd.Timestamp, ts.max())


class TagStore:
    """Read-only view over ``root/<source_id>/<point_id>.parquet`` files."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def archive_path(self, identity: TagIdentity) -> Path:
        """The path ``root/<source_id>/<point_id>.parquet`` of ``identity``, existing or not.

        Examples:
            >>> import tsdive
            >>> store = tsdive.TagStore("data")
            >>> store.archive_path(tsdive.TagIdentity("demo", "fic101_demo")).as_posix()
            'data/demo/fic101_demo.parquet'
        """
        return self.root / identity.source_id / f"{identity.point_id}.parquet"

    def list_tags(self) -> list[TagIdentity]:
        """One identity per ``root/<source_id>/<point_id>.parquet`` file, sorted.

        The identity is read off the path, not off the file's ``tsdive.meta``.

        Examples:
            >>> import tsdive
            >>> store = tsdive.TagStore("data")
            >>> [str(tag) for tag in store.list_tags()][:2]
            ['demo:fic101_demo', 'demo:tic101_demo']
        """
        found: list[TagIdentity] = []
        for source_dir in sorted(self.root.iterdir()) if self.root.exists() else []:
            if not source_dir.is_dir():
                continue
            for f in sorted(source_dir.glob("*.parquet")):
                found.append(TagIdentity(source_id=source_dir.name, point_id=f.stem))
        return found

    def read_window(
        self,
        identity: TagIdentity,
        start: datetime,
        end: datetime,
        contract: SamplingContract,
        *,
        min_severity: Severity = Severity.GOOD,
        tz_names: tuple[str, ...] = (),
    ) -> Window:
        """Return a window plus its data-physics block.

        ``contract`` is a required positional argument: every read states
        how its numbers were produced.

        Coverage accounts for the whole requested span, not just the part
        between the first and last sample: the interval from ``start`` to
        the first sample and from the last sample to ``end`` are classified
        by the same gap rules and count toward ``n_gaps``, ``longest_gap_s``
        and coverage. A window with no samples reports ``coverage=0.0`` and
        one ``unknown`` gap spanning the window - never a fake 1.0.

        A window whose timestamps go backwards is refused with
        [`NonMonotonicIndex`][tsdive.NonMonotonicIndex] before any physics is
        computed. Gaps and coverage over a re-sorted index would describe
        an ordering the historian never produced.
        """
        path = self.archive_path(identity)
        if not path.exists():
            raise FileNotFoundError(f"no archive at {path}")
        table = parquet.read_table(path)
        df = table.to_pandas()
        validate_schema(df)
        require_utc(df["timestamp"])

        start_ts = cast(pd.Timestamp, pd.Timestamp(start))
        end_ts = cast(pd.Timestamp, pd.Timestamp(end))
        in_window = df[(df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)]
        meta = meta_from_parquet(path)
        annotated = annotate_severity(in_window, meta.quality_codes)
        annotated = null_digital_state_values(
            annotated, role=meta.role, tag=identity, codes=meta.quality_codes
        )

        # A MODE tag's value is a state label, not a measurement: "R1" is
        # present and usable, and asking whether it is a finite float is
        # the wrong question.
        state_valued = meta.role is Role.MODE
        value_ok = (
            annotated["value"].notna()
            if state_valued
            else pd.to_numeric(annotated["value"], errors="coerce").notna()
        )
        valid_mask = severity_meets(annotated["severity"], min_severity) & value_ok
        # Explicit bool dtype: an empty mask infers float64, and
        # `frame[frame["valid"]]` on a non-bool series reads as column
        # selection instead of a row mask.
        annotated["valid"] = pd.Series(
            valid_mask.to_numpy(dtype=bool), dtype=bool, index=annotated.index
        )

        ts = cast(pd.Series, annotated["timestamp"].reset_index(drop=True))
        audit = timestamp_audit(identity, annotated, tz_names=tz_names)
        if audit.non_monotonic_positions:
            first = audit.non_monotonic_positions[0]
            raise NonMonotonicIndex(
                f"{identity}: timestamp at window position {first} "
                f"({ts.iloc[first].isoformat()}) precedes the sample before it "
                f"({ts.iloc[first - 1].isoformat()}); coverage, gaps and averages "
                "would describe an ordering the historian never had - fix the "
                "export rather than have tsdive sort it",
                offending_positions=audit.non_monotonic_positions,
            )

        # Monotonic by the refusal above, so every interval is >= 0.
        intervals = ts.diff().dt.total_seconds().to_numpy()[1:]
        median_interval = (
            float(pd.Series(intervals).median()) if len(intervals) > 0 else None
        )
        gap_threshold = (
            max(2.5 * median_interval, 60.0) if median_interval else 3600.0
        )
        gap_context = gaps_mod.GapContext(
            contract_stepped=contract.stepped,
            observed_median_interval_s=median_interval,
            declared_sample_rate_s=meta.sample_rate_s,
        )
        # An edge gap has a sample on one side only. compression_steady
        # claims the value was unchanged across the hole, which needs a
        # sample on both sides to assert, so that rule is withheld at the
        # edges; spacing rules still apply.
        edge_context = replace(gap_context, contract_stepped=False)
        findings: list[gaps_mod.GapFinding] = []
        # Window edges are gaps too: samples that stop 19 minutes before the
        # window ends leave 19 minutes uncovered, whatever happened between
        # the samples that are present.
        raw_gaps: list[tuple[pd.Timestamp, pd.Timestamp, float, gaps_mod.GapContext]] = []
        leading_slack = 0.0
        trailing_slack = 0.0
        if len(ts) == 0:
            whole = (end_ts - start_ts).total_seconds()
            if whole > 0:
                findings.append(
                    gaps_mod.GapFinding(
                        start=start_ts,
                        end=end_ts,
                        duration_s=whole,
                        classification=gaps_mod.GapClassification(
                            gaps_mod.GapClass.UNKNOWN,
                            "no samples in window; nothing observed",
                        ),
                    )
                )
        else:
            first_ts = cast(pd.Timestamp, ts.min())
            last_ts = cast(pd.Timestamp, ts.max())
            leading = (first_ts - start_ts).total_seconds()
            if leading > gap_threshold:
                raw_gaps.append((start_ts, first_ts, leading, edge_context))
            else:
                leading_slack = max(0.0, leading)
            raw_gaps.extend(
                (g_start, g_end, duration, gap_context)
                for g_start, g_end, duration in gaps_mod.detect_gaps(ts, gap_threshold)
            )
            trailing = (end_ts - last_ts).total_seconds()
            if trailing > gap_threshold:
                raw_gaps.append((last_ts, end_ts, trailing, edge_context))
            else:
                trailing_slack = max(0.0, trailing)
        for g_start, g_end, duration, ctx in raw_gaps:
            findings.append(
                gaps_mod.GapFinding(
                    start=g_start,
                    end=g_end,
                    duration_s=duration,
                    classification=gaps_mod.classify_gap(duration, ctx),
                )
            )

        counts = severity_counts(annotated)
        # is_unmapped is a pure function of the raw code, so the distinct
        # codes decide the answer for every row carrying them.
        unmapped = sorted(
            {
                str(q)
                for q in annotated["quality"].drop_duplicates()
                if is_unmapped(q, meta.quality_codes)
            }
        )
        looks_opc_da = (
            bool(unmapped)
            and meta.quality_codes is None
            and all(is_byte_sized_code(q) for q in unmapped)
        )

        physics = DataPhysics(
            coverage=gaps_mod.summarize_coverage(start_ts, end_ts, tuple(findings)),
            severity_counts=counts,
            clipping=clipped_fraction(
                annotated["value"], annotated["quality"], meta.eng_range
            ),
            timestamp_audit=audit,
            unit=units_mod.resolve_unit(meta.unit_raw),
            valid_fraction=(
                float(annotated["valid"].sum()) / len(annotated) if len(annotated) else None
            ),
            unmapped_quality_codes=unmapped,
            edge_slack_s=leading_slack + trailing_slack,
            edge_slack_start_s=leading_slack,
            edge_slack_end_s=trailing_slack,
            unmapped_look_like_opc_da=looks_opc_da,
            implausible_magnitude_count=_beyond_float32(annotated["value"]),
        )

        return Window(
            identity=identity,
            meta=meta,
            contract=contract,
            start=start_ts,
            end=end_ts,
            frame=annotated.reset_index(drop=True),
            physics=physics,
        )


FLOAT32_MAX = float(np.finfo(np.float32).max)


def _beyond_float32(values: pd.Series) -> int:
    """Rows whose |value| exceeds the largest finite float32."""
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    return int((np.abs(numeric) > FLOAT32_MAX).sum())


def assert_windows_comparable(a: Window, b: Window) -> None:
    """Refuse mixing windows produced under different calculation bases."""
    if not a.contract.comparable_with(b.contract):
        raise IncomparableSamplingError(
            f"cannot mix windows: {a.identity} is {a.contract.calculation_basis.value}, "
            f"{b.identity} is {b.contract.calculation_basis.value}; these are different "
            "quantities"
        )


class SingleFileStore(TagStore):
    """Read-only view over one explicit parquet file (no layout assumptions)."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        super().__init__(self.path.parent)

    def archive_path(self, identity: TagIdentity) -> Path:
        """The store's one file, whatever ``identity`` names."""
        return self.path

    def list_tags(self) -> list[TagIdentity]:
        """The identity the file's ``tsdive.meta`` declares, as a one-item list.

        Examples:
            >>> import tsdive
            >>> store = tsdive.SingleFileStore("data/demo/fic101_demo.parquet")
            >>> [str(tag) for tag in store.list_tags()]
            ['demo:FIC101.PV']
        """
        return [meta_from_parquet(self.path).identity]

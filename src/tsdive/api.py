"""Library entry points behind the CLI commands.

:func:`profile` runs the whole profiling flow (parse the window, open the
archive, read the span, optionally assess flatline) and hands back
objects. Rendering lives in :mod:`tsdive.report`; only the CLI turns a
:class:`Profile` into lines, so a caller who wants the numbers never has
to parse text back out of a report.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import pandas as pd

from tsdive.detectors.flatline import FlatlineVerdict, assess_flatline
from tsdive.errors import SchemaError
from tsdive.features.window_features import WindowStats, compute_stats
from tsdive.report import render_window_report
from tsdive.store.identity import TagIdentity, TagMeta
from tsdive.store.quality import (
    Severity,
    annotate_severity,
    good_mask,
    null_digital_state_values,
)
from tsdive.store.sampling_contract import (
    AggregateType,
    CalculationBasis,
    RetrievalMode,
    SamplingContract,
)
from tsdive.store.tagstore import (
    DataPhysics,
    SingleFileStore,
    Window,
    archive_extent,
    meta_from_dict,
    meta_from_parquet,
    safe_filename,
    write_tag,
)


def parse_window(spec: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Parse ``START/END`` into two UTC-aware timestamps.

    Naive bounds are rejected: a window whose offset is unstated is not a
    window, and tsdive stores UTC only.
    """
    if "/" not in spec:
        raise ValueError("window must be <START>/<END> in ISO 8601 UTC")
    start_s, end_s = spec.split("/", maxsplit=1)
    start = pd.Timestamp(start_s)
    end = pd.Timestamp(end_s)
    for name, ts in (("START", start), ("END", end)):
        if ts.tz is None:
            raise ValueError(
                f"window {name} is naive ({ts}); append 'Z' or '+00:00' - "
                "tsdive stores UTC only"
            )
    return cast(pd.Timestamp, start), cast(pd.Timestamp, end)


def validate_tz_names(names: Sequence[str]) -> None:
    """Reject unknown IANA names before any archive is read.

    ZoneInfoNotFoundError subclasses KeyError, which no caller catches;
    without this a typo surfaces as a traceback.
    """
    for name in names:
        try:
            ZoneInfo(name)
        except ZoneInfoNotFoundError as e:
            raise ValueError(f"unknown tz {name!r}: not an IANA time zone name") from e


def reference_history(
    path: Path, start: pd.Timestamp, end: pd.Timestamp
) -> tuple[list[float], list[int]]:
    """Pooled change intervals and per-window distinct counts from preceding spans.

    Thresholds come from the same archive, strictly before the assessed
    window.
    """
    table = cast(pd.DataFrame, pd.read_parquet(path))
    # The tag's own quality_codes decide what GOOD means here, exactly as
    # they do on the assessed window. Without them a source that codes
    # quality in its own vocabulary has zero GOOD reference samples, and
    # the thresholds silently come back empty.
    codes = meta_from_parquet(Path(path)).quality_codes
    table = table.sort_values("timestamp", kind="stable").reset_index(drop=True)
    ts = cast(pd.Series, table["timestamp"])
    span = end - start

    # The window edges, walked exactly as before: cursor from the first
    # sample, one span at a time, stopping before the assessed window.
    edges: list[pd.Timestamp] = []
    cursor = cast(pd.Timestamp, ts.min())
    while bool(cursor + span < start):
        edges.append(cursor)
        cursor = cursor + span
    if not edges:
        return [], []
    boundaries = pd.DatetimeIndex([*edges, edges[-1] + span])

    good = table[good_mask(table, codes) & table["value"].notna()].reset_index(drop=True)
    # Which reference window each GOOD sample belongs to. Rows at or after
    # the last boundary sit between the references and the assessed
    # window; searchsorted puts them out of range and they are dropped.
    # Compared as UTC nanoseconds: searchsorted over boxed Timestamps
    # would be a Python compare per sample.
    window_of = (
        np.searchsorted(
            pd.Series(boundaries).astype("int64").to_numpy(),
            good["timestamp"].astype("int64").to_numpy(),
            side="right",
        )
        - 1
    )
    inside = (window_of >= 0) & (window_of < len(edges))
    good = good[inside].reset_index(drop=True)
    window_of = window_of[inside]

    # Values are compared as stored, never coerced to float, so a MODE
    # tag's states ("R0" -> "R1") count as changes like any other.
    values = cast(pd.Series, good["value"])
    counts = values.groupby(window_of).nunique() if len(good) else pd.Series(dtype="int64")
    distinct_counts = [int(counts.get(i, 0)) for i in range(len(edges))]

    if len(good) < 2:
        return [], distinct_counts
    # A change EVENT is a row whose value differs from the previous row of
    # the same reference window. The first row of a window compares
    # against nothing: a pair straddling two windows was never compared
    # before and is not compared now, because each window's history is
    # its own.
    changed = values.ne(values.shift()).to_numpy(dtype=bool, na_value=False)
    first_of_window = np.empty(len(good), dtype=bool)
    first_of_window[0] = True
    first_of_window[1:] = window_of[1:] != window_of[:-1]
    at = np.flatnonzero(changed & ~first_of_window)
    if at.size < 2:
        return [], distinct_counts

    # The interval is the time BETWEEN successive change events, not the
    # sample spacing at which a change happened to be noticed. Signal 1
    # measures how long the value has sat still and holds it against this
    # p99, so the reference has to be the same quantity: a tag sampled
    # every second that moves every second has a 1 s interval, and one
    # that moves every tenth sample has a 10 s interval, where the
    # spacing would call both of them 1 s.
    stamps = good["timestamp"].astype("int64").to_numpy()[at]
    in_window = window_of[at]
    gaps = np.diff(stamps)
    same_window = in_window[1:] == in_window[:-1]
    intervals = [float(g) / 1e9 for g in gaps[same_window]]
    return intervals, distinct_counts


def read_meta_json(path: str | Path) -> TagMeta:
    """Load a :class:`TagMeta` from a JSON file.

    The file uses the same object as the archive's own ``tsdive.meta``
    block, so metadata written for ingest is readable back off the
    archive without a second format to keep in step.
    """
    text = Path(path).read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as e:
        raise SchemaError(f"{Path(path).name}: not valid JSON ({e})") from e
    if not isinstance(payload, dict):
        raise SchemaError(f"{Path(path).name}: expected a JSON object of tag metadata")
    severities = {s.value for s in Severity}
    for code, declared in (payload.get("quality_codes") or {}).items():
        if not isinstance(declared, str) or declared.strip().upper() not in severities:
            raise SchemaError(
                f"{Path(path).name}: quality_codes maps {code!r} to {declared!r}; "
                f"expected one of {', '.join(sorted(severities))}"
            )
    return meta_from_dict(payload)


def _read_source(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return cast(pd.DataFrame, pd.read_csv(path))
    if path.suffix.lower() in {".parquet", ".pq"}:
        return cast(pd.DataFrame, pd.read_parquet(path))
    raise SchemaError(
        f"{path.name}: unsupported input {path.suffix!r}; ingest reads .csv and .parquet"
    )


def _parse_timestamps(raw: pd.Series, column: str) -> list[pd.Timestamp]:
    """Parse each element to a Timestamp, refusing the first unparseable one.

    Element-wise on purpose. ``pd.to_datetime`` over a whole column whose
    rows carry different UTC offsets returns object dtype (and warns that
    a future pandas will raise), which makes every ``.dt`` accessor
    downstream fail with an AttributeError instead of a typed refusal.
    """
    stamps: list[pd.Timestamp] = []
    for v in raw:
        ts = cast(pd.Timestamp, pd.to_datetime(v, errors="coerce"))
        if ts is pd.NaT or pd.isna(ts):
            raise SchemaError(f"{column}: {v!r} is not a timestamp tsdive can parse")
        stamps.append(ts)
    return stamps


def _to_utc(raw: pd.Series, tz: str | None, column: str) -> pd.Series:
    """Parse a source timestamp column into UTC, or refuse.

    Naive timestamps carry no offset, so tsdive cannot know what
    instant they name. With ``tz`` the caller states the source's zone and
    the column is localised then converted; without it, the ingest is
    refused rather than assuming UTC.

    Rows with different UTC offsets all name real instants and are
    converted individually. A column that mixes naive and offset-bearing
    rows is refused: ``tz`` would have to be applied to some rows and not
    others, and the export is telling two stories about its own clock.
    """
    stamps = _parse_timestamps(raw, column)
    aware = [ts.tz is not None for ts in stamps]
    if any(aware) and not all(aware):
        raise SchemaError(
            f"{column}: mixed naive and offset-bearing timestamps; the naive rows "
            "name no instant tsdive can place next to the aware ones - export "
            "the whole column with UTC offsets"
        )
    if stamps and all(aware):
        return pd.Series(
            pd.DatetimeIndex([ts.tz_convert("UTC") for ts in stamps]), index=raw.index
        )
    if tz is None:
        raise SchemaError(
            f"{column} is naive (no UTC offset); pass --tz <IANA zone> to state what "
            "zone the source is in - tsdive will not assume UTC"
        )
    validate_tz_names((tz,))
    local = pd.Series(pd.DatetimeIndex(stamps), index=raw.index)
    try:
        localised = local.dt.tz_localize(tz)
    except Exception as e:
        raise SchemaError(
            f"{column}: cannot localise to {tz} ({e}); a DST transition makes some of "
            "these local times ambiguous or nonexistent - export with UTC offsets"
        ) from e
    return cast(pd.Series, localised.dt.tz_convert("UTC"))


def _resolve_quality(
    frame: pd.DataFrame,
    *,
    label: str,
    quality_col: str | None,
    assume_quality: str | None,
) -> tuple[pd.Series, bool]:
    """The quality column to archive, and whether tsdive wrote it.

    Returns the source's own column when ``quality_col`` names one, else a
    constant column of ``assume_quality``. Raises SchemaError when both
    are present, when neither is, or when ``assume_quality`` names no
    severity.
    """
    if quality_col is not None and quality_col in frame.columns:
        if assume_quality is not None:
            raise SchemaError(
                f"{label}: --assume-quality was given but {quality_col!r} exists; "
                "refusing to overwrite the source's own quality codes"
            )
        return frame[quality_col], False
    if assume_quality is None:
        raise SchemaError(
            f"{label}: no quality column {quality_col!r}. A value without a "
            "quality code is not a measurement; name the right column with "
            "--quality-col, or state --assume-quality GOOD|UNCERTAIN|BAD, which "
            "is recorded on the archive and reported in every profile"
        )
    declared = assume_quality.strip().upper()
    if declared not in {s.value for s in Severity}:
        raise SchemaError(
            f"--assume-quality {assume_quality!r} is not a severity; "
            f"expected one of {', '.join(s.value for s in Severity)}"
        )
    return pd.Series([declared] * len(frame), index=frame.index), True


def _prepare_archive(
    *,
    timestamps: pd.Series,
    values: pd.Series,
    quality: pd.Series,
    assumed: bool,
    meta: TagMeta,
) -> tuple[pd.DataFrame, TagMeta]:
    """Assemble one archive frame and check it, without writing."""
    out_frame = pd.DataFrame(
        {"timestamp": timestamps, "value": values, "quality": quality}
    ).reset_index(drop=True)
    stamped = replace(meta, quality_assumed=assumed or meta.quality_assumed)
    # Refuse a bad export here rather than at read time: everything the
    # read path checks about values is checkable now.
    null_digital_state_values(
        annotate_severity(out_frame, stamped.quality_codes),
        role=stamped.role,
        tag=stamped.identity,
    )
    return out_frame, stamped


def ingest(
    source: str | Path,
    *,
    out: str | Path,
    meta: TagMeta,
    timestamp_col: str = "timestamp",
    value_col: str = "value",
    quality_col: str = "quality",
    tz: str | None = None,
    assume_quality: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Turn a CSV or parquet export into a tsdive archive.

    Only the three declared columns are carried over; everything else in
    the export is left behind, so an ingested archive contains exactly
    what its schema promises.

    A missing quality column is a refusal, not a default: a value whose
    trustworthiness is unknown is not a measurement. ``assume_quality``
    is the explicit override, and it is recorded on the archive
    (``quality_assumed``) and printed by every profile of it.

    Raises:
        SchemaError: unreadable input, missing column, naive timestamps
            without ``tz``, or an unusable ``assume_quality`` value.
        FileExistsError: ``out`` exists and ``overwrite`` is False.
    """
    src = Path(source)
    frame = _read_source(src)
    for name, column in (("timestamp", timestamp_col), ("value", value_col)):
        if column not in frame.columns:
            raise SchemaError(
                f"{src.name}: no {name} column {column!r}; columns are "
                f"{', '.join(map(str, frame.columns))}"
            )
    quality, assumed = _resolve_quality(
        frame, label=src.name, quality_col=quality_col, assume_quality=assume_quality
    )
    out_frame, stamped = _prepare_archive(
        timestamps=_to_utc(frame[timestamp_col], tz, timestamp_col),
        values=frame[value_col],
        quality=quality,
        assumed=assumed,
        meta=meta,
    )
    return write_tag(out, out_frame, stamped, overwrite=overwrite)


def _wide_columns(
    frame: pd.DataFrame,
    *,
    label: str,
    timestamp_col: str,
    tags: Sequence[str] | None,
    quality_suffix: str | None,
) -> list[str]:
    """The tag columns of a wide export, each checked to exist."""
    columns = [str(c) for c in frame.columns]
    listed = ", ".join(columns)
    if timestamp_col not in columns:
        raise SchemaError(
            f"{label}: no timestamp column {timestamp_col!r}; columns are {listed}"
        )
    if tags is not None:
        chosen = list(dict.fromkeys(str(t) for t in tags))
        for tag in chosen:
            if tag not in columns:
                raise SchemaError(
                    f"{label}: no column {tag!r} named in --tags; columns are {listed}"
                )
    else:
        quality_cols = {f"{c}{quality_suffix}" for c in columns} if quality_suffix else set()
        chosen = [c for c in columns if c != timestamp_col and c not in quality_cols]
    if not chosen:
        raise SchemaError(f"{label}: no tag columns besides {timestamp_col!r}")
    return chosen


def _wide_quality_columns(
    frame: pd.DataFrame,
    chosen: Sequence[str],
    *,
    label: str,
    quality_suffix: str | None,
) -> dict[str, str | None]:
    """Each tag's quality column, ``None`` for every tag when there is no suffix."""
    quality_cols: dict[str, str | None] = {}
    for tag in chosen:
        qcol = f"{tag}{quality_suffix}" if quality_suffix is not None else None
        if qcol is not None and qcol not in frame.columns:
            raise SchemaError(
                f"{label}: no quality column {qcol!r} for tag {tag!r}; columns are "
                f"{', '.join(map(str, frame.columns))}"
            )
        quality_cols[tag] = qcol
    return quality_cols


def _check_distinct_filenames(names: Sequence[str], *, what: str) -> None:
    """Raise ValueError when two names share one :func:`safe_filename`."""
    seen: dict[str, str] = {}
    for name in names:
        key = safe_filename(name)
        if key in seen:
            if seen[key] == name:
                raise ValueError(f"{what} {name!r} appears twice; each needs its own file")
            raise ValueError(
                f"{what} {seen[key]!r} and {name!r} both map to the file name {key!r}; "
                "rename one of them"
            )
        seen[key] = name


def ingest_wide(
    source: str | Path,
    *,
    out_dir: str | Path,
    meta_dir: str | Path,
    timestamp_col: str = "timestamp",
    tags: Sequence[str] | None = None,
    quality_suffix: str | None = None,
    tz: str | None = None,
    assume_quality: str | None = None,
    overwrite: bool = False,
) -> list[Path]:
    """Turn a wide export, one column per tag, into one archive per tag.

    The file is read once. Tag columns are ``tags`` when given, else every
    column that is not ``timestamp_col`` and not a quality column; with
    ``quality_suffix`` a tag's quality column is ``f"{tag}{quality_suffix}"``.
    Metadata for a tag comes from ``meta_dir / f"{safe_filename(tag)}.json"``
    and its archive goes to ``out_dir / f"{safe_filename(point_id)}.parquet"``.
    Every check runs before the first archive is written, so a refusal
    leaves ``out_dir`` as it was.

    Raises:
        SchemaError: unreadable input; a missing timestamp, tag, quality
            or metadata file; naive timestamps without ``tz``; or
            ``quality_suffix`` given together with ``assume_quality``.
        ValueError: two tags or two point ids that share one file name.
        FileExistsError: an archive exists and ``overwrite`` is False.
    """
    src = Path(source)
    frame = _read_source(src)
    label = src.name
    chosen = _wide_columns(
        frame,
        label=label,
        timestamp_col=timestamp_col,
        tags=tags,
        quality_suffix=quality_suffix,
    )
    _check_distinct_filenames(chosen, what="tags")
    if quality_suffix is not None and assume_quality is not None:
        raise SchemaError(
            f"{label}: --assume-quality was given with --quality-suffix "
            f"{quality_suffix!r}; the export carries its own quality codes, so "
            "drop one of the two"
        )
    if quality_suffix is None and assume_quality is None:
        raise SchemaError(
            f"{label}: no --quality-suffix and no --assume-quality; name the suffix "
            "of the per-tag quality columns with --quality-suffix, or state "
            "--assume-quality GOOD|UNCERTAIN|BAD, which is recorded on every "
            "archive and reported in every profile"
        )
    quality_cols = _wide_quality_columns(
        frame, chosen, label=label, quality_suffix=quality_suffix
    )
    meta_root = Path(meta_dir)
    metas: dict[str, TagMeta] = {}
    for tag in chosen:
        meta_path = meta_root / f"{safe_filename(tag)}.json"
        if not meta_path.exists():
            raise SchemaError(
                f"tag {tag!r}: no metadata file {meta_path.as_posix()}; write it by "
                "hand or with --init-meta"
            )
        metas[tag] = read_meta_json(meta_path)
    _check_distinct_filenames(
        [m.identity.point_id for m in metas.values()], what="point ids"
    )
    root = Path(out_dir)
    targets = {
        tag: root / f"{safe_filename(metas[tag].identity.point_id)}.parquet"
        for tag in chosen
    }
    if not overwrite:
        for target in targets.values():
            if target.exists():
                raise FileExistsError(
                    f"{target} already exists; pass overwrite=True to replace it"
                )
    timestamps = _to_utc(frame[timestamp_col], tz, timestamp_col)
    prepared: list[tuple[Path, pd.DataFrame, TagMeta]] = []
    for tag in chosen:
        quality, assumed = _resolve_quality(
            frame, label=label, quality_col=quality_cols[tag], assume_quality=assume_quality
        )
        out_frame, stamped = _prepare_archive(
            timestamps=timestamps,
            values=frame[tag],
            quality=quality,
            assumed=assumed,
            meta=metas[tag],
        )
        prepared.append((targets[tag], out_frame, stamped))
    return [
        write_tag(target, out_frame, stamped, overwrite=overwrite)
        for target, out_frame, stamped in prepared
    ]


# The keys of a metadata template, in the order docs/SCHEMA.md lists them.
_META_TEMPLATE_KEYS = (
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


def _quality_code_template(raw_codes: pd.Series) -> dict[str, str | None]:
    """One entry per distinct raw quality value, filled only where the value names a severity."""
    severities = {s.value for s in Severity}
    template: dict[str, str | None] = {}
    for raw in sorted({str(v) for v in raw_codes.dropna()}):
        folded = raw.strip().upper()
        template[raw] = folded if folded in severities else None
    return template


def init_meta(
    source: str | Path,
    *,
    out_dir: str | Path,
    source_id: str,
    timestamp_col: str = "timestamp",
    tags: Sequence[str] | None = None,
    quality_suffix: str | None = None,
    overwrite: bool = False,
) -> list[Path]:
    """Write one metadata template per tag column of a wide export.

    Each template at ``out_dir / f"{safe_filename(tag)}.json"`` carries
    ``identity`` (``source_id`` and the column name as ``point_id``),
    ``name`` (the column name) and every optional key of ``tsdive.meta``
    set to ``null``. Nothing about units, ranges or sample rate is read
    off the data. With ``quality_suffix``, ``quality_codes`` lists every
    distinct raw value of the tag's quality column, mapped to a severity
    only where the value spells ``GOOD``, ``UNCERTAIN`` or ``BAD``
    itself; every other code stays ``null`` for the reader to fill, and
    :func:`read_meta_json` refuses the file until they are.

    Raises:
        SchemaError: unreadable input, or a missing timestamp, tag or
            quality column.
        ValueError: two tags that share one file name.
        FileExistsError: a template exists and ``overwrite`` is False.
    """
    src = Path(source)
    frame = _read_source(src)
    chosen = _wide_columns(
        frame,
        label=src.name,
        timestamp_col=timestamp_col,
        tags=tags,
        quality_suffix=quality_suffix,
    )
    _check_distinct_filenames(chosen, what="tags")
    quality_cols = _wide_quality_columns(
        frame, chosen, label=src.name, quality_suffix=quality_suffix
    )
    root = Path(out_dir)
    targets = {tag: root / f"{safe_filename(tag)}.json" for tag in chosen}
    if not overwrite:
        for target in targets.values():
            if target.exists():
                raise FileExistsError(
                    f"{target} already exists; pass overwrite=True to replace it"
                )
    root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for tag in chosen:
        payload: dict[str, object] = {
            "identity": {"source_id": source_id, "point_id": tag},
            "name": tag,
            **dict.fromkeys(_META_TEMPLATE_KEYS),
        }
        qcol = quality_cols[tag]
        if qcol is not None:
            payload["quality_codes"] = _quality_code_template(frame[qcol])
        targets[tag].write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        written.append(targets[tag])
    return written


@dataclass(frozen=True)
class Profile:
    """One archive window, its statistics, and an optional flatline verdict.

    Holds objects, not text: :meth:`render` is the only place the report
    lines are produced, and it is the same renderer the CLI prints.
    """

    window: Window
    stats: WindowStats
    flatline: FlatlineVerdict | None = None

    def render(self) -> str:
        return render_window_report(self.window, self.flatline, self.stats)

    @property
    def identity(self) -> TagIdentity:
        return self.window.identity

    @property
    def meta(self) -> TagMeta:
        return self.window.meta

    @property
    def physics(self) -> DataPhysics:
        return self.window.physics

    @property
    def frame(self) -> pd.DataFrame:
        return self.window.frame


def profile(
    path: str | Path,
    window: str | None = None,
    *,
    tz: str | Sequence[str] | None = None,
    basis: str | CalculationBasis = CalculationBasis.TIME_WEIGHTED,
    stepped: bool = False,
    flatline: bool = False,
) -> Profile:
    """Profile one window of a single-tag archive.

    Args:
        path: parquet archive carrying ``tsdive.meta``.
        window: ``START/END`` in ISO 8601 UTC. Omitted, the window is the
            archive's own extent (first to last timestamp), which the
            rendered report states like any other window.
        tz: IANA name(s) to audit for DST transitions inside the window.
        basis: calculation basis declared on the read.
        stepped: stepped interpolation between samples.
        flatline: assess flatline using prior equal-size windows of the
            same archive as references.

    Raises:
        ValueError: malformed or naive window, unknown tz name.
        TSDiveError: any typed refusal from the read path.
    """
    tz_names = (tz,) if isinstance(tz, str) else tuple(tz or ())
    validate_tz_names(tz_names)
    store = SingleFileStore(Path(path))
    start, end = parse_window(window) if window else archive_extent(store.path)
    meta = meta_from_parquet(store.path)
    contract = SamplingContract(
        calculation_basis=CalculationBasis(basis),
        retrieval_mode=RetrievalMode.RECORDED,
        aggregate_type=AggregateType.NONE,
        stepped=stepped,
    )
    read = store.read_window(
        meta.identity,
        start.to_pydatetime(),
        end.to_pydatetime(),
        contract,
        tz_names=tz_names,
    )
    verdict = None
    if flatline:
        intervals, counts = reference_history(store.path, start, end)
        verdict = assess_flatline(
            read,
            reference_change_intervals_s=intervals,
            reference_distinct_counts=counts,
        )
    return Profile(window=read, stats=compute_stats(read), flatline=verdict)

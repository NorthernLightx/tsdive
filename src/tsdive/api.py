"""Library entry points behind the CLI commands.

:func:`profile` runs the whole profiling flow (parse the window, open the
archive, read the span, optionally assess flatline) and hands back
objects. Rendering lives in :mod:`tsdive.report`; only the CLI turns a
:class:`Profile` into lines, so a caller who wants the numbers never has
to parse text back out of a report.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import pandas as pd
import pyarrow as pa

from tsdive.detectors.flatline import FlatlineVerdict, assess_flatline
from tsdive.errors import SchemaError
from tsdive.features.window_features import WindowStats, compute_stats
from tsdive.report import render_window_report
from tsdive.store.identity import TagIdentity, TagMeta
from tsdive.store.quality import (
    Severity,
    annotate_severity,
    declared_value_state,
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
    read_archive_table,
    safe_filename,
    write_tag,
)
from tsdive.switchback.archive import SwitchbackAnalysis, analyze_archives, with_power
from tsdive.switchback.plan import SwitchbackPlan, make_plan

_WINDOW_FORMS = (
    "window must be <START>/<END>, <START>/<DURATION>, <DURATION>/<END> "
    "or <DATE> in ISO 8601 UTC"
)
_BARE_DATE = re.compile(r"\d{4}-\d{2}-\d{2}\Z")
_DURATION = re.compile(
    r"P(?:(?P<days>\d+(?:\.\d+)?)D)?"
    r"(?:T(?:(?P<hours>\d+(?:\.\d+)?)H)?"
    r"(?:(?P<minutes>\d+(?:\.\d+)?)M)?"
    r"(?:(?P<seconds>\d+(?:\.\d+)?)S)?)?\Z"
)


def _window_bound(name: str, text: str) -> pd.Timestamp:
    """Parse one window bound; a bare date is 00:00:00 UTC of that day."""
    if _BARE_DATE.match(text):
        return cast(pd.Timestamp, pd.Timestamp(text, tz="UTC"))
    try:
        ts = pd.Timestamp(text)
    except (ValueError, TypeError) as exc:
        raise ValueError(_WINDOW_FORMS) from exc
    if ts.tz is None:
        raise ValueError(
            f"window {name} is naive ({ts}); append 'Z' or '+00:00' - "
            "tsdive stores UTC only"
        )
    return cast(pd.Timestamp, ts)


def _window_duration(text: str) -> pd.Timedelta:
    """Parse an ISO 8601 duration in days, hours, minutes and seconds.

    A month or year unit raises ``ValueError`` naming the accepted units.
    P1M is the reason the units are read here instead of by
    ``pd.Timedelta``: pandas reads P1M as one minute, so a caller who
    means a month would get a 60-second window and no warning.
    """
    match = _DURATION.match(text)
    parts = {k: float(v) for k, v in match.groupdict().items() if v} if match else {}
    if not parts:
        raise ValueError(
            f"window duration {text} is not days, hours, minutes and seconds; "
            "write it like PT5H or P2DT6H"
        )
    return pd.Timedelta(dt.timedelta(**parts))


def parse_window(spec: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Parse an ISO 8601 interval into two UTC-aware timestamps.

    ``START/END``, ``START/DURATION``, ``DURATION/END`` and a bare date
    standing for one UTC day all name a window. Naive bounds are
    rejected: a window whose offset is unstated is not a window, and
    tsdive stores UTC only.
    """
    if "/" not in spec:
        if _BARE_DATE.match(spec):
            day = _window_bound("START", spec)
            return day, cast(pd.Timestamp, day + dt.timedelta(days=1))
        raise ValueError(_WINDOW_FORMS)
    left, right = spec.split("/", maxsplit=1)
    left_is_duration = left.startswith("P")
    right_is_duration = right.startswith("P")
    if left_is_duration and right_is_duration:
        raise ValueError(_WINDOW_FORMS)
    if right_is_duration:
        start = _window_bound("START", left)
        end = start + _window_duration(right)
    elif left_is_duration:
        end = _window_bound("END", right)
        start = end - _window_duration(left)
    else:
        start = _window_bound("START", left)
        end = _window_bound("END", right)
    if end <= start:
        raise ValueError("window END is not after START")
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
    table = cast(pd.DataFrame, read_archive_table(path).to_pandas())
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

    # A value-column state the tag declares in quality_codes (PI's "I/O
    # Timeout") is no reading, exactly as on the assessed window.
    stated = pd.Series(False, index=table.index)
    if codes and not pd.api.types.is_numeric_dtype(table["value"]):
        stated = table["value"].map(lambda v: declared_value_state(v, codes) is not None)
    usable = good_mask(table, codes) & table["value"].notna() & ~stated.astype(bool)
    good = table[usable].reset_index(drop=True)
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
    """Load a [`TagMeta`][tsdive.TagMeta] from a JSON file.

    The file uses the same object as the archive's own ``tsdive.meta``
    block, so metadata written for ingest is readable back off the
    archive without a second format to keep in step.

    Every key is checked, at the top level and under ``identity``. A key
    tsdive does not read raises ``SchemaError`` naming the closest known
    key, so a misspelt ``"unit"`` cannot leave the archive without its
    unit.

    Raises:
        SchemaError: invalid JSON, an unknown or missing key, a value of
            the wrong type, one end of the engineering range without the
            other, or a ``quality_codes`` entry that names no severity.

    Examples:
        >>> import json
        >>> import tsdive
        >>> meta = {"identity": {"source_id": "plant1", "point_id": "FIC101.PV"},
        ...         "name": "FIC-101 flow", "unit_raw": "m3/h", "sample_rate_s": 3.0}
        >>> with open("FIC101.PV.json", "w", encoding="utf-8") as f:
        ...     json.dump(meta, f)
        >>> read = tsdive.read_meta_json("FIC101.PV.json")
        >>> str(read.identity), read.unit_raw, read.sample_rate_s
        ('plant1:FIC101.PV', 'm3/h', 3.0)

        A misspelt key raises instead of being dropped:

        >>> meta["unit"] = meta.pop("unit_raw")
        >>> with open("FIC101.PV.json", "w", encoding="utf-8") as f:
        ...     json.dump(meta, f)
        >>> tsdive.read_meta_json("FIC101.PV.json")
        Traceback (most recent call last):
        ...
        tsdive.errors.SchemaError: FIC101.PV.json: unknown key 'unit'; did you mean 'unit_raw'?
    """
    text = Path(path).read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as e:
        raise SchemaError(f"{Path(path).name}: not valid JSON ({e})") from e
    return meta_from_dict(payload, label=Path(path).name)


def _read_source(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return cast(pd.DataFrame, pd.read_csv(path))
    if path.suffix.lower() in {".parquet", ".pq"}:
        try:
            return cast(pd.DataFrame, pd.read_parquet(path))
        except pa.ArrowInvalid as e:
            raise SchemaError(f"{path.name}: not a readable parquet file ({e})") from e
    raise SchemaError(
        f"{path.name}: unsupported input {path.suffix!r}; ingest reads .csv and .parquet"
    )


# A numeric date whose first two fields are day and month in some order:
# 01/02/2026, 1.2.26, 13-02-2026. Year-first forms such as ISO 8601 do
# not match, so they never reach the day/month decision.
_DAY_MONTH = re.compile(r"\s*(\d{1,2})([/.\-])(\d{1,2})\2(\d{4}|\d{2})(?!\d)(.*)\Z", re.DOTALL)


def _date_order(raw: pd.Series, column: str, *, dayfirst: bool) -> bool:
    """True when the column's numeric dates read day first, False when month first.

    A row whose first field exceeds 12 reads only day first, one whose
    second field exceeds 12 only month first, and one with both fields at
    12 or less reads both ways. Without ``dayfirst`` a row that reads both
    ways raises ``SchemaError``, and so does a column holding rows of both
    one-way kinds: one parse must never read two rows in two orders.
    """
    ambiguous: str | None = None
    day_only: str | None = None
    month_only: str | None = None
    for v in raw:
        m = _DAY_MONTH.match(v) if isinstance(v, str) else None
        if m is None:
            continue
        first, second = int(m.group(1)), int(m.group(3))
        if dayfirst and second > 12:
            raise SchemaError(
                f"{column}: {v!r} has no month {second} when read day first; "
                "drop --dayfirst, or pass --timestamp-format with the order the "
                "export uses"
            )
        if first <= 12 and second <= 12:
            ambiguous = ambiguous or v
        elif first > 12 and second <= 12:
            day_only = day_only or v
        elif second > 12 and first <= 12:
            month_only = month_only or v
    if dayfirst:
        return True
    if ambiguous is not None:
        m = cast(re.Match[str], _DAY_MONTH.match(ambiguous))
        first, second, year = m.group(1), m.group(3), m.group(4)
        raise SchemaError(
            f"{column}: {ambiguous!r} reads as day {first} of month {second} or as "
            f"month {first}, day {second} of {year}; pass --dayfirst to read day "
            "first, or --timestamp-format with a strptime format such as "
            "'%d/%m/%Y %H:%M:%S'"
        )
    if day_only is not None and month_only is not None:
        raise SchemaError(
            f"{column}: {day_only!r} reads only day first and {month_only!r} only "
            "month first; export the column in one date order, or pass "
            "--timestamp-format to state it"
        )
    return day_only is not None


def _month_first(value: object, *, dayfirst: bool) -> object:
    """``value`` with a numeric day/month date rewritten month first, slash-separated.

    pandas reads such a date month first, and a ``dayfirst`` hint also
    swaps ISO 8601 dates (2024-03-01 becomes 3 January), so the order is
    fixed here and pandas never sees the hint.
    """
    m = _DAY_MONTH.match(value) if isinstance(value, str) else None
    if m is None:
        return value
    first, _, second, year, rest = m.groups()
    day, month = (first, second) if dayfirst else (second, first)
    return f"{month}/{day}/{year}{rest}"


def _parse_timestamps(
    raw: pd.Series,
    column: str,
    *,
    timestamp_format: str | None = None,
    dayfirst: bool = False,
) -> list[pd.Timestamp]:
    """Parse each element to a Timestamp, refusing the first unparseable one.

    Element-wise on purpose. ``pd.to_datetime`` over a whole column whose
    rows carry different UTC offsets returns object dtype (and warns that
    a future pandas will raise), which makes every ``.dt`` accessor
    downstream fail with an AttributeError instead of a typed refusal.

    With ``timestamp_format`` every element must match that strptime
    format. Without it, numeric day/month dates are read in the one order
    :func:`_date_order` finds, or raise ``SchemaError``.
    """
    if timestamp_format is not None and dayfirst:
        raise ValueError("pass --timestamp-format or --dayfirst, not both")
    stamps: list[pd.Timestamp] = []
    if timestamp_format is not None:
        for v in raw:
            try:
                ts = cast(pd.Timestamp, pd.to_datetime(v, format=timestamp_format))
            except (ValueError, TypeError):
                ts = pd.NaT
            if ts is pd.NaT or pd.isna(ts):
                raise SchemaError(
                    f"{column}: {v!r} does not match --timestamp-format {timestamp_format!r}"
                )
            stamps.append(ts)
        return stamps
    order = _date_order(raw, column, dayfirst=dayfirst)
    for v in raw:
        ts = cast(
            pd.Timestamp, pd.to_datetime(_month_first(v, dayfirst=order), errors="coerce")
        )
        if ts is pd.NaT or pd.isna(ts):
            raise SchemaError(f"{column}: {v!r} is not a timestamp tsdive can parse")
        stamps.append(ts)
    return stamps


def _to_utc(
    raw: pd.Series,
    tz: str | None,
    column: str,
    *,
    timestamp_format: str | None = None,
    dayfirst: bool = False,
) -> pd.Series:
    """Parse a source timestamp column into UTC, or raise ``SchemaError``.

    Naive timestamps carry no offset, so tsdive cannot know what
    instant they name. With ``tz`` the caller states the source's zone and
    the column is localised then converted; without it, a naive column
    raises ``SchemaError`` naming the column and the ``--tz`` flag.

    Rows with different UTC offsets all name real instants and are
    converted individually. A column that mixes naive and offset-bearing
    rows raises ``SchemaError``: ``tz`` would have to be applied to some
    rows and not others, and the export is telling two stories about its
    own clock.
    """
    stamps = _parse_timestamps(
        raw, column, timestamp_format=timestamp_format, dayfirst=dayfirst
    )
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
        codes=stamped.quality_codes,
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
    timestamp_format: str | None = None,
    dayfirst: bool = False,
) -> Path:
    """Turn a CSV or parquet export into a tsdive archive.

    Only the three declared columns are carried over; everything else in
    the export is left behind, so an ingested archive contains exactly
    what its schema promises.

    A missing quality column is a refusal, not a default: a value whose
    trustworthiness is unknown is not a measurement. ``assume_quality``
    is the explicit override, and it is recorded on the archive
    (``quality_assumed``) and printed by every profile of it.

    Timestamps are parsed row by row, so rows with different UTC offsets
    each keep their own. A numeric date such as ``01/02/2026`` reads as 1
    February day first and 2 January month first; a column holding one
    raises ``SchemaError`` unless ``dayfirst`` or ``timestamp_format`` (a
    strptime format every row must match) states the order. A column
    whose dates read only one way, such as ``3/14/2024 1:05 PM``, parses
    without either.

    Raises:
        SchemaError: unreadable input, missing column, naive timestamps
            without ``tz``, a date that reads day first and month first
            with no order stated, a row that does not match
            ``timestamp_format``, or an unusable ``assume_quality`` value.
        ValueError: both ``timestamp_format`` and ``dayfirst``.
        FileExistsError: ``out`` exists and ``overwrite`` is False.

    Examples:
        A CSV export of the demo flow tag, ingested under a new identity:

        >>> import pandas as pd
        >>> import tsdive
        >>> pd.read_parquet("data/demo/fic101_demo.parquet").to_csv("fic101.csv", index=False)
        >>> meta = tsdive.TagMeta(identity=tsdive.TagIdentity("plant1", "FIC101.PV"),
        ...                       name="FIC-101 flow", unit_raw="m3/h")
        >>> out = tsdive.ingest("fic101.csv", out="archive/plant1/FIC101.PV.parquet", meta=meta)
        >>> out.as_posix(), len(pd.read_parquet(out))
        ('archive/plant1/FIC101.PV.parquet', 562)

        An export whose dates read both ways states its order:

        >>> pd.DataFrame({"timestamp": ["01/02/2026 08:00:00", "13/02/2026 08:00:00"],
        ...               "value": [61.0, 62.0], "quality": ["GOOD", "GOOD"]}
        ...              ).to_csv("eu.csv", index=False)
        >>> tsdive.ingest("eu.csv", out="eu.parquet", meta=meta, tz="Europe/Paris")
        Traceback (most recent call last):
        ...
        tsdive.errors.SchemaError: timestamp: '01/02/2026 08:00:00' reads as day 01 of month 02 ...
        >>> out = tsdive.ingest("eu.csv", out="eu.parquet", meta=meta, tz="Europe/Paris",
        ...                     dayfirst=True)
        >>> list(pd.read_parquet(out)["timestamp"].dt.strftime("%Y-%m-%d %H:%M"))
        ['2026-02-01 07:00', '2026-02-13 07:00']
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
        timestamps=_to_utc(
            frame[timestamp_col],
            tz,
            timestamp_col,
            timestamp_format=timestamp_format,
            dayfirst=dayfirst,
        ),
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
    timestamp_format: str | None = None,
    dayfirst: bool = False,
) -> list[Path]:
    """Turn a wide export, one column per tag, into one archive per tag.

    The file is read once. Tag columns are ``tags`` when given, else every
    column that is not ``timestamp_col`` and not a quality column; with
    ``quality_suffix`` a tag's quality column is ``f"{tag}{quality_suffix}"``.
    Metadata for a tag comes from ``meta_dir / f"{safe_filename(tag)}.json"``
    and its archive goes to ``out_dir / f"{safe_filename(point_id)}.parquet"``.
    Every check runs before the first archive is written, so a refusal
    leaves ``out_dir`` as it was. Timestamps are parsed as
    [`ingest`][tsdive.ingest] parses them, ``timestamp_format`` and
    ``dayfirst`` included.

    Raises:
        SchemaError: unreadable input; a missing timestamp, tag, quality
            or metadata file; naive timestamps without ``tz``; a date that
            reads day first and month first with no order stated; or
            ``quality_suffix`` given together with ``assume_quality``.
        ValueError: two tags or two point ids that share one file name, or
            both ``timestamp_format`` and ``dayfirst``.
        FileExistsError: an archive exists and ``overwrite`` is False.

    Examples:
        A wide export of the two demo tags, without a quality column:

        >>> import pandas as pd
        >>> import tsdive
        >>> fic = pd.read_parquet("data/demo/fic101_demo.parquet")
        >>> tic = pd.read_parquet("data/demo/tic101_demo.parquet")
        >>> wide = pd.DataFrame({"ts": fic["timestamp"], "FIC101.PV": fic["value"],
        ...                      "TIC101.PV": tic["value"]})
        >>> wide.to_csv("export.csv", index=False)
        >>> _ = tsdive.init_meta("export.csv", out_dir="meta", source_id="plant1",
        ...                      timestamp_col="ts")
        >>> archives = tsdive.ingest_wide("export.csv", out_dir="archive/plant1",
        ...                               meta_dir="meta", timestamp_col="ts",
        ...                               assume_quality="GOOD")
        >>> [path.as_posix() for path in archives]
        ['archive/plant1/FIC101.PV.parquet', 'archive/plant1/TIC101.PV.parquet']
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
    timestamps = _to_utc(
        frame[timestamp_col],
        tz,
        timestamp_col,
        timestamp_format=timestamp_format,
        dayfirst=dayfirst,
    )
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
    [`read_meta_json`][tsdive.read_meta_json] refuses the file until they are.

    Raises:
        SchemaError: unreadable input, or a missing timestamp, tag or
            quality column.
        ValueError: two tags that share one file name.
        FileExistsError: a template exists and ``overwrite`` is False.

    Examples:
        >>> import pandas as pd
        >>> import tsdive
        >>> fic = pd.read_parquet("data/demo/fic101_demo.parquet")
        >>> tic = pd.read_parquet("data/demo/tic101_demo.parquet")
        >>> wide = pd.DataFrame({"ts": fic["timestamp"], "FIC101.PV": fic["value"],
        ...                      "TIC101.PV": tic["value"]})
        >>> wide.to_csv("export.csv", index=False)
        >>> paths = tsdive.init_meta("export.csv", out_dir="meta", source_id="plant1",
        ...                          timestamp_col="ts")
        >>> [path.as_posix() for path in paths]
        ['meta/FIC101.PV.json', 'meta/TIC101.PV.json']
        >>> template = tsdive.read_meta_json(paths[0])
        >>> str(template.identity), template.name, template.unit_raw
        ('plant1:FIC101.PV', 'FIC101.PV', None)
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

    Holds objects, not text: [`render`][tsdive.Profile.render] is the only place the report
    lines are produced, and it is the same renderer the CLI prints.
    """

    window: Window
    stats: WindowStats
    flatline: FlatlineVerdict | None = None

    def render(self) -> str:
        """The report ``tsdive profile`` prints for this window, without colour.

        Examples:
            >>> import tsdive
            >>> p = tsdive.profile("data/demo/fic101_demo.parquet",
            ...                    "2024-03-30T20:00:00Z/2024-03-31T06:00:00Z")
            >>> for line in p.render().splitlines()[:2]:
            ...     print(line)
            demo:FIC101.PV  FIC-101 flow
            coverage 0.933   GOOD 561/562   censored yes   gaps 1
        """
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

    Examples:
        >>> import tsdive
        >>> p = tsdive.profile("data/demo/fic101_demo.parquet",
        ...                    "2024-03-30T20:00:00Z/2024-03-31T06:00:00Z")
        >>> str(p.identity), round(p.physics.coverage.coverage, 3)
        ('demo:FIC101.PV', 0.933)
        >>> p.physics.clipping.censored
        True
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


# --- switchback --------------------------------------------------------------


def _instant(value: str | pd.Timestamp | dt.datetime, name: str) -> pd.Timestamp:
    """One UTC instant from ISO 8601 text or an aware timestamp."""
    if isinstance(value, str):
        return _window_bound(name, value)
    stamp = cast(pd.Timestamp, pd.Timestamp(value))
    if stamp.tz is None:
        raise ValueError(f"{name} {stamp} is naive; state it in UTC")
    return stamp


def _whole_seconds(value: str | int | float | dt.timedelta | pd.Timedelta, name: str) -> int:
    """A duration in whole seconds from an ISO 8601 duration, a number of seconds or a timedelta."""
    if isinstance(value, str):
        try:
            delta = _window_duration(value)
        except ValueError:
            raise ValueError(
                f"{name} {value} is not days, hours, minutes and seconds; write it like PT1H"
            ) from None
    elif isinstance(value, dt.timedelta | pd.Timedelta):
        delta = pd.Timedelta(value)
    elif isinstance(value, int | float) and not isinstance(value, bool):
        delta = pd.Timedelta(float(value), unit="s")
    else:
        raise ValueError(f"{name} must be an ISO 8601 duration like PT1H or a number of seconds")
    seconds = delta.total_seconds()
    if seconds != int(seconds):
        raise ValueError(f"{name} of {seconds} s is not a whole number of seconds")
    return int(seconds)


def switchback_plan(
    start: str | pd.Timestamp | dt.datetime,
    end: str | pd.Timestamp | dt.datetime,
    block: str | int | float | dt.timedelta,
    washout: str | int | float | dt.timedelta,
    seed: int,
    *,
    history: str | Path | None = None,
    history_window: str | tuple[pd.Timestamp, pd.Timestamp] | None = None,
) -> SwitchbackPlan:
    """Plan a balanced random schedule of settings A and B over one window.

    The window from ``start`` is cut into K blocks of ``block`` (the last
    block is dropped when K is odd), and ``seed`` assigns exactly K/2 of
    them to B. The first ``washout`` of every block is left out of the
    analysis. The same arguments give the same plan and digest.

    Args:
        start: ISO 8601 UTC instant or an aware timestamp.
        end: ISO 8601 UTC instant or an aware timestamp.
        block: block length, an ISO 8601 duration such as ``PT1H`` or
            seconds; whole seconds only.
        washout: time dropped at the start of every block, as ``block``;
            0 or more and shorter than a block.
        seed: seed of the assignment, 0 or more.
        history: one archive of the target where nothing was switched;
            with ``history_window`` it adds a power readout.
        history_window: ``START/END`` in ISO 8601 UTC, or a pair of aware
            timestamps, at least as long as the schedule.

    Raises:
        ValueError: a malformed or naive bound, a block or washout out of
            range, or only one of ``history`` and ``history_window``.
        DesignTooSmall: the window holds too few blocks for a
            randomization test at the 5% level.

    Examples:
        >>> import tsdive
        >>> plan = tsdive.switchback_plan("2024-06-03T00:00:00Z", "2024-06-04T00:00:00Z",
        ...                               block="PT1H", washout="PT15M", seed=7)
        >>> plan.k, plan.digest[:12]
        (24, '66da65ede04f')
        >>> "".join(b.setting for b in plan.blocks)
        'AAABBABAABBBBBBAAAAABBAB'
        >>> plan.write_json("plan.json").name
        'plan.json'
    """
    if (history is None) != (history_window is None):
        raise ValueError("a power readout needs both history and history_window")
    plan = make_plan(
        _instant(start, "START"),
        _instant(end, "END"),
        _whole_seconds(block, "block"),
        _whole_seconds(washout, "washout"),
        int(seed),
    )
    if history is None or history_window is None:
        return plan
    if isinstance(history_window, str):
        h_span = parse_window(history_window)
    else:
        h_start = _instant(history_window[0], "history START")
        h_end = _instant(history_window[1], "history END")
        if h_end <= h_start:
            raise ValueError("history window END is not after START")
        h_span = (h_start, h_end)
    return with_power(plan, history, h_span)


def switchback_analyze(
    archives: Sequence[str | Path],
    plan: SwitchbackPlan | str | Path,
    *,
    target: str,
    covariates: Sequence[str] = (),
) -> SwitchbackAnalysis:
    """The difference between settings A and B on ``target`` under a verified plan.

    The plan is checked first: its digest against its blocks, the block
    times, the balance of the settings, and the settings against the ones
    its seed draws. The archives are read over the schedule under the
    sampling contract ``compare`` uses, and GOOD numeric samples at least
    the washout into their block are kept.

    Args:
        archives: single-tag parquet archives holding the target and every
            covariate; others are listed as unused.
        plan: a plan, or the path of a plan file.
        target: the tag, as ``source:point`` or a point id that one
            archive carries.
        covariates: tags declared before the analysis for the adjusted
            estimate.

    Raises:
        ScheduleMismatch: the plan was edited or is not balanced.
        DesignTooSmall: the plan holds too few blocks.
        ValueError: a tag that matches no archive or is named twice.
        IncomparableSamplingError: two reads under different sampling
            contracts.
        SchemaError: a covariate repeats a timestamp among its valid
            samples.
        TSDiveError: any typed refusal from the read path.

    Examples:
        >>> import tsdive
        >>> plan = tsdive.switchback_plan("2024-06-03T00:00:00Z", "2024-06-04T00:00:00Z",
        ...                               block="PT1H", washout="PT15M", seed=7)
        >>> archives = ["data/switchback_demo/ti201.parquet",
        ...             "data/switchback_demo/fi200.parquet",
        ...             "data/switchback_demo/tt001.parquet"]
        >>> result = tsdive.switchback_analyze(archives, plan, target="TI201.PV",
        ...                                    covariates=["FI200.PV", "TT001.PV"])
        >>> round(result.direct.estimate, 4), round(result.direct.p_value, 3)
        (0.6024, 0.154)
        >>> round(result.adjusted.estimate, 4), round(result.adjusted.p_value, 3)
        (0.2847, 0.001)
    """
    resolved = plan if isinstance(plan, SwitchbackPlan) else SwitchbackPlan.read_json(plan)
    return analyze_archives(archives, resolved, target=target, covariates=tuple(covariates))

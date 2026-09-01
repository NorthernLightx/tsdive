"""Quality codes: mandatory column, verbatim preservation, derived severity.

Invariants enforced here:

- A value column without a quality column is a schema error.
- Raw vendor codes are preserved verbatim; derived severity lives beside
  them, never replaces them.
- Digital states are never coerced to float. A row whose raw code is a
  known digital state has its value forced to null downstream, so a state
  number like 257 can never masquerade as 257.0 engineering units.
- Unknown quality codes map to UNCERTAIN, never GOOD.
  A numeric code is GOOD only when it matches a registered Good status
  code, not merely because its OPC UA severity bits read ``00``.
- Numeric codes are read as OPC UA 32-bit StatusCodes. Classic OPC DA
  byte qualities (0-255: 192 Good, 64 Uncertain, 0 Bad, plus sub-status
  variants across 0xC0-0xFF, 0x40-0xBF and 0x00-0x3F) are a different
  code space that disagrees with UA at both ends, and an export in it
  must declare ``quality_codes`` on the tag -- ``{"192": "GOOD",
  "64": "UNCERTAIN", "0": "BAD"}`` -- rather than be sniffed for.
- A non-numeric value on a measurement tag is a schema error, not a null.
  Only tags declared ``role=MODE`` may carry string states.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum

import numpy as np
import pandas as pd

from tsdive.errors import SchemaError
from tsdive.store.identity import Role


class Severity(StrEnum):
    GOOD = "GOOD"
    UNCERTAIN = "UNCERTAIN"
    BAD = "BAD"

    @property
    def rank(self) -> int:
        """Rank order: BAD < UNCERTAIN < GOOD."""
        return SEVERITY_RANK[self]


# Severity is a StrEnum, so bare comparison operators are lexicographic
# ("UNCERTAIN" >= "GOOD"). Rank order must come from this table instead.
SEVERITY_RANK: dict[Severity, int] = {
    Severity.BAD: 0,
    Severity.UNCERTAIN: 1,
    Severity.GOOD: 2,
}


def meets_severity(severity: Severity, minimum: Severity) -> bool:
    """True when ``severity`` is at least as trustworthy as ``minimum``."""
    return SEVERITY_RANK[severity] >= SEVERITY_RANK[minimum]


# Explicit string-code table. Extend per site; unknown strings fall through
# to UNCERTAIN.
_STRING_CODES: dict[str, Severity] = {
    "GOOD": Severity.GOOD,
    "UNCERTAIN": Severity.UNCERTAIN,
    "SUBSTITUTED": Severity.UNCERTAIN,
    "SCAN OFF": Severity.UNCERTAIN,
    "BAD": Severity.BAD,
    "OVER RANGE": Severity.BAD,
    "UNDER RANGE": Severity.BAD,
    "COMM FAILURE": Severity.BAD,
}

# Archive digital-state codes (numeric). The value stored alongside these is
# a state number, not a measurement. Codes below follow the plan's example
# (257 "Over Range"); sites should register their own sets explicitly.
DIGITAL_STATE_LABELS: dict[int, str] = {
    248: "Bad",
    249: "Comms Outage",
    250: "Scan Off",
    251: "Substituted",
    257: "Over Range",
}

_DIGITAL_STATE_SEVERITY: dict[int, Severity] = {
    248: Severity.BAD,
    249: Severity.BAD,
    250: Severity.UNCERTAIN,
    251: Severity.UNCERTAIN,
    257: Severity.BAD,
}


# Registered OPC UA Good status codes (OPC UA Part 4, StatusCode table),
# keyed by the 32-bit code with its info bits cleared. Severity lives in
# bits 31-30 and the subcode in bits 27-16, so a Good status is
# ``0x00SS0000``. This table is the whitelist: a top-bits-``00`` integer
# that is not in it is not read as Good. Register your site's additions
# here rather than widening the rule.
OPC_UA_GOOD_CODES: dict[int, str] = {
    0x0000_0000: "Good",
    0x002D_0000: "GoodSubscriptionTransferred",
    0x002E_0000: "GoodCompletesAsynchronously",
    0x002F_0000: "GoodOverload",
    0x0030_0000: "GoodClamped",
    0x0096_0000: "GoodLocalOverride",
    0x00A2_0000: "GoodEntryInserted",
    0x00A3_0000: "GoodEntryReplaced",
    0x00A5_0000: "GoodNoData",
    0x00A6_0000: "GoodMoreData",
    0x00BA_0000: "GoodResultsMayBeIncomplete",
    0x00D9_0000: "GoodDataIgnored",
    0x00DC_0000: "GoodEdited",
    0x00E0_0000: "GoodDependentValueChanged",
}

_INFO_MASK = 0xFFFF
_INFO_TYPE_MASK = 0b11 << 10


def _opc_status_base(code: int) -> int | None:
    """Status code with its info bits stripped, or None if it is not one.

    Bits 15-0 qualify a status (InfoType in bits 11-10 plus that type's
    info bits); they never change which status it is. When InfoType is
    ``00`` the spec requires the remaining info bits to be zero, so a
    lone ``1`` or ``192`` is not a *UA* status code, so it must not
    inherit Good from its top two bits. ``192`` is Good
    in the OPC DA byte space, but that space is declared through
    ``quality_codes``, never inferred here.
    """
    if code < 0:
        return None
    info = code & _INFO_MASK
    if info and not info & _INFO_TYPE_MASK:
        return None
    return code & ~_INFO_MASK


def _numeric_code(raw: object) -> int | None:
    """Integer view of a raw code when it is numeric (int, integral float, digit-string)."""
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        return int(raw) if raw.is_integer() else None
    if isinstance(raw, str):
        s = raw.strip()
        try:
            f = float(s)
        except ValueError:
            return None
        return int(f) if f.is_integer() else None
    return None


QualityCodes = Mapping[str, str] | None


def _declared_severity(raw: object, codes: QualityCodes) -> Severity | None:
    """Severity from a tag's own ``quality_codes`` map, if it names this code.

    Matched exactly first, then trimmed and upper-cased, so a map written
    as ``{"SUB": "UNCERTAIN"}`` also catches ``" sub "``.
    """
    if not codes:
        return None
    key = str(raw)
    declared = codes.get(key)
    if declared is None:
        folded = key.strip().upper()
        declared = next(
            (v for k, v in codes.items() if k.strip().upper() == folded), None
        )
    if declared is None:
        return None
    try:
        return Severity(declared.strip().upper())
    except ValueError as e:
        raise SchemaError(
            f"quality_codes maps {key!r} to {declared!r}; "
            f"expected one of {', '.join(s.value for s in Severity)}"
        ) from e


def derive_severity(raw: object, codes: QualityCodes = None) -> Severity:
    """Map a raw vendor quality code to {GOOD, UNCERTAIN, BAD}.

    Precedence: the tag's own ``quality_codes`` map, then the numeric
    digital-state table, the string table, OPC UA status codes, and
    finally UNCERTAIN for anything unmapped. Total over ``raw``; the only
    refusal is a ``quality_codes`` entry naming a severity that does not
    exist.

    Numeric codes are read as OPC UA StatusCodes. An OPC DA byte quality
    (192 Good, 64 Uncertain, 0 Bad) means something else entirely and
    must be declared in ``codes``; nothing here sniffs for it.

    The status-code rule is asymmetric on purpose. Bits 31-30 reading
    ``01`` or ``1x`` are enough to mark a sample UNCERTAIN or BAD, but
    they are not enough to mark one GOOD: that requires a match in
    :data:`OPC_UA_GOOD_CODES`. Every other top-bits-``00`` integer is
    UNCERTAIN and reported as unmapped, because "no bits set" is what an
    uninitialised field, a row number and a real Good status all look
    like.
    """
    declared = _declared_severity(raw, codes)
    if declared is not None:
        return declared
    code = _numeric_code(raw)
    if code is not None:
        if code in _DIGITAL_STATE_SEVERITY:
            return _DIGITAL_STATE_SEVERITY[code]
        top_bits = code >> 30
        if top_bits == 0:
            base = _opc_status_base(code)
            if base is not None and base in OPC_UA_GOOD_CODES:
                return Severity.GOOD
            return Severity.UNCERTAIN
        if top_bits == 1:
            return Severity.UNCERTAIN
        return Severity.BAD
    key = str(raw).strip().upper()
    return _STRING_CODES.get(key, Severity.UNCERTAIN)


def digital_state_label(raw: object) -> str | None:
    """Human label when the raw code is a known digital state, else None."""
    code = _numeric_code(raw)
    return DIGITAL_STATE_LABELS.get(code) if code is not None else None


def is_byte_sized_code(raw: object) -> bool:
    """True when ``raw`` is an integer in 0..255, the OPC DA quality range.

    A whole column of these is the fingerprint of a classic OPC DA export
    read through the OPC UA rule. It is grounds for a hint in the report,
    never for reinterpreting the codes.
    """
    code = _numeric_code(raw)
    return code is not None and 0 <= code <= 255


REQUIRED_COLUMNS = ("timestamp", "value", "quality")


def validate_schema(df: pd.DataFrame) -> None:
    """Enforce mandatory columns and UTC-aware timestamps.

    Value without quality is a schema error. Naive timestamps at ingest
    are a schema error (UTC-only invariant).
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"missing required column(s): {', '.join(missing)}")
    ts = df["timestamp"]
    if len(ts) > 0:
        if not pd.api.types.is_datetime64_any_dtype(ts):
            raise SchemaError("timestamp column must be datetime64")
        if ts.dt.tz is None:
            raise SchemaError(
                "naive timestamps rejected at ingest: tsdive stores UTC only"
            )


def per_distinct_code(quality: pd.Series, fn) -> np.ndarray:
    """Apply ``fn`` once per distinct raw code and scatter the answers back.

    Every rule in this module is a pure function of the raw code, so a
    column of a million rows carrying two codes costs two calls. Missing
    codes keep their own bucket (``use_na_sentinel=False``) rather than
    being dropped, because a null quality is itself a code to classify.
    """
    positions, uniques = pd.factorize(quality, use_na_sentinel=False)
    answers = np.empty(len(uniques), dtype=object)
    for i, raw in enumerate(uniques):
        answers[i] = fn(raw)
    out = np.empty(len(positions), dtype=object)
    if len(positions):
        out[:] = answers[positions]
    return out


def annotate_severity(df: pd.DataFrame, codes: QualityCodes = None) -> pd.DataFrame:
    """Return a copy with ``severity`` derived from verbatim ``quality``."""
    validate_schema(df)
    out = df.copy()
    out["severity"] = per_distinct_code(out["quality"], lambda q: derive_severity(q, codes))
    return out


def _to_float_or_none(v: object) -> float | None:
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def null_digital_state_values(
    df: pd.DataFrame,
    *,
    role: Role | None = None,
    tag: object = None,
) -> pd.DataFrame:
    """Force value to null wherever quality is a known digital state.

    This is the guard against coercion: an archive that stores the state
    number in the value field must not leak 257.0 into statistics.

    What survives that guard must be numeric. A leftover string on a
    measurement tag is a schema error naming the tag and the first
    offending value, because nulling it would delete data the caller never
    said was unusable. ``role=MODE`` is the one legal string-valued path: a
    mode tag's value *is* a state label ("R1"), so it is left verbatim
    and no numeric physics is computed from it.
    """
    out = df.copy()
    is_state = per_distinct_code(
        out["quality"], lambda q: digital_state_label(q) is not None
    ).astype(bool)
    out.loc[pd.Series(is_state, index=out.index), "value"] = pd.NA
    if role is Role.MODE:
        return out
    values = out["value"]
    coerced = pd.to_numeric(values, errors="coerce").to_numpy(
        dtype="float64", na_value=np.nan
    )
    # ``to_numeric`` and ``float()`` disagree on a handful of spellings, so
    # anything it could not read is re-checked one row at a time. Normally
    # that is zero rows: this is the refusal path, not the fast path.
    unread = np.flatnonzero(np.isnan(coerced) & values.notna().to_numpy())
    for pos in unread:
        v = values.iloc[int(pos)]
        parsed = _to_float_or_none(v)
        if parsed is None:
            where = f"tag {tag}: " if tag is not None else ""
            raise SchemaError(
                f"{where}value {v!r} is not numeric and no digital state explains it; "
                "declare the tag role=MODE if it carries string states, otherwise fix "
                "the export - tsdive will not null a value it was not told to drop"
            )
        coerced[int(pos)] = parsed
    out["value"] = pd.array(coerced, dtype="Float64")
    return out


def is_unmapped(raw: object, codes: QualityCodes = None) -> bool:
    """True when ``raw`` reached the UNCERTAIN fallback without matching any rule."""
    if derive_severity(raw, codes) != Severity.UNCERTAIN:
        return False
    if _declared_severity(raw, codes) is not None:
        return False
    if str(raw).strip().upper() in _STRING_CODES:
        return False
    if digital_state_label(raw) is not None:
        return False
    code = _numeric_code(raw)
    if code is None:
        return True
    return (code >> 30) != 1  # matched the OPC UA uncertain-bit rule when == 1


def severity_series(frame: pd.DataFrame, codes: QualityCodes = None) -> pd.Series:
    """Derived severity for every row, index-aligned with ``frame``.

    Always derived from the verbatim ``quality`` column, never read off an
    existing ``severity`` column, so one rule decides severity everywhere.
    """
    if "quality" not in frame.columns:
        raise SchemaError("missing required column(s): quality")
    return pd.Series(
        per_distinct_code(frame["quality"], lambda q: derive_severity(q, codes)),
        index=frame.index,
        dtype=object,
    )


def good_mask(frame: pd.DataFrame, codes: QualityCodes = None) -> pd.Series:
    """Boolean mask of rows whose derived severity is GOOD.

    The one way to ask "is this row trustworthy". Comparing the raw code
    against the string ``"GOOD"`` misses every archive that codes quality
    numerically, and silently treats its GOOD rows as unusable.
    """
    return severity_series(frame, codes) == Severity.GOOD


def usable_mask(frame: pd.DataFrame, codes: QualityCodes = None) -> pd.Series:
    """Boolean mask of rows a statistic may be computed over.

    A frame that came out of a window read already carries ``valid``,
    derived there from the tag's own ``quality_codes`` and from whether
    the row holds a usable value. That column is the answer: re-deriving
    validity from the raw codes without the tag's vocabulary would overrule
    the read, and an archive coding quality numerically (192 GOOD) would
    have every row judged by the default one instead.

    A frame that never went through a read has no such column and falls
    back to :func:`good_mask` plus a present value.
    """
    if "valid" in frame.columns:
        return frame["valid"].astype(bool)
    return good_mask(frame, codes) & frame["value"].notna()


def severity_meets(severity: pd.Series, minimum: Severity) -> pd.Series:
    """Row-wise :func:`meets_severity` over a derived severity column."""
    trusted = [s for s in Severity if meets_severity(s, minimum)]
    return severity.isin(trusted)


def severity_counts(df: pd.DataFrame) -> dict[Severity, int]:
    """Counts per derived severity over the given (annotated) frame."""
    counts = {s: 0 for s in Severity}
    for value, n in df["severity"].value_counts().items():
        counts[Severity(value)] += int(n)
    return counts

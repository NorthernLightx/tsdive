"""Immutable tag identity and metadata.

Identity is ``(source_id, point_id)``, never the display name. Display
names get renamed by control engineers; identity does not.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Role(StrEnum):
    """Control-loop role of a tag. Nullable on TagMeta; populated when known."""

    PV = "PV"
    SP = "SP"
    OP = "OP"
    MODE = "MODE"


@dataclass(frozen=True)
class TagIdentity:
    """Immutable archive coordinates of a tag."""

    source_id: str
    point_id: str

    def __str__(self) -> str:
        return f"{self.source_id}:{self.point_id}"


@dataclass(frozen=True)
class EngRange:
    """Engineering range as zero/span (the historian's own representation)."""

    zero: float
    span: float

    @property
    def high(self) -> float:
        return self.zero + self.span


@dataclass(frozen=True)
class TagMeta:
    """Static metadata for one tag.

    A field the source does not state is ``None``, never a placeholder
    like ``1.0`` or ``0``.
    """

    identity: TagIdentity
    name: str
    unit_raw: str | None = None
    unit_canonical: str | None = None
    eng_range: EngRange | None = None
    sample_rate_s: float | None = None
    asset: str | None = None
    loop_id: str | None = None
    role: Role | None = None
    # Raw vendor code -> "GOOD" / "UNCERTAIN" / "BAD", for codes only this
    # source's owner can interpret. Consulted before the global tables, so
    # a site declares what its own codes mean instead of editing library
    # tables or letting them fall through to UNCERTAIN.
    quality_codes: dict[str, str] | None = None
    # True when the quality column was written by tsdive at ingest
    # instead of coming from the source. Every report says so out loud:
    # an assumed quality is a caller's promise, not an observation.
    quality_assumed: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.identity, TagIdentity):
            raise TypeError("TagMeta.identity must be a TagIdentity")
        if self.quality_codes is not None and not all(
            isinstance(k, str) and isinstance(v, str) for k, v in self.quality_codes.items()
        ):
            raise TypeError("TagMeta.quality_codes must map str raw codes to str severities")

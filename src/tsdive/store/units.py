"""Unit resolution over an explicit alias table.

Rules:

- Unknown unit strings resolve to ``unit_canonical=None`` and are surfaced
  in reports. No spelling is resolved by inference.
- Reference-condition units (Nm3/h, Sm3/h, scfm) share a dimension with
  their plain cousins but are NOT comparable without a declared reference
  state. Comparing Nm3/h to m3/h is refused, not converted silently.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import TYPE_CHECKING

from tsdive.errors import IncomparableUnitsError, UnresolvedUnitError

if TYPE_CHECKING:
    import pint


@lru_cache(maxsize=1)
def unit_registry() -> pint.UnitRegistry:
    """The process-wide pint registry, built on first use.

    Building one costs ~150 ms and parses pint's whole default definition
    file, which is wasted on the many callers that only ever resolve unit
    strings against the alias table. Importing pint here keeps that cost
    off ``import tsdive`` too.
    """
    import pint

    return pint.UnitRegistry()


@dataclass(frozen=True)
class UnitResolution:
    raw: str
    canonical: str | None
    pint_units: str | None
    reference_condition: bool

    @property
    def resolved(self) -> bool:
        return self.canonical is not None


@lru_cache(maxsize=1)
def _alias_table() -> tuple[dict[str, UnitResolution], dict[str, UnitResolution]]:
    text = (
        resources.files("tsdive.store")
        .joinpath("data")
        .joinpath("engunits_aliases.csv")
        .read_text(encoding="utf-8")
    )
    exact: dict[str, UnitResolution] = {}
    folded: dict[str, UnitResolution] = {}
    for row in csv.DictReader(io.StringIO(text)):
        res = UnitResolution(
            raw=row["alias_raw"],
            canonical=row["canonical"],
            pint_units=row["pint_units"],
            reference_condition=row["reference_condition"] == "true",
        )
        exact[row["alias_raw"]] = res
        folded[row["alias_raw"].casefold()] = res
    return exact, folded


def resolve_unit(raw: str | None) -> UnitResolution:
    """Resolve a raw unit string against the alias table.

    Unknown input yields ``canonical=None`` (surfaced downstream) and is
    not matched into a known family by inference.
    """
    if raw is None:
        return UnitResolution(raw="", canonical=None, pint_units=None, reference_condition=False)
    exact, folded = _alias_table()
    hit = exact.get(raw) or folded.get(raw.strip().casefold())
    if hit is not None:
        return hit
    return UnitResolution(raw=raw, canonical=None, pint_units=None, reference_condition=False)


def _dimensionality(pint_units: str):
    ureg = unit_registry()
    return ureg.get_dimensionality(ureg.parse_units(pint_units))


def check_comparable(
    a_raw: str | None,
    b_raw: str | None,
    *,
    reference_states_declared: bool = False,
) -> None:
    """Refuse incomparable unit pairs before any conversion happens.

    Raises:
        UnresolvedUnitError: either unit is unknown.
        IncomparableUnitsError: reference-condition unit without declared
            reference state, or differing dimensions.
    """
    ra, rb = resolve_unit(a_raw), resolve_unit(b_raw)
    for name, r in (("a_raw", ra), ("b_raw", rb)):
        if not r.resolved:
            if not r.raw:
                raise UnresolvedUnitError(
                    f"{name} names no unit; declare the tag's unit_raw before "
                    "comparing units"
                )
            raise UnresolvedUnitError(
                f"unit {r.raw!r} is not in the alias table; "
                "register it or declare the reference state"
            )
    assert ra.pint_units is not None and rb.pint_units is not None
    if (ra.reference_condition or rb.reference_condition) and not reference_states_declared:
        flagged = [r.raw for r in (ra, rb) if r.reference_condition]
        raise IncomparableUnitsError(
            f"reference-condition unit(s) {flagged} require a declared reference "
            "state before cross-tag comparison"
        )
    if _dimensionality(ra.pint_units) != _dimensionality(rb.pint_units):
        raise IncomparableUnitsError(
            f"units {ra.raw!r} and {rb.raw!r} have different dimensions "
            f"({_dimensionality(ra.pint_units)} vs {_dimensionality(rb.pint_units)})"
        )


def convert(value: float, from_raw: str, to_raw: str) -> float:
    """Convert between two resolved, comparable units."""
    check_comparable(from_raw, to_raw, reference_states_declared=True)
    rf, rt = resolve_unit(from_raw), resolve_unit(to_raw)
    assert rf.pint_units is not None and rt.pint_units is not None
    ureg = unit_registry()
    quantity = ureg.Quantity(value, ureg.parse_units(rf.pint_units))
    return float(quantity.to(rt.pint_units).magnitude)

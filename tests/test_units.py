"""Fixture (g) and unit-resolution rules."""

from __future__ import annotations

import subprocess
import sys

import pytest

from tsdive.errors import IncomparableUnitsError, UnresolvedUnitError
from tsdive.store.units import (
    _alias_table,
    _dimensionality,
    check_comparable,
    convert,
    resolve_unit,
    unit_registry,
)


def test_nm3_vs_m3_refused():
    """Reference-condition flow is not plain flow; no silent conversion."""
    with pytest.raises(IncomparableUnitsError) as err:
        check_comparable("Nm3/h", "m3/h")
    assert "reference state" in str(err.value)


def test_sm3_and_scfm_flagged():
    assert resolve_unit("Sm3/h").reference_condition
    assert resolve_unit("scfm").reference_condition
    with pytest.raises(IncomparableUnitsError):
        check_comparable("Sm3/h", "Nm3/h")


def test_declared_reference_state_allows_comparison():
    check_comparable("Nm3/h", "m3/h", reference_states_declared=True)
    check_comparable("m3/h", "m3/h")


def test_dimension_mismatch_refused():
    with pytest.raises(IncomparableUnitsError):
        check_comparable("kg/h", "m3/h")


def test_unknown_unit_resolves_null_never_guessed():
    r = resolve_unit("bogus-unit-42")
    assert r.canonical is None
    assert not r.resolved
    with pytest.raises(UnresolvedUnitError):
        check_comparable("bogus-unit-42", "m3/h")


def test_case_insensitive_alias():
    assert resolve_unit("nm3/H").reference_condition
    assert resolve_unit("kpa").pint_units == "kPa"


def test_convert_same_family():
    assert convert(100.0, "degF", "degC") == pytest.approx(37.7778, abs=1e-3)


def test_pint_is_not_imported_by_importing_tsdive():
    """The registry costs ~150 ms to build; only unit comparison pays it."""
    probe = (
        "import sys, tsdive, tsdive.store.units as u;"
        "print('pint' in sys.modules);"
        "u.resolve_unit('m3/h');"
        "print('pint' in sys.modules);"
        "u.check_comparable('m3/h', 'm3/h');"
        "print('pint' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert out.stdout.split() == ["False", "False", "True"]


def test_the_registry_is_built_once():
    assert unit_registry() is unit_registry()


def test_every_alias_parses_through_pint():
    """A row whose pint_units pint cannot read is a silent dead alias."""
    exact, _ = _alias_table()
    for alias, resolution in exact.items():
        assert resolution.pint_units, f"{alias} has no pint_units"
        _dimensionality(resolution.pint_units)  # raises if pint cannot read it


@pytest.mark.parametrize(
    ("raw", "canonical"),
    [
        ("Pa", "pascals"),
        ("oC", "degrees Celsius"),
        ("°C", "degrees Celsius"),
        ("%", "percent"),
        ("m3/s", "cubic meters per second"),
        ("m³/h", "cubic meters per hour"),
        ("m³/s", "cubic meters per second"),
        ("t/h", "metric tons per hour"),
        ("mbar", "millibar"),
        ("MPa", "megapascals"),
        ("l/min", "liters per minute"),
        ("m/s", "meters per second"),
        ("rpm", "revolutions per minute"),
        ("mA", "milliamperes"),
        ("V", "volts"),
    ],
)
def test_si_spellings_resolve(raw, canonical):
    assert resolve_unit(raw).canonical == canonical


def test_the_3w_units_file_now_resolves():
    """`dataset.ini` spells its four units this way; all four were unresolved."""
    for raw in ("Pa", "oC", "%", "m3/s"):
        assert resolve_unit(raw).resolved


def test_celsius_spellings_are_the_same_unit():
    check_comparable("oC", "degC")
    check_comparable("°C", "degC")
    assert convert(100.0, "oC", "degF") == pytest.approx(212.0, abs=1e-6)


@pytest.mark.parametrize("raw", ["C", "barg", "psig"])
def test_ambiguous_and_gauge_spellings_stay_unresolved(raw):
    """`C` is coulombs as readily as Celsius; a gauge unit names no datum.

    The table carries no reference-datum column, so `barg` and `psig`
    would have to resolve to their absolute cousins and be off by one
    atmosphere. Unresolved is the honest answer - see docs/SCHEMA.md.
    """
    assert not resolve_unit(raw).resolved
    with pytest.raises(UnresolvedUnitError):
        check_comparable(raw, "bar")

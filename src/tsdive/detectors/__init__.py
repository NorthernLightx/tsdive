"""Detectors consult the data-physics layer before asserting anything."""

from __future__ import annotations

from tsdive.detectors.flatline import FlatlineVerdict, SignalFinding, assess_flatline

__all__ = ["FlatlineVerdict", "SignalFinding", "assess_flatline"]

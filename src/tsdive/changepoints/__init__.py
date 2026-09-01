"""Changepoint segmentation: regimes for a tag that has no MODE tag."""

from __future__ import annotations

from tsdive.changepoints.pelt import (
    DEFAULT_PENALTY_MULTIPLIER,
    METHOD,
    Segment,
    Segmentation,
    default_penalty,
    pelt_l2,
    segment_window,
)

__all__ = [
    "DEFAULT_PENALTY_MULTIPLIER",
    "METHOD",
    "Segment",
    "Segmentation",
    "default_penalty",
    "pelt_l2",
    "segment_window",
]

"""Multivariate SPC."""

from __future__ import annotations

from tsdive.mspc.pca import (
    AlignedMatrix,
    MspcDetection,
    PcaModel,
    align_windows,
    common_rate,
    detect,
    fit_pca,
)

__all__ = [
    "AlignedMatrix",
    "MspcDetection",
    "PcaModel",
    "align_windows",
    "common_rate",
    "detect",
    "fit_pca",
]

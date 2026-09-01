"""Multivariate SPC via PCA, Hotelling T2 and SPE residuals.

Rules enforced here:

- Alignment refuses unless every tag shares the common UTC grid at the
  declared rate with at least ``min_coverage`` coverage, and contracts
  are mutually comparable. No silent interpolation across tags.
- Control limits are EMPIRICAL percentiles of training statistics. They
  carry no chi-square or F distributional claim.
- SVD signs are fixed deterministically so models replay identically.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

from tsdive.errors import MspcAlignmentError
from tsdive.store.tagstore import Window


@dataclass(frozen=True)
class AlignedMatrix:
    index: pd.DatetimeIndex
    columns: list[str]
    matrix: np.ndarray  # shape (n_times, n_tags)
    coverage: float  # fraction of grid cells filled after alignment


def common_rate(windows: Sequence[Window]) -> int:
    """The one rate every archive declares, or a refusal naming what they say.

    The grid rate decides which samples line up across tags, so picking
    one for a caller who never stated it - or averaging two declarations -
    would invent the alignment every multivariate statistic rests on.
    """
    declared = {str(w.identity): w.meta.sample_rate_s for w in windows}
    silent = sorted(tag for tag, rate in declared.items() if rate is None)
    if silent:
        raise ValueError(
            f"no sample_rate_s declared by {', '.join(silent)}; pass --rate-s to "
            "state the grid"
        )
    distinct = {float(r) for r in declared.values() if r is not None}
    if len(distinct) != 1:
        spread = ", ".join(f"{tag}={rate}" for tag, rate in sorted(declared.items()))
        raise ValueError(
            f"archives declare different sample rates ({spread}); pass --rate-s to "
            "state the grid"
        )
    rate = distinct.pop()
    if rate != int(rate):
        raise ValueError(
            f"declared sample rate {rate} s is not a whole number of seconds; pass "
            "--rate-s to state the grid"
        )
    return int(rate)


def _grid(start: pd.Timestamp, end: pd.Timestamp, rate_s: int) -> pd.DatetimeIndex:
    steps = int((end - start).total_seconds() // rate_s) + 1
    return pd.DatetimeIndex([start + pd.Timedelta(rate_s * i, unit="s") for i in range(steps)])


def align_windows(
    windows: list[Window],
    *,
    rate_s: int,
    min_coverage: float = 0.95,
) -> AlignedMatrix:
    """Assemble a common-grid sample matrix from multiple store windows."""
    if len(windows) < 2:
        raise MspcAlignmentError("need at least two windows to align")
    start = cast(pd.Timestamp, max(w.start for w in windows))
    end = cast(pd.Timestamp, min(w.end for w in windows))
    if end <= start:
        raise MspcAlignmentError("window time spans do not overlap")
    grid = _grid(pd.Timestamp(start), pd.Timestamp(end), rate_s)

    cols: dict[str, np.ndarray] = {}
    filled_cells = 0
    for w in windows:
        name = str(w.identity)
        good = w.frame[w.frame["valid"]]
        series = pd.Series(
            good["value"].astype(float).to_numpy(), index=pd.DatetimeIndex(good["timestamp"])
        )
        reindexed = series.reindex(grid, method=None)
        vals = reindexed.to_numpy(dtype=float)
        filled_cells += int(np.isfinite(vals).sum())
        cols[name] = vals
    total_cells = len(grid) * len(windows)
    coverage = filled_cells / total_cells if total_cells else 0.0
    if coverage < min_coverage:
        raise MspcAlignmentError(
            f"aligned coverage {coverage:.3f} below required {min_coverage}; "
            "refusing to interpolate across tags"
        )
    matrix = np.column_stack([cols[c] for c in sorted(cols)])
    # Fill residual holes column-wise with that tag's training median is NOT
    # allowed silently; instead rows containing any NaN are dropped and the
    # drop is reported in coverage math above. Rows kept are all-finite:
    keep = ~np.isnan(matrix).any(axis=1)
    if not keep.any():
        raise MspcAlignmentError("no fully-observed aligned rows remain")
    return AlignedMatrix(
        index=grid[keep],
        columns=sorted(cols),
        matrix=matrix[keep],
        coverage=coverage,
    )


@dataclass(frozen=True)
class PcaModel:
    columns: list[str]
    mean: np.ndarray
    components: np.ndarray  # (k, n_tags), rows unit-norm, sign-fixed
    score_var: np.ndarray  # training variance of each score axis (for T2)
    t2_limit: float
    spe_limit: float
    explained_variance: tuple[float, ...]


def _fix_signs(components: np.ndarray) -> np.ndarray:
    out = components.copy()
    for i in range(out.shape[0]):
        j = int(np.argmax(np.abs(out[i])))
        if out[i, j] < 0:
            out[i] *= -1.0
    return out


def fit_pca(
    train: AlignedMatrix,
    *,
    variance_threshold: float = 0.95,
    limit_quantile: float = 0.99,
) -> PcaModel:
    """Fit PCA on fully-observed aligned rows; limits are empirical."""
    x = train.matrix
    mean = x.mean(axis=0)
    xc = x - mean
    _, s, vt = np.linalg.svd(xc, full_matrices=False)
    var = s**2 / max(len(x) - 1, 1)
    ratio = var / var.sum()
    k = max(1, int(np.searchsorted(np.cumsum(ratio), variance_threshold) + 1))
    k = min(k, len(ratio))
    comps = _fix_signs(vt[:k])

    scores = xc @ comps.T
    t2 = np.einsum("ij,ij->i", scores / np.maximum(var[:k], 1e-12), scores)
    resid = xc - scores @ comps
    spe = (resid**2).sum(axis=1)

    return PcaModel(
        columns=list(train.columns),
        mean=mean,
        components=comps,
        score_var=var[:k].copy(),
        t2_limit=float(np.quantile(t2, limit_quantile)),
        spe_limit=float(np.quantile(spe, limit_quantile)),
        explained_variance=tuple(float(r) for r in ratio[:k]),
    )


@dataclass(frozen=True)
class MspcDetection:
    timestamps: pd.DatetimeIndex
    t2: np.ndarray
    spe: np.ndarray
    t2_breaches: list[pd.Timestamp]
    spe_breaches: list[pd.Timestamp]
    top_contributors: dict[str, list[str]]  # breach ts iso -> contributing tags (max 3)


def detect(model: PcaModel, monitor: AlignedMatrix) -> MspcDetection:
    if list(monitor.columns) != model.columns:
        raise MspcAlignmentError("monitor matrix columns differ from trained model")
    xc = monitor.matrix - model.mean
    scores = xc @ model.components.T
    t2 = np.einsum("ij,ij->i", scores / np.maximum(model.score_var, 1e-12), scores)
    resid = xc - scores @ model.components
    spe = (resid**2).sum(axis=1)

    t2_mask = t2 > model.t2_limit
    spe_mask = spe > model.spe_limit
    contributors: dict[str, list[str]] = {}
    for idx in np.where(t2_mask | spe_mask)[0]:
        contrib = np.abs(resid[idx]) * np.abs(model.components).sum(axis=0)
        top = [model.columns[j] for j in np.argsort(contrib)[::-1][:3]]
        contributors[monitor.index[idx].isoformat()] = top

    return MspcDetection(
        timestamps=monitor.index,
        t2=t2,
        spe=spe,
        t2_breaches=[cast(pd.Timestamp, monitor.index[i]) for i in np.where(t2_mask)[0]],
        spe_breaches=[cast(pd.Timestamp, monitor.index[i]) for i in np.where(spe_mask)[0]],
        top_contributors=contributors,
    )

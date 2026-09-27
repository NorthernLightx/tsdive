"""Before/after shift intervals on ordered series: estimators, refusal rules, generator.

The level estimate is mean(after) - mean(before). Three interval methods:

- ``naive``: Welch-style standard error that treats the samples as independent.
- ``hac``: per-period long-run variance, Newey-West Bartlett kernel with the
  Andrews (1991) AR(1) plug-in bandwidth capped at n - 1.
- ``block_bootstrap``: moving block bootstrap inside each period, percentile
  interval on the difference of resampled means.

The spread estimate is log(SD after / SD before), with a ``naive`` interval
from the normal-theory variance of a log variance ratio and a
``block_bootstrap`` interval from the same resampler.

``adjusted`` fits OLS of the target on the covariates over the before period
only, applies the coefficients to both periods and runs the same method on the
residuals. ``raw`` runs the method on the target itself.

Rules, each recorded on its row:

- R0 ``too_few``: fewer than 30 samples in either period. ``no_spread``: the
  target's before-period MAD is 0, or the before-period residual MAD is 0.
- R1 ``before_trend``: the HAC t statistic of an OLS slope over the before
  period exceeds 1.96 in absolute value, computed on the series the interval
  reads (the target for ``raw``, the residuals for ``adjusted``).
- R2 ``covariate_outside``: a covariate's after-period median lies outside its
  before-period 1st to 99th percentile. Adjusted arm only.
- R3 ``covariate_shifted``: a covariate's own ``hac`` level interval excludes 0.
  Adjusted arm only.

R0 and R2 refuse the row. R1 and R3 are flags; ``REFUSAL_SETS`` turns them
into refusals when a summary is computed.

The generator draws y_t = delta 1[after] + rho x_t + sqrt(1 - rho^2) e_t + drift_t,
with x and e independent unit-variance AR(1) series sharing phi. The recorded
covariate can carry its own step after the change date, which y does not read.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

Z95 = 1.959964
ALPHA = 0.05
MIN_SAMPLES = 30
TREND_T = 1.96
BOOT_REPLICATES = 200
BOOT_SEED = 42
BOOT_MIN_BLOCK = 10  # the block rule of tsdive.compare._block_bootstrap
ANDREWS_BARTLETT = 1.1447
AR1_CLIP = 0.9999  # keeps the Andrews plug-in finite; the n - 1 cap binds first
OUTSIDE_LO, OUTSIDE_HI = 1.0, 99.0
# An OLS residual of an exact linear copy is float noise, not zero; below this
# fraction of the target's before MAD the residual MAD counts as 0.
RESIDUAL_MAD_RTOL = 1e-9

LEVEL = "level"
SPREAD = "spread"
NAIVE = "naive"
HAC = "hac"
BOOTSTRAP = "block_bootstrap"
LEVEL_METHODS = (NAIVE, HAC, BOOTSTRAP)
SPREAD_METHODS = (NAIVE, BOOTSTRAP)
QUANTITY_METHODS = ((LEVEL, LEVEL_METHODS), (SPREAD, SPREAD_METHODS))
RAW = "raw"
ADJUSTED = "adjusted"
ADJUSTMENTS = (RAW, ADJUSTED)

TOO_FEW = "too_few"
NO_SPREAD = "no_spread"
NO_SPREAD_AFTER = "no_spread_after"
NO_COVARIATE = "no_covariate"
COVARIATE_OUTSIDE = "covariate_outside"
BEFORE_TREND = "before_trend"
COVARIATE_SHIFTED = "covariate_shifted"

# Hard refusals (R0, R2) apply in every set; the flags listed here are added.
REFUSAL_SETS = {
    "base": (),
    "base+R1": (BEFORE_TREND,),
    "base+R3": (COVARIATE_SHIFTED,),
    "base+R1+R3": (BEFORE_TREND, COVARIATE_SHIFTED),
}
RAW_REFUSAL_SETS = ("base", "base+R1")


# ---------------------------------------------------------------- basics


def mad(values: np.ndarray) -> float:
    """Median absolute deviation from the median, unscaled."""
    return float(np.median(np.abs(values - np.median(values))))


def normal_p(z: float) -> float:
    """Two-sided normal p-value of a z statistic."""
    if not math.isfinite(z):
        return 0.0
    return math.erfc(abs(z) / math.sqrt(2.0))


def autocovariances(u: np.ndarray) -> np.ndarray:
    """gamma_0 .. gamma_{n-1} of the demeaned series, each divided by n."""
    n = len(u)
    c = u - u.mean()
    size = 1 << (2 * n - 1).bit_length()
    spec = np.fft.rfft(c, size)
    return np.fft.irfft(spec * np.conj(spec), size)[:n] / n


def andrews_bandwidth(u: np.ndarray) -> float:
    """Andrews (1991) AR(1) plug-in bandwidth for the Bartlett kernel, capped at n - 1."""
    n = len(u)
    c = u - u.mean()
    denom = float(np.dot(c[:-1], c[:-1]))
    if n < 3 or denom <= 0:
        return 0.0
    rho = float(np.dot(c[1:], c[:-1])) / denom
    rho = min(max(rho, -AR1_CLIP), AR1_CLIP)
    alpha = 4.0 * rho**2 / ((1.0 - rho) ** 2 * (1.0 + rho) ** 2)
    return min(ANDREWS_BARTLETT * (alpha * n) ** (1.0 / 3.0), float(n - 1))


def long_run_variance(u: np.ndarray) -> float:
    """Newey-West Bartlett long-run variance with the Andrews bandwidth S.

    Lag j carries weight 1 - j / S for j < S, so the estimate is never negative.
    """
    gamma = autocovariances(u)
    bandwidth = andrews_bandwidth(u)
    lags = np.arange(1, math.ceil(bandwidth))
    lags = lags[lags < bandwidth]
    if lags.size == 0:
        return float(gamma[0])
    weights = 1.0 - lags / bandwidth
    return float(gamma[0] + 2.0 * np.dot(weights, gamma[lags]))


def trend_t(u: np.ndarray) -> float:
    """HAC t statistic of the OLS slope of u on its sample index."""
    n = len(u)
    t = np.arange(n, dtype=float)
    tc = t - t.mean()
    sxx = float(np.dot(tc, tc))
    slope = float(np.dot(tc, u)) / sxx
    resid = u - u.mean() - slope * tc
    lrv = long_run_variance(tc * resid)
    var = n * lrv / sxx**2
    if var <= 0:
        return 0.0 if slope == 0 else math.copysign(math.inf, slope)
    return slope / math.sqrt(var)


def block_size(n: int) -> int:
    return min(max(BOOT_MIN_BLOCK, int(n ** (1 / 3))), n)


@lru_cache(maxsize=64)
def _resample_indices(n_before: int, n_after: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """(B, n) moving-block indices for the before and then the after period."""
    rng = np.random.default_rng(seed)
    out = []
    for n in (n_before, n_after):
        block = block_size(n)
        n_blocks = -(-n // block)
        starts = rng.integers(0, n - block + 1, size=(BOOT_REPLICATES, n_blocks))
        idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(
            BOOT_REPLICATES, -1
        )[:, :n]
        idx.setflags(write=False)
        out.append(idx)
    return out[0], out[1]


def resample_indices(n_before: int, n_after: int, seed: int = BOOT_SEED):
    """Moving block bootstrap indices, one resampler per period, deterministic in seed."""
    return _resample_indices(int(n_before), int(n_after), int(seed))


# ---------------------------------------------------------------- intervals


@dataclass(frozen=True)
class Interval:
    """An estimate, its 95% interval and a two-sided p-value, in the input units."""

    estimate: float
    lo: float
    hi: float
    p_value: float

    @property
    def clears(self) -> bool:
        return bool(self.lo > 0 or self.hi < 0)


def _normal_interval(estimate: float, se: float) -> Interval:
    if se <= 0:
        z = 0.0 if estimate == 0 else math.copysign(math.inf, estimate)
        return Interval(estimate, estimate, estimate, normal_p(z))
    return Interval(estimate, estimate - Z95 * se, estimate + Z95 * se, normal_p(estimate / se))


def _percentile_interval(estimate: float, draws: np.ndarray) -> Interval:
    lo, hi = np.quantile(draws, [ALPHA / 2, 1 - ALPHA / 2])
    p = 2.0 * min(float(np.mean(draws <= 0)), float(np.mean(draws >= 0)))
    return Interval(estimate, float(lo), float(hi), min(p, 1.0))


def level_interval(before: np.ndarray, after: np.ndarray, method: str) -> Interval:
    """Interval for mean(after) - mean(before)."""
    estimate = float(after.mean() - before.mean())
    nb, na = len(before), len(after)
    if method == NAIVE:
        se = math.sqrt(before.var(ddof=1) / nb + after.var(ddof=1) / na)
        return _normal_interval(estimate, se)
    if method == HAC:
        se = math.sqrt(long_run_variance(before) / nb + long_run_variance(after) / na)
        return _normal_interval(estimate, se)
    if method == BOOTSTRAP:
        ib, ia = resample_indices(nb, na)
        draws = after[ia].mean(axis=1) - before[ib].mean(axis=1)
        return _percentile_interval(estimate, draws)
    raise ValueError(f"unknown level method {method!r}")


def spread_interval(before: np.ndarray, after: np.ndarray, method: str) -> Interval | None:
    """Interval for log(SD after / SD before), or None when a resample has no spread."""
    nb, na = len(before), len(after)
    estimate = float(math.log(after.std(ddof=1) / before.std(ddof=1)))
    if method == NAIVE:
        se = 0.5 * math.sqrt(2.0 / (na - 1) + 2.0 / (nb - 1))
        return _normal_interval(estimate, se)
    if method == BOOTSTRAP:
        ib, ia = resample_indices(nb, na)
        sd_b = before[ib].std(axis=1, ddof=1)
        sd_a = after[ia].std(axis=1, ddof=1)
        if not (np.all(sd_b > 0) and np.all(sd_a > 0)):
            return None
        return _percentile_interval(estimate, np.log(sd_a / sd_b))
    raise ValueError(f"unknown spread method {method!r}")


# ---------------------------------------------------------------- adjustment


def adjust(
    y_before: np.ndarray, y_after: np.ndarray, x_before: np.ndarray, x_after: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Residuals of both periods from an OLS fit over the before period, and the
    variance reduction 1 - var(residual before) / var(target before)."""
    design = np.column_stack([np.ones(len(y_before)), x_before])
    coef, *_ = np.linalg.lstsq(design, y_before, rcond=None)
    resid_b = y_before - design @ coef
    resid_a = y_after - np.column_stack([np.ones(len(y_after)), x_after]) @ coef
    var_y = float(y_before.var(ddof=1))
    reduction = 1.0 - float(resid_b.var(ddof=1)) / var_y if var_y > 0 else math.nan
    return resid_b, resid_a, reduction


def covariate_outside(x_before: np.ndarray, x_after: np.ndarray) -> bool:
    """True when a column's after median lies outside its before 1st-99th percentile."""
    lo, hi = np.percentile(x_before, [OUTSIDE_LO, OUTSIDE_HI], axis=0)
    median = np.median(x_after, axis=0)
    return bool(np.any((median < lo) | (median > hi)))


def covariate_shifted(x_before: np.ndarray, x_after: np.ndarray) -> bool:
    """True when a column's own ``hac`` level interval excludes 0."""
    return any(
        level_interval(x_before[:, j], x_after[:, j], HAC).clears
        for j in range(x_before.shape[1])
    )


def usable(before: np.ndarray, after: np.ndarray) -> str:
    """'' when a tag passes R0 on its finite samples, else the reason."""
    b = before[np.isfinite(before)]
    a = after[np.isfinite(after)]
    if len(b) < MIN_SAMPLES or len(a) < MIN_SAMPLES:
        return TOO_FEW
    if mad(b) == 0:
        return NO_SPREAD
    return ""


# ---------------------------------------------------------------- one target


def _row(
    quantity: str,
    method: str,
    adjustment: str,
    n_before: int,
    n_after: int,
    n_covariates: int,
) -> dict:
    return {
        "quantity": quantity,
        "method": method,
        "adjustment": adjustment,
        "n_before": n_before,
        "n_after": n_after,
        "n_covariates": n_covariates,
        "scale": math.nan,
        "estimate": math.nan,
        "lo": math.nan,
        "hi": math.nan,
        "p_value": math.nan,
        "clears": None,
        "refused": False,
        "reason": "",
        "trend_t": math.nan,
        "before_trend": None,
        "covariate_outside": None,
        "covariate_shifted": None,
        "variance_reduction": math.nan,
    }


def _refused_rows(
    quantities: tuple[str, ...],
    adjustment: str,
    reason: str,
    nb: int,
    na: int,
    k: int,
    **flags,
) -> list[dict]:
    rows = []
    for quantity, methods in QUANTITY_METHODS:
        if quantity not in quantities:
            continue
        for method in methods:
            row = _row(quantity, method, adjustment, nb, na, k)
            row.update(refused=True, reason=reason, **flags)
            rows.append(row)
    return rows


def _arm_rows(
    quantities: tuple[str, ...],
    before: np.ndarray,
    after: np.ndarray,
    scale: float,
    adjustment: str,
    n_covariates: int,
    **extra,
) -> list[dict]:
    """Every (quantity, method) row of one arm that passed its hard refusals."""
    nb, na = len(before), len(after)
    t = trend_t(before)
    rows = []
    for quantity, methods in QUANTITY_METHODS:
        if quantity not in quantities:
            continue
        for method in methods:
            row = _row(quantity, method, adjustment, nb, na, n_covariates)
            row.update(trend_t=t, before_trend=bool(abs(t) > TREND_T), scale=scale, **extra)
            if quantity == LEVEL:
                interval = level_interval(before, after, method)
            elif after.std(ddof=1) == 0:
                row.update(refused=True, reason=NO_SPREAD_AFTER)
                rows.append(row)
                continue
            else:
                interval = spread_interval(before, after, method)
                if interval is None:
                    row.update(refused=True, reason=NO_SPREAD_AFTER)
                    rows.append(row)
                    continue
            row.update(
                estimate=interval.estimate,
                lo=interval.lo,
                hi=interval.hi,
                p_value=interval.p_value,
                clears=interval.clears,
            )
            rows.append(row)
    return rows


def score_target(
    before: dict[str, np.ndarray],
    after: dict[str, np.ndarray],
    target: str,
    covariates: list[str],
    quantities: tuple[str, ...] = (LEVEL, SPREAD),
) -> list[dict]:
    """Rows for one target: raw and adjusted arms, every quantity and method.

    ``before`` and ``after`` map each tag to one period's samples, NaN where
    missing and position-aligned across tags. ``covariates`` are the tags the
    adjusted arm regresses on, declared before any interval is computed.
    Level rows carry values in the target's units; ``scale`` is the target's
    before-period SD over the rows the arm used.
    """
    yb_all, ya_all = before[target], after[target]
    fb, fa = np.isfinite(yb_all), np.isfinite(ya_all)
    yb, ya = yb_all[fb], ya_all[fa]
    rows: list[dict] = []
    reason = usable(yb_all, ya_all)
    if reason:
        rows += _refused_rows(quantities, RAW, reason, len(yb), len(ya), 0)
    else:
        rows += _arm_rows(quantities, yb, ya, float(yb.std(ddof=1)), RAW, 0)

    k = len(covariates)
    if k == 0:
        return rows + _refused_rows(quantities, ADJUSTED, NO_COVARIATE, len(yb), len(ya), 0)
    xb_all = np.column_stack([before[c] for c in covariates])
    xa_all = np.column_stack([after[c] for c in covariates])
    jb = fb & np.all(np.isfinite(xb_all), axis=1)
    ja = fa & np.all(np.isfinite(xa_all), axis=1)
    yb2, ya2, xb, xa = yb_all[jb], ya_all[ja], xb_all[jb], xa_all[ja]
    nb, na = len(yb2), len(ya2)
    if nb < MIN_SAMPLES or na < MIN_SAMPLES:
        return rows + _refused_rows(quantities, ADJUSTED, TOO_FEW, nb, na, k)
    if mad(yb2) == 0:
        return rows + _refused_rows(quantities, ADJUSTED, NO_SPREAD, nb, na, k)
    outside = covariate_outside(xb, xa)
    shifted = covariate_shifted(xb, xa)
    flags = {"covariate_outside": outside, "covariate_shifted": shifted}
    if outside:
        return rows + _refused_rows(quantities, ADJUSTED, COVARIATE_OUTSIDE, nb, na, k, **flags)
    resid_b, resid_a, reduction = adjust(yb2, ya2, xb, xa)
    if mad(resid_b) <= RESIDUAL_MAD_RTOL * mad(yb2):
        return rows + _refused_rows(
            quantities, ADJUSTED, NO_SPREAD, nb, na, k, variance_reduction=reduction, **flags
        )
    return rows + _arm_rows(
        quantities,
        resid_b,
        resid_a,
        float(yb2.std(ddof=1)),
        ADJUSTED,
        k,
        variance_reduction=reduction,
        **flags,
    )


def refused_under(row: dict, refusal_set: str) -> bool:
    """True when the row is refused once the set's flags count as refusals."""
    if row["refused"]:
        return True
    return any(bool(row.get(flag)) for flag in REFUSAL_SETS[refusal_set])


# ---------------------------------------------------------------- generator


def cell_seed(params: dict) -> int:
    """Seed derived from the cell's parameters alone, so cell order does not matter."""
    key = "|".join(f"{name}={params[name]!r}" for name in sorted(params))
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "little")


def ar1(rng: np.random.Generator, phi: float, replicates: int, length: int) -> np.ndarray:
    """(replicates, length) stationary AR(1) paths with unit marginal variance."""
    out = np.empty((replicates, length))
    innov = rng.standard_normal((replicates, length))
    out[:, 0] = innov[:, 0]
    scale = math.sqrt(1.0 - phi**2)
    for t in range(1, length):
        out[:, t] = phi * out[:, t - 1] + scale * innov[:, t]
    return out


def simulate(
    *,
    phi: float,
    rho: float,
    drift: float,
    delta: float,
    n: int,
    replicates: int,
    sd_ratio: float = 1.0,
    covariate_step: float = 0.0,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """(y, x), each (replicates, 2n): n samples before the change date, n after.

    y = rho x0 + sqrt(1 - rho^2) e, times ``sd_ratio`` after the change date,
    plus ``delta`` after it and a linear drift rising by ``drift`` over the
    2n samples. The recorded covariate is x0 plus ``covariate_step`` after the
    change date; y reads x0, so the target's true shift is ``delta``.
    """
    rng = np.random.default_rng(seed)
    x0 = ar1(rng, phi, replicates, 2 * n)
    e = ar1(rng, phi, replicates, 2 * n)
    y = rho * x0 + math.sqrt(1.0 - rho**2) * e
    y[:, n:] *= sd_ratio
    y[:, n:] += delta
    y += drift * np.arange(2 * n) / (2 * n - 1)
    x = x0.copy()
    x[:, n:] += covariate_step
    return y, x

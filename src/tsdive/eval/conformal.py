"""Conformal p-values and the test martingale alarm built on them.

A nonconformity score per observation, a calibration set of scores drawn
from the in-control record, and a stream of scores to watch. Each stream
score gets a conformal p-value against the pool it is exchangeable with,
the p-values feed a test martingale, and the martingale crossing
``1 / delta`` is the alarm. Under exchangeability the martingale exceeds
``1 / delta`` at any time with probability at most ``delta`` (Ville's
inequality), so ``delta`` is a false-alarm bound for the whole record
rather than for one window.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Iterable

import numpy as np

DEFAULT_EPSILONS: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def _finite_array(values: Iterable[float], name: str) -> np.ndarray:
    arr = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} holds a non-finite value")
    return arr


def conformal_p_values(
    calibration: Iterable[float], stream: Iterable[float], *, online: bool = True
) -> np.ndarray:
    """Conformal p-value of each stream score against its pool, in stream order.

    For stream position ``t`` the pool is ``calibration`` plus
    ``stream[:t]`` when ``online`` is true, and ``calibration`` alone
    otherwise. The p-value is
    ``(count(pool > x_t) + count(pool == x_t) + 1) / (len(pool) + 1)``:
    ties count against the new score, so the value is conservative and
    the function is deterministic, with no randomised tie-breaking.

    Under exchangeability of the calibration and stream scores each
    ``p_t`` is stochastically no smaller than a uniform variable on
    ``(0, 1]``, and with ``online=True`` the ``p_t`` are independent,
    which is the property ``power_martingale`` and ``mixture_martingale``
    need. Raises ``ValueError`` when ``calibration`` is empty or either
    input holds a non-finite value.
    """
    cal = _finite_array(calibration, "calibration")
    obs = _finite_array(stream, "stream")
    if cal.size == 0:
        raise ValueError("conformal_p_values needs at least one calibration score")
    pool = sorted(cal.tolist())
    out = np.empty(obs.size, dtype=float)
    for t, x in enumerate(obs.tolist()):
        lo = bisect.bisect_left(pool, x)
        n_ge = len(pool) - lo
        out[t] = (n_ge + 1) / (len(pool) + 1)
        if online:
            pool.insert(lo, x)
    return out


def _p_array(p_values: Iterable[float]) -> np.ndarray:
    p = _finite_array(p_values, "p_values")
    if p.size and (np.any(p <= 0.0) or np.any(p > 1.0)):
        raise ValueError("p_values must lie in (0, 1]")
    return p


def power_martingale(p_values: Iterable[float], epsilon: float) -> np.ndarray:
    """Natural log of the power martingale ``prod(epsilon * p ** (epsilon - 1))``.

    One value per stream position, cumulative from the first. ``epsilon``
    must lie in ``(0, 1]``; ``epsilon == 1`` gives zeros. Raises
    ``ValueError`` for ``epsilon`` outside that interval or a p-value
    outside ``(0, 1]``.
    """
    eps = float(epsilon)
    if not (0.0 < eps <= 1.0) or not math.isfinite(eps):
        raise ValueError(f"epsilon must lie in (0, 1], got {epsilon!r}")
    p = _p_array(p_values)
    return np.cumsum(math.log(eps) + (eps - 1.0) * np.log(p))


def mixture_martingale(
    p_values: Iterable[float], epsilons: Iterable[float] = DEFAULT_EPSILONS
) -> np.ndarray:
    """Natural log of the mean over ``epsilons`` of the power martingales.

    One value per stream position, computed in log space with
    ``np.logaddexp.reduce``. A mixture of martingales is a martingale, so
    ``martingale_alarm`` keeps its bound. Raises ``ValueError`` when
    ``epsilons`` is empty or any epsilon lies outside ``(0, 1]``.
    """
    eps = [float(e) for e in epsilons]
    if not eps:
        raise ValueError("mixture_martingale needs at least one epsilon")
    p = _p_array(p_values)
    logs = np.stack([power_martingale(p, e) for e in eps], axis=0)
    return np.logaddexp.reduce(logs, axis=0) - math.log(len(eps))


def martingale_alarm(log_values: Iterable[float], delta: float) -> int | None:
    """First index where the log martingale reaches ``log(1 / delta)``, else ``None``.

    Ville's inequality: under exchangeability of the calibration and
    stream scores, the probability that the martingale ever reaches
    ``1 / delta`` is at most ``delta``, so the bound covers the whole
    record and not each window separately. The bound rests on that
    exchangeability; any fitted centre and scale that produce the scores
    must come from a fit set disjoint from the calibration and stream.
    ``delta`` must lie in ``(0, 1)``, else ``ValueError``.
    """
    d = float(delta)
    if not (0.0 < d < 1.0) or not math.isfinite(d):
        raise ValueError(f"delta must lie in (0, 1), got {delta!r}")
    values = _finite_array(log_values, "log_values")
    hits = np.flatnonzero(values >= -math.log(d))
    return int(hits[0]) if hits.size else None

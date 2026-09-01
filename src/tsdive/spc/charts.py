"""SPC charts with per-rule reporting.

Control limits come from baseline objects, never from the monitored
window itself. Each rule reports its own hits; there is no blended alarm
score. Constants are the classic Shewhart tables (A2, D3, D4 for xbar/R
with n = 2..10).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

# Shewhart constants for subgroup size n (ASTM-style tables):
XBAR_A2 = {
    2: 1.880,
    3: 1.023,
    4: 0.729,
    5: 0.577,
    6: 0.483,
    7: 0.419,
    8: 0.373,
    9: 0.337,
    10: 0.308,
}
R_D3 = {2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.076, 8: 0.136, 9: 0.184, 10: 0.223}
R_D4 = {n: 3.267 for n in range(2, 7)} | {
    7: 1.924,
    8: 1.864,
    9: 1.816,
    10: 1.777,
}


@dataclass(frozen=True)
class ControlLimits:
    center: float
    ucl: float
    lcl: float
    basis: str


@dataclass(frozen=True)
class RuleHit:
    rule: str
    timestamp: pd.Timestamp
    detail: str


def individuals_limits(center: float, sigma: float) -> ControlLimits:
    return ControlLimits(
        center=center, ucl=center + 3 * sigma, lcl=center - 3 * sigma, basis="individuals 3-sigma"
    )


def xbar_r_limits(subgroups: list[list[float]]) -> tuple[ControlLimits, ControlLimits]:
    """xbar and R chart limits from historical subgroups (equal size n)."""
    sizes = {len(s) for s in subgroups}
    if len(sizes) != 1:
        raise ValueError("subgroups must be equal size")
    n = sizes.pop()
    if n not in XBAR_A2:
        raise ValueError(f"subgroup size {n} outside supported table (2..10)")
    xbars = [float(np.mean(s)) for s in subgroups]
    ranges = [float(np.ptp(s)) for s in subgroups]
    xbb = float(np.mean(xbars))
    rbar = float(np.mean(ranges))
    xbar_limits = ControlLimits(
        center=xbb, ucl=xbb + XBAR_A2[n] * rbar, lcl=xbb - XBAR_A2[n] * rbar, basis=f"xbar n={n}"
    )
    r_limits = ControlLimits(
        center=rbar, ucl=R_D4[n] * rbar, lcl=R_D3[n] * rbar, basis=f"R n={n}"
    )
    return xbar_limits, r_limits


def _runs_hits(
    ts: list[pd.Timestamp], values: list[float], limits: ControlLimits
) -> list[RuleHit]:
    hits: list[RuleHit] = []
    side = [1 if v > limits.center else (-1 if v < limits.center else 0) for v in values]

    run_len = 0
    for i, s in enumerate(side):
        run_len = run_len + 1 if s != 0 and (i > 0 and side[i - 1] == s) else (1 if s != 0 else 0)
        if run_len == 9:
            hits.append(
                RuleHit(
                    "RUN_9_SAMESIDE",
                    ts[i],
                    f"9 consecutive points on one side of center {limits.center:.4g}",
                )
            )

    trend = 0
    for i in range(1, len(values)):
        d = values[i] - values[i - 1]
        prev_d = values[i - 1] - values[i - 2] if i >= 2 else 0.0
        if not math.isfinite(d):
            trend = 0
            continue
        same = (d > 0 and prev_d > 0) or (d < 0 and prev_d < 0)
        trend = trend + 1 if (same and trend > 0) or trend == 0 else 1
        if trend >= 5:  # 6 points steadily rising/falling incl. first
            hits.append(RuleHit("TREND_6", ts[i], "6 consecutively increasing/decreasing points"))
    return hits


def apply_rules(
    timestamps: pd.Series, values: pd.Series, limits: ControlLimits
) -> list[RuleHit]:
    """Evaluate the SPC rule set over a monitored series."""
    ts_list = [cast(pd.Timestamp, pd.Timestamp(t)) for t in timestamps]
    val_list = [float(v) for v in values]

    def _beyond_hit(i: int) -> RuleHit:
        return RuleHit(
            "BEYOND_3SIGMA",
            ts_list[i],
            f"value {val_list[i]:.4g} outside [{limits.lcl:.4g}, {limits.ucl:.4g}]",
        )

    hits: list[RuleHit] = [
        _beyond_hit(i) for i, v in enumerate(val_list) if v > limits.ucl or v < limits.lcl
    ]
    hits.extend(_runs_hits(ts_list, val_list, limits))
    return hits

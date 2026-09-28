"""Randomized switchback schedules of two settings, and their randomization analysis.

:mod:`tsdive.switchback.design` draws a balanced schedule of K blocks and
the reference assignments; :mod:`tsdive.switchback.inference` computes
the B - A difference on the kept samples, its randomization p-value and
the interval that inverts the test; :mod:`tsdive.switchback.plan` holds
the schedule as a plan with a digest, checks it, and reads its power off
a history window; :mod:`tsdive.switchback.archive` reads the target and
covariates through the store and runs the analysis, and
:mod:`tsdive.switchback.render` writes both as text and JSON. The package
imports numpy and pandas and nothing heavier.
"""

from __future__ import annotations

from tsdive.switchback.design import (
    ALPHA,
    MIN_ASSIGNMENTS,
    PERMUTATIONS,
    Blocks,
    Design,
    cut_blocks,
    design_size,
    make_design,
    schedule_blocks,
    setting,
)
from tsdive.switchback.inference import (
    COLLINEAR,
    EMPTY_BLOCK,
    NO_SPREAD,
    REASONS,
    TOO_FEW,
    TOO_MANY_COVARIATES,
    Analysis,
    analyze,
    lag_response,
)
from tsdive.switchback.plan import (
    POWER_DELTAS,
    POWER_DRAWS,
    PlannedBlock,
    PowerReadout,
    SwitchbackPlan,
    make_plan,
    plan_digest,
    verify_plan,
)

__all__ = [
    "ALPHA",
    "COLLINEAR",
    "EMPTY_BLOCK",
    "MIN_ASSIGNMENTS",
    "NO_SPREAD",
    "PERMUTATIONS",
    "POWER_DELTAS",
    "POWER_DRAWS",
    "REASONS",
    "TOO_FEW",
    "TOO_MANY_COVARIATES",
    "Analysis",
    "Blocks",
    "Design",
    "PlannedBlock",
    "PowerReadout",
    "SwitchbackPlan",
    "analyze",
    "cut_blocks",
    "design_size",
    "lag_response",
    "make_design",
    "make_plan",
    "plan_digest",
    "schedule_blocks",
    "setting",
    "verify_plan",
]

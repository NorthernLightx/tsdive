"""Baselines: provisional and refined.

A refined baseline improves on a provisional one in two directions.
:mod:`~tsdive.baselines.regime` conditions on a MODE tag, so a limit
means something different in each plant state.
:mod:`~tsdive.baselines.population` conditions on the asset and
reaches the case neither can: a tag frozen for its whole life, whose own
history says nothing because its own history never moved.
"""

from __future__ import annotations

from tsdive.baselines.population import (
    MIN_BASELINE_WINDOWS,
    PopulationBaseline,
    PopulationScreen,
    match_population,
    population_baseline,
    screen_population,
)
from tsdive.baselines.provisional import (
    ProvisionalBaseline,
    ScreenResult,
    mad_baseline,
    moving_range_baseline,
    screen,
)
from tsdive.baselines.regime import (
    RegimeBaseline,
    match_regime,
    regime_baselines,
    screen_regime,
)

__all__ = [
    "MIN_BASELINE_WINDOWS",
    "PopulationBaseline",
    "PopulationScreen",
    "ProvisionalBaseline",
    "RegimeBaseline",
    "ScreenResult",
    "mad_baseline",
    "match_population",
    "match_regime",
    "moving_range_baseline",
    "population_baseline",
    "regime_baselines",
    "screen",
    "screen_population",
    "screen_regime",
]

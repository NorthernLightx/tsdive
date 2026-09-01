"""Population baselines: indicting a tag its own history cannot."""

from __future__ import annotations

import pandas as pd
import pytest

from tsdive.baselines import (
    match_population,
    population_baseline,
    screen_population,
)
from tsdive.errors import PopulationTooSparse


def feature_rows(
    asset: str, variable: str, distinct: list[float], stall: list[float] | None = None
) -> pd.DataFrame:
    stall = stall if stall is not None else [1.0] * len(distinct)
    return pd.DataFrame(
        {
            "asset": [asset] * len(distinct),
            "variable": [variable] * len(distinct),
            "distinct_count": distinct,
            "stall_s": stall,
        }
    )


def moving_pair() -> pd.DataFrame:
    """A busy transmitter: 40 windows, hundreds of distinct values each."""
    return feature_rows("ASSET-A", "P-TPT", [800.0 + i for i in range(40)])


def still_pair() -> pd.DataFrame:
    """A transmitter that is flat on this asset: 40 windows, all frozen."""
    return feature_rows("ASSET-A", "P-PDG", [1.0] * 40, [3599.0] * 40)


def test_baseline_summarises_each_pair():
    bases = population_baseline(pd.concat([moving_pair(), still_pair()]))
    assert set(bases) == {("ASSET-A", "P-TPT"), ("ASSET-A", "P-PDG")}
    moving = bases[("ASSET-A", "P-TPT")]
    assert moving.n_windows == 40
    assert moving.p05_distinct > 1
    assert moving.n_frozen_windows == 0
    assert moving.sufficient
    still = bases[("ASSET-A", "P-PDG")]
    assert still.p05_distinct == 1.0
    assert still.n_frozen_windows == 40
    assert still.frozen_share == 1.0


def test_fires_on_a_frozen_window_of_a_pair_that_normally_moves():
    base = population_baseline(moving_pair())[("ASSET-A", "P-TPT")]
    result = screen_population(pd.Series({"distinct_count": 1.0}), base)
    assert result.fired
    assert result.observed_distinct == 1
    assert result.n_baseline_windows == 40
    # The evidence names the size of the population the claim rests on.
    assert "40 training windows" in result.evidence


def test_does_not_fire_on_a_pair_that_is_normally_still():
    base = population_baseline(still_pair())[("ASSET-A", "P-PDG")]
    result = screen_population(pd.Series({"distinct_count": 1.0}), base)
    assert not result.fired
    assert "40 of its 40 training windows" in result.evidence


def test_does_not_fire_on_a_window_that_moved():
    base = population_baseline(moving_pair())[("ASSET-A", "P-TPT")]
    result = screen_population(pd.Series({"distinct_count": 640.0}), base)
    assert not result.fired
    assert "not frozen" in result.evidence


def test_refuses_a_baseline_below_the_declared_size():
    rows = feature_rows("ASSET-A", "P-TPT", [800.0] * 8)
    base = population_baseline(rows, min_windows=20)[("ASSET-A", "P-TPT")]
    assert not base.sufficient
    assert base.n_windows == 8
    with pytest.raises(PopulationTooSparse, match="8 training window"):
        screen_population(pd.Series({"distinct_count": 1.0}), base)


def test_refuses_a_pair_that_never_appeared_in_training():
    bases = population_baseline(moving_pair())
    with pytest.raises(PopulationTooSparse, match="only in the fold under test"):
        match_population(bases, "ASSET-B", "P-TPT")


def test_refuses_a_window_with_no_distinct_count():
    base = population_baseline(moving_pair())[("ASSET-A", "P-TPT")]
    with pytest.raises(PopulationTooSparse, match="nothing to compare"):
        screen_population(pd.Series({"distinct_count": None}), base)


def test_missing_column_refused():
    with pytest.raises(KeyError, match="stall_s"):
        population_baseline(moving_pair().drop(columns=["stall_s"]))


def test_pairs_do_not_pool_across_assets():
    """A busy asset must not indict a quiet asset's sensor."""
    quiet = feature_rows("ASSET-B", "P-TPT", [1.0] * 40, [3599.0] * 40)
    bases = population_baseline(pd.concat([moving_pair(), quiet]))
    result = screen_population(
        pd.Series({"distinct_count": 1.0}), bases[("ASSET-B", "P-TPT")]
    )
    assert not result.fired

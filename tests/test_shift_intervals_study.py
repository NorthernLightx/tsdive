"""The shift interval study on synthetic inputs: no network, no real data."""

from __future__ import annotations

import filecmp
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parents[1]
STUDY = ROOT / "examples" / "studies" / "shift_intervals"
sys.path.insert(0, str(STUDY))

import run_shift as rs  # noqa: E402
import shift as sh  # noqa: E402


def _ar1(phi: float, n: int, seed: int) -> np.ndarray:
    return sh.ar1(np.random.default_rng(seed), phi, 1, n)[0]


def _rows(rows: list[dict], adjustment: str) -> list[dict]:
    return [row for row in rows if row["adjustment"] == adjustment]


# ---------------------------------------------------------------- estimators


def test_hac_matches_naive_on_white_noise():
    rng = np.random.default_rng(0)
    before, after = rng.standard_normal(2000), rng.standard_normal(2000)
    naive = sh.level_interval(before, after, sh.NAIVE)
    hac = sh.level_interval(before, after, sh.HAC)
    ratio = (hac.hi - hac.lo) / (naive.hi - naive.lo)
    assert 0.9 < ratio < 1.1
    assert naive.estimate == hac.estimate


def test_hac_is_wider_than_naive_on_an_autocorrelated_series():
    before, after = _ar1(0.9, 480, 1), _ar1(0.9, 480, 2)
    naive = sh.level_interval(before, after, sh.NAIVE)
    hac = sh.level_interval(before, after, sh.HAC)
    assert (hac.hi - hac.lo) > 2 * (naive.hi - naive.lo)


def test_bandwidth_is_capped_at_n_minus_one():
    ramp = np.arange(50, dtype=float)
    assert sh.andrews_bandwidth(ramp) == 49.0
    assert sh.andrews_bandwidth(np.full(50, 3.0)) == 0.0


def test_block_bootstrap_is_deterministic_under_its_seed():
    rng = np.random.default_rng(3)
    before, after = rng.standard_normal(200), rng.standard_normal(150)
    first = sh.level_interval(before, after, sh.BOOTSTRAP)
    second = sh.level_interval(before.copy(), after.copy(), sh.BOOTSTRAP)
    assert first == second
    ib, ia = sh.resample_indices(200, 150)
    assert ib.shape == (sh.BOOT_REPLICATES, 200)
    assert ia.shape == (sh.BOOT_REPLICATES, 150)
    other_b, _ = sh.resample_indices(200, 150, seed=7)
    assert not np.array_equal(ib, other_b)
    assert sh.block_size(120) == 10
    assert sh.block_size(20_000) == 27


def test_spread_interval_reads_the_sd_ratio():
    rng = np.random.default_rng(4)
    before, after = rng.standard_normal(2000), 0.5 * rng.standard_normal(2000)
    for method in sh.SPREAD_METHODS:
        interval = sh.spread_interval(before, after, method)
        assert interval.lo < math.log(0.5) < interval.hi
        assert interval.clears


# ---------------------------------------------------------------- adjustment


def test_adjustment_removes_an_exact_linear_dependence():
    rng = np.random.default_rng(5)
    x = rng.standard_normal((400, 2))
    noise = 0.01 * rng.standard_normal(400)
    y = 3.0 + 2.0 * x[:, 0] - x[:, 1] + noise
    resid_b, resid_a, reduction = sh.adjust(y[:200], y[200:] + 1.0, x[:200], x[200:])
    assert reduction > 0.999
    assert abs(resid_b.mean()) < 1e-9
    assert abs(resid_a.mean() - 1.0) < 0.01


def test_a_residual_with_no_spread_is_refused():
    rng = np.random.default_rng(6)
    x = rng.standard_normal(200)
    before = {"y": 2.0 * x[:100], "x": x[:100]}
    after = {"y": 2.0 * x[100:], "x": x[100:]}
    rows = sh.score_target(before, after, "y", ["x"])
    adjusted = _rows(rows, sh.ADJUSTED)
    assert adjusted
    assert all(row["refused"] and row["reason"] == sh.NO_SPREAD for row in adjusted)
    assert not any(row["refused"] for row in _rows(rows, sh.RAW))


def test_no_covariate_refuses_the_adjusted_arm_only():
    rng = np.random.default_rng(7)
    before, after = {"y": rng.standard_normal(60)}, {"y": rng.standard_normal(60)}
    rows = sh.score_target(before, after, "y", [])
    assert all(row["reason"] == sh.NO_COVARIATE for row in _rows(rows, sh.ADJUSTED))
    assert not any(row["refused"] for row in _rows(rows, sh.RAW))


# ---------------------------------------------------------------- rules


def _pair(n_before: int, n_after: int, seed: int) -> tuple[dict, dict]:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n_before + n_after)
    y = 0.6 * x + 0.8 * rng.standard_normal(n_before + n_after)
    return (
        {"y": y[:n_before], "x": x[:n_before]},
        {"y": y[n_before:], "x": x[n_before:]},
    )


def test_r0_too_few_fires_below_30_samples():
    before, after = _pair(29, 40, 8)
    rows = sh.score_target(before, after, "y", ["x"])
    assert all(row["reason"] == sh.TOO_FEW for row in rows)
    before, after = _pair(30, 30, 8)
    assert not any(row["refused"] for row in sh.score_target(before, after, "y", ["x"]))


def test_r0_too_few_counts_finite_samples_only():
    before, after = _pair(40, 40, 9)
    before["y"][:11] = np.nan
    rows = sh.score_target(before, after, "y", ["x"])
    assert all(row["reason"] == sh.TOO_FEW for row in rows)


def test_r0_no_spread_fires_on_a_frozen_before_period():
    before, after = _pair(60, 60, 10)
    before["y"] = np.full(60, 4.0)
    assert sh.usable(before["y"], after["y"]) == sh.NO_SPREAD
    rows = sh.score_target(before, after, "y", ["x"])
    assert all(row["reason"] == sh.NO_SPREAD for row in rows)
    assert sh.usable(*_pair(60, 60, 10)[0].values()) == ""


def test_r1_flags_a_before_period_trend():
    rng = np.random.default_rng(11)
    ramp = np.linspace(0.0, 5.0, 200) + rng.standard_normal(200)
    assert abs(sh.trend_t(ramp)) > sh.TREND_T
    rows = sh.score_target({"y": ramp}, {"y": rng.standard_normal(200)}, "y", [])
    raw = _rows(rows, sh.RAW)
    assert all(row["before_trend"] and not row["refused"] for row in raw)
    assert all(sh.refused_under(row, "base+R1") for row in raw)
    assert not any(sh.refused_under(row, "base") for row in raw)
    quiet = rng.standard_normal(200)
    assert abs(sh.trend_t(quiet)) < sh.TREND_T


def test_r2_refuses_the_adjusted_arm_when_a_covariate_leaves_its_range():
    before, after = _pair(100, 100, 12)
    after["x"] = after["x"] + 10.0
    rows = sh.score_target(before, after, "y", ["x"])
    adjusted = _rows(rows, sh.ADJUSTED)
    assert all(row["reason"] == sh.COVARIATE_OUTSIDE for row in adjusted)
    assert all(row["covariate_outside"] for row in adjusted)
    assert not any(row["refused"] for row in _rows(rows, sh.RAW))
    before, after = _pair(100, 100, 12)
    adjusted = _rows(sh.score_target(before, after, "y", ["x"]), sh.ADJUSTED)
    assert not any(row["refused"] or row["covariate_outside"] for row in adjusted)


def test_r3_flags_a_shifted_covariate_inside_its_range():
    before, after = _pair(480, 480, 13)
    after["x"] = after["x"] + 0.5
    adjusted = _rows(sh.score_target(before, after, "y", ["x"]), sh.ADJUSTED)
    assert all(row["covariate_shifted"] for row in adjusted)
    assert not any(row["refused"] or row["covariate_outside"] for row in adjusted)
    assert all(sh.refused_under(row, "base+R3") for row in adjusted)
    before, after = _pair(480, 480, 13)
    adjusted = _rows(sh.score_target(before, after, "y", ["x"]), sh.ADJUSTED)
    assert not any(row["covariate_shifted"] for row in adjusted)


# ---------------------------------------------------------------- generator


def test_generator_is_deterministic_in_its_parameters():
    params = {"phi": 0.9, "rho": 0.5, "drift": 1.0, "delta": 0.25, "n": 50, "replicates": 4}
    first = sh.simulate(**params, seed=sh.cell_seed(params))
    second = sh.simulate(**params, seed=sh.cell_seed(dict(reversed(params.items()))))
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    third = sh.simulate(**{**params, "phi": 0.5}, seed=sh.cell_seed({**params, "phi": 0.5}))
    assert not np.array_equal(first[0], third[0])


def test_generator_places_the_shift_and_the_covariate_step():
    y, x = sh.simulate(
        phi=0.0,
        rho=0.0,
        drift=0.0,
        delta=5.0,
        n=2000,
        replicates=1,
        covariate_step=2.0,
        seed=1,
    )
    assert abs(y[0, 2000:].mean() - y[0, :2000].mean() - 5.0) < 0.2
    assert abs(x[0, 2000:].mean() - x[0, :2000].mean() - 2.0) < 0.2
    assert abs(y[0, :2000].std() - 1.0) < 0.1


def test_cell_seeds_do_not_depend_on_cell_order():
    cells = rs.synthetic_cells()
    seeds = {rs.cell_label(c): sh.cell_seed(c) for c in cells}
    shuffled = list(cells)
    random.Random(0).shuffle(shuffled)
    assert {rs.cell_label(c): sh.cell_seed(c) for c in shuffled} == seeds
    assert len(set(seeds.values())) == len(cells)


@pytest.fixture
def small_grid(monkeypatch):
    monkeypatch.setattr(rs, "PHIS", (0.0, 0.9))
    monkeypatch.setattr(rs, "RHOS", (0.0, 0.9))
    monkeypatch.setattr(rs, "DRIFTS", (0.0,))
    monkeypatch.setattr(rs, "DELTAS", (0.0, 1.0))
    monkeypatch.setattr(rs, "NS", (60,))
    monkeypatch.setattr(rs, "AFFECTED_DELTAS", (1.0,))
    monkeypatch.setattr(rs, "SD_RATIOS", (1.0, 0.5))


def test_synthetic_run_writes_identical_csvs_on_rerun(small_grid, tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    info = rs.run(first, beds=(rs.BED_SYNTHETIC,), replicates=6)
    rs.run(second, beds=(rs.BED_SYNTHETIC,), replicates=6)
    assert filecmp.cmp(first / "synthetic.csv", second / "synthetic.csv", shallow=False)
    section = info["beds"][rs.BED_SYNTHETIC]
    assert section["replicates"] == 6
    assert section["n_cells"] == {"level": 8, "affected_covariate": 2, "spread": 4}
    frame = pd.read_csv(first / "synthetic.csv")
    raw_sets = set(frame.loc[frame["adjustment"] == sh.RAW, "refusal_set"])
    assert raw_sets == set(sh.RAW_REFUSAL_SETS)
    assert set(frame.loc[frame["adjustment"] == sh.ADJUSTED, "refusal_set"]) == set(
        sh.REFUSAL_SETS
    )
    level = frame[frame["quantity"] == sh.LEVEL]
    assert set(level["method"]) == set(sh.LEVEL_METHODS)
    assert level.loc[level["delta"] == 0, "detect_rate"].isna().all()
    assert level.loc[level["delta"] > 0, "claim_rate"].isna().all()

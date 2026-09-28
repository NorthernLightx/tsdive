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

import build_tep_cache as btc  # noqa: E402
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


def test_r4_flags_a_persistent_series_and_not_white_noise():
    persistent = {"y": _ar1(0.98, 240, 14)}
    rows = sh.score_target(
        {"y": persistent["y"][:120]}, {"y": persistent["y"][120:]}, "y", []
    )
    raw = _rows(rows, sh.RAW)
    assert all(row["too_persistent"] and not row["refused"] for row in raw)
    assert all(sh.refused_under(row, "base+R4") for row in raw)
    assert not any(sh.refused_under(row, "base") for row in raw)
    rng = np.random.default_rng(14)
    quiet = sh.score_target(
        {"y": rng.standard_normal(120)}, {"y": rng.standard_normal(120)}, "y", []
    )
    assert not any(row["too_persistent"] for row in _rows(quiet, sh.RAW))
    assert sh.effective_n(100, 0.5) == pytest.approx(100 / 3)
    assert sh.effective_n(100, -0.4) == 100
    assert sh.effective_n(100, math.nan) == 100


def test_too_many_covariates_refuses_the_adjusted_arm_only():
    rng = np.random.default_rng(15)
    names = [f"x{i}" for i in range(11)]
    before = {name: rng.standard_normal(100) for name in [*names, "y"]}
    after = {name: rng.standard_normal(100) for name in [*names, "y"]}
    rows = sh.score_target(before, after, "y", names)
    assert all(row["reason"] == sh.TOO_MANY_COVARIATES for row in _rows(rows, sh.ADJUSTED))
    assert not any(row["refused"] for row in _rows(rows, sh.RAW))
    rows = sh.score_target(before, after, "y", names[:10])
    assert not any(
        row["reason"] == sh.TOO_MANY_COVARIATES for row in _rows(rows, sh.ADJUSTED)
    )


# ---------------------------------------------------------------- ewc


def test_ewc_terms_follow_the_published_rule():
    assert sh.ewc_terms(120) == 9
    assert sh.ewc_terms(480) == 24
    assert sh.ewc_terms(2) == 1


def test_ewc_long_run_variance_matches_the_cosine_sum():
    rng = np.random.default_rng(16)
    u = rng.standard_normal(50)
    n, nu = 50, sh.ewc_terms(50)
    z = u - u.mean()
    t = np.arange(1, n + 1)
    brute = [
        math.sqrt(2 / n) * sum(math.cos(math.pi * j * (t[i] - 0.5) / n) * z[i] for i in range(n))
        for j in range(1, nu + 1)
    ]
    omega, terms = sh.ewc_long_run_variance(u)
    assert terms == nu
    assert omega == pytest.approx(sum(v * v for v in brute) / nu, rel=1e-10)


def test_ewc_uses_t_critical_values_and_widens_on_persistent_series():
    rng = np.random.default_rng(17)
    before, after = rng.standard_normal(4000), rng.standard_normal(4000)
    naive = sh.level_interval(before, after, sh.NAIVE)
    ewc = sh.level_interval(before, after, sh.EWC)
    assert 0.85 < (ewc.hi - ewc.lo) / (naive.hi - naive.lo) < 1.2
    before, after = _ar1(0.9, 480, 18), _ar1(0.9, 480, 19)
    naive = sh.level_interval(before, after, sh.NAIVE)
    ewc = sh.level_interval(before, after, sh.EWC)
    assert (ewc.hi - ewc.lo) > 2 * (naive.hi - naive.lo)
    # One period with no variation leaves nu of the other as the degrees of freedom.
    flat = np.full(120, 3.0)
    noisy = rng.standard_normal(120)
    interval = sh.level_interval(flat, noisy, sh.EWC)
    omega, nu = sh.ewc_long_run_variance(noisy)
    se = math.sqrt(omega / 120)
    crit = (interval.hi - interval.lo) / (2 * se)
    assert crit == pytest.approx(float(sh.stats.t.ppf(0.975, nu)), rel=1e-9)


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


# ---------------------------------------------------------------- real-shaped beds
#
# Values are integers on short periodic patterns (periods 6 and 4) and every
# period holds a whole number of cycles, so a placebo's before and after
# arrays are identical and its estimate is exactly 0.

PATTERN_A = np.array([0, 2, 5, 1, 3, 4] * 10, dtype=float)
PATTERN_B = np.array([1, 0, 3, 2] * 15, dtype=float)
FROZEN = 7.0
T0 = pd.Timestamp("2015-01-01T00:00:00+00:00")
SKAB_T0 = pd.Timestamp("2020-03-09T10:00:00+00:00")
SKAB_ONSET, SKAB_END, SKAB_ROWS = 960, 1260, 1500
SKAB_GAP_AT, SKAB_GAP_S = 1400, 90
MANIFEST = '{"common_variables": ["P-PDG", "P-TPT", "T-TPT"], "subgroups": 60}\n'


def _minutes(values: np.ndarray) -> dict:
    return {f"s{i:02d}": float(values[i]) for i in range(60)}


def _instance(
    instance: str, well: str, indices: list[int], labels: list[int], pdg_levels: list[float]
) -> tuple[list, list]:
    windows, medians = [], []
    for index, label, level in zip(indices, labels, pdg_levels, strict=True):
        start = T0 + pd.Timedelta(3600 * index, unit="s")
        windows.append(
            {
                "instance": instance,
                "well": well,
                "window_index": index,
                "window_start": start.isoformat(),
                "label": label,
                "design_label": label,
            }
        )
        for variable, values in (
            ("P-PDG", level + PATTERN_A),
            ("P-TPT", 50.0 + PATTERN_B),
            ("T-TPT", np.full(60, FROZEN)),
        ):
            medians.append(
                {"instance": instance, "window_index": index, "variable": variable}
                | _minutes(values)
            )
    return windows, medians


def _write_3w(path: Path, parts: list[tuple[list, list]]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    windows = [w for part in parts for w in part[0]]
    medians = [m for part in parts for m in part[1]]
    pd.DataFrame(windows).to_parquet(path / "windows.parquet", index=False)
    pd.DataFrame(medians).to_parquet(path / "subgroup_medians.parquet", index=False)
    (path / "MANIFEST.json").write_text(MANIFEST, encoding="utf-8", newline="\n")


def build_3w_normal(path: Path) -> None:
    _write_3w(
        path,
        [
            _instance("WELL-00001_a", "WELL-00001", [1, 2, 3, 4, 5], [0] * 5, [100.0] * 5),
            _instance("WELL-00002_b", "WELL-00002", [3, 4, 5, 6], [0] * 4, [100.0] * 4),
            _instance("WELL-00002_fault", "WELL-00002", [1, 2, 3, 4], [0, 0, 0, 1], [100.0] * 4),
            _instance("WELL-00003_short", "WELL-00003", [1, 2, 4, 5], [0] * 4, [100.0] * 4),
        ],
    )


def build_3w_aligned(path: Path) -> None:
    windows, medians = _instance(
        "WELL-00009_step",
        "WELL-00009",
        [1, 2, 3, 4, 6, 7],
        [0, 0, 0, 0, 1, 1],
        [1000.0] * 4 + [2000.0] * 2,
    )
    for row, offset in zip(windows, (-5, -4, -3, -2, 0, 1), strict=True):
        row["onset_offset"] = offset
    _write_3w(path, [(windows, medians)])


def _skab_record(directory: Path, rows: int, step: float, labelled: bool) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    position = np.arange(rows)
    stamps = SKAB_T0 + pd.to_timedelta(position, unit="s")
    if labelled:
        stamps = stamps.where(
            position < SKAB_GAP_AT, stamps + pd.Timedelta(SKAB_GAP_S, unit="s")
        )
    in_span = (position >= SKAB_ONSET) & (position < SKAB_END) & labelled
    columns = {
        "Current": 10.0 + np.resize(PATTERN_A, rows) + step * in_span,
        "Voltage": np.resize(PATTERN_B, rows),
        "Pressure": np.full(rows, FROZEN),
    }
    for tag, values in columns.items():
        pd.DataFrame({"timestamp": stamps, "value": values, "quality": "GOOD"}).to_parquet(
            directory / f"{tag}.parquet", index=False
        )
    if labelled:
        pd.DataFrame(
            {"timestamp": stamps, "anomaly": in_span.astype(int), "changepoint": 0}
        ).to_parquet(directory / "labels.parquet", index=False)


def build_skab(path: Path) -> None:
    _skab_record(path / "valve1__1", SKAB_ROWS, 50.0, labelled=True)
    _skab_record(path / "anomaly-free__anomaly-free", 2500, 0.0, labelled=False)


@pytest.fixture(scope="module")
def real_beds(tmp_path_factory) -> tuple[dict, Path, Path]:
    root = tmp_path_factory.mktemp("shift_intervals")
    build_3w_normal(root / "windows")
    build_3w_aligned(root / "aligned")
    build_skab(root / "archives")
    kwargs = {
        "windows": root / "windows",
        "aligned": root / "aligned",
        "archives": root / "archives",
        "source_3w": None,
        "source_skab": None,
    }
    beds = (rs.BED_PLACEBO, rs.BED_KNOWN)
    info = rs.run(root / "a", beds, **kwargs)
    rs.run(root / "b", beds, **kwargs)
    return info, root / "a", root / "b"


def test_first_run_finds_the_first_block_of_consecutive_windows():
    assert rs.first_run([1, 2, 3, 5, 6, 7, 8], 4) == 3
    assert rs.first_run([1, 2, 3, 4], 4) == 0
    assert rs.first_run([1, 2, 4, 5], 4) is None


def test_placebo_bed_scores_the_normal_instances_only(real_beds):
    info, out, _ = real_beds
    counts = info["beds"][rs.BED_PLACEBO]["3w_cache"]
    assert counts["n_instances_no_fault_window"] == 3
    assert counts["n_instances_scored"] == 2
    assert counts["n_instances_too_short"] == 1
    placebo = rs.read_rows(out / rs.PLACEBO_CSV)
    assert set(placebo.loc[placebo["design"] == rs.DESIGN_3W_PLACEBO, "record"]) == {
        "WELL-00001_a",
        "WELL-00002_b",
    }
    designs = info["beds"][rs.BED_PLACEBO]["designs"]
    assert designs[rs.DESIGN_SKAB_FREE]["n_splits"] == 2
    assert designs[rs.DESIGN_SKAB_PRE]["n_splits"] == 1
    first = placebo[placebo["record"] == "WELL-00001_a"].iloc[0]
    assert first["n_before"] == 120 and first["n_after"] == 120


def test_a_periodic_placebo_does_not_clear(real_beds):
    _, out, _ = real_beds
    placebo = rs.read_rows(out / rs.PLACEBO_CSV)
    answered = placebo[placebo["refused"] == 0]
    assert len(answered) > 0
    assert (answered["clears"] == 0).all()
    assert (answered["estimate"] == 0).all()


def test_the_step_clears(real_beds):
    info, out, _ = real_beds
    known = rs.read_rows(out / rs.KNOWN_CSV)
    step = known[
        (known["record"] == "WELL-00009_step")
        & (known["tag"] == "P-PDG")
        & (known["quantity"] == sh.LEVEL)
    ]
    assert len(step) == 2 * len(sh.LEVEL_METHODS)
    assert (step["refused"] == 0).all()
    assert (step["clears"] == 1).all()
    adjusted = step[step["adjustment"] == sh.ADJUSTED]
    assert adjusted["variance_ratio_after"].notna().all()
    assert step.loc[step["adjustment"] == sh.RAW, "variance_ratio_after"].isna().all()
    assert set(step["n_before"]) == {180} and set(step["n_after"]) == {120}
    skab = known[
        (known["design"] == rs.DESIGN_SKAB_ONSET)
        & (known["tag"] == "Current")
        & (known["quantity"] == sh.LEVEL)
        & (known["adjustment"] == sh.RAW)
    ]
    assert (skab["clears"] == 1).all()
    assert set(skab["n_before"]) == {SKAB_ONSET}
    assert set(skab["n_after"]) == {SKAB_END - SKAB_ONSET}
    counts = info["beds"][rs.BED_KNOWN]["3w_cache"]
    assert counts["before_onset_offsets"] == [-4, -3, -2]
    assert counts["after_onset_offsets"] == [0, 1]


def test_refused_rows_carry_their_reason(real_beds):
    info, out, _ = real_beds
    known = rs.read_rows(out / rs.KNOWN_CSV)
    frozen = known[known["tag"].isin(["T-TPT", "Pressure"])]
    assert len(frozen) > 0
    assert (frozen["refused"] == 1).all()
    assert set(frozen["reason"]) == {sh.NO_SPREAD}
    moved_covariate = known[
        (known["record"] == "WELL-00009_step")
        & (known["tag"] == "P-TPT")
        & (known["adjustment"] == sh.ADJUSTED)
    ]
    assert set(moved_covariate["reason"]) == {sh.COVARIATE_OUTSIDE}
    assert (moved_covariate["covariate_outside"] == 1).all()
    assert known.loc[known["refused"] == 0, "reason"].isna().all()
    gaps = info["beds"][rs.BED_KNOWN]["skab_gaps_over_60_s"]
    assert gaps == [
        {
            "record": "valve1__1",
            "rows": SKAB_ROWS,
            "n_gaps": 1,
            "longest_gap_s": SKAB_GAP_S + 1.0,
        }
    ]


def test_summary_reads_the_rows(real_beds):
    _, out, _ = real_beds
    summary = pd.read_csv(out / rs.SUMMARY_CSV)
    placebo = summary[
        (summary["design"] == rs.DESIGN_3W_PLACEBO)
        & (summary["quantity"] == sh.LEVEL)
        & (summary["method"] == sh.HAC)
        & (summary["refusal_set"] == "base")
    ]
    assert set(placebo["adjustment"]) == {sh.RAW, sh.ADJUSTED}
    assert (placebo["clear_rate_pooled"] == 0).all()
    assert (placebo["record_clear_rate"] == 0).all()
    assert (placebo["n_groups"] == 2).all()
    onset = summary[
        (summary["design"] == rs.DESIGN_3W_ONSET)
        & (summary["quantity"] == sh.LEVEL)
        & (summary["adjustment"] == sh.RAW)
        & (summary["refusal_set"] == "base")
    ]
    assert (onset["record_clear_rate"] == 1).all()
    raw_sets = summary.loc[summary["adjustment"] == sh.RAW, "refusal_set"]
    assert set(raw_sets) == set(sh.RAW_REFUSAL_SETS)


def test_a_rerun_writes_identical_csvs(real_beds):
    _, first, second = real_beds
    for name in (rs.PLACEBO_CSV, rs.KNOWN_CSV, rs.SUMMARY_CSV):
        assert filecmp.cmp(first / name, second / name, shallow=False), name


def test_run_json_carries_no_machine_path(real_beds):
    _, out, _ = real_beds
    text = (out / "run.json").read_text("utf-8")
    for fragment in (":\\\\", ":/", "Users", "tmp"):
        assert fragment not in text, fragment


# ---------------------------------------------------------------- turbine-shaped bed
#
# Every column repeats with a period that divides one week of 10 min rows
# (1008), so two placebo weeks hold identical arrays and the placebo
# estimate is exactly 0.

WEEK_ROWS = 1008
TURBINE_T0 = pd.Timestamp("2010-04-29 06:20")
PITCH_AFTER = 600


def _periodic(values: list[float], rows: int) -> np.ndarray:
    return np.resize(np.array(values, dtype=float), rows)


def _turbine_frame(rows: int) -> pd.DataFrame:
    v = _periodic([4.0, 6.0, 8.0, 10.0, 12.0, 7.0], rows)
    frame = pd.DataFrame(
        {
            "V": v,
            "D": _periodic([100.0, 140.0, 180.0, 220.0], rows),
            "rho": _periodic([1.1, 1.2, 1.15], rows),
            "S": _periodic([0.2, 0.4, 0.3, 0.5], rows),
            "I": _periodic([0.05, 0.09, 0.07, 0.06, 0.08, 0.04, 0.1], rows),
            "VcosD": _periodic([1.0, -2.0, 3.0, 0.5, -1.5, 2.5, -0.5, 1.5, 0.0], rows),
            "VsinD": _periodic([2.0, 0.0, -1.0, 1.0, 3.0, -2.0] * 2, rows),
        }
    )
    pattern = _periodic([0.0, 0.03, 0.01, 0.05, 0.02, 0.04, 0.06, 0.0, 0.02] * 2, rows)
    frame["y_test"] = 0.08 * v + pattern
    frame["y_ctrl"] = 0.08 * v + _periodic([0.01, 0.0, 0.03, 0.02] * 3 + [0.0, 0.01], rows)
    return frame


def _stamps(rows: int, gap_at: int | None = None) -> pd.Series:
    step = pd.to_timedelta(np.arange(rows) * 10, unit="min")
    stamps = pd.Series(TURBINE_T0 + step)
    if gap_at is not None:
        stamps[gap_at:] = stamps[gap_at:] + pd.Timedelta(2, unit="D")
    return stamps


def _write_pair(path: Path, frame: pd.DataFrame, stamps: pd.Series, status, names) -> None:
    status_name, y_name, c_name = names
    out = pd.DataFrame(
        {
            "time": stamps.dt.strftime("%m/%d/%Y %H:%M"),
            status_name: status,
            "V": frame["V"],
            "D": frame["D"],
            "rho": frame["rho"],
            "S": frame["S"],
            "I": frame["I"],
            "VcosD": frame["VcosD"],
            "VsinD": frame["VsinD"],
            y_name: frame["y_test"],
            c_name: frame["y_ctrl"],
        }
    )
    out.index = range(1, len(out) + 1)
    out.to_csv(path, lineterminator="\n")


def build_turbine(root: Path) -> None:
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    vg_rows = 4 * WEEK_ROWS + 500
    vg = _turbine_frame(vg_rows)
    vg_status = (np.arange(vg_rows) >= 4 * WEEK_ROWS).astype(int)
    _write_pair(
        raw / "Turbine Upgrade Dataset(VG Pair).csv",
        vg,
        _stamps(vg_rows),
        vg_status,
        ("upgrade status", "y_test (normalized)", "y_ctrl (normalized)"),
    )
    pitch_rows = 2 * WEEK_ROWS + PITCH_AFTER
    pitch = _turbine_frame(pitch_rows)
    upgraded = np.arange(pitch_rows) >= 2 * WEEK_ROWS
    windy = upgraded & (pitch["V"].to_numpy() > 9.0)
    base = pitch["y_test"].to_numpy().copy()
    table = {}
    for r in rs.TURBINE_TABLE_R:
        table[f"y_test(r={r:.2f}, normalized)"] = np.where(windy, base * (1 + r), base)
    main = pitch.assign(y_test=table["y_test(r=0.05, normalized)"])
    stamps = _stamps(pitch_rows, gap_at=WEEK_ROWS + 3)
    _write_pair(
        raw / "Turbine Upgrade Dataset(Pitch Angle Pair).csv",
        main,
        stamps,
        upgraded.astype(int),
        ("upgrade status", "y_test(normalized)", "y_ctrl(normalized)"),
    )
    t73 = pd.DataFrame(
        {
            "time": stamps.dt.strftime("%m/%d/%Y %H:%M"),
            "upgrade.status": upgraded.astype(int),
            **{c: pitch[c] for c in ("V", "D", "rho", "S", "I", "VcosD", "VsinD")},
            **table,
            "y_ctrl(normalized)": pitch["y_ctrl"],
        }
    )
    t73.to_csv(
        raw / "Turbine Upgrade Dataset(Pitch Angle Pair, Table7.3).csv",
        index=False,
        lineterminator="\n",
    )


@pytest.fixture(scope="module")
def turbine_bed(tmp_path_factory) -> tuple[dict, Path, Path]:
    root = tmp_path_factory.mktemp("turbine")
    build_turbine(root / "turbine")
    info = rs.run(root / "a", (rs.BED_TURBINE,), turbine=root / "turbine")
    rs.run(root / "b", (rs.BED_TURBINE,), turbine=root / "turbine")
    return info, root / "a", root / "b"


def test_turbine_loader_reads_the_published_headers(turbine_bed):
    info, _, _ = turbine_bed
    dataset = info["beds"][rs.BED_TURBINE]["dataset"]
    table = dataset["pitch_table"]
    assert table["modified_rows_match_upgraded_and_v_over_9"]
    assert table["main_file_y_test_equals_r_0.05"]
    assert table["n_rows_modified"] == 200  # V in {10, 12}: 2 of every 6 upgraded rows
    vg = dataset["pairs"][rs.PAIR_VG]
    assert vg["n_rows"] == 4 * WEEK_ROWS + 500
    assert vg["n_upgraded_rows"] == 500
    assert dataset["pairs"][rs.PAIR_PITCH]["steps_minutes"]["over_1440"] == 1


def test_turbine_placebo_weeks_do_not_clear_and_injection_carries_its_truth(turbine_bed):
    info, out, _ = turbine_bed
    rows = rs.read_rows(out / rs.TURBINE_CSV)
    designs = info["beds"][rs.BED_TURBINE]["designs"]
    assert designs[rs.DESIGN_TURBINE_PLACEBO]["n_splits"] == 3  # 2 VG pairs, 1 pitch pair
    level = rows[rows["quantity"] == sh.LEVEL]
    placebo = level[(level["design"] == rs.DESIGN_TURBINE_PLACEBO) & (level["refused"] == 0)]
    assert len(placebo) > 0
    assert (placebo["estimate"] == 0).all()
    assert (placebo["clears"] == 0).all()
    assert (placebo["truth"] == 0).all()
    injected = level[level["design"] == rs.DESIGN_TURBINE_INJECTED.format(r=0.09)]
    answered = injected[injected["refused"] == 0]
    assert len(answered) > 0
    assert (answered["truth"] > 0).all()
    assert (answered["estimate"] - answered["truth"]).abs().max() < 1e-3
    hac = answered[answered["method"] == sh.HAC]
    assert (hac["clears"] == 1).all()


def test_turbine_pitch_change_scores_each_published_r(turbine_bed):
    _, out, _ = turbine_bed
    rows = rs.read_rows(out / rs.TURBINE_CSV)
    pitch = rows[
        (rows["design"] == rs.DESIGN_TURBINE_PITCH)
        & (rows["quantity"] == sh.LEVEL)
        & (rows["method"] == sh.HAC)
        & (rows["adjustment"] == sh.RAW)
    ]
    assert len(pitch) == 1 + len(rs.TURBINE_TABLE_R)
    assert set(pitch["n_before"]) == {PITCH_AFTER} and set(pitch["n_after"]) == {PITCH_AFTER}
    truth = pitch.set_index("record")["truth"]
    assert truth["pitch_pair_r0.00"] == 0
    assert truth["pitch_pair_r0.02"] < truth["pitch_pair_r0.05"] < truth["pitch_pair_r0.09"]
    vg = rows[(rows["design"] == rs.DESIGN_TURBINE_VG) & (rows["quantity"] == sh.LEVEL)]
    assert vg["truth"].isna().all()
    assert set(vg["n_after"]) == {500}


def test_turbine_rerun_writes_identical_csvs(turbine_bed):
    _, first, second = turbine_bed
    for name in (rs.TURBINE_CSV, rs.SUMMARY_CSV):
        assert filecmp.cmp(first / name, second / name, shallow=False), name
    summary = pd.read_csv(first / rs.SUMMARY_CSV)
    injected = summary[
        (summary["design"] == rs.DESIGN_TURBINE_INJECTED.format(r=0.09))
        & (summary["quantity"] == sh.LEVEL)
        & (summary["method"] == sh.HAC)
        & (summary["adjustment"] == sh.RAW)
        & (summary["refusal_set"] == "base")
    ]
    assert injected["coverage"].notna().all()


# ---------------------------------------------------------------- TEP-shaped bed
TEP_RUNS = 12


def _tep_frame(faults: list[int], runs: int, samples: int, seed: int) -> pd.DataFrame:
    """White-noise runs; fault 1 steps xmeas_1 by 3 after sample 160, fault 6 zeroes it."""
    rng = np.random.default_rng(seed)
    parts = []
    for fault in faults:
        for run in range(1, runs + 1):
            sample = np.arange(1, samples + 1)
            values = rng.standard_normal((samples, len(btc.VARIABLES)))
            if fault == 1:
                values[sample > 160, 0] += 3.0
            if fault == 6:
                values[sample > 160, 0] = 0.0
            frame = pd.DataFrame(values, columns=btc.VARIABLES)
            frame.insert(0, "sample", sample)
            frame.insert(0, "simulationRun", run)
            frame.insert(0, "faultNumber", fault)
            parts.append(frame)
    return pd.concat(parts, ignore_index=True)


def test_tep_cache_builder_keeps_runs_and_samples(tmp_path):
    pyreadr = pytest.importorskip("pyreadr")
    source = tmp_path / "tep"
    source.mkdir()
    pyreadr.write_rdata(
        str(source / btc.FILES["fault_free"]), _tep_frame([0], 3, 330, 1), df_name="fault_free"
    )
    pyreadr.write_rdata(
        str(source / btc.FILES["faulty"]), _tep_frame([1, 6], 3, 330, 2), df_name="faulty"
    )
    (source / "MANIFEST.sha256.json").write_text("{}\n", encoding="utf-8")
    manifest = btc.build(source, tmp_path / "cache", runs=2)
    assert manifest["n_rows"] == 3 * 2 * btc.LAST_SAMPLE
    assert manifest["faults"] == [0, 1, 6]
    assert manifest["non_finite_values"] == 0
    onset = manifest["onset_check"]
    assert onset["fault_mean"][onset["samples"].index(161)] == 0.0
    cache = pd.read_parquet(tmp_path / "cache" / btc.CACHE_NAME)
    assert cache["run"].max() == 2 and cache["sample"].max() == btc.LAST_SAMPLE
    assert cache["xmeas_1"].dtype == np.float32


@pytest.fixture(scope="module")
def tep_bed(tmp_path_factory) -> tuple[dict, Path, Path, Path]:
    root = tmp_path_factory.mktemp("tep")
    cache = root / "cache"
    cache.mkdir()
    frame = _tep_frame([0, 1], TEP_RUNS, 320, 3)
    frame = frame.rename(columns={"faultNumber": "fault", "simulationRun": "run"})
    frame.to_parquet(cache / rs.TEP_CACHE_FILE, index=False)
    (cache / "MANIFEST.json").write_text(
        '{"source_manifest_sha256": "0123456789ab", "cache_sha256": "ba9876543210", '
        '"runs": [1, 12], "samples": [1, 320]}\n',
        encoding="utf-8",
    )
    patch = pytest.MonkeyPatch()
    patch.setattr(rs, "TEP_TRUTH_RUNS", 8)
    patch.setattr(rs, "TEP_SCORED_RUNS", (9, TEP_RUNS))
    try:
        info = rs.run(root / "a", (rs.BED_TEP,), tep_cache=cache)
        rs.run(root / "b", (rs.BED_TEP,), tep_cache=cache)
    finally:
        patch.undo()
    return info, root / "a", root / "b", cache


def test_tep_bed_scores_the_step_against_its_ensemble_truth(tep_bed):
    info, out, _, cache = tep_bed
    table = pd.read_csv(out / rs.TEP_CSV)
    assert set(table["condition"]) == {"fault_free", "fault_1"}
    assert set(table["variable_group"]) == set(rs.TEP_GROUPS)
    base = table[(table["refusal_set"] == "base") & (table["quantity"] == sh.LEVEL)]
    free = base[base["condition"] == "fault_free"]
    assert free["claim_rate"].notna().all() and free["detect_rate"].isna().all()
    rows = pd.read_parquet(cache / rs.TEP_ROWS_FILE)
    step = rows[
        (rows["fault"] == 1)
        & (rows["tag"] == "xmeas_1")
        & (rows["quantity"] == sh.LEVEL)
        & (rows["adjustment"] == sh.RAW)
    ]
    assert (step["clears"] == 1).all()
    level_truth = step["truth"] * step["scale"]
    assert ((level_truth - 3.0).abs() < 0.2).all()
    fault_free = rows[(rows["fault"] == 0) & (rows["quantity"] == sh.LEVEL)]
    assert (fault_free["truth"].fillna(0) == 0).all()
    spread = rows[(rows["fault"] == 0) & (rows["quantity"] == sh.SPREAD)]
    assert spread["truth"].notna().all() and (spread["truth"] != 0).any()
    adjusted = rows[rows["adjustment"] == sh.ADJUSTED]
    assert set(adjusted["reason"]) == {sh.TOO_MANY_COVARIATES}
    section = info["beds"][rs.BED_TEP]
    assert section["adjusted_refusals"] == {sh.TOO_MANY_COVARIATES: len(adjusted)}
    assert section["truth_runs"] == [1, 8] and section["scored_runs"] == [9, TEP_RUNS]


def test_tep_rerun_writes_an_identical_table(tep_bed):
    _, first, second, _ = tep_bed
    assert filecmp.cmp(first / rs.TEP_CSV, second / rs.TEP_CSV, shallow=False)

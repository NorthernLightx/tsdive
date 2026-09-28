"""The switchback schedule study on synthetic inputs: no network, no real data."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

ROOT = Path(__file__).parents[1]
STUDY = ROOT / "examples" / "studies" / "switchback"
sys.path.insert(0, str(STUDY))

import switchback as sb  # noqa: E402

sh = sb.sh


def _regular(t: int, length: int) -> tuple[np.ndarray, sb.Blocks]:
    times = np.arange(t, dtype=float)
    return times, sb.cut_blocks(times, t, length)


def _fit(y, x, blocks, washout, seed):
    frame = sb.make_frame(y, x, blocks, washout)
    design = sb.make_design(blocks.k, seed)
    terms = sb.assignment_terms(frame, design)
    return frame, design, terms


# ---------------------------------------------------------------- schedule


@pytest.mark.parametrize("k", [8, 16])
def test_balanced_schedule_is_deterministic_and_balanced(k):
    first, second = sb.make_design(k, 11), sb.make_design(k, 11)
    assert np.array_equal(first.observed, second.observed)
    assert np.array_equal(first.assignments, second.assignments)
    assert first.observed.sum() == k // 2
    assert np.all(first.assignments.sum(axis=1) == k // 2)
    other = sb.make_design(k, 12)
    assert not (
        np.array_equal(first.observed, other.observed)
        and np.array_equal(first.assignments, other.assignments)
    )


def test_odd_block_count_drops_the_last_block():
    _, blocks = _regular(75, 15)
    assert blocks.k == 4
    assert np.all(blocks.block[60:] == -1)
    assert blocks.block[59] == 3 and blocks.offset[59] == 14


def test_enumeration_is_used_up_to_1000_assignments():
    twelve = sb.make_design(12, 3)
    assert twelve.enumerated and twelve.n_assignments == 924
    assert len(twelve.assignments) == 924
    assert len({row.tobytes() for row in twelve.assignments}) == 924
    assert any(np.array_equal(row, twelve.observed) for row in twelve.assignments)
    fourteen = sb.make_design(14, 3)
    assert not fourteen.enumerated and fourteen.n_assignments == 3432
    assert len(fourteen.assignments) == sb.PERMUTATIONS


def test_design_too_small_fires_at_four_and_six_blocks():
    assert sb.design_refusal(4) == sb.DESIGN_TOO_SMALL  # 6 assignments
    assert sb.design_refusal(6) == sb.DESIGN_TOO_SMALL  # 20 assignments, p >= 0.1
    assert sb.design_refusal(8) == ""  # 70 assignments, p >= 2/70
    with pytest.raises(ValueError, match=sb.DESIGN_TOO_SMALL):
        sb.make_design(4, 0)


# ---------------------------------------------------------------- randomization


def test_randomization_p_is_near_uniform_on_white_noise():
    rng = np.random.default_rng(5)
    _, blocks = _regular(240, 15)
    p = []
    for rep in range(400):
        y = rng.standard_normal(240)
        frame, design, terms = _fit(y, None, blocks, 0, rep)
        p.append(sb.randomization(design, terms, frame.resid_sums[:, None], 1.0).p_value[0])
    p = np.array(p)
    assert 0.02 <= np.mean(p <= 0.05) <= 0.09
    assert 0.42 <= p.mean() <= 0.58
    assert stats.kstest(p, "uniform").pvalue > 0.01


def test_raw_estimate_is_the_difference_in_means():
    rng = np.random.default_rng(6)
    _, blocks = _regular(240, 30)
    y = rng.standard_normal(240)
    frame, design, terms = _fit(y, None, blocks, 4, 1)
    est = sb.randomization(design, terms, frame.resid_sums[:, None], 1.0).estimate[0]
    z = design.observed[frame.block]
    assert est == pytest.approx(frame.y[z == 1].mean() - frame.y[z == 0].mean(), abs=1e-12)


@pytest.mark.parametrize("length", [15, 30])
def test_inverted_interval_holds_the_shift_and_ends_where_p_crosses(length):
    rng = np.random.default_rng(8)
    times, blocks = _regular(240, length)
    base = rng.standard_normal(240)
    design = sb.make_design(blocks.k, 21)
    shift = 1.5
    y = base + shift * sb.lag_response(sb.setting(blocks, design.observed), times, 0.0)
    frame = sb.make_frame(y, None, blocks, 0)
    terms = sb.assignment_terms(frame, design)
    result = sb.randomization(design, terms, frame.resid_sums[:, None], 1.0)
    lo, hi = result.lo[0], result.hi[0]
    assert lo <= shift <= hi
    z = terms.z_kept

    def p_at(theta: float) -> float:
        return sb.randomization_p(frame, design, frame.y - theta * z, 1.0)

    eps = 1e-6 * (hi - lo)
    assert p_at(lo - eps) <= sb.ALPHA < p_at(lo + eps)
    assert p_at(hi + eps) <= sb.ALPHA < p_at(hi - eps)
    assert p_at(result.estimate[0]) == 1.0


def test_frisch_waugh_adjusted_equals_the_direct_ols_coefficient():
    rng = np.random.default_rng(9)
    _, blocks = _regular(240, 15)
    x = rng.standard_normal((240, 3))
    x[:, 2] = x[:, 0] + x[:, 1]  # a repeated direction is dropped, not inverted
    y = x[:, :2] @ [1.0, -0.5] + rng.standard_normal(240)
    frame, design, terms = _fit(y, x, blocks, 5, 2)
    result = sb.randomization(design, terms, frame.resid_sums[:, None], 1.0)
    xk = x[frame.positions]

    def direct(row: np.ndarray) -> float:
        z = row[frame.block].astype(float)
        design_matrix = np.column_stack([np.ones(frame.n), z, xk])
        return float(np.linalg.lstsq(design_matrix, frame.y, rcond=None)[0][1])

    assert result.estimate[0] == pytest.approx(direct(design.observed), abs=1e-10)
    for j in range(5):
        fwl = (design.matrix[j] @ frame.resid_sums) / terms.denom[j]
        assert fwl == pytest.approx(direct(design.assignments[j]), abs=1e-10)
    arms = sb.sample_arms(frame, terms, frame.resid[:, None])
    for arm in (sb.NAIVE, sb.HAC, sb.EWC):
        assert arms[arm].estimate[0] == pytest.approx(result.estimate[0], abs=1e-12)


def test_too_many_covariates_and_too_few_samples_refuse_the_frame():
    rng = np.random.default_rng(10)
    _, blocks = _regular(240, 30)
    y = rng.standard_normal(240)
    assert sb.make_frame(y, rng.standard_normal((240, 6)), blocks, 23) == sb.TOO_MANY_COVARIATES
    assert sb.make_frame(y, None, blocks, 27) == sb.TOO_FEW
    assert sb.make_frame(np.full(240, 2.0), None, blocks, 0) == sb.NO_SPREAD


# ---------------------------------------------------------------- response and washout


def test_first_order_response_reaches_one_minus_exp_minus_three_at_three_tau():
    tau = 5.0
    times = np.arange(40, dtype=float)
    s = sb.lag_response(np.ones(40), times, tau)
    assert s[int(3 * tau) - 1] == pytest.approx(1 - math.exp(-3), abs=1e-12)
    assert np.all(np.diff(s) > 0)
    assert np.array_equal(sb.lag_response(np.ones(40), times, 0.0), np.ones(40))
    assert sb.washout_steps(tau) == 15 and sb.washout_steps(1.5) == 5
    assert sb.washout_steps(3.75) == 12 and sb.washout_steps(0.0) == 0


def test_a_gap_moves_the_response_by_its_length():
    s = sb.lag_response(np.ones(3), np.array([0.0, 1.0, 11.0]), 2.0)
    assert s[2] == pytest.approx(1 - math.exp(-1 / 2) * math.exp(-1 / 2) * math.exp(-10 / 2))


def test_washout_removes_the_carryover_bias():
    rng = np.random.default_rng(12)
    times, blocks = _regular(480, 30)
    tau = 7.5
    biases = {}
    for washout in (0, sb.washout_steps(tau)):
        errors = []
        for seed in range(20):
            design = sb.make_design(blocks.k, seed)
            s = sb.lag_response(sb.setting(blocks, design.observed), times, tau)
            y = s + 1e-6 * rng.standard_normal(480)
            frame = sb.make_frame(y, None, blocks, washout)
            terms = sb.assignment_terms(frame, design)
            est = sb.randomization(design, terms, frame.resid_sums[:, None], 1.0).estimate[0]
            errors.append(est - 1.0)
        biases[washout] = np.median(errors)
    assert biases[0] < -0.2
    assert abs(biases[sb.washout_steps(tau)]) < math.exp(-3)


# ---------------------------------------------------------------- other arms


def test_ewc_through_the_dct_matches_the_shift_study_estimator():
    rng = np.random.default_rng(13)
    for n in (60, 300, 961):
        u = rng.standard_normal(n).cumsum() * 0.1 + rng.standard_normal(n)
        omega, nu = sb.ewc_long_run_variance(u[:, None])
        ref, ref_nu = sh.ewc_long_run_variance(u)
        assert nu == ref_nu
        assert omega[0] == pytest.approx(ref, rel=1e-10)


def test_block_t_is_welch_on_the_block_means():
    rng = np.random.default_rng(14)
    _, blocks = _regular(240, 15)
    y = rng.standard_normal(240)
    frame, design, terms = _fit(y, None, blocks, 3, 4)
    result = sb.sample_arms(frame, terms, frame.resid[:, None])[sb.BLOCK_T]
    means = np.array([frame.y[frame.block == b].mean() for b in range(blocks.k)])
    b, a = means[design.observed == 1], means[design.observed == 0]
    welch = stats.ttest_ind(b, a, equal_var=False)
    assert result.estimate[0] == pytest.approx(b.mean() - a.mean(), abs=1e-12)
    assert result.p_value[0] == pytest.approx(welch.pvalue, rel=1e-9)
    ci = welch.confidence_interval(0.95)
    assert (result.lo[0], result.hi[0]) == pytest.approx((ci.low, ci.high), rel=1e-9)


def test_prepost_reads_the_shift_study_interval():
    rng = np.random.default_rng(15)
    before, after = rng.standard_normal(120), rng.standard_normal(120) + 0.3
    got = sb.prepost(before, after, sh.HAC)
    ref = sh.level_interval(before, after, sh.HAC)
    assert got == (ref.estimate, ref.lo, ref.hi, ref.p_value)
    assert sb.prepost(np.full(120, 1.0), after, sh.EWC) == sb.NO_SPREAD
    x_b, x_a = rng.standard_normal((120, 2)), rng.standard_normal((120, 2)) + 10.0
    assert sb.prepost(before, after, sh.HAC, x_b, x_a) == sb.COVARIATE_OUTSIDE

"""Switchback schedules, randomization inference and plans, on synthetic arrays."""

from __future__ import annotations

import json
import math
import re
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from tsdive.errors import DesignTooSmall, ScheduleMismatch
from tsdive.switchback import design as sd
from tsdive.switchback import inference as si
from tsdive.switchback import plan as sp

START = pd.Timestamp("2024-03-04 00:00:00+00:00")


def _regular(t: int, length: int) -> tuple[np.ndarray, sd.Blocks]:
    times = np.arange(t, dtype=float)
    return times, sd.cut_blocks(times, t, length)


def _fit(y, x, blocks, washout, seed):
    frame = si.make_frame(y, x, blocks, washout)
    design = sd.make_design(blocks.k, seed)
    terms = si.assignment_terms(frame, design)
    return frame, design, terms


def _plan(**kw) -> sp.SwitchbackPlan:
    args = {
        "start": START,
        "end": START + pd.Timedelta(16, unit="h"),
        "block_s": 3600,
        "washout_s": 600,
        "seed": 42,
    }
    args.update(kw)
    return sp.make_plan(**args)


def _resigned(plan: sp.SwitchbackPlan, blocks) -> sp.SwitchbackPlan:
    """``plan`` with other blocks and a digest recomputed over them."""
    edited = replace(plan, blocks=tuple(blocks))
    return replace(edited, digest=sp.plan_digest(edited))


# ---------------------------------------------------------------- schedule


@pytest.mark.parametrize("k", [8, 16])
def test_balanced_schedule_is_deterministic_and_balanced(k):
    first, second = sd.make_design(k, 11), sd.make_design(k, 11)
    assert np.array_equal(first.observed, second.observed)
    assert np.array_equal(first.assignments, second.assignments)
    assert first.observed.sum() == k // 2
    assert np.all(first.assignments.sum(axis=1) == k // 2)
    other = sd.make_design(k, 12)
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
    twelve = sd.make_design(12, 3)
    assert twelve.enumerated and twelve.n_assignments == 924
    assert len(twelve.assignments) == 924
    assert len({row.tobytes() for row in twelve.assignments}) == 924
    assert any(np.array_equal(row, twelve.observed) for row in twelve.assignments)
    fourteen = sd.make_design(14, 3)
    assert not fourteen.enumerated and fourteen.n_assignments == 3432
    assert len(fourteen.assignments) == sd.PERMUTATIONS


def test_design_too_small_fires_at_four_and_six_blocks():
    """SCOPE claim: a design too small for a 5% randomization test raises DesignTooSmall."""
    assert sd.design_refusal(4) == sd.DESIGN_TOO_SMALL  # 6 assignments
    assert sd.design_refusal(6) == sd.DESIGN_TOO_SMALL  # 20 assignments, p >= 0.1
    assert sd.design_refusal(8) == ""  # 70 assignments, p >= 2/70
    with pytest.raises(DesignTooSmall, match="4 blocks give 6 balanced assignments"):
        sd.make_design(4, 0)


# ---------------------------------------------------------------- randomization


def test_randomization_p_is_near_uniform_on_white_noise():
    rng = np.random.default_rng(5)
    _, blocks = _regular(240, 15)
    p = []
    for rep in range(400):
        y = rng.standard_normal(240)
        frame, design, terms = _fit(y, None, blocks, 0, rep)
        p.append(si.randomization(design, terms, frame.resid_sums[:, None], 1.0).p_value[0])
    p = np.sort(np.array(p))
    assert 0.02 <= np.mean(p <= 0.05) <= 0.09
    assert 0.42 <= p.mean() <= 0.58
    # Kolmogorov-Smirnov distance to the uniform, against its 1% critical value.
    grid = np.arange(1, len(p) + 1) / len(p)
    ks = max(np.max(grid - p), np.max(p - (grid - 1 / len(p))))
    assert ks < 1.63 / math.sqrt(len(p))


@pytest.mark.parametrize("washout", [0, 3])
def test_an_enumerated_design_rejects_at_most_alpha_of_its_assignments(washout):
    """SCOPE claim: under the declared randomization the test rejects at most 5% of assignments.

    Over every assignment as the observed one, p <= 0.05 holds for
    floor(0.05 C) of them, with and without covariates.
    """
    rng = np.random.default_rng(7)
    _, blocks = _regular(240, 24)
    y = rng.standard_normal(240).cumsum()  # a drifting record: exactness needs no stationarity
    x = rng.standard_normal((240, 2)) + y[:, None] * 0.1
    for covariates in (None, x):
        frame = si.make_frame(y, covariates, blocks, washout)
        rows = sd.all_assignments(blocks.k)
        rejected = 0
        for index in range(len(rows)):
            design = sd.Design(
                blocks.k, rows[index].copy(), rows, rows.astype(float), True, len(rows), 0.0
            )
            terms = si.assignment_terms(frame, design)
            p = si.randomization(design, terms, frame.resid_sums[:, None], 1.0).p_value[0]
            rejected += p <= sd.ALPHA
        assert rejected == math.floor(sd.ALPHA * len(rows))


def test_exact_rejection_counts_what_the_test_rejects_over_every_assignment():
    rng = np.random.default_rng(17)
    _, blocks = _regular(240, 30)
    y = np.round(rng.standard_normal(240).cumsum(), 1)  # rounded: ties between assignments
    x = rng.standard_normal((240, 2)) + y[:, None] * 0.2
    rows = sd.all_assignments(blocks.k)
    for covariates in (None, x):
        frame = si.make_frame(y, covariates, blocks, 0)
        rejected = 0
        for index in range(len(rows)):
            design = sd.Design(
                blocks.k, rows[index].copy(), rows, rows.astype(float), True, len(rows), 0.0
            )
            terms = si.assignment_terms(frame, design)
            p = si.randomization(design, terms, frame.resid_sums[:, None], 1.0).p_value[0]
            rejected += p <= sd.ALPHA
        assert si.exact_rejection(frame, blocks.k, 1.0) == pytest.approx(rejected / len(rows))
        assert si.exact_rejection(frame, blocks.k, 1.0) <= math.floor(0.05 * 70) / 70


def test_raw_estimate_is_the_difference_in_means():
    rng = np.random.default_rng(6)
    _, blocks = _regular(240, 30)
    y = rng.standard_normal(240)
    frame, design, terms = _fit(y, None, blocks, 4, 1)
    est = si.randomization(design, terms, frame.resid_sums[:, None], 1.0).estimate[0]
    z = design.observed[frame.block]
    assert est == pytest.approx(frame.y[z == 1].mean() - frame.y[z == 0].mean(), abs=1e-12)


@pytest.mark.parametrize("length", [15, 30])
def test_inverted_interval_holds_the_shift_and_ends_where_p_crosses(length):
    rng = np.random.default_rng(8)
    times, blocks = _regular(240, length)
    base = rng.standard_normal(240)
    design = sd.make_design(blocks.k, 21)
    shift = 1.5
    y = base + shift * si.lag_response(sd.setting(blocks, design.observed), times, 0.0)
    frame = si.make_frame(y, None, blocks, 0)
    terms = si.assignment_terms(frame, design)
    result = si.randomization(design, terms, frame.resid_sums[:, None], 1.0)
    lo, hi = result.lo[0], result.hi[0]
    assert lo <= shift <= hi
    z = terms.z_kept

    def p_at(theta: float) -> float:
        return si.randomization_p(frame, design, frame.y - theta * z, 1.0)

    eps = 1e-6 * (hi - lo)
    assert p_at(lo - eps) <= sd.ALPHA < p_at(lo + eps)
    assert p_at(hi + eps) <= sd.ALPHA < p_at(hi - eps)
    assert p_at(result.estimate[0]) == 1.0


def test_frisch_waugh_adjusted_equals_the_direct_ols_coefficient():
    rng = np.random.default_rng(9)
    _, blocks = _regular(240, 15)
    x = rng.standard_normal((240, 3))
    x[:, 2] = x[:, 0] + x[:, 1]  # a repeated direction is dropped, not inverted
    y = x[:, :2] @ [1.0, -0.5] + rng.standard_normal(240)
    frame, design, terms = _fit(y, x, blocks, 5, 2)
    result = si.randomization(design, terms, frame.resid_sums[:, None], 1.0)
    xk = x[frame.positions]

    def direct(row: np.ndarray) -> float:
        z = row[frame.block].astype(float)
        design_matrix = np.column_stack([np.ones(frame.n), z, xk])
        return float(np.linalg.lstsq(design_matrix, frame.y, rcond=None)[0][1])

    assert result.estimate[0] == pytest.approx(direct(design.observed), abs=1e-10)
    for j in range(5):
        fwl = (design.matrix[j] @ frame.resid_sums) / terms.denom[j]
        assert fwl == pytest.approx(direct(design.assignments[j]), abs=1e-10)
    whole = si.analyze(y, x, blocks, 5, design)
    assert whole.estimate == pytest.approx(direct(design.observed), abs=1e-10)


def test_too_many_covariates_and_too_few_samples_refuse_the_frame():
    rng = np.random.default_rng(10)
    _, blocks = _regular(240, 30)
    y = rng.standard_normal(240)
    assert si.make_frame(y, rng.standard_normal((240, 6)), blocks, 23) == si.TOO_MANY_COVARIATES
    assert si.make_frame(y, None, blocks, 27) == si.TOO_FEW
    assert si.make_frame(np.full(240, 2.0), None, blocks, 0) == si.NO_SPREAD


def test_an_empty_block_refuses_the_analysis_and_keeps_its_counts():
    rng = np.random.default_rng(14)
    _, blocks = _regular(240, 30)
    y = rng.standard_normal(240)
    y[60:90] = np.nan  # block 2 holds no finite sample
    design = sd.make_design(blocks.k, 3)
    result = si.analyze(y, None, blocks, 0, design)
    assert result.reason == si.EMPTY_BLOCK
    assert result.counts.tolist() == [30, 30, 0, 30, 30, 30, 30, 30]
    assert math.isnan(result.estimate)


def test_a_covariate_that_follows_the_schedule_is_collinear():
    rng = np.random.default_rng(15)
    _, blocks = _regular(240, 30)
    design = sd.make_design(blocks.k, 4)
    z = sd.setting(blocks, design.observed)
    y = z + rng.standard_normal(240)
    result = si.analyze(y, (2.0 * z + 1.0)[:, None], blocks, 0, design)
    assert result.reason == si.COLLINEAR


def test_a_side_the_accepted_set_does_not_close_on_is_infinite():
    # Two of three assignments with |c| > 1 are as extreme at every large
    # |theta|, so the accepted set runs off both ends.
    a = np.array([[0.5], [0.3], [0.2]])
    c = np.array([2.0, -3.0, 0.1])
    lo, hi = si.invert(a, c, np.array([1.0]), np.ones(3, dtype=np.int64), 0, 2)
    assert lo[0] == -math.inf and hi[0] == math.inf
    lo, hi = si.invert(a, c, np.array([1.0]), np.ones(3, dtype=np.int64), 2, 2)
    assert lo[0] == -math.inf and hi[0] == math.inf


def test_a_block_level_covariate_can_leave_the_adjusted_interval_open():
    # With 8 blocks and a covariate constant within each block, enough
    # assignments lie closer to the covariate than the observed one that
    # every shift is accepted: the interval is open on both sides.
    rng = np.random.default_rng(9)
    _, blocks = _regular(240, 30)
    design = sd.make_design(blocks.k, 9)
    x = rng.standard_normal(blocks.k)[blocks.block][:, None]
    x = x + 0.05 * rng.standard_normal((240, 1))
    result = si.analyze(rng.standard_normal(240), x, blocks, 0, design)
    assert result.reason == ""
    assert result.lo == -math.inf and result.hi == math.inf
    assert math.isfinite(result.estimate) and result.p_value > 0.05


# ---------------------------------------------------------------- response and washout


def test_first_order_response_reaches_one_minus_exp_minus_three_at_three_tau():
    tau = 5.0
    times = np.arange(40, dtype=float)
    s = si.lag_response(np.ones(40), times, tau)
    assert s[int(3 * tau) - 1] == pytest.approx(1 - math.exp(-3), abs=1e-12)
    assert np.all(np.diff(s) > 0)
    assert np.array_equal(si.lag_response(np.ones(40), times, 0.0), np.ones(40))


def test_a_gap_moves_the_response_by_its_length():
    s = si.lag_response(np.ones(3), np.array([0.0, 1.0, 11.0]), 2.0)
    assert s[2] == pytest.approx(1 - math.exp(-1 / 2) * math.exp(-1 / 2) * math.exp(-10 / 2))


def test_washout_removes_the_carryover_bias():
    rng = np.random.default_rng(12)
    times, blocks = _regular(480, 30)
    tau = 7.5
    washout = math.ceil(3 * tau)
    biases = {}
    for w in (0, washout):
        errors = []
        for seed in range(20):
            design = sd.make_design(blocks.k, seed)
            s = si.lag_response(sd.setting(blocks, design.observed), times, tau)
            y = s + 1e-6 * rng.standard_normal(480)
            frame = si.make_frame(y, None, blocks, w)
            terms = si.assignment_terms(frame, design)
            est = si.randomization(design, terms, frame.resid_sums[:, None], 1.0).estimate[0]
            errors.append(est - 1.0)
        biases[w] = np.median(errors)
    assert biases[0] < -0.2
    assert abs(biases[washout]) < math.exp(-3)


# ---------------------------------------------------------------- plans


def test_a_plan_is_deterministic_under_its_seed():
    """SCOPE claim: a plan is a deterministic function of its window, blocks and seed."""
    first, second = _plan(), _plan()
    assert first == second
    assert first.k == 16 and int(first.observed.sum()) == 8
    assert first.blocks[3].start == START + pd.Timedelta(3, unit="h")
    assert first.blocks[3].washout_end == START + pd.Timedelta(3 * 60 + 10, unit="min")
    assert _plan(seed=43).digest != first.digest
    assert not first.enumerated and first.n_assignments == 12870
    assert first.min_p == pytest.approx(1 / 1001)


def test_the_digest_reads_integers_and_timestamps_and_is_pinned():
    """SCOPE claim: the plan digest is the same on every platform.

    The digest input holds integers and ISO 8601 UTC timestamps only, and
    the digest of this plan is pinned, so a platform or numpy stream
    change that moves the schedule fails here.
    """
    plan = _plan()
    text = sp.canonical_schedule(
        plan.start, plan.end, plan.block_s, plan.washout_s, plan.seed, plan.blocks
    )
    stamp = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+00:00"
    for line in text.splitlines()[1:]:
        assert re.fullmatch(
            rf"(start|end) {stamp}|(block_s|washout_s|seed) \d+|"
            rf"block \d+ {stamp} {stamp} {stamp} [AB]",
            line,
        ), line
    assert "".join(b.setting for b in plan.blocks) == "ABAABAAABBBABABB"
    assert plan.digest == (
        "c422284131f3452b47111c334a06aa96af43bd6e3e352406fa2cde3e0e86f06a"
    )


def test_a_window_of_four_blocks_raises_design_too_small():
    """SCOPE claim: a plan too small for a 5% randomization test raises DesignTooSmall."""
    with pytest.raises(DesignTooSmall):
        _plan(end=START + pd.Timedelta(4, unit="h"))
    assert _plan(end=START + pd.Timedelta(9, unit="h") + pd.Timedelta(59, unit="min")).k == 8


def test_plan_arguments_outside_their_range_raise_value_error():
    with pytest.raises(ValueError, match="washout"):
        _plan(washout_s=3600)
    with pytest.raises(ValueError, match="not after"):
        _plan(end=START)
    with pytest.raises(ValueError, match="naive"):
        _plan(start=pd.Timestamp("2024-03-04 00:00:00"))
    with pytest.raises(ValueError, match="seed"):
        _plan(seed=-1)


def test_the_plan_json_round_trips(tmp_path):
    plan = _plan()
    path = plan.write_json(tmp_path / "plan.json")
    back = sp.SwitchbackPlan.read_json(path)
    assert back == plan
    assert sp.verify_plan(back).observed.tolist() == plan.observed.tolist()
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["digest"] == plan.digest and doc["reference"] == "sampled"
    assert [b["setting"] for b in doc["schedule"]] == [b.setting for b in plan.blocks]
    with pytest.raises(FileExistsError, match="another path"):
        plan.write_json(path)


def test_a_file_that_is_no_plan_raises_value_error(tmp_path):
    path = tmp_path / "other.json"
    path.write_text('{"format": "something-else"}', encoding="utf-8")
    with pytest.raises(ValueError, match="not a switchback plan"):
        sp.SwitchbackPlan.read_json(path)


def test_an_edited_plan_is_refused():
    """SCOPE claim: a plan whose blocks no longer match its digest raises ScheduleMismatch."""
    plan = _plan()
    flipped = replace(plan.blocks[0], setting="A" if plan.blocks[0].setting == "B" else "B")
    edited = replace(plan, blocks=(flipped, *plan.blocks[1:]))
    with pytest.raises(ScheduleMismatch, match="does not match its schedule"):
        sp.verify_plan(edited)


def test_an_unbalanced_schedule_is_refused():
    """SCOPE claim: an unbalanced schedule raises ScheduleMismatch, digest recomputed or not."""
    plan = _plan()
    a_block = next(b for b in plan.blocks if b.setting == "A")
    blocks = [replace(b, setting="B") if b is a_block else b for b in plan.blocks]
    with pytest.raises(ScheduleMismatch, match="not balanced: 9 of 16"):
        sp.verify_plan(_resigned(plan, blocks))


def test_settings_the_seed_did_not_draw_are_refused():
    plan = _plan()
    i = next(b.index for b in plan.blocks if b.setting == "A")
    j = next(b.index for b in plan.blocks if b.setting == "B")
    blocks = list(plan.blocks)
    blocks[i] = replace(blocks[i], setting="B")
    blocks[j] = replace(blocks[j], setting="A")
    with pytest.raises(ScheduleMismatch, match=f"seed {plan.seed}"):
        sp.verify_plan(_resigned(plan, blocks))


def test_a_moved_block_is_refused():
    plan = _plan()
    moved = replace(plan.blocks[2], end=plan.blocks[2].end + pd.Timedelta(1, unit="min"))
    blocks = [moved if b.index == 2 else b for b in plan.blocks]
    with pytest.raises(ScheduleMismatch, match="block 2 runs"):
        sp.verify_plan(_resigned(plan, blocks))


# ---------------------------------------------------------------- power


def test_power_rates_hold_the_size_and_grow_with_the_shift():
    rng = np.random.default_rng(16)
    times = pd.Series(START + pd.to_timedelta(np.arange(480) * 60, unit="s"))
    blocks = sp.schedule_offsets(times, START, 1800, 16)
    y = rng.standard_normal(480)
    out = sp.power_rates(y, blocks, 0, seed=5, draws=200)
    assert not isinstance(out, str)
    sigma, kept, rates = out
    assert kept == 480
    assert sigma == pytest.approx(si.MAD_TO_SD * si.mad(y))
    assert rates[0] <= 0.05 + 3.5 * math.sqrt(0.05 * 0.95 / 200)
    assert list(rates) == sorted(rates)
    assert rates[-1] == 1.0
    assert sp.smallest_detected(sp.POWER_DELTAS, rates) in sp.POWER_DELTAS[1:]
    again = sp.power_rates(y, blocks, 0, seed=5, draws=200)
    assert again == out


def test_power_rates_refuse_a_flat_or_holed_history():
    times = pd.Series(START + pd.to_timedelta(np.arange(480) * 60, unit="s"))
    blocks = sp.schedule_offsets(times, START, 1800, 16)
    assert sp.power_rates(np.full(480, 3.0), blocks, 0, seed=1) == si.NO_SPREAD
    y = np.random.default_rng(2).standard_normal(480)
    y[30:60] = np.nan
    assert sp.power_rates(y, blocks, 0, seed=1) == si.EMPTY_BLOCK
    assert sp.smallest_detected(sp.POWER_DELTAS, (0.05, 0.2, 0.4, 0.6, 0.79)) is None

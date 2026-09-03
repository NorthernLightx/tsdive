"""tsdive.eval.conformal: p-values, martingales, validity and power."""

from __future__ import annotations

import numpy as np
import pytest

import tsdive.eval as ev
from tsdive.eval import (
    conformal_p_values,
    martingale_alarm,
    mixture_martingale,
    power_martingale,
)

CAL = [1.0, 2.0, 3.0, 4.0, 5.0]


def test_exports_are_sorted_and_present():
    names = ("conformal_p_values", "martingale_alarm", "mixture_martingale", "power_martingale")
    for name in names:
        assert name in ev.__all__
    assert ev.__all__ == sorted(ev.__all__)


def test_p_values_hand_computed_with_a_tie():
    # t0: pool {1,2,3,4,5}, x=3: greater 2, equal 1 -> (2+1+1)/6
    # t1: pool {1,2,3,3,4,5}, x=6: greater 0 -> 1/7
    # t2: pool {1,2,3,3,4,5,6}, x=0: greater 7 -> 8/8
    # t3: pool {0,1,2,3,3,4,5,6}, x=3: greater 3, equal 2 -> 6/9
    stream = [3.0, 6.0, 0.0, 3.0]
    p = conformal_p_values(CAL, stream)
    assert p == pytest.approx([4 / 6, 1 / 7, 1.0, 6 / 9])


def test_offline_p_values_use_the_calibration_only():
    stream = [3.0, 6.0, 0.0, 3.0]
    p = conformal_p_values(CAL, stream, online=False)
    assert p == pytest.approx([4 / 6, 1 / 6, 1.0, 4 / 6])


def test_p_values_are_deterministic():
    rng = np.random.default_rng(3)
    cal, stream = rng.normal(size=40), rng.normal(size=90)
    a = conformal_p_values(cal, stream)
    b = conformal_p_values(cal, stream)
    assert np.array_equal(a, b)
    assert np.array_equal(mixture_martingale(a), mixture_martingale(b))


@pytest.mark.parametrize(("delta", "slack"), [(0.05, 0.02), (0.01, 0.02)])
def test_validity_under_exchangeability(delta, slack):
    alarms = 0
    for seed in range(400):
        rng = np.random.default_rng(seed)
        cal = rng.normal(size=60)
        stream = rng.normal(size=180)
        log_m = mixture_martingale(conformal_p_values(cal, stream))
        alarms += martingale_alarm(log_m, delta) is not None
    assert alarms / 400 <= delta + slack


def test_power_after_a_four_sigma_shift():
    hits = 0
    for seed in range(100):
        rng = np.random.default_rng(1000 + seed)
        cal = rng.normal(size=60)
        stream = np.concatenate([rng.normal(size=60), rng.normal(loc=4.0, size=120)])
        log_m = mixture_martingale(conformal_p_values(cal, stream))
        alarm = martingale_alarm(log_m, 0.01)
        hits += alarm is not None and alarm >= 60
    assert hits >= 95


def test_power_martingale_epsilon_one_is_flat_and_mixture_is_bounded():
    rng = np.random.default_rng(11)
    p = conformal_p_values(rng.normal(size=30), rng.normal(size=50))
    assert np.array_equal(power_martingale(p, 1.0), np.zeros(50))
    parts = np.stack([power_martingale(p, e) for e in (0.1, 0.5, 0.9)])
    mix = mixture_martingale(p, epsilons=(0.1, 0.5, 0.9))
    assert np.all(mix >= parts.min(axis=0) - 1e-12)
    assert np.all(mix <= parts.max(axis=0) + 1e-12)


def test_alarm_index_is_the_first_crossing():
    log_m = np.array([0.0, 1.0, 3.5, 2.0, 5.0])
    assert martingale_alarm(log_m, 0.05) == 2  # log(20) = 2.996
    assert martingale_alarm(log_m, 0.001) is None
    assert martingale_alarm([], 0.05) is None


@pytest.mark.parametrize(
    "call",
    [
        lambda: conformal_p_values([], [1.0]),
        lambda: conformal_p_values([1.0, np.nan], [1.0]),
        lambda: conformal_p_values([1.0], [np.inf]),
        lambda: power_martingale([0.5], 0.0),
        lambda: power_martingale([0.5], 1.5),
        lambda: power_martingale([0.0], 0.5),
        lambda: mixture_martingale([0.5], epsilons=()),
        lambda: martingale_alarm([0.0], 1.0),
        lambda: martingale_alarm([0.0], 0.0),
    ],
)
def test_bad_input_raises_value_error(call):
    with pytest.raises(ValueError):
        call()

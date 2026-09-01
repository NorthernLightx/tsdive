"""Provisional baselines, regime baselines, SPC rules."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import make_meta, stepped_contract
from tsdive.baselines import (
    mad_baseline,
    match_regime,
    moving_range_baseline,
    regime_baselines,
    screen,
    screen_regime,
)
from tsdive.errors import InsufficientQuality, RegimeTooSparse
from tsdive.spc import apply_rules, individuals_limits, xbar_r_limits
from tsdive.store.quality import good_mask, usable_mask
from tsdive.store.tagstore import SingleFileStore, Window

RNG = np.random.default_rng(3)


def _history(n: int = 300, mu: float = 50.0, sigma: float = 1.0) -> pd.DataFrame:
    vals = RNG.normal(mu, sigma, size=n)
    base = pd.Timestamp("2025-01-01 00:00:00+00:00")
    return pd.DataFrame(
        {
            "timestamp": [base + pd.Timedelta(i, unit="min") for i in range(n)],
            "value": vals,
            "quality": ["GOOD"] * n,
        }
    )


def test_mad_baseline_recovers_sigma():
    b = mad_baseline(_history(mu=50.0, sigma=2.0))
    assert b.center == pytest.approx(50.0, abs=0.5)
    assert b.scale == pytest.approx(2.0, rel=0.25)
    assert "provisional" in b.caveat


def test_moving_range_baseline_shape():
    b = moving_range_baseline(_history(sigma=1.0))
    assert b.scale == pytest.approx(1.0, rel=0.35)
    assert "process motion" in b.caveat


def test_insufficient_history_refused():
    with pytest.raises(InsufficientQuality):
        mad_baseline(_history(n=10))


def test_screen_flags_shift():
    hist = _history(mu=50.0, sigma=1.0)
    b = mad_baseline(hist)
    shifted = hist.copy()
    shifted["value"] = shifted["value"] + 6.0  # 6 sigma
    result = screen(b, shifted)
    assert result.n_flagged > len(shifted) * 0.9
    clean = screen(b, hist)
    assert clean.n_flagged <= len(hist) * 0.03


def test_regime_baselines_and_sparse_refusal():
    hist = _history(n=200)
    modes = pd.Series(["R1"] * 100 + ["R2"] * 100)
    bases = regime_baselines(hist, modes)
    assert set(bases) == {"R1", "R2"}
    with pytest.raises(RegimeTooSparse):
        regime_baselines(hist.head(30), pd.Series(["R9"] * 30), min_samples=50)
    with pytest.raises(RegimeTooSparse):
        match_regime(bases, "UNKNOWN")


def test_regime_screen_uses_per_regime_limits():
    rng = np.random.default_rng(11)
    rows = []
    for mu in (40.0, 80.0):
        for i in range(120):
            rows.append(
                {
                    "timestamp": pd.Timestamp("2025-02-01") + pd.Timedelta(i, unit="min"),
                    "value": float(rng.normal(mu, 1.0)),
                    "quality": "GOOD",
                }
            )
    frame = pd.DataFrame(rows)
    modes = pd.Series(["R1"] * 120 + ["R2"] * 120)
    bases = regime_baselines(frame, modes)

    # Values at their OWN regime centers: a global MAD (center ~60) would
    # flag both; per-regime limits must not.
    monitor = pd.DataFrame(
        {
            "timestamp": [
                pd.Timestamp("2025-02-02"),
                pd.Timestamp("2025-02-02 00:01:00"),
            ],
            "value": [40.2, 80.3],
            "quality": ["GOOD"] * 2,
        }
    )
    result = screen_regime(bases, monitor, pd.Series(["R1", "R2"]))
    assert result.n_flagged == 0
    # A true per-regime anomaly IS flagged:
    bad = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2025-02-03")],
            "value": [47.0],  # 7 sigma off R1's center
            "quality": ["GOOD"],
        }
    )
    assert screen_regime(bases, bad, pd.Series(["R1"])).n_flagged >= 1


def test_individuals_rules_fire_independently():
    ts = pd.Series([pd.Timestamp("2025-01-01") + pd.Timedelta(i, unit="h") for i in range(14)])
    limits = individuals_limits(center=0.0, sigma=1.0)
    shift = pd.Series([0.0] * 4 + [5.0] * 10)  # step beyond UCL, stays high
    hits = apply_rules(ts, shift, limits)
    by_rule = {h.rule for h in hits}
    assert "BEYOND_3SIGMA" in by_rule
    assert "RUN_9_SAMESIDE" in by_rule


N_CODED_GOOD = 120
N_CODED_BAD = 10


def _numerically_coded_window(archive_factory) -> Window:
    """A read of an archive whose GOOD code is 192 and BAD code is 0.

    Read through the store, so the frame carries the ``valid`` column the
    tag's own vocabulary produced. Under the default vocabulary the two
    codes swap places: 192 is not an OPC UA status code at all, and 0 is
    UA ``Good``.
    """
    base = pd.Timestamp("2025-03-01 00:00:00+00:00")
    n = N_CODED_GOOD + N_CODED_BAD
    frame = pd.DataFrame(
        {
            "timestamp": [base + pd.Timedelta(i, unit="min") for i in range(n)],
            "value": [50.0 + (i % 5) for i in range(N_CODED_GOOD)] + [999.0] * N_CODED_BAD,
            "quality": [192] * N_CODED_GOOD + [0] * N_CODED_BAD,
        }
    )
    meta = make_meta(
        sample_rate_s=60.0, quality_codes={"192": "GOOD", "0": "BAD"}
    )
    store = SingleFileStore(archive_factory(frame, meta))
    return store.read_window(
        meta.identity,
        base.to_pydatetime(),
        (base + pd.Timedelta(n, unit="min")).to_pydatetime(),
        stepped_contract(),
    )


def test_mad_baseline_honours_the_tags_own_quality_codes(archive_factory):
    window = _numerically_coded_window(archive_factory)
    assert int(good_mask(window.frame).sum()) == N_CODED_BAD  # the inversion
    assert int(usable_mask(window.frame).sum()) == N_CODED_GOOD

    base = mad_baseline(window.frame)
    assert base.n_history_good == N_CODED_GOOD
    assert base.center == pytest.approx(52.0)
    # Without the read's verdict there are 10 rows the default vocabulary
    # calls GOOD, which is below the minimum history.
    with pytest.raises(InsufficientQuality):
        mad_baseline(window.frame.drop(columns="valid"))


def test_regime_baselines_honour_the_tags_own_quality_codes(archive_factory):
    window = _numerically_coded_window(archive_factory)
    modes = pd.Series(["R0"] * 60 + ["R1"] * 60 + ["R0"] * N_CODED_BAD)
    bases = regime_baselines(window.frame, modes)
    assert {name: b.n_good for name, b in bases.items()} == {"R0": 60, "R1": 60}
    with pytest.raises(RegimeTooSparse):
        regime_baselines(window.frame.drop(columns="valid"), modes)


def test_usable_mask_falls_back_to_good_mask_on_a_plain_frame():
    base = pd.Timestamp("2025-03-01 00:00:00+00:00")
    frame = pd.DataFrame(
        {
            "timestamp": [base + pd.Timedelta(i, unit="min") for i in range(4)],
            "value": [50.0, np.nan, 52.0, 53.0],
            "quality": ["GOOD", "GOOD", "BAD", "GOOD"],
        }
    )
    assert "valid" not in frame.columns
    assert list(usable_mask(frame)) == [True, False, False, True]
    assert list(usable_mask(frame)) == list(good_mask(frame) & frame["value"].notna())


def test_xbar_r_limits_table():
    subgroups = [[49.0, 51.0], [50.0, 52.0], [48.0, 50.0]]
    xbar_l, r_l = xbar_r_limits(subgroups)
    assert xbar_l.center == pytest.approx(50.0)
    assert xbar_l.ucl == pytest.approx(50.0 + 1.880 * 2.0)  # A2[2]=1.880, Rbar=2
    assert r_l.ucl == pytest.approx(3.267 * 2.0)
    with pytest.raises(ValueError):
        xbar_r_limits([[1.0, 2.0], [3.0]])

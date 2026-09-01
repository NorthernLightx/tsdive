"""One provoking test per gap class, coverage semantics, and properties."""

from __future__ import annotations

from typing import cast

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tsdive.errors import NonMonotonicIndex
from tsdive.store.gaps import (
    DATA_LOSS_CLASSES,
    GapClass,
    GapContext,
    GapFinding,
    classify_gap,
    detect_gaps,
    summarize_coverage,
)


def ts(s: str) -> pd.Timestamp:
    return cast(pd.Timestamp, pd.Timestamp(s))


def test_compression_steady():
    c = GapContext(contract_stepped=True, observed_median_interval_s=3.0)
    r = classify_gap(120.0, c)
    assert r.cls is GapClass.COMPRESSION_STEADY
    assert "compression" in r.rule


def test_comm_outage_multitag():
    c = GapContext(peer_gap_overlap=True)
    r = classify_gap(3600.0, c)
    assert r.cls is GapClass.COMM_OUTAGE_MULTITAG
    assert "outage" in r.rule


def test_scan_off():
    periods = ((ts("2024-03-01 02:00:00+00:00"), ts("2024-03-01 03:00:00+00:00")),)
    c = GapContext(scan_off_periods=periods)
    r = classify_gap(300.0, c)
    assert r.cls is GapClass.SCAN_OFF


def test_sparse_by_design():
    c = GapContext(declared_sample_rate_s=600.0)
    r = classify_gap(1500.0, c)
    assert r.cls is GapClass.SPARSE_BY_DESIGN


def test_unknown_is_the_honest_default():
    c = GapContext(contract_stepped=False, observed_median_interval_s=3.0)
    r = classify_gap(100_000.0, c)
    assert r.cls is GapClass.UNKNOWN
    assert r.rule == "no rule matched"
    assert r.cls in DATA_LOSS_CLASSES


def test_precedence_scan_off_beats_outage():
    periods = ((ts("2024-03-01 02:00:00+00:00"), ts("2024-03-01 03:00:00+00:00")),)
    c = GapContext(peer_gap_overlap=True, scan_off_periods=periods)
    assert classify_gap(60.0, c).cls is GapClass.SCAN_OFF


def test_detect_gaps_threshold():
    stamps = [
        ts("2024-03-01 00:00:00+00:00"),
        ts("2024-03-01 00:00:03+00:00"),
        ts("2024-03-01 00:10:00+00:00"),
        ts("2024-03-01 00:10:03+00:00"),
    ]
    gaps = detect_gaps(pd.Series(pd.DatetimeIndex(stamps)), threshold_s=60.0)
    assert len(gaps) == 1
    start, end, dur = gaps[0]
    assert dur == 597.0
    assert end > start


def test_detect_gaps_refuses_unsorted_input_instead_of_sorting():
    """Sorting would put 00:10:00 next to 00:00:03 and invent a 597s gap."""
    stamps = [
        ts("2024-03-01 00:00:00+00:00"),
        ts("2024-03-01 00:10:00+00:00"),
        ts("2024-03-01 00:00:03+00:00"),
    ]
    with pytest.raises(NonMonotonicIndex) as err:
        detect_gaps(pd.Series(pd.DatetimeIndex(stamps)), threshold_s=60.0)
    assert err.value.offending_positions == [2]


def _finding(cls_ctx: GapContext, duration_s: float) -> GapFinding:
    start = ts("2024-03-01 00:10:00+00:00")
    return GapFinding(
        start=start,
        end=cast(pd.Timestamp, start + pd.Timedelta(duration_s, unit="s")),
        duration_s=duration_s,
        classification=classify_gap(duration_s, cls_ctx),
    )


def test_benign_gaps_do_not_reduce_coverage():
    benign = _finding(GapContext(contract_stepped=True, observed_median_interval_s=3.0), 120.0)
    s = summarize_coverage(
        ts("2024-03-01 00:00:00+00:00"), ts("2024-03-01 01:00:00+00:00"), (benign,)
    )
    assert s.coverage == 1.0
    assert s.n_gaps == 1
    assert s.longest_gap_s == 120.0


def test_data_loss_gaps_reduce_coverage():
    loss = _finding(GapContext(peer_gap_overlap=True), 600.0)
    assert loss.classification.cls is GapClass.COMM_OUTAGE_MULTITAG
    s = summarize_coverage(
        ts("2024-03-01 00:00:00+00:00"), ts("2024-03-01 01:00:00+00:00"), (loss,)
    )
    assert s.coverage == pytest.approx((3600 - 600) / 3600)


def test_empty_window_coverage_is_null_not_fake():
    s = summarize_coverage(ts("2024-03-01 01:00:00+00:00"), ts("2024-03-01 00:00:00+00:00"), ())
    assert s.coverage is None


@given(
    st.floats(min_value=0, max_value=10**7),
    st.booleans(),
    st.one_of(st.none(), st.floats(min_value=0.001, max_value=10**5)),
    st.one_of(st.none(), st.floats(min_value=0.001, max_value=10**5)),
    st.booleans(),
)
@settings(deadline=None, max_examples=300)
def test_classifier_total_and_honest(duration, stepped, median, rate, peer):
    """Total over arbitrary inputs; rule text always present; classes valid."""
    ctx = GapContext(
        contract_stepped=stepped,
        observed_median_interval_s=median,
        declared_sample_rate_s=rate,
        peer_gap_overlap=peer,
    )
    result = classify_gap(duration, ctx)
    assert isinstance(result.cls, GapClass)
    assert result.rule
    if result.cls is GapClass.COMPRESSION_STEADY:
        assert stepped and median is not None and duration <= 50 * median
    if result.cls is GapClass.SPARSE_BY_DESIGN:
        assert rate is not None and duration <= 3 * rate
    if result.cls is GapClass.COMM_OUTAGE_MULTITAG:
        assert peer

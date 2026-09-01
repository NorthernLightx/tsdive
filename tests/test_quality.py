"""Fixture (b) and quality-mapping properties."""

from __future__ import annotations

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tsdive.errors import SchemaError
from tsdive.store.identity import Role
from tsdive.store.quality import (
    DIGITAL_STATE_LABELS,
    OPC_UA_GOOD_CODES,
    Severity,
    annotate_severity,
    derive_severity,
    digital_state_label,
    good_mask,
    is_unmapped,
    null_digital_state_values,
    severity_counts,
    severity_series,
)


def _frame_with_state(value_at_state: float = 257.0):
    ts = [pd.Timestamp(f"2024-03-01 00:00:0{s}+00:00") for s in range(5)]
    values = [10.0, 11.0, value_at_state, 12.0, 13.0]
    qualities = ["GOOD", "GOOD", 257, "GOOD", "GOOD"]
    return pd.DataFrame({"timestamp": ts, "value": values, "quality": qualities})


def test_digital_state_257_never_coerced_to_float():
    raw = _frame_with_state(257.0)
    out = null_digital_state_values(annotate_severity(raw))
    assert digital_state_label(257) == "Over Range"
    row = out[out["quality"] == 257]
    assert len(row) == 1
    assert pd.isna(row.iloc[0]["value"])
    # The state number never leaks downstream as a value:
    numeric = [float(v) for v in out["value"].dropna()]
    assert 257.0 not in numeric


def test_mean_excludes_digital_state_rows():
    raw = _frame_with_state()
    out = null_digital_state_values(annotate_severity(raw))
    counts = severity_counts(out)
    assert counts[Severity.BAD] == 1
    vals = out.loc[out["severity"] == Severity.GOOD, "value"].astype(float)
    assert vals.mean() == pytest.approx((10.0 + 11.0 + 12.0 + 13.0) / 4)


def test_value_without_quality_is_schema_error():
    df = pd.DataFrame(
        {
            "timestamp": ["2024-03-01 00:00:00+00:00"],
            "value": [1.0],
        }
    )
    with pytest.raises(SchemaError):
        annotate_severity(df)


def test_raw_codes_preserved_verbatim():
    out = annotate_severity(_frame_with_state())
    assert list(out["quality"]) == ["GOOD", "GOOD", 257, "GOOD", "GOOD"]


def test_opc_bit_rules():
    assert derive_severity(0) is Severity.GOOD
    assert derive_severity(0x4000_0000) is Severity.UNCERTAIN
    assert derive_severity(0xC000_0000) is Severity.BAD
    # Explicit table beats the bit rule (257 has Good top bits):
    assert derive_severity(257) is Severity.BAD


def test_unknown_string_maps_uncertain_and_flagged():
    assert derive_severity("MYSTERY CODE") is Severity.UNCERTAIN
    assert is_unmapped("MYSTERY CODE")
    assert not is_unmapped("SCAN OFF")


@given(st.one_of(st.text(), st.integers(-2**40, 2**40), st.floats(allow_nan=False)))
@settings(deadline=None, max_examples=300)
def test_derive_severity_total(raw):
    """Total function over arbitrary input: never raises, always a Severity."""
    result = derive_severity(raw)
    assert isinstance(result, Severity)


@given(st.integers(-2**40, 2**40))
@settings(deadline=None, max_examples=500)
def test_no_digital_state_ever_maps_good(code):
    if code in DIGITAL_STATE_LABELS:
        assert derive_severity(code) is not Severity.GOOD


def _baseline_frame(qualities: list[object]) -> pd.DataFrame:
    base = pd.Timestamp("2024-03-01 00:00:00+00:00")
    n = len(qualities)
    return pd.DataFrame(
        {
            "timestamp": [base + pd.Timedelta(3 * i, unit="s") for i in range(n)],
            "value": [50.0 + (i % 7) for i in range(n)],
            "quality": qualities,
        }
    )


def test_good_mask_reads_numeric_opc_codes_like_their_string_equivalents():
    """A numerically coded archive baselines identically to a string-coded one.

    0 is Good, 0x40000000 Uncertain, 0x80000000 Bad. Comparing the raw
    code against "GOOD" would call every one of these rows unusable and
    refuse the baseline with InsufficientQuality.
    """
    from tsdive.baselines import mad_baseline

    pattern = ["GOOD"] * 38 + ["UNCERTAIN", "BAD"]
    numeric = [0] * 38 + [0x4000_0000, 0x8000_0000]

    strings = _baseline_frame(pattern)
    codes = _baseline_frame(numeric)

    assert list(good_mask(codes)) == list(good_mask(strings))
    assert good_mask(codes).sum() == 38

    from_strings = mad_baseline(strings)
    from_codes = mad_baseline(codes)
    assert (from_codes.center, from_codes.scale) == (from_strings.center, from_strings.scale)
    assert from_codes.n_history_good == 38


def test_good_mask_is_index_aligned_with_its_frame():
    frame = _baseline_frame(["GOOD", "BAD", "GOOD"]).iloc[1:]
    mask = good_mask(frame)
    assert list(mask.index) == [1, 2]
    assert list(frame[mask]["quality"]) == ["GOOD"]


def test_severity_series_needs_a_quality_column():
    with pytest.raises(SchemaError, match="quality"):
        severity_series(pd.DataFrame({"value": [1.0]}))


def test_top_bits_zero_is_not_enough_to_be_good():
    """Only registered Good status codes are GOOD; the rest are unmapped.

    1 and 192 have OPC UA severity bits 00, but neither is a status code:
    with InfoType 00 the spec requires the remaining info bits to be zero.
    """
    for code in (1, 192, 0x0007_0000):
        assert derive_severity(code) is Severity.UNCERTAIN, code
        assert is_unmapped(code), code


def test_opc_ua_good_code_table_matches_its_registered_values():
    """Pin the table: a typo here would silently promote rows to GOOD."""
    assert OPC_UA_GOOD_CODES == {
        0x0000_0000: "Good",
        0x002D_0000: "GoodSubscriptionTransferred",
        0x002E_0000: "GoodCompletesAsynchronously",
        0x002F_0000: "GoodOverload",
        0x0030_0000: "GoodClamped",
        0x0096_0000: "GoodLocalOverride",
        0x00A2_0000: "GoodEntryInserted",
        0x00A3_0000: "GoodEntryReplaced",
        0x00A5_0000: "GoodNoData",
        0x00A6_0000: "GoodMoreData",
        0x00BA_0000: "GoodResultsMayBeIncomplete",
        0x00D9_0000: "GoodDataIgnored",
        0x00DC_0000: "GoodEdited",
        0x00E0_0000: "GoodDependentValueChanged",
    }


def test_registered_good_status_codes_are_good():
    for code, name in OPC_UA_GOOD_CODES.items():
        assert derive_severity(code) is Severity.GOOD, name
        assert not is_unmapped(code), name


def test_info_bits_qualify_a_good_status_without_changing_it():
    # InfoType 01 (bit 10) plus DataValue info bits: still GoodNoData.
    assert derive_severity(0x00A5_0000 | 0x0400 | 0x03) is Severity.GOOD


def test_uncertain_and_bad_bit_rules_are_unchanged():
    assert derive_severity(0x4000_0000) is Severity.UNCERTAIN
    assert not is_unmapped(0x4000_0000)
    assert derive_severity(0x8000_0000) is Severity.BAD
    assert derive_severity(0xC000_0000) is Severity.BAD


def test_string_value_on_a_measurement_tag_is_refused():
    frame = annotate_severity(_baseline_frame(["GOOD", "GOOD"]))
    frame["value"] = frame["value"].astype(object)
    frame.loc[frame.index[1], "value"] = "R1"
    with pytest.raises(SchemaError, match="R1"):
        null_digital_state_values(frame, tag="plant1:FIC101.PV")


def test_a_number_only_python_can_read_survives_the_vectorised_coercion():
    """``pd.to_numeric`` refuses ``1_000``; ``float`` reads it. The value stays."""
    frame = annotate_severity(_baseline_frame(["GOOD", "GOOD"]))
    frame["value"] = frame["value"].astype(object)
    frame.loc[frame.index[1], "value"] = "1_000"
    out = null_digital_state_values(frame, tag="plant1:FIC101.PV")
    assert out["value"].iloc[1] == 1000.0


def test_mode_tag_keeps_its_string_states():
    frame = annotate_severity(_baseline_frame(["GOOD", "GOOD", 257]))
    frame["value"] = ["R0", "R1", "R1"]
    out = null_digital_state_values(frame, role=Role.MODE)
    assert list(out["value"][:2]) == ["R0", "R1"]
    assert pd.isna(out["value"].iloc[2])  # digital state still nulls the value


VENDOR_CODES = {"SUB": "UNCERTAIN", "OK": "GOOD", "COMM FAILURE": "UNCERTAIN"}


def test_per_source_codes_are_consulted_before_the_global_tables():
    assert derive_severity("SUB") is Severity.UNCERTAIN  # global fallback
    assert is_unmapped("SUB")  # ... and honestly flagged as a guess

    assert derive_severity("SUB", VENDOR_CODES) is Severity.UNCERTAIN
    assert not is_unmapped("SUB", VENDOR_CODES)  # now it is declared
    assert derive_severity("OK", VENDOR_CODES) is Severity.GOOD
    # The map wins over the shipped string table, not the other way round.
    assert derive_severity("COMM FAILURE") is Severity.BAD
    assert derive_severity("COMM FAILURE", VENDOR_CODES) is Severity.UNCERTAIN


def test_per_source_codes_match_trimmed_and_case_folded():
    assert derive_severity(" sub ", VENDOR_CODES) is Severity.UNCERTAIN


def test_good_mask_honours_per_source_codes():
    frame = _baseline_frame(["OK", "SUB", "OK"])
    assert list(good_mask(frame)) == [False, False, False]
    assert list(good_mask(frame, VENDOR_CODES)) == [True, False, True]


def test_unknown_severity_name_in_the_map_is_refused():
    with pytest.raises(SchemaError, match="EXCELLENT"):
        derive_severity("OK", {"OK": "EXCELLENT"})

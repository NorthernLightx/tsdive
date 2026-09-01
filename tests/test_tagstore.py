"""Store read path: physics block, schema refusals, read-only guarantee (h)."""

from __future__ import annotations

import datetime as dt
import hashlib
import inspect

import pandas as pd
import pytest

from conftest import (
    EngRange,
    Role,
    good_frame,
    make_meta,
    stepped_contract,
    write_archive,
)
from tsdive.errors import NonMonotonicIndex, SchemaError
from tsdive.report import render_window_report
from tsdive.store.gaps import GapClass
from tsdive.store.quality import Severity
from tsdive.store.tagstore import (
    SingleFileStore,
    TagStore,
    meta_from_dict,
    meta_from_parquet,
    meta_to_dict,
    write_tag,
)

UTC = dt.UTC


def _hourly(n: int = 12, start_day: str = "2024-03-01"):
    start = pd.Timestamp(f"{start_day} 00:00:00+00:00")
    return [start + pd.Timedelta(i, unit="min") for i in range(n)]


def test_window_carries_quality_and_physics(archive_factory, store):
    stamps = _hourly()
    values = [10.0 + i for i in range(len(stamps))]
    meta = make_meta(
        loop_id="FIC101",
        role=Role.PV,
        eng_range=EngRange(zero=0.0, span=100.0),
    )
    archive_factory(good_frame(stamps, values), meta)
    window = store.read_window(
        meta.identity,
        dt.datetime(2024, 3, 1, tzinfo=UTC),
        dt.datetime(2024, 3, 1, 0, 30, tzinfo=UTC),
        stepped_contract(),
    )
    assert "quality" in window.frame.columns
    assert "severity" in window.frame.columns
    assert "valid" in window.frame.columns
    assert set(window.frame["severity"]) == {Severity.GOOD}
    p = window.physics
    # 12 samples one minute apart cover 00:00-00:11 of a 30 min window; the
    # remaining 1140 s are a trailing edge gap, not free coverage.
    assert p.coverage.n_gaps == 1
    assert p.coverage.coverage == pytest.approx((1800 - 1140) / 1800)
    assert p.severity_counts[Severity.GOOD] == len(stamps)
    assert not p.clipping.censored
    # loop_id/role round-trip through parquet metadata — no schema theatre:
    assert window.meta.loop_id == "FIC101"
    assert window.meta.role is Role.PV


def test_gap_classified_in_physics(tmp_path):
    stamps = [
        pd.Timestamp("2024-03-01 00:00:00+00:00"),
        pd.Timestamp("2024-03-01 00:00:03+00:00"),
        pd.Timestamp("2024-03-01 00:20:00+00:00"),
        pd.Timestamp("2024-03-01 00:20:03+00:00"),
    ]
    meta = make_meta(source_id="plant9", point_id="T.PV")
    write_archive(tmp_path / "plant9" / "T.PV.parquet", good_frame(stamps, [1, 2, 3, 4]), meta)
    store = TagStore(tmp_path)
    window = store.read_window(
        meta.identity,
        dt.datetime(2024, 3, 1, tzinfo=UTC),
        dt.datetime(2024, 3, 1, 1, tzinfo=UTC),
        stepped_contract(),
    )
    classes = [g.classification.cls for g in window.physics.coverage.gaps]
    # the 1197s hole between sample 2 and 3, plus the 2397s trailing edge.
    assert len(classes) == 2
    # both are too big for compression against a median interval of 3s.
    assert classes == [GapClass.UNKNOWN, GapClass.UNKNOWN]
    assert window.physics.coverage.coverage == pytest.approx((3600 - 1197 - 2397) / 3600)


def _burst_archive(archive_factory, first: str):
    """Ten samples 3 s apart (27 s of data), starting at ``first``."""
    base = pd.Timestamp(first)
    stamps = [base + pd.Timedelta(3 * i, unit="s") for i in range(10)]
    meta = make_meta(source_id="p", point_id="EDGE.PV")
    archive_factory(good_frame(stamps, [float(i) for i in range(10)]), meta)
    return meta


@pytest.mark.parametrize(
    ("first_sample", "edge"),
    [("2024-03-01 00:00:00+00:00", "trailing"), ("2024-03-01 00:59:33+00:00", "leading")],
)
def test_window_edge_counts_against_coverage(archive_factory, store, first_sample, edge):
    """27 s of samples in a 3600 s window is 0.0075 coverage, not 1.0."""
    meta = _burst_archive(archive_factory, first_sample)
    window = store.read_window(
        meta.identity,
        dt.datetime(2024, 3, 1, tzinfo=UTC),
        dt.datetime(2024, 3, 1, 1, tzinfo=UTC),
        stepped_contract(),
    )
    cov = window.physics.coverage
    assert cov.n_gaps == 1
    assert cov.longest_gap_s == pytest.approx(3600 - 27)
    assert cov.coverage == pytest.approx(27 / 3600)
    assert cov.gaps[0].classification.cls is GapClass.UNKNOWN
    if edge == "trailing":
        assert cov.gaps[0].end == window.end
    else:
        assert cov.gaps[0].start == window.start


def test_empty_window_coverage_is_zero_not_one(archive_factory, store):
    meta = _burst_archive(archive_factory, "2024-03-01 12:00:00+00:00")
    window = store.read_window(
        meta.identity,
        dt.datetime(2024, 3, 1, tzinfo=UTC),
        dt.datetime(2024, 3, 1, 1, tzinfo=UTC),
        stepped_contract(),
    )
    assert len(window.frame) == 0
    cov = window.physics.coverage
    assert cov.coverage == 0.0
    assert cov.n_gaps == 1
    assert cov.longest_gap_s == 3600.0
    assert cov.gaps[0].classification.cls is GapClass.UNKNOWN
    assert (cov.gaps[0].start, cov.gaps[0].end) == (window.start, window.end)


def test_empty_window_keeps_a_bool_valid_column(archive_factory, store):
    """Object dtype here turns `frame[frame["valid"]]` into column selection."""
    meta = _burst_archive(archive_factory, "2024-03-01 12:00:00+00:00")
    window = store.read_window(
        meta.identity,
        dt.datetime(2024, 3, 1, tzinfo=UTC),
        dt.datetime(2024, 3, 1, 1, tzinfo=UTC),
        stepped_contract(),
    )
    assert len(window.frame) == 0
    assert window.frame["valid"].dtype == bool
    assert list(window.frame[window.frame["valid"]].columns) == list(window.frame.columns)


def test_contract_is_positional_and_required(archive_factory, store):
    stamps = _hourly(3)
    meta = make_meta(source_id="p", point_id="X.PV")
    archive_factory(good_frame(stamps, [1.0, 2.0, 3.0]), meta)
    start = dt.datetime(2024, 3, 1, tzinfo=UTC)
    end = dt.datetime(2024, 3, 1, 1, tzinfo=UTC)
    with pytest.raises(TypeError):
        store.read_window(meta.identity, start, end)  # type: ignore[call-arg]


def test_missing_quality_column_refused(tmp_path):
    df = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-03-01 00:00:00+00:00")],
            "value": [1.0],
        }
    )
    meta = make_meta(source_id="p", point_id="Q.PV")
    with pytest.raises(SchemaError):
        write_tag(tmp_path / "p" / "refused.parquet", df, meta)
    path = write_archive(tmp_path / "p" / "Q.PV.parquet", df, meta, validate=False)
    store = SingleFileStore(path)
    with pytest.raises(SchemaError):
        store.read_window(
            meta.identity,
            dt.datetime(2024, 3, 1, tzinfo=UTC),
            dt.datetime(2024, 3, 1, 1, tzinfo=UTC),
            stepped_contract(),
        )


def test_write_tag_round_trips_and_refuses_to_overwrite(tmp_path):
    """write_tag creates archives; it is not an edit path for existing ones."""
    meta = make_meta(source_id="new", point_id="W.PV", loop_id="FIC9", role=Role.PV)
    frame = good_frame(_hourly(4), [1.0, 2.0, 3.0, 4.0])
    path = write_tag(tmp_path / "new" / "W.PV.parquet", frame, meta)

    store = SingleFileStore(path)
    window = store.read_window(
        meta.identity,
        dt.datetime(2024, 3, 1, tzinfo=UTC),
        dt.datetime(2024, 3, 1, 1, tzinfo=UTC),
        stepped_contract(),
    )
    assert list(window.frame["value"]) == [1.0, 2.0, 3.0, 4.0]
    assert window.meta.identity == meta.identity
    assert window.meta.role is Role.PV

    before = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError) as err:
        write_tag(path, frame, meta)
    assert "does not mutate archives" in str(err.value)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before

    write_tag(path, good_frame(_hourly(2), [9.0, 9.0]), meta, overwrite=True)
    assert hashlib.sha256(path.read_bytes()).hexdigest() != before


def test_meta_missing_key_is_a_schema_error_naming_the_key():
    d = meta_to_dict(make_meta())
    del d["name"]
    with pytest.raises(SchemaError) as err:
        meta_from_dict(d)
    assert "'name'" in str(err.value)


def test_readonly_public_surface():
    """(h): the store exposes no archive-mutation API."""
    forbidden = ("write_", "save_", "update_", "delete_", "ingest_", "mutate_", "set_", "append_")
    for cls in (TagStore, SingleFileStore):
        mutators = [
            name
            for name, obj in inspect.getmembers(cls, predicate=inspect.isfunction)
            if not name.startswith("_") and name.lower().startswith(forbidden)
        ]
        assert mutators == [], f"{cls.__name__} exposes mutation API: {mutators}"


def test_reads_leave_archive_bytes_identical(archive_factory, store):
    stamps = _hourly(6)
    meta = make_meta(source_id="p", point_id="R.PV")
    archive_factory(good_frame(stamps, [float(i) for i in range(6)]), meta)
    path = store.archive_path(meta.identity)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    for _ in range(3):
        store.read_window(
            meta.identity,
            dt.datetime(2024, 3, 1, tzinfo=UTC),
            dt.datetime(2024, 3, 1, 0, 5, tzinfo=UTC),
            stepped_contract(),
        )
    after = hashlib.sha256(path.read_bytes()).hexdigest()
    assert before == after


def test_unresolved_unit_surfaced_in_physics(archive_factory, store):
    stamps = _hourly(3)
    meta = make_meta(unit_raw="furlongs/fortnight")
    archive_factory(good_frame(stamps, [1.0, 2.0, 3.0]), meta)
    window = store.read_window(
        meta.identity,
        dt.datetime(2024, 3, 1, tzinfo=UTC),
        dt.datetime(2024, 3, 1, 1, tzinfo=UTC),
        stepped_contract(),
    )
    assert window.physics.unit.canonical is None
    assert window.physics.unit.raw == "furlongs/fortnight"


def test_bad_severity_rows_masked_from_valid(archive_factory, store):
    stamps = [pd.Timestamp(f"2024-03-01 00:00:0{i}+00:00") for i in range(4)]
    df = pd.DataFrame(
        {
            "timestamp": stamps,
            "value": [10.0, 11.0, 257.0, 13.0],
            "quality": ["GOOD", "GOOD", 257, "GOOD"],
        }
    )
    meta = make_meta(source_id="p", point_id="S.PV")
    archive_factory(df, meta)
    window = store.read_window(
        meta.identity,
        dt.datetime(2024, 3, 1, tzinfo=UTC),
        dt.datetime(2024, 3, 1, 1, tzinfo=UTC),
        stepped_contract(),
    )
    valid_values = window.frame.loc[window.frame["valid"], "value"]
    assert all(float(v) != 257.0 for v in valid_values)


def _mixed_severity_archive(archive_factory):
    stamps = [pd.Timestamp(f"2024-03-01 00:00:0{i}+00:00") for i in range(3)]
    df = pd.DataFrame(
        {
            "timestamp": stamps,
            "value": [10.0, 11.0, 12.0],
            "quality": ["GOOD", "SUBSTITUTED", "BAD"],
        }
    )
    meta = make_meta(source_id="p", point_id="M.PV")
    archive_factory(df, meta)
    return meta


@pytest.mark.parametrize(
    ("min_severity", "expected"),
    [
        (Severity.GOOD, [Severity.GOOD]),
        (Severity.UNCERTAIN, [Severity.GOOD, Severity.UNCERTAIN]),
        (Severity.BAD, [Severity.GOOD, Severity.UNCERTAIN, Severity.BAD]),
    ],
)
def test_min_severity_uses_trust_rank_not_alphabetical_order(
    archive_factory, store, min_severity, expected
):
    """StrEnum comparison would rank UNCERTAIN above GOOD; rank must not."""
    meta = _mixed_severity_archive(archive_factory)
    window = store.read_window(
        meta.identity,
        dt.datetime(2024, 3, 1, tzinfo=UTC),
        dt.datetime(2024, 3, 1, 1, tzinfo=UTC),
        stepped_contract(),
        min_severity=min_severity,
    )
    valid_sev = list(window.frame.loc[window.frame["valid"], "severity"])
    assert valid_sev == expected


def _string_valued_frame(values: list[str]) -> pd.DataFrame:
    stamps = _hourly(len(values))
    return pd.DataFrame(
        {
            "timestamp": stamps,
            "value": values,
            "quality": ["GOOD"] * len(values),
        }
    )


def test_mode_tag_string_states_survive_the_read(archive_factory, store):
    """A MODE tag's states are its values; nulling them would erase the regime."""
    meta = make_meta(point_id="FIC101.MODE", role=Role.MODE, unit_raw=None)
    archive_factory(_string_valued_frame(["R0", "R0", "R1", "R2"]), meta)
    window = store.read_window(
        meta.identity,
        dt.datetime(2024, 3, 1, tzinfo=UTC),
        dt.datetime(2024, 3, 1, 0, 30, tzinfo=UTC),
        stepped_contract(),
    )
    assert list(window.frame["value"]) == ["R0", "R0", "R1", "R2"]
    assert list(window.frame["valid"]) == [True] * 4


def test_string_value_on_a_measurement_tag_is_refused_on_read(archive_factory, store):
    meta = make_meta(role=Role.PV)
    archive_factory(_string_valued_frame(["50.0", "50.5", "R1", "51.0"]), meta)
    with pytest.raises(SchemaError) as excinfo:
        store.read_window(
            meta.identity,
            dt.datetime(2024, 3, 1, tzinfo=UTC),
            dt.datetime(2024, 3, 1, 0, 30, tzinfo=UTC),
            stepped_contract(),
        )
    message = str(excinfo.value)
    assert str(meta.identity) in message  # names the tag
    assert "'R1'" in message  # and the first offending value


def test_sub_threshold_edge_slack_is_reported_though_coverage_stays_1(
    archive_factory, store
):
    """Samples stop 100s before the window ends; the 150s threshold hides it."""
    stamps = _hourly(4)  # one minute apart -> gap threshold 150s
    meta = make_meta(role=Role.PV, sample_rate_s=None)
    archive_factory(good_frame(stamps, [1.0, 2.0, 3.0, 4.0]), meta)
    window = store.read_window(
        meta.identity,
        dt.datetime(2024, 3, 1, tzinfo=UTC),
        dt.datetime(2024, 3, 1, 0, 4, 40, tzinfo=UTC),
        stepped_contract(),
    )
    p = window.physics
    assert p.coverage.coverage == 1.0
    assert p.coverage.n_gaps == 0
    assert p.edge_slack_s == 100.0
    assert (p.edge_slack_start_s, p.edge_slack_end_s) == (0.0, 100.0)
    assert (
        "  edge slack 100 s (start 0 s, end 100 s; under the gap threshold, covered)"
        in render_window_report(window)
    )


def test_coverage_counts_rows_and_valid_fraction_counts_values(archive_factory, store):
    """A historian that writes a row per scan reports 1.000 over NaN holes."""
    stamps = _hourly(10)
    values = [1.0, 2.0, float("nan"), 4.0, float("nan"), 6.0, 7.0, 8.0, 9.0, 10.0]
    meta = make_meta(role=Role.PV, sample_rate_s=60.0)
    archive_factory(good_frame(stamps, values), meta)
    window = store.read_window(
        meta.identity,
        dt.datetime(2024, 3, 1, tzinfo=UTC),
        dt.datetime(2024, 3, 1, 0, 9, tzinfo=UTC),
        stepped_contract(),
    )
    p = window.physics
    assert p.coverage.coverage == 1.0
    assert p.valid_fraction == pytest.approx(0.8)
    assert "valid 0.800" in render_window_report(window)


def test_valid_fraction_is_null_when_the_window_holds_no_rows(archive_factory, store):
    """A fraction of nothing is not 0.0 and not 1.0."""
    meta = make_meta(role=Role.PV, sample_rate_s=60.0)
    archive_factory(good_frame(_hourly(4), [1.0, 2.0, 3.0, 4.0]), meta)
    window = store.read_window(
        meta.identity,
        dt.datetime(2030, 1, 1, tzinfo=UTC),
        dt.datetime(2030, 1, 2, tzinfo=UTC),
        stepped_contract(),
    )
    assert window.physics.valid_fraction is None
    assert "valid null" in render_window_report(window)


def test_values_beyond_float32_range_are_counted_not_censored(archive_factory, store):
    """3W's -1.180116e+42 null sentinel: a fact, reported without a verdict."""
    stamps = _hourly(4)
    meta = make_meta(role=Role.PV, sample_rate_s=60.0)
    archive_factory(good_frame(stamps, [-1.180116e42, -1.180116e42, 3.0, 4.0]), meta)
    window = store.read_window(
        meta.identity,
        dt.datetime(2024, 3, 1, tzinfo=UTC),
        dt.datetime(2024, 3, 1, 0, 3, tzinfo=UTC),
        stepped_contract(),
    )
    p = window.physics
    assert p.implausible_magnitude_count == 2
    # No eng_range, so nothing censored it and the report says why.
    assert not p.clipping.censored
    assert (
        "  beyond float32 range 2 (not censored: eng range unknown)"
        in render_window_report(window)
    )


def test_no_beyond_float32_line_when_every_value_fits(archive_factory, store):
    stamps = _hourly(4)
    meta = make_meta(role=Role.PV, sample_rate_s=60.0)
    archive_factory(good_frame(stamps, [1.0, 2.0, 3.4e38, -3.4e38]), meta)
    window = store.read_window(
        meta.identity,
        dt.datetime(2024, 3, 1, tzinfo=UTC),
        dt.datetime(2024, 3, 1, 0, 3, tzinfo=UTC),
        stepped_contract(),
    )
    assert window.physics.implausible_magnitude_count == 0
    assert "beyond float32" not in render_window_report(window)


def test_no_edge_slack_line_when_the_window_ends_on_a_sample(archive_factory, store):
    stamps = _hourly(4)
    meta = make_meta(role=Role.PV, sample_rate_s=None)
    archive_factory(good_frame(stamps, [1.0, 2.0, 3.0, 4.0]), meta)
    window = store.read_window(
        meta.identity,
        dt.datetime(2024, 3, 1, tzinfo=UTC),
        dt.datetime(2024, 3, 1, 0, 3, tzinfo=UTC),
        stepped_contract(),
    )
    assert window.physics.edge_slack_s == 0.0
    assert "edge slack" not in render_window_report(window)


def test_read_window_refuses_backwards_timestamps(archive_factory, store):
    """The audit already saw it; the read refuses rather than sorting on."""
    stamps = [
        pd.Timestamp("2024-03-01 00:00:00+00:00"),
        pd.Timestamp("2024-03-01 00:10:00+00:00"),
        pd.Timestamp("2024-03-01 00:00:03+00:00"),
    ]
    meta = make_meta(role=Role.PV)
    archive_factory(good_frame(stamps, [1.0, 2.0, 3.0]), meta)
    with pytest.raises(NonMonotonicIndex) as err:
        store.read_window(
            meta.identity,
            dt.datetime(2024, 3, 1, tzinfo=UTC),
            dt.datetime(2024, 3, 1, 1, tzinfo=UTC),
            stepped_contract(),
        )
    assert err.value.offending_positions == [2]
    assert "position 2" in str(err.value)
    assert "2024-03-01T00:00:03" in str(err.value)


def _window_with_qualities(archive_factory, store, qualities, meta):
    stamps = _hourly(len(qualities))
    frame = pd.DataFrame(
        {
            "timestamp": stamps,
            "value": [50.0 + i for i in range(len(qualities))],
            "quality": qualities,
        }
    )
    archive_factory(frame, meta)
    return store.read_window(
        meta.identity,
        dt.datetime(2024, 3, 1, tzinfo=UTC),
        dt.datetime(2024, 3, 1, 0, 30, tzinfo=UTC),
        stepped_contract(),
    )


def test_byte_sized_unmapped_codes_hint_at_an_opc_da_export(archive_factory, store):
    """192/64 are Good/Uncertain in OPC DA and nothing in OPC UA."""
    window = _window_with_qualities(
        archive_factory, store, [192, 192, 64, 192], make_meta(role=Role.PV)
    )
    assert window.physics.unmapped_quality_codes == ["192", "64"]
    assert window.physics.unmapped_look_like_opc_da
    assert (
        "  byte-sized codes: OPC DA export? declare quality_codes in tag meta"
        in render_window_report(window).splitlines()
    )


def test_no_opc_da_hint_once_the_codes_are_declared(archive_factory, store):
    meta = make_meta(role=Role.PV, quality_codes={"192": "GOOD", "64": "UNCERTAIN"})
    window = _window_with_qualities(archive_factory, store, [192, 192, 64, 192], meta)
    assert window.physics.unmapped_quality_codes == []
    assert not window.physics.unmapped_look_like_opc_da
    assert "OPC DA export?" not in render_window_report(window)


def test_no_opc_da_hint_when_an_unmapped_code_is_not_byte_sized(archive_factory, store):
    window = _window_with_qualities(
        archive_factory, store, [192, "MYSTERY", 64, 192], make_meta(role=Role.PV)
    )
    assert window.physics.unmapped_quality_codes == ["192", "64", "MYSTERY"]
    assert not window.physics.unmapped_look_like_opc_da
    assert "OPC DA export?" not in render_window_report(window)


def test_per_source_quality_codes_round_trip_through_the_archive(archive_factory, store):
    """A vendor code only this source can explain, declared in its metadata."""
    meta = make_meta(role=Role.PV, quality_codes={"SUB": "UNCERTAIN", "OK": "GOOD"})
    stamps = _hourly(4)
    frame = pd.DataFrame(
        {
            "timestamp": stamps,
            "value": [50.0, 50.5, 51.0, 51.5],
            "quality": ["OK", "OK", "SUB", "OK"],
        }
    )
    path = archive_factory(frame, meta)
    assert meta_from_parquet(path).quality_codes == {"SUB": "UNCERTAIN", "OK": "GOOD"}

    window = store.read_window(
        meta.identity,
        dt.datetime(2024, 3, 1, tzinfo=UTC),
        dt.datetime(2024, 3, 1, 0, 30, tzinfo=UTC),
        stepped_contract(),
    )
    assert window.physics.severity_counts[Severity.GOOD] == 3
    assert window.physics.severity_counts[Severity.UNCERTAIN] == 1
    # Declared codes are mapped, so nothing is reported as a guess.
    assert window.physics.unmapped_quality_codes == []
    assert list(window.frame["valid"]) == [True, True, False, True]

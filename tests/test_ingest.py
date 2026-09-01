"""The ingest path: exports in, archives out, refusals where guessing would start."""

from __future__ import annotations

import json

import pandas as pd
import pytest

import tsdive
from tsdive.cli import cmd_ingest, cmd_profile
from tsdive.errors import SchemaError
from tsdive.store.tagstore import meta_from_parquet

META = {
    "identity": {"source_id": "plant1", "point_id": "FIC101.PV"},
    "name": "FIC-101 flow",
    "unit_raw": "m3/h",
    "eng_range_zero": 0.0,
    "eng_range_span": 100.0,
    "sample_rate_s": 60.0,
    "role": "PV",
}


def _meta_file(tmp_path, **overrides):
    payload = {**META, **overrides}
    path = tmp_path / "meta.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _csv(tmp_path, *, stamps, quality=True, name="export.csv"):
    rows = {"ts": stamps, "v": [50.0 + i for i in range(len(stamps))]}
    if quality:
        rows["q"] = ["GOOD"] * len(stamps)
    path = tmp_path / name
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


AWARE = ["2024-03-01T00:00:00Z", "2024-03-01T00:01:00Z", "2024-03-01T00:02:00Z"]
NAIVE = ["2024-03-01 00:00:00", "2024-03-01 00:01:00", "2024-03-01 00:02:00"]


def test_csv_round_trips_into_a_profileable_archive(tmp_path, capsys):
    src = _csv(tmp_path, stamps=AWARE)
    out = tmp_path / "FIC101.PV.parquet"
    rc = cmd_ingest(
        [
            str(src),
            "--out",
            str(out),
            "--meta",
            str(_meta_file(tmp_path)),
            "--timestamp-col",
            "ts",
            "--value-col",
            "v",
            "--quality-col",
            "q",
        ]
    )
    assert rc == 0
    assert capsys.readouterr().out.splitlines() == [
        f"wrote     {out.as_posix()}",
        "tag       plant1:FIC101.PV   3 rows   quality from column q",
    ]

    meta = meta_from_parquet(out)
    assert meta.identity.point_id == "FIC101.PV"
    assert meta.quality_assumed is False

    assert cmd_profile([str(out)]) == 0
    report = capsys.readouterr().out
    assert "GOOD 3   UNCERTAIN 0   BAD 0" in report
    assert "quality ASSUMED" not in report


def test_ingest_states_the_zone_it_converted_from(tmp_path, capsys):
    src = _csv(tmp_path, stamps=NAIVE)
    out = tmp_path / "local.parquet"
    rc = cmd_ingest(
        [
            str(src),
            "--out",
            str(out),
            "--meta",
            str(_meta_file(tmp_path)),
            "--timestamp-col",
            "ts",
            "--value-col",
            "v",
            "--quality-col",
            "q",
            "--tz",
            "America/Chicago",
        ]
    )
    assert rc == 0
    assert capsys.readouterr().out.splitlines()[2] == (
        "tz        America/Chicago -> UTC"
    )


def test_naive_timestamps_are_localised_by_tz(tmp_path):
    src = _csv(tmp_path, stamps=NAIVE)
    out = tmp_path / "local.parquet"
    tsdive.ingest(
        src,
        out=out,
        meta=tsdive.read_meta_json(_meta_file(tmp_path)),
        timestamp_col="ts",
        value_col="v",
        quality_col="q",
        tz="America/Chicago",
    )
    first = pd.read_parquet(out)["timestamp"].iloc[0]
    assert first == pd.Timestamp("2024-03-01 06:00:00+00:00")  # CST is UTC-6


def test_naive_timestamps_without_tz_are_refused(tmp_path, capsys):
    src = _csv(tmp_path, stamps=NAIVE)
    rc = cmd_ingest(
        [
            str(src),
            "--out",
            str(tmp_path / "no.parquet"),
            "--meta",
            str(_meta_file(tmp_path)),
            "--timestamp-col",
            "ts",
            "--value-col",
            "v",
            "--quality-col",
            "q",
        ]
    )
    err = capsys.readouterr().err
    assert rc == 2
    assert "naive" in err
    assert "--tz" in err
    assert not (tmp_path / "no.parquet").exists()


MIXED_OFFSETS = [
    "2024-01-01T00:00:00+00:00",
    "2024-06-01T00:00:00+01:00",
    "2024-07-01T00:00:00+02:00",
]


def test_mixed_utc_offsets_are_converted_row_by_row(tmp_path, recwarn):
    src = _csv(tmp_path, stamps=MIXED_OFFSETS)
    out = tmp_path / "offsets.parquet"
    tsdive.ingest(
        src,
        out=out,
        meta=tsdive.read_meta_json(_meta_file(tmp_path)),
        timestamp_col="ts",
        value_col="v",
        quality_col="q",
    )
    ts = pd.read_parquet(out)["timestamp"]
    assert str(ts.dt.tz) == "UTC"
    assert list(ts) == [
        pd.Timestamp("2024-01-01 00:00:00+00:00"),
        pd.Timestamp("2024-05-31 23:00:00+00:00"),
        pd.Timestamp("2024-06-30 22:00:00+00:00"),
    ]
    assert [w for w in recwarn if issubclass(w.category, FutureWarning)] == []


def test_naive_and_aware_timestamps_in_one_column_are_refused(tmp_path):
    src = _csv(tmp_path, stamps=["2024-03-01T00:00:00Z", "2024-03-01 00:01:00"])
    with pytest.raises(SchemaError, match="mixed naive and offset-bearing"):
        tsdive.ingest(
            src,
            out=tmp_path / "mixed.parquet",
            meta=tsdive.read_meta_json(_meta_file(tmp_path)),
            timestamp_col="ts",
            value_col="v",
            quality_col="q",
            tz="UTC",
        )


def test_unparseable_timestamp_names_the_offending_value(tmp_path):
    src = _csv(tmp_path, stamps=["2024-03-01T00:00:00Z", "not a time"])
    with pytest.raises(SchemaError, match="is not a timestamp tsdive can parse"):
        tsdive.ingest(
            src,
            out=tmp_path / "garbage.parquet",
            meta=tsdive.read_meta_json(_meta_file(tmp_path)),
            timestamp_col="ts",
            value_col="v",
            quality_col="q",
        )


def test_missing_quality_column_is_refused(tmp_path, capsys):
    src = _csv(tmp_path, stamps=AWARE, quality=False)
    rc = cmd_ingest(
        [
            str(src),
            "--out",
            str(tmp_path / "no.parquet"),
            "--meta",
            str(_meta_file(tmp_path)),
            "--timestamp-col",
            "ts",
            "--value-col",
            "v",
        ]
    )
    err = capsys.readouterr().err
    assert rc == 2
    assert "--assume-quality" in err
    assert not (tmp_path / "no.parquet").exists()


def test_assumed_quality_is_recorded_and_reported(tmp_path, capsys):
    src = _csv(tmp_path, stamps=AWARE, quality=False)
    out = tmp_path / "assumed.parquet"
    rc = cmd_ingest(
        [
            str(src),
            "--out",
            str(out),
            "--meta",
            str(_meta_file(tmp_path)),
            "--timestamp-col",
            "ts",
            "--value-col",
            "v",
            "--assume-quality",
            "GOOD",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out.splitlines()[1] == (
        "tag       plant1:FIC101.PV   3 rows   quality assumed GOOD"
    )
    assert "warning: quality assumed GOOD" in captured.err
    assert meta_from_parquet(out).quality_assumed is True

    assert cmd_profile([str(out)]) == 0
    assert "quality ASSUMED at ingest" in capsys.readouterr().out


def test_assume_quality_never_overrides_a_real_column(tmp_path):
    src = _csv(tmp_path, stamps=AWARE)
    with pytest.raises(SchemaError, match="refusing to overwrite"):
        tsdive.ingest(
            src,
            out=tmp_path / "clash.parquet",
            meta=tsdive.read_meta_json(_meta_file(tmp_path)),
            timestamp_col="ts",
            value_col="v",
            quality_col="q",
            assume_quality="GOOD",
        )


def test_assume_quality_must_name_a_severity(tmp_path):
    src = _csv(tmp_path, stamps=AWARE, quality=False)
    with pytest.raises(SchemaError, match="not a severity"):
        tsdive.ingest(
            src,
            out=tmp_path / "bad.parquet",
            meta=tsdive.read_meta_json(_meta_file(tmp_path)),
            timestamp_col="ts",
            value_col="v",
            assume_quality="PROBABLY FINE",
        )


def test_missing_value_column_names_what_the_export_has(tmp_path):
    src = _csv(tmp_path, stamps=AWARE)
    with pytest.raises(SchemaError, match="ts, v, q"):
        tsdive.ingest(
            src,
            out=tmp_path / "x.parquet",
            meta=tsdive.read_meta_json(_meta_file(tmp_path)),
            timestamp_col="ts",
            value_col="value",
        )


def test_meta_json_missing_a_required_key_is_refused(tmp_path):
    path = tmp_path / "meta.json"
    path.write_text(json.dumps({"identity": {"source_id": "plant1"}}), encoding="utf-8")
    with pytest.raises(SchemaError, match="point_id"):
        tsdive.read_meta_json(path)


def test_ingest_refuses_an_unsupported_input(tmp_path):
    src = tmp_path / "export.xlsx"
    src.write_bytes(b"not really a spreadsheet")
    with pytest.raises(SchemaError, match=r"\.xlsx"):
        tsdive.ingest(
            src,
            out=tmp_path / "x.parquet",
            meta=tsdive.read_meta_json(_meta_file(tmp_path)),
        )


def test_ingest_does_not_overwrite_an_archive(tmp_path):
    src = _csv(tmp_path, stamps=AWARE)
    meta = tsdive.read_meta_json(_meta_file(tmp_path))
    out = tmp_path / "once.parquet"
    kwargs = {"timestamp_col": "ts", "value_col": "v", "quality_col": "q"}
    tsdive.ingest(src, out=out, meta=meta, **kwargs)
    with pytest.raises(FileExistsError):
        tsdive.ingest(src, out=out, meta=meta, **kwargs)
    assert tsdive.ingest(src, out=out, meta=meta, overwrite=True, **kwargs) == out


def test_ingest_refuses_a_string_value_on_a_measurement_tag(tmp_path):
    path = tmp_path / "mixed.csv"
    pd.DataFrame(
        {"ts": AWARE, "v": ["50.0", "R1", "51.0"], "q": ["GOOD"] * 3}
    ).to_csv(path, index=False)
    with pytest.raises(SchemaError, match="R1"):
        tsdive.ingest(
            path,
            out=tmp_path / "mixed.parquet",
            meta=tsdive.read_meta_json(_meta_file(tmp_path)),
            timestamp_col="ts",
            value_col="v",
            quality_col="q",
        )


def test_mode_tag_string_states_ingest_cleanly(tmp_path):
    path = tmp_path / "mode.csv"
    pd.DataFrame({"ts": AWARE, "v": ["R0", "R1", "R1"], "q": ["GOOD"] * 3}).to_csv(
        path, index=False
    )
    out = tsdive.ingest(
        path,
        out=tmp_path / "mode.parquet",
        meta=tsdive.read_meta_json(
            _meta_file(tmp_path, role="MODE", eng_range_zero=None, eng_range_span=None)
        ),
        timestamp_col="ts",
        value_col="v",
        quality_col="q",
    )
    assert list(tsdive.profile(out).frame["value"]) == ["R0", "R1", "R1"]

"""The ingest path: exports in, archives out, refusals where guessing would start."""

from __future__ import annotations

import json

import pandas as pd
import pytest

import tsdive
from tsdive.cli import cmd_ingest, cmd_profile
from tsdive.errors import NonMonotonicIndex, SchemaError
from tsdive.store.tagstore import meta_from_parquet, safe_filename

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
    assert rc == 3
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


EU_STAMPS = ["01/02/2026 08:00:00", "01/02/2026 08:01:00", "13/02/2026 08:02:00"]


def _ingest_stamps(tmp_path, stamps, name="dates", **kwargs):
    out = tmp_path / f"{name}.parquet"
    tsdive.ingest(
        _csv(tmp_path, stamps=stamps, name=f"{name}.csv"),
        out=out,
        meta=tsdive.read_meta_json(_meta_file(tmp_path)),
        timestamp_col="ts",
        value_col="v",
        quality_col="q",
        **kwargs,
    )
    return [t.strftime("%Y-%m-%d %H:%M") for t in pd.read_parquet(out)["timestamp"]]


def test_a_date_that_reads_both_ways_is_refused(tmp_path, capsys):
    """SCOPE claim: a day/month-ambiguous date with no stated order raises SchemaError."""
    with pytest.raises(SchemaError) as info:
        _ingest_stamps(tmp_path, EU_STAMPS, tz="Europe/Paris")
    message = str(info.value)
    assert "'01/02/2026 08:00:00'" in message
    assert "pass dayfirst=True to read day first, or timestamp_format with" in message

    out = tmp_path / "cli.parquet"
    rc = cmd_ingest([str(_csv(tmp_path, stamps=EU_STAMPS, name="cli.csv")), "--out", str(out),
                     "--meta", str(_meta_file(tmp_path)), "--timestamp-col", "ts",
                     "--value-col", "v", "--quality-col", "q", "--tz", "Europe/Paris"])
    assert rc != 0
    err = capsys.readouterr().err
    assert "reads as day 01 of month 02 or as month 01, day 02" in err
    assert "pass --dayfirst to read day first, or --timestamp-format with" in err
    assert not out.exists()


@pytest.mark.parametrize(
    ("stamps", "suggested", "first_utc"),
    [
        (["01/02/2026 00:00", "01/02/2026 00:01"], "%d/%m/%Y %H:%M", "2026-01-31 23:00"),
        (["01.02.26 08:00:00", "01.02.26 08:01:00"], "%d.%m.%y %H:%M:%S", "2026-02-01 07:00"),
        (
            ["01-02-2026T08:00:00.250", "01-02-2026T08:01:00.250"],
            "%d-%m-%YT%H:%M:%S.%f",
            "2026-02-01 07:00",
        ),
    ],
)
def test_the_suggested_date_format_has_the_shape_of_the_value(
    tmp_path, stamps, suggested, first_utc
):
    """The refusal suggests a format that parses the value it quotes."""
    with pytest.raises(SchemaError) as info:
        _ingest_stamps(tmp_path, stamps, tz="Europe/Paris")
    assert f"strptime format such as '{suggested}'" in str(info.value)
    read = _ingest_stamps(
        tmp_path, stamps, name="suggested", tz="Europe/Paris", timestamp_format=suggested
    )
    assert read[0] == first_utc


@pytest.mark.parametrize(
    "order",
    [{"dayfirst": True}, {"timestamp_format": "%d/%m/%Y %H:%M:%S"}],
    ids=["dayfirst", "format"],
)
def test_a_stated_order_reads_every_row_the_same_way(tmp_path, order):
    assert _ingest_stamps(tmp_path, EU_STAMPS, tz="Europe/Paris", **order) == [
        "2026-02-01 07:00",
        "2026-02-01 07:01",
        "2026-02-13 07:02",
    ]


def test_cli_dayfirst_and_timestamp_format_reach_the_parser(tmp_path, capsys):
    common = ["--meta", str(_meta_file(tmp_path)), "--timestamp-col", "ts", "--value-col",
              "v", "--quality-col", "q", "--tz", "Europe/Paris"]
    src = str(_csv(tmp_path, stamps=EU_STAMPS, name="eu.csv"))
    assert cmd_ingest([src, "--out", str(tmp_path / "a.parquet"), "--dayfirst", *common]) == 0
    fmt = ["--timestamp-format", "%d/%m/%Y %H:%M:%S"]
    assert cmd_ingest([src, "--out", str(tmp_path / "b.parquet"), *fmt, *common]) == 0
    a = pd.read_parquet(tmp_path / "a.parquet")["timestamp"]
    assert list(a) == list(pd.read_parquet(tmp_path / "b.parquet")["timestamp"])
    assert a.iloc[2] == pd.Timestamp("2026-02-13 07:02:00+00:00")
    with pytest.raises(SystemExit) as info:
        cmd_ingest([src, "--out", str(tmp_path / "c.parquet"), "--dayfirst", *fmt, *common])
    assert info.value.code == 2
    assert "pass one" in capsys.readouterr().err


def test_us_dates_that_read_one_way_still_parse(tmp_path):
    stamps = ["3/14/2024 1:05 PM", "3/14/2024 1:06 PM", "3/15/2024 9:00 AM"]
    assert _ingest_stamps(tmp_path, stamps, tz="America/New_York") == [
        "2024-03-14 17:05",
        "2024-03-14 17:06",
        "2024-03-15 13:00",
    ]


def test_dates_in_two_one_way_orders_are_refused(tmp_path):
    with pytest.raises(SchemaError, match="reads only day first and '02/14/2026 08:01:00' only "
                       "month first"):
        _ingest_stamps(tmp_path, ["13/02/2026 08:00:00", "02/14/2026 08:01:00"], tz="UTC")


def test_dayfirst_refuses_a_month_above_12(tmp_path):
    with pytest.raises(SchemaError, match="'02/14/2026 08:00:00' has no month 14 when read day"):
        _ingest_stamps(tmp_path, ["02/14/2026 08:00:00"], tz="UTC", dayfirst=True)


def test_dayfirst_leaves_iso_dates_alone(tmp_path):
    assert _ingest_stamps(tmp_path, AWARE, dayfirst=True) == [
        "2024-03-01 00:00",
        "2024-03-01 00:01",
        "2024-03-01 00:02",
    ]


def test_a_row_off_the_stated_format_names_value_and_format(tmp_path):
    with pytest.raises(SchemaError, match=r"'2024-03-01T00:00:00Z' does not match "
                       r"timestamp_format '%d/%m/%Y %H:%M:%S'"):
        _ingest_stamps(tmp_path, AWARE, timestamp_format="%d/%m/%Y %H:%M:%S")


def test_unparseable_timestamp_names_the_offending_value(tmp_path):
    src = _csv(tmp_path, stamps=["2024-03-01T00:00:00Z", "not a time", "never"])
    with pytest.raises(SchemaError, match="'not a time' is not a timestamp tsdive can parse"):
        tsdive.ingest(
            src,
            out=tmp_path / "garbage.parquet",
            meta=tsdive.read_meta_json(_meta_file(tmp_path)),
            timestamp_col="ts",
            value_col="v",
            quality_col="q",
        )


def test_one_parse_reads_a_uniform_column_and_other_layouts_row_by_row():
    from tsdive.api import _parse_timestamps

    minutes = pd.date_range("2024-03-01", periods=1000, freq="min")
    uniform = pd.Series(minutes.strftime("%d/%m/%Y %H:%M"))
    parsed = _parse_timestamps(uniform, "ts", dayfirst=True)
    assert str(parsed.dtype) == "datetime64[ns]"
    assert parsed.iloc[-1] == pd.Timestamp("2024-03-01 16:39")

    # The first row fixes the inferred format; the rows that miss it parse one by one.
    mixed = pd.Series(["2024-03-01", "2024-03-01 00:01:00", "2024-03-01T00:02:00"])
    assert list(_parse_timestamps(mixed, "ts")) == list(
        pd.date_range("2024-03-01", periods=3, freq="min")
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
    assert rc == 3
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


def test_unknown_meta_keys_are_refused_with_the_closest_known_key(tmp_path, capsys):
    """SCOPE claim: a metadata key tsdive does not read raises SchemaError."""
    payload = {k: v for k, v in META.items() if k not in ("unit_raw", "eng_range_zero",
                                                          "eng_range_span")}
    path = _meta_file(tmp_path)
    path.write_text(json.dumps({**payload, "unit": "m3/h"}), encoding="utf-8")
    with pytest.raises(SchemaError, match=r"meta\.json: unknown key 'unit'; did you mean "
                       r"'unit_raw'\?"):
        tsdive.read_meta_json(path)

    out = tmp_path / "FIC101.PV.parquet"
    rc = cmd_ingest([str(_csv(tmp_path, stamps=AWARE)), "--out", str(out), "--meta", str(path),
                     "--timestamp-col", "ts", "--value-col", "v", "--quality-col", "q"])
    assert rc != 0
    assert "unknown key 'unit'" in capsys.readouterr().err
    assert not out.exists()


def test_a_nested_eng_range_is_refused_naming_the_flat_keys(tmp_path):
    path = _meta_file(tmp_path)
    payload = {k: v for k, v in META.items() if not k.startswith("eng_range")}
    path.write_text(json.dumps({**payload, "eng_range": {"zero": 0, "span": 200}}),
                    encoding="utf-8")
    with pytest.raises(SchemaError, match="unknown key 'eng_range'; write the range as "
                       "eng_range_zero and eng_range_span"):
        tsdive.read_meta_json(path)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"identity": {"source": "plant1", "point_id": "X"}},
         r"unknown key 'identity\.source'; did you mean 'identity\.source_id'\?"),
        ({"eng_range_span": None}, "eng_range_zero is set and eng_range_span is not"),
        ({"eng_range_zero": None}, "eng_range_span is set and eng_range_zero is not"),
        ({"eng_range_span": 0}, "eng_range_span must be greater than 0"),
        ({"eng_range_zero": "0"}, "eng_range_zero must be a finite number"),
        ({"sample_rate_s": -1}, "sample_rate_s must be a number of seconds greater than 0"),
        ({"role": "pv"}, "role 'pv' is not one of PV, SP, OP, MODE"),
        ({"unit_raw": 3}, "unit_raw must be a string or null"),
        ({"quality_assumed": "yes"}, "quality_assumed must be true or false"),
        ({"flow_units": "m3/h"}, r"unknown key 'flow_units'; the known keys are identity, "),
    ],
)
def test_malformed_meta_values_are_refused(tmp_path, overrides, message):
    with pytest.raises(SchemaError, match=message):
        tsdive.read_meta_json(_meta_file(tmp_path, **overrides))


def test_wide_ingest_refuses_an_unknown_key_in_a_meta_file_before_writing(tmp_path):
    meta_dir = _wide_meta_dir(tmp_path)
    path = meta_dir / "TIC201.PV.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps({**payload, "units": "degC"}), encoding="utf-8")
    with pytest.raises(SchemaError, match=r"TIC201\.PV\.json: unknown key 'units'"):
        tsdive.ingest_wide(
            _wide_csv(tmp_path),
            out_dir=tmp_path / "archive",
            meta_dir=meta_dir,
            timestamp_col="ts",
            quality_suffix="_q",
            tz="Europe/London",
        )
    assert not (tmp_path / "archive").exists()


def test_an_archive_whose_meta_carries_an_unknown_key_is_refused(tmp_path):
    import pyarrow as pa
    from pyarrow import parquet

    frame = pd.DataFrame({"timestamp": pd.to_datetime(AWARE), "value": [1.0, 2.0, 3.0],
                          "quality": ["GOOD"] * 3})
    table = pa.Table.from_pandas(frame, preserve_index=False).replace_schema_metadata(
        {"tsdive.meta": json.dumps({**META, "unit": "m3/h"})}
    )
    path = tmp_path / "odd.parquet"
    parquet.write_table(table, path)
    with pytest.raises(SchemaError, match=r"odd\.parquet: unknown key 'unit'"):
        meta_from_parquet(path)


def test_ingest_refuses_an_unsupported_input(tmp_path):
    src = tmp_path / "export.xlsx"
    src.write_bytes(b"not really a spreadsheet")
    with pytest.raises(SchemaError, match=r"\.xlsx"):
        tsdive.ingest(
            src,
            out=tmp_path / "x.parquet",
            meta=tsdive.read_meta_json(_meta_file(tmp_path)),
        )


# A German-locale Excel CSV: ';' between columns, ',' as the decimal mark,
# cp1252 text with a degree sign in the header, and one PI digital state.
GERMAN_ROWS = [
    "Zeitstempel;Wert \N{DEGREE SIGN}C;Status",
    "28.10.2026 06:00:00;40,06;Good",
    "28.10.2026 06:05:00;40,11;Good",
    "28.10.2026 06:10:00;Shutdown;Bad",
    "28.10.2026 06:15:00;40,02;Good",
]
GERMAN_ARGS = [
    "--timestamp-col", "Zeitstempel", "--value-col", "Wert \N{DEGREE SIGN}C",
    "--quality-col", "Status", "--tz", "Europe/Berlin", "--dayfirst",
]


def _german_csv(tmp_path):
    path = tmp_path / "de.csv"
    path.write_bytes(("\r\n".join(GERMAN_ROWS) + "\r\n").encode("cp1252"))
    return path


def test_sep_decimal_and_encoding_read_a_german_excel_export(tmp_path, capsys):
    out = tmp_path / "de.parquet"
    meta = _meta_file(tmp_path, quality_codes={"Good": "GOOD", "Bad": "BAD", "Shutdown": "BAD"})
    rc = cmd_ingest(
        [str(_german_csv(tmp_path)), "--out", str(out), "--meta", str(meta), *GERMAN_ARGS,
         "--sep", ";", "--decimal", ",", "--encoding", "cp1252"]
    )
    assert rc == 0, capsys.readouterr().err
    stored = pd.read_parquet(out)
    assert list(stored["value"]) == ["40.06", "40.11", "Shutdown", "40.02"]
    assert tsdive.profile(out).stats.features.max == 40.11


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (
            [],
            "[SchemaError] de.csv: byte 0xb0 at offset 17 is not utf-8; pass --encoding NAME",
        ),
        (
            ["--encoding", "cp1252"],
            "[SchemaError] de.csv: the header reads as one column 'Zeitstempel;Wert "
            "\N{DEGREE SIGN}C;Status'; the export separates its columns with ';', so pass "
            "--sep ';'",
        ),
    ],
    ids=["encoding", "sep"],
)
def test_a_csv_read_with_the_wrong_format_names_the_flag(tmp_path, capsys, extra, message):
    out = tmp_path / "de.parquet"
    rc = cmd_ingest(
        [str(_german_csv(tmp_path)), "--out", str(out), "--meta", str(_meta_file(tmp_path)),
         *GERMAN_ARGS, *extra]
    )
    assert rc == 3
    assert message in capsys.readouterr().err
    assert not out.exists()


def test_csv_options_reach_wide_long_and_template_reads(tmp_path):
    options = {"sep": ";", "decimal": ",", "encoding": "cp1252"}
    wide = tmp_path / "wide.csv"
    wide.write_bytes("ts;FIC101.PV;TIC201.PV\r\n2024-03-01T00:00:00Z;1,5;2,5\r\n".encode("cp1252"))
    (path,) = tsdive.init_meta(wide, out_dir=tmp_path / "wmeta", source_id="plant1",
                               timestamp_col="ts", tags=["FIC101.PV"], **options)
    written = tsdive.ingest_wide(wide, out_dir=tmp_path / "wide", meta_dir=path.parent,
                                 timestamp_col="ts", tags=["FIC101.PV"],
                                 assume_quality="GOOD", **options)
    assert list(pd.read_parquet(written[0])["value"]) == [1.5]

    long = tmp_path / "long.csv"
    long.write_bytes(
        "tag;ts;v\r\nA;2024-03-01T00:00:00Z;1,5\r\nB;2024-03-01T00:00:00Z;2,5\r\n".encode("cp1252")
    )
    meta_dir = tmp_path / "lmeta"
    tsdive.init_long_meta(long, out_dir=meta_dir, source_id="plant1", tag_col="tag",
                          timestamp_col="ts", value_col="v", **options)
    written = tsdive.ingest_long(long, out_dir=tmp_path / "long", meta_dir=meta_dir,
                                 tag_col="tag", timestamp_col="ts", value_col="v",
                                 assume_quality="GOOD", **options)
    assert [list(pd.read_parquet(p)["value"]) for p in written] == [[1.5], [2.5]]

    template = tsdive.init_tag_meta(_german_csv(tmp_path), out=tmp_path / "de.json",
                                    timestamp_col="Zeitstempel",
                                    value_col="Wert \N{DEGREE SIGN}C", quality_col="Status",
                                    **options)
    codes = json.loads(template.read_text(encoding="utf-8"))["quality_codes"]
    assert codes == {"Bad": "BAD", "Good": "GOOD", "Shutdown": None}


def test_python_callers_see_the_keyword_and_parquet_takes_no_csv_option(tmp_path):
    meta = tsdive.read_meta_json(_meta_file(tmp_path))
    with pytest.raises(SchemaError, match=r"so pass sep=';'$"):
        tsdive.ingest(_german_csv(tmp_path), out=tmp_path / "x.parquet", meta=meta,
                      encoding="cp1252")
    with pytest.raises(ValueError, match=r"fic101\.parquet: sep applies to CSV input"):
        tsdive.ingest(_parquet_export(tmp_path), out=tmp_path / "y.parquet", meta=meta, sep=";")


def _parquet_export(tmp_path):
    path = tmp_path / "fic101.parquet"
    pd.DataFrame({"timestamp": AWARE, "value": [1.0, 2.0, 3.0], "quality": ["GOOD"] * 3}
                 ).to_parquet(path)
    return path


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


def _pi_export(tmp_path):
    path = tmp_path / "pi.csv"
    pd.DataFrame(
        {
            "ts": ["2024-03-01T00:00:00Z", "2024-03-01T00:01:00Z", "2024-03-01T00:02:00Z"],
            "v": ["50.0", "I/O Timeout", "52.0"],
            "q": ["Good", "Bad", "Good"],
        }
    ).to_csv(path, index=False)
    return path


def test_a_pi_digital_state_in_the_value_column_ingests_once_declared(tmp_path, capsys):
    common = dict(timestamp_col="ts", value_col="v", quality_col="q")
    with pytest.raises(SchemaError, match=r"quality_codes, for example \{\"I/O Timeout\": "
                       r"\"BAD\"\}"):
        tsdive.ingest(_pi_export(tmp_path), out=tmp_path / "bare.parquet",
                      meta=tsdive.read_meta_json(_meta_file(tmp_path)), **common)

    codes = {"Good": "GOOD", "Bad": "BAD", "I/O Timeout": "BAD"}
    out = tsdive.ingest(_pi_export(tmp_path), out=tmp_path / "pi.parquet",
                        meta=tsdive.read_meta_json(_meta_file(tmp_path, quality_codes=codes)),
                        **common)
    stored = pd.read_parquet(out)
    assert list(stored["value"]) == ["50.0", "I/O Timeout", "52.0"]  # verbatim
    p = tsdive.profile(out)
    counts = {k.value: n for k, n in p.physics.severity_counts.items()}
    assert counts == {"GOOD": 2, "UNCERTAIN": 0, "BAD": 1}
    assert pd.isna(p.frame["value"].iloc[1])
    assert p.stats.features.max == 52.0


def _three_state_export(tmp_path):
    path = tmp_path / "states.csv"
    pd.DataFrame(
        {
            "ts": [f"2024-03-01T00:0{k}:00Z" for k in range(6)],
            "v": ["50.0", "Shutdown", "I/O Timeout", "I/O Timeout", "Comm Fail", "52.0"],
        }
    ).to_csv(path, index=False)
    return path


def test_every_undeclared_state_is_named_in_one_refusal(tmp_path):
    with pytest.raises(SchemaError) as info:
        tsdive.ingest(_three_state_export(tmp_path), out=tmp_path / "s.parquet",
                      meta=tsdive.read_meta_json(_meta_file(tmp_path)), timestamp_col="ts",
                      value_col="v", assume_quality="GOOD")
    assert str(info.value).startswith(
        "tag plant1:FIC101.PV: values 'Shutdown' (1 row), 'I/O Timeout' (2 rows) and "
        "'Comm Fail' (1 row) are not numeric"
    )
    assert '{"Shutdown": "BAD", "I/O Timeout": "BAD", "Comm Fail": "BAD"}' in str(info.value)


def test_the_template_lists_every_state_and_ingest_waits_for_their_severity(tmp_path):
    template = tsdive.init_tag_meta(_three_state_export(tmp_path), out=tmp_path / "t.json",
                                    timestamp_col="ts", value_col="v")
    payload = json.loads(template.read_text(encoding="utf-8"))
    assert payload["quality_codes"] == {"Shutdown": None, "I/O Timeout": None, "Comm Fail": None}

    payload.pop("_comments")
    payload.update(identity=META["identity"], name=META["name"])
    template.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SchemaError, match="quality_codes"):
        tsdive.read_meta_json(template)

    payload["quality_codes"] = dict.fromkeys(payload["quality_codes"], "BAD")
    template.write_text(json.dumps(payload), encoding="utf-8")
    out = tsdive.ingest(_three_state_export(tmp_path), out=tmp_path / "s.parquet",
                        meta=tsdive.read_meta_json(template), timestamp_col="ts",
                        value_col="v", assume_quality="GOOD")
    counts = {k.value: n for k, n in tsdive.profile(out).physics.severity_counts.items()}
    assert counts == {"GOOD": 2, "UNCERTAIN": 0, "BAD": 4}


def test_a_template_adds_value_states_after_the_quality_codes(tmp_path):
    template = tsdive.init_tag_meta(_pi_export(tmp_path), out=tmp_path / "t.json",
                                    timestamp_col="ts", value_col="v", quality_col="q")
    codes = json.loads(template.read_text(encoding="utf-8"))["quality_codes"]
    assert codes == {"Bad": "BAD", "Good": "GOOD", "I/O Timeout": None}


def test_a_column_of_string_states_adds_nothing_to_the_template(tmp_path):
    """A MODE tag's values are all states, so none of them is listed as a quality code."""
    path = tmp_path / "mode.csv"
    pd.DataFrame({"ts": AWARE, "v": ["R0", "R1", "R1"]}).to_csv(path, index=False)
    template = tsdive.init_tag_meta(path, out=tmp_path / "t.json", timestamp_col="ts",
                                    value_col="v")
    assert json.loads(template.read_text(encoding="utf-8"))["quality_codes"] is None


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


# ------------------------------------------------- several tags in one export

LONG_TAGS = {"PI101.PV": 3.5, "FI102.PV": 40.0, "TI103.PV": 181.0}
LONG_INSTANTS = pd.date_range("2026-03-01", periods=288, freq="5min", tz="UTC")


def _long_csv(tmp_path, *, sort_by="Timestamp", name="long.csv"):
    """Three tags x 288 rows at 5 min in one Tag/Timestamp/Value/Status export."""
    rows = [
        {"Tag": tag, "Timestamp": t.isoformat(), "Value": level + 0.01 * k, "Status": "Good"}
        for tag, level in LONG_TAGS.items()
        for k, t in enumerate(LONG_INSTANTS)
    ]
    path = tmp_path / name
    pd.DataFrame(rows).sort_values(sort_by, kind="stable").to_csv(path, index=False)
    return path


@pytest.mark.parametrize("sort_by", ["Timestamp", "Tag"], ids=["time-sorted", "tag-sorted"])
def test_a_merged_multi_tag_export_is_refused(tmp_path, capsys, sort_by):
    out = tmp_path / "merged.parquet"
    rc = cmd_ingest(
        [
            str(_long_csv(tmp_path, sort_by=sort_by)),
            "--out",
            str(out),
            "--meta",
            str(_meta_file(tmp_path)),
            "--timestamp-col",
            "Timestamp",
            "--value-col",
            "Value",
            "--assume-quality",
            "GOOD",
        ]
    )
    err = capsys.readouterr().err
    assert rc == 3
    assert "[SchemaError] long.csv: 864 rows share a timestamp" in err
    assert "column 'Tag' splits the rows into 3 overlapping series" in err
    assert "pass --tag-col Tag to write one archive per tag" in err
    assert not out.exists()


def test_a_merged_export_on_offset_clocks_is_refused_naming_ingest_long(tmp_path):
    """Tags sampled at different instants never share a timestamp but still run backwards."""
    frame = pd.DataFrame(
        {
            "Tag": ["A"] * 3 + ["B"] * 3,
            "Timestamp": [
                "2026-03-01T00:00:00Z",
                "2026-03-01T00:01:00Z",
                "2026-03-01T00:02:00Z",
                "2026-03-01T00:00:30Z",
                "2026-03-01T00:01:30Z",
                "2026-03-01T00:02:30Z",
            ],
            "Value": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
        }
    )
    frame.to_csv(tmp_path / "offset.csv", index=False)
    with pytest.raises(
        SchemaError,
        match=r"the timestamps run backwards at data row 4, and column 'Tag' splits the rows "
        r"into 2 overlapping series.*call ingest_long with tag_col='Tag'",
    ):
        tsdive.ingest(
            tmp_path / "offset.csv",
            out=tmp_path / "offset.parquet",
            meta=tsdive.read_meta_json(_meta_file(tmp_path)),
            timestamp_col="Timestamp",
            value_col="Value",
            assume_quality="GOOD",
        )
    assert not (tmp_path / "offset.parquet").exists()


def test_a_repeated_timestamp_in_a_single_tag_export_still_ingests(tmp_path):
    """Two events at one instant: a row id and a batch column do not read as tag columns."""
    stamps = [
        "2024-03-01T00:00:00Z",
        "2024-03-01T00:01:00Z",
        "2024-03-01T00:01:00Z",
        "2024-03-01T00:02:00Z",
    ]
    pd.DataFrame(
        {
            "ts": stamps,
            "v": [50.0, 51.0, 51.5, 52.0],
            "q": ["GOOD"] * 4,
            "row": [1, 2, 3, 4],
            "batch": ["B1", "B1", "B2", "B2"],
        }
    ).to_csv(tmp_path / "events.csv", index=False)
    out = tsdive.ingest(
        tmp_path / "events.csv",
        out=tmp_path / "events.parquet",
        meta=tsdive.read_meta_json(_meta_file(tmp_path)),
        timestamp_col="ts",
        value_col="v",
        quality_col="q",
    )
    assert list(pd.read_parquet(out)["value"]) == [50.0, 51.0, 51.5, 52.0]


def test_a_backwards_export_is_refused_before_writing(tmp_path):
    src = _csv(tmp_path, stamps=[AWARE[0], AWARE[2], AWARE[1]])
    stamp = r"data row 3 \(2024-03-01T00:01:00\+00:00\)"
    with pytest.raises(NonMonotonicIndex, match=stamp) as info:
        tsdive.ingest(
            src,
            out=tmp_path / "back.parquet",
            meta=tsdive.read_meta_json(_meta_file(tmp_path)),
            timestamp_col="ts",
            value_col="v",
            quality_col="q",
        )
    assert info.value.offending_positions == [2]
    assert not (tmp_path / "back.parquet").exists()


# ---------------------------------------------------------------- wide exports

WIDE_TAGS = ("FIC101.PV", "TIC201.PV", "PIC301.PV")
WIDE_CODES = ("Good", "Bad", "Questionable")
# One hour of real instants at 60 s across the 2024-03-31 01:00Z spring
# forward. Written as naive Europe/London wall-clock, the column runs
# 00:30..00:59 and then 02:00..02:30.
WIDE_INSTANTS = pd.date_range("2024-03-31T00:30:00Z", "2024-03-31T01:30:00Z", freq="60s")


def _wide_csv(tmp_path, *, tags=WIDE_TAGS, quality=True, suffix="_q", name="wide.csv"):
    local = WIDE_INSTANTS.tz_convert("Europe/London").strftime("%Y-%m-%d %H:%M:%S")
    rows: dict[str, list] = {"ts": list(local)}
    n = len(WIDE_INSTANTS)
    for i, tag in enumerate(tags):
        rows[tag] = [50.0 * (i + 1) + k for k in range(n)]
        if quality:
            rows[f"{tag}{suffix}"] = [WIDE_CODES[(k + i) % 3] for k in range(n)]
    path = tmp_path / name
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _wide_meta_dir(tmp_path, tags=WIDE_TAGS, codes=None):
    codes = codes or {"Good": "GOOD", "Bad": "BAD", "Questionable": "UNCERTAIN"}
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir(exist_ok=True)
    for tag in tags:
        payload = {
            **META,
            "identity": {"source_id": "plant1", "point_id": tag},
            "name": tag,
            "quality_codes": codes,
        }
        (meta_dir / f"{safe_filename(tag)}.json").write_text(json.dumps(payload), encoding="utf-8")
    return meta_dir


def test_wide_export_writes_one_archive_per_column_across_dst(tmp_path):
    out_dir = tmp_path / "archive"
    written = tsdive.ingest_wide(
        _wide_csv(tmp_path),
        out_dir=out_dir,
        meta_dir=_wide_meta_dir(tmp_path),
        timestamp_col="ts",
        quality_suffix="_q",
        tz="Europe/London",
    )
    assert [p.name for p in written] == [f"{tag}.parquet" for tag in WIDE_TAGS]
    assert all(p.exists() and p.parent == out_dir for p in written)

    meta = meta_from_parquet(written[1])
    assert meta.identity.point_id == "TIC201.PV"
    assert meta.quality_assumed is False
    frame = pd.read_parquet(written[1])
    assert list(frame["timestamp"]) == list(WIDE_INSTANTS)
    assert frame["value"].iloc[0] == 100.0

    p = tsdive.profile(written[1], tz="Europe/London")
    assert p.physics.coverage.coverage == 1.0
    assert "coverage 1.000" in p.render()
    (transition,) = p.physics.timestamp_audit.dst_transitions
    assert transition.instant_utc == pd.Timestamp("2024-03-31 01:00:00+00:00")
    assert "DST (Europe/London) 2024-03-31 01:00:00Z  +0h -> +1h" in p.render()
    assert sum(p.physics.severity_counts.values()) == len(WIDE_INSTANTS)
    assert p.physics.unmapped_quality_codes == []


# 48 h at 5 min across the 2026-10-25 01:00Z fall back in Berlin: written as
# naive wall clock, 02:00..02:55 appears twice and the column has 577 rows.
AUTUMN_INSTANTS = pd.date_range(
    pd.Timestamp("2026-10-24 00:00", tz="Europe/Berlin").tz_convert("UTC"),
    periods=577,
    freq="5min",
)


def _autumn_csv(tmp_path, local=None, name="autumn.csv"):
    if local is None:
        local = list(AUTUMN_INSTANTS.tz_convert("Europe/Berlin").strftime("%Y-%m-%d %H:%M:%S"))
    path = tmp_path / name
    pd.DataFrame({"ts": local, "FIC101.PV": [50.0 + 0.01 * k for k in range(len(local))]}).to_csv(
        path, index=False
    )
    return path


def _ingest_autumn(tmp_path, src):
    return tsdive.ingest_wide(
        src,
        out_dir=tmp_path / "archive",
        meta_dir=_wide_meta_dir(tmp_path, tags=("FIC101.PV",)),
        timestamp_col="ts",
        tz="Europe/Berlin",
        assume_quality="GOOD",
    )


def test_the_repeated_autumn_hour_is_placed_by_row_order(tmp_path):
    (written,) = _ingest_autumn(tmp_path, _autumn_csv(tmp_path))
    stamps = pd.read_parquet(written)["timestamp"]
    assert list(stamps) == list(AUTUMN_INSTANTS)
    assert set(stamps.diff().dt.total_seconds().iloc[1:]) == {300.0}


def _autumn_local():
    return list(AUTUMN_INSTANTS.tz_convert("Europe/Berlin").strftime("%Y-%m-%d %H:%M:%S"))


def _one_sample_in_the_hour():
    local = _autumn_local()
    i = local.index("2026-10-25 02:00:00")
    return [*local[:i], "2026-10-25 02:30:00", *local[i + 24 :]]


def _a_repeated_row_in_the_hour():
    local = _autumn_local()
    i = local.index("2026-10-25 02:10:00")
    return [*local[: i + 1], local[i], *local[i + 1 :]]


def _the_hour_sorted_by_wall_clock():
    local = _autumn_local()
    i = local.index("2026-10-25 02:00:00")
    first, second = local[i : i + 12], local[i + 12 : i + 24]
    paired = [t for pair in zip(first, second, strict=True) for t in pair]
    return [*local[:i], *paired, *local[i + 24 :]]


def _a_skipped_spring_hour():
    return list(pd.date_range("2026-03-29 01:00", "2026-03-29 03:00", freq="5min").astype(str))


@pytest.mark.parametrize(
    ("stamps", "message"),
    [
        (
            _one_sample_in_the_hour,
            r"ts: 2026-10-25 02:30:00 falls in the hour Europe/Berlin repeats when the clocks "
            r"go back, and the export holds that hour once",
        ),
        (
            _a_repeated_row_in_the_hour,
            r"ts: the rows of the hour Europe/Berlin repeats when the clocks go back, from "
            r"2026-10-25 02:00:00, are out of time order or repeat",
        ),
        (
            _the_hour_sorted_by_wall_clock,
            r"ts: the rows of the hour Europe/Berlin repeats when the clocks go back, from "
            r"2026-10-25 02:00:00, are out of time order or repeat",
        ),
        (
            _a_skipped_spring_hour,
            r"ts: 2026-03-29 02:00:00 does not exist in Europe/Berlin, because the clocks "
            r"skip it when they go forward",
        ),
    ],
    ids=["one-sample", "repeated-row", "wall-clock-order", "spring"],
)
def test_local_times_the_row_order_cannot_place_are_refused(tmp_path, stamps, message):
    tail = ".*; export the stretch around the change with UTC offsets$"
    with pytest.raises(SchemaError, match=message + tail):
        _ingest_autumn(tmp_path, _autumn_csv(tmp_path, stamps()))
    assert not (tmp_path / "archive").exists()


def test_wide_ingest_takes_the_date_order_flags(tmp_path, capsys):
    src = tmp_path / "eu_wide.csv"
    pd.DataFrame({"ts": EU_STAMPS, "FIC101.PV": [1.0, 2.0, 3.0]}).to_csv(src, index=False)
    meta_dir = _wide_meta_dir(tmp_path, tags=["FIC101.PV"])
    argv = [str(src), "--wide", "--out", str(tmp_path / "archive"), "--meta-dir", str(meta_dir),
            "--timestamp-col", "ts", "--tz", "Europe/Paris", "--assume-quality", "GOOD"]
    assert cmd_ingest(argv) != 0
    assert "'01/02/2026 08:00:00' reads as day 01" in capsys.readouterr().err
    assert cmd_ingest([*argv, "--dayfirst"]) == 0
    stamps = pd.read_parquet(tmp_path / "archive" / "FIC101.PV.parquet")["timestamp"]
    assert stamps.iloc[0] == pd.Timestamp("2026-02-01 07:00:00+00:00")
    fmt = ["--timestamp-format", "%d/%m/%Y %H:%M:%S", "--overwrite"]
    assert cmd_ingest([*argv, *fmt]) == 0


def test_wide_tags_subset_with_assumed_quality(tmp_path):
    written = tsdive.ingest_wide(
        _wide_csv(tmp_path, quality=False),
        out_dir=tmp_path / "archive",
        meta_dir=_wide_meta_dir(tmp_path),
        timestamp_col="ts",
        tags=["TIC201.PV"],
        tz="Europe/London",
        assume_quality="good",
    )
    assert [p.name for p in written] == ["TIC201.PV.parquet"]
    assert sorted(q.name for q in (tmp_path / "archive").iterdir()) == ["TIC201.PV.parquet"]
    meta = meta_from_parquet(written[0])
    assert meta.quality_assumed is True
    assert set(pd.read_parquet(written[0])["quality"]) == {"GOOD"}


def test_wide_missing_meta_file_names_tag_and_path(tmp_path):
    meta_dir = _wide_meta_dir(tmp_path)
    (meta_dir / "PIC301.PV.json").unlink()
    with pytest.raises(SchemaError, match=r"'PIC301\.PV'.*meta/PIC301\.PV\.json") as info:
        tsdive.ingest_wide(
            _wide_csv(tmp_path),
            out_dir=tmp_path / "archive",
            meta_dir=meta_dir,
            timestamp_col="ts",
            quality_suffix="_q",
            tz="Europe/London",
        )
    assert "no metadata file" in str(info.value)
    assert not (tmp_path / "archive").exists()


def test_wide_missing_quality_column_is_refused(tmp_path):
    with pytest.raises(SchemaError, match=r"no quality column 'FIC101\.PV_q' for tag 'FIC101\.PV'"):
        tsdive.ingest_wide(
            _wide_csv(tmp_path, quality=False),
            out_dir=tmp_path / "archive",
            meta_dir=_wide_meta_dir(tmp_path),
            timestamp_col="ts",
            quality_suffix="_q",
            tz="Europe/London",
        )


def test_wide_suffix_and_assume_quality_together_are_refused(tmp_path):
    with pytest.raises(SchemaError, match="assume_quality was given with quality_suffix"):
        tsdive.ingest_wide(
            _wide_csv(tmp_path),
            out_dir=tmp_path / "archive",
            meta_dir=_wide_meta_dir(tmp_path),
            timestamp_col="ts",
            quality_suffix="_q",
            tz="Europe/London",
            assume_quality="GOOD",
        )


def test_wide_naive_stamps_without_tz_are_refused(tmp_path):
    with pytest.raises(SchemaError, match="naive"):
        tsdive.ingest_wide(
            _wide_csv(tmp_path),
            out_dir=tmp_path / "archive",
            meta_dir=_wide_meta_dir(tmp_path),
            timestamp_col="ts",
            quality_suffix="_q",
        )
    assert not (tmp_path / "archive").exists()


def test_wide_backwards_rows_are_refused_before_any_archive(tmp_path):
    src = _wide_csv(tmp_path)
    frame = pd.read_csv(src)
    frame.iloc[::-1].to_csv(src, index=False)
    with pytest.raises(NonMonotonicIndex, match=r"wide\.csv: the timestamp of data row 2"):
        tsdive.ingest_wide(
            src,
            out_dir=tmp_path / "archive",
            meta_dir=_wide_meta_dir(tmp_path),
            timestamp_col="ts",
            quality_suffix="_q",
            tz="Europe/London",
        )
    assert not (tmp_path / "archive").exists()


def test_wide_unsafe_tag_name_lands_as_a_safe_filename(tmp_path):
    tags = ("FIC101/PV:1",)
    assert safe_filename(tags[0]) == "FIC101_PV_1"
    written = tsdive.ingest_wide(
        _wide_csv(tmp_path, tags=tags),
        out_dir=tmp_path / "archive",
        meta_dir=_wide_meta_dir(tmp_path, tags=tags),
        timestamp_col="ts",
        quality_suffix="_q",
        tz="Europe/London",
    )
    assert [p.name for p in written] == ["FIC101_PV_1.parquet"]
    assert meta_from_parquet(written[0]).identity.point_id == "FIC101/PV:1"


def test_wide_tags_colliding_after_safe_filename_are_refused(tmp_path):
    tags = ("A/B", "A:B")
    with pytest.raises(ValueError, match=r"'A/B' and 'A:B' both map to the file name 'A_B'"):
        tsdive.ingest_wide(
            _wide_csv(tmp_path, tags=tags),
            out_dir=tmp_path / "archive",
            meta_dir=_wide_meta_dir(tmp_path, tags=("A_B",)),
            timestamp_col="ts",
            quality_suffix="_q",
            tz="Europe/London",
        )
    assert not (tmp_path / "archive").exists()


# --------------------------------------- long exports, one row per tag and time

LONG_ARGS = ["--timestamp-col", "Timestamp", "--value-col", "Value", "--quality-col", "Status"]


def test_tag_col_templates_then_archives_one_per_tag(tmp_path, capsys):
    src = _long_csv(tmp_path)
    meta_dir = tmp_path / "meta"
    rc = cmd_ingest(
        [str(src), "--tag-col", "Tag", *LONG_ARGS, "--init-meta", str(meta_dir),
         "--source-id", "plant1"]
    )
    assert rc == 0
    assert capsys.readouterr().out.splitlines() == [
        f"wrote     {(meta_dir / 'PI101.PV.json').as_posix()}",
        f"wrote     {(meta_dir / 'FI102.PV.json').as_posix()}",
        f"wrote     {(meta_dir / 'TI103.PV.json').as_posix()}",
    ]
    template = json.loads((meta_dir / "FI102.PV.json").read_text(encoding="utf-8"))
    assert template["identity"] == {"source_id": "plant1", "point_id": "FI102.PV"}
    assert template["quality_codes"] == {"Good": "GOOD"}

    out_dir = tmp_path / "archive"
    rc = cmd_ingest(
        [str(src), "--tag-col", "Tag", *LONG_ARGS, "--out", str(out_dir), "--meta-dir",
         str(meta_dir)]
    )
    assert rc == 0
    assert capsys.readouterr().out.splitlines() == [
        f"wrote     {(out_dir / 'PI101.PV.parquet').as_posix()}",
        f"wrote     {(out_dir / 'FI102.PV.parquet').as_posix()}",
        f"wrote     {(out_dir / 'TI103.PV.parquet').as_posix()}",
    ]
    for tag, level in LONG_TAGS.items():
        frame = pd.read_parquet(out_dir / f"{tag}.parquet")
        assert list(frame["timestamp"]) == list(LONG_INSTANTS)
        assert frame["value"].iloc[0] == level
        assert meta_from_parquet(out_dir / f"{tag}.parquet").identity.point_id == tag


def test_tag_sorted_and_time_sorted_exports_give_the_same_archives(tmp_path):
    meta_dir = _wide_meta_dir(tmp_path, tags=tuple(LONG_TAGS), codes={"Good": "GOOD"})
    frames = {}
    for sort_by in ("Timestamp", "Tag"):
        written = tsdive.ingest_long(
            _long_csv(tmp_path, sort_by=sort_by, name=f"{sort_by}.csv"),
            out_dir=tmp_path / sort_by,
            meta_dir=meta_dir,
            tag_col="Tag",
            timestamp_col="Timestamp",
            value_col="Value",
            quality_col="Status",
        )
        frames[sort_by] = {p.name: pd.read_parquet(p) for p in written}
    assert sorted(frames["Tag"]) == sorted(frames["Timestamp"])
    for name, frame in frames["Tag"].items():
        pd.testing.assert_frame_equal(frame, frames["Timestamp"][name])


def test_tag_col_writes_nothing_when_one_tag_has_no_metadata(tmp_path):
    with pytest.raises(SchemaError, match=r"tag 'TI103\.PV': no metadata file .*init_long_meta"):
        tsdive.ingest_long(
            _long_csv(tmp_path),
            out_dir=tmp_path / "archive",
            meta_dir=_wide_meta_dir(tmp_path, tags=("PI101.PV", "FI102.PV")),
            tag_col="Tag",
            timestamp_col="Timestamp",
            value_col="Value",
            assume_quality="GOOD",
        )
    assert not (tmp_path / "archive").exists()


def test_tag_col_takes_a_subset_and_refuses_an_absent_tag(tmp_path):
    kwargs = {
        "out_dir": tmp_path / "archive",
        "meta_dir": _wide_meta_dir(tmp_path, tags=tuple(LONG_TAGS)),
        "tag_col": "Tag",
        "timestamp_col": "Timestamp",
        "value_col": "Value",
        "assume_quality": "GOOD",
    }
    src = _long_csv(tmp_path)
    written = tsdive.ingest_long(src, tags=["TI103.PV"], **kwargs)
    assert [p.name for p in written] == ["TI103.PV.parquet"]
    with pytest.raises(
        SchemaError,
        match=r"column 'Tag' holds no tag 'TI104\.PV' named in tags; it holds PI101\.PV, "
        r"FI102\.PV, TI103\.PV",
    ):
        tsdive.ingest_long(src, tags=["TI104.PV"], overwrite=True, **kwargs)


def test_tag_col_refuses_rows_without_a_tag(tmp_path):
    src = _long_csv(tmp_path)
    frame = pd.read_csv(src)
    frame.loc[[3, 7], "Tag"] = None
    frame.to_csv(src, index=False)
    with pytest.raises(SchemaError, match="2 rows have no tag in column 'Tag'"):
        tsdive.init_long_meta(src, out_dir=tmp_path / "meta", source_id="plant1", tag_col="Tag",
                              timestamp_col="Timestamp", value_col="Value")


def test_tag_col_types_each_tags_values_on_their_own(tmp_path):
    """One tag's digital state leaves the other tags' archives numeric."""
    src = _long_csv(tmp_path)
    frame = pd.read_csv(src)
    frame["Value"] = frame["Value"].astype(object)
    frame.loc[frame.index[frame["Tag"] == "PI101.PV"][5], "Value"] = "Shutdown"
    frame.to_csv(src, index=False)
    meta_dir = _wide_meta_dir(tmp_path, tags=tuple(LONG_TAGS), codes={"Shutdown": "BAD"})
    written = tsdive.ingest_long(
        src,
        out_dir=tmp_path / "archive",
        meta_dir=meta_dir,
        tag_col="Tag",
        timestamp_col="Timestamp",
        value_col="Value",
        assume_quality="GOOD",
    )
    dtypes = {p.stem: str(pd.read_parquet(p)["value"].dtype) for p in written}
    assert dtypes == {"PI101.PV": "object", "FI102.PV": "float64", "TI103.PV": "float64"}

    templates = tsdive.init_long_meta(src, out_dir=tmp_path / "templates", source_id="plant1",
                                      tag_col="Tag", timestamp_col="Timestamp",
                                      value_col="Value")
    codes = {p.stem: json.loads(p.read_text(encoding="utf-8"))["quality_codes"]
             for p in templates}
    assert codes == {"PI101.PV": {"Shutdown": None}, "FI102.PV": None, "TI103.PV": None}


def test_tag_col_refuses_a_tag_that_runs_backwards_naming_the_file_row(tmp_path):
    src = _long_csv(tmp_path, sort_by="Tag")
    frame = pd.read_csv(src)
    frame.iloc[[300, 301]] = frame.iloc[[301, 300]].to_numpy()
    frame.to_csv(src, index=False)
    with pytest.raises(NonMonotonicIndex, match=r"long\.csv, tag 'PI101\.PV': the timestamp "
                       r"of data row 302 "):
        tsdive.ingest_long(
            src,
            out_dir=tmp_path / "archive",
            meta_dir=_wide_meta_dir(tmp_path, tags=tuple(LONG_TAGS)),
            tag_col="Tag",
            timestamp_col="Timestamp",
            value_col="Value",
            assume_quality="GOOD",
        )
    assert not (tmp_path / "archive").exists()


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--wide"], "--tag-col does not apply with --wide"),
        (["--quality-suffix", "_q"], "--quality-suffix requires --wide"),
        (["--meta", "m.json"], "--meta does not apply with --tag-col; use --meta-dir"),
        ([], "--tag-col requires --meta-dir"),
    ],
)
def test_tag_col_flags_that_belong_to_another_form_are_usage_errors(
    tmp_path, capsys, extra, message
):
    with pytest.raises(SystemExit) as info:
        cmd_ingest(["long.csv", "--tag-col", "Tag", "--out", str(tmp_path / "x"), *extra])
    assert info.value.code == 2
    assert message in capsys.readouterr().err


# ---------------------------------------------------------- metadata templates

TEMPLATE_KEYS = [
    "identity",
    "name",
    "unit_raw",
    "unit_canonical",
    "eng_range_zero",
    "eng_range_span",
    "sample_rate_s",
    "retrieval_mode",
    "asset",
    "loop_id",
    "role",
    "quality_codes",
    "quality_assumed",
]


def test_init_meta_writes_one_template_per_tag_in_schema_order(tmp_path):
    written = tsdive.init_meta(
        _wide_csv(tmp_path),
        out_dir=tmp_path / "meta",
        source_id="plant1",
        timestamp_col="ts",
        quality_suffix="_q",
    )
    assert [p.name for p in written] == [f"{tag}.json" for tag in WIDE_TAGS]
    text = written[0].read_text(encoding="utf-8")
    assert text.endswith("}\n") and "\n  \"name\"" in text
    payload = json.loads(text)
    assert list(payload) == TEMPLATE_KEYS
    assert payload["identity"] == {"source_id": "plant1", "point_id": "FIC101.PV"}
    assert payload["name"] == "FIC101.PV"
    assert payload["quality_codes"] == {"Bad": "BAD", "Good": "GOOD", "Questionable": None}
    assert all(payload[k] is None for k in TEMPLATE_KEYS[2:11])
    assert payload["quality_assumed"] is None


def test_init_meta_without_quality_columns_leaves_quality_codes_null(tmp_path):
    (written,) = tsdive.init_meta(
        _wide_csv(tmp_path, quality=False),
        out_dir=tmp_path / "meta",
        source_id="plant1",
        timestamp_col="ts",
        tags=["TIC201.PV"],
    )
    assert json.loads(written.read_text(encoding="utf-8"))["quality_codes"] is None


def test_init_meta_does_not_overwrite_a_template(tmp_path):
    kwargs = dict(out_dir=tmp_path / "meta", source_id="plant1", timestamp_col="ts")
    src = _wide_csv(tmp_path, quality=False)
    tsdive.init_meta(src, **kwargs)
    with pytest.raises(FileExistsError, match=r"FIC101\.PV\.json already exists"):
        tsdive.init_meta(src, **kwargs)
    assert len(tsdive.init_meta(src, overwrite=True, **kwargs)) == 3


def test_filled_templates_feed_ingest_wide(tmp_path):
    src = _wide_csv(tmp_path)
    meta_dir = tmp_path / "meta"
    templates = tsdive.init_meta(
        src, out_dir=meta_dir, source_id="plant1", timestamp_col="ts", quality_suffix="_q"
    )
    for path in templates:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["quality_codes"]["Questionable"] = "UNCERTAIN"
        payload["sample_rate_s"] = 60.0
        path.write_text(json.dumps(payload), encoding="utf-8")
    written = tsdive.ingest_wide(
        src,
        out_dir=tmp_path / "archive",
        meta_dir=meta_dir,
        timestamp_col="ts",
        quality_suffix="_q",
        tz="Europe/London",
    )
    assert len(written) == 3
    p = tsdive.profile(written[0])
    assert p.physics.unmapped_quality_codes == []
    assert sum(p.physics.severity_counts.values()) == len(WIDE_INSTANTS)
    assert p.meta.sample_rate_s == 60.0


def test_template_with_an_unmapped_code_is_refused_at_read(tmp_path):
    src = _wide_csv(tmp_path)
    meta_dir = tmp_path / "meta"
    tsdive.init_meta(
        src, out_dir=meta_dir, source_id="plant1", timestamp_col="ts", quality_suffix="_q"
    )
    with pytest.raises(SchemaError, match="quality_codes maps 'Questionable' to None"):
        tsdive.read_meta_json(meta_dir / "FIC101.PV.json")
    with pytest.raises(SchemaError, match=r"FIC101\.PV\.json: quality_codes maps 'Questionable'"):
        tsdive.ingest_wide(
            src,
            out_dir=tmp_path / "archive",
            meta_dir=meta_dir,
            timestamp_col="ts",
            quality_suffix="_q",
            tz="Europe/London",
        )
    assert not (tmp_path / "archive").exists()


def test_python_messages_name_keywords_where_the_cli_names_flags(tmp_path, capsys):
    src = _wide_csv(tmp_path)
    meta_dir = _wide_meta_dir(tmp_path)
    with pytest.raises(SchemaError, match=r"no quality_suffix and no assume_quality; name the "
                       r"suffix .* assume_quality=GOOD\|UNCERTAIN\|BAD, which"):
        tsdive.ingest_wide(src, out_dir=tmp_path / "a", meta_dir=meta_dir, timestamp_col="ts",
                           tz="Europe/London")
    rc = cmd_ingest([str(src), "--wide", "--out", str(tmp_path / "a"), "--meta-dir",
                     str(meta_dir), "--timestamp-col", "ts", "--tz", "Europe/London"])
    assert rc != 0
    assert "no --quality-suffix and no --assume-quality" in capsys.readouterr().err

    naive = _csv(tmp_path, stamps=NAIVE, name="naive.csv")
    with pytest.raises(SchemaError, match="pass tz=<IANA zone> to state"):
        tsdive.ingest(naive, out=tmp_path / "n.parquet",
                      meta=tsdive.read_meta_json(_meta_file(tmp_path)), timestamp_col="ts",
                      value_col="v", quality_col="q")


def test_a_single_tag_template_leaves_the_tag_to_the_reader(tmp_path, capsys):
    src = tmp_path / "pi.csv"
    pd.DataFrame(
        {
            "ts": AWARE,
            "v": [50.0, 51.0, 52.0],
            "q": ["Good", "Questionable", "Good"],
            "note": ["", "", ""],
        }
    ).to_csv(src, index=False)
    meta_path = tmp_path / "meta" / "fic.json"
    argv = [str(src), "--init-meta", str(meta_path), "--timestamp-col", "ts", "--value-col",
            "v", "--quality-col", "q"]
    assert cmd_ingest(argv) == 0
    assert capsys.readouterr().out.strip() == f"wrote     {meta_path.as_posix()}"
    template = json.loads(meta_path.read_text(encoding="utf-8"))
    assert list(template) == ["_comments", *TEMPLATE_KEYS]
    assert template["identity"] == {"source_id": None, "point_id": None}
    assert (template["name"], template["unit_raw"]) == (None, None)
    assert template["quality_codes"] == {"Good": "GOOD", "Questionable": None}
    assert template["_comments"]["columns"] == (
        "timestamp 'ts', value 'v', quality 'q'; the export has ts, v, q, note"
    )
    assert set(template["_comments"]) == {*TEMPLATE_KEYS, "columns"}
    with pytest.raises(SchemaError, match=r"identity\.source_id must be a non-empty string"):
        tsdive.read_meta_json(meta_path)

    assert cmd_ingest(argv) == 2
    assert "already exists; pass --overwrite" in capsys.readouterr().err
    assert cmd_ingest([*argv, "--overwrite"]) == 0
    capsys.readouterr()

    template.update(
        identity={"source_id": "plant1", "point_id": "FIC101.PV"},
        name="FIC-101 flow",
        unit_raw="m3/h",
        quality_codes={"Good": "GOOD", "Questionable": "UNCERTAIN"},
    )
    meta_path.write_text(json.dumps(template), encoding="utf-8")
    out = tmp_path / "fic.parquet"
    assert cmd_ingest([str(src), "--out", str(out), "--meta", str(meta_path), "--timestamp-col",
                       "ts", "--value-col", "v", "--quality-col", "q"]) == 0
    assert meta_from_parquet(out).unit_raw == "m3/h"


def test_a_single_tag_template_takes_no_ingest_flags(tmp_path, capsys):
    src = _csv(tmp_path, stamps=AWARE)
    with pytest.raises(SystemExit) as info:
        cmd_ingest([str(src), "--init-meta", str(tmp_path / "m.json"), "--out", "x.parquet"])
    assert info.value.code == 2
    assert "--out does not apply with --init-meta" in capsys.readouterr().err


def test_a_misspelt_comment_key_is_refused(tmp_path):
    path = _meta_file(tmp_path, _comment={"unit_raw": "m3/h"})
    with pytest.raises(SchemaError, match="unknown key '_comment'; did you mean '_comments'"):
        tsdive.read_meta_json(path)


def test_a_declared_retrieval_mode_reaches_every_read(tmp_path, capsys):
    out = tmp_path / "interp.parquet"
    tsdive.ingest(
        _csv(tmp_path, stamps=AWARE),
        out=out,
        meta=tsdive.read_meta_json(_meta_file(tmp_path, retrieval_mode="INTERPOLATED")),
        timestamp_col="ts",
        value_col="v",
        quality_col="q",
    )
    assert meta_from_parquet(out).retrieval_mode == tsdive.RetrievalMode.INTERPOLATED
    p = tsdive.profile(out)
    assert p.window.contract.retrieval_mode == tsdive.RetrievalMode.INTERPOLATED
    assert "contract  TIME_WEIGHTED  INTERPOLATED  NONE" in p.render()
    assert cmd_profile([str(out), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["contract"]["retrieval_mode"] == "INTERPOLATED"

    plain = tmp_path / "recorded.parquet"
    tsdive.ingest(_csv(tmp_path, stamps=AWARE, name="plain.csv"), out=plain,
                  meta=tsdive.read_meta_json(_meta_file(tmp_path)), timestamp_col="ts",
                  value_col="v", quality_col="q")
    recorded = tsdive.profile(plain).window.contract
    assert recorded.retrieval_mode == tsdive.RetrievalMode.RECORDED
    assert recorded.digest() != p.window.contract.digest()


def test_an_unknown_retrieval_mode_is_refused(tmp_path):
    with pytest.raises(SchemaError, match="retrieval_mode 'interpolated' is not one of "
                       "RECORDED, INTERPOLATED"):
        tsdive.read_meta_json(_meta_file(tmp_path, retrieval_mode="interpolated"))


# ------------------------------------------------------------- ingest --json


def _json_out(capsys) -> dict:
    return json.loads(capsys.readouterr().out)


def test_ingest_json_lists_the_archive_it_wrote(tmp_path, capsys):
    out = tmp_path / "FIC101.PV.parquet"
    argv = [str(_csv(tmp_path, stamps=AWARE)), "--out", str(out), "--meta",
            str(_meta_file(tmp_path)), "--timestamp-col", "ts", "--value-col", "v",
            "--quality-col", "q", "--json"]
    assert cmd_ingest(argv) == 0
    assert _json_out(capsys) == {
        "result_kind": "ingest",
        "tsdive_version": tsdive.__version__,
        "form": "single",
        "archives": [
            {
                "path": out.as_posix(),
                "tag": "plant1:FIC101.PV",
                "identity": {"source_id": "plant1", "point_id": "FIC101.PV"},
                "rows": 3,
                "first": "2024-03-01T00:00:00+00:00",
                "last": "2024-03-01T00:02:00+00:00",
                "quality_source": "column",
                "assumed_quality": None,
            }
        ],
    }


def test_ingest_json_on_a_wide_export_names_assumed_quality(tmp_path, capsys):
    out_dir = tmp_path / "archive"
    argv = [str(_wide_csv(tmp_path, quality=False)), "--wide", "--out", str(out_dir),
            "--meta-dir", str(_wide_meta_dir(tmp_path)), "--timestamp-col", "ts",
            "--tz", "Europe/London", "--assume-quality", "good", "--json"]
    assert cmd_ingest(argv) == 0
    captured = capsys.readouterr()
    doc = json.loads(captured.out)
    assert doc["form"] == "wide"
    assert [a["identity"]["point_id"] for a in doc["archives"]] == list(WIDE_TAGS)
    assert {(a["quality_source"], a["assumed_quality"]) for a in doc["archives"]} == {
        ("assumed", "GOOD")
    }
    assert {a["rows"] for a in doc["archives"]} == {len(WIDE_INSTANTS)}
    assert doc["archives"][0]["first"] == WIDE_INSTANTS[0].isoformat()
    assert captured.err.startswith("warning: quality assumed GOOD")


def test_ingest_json_on_a_long_export_and_its_templates(tmp_path, capsys):
    src, meta_dir, out_dir = _long_csv(tmp_path), tmp_path / "meta", tmp_path / "archive"
    init = [str(src), "--tag-col", "Tag", *LONG_ARGS, "--init-meta", str(meta_dir),
            "--source-id", "plant1", "--json"]
    assert cmd_ingest(init) == 0
    assert _json_out(capsys) == {
        "result_kind": "ingest",
        "tsdive_version": tsdive.__version__,
        "form": "long",
        "templates": [(meta_dir / f"{tag}.json").as_posix() for tag in LONG_TAGS],
    }
    argv = [str(src), "--tag-col", "Tag", *LONG_ARGS, "--out", str(out_dir), "--meta-dir",
            str(meta_dir), "--json"]
    assert cmd_ingest(argv) == 0
    doc = _json_out(capsys)
    assert [a["path"] for a in doc["archives"]] == [
        (out_dir / f"{tag}.parquet").as_posix() for tag in LONG_TAGS
    ]
    assert {a["quality_source"] for a in doc["archives"]} == {"column"}
    assert {a["last"] for a in doc["archives"]} == {LONG_INSTANTS[-1].isoformat()}


def test_ingest_json_prints_the_refusal_object_on_exit_3(tmp_path, capsys):
    argv = [str(_long_csv(tmp_path)), "--out", str(tmp_path / "merged.parquet"), "--meta",
            str(_meta_file(tmp_path)), "--timestamp-col", "Timestamp", "--value-col", "Value",
            "--assume-quality", "GOOD", "--json"]
    assert cmd_ingest(argv) == 3
    doc = _json_out(capsys)
    assert (doc["result_kind"], doc["error_type"]) == ("refusal", "SchemaError")
    assert "pass --tag-col Tag" in doc["cause"]

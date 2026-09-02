"""The ingest path: exports in, archives out, refusals where guessing would start."""

from __future__ import annotations

import json

import pandas as pd
import pytest

import tsdive
from tsdive.cli import cmd_ingest, cmd_profile
from tsdive.errors import SchemaError
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
    with pytest.raises(SchemaError, match="--assume-quality was given with --quality-suffix"):
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


# ---------------------------------------------------------- metadata templates

TEMPLATE_KEYS = [
    "identity",
    "name",
    "unit_raw",
    "unit_canonical",
    "eng_range_zero",
    "eng_range_span",
    "sample_rate_s",
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
    assert all(payload[k] is None for k in TEMPLATE_KEYS[2:10])
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

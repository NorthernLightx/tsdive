"""The SKAB study scripts on a synthetic fixture: no network, no real data.

The fixture is three SKAB-shaped files plus one with a duplicated
timestamp. Record ``a`` carries a level shift on two sensors in its
fifth minute that returns to normal; record ``b`` is stable; the
anomaly-free file has no label columns. Values follow short periodic
patterns rather than noise so that no SPC run rule fires by chance and
the stream, at half the baseline amplitude, stays under every worst-
baseline threshold unless the shift is in the window.
"""

from __future__ import annotations

import filecmp
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parents[1]
STUDY = ROOT / "examples" / "studies" / "skab"
sys.path.insert(0, str(STUDY))

import build_archives as ba  # noqa: E402
import run_skab as rs  # noqa: E402

T0 = pd.Timestamp("2020-03-09 10:00:00")
P4 = np.array([1.0, 0.3, -1.0, -0.3])
P6 = np.array([1.0, 0.5, -0.2, -1.0, -0.5, 0.2])
SHIFT_ROWS = range(240, 300)
SHIFT_TAGS = ("Current", "Voltage")
BASELINE_ROWS = 180
STREAM_AMPLITUDE = 0.5


def _pattern(n: int, j: int) -> np.ndarray:
    t = np.arange(n)
    return (1.0 + 0.1 * j) * P4[(t + j) % 4] + (0.7 + 0.05 * j) * P6[(t + 2 * j) % 6]


def _frame(n: int, *, shift: bool, labels: bool) -> pd.DataFrame:
    stamps = [(T0 + pd.Timedelta(i, unit="s")).strftime("%Y-%m-%d %H:%M:%S") for i in range(n)]
    frame = pd.DataFrame({ba.TIMESTAMP_COL: stamps})
    amplitude = np.where(np.arange(n) < BASELINE_ROWS, 1.0, STREAM_AMPLITUDE)
    for j, tag in enumerate(ba.SENSORS):
        if tag == "Pressure":
            frame[tag] = 0.0
            continue
        values = 10.0 * (j + 1) + amplitude * _pattern(n, j)
        if shift and tag in SHIFT_TAGS:
            values[list(SHIFT_ROWS)] += 25.0
        frame[tag] = np.round(values, 4)
    if labels:
        anomaly = np.zeros(n, dtype=int)
        changepoint = np.zeros(n, dtype=int)
        if shift:
            anomaly[list(SHIFT_ROWS)] = 1
            changepoint[[SHIFT_ROWS.start, SHIFT_ROWS.stop]] = 1
        frame["anomaly"] = anomaly
        frame["changepoint"] = changepoint
    return frame


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep=ba.SEPARATOR, index=False, lineterminator="\n")


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("skab")
    raw = root / "raw"
    _write(_frame(360, shift=True, labels=True), raw / "valve1" / "a.csv")
    _write(_frame(360, shift=False, labels=True), raw / "valve1" / "b.csv")
    dup = _frame(300, shift=False, labels=True)
    dup.loc[10, ba.TIMESTAMP_COL] = dup.loc[9, ba.TIMESTAMP_COL]
    _write(dup, raw / "valve1" / "dup.csv")
    _write(_frame(300, shift=False, labels=False), raw / "anomaly-free" / "anomaly-free.csv")
    archives = root / "archives"
    converted, refusals = ba.build(raw, archives, root / "work")
    return archives, converted, refusals


@pytest.fixture(scope="module")
def result(built):
    archives, _, _ = built
    return rs.run_study(archives, window_s=60, k=3, seeds=2)


# ------------------------------------------------------------- build


def test_build_writes_eight_archives_and_labels_per_experiment(built):
    archives, converted, _ = built
    assert [c["experiment"] for c in converted] == [
        "anomaly-free__anomaly-free",
        "valve1__a",
        "valve1__b",
    ]
    for record in converted:
        assert len(record["archives"]) == 8
        names = sorted(p.stem for p in (archives / record["experiment"]).glob("*.parquet"))
        assert names == sorted([*ba.SENSORS, "labels"] if record["labels"] else ba.SENSORS)
    labels = pd.read_parquet(archives / "valve1__a" / "labels.parquet")
    assert list(labels.columns) == ["timestamp", "anomaly", "changepoint"]
    assert int(labels["anomaly"].sum()) == 60
    assert int(labels["changepoint"].sum()) == 2
    assert not (archives / "anomaly-free__anomaly-free" / "labels.parquet").exists()


def test_duplicated_timestamp_is_a_refusal_row(built):
    archives, _, refusals = built
    assert [r["experiment"] for r in refusals] == ["valve1__dup"]
    assert refusals[0]["refusal"] == "SchemaError"
    assert "duplicate timestamps" in refusals[0]["message"]
    on_disk = json.loads((archives / "refusals.json").read_text("utf-8"))
    assert on_disk == refusals
    assert not (archives / "valve1__dup").exists()


# ------------------------------------------------------------- run


def test_every_window_is_scored_or_refused(result):
    scores = result.scores
    scored = scores["score"].notna()
    refused = scores["refusal"] != ""
    assert (scored ^ refused).all()
    assert sorted(scores["tool"].unique()) == sorted(rs.TOOLS)
    per_tool = scores.groupby("tool").size()
    assert per_tool.nunique() == 1
    assert int(per_tool.iloc[0]) == len(result.windows)
    assert set(result.windows["experiment"]) == {
        "anomaly-free__anomaly-free",
        "valve1__a",
        "valve1__b",
    }


def test_profile_finds_the_frozen_tag(result):
    prof = result.profile_per_tag
    frozen = prof[prof["frozen_whole_record"]]
    assert sorted(frozen["tag"].unique()) == ["Pressure"]
    assert len(frozen) == 3
    assert (prof["quality_assumed"]).all()
    assert (prof["interval_median_s"] == 1.0).all()
    usability = result.tag_usability
    unusable = usability[~usability["usable"]]
    assert sorted(unusable["tag"].unique()) == ["Pressure"]
    assert unusable["reason"].str.startswith("frozen").all()
    summary = result.profile_summary.set_index("tag")
    assert int(summary.loc["Pressure", "n_frozen_whole_record"]) == 3
    assert int(summary.loc["Current", "n_frozen_whole_record"]) == 0


def test_shift_is_detected_with_no_delay_and_recovers(result):
    per_record = result.per_record.set_index(["tool", "experiment"])
    a_windows = result.windows[result.windows["experiment"] == "valve1__a"]
    assert a_windows.loc[a_windows["label"] == 1, "window_index"].tolist() == [4]
    for tool in (rs.TOOL_SCREEN, rs.TOOL_SPC, rs.TOOL_T2, rs.TOOL_SPE):
        row = per_record.loc[(tool, "valve1__a")]
        assert row["refusal"] == "", (tool, row["refusal"])
        assert row["detected"] is True or row["detected"] == True  # noqa: E712
        assert row["delay_windows"] == 0
        assert row["recovered"] == True  # noqa: E712
        assert row["n_pre_fired"] == 0 and row["n_post_fired"] == 0
        assert row["roc_auc"] == 1.0


def test_stable_records_raise_no_alarm(result):
    per_record = result.per_record.set_index(["tool", "experiment"])
    for tool in (rs.TOOL_SCREEN, rs.TOOL_SPC, rs.TOOL_T2, rs.TOOL_SPE):
        for experiment in ("valve1__b", "anomaly-free__anomaly-free"):
            row = per_record.loc[(tool, experiment)]
            assert row["refusal"] == "", (tool, experiment, row["refusal"])
            assert row["n_pre_fired"] == 0
            assert pd.isna(row["detected"])
    clock = per_record.loc[(rs.TOOL_CLOCK, "valve1__b")]
    assert clock["n_pre_fired"] == clock["n_pre"] == 3
    free = per_record.loc[(rs.TOOL_SCREEN, "anomaly-free__anomaly-free")]
    assert free["auc_refusal"] == rs.REFUSAL_ONE_CLASS


def test_summary_matches_per_record(result):
    summary = result.summary.set_index(["tool", "group"])
    per_record = result.per_record
    scores = result.scores
    for tool in rs.TOOLS:
        row = summary.loc[(tool, rs.GROUP_LABELLED)]
        pr = per_record[(per_record["tool"] == tool) & (per_record["group"] == rs.GROUP_LABELLED)]
        with_threshold = pr[pr["threshold"].notna()]
        assert row["n_records"] == 2
        assert row["n_records_with_threshold"] == len(with_threshold)
        assert row["n_pre_windows"] == with_threshold["n_pre"].sum()
        assert row["far_pre"] == round(
            with_threshold["n_pre_fired"].sum() / with_threshold["n_pre"].sum(), 4
        )
        assert row["far_floor"] == 0.25
        assert row["over_floor"] == round(row["far_pre"] - 0.25, 4)
        assert row["n_detected"] == with_threshold["detected"].eq(True).sum()
        stream = scores[
            (scores["tool"] == tool)
            & (scores["group"] == rs.GROUP_LABELLED)
            & (scores["role"] == rs.ROLE_STREAM)
        ]
        assert row["n_windows_scored"] == (stream["refusal"] == "").sum()
        assert row["n_windows_refused"] == (stream["refusal"] != "").sum()
    clock = summary.loc[(rs.TOOL_CLOCK, rs.GROUP_LABELLED)]
    assert clock["far_pre"] == 1.0
    assert clock["recovered_rate"] == 0.0


def test_every_folder_auc_carries_a_value_or_a_reason(result):
    """The fixture holds valve1 records only, so valve2 and other are refused."""
    summary = result.summary
    for row in summary[summary["group"] == rs.GROUP_LABELLED].itertuples():
        assert row.roc_auc_valve1 is not None and not pd.isna(row.roc_auc_valve1)
        assert row.roc_auc_valve1_refusal == ""
        for folder in ("valve2", "other"):
            assert pd.isna(getattr(row, f"roc_auc_{folder}"))
            assert getattr(row, f"roc_auc_{folder}_refusal") == "no scored window"


def test_conformal_bed_outcomes(result):
    conformal = result.conformal.set_index(["experiment", "delta"])
    assert conformal.loc[("valve1__a", 0.05), "outcome"] == rs.OUTCOME_DETECT
    assert conformal.loc[("valve1__b", 0.05), "outcome"] == rs.OUTCOME_NONE
    assert conformal.loc[("anomaly-free__anomaly-free", 0.05), "outcome"] == rs.OUTCOME_NONE
    assert (conformal["n_usable_tags"] == 7).all()
    summary = result.conformal_summary.set_index(["group", "delta"])
    assert summary.loc[(rs.GROUP_LABELLED, 0.05), "n_detect"] == 1
    assert summary.loc[(rs.GROUP_ANOMALY_FREE, 0.05), "n_no_alarm"] == 1
    assert summary.loc[(rs.GROUP_LABELLED, 0.05), "permutation_seeds"] == 2
    assert len(result.permutation) == 3 * 2 * 2


def test_two_runs_write_identical_files(built, result, tmp_path):
    archives, _, _ = built
    rs.write_result(result, tmp_path / "one")
    rs.write_result(rs.run_study(archives, window_s=60, k=3, seeds=2), tmp_path / "two")
    names = sorted(p.name for p in (tmp_path / "one").iterdir())
    assert "scores.csv" in names and "summary.csv" in names
    for name in names:
        assert filecmp.cmp(tmp_path / "one" / name, tmp_path / "two" / name, shallow=False), name
    text = (tmp_path / "one" / "per_record.csv").read_text("utf-8")
    assert "\r" not in text

"""The SKAB conversion script on a synthetic fixture: no network, no real data.

The fixture is three SKAB-shaped files plus one with a duplicated
timestamp. Record ``a`` carries a level shift on two sensors in its
fifth minute that returns to normal; record ``b`` is stable; the
anomaly-free file has no label columns. Values follow short periodic
patterns rather than noise so that no SPC run rule fires by chance and
the stream, at half the baseline amplitude, stays under every worst-
baseline threshold unless the shift is in the window.
"""

from __future__ import annotations

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

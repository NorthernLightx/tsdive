"""Features, backbone, splits."""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from conftest import stepped_contract
from tsdive.data import generate_backbone, load_truth, stratified_split
from tsdive.features import extract
from tsdive.store.tagstore import TagStore

UTC = dt.UTC


def test_backbone_deterministic(tmp_path):
    generate_backbone(tmp_path / "a", seed=42)
    generate_backbone(tmp_path / "b", seed=42)
    assert (tmp_path / "a" / "sim-unit-0" / "FIC000.PV.parquet").read_bytes() == (
        tmp_path / "b" / "sim-unit-0" / "FIC000.PV.parquet"
    ).read_bytes()
    assert len(load_truth(tmp_path / "a")) == 4


def test_backbone_truth_matches_loops(tmp_path):
    bb = generate_backbone(tmp_path / "bb", seed=7, n_loops=4)
    truth = load_truth(bb.root)
    assert len(truth) == 4
    assert set(truth["loop"]) == {0, 1, 2, 3}


def test_features_from_backbone_window(tmp_path):
    bb = generate_backbone(tmp_path / "bk")
    store = TagStore(bb.root)
    pv_identity = next(t for t in store.list_tags() if t.point_id.endswith("PV"))
    window = store.read_window(
        pv_identity,
        dt.datetime(2025, 1, 1, 1, tzinfo=UTC),
        dt.datetime(2025, 1, 1, 3, tzinfo=UTC),
        stepped_contract(),
    )
    feats = extract(window)
    row = feats.to_row()
    assert row["n_good"] > 100
    assert row["coverage"] == pytest.approx(1.0)
    assert row["weighted_mean"] is not None
    assert isinstance(row["notes"], str)


def test_features_null_for_nonnumeric_mode_tag(tmp_path):
    """The MODE tag stores states; numeric features honestly come back None."""
    bb = generate_backbone(tmp_path / "bk3")
    store = TagStore(bb.root)
    mode_identity = next(t for t in store.list_tags() if t.point_id.endswith("MODE"))
    window = store.read_window(
        mode_identity,
        dt.datetime(2025, 1, 1, 1, tzinfo=UTC),
        dt.datetime(2025, 1, 1, 3, tzinfo=UTC),
        stepped_contract(),
    )
    feats = extract(window)
    row = feats.to_row()
    assert row["weighted_mean"] is None
    assert any("mean refused" in n for n in feats.notes)


def test_features_note_censoring(tmp_path):
    bb = generate_backbone(tmp_path / "bk2")
    store = TagStore(bb.root)
    pv_identity = next(t for t in store.list_tags() if t.point_id.endswith("PV"))
    # Fault-free first day has no censoring; check the note machinery via
    # a window that includes the saturation-free span anyway:
    window = store.read_window(
        pv_identity,
        dt.datetime(2025, 1, 1, tzinfo=UTC),
        dt.datetime(2025, 1, 1, 6, tzinfo=UTC),
        stepped_contract(),
    )
    feats = extract(window)
    assert not any("censored" in n for n in feats.notes)


def test_split_deterministic_and_stratified():
    frame = pd.DataFrame(
        {
            "instance_source": ["u0"] * 120 + ["u1"] * 120,
            "fault_type": (["normal"] * 60 + ["fault"] * 60) * 2,
        }
    )
    s1 = stratified_split(frame, seed=42)
    s2 = stratified_split(frame, seed=42)
    assert s1.train == s2.train and s1.val == s2.val and s1.test == s2.test
    train_n, val_n, test_n = s1.sizes()
    assert train_n + val_n + test_n == len(frame)
    # Stratified assignment is balanced within each stratum:
    for counts in s1.strata_counts.values():
        total = sum(counts.values())
        assert counts["train"] == round(total * 0.7) or abs(counts["train"] / total - 0.70) < 0.02
        assert abs(counts["val"] / total - 0.15) < 0.02
        assert abs(counts["test"] / total - 0.15) < 0.02


def test_split_missing_column_refused():
    with pytest.raises(KeyError):
        stratified_split(pd.DataFrame({"a": [1]}), by=("nope",))

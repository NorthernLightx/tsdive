"""The 3W detector study, on a synthetic 3W-shaped fixture. Zero network.

The fixture is three instances across two fake wells, laid out exactly as
``scripts/convert_3w.py`` writes real ones: one parquet per variable plus
``LABEL_class`` and ``LABEL_state`` MODE tags, an ``INSTANCE.json``
sidecar, an unlabelled prefix, and assumed quality codes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parents[1]
STUDY = ROOT / "examples" / "studies" / "3w_detectors"
sys.path.insert(0, str(STUDY))

import build_onsets  # noqa: E402
import build_windows  # noqa: E402
import run_detectors  # noqa: E402

from tsdive.store.identity import Role, TagIdentity, TagMeta  # noqa: E402
from tsdive.store.tagstore import write_tag  # noqa: E402

QUALITY_CODES = {"GOOD_ASSUMED": "GOOD", "NO_DATA": "BAD"}
VARIABLES = ("P-TPT", "P-ANULAR", "T-TPT")
N_ROWS = 2400
PREFIX_ROWS = 120
WINDOW_S = 60
SUBGROUPS = 6


def _write(path: Path, stamps, values, meta: TagMeta) -> None:
    quality = ["NO_DATA" if v is None or (isinstance(v, float) and np.isnan(v))
               else "GOOD_ASSUMED" for v in values]
    write_tag(
        path,
        pd.DataFrame({"timestamp": stamps, "value": values, "quality": quality}),
        meta,
        overwrite=True,
    )


def _meta(instance: str, well: str, point: str, role: Role, unit: str | None) -> TagMeta:
    return TagMeta(
        identity=TagIdentity(source_id=instance, point_id=point),
        name=f"{point} fixture",
        unit_raw=unit,
        eng_range=None,
        sample_rate_s=1.0,
        asset=well,
        loop_id=instance,
        role=role,
        quality_codes=dict(QUALITY_CODES),
        quality_assumed=True,
    )


def write_instance(
    root: Path,
    instance: str,
    well: str,
    folder: str,
    classes: list[int | None],
    *,
    seed: int,
    frozen: tuple[str, ...] = (),
) -> Path:
    """One 3W-shaped instance directory. ``classes`` is per row."""
    out = root / instance
    out.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp("2020-01-01T00:00:00+00:00")
    stamps = [start + pd.Timedelta(i, unit="s") for i in range(len(classes))]
    rng = np.random.default_rng(seed)
    for i, variable in enumerate(VARIABLES):
        if variable in frozen:
            values = np.full(len(classes), 1.0e6 + i)
        else:
            base = 1.0e6 * (i + 1) + np.cumsum(rng.normal(0, 40.0, len(classes)))
            bump = np.array([300.0 if c not in (None, 0) else 0.0 for c in classes])
            values = base + bump
        _write(
            out / f"{variable}.parquet",
            stamps,
            list(values),
            _meta(instance, well, variable, Role.PV, "Pa"),
        )
    _write(
        out / "LABEL_class.parquet",
        stamps,
        [None if c is None else str(c) for c in classes],
        _meta(instance, well, "LABEL_class", Role.MODE, None),
    )
    _write(
        out / "LABEL_state.parquet",
        stamps,
        [None if c is None else ("0" if c in (0,) else "1") for c in classes],
        _meta(instance, well, "LABEL_state", Role.MODE, None),
    )
    histogram = {"NA" if c is None else str(c): 0 for c in classes}
    for c in classes:
        histogram["NA" if c is None else str(c)] += 1
    (out / "INSTANCE.json").write_text(
        json.dumps(
            {
                "instance": instance,
                "instance_source": "REAL",
                "well": well,
                "folder_label": folder,
                "first_timestamp_utc": stamps[0].isoformat(),
                "last_timestamp_utc": stamps[-1].isoformat(),
                "n_rows": len(classes),
                "present_variables": list(VARIABLES),
                "absent_variables": [],
                "archives_written": [*VARIABLES, "LABEL_class", "LABEL_state"],
                "quality_assumed": True,
                "quality_codes": dict(QUALITY_CODES),
                "row_label_histogram": {"class": histogram, "state": histogram},
                "unlabelled_prefix_rows": sum(1 for c in classes if c is None),
                "timestamp_tz_assumed": "UTC",
                "source_relpath": f"fixture/{instance}.parquet",
                "source_sha256": "0" * 64,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return out


def classes_for(kind: str) -> list[int | None]:
    """Per-row class column: prefix of nulls, then a scripted event."""
    rows: list[int | None] = [None] * PREFIX_ROWS
    body = N_ROWS - PREFIX_ROWS
    if kind == "clean":
        rows += [0] * body
    elif kind == "transient_then_steady":
        # 60 s of transient (one whole window), then steady fault.
        rows += [0] * (body - 900) + [101] * 60 + [1] * 840
    elif kind == "one_second_of_fault":
        rows += [0] * (body - 1) + [4]
    else:  # pragma: no cover - guarded by the callers below
        raise ValueError(kind)
    return rows


@pytest.fixture(scope="module")
def archives(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("3w_fixture") / "archives"
    root.mkdir(parents=True)
    write_instance(root, "WELL-A_0001", "WELL-A", "0", classes_for("clean"), seed=1)
    write_instance(
        root,
        "WELL-A_0002",
        "WELL-A",
        "7",
        classes_for("transient_then_steady"),
        seed=2,
    )
    write_instance(
        root,
        "WELL-B_0001",
        "WELL-B",
        "4",
        classes_for("one_second_of_fault"),
        seed=3,
        frozen=("P-ANULAR",),
    )
    return root


@pytest.fixture(scope="module")
def built(archives: Path, tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("3w_windows")
    code = build_windows.main(
        [
            "--archives",
            str(archives),
            "--out",
            str(out),
            "--window-s",
            str(WINDOW_S),
            "--subgroups",
            str(SUBGROUPS),
            "--jobs",
            "1",
        ]
    )
    assert code == 0
    return out


def test_common_variables_come_off_the_instance_sidecars(archives: Path):
    records = build_windows.instance_records(archives)
    assert build_windows.common_variables(records, 0.6) == sorted(VARIABLES)
    # MODE tags are never candidates, whatever their share.
    assert not any(
        v.startswith(("ESTADO-", "LABEL_"))
        for v in build_windows.common_variables(records, 0.0)
    )


def test_unlabelled_prefix_windows_are_dropped_not_filled(built: Path):
    manifest = json.loads((built / "MANIFEST.json").read_text("utf-8"))
    windows = pd.read_parquet(built / "windows.parquet")
    # 120 null rows over 60 s windows is exactly two windows per instance.
    assert manifest["n_windows_dropped_unlabelled_prefix"] == 2 * 3
    assert windows["window_index"].min() == 2
    first = windows[windows["instance"] == "WELL-A_0001"].iloc[0]
    assert pd.Timestamp(first["window_start"]) == pd.Timestamp(
        "2020-01-01T00:02:00+00:00"
    )


def test_label_is_an_or_over_the_rows(built: Path):
    """One faulted second in 60 makes the window positive."""
    windows = pd.read_parquet(built / "windows.parquet")
    b = windows[windows["instance"] == "WELL-B_0001"]
    assert b["label"].sum() == 1
    hit = b[b["label"] == 1].iloc[0]
    assert hit["window_index"] == b["window_index"].max()
    assert hit["classes"] == "0;4"
    assert not bool(hit["transient_only"])


def test_transient_only_flag_separates_run_up_from_fault(built: Path):
    windows = pd.read_parquet(built / "windows.parquet")
    a2 = windows[windows["instance"] == "WELL-A_0002"].set_index("window_index")
    transient = a2[a2["classes"] == "101"]
    assert len(transient) == 1
    assert bool(transient.iloc[0]["transient_only"])
    assert int(transient.iloc[0]["label"]) == 1
    steady = a2[a2["classes"] == "1"]
    assert len(steady) == 14
    assert not steady["transient_only"].any()
    assert steady["label"].all()


def test_feature_and_grid_tables_line_up_with_the_windows(built: Path):
    windows = pd.read_parquet(built / "windows.parquet")
    features = pd.read_parquet(built / "features.parquet")
    grid = pd.read_parquet(built / "subgroup_medians.parquet")
    assert len(features) == len(windows) * len(VARIABLES)
    assert len(grid) == len(features)
    assert [c for c in grid.columns if c.startswith("s")] == [
        f"s{i:02d}" for i in range(SUBGROUPS)
    ]
    # The frozen variable on WELL-B holds one distinct value per window.
    frozen = features[
        (features["instance"] == "WELL-B_0001") & (features["variable"] == "P-ANULAR")
    ]
    assert (frozen["distinct_count"] == 1).all()


BASELINE_WINDOWS = 6  # 6 windows x 6 sub-groups = 36 >= MIN_HISTORY_GOOD


@pytest.fixture(scope="module")
def detector_run(built: Path, tmp_path_factory) -> Path:
    source = tmp_path_factory.mktemp("3w_source")
    (source / "MANIFEST.sha256.json").write_text("{}", encoding="utf-8")
    out = tmp_path_factory.mktemp("3w_results")
    code = run_detectors.main(
        [
            "--windows",
            str(built),
            "--source",
            str(source),
            "--out",
            str(out),
            "--n-folds",
            "2",
            "--min-baseline-windows",
            "5",
            "--baseline-windows",
            str(BASELINE_WINDOWS),
        ]
    )
    assert code == 0
    return out


def test_run_detectors_smoke(detector_run: Path, built: Path):
    """Every tool runs, every window gets a score or a named refusal."""
    out = detector_run
    per_fold = pd.read_csv(out / "per_fold.csv")
    run = json.loads((out / "run.json").read_text("utf-8"))
    windows = pd.read_parquet(built / "windows.parquet")

    assert set(per_fold["tool"]) == set(run_detectors.ALL_TOOLS)
    assert set(per_fold["variant"]) == set(run_detectors.OWN_HISTORY_VARIANTS)
    assert set(per_fold["design"]) == {
        run_detectors.DESIGN_POOLED,
        run_detectors.DESIGN_POOLED_MATCHED,
        run_detectors.DESIGN_OWN,
        run_detectors.DESIGN_STANDARDISED,
    }
    # Nothing is quietly dropped: a window is scored or refused, never both
    # and never neither.
    counted = per_fold[(per_fold["variant"] == "all") & (per_fold["split"] == "well")]
    for _, row in counted.iterrows():
        assert row["n_scored"] + row["n_refused"] == row["n_windows"]
    whole = per_fold[per_fold["design"] != run_detectors.DESIGN_POOLED_MATCHED]
    for split in ("well", "instance", "own-history"):
        totals = whole[
            (whole["variant"] == "all") & (whole["split"] == split)
        ].groupby(["tool", "design"])["n_windows"].sum()
        assert (totals == len(windows)).all()
    # The matched control reads the pooled scores back over exactly the
    # windows the own-history design scores, and no others.
    own = json.loads((out / "run.json").read_text("utf-8"))["provenance"]["own_history"]
    matched = per_fold[
        (per_fold["design"] == run_detectors.DESIGN_POOLED_MATCHED)
        & (per_fold["variant"] == "all")
    ]
    assert (matched.groupby("tool")["n_windows"].sum() == own["n_scored_windows"]).all()

    # Whole wells stay whole.
    folds = run["provenance"]["splits"]["well"]["folds"]
    assert sorted(w for fold in folds for w in fold) == ["WELL-A", "WELL-B"]
    assert all(len(set(fold)) == len(fold) for fold in folds)
    assert run["provenance"]["dataset_manifest_sha256"]

    scores = pd.read_csv(out / "scores.csv")
    assert set(scores["window_key"]) <= set(range(len(windows)))
    assert scores["score"].notna().all()


def test_own_history_design_scores_every_post_baseline_window(detector_run: Path):
    """The own-history tools score the later windows and refuse the rest by name."""
    run = json.loads((detector_run / "run.json").read_text("utf-8"))
    own = run["provenance"]["own_history"]
    assert own["baseline_windows"] == BASELINE_WINDOWS
    # Three instances, each long enough: 6 baseline windows apiece.
    assert own["n_instances_scored"] == 3
    assert own["n_instances_too_short"] == 0
    assert own["n_baseline_windows"] == 3 * BASELINE_WINDOWS
    assert own["n_pairs_with_a_usable_baseline"] > 0

    per_fold = pd.read_csv(detector_run / "per_fold.csv")
    own_rows = per_fold[
        (per_fold["design"] == run_detectors.DESIGN_OWN) & (per_fold["variant"] == "all")
    ]
    assert set(own_rows["tool"]) == set(run_detectors.OWN_HISTORY_TOOLS)
    for _, row in own_rows.iterrows():
        assert row["n_scored"] == own["n_scored_windows"]
        # Every refusal here is the design's own, not the data's.
        assert row["n_refused_by_design"] == own["n_baseline_windows"]

    # Group holdout is irrelevant to this design and the run says so.
    kind = run["provenance"]["splits"]["own-history"]["kind"]
    assert "no group holdout" in kind

    # The clock control reads no sensor value: its score is the window's
    # position after the baseline, restarting at 0 for each instance.
    scores = pd.read_csv(detector_run / "scores.csv")
    clock = scores[scores["tool"] == "clock"].sort_values("window_key")
    windows = pd.read_csv(detector_run / "windows.csv").set_index("window_key")
    joined = clock.join(windows[["instance", "window_index"]], on="window_key")
    for _, group in joined.groupby("instance"):
        ordered = group.sort_values("window_index")
        assert list(ordered["score"]) == list(range(len(ordered)))


def test_own_history_baseline_is_label_blind_and_refuses_short_instances():
    """First k windows by position, whatever their labels; shorter is refused."""
    windows = pd.DataFrame(
        {
            "instance": ["A"] * 5 + ["B"] * 2,
            "window_index": [4, 0, 3, 1, 2, 1, 0],
            "window_key": [40, 10, 30, 20, 25, 61, 60],
            "label": [0, 1, 0, 1, 0, 0, 0],
        }
    )
    layout = run_detectors.own_history_layout(windows, 3)
    # Ordered by window_index, not by row order and not by label.
    assert layout.baseline == {"A": [10, 20, 25]}
    assert layout.scored == {"A": [30, 40]}
    assert set(layout.short) == {"B"}
    assert layout.short["B"].startswith(run_detectors.DESIGN_REFUSAL)
    assert "2 window(s)" in layout.short["B"]

    out = run_detectors.ToolScores.empty()
    run_detectors._design_refusals(out, layout, windows)
    assert set(out.refusals) == {10, 20, 25, 60, 61}
    assert all(r.startswith(run_detectors.DESIGN_REFUSAL) for r in out.refusals.values())


def test_per_instance_standardisation_uses_only_the_baseline_windows():
    """Mean 0 / std 1 on the baseline; later windows keep their offset."""
    columns = ["v"]
    frame = pd.DataFrame(
        {
            "instance": ["A"] * 5,
            "variable": ["P"] * 5,
            "window_index": [0, 1, 2, 3, 4],
            "window_key": [0, 1, 2, 3, 4],
            "v": [1.0, 2.0, 3.0, 100.0, -100.0],
        }
    )
    windows = frame[["instance", "window_index", "window_key"]].assign(label=0)
    layout = run_detectors.own_history_layout(windows, 3)
    out, detail = run_detectors._instance_zscore(frame, layout, columns)

    # Baseline mean 2.0, population std sqrt(2/3); the later windows are
    # z-scored against those and nothing else.
    scale = float(np.std([1.0, 2.0, 3.0]))
    assert list(out["window_key"]) == [3, 4]
    assert out["v"].tolist() == pytest.approx(
        [(100.0 - 2.0) / scale, (-100.0 - 2.0) / scale]
    )
    assert detail["n_zero_scale_columns"] == 0
    assert detail["n_scored_rows_without_baseline_stats"] == 0

    # A frozen tag has no scale; it is centred, left unscaled, and counted.
    frozen = frame.assign(v=[7.0, 7.0, 7.0, 9.0, 5.0])
    out, detail = run_detectors._instance_zscore(frozen, layout, columns)
    assert detail["n_zero_scale_columns"] == 1
    assert out["v"].tolist() == pytest.approx([2.0, -2.0])


def test_values_beyond_float32_range_are_blanked_before_any_detector_sees_them():
    """3W's -1.180116e+42 sentinel: masked once, counted, everything else kept."""
    frame = pd.DataFrame(
        {
            "instance": ["A", "A", "B"],
            "variable": ["P-TPT", "P-TPT", "P-PDG"],
            "s0": [1.0, 1e30, -1.180116e42],
            "s1": [2.0, 1e40, -1.180116e42],
        }
    )
    out, detail = run_detectors.mask_beyond_float32(frame, ["s0", "s1"])

    # 1e40 is beyond float32 and 1e30 is not; the row keeps the value that fits.
    assert detail == {"n_cells": 3, "n_rows": 2}
    assert out["s0"].tolist() == pytest.approx([1.0, 1e30, np.nan], nan_ok=True)
    assert np.isnan(out.loc[1, "s1"])
    assert out.loc[[0], ["s0", "s1"]].to_numpy().tolist() == [[1.0, 2.0]]
    # The caller's frame is not mutated: the mask is applied to a copy.
    assert frame.loc[2, "s0"] == -1.180116e42

    # Nothing to mask leaves the frame alone and says so.
    clean = frame.assign(s0=[1.0, 2.0, 3.0], s1=[4.0, 5.0, 6.0])
    same, detail = run_detectors.mask_beyond_float32(clean, ["s0", "s1"])
    assert detail == {"n_cells": 0, "n_rows": 0}
    assert same is clean


# ------------------------------------------------------ onset-aligned design

ALIGNED_K = 6  # 6 windows x 6 sub-groups = 36 >= MIN_HISTORY_GOOD
ONSET_ROWS = (600, 900, 1500)


def classes_with_onset(onset_row: int, transient_rows: int = 300) -> list[int | None]:
    """Null prefix, then normal, then a transient run-up, then the steady fault."""
    rows: list[int | None] = [None] * PREFIX_ROWS
    rows += [0] * (onset_row - PREFIX_ROWS)
    rows += [101] * transient_rows
    rows += [1] * (N_ROWS - onset_row - transient_rows)
    return rows


def test_onset_is_the_first_fault_row_and_the_prefix_cannot_hold_one():
    stamps = pd.Series(pd.date_range("2020-01-01", periods=8, freq="s", tz="UTC"))

    # The unlabelled prefix is null, not zero, so it is never an onset.
    prefixed = pd.Series([None, None, "0", "0", "101", "101", "1", "1"])
    onset, klass = build_onsets.onset_of(prefixed, stamps)
    assert klass == 101
    assert onset == stamps.iloc[4]

    # The transient run-up counts: 107 is an onset, not a lead-in to one.
    transient = pd.Series(["0", "0", "0", "107", "107", "7", "7", "7"])
    onset, klass = build_onsets.onset_of(transient, stamps)
    assert klass == 107
    assert onset == stamps.iloc[3]

    # A steady class with no run-up in front of it is its own onset.
    steady = pd.Series(["0", "0", "4", "4", "4", "4", "4", "4"])
    assert build_onsets.onset_of(steady, stamps)[1] == 4

    # No fault row at all is not an error and not a zero.
    assert build_onsets.onset_of(pd.Series(["0"] * 8), stamps) == (None, None)
    assert build_onsets.onset_of(pd.Series([None] * 8), stamps) == (None, None)


def test_onset_row_records_the_spans_and_the_evaluability(archives: Path):
    """The fixture's three instances cover all three outcomes."""
    rows = {
        d.name: build_onsets.instance_onset(
            {"instance_dir": str(d), "window_s": WINDOW_S, "baseline_windows": 3}
        )
        for d in sorted(archives.iterdir())
        if (d / "INSTANCE.json").exists()
    }

    clean = rows["WELL-A_0001"]
    assert clean["t_onset"] is None
    assert clean["onset_kind"] == "none"
    assert not clean["evaluable"]

    run_up = rows["WELL-A_0002"]
    assert run_up["onset_class"] == 101
    assert run_up["onset_kind"] == "transient"
    # 120 null rows, then 1,380 normal: onset is 1,500 s into the record.
    assert run_up["pre_onset_s"] == 1500.0
    assert run_up["pre_onset_labelled_s"] == 1380.0
    assert run_up["pre_onset_windows"] == 25
    assert run_up["post_onset_windows"] == 15
    assert run_up["evaluable"]

    # One faulted second in the last row: an onset with nothing after it.
    late = rows["WELL-B_0001"]
    assert late["onset_class"] == 4
    assert late["onset_kind"] == "steady"
    assert late["post_onset_windows"] == 0
    assert not late["has_2_post_windows"]
    assert not late["evaluable"]


@pytest.fixture(scope="module")
def aligned_archives(tmp_path_factory) -> Path:
    """Three instances on three wells, faults starting at three different hours."""
    root = tmp_path_factory.mktemp("3w_aligned_fixture") / "archives"
    root.mkdir(parents=True)
    for i, (well, onset_row) in enumerate(zip("CDE", ONSET_ROWS, strict=True)):
        write_instance(
            root,
            f"WELL-{well}_0001",
            f"WELL-{well}",
            "7",
            classes_with_onset(onset_row),
            seed=10 + i,
        )
    return root


@pytest.fixture(scope="module")
def aligned_built(aligned_archives: Path, tmp_path_factory) -> tuple[Path, Path]:
    out = tmp_path_factory.mktemp("3w_aligned_windows")
    results = tmp_path_factory.mktemp("3w_aligned_onsets")
    assert (
        build_onsets.main(
            [
                "--archives", str(aligned_archives),
                "--out", str(out),
                "--results", str(results),
                "--window-s", str(WINDOW_S),
                "--baseline-windows", str(ALIGNED_K),
                "--jobs", "1",
            ]
        )
        == 0
    )
    assert (
        build_windows.main(
            [
                "--aligned",
                "--archives", str(aligned_archives),
                "--out", str(out),
                "--onsets", str(out / "onsets.csv"),
                "--window-s", str(WINDOW_S),
                "--subgroups", str(SUBGROUPS),
                "--baseline-windows", str(ALIGNED_K),
                "--jobs", "1",
            ]
        )
        == 0
    )
    return out, results


def test_aligned_windows_sit_at_exact_offsets_and_skip_the_guard(aligned_built):
    out, _ = aligned_built
    windows = pd.read_parquet(out / "windows.parquet")
    onsets = pd.read_csv(out / "onsets.csv").set_index("instance")
    manifest = json.loads((out / "MANIFEST.json").read_text("utf-8"))

    assert manifest["n_instances"] == 3
    assert manifest["n_windows_per_role"] == {
        "baseline": 3 * ALIGNED_K,
        "pre": 3,
        "post0": 3,
        "post1": 3,
    }
    # The guard window is the one offset the design does not build.
    offsets = sorted(set(windows["onset_offset"]))
    assert offsets == [*range(-(ALIGNED_K + 2), -1), 0, 1]
    assert -1 not in offsets

    for instance, group in windows.groupby("instance"):
        onset = pd.Timestamp(onsets.loc[instance, "t_onset"])
        for _, row in group.iterrows():
            start = pd.Timestamp(row["window_start"])
            assert start == onset + pd.Timedelta(WINDOW_S * row["onset_offset"], "s")
            assert pd.Timestamp(row["window_end"]) == start + pd.Timedelta(
                WINDOW_S - 1, "s"
            )
        post0 = group[group["role"] == "post0"].iloc[0]
        assert pd.Timestamp(post0["window_start"]) == onset

    # The label the design asserts is the label the rows carry.
    assert manifest["n_test_windows_whose_rows_disagree_with_the_design_label"] == 0
    assert (windows["design_label"] == windows["label"]).all()
    assert not windows[windows["role"].isin(("baseline", "pre"))]["label"].any()
    assert windows[windows["role"].isin(("post0", "post1"))]["label"].all()


@pytest.fixture(scope="module")
def aligned_run(aligned_built, tmp_path_factory) -> Path:
    built, results = aligned_built
    source = tmp_path_factory.mktemp("3w_aligned_source")
    (source / "MANIFEST.sha256.json").write_text("{}", encoding="utf-8")
    out = tmp_path_factory.mktemp("3w_aligned_results")
    assert (
        run_detectors.main(
            [
                "--aligned",
                "--windows", str(built),
                "--source", str(source),
                "--out", str(out),
                "--onsets", str(built / "onsets.csv"),
                "--onset-summary", str(results / "onsets.json"),
                "--n-folds", "2",
            ]
        )
        == 0
    )
    return out


def test_the_clock_control_is_worth_nothing_once_the_design_offset_is_out(aligned_run):
    """The offset-free clock ties every comparison, so its AUC is exactly 0.5."""
    from sklearn.metrics import roc_auc_score

    summary = pd.read_csv(aligned_run / "aligned_summary.csv")
    flat = summary[
        (summary["tool"] == "clock_flat") & (summary["variant"] == "all")
    ].iloc[0]
    assert flat["roc_auc"] == 0.5

    windows = pd.read_csv(aligned_run / "aligned_windows.csv").set_index("window_key")
    scores = pd.read_csv(aligned_run / "aligned_scores.csv")
    flat_scores = scores[scores["tool"] == "clock_flat"].set_index("window_key")["score"]
    # It is one number per instance, the same for all of its windows.
    joined = windows.join(flat_scores.rename("score"), how="inner")
    assert (joined.groupby("instance")["score"].nunique() == 1).all()

    # The plain clock is above chance here, and the whole of the excess is
    # the fixed spacing the design puts between pre and post: take the
    # offset back out of its own scores and the AUC returns to 0.5.
    clock = scores[scores["tool"] == "clock"].set_index("window_key")["score"]
    test = windows.join(clock.rename("score"), how="inner")
    test = test[test["role"] != "baseline"]
    y = (test["role"] != "pre").astype(int)
    assert roc_auc_score(y, test["score"]) > 0.5
    assert roc_auc_score(y, test["score"] - test["onset_offset"]) == 0.5


def test_aligned_run_scores_the_baseline_it_takes_its_threshold_from(aligned_run):
    per_instance = pd.read_csv(aligned_run / "aligned_per_instance.csv")
    assert set(per_instance["tool"]) == set(run_detectors.ALIGNED_TOOLS)
    own_history = per_instance[per_instance["tool"].isin(("mad", "spc", "clock"))]
    # A threshold needs baseline scores, so the baseline windows are
    # scored under this design rather than refused as they are under the
    # unaligned own-history one.
    assert (own_history["n_baseline_scored"] == ALIGNED_K).all()
    assert own_history["baseline_max"].notna().all()
    assert own_history[["pre", "post0", "post1"]].notna().all().all()

    run = json.loads((aligned_run / "aligned_run.json").read_text("utf-8"))
    assert run["provenance"]["refusals"]["n_instances_evaluable"] == 3
    assert run["provenance"]["split"]["group_col"] == "well"


def _toy_scores() -> pd.DataFrame:
    """Four instances, one per outcome the detection metrics have to name."""
    return pd.DataFrame(
        [
            # quiet baseline, detected in the first hour after onset
            {"instance": "A", "baseline_max": 1.0, "pre": 0.5, "post0": 2.0,
             "post1": 3.0, "fold": 0},
            # fires before onset and never after it
            {"instance": "B", "baseline_max": 1.0, "pre": 2.0, "post0": 0.5,
             "post1": 0.5, "fold": 0},
            # detected only in the second hour
            {"instance": "C", "baseline_max": 1.0, "pre": 0.5, "post0": 0.5,
             "post1": 5.0, "fold": 1},
            # no baseline score, so no threshold and no verdict
            {"instance": "D", "baseline_max": None, "pre": 0.5, "post0": 1.0,
             "post1": 1.0, "fold": 1},
        ]
    )


def test_paired_and_threshold_metrics_on_a_toy_score_table():
    out = run_detectors.aligned_metrics(_toy_scores())

    assert out["n_instances"] == 4
    assert out["n_instances_with_threshold"] == 3
    assert out["n_instances_refused"] == 1

    # Only A beats its own pre window in the first hour; A and C in the second.
    assert out["paired_hit_post0"] == pytest.approx(1 / 3, abs=1e-4)
    assert out["paired_hit_post1"] == pytest.approx(2 / 3, abs=1e-4)

    # B is the false alarm; the fired-on test is strict, so C's post0 of 0.5
    # against a baseline of 1.0 is not a detection.
    assert out["false_alarm_rate_pre"] == pytest.approx(1 / 3, abs=1e-4)
    assert out["detection_rate_post0"] == pytest.approx(1 / 3, abs=1e-4)
    assert out["detection_rate_post1"] == pytest.approx(2 / 3, abs=1e-4)

    # A is 0 windows late, C is 1, B is never detected and carries no delay.
    assert out["n_detected"] == 2
    assert out["n_never_detected"] == 1
    assert out["median_delay_windows"] == 0.5
    assert out["roc_auc"] == pytest.approx(11.5 / 18, abs=1e-4)


def test_a_tool_that_never_clears_its_own_threshold_has_no_delay():
    frame = _toy_scores()
    frame[["post0", "post1"]] = 0.1
    out = run_detectors.aligned_metrics(frame)
    assert out["n_detected"] == 0
    assert out["n_never_detected"] == 3
    assert out["median_delay_windows"] is None
    assert out["detection_rate_post0"] == 0.0
    assert out["detection_rate_post1"] == 0.0
    # The pre window's own false alarm is unaffected by what follows it.
    assert out["false_alarm_rate_pre"] == pytest.approx(1 / 3, abs=1e-4)

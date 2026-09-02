"""The 3W Chronos-Bolt study on a synthetic fixture with a stub forecaster.

No torch, no chronos, no network: the model call sits behind the
``forecaster`` argument and the fixture is three instances of archives
laid out as ``data/3w_archives/<instance>/<VARIABLE>.parquet``.
"""

from __future__ import annotations

import filecmp
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parents[1]
STUDY = ROOT / "examples" / "studies" / "3w_chronos"
sys.path.insert(0, str(STUDY))

import run_chronos as rc  # noqa: E402

DETECTOR_RESULTS = ROOT / "examples" / "studies" / "3w_detectors" / "results"
VARIABLES = ["P-ANULAR", "P-JUS-CKGL", "P-MON-CKP", "P-PDG", "P-TPT", "T-TPT"]
T0 = pd.Timestamp("2020-01-01T00:00:00+00:00")
RATE_S = 5
HOURS = 9
OFFSETS = {-5: "baseline", -4: "baseline", -3: "baseline", -2: "pre", 0: "post0", 1: "post1"}


def stub_forecaster(contexts: np.ndarray) -> np.ndarray:
    """q50 = mean of the last ten context minutes, band = one context std."""
    level = contexts[:, -10:].mean(axis=1, keepdims=True)
    spread = np.maximum(contexts.std(axis=1, keepdims=True), 1e-6)
    q = np.stack([level - spread, level, level + spread], axis=-1)
    return np.repeat(q, rc.TARGET_MINUTES, axis=1)


def _signal(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n) * RATE_S / 3600.0
    return 100.0 + 3.0 * np.sin(2 * np.pi * t / 2.0) + rng.normal(0.0, 0.5, n)


def _write(archives: Path, instance: str, variable: str, values: np.ndarray) -> None:
    stamps = T0 + pd.to_timedelta(np.arange(len(values)) * RATE_S, unit="s")
    quality = np.where(np.isfinite(values), "GOOD_ASSUMED", "NO_DATA")
    frame = pd.DataFrame({"timestamp": stamps, "value": values, "quality": quality})
    (archives / instance).mkdir(parents=True, exist_ok=True)
    frame.to_parquet(archives / instance / f"{variable}.parquet", index=False)


def _windows(instance: str, well: str, onset_index: int) -> list[dict]:
    rows = []
    for offset, role in OFFSETS.items():
        index = onset_index + offset
        rows.append(
            {
                "instance": instance,
                "well": well,
                "folder_label": "1",
                "window_index": index,
                "onset_offset": offset,
                "role": role,
                "window_start": (T0 + pd.Timedelta(index, "h")).isoformat(),
            }
        )
    return rows


@pytest.fixture(scope="module")
def fixture(tmp_path_factory):
    archives = tmp_path_factory.mktemp("archives")
    n = HOURS * 3600 // RATE_S
    rows = []
    # A: onset at hour 6, so its -5 h window opens at hour 1 with exactly 60
    # minutes of record before it; each post window carries its own step.
    for k, variable in enumerate(VARIABLES):
        values = _signal(n, seed=k)
        values[6 * 3600 // RATE_S : 7 * 3600 // RATE_S] += 40.0
        values[7 * 3600 // RATE_S : 8 * 3600 // RATE_S] += 90.0
        if variable == "P-TPT":
            # A missing quarter hour inside the pre window's context.
            values[(3 * 3600 + 600) // RATE_S : (3 * 3600 + 1500) // RATE_S] = np.nan
        _write(archives, "A", variable, values)
    rows += _windows("A", "WELL-A", onset_index=6)
    # B: no step; P-PDG carries no data at all, so five variables score.
    for k, variable in enumerate(VARIABLES):
        values = _signal(n, seed=10 + k)
        if variable == "P-PDG":
            values[:] = np.nan
        _write(archives, "B", variable, values)
    rows += _windows("B", "WELL-B", onset_index=6)
    # C: onset at hour 5, so its -5 h window opens the record and has no
    # context; the instance has no threshold.
    for k, variable in enumerate(VARIABLES):
        _write(archives, "C", variable, _signal(n, seed=20 + k))
    rows += _windows("C", "WELL-C", onset_index=5)
    windows = pd.DataFrame(rows)
    folds = pd.DataFrame(
        {"instance": ["A", "B", "C"], "fold": [0, 1, 1], "onset_kind": ["transient"] * 3}
    )
    return archives, windows, folds


@pytest.fixture(scope="module")
def result(fixture):
    archives, windows, folds = fixture
    return rc.run_study(windows, VARIABLES, archives, stub_forecaster, folds)


# ------------------------------------------------------------ aggregation


def _frame(minutes_with_data: list[int], value: float = 1.0, per_minute: int = 3):
    stamps, values = [], []
    for m in minutes_with_data:
        for j in range(per_minute):
            stamps.append(T0 + pd.Timedelta(60 * m + 20 * j, "s"))
            values.append(value + j)
    return pd.DataFrame(
        {"timestamp": stamps, "value": values, "quality": ["GOOD_ASSUMED"] * len(stamps)}
    )


def test_minute_means_average_good_rows_only():
    frame = _frame([0, 2], value=1.0)
    frame.loc[len(frame)] = [T0 + pd.Timedelta(60, "s"), 999.0, "NO_DATA"]
    out = rc.minute_means(frame, T0, 3)
    assert out[0] == pytest.approx(2.0)
    assert np.isnan(out[1])
    assert out[2] == pytest.approx(2.0)
    # A row past the span is ignored.
    frame.loc[len(frame)] = [T0 + pd.Timedelta(180, "s"), 5.0, "GOOD_ASSUMED"]
    assert np.isnan(rc.minute_means(frame, T0, 3)).sum() == 1


def test_context_and_target_thresholds_are_refusals():
    start = T0 + pd.Timedelta(rc.CONTEXT_MINUTES, "min")
    rng = np.random.default_rng(0)

    def build(context_minutes: int, target_minutes: int) -> pd.DataFrame:
        ctx = _frame(list(range(context_minutes)))
        ctx["value"] = rng.normal(size=len(ctx))
        tgt = _frame(list(range(rc.CONTEXT_MINUTES, rc.CONTEXT_MINUTES + target_minutes)))
        tgt["value"] = rng.normal(size=len(tgt))
        return pd.concat([ctx, tgt], ignore_index=True)

    short_ctx = rc.prepare_variable(build(59, 60), start, 0, "v")
    assert short_ctx.refusal == "context: 59 of 240 minutes with data, fewer than 60"
    short_tgt = rc.prepare_variable(build(60, 29), start, 0, "v")
    assert short_tgt.refusal == "target: 29 of 60 minutes with data, fewer than 30"
    ok = rc.prepare_variable(build(60, 30), start, 0, "v")
    assert ok.refusal is None
    assert ok.n_context == 60 and ok.n_target == 30
    # The 180 missing context minutes are filled and counted.
    assert ok.n_context_filled == 180
    assert np.isfinite(ok.context).all()
    flat = rc.prepare_variable(_frame(list(range(300)), value=7.0), start, 0, "v")
    assert flat.refusal == "context is constant: the surprise has no scale"


def test_is_constant_tolerates_float_noise_only():
    assert rc.is_constant(np.array([18.35756, 18.35756 + 4e-15, np.nan, 18.35756]))
    assert rc.is_constant(np.array([0.0, 0.0, 5e-10]))
    assert not rc.is_constant(np.array([18.35756, 18.35757]))
    assert not rc.is_constant(np.array([0.0, 1e-6]))
    assert rc.is_constant(np.array([np.nan, np.nan]))


def test_forward_fill_counts_leading_and_interior_gaps():
    filled, n = rc.forward_fill(np.array([np.nan, np.nan, 1.0, np.nan, 3.0]))
    assert filled.tolist() == [1.0, 1.0, 1.0, 1.0, 3.0]
    assert n == 3
    with pytest.raises(ValueError):
        rc.forward_fill(np.array([np.nan, np.nan]))


# --------------------------------------------------------------- surprise


def test_surprise_by_hand():
    context = np.array([0.0, 2.0] * 10)  # std 1.0
    q = np.zeros((rc.TARGET_MINUTES, 3))
    q[:, 0], q[:, 1], q[:, 2] = 9.0, 10.0, 11.0  # band 2.0 above the floor 0.05
    target = np.full(rc.TARGET_MINUTES, 10.5)  # |diff| 0.5 -> 0.25 per minute
    target[:6] = np.nan
    value, skipped = rc.surprise(target, q, context)
    assert value == pytest.approx(0.25)
    assert skipped == 6
    # A zero-width band falls back to 5 percent of the context std.
    q[:, 0] = q[:, 2] = 10.0
    value, skipped = rc.surprise(np.full(rc.TARGET_MINUTES, 10.1), q, context)
    assert value == pytest.approx(0.1 / 0.05)
    assert skipped == 0


def test_forecaster_shape_is_checked(fixture):
    archives, windows, folds = fixture

    def bad(contexts):
        return np.zeros((len(contexts), 10, 3))

    with pytest.raises(ValueError, match="forecaster returned"):
        rc.run_study(windows, VARIABLES, archives, bad, folds)


# ------------------------------------------------------------- the tables


def test_per_instance_columns_match_the_detector_study(result):
    header = (DETECTOR_RESULTS / "aligned_per_instance.csv").read_text("utf-8").splitlines()[0]
    assert list(result.per_instance.columns) == header.split(",")
    summary_header = (DETECTOR_RESULTS / "aligned_summary.csv").read_text("utf-8")
    expected = [*summary_header.splitlines()[0].split(","), "far_floor", "over_floor"]
    assert list(result.summary.columns) == expected


def test_window_refusals_and_scores(result):
    windows = result.windows.set_index(["instance", "onset_offset"])
    # C's -5 h window opens the record: every variable lacks context.
    c5 = windows.loc[("C", -5)]
    assert c5["n_variables_scored"] == 0
    assert c5["refusal"].startswith("0 of 6 variables scored, fewer than 4; P-ANULAR: context: 0")
    assert pd.isna(c5["score"])
    # B scores five variables everywhere; P-PDG is a refusal row, not a gap.
    assert (windows.loc["B", "n_variables_scored"] == 5).all()
    b_pdg = result.window_scores.query("instance == 'B' and variable == 'P-PDG'")
    assert len(b_pdg) == 6
    assert (b_pdg["refusal"] == "context: 0 of 240 minutes with data, fewer than 60").all()
    # A's -5 h window has exactly 60 context minutes and scores.
    a5 = result.window_scores.query("instance == 'A' and onset_offset == -5")
    assert (a5["n_context_minutes"] == 60).all()
    assert (a5["n_context_filled"] == 180).all()
    # The gap inside A's P-TPT pre-window context is filled and counted.
    gap = result.window_scores.query(
        "instance == 'A' and variable == 'P-TPT' and onset_offset == -2"
    ).iloc[0]
    assert gap["n_context_filled"] == 15
    assert result.counts["n_instances_scored"] == 2
    assert result.counts["n_instances_refused"] == 1
    assert result.counts["variable_refusals_by_cause"] == {"context": 12}


def test_per_instance_threshold_fires_paired_and_delay(result):
    table = result.per_instance.set_index(["tool", "instance"])
    a = table.loc[(rc.TOOL, "A")]
    b = table.loc[(rc.TOOL, "B")]
    c = table.loc[(rc.TOOL, "C")]
    scores = result.windows.set_index(["instance", "onset_offset"])["score"]
    assert a["n_baseline_scored"] == 3
    assert a["baseline_max"] == pytest.approx(max(scores.loc[("A", o)] for o in (-5, -4, -3)))
    assert a["fold"] == 0 and a["onset_kind"] == "transient" and a["well"] == "WELL-A"
    # The step at onset is a surprise above every baseline and above pre.
    assert a["post0"] > a["baseline_max"] and a["post0"] > a["pre"]
    assert pd.isna(a["refusal"])
    assert pd.isna(b["refusal"])
    assert c["n_baseline_scored"] == 2
    assert pd.isna(c["baseline_max"])
    assert "baseline: 2 of 3 windows scored, no threshold" in c["refusal"]

    row = result.summary.set_index("tool").loc[rc.TOOL]
    assert row["n_instances"] == 3
    assert row["n_instances_with_threshold"] == 2
    assert row["n_instances_refused"] == 1
    assert row["n_pre_windows"] == 2 and row["n_post_windows"] == 4
    usable = result.per_instance.query("tool == @rc.TOOL and baseline_max.notna()")
    hit0 = (usable["post0"] > usable["pre"]).mean()
    far = (usable["pre"] > usable["baseline_max"]).mean()
    assert row["paired_hit_post0"] == pytest.approx(hit0)
    assert row["false_alarm_rate_pre"] == pytest.approx(far)
    assert row["far_floor"] == pytest.approx(0.25)
    assert row["over_floor"] == pytest.approx(far - 0.25)
    assert row["n_detected"] + row["n_never_detected"] == 2
    assert row["n_detected"] >= 1 and row["median_delay_windows"] == 0.0
    assert row["detection_rate_post0"] == pytest.approx(
        (usable["post0"] > usable["baseline_max"]).mean()
    )
    assert row["n_folds_scored"] == 2
    # The clock is the window position, so every paired comparison is a hit.
    clock = result.summary.set_index("tool").loc[rc.CLOCK]
    assert clock["paired_hit_post0"] == 1.0 and clock["paired_hit_post1"] == 1.0
    assert clock["n_instances_with_threshold"] == 3
    same = result.summary.set_index("tool").loc[rc.CLOCK_SAME]
    assert same["n_instances"] == 2
    assert set(result.per_fold["tool"]) == {rc.TOOL, rc.CLOCK, rc.CLOCK_SAME}


def test_two_runs_write_identical_files(fixture, tmp_path):
    archives, windows, folds = fixture
    for name in ("one", "two"):
        out = rc.run_study(windows, VARIABLES, archives, stub_forecaster, folds)
        rc.write_result(out, tmp_path / name)
    names = sorted(p.name for p in (tmp_path / "one").iterdir())
    assert names == ["per_fold.csv", "per_instance.csv", "summary.csv", "window_scores.csv",
                     "windows.csv"]
    for name in names:
        assert filecmp.cmp(tmp_path / "one" / name, tmp_path / "two" / name, shallow=False)
    text = (tmp_path / "one" / "per_instance.csv").read_text("utf-8")
    assert "\r" not in text
    # Integer columns print as integers and floats carry at most 4 decimals.
    assert ",0,3," in text
    for cell in text.splitlines()[1].split(","):
        if "." in cell:
            assert len(cell.split(".")[1]) <= 4

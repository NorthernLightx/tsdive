"""The baseline drift study on synthetic caches: no network, no real data.

The 3W-shaped cache holds four instances: ``drift`` ramps by one unit a
minute and carries no fault, ``step`` jumps once and stays up, ``brief``
is too short for the alarm rule and ``tiny`` is shorter than the
baseline. The SKAB-shaped archives hold a ramping record with no labels
and a record whose eighth window is labelled and then returns. Values
are integers on short periodic patterns, so every score a scoring gives
a window repeats exactly and nothing fires by rounding.
"""

from __future__ import annotations

import filecmp
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parents[1]
STUDY = ROOT / "examples" / "studies" / "baseline_drift"
sys.path.insert(0, str(STUDY))

import run_drift as rd  # noqa: E402

VARIABLES = ["P-PDG", "P-TPT"]
PATTERN = np.array([0, 2, 5, 1, 3, 4] * 10, dtype=float)  # 60 minutes, MAD > 0
FROZEN = 7.0
T0 = pd.Timestamp("2015-01-01T00:00:00+00:00")
STEP_ONSET = 8
WOBBLE = [0, 1, 3]
SKAB_T0 = pd.Timestamp("2020-03-09T10:00:00+00:00")
SKAB_FAULT = 7


def _minutes(values: np.ndarray) -> dict:
    return {f"s{i:02d}": float(values[i]) for i in range(60)}


def _instance_rows(
    instance: str,
    levels: list[float],
    labels: list[int],
    indices: list[int] | None = None,
) -> tuple[list, list]:
    """Window rows and sub-group rows for one 3W instance."""
    indices = indices or [position + 1 for position in range(len(levels))]
    windows, medians = [], []
    for position, (level, label) in enumerate(zip(levels, labels, strict=True)):
        index = indices[position]
        start = T0 + pd.Timedelta(3600 * index, unit="s")
        windows.append(
            {
                "instance": instance,
                "well": "WELL-00099",
                "folder_label": "1",
                "window_index": index,
                "window_start": start.isoformat(),
                "window_end": (start + pd.Timedelta(3600, unit="s")).isoformat(),
                "label": label,
                "design_label": label,
                "onset_offset": position,
                "role": "stream",
                "n_variables_present": len(VARIABLES),
            }
        )
        medians.append(
            {"instance": instance, "window_index": index, "variable": "P-PDG"}
            | _minutes(np.full(60, FROZEN))
        )
        medians.append(
            {"instance": instance, "window_index": index, "variable": "P-TPT"}
            | _minutes(level + PATTERN)
        )
    return windows, medians


def _ramp_levels(n: int) -> list[float]:
    return [1000.0 + 60.0 * position for position in range(n)]


def build_3w_cache(path: Path, *, aligned: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    windows, medians = [], []
    if aligned:
        offsets = (-5, -4, -3, -2, 0, 1)
        roles = ("baseline", "baseline", "baseline", "pre", "post0", "post1")
        for name in ("alpha", "beta"):
            levels = [1000.0, 1000.0, 1000.0, 1000.0, 2000.0, 2000.0]
            labels = [0, 0, 0, 0, 1, 1]
            w, m = _instance_rows(
                f"WELL-00099_{name}", levels, labels, [1, 2, 3, 4, 6, 7]
            )
            for row, offset, role in zip(w, offsets, roles, strict=True):
                row["onset_offset"] = offset
                row["role"] = role
            windows += w
            medians += m
    else:
        plan = [
            ("drift", _ramp_levels(10), [0] * 10),
            (
                "step",
                [1000.0] * STEP_ONSET + [4000.0] * 4,
                [0] * STEP_ONSET + [1] * 4,
            ),
            ("brief", [1000.0] * 5, [0] * 5),
            ("tiny", [1000.0] * 3, [0] * 3),
        ]
        for name, levels, labels in plan:
            w, m = _instance_rows(f"WELL-00099_{name}", levels, labels)
            windows += w
            medians += m
    pd.DataFrame(windows).to_parquet(path / "windows.parquet", index=False)
    pd.DataFrame(medians).to_parquet(path / "subgroup_medians.parquet", index=False)
    (path / "MANIFEST.json").write_text(
        '{"common_variables": ["P-PDG", "P-TPT"], "window_s": 3600.0, "subgroups": 60}\n',
        encoding="utf-8",
        newline="\n",
    )


def _skab_levels(n: int, ramp: float, fault: dict[int, float]) -> list[float]:
    return [
        fault.get(position, 100.0 + ramp * position + WOBBLE[position % 3])
        for position in range(n)
    ]


def build_skab_archives(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    plan = {
        "free__ramp": (_skab_levels(12, 10.0, {}), None),
        "valve__step": (_skab_levels(13, 0.0, {SKAB_FAULT: 201.0}), SKAB_FAULT),
    }
    for name, (levels, fault) in plan.items():
        directory = path / name
        directory.mkdir(parents=True, exist_ok=True)
        stamps = pd.date_range(SKAB_T0, periods=60 * len(levels), freq="s")
        values = np.repeat(np.array(levels, dtype=float), 60)
        for tag in ("Current", "Pressure"):
            column = values if tag == "Current" else np.full(values.shape, FROZEN)
            pd.DataFrame(
                {"timestamp": stamps, "value": column, "quality": "GOOD"}
            ).to_parquet(directory / f"{tag}.parquet", index=False)
        if fault is None:
            continue
        anomaly = np.zeros(len(stamps), dtype=int)
        anomaly[fault * 60 : (fault + 1) * 60] = 1
        pd.DataFrame(
            {"timestamp": stamps, "anomaly": anomaly, "changepoint": 0}
        ).to_parquet(directory / "labels.parquet", index=False)


@pytest.fixture(scope="module")
def study(tmp_path_factory) -> tuple[rd.StudyResult, Path]:
    root = tmp_path_factory.mktemp("baseline_drift")
    build_3w_cache(root / "windows", aligned=False)
    build_3w_cache(root / "aligned", aligned=True)
    build_skab_archives(root / "archives")
    out = root / "results"
    result = rd.run(root / "windows", root / "aligned", root / "archives", out)
    return result, out


def _record(result: rd.StudyResult, record: str, scoring: str) -> pd.Series:
    rows = result.per_record[
        (result.per_record["record"] == record) & (result.per_record["scoring"] == scoring)
    ]
    assert len(rows) == 1, record
    return rows.iloc[0]


# ---------------------------------------------------------------- 3W


def test_static_fires_on_the_drifting_record_and_the_others_do_not(study):
    result, _ = study
    static = _record(result, "WELL-00099_drift", rd.SCORING_STATIC)
    assert static["n_pre"] == 4
    assert static["n_pre_fired"] > 0
    for scoring in (rd.SCORING_DIFF, rd.SCORING_ROLLING):
        row = _record(result, "WELL-00099_drift", scoring)
        assert row["n_pre"] == 4
        assert row["n_pre_fired"] == 0, scoring
    assert _record(result, "WELL-00099_drift", rd.SCORING_CLOCK)["n_pre_fired"] == 4


def test_every_scoring_detects_the_step(study):
    result, _ = study
    for scoring in (rd.SCORING_STATIC, rd.SCORING_DIFF, rd.SCORING_ROLLING):
        row = _record(result, "WELL-00099_step", scoring)
        assert row["detected"], scoring
        assert row["n_pre_fired"] == 0, scoring
    rolling = _record(result, "WELL-00099_step", rd.SCORING_ROLLING)
    static = _record(result, "WELL-00099_step", rd.SCORING_STATIC)
    assert rolling["n_consecutive_fault_fires"] < static["n_consecutive_fault_fires"]


def test_short_records_are_refused_and_counted(study):
    result, _ = study
    brief = _record(result, "WELL-00099_brief", rd.SCORING_STATIC)
    assert brief["group"] == rd.GROUP_NORMAL
    assert "the rule needs 7" in brief["refusal"]
    assert pd.isna(brief["threshold"])
    assert result.scores[result.scores["record"] == "WELL-00099_brief"].empty
    assert "WELL-00099_tiny" not in set(result.per_record["record"])
    short = result.summary[
        (result.summary["bed"] == rd.BED_3W_OWN) & (result.summary["group"] == rd.GROUP_SHORT)
    ]
    assert len(short) == 1
    assert int(short.iloc[0]["n_records"]) == 1


def test_frozen_variable_is_unusable_for_every_scoring(study):
    result, _ = study
    frozen = result.usability[
        (result.usability["record"] == "WELL-00099_drift")
        & (result.usability["variable"] == "P-PDG")
    ]
    assert set(frozen["scoring"]) == {
        rd.SCORING_STATIC,
        rd.SCORING_DIFF,
        rd.SCORING_ROLLING,
    }
    assert set(frozen["reason"]) == {rd.REFUSAL_ZERO_SCALE}


def test_rolling_is_refused_on_the_aligned_bed(study):
    result, _ = study
    rolling = result.per_record[
        (result.per_record["bed"] == rd.BED_3W_ALIGNED)
        & (result.per_record["scoring"] == rd.SCORING_ROLLING)
    ]
    assert len(rolling) == 2
    assert all("no predecessors in the cache" in reason for reason in rolling["refusal"])
    static = result.per_record[
        (result.per_record["bed"] == rd.BED_3W_ALIGNED)
        & (result.per_record["scoring"] == rd.SCORING_STATIC)
    ]
    assert all(static["detected"])


# ---------------------------------------------------------------- SKAB


def test_skab_ramp_separates_the_scorings(study):
    result, _ = study
    assert _record(result, "free__ramp", rd.SCORING_STATIC)["n_pre_fired"] > 0
    for scoring in (rd.SCORING_DIFF, rd.SCORING_ROLLING):
        assert _record(result, "free__ramp", scoring)["n_pre_fired"] == 0, scoring


def test_rolling_recovers_within_two_windows_on_skab(study):
    result, _ = study
    rolling = _record(result, "valve__step", rd.SCORING_ROLLING)
    assert rolling["detected"]
    assert rolling["recovered"]
    assert rolling["delay_windows"] == 0


# ---------------------------------------------------------------- tables


def test_every_window_of_a_scored_record_is_scored_or_refused(study):
    result, _ = study
    scored = set(result.scores["record"])
    assert scored == set(result.windows["record"])
    for (record, scoring), rows in result.scores.groupby(["record", "scoring"]):
        n = int((result.windows["record"] == record).sum())
        assert len(rows) == n, (record, scoring)
        assert sorted(rows["position"]) == list(range(n))
        has_score = rows["score"].notna()
        has_refusal = rows["refusal"].fillna("").ne("")
        assert bool((has_score ^ has_refusal).all()), (record, scoring)


def test_summary_arithmetic_matches_the_record_rows(study):
    result, _ = study
    for row in result.summary.itertuples():
        if row.scoring == "none":
            continue
        pr = result.per_record[
            (result.per_record["bed"] == row.bed)
            & (result.per_record["group"] == row.group)
            & (result.per_record["scoring"] == row.scoring)
        ]
        assert row.n_records == len(pr)
        ok = pr[pr["threshold"].notna()]
        assert row.n_records_scored == len(ok)
        assert row.n_records_refused == len(pr) - len(ok)
        assert row.n_pre_windows == int(ok["n_pre"].fillna(0).sum())
        if row.n_pre_windows:
            fired = int(ok["n_pre_fired"].fillna(0).sum())
            assert row.far_window == pytest.approx(
                round(fired / row.n_pre_windows, 4), abs=1e-9
            )
            assert row.over_floor == pytest.approx(
                round(row.far_window - rd.far_floor(3), 4), abs=1e-9
            )


def test_the_run_repeats_byte_for_byte(study, tmp_path):
    _, out = study
    root = out.parent
    again = tmp_path / "again"
    rd.run(root / "windows", root / "aligned", root / "archives", again)
    for name in ("windows.csv", "scores.csv", "per_record.csv", "summary.csv", "usability.csv"):
        assert filecmp.cmp(out / name, again / name, shallow=False), name

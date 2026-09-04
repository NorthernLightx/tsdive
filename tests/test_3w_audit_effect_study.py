"""The audit-effect study on a synthetic 3W cache: no network, no real data.

Three instances. ``a`` carries a fault from its sixth window, a tag
frozen at one value, a tag rounded to a lattice whose window median
moves off the baseline centre, and a tag with 21 finite baseline minutes,
which is under the 30 ``mad_baseline`` requires. ``b`` carries no fault
and a lattice tag whose third baseline window sits off the centre, so the
unchecked variant gives that instance an infinite threshold. ``c`` is
shorter than the baseline.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parents[1]
STUDY = ROOT / "examples" / "studies" / "3w_audit_effect"
sys.path.insert(0, str(STUDY))

import run_audit_effect as ra  # noqa: E402

VARIABLES = ["P-ANULAR", "P-MON-CKP", "P-PDG", "P-TPT"]
PATTERN = np.array([0.0, 2.0, 5.0, 1.0, 3.0, 4.0] * 10)
FROZEN = 7.0
LATTICE_TAIL = np.arange(1.0, 21.0)
T0 = pd.Timestamp("2015-01-01T00:00:00+00:00")


def _lattice(centre: float) -> np.ndarray:
    return np.concatenate([np.full(40, centre), LATTICE_TAIL])


def _sparse(values: np.ndarray) -> np.ndarray:
    out = np.full(60, np.nan)
    out[: len(values)] = values
    return out


def _instance_a() -> dict[str, list[np.ndarray]]:
    return {
        "P-TPT": [100.0 + PATTERN + (500.0 if w >= 5 else 0.0) for w in range(8)],
        "P-PDG": [np.full(60, FROZEN) for _ in range(8)],
        "P-ANULAR": [_lattice(5.0 if w < 3 else 9.0) for w in range(8)],
        "P-MON-CKP": [
            _sparse(np.arange(10.0, 17.0)) if w < 3 else 10.0 + PATTERN for w in range(8)
        ],
    }


def _instance_b() -> dict[str, list[np.ndarray]]:
    return {
        "P-TPT": [100.0 + PATTERN for _ in range(7)],
        "P-PDG": [np.full(60, FROZEN) for _ in range(7)],
        "P-ANULAR": [np.full(60, 6.0 if w == 2 else 5.0) for w in range(7)],
        "P-MON-CKP": [10.0 + PATTERN for _ in range(7)],
    }


def _instance_c() -> dict[str, list[np.ndarray]]:
    return {name: [100.0 + PATTERN for _ in range(3)] for name in VARIABLES}


PLAN = {
    "WELL-00099_a": (_instance_a, [0, 0, 0, 0, 0, 1, 1, 1]),
    "WELL-00099_b": (_instance_b, [0] * 7),
    "WELL-00099_c": (_instance_c, [0] * 3),
}


def build_cache(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    windows, features, grid = [], [], []
    for instance, (builder, labels) in PLAN.items():
        series = builder()
        for position, label in enumerate(labels):
            start = T0 + pd.Timedelta(3600 * position, unit="s")
            windows.append(
                {
                    "instance": instance,
                    "well": "WELL-00099",
                    "folder_label": "1",
                    "window_index": position + 1,
                    "window_start": start.isoformat(),
                    "window_end": (start + pd.Timedelta(3600, unit="s")).isoformat(),
                    "label": label,
                }
            )
            for variable in VARIABLES:
                values = np.asarray(series[variable][position], dtype=float)
                grid.append(
                    {"instance": instance, "window_index": position + 1, "variable": variable}
                    | {f"s{i:02d}": float(values[i]) for i in range(60)}
                )
                features.append(
                    {
                        "instance": instance,
                        "window_index": position + 1,
                        "variable": variable,
                        "median": float(np.nanmedian(values)),
                    }
                )
    pd.DataFrame(windows).to_parquet(path / "windows.parquet", index=False)
    pd.DataFrame(features).to_parquet(path / "features.parquet", index=False)
    pd.DataFrame(grid).to_parquet(path / "subgroup_medians.parquet", index=False)
    (path / "MANIFEST.json").write_text(
        '{"common_variables": ["P-ANULAR", "P-MON-CKP", "P-PDG", "P-TPT"], '
        '"window_s": 3600.0, "subgroups": 60}\n',
        encoding="utf-8",
        newline="\n",
    )


@pytest.fixture(scope="module")
def study(tmp_path_factory) -> tuple[ra.StudyResult, Path]:
    root = tmp_path_factory.mktemp("audit_effect")
    build_cache(root / "windows")
    out = root / "results"
    result = ra.run(root / "windows", out, per_tag=root / "missing.csv")
    return result, root


def _pair(result: ra.StudyResult, variant: str, instance: str, variable: str) -> pd.Series:
    frame = result.pairs
    hit = frame[
        (frame["variant"] == variant)
        & (frame["instance"] == instance)
        & (frame["variable"] == variable)
    ]
    assert len(hit) == 1, (instance, variable)
    return hit.iloc[0]


def _record(result: ra.StudyResult, variant: str, detector: str, instance: str) -> pd.Series:
    frame = result.per_record
    hit = frame[
        (frame["variant"] == variant)
        & (frame["detector"] == detector)
        & (frame["instance"] == instance)
    ]
    assert len(hit) == 1, (variant, detector, instance)
    return hit.iloc[0]


# ---------------------------------------------------------------- pairs


def test_frozen_and_lattice_tags_are_dropped_by_the_zero_scale_check(study):
    result, _ = study
    frozen = _pair(result, ra.VARIANT_CHECKED, "WELL-00099_a", "P-PDG")
    assert frozen["status"] == ra.STATUS_ZERO_SCALE
    assert frozen["baseline_distinct"] == 1
    lattice = _pair(result, ra.VARIANT_CHECKED, "WELL-00099_a", "P-ANULAR")
    assert lattice["status"] == ra.STATUS_ZERO_SCALE
    assert lattice["baseline_distinct"] > 1
    for variable in ("P-PDG", "P-ANULAR"):
        kept = _pair(result, ra.VARIANT_UNCHECKED, "WELL-00099_a", variable)
        assert kept["status"] == ra.STATUS_ZERO_SCALE_KEPT


def test_a_short_baseline_history_is_dropped_by_the_quality_check(study):
    result, _ = study
    sparse = _pair(result, ra.VARIANT_CHECKED, "WELL-00099_a", "P-MON-CKP")
    assert sparse["status"] == ra.STATUS_QUALITY
    assert 0 < sparse["n_finite_baseline"] < 30
    kept = _pair(result, ra.VARIANT_UNCHECKED, "WELL-00099_a", "P-MON-CKP")
    assert kept["status"] == ra.STATUS_KEPT
    assert kept["scale"] > 0


# ---------------------------------------------------------------- scores


def test_unchecked_screen_produces_infinite_windows(study):
    result, _ = study
    unchecked = _record(result, ra.VARIANT_UNCHECKED, ra.TOOL_SCREEN, "WELL-00099_a")
    assert unchecked["n_inf_windows"] > 0
    checked = _record(result, ra.VARIANT_CHECKED, ra.TOOL_SCREEN, "WELL-00099_a")
    assert checked["n_inf_windows"] == 0
    assert checked["detected"] and unchecked["detected"]


def test_an_infinite_baseline_score_silences_the_alarm(study):
    result, _ = study
    unchecked = _record(result, ra.VARIANT_UNCHECKED, ra.TOOL_SCREEN, "WELL-00099_b")
    assert math.isinf(float(unchecked["threshold"]))
    assert unchecked["n_pre_fired"] == 0
    checked = _record(result, ra.VARIANT_CHECKED, ra.TOOL_SCREEN, "WELL-00099_b")
    assert math.isfinite(float(checked["threshold"]))


def test_the_clock_control_is_the_same_under_both_variants(study):
    result, _ = study
    columns = ["instance", "threshold", "n_pre", "n_pre_fired", "detected"]
    frame = result.per_record[result.per_record["detector"] == ra.TOOL_CLOCK]
    checked = frame[frame["variant"] == ra.VARIANT_CHECKED][columns].reset_index(drop=True)
    unchecked = frame[frame["variant"] == ra.VARIANT_UNCHECKED][columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(checked, unchecked)


def test_the_short_instance_is_refused_and_counted(study):
    result, _ = study
    row = _record(result, ra.VARIANT_CHECKED, ra.TOOL_SCREEN, "WELL-00099_c")
    assert row["group"] == ra.GROUP_SHORT
    assert "are its baseline" in row["refusal"]
    assert pd.isna(row["threshold"])
    short = result.summary[
        (result.summary["group"] == ra.GROUP_SHORT)
        & (result.summary["detector"] == ra.TOOL_SCREEN)
        & (result.summary["variant"] == ra.VARIANT_CHECKED)
    ]
    assert len(short) == 1
    assert int(short.iloc[0]["n_instances"]) == 1
    assert result.pairs[result.pairs["instance"] == "WELL-00099_c"].empty


# ---------------------------------------------------------------- tables


def test_unchecked_keeps_every_pair_the_checks_drop(study):
    result, _ = study
    pooled = result.summary[result.summary["group"] == "pooled"]
    checked = pooled[pooled["variant"] == ra.VARIANT_CHECKED].iloc[0]
    unchecked = pooled[pooled["variant"] == ra.VARIANT_UNCHECKED].iloc[0]
    assert unchecked["n_pairs"] == checked["n_pairs"]
    assert unchecked["n_pairs_kept"] == (
        checked["n_pairs_kept"]
        + checked["n_pairs_zero_scale"]
        + checked["n_pairs_insufficient_quality"]
    )


def test_summary_arithmetic_matches_the_record_rows(study):
    result, _ = study
    for row in result.summary.itertuples():
        pr = result.per_record[
            (result.per_record["variant"] == row.variant)
            & (result.per_record["detector"] == row.detector)
        ]
        pr = pr[pr["group"] != ra.GROUP_SHORT] if row.group == "pooled" else pr[
            pr["group"] == row.group
        ]
        assert row.n_instances == len(pr)
        ok = pr[pr["threshold"].notna()]
        assert row.n_instances_with_threshold == len(ok)
        assert row.n_pre_windows == int(ok["n_pre"].fillna(0).sum())
        if row.n_pre_windows:
            fired = int(ok["n_pre_fired"].fillna(0).sum())
            assert row.far_window == pytest.approx(
                round(fired / row.n_pre_windows, 4), abs=1e-9
            )


def test_the_run_repeats_byte_for_byte(study, tmp_path):
    result, root = study
    again = tmp_path / "again"
    ra.run(root / "windows", again, per_tag=root / "missing.csv")
    for name in ("pairs.csv", "per_record.csv", "summary.csv"):
        first = (root / "results" / name).read_bytes()
        assert first == (again / name).read_bytes(), name
    assert result.run_info["frozen_population"] == {"available": False}

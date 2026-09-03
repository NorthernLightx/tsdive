"""The 3W conformal martingale study on a synthetic pair of window caches."""

from __future__ import annotations

import filecmp
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parents[1]
STUDY = ROOT / "examples" / "studies" / "3w_conformal"
sys.path.insert(0, str(STUDY))

import run_conformal as rc  # noqa: E402

VARIABLES = ["P-ANULAR", "P-JUS-CKGL", "P-MON-CKP", "P-PDG", "P-TPT", "T-TPT"]
OFFSETS = {-5: "baseline", -4: "baseline", -3: "baseline", -2: "pre", 0: "post0", 1: "post1"}
DELTAS = (0.05, 0.01)
STABLE, SHIFT, TIE, SHORT = "WELL-A_stable", "WELL-B_shift", "WELL-C_tie", "WELL-D_short"


def _subgroups(rng: np.random.Generator, shift: float, constant: bool) -> np.ndarray:
    """(6, 60) sub-group medians: unit noise around a per-variable level."""
    levels = 100.0 + 10.0 * np.arange(len(VARIABLES))[:, None]
    values = levels + rng.normal(0.0, 1.0, (len(VARIABLES), 60)) + shift
    if constant:
        values[0, :] = 42.0
    return values


def _instance_windows(instance: str, n: int, labels: list[int], aligned: bool) -> list[dict]:
    rows = []
    offsets = list(OFFSETS)[:n]
    for position in range(n):
        index = 10 + offsets[position] if aligned else position + 1
        row = {
            "instance": instance,
            "well": instance.split("_")[0],
            "folder_label": "1" if any(labels) else "0",
            "window_index": index,
            "window_start": f"2020-01-01T{index:02d}:00:00+00:00",
            "window_end": f"2020-01-01T{index:02d}:59:59+00:00",
            "label": labels[position],
            "transient_only": False,
            "classes": str(labels[position]),
            "state_mode": str(labels[position]),
            "n_state_values": 1,
            "n_variables_present": 6,
        }
        if aligned:
            row["onset_offset"] = offsets[position]
            row["role"] = OFFSETS[offsets[position]]
            row["design_label"] = labels[position]
        rows.append(row)
    return rows


def _build_cache(root: Path, aligned: bool) -> Path:
    rng = np.random.default_rng(20260903)
    n_full = 6 if aligned else 8
    fault_from = 4 if aligned else 5
    plan = {
        STABLE: (n_full, [0] * n_full, 0.0, False),
        SHIFT: (n_full, [0] * fault_from + [1] * (n_full - fault_from), 6.0, False),
        TIE: (n_full, [0] * n_full, 0.0, True),
        SHORT: (3, [0, 0, 0], 0.0, False),
    }
    windows, sub = [], []
    for instance, (n, labels, shift, constant) in plan.items():
        rows = _instance_windows(instance, n, labels, aligned)
        windows += rows
        for row in rows:
            values = _subgroups(rng, shift if row["label"] else 0.0, constant)
            for j, variable in enumerate(VARIABLES):
                sub.append(
                    {
                        "instance": instance,
                        "window_index": row["window_index"],
                        "variable": variable,
                        **{f"s{i:02d}": values[j, i] for i in range(60)},
                    }
                )
    cache = root / ("aligned" if aligned else "own")
    cache.mkdir()
    frame = pd.DataFrame(windows)
    frame.to_parquet(cache / "windows.parquet", index=False)
    pd.DataFrame(sub).to_parquet(cache / "subgroup_medians.parquet", index=False)
    (cache / "MANIFEST.json").write_text(
        json.dumps({"n_windows": len(frame), "n_instances": int(frame["instance"].nunique())}),
        encoding="utf-8",
    )
    return cache


@pytest.fixture(scope="module")
def outputs(tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("caches")
    own = _build_cache(root, aligned=False)
    aligned = _build_cache(root, aligned=True)
    outs = []
    for name in ("first", "second"):
        out = root / name
        rc.run(own, aligned, out, deltas=DELTAS, seeds=3)
        outs.append(out)
    return {
        "dirs": outs,
        "per_instance": pd.read_csv(outs[0] / "per_instance.csv"),
        "summary": pd.read_csv(outs[0] / "summary.csv"),
        "permutation": pd.read_csv(outs[0] / "permutation.csv"),
        "run": json.loads((outs[0] / "run.json").read_text("utf-8")),
    }


def _row(per_instance: pd.DataFrame, bed: str, instance: str, method: str, delta=None):
    g = per_instance[
        (per_instance["bed"] == bed)
        & (per_instance["instance"] == instance)
        & (per_instance["method"] == method)
    ]
    g = g[g["delta"].isna()] if delta is None else g[g["delta"] == delta]
    assert len(g) == 1
    return g.iloc[0]


def test_every_instance_has_a_row_per_method_and_delta(outputs):
    pi = outputs["per_instance"]
    assert set(pi["outcome"]) <= set(rc.OUTCOMES)
    expected = {(rc.METHOD_WORST, None), *((m, d) for m in rc.MARTINGALE_METHODS for d in DELTAS)}
    for bed in (rc.BED_OWN, rc.BED_ALIGNED):
        for instance in (STABLE, SHIFT, TIE, SHORT):
            g = pi[(pi["bed"] == bed) & (pi["instance"] == instance)]
            keys = {(r.method, None if pd.isna(r.delta) else r.delta) for r in g.itertuples()}
            assert keys == expected, (bed, instance)


def test_shifted_instance_is_detected_with_no_delay(outputs):
    pi = outputs["per_instance"]
    for bed in (rc.BED_OWN, rc.BED_ALIGNED):
        row = _row(pi, bed, SHIFT, rc.METHOD_CONFORMAL, 0.05)
        assert row["outcome"] == "detection"
        assert row["delay_windows"] == 0
        assert row["alarm_window"] == row["first_fault_window"]
        assert row["max_log10_martingale"] > -np.log10(0.05)


def test_stable_instance_has_no_alarm_and_the_tie_variable_is_unusable(outputs):
    pi = outputs["per_instance"]
    for bed in (rc.BED_OWN, rc.BED_ALIGNED):
        for delta in DELTAS:
            assert _row(pi, bed, STABLE, rc.METHOD_CONFORMAL, delta)["outcome"] == "none"
        assert _row(pi, bed, TIE, rc.METHOD_CONFORMAL, 0.05)["n_usable_variables"] == 5
        assert _row(pi, bed, STABLE, rc.METHOD_CONFORMAL, 0.05)["n_usable_variables"] == 6


def test_short_instance_is_a_design_refusal(outputs):
    pi = outputs["per_instance"]
    short = pi[pi["instance"] == SHORT]
    assert set(short["outcome"]) == {"refused"}
    assert short["refusal"].str.startswith(rc.DESIGN_REFUSAL).all()
    assert set(short[short["bed"] == rc.BED_OWN]["group"]) == {rc.GROUP_SHORT}
    assert outputs["run"]["beds"][rc.BED_OWN]["refusals"] == {rc.DESIGN_REFUSAL: 1}


def test_groups_follow_the_labels(outputs):
    pi = outputs["per_instance"]
    own = pi[pi["bed"] == rc.BED_OWN].drop_duplicates("instance").set_index("instance")["group"]
    assert own[STABLE] == rc.GROUP_NORMAL
    assert own[SHIFT] == rc.GROUP_MIXED
    assert own[TIE] == rc.GROUP_NORMAL
    assert set(pi[pi["bed"] == rc.BED_ALIGNED]["group"]) == {rc.GROUP_ALIGNED}


def test_two_runs_write_identical_csvs(outputs):
    first, second = outputs["dirs"]
    for name in ("per_instance.csv", "summary.csv", "permutation.csv"):
        assert filecmp.cmp(first / name, second / name, shallow=False), name


def test_summary_arithmetic_matches_per_instance(outputs):
    pi, summary = outputs["per_instance"], outputs["summary"]
    for row in summary.itertuples():
        if row.method == rc.METHOD_PERMUTATION:
            continue
        g = pi[(pi["bed"] == row.bed) & (pi["group"] == row.group) & (pi["method"] == row.method)]
        g = g[g["delta"].isna()] if pd.isna(row.delta) else g[g["delta"] == row.delta]
        scored = g[g["outcome"] != "refused"]
        assert row.n_instances == len(g)
        assert row.n_refused == (g["outcome"] == "refused").sum()
        assert row.n_alarm == (scored["outcome"] != "none").sum()
        if len(scored) and row.group != rc.GROUP_POSITIVE_BASELINE:
            far = (scored["outcome"] == "false_alarm").sum() / len(scored)
            assert row.far == pytest.approx(far, abs=1e-4)
        if len(scored) and row.group in (rc.GROUP_MIXED, rc.GROUP_ALIGNED):
            detect = (scored["outcome"] == "detection").sum() / len(scored)
            assert row.detect == pytest.approx(detect, abs=1e-4)


def test_permutation_rows_average_into_the_summary(outputs):
    perm, summary = outputs["permutation"], outputs["summary"]
    assert perm["alarm_fraction"].between(0.0, 1.0).all()
    assert set(perm["seed"]) == {0, 1, 2}
    means = perm.groupby(["bed", "group", "delta"])["alarm_fraction"].mean()
    rows = summary[summary["method"] == rc.METHOD_PERMUTATION]
    assert len(rows) == len(means)
    for row in rows.itertuples():
        assert row.alarm_rate == pytest.approx(means[(row.bed, row.group, row.delta)], abs=1e-4)

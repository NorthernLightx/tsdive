"""The switchback schedule study on synthetic inputs: no network, no real data."""

from __future__ import annotations

import filecmp
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

ROOT = Path(__file__).parents[1]
STUDY = ROOT / "examples" / "studies" / "switchback"
sys.path.insert(0, str(STUDY))

import run_switchback as rsw  # noqa: E402
import switchback as sb  # noqa: E402

sh = sb.sh


def _regular(t: int, length: int) -> tuple[np.ndarray, sb.Blocks]:
    times = np.arange(t, dtype=float)
    return times, sb.cut_blocks(times, t, length)


def _fit(y, x, blocks, washout, seed):
    frame = sb.make_frame(y, x, blocks, washout)
    design = sb.make_design(blocks.k, seed)
    terms = sb.assignment_terms(frame, design)
    return frame, design, terms


# ---------------------------------------------------------------- schedule


@pytest.mark.parametrize("k", [8, 16])
def test_balanced_schedule_is_deterministic_and_balanced(k):
    first, second = sb.make_design(k, 11), sb.make_design(k, 11)
    assert np.array_equal(first.observed, second.observed)
    assert np.array_equal(first.assignments, second.assignments)
    assert first.observed.sum() == k // 2
    assert np.all(first.assignments.sum(axis=1) == k // 2)
    other = sb.make_design(k, 12)
    assert not (
        np.array_equal(first.observed, other.observed)
        and np.array_equal(first.assignments, other.assignments)
    )


def test_odd_block_count_drops_the_last_block():
    _, blocks = _regular(75, 15)
    assert blocks.k == 4
    assert np.all(blocks.block[60:] == -1)
    assert blocks.block[59] == 3 and blocks.offset[59] == 14


def test_enumeration_is_used_up_to_1000_assignments():
    twelve = sb.make_design(12, 3)
    assert twelve.enumerated and twelve.n_assignments == 924
    assert len(twelve.assignments) == 924
    assert len({row.tobytes() for row in twelve.assignments}) == 924
    assert any(np.array_equal(row, twelve.observed) for row in twelve.assignments)
    fourteen = sb.make_design(14, 3)
    assert not fourteen.enumerated and fourteen.n_assignments == 3432
    assert len(fourteen.assignments) == sb.PERMUTATIONS


def test_design_too_small_fires_at_four_and_six_blocks():
    assert sb.design_refusal(4) == sb.DESIGN_TOO_SMALL  # 6 assignments
    assert sb.design_refusal(6) == sb.DESIGN_TOO_SMALL  # 20 assignments, p >= 0.1
    assert sb.design_refusal(8) == ""  # 70 assignments, p >= 2/70
    with pytest.raises(ValueError, match=sb.DESIGN_TOO_SMALL):
        sb.make_design(4, 0)


# ---------------------------------------------------------------- randomization


def test_randomization_p_is_near_uniform_on_white_noise():
    rng = np.random.default_rng(5)
    _, blocks = _regular(240, 15)
    p = []
    for rep in range(400):
        y = rng.standard_normal(240)
        frame, design, terms = _fit(y, None, blocks, 0, rep)
        p.append(sb.randomization(design, terms, frame.resid_sums[:, None], 1.0).p_value[0])
    p = np.array(p)
    assert 0.02 <= np.mean(p <= 0.05) <= 0.09
    assert 0.42 <= p.mean() <= 0.58
    assert stats.kstest(p, "uniform").pvalue > 0.01


@pytest.mark.parametrize("washout", [0, 3])
def test_an_enumerated_design_rejects_at_most_alpha_of_its_assignments(washout):
    """Over every assignment as the observed one, p <= 0.05 holds for floor(0.05 C) of them."""
    rng = np.random.default_rng(7)
    _, blocks = _regular(240, 24)
    y = rng.standard_normal(240).cumsum()  # a drifting record: exactness needs no stationarity
    x = rng.standard_normal((240, 2)) + y[:, None] * 0.1
    for covariates in (None, x):
        frame = sb.make_frame(y, covariates, blocks, washout)
        rows = sb.all_assignments(blocks.k)
        rejected = 0
        for index in range(len(rows)):
            design = sb.Design(
                blocks.k, rows[index].copy(), rows, rows.astype(float), True, len(rows), 0.0
            )
            terms = sb.assignment_terms(frame, design)
            p = sb.randomization(design, terms, frame.resid_sums[:, None], 1.0).p_value[0]
            rejected += p <= sb.ALPHA
        assert rejected == math.floor(sb.ALPHA * len(rows))


def test_raw_estimate_is_the_difference_in_means():
    rng = np.random.default_rng(6)
    _, blocks = _regular(240, 30)
    y = rng.standard_normal(240)
    frame, design, terms = _fit(y, None, blocks, 4, 1)
    est = sb.randomization(design, terms, frame.resid_sums[:, None], 1.0).estimate[0]
    z = design.observed[frame.block]
    assert est == pytest.approx(frame.y[z == 1].mean() - frame.y[z == 0].mean(), abs=1e-12)


@pytest.mark.parametrize("length", [15, 30])
def test_inverted_interval_holds_the_shift_and_ends_where_p_crosses(length):
    rng = np.random.default_rng(8)
    times, blocks = _regular(240, length)
    base = rng.standard_normal(240)
    design = sb.make_design(blocks.k, 21)
    shift = 1.5
    y = base + shift * sb.lag_response(sb.setting(blocks, design.observed), times, 0.0)
    frame = sb.make_frame(y, None, blocks, 0)
    terms = sb.assignment_terms(frame, design)
    result = sb.randomization(design, terms, frame.resid_sums[:, None], 1.0)
    lo, hi = result.lo[0], result.hi[0]
    assert lo <= shift <= hi
    z = terms.z_kept

    def p_at(theta: float) -> float:
        return sb.randomization_p(frame, design, frame.y - theta * z, 1.0)

    eps = 1e-6 * (hi - lo)
    assert p_at(lo - eps) <= sb.ALPHA < p_at(lo + eps)
    assert p_at(hi + eps) <= sb.ALPHA < p_at(hi - eps)
    assert p_at(result.estimate[0]) == 1.0


def test_frisch_waugh_adjusted_equals_the_direct_ols_coefficient():
    rng = np.random.default_rng(9)
    _, blocks = _regular(240, 15)
    x = rng.standard_normal((240, 3))
    x[:, 2] = x[:, 0] + x[:, 1]  # a repeated direction is dropped, not inverted
    y = x[:, :2] @ [1.0, -0.5] + rng.standard_normal(240)
    frame, design, terms = _fit(y, x, blocks, 5, 2)
    result = sb.randomization(design, terms, frame.resid_sums[:, None], 1.0)
    xk = x[frame.positions]

    def direct(row: np.ndarray) -> float:
        z = row[frame.block].astype(float)
        design_matrix = np.column_stack([np.ones(frame.n), z, xk])
        return float(np.linalg.lstsq(design_matrix, frame.y, rcond=None)[0][1])

    assert result.estimate[0] == pytest.approx(direct(design.observed), abs=1e-10)
    for j in range(5):
        fwl = (design.matrix[j] @ frame.resid_sums) / terms.denom[j]
        assert fwl == pytest.approx(direct(design.assignments[j]), abs=1e-10)
    arms = sb.sample_arms(frame, terms, frame.resid[:, None])
    for arm in (sb.NAIVE, sb.HAC, sb.EWC):
        assert arms[arm].estimate[0] == pytest.approx(result.estimate[0], abs=1e-12)


def test_too_many_covariates_and_too_few_samples_refuse_the_frame():
    rng = np.random.default_rng(10)
    _, blocks = _regular(240, 30)
    y = rng.standard_normal(240)
    assert sb.make_frame(y, rng.standard_normal((240, 6)), blocks, 23) == sb.TOO_MANY_COVARIATES
    assert sb.make_frame(y, None, blocks, 27) == sb.TOO_FEW
    assert sb.make_frame(np.full(240, 2.0), None, blocks, 0) == sb.NO_SPREAD


# ---------------------------------------------------------------- response and washout


def test_first_order_response_reaches_one_minus_exp_minus_three_at_three_tau():
    tau = 5.0
    times = np.arange(40, dtype=float)
    s = sb.lag_response(np.ones(40), times, tau)
    assert s[int(3 * tau) - 1] == pytest.approx(1 - math.exp(-3), abs=1e-12)
    assert np.all(np.diff(s) > 0)
    assert np.array_equal(sb.lag_response(np.ones(40), times, 0.0), np.ones(40))
    assert sb.washout_steps(tau) == 15 and sb.washout_steps(1.5) == 5
    assert sb.washout_steps(3.75) == 12 and sb.washout_steps(0.0) == 0


def test_a_gap_moves_the_response_by_its_length():
    s = sb.lag_response(np.ones(3), np.array([0.0, 1.0, 11.0]), 2.0)
    assert s[2] == pytest.approx(1 - math.exp(-1 / 2) * math.exp(-1 / 2) * math.exp(-10 / 2))


def test_washout_removes_the_carryover_bias():
    rng = np.random.default_rng(12)
    times, blocks = _regular(480, 30)
    tau = 7.5
    biases = {}
    for washout in (0, sb.washout_steps(tau)):
        errors = []
        for seed in range(20):
            design = sb.make_design(blocks.k, seed)
            s = sb.lag_response(sb.setting(blocks, design.observed), times, tau)
            y = s + 1e-6 * rng.standard_normal(480)
            frame = sb.make_frame(y, None, blocks, washout)
            terms = sb.assignment_terms(frame, design)
            est = sb.randomization(design, terms, frame.resid_sums[:, None], 1.0).estimate[0]
            errors.append(est - 1.0)
        biases[washout] = np.median(errors)
    assert biases[0] < -0.2
    assert abs(biases[sb.washout_steps(tau)]) < math.exp(-3)


# ---------------------------------------------------------------- other arms


def test_ewc_through_the_dct_matches_the_shift_study_estimator():
    rng = np.random.default_rng(13)
    for n in (60, 300, 961):
        u = rng.standard_normal(n).cumsum() * 0.1 + rng.standard_normal(n)
        omega, nu = sb.ewc_long_run_variance(u[:, None])
        ref, ref_nu = sh.ewc_long_run_variance(u)
        assert nu == ref_nu
        assert omega[0] == pytest.approx(ref, rel=1e-10)


def test_block_t_is_welch_on_the_block_means():
    rng = np.random.default_rng(14)
    _, blocks = _regular(240, 15)
    y = rng.standard_normal(240)
    frame, design, terms = _fit(y, None, blocks, 3, 4)
    result = sb.sample_arms(frame, terms, frame.resid[:, None])[sb.BLOCK_T]
    means = np.array([frame.y[frame.block == b].mean() for b in range(blocks.k)])
    b, a = means[design.observed == 1], means[design.observed == 0]
    welch = stats.ttest_ind(b, a, equal_var=False)
    assert result.estimate[0] == pytest.approx(b.mean() - a.mean(), abs=1e-12)
    assert result.p_value[0] == pytest.approx(welch.pvalue, rel=1e-9)
    ci = welch.confidence_interval(0.95)
    assert (result.lo[0], result.hi[0]) == pytest.approx((ci.low, ci.high), rel=1e-9)


def test_prepost_reads_the_shift_study_interval():
    rng = np.random.default_rng(15)
    before, after = rng.standard_normal(120), rng.standard_normal(120) + 0.3
    got = sb.prepost(before, after, sh.HAC)
    ref = sh.level_interval(before, after, sh.HAC)
    assert got == (ref.estimate, ref.lo, ref.hi, ref.p_value)
    assert sb.prepost(np.full(120, 1.0), after, sh.EWC) == sb.NO_SPREAD
    x_b, x_a = rng.standard_normal((120, 2)), rng.standard_normal((120, 2)) + 10.0
    assert sb.prepost(before, after, sh.HAC, x_b, x_a) == sb.COVARIATE_OUTSIDE


# ---------------------------------------------------------------- real-shaped beds

T0 = pd.Timestamp("2016-05-01T00:00:00+00:00")
TAGS_3W = ["P-PDG", "P-TPT", "T-TPT"]


def _minutes(values: np.ndarray) -> dict:
    return {f"s{i:02d}": float(values[i]) for i in range(60)}


def build_3w(path: Path) -> None:
    """Three normal instances with 4 consecutive windows, one short, one with a fault window."""
    rng = np.random.default_rng(20)
    windows, medians = [], []
    layout = [
        ("WELL-00001_a", "WELL-00001", [1, 2, 3, 4], [0] * 4),
        ("WELL-00001_b", "WELL-00001", [2, 3, 4, 5, 6], [0] * 5),
        ("WELL-00002_c", "WELL-00002", [1, 2, 3, 4], [0] * 4),
        ("WELL-00002_short", "WELL-00002", [1, 2, 4, 5], [0] * 4),
        ("WELL-00003_fault", "WELL-00003", [1, 2, 3, 4], [0, 0, 0, 1]),
    ]
    for instance, well, indices, labels in layout:
        for index, label in zip(indices, labels, strict=True):
            windows.append(
                {"instance": instance, "well": well, "window_index": index, "label": label}
            )
            for tag in TAGS_3W:
                frozen = tag == "T-TPT"
                values = np.full(60, 7.0) if frozen else 50.0 + rng.standard_normal(60)
                medians.append(
                    {"instance": instance, "window_index": index, "variable": tag}
                    | _minutes(values)
                )
    path.mkdir(parents=True)
    pd.DataFrame(windows).to_parquet(path / "windows.parquet", index=False)
    pd.DataFrame(medians).to_parquet(path / "subgroup_medians.parquet", index=False)
    (path / "MANIFEST.json").write_text(
        json.dumps({"common_variables": TAGS_3W}) + "\n", encoding="utf-8"
    )


def build_tep(path: Path) -> None:
    rng = np.random.default_rng(21)
    parts = []
    for run in (251, 252):
        frame = pd.DataFrame(
            rng.standard_normal((960, len(rsw.TEP_VARIABLES))).astype("float32"),
            columns=list(rsw.TEP_VARIABLES),
        )
        frame.insert(0, "sample", np.arange(1, 961, dtype="int16"))
        frame.insert(0, "run", np.int16(run))
        parts.append(frame)
    path.mkdir(parents=True)
    pd.concat(parts, ignore_index=True).to_parquet(path / "cache.parquet", index=False)
    manifest = {
        "cache": "cache.parquet",
        "cache_sha256": "0" * 12,
        "source_manifest_sha256": "1" * 12,
        "runs": [251, 252],
    }
    (path / "MANIFEST.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")


TURBINE_ROWS_BEFORE = 16 * 144  # 16 days of 10 min rows


def _turbine_frame(rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    v = 8.0 + 2.0 * rng.standard_normal(rows)
    d = rng.uniform(0, 360, rows)
    frame = pd.DataFrame(
        {
            "V": v,
            "D": d,
            "rho": 1.2 + 0.01 * rng.standard_normal(rows),
            "S": 0.3 + 0.05 * rng.standard_normal(rows),
            "I": 0.08 + 0.01 * rng.standard_normal(rows),
            "VcosD": v * np.cos(np.radians(d)),
            "VsinD": v * np.sin(np.radians(d)),
        }
    )
    frame["y_test"] = 0.08 * v + 0.02 * rng.standard_normal(rows)
    frame["y_ctrl"] = 0.08 * v + 0.02 * rng.standard_normal(rows)
    return frame


def _write_pair(path: Path, frame, stamps, status, names) -> None:
    status_name, y_name, c_name = names
    out = pd.DataFrame(
        {
            "time": stamps.dt.strftime("%m/%d/%Y %H:%M"),
            status_name: status,
            **{c: frame[c] for c in ("V", "D", "rho", "S", "I", "VcosD", "VsinD")},
            y_name: frame["y_test"],
            c_name: frame["y_ctrl"],
        }
    )
    out.index = range(1, len(out) + 1)
    out.to_csv(path, lineterminator="\n")


def build_turbine(root: Path) -> None:
    raw = root / "raw"
    raw.mkdir(parents=True)
    rows = TURBINE_ROWS_BEFORE + 300
    status = (np.arange(rows) >= TURBINE_ROWS_BEFORE).astype(int)
    step = pd.to_timedelta(np.arange(rows) * 10, unit="min")
    stamps = pd.Series(pd.Timestamp("2010-06-01 00:00") + step)
    vg = _turbine_frame(rows, 22)
    _write_pair(
        raw / "Turbine Upgrade Dataset(VG Pair).csv",
        vg,
        stamps,
        status,
        ("upgrade status", "y_test (normalized)", "y_ctrl (normalized)"),
    )
    pitch = _turbine_frame(rows, 23)
    gapped = stamps.copy()
    gapped[500:] = gapped[500:] + pd.Timedelta(1, unit="D")
    windy = (status == 1) & (pitch["V"].to_numpy() > 9.0)
    base = pitch["y_test"].to_numpy()
    table = {
        f"y_test(r={r:.2f}, normalized)": np.where(windy, base * (1 + r), base)
        for r in rsw.rs.TURBINE_TABLE_R
    }
    main = pitch.assign(y_test=table["y_test(r=0.05, normalized)"])
    _write_pair(
        raw / "Turbine Upgrade Dataset(Pitch Angle Pair).csv",
        main,
        gapped,
        status,
        ("upgrade status", "y_test(normalized)", "y_ctrl(normalized)"),
    )
    pd.DataFrame(
        {
            "time": gapped.dt.strftime("%m/%d/%Y %H:%M"),
            "upgrade.status": status,
            **{c: pitch[c] for c in ("V", "D", "rho", "S", "I", "VcosD", "VsinD")},
            **table,
            "y_ctrl(normalized)": pitch["y_ctrl"],
        }
    ).to_csv(
        raw / "Turbine Upgrade Dataset(Pitch Angle Pair, Table7.3).csv",
        index=False,
        lineterminator="\n",
    )


def build_skab(path: Path) -> None:
    rng = np.random.default_rng(24)
    for name, rows, labelled in (
        ("anomaly-free__anomaly-free", 2400, False),
        ("valve1__1", 600, True),
    ):
        directory = path / name
        directory.mkdir(parents=True)
        stamps = T0 + pd.to_timedelta(np.arange(rows), unit="s")
        columns = {
            "Current": 10.0 + rng.standard_normal(rows),
            "Voltage": 230.0 + np.cumsum(0.05 * rng.standard_normal(rows)),
            "Pressure": np.full(rows, 1.5),
        }
        for tag, values in columns.items():
            pd.DataFrame({"timestamp": stamps, "value": values, "quality": "GOOD"}).to_parquet(
                directory / f"{tag}.parquet", index=False
            )
        if labelled:
            pd.DataFrame({"timestamp": stamps, "anomaly": 0, "changepoint": 0}).to_parquet(
                directory / "labels.parquet", index=False
            )


OUT_CSVS = (
    rsw.SUMMARY_CSV,
    rsw.DESIGNS_CSV,
    rsw.REFUSALS_CSV,
    rsw.PER_WELL_CSV,
    rsw.PAIRS_CSV,
)
FIXTURE_DRAWS = {rsw.BED_3W: 2, rsw.BED_TEP: 2, rsw.BED_TURBINE: 3, rsw.BED_SKAB: 3}


def _summary(path: Path) -> pd.DataFrame:
    return pd.read_csv(path / rsw.SUMMARY_CSV, keep_default_na=False, na_values=[""])


@pytest.fixture(scope="module")
def beds(tmp_path_factory):
    root = tmp_path_factory.mktemp("switchback")
    build_3w(root / "windows")
    build_tep(root / "tep")
    build_turbine(root / "turbine")
    build_skab(root / "archives")
    kwargs = {
        "windows": root / "windows",
        "tep_cache": root / "tep",
        "turbine": root / "turbine",
        "archives": root / "archives",
        "draws": FIXTURE_DRAWS,
    }
    saved = rsw.TEP_VARIABLES
    rsw.TEP_VARIABLES = saved[:5]  # five variables keep the fixture fast
    try:
        info = rsw.run(root / "a", rsw.BEDS, rows_dir=root / "rows", **kwargs)
        rsw.run(root / "b", rsw.BEDS, rows_dir=None, **kwargs)
    finally:
        rsw.TEP_VARIABLES = saved
    return info, root


def test_a_rerun_writes_identical_csvs(beds):
    _, root = beds
    for name in OUT_CSVS:
        assert filecmp.cmp(root / "a" / name, root / "b" / name, shallow=False), name


def test_every_bed_writes_every_cell(beds):
    info, root = beds
    summary = _summary(root / "a")
    for bed in rsw.BEDS:
        assert (summary["bed"] == bed).sum() == len(rsw.bed_cells(bed))
    rows = pd.read_parquet(root / "rows" / "skab.parquet")
    assert len(rows) == info["beds"]["skab"]["n_rows"]


def test_small_designs_are_refused_with_their_reason(beds):
    _, root = beds
    designs = pd.read_csv(root / "a" / rsw.DESIGNS_CSV, keep_default_na=False)
    refused = designs[designs["refusal"] == sb.DESIGN_TOO_SMALL]
    assert set(zip(refused["bed"], refused["block"], strict=True)) == {
        ("3w", 60),
        ("turbine", 432),
        ("skab", 600),
    }
    summary = _summary(root / "a")
    cut = summary[(summary["bed"] == "3w") & (summary["block"] == 60)]
    assert (cut["n_answered"] == 0).all()
    assert (cut["top_reason"] == sb.DESIGN_TOO_SMALL).all()


def test_beds_read_what_the_loaders_give(beds):
    info, root = beds
    sections = info["beds"]
    assert sections["3w"]["n_records"] == 3  # the short and the fault instance are left out
    assert sections["3w"]["n_usable_targets"] == 6  # T-TPT is frozen
    assert sections["tep"]["n_records"] == 2
    assert sections["turbine"]["n_records"] == 2
    assert sections["skab"]["n_records"] == 1  # the labelled record is left out
    refusals = pd.read_csv(root / "a" / rsw.REFUSALS_CSV, keep_default_na=False)
    frozen = refusals[(refusals["bed"] == "3w") & (refusals["reason"] == sb.NO_SPREAD)]
    assert not frozen.empty


def test_prepost_rows_carry_one_unit_per_target(beds):
    _, root = beds
    summary = _summary(root / "a")
    tep = summary[summary["bed"] == "tep"]
    assert (tep.loc[tep["arm"] == sb.PREPOST_HAC, "n_rows"] == 2 * 5).all()
    assert (tep.loc[tep["arm"] == sb.RANDOMIZATION, "n_rows"] == 2 * 5 * 2).all()


def test_injected_shift_raises_detection(beds):
    _, root = beds
    summary = _summary(root / "a")
    rand = summary[
        (summary["bed"] == "turbine")
        & (summary["block"] == 144)
        & (summary["arm"] == sb.RANDOMIZATION)
        & (summary["adjustment"] == sb.RAW)
        & (summary["tau"] == 0)
    ].set_index("delta")
    assert rand.loc[0.5, "rate"] > rand.loc[0.0, "rate"]
    assert summary["coverage"].dropna().between(0, 1).all()


def test_run_json_carries_no_machine_path(beds):
    _, root = beds
    text = (root / "a" / "run.json").read_text("utf-8")
    assert ":\\\\" not in text and str(root) not in text
    assert str(Path.home()) not in text


def test_worker_processes_write_the_same_csvs(beds, tmp_path):
    _, root = beds
    kwargs = {
        "windows": root / "windows",
        "tep_cache": root / "tep",
        "turbine": root / "turbine",
        "archives": root / "archives",
        "draws": FIXTURE_DRAWS,
    }
    rsw.run(tmp_path, (rsw.BED_SKAB,), rows_dir=None, workers=2, **kwargs)
    single = _summary(root / "a")
    pooled = _summary(tmp_path)
    pd.testing.assert_frame_equal(single[single["bed"] == "skab"].reset_index(drop=True), pooled)


def test_tep_cache_builder_keeps_runs_251_to_350_of_the_fault_free_file(tmp_path):
    pyreadr = pytest.importorskip("pyreadr")
    spec = importlib.util.spec_from_file_location(
        "switchback_build_tep_cache", STUDY / "build_tep_cache.py"
    )
    btc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(btc)
    rng = np.random.default_rng(25)
    parts = []
    for run in (250, 251, 350, 351):
        frame = pd.DataFrame(rng.standard_normal((btc.SAMPLES, 52)), columns=btc.VARIABLES)
        frame.insert(0, "sample", np.arange(1, btc.SAMPLES + 1))
        frame.insert(0, "simulationRun", run)
        frame.insert(0, "faultNumber", 0)
        parts.append(frame)
    source = tmp_path / "tep"
    source.mkdir()
    pyreadr.write_rdata(str(source / btc.SOURCE_FILE), pd.concat(parts), df_name="fault_free")
    (source / "MANIFEST.sha256.json").write_text("{}\n", encoding="utf-8")
    manifest = btc.build(source, tmp_path / "cache")
    assert manifest["n_runs"] == 2 and manifest["n_rows"] == 2 * btc.SAMPLES
    cache = pd.read_parquet(tmp_path / "cache" / btc.CACHE_NAME)
    assert sorted(cache["run"].unique()) == [251, 350]
    assert cache["xmeas_1"].dtype == np.float32

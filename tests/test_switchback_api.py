"""``switchback_plan`` and ``switchback_analyze`` over synthetic archives."""

from __future__ import annotations

import datetime as dt
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import tsdive
from conftest import make_meta, write_archive
from tsdive.errors import IncomparableSamplingError, ScheduleMismatch, SchemaError
from tsdive.store.sampling_contract import CalculationBasis, RetrievalMode, SamplingContract
from tsdive.switchback import archive as sa
from tsdive.switchback.inference import lag_response
from tsdive.switchback.plan import plan_digest

START = pd.Timestamp("2024-03-04 00:00:00+00:00")
TRIAL_END = START + pd.Timedelta(16, unit="h")


def _write(path: Path, stamps, values, point: str, unit: str = "degC", quality=None) -> Path:
    frame = pd.DataFrame(
        {
            "timestamp": stamps,
            "value": values,
            "quality": quality if quality is not None else ["GOOD"] * len(stamps),
        }
    )
    return write_archive(path, frame, make_meta("unit1", point, unit_raw=unit, sample_rate_s=60.0))


def _minutes(start: pd.Timestamp, n: int) -> pd.DatetimeIndex:
    return start + pd.to_timedelta(np.arange(n) * 60, unit="s")


def _trial(tmp_path: Path, plan, *, shift: float, seed: int = 1) -> tuple[Path, Path]:
    """A target and a drifting feed covariate over the plan, the target following the plan.

    The target moves by ``shift`` in B blocks through a 3-minute lag and
    by 0.5 per unit of feed.
    """
    n = int((plan.schedule_end - plan.start).total_seconds() // 60)
    stamps = _minutes(plan.start, n)
    rng = np.random.default_rng(seed)
    feed = 50.0 + np.cumsum(rng.normal(0.0, 0.2, n))
    block = np.arange(n) // (plan.block_s // 60)
    z = plan.observed[block].astype(float)
    response = lag_response(z, np.arange(n, dtype=float), 3.0)
    target = 120.0 + 0.5 * (feed - 50.0) + shift * response + rng.normal(0.0, 0.3, n)
    return (
        _write(tmp_path / "ti201.parquet", stamps, target, "TI201.PV"),
        _write(tmp_path / "fi200.parquet", stamps, feed, "FI200.PV", unit="m3/h"),
    )


def _plan(**kw):
    args = {"start": START, "end": TRIAL_END, "block": "PT1H", "washout": "PT10M", "seed": 42}
    args.update(kw)
    return tsdive.switchback_plan(**args)


# ---------------------------------------------------------------- analyze


def test_an_injected_shift_clears_through_the_archives(tmp_path):
    plan = _plan()
    target, feed = _trial(tmp_path, plan, shift=0.4)
    a = tsdive.switchback_analyze(
        [target, feed], plan, target="unit1:TI201.PV", covariates=["FI200.PV"]
    )
    assert isinstance(a, tsdive.SwitchbackAnalysis)
    assert a.direct.kept_per_block == (50,) * 16
    adjusted = a.adjusted
    assert adjusted is not None and adjusted.reason is None
    assert adjusted.lo > 0 and adjusted.lo <= 0.4 <= adjusted.hi
    assert adjusted.p_value == pytest.approx(1 / 1001)
    assert adjusted.hi - adjusted.lo < a.direct.hi - a.direct.lo
    assert a.unit == "degrees Celsius"
    assert a.unused == ()
    assert "washout 10 min" in a.render()
    assert a.frame["kept"].sum() == 800


def test_a_zero_shift_clears_at_most_the_nominal_rate_over_400_seeds(tmp_path):
    """SCOPE claim: under the declared randomization the claim rate at a zero shift is <= 5%.

    One AR(1) record where nothing was switched, 400 plan seeds. The
    claim rate stays within 0.05 + 3.5 Monte Carlo SE.
    """
    n = 8 * 60
    rng = np.random.default_rng(3)
    noise = rng.standard_normal(n)
    values = np.empty(n)
    values[0] = noise[0]
    for i in range(1, n):
        values[i] = 0.8 * values[i - 1] + noise[i]
    path = _write(tmp_path / "y.parquet", _minutes(START, n), values, "Y.PV")
    claims = 0
    draws = 400
    for seed in range(draws):
        plan = _plan(end=START + pd.Timedelta(8, unit="h"), block="PT30M", washout=0, seed=seed)
        est = tsdive.switchback_analyze([path], plan, target="Y.PV").direct
        claims += bool(est.lo > 0 or est.hi < 0)
    assert claims / draws <= 0.05 + 3.5 * math.sqrt(0.05 * 0.95 / draws)


def test_a_hole_over_a_block_refuses_with_empty_block(tmp_path):
    plan = _plan()
    n = 16 * 60
    stamps = _minutes(START, n)
    hole = (stamps >= START + pd.Timedelta(3, unit="h")) & (
        stamps < START + pd.Timedelta(4, unit="h")
    )
    keep = ~hole
    rng = np.random.default_rng(4)
    values = rng.standard_normal(int(keep.sum()))
    path = _write(tmp_path / "t.parquet", stamps[keep], values, "T.PV")
    a = tsdive.switchback_analyze([path], plan, target="T.PV")
    assert a.direct.reason == "empty_block"
    assert a.direct.kept_per_block[3] == 0
    assert "first block 3 from 2024-03-04T03:00:00+00:00" in (a.direct.detail or "")
    text = a.render()
    assert text.splitlines()[0] == "unit1:T.PV   refused empty_block"
    assert a.to_dict()["direct"]["interval"] is None


def test_microsecond_timestamps_cut_the_same_blocks(tmp_path):
    plan = _plan()
    n = 16 * 60
    values = np.random.default_rng(7).standard_normal(n)
    stamps = _minutes(START, n)
    ns = _write(tmp_path / "ns.parquet", stamps.as_unit("ns"), values, "NS.PV")
    us = _write(tmp_path / "us.parquet", stamps.as_unit("us"), values, "US.PV")
    by_ns = tsdive.switchback_analyze([ns], plan, target="NS.PV").direct
    by_us = tsdive.switchback_analyze([us], plan, target="US.PV").direct
    assert by_us.kept_per_block == by_ns.kept_per_block == (50,) * 16
    assert by_us.estimate == by_ns.estimate


def test_bad_quality_samples_are_left_out(tmp_path):
    plan = _plan()
    n = 16 * 60
    rng = np.random.default_rng(5)
    quality = ["GOOD"] * n
    quality[70:80] = ["BAD"] * 10  # inside block 1, past its washout
    path = _write(tmp_path / "t.parquet", _minutes(START, n), rng.standard_normal(n), "T.PV",
                  quality=quality)
    a = tsdive.switchback_analyze([path], plan, target="T.PV")
    assert a.direct.kept_per_block[1] == 40
    assert a.good_share == pytest.approx((n - 10) / n)


def test_an_edited_plan_file_is_refused(tmp_path):
    """SCOPE claim: ``switchback analyze`` refuses a plan edited after it was written."""
    plan = _plan()
    target, feed = _trial(tmp_path, plan, shift=0.0)
    path = plan.write_json(tmp_path / "plan.json")
    doc = json.loads(path.read_text(encoding="utf-8"))
    first = doc["schedule"][0]
    first["setting"] = "A" if first["setting"] == "B" else "B"
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ScheduleMismatch, match="does not match its schedule"):
        tsdive.switchback_analyze([target, feed], path, target="TI201.PV")


def test_an_unbalanced_plan_is_refused(tmp_path):
    """SCOPE claim: ``switchback analyze`` refuses an unbalanced schedule."""
    plan = _plan()
    target, _ = _trial(tmp_path, plan, shift=0.0)
    blocks = [replace(b, setting="B") for b in plan.blocks]
    edited = replace(plan, blocks=tuple(blocks))
    edited = replace(edited, digest=plan_digest(edited))
    with pytest.raises(ScheduleMismatch, match="not balanced: 16 of 16"):
        tsdive.switchback_analyze([target], edited, target="TI201.PV")


def test_reads_under_different_sampling_contracts_raise(tmp_path, monkeypatch):
    plan = _plan()
    target, feed = _trial(tmp_path, plan, shift=0.0)
    read = sa.read_span

    def event_weighted_covariate(path, start, end):
        window = read(path, start, end)
        if Path(path) == feed:
            contract = SamplingContract(CalculationBasis.EVENT_WEIGHTED, RetrievalMode.RECORDED)
            window = replace(window, contract=contract)
        return window

    monkeypatch.setattr(sa, "read_span", event_weighted_covariate)
    with pytest.raises(IncomparableSamplingError):
        tsdive.switchback_analyze([target, feed], plan, target="TI201.PV", covariates=["FI200.PV"])


def test_a_covariate_repeating_a_timestamp_raises_schema_error(tmp_path):
    plan = _plan()
    target, _ = _trial(tmp_path, plan, shift=0.0)
    stamps = _minutes(START, 16 * 60)
    stamps = stamps.insert(10, stamps[10])
    dup = _write(tmp_path / "d.parquet", stamps, np.arange(len(stamps), dtype=float), "D.PV")
    with pytest.raises(SchemaError, match="duplicate timestamps"):
        tsdive.switchback_analyze([target, dup], plan, target="TI201.PV", covariates=["D.PV"])


def test_tags_are_resolved_and_unused_archives_are_listed(tmp_path):
    plan = _plan()
    target, feed = _trial(tmp_path, plan, shift=0.0)
    a = tsdive.switchback_analyze([target, feed], plan, target="TI201.PV")
    assert a.unused == ("unit1:FI200.PV",)
    assert a.adjusted is None
    assert "unused    unit1:FI200.PV" in a.render()
    with pytest.raises(ValueError, match=r"no archive carries the target XX\.PV"):
        tsdive.switchback_analyze([target, feed], plan, target="XX.PV")
    with pytest.raises(ValueError, match="named twice"):
        tsdive.switchback_analyze(
            [target, feed], plan, target="TI201.PV", covariates=["FI200.PV", "FI200.PV"]
        )
    with pytest.raises(ValueError, match="also named as a covariate"):
        tsdive.switchback_analyze(
            [target, feed], plan, target="TI201.PV", covariates=["unit1:TI201.PV"]
        )


def test_too_many_covariates_refuse_the_adjusted_analysis_only(tmp_path):
    plan = _plan(end=START + pd.Timedelta(8, unit="h"), block="PT30M", washout="PT28M")
    n = 8 * 60
    rng = np.random.default_rng(6)
    stamps = _minutes(START, n)
    paths = [_write(tmp_path / "t.parquet", stamps, rng.standard_normal(n), "T.PV")]
    names = []
    for i in range(4):
        paths.append(_write(tmp_path / f"x{i}.parquet", stamps, rng.standard_normal(n), f"X{i}.PV"))
        names.append(f"X{i}.PV")
    a = tsdive.switchback_analyze(paths, plan, target="T.PV", covariates=names)
    assert a.direct.reason is None and a.direct.n_kept == 32
    assert a.adjusted is not None and a.adjusted.reason == "too_many_covariates"
    assert "4 covariates over 32 kept samples" in (a.adjusted.detail or "")


def test_an_open_interval_renders_as_unbounded(tmp_path):
    plan = _plan()
    target, feed = _trial(tmp_path, plan, shift=0.4)
    a = tsdive.switchback_analyze([target, feed], plan, target="TI201.PV", covariates=["FI200.PV"])
    assert a.adjusted is not None
    open_both = replace(a, adjusted=replace(a.adjusted, lo=-math.inf, hi=math.inf))
    assert "  95%       [unbounded, unbounded]" in open_both.render().splitlines()
    doc = json.loads(json.dumps(open_both.to_dict()))
    assert doc["adjusted"]["interval"] == {
        "lo": None,
        "hi": None,
        "lo_unbounded": True,
        "hi_unbounded": True,
    }
    open_low = replace(a, direct=replace(a.direct, lo=-math.inf))
    interval = [line for line in open_low.render().splitlines() if line.startswith("  95%")]
    assert interval[0].startswith("  95%       [unbounded, +")
    assert open_low.to_dict()["direct"]["interval"]["hi"] == pytest.approx(a.direct.hi)


def test_render_and_json_carry_the_same_numbers(tmp_path):
    plan = _plan()
    target, feed = _trial(tmp_path, plan, shift=0.4)
    a = tsdive.switchback_analyze([target, feed], plan, target="TI201.PV", covariates=["FI200.PV"])
    doc = json.loads(json.dumps(a.to_dict()))
    assert doc["plan"]["digest"] == plan.digest and doc["plan"]["verified"] is True
    assert doc["plan"]["assignments"] == 12870
    assert doc["settings"] == [b.setting for b in plan.blocks]
    assert doc["direct"]["estimate"] == pytest.approx(a.direct.estimate)
    assert doc["adjusted"]["covariates"] == ["unit1:FI200.PV"]
    assert doc["assumptions"] == sa.ASSUMPTIONS
    lines = a.render().splitlines()
    assert lines[0].startswith("unit1:TI201.PV   B - A ")
    assert lines[-3:] == [
        "  difference between settings A and B under the declared random schedule, by",
        "  intended assignment; holds if the schedule was followed and carryover ended",
        "  within the washout",
    ]
    assert all(len(line) < 80 for line in lines)


# ---------------------------------------------------------------- plan


def test_plan_arguments_take_iso_durations_seconds_and_timedeltas():
    first = _plan()
    assert _plan(block=3600, washout=600) == first
    assert _plan(block=dt.timedelta(hours=1), washout=pd.Timedelta(10, unit="min")) == first
    assert _plan(start="2024-03-04T00:00:00Z", end="2024-03-04T16:00:00Z") == first
    with pytest.raises(ValueError, match="whole number of seconds"):
        _plan(block=1.5)
    with pytest.raises(ValueError, match="washout P1M is not days, hours, minutes and seconds"):
        _plan(washout="P1M")
    with pytest.raises(ValueError, match="both history and history_window"):
        _plan(history="x.parquet")
    with pytest.raises(ValueError, match="naive"):
        _plan(start=pd.Timestamp("2024-03-04 00:00:00"))


def _history(tmp_path: Path, days: int = 2) -> tuple[Path, str]:
    """Two days of the target before the trial, nothing switched."""
    h_start = START - pd.Timedelta(days, unit="D")
    n = days * 24 * 60
    rng = np.random.default_rng(8)
    path = _write(tmp_path / "hist.parquet", _minutes(h_start, n), rng.normal(0, 0.3, n), "T.PV")
    return path, f"{h_start.isoformat()}/{START.isoformat()}"


def test_a_plan_over_history_reads_its_power(tmp_path):
    path, window = _history(tmp_path)
    plan = _plan(history=path, history_window=window)
    power = plan.power
    assert power is not None and power.reason is None
    assert power.tag == "unit1:T.PV" and power.draws == 200
    assert power.end - power.start == pd.Timedelta(16, unit="h")
    assert power.n_kept == 800
    assert power.rates[0] <= 0.05 + 3.5 * math.sqrt(0.05 * 0.95 / 200)
    assert list(power.rates) == sorted(power.rates)
    assert power.smallest is not None and power.smallest in power.deltas
    assert plan.digest == _plan().digest
    text = plan.render()
    smallest = next(line for line in text.splitlines() if line.startswith("  smallest"))
    assert smallest.endswith(" with detection >= 0.8")
    assert f"{power.smallest:g} sigma (" in smallest
    back = tsdive.SwitchbackPlan.read_json(plan.write_json(tmp_path / "plan.json"))
    assert back == plan


def test_a_short_history_refuses_the_readout_not_the_plan(tmp_path):
    path, _ = _history(tmp_path)
    window = f"{START - pd.Timedelta(8, unit='h')}/{START}".replace(" ", "T")
    plan = _plan(history=path, history_window=window)
    assert plan.k == 16
    assert plan.power is not None and plan.power.reason == "history_short"
    assert "the history window spans 8 h and the schedule 16 h" in (plan.power.detail or "")
    assert "refused   history_short" in plan.render()


def test_a_covariate_the_setting_moves_is_flagged_not_refused(tmp_path):
    plan = _plan()
    target, feed = _trial(tmp_path, plan, shift=0.0)
    n = int((plan.schedule_end - plan.start).total_seconds() // 60)
    z = plan.observed[np.arange(n) // 60].astype(float)
    rng = np.random.default_rng(3)
    output = 40.0 + 6.0 * lag_response(z, np.arange(n, dtype=float), 3.0) + rng.normal(0, 0.2, n)
    stamps = _minutes(plan.start, n)
    target = _write(tmp_path / "ti201.parquet", stamps,
                    120.0 + 0.1 * (output - 40.0) + rng.normal(0, 0.3, n), "TI201.PV")
    op = _write(tmp_path / "fc200.parquet", stamps, output, "FC200.OP", unit="%")
    a = tsdive.switchback_analyze([target, feed, op], plan, target="TI201.PV",
                                  covariates=["FI200.PV", "FC200.OP"])
    assert [c.tag for c in a.covariate_checks] == ["unit1:FI200.PV", "unit1:FC200.OP"]
    assert [c.tag for c in a.moving_covariates] == ["unit1:FC200.OP"]
    moved = a.moving_covariates[0].difference
    assert moved.p_value is not None and moved.p_value < 0.05
    assert not a.direct.refused and a.adjusted is not None and not a.adjusted.refused
    text = a.render()
    assert "covariate unit1:FC200.OP moves with the setting (p " in text
    assert "report the unadjusted one" in " ".join(text.split())
    assert "covariate unit1:FI200.PV moves" not in text
    checks = a.to_dict()["covariate_checks"]
    assert [c["moves_with_setting"] for c in checks] == [False, True]
    assert checks[1]["unit"] == "percent" and checks[1]["p_value"] == moved.p_value


def test_an_edited_plan_names_the_digest_to_restore(tmp_path):
    plan = _plan()
    target, feed = _trial(tmp_path, plan, shift=0.0)
    path = plan.write_json(tmp_path / "plan.json")
    doc = json.loads(path.read_text(encoding="utf-8"))
    first = doc["schedule"][0]
    first["setting"] = "A" if first["setting"] == "B" else "B"
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ScheduleMismatch) as info:
        tsdive.switchback_analyze([target, feed], path, target="TI201.PV")
    message = str(info.value)
    assert f"restore the plan file whose digest starts {plan.digest[:12]}" in message
    assert "the analysis needs the plan as drawn" in message
    assert "plan the trial again" not in message


def test_a_short_history_line_states_the_window_passed_and_the_span_needed(tmp_path):
    path, _ = _history(tmp_path)
    h_start = START - pd.Timedelta(8, unit="h")
    plan = _plan(history=path, history_window=f"{h_start}/{START}".replace(" ", "T"))
    assert plan.power is not None
    assert (plan.power.start, plan.power.end) == (h_start, START)
    history = next(ln for ln in plan.render().splitlines() if ln.startswith("  history"))
    assert history.endswith("2024-03-03 16:00:00Z -> 2024-03-04 00:00:00Z   "
                            "(the schedule needs 16 h)")


def test_the_plan_window_help_states_the_offset_conversion():
    from tsdive.cli import _parser_switchback_plan

    helps = {a.dest: a.help for a in _parser_switchback_plan()._actions}
    assert "converted to UTC" in helps["window"]
    assert "converted to UTC" in helps["history_window"]
    plan = _plan(start=pd.Timestamp("2024-03-04 02:00:00+02:00"),
                 end=pd.Timestamp("2024-03-04 18:00:00+02:00"))
    assert plan.start == START

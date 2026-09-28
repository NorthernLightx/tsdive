"""``tsdive switchback plan`` and ``tsdive switchback analyze`` end to end."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import tsdive
from conftest import make_meta, write_archive
from tsdive.cli import cmd_run, main
from tsdive.switchback.inference import lag_response

START = pd.Timestamp("2024-03-04 00:00:00+00:00")
WINDOW = "2024-03-04T00:00:00Z/2024-03-04T16:00:00Z"
PLAN_ARGS = ["--window", WINDOW, "--block", "PT1H", "--washout", "PT10M", "--seed", "42"]


def _write(path: Path, stamps, values, point: str, unit: str) -> Path:
    frame = pd.DataFrame({"timestamp": stamps, "value": values, "quality": ["GOOD"] * len(stamps)})
    return write_archive(path, frame, make_meta("unit1", point, unit_raw=unit, sample_rate_s=60.0))


def _trial(tmp_path: Path, plan: tsdive.SwitchbackPlan) -> tuple[Path, Path]:
    n = 16 * 60
    stamps = START + pd.to_timedelta(np.arange(n) * 60, unit="s")
    rng = np.random.default_rng(1)
    feed = 50.0 + np.cumsum(rng.normal(0.0, 0.2, n))
    z = plan.observed[np.arange(n) // 60].astype(float)
    target = (
        120.0
        + 0.5 * (feed - 50.0)
        + 0.4 * lag_response(z, np.arange(n, dtype=float), 3.0)
        + rng.normal(0.0, 0.3, n)
    )
    return (
        _write(tmp_path / "ti201.parquet", stamps, target, "TI201.PV", "degC"),
        _write(tmp_path / "fi200.parquet", stamps, feed, "FI200.PV", "m3/h"),
    )


def _run(capsys, argv: list[str]) -> tuple[int, str, str]:
    rc = main(argv)
    captured = capsys.readouterr()
    return rc, captured.out, captured.err


def test_plan_writes_the_file_and_prints_the_schedule(tmp_path, capsys):
    out = tmp_path / "plan.json"
    rc, text, _ = _run(capsys, ["switchback", "plan", *PLAN_ARGS, "-o", str(out), "--no-color"])
    assert rc == 0
    lines = text.splitlines()
    plan = tsdive.SwitchbackPlan.read_json(out)
    assert lines[0] == f"switchback plan   16 blocks of 1 h   A 8   B 8   digest {plan.digest[:12]}"
    assert f"wrote     {out.as_posix()}" in lines
    assert "washout   10 min at the start of every block" in lines
    assert plan == tsdive.switchback_plan(START, START + pd.Timedelta(16, unit="h"), 3600, 600, 42)
    assert all(len(line) < 80 for line in lines if not line.startswith("wrote"))


def test_a_zero_washout_prints_as_none(tmp_path, capsys):
    args = [*PLAN_ARGS[:4], "--washout", "PT0S", *PLAN_ARGS[6:]]
    rc, text, _ = _run(capsys, ["switchback", "plan", *args, "-o", str(tmp_path / "p.json")])
    assert rc == 0
    assert "washout   none" in text.splitlines()


def test_plan_json_prints_the_plan_document(tmp_path, capsys):
    out = tmp_path / "plan.json"
    rc, text, _ = _run(capsys, ["switchback", "plan", *PLAN_ARGS, "-o", str(out), "--json"])
    assert rc == 0
    assert json.loads(text) == json.loads(out.read_text(encoding="utf-8"))


def test_json_before_the_command_name_reaches_the_subcommand(tmp_path, capsys):
    out = tmp_path / "plan.json"
    rc, text, _ = _run(capsys, ["--json", "switchback", "plan", *PLAN_ARGS, "-o", str(out)])
    assert rc == 0
    assert json.loads(text)["digest"] == tsdive.SwitchbackPlan.read_json(out).digest


def test_plan_refuses_an_existing_file_and_a_small_design(tmp_path, capsys):
    out = tmp_path / "plan.json"
    out.write_text("{}", encoding="utf-8")
    rc, _, err = _run(capsys, ["switchback", "plan", *PLAN_ARGS, "-o", str(out)])
    assert rc == 2 and "exists; a plan file records one randomization" in err
    rc, _, err = _run(
        capsys,
        [
            "switchback", "plan", "--window", "2024-03-04T00:00:00Z/PT4H", "--block", "PT1H",
            "--washout", "PT0S", "--seed", "1", "-o", str(tmp_path / "small.json"),
        ],
    )
    assert rc == 2
    assert err.startswith("[DesignTooSmall] 4 blocks give 6 balanced assignments")
    assert not (tmp_path / "small.json").exists()


def test_analyze_prints_text_and_json(tmp_path, capsys):
    plan_path = tmp_path / "plan.json"
    assert main(["switchback", "plan", *PLAN_ARGS, "-o", str(plan_path)]) == 0
    capsys.readouterr()
    target, feed = _trial(tmp_path, tsdive.SwitchbackPlan.read_json(plan_path))
    argv = [
        "switchback", "analyze", str(target), str(feed), "--plan", str(plan_path),
        "--target", "TI201.PV", "--covariate", "FI200.PV",
    ]
    rc, text, _ = _run(capsys, [*argv, "--no-color"])
    assert rc == 0
    analysis = tsdive.switchback_analyze(
        [target, feed], plan_path, target="TI201.PV", covariates=["FI200.PV"]
    )
    assert text == analysis.render() + "\n"
    assert text.splitlines()[0].startswith("unit1:TI201.PV   B - A ")
    assert "Adjusted  (OLS on 1 covariate)" in text
    rc, payload, _ = _run(capsys, [*argv, "--json"])
    assert rc == 0
    assert json.loads(payload) == json.loads(json.dumps(analysis.to_dict()))


def test_analyze_refusals_print_the_error_class(tmp_path, capsys):
    plan_path = tmp_path / "plan.json"
    assert main(["switchback", "plan", *PLAN_ARGS, "-o", str(plan_path)]) == 0
    target, _ = _trial(tmp_path, tsdive.SwitchbackPlan.read_json(plan_path))
    capsys.readouterr()
    argv = ["switchback", "analyze", str(target), "--plan", str(plan_path), "--target"]
    rc, _, err = _run(capsys, [*argv, "NOPE"])
    assert rc == 2 and err.startswith("error: no archive carries the target NOPE")
    doc = json.loads(plan_path.read_text(encoding="utf-8"))
    doc["seed"] = 43
    plan_path.write_text(json.dumps(doc), encoding="utf-8")
    rc, _, err = _run(capsys, [*argv, "TI201.PV"])
    assert rc == 2 and err.startswith("[ScheduleMismatch] plan digest ")


@pytest.mark.parametrize(
    ("argv", "code", "phrase"),
    [
        (["switchback"], 2, "tsdive switchback - plan and analyze"),
        (["switchback", "--help"], 0, "--help after plan or analyze"),
        (["switchback", "fit"], 2, "unknown switchback command 'fit'"),
    ],
)
def test_switchback_without_a_subcommand_prints_its_usage(capsys, argv, code, phrase):
    rc, _, err = _run(capsys, argv)
    assert rc == code and phrase in err


def test_the_main_help_names_both_commands(capsys):
    main(["--help"])
    err = capsys.readouterr().err
    assert "switchback plan --window START/END" in err
    assert "switchback analyze <parquet...> --plan PLAN.json" in err


def test_a_run_plan_walks_switchback_with_a_plan_file_beside_it(tmp_path):
    plan = tsdive.switchback_plan(START, START + pd.Timedelta(16, unit="h"), 3600, 600, 42)
    plan.write_json(tmp_path / "trial.json")
    _trial(tmp_path, plan)
    toml = tmp_path / "run.toml"
    toml.write_text(
        'archives = ["*.parquet"]\n'
        'steps = ["switchback"]\n\n'
        "[options.switchback]\n"
        'plan = "trial.json"\n'
        'target = "TI201.PV"\n'
        'covariate = ["FI200.PV"]\n',
        encoding="utf-8",
    )
    out = tmp_path / "out"
    assert cmd_run([str(toml), "-o", str(out)]) == 0
    ledger = json.loads((out / "ledger.json").read_text(encoding="utf-8"))
    (finding,) = ledger["findings"]
    assert finding["step"] == "switchback"
    assert finding["text"].startswith("unit1:TI201.PV   B - A ")
    assert ledger["refusals"] == []

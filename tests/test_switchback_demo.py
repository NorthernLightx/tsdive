"""The switchback demo trial, checked against what docs/SWITCHBACK.md reads off it."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

import tsdive
from tsdive.store.tagstore import write_tag

DEMO = Path(__file__).parents[1] / "examples" / "switchback" / "make_trial.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("make_trial", DEMO)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["make_trial"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def trial(tmp_path_factory) -> tuple[ModuleType, dict[str, Path]]:
    module = _load()
    out = tmp_path_factory.mktemp("switchback_demo")
    paths = {
        name: write_tag(out / f"{name}.parquet", frame, meta, overwrite=True)
        for name, (frame, meta) in module.build().items()
    }
    return module, paths


def test_the_covariates_recover_the_shift_the_direct_estimate_misses(trial):
    module, paths = trial
    plan = tsdive.switchback_plan(*module.TRIAL, module.BLOCK, module.WASHOUT, module.SEED)
    a = tsdive.switchback_analyze(
        list(paths.values()), plan, target="TI201.PV", covariates=["FI200.PV", "TT001.PV"]
    )
    assert a.direct.lo < 0 < a.direct.hi
    assert a.adjusted is not None
    assert 0 < a.adjusted.lo <= module.SHIFT <= a.adjusted.hi
    assert a.direct.kept_per_block == (45,) * 24


def test_the_history_readout_reaches_no_grid_shift_at_0_8(trial):
    module, paths = trial
    plan = tsdive.switchback_plan(
        *module.TRIAL,
        module.BLOCK,
        module.WASHOUT,
        module.SEED,
        history=paths["ti201"],
        history_window="2024-06-02T00:00:00Z/2024-06-03T00:00:00Z",
    )
    assert plan.power is not None and plan.power.reason is None
    assert plan.power.smallest is None
    assert plan.power.rates[-1] < 0.8

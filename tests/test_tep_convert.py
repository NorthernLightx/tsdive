"""The TEP converter, exercised offline against an in-test source frame.

No ``pyreadr`` and no ``.RData``: the frame those two would have produced
is built here, so what is tested is the converter's decisions rather than
the R reader. Those decisions are the ones TEP forces - a simulation has
no clock, no quality column, and a fault that starts at a known sample
rather than at the top of the file.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

import tsdive
from tsdive.store.identity import Role
from tsdive.store.quality import Severity
from tsdive.store.tagstore import meta_from_parquet
from tsdive.store.units import _dimensionality, resolve_unit

SCRIPTS = Path(__file__).parents[1] / "scripts"


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


convert_tep = _load("convert_tep")

N_SAMPLES = 240  # 12 h at 180 s, so four 3 h windows fit
FAULT = 4
RUN = 7

# The four spellings the shipped alias table does not carry. Pinned here
# because the study publishes the count: if the table gains one of them,
# this test says so instead of the number quietly moving.
UNRESOLVED_SPELLINGS = {"kscmh", "kPa gauge", "mol%", "kW"}


def source_frame(fault: int = FAULT, run: int = RUN, n: int = N_SAMPLES) -> pd.DataFrame:
    """The columns ``pyreadr`` hands back from a TEP ``.RData``."""
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "faultNumber": np.full(n, float(fault)),
            "simulationRun": np.full(n, float(run)),
            "sample": np.arange(1, n + 1, dtype="int32"),
        }
    )
    for i, variable in enumerate(convert_tep.VARIABLES):
        frame[variable] = rng.normal(loc=float(i + 1), scale=0.5, size=n)
    return frame


@pytest.fixture()
def converted(tmp_path: Path) -> tuple[Path, dict]:
    record = convert_tep.convert_run(
        source_frame(),
        tmp_path / "archives",
        split="train",
        fault=FAULT,
        run=RUN,
        source="TEP_Faulty_Training.RData",
        source_md5="0" * 32,
    )
    return convert_tep.run_dir(tmp_path / "archives", "train", FAULT, RUN), record


def test_every_variable_plus_the_label_becomes_an_archive(
    converted: tuple[Path, dict],
) -> None:
    out_dir, record = converted
    assert len(convert_tep.VARIABLES) == 52
    assert record["archives_written"] == [*convert_tep.VARIABLES, "LABEL_fault"]
    assert len(list(out_dir.glob("*.parquet"))) == 53
    assert out_dir.name == "fault04_run007"
    assert out_dir.parent.name == "train"


def test_timestamps_are_synthetic_and_say_so(converted: tuple[Path, dict]) -> None:
    """TEP numbers its rows; it does not stamp them. The clock is invented."""
    out_dir, record = converted
    assert record["timestamp_synthetic"] is True
    assert record["t0_utc"] == "2000-01-01T00:00:00+00:00"
    assert record["sample_rate_s"] == 180.0
    on_disk = json.loads((out_dir / "RUN.json").read_text("utf-8"))
    assert on_disk["timestamp_synthetic"] is True

    frame = pd.read_parquet(out_dir / "xmeas_1.parquet")
    stamps = pd.DatetimeIndex(frame["timestamp"])
    assert str(stamps[0]) == "2000-01-01 00:00:00+00:00"
    assert (stamps.to_series().diff().dropna() == pd.Timedelta(180, unit="s")).all()
    assert stamps[-1] - stamps[0] == pd.Timedelta(180 * (N_SAMPLES - 1), unit="s")


def test_label_fault_is_zero_until_the_onset_sample(converted: tuple[Path, dict]) -> None:
    out_dir, record = converted
    assert record["onset_sample"] == 20
    frame = pd.read_parquet(out_dir / "LABEL_fault.parquet")
    values = list(frame["value"])
    assert values[:19] == ["0"] * 19  # samples 1-19
    assert values[19] == "4"  # sample 20 is the first faulty row
    assert set(values[19:]) == {"4"}
    assert record["n_faulty_rows"] == N_SAMPLES - 19

    meta = meta_from_parquet(out_dir / "LABEL_fault.parquet")
    assert meta.role is Role.MODE
    assert meta.unit_raw is None


def test_the_testing_split_steps_the_fault_in_at_sample_160(tmp_path: Path) -> None:
    record = convert_tep.convert_run(
        source_frame(fault=2, run=1, n=200),
        tmp_path / "archives",
        split="test",
        fault=2,
        run=1,
        source="TEP_Faulty_Testing.RData",
        source_md5=None,
    )
    assert record["onset_sample"] == 160
    out_dir = convert_tep.run_dir(tmp_path / "archives", "test", 2, 1)
    values = list(pd.read_parquet(out_dir / "LABEL_fault.parquet")["value"])
    assert values[158] == "0"
    assert values[159] == "2"


def test_a_fault_free_run_is_labelled_zero_throughout(tmp_path: Path) -> None:
    """The same rule, not a special case: fault 0 after the onset is still "0"."""
    convert_tep.convert_run(
        source_frame(fault=0, run=3, n=60),
        tmp_path / "archives",
        split="train",
        fault=0,
        run=3,
        source="TEP_FaultFree_Training.RData",
        source_md5=None,
    )
    out_dir = convert_tep.run_dir(tmp_path / "archives", "train", 0, 3)
    values = set(pd.read_parquet(out_dir / "LABEL_fault.parquet")["value"])
    assert values == {"0"}


def test_quality_is_assumed_and_the_simulated_code_is_declared(
    converted: tuple[Path, dict],
) -> None:
    out_dir, record = converted
    assert record["quality_codes"] == {"SIMULATED": "GOOD"}
    meta = meta_from_parquet(out_dir / "xmeas_9.parquet")
    assert meta.quality_assumed is True
    assert meta.quality_codes == {"SIMULATED": "GOOD"}
    assert meta.unit_raw == "degC"
    assert meta.eng_range is None
    assert meta.sample_rate_s == 180.0
    assert meta.asset == "TEP"
    assert meta.loop_id == "fault04_run007"
    assert meta.role is Role.PV
    profile = tsdive.profile(out_dir / "xmeas_9.parquet")
    assert profile.physics.severity_counts[Severity.GOOD] == N_SAMPLES
    assert "quality ASSUMED at ingest" in profile.render()


def test_a_non_finite_value_is_refused_rather_than_coded_good(tmp_path: Path) -> None:
    """Writing SIMULATED=GOOD over a NaN would be a lie about the source."""
    frame = source_frame(n=30)
    frame.loc[5, "xmeas_3"] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        convert_tep.convert_run(
            frame,
            tmp_path / "archives",
            split="train",
            fault=FAULT,
            run=RUN,
            source="x.RData",
            source_md5=None,
        )


def test_manipulated_variables_carry_the_op_role(converted: tuple[Path, dict]) -> None:
    out_dir, _ = converted
    assert meta_from_parquet(out_dir / "xmv_1.parquet").role is Role.OP
    assert meta_from_parquet(out_dir / "xmeas_41.parquet").role is Role.PV


def test_conversion_is_idempotent(tmp_path: Path) -> None:
    kwargs = {
        "split": "train",
        "fault": FAULT,
        "run": RUN,
        "source": "x.RData",
        "source_md5": None,
    }
    first = convert_tep.convert_run(source_frame(n=30), tmp_path / "a", **kwargs)
    again = convert_tep.convert_run(source_frame(n=30), tmp_path / "a", **kwargs)
    assert first["skipped"] is False
    assert again["skipped"] is True
    assert again["archives_written"] == first["archives_written"]


def test_the_unit_table_covers_every_variable_once() -> None:
    assert len(convert_tep.UNITS) == 52
    assert set(convert_tep.UNITS) == set(convert_tep.VARIABLES)
    # Downs & Vogel put the 19 composition analysers on a mole basis and
    # every manipulated variable on a percent-of-range basis.
    assert {v for v, u in convert_tep.UNITS.items() if u == "mol%"} == {
        f"xmeas_{i}" for i in range(23, 42)
    }
    assert {v for v, u in convert_tep.UNITS.items() if u == "%"} == (
        {f"xmv_{i}" for i in range(1, 12)} | {"xmeas_8", "xmeas_12", "xmeas_15"}
    )


def test_every_resolvable_tep_unit_parses_and_the_rest_stay_unresolved() -> None:
    """A spelling that resolves must reach pint; the four that do not are named."""
    unresolved = set()
    for spelling in set(convert_tep.UNITS.values()):
        resolution = resolve_unit(spelling)
        if resolution.canonical is None:
            unresolved.add(spelling)
            continue
        assert resolution.pint_units, f"{spelling} resolved with no pint_units"
        _dimensionality(resolution.pint_units)  # raises if pint cannot read it
    assert unresolved == UNRESOLVED_SPELLINGS


def test_payloads_split_by_fault_and_run_and_honour_the_caps(tmp_path: Path) -> None:
    frames = [source_frame(fault=f, run=r, n=10) for f in (0, 1) for r in (1, 2, 3)]
    frame = pd.concat(frames, ignore_index=True)
    jobs = convert_tep.payloads_for(
        frame,
        tmp_path / "out",
        split="train",
        source="x.RData",
        source_md5=None,
        faults={1},
        runs_per_fault=2,
        force=False,
    )
    assert [(j["fault"], j["run"]) for j in jobs] == [(1, 1), (1, 2)]
    assert list(jobs[0]["frame"].columns) == [
        "faultNumber",
        "simulationRun",
        "sample",
        *convert_tep.VARIABLES,
    ]

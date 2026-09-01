"""The demo builders, checked against what mspc says about their burst.

The README transcripts are generated from these two tags, so the comments
describing the burst are claims about a run that can be made. This makes
that run.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from tsdive.cli import cmd_mspc
from tsdive.store.tagstore import write_tag

SCRIPTS = Path(__file__).parents[1] / "scripts"

# The window the README's mspc block uses.
BASELINE = "2024-03-30T20:00:00Z/2024-03-30T23:00:00Z"
WINDOW = "2024-03-31T04:00:00Z/2024-03-31T06:00:00Z"


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


make_demo_archive = _load("make_demo_archive")


def test_the_demo_burst_lands_in_spe_and_in_part_of_t2(tmp_path, capsys):
    flow, flow_meta = make_demo_archive.build_fic101()
    temp, temp_meta = make_demo_archive.build_tic101(flow)
    flow_path = write_tag(tmp_path / "fic101.parquet", flow, flow_meta, overwrite=True)
    temp_path = write_tag(tmp_path / "tic101.parquet", temp, temp_meta, overwrite=True)

    rc = cmd_mspc(
        [
            str(flow_path),
            str(temp_path),
            "--baseline",
            BASELINE,
            "--window",
            WINDOW,
        ]
    )
    out = capsys.readouterr().out.splitlines()
    assert rc == 0
    assert any(ln.startswith("window    ") and "rows 121" in ln for ln in out)
    # What build_tic101's comment claims: SPE carries most of the burst
    # and T2 is not quiet through it.
    assert "SPE breaches  108" in out
    assert "T2 breaches  22" in out

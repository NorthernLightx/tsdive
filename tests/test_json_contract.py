"""The key paths of every JSON document tsdive writes, against a committed list.

A script or an MCP client reads these keys. The test writes the demo
data, runs each command under ``--json``, ``tsdive run`` for its
``ledger.json``, and the MCP ``screen`` and ``spc`` tools, and lists
every key path of each document, reading the first element of each
list. ``tests/data/json_keys.json`` holds the list. After an intended
change, regenerate it with:

    uv run python tests/test_json_contract.py
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

import tsdive
from tsdive import mcp_server
from tsdive.cli import main
from tsdive.store.tagstore import meta_from_parquet, meta_to_dict

KEYS = Path(__file__).parent / "data" / "json_keys.json"

FIC = "data/demo/fic101_demo.parquet"
TIC = "data/demo/tic101_demo.parquet"
TRIAL = sorted(f"data/switchback_demo/{n}.parquet" for n in ("fi200", "ti201", "tt001"))
BASELINE = "2024-03-30T20:00:00Z/2024-03-31T01:00:00Z"
WINDOW = "2024-03-31T01:00:00Z/2024-03-31T06:00:00Z"
BEFORE = "2024-03-30T20:00:00Z/2024-03-30T23:00:00Z"
AFTER = "2024-03-31T04:00:00Z/2024-03-31T06:00:00Z"
# The flow sits at full scale from 02:01, so this baseline is censored.
CENSORED = "2024-03-31T02:00:00Z/2024-03-31T02:30:00Z"

PLAN = """\
archives = ["data/demo/*.parquet"]
window   = "2024-03-31T01:00:00Z/2024-03-31T06:00:00Z"
baseline = "2024-03-30T20:00:00Z/2024-03-31T01:00:00Z"
before   = "2024-03-30T20:00:00Z/2024-03-30T23:00:00Z"
after    = "2024-03-31T04:00:00Z/2024-03-31T06:00:00Z"
steps    = ["profile", "segment", "screen", "spc", "mspc", "compare"]

[options.profile]
flatline = true
"""

# Each CLI document: (name, argv, the exit status the command returns).
COMMANDS: list[tuple[str, list[str], int]] = [
    ("profile", ["profile", FIC, "--flatline", "--tz", "Europe/London"], 0),
    ("segment", ["segment", FIC], 0),
    ("screen", ["screen", FIC, "--baseline", BASELINE, "--window", WINDOW], 0),
    ("spc", ["spc", FIC, "--baseline", BASELINE, "--window", WINDOW], 0),
    ("mspc", ["mspc", FIC, TIC, "--baseline", BEFORE, "--window", AFTER], 0),
    ("compare", ["compare", FIC, TIC, "--before", BEFORE, "--after", AFTER], 0),
    (
        "switchback plan",
        [
            "switchback", "plan", "--window", "2024-06-03T00:00:00Z/2024-06-04T00:00:00Z",
            "--block", "PT1H", "--washout", "PT15M", "--seed", "7",
            "--history", "data/switchback_demo/ti201.parquet",
            "--history-window", "2024-06-02T00:00:00Z/2024-06-03T00:00:00Z",
            "-o", "plan.json",
        ],
        0,
    ),
    (
        "switchback analyze",
        [
            "switchback", "analyze", *TRIAL, "--plan", "plan.json", "--target", "TI201.PV",
            "--covariate", "FI200.PV", "--covariate", "TT001.PV",
        ],
        0,
    ),
    ("ingest", ["ingest", "fic101.csv", "--out", "FIC101.PV.parquet", "--meta", "fic101.json"], 0),
    ("ingest --init-meta", ["ingest", "fic101.csv", "--init-meta", "template.json"], 0),
    ("refusal", ["screen", FIC, "--baseline", CENSORED, "--window", AFTER], 3),
]


def key_paths(value: object, prefix: str = "") -> list[str]:
    """Every key path under ``value``; a list contributes its first element as ``[]``."""
    if isinstance(value, dict):
        paths: list[str] = []
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            paths += [path, *key_paths(item, path)]
        return paths
    if isinstance(value, list) and value:
        return key_paths(value[0], f"{prefix}[]")
    return []


@contextlib.contextmanager
def _inside(directory: Path) -> Iterator[None]:
    before = Path.cwd()
    os.chdir(directory)
    try:
        yield
    finally:
        os.chdir(before)


def _json(argv: list[str], status: int) -> dict:
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
        code = main(["--json", *argv])
    assert code == status, f"tsdive {' '.join(argv)} exited {code}"
    return json.loads(out.getvalue())


def documents(work: Path) -> dict[str, dict]:
    """Every JSON document, by name, written in the directory ``work``."""
    tsdive.write_demo_data(work / "data")
    with _inside(work):
        pd.read_parquet(FIC).to_csv("fic101.csv", index=False)
        meta = meta_to_dict(meta_from_parquet(Path(FIC)))
        Path("fic101.json").write_text(json.dumps(meta), encoding="utf-8")
        docs = {name: _json(argv, status) for name, argv, status in COMMANDS}
        Path("plan.toml").write_text(PLAN, encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()):
            assert main(["run", "plan.toml", "-o", "run"]) == 0
        docs["run ledger.json"] = json.loads(Path("run/ledger.json").read_text(encoding="utf-8"))
        docs["mcp screen"] = mcp_server.screen(archive=FIC, baseline=BASELINE, window=WINDOW)
        docs["mcp spc"] = mcp_server.spc(archive=FIC, baseline=BASELINE, window=WINDOW)
    return docs


def contract(work: Path) -> dict[str, list[str]]:
    return {name: sorted(key_paths(doc)) for name, doc in documents(work).items()}


def test_every_document_names_its_contract(tmp_path):
    for name, doc in documents(tmp_path).items():
        assert doc["tsdive_version"] == tsdive.__version__, name
        assert doc["result_kind"] in {"evidence", "refusal", "ingest", "plan", "ledger"}, name


def test_the_key_paths_match_the_committed_contract(tmp_path):
    expected = json.loads(KEYS.read_text(encoding="utf-8"))
    assert contract(tmp_path) == expected


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        found = contract(Path(tmp))
    KEYS.parent.mkdir(exist_ok=True)
    KEYS.write_text(json.dumps(found, indent=2) + "\n", encoding="utf-8", newline="\n")
    sys.stdout.write(f"wrote {KEYS.as_posix()}: {len(found)} documents\n")

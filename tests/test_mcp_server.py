"""The MCP surface, over demo archives built into a tmp dir.

Tests that take the ``server`` fixture skip without the ``mcp`` extra;
the ones that call the wrapper functions run either way. The archives
come from scripts/make_demo_archive.py, so no test here touches data/
and none of them reaches the network.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from tsdive import mcp_server

SCRIPTS = Path(__file__).parents[1] / "scripts"

# The demo flow archive pins at full scale here (eng range 0..100), so
# the window is censored and may not serve as a baseline.
SATURATED = "2024-03-31T02:00:00Z/2024-03-31T02:30:00Z"
BASELINE = "2024-03-30T20:00:00Z/2024-03-30T23:00:00Z"
WINDOW = "2024-03-31T04:00:00Z/2024-03-31T06:00:00Z"

TOOL_NAMES = {"profile", "segment", "screen", "spc", "compare", "switchback_analyze"}


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def demo(tmp_path: Path) -> tuple[str, str]:
    """Paths to the demo flow and temperature archives, freshly written."""
    from tsdive.store.tagstore import write_tag

    builders = _load("make_demo_archive")
    flow, flow_meta = builders.build_fic101()
    temp, temp_meta = builders.build_tic101(flow)
    flow_path = write_tag(tmp_path / "fic101.parquet", flow, flow_meta, overwrite=True)
    temp_path = write_tag(tmp_path / "tic101.parquet", temp, temp_meta, overwrite=True)
    return str(flow_path), str(temp_path)


@pytest.fixture()
def server() -> Any:
    pytest.importorskip("mcp", reason="the mcp extra is not installed")
    return mcp_server.build_server()


def call(server: Any, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """The structured content of one tool call, with isError asserted false."""
    result = asyncio.run(server.call_tool(name, arguments))
    assert result.is_error is False
    assert result.structured_content is not None
    return dict(result.structured_content)


def test_server_exposes_the_six_tools(server: Any) -> None:
    tools = asyncio.run(server.list_tools())
    assert {t.name for t in tools} == TOOL_NAMES


def test_every_tool_is_annotated_read_only(server: Any) -> None:
    tools = asyncio.run(server.list_tools())
    assert all(t.annotations is not None for t in tools)
    assert all(t.annotations.read_only_hint for t in tools)
    assert all(t.annotations.destructive_hint is False for t in tools)


def test_every_tool_description_states_the_result_union(server: Any) -> None:
    tools = asyncio.run(server.list_tools())
    for tool in tools:
        assert tool.description is not None
        assert '"result_kind": "evidence"' in tool.description
        assert '"result_kind": "refusal"' in tool.description


def test_profile_over_the_whole_extent_returns_evidence(
    server: Any, demo: tuple[str, str]
) -> None:
    flow, _ = demo
    payload = call(server, "profile", {"archive": flow})
    assert payload["result_kind"] == "evidence"
    assert payload["tag"] == "demo:FIC101.PV"
    # The demo archive drops a 40-minute stretch, so the read reports a gap.
    assert payload["coverage"]["gaps"]


def test_segment_and_spc_return_evidence(server: Any, demo: tuple[str, str]) -> None:
    flow, _ = demo
    segmented = call(server, "segment", {"archive": flow, "window": WINDOW})
    assert segmented["result_kind"] == "evidence"
    assert segmented["segments"]

    charted = call(
        server, "spc", {"archive": flow, "baseline": BASELINE, "window": WINDOW}
    )
    assert charted["result_kind"] == "evidence"
    assert {r["rule"] for r in charted["rules"]} == {
        "BEYOND_3SIGMA",
        "RUN_9_SAMESIDE",
        "TREND_6",
    }


def test_screen_returns_evidence_and_honours_k(server: Any, demo: tuple[str, str]) -> None:
    flow, _ = demo
    payload = call(
        server,
        "screen",
        {"archive": flow, "baseline": BASELINE, "window": WINDOW, "k": 2.0},
    )
    assert payload["result_kind"] == "evidence"
    assert payload["k"] == 2.0
    assert payload["method"] == "MAD"


def test_compare_takes_several_archives(server: Any, demo: tuple[str, str]) -> None:
    flow, temp = demo
    payload = call(
        server,
        "compare",
        {"archives": [flow, temp], "before": BASELINE, "after": WINDOW},
    )
    assert payload["result_kind"] == "evidence"
    assert {t["tag"] for t in payload["tags"]} == {"demo:FIC101.PV", "demo:TIC101.PV"}


@pytest.fixture()
def trial(tmp_path: Path) -> tuple[list[str], str]:
    """The switchback demo trial's archives and its plan file."""
    import tsdive
    from tsdive.store.tagstore import write_tag

    spec = importlib.util.spec_from_file_location(
        "make_trial", SCRIPTS.parent / "examples" / "switchback" / "make_trial.py"
    )
    assert spec is not None and spec.loader is not None
    builders = importlib.util.module_from_spec(spec)
    sys.modules["make_trial"] = builders
    spec.loader.exec_module(builders)
    paths = [
        str(write_tag(tmp_path / f"{name}.parquet", frame, meta, overwrite=True))
        for name, (frame, meta) in builders.build().items()
    ]
    plan = tsdive.switchback_plan(*builders.TRIAL, builders.BLOCK, builders.WASHOUT, builders.SEED)
    return paths, str(plan.write_json(tmp_path / "plan.json"))


def test_switchback_analyze_returns_the_cli_fields(server: Any, trial) -> None:
    archives, plan = trial
    payload = call(
        server,
        "switchback_analyze",
        {"archives": archives, "plan": plan, "target": "TI201.PV", "covariates": ["FI200.PV"]},
    )
    assert payload["result_kind"] == "evidence"
    assert payload["tag"] == "demo:TI201.PV"
    assert payload["plan"]["verified"] is True
    assert payload["adjusted"]["covariates"] == ["demo:FI200.PV"]
    assert payload["unused"] == ["demo:TT001.PV"]


def test_an_edited_plan_comes_back_as_a_schedule_mismatch(trial) -> None:
    import json

    archives, plan = trial
    doc = json.loads(Path(plan).read_text(encoding="utf-8"))
    doc["washout_s"] = 0
    Path(plan).write_text(json.dumps(doc), encoding="utf-8")
    payload = mcp_server.switchback_analyze(archives=archives, plan=plan, target="TI201.PV")
    assert payload["result_kind"] == "refusal"
    assert payload["error_type"] == "ScheduleMismatch"


def test_a_censored_baseline_comes_back_as_a_refusal(
    server: Any, demo: tuple[str, str]
) -> None:
    flow, _ = demo
    payload = call(
        server,
        "screen",
        {"archive": flow, "baseline": SATURATED, "window": WINDOW},
    )
    assert payload["result_kind"] == "refusal"
    assert payload["error_type"] == "InsufficientQuality"
    assert "censored" in payload["cause"]


def test_a_refusal_is_not_a_transport_error(server: Any, demo: tuple[str, str]) -> None:
    flow, _ = demo
    result = asyncio.run(
        server.call_tool(
            "screen", {"archive": flow, "baseline": SATURATED, "window": WINDOW}
        )
    )
    assert result.is_error is False


def test_an_overlapping_baseline_raises_instead_of_refusing(
    demo: tuple[str, str],
) -> None:
    flow, _ = demo
    with pytest.raises(ValueError, match="baseline and window overlap"):
        mcp_server.screen(archive=flow, baseline=WINDOW, window=WINDOW)


def test_a_rejected_option_raises_with_the_parser_message(demo: tuple[str, str]) -> None:
    flow, _ = demo
    with pytest.raises(ValueError, match="--basis"):
        mcp_server.profile(archive=flow, basis="MASS_WEIGHTED")


def test_a_malformed_window_reaches_the_client_with_its_message(
    server: Any, demo: tuple[str, str]
) -> None:
    from mcp.server.mcpserver.exceptions import ToolError

    flow, _ = demo
    with pytest.raises(ToolError, match="ISO 8601 UTC"):
        asyncio.run(server.call_tool("profile", {"archive": flow, "window": "bogus"}))


def test_a_missing_archive_reaches_the_client_with_its_path(server: Any) -> None:
    from mcp.server.mcpserver.exceptions import ToolError

    with pytest.raises(ToolError, match=r"absent\.parquet"):
        asyncio.run(server.call_tool("profile", {"archive": "absent.parquet"}))


# Two days at 60 s on the switchback demo temperature, charted against a
# 12 h baseline: 849 BEYOND_3SIGMA hits in 34 runs.
TWO_DAYS = {
    "baseline": "2024-06-01/PT12H",
    "window": "2024-06-02T00:00:00Z/2024-06-03T23:59:00Z",
}


@pytest.fixture(scope="module")
def ti201(tmp_path_factory: pytest.TempPathFactory) -> str:
    import tsdive

    paths = tsdive.write_demo_data(tmp_path_factory.mktemp("demo"))
    return str(next(p for p in paths if p.name == "ti201.parquet"))


def _answer_size(server: Any, name: str, arguments: dict[str, Any]) -> tuple[int, dict]:
    """Characters of text and structured content of one call, and the structured content."""
    import json

    result = asyncio.run(server.call_tool(name, arguments))
    assert result.is_error is False
    text = sum(len(c.text) for c in result.content if hasattr(c, "text"))
    return text + len(json.dumps(result.structured_content)), dict(result.structured_content)


def test_a_two_day_spc_answer_is_bounded_and_its_counts_exact(server: Any, ti201: str) -> None:
    from tsdive.cli import _parser_spc, json_spc
    from tsdive.ui.jsonout import to_jsonable

    size, payload = _answer_size(server, "spc", {"archive": ti201, **TWO_DAYS})
    assert size < 40_000
    full = json_spc(
        _parser_spc().parse_args(
            ["--baseline", TWO_DAYS["baseline"], "--window", TWO_DAYS["window"], ti201]
        )
    )
    assert payload["n_hits"] == full["n_hits"] == 987
    for trimmed, entry in zip(payload["rules"], full["rules"], strict=True):
        assert trimmed["n"] == entry["n"] == len(entry["hits"])
        assert trimmed["hits"] == []
        assert trimmed["hits_dropped"] == entry["n"]
    assert payload["runs"] == to_jsonable(full["runs"])
    beyond = [r for r in payload["runs"] if r["rule"] == "BEYOND_3SIGMA"]
    assert (len(beyond), sum(r["n"] for r in beyond)) == (34, 849)


def test_max_events_lists_that_many_events(server: Any, ti201: str) -> None:
    _, charted = _answer_size(server, "spc", {"archive": ti201, **TWO_DAYS, "max_events": 3})
    assert [len(r["hits"]) for r in charted["rules"]] == [3, 3, 3]
    assert [r["hits_dropped"] for r in charted["rules"]] == [846, 16, 116]
    _, screened = _answer_size(server, "screen", {"archive": ti201, **TWO_DAYS})
    assert (screened["n_flagged"], screened["flagged"], screened["flagged_dropped"]) == (
        849,
        [],
        849,
    )


def test_a_negative_max_events_raises_a_tool_error(server: Any, ti201: str) -> None:
    from mcp.server.mcpserver.exceptions import ToolError

    with pytest.raises(ToolError, match="max_events must be 0 or more"):
        asyncio.run(server.call_tool("spc", {"archive": ti201, **TWO_DAYS, "max_events": -1}))


def test_a_directory_reaches_the_client_with_its_path(server: Any, tmp_path: Path) -> None:
    from mcp.server.mcpserver.exceptions import ToolError

    folder = tmp_path / "exports"
    folder.mkdir()
    with pytest.raises(ToolError, match="exports") as info:
        asyncio.run(server.call_tool("profile", {"archive": str(folder)}))
    assert "\n" not in str(info.value)


def test_build_server_without_the_extra_names_the_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # None in sys.modules makes the import raise, which is how an install
    # without the mcp extra behaves.
    for name in ("mcp", "mcp.server", "mcp.server.mcpserver", "mcp.types"):
        monkeypatch.setitem(sys.modules, name, None)
    with pytest.raises(ImportError, match="the mcp extra") as info:
        mcp_server.build_server()
    message = str(info.value)
    assert 'pip install "tsdive[mcp] @ git+https://github.com/NorthernLightx/tsdive.git"' in message
    version = mcp_server.__version__
    releases = "https://github.com/NorthernLightx/tsdive/releases/download"
    wheel = f"{releases}/v{version}/tsdive-{version}-py3-none-any.whl"
    assert f'pip install "tsdive[mcp] @ {wheel}"' in message
    assert "clone" not in message

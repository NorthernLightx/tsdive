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

TOOL_NAMES = {"profile", "segment", "screen", "spc", "compare"}


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


def test_server_exposes_the_five_tools(server: Any) -> None:
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


def test_build_server_without_the_extra_names_the_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # None in sys.modules makes the import raise, which is how an install
    # without the mcp extra behaves.
    for name in ("mcp", "mcp.server", "mcp.server.mcpserver", "mcp.types"):
        monkeypatch.setitem(sys.modules, name, None)
    with pytest.raises(ImportError, match="the mcp extra"):
        mcp_server.build_server()

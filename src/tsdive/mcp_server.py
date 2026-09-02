"""A read-only MCP server over the tsdive analyses.

Five tools wrap the functions behind ``tsdive <command> --json``, so a
caller over MCP and a caller over the CLI read the same fields. Each tool
builds the command line the CLI parser already validates, so option
names, choices and defaults are declared once, in :mod:`tsdive.cli`.

Every tool returns one JSON object carrying ``result_kind``. A typed
refusal (:class:`tsdive.errors.TSDiveError`) comes back as
``result_kind="refusal"`` with the class name and the message, and the
call itself succeeds. Any other exception propagates, so a malformed
window or an unreadable path reaches the client as a tool error carrying
the same message the CLI prints.

The server reads local parquet archives and opens no network connection.
It writes nothing, and the store exposes no mutation API. The transport
is stdio.

The ``mcp`` package is an optional extra. Importing this module never
imports it; :func:`build_server` does, and raises :class:`ImportError`
naming the extra when it is absent.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import inspect
import io
import sys
from collections.abc import Callable, Sequence
from typing import Any

from tsdive import __version__
from tsdive.cli import (
    _parser_compare,
    _parser_profile,
    _parser_screen,
    _parser_segment,
    _parser_spc,
    json_compare,
    json_profile,
    json_screen,
    json_segment,
    json_spc,
)
from tsdive.errors import TSDiveError
from tsdive.ui.jsonout import to_jsonable

MISSING_MCP = (
    "the tsdive MCP server needs the mcp package (>=2.1); install tsdive with "
    "the mcp extra, for example pip install -e '.[mcp]' from a clone"
)

RESULT_CONTRACT = """\
Returns one JSON object, discriminated on result_kind:

  {"result_kind": "evidence", ...}
      the analysis, carrying the fields `tsdive <command> --json` prints.

  {"result_kind": "refusal", "error_type": "<class>", "cause": "<message>"}
      a check that has no answer on this data. error_type names a
      tsdive.errors class: SchemaError, InsufficientQuality,
      IncomparableSamplingError, NonMonotonicIndex, UnresolvedUnitError,
      IncomparableUnitsError, RegimeTooSparse, PopulationTooSparse,
      GroupLeakage, MspcAlignmentError or NarratorUnavailable.

A refusal comes back with isError false, so the call succeeds. A
malformed window, a rejected option or an unreadable path raises a tool
error instead, and its message names what to fix."""

INSTRUCTIONS = """\
tsdive profiles process time series held in single-tag parquet archives
and reports what the store did to a window. Every tool is read-only and
takes local archive paths plus ISO 8601 UTC windows written START/END
like 2024-03-01T00:00:00Z/2024-03-01T01:00:00Z, START/PT1H, PT1H/END,
or a date for one whole UTC day.

Read result_kind on every answer. A refusal names the check that has no
answer on this data and why. It is the result, so report it instead of
retrying with different options."""


def _sdk() -> tuple[Any, Any, Any]:
    """``MCPServer``, ``ToolAnnotations`` and ``ToolError`` from the mcp package.

    Raises:
        ImportError: the mcp extra is not installed.
    """
    try:
        from mcp.server.mcpserver import MCPServer
        from mcp.server.mcpserver.exceptions import ToolError
        from mcp.types import ToolAnnotations
    except ImportError as e:
        raise ImportError(MISSING_MCP) from e
    return MCPServer, ToolAnnotations, ToolError


def _flag(name: str, value: object | None) -> list[str]:
    """``[name, str(value)]`` when ``value`` is set, an empty list otherwise.

    An unset option is left off the command line, so the default declared
    on the CLI parser applies and is never restated here.
    """
    return [] if value is None else [name, str(value)]


def _switch(name: str, on: bool) -> list[str]:
    return [name] if on else []


def _namespace(parser: argparse.ArgumentParser, argv: list[str]) -> argparse.Namespace:
    """Parsed arguments, with argparse's exit turned into a ValueError.

    argparse writes usage to stderr and raises SystemExit on a rejected
    value. SystemExit out of a tool call would end the server process, so
    the message is captured and re-raised.
    """
    captured = io.StringIO()
    try:
        with contextlib.redirect_stderr(captured):
            return parser.parse_args(argv)
    except SystemExit as e:
        lines = captured.getvalue().strip().splitlines()
        raise ValueError(lines[-1] if lines else f"{parser.prog}: invalid arguments") from e


def _answer(
    to_json: Callable[[argparse.Namespace], dict[str, object]],
    parser: argparse.ArgumentParser,
    argv: list[str],
) -> dict[str, Any]:
    """The command's JSON payload tagged ``evidence``, or its typed refusal.

    Argument parsing sits outside the caught block: a rejected option is
    a caller error, not a statement about the data.
    """
    args = _namespace(parser, argv)
    try:
        return {"result_kind": "evidence", **to_jsonable(to_json(args))}
    except TSDiveError as e:
        return {
            "result_kind": "refusal",
            "error_type": type(e).__name__,
            "cause": str(e),
        }


def profile(
    archive: str,
    window: str | None = None,
    basis: str | None = None,
    stepped: bool = False,
    tz: list[str] | None = None,
    flatline: bool = False,
) -> dict[str, Any]:
    """Report the data physics and statistics of one window.

    Covers coverage and gap classes, clipping against the engineering
    range, the timestamp audit, quality counts and unit resolution.

    Args:
        archive: single-tag parquet archive carrying tsdive.meta.
        window: START/END, START/PT5H, PT5H/END or a date for one day, in
            ISO 8601 UTC. Omitted, the whole archive extent.
        basis: TIME_WEIGHTED or EVENT_WEIGHTED. Omitted, TIME_WEIGHTED.
        stepped: stepped interpolation between samples.
        tz: IANA zone names to audit for DST transitions inside the window.
        flatline: run the flatline signals against prior equal-size windows.
    """
    argv = [
        *_flag("--window", window),
        *_flag("--basis", basis),
        *_switch("--stepped", stepped),
        *_switch("--flatline", flatline),
    ]
    for name in tz or ():
        argv += ["--tz", name]
    return _answer(json_profile, _parser_profile(), [*argv, "--", archive])


def segment(
    archive: str,
    window: str | None = None,
    penalty: float | None = None,
    min_size: int | None = None,
    basis: str | None = None,
    stepped: bool = False,
) -> dict[str, Any]:
    """Cut a window into the piecewise-constant segments the samples show.

    Args:
        archive: single-tag parquet archive carrying tsdive.meta.
        window: START/END, START/PT5H, PT5H/END or a date for one day, in
            ISO 8601 UTC. Omitted, the whole archive extent.
        penalty: cost one more breakpoint has to buy. Omitted, a multiple
            of log(n) on the scaled series.
        min_size: samples a segment must hold, at least.
        basis: TIME_WEIGHTED or EVENT_WEIGHTED. Omitted, TIME_WEIGHTED.
        stepped: stepped interpolation between samples.
    """
    argv = [
        *_flag("--window", window),
        *_flag("--penalty", penalty),
        *_flag("--min-size", min_size),
        *_flag("--basis", basis),
        *_switch("--stepped", stepped),
    ]
    return _answer(json_segment, _parser_segment(), [*argv, "--", archive])


def screen(
    archive: str,
    baseline: str,
    window: str,
    method: str | None = None,
    k: float | None = None,
    mode: str | None = None,
    basis: str | None = None,
    stepped: bool = False,
) -> dict[str, Any]:
    """Flag samples outside a baseline built from a separate window.

    A censored baseline returns an InsufficientQuality refusal. A
    baseline overlapping the screened window raises a tool error.

    Args:
        archive: single-tag parquet archive carrying tsdive.meta.
        baseline: START/END, START/PT5H, PT5H/END or a date for one day, in
            ISO 8601 UTC for the history.
        window: START/END, START/PT5H, PT5H/END or a date for one day, in
            ISO 8601 UTC to screen.
        method: mad or moving-range. Omitted, mad.
        k: flag beyond k*scale from center. Omitted, 3.0.
        mode: MODE archive path; one baseline per regime instead of one
            for the window.
        basis: TIME_WEIGHTED or EVENT_WEIGHTED. Omitted, TIME_WEIGHTED.
        stepped: stepped interpolation between samples.
    """
    argv = [
        "--baseline",
        baseline,
        "--window",
        window,
        *_flag("--method", method),
        *_flag("--k", k),
        *_flag("--mode", mode),
        *_flag("--basis", basis),
        *_switch("--stepped", stepped),
    ]
    return _answer(json_screen, _parser_screen(), [*argv, "--", archive])


def spc(
    archive: str,
    baseline: str,
    window: str,
    basis: str | None = None,
    stepped: bool = False,
) -> dict[str, Any]:
    """Chart a window against individuals limits fitted on a baseline.

    Reports the limits and the hits of each rule: BEYOND_3SIGMA,
    RUN_9_SAMESIDE and TREND_6.

    Args:
        archive: single-tag parquet archive carrying tsdive.meta.
        baseline: START/END, START/PT5H, PT5H/END or a date for one day, in
            ISO 8601 UTC the limits are computed from.
        window: START/END, START/PT5H, PT5H/END or a date for one day, in
            ISO 8601 UTC to chart.
        basis: TIME_WEIGHTED or EVENT_WEIGHTED. Omitted, TIME_WEIGHTED.
        stepped: stepped interpolation between samples.
    """
    argv = [
        "--baseline",
        baseline,
        "--window",
        window,
        *_flag("--basis", basis),
        *_switch("--stepped", stepped),
    ]
    return _answer(json_spc, _parser_spc(), [*argv, "--", archive])


def compare(
    archives: list[str],
    before: str,
    after: str,
    top: int | None = None,
    rate_s: int | None = None,
) -> dict[str, Any]:
    """Report what changed between two periods of one unit.

    Covers per-tag level, spread and flagged share, the pairwise
    correlation table and the joint MSPC structure. A tag that has no
    answer carries its own reason in the row rather than ending the call.

    Args:
        archives: every single-tag parquet archive of one unit.
        before: START/END, START/PT5H, PT5H/END or a date for one day, in
            ISO 8601 UTC for the earlier period.
        after: START/END, START/PT5H, PT5H/END or a date for one day, in
            ISO 8601 UTC for the later period.
        top: rows each table reports before it counts the rest.
        rate_s: grid rate in seconds. Omitted, the rate every archive declares.
    """
    argv = [
        "--before",
        before,
        "--after",
        after,
        *_flag("--top", top),
        *_flag("--rate-s", rate_s),
    ]
    return _answer(json_compare, _parser_compare(), [*argv, "--", *archives])


TOOLS: tuple[Callable[..., dict[str, Any]], ...] = (profile, segment, screen, spc, compare)


def _description(fn: Callable[..., Any]) -> str:
    """The tool's docstring followed by the result contract."""
    return f"{inspect.getdoc(fn) or ''}\n\n{RESULT_CONTRACT}"


def _carrying_the_message(
    fn: Callable[..., dict[str, Any]], tool_error: Any
) -> Callable[..., dict[str, Any]]:
    """``fn`` with caller-input errors re-raised as the SDK's ToolError.

    The SDK replaces any other exception with a generic crash message, so
    the reason a window, an option or a path was rejected would stay on
    the server. This is the pair the CLI turns into ``error: <message>``.
    functools.wraps keeps the signature the tool schema is built from.
    """

    @functools.wraps(fn)
    def call(*args: Any, **kwargs: Any) -> dict[str, Any]:
        try:
            return fn(*args, **kwargs)
        except (ValueError, FileNotFoundError) as e:
            raise tool_error(str(e)) from e

    return call


def build_server() -> Any:
    """An MCPServer carrying the five tsdive tools, all marked read-only.

    Raises:
        ImportError: the mcp extra is not installed.
    """
    mcp_server, tool_annotations, tool_error = _sdk()
    server = mcp_server(
        name="tsdive",
        version=__version__,
        instructions=INSTRUCTIONS,
    )
    read_only = tool_annotations(
        read_only_hint=True,
        destructive_hint=False,
        idempotent_hint=True,
        open_world_hint=False,
    )
    for fn in TOOLS:
        server.add_tool(
            _carrying_the_message(fn, tool_error),
            description=_description(fn),
            annotations=read_only,
        )
    return server


def main(argv: Sequence[str] | None = None) -> int:
    """Serve the tsdive tools over the MCP stdio transport."""
    parser = argparse.ArgumentParser(
        prog="tsdive-mcp",
        description="serve the tsdive analyses to an MCP client over stdio",
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.parse_args(argv)
    try:
        server = build_server()
    except ImportError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    server.run(transport="stdio")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""``tsdive`` entry points.

The working loop a new archive walks:

    ingest -> profile -> segment -> screen -> spc -> mspc

Every command after ``ingest`` takes the same window syntax
(``START/END``, ``START/PT5H``, ``PT5H/END``, or a date for one UTC day)
and refuses the same way: ``[ErrorName] message`` on stderr and exit
status 3 (``REFUSED``), with the refusal object on stdout under
``--json``. A usage error or invalid input exits 2 (``USAGE``). On
``profile`` an omitted window means the archive's whole extent, which
the report states like any other window.

Argument parsing, printing and exit codes live here; the profiling flow
itself is :func:`tsdive.api.profile` and the other analyses are the
functions in :mod:`tsdive.analyses`, so the library and the CLI can
never disagree about what a result is.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import pandas as pd
from pyarrow import parquet

from tsdive import __version__, analyses
from tsdive._naming import cli_names
from tsdive.analyses import (
    CompareAnalysis,
    MspcAnalysis,
    ScreenAnalysis,
    SegmentAnalysis,
    SpcAnalysis,
)
from tsdive.analyses_render import render_lines
from tsdive.api import (
    Profile,
    ingest,
    ingest_long,
    ingest_wide,
    init_long_meta,
    init_meta,
    init_tag_meta,
    parse_window,
    profile,
    read_meta_json,
    switchback_analyze,
    switchback_plan,
)
from tsdive.changepoints import DEFAULT_PENALTY_MULTIPLIER
from tsdive.demo import DEFAULT_DIR as DEMO_DEFAULT_DIR
from tsdive.demo import write_demo_data
from tsdive.errors import TSDiveError
from tsdive.features.window_features import compute_stats
from tsdive.report import (
    SEP,
    continued,
    fmt_span,
    label_line,
    plural,
    render_window_report,
)
from tsdive.store.sampling_contract import (
    CalculationBasis,
    RetrievalMode,
    SamplingContract,
)
from tsdive.store.tagstore import (
    SingleFileStore,
    archive_extent,
    meta_from_parquet,
)
from tsdive.switchback.archive import SwitchbackAnalysis
from tsdive.switchback.plan import SwitchbackPlan
from tsdive.switchback.render import plan_lines
from tsdive.ui.jsonout import error_fields, error_text, refusal_json, to_jsonable
from tsdive.ui.term import colour_enabled, colourise, red

MAIN_DOC = """tsdive - data-quality profiling and monitoring for process time series

commands, in the order an archive walks them:
  demo [DIR]                              write the demo archives into DIR
                                          (default tsdive-demo/) to try the rest
  ingest <csv|parquet> --out ARCHIVE --meta META.json
                                          build an archive from an export;
                                          --init-meta META.json writes the
                                          metadata template first
  ingest <csv|parquet> --wide --out DIR --meta-dir DIR
                                          one archive per column of a wide export
  ingest <csv|parquet> --tag-col COL --out DIR --meta-dir DIR
                                          one archive per tag of a long export,
                                          the tag of each row in column COL
  profile <parquet> [--window START/END]  data physics and statistics for a window
  segment <parquet> [--window START/END]  regimes read off the samples themselves,
                                          for a tag with no MODE tag to key them
  screen <parquet> --baseline START/END --window START/END [--mode PARQUET]
                                          flag samples outside a MAD baseline, or a
                                          per-regime one with --mode
  spc <parquet> --baseline START/END --window START/END
                                          individuals chart; every rule reports its
                                          own hits, zero included
  mspc <parquet...> --baseline START/END --window START/END
                                          PCA T2 and SPE over aligned tags
  compare <parquet...> --before START/END --after START/END
                                          what changed between two periods: tags,
                                          pairs of tags, and their joint structure
  switchback plan --window START/END --block PT1H --washout PT10M --seed N
                  -o PLAN.json [--history PARQUET --history-window START/END]
                                          a balanced random schedule of settings A
                                          and B, with its power over a history
  switchback analyze <parquet...> --plan PLAN.json --target TAG [--covariate TAG]
                                          the difference between settings A and B
                                          under the plan, by randomization inference
  run <plan.toml> [-o DIR] [--strict]     walk one plan over several archives into
                                          one evidence ledger
  report-html <parquet...> [--window START/END] [-o FILE]
                                          render a static HTML evidence snapshot

options, on every command:
  --json                                  one JSON object on stdout instead of
                                          text (the analysis commands)
  --no-color                              plain text; NO_COLOR does the same
  --version                               print the tsdive version

windows, wherever START/END appears above:
  START/END, START/PT5H, PT5H/END, or a date for one whole UTC day

exit status:
  0                                       the answer was printed
  2                                       usage error or invalid input
  3                                       refusal: a typed tsdive error, printed
                                          as [ErrorName] on stderr and, under
                                          --json, as one object on stdout
"""

# Exit status of a command. ``run`` keeps its own rule, stated in its help.
OK = 0
USAGE = 2
REFUSED = 3


def _positive_int(text: str) -> int:
    """An ``argparse`` type for counts a computation can actually step by.

    Caught at parse time because the alternative is a divide by zero or a
    backwards grid deep inside the numerics, which reaches the caller as a
    traceback or as a coverage number that misstates why it refused.
    """
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not an integer") from None
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer, not {value}")
    return value


def _positive_float(text: str) -> float:
    """An ``argparse`` type for a strictly positive multiplier.

    A non-positive ``k`` puts the lower limit above the upper one, so the
    printed interval reads backwards and every sample falls outside it.
    """
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from None
    if not value > 0.0:
        raise argparse.ArgumentTypeError(f"must be greater than 0, not {value}")
    return value


def _non_negative_int(text: str) -> int:
    """An ``argparse`` type for a seed: an integer of 0 or more."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not an integer") from None
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be 0 or more, not {value}")
    return value


def _unit_fraction(text: str) -> float:
    """An ``argparse`` type for a share in (0, 1].

    Zero keeps nothing and above one is not a share; both are questions
    the statistic cannot answer.
    """
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from None
    if not 0.0 < value <= 1.0:
        raise argparse.ArgumentTypeError(f"must be in (0, 1], not {value}")
    return value


def _combined_extent(paths: Sequence[str]) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Span covering every archive that can be read, for an omitted --window.

    One window is applied to every archive, so the default has to cover
    all of them. Archives that cannot be read contribute no extent; they
    surface as refusals in the report itself.
    """
    extents = []
    for p in paths:
        try:
            extents.append(archive_extent(p))
        except (TSDiveError, OSError):
            continue
    if not extents:
        raise ValueError(
            "no readable archive to take a window from; pass --window explicitly"
        )
    return min(e[0] for e in extents), max(e[1] for e in extents)


StepRunner = Callable[[argparse.Namespace], list[str]]
StepJson = Callable[[argparse.Namespace], dict[str, object]]


def _no_color(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "no_color", False))


def _add_output_flags(
    parser: argparse.ArgumentParser, *, json_flag: bool = True
) -> argparse.ArgumentParser:
    """Add the flags that decide how a command's answer is written."""
    if json_flag:
        parser.add_argument(
            "--json",
            action="store_true",
            help="print one JSON object and nothing else",
        )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="plain text, no ANSI colour (NO_COLOR does the same)",
    )
    return parser


def _print_refusal(prefix: str, message: str, args: argparse.Namespace) -> None:
    on = colour_enabled(sys.stderr, no_color=_no_color(args))
    print(f"{red(prefix, on)} {message}", file=sys.stderr)


def _refused(error: TSDiveError, args: argparse.Namespace) -> int:
    """Report a typed refusal and return its exit status, ``REFUSED``.

    The ``[ErrorName] message`` line goes to stderr. Under ``--json`` the
    refusal object the MCP server returns is also printed on stdout, so a
    script that parses stdout reads a refusal as JSON too.
    """
    _print_refusal(f"[{type(error).__name__}]", str(error), args)
    if getattr(args, "json", False):
        print(json.dumps(refusal_json(error), indent=2))
    return REFUSED


def _print_lines(lines: Sequence[str], args: argparse.Namespace) -> None:
    """Write a command's answer to stdout, coloured only for a terminal."""
    on = colour_enabled(sys.stdout, no_color=_no_color(args))
    print("\n".join(colourise(lines, enabled=on)))


def _report_and_exit(
    fn: StepRunner, args: argparse.Namespace, *, to_json: StepJson | None = None
) -> int:
    """Print what a step returned, or map its refusal to an exit code.

    Steps raise rather than print, so a caller that wants the lines
    (``tsdive run``) and a caller that wants a process exit code share
    one body and can never disagree about what a refusal is. Colour is
    applied here and nowhere else, so every renderer stays plain text.
    Messages raised inside name CLI flags (``--rate-s``), not keywords.

    Returns ``OK``, ``REFUSED`` for a ``TSDiveError``, and ``USAGE`` for a
    malformed argument, an existing output or a path the OS cannot read.
    """
    try:
        with cli_names():
            if getattr(args, "json", False) and to_json is not None:
                print(json.dumps(to_jsonable(to_json(args)), indent=2))
                return OK
            _print_lines(fn(args), args)
        return OK
    except TSDiveError as e:
        return _refused(e, args)
    except (ValueError, OSError) as e:
        _print_refusal("error:", error_text(e), args)
        return USAGE


def _parser_screen() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tsdive screen")
    parser.add_argument("parquet", help="single-tag parquet archive to screen")
    parser.add_argument(
        "--baseline", required=True, help="ISO 8601 START/END in UTC for the history"
    )
    parser.add_argument(
        "--window", required=True, help="ISO 8601 START/END in UTC to screen"
    )
    parser.add_argument(
        "--method",
        default="mad",
        choices=["mad", "moving-range"],
        help="scale of the baseline: MAD, or the mean moving range; ignored with --mode",
    )
    parser.add_argument(
        "--k", type=_positive_float, default=3.0, help="flag beyond k*scale from center"
    )
    parser.add_argument(
        "--mode",
        default=None,
        metavar="PARQUET",
        help="MODE archive; compute one baseline per regime instead of one for the window",
    )
    parser.add_argument(
        "--basis",
        default="TIME_WEIGHTED",
        choices=[b.value for b in CalculationBasis],
        help="calculation basis declared on the read",
    )
    parser.add_argument(
        "--stepped", action="store_true", help="stepped interpolation between samples"
    )
    return _add_output_flags(parser)


def _screen_run(args: argparse.Namespace) -> ScreenAnalysis:
    return analyses.screen(
        args.parquet,
        args.baseline,
        args.window,
        method=args.method,
        k=args.k,
        mode=args.mode or None,
        basis=args.basis,
        stepped=args.stepped,
    )


def run_screen(args: argparse.Namespace) -> list[str]:
    """Screen a window against a baseline: MAD, or regime-keyed with --mode."""
    return render_lines(_screen_run(args).render())


def json_screen(args: argparse.Namespace) -> dict[str, object]:
    return _screen_run(args).to_dict()


def cmd_screen(argv: Sequence[str] | None = None) -> int:
    """Screen a window against a baseline: MAD, or regime-keyed with --mode."""
    return _report_and_exit(
        run_screen, _parser_screen().parse_args(argv), to_json=json_screen
    )


def _parser_segment() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tsdive segment")
    parser.add_argument("parquet", help="single-tag parquet archive to segment")
    parser.add_argument(
        "--window",
        default=None,
        help="ISO 8601 START/END in UTC; omitted, the whole archive extent",
    )
    parser.add_argument(
        "--penalty",
        type=_positive_float,
        default=None,
        help="cost one more breakpoint has to buy; omitted, "
        f"{DEFAULT_PENALTY_MULTIPLIER:g}*log(n) on the scaled series",
    )
    parser.add_argument(
        "--min-size",
        type=_positive_int,
        default=10,
        help="samples a segment must hold, at least",
    )
    parser.add_argument(
        "--basis",
        default="TIME_WEIGHTED",
        choices=[b.value for b in CalculationBasis],
        help="calculation basis declared on the read",
    )
    parser.add_argument(
        "--stepped", action="store_true", help="stepped interpolation between samples"
    )
    parser.add_argument(
        "--mode-out",
        default=None,
        metavar="FILE",
        help="write the segments as a MODE archive, one label per sample, for screen --mode",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing archive at --mode-out",
    )
    return _add_output_flags(parser)


def _segment_run(args: argparse.Namespace) -> SegmentAnalysis:
    return analyses.segment(
        args.parquet,
        args.window,
        penalty=args.penalty,
        min_size=args.min_size,
        basis=args.basis,
        stepped=args.stepped,
    )


def _segment_mode_out(analysis: SegmentAnalysis, args: argparse.Namespace) -> str | None:
    """Write ``--mode-out`` when asked and return the path written, as posix."""
    mode_out = getattr(args, "mode_out", None)
    if mode_out is None:
        return None
    return analysis.write_mode_archive(mode_out, overwrite=args.overwrite).as_posix()


def run_segment(args: argparse.Namespace) -> list[str]:
    """Segment a window into regimes the samples themselves show."""
    analysis = _segment_run(args)
    lines = render_lines(analysis.render())
    written = _segment_mode_out(analysis, args)
    if written is not None:
        lines += ["", label_line("wrote", written)]
    return lines


def json_segment(args: argparse.Namespace) -> dict[str, object]:
    analysis = _segment_run(args)
    doc = analysis.to_dict()
    written = _segment_mode_out(analysis, args)
    if written is not None:
        doc["mode_out"] = written
    return doc


def cmd_segment(argv: Sequence[str] | None = None) -> int:
    """Segment a window into regimes the samples themselves show."""
    return _report_and_exit(
        run_segment, _parser_segment().parse_args(argv), to_json=json_segment
    )


def _parser_spc() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tsdive spc")
    parser.add_argument("parquet", help="single-tag parquet archive to chart")
    parser.add_argument(
        "--baseline",
        required=True,
        help="ISO 8601 START/END in UTC the limits are computed from",
    )
    parser.add_argument(
        "--window", required=True, help="ISO 8601 START/END in UTC to chart"
    )
    parser.add_argument(
        "--basis",
        default="TIME_WEIGHTED",
        choices=[b.value for b in CalculationBasis],
        help="calculation basis declared on the read",
    )
    parser.add_argument(
        "--stepped", action="store_true", help="stepped interpolation between samples"
    )
    return _add_output_flags(parser)


def _spc_run(args: argparse.Namespace) -> SpcAnalysis:
    return analyses.spc(
        args.parquet, args.baseline, args.window, basis=args.basis, stepped=args.stepped
    )


def run_spc(args: argparse.Namespace) -> list[str]:
    """Chart a window against baseline control limits."""
    return render_lines(_spc_run(args).render())


def json_spc(args: argparse.Namespace) -> dict[str, object]:
    return _spc_run(args).to_dict()


def cmd_spc(argv: Sequence[str] | None = None) -> int:
    """Chart a window against baseline control limits."""
    return _report_and_exit(run_spc, _parser_spc().parse_args(argv), to_json=json_spc)


def _parser_mspc() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tsdive mspc")
    parser.add_argument("parquet", nargs="+", help="two or more single-tag archives")
    parser.add_argument(
        "--baseline",
        required=True,
        help="ISO 8601 START/END in UTC the model is fitted on",
    )
    parser.add_argument(
        "--window", required=True, help="ISO 8601 START/END in UTC to monitor"
    )
    parser.add_argument(
        "--rate-s",
        type=_positive_int,
        default=None,
        help="grid rate in seconds; omitted, the rate every archive declares",
    )
    parser.add_argument(
        "--variance",
        type=_unit_fraction,
        default=0.95,
        help="cumulative variance kept",
    )
    parser.add_argument(
        "--quantile",
        type=_unit_fraction,
        default=0.99,
        help="empirical quantile for the limits",
    )
    parser.add_argument(
        "--min-coverage",
        type=_unit_fraction,
        default=0.95,
        help="grid cells that must be filled before alignment is accepted",
    )
    return _add_output_flags(parser)


def _mspc_run(args: argparse.Namespace) -> MspcAnalysis:
    return analyses.mspc(
        args.parquet,
        args.baseline,
        args.window,
        rate_s=args.rate_s,
        variance=args.variance,
        quantile=args.quantile,
        min_coverage=args.min_coverage,
    )


def run_mspc(args: argparse.Namespace) -> list[str]:
    """Detect multivariate departures with PCA T2 and SPE."""
    return render_lines(_mspc_run(args).render())


def json_mspc(args: argparse.Namespace) -> dict[str, object]:
    return _mspc_run(args).to_dict()


def cmd_mspc(argv: Sequence[str] | None = None) -> int:
    """Detect multivariate departures with PCA T2 and SPE."""
    return _report_and_exit(run_mspc, _parser_mspc().parse_args(argv), to_json=json_mspc)


def _parser_compare() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tsdive compare")
    parser.add_argument("parquet", nargs="+", help="every single-tag archive of one unit")
    parser.add_argument(
        "--before",
        required=True,
        help="ISO 8601 START/END in UTC for the earlier period",
    )
    parser.add_argument(
        "--after", required=True, help="ISO 8601 START/END in UTC for the later period"
    )
    parser.add_argument(
        "--top",
        type=_positive_int,
        default=10,
        help="rows each table prints before it counts the rest",
    )
    parser.add_argument(
        "--rate-s",
        type=_positive_int,
        default=None,
        help="grid rate in seconds; omitted, the rate every archive declares",
    )
    return _add_output_flags(parser)


def _compared(args: argparse.Namespace) -> CompareAnalysis:
    return analyses.compare(
        args.parquet, args.before, args.after, top=args.top, rate_s=args.rate_s
    )


def run_compare(args: argparse.Namespace) -> list[str]:
    """Report what changed between two periods of one unit."""
    return render_lines(_compared(args).render())


def json_compare(args: argparse.Namespace) -> dict[str, object]:
    return _compared(args).to_dict()


def cmd_compare(argv: Sequence[str] | None = None) -> int:
    """Report what changed between two periods of one unit."""
    return _report_and_exit(
        run_compare, _parser_compare().parse_args(argv), to_json=json_compare
    )


SWITCHBACK_DOC = """tsdive switchback - plan and analyze a randomized trial of settings A and B

  switchback plan --window START/END --block PT1H --washout PT10M --seed N
                  -o PLAN.json [--history PARQUET --history-window START/END]
  switchback analyze <parquet...> --plan PLAN.json --target TAG [--covariate TAG]

--help after plan or analyze lists every option.
"""


def _parser_switchback_plan() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tsdive switchback plan")
    parser.add_argument(
        "--window",
        required=True,
        help="ISO 8601 START/END the schedule runs over; bounds with a UTC offset "
        "are converted to UTC",
    )
    parser.add_argument(
        "--block", required=True, help="block length as an ISO 8601 duration, like PT1H"
    )
    parser.add_argument(
        "--washout",
        required=True,
        help="time left out at the start of every block, like PT10M; PT0S for none",
    )
    parser.add_argument(
        "--seed", required=True, type=_non_negative_int, help="seed of the assignment"
    )
    parser.add_argument(
        "-o",
        "--out",
        required=True,
        metavar="PLAN.json",
        help="file the plan is written to; an existing file is refused",
    )
    parser.add_argument(
        "--history",
        default=None,
        metavar="PARQUET",
        help="archive of the target where nothing was switched, for the power readout",
    )
    parser.add_argument(
        "--history-window",
        default=None,
        help="ISO 8601 START/END of the history, at least as long as the schedule; "
        "bounds with a UTC offset are converted to UTC",
    )
    return _add_output_flags(parser)


def _planned(args: argparse.Namespace) -> tuple[SwitchbackPlan, Path]:
    """The plan the arguments describe, written to ``--out``."""
    start, end = parse_window(args.window)
    plan = switchback_plan(
        start,
        end,
        args.block,
        args.washout,
        args.seed,
        history=args.history,
        history_window=args.history_window,
    )
    return plan, plan.write_json(args.out)


def run_switchback_plan(args: argparse.Namespace) -> list[str]:
    """Write a balanced random schedule and report it."""
    plan, out = _planned(args)
    return plan_lines(plan, wrote=out.as_posix())


def json_switchback_plan(args: argparse.Namespace) -> dict[str, object]:
    plan, _ = _planned(args)
    return plan.to_dict()


def _parser_switchback_analyze() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tsdive switchback analyze")
    parser.add_argument(
        "parquet", nargs="+", help="archives holding the target and every covariate"
    )
    parser.add_argument(
        "--plan",
        required=True,
        metavar="PLAN.json",
        help="plan file written by tsdive switchback plan",
    )
    parser.add_argument(
        "--target", required=True, help="tag to analyse, as source:point or a point id"
    )
    parser.add_argument(
        "--covariate",
        action="append",
        default=[],
        metavar="TAG",
        help="tag for the adjusted estimate, declared before the analysis; repeat for more",
    )
    return _add_output_flags(parser)


def _switchback_analyzed(args: argparse.Namespace) -> SwitchbackAnalysis:
    return switchback_analyze(
        args.parquet, args.plan, target=args.target, covariates=args.covariate
    )


def run_switchback_analyze(args: argparse.Namespace) -> list[str]:
    """Report the difference between settings A and B under a plan."""
    return render_lines(_switchback_analyzed(args).render())


def json_switchback_analyze(args: argparse.Namespace) -> dict[str, object]:
    return _switchback_analyzed(args).to_dict()


def _switchback_analyze_status(args: argparse.Namespace) -> int:
    """Print the analysis; ``REFUSED`` when its raw estimate is refused.

    The report is printed either way, so the refusal reason stays on the
    page. A refused adjusted estimate with a raw one standing keeps
    ``OK``: the row says why the adjustment has no answer.
    """
    done: list[SwitchbackAnalysis] = []

    def analysed(a: argparse.Namespace) -> SwitchbackAnalysis:
        done.append(_switchback_analyzed(a))
        return done[-1]

    status = _report_and_exit(
        lambda a: render_lines(analysed(a).render()),
        args,
        to_json=lambda a: analysed(a).to_dict(),
    )
    if status == OK and done and done[-1].direct.refused:
        return REFUSED
    return status


def cmd_switchback(argv: Sequence[str] | None = None) -> int:
    """Dispatch ``tsdive switchback plan`` and ``tsdive switchback analyze``."""
    rest = list(sys.argv[2:] if argv is None else argv)
    # main() moves a leading --json or --no-color behind "switchback"; they
    # belong to the subcommand's parser, so they move behind it again.
    lead = 0
    while lead < len(rest) and rest[lead] in {"--json", "--no-color"}:
        lead += 1
    flags, rest = rest[:lead], rest[lead:]
    if not rest or rest[0] in {"-h", "--help"}:
        print(SWITCHBACK_DOC.strip(), file=sys.stderr)
        return 0 if rest else 2
    sub, tail = rest[0], [*rest[1:], *flags]
    if sub == "plan":
        return _report_and_exit(
            run_switchback_plan,
            _parser_switchback_plan().parse_args(tail),
            to_json=json_switchback_plan,
        )
    if sub == "analyze":
        return _switchback_analyze_status(_parser_switchback_analyze().parse_args(tail))
    print(f"unknown switchback command {sub!r}; try 'plan' or 'analyze'", file=sys.stderr)
    return 2


def _parser_profile() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tsdive profile")
    parser.add_argument("parquet", help="path to a single-tag parquet archive with tsdive.meta")
    parser.add_argument(
        "--window",
        default=None,
        help="ISO 8601 START/END in UTC, for example "
        "2024-03-01T00:00:00Z/2024-03-01T01:00:00Z; omitted, the whole archive extent "
        "is profiled",
    )
    parser.add_argument(
        "--basis",
        default="TIME_WEIGHTED",
        choices=[b.value for b in CalculationBasis],
        help="calculation basis declared on the read",
    )
    parser.add_argument(
        "--stepped", action="store_true", help="stepped interpolation between samples"
    )
    parser.add_argument(
        "--tz",
        action="append",
        default=[],
        dest="tz_names",
        help="IANA tz to audit for DST transitions inside the window (repeatable)",
    )
    parser.add_argument(
        "--flatline",
        action="store_true",
        help="run flatline signals using prior equal-size windows as references",
    )
    return _add_output_flags(parser)


def _profiled(args: argparse.Namespace) -> Profile:
    return profile(
        args.parquet,
        args.window,
        tz=args.tz_names,
        basis=args.basis,
        stepped=args.stepped,
        flatline=args.flatline,
    )


def run_profile(args: argparse.Namespace) -> list[str]:
    """Report the data physics and statistics of one window."""
    return render_lines(_profiled(args).render())


def json_profile(args: argparse.Namespace) -> dict[str, object]:
    return _profiled(args).to_dict()


def cmd_profile(argv: Sequence[str] | None = None) -> int:
    """Report the data physics and statistics of one window."""
    return _report_and_exit(
        run_profile, _parser_profile().parse_args(argv), to_json=json_profile
    )


def _split_tags(text: str | None) -> list[str] | None:
    return None if text is None else [t.strip() for t in text.split(",") if t.strip()]


def _csv_options(args: argparse.Namespace) -> dict[str, str | None]:
    """The CSV reading options of an ingest command line."""
    return {"sep": args.sep, "decimal": args.decimal, "encoding": args.encoding}


def _warn_assumed_quality(assume_quality: str | None) -> None:
    if assume_quality:
        print(
            f"warning: quality assumed {assume_quality.strip().upper()} for every "
            "sample; the archive records this and every profile of it says so",
            file=sys.stderr,
        )


def _parser_ingest() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tsdive ingest")
    parser.add_argument(
        "source",
        help="CSV or parquet export: one tag, one column per tag with --wide, or one row "
        "per tag and timestamp with --tag-col",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="archive to create; with --wide or --tag-col, directory for one archive per tag",
    )
    parser.add_argument(
        "--meta",
        default=None,
        help="JSON file of tag metadata (the tsdive.meta object); a key it does not "
        "define raises SchemaError",
    )
    parser.add_argument(
        "--timestamp-col", default="timestamp", help="timestamp column of the export"
    )
    parser.add_argument(
        "--timestamp-format",
        default=None,
        metavar="FORMAT",
        help="strptime format every timestamp must match, like '%%d/%%m/%%Y %%H:%%M:%%S'",
    )
    parser.add_argument(
        "--dayfirst",
        action="store_true",
        help="read numeric dates day first, so 01/02/2026 is 1 February; without it "
        "or --timestamp-format a date that reads both ways raises SchemaError",
    )
    parser.add_argument(
        "--sep",
        default=None,
        metavar="CHAR",
        help="column separator of a CSV export, like ';' (default ,)",
    )
    parser.add_argument(
        "--decimal",
        default=None,
        metavar="CHAR",
        help="decimal mark of a CSV export, like ',' (default .)",
    )
    parser.add_argument(
        "--encoding",
        default=None,
        metavar="NAME",
        help="text encoding of a CSV export, like cp1252 (default utf-8)",
    )
    parser.add_argument(
        "--value-col",
        default=None,
        help="value column of a single-tag or --tag-col export (default value)",
    )
    parser.add_argument(
        "--quality-col",
        default=None,
        help="quality column of a single-tag or --tag-col export (default quality)",
    )
    parser.add_argument(
        "--init-meta",
        default=None,
        metavar="FILE|DIR",
        help="write a metadata template to FILE and stop, then fill it in and ingest "
        "with --meta FILE; with --wide or --tag-col, write a <tag>.json template per "
        "tag into DIR, then ingest with --meta-dir DIR",
    )
    several = parser.add_argument_group("exports of several tags")
    several.add_argument(
        "--wide",
        action="store_true",
        help="read one column per tag and write one archive per tag into --out, named "
        "by point_id",
    )
    several.add_argument(
        "--tag-col",
        default=None,
        metavar="COL",
        help="read the tag of each row from COL and write one archive per tag into "
        "--out, named by point_id",
    )
    several.add_argument(
        "--meta-dir",
        default=None,
        metavar="DIR",
        help="directory of <tag>.json metadata files, one per tag",
    )
    several.add_argument(
        "--tags",
        default=None,
        metavar="A,B,...",
        help="tags to ingest; omitted, every column that is not the timestamp or a "
        "quality column (--wide), or every value of --tag-col",
    )
    several.add_argument(
        "--quality-suffix",
        default=None,
        metavar="S",
        help="with --wide, each tag's quality column is <tag><S>",
    )
    several.add_argument(
        "--source-id",
        default=None,
        metavar="ID",
        help="identity.source_id written into every template (--init-meta)",
    )
    parser.add_argument(
        "--tz",
        default=None,
        help="IANA zone the source's naive timestamps are in; they are localised to it "
        "and converted to UTC. Timestamps that already carry an offset ignore this",
    )
    parser.add_argument(
        "--assume-quality",
        default=None,
        metavar="GOOD|UNCERTAIN|BAD",
        help="declare a quality for a source that has no quality column; recorded on "
        "the archive and reported by every profile of it",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing archive at --out, or template under --init-meta",
    )
    return _add_output_flags(parser, json_flag=False)


def _check_ingest_flags(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Reject flags that belong to another ingest form."""
    long = args.tag_col is not None
    # A long export names its value and quality columns as a single-tag one does.
    single_only = [("--meta", args.meta)]
    if not long:
        single_only += [("--value-col", args.value_col), ("--quality-col", args.quality_col)]
    several_only = (("--meta-dir", args.meta_dir), ("--tags", args.tags))
    init_only = (("--source-id", args.source_id),)
    ingest_only = (
        ("--out", args.out),
        ("--meta-dir", args.meta_dir),
        ("--tz", args.tz),
        ("--assume-quality", args.assume_quality),
        ("--timestamp-format", args.timestamp_format),
        ("--dayfirst", args.dayfirst or None),
    )
    if args.timestamp_format is not None and args.dayfirst:
        parser.error("--timestamp-format and --dayfirst both state the date order; pass one")
    if args.wide and long:
        parser.error("--tag-col does not apply with --wide; pass one")
    if long and args.quality_suffix is not None:
        parser.error("--quality-suffix requires --wide")
    if args.wide or long:
        form = "--wide" if args.wide else "--tag-col"
        for flag, value in single_only:
            if value is not None:
                parser.error(f"{flag} does not apply with {form}; use --meta-dir")
        if args.init_meta is not None:
            for flag, value in ingest_only:
                if value is not None:
                    parser.error(f"{flag} does not apply with --init-meta")
            if args.source_id is None:
                parser.error("--init-meta requires --source-id")
        else:
            for flag, value in init_only:
                if value is not None:
                    parser.error(f"{flag} requires --init-meta")
            if args.out is None:
                parser.error("the following arguments are required: --out")
            if args.meta_dir is None:
                parser.error(f"{form} requires --meta-dir")
    else:
        if args.quality_suffix is not None:
            parser.error("--quality-suffix requires --wide")
        for flag, value in (*several_only, *init_only):
            if value is not None:
                parser.error(f"{flag} requires --wide or --tag-col")
        if args.init_meta is not None:
            for flag, value in (*ingest_only, ("--meta", args.meta)):
                if value is not None:
                    parser.error(f"{flag} does not apply with --init-meta")
            return
        if args.out is None:
            parser.error("the following arguments are required: --out")
        if args.meta is None:
            parser.error("the following arguments are required: --meta")


def _parser_demo() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tsdive demo")
    parser.add_argument(
        "directory",
        nargs="?",
        default=DEMO_DEFAULT_DIR,
        metavar="DIR",
        help="directory to write the demo archives into; existing files are refused",
    )
    return _add_output_flags(parser, json_flag=False)


def cmd_demo(argv: Sequence[str] | None = None) -> int:
    """Write the demo archives into a directory and print the command to try next."""
    args = _parser_demo().parse_args(argv)
    try:
        paths = write_demo_data(args.directory)
    except (OSError, ValueError) as e:
        _print_refusal("error:", error_text(e), args)
        return USAGE
    lines = []
    for i, path in enumerate(paths):
        rows = parquet.read_metadata(path).num_rows
        text = f"{path.as_posix()}   {rows} samples"
        lines.append(label_line("wrote", text) if i == 0 else continued(text))
    lines.append(label_line("next", f"tsdive profile {paths[0].as_posix()}"))
    _print_lines(lines, args)
    return OK


def cmd_ingest(argv: Sequence[str] | None = None) -> int:
    """Build an archive from a CSV or parquet export."""
    parser = _parser_ingest()
    args = parser.parse_args(argv)
    _check_ingest_flags(parser, args)
    with cli_names():
        return _ingest(args)


def _ingest(args: argparse.Namespace) -> int:
    try:
        if args.init_meta is not None and not args.wide and args.tag_col is None:
            template = init_tag_meta(
                args.source,
                out=args.init_meta,
                timestamp_col=args.timestamp_col,
                value_col=args.value_col or "value",
                quality_col=args.quality_col or "quality",
                overwrite=args.overwrite,
                **_csv_options(args),
            )
            _print_lines([label_line("wrote", template.as_posix())], args)
            return OK
        if args.init_meta is not None and args.tag_col is not None:
            templates = init_long_meta(
                args.source,
                out_dir=args.init_meta,
                source_id=args.source_id,
                tag_col=args.tag_col,
                timestamp_col=args.timestamp_col,
                value_col=args.value_col or "value",
                quality_col=args.quality_col or "quality",
                tags=_split_tags(args.tags),
                overwrite=args.overwrite,
                **_csv_options(args),
            )
            _print_lines([label_line("wrote", path.as_posix()) for path in templates], args)
            return OK
        if args.init_meta is not None:
            templates = init_meta(
                args.source,
                out_dir=args.init_meta,
                source_id=args.source_id,
                timestamp_col=args.timestamp_col,
                tags=_split_tags(args.tags),
                quality_suffix=args.quality_suffix,
                overwrite=args.overwrite,
                **_csv_options(args),
            )
            _print_lines([label_line("wrote", path.as_posix()) for path in templates], args)
            return 0
        if args.tag_col is not None:
            written = ingest_long(
                args.source,
                out_dir=args.out,
                meta_dir=args.meta_dir,
                tag_col=args.tag_col,
                timestamp_col=args.timestamp_col,
                value_col=args.value_col or "value",
                quality_col=args.quality_col or "quality",
                tags=_split_tags(args.tags),
                tz=args.tz,
                assume_quality=args.assume_quality,
                overwrite=args.overwrite,
                timestamp_format=args.timestamp_format,
                dayfirst=args.dayfirst,
                **_csv_options(args),
            )
            _print_lines([label_line("wrote", path.as_posix()) for path in written], args)
            _warn_assumed_quality(args.assume_quality)
            return OK
        if args.wide:
            written = ingest_wide(
                args.source,
                out_dir=args.out,
                meta_dir=args.meta_dir,
                timestamp_col=args.timestamp_col,
                tags=_split_tags(args.tags),
                quality_suffix=args.quality_suffix,
                tz=args.tz,
                assume_quality=args.assume_quality,
                overwrite=args.overwrite,
                timestamp_format=args.timestamp_format,
                dayfirst=args.dayfirst,
                **_csv_options(args),
            )
            _print_lines([label_line("wrote", path.as_posix()) for path in written], args)
            _warn_assumed_quality(args.assume_quality)
            return 0
        value_col = args.value_col or "value"
        quality_col = args.quality_col or "quality"
        meta = read_meta_json(args.meta)
        out = ingest(
            args.source,
            out=args.out,
            meta=meta,
            timestamp_col=args.timestamp_col,
            value_col=value_col,
            quality_col=quality_col,
            tz=args.tz,
            assume_quality=args.assume_quality,
            overwrite=args.overwrite,
            timestamp_format=args.timestamp_format,
            dayfirst=args.dayfirst,
            **_csv_options(args),
        )
        quality = (
            f"quality assumed {args.assume_quality.strip().upper()}"
            if args.assume_quality
            else f"quality from column {quality_col}"
        )
        lines = [
            label_line("wrote", Path(out).as_posix()),
            label_line(
                "tag",
                f"{meta.identity}{SEP}"
                f"{plural(parquet.read_metadata(out).num_rows, 'row')}{SEP}{quality}",
            ),
        ]
        if args.tz:
            lines.append(label_line("tz", f"{args.tz} -> UTC"))
        _print_lines(lines, args)
        _warn_assumed_quality(args.assume_quality)
        return OK
    except TSDiveError as e:
        return _refused(e, args)
    except (ValueError, OSError) as e:
        _print_refusal("error:", error_text(e), args)
        return USAGE


def _parser_report_html() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tsdive report-html")
    parser.add_argument("parquet", nargs="+", help="archive(s) to profile into the report")
    parser.add_argument(
        "--window",
        default=None,
        help="ISO 8601 START/END in UTC applied to every archive; omitted, the span "
        "covering every readable archive's extent",
    )
    parser.add_argument(
        "-o", "--out", default="tsdive-report.html", help="path of the HTML file to write"
    )
    return _add_output_flags(parser, json_flag=False)


def cmd_report_html(argv: Sequence[str] | None = None) -> int:
    """Render one static HTML page that profiles each archive over one window."""
    args = _parser_report_html().parse_args(argv)
    with cli_names():
        return _report_html(args)


def _report_html(args: argparse.Namespace) -> int:
    from tsdive.narrate import EvidenceLedger
    from tsdive.ui.static_report import render_static_report, write_static_report
    from tsdive.ui.svg import window_figure

    try:
        start, end = (
            parse_window(args.window) if args.window else _combined_extent(args.parquet)
        )
        window_label = f"{start.isoformat()}/{end.isoformat()}"
        profiles: list[str] = []
        figures: list[str] = []
        rows: list[dict[str, str]] = []
        contract = SamplingContract(CalculationBasis.TIME_WEIGHTED, RetrievalMode.RECORDED)
        for path in args.parquet:
            try:
                store = SingleFileStore(Path(path))
                meta = meta_from_parquet(store.path)
                window = store.read_window(
                    meta.identity,
                    start.to_pydatetime(),
                    end.to_pydatetime(),
                    contract,
                )
                profiles.append(
                    render_window_report(window, stats=compute_stats(window))
                )
                figures.append(window_figure(window))
            except (TSDiveError, OSError) as e:
                rows.append({"step": "profile", "tags": str(path), **error_fields(e)})
        refusals = [f"[{row['error_type']}] {row['cause']}" for row in rows]
        ledger = EvidenceLedger(
            title=f"snapshot {window_label}", profiles=profiles, refusals=rows
        )
        html_text = render_static_report(
            title="tsdive evidence snapshot",
            profiles=profiles,
            benchmarks_markdown_rows=[
                ("profiles rendered", str(len(profiles))),
                ("refusals recorded", str(len(refusals))),
                ("ledger", ", ".join(f"{k}={v}" for k, v in ledger.summary_stats().items())),
            ],
            refusal_log=refusals,
            figures=figures,
        )
        out = write_static_report(html_text, Path(args.out))
        _print_lines(
            [
                label_line("wrote", Path(out).as_posix()),
                label_line("window", fmt_span(start, end)),
                label_line(
                    "profiles",
                    f"{len(profiles)}{SEP}refusals {len(refusals)}",
                ),
            ],
            args,
        )
        return OK
    except TSDiveError as e:
        return _refused(e, args)
    except (ValueError, OSError) as e:
        _print_refusal("error:", error_text(e), args)
        return USAGE


# The analysis steps a plan may name, in pipeline order: a plan's listing
# order never decides the run order, because stage n reads what stage n-1
# established. Each entry pairs a step's own parser with its runner, so a
# plan can express nothing a command line cannot and both go through the
# same validators. A dict, deliberately: no registry, no entry points.
STEPS: dict[str, tuple[Callable[[], argparse.ArgumentParser], StepRunner]] = {
    "profile": (_parser_profile, run_profile),
    "segment": (_parser_segment, run_segment),
    "screen": (_parser_screen, run_screen),
    "spc": (_parser_spc, run_spc),
    "mspc": (_parser_mspc, run_mspc),
    "compare": (_parser_compare, run_compare),
    "switchback": (_parser_switchback_analyze, run_switchback_analyze),
}

# The same steps returning their analysis objects instead of text, for a
# caller that draws or inspects a result after rendering it. Each object
# has ``render()``, and ``STEPS`` prints exactly its lines.
ANALYSES: dict[str, Callable[[argparse.Namespace], object]] = {
    "profile": _profiled,
    "segment": _segment_run,
    "screen": _screen_run,
    "spc": _spc_run,
    "mspc": _mspc_run,
    "compare": _compared,
    "switchback": _switchback_analyzed,
}

# mspc, compare and switchback read every archive at once; the other
# steps run once per archive.
MULTI_TAG_STEPS = frozenset({"mspc", "compare", "switchback"})

# compare grades an after period against a before period, so a plan hands
# it --before and --after and neither --window nor --baseline.
TWO_PERIOD_STEPS = frozenset({"compare"})

# profile and segment default an omitted window to the archive's own
# extent. The screen, spc and mspc steps grade one window against another
# and have no such default, so a plan without a window refuses them by
# name.
EXTENT_DEFAULTED = frozenset({"profile", "segment"})

# Steps that need a history window to compare against.
BASELINED = frozenset({"screen", "spc", "mspc"})

# switchback reads its schedule from the plan file its options name, so a
# plan hands it neither a window nor periods.
SCHEDULED_STEPS = frozenset({"switchback"})


def _parser_run() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tsdive run",
        epilog="exit status: 0 when the ledger holds at least one profile or finding, "
        "refused and failed steps included as rows of the ledger; 2 when the plan cannot "
        "be read or no step produced a result. With --strict: 2 when a step filed an "
        "error, else 3 when a step was refused, else 0",
    )
    parser.add_argument("plan", help="TOML plan naming archives, windows and steps")
    parser.add_argument(
        "-o",
        "--out",
        default=None,
        metavar="DIR",
        help="directory for ledger.json, ledger.txt and report.html; omitted, "
        "tsdive-run/ beside the plan",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit 2 when a step filed an error and 3 when a step was refused, as the "
        "single commands do",
    )
    return _add_output_flags(parser, json_flag=False)


def cmd_run(argv: Sequence[str] | None = None) -> int:
    """Walk one plan over several archives into one evidence ledger.

    A refused or failed step is a row of the ledger and does not change
    the exit status: 0 when the ledger holds at least one profile or
    finding, 2 when the plan cannot be read or no step produced a result.
    With ``--strict`` the rows decide it: 2 when ``errors`` is not empty,
    else 3 when ``refusals`` is not empty, else 0.
    """
    args = _parser_run().parse_args(argv)
    with cli_names():
        return _run(args)


def _run(args: argparse.Namespace) -> int:
    from tsdive.plan import execute, load_plan, write_run

    plan_path = Path(args.plan)
    try:
        plan = load_plan(plan_path)
    except TSDiveError as e:
        return _refused(e, args)
    except (ValueError, OSError) as e:
        # Nothing is written for a plan that never started: an unknown
        # step or an unmatched glob leaves no half-run output directory.
        _print_refusal("error:", error_text(e), args)
        return USAGE
    walk = execute(plan)
    out_dir = Path(args.out) if args.out else plan_path.parent / "tsdive-run"
    _, _, lines = write_run(plan, out_dir, walk)
    _print_lines(lines, args)
    if args.strict:
        return USAGE if walk.errors else REFUSED if walk.refusals else OK
    # A refused or failed step is a row of the ledger, so it does not
    # decide the status; a ledger with no profile and no finding does.
    return OK if walk.profiles or walk.findings else USAGE


def main(argv: Sequence[str] | None = None) -> int:
    argv_l = list(sys.argv[1:] if argv is None else argv)
    # MAIN_DOC calls --json and --no-color global, but each one is owned by
    # the command's own parser, so a leading run of them moves behind the
    # command name before dispatch.
    lead = 0
    while lead < len(argv_l) and argv_l[lead] in {"--json", "--no-color"}:
        lead += 1
    if 0 < lead < len(argv_l):
        argv_l = [argv_l[lead], *argv_l[:lead], *argv_l[lead + 1 :]]
    if not argv_l or argv_l[0] in {"-h", "--help"}:
        print(MAIN_DOC.strip(), file=sys.stderr)
        return 0 if argv_l else 2
    if argv_l[0] in {"-V", "--version"}:
        print(f"tsdive {__version__}")
        return 0
    if argv_l[0] == "demo":
        return cmd_demo(argv_l[1:])
    if argv_l[0] == "profile":
        return cmd_profile(argv_l[1:])
    if argv_l[0] == "ingest":
        return cmd_ingest(argv_l[1:])
    if argv_l[0] == "report-html":
        return cmd_report_html(argv_l[1:])
    if argv_l[0] == "segment":
        return cmd_segment(argv_l[1:])
    if argv_l[0] == "screen":
        return cmd_screen(argv_l[1:])
    if argv_l[0] == "spc":
        return cmd_spc(argv_l[1:])
    if argv_l[0] == "mspc":
        return cmd_mspc(argv_l[1:])
    if argv_l[0] == "compare":
        return cmd_compare(argv_l[1:])
    if argv_l[0] == "switchback":
        return cmd_switchback(argv_l[1:])
    if argv_l[0] == "run":
        return cmd_run(argv_l[1:])
    print(
        f"unknown command {argv_l[0]!r}; try 'demo', 'profile', 'segment', 'screen', "
        "'spc', 'mspc', 'compare', 'switchback', 'run', 'ingest' or 'report-html'",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())

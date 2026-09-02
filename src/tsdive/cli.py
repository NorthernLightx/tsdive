"""``tsdive`` entry points.

The working loop a new archive walks:

    ingest -> profile -> segment -> screen -> spc -> mspc

Every command after ``ingest`` takes the same ``START/END`` window
syntax and refuses the same way: ``[ErrorName] message`` on stderr, exit
2. On ``profile`` an omitted window means the archive's whole extent,
which the report states like any other window.

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
from tsdive.analyses import (
    CompareAnalysis,
    MspcAnalysis,
    ScreenAnalysis,
    SegmentAnalysis,
    SpcAnalysis,
)
from tsdive.analyses_render import window_json
from tsdive.api import Profile, ingest, parse_window, profile, read_meta_json
from tsdive.changepoints import DEFAULT_PENALTY_MULTIPLIER
from tsdive.detectors.flatline import FlatlineVerdict
from tsdive.errors import TSDiveError
from tsdive.features.window_features import compute_stats
from tsdive.report import (
    SEP,
    fmt_span,
    label_line,
    plural,
    render_window_report,
)
from tsdive.store.gaps import DATA_LOSS_CLASSES
from tsdive.store.quality import Severity
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
from tsdive.ui.jsonout import to_jsonable
from tsdive.ui.term import colour_enabled, colourise, red

MAIN_DOC = """tsdive - data-quality profiling and monitoring for process time series

commands, in the order an archive walks them:
  ingest <csv|parquet> --out ARCHIVE --meta META.json
                                          build an archive from an export
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
  run <plan.toml> [-o DIR]                walk one plan over several archives into
                                          one evidence ledger
  report-html <parquet...> [--window START/END] [-o FILE]
                                          render a static HTML evidence snapshot

options, on every command:
  --json                                  one JSON object on stdout instead of
                                          text (the analysis commands)
  --no-color                              plain text; NO_COLOR does the same
  --version                               print the tsdive version
"""


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


def _lines(text: str) -> list[str]:
    """A rendered report back as the lines it was joined from.

    ``render()`` joins its lines with newlines and carries no trailing
    one, so splitting on newlines is exact.
    """
    return text.split("\n")


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
    """
    try:
        if getattr(args, "json", False) and to_json is not None:
            print(json.dumps(to_jsonable(to_json(args)), indent=2))
            return 0
        _print_lines(fn(args), args)
        return 0
    except TSDiveError as e:
        _print_refusal(f"[{type(e).__name__}]", str(e), args)
        return 2
    except (ValueError, FileNotFoundError) as e:
        _print_refusal("error:", str(e), args)
        return 2


def _parser_screen() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tsdive screen")
    parser.add_argument("parquet", help="single-tag parquet archive to screen")
    parser.add_argument(
        "--baseline", required=True, help="ISO 8601 START/END in UTC for the history"
    )
    parser.add_argument(
        "--window", required=True, help="ISO 8601 START/END in UTC to screen"
    )
    parser.add_argument("--method", default="mad", choices=["mad", "moving-range"])
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
        "--basis", default="TIME_WEIGHTED", choices=[b.value for b in CalculationBasis]
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
    return _lines(_screen_run(args).render())


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
        "--basis", default="TIME_WEIGHTED", choices=[b.value for b in CalculationBasis]
    )
    parser.add_argument(
        "--stepped", action="store_true", help="stepped interpolation between samples"
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


def run_segment(args: argparse.Namespace) -> list[str]:
    """Segment a window into regimes the samples themselves show."""
    return _lines(_segment_run(args).render())


def json_segment(args: argparse.Namespace) -> dict[str, object]:
    return _segment_run(args).to_dict()


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
        "--basis", default="TIME_WEIGHTED", choices=[b.value for b in CalculationBasis]
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
    return _lines(_spc_run(args).render())


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
    return _lines(_mspc_run(args).render())


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
    return _lines(_compared(args).render())


def json_compare(args: argparse.Namespace) -> dict[str, object]:
    return _compared(args).to_dict()


def cmd_compare(argv: Sequence[str] | None = None) -> int:
    """Report what changed between two periods of one unit."""
    return _report_and_exit(
        run_compare, _parser_compare().parse_args(argv), to_json=json_compare
    )


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
        "--basis", default="TIME_WEIGHTED", choices=[b.value for b in CalculationBasis]
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
    return _lines(_profiled(args).render())


def _flatline_json(verdict: FlatlineVerdict) -> dict[str, object]:
    if verdict.saturated_not_frozen or verdict.not_assessed is not None:
        text = "NOT ASSESSED"
    else:
        text = "FLATLINE SUSPECTED" if verdict.fired else "no flatline"
    return {
        "verdict": text,
        "fired": verdict.fired,
        "saturated_not_frozen": verdict.saturated_not_frozen,
        "not_assessed": verdict.not_assessed,
        "signals": list(verdict.signals),
    }


def _profile_json(result: Profile) -> dict[str, object]:
    """Every number the profile report states, keyed for a program."""
    window = result.window
    p = window.physics
    cov = p.coverage
    ta = p.timestamp_audit
    u = p.unit
    stats = result.stats
    f = stats.features
    c = window.contract
    return {
        "tag": str(window.identity),
        "name": window.meta.name,
        "window": window_json(window),
        "contract": {
            "calculation_basis": c.calculation_basis,
            "retrieval_mode": c.retrieval_mode,
            "aggregate_type": c.aggregate_type,
            "stepped": c.stepped,
            "digest": c.digest(),
        },
        "units": {
            "raw": u.raw,
            "canonical": u.canonical,
            "pint_units": u.pint_units,
            "reference_condition": u.reference_condition,
            "resolved": u.resolved,
        },
        "coverage": {
            "coverage": cov.coverage,
            "valid_fraction": p.valid_fraction,
            "n_gaps": cov.n_gaps,
            "n_data_loss_gaps": sum(
                1 for g in cov.gaps if g.classification.cls in DATA_LOSS_CLASSES
            ),
            "longest_gap_s": cov.longest_gap_s,
            "edge_slack_s": p.edge_slack_s,
            "edge_slack_start_s": p.edge_slack_start_s,
            "edge_slack_end_s": p.edge_slack_end_s,
            "gaps": [
                {
                    "start": g.start,
                    "end": g.end,
                    "duration_s": g.duration_s,
                    "class": g.classification.cls,
                    "rule": g.classification.rule,
                }
                for g in cov.gaps
            ],
        },
        "quality": {
            "counts": {s.value: p.severity_counts.get(s, 0) for s in Severity},
            "assumed_at_ingest": window.meta.quality_assumed,
            "unmapped_codes": p.unmapped_quality_codes,
            "unmapped_look_like_opc_da": p.unmapped_look_like_opc_da,
        },
        "range": {
            "clipped_fraction": p.clipping.fraction,
            "n_clipped": p.clipping.n_clipped,
            "censored": p.clipping.censored,
            "beyond_float32_count": p.implausible_magnitude_count,
        },
        "timestamps": {
            "audited": ta.n_samples,
            "duplicates": len(ta.duplicate_timestamps),
            "non_monotonic": len(ta.non_monotonic_positions),
            "dst_transitions": [
                {
                    "tz": t.tz,
                    "instant": t.instant_utc,
                    "offset_before_s": t.offset_before_s,
                    "offset_after_s": t.offset_after_s,
                }
                for t in ta.dst_transitions
            ],
        },
        "values": {
            "n_good": f.n_good,
            "n_samples": f.n_samples,
            "state_valued": stats.state_valued,
            "state_counts": [{"state": s, "n": n} for s, n in stats.state_counts],
            "min": f.min,
            "p05": stats.p05,
            "median": f.median,
            "p95": stats.p95,
            "max": f.max,
            "mean": f.weighted_mean,
            "mean_basis": c.calculation_basis,
            "mean_note": stats.mean_note,
            "std": f.std,
            "mad": stats.mad,
            "distinct": stats.distinct_count,
            "stall_s": f.stall_s,
            "changes_per_hour": f.changes_per_hour,
            "interval_median_s": stats.interval_median_s,
            "interval_p05_s": stats.interval_p05_s,
            "interval_p95_s": stats.interval_p95_s,
            "declared_rate_s": stats.declared_rate_s,
            "interval_differs_from_declared": stats.interval_differs_from_declared,
        },
        "flatline": (
            None if result.flatline is None else _flatline_json(result.flatline)
        ),
    }


def json_profile(args: argparse.Namespace) -> dict[str, object]:
    return _profile_json(_profiled(args))


def cmd_profile(argv: Sequence[str] | None = None) -> int:
    """Report the data physics and statistics of one window."""
    return _report_and_exit(
        run_profile, _parser_profile().parse_args(argv), to_json=json_profile
    )


def cmd_ingest(argv: Sequence[str] | None = None) -> int:
    """Build an archive from a CSV or parquet export."""
    parser = argparse.ArgumentParser(prog="tsdive ingest")
    parser.add_argument("source", help="CSV or parquet export of one tag")
    parser.add_argument("--out", required=True, help="archive to create")
    parser.add_argument(
        "--meta", required=True, help="JSON file of tag metadata (the tsdive.meta object)"
    )
    parser.add_argument("--timestamp-col", default="timestamp")
    parser.add_argument("--value-col", default="value")
    parser.add_argument("--quality-col", default="quality")
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
        "--overwrite", action="store_true", help="replace an existing archive at --out"
    )
    _add_output_flags(parser, json_flag=False)
    args = parser.parse_args(argv)

    try:
        meta = read_meta_json(args.meta)
        out = ingest(
            args.source,
            out=args.out,
            meta=meta,
            timestamp_col=args.timestamp_col,
            value_col=args.value_col,
            quality_col=args.quality_col,
            tz=args.tz,
            assume_quality=args.assume_quality,
            overwrite=args.overwrite,
        )
        quality = (
            f"quality assumed {args.assume_quality.strip().upper()}"
            if args.assume_quality
            else f"quality from column {args.quality_col}"
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
        if args.assume_quality:
            print(
                f"warning: quality assumed {args.assume_quality.strip().upper()} for every "
                "sample; the archive records this and every profile of it says so",
                file=sys.stderr,
            )
        return 0
    except TSDiveError as e:
        _print_refusal(f"[{type(e).__name__}]", str(e), args)
        return 2
    except (ValueError, OSError) as e:
        _print_refusal("error:", str(e), args)
        return 2


def cmd_report_html(argv: Sequence[str] | None = None) -> int:
    """Render a static HTML evidence snapshot from demo archives."""
    from tsdive.narrate import EvidenceLedger
    from tsdive.ui.static_report import render_static_report, write_static_report

    parser = argparse.ArgumentParser(prog="tsdive report-html")
    parser.add_argument("parquet", nargs="+", help="archive(s) to profile into the report")
    parser.add_argument(
        "--window",
        default=None,
        help="ISO 8601 START/END in UTC applied to every archive; omitted, the span "
        "covering every readable archive's extent",
    )
    parser.add_argument("-o", "--out", default="tsdive-report.html")
    _add_output_flags(parser, json_flag=False)
    args = parser.parse_args(argv)

    try:
        start, end = (
            parse_window(args.window) if args.window else _combined_extent(args.parquet)
        )
        window_label = f"{start.isoformat()}/{end.isoformat()}"
        profiles: list[str] = []
        refusals: list[str] = []
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
            except TSDiveError as e:
                refusals.append(f"[{type(e).__name__}] {e}")
            except OSError as e:
                refusals.append(f"[FileNotFoundError] {e}")
        ledger = EvidenceLedger(
            title=f"snapshot {window_label}", profiles=profiles, refusals=refusals
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
        return 0
    except TSDiveError as e:
        _print_refusal(f"[{type(e).__name__}]", str(e), args)
        return 2
    except (ValueError, FileNotFoundError) as e:
        _print_refusal("error:", str(e), args)
        return 2


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
}

# mspc reads every archive at once; the other steps run once per archive.
MULTI_TAG_STEPS = frozenset({"mspc"})

# profile and segment default an omitted window to the archive's own
# extent. The screen, spc and mspc steps grade one window against another
# and have no such default, so a plan without a window refuses them by
# name.
EXTENT_DEFAULTED = frozenset({"profile", "segment"})

# Steps that need a history window to compare against.
BASELINED = frozenset({"screen", "spc", "mspc"})


def _parser_run() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tsdive run")
    parser.add_argument("plan", help="TOML plan naming archives, windows and steps")
    parser.add_argument(
        "-o",
        "--out",
        default=None,
        metavar="DIR",
        help="directory for ledger.json, ledger.txt and report.html; omitted, "
        "tsdive-run/ beside the plan",
    )
    return _add_output_flags(parser, json_flag=False)


def cmd_run(argv: Sequence[str] | None = None) -> int:
    """Walk one plan over several archives into one evidence ledger."""
    from tsdive.plan import execute, load_plan, write_run

    args = _parser_run().parse_args(argv)
    plan_path = Path(args.plan)
    try:
        plan = load_plan(plan_path)
    except TSDiveError as e:
        _print_refusal(f"[{type(e).__name__}]", str(e), args)
        return 2
    except (ValueError, OSError) as e:
        # Nothing is written for a plan that never started: an unknown
        # step or an unmatched glob leaves no half-run output directory.
        _print_refusal("error:", str(e), args)
        return 2
    profiles, findings, refusals = execute(plan)
    out_dir = Path(args.out) if args.out else plan_path.parent / "tsdive-run"
    _, _, lines = write_run(plan, out_dir, profiles, findings, refusals)
    _print_lines(lines, args)
    return 0 if profiles or findings else 2


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
    if argv_l[0] == "run":
        return cmd_run(argv_l[1:])
    print(
        f"unknown command {argv_l[0]!r}; try 'profile', 'segment', 'screen', 'spc', "
        "'mspc', 'compare', 'run', 'ingest' or 'report-html'",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())

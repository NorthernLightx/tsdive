"""``tsdive`` entry points.

The working loop a new archive walks:

    ingest -> profile -> segment -> screen -> spc -> mspc

Every command after ``ingest`` takes the same ``START/END`` window
syntax and refuses the same way: ``[ErrorName] message`` on stderr, exit
2. On ``profile`` an omitted window means the archive's whole extent,
which the report states like any other window.

Argument parsing, printing and exit codes live here; the profiling flow
itself is :func:`tsdive.api.profile`, so the library and the CLI can
never disagree about what a profile is.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

import pandas as pd
from pyarrow import parquet

from tsdive import __version__
from tsdive.api import Profile, ingest, parse_window, profile, read_meta_json
from tsdive.baselines.provisional import (
    ProvisionalBaseline,
    ScreenResult,
    mad_baseline,
    moving_range_baseline,
    screen,
)
from tsdive.baselines.regime import RegimeBaseline, regime_baselines, screen_regime
from tsdive.changepoints import DEFAULT_PENALTY_MULTIPLIER, segment_window
from tsdive.changepoints.pelt import Segmentation
from tsdive.compare import (
    NO_INTERVAL,
    CompareResult,
    JointStructure,
    PairChange,
    PairTable,
    TagChange,
    compare,
)
from tsdive.detectors.flatline import FlatlineVerdict
from tsdive.errors import InsufficientQuality, SchemaError, TSDiveError
from tsdive.features.window_features import compute_stats
from tsdive.mspc.pca import (
    AlignedMatrix,
    MspcDetection,
    PcaModel,
    align_windows,
    common_rate,
    detect,
    fit_pca,
)
from tsdive.report import (
    MAX_WIDTH,
    SEP,
    continued,
    fits,
    fmt_duration,
    fmt_num,
    fmt_span,
    fmt_ts,
    indent,
    label_line,
    more_line,
    plural,
    render_window_report,
    wrapped,
    yes_no,
)
from tsdive.spc.charts import ControlLimits, RuleHit, apply_rules, individuals_limits
from tsdive.store.clipping import assert_usable_baseline
from tsdive.store.gaps import DATA_LOSS_CLASSES
from tsdive.store.quality import Severity
from tsdive.store.sampling_contract import (
    AggregateType,
    CalculationBasis,
    RetrievalMode,
    SamplingContract,
)
from tsdive.store.tagstore import (
    SingleFileStore,
    Window,
    archive_extent,
    meta_from_parquet,
)
from tsdive.ui.jsonout import to_jsonable
from tsdive.ui.term import colour_enabled, colourise, red, rule

# Enough flagged timestamps to see where the screen fired without pasting
# a monitored window back at the caller.
FLAGGED_SHOWN = 3
HITS_SHOWN = 5

# The rule set apply_rules emits, in report order. A rule with no hits
# prints a zero count, so the report states which rules ran.
SPC_RULES = ("BEYOND_3SIGMA", "RUN_9_SAMESIDE", "TREND_6")

# MspcDetection.top_contributors keeps this many tags, so a model holding
# no more than that names all of them on every breach and ranks nothing.
CONTRIBUTORS_KEPT = 3

# compare's table 1: the fixed columns, as (header, minimum width,
# right-aligned). The tag column is elastic and comes first.
_CHANGE_COLUMNS = (
    ("quality", 8, False),
    ("sigma", 5, True),
    ("spread", 6, True),
    ("flagged", 7, True),
    ("changed at", 20, False),
)

# What the tag column takes when the fixed columns leave it the room: 16
# characters, until a quality value as wide as "UNCERTAIN 100%" claims
# some of it. Below TAG_COLUMN_MIN a truncated tag names nothing.
TAG_COLUMN_MAX = 16
TAG_COLUMN_MIN = 8

# compare's table 2, same shape; its elastic column holds two tag labels
# and the tilde between them.
_PAIR_COLUMNS = (
    ("before", 6, True),
    ("after", 6, True),
    ("delta", 6, True),
    ("interval", 14, False),
)
PAIR_COLUMN_MAX = 2 * TAG_COLUMN_MAX + 3

PAIR_METHOD = "(Pearson on first differences, 95% block bootstrap)"

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


def _n_good(window: Window) -> int:
    """Rows the window read judged trustworthy *and* usable."""
    return int(window.frame["valid"].sum())


def _window_json(window: Window) -> dict[str, object]:
    """The span a step read, as a program wants it."""
    return {
        "start": window.start,
        "end": window.end,
        "duration_s": (window.end - window.start).total_seconds(),
    }


def _baseline_line(window: Window) -> str:
    """States the censoring verdict the baseline was admitted on."""
    return label_line(
        "baseline",
        f"{fmt_span(window.start, window.end)}{SEP}GOOD {_n_good(window)}{SEP}"
        f"censored {yes_no(window.physics.clipping.censored)}",
    )


def _window_line(window: Window) -> str:
    return label_line("window", fmt_span(window.start, window.end))


def _baseline_json(window: Window) -> dict[str, object]:
    return {
        **_window_json(window),
        "n_good": _n_good(window),
        "censored": window.physics.clipping.censored,
    }


def _baseline_and_monitor(
    path: str,
    baseline: str,
    window: str,
    *,
    basis: str,
    stepped: bool,
) -> tuple[Window, Window]:
    """Read a baseline window and a monitored window from one archive.

    Two refusals live here rather than in the baseline, SPC and MSPC
    library calls, which are handed frames and never see where those
    frames came from: a baseline overlapping the monitored window grades
    those samples against themselves, and a censored baseline carries no
    information about normal operation (:func:`assert_usable_baseline`).
    """
    b_start, b_end = parse_window(baseline)
    m_start, m_end = parse_window(window)
    # Touching bounds are not an overlap, but reads are inclusive at both
    # ends, so an adjacent pair would share the sample on the boundary.
    # It is given to the monitored window and dropped from the baseline
    # frame below.
    if b_start < m_end and m_start < b_end:
        raise ValueError("baseline and window overlap")
    store = SingleFileStore(Path(path))
    meta = meta_from_parquet(store.path)
    contract = SamplingContract(
        calculation_basis=CalculationBasis(basis),
        retrieval_mode=RetrievalMode.RECORDED,
        aggregate_type=AggregateType.NONE,
        stepped=stepped,
    )
    reads = [
        store.read_window(meta.identity, s.to_pydatetime(), e.to_pydatetime(), contract)
        for s, e in ((b_start, b_end), (m_start, m_end))
    ]
    base, monitor = reads
    assert_usable_baseline(meta, base.physics.clipping)
    if b_end == m_start:
        # The frame loses the boundary sample; the physics block keeps
        # it, so the censoring verdict the baseline was admitted on is
        # still the one taken over the whole read - the conservative side.
        base = replace(base, frame=base.frame[base.frame["timestamp"] < m_start])
    return base, monitor


def _aligned_modes(
    frame: pd.DataFrame, modes: Window, label: str
) -> tuple[pd.DataFrame, pd.Series, int]:
    """Inner-join a frame to its MODE rows on the exact timestamp.

    Exact only: a regime label carried over from a neighbouring second is
    a guess about what the plant was doing. Rows with no mode sample of
    their own are dropped and counted, and a window that loses more than
    half of them is refused rather than screened against a fragment.
    """
    usable = modes.frame.loc[modes.frame["valid"], ["timestamp", "value"]]
    if bool(usable["timestamp"].duplicated().any()):
        raise SchemaError(
            f"{modes.identity}: duplicate timestamps in the mode archive; joining "
            f"them would multiply {label} rows"
        )
    joined = frame.merge(
        usable.rename(columns={"value": "mode"}), on="timestamp", how="inner"
    ).reset_index(drop=True)
    dropped = len(frame) - len(joined)
    if dropped * 2 > len(frame):
        raise InsufficientQuality(
            f"{dropped} of {len(frame)} {label} rows carry no mode sample at their "
            "own timestamp; more than half the window would be screened blind"
        )
    return joined, cast(pd.Series, joined["mode"].astype(str)), dropped


@dataclass(frozen=True)
class _Alignment:
    """Rows that carried a mode sample of their own, over rows read."""

    kept: int
    total: int


@dataclass(frozen=True)
class _SegmentRun:
    window: Window
    found: Segmentation
    penalty_is_default: bool


@dataclass(frozen=True)
class _ScreenRun:
    """One screen, whether it used one baseline or one per regime.

    ``provisional`` and ``regimes`` are the two ways a baseline is built;
    exactly one is populated, and both the text and the JSON read the same
    object rather than screening the window twice.
    """

    baseline: Window
    monitor: Window
    result: ScreenResult
    k: float
    mode_path: str | None = None
    provisional: ProvisionalBaseline | None = None
    regimes: dict[str, RegimeBaseline] | None = None
    alignment: tuple[_Alignment, _Alignment] | None = None


@dataclass(frozen=True)
class _SpcRun:
    baseline: Window
    monitor: Window
    limits: ControlLimits
    sigma: float
    n_monitored: int
    hits: list[RuleHit]


@dataclass(frozen=True)
class _MspcRun:
    baseline: Window
    monitor: Window
    tags: list[str]
    rate_s: int
    rate_source: str
    quantile: float
    train: AlignedMatrix
    test: AlignedMatrix
    model: PcaModel
    found: MspcDetection

    @property
    def ranked(self) -> bool:
        """True when the model holds more tags than a top-N list would name."""
        return len(self.model.columns) > CONTRIBUTORS_KEPT


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


def _screen_run(args: argparse.Namespace) -> _ScreenRun:
    """Read both windows, build the baseline(s), and screen the monitored one."""
    base_w, monitor_w = _baseline_and_monitor(
        args.parquet, args.baseline, args.window, basis=args.basis, stepped=args.stepped
    )
    if not args.mode:
        base = (
            mad_baseline(base_w.frame)
            if args.method == "mad"
            else moving_range_baseline(base_w.frame)
        )
        return _ScreenRun(
            baseline=base_w,
            monitor=monitor_w,
            result=screen(base, monitor_w.frame, k=args.k),
            k=args.k,
            provisional=base,
        )
    mode_base, mode_monitor = _baseline_and_monitor(
        args.mode, args.baseline, args.window, basis=args.basis, stepped=args.stepped
    )
    b_frame, b_modes, b_dropped = _aligned_modes(base_w.frame, mode_base, "baseline")
    m_frame, m_modes, m_dropped = _aligned_modes(monitor_w.frame, mode_monitor, "window")
    regimes = regime_baselines(b_frame, b_modes)
    return _ScreenRun(
        baseline=base_w,
        monitor=monitor_w,
        result=screen_regime(regimes, m_frame, m_modes, k=args.k),
        k=args.k,
        mode_path=args.mode,
        regimes=regimes,
        alignment=(
            _Alignment(len(b_frame), len(b_frame) + b_dropped),
            _Alignment(len(m_frame), len(m_frame) + m_dropped),
        ),
    )


def _flagged_share(result: ScreenResult) -> str:
    if not result.n_screened:
        return "n/a"
    return f"{100.0 * result.n_flagged / result.n_screened:.1f}%"


def run_screen(args: argparse.Namespace) -> list[str]:
    """Screen a window against a baseline: MAD, or regime-keyed with --mode."""
    run = _screen_run(args)
    result = run.result
    lines = [
        f"{run.baseline.identity}  flagged {result.n_flagged} of "
        f"{result.n_screened} ({_flagged_share(result)})",
        "",
        _baseline_line(run.baseline),
        _window_line(run.monitor),
    ]
    if run.mode_path is not None:
        lines.append(label_line("mode", run.mode_path))
    if run.alignment is not None:
        base_a, monitor_a = run.alignment
        lines.append(
            label_line(
                "alignment",
                f"baseline {base_a.kept}/{base_a.total}{SEP}"
                f"window {monitor_a.kept}/{monitor_a.total}",
            )
        )
    if run.regimes is not None:
        lines.append(label_line("method", f"REGIME_MAD{SEP}k {run.k}"))
        lines.extend(
            rule("Regimes")
            + indent(
                f"regime {name}{SEP}center {fmt_num(r.center)}{SEP}"
                f"scale {fmt_num(r.scale)}{SEP}n {r.n_good}"
                for name, r in run.regimes.items()
            )
        )
    else:
        base = run.provisional
        assert base is not None
        lo, hi = base.limits(run.k)
        lines.append(
            label_line(
                "method",
                f"{base.kind}{SEP}center {fmt_num(base.center)}{SEP}"
                f"scale {fmt_num(base.scale)}{SEP}k {run.k}{SEP}"
                f"limits [{fmt_num(lo)}, {fmt_num(hi)}]",
            )
        )
        lines.append(label_line("caveat", base.caveat))
    if result.flagged_timestamps:
        # Regime screening walks regime by regime, so its timestamps come
        # back grouped rather than in time order; "first" has to mean first.
        stamps = sorted(result.flagged_timestamps)
        lines.extend(rule("Flagged"))
        lines.extend(indent(fmt_ts(t) for t in stamps[:FLAGGED_SHOWN]))
        lines.extend(more_line(len(stamps) - FLAGGED_SHOWN))
    return lines


def json_screen(args: argparse.Namespace) -> dict[str, object]:
    run = _screen_run(args)
    result = run.result
    base = run.provisional
    return {
        "tag": str(run.baseline.identity),
        "mode": run.mode_path,
        "baseline": _baseline_json(run.baseline),
        "window": _window_json(run.monitor),
        "method": result.kind,
        "k": run.k,
        "center": None if base is None else base.center,
        "scale": None if base is None else base.scale,
        "limits": None if base is None else list(base.limits(run.k)),
        "caveat": None if base is None else base.caveat,
        "regimes": (
            None
            if run.regimes is None
            else [
                {
                    "regime": name,
                    "center": r.center,
                    "scale": r.scale,
                    "n_good": r.n_good,
                    "method": r.method,
                }
                for name, r in run.regimes.items()
            ]
        ),
        "alignment": (
            None
            if run.alignment is None
            else {
                "baseline": {
                    "kept": run.alignment[0].kept,
                    "total": run.alignment[0].total,
                },
                "window": {
                    "kept": run.alignment[1].kept,
                    "total": run.alignment[1].total,
                },
            }
        ),
        "n_screened": result.n_screened,
        "n_flagged": result.n_flagged,
        "flagged": sorted(result.flagged_timestamps),
    }


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


def _segment_run(args: argparse.Namespace) -> _SegmentRun:
    """Read the window and cut it into piecewise-constant segments."""
    store = SingleFileStore(Path(args.parquet))
    meta = meta_from_parquet(store.path)
    start, end = parse_window(args.window) if args.window else archive_extent(store.path)
    contract = SamplingContract(
        calculation_basis=CalculationBasis(args.basis),
        retrieval_mode=RetrievalMode.RECORDED,
        aggregate_type=AggregateType.NONE,
        stepped=args.stepped,
    )
    window = store.read_window(
        meta.identity, start.to_pydatetime(), end.to_pydatetime(), contract
    )
    return _SegmentRun(
        window=window,
        found=segment_window(window, penalty=args.penalty, min_size=args.min_size),
        penalty_is_default=args.penalty is None,
    )


# Minimum width of each segment-table column, so the table keeps its shape
# on a short archive and widens only for a value that does not fit.
_SEGMENT_COLUMNS = (("#", 1, True), ("start", 20, False), ("end", 20, False),
                    ("n", 3, True), ("median", 7, True), ("mad", 6, True))


def _segment_table(found: Segmentation) -> list[str]:
    """The segment table: one row per segment, numbers right-aligned."""
    rows = [
        (
            str(i),
            fmt_ts(s.start),
            fmt_ts(s.end),
            str(s.n),
            fmt_num(s.median),
            fmt_num(s.mad),
        )
        for i, s in enumerate(found.segments, start=1)
    ]
    headers = tuple(c[0] for c in _SEGMENT_COLUMNS)
    widths = [
        max(minimum, len(head), *(len(r[i]) for r in rows))
        for i, (head, minimum, _) in enumerate(_SEGMENT_COLUMNS)
    ]
    def line(cells: tuple[str, ...]) -> str:
        return "  " + "  ".join(
            cell.rjust(w) if right else cell.ljust(w)
            for cell, w, (_, _, right) in zip(cells, widths, _SEGMENT_COLUMNS, strict=True)
        )

    return ["", line(headers), *(line(r) for r in rows)]


def run_segment(args: argparse.Namespace) -> list[str]:
    """Segment a window into regimes the samples themselves show."""
    run = _segment_run(args)
    window, found = run.window, run.found
    default_note = (
        f" ({DEFAULT_PENALTY_MULTIPLIER:g}*log n)" if run.penalty_is_default else ""
    )
    return [
        f"{window.identity}  {plural(len(found.segments), 'segment')}{SEP}"
        f"{plural(len(found.breakpoints), 'breakpoint')}{SEP}"
        # Not a refusal - a segment table is not a baseline - but a
        # segment pinned at full scale is a saturation, not a regime.
        f"censored {yes_no(window.physics.clipping.censored)}",
        "",
        label_line(
            "window",
            f"{fmt_span(window.start, window.end)}{SEP}usable {found.n_used}",
        ),
        label_line(
            "method",
            f"{found.method}{SEP}penalty {fmt_num(found.penalty)}{default_note}"
            f"{SEP}min-size {found.min_size}",
        ),
        *_segment_table(found),
    ]


def json_segment(args: argparse.Namespace) -> dict[str, object]:
    run = _segment_run(args)
    found = run.found
    return {
        "tag": str(run.window.identity),
        "window": _window_json(run.window),
        "censored": run.window.physics.clipping.censored,
        "usable": found.n_used,
        "method": found.method,
        "penalty": found.penalty,
        "penalty_is_default": run.penalty_is_default,
        "min_size": found.min_size,
        "breakpoints": list(found.breakpoints),
        "segments": [
            {
                "index": i,
                "start": s.start,
                "end": s.end,
                "n": s.n,
                "median": s.median,
                "mad": s.mad,
            }
            for i, s in enumerate(found.segments, start=1)
        ],
    }


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


def _spc_run(args: argparse.Namespace) -> _SpcRun:
    """Fit individuals limits on the baseline and run every rule over the window."""
    base_w, monitor_w = _baseline_and_monitor(
        args.parquet, args.baseline, args.window, basis=args.basis, stepped=args.stepped
    )
    base = mad_baseline(base_w.frame)
    limits = individuals_limits(base.center, base.scale)
    good = monitor_w.frame[monitor_w.frame["valid"]]
    return _SpcRun(
        baseline=base_w,
        monitor=monitor_w,
        limits=limits,
        sigma=base.scale,
        n_monitored=len(good),
        hits=apply_rules(
            good["timestamp"], cast(pd.Series, good["value"].astype(float)), limits
        ),
    )


def run_spc(args: argparse.Namespace) -> list[str]:
    """Chart a window against baseline control limits."""
    run = _spc_run(args)
    limits = run.limits
    lines = [
        f"{run.baseline.identity}  {plural(len(run.hits), 'rule hit')} in "
        f"{plural(run.n_monitored, 'sample')}",
        "",
        _baseline_line(run.baseline),
        _window_line(run.monitor),
        label_line(
            "limits",
            f"center {fmt_num(limits.center)}{SEP}sigma {fmt_num(run.sigma)}{SEP}"
            f"lcl {fmt_num(limits.lcl)}{SEP}ucl {fmt_num(limits.ucl)}",
        ),
        label_line("basis", limits.basis),
    ]
    for name in SPC_RULES:
        fired = [h for h in run.hits if h.rule == name]
        lines.extend(rule(name, str(len(fired))))
        lines.extend(
            indent(f"{fmt_ts(h.timestamp)}{SEP}{h.detail}" for h in fired[:HITS_SHOWN])
        )
        lines.extend(more_line(len(fired) - HITS_SHOWN))
    return lines


def json_spc(args: argparse.Namespace) -> dict[str, object]:
    run = _spc_run(args)
    return {
        "tag": str(run.baseline.identity),
        "baseline": _baseline_json(run.baseline),
        "window": _window_json(run.monitor),
        "limits": {
            "center": run.limits.center,
            "sigma": run.sigma,
            "lcl": run.limits.lcl,
            "ucl": run.limits.ucl,
            "basis": run.limits.basis,
        },
        "n_monitored": run.n_monitored,
        "n_hits": len(run.hits),
        "rules": [
            {
                "rule": name,
                "n": sum(1 for h in run.hits if h.rule == name),
                "hits": [
                    {"timestamp": h.timestamp, "detail": h.detail}
                    for h in run.hits
                    if h.rule == name
                ],
            }
            for name in SPC_RULES
        ],
    }


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


def _mspc_run(args: argparse.Namespace) -> _MspcRun:
    """Align every archive on one grid, fit PCA on the baseline, monitor the window."""
    pairs = [
        _baseline_and_monitor(
            path, args.baseline, args.window, basis="TIME_WEIGHTED", stepped=False
        )
        for path in args.parquet
    ]
    bases = [b for b, _ in pairs]
    monitors = [m for _, m in pairs]
    rate = args.rate_s if args.rate_s is not None else common_rate(bases)
    train = align_windows(bases, rate_s=rate, min_coverage=args.min_coverage)
    model = fit_pca(train, variance_threshold=args.variance, limit_quantile=args.quantile)
    test = align_windows(monitors, rate_s=rate, min_coverage=args.min_coverage)
    return _MspcRun(
        baseline=bases[0],
        monitor=monitors[0],
        tags=[str(b.identity) for b in bases],
        rate_s=rate,
        rate_source="declared" if args.rate_s is None else "--rate-s",
        quantile=args.quantile,
        train=train,
        test=test,
        model=model,
        found=detect(model, test),
    )


def _mspc_headline(run: _MspcRun) -> tuple[str, list[str]]:
    """The headline, and the ``tags`` lines it needs when the list will not fit."""
    found = run.found
    counts = (
        f"T2 breaches {len(found.t2_breaches)}{SEP}"
        f"SPE breaches {len(found.spe_breaches)}{SEP}"
        f"of {plural(len(run.test.index), 'row')}"
    )
    listed = ", ".join(run.tags)
    if fits(f"{listed}  {counts}"):
        return f"{listed}  {counts}", []
    headline = f"{plural(len(run.tags), 'tag')}  {counts}"
    if fits(label_line("tags", listed)):
        return headline, [label_line("tags", listed)]
    return headline, [
        label_line("tags", run.tags[0]),
        *(continued(tag) for tag in run.tags[1:]),
    ]


def _breach_lines(run: _MspcRun, stamps: list[pd.Timestamp]) -> list[str]:
    """Up to HITS_SHOWN breach timestamps, with contributors only when they rank."""
    lines: list[str] = []
    for ts in stamps[:HITS_SHOWN]:
        text = fmt_ts(ts)
        if run.ranked:
            named = ", ".join(run.found.top_contributors[ts.isoformat()])
            text += f"{SEP}contributors {named}"
        lines.extend(wrapped(text))
    return lines + more_line(len(stamps) - HITS_SHOWN)


def run_mspc(args: argparse.Namespace) -> list[str]:
    """Detect multivariate departures with PCA T2 and SPE."""
    run = _mspc_run(args)
    model = run.model
    headline, tag_lines = _mspc_headline(run)
    lines = [headline, "", *tag_lines]
    lines.extend(
        [
            label_line(
                "baseline",
                f"{fmt_span(run.baseline.start, run.baseline.end)}{SEP}"
                f"rows {len(run.train.index)}{SEP}"
                f"coverage {fmt_num(run.train.coverage)}",
            ),
            label_line(
                "window",
                f"{fmt_span(run.monitor.start, run.monitor.end)}{SEP}"
                f"rows {len(run.test.index)}{SEP}"
                f"coverage {fmt_num(run.test.coverage)}",
            ),
            label_line(
                "model",
                f"rate {run.rate_s} s ({run.rate_source}){SEP}"
                f"components {len(model.components)} of {len(model.columns)}{SEP}"
                f"explained {', '.join(fmt_num(v) for v in model.explained_variance)}",
            ),
            label_line(
                "limits",
                f"T2 {fmt_num(model.t2_limit)}{SEP}SPE {fmt_num(model.spe_limit)}"
                f"{SEP}(empirical q{run.quantile})",
            ),
        ]
    )
    if not run.ranked:
        lines.append(
            label_line(
                "contributors",
                f"not ranked ({plural(len(model.columns), 'tag')}; "
                f"top-{CONTRIBUTORS_KEPT} would list every one)",
            )
        )
    for name, breaches in (
        ("T2 breaches", run.found.t2_breaches),
        ("SPE breaches", run.found.spe_breaches),
    ):
        lines.extend(rule(name, str(len(breaches))))
        lines.extend(_breach_lines(run, breaches))
    return lines


def json_mspc(args: argparse.Namespace) -> dict[str, object]:
    run = _mspc_run(args)
    model = run.model
    return {
        "tags": run.tags,
        "rate_s": run.rate_s,
        "rate_source": run.rate_source,
        "baseline": {
            **_window_json(run.baseline),
            "rows": len(run.train.index),
            "coverage": run.train.coverage,
        },
        "window": {
            **_window_json(run.monitor),
            "rows": len(run.test.index),
            "coverage": run.test.coverage,
        },
        "model": {
            "components": len(model.components),
            "n_columns": len(model.columns),
            "explained_variance": list(model.explained_variance),
        },
        "limits": {
            "t2": model.t2_limit,
            "spe": model.spe_limit,
            "quantile": run.quantile,
        },
        "contributors_ranked": run.ranked,
        "t2_breaches": run.found.t2_breaches,
        "spe_breaches": run.found.spe_breaches,
        "contributors": run.found.top_contributors if run.ranked else {},
    }


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


def _compared(args: argparse.Namespace) -> CompareResult:
    return compare(args.parquet, args.before, args.after, rate_s=args.rate_s)


def _clip(text: str, width: int) -> str:
    """``text`` inside ``width`` columns, the cut marked with ``..``."""
    return text if len(text) <= width else f"{text[: width - 2]}.."


def _table(
    head: str,
    rows: Sequence[tuple[str, ...]],
    columns: Sequence[tuple[str, int, bool]],
    *,
    elastic_max: int,
) -> list[str]:
    """A header row and its rows, the first column shrunk to fit 80 columns."""
    widths = [
        max(minimum, len(header), *(len(r[i + 1]) for r in rows))
        for i, (header, minimum, _) in enumerate(columns)
    ]
    fixed = sum(2 + w for w in widths)
    wanted = max(len(head), *(len(r[0]) for r in rows))
    first = max(TAG_COLUMN_MIN, min(elastic_max, wanted, MAX_WIDTH - 3 - fixed))

    def line(cells: tuple[str, ...]) -> str:
        out = "  " + _clip(cells[0], first).ljust(first)
        for cell, w, (_, _, right) in zip(cells[1:], widths, columns, strict=True):
            out += "  " + (cell.rjust(w) if right else cell.ljust(w))
        return out.rstrip()

    return [line((head, *(c[0] for c in columns))), *(line(r) for r in rows)]


def _share(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.0f}%"


def _changed_at(change: TagChange) -> str:
    if change.changed_at is not None:
        return fmt_ts(change.changed_at)
    return "n/a" if change.changed_at_reason else "none"


def _change_row(change: TagChange) -> tuple[str, ...]:
    return (
        change.label,
        change.quality,
        "n/a" if change.level_shift_sigma is None else f"{change.level_shift_sigma:+.1f}",
        "n/a" if change.spread_ratio is None else f"x{change.spread_ratio:.1f}",
        _share(change.flagged_fraction),
        _changed_at(change),
    )


def _more_changes(hidden: Sequence[TagChange]) -> list[str]:
    """``(+N more within X sigma)``, X being the largest shift left out."""
    if not hidden:
        return []
    shifts = [abs(c.level_shift_sigma) for c in hidden if c.level_shift_sigma is not None]
    if not shifts:
        return [f"  (+{len(hidden)} more)"]
    return [f"  (+{len(hidden)} more within {max(shifts):.1f} sigma)"]


def _changed_section(result: CompareResult, top: int) -> list[str]:
    shown = result.tags[:top]
    return (
        rule("Tags that changed")
        + _table(
            "tag",
            [_change_row(c) for c in shown],
            _CHANGE_COLUMNS,
            elastic_max=TAG_COLUMN_MAX,
        )
        + _more_changes(result.tags[top:])
    )


def _pair_row(pair: PairChange) -> tuple[str, ...]:
    # Each side is clipped on its own, so a long pair keeps both names
    # rather than spending the column on the first one.
    return (
        f"{_clip(pair.left, TAG_COLUMN_MAX)} ~ {_clip(pair.right, TAG_COLUMN_MAX)}",
        f"{pair.pearson_before:.2f}",
        f"{pair.pearson_after:.2f}",
        f"{pair.delta:+.2f}",
        "n/a" if pair.lo is None or pair.hi is None else f"[{pair.lo:+.2f}, {pair.hi:+.2f}]",
    )


def _pairs_section(table: PairTable, top: int) -> list[str]:
    if table.reason is not None:
        return rule("Pairs that decoupled") + wrapped(table.reason)
    clearing = table.clearing
    shown = clearing[:top]
    lines = rule("Pairs that decoupled", PAIR_METHOD)
    # A header row over no rows states the columns of a table that has
    # none; the count below says everything there is to say.
    if shown:
        lines.extend(
            _table(
                "pair",
                [_pair_row(p) for p in shown],
                _PAIR_COLUMNS,
                elastic_max=PAIR_COLUMN_MAX,
            )
        )
    left = table.n_pairs - len(shown)
    if left:
        beyond = len(clearing) - len(shown)
        tail = (
            f"{beyond} clearing the interval"
            if beyond
            else "none clearing the interval"
        )
        lines.append(f"  (+{plural(left, 'more pair')}, {tail})")
    if table.n_undefined:
        lines.extend(
            wrapped(
                f"{plural(table.n_undefined, 'pair')} hold a tag with no spread in "
                "one period, where a correlation is undefined"
            )
        )
    if table.n_no_interval:
        lines.extend(
            wrapped(
                f"{plural(table.n_no_interval, 'pair')} carry a delta and no "
                f"interval: {NO_INTERVAL}. --json states them"
            )
        )
    return lines


def _joint_section(joint: JointStructure) -> list[str]:
    if joint.reason is not None:
        return rule("Joint structure") + wrapped(joint.reason)
    body = [
        f"components {joint.n_components} of {joint.n_aligned}{SEP}"
        f"explained {fmt_num(joint.explained_before)} -> "
        f"{fmt_num(joint.explained_after)} on after",
        f"rows {joint.rows}{SEP}T2 breaches {joint.t2_breaches}{SEP}"
        f"SPE breaches {joint.spe_breaches}",
    ]
    named = SEP.join(f"{name} {_share(share)}" for name, share in joint.contributors)
    if joint.other_share is not None:
        named += f"{SEP}other {_share(joint.other_share)}"
    header = rule(
        "Joint structure",
        f"(PCA fitted on before, {joint.n_aligned} of {joint.n_offered} tags aligned)",
    )
    lines = header + indent(body)
    if named:
        lines.extend(wrapped(f"SPE contributors{SEP}{named}"))
    else:
        lines.extend(indent(["SPE contributors   none (no SPE breach)"]))
    return lines


def _compare_headline(result: CompareResult) -> str:
    table = result.pairs
    pairs = (
        "pairs refused"
        if table.reason is not None
        else f"pairs {len(table.clearing)} of {table.n_pairs} clearing"
    )
    counts = f"refused {result.n_refused}{SEP}{pairs}"
    tags = plural(len(result.tags), "tag")
    if result.source_id is None:
        return f"{tags}  {counts}"
    return f"{result.source_id}  {tags}{SEP}{counts}"


def _period_line(label: str, span: tuple[pd.Timestamp, pd.Timestamp]) -> str:
    start, end = span
    return label_line(
        label,
        f"{fmt_span(start, end)}{SEP}({fmt_duration((end - start).total_seconds())})",
    )


def run_compare(args: argparse.Namespace) -> list[str]:
    """Report what changed between two periods of one unit."""
    result = _compared(args)
    table = result.pairs
    lines = [
        _compare_headline(result),
        "",
        _period_line("before", result.before),
        _period_line("after", result.after),
    ]
    if table.rate_s is not None and table.coverage_before is not None:
        lines.append(
            label_line(
                "grid",
                f"rate {table.rate_s} s ({table.rate_source}){SEP}"
                f"coverage {fmt_num(table.coverage_before)} -> "
                f"{fmt_num(table.coverage_after)}",
            )
        )
    lines.extend(_changed_section(result, args.top))
    lines.extend(_pairs_section(table, args.top))
    lines.extend(_joint_section(result.joint))
    return lines


def _span_json(span: tuple[pd.Timestamp, pd.Timestamp]) -> dict[str, object]:
    start, end = span
    return {
        "start": start,
        "end": end,
        "duration_s": (end - start).total_seconds(),
    }


def json_compare(args: argparse.Namespace) -> dict[str, object]:
    result = _compared(args)
    table = result.pairs
    joint = result.joint
    return {
        "before": _span_json(result.before),
        "after": _span_json(result.after),
        "source_id": result.source_id,
        "grid": {
            "rate_s": table.rate_s,
            "rate_source": table.rate_source,
            "coverage_before": table.coverage_before,
            "coverage_after": table.coverage_after,
        },
        "tags": [
            {
                "tag": c.tag,
                "quality": c.quality,
                "refused": c.refused,
                "level_shift_sigma": c.level_shift_sigma,
                "level_shift_reason": c.level_shift_reason,
                "spread_ratio": c.spread_ratio,
                "spread_reason": c.spread_reason,
                "flagged_fraction": c.flagged_fraction,
                "flagged_reason": c.flagged_reason,
                "changed_at": c.changed_at,
                "changed_at_reason": c.changed_at_reason,
            }
            for c in result.tags
        ],
        "pairs": (
            None
            if table.reason is not None
            else [
                {
                    "left": p.left,
                    "right": p.right,
                    "pearson_before": p.pearson_before,
                    "pearson_after": p.pearson_after,
                    "delta": p.delta,
                    "interval": None if p.lo is None else [p.lo, p.hi],
                    "interval_reason": None if p.lo is not None else NO_INTERVAL,
                    "clears": p.clears,
                    "spearman_before": p.spearman_before,
                    "spearman_after": p.spearman_after,
                    "spearman_delta": p.spearman_delta,
                }
                for p in table.pairs
            ]
        ),
        "pairs_reason": table.reason,
        "pairs_undefined": table.n_undefined,
        "pairs_without_interval": table.n_no_interval,
        "joint": (
            None
            if joint.reason is not None
            else {
                "tags_offered": joint.n_offered,
                "tags_aligned": joint.n_aligned,
                "components": joint.n_components,
                "rows": joint.rows,
                "explained_before": joint.explained_before,
                "explained_after": joint.explained_after,
                "t2_breaches": joint.t2_breaches,
                "spe_breaches": joint.spe_breaches,
                "spe_contributors": [
                    {"tag": name, "share": share} for name, share in joint.contributors
                ],
                "spe_other_share": joint.other_share,
            }
        ),
        "joint_reason": joint.reason,
    }


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
    # render() joins its own lines and carries no trailing newline, so
    # splitting it back is exact.
    return _profiled(args).render().splitlines()


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
        "window": _window_json(window),
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

"""Deterministic rendering of a window's data-physics report.

Shared by the CLI and the reference cases so both can never disagree.
All numbers are fixed-precision; ordering is stable; unpopulatable
values render as ``null``, never as ``1.0`` or ``0`` placeholders.

Layout: line 1 is the headline, then a block of ``label     value``
lines, then one section per group of findings. Every line fits in 80
columns and carries no escape codes; :mod:`tsdive.ui.term` adds
colour on the way to a terminal.
"""

from __future__ import annotations

import math
import textwrap
from collections.abc import Iterable

import pandas as pd

from tsdive.detectors.flatline import FlatlineVerdict
from tsdive.features.window_features import MAX_STATES_LISTED, WindowStats
from tsdive.store.gaps import DATA_LOSS_CLASSES, GapFinding
from tsdive.store.quality import Severity
from tsdive.store.tagstore import Window
from tsdive.ui.term import rule

SIGNIFICANT_DIGITS = 4

# Labels sit in the first 10 columns; two spaces separate the longest one
# from its value, so every value in a block starts in the same column.
LABEL_WIDTH = 8

# Between the fields of one line. Wide enough that `a 1   b 2` reads as
# two pairs rather than four words.
SEP = "   "


def _fmt(v: float | None, nd: int = 3) -> str:
    return "null" if v is None else f"{v:.{nd}f}"


def yes_no(flag: bool) -> str:
    """``yes``/``no``: a verdict, not a Python repr."""
    return "yes" if flag else "no"


def plural(n: int, noun: str) -> str:
    """``6 segments`` / ``1 segment``: a count and a noun that agree."""
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def label_line(label: str, value: str) -> str:
    """``label`` in the first 10 columns, then ``value``."""
    return f"{label:<{LABEL_WIDTH}}  {value}"


def continued(value: str) -> str:
    """A further line under a label, aligned with the first one's value."""
    return f"{'':<{LABEL_WIDTH}}  {value}"


def indent(lines: Iterable[str]) -> list[str]:
    """Section content, two columns in from its header."""
    return [f"  {line}" for line in lines]


def more_line(n: int) -> list[str]:
    """``(+N more)`` for a list that was cut short, or nothing."""
    return [f"  (+{n} more)"] if n > 0 else []


# Every line is written for a terminal 80 columns wide. Folded text stops
# two columns short so a wrapped line never reaches the edge.
MAX_WIDTH = 80
WRAP_WIDTH = 78


def fits(text: str) -> bool:
    """True when the line is inside the 80-column budget."""
    return len(text) < MAX_WIDTH


def wrapped(text: str, *, width: int = WRAP_WIDTH) -> list[str]:
    """Section content folded to ``width``, continuations at the same indent."""
    return textwrap.wrap(
        text,
        width=width,
        initial_indent="  ",
        subsequent_indent="  ",
        break_long_words=False,
        break_on_hyphens=False,
    ) or ["  "]


def fmt_num(v: float | None, digits: int = SIGNIFICANT_DIGITS) -> str:
    """``digits`` significant figures in fixed point; ``n/a`` for None.

    Fixed point on purpose: an operator reading a flow rate should not
    have to decode ``6.230e+01``.
    """
    if v is None:
        return "n/a"
    f = float(v)
    if not math.isfinite(f):
        # log10 cannot pick a precision for these; print them as spelled.
        return str(f)
    if f == 0.0:
        return "0"
    decimals = min(max(digits - 1 - math.floor(math.log10(abs(f))), 0), 12)
    return f"{f:.{decimals}f}"


def fmt_seconds(v: float | None) -> str:
    """Seconds, integral when whole. ``n/a`` carries no unit."""
    if v is None:
        return "n/a"
    f = float(v)
    if not math.isfinite(f):
        return f"{f} s"
    return f"{int(f)} s" if f.is_integer() else f"{fmt_num(f)} s"


def fmt_ts(value: object) -> str:
    """One instant as ``2024-03-30 20:00:00Z``: UTC, space, trailing Z.

    A sub-second part is kept, with its trailing zeros dropped; a naive
    timestamp is read as UTC, which is the only zone an archive stores.
    """
    stamp = pd.Timestamp(value)  # type: ignore[arg-type]
    if stamp.tz is None:
        stamp = stamp.tz_localize("UTC")
    stamp = stamp.tz_convert("UTC")
    text = stamp.strftime("%Y-%m-%d %H:%M:%S")
    fraction = stamp.strftime("%f").rstrip("0")
    return f"{text}.{fraction}Z" if fraction else f"{text}Z"


def fmt_span(start: object, end: object) -> str:
    """``start -> end``; the end drops its date when it repeats the start's."""
    left, right = fmt_ts(start), fmt_ts(end)
    if left[:10] == right[:10]:
        right = right[11:]
    return f"{left} -> {right}"


def fmt_duration(seconds: float | None) -> str:
    """A span in the largest unit that divides it whole: ``10 h``, ``40 min``.

    Falls back to seconds, with one decimal when the count is fractional.
    ``n/a`` carries no unit.
    """
    if seconds is None:
        return "n/a"
    value = float(seconds)
    if not math.isfinite(value):
        return f"{value} s"
    for size, unit in ((86400.0, "d"), (3600.0, "h"), (60.0, "min")):
        if abs(value) >= size and value % size == 0:
            return f"{int(value / size)} {unit}"
    return f"{int(value)} s" if value.is_integer() else f"{value:.1f} s"


def render_gap(gap: GapFinding) -> list[str]:
    """One gap: when it ran, how long, and the class with the rule that named it."""
    return [
        f"{fmt_span(gap.start, gap.end)}{SEP}{fmt_duration(gap.duration_s)}"
        f"{SEP}{gap.classification.cls.value} ({gap.classification.rule})"
    ]


def _flatline_token(verdict: FlatlineVerdict) -> str:
    """The headline word for a verdict, whose section carries the rest."""
    if verdict.saturated_not_frozen or verdict.not_assessed is not None:
        return "NOT ASSESSED"
    return "SUSPECTED" if verdict.fired else "none"


def render_stats(window: Window, stats: WindowStats) -> list[str]:
    """The Values section: distribution, movement, and the clock."""
    f = stats.features
    body: list[str] = []
    if f.n_good == 0:
        head = "none GOOD"
    elif stats.state_valued:
        head = f"states  GOOD n={f.n_good}"
        shown = stats.state_counts[:MAX_STATES_LISTED]
        line = SEP.join(f"{s} {n}" for s, n in shown)
        if len(stats.state_counts) > MAX_STATES_LISTED:
            line += f"{SEP}+{len(stats.state_counts) - MAX_STATES_LISTED} more"
        body.append(line)
    else:
        head = f"GOOD n={f.n_good}"
        basis = window.contract.calculation_basis.value.lower().replace("_", "-")
        mean = stats.mean_note or f"mean {fmt_num(f.weighted_mean)} ({basis})"
        body.append(
            SEP.join(
                (
                    f"min {fmt_num(f.min)}",
                    f"p05 {fmt_num(stats.p05)}",
                    f"median {fmt_num(f.median)}",
                    f"p95 {fmt_num(stats.p95)}",
                    f"max {fmt_num(f.max)}",
                )
            )
        )
        body.append(
            SEP.join((mean, f"std {fmt_num(f.std)}", f"mad {fmt_num(stats.mad)}"))
        )
    if f.n_good > 0:
        body.append(
            SEP.join(
                (
                    f"distinct {stats.distinct_count}",
                    f"stall {fmt_seconds(f.stall_s)}",
                    f"changes/h {fmt_num(f.changes_per_hour)}",
                )
            )
        )
    declared = (
        "none" if stats.declared_rate_s is None else fmt_seconds(stats.declared_rate_s)
    )
    interval = (
        f"interval {fmt_seconds(stats.interval_median_s)} "
        f"(p05 {fmt_seconds(stats.interval_p05_s)}, "
        f"p95 {fmt_seconds(stats.interval_p95_s)}){SEP}declared {declared}"
    )
    if stats.interval_differs_from_declared:
        interval += f"{SEP}(differs from declared)"
    body.append(interval)
    return rule("Values", head) + indent(body)


def _headline(window: Window, flatline: FlatlineVerdict | None) -> list[str]:
    p = window.physics
    parts = [
        f"coverage {_fmt(p.coverage.coverage)}",
        f"GOOD {int(window.frame['valid'].sum())}/{len(window.frame)}",
        f"censored {yes_no(p.clipping.censored)}",
        f"gaps {p.coverage.n_gaps}",
    ]
    if flatline is not None:
        parts.append(f"flatline {_flatline_token(flatline)}")
    return [f"{window.identity}  {window.meta.name}", SEP.join(parts)]


def _identity_block(window: Window) -> list[str]:
    c = window.contract
    span = (window.end - window.start).total_seconds()
    lines = [
        label_line(
            "window", f"{fmt_span(window.start, window.end)}  ({fmt_duration(span)})"
        ),
        label_line(
            "contract",
            "  ".join(
                (
                    c.calculation_basis.value,
                    c.retrieval_mode.value,
                    c.aggregate_type.value,
                    f"stepped {yes_no(c.stepped)}",
                    f"digest {c.digest()[:12]}",
                )
            ),
        ),
    ]
    u = window.physics.unit
    if not u.resolved:
        lines.append(
            label_line("units", f"{u.raw!r} unresolved (null); comparison refused")
        )
        return lines
    lines.append(label_line("units", f"{u.raw} -> {u.canonical}"))
    if u.reference_condition:
        lines.append(
            continued("reference-condition: declare state before cross-tag comparison")
        )
    return lines


def _coverage_section(window: Window) -> list[str]:
    p = window.physics
    cov = p.coverage
    n_data_loss = sum(1 for g in cov.gaps if g.classification.cls in DATA_LOSS_CLASSES)
    body = [
        SEP.join(
            (
                f"coverage {_fmt(cov.coverage)}",
                f"valid {_fmt(p.valid_fraction)}",
                f"gaps {cov.n_gaps}",
                f"data-loss gaps {n_data_loss}",
                f"longest {fmt_duration(cov.longest_gap_s)}",
            )
        )
    ]
    for gap in cov.gaps:
        body.extend(render_gap(gap))
    if p.edge_slack_s > 0:
        body.append(
            f"edge slack {fmt_duration(p.edge_slack_s)} "
            f"(start {fmt_duration(p.edge_slack_start_s)}, "
            f"end {fmt_duration(p.edge_slack_end_s)}; "
            "under the gap threshold, covered)"
        )
    return rule("Coverage") + indent(body)


def _quality_section(window: Window) -> list[str]:
    p = window.physics
    counts = p.severity_counts
    body = [SEP.join(f"{s.value} {counts.get(s, 0)}" for s in Severity)]
    if window.meta.quality_assumed:
        body.append("quality ASSUMED at ingest: a caller's declaration, not observed")
    if p.unmapped_quality_codes:
        body.append(
            "unmapped codes, treated UNCERTAIN: "
            + ", ".join(p.unmapped_quality_codes)
        )
        if p.unmapped_look_like_opc_da:
            body.append(
                "byte-sized codes: OPC DA export? declare quality_codes in tag meta"
            )
    return rule("Quality") + indent(body)


def _range_section(window: Window) -> list[str]:
    p = window.physics
    clipped = (
        "clipped null (eng range unknown)"
        if p.clipping.fraction is None
        else f"clipped {p.clipping.fraction:.4f}"
    )
    body = [f"{clipped}{SEP}censored {yes_no(p.clipping.censored)}"]
    if p.implausible_magnitude_count > 0:
        why = (
            "not censored: eng range unknown"
            if window.meta.eng_range is None
            else "eng range declared; see clipped above"
        )
        body.append(f"beyond float32 range {p.implausible_magnitude_count} ({why})")
    return rule("Range") + indent(body)


def _timestamp_section(window: Window) -> list[str]:
    ta = window.physics.timestamp_audit
    body = [
        SEP.join(
            (
                f"audited {ta.n_samples}",
                f"duplicates {len(ta.duplicate_timestamps)}",
                f"non-monotonic {len(ta.non_monotonic_positions)}",
            )
        )
    ]
    body.extend(
        f"DST ({t.tz}) {fmt_ts(t.instant_utc)}  "
        f"{t.offset_before_s / 3600:+.0f}h -> {t.offset_after_s / 3600:+.0f}h"
        for t in ta.dst_transitions
    )
    return rule("Timestamps") + indent(body)


def _flatline_section(verdict: FlatlineVerdict) -> list[str]:
    if verdict.saturated_not_frozen:
        body = ["NOT ASSESSED (window censored; saturated != frozen)"]
    elif verdict.not_assessed is not None:
        body = [f"NOT ASSESSED ({verdict.not_assessed})"]
    else:
        body = ["FLATLINE SUSPECTED" if verdict.fired else "no flatline"]
        body.extend(
            f"[{'FIRED' if s.fired else 'ok   '}] {s.signal}: {s.evidence}"
            for s in verdict.signals
        )
    return rule("Flatline") + indent(body)


def render_window_report(
    window: Window,
    flatline: FlatlineVerdict | None = None,
    stats: WindowStats | None = None,
) -> str:
    """Render the full data-physics report for one window."""
    lines = _headline(window, flatline)
    lines.append("")
    lines.extend(_identity_block(window))
    lines.extend(_coverage_section(window))
    lines.extend(_quality_section(window))
    lines.extend(_range_section(window))
    lines.extend(_timestamp_section(window))
    if stats is not None:
        lines.extend(render_stats(window, stats))
    if flatline is not None:
        lines.extend(_flatline_section(flatline))
    return "\n".join(lines)

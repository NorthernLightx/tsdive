"""Text lines and JSON payloads for the results in :mod:`tsdive.analyses`.

Each ``X_lines`` function returns the plain text ``tsdive X`` prints, one
line per element, and each ``X_json`` the object ``tsdive X --json``
prints before serialisation. Colour is applied by the CLI and nowhere
else, so every renderer here stays plain text.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import pandas as pd

from tsdive.changepoints import DEFAULT_PENALTY_MULTIPLIER
from tsdive.changepoints.pelt import Segmentation
from tsdive.compare import (
    NO_INTERVAL,
    CompareResult,
    JointStructure,
    PairChange,
    PairTable,
    TagChange,
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
    rule,
    wrapped,
    yes_no,
)
from tsdive.store.tagstore import Window

if TYPE_CHECKING:
    from tsdive.analyses import (
        CompareAnalysis,
        MspcAnalysis,
        ScreenAnalysis,
        SegmentAnalysis,
        SpcAnalysis,
    )


def render_lines(text: str) -> list[str]:
    """A rendered report back as the lines it was joined from.

    ``render()`` joins its lines with newlines and carries no trailing
    one, so splitting on newlines is exact.
    """
    return text.split("\n")


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


def _n_good(window: Window) -> int:
    """Rows the window read judged trustworthy *and* usable."""
    return int(window.frame["valid"].sum())


def window_json(window: Window) -> dict[str, object]:
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
        **window_json(window),
        "n_good": _n_good(window),
        "censored": window.physics.clipping.censored,
    }


# --- screen -----------------------------------------------------------------


def _flagged_share(a: ScreenAnalysis) -> str:
    result = a.result
    if not result.n_screened:
        return "n/a"
    return f"{100.0 * result.n_flagged / result.n_screened:.1f}%"


def screen_lines(a: ScreenAnalysis) -> list[str]:
    """The ``tsdive screen`` report for one analysis."""
    result = a.result
    lines = [
        f"{a.baseline.identity}  flagged {result.n_flagged} of "
        f"{result.n_screened} ({_flagged_share(a)})",
        "",
        _baseline_line(a.baseline),
        _window_line(a.monitor),
    ]
    if a.mode_path is not None:
        lines.append(label_line("mode", a.mode_path))
    if a.alignment is not None:
        base_a, monitor_a = a.alignment
        lines.append(
            label_line(
                "alignment",
                f"baseline {base_a.kept}/{base_a.total}{SEP}"
                f"window {monitor_a.kept}/{monitor_a.total}",
            )
        )
    if a.regimes is not None:
        lines.append(label_line("method", f"REGIME_MAD{SEP}k {a.k}"))
        lines.extend(
            rule("Regimes")
            + indent(
                f"regime {name}{SEP}center {fmt_num(r.center)}{SEP}"
                f"scale {fmt_num(r.scale)}{SEP}n {r.n_good}"
                for name, r in a.regimes.items()
            )
        )
    else:
        base = a.provisional
        assert base is not None
        lo, hi = base.limits(a.k)
        lines.append(
            label_line(
                "method",
                f"{base.kind}{SEP}center {fmt_num(base.center)}{SEP}"
                f"scale {fmt_num(base.scale)}{SEP}k {a.k}{SEP}"
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


def screen_json(a: ScreenAnalysis) -> dict[str, object]:
    """The ``tsdive screen --json`` payload for one analysis."""
    result = a.result
    base = a.provisional
    return {
        "tag": str(a.baseline.identity),
        "mode": a.mode_path,
        "baseline": _baseline_json(a.baseline),
        "window": window_json(a.monitor),
        "method": result.kind,
        "k": a.k,
        "center": None if base is None else base.center,
        "scale": None if base is None else base.scale,
        "limits": None if base is None else list(base.limits(a.k)),
        "caveat": None if base is None else base.caveat,
        "regimes": (
            None
            if a.regimes is None
            else [
                {
                    "regime": name,
                    "center": r.center,
                    "scale": r.scale,
                    "n_good": r.n_good,
                    "method": r.method,
                }
                for name, r in a.regimes.items()
            ]
        ),
        "alignment": (
            None
            if a.alignment is None
            else {
                "baseline": {
                    "kept": a.alignment[0].kept,
                    "total": a.alignment[0].total,
                },
                "window": {
                    "kept": a.alignment[1].kept,
                    "total": a.alignment[1].total,
                },
            }
        ),
        "n_screened": result.n_screened,
        "n_flagged": result.n_flagged,
        "flagged": sorted(result.flagged_timestamps),
    }


# --- segment ----------------------------------------------------------------

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


def segment_lines(a: SegmentAnalysis) -> list[str]:
    """The ``tsdive segment`` report for one analysis."""
    window, found = a.window, a.found
    default_note = (
        f" ({DEFAULT_PENALTY_MULTIPLIER:g}*log n)" if a.penalty_is_default else ""
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


def segment_json(a: SegmentAnalysis) -> dict[str, object]:
    """The ``tsdive segment --json`` payload for one analysis."""
    found = a.found
    return {
        "tag": str(a.window.identity),
        "window": window_json(a.window),
        "censored": a.window.physics.clipping.censored,
        "usable": found.n_used,
        "method": found.method,
        "penalty": found.penalty,
        "penalty_is_default": a.penalty_is_default,
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


# --- spc --------------------------------------------------------------------


def spc_lines(a: SpcAnalysis) -> list[str]:
    """The ``tsdive spc`` report for one analysis."""
    limits = a.limits
    lines = [
        f"{a.baseline.identity}  {plural(len(a.hits), 'rule hit')} in "
        f"{plural(a.n_monitored, 'sample')}",
        "",
        _baseline_line(a.baseline),
        _window_line(a.monitor),
        label_line(
            "limits",
            f"center {fmt_num(limits.center)}{SEP}sigma {fmt_num(a.sigma)}{SEP}"
            f"lcl {fmt_num(limits.lcl)}{SEP}ucl {fmt_num(limits.ucl)}",
        ),
        label_line("basis", limits.basis),
    ]
    for name in SPC_RULES:
        fired = [h for h in a.hits if h.rule == name]
        lines.extend(rule(name, str(len(fired))))
        lines.extend(
            indent(f"{fmt_ts(h.timestamp)}{SEP}{h.detail}" for h in fired[:HITS_SHOWN])
        )
        lines.extend(more_line(len(fired) - HITS_SHOWN))
    return lines


def spc_json(a: SpcAnalysis) -> dict[str, object]:
    """The ``tsdive spc --json`` payload for one analysis."""
    return {
        "tag": str(a.baseline.identity),
        "baseline": _baseline_json(a.baseline),
        "window": window_json(a.monitor),
        "limits": {
            "center": a.limits.center,
            "sigma": a.sigma,
            "lcl": a.limits.lcl,
            "ucl": a.limits.ucl,
            "basis": a.limits.basis,
        },
        "n_monitored": a.n_monitored,
        "n_hits": len(a.hits),
        "rules": [
            {
                "rule": name,
                "n": sum(1 for h in a.hits if h.rule == name),
                "hits": [
                    {"timestamp": h.timestamp, "detail": h.detail}
                    for h in a.hits
                    if h.rule == name
                ],
            }
            for name in SPC_RULES
        ],
    }


# --- mspc -------------------------------------------------------------------


def _mspc_headline(a: MspcAnalysis) -> tuple[str, list[str]]:
    """The headline, and the ``tags`` lines it needs when the list will not fit."""
    found = a.found
    counts = (
        f"T2 breaches {len(found.t2_breaches)}{SEP}"
        f"SPE breaches {len(found.spe_breaches)}{SEP}"
        f"of {plural(len(a.test.index), 'row')}"
    )
    listed = ", ".join(a.tags)
    if fits(f"{listed}  {counts}"):
        return f"{listed}  {counts}", []
    headline = f"{plural(len(a.tags), 'tag')}  {counts}"
    if fits(label_line("tags", listed)):
        return headline, [label_line("tags", listed)]
    return headline, [
        label_line("tags", a.tags[0]),
        *(continued(tag) for tag in a.tags[1:]),
    ]


def _breach_lines(a: MspcAnalysis, stamps: list[pd.Timestamp]) -> list[str]:
    """Up to HITS_SHOWN breach timestamps, with contributors only when they rank."""
    lines: list[str] = []
    for ts in stamps[:HITS_SHOWN]:
        text = fmt_ts(ts)
        if a.ranked:
            named = ", ".join(a.found.top_contributors[ts.isoformat()])
            text += f"{SEP}contributors {named}"
        lines.extend(wrapped(text))
    return lines + more_line(len(stamps) - HITS_SHOWN)


def mspc_lines(a: MspcAnalysis) -> list[str]:
    """The ``tsdive mspc`` report for one analysis."""
    model = a.model
    headline, tag_lines = _mspc_headline(a)
    lines = [headline, "", *tag_lines]
    lines.extend(
        [
            label_line(
                "baseline",
                f"{fmt_span(a.baseline.start, a.baseline.end)}{SEP}"
                f"rows {len(a.train.index)}{SEP}"
                f"coverage {fmt_num(a.train.coverage)}",
            ),
            label_line(
                "window",
                f"{fmt_span(a.monitor.start, a.monitor.end)}{SEP}"
                f"rows {len(a.test.index)}{SEP}"
                f"coverage {fmt_num(a.test.coverage)}",
            ),
            label_line(
                "model",
                f"rate {a.rate_s} s ({a.rate_source}){SEP}"
                f"components {len(model.components)} of {len(model.columns)}{SEP}"
                f"explained {', '.join(fmt_num(v) for v in model.explained_variance)}",
            ),
            label_line(
                "limits",
                f"T2 {fmt_num(model.t2_limit)}{SEP}SPE {fmt_num(model.spe_limit)}"
                f"{SEP}(empirical q{a.quantile})",
            ),
        ]
    )
    if not a.ranked:
        lines.append(
            label_line(
                "contributors",
                f"not ranked ({plural(len(model.columns), 'tag')}; "
                f"top-{CONTRIBUTORS_KEPT} would list every one)",
            )
        )
    for name, breaches in (
        ("T2 breaches", a.found.t2_breaches),
        ("SPE breaches", a.found.spe_breaches),
    ):
        lines.extend(rule(name, str(len(breaches))))
        lines.extend(_breach_lines(a, breaches))
    return lines


def mspc_json(a: MspcAnalysis) -> dict[str, object]:
    """The ``tsdive mspc --json`` payload for one analysis."""
    model = a.model
    return {
        "tags": a.tags,
        "rate_s": a.rate_s,
        "rate_source": a.rate_source,
        "baseline": {
            **window_json(a.baseline),
            "rows": len(a.train.index),
            "coverage": a.train.coverage,
        },
        "window": {
            **window_json(a.monitor),
            "rows": len(a.test.index),
            "coverage": a.test.coverage,
        },
        "model": {
            "components": len(model.components),
            "n_columns": len(model.columns),
            "explained_variance": list(model.explained_variance),
        },
        "limits": {
            "t2": model.t2_limit,
            "spe": model.spe_limit,
            "quantile": a.quantile,
        },
        "contributors_ranked": a.ranked,
        "t2_breaches": a.found.t2_breaches,
        "spe_breaches": a.found.spe_breaches,
        "contributors": a.found.top_contributors if a.ranked else {},
    }


# --- compare ----------------------------------------------------------------


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


def compare_lines(a: CompareAnalysis) -> list[str]:
    """The ``tsdive compare`` report for one analysis."""
    table = a.pairs
    lines = [
        _compare_headline(a),
        "",
        _period_line("before", a.before),
        _period_line("after", a.after),
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
    lines.extend(_changed_section(a, a.top))
    lines.extend(_pairs_section(table, a.top))
    lines.extend(_joint_section(a.joint))
    return lines


def _span_json(span: tuple[pd.Timestamp, pd.Timestamp]) -> dict[str, object]:
    start, end = span
    return {
        "start": start,
        "end": end,
        "duration_s": (end - start).total_seconds(),
    }


def compare_json(a: CompareAnalysis) -> dict[str, object]:
    """The ``tsdive compare --json`` payload for one analysis."""
    table = a.pairs
    joint = a.joint
    return {
        "before": _span_json(a.before),
        "after": _span_json(a.after),
        "source_id": a.source_id,
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
            for c in a.tags
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

"""Text lines and JSON payloads for switchback plans and analyses.

``plan_lines`` and ``analysis_lines`` return the plain text the two
``tsdive switchback`` commands print, ``analysis_json`` the object
``tsdive switchback analyze --json`` prints. The plan's own document is
:func:`tsdive.switchback.plan.plan_to_dict`. Colour is applied by the CLI
and nowhere else.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from tsdive.report import (
    SEP,
    fmt_duration,
    fmt_num,
    fmt_span,
    fmt_ts,
    indent,
    label_line,
    more_line,
    rule,
    wrapped,
    yes_no_unknown,
)
from tsdive.switchback.design import PERMUTATIONS, order_of_magnitude
from tsdive.switchback.plan import POWER_TARGET, SETTING_A, SETTING_B, json_count
from tsdive.ui.jsonout import to_jsonable

if TYPE_CHECKING:
    from tsdive.switchback.archive import SwitchbackAnalysis, SwitchbackEstimate
    from tsdive.switchback.plan import PowerReadout, SwitchbackPlan

# Enough blocks to read the first hours of a schedule off the terminal;
# the plan file lists every block.
BLOCKS_SHOWN = 24

DIGEST_SHOWN = 12


def _signed(value: float) -> str:
    return f"+{fmt_num(value)}" if value >= 0 else fmt_num(value)


def _p(value: float) -> str:
    return fmt_num(value, 3)


def _with_unit(text: str, unit: str | None) -> str:
    return f"{text} {unit}" if unit else text


# Counts at or above this print as a power of ten.
COUNT_PRINTED = 10**6


def _design_line(plan: SwitchbackPlan) -> str:
    n = plan.n_assignments
    count = str(n) if n < COUNT_PRINTED else f"over 10^{order_of_magnitude(n)}"
    reference = "enumerated" if plan.enumerated else f"{PERMUTATIONS} drawn"
    return label_line(
        "design",
        f"{count} balanced assignments{SEP}{reference}{SEP}smallest p {_p(plan.min_p)}",
    )


def _washout(plan: SwitchbackPlan) -> str:
    return "none" if plan.washout_s == 0 else fmt_duration(plan.washout_s)


def _counts(plan: SwitchbackPlan) -> str:
    n_b = sum(1 for b in plan.blocks if b.setting == SETTING_B)
    return f"{SETTING_A} {plan.k - n_b}{SEP}{SETTING_B} {n_b}"


# ---------------------------------------------------------------- plan


def _power_lines(power: PowerReadout) -> list[str]:
    header = rule(
        "Power",
        f"({power.draws} schedules laid over the history, shift added in B blocks)",
    )
    head = [
        label_line("history", f"{power.tag}{SEP}{fmt_span(power.start, power.end)}"),
    ]
    if power.reason is not None:
        return header + indent(head) + wrapped(f"refused   {power.reason}: {power.detail}")
    sigma = power.sigma if power.sigma is not None else math.nan
    head.append(label_line("sigma", f"{_with_unit(fmt_num(sigma), power.unit)} (1.4826 MAD)"))
    width = max(len(f"{d:g}") for d in power.deltas) + 3
    head.append(
        label_line("shift", "".join(f"{d:g}".ljust(width) for d in power.deltas) + "sigma")
    )
    head.append(
        label_line("claimed", "".join(f"{r:.3f}".ljust(width) for r in power.rates).rstrip())
    )
    if power.smallest is None:
        smallest = "none on the grid"
    else:
        smallest = (
            f"{power.smallest:g} sigma "
            f"({_with_unit(fmt_num(power.smallest * sigma), power.unit)})"
        )
    head.append(label_line("smallest", f"{smallest} with detection >= {POWER_TARGET:g}"))
    return header + indent(head)


def plan_lines(plan: SwitchbackPlan, wrote: str | None = None) -> list[str]:
    """The ``tsdive switchback plan`` report for one plan; ``wrote`` names the plan file."""
    lines = [
        f"switchback plan{SEP}{plan.k} blocks of {fmt_duration(plan.block_s)}{SEP}"
        f"{_counts(plan)}{SEP}digest {plan.digest[:DIGEST_SHOWN]}",
        "",
        label_line(
            "window",
            f"{fmt_span(plan.start, plan.end)}{SEP}"
            f"({fmt_duration((plan.end - plan.start).total_seconds())})",
        ),
        label_line(
            "schedule",
            f"{fmt_span(plan.start, plan.schedule_end)}{SEP}seed {plan.seed}",
        ),
        label_line(
            "washout",
            "none" if plan.washout_s == 0 else f"{_washout(plan)} at the start of every block",
        ),
        _design_line(plan),
    ]
    if wrote is not None:
        lines.append(label_line("wrote", wrote))
    lines.extend(rule("Schedule", "(the plan file lists every block)"))
    lines.extend(
        indent(f"{b.index:>4}  {fmt_ts(b.start)}  {b.setting}" for b in plan.blocks[:BLOCKS_SHOWN])
    )
    lines.extend(more_line(plan.k - BLOCKS_SHOWN))
    if plan.power is not None:
        lines.extend(_power_lines(plan.power))
    return lines


# ---------------------------------------------------------------- analysis


def _interval(est: SwitchbackEstimate) -> str:
    lo = "unbounded" if est.lo_unbounded else _signed(est.lo or 0.0)
    hi = "unbounded" if est.hi_unbounded else _signed(est.hi or 0.0)
    return f"[{lo}, {hi}]"


def _kept_line(est: SwitchbackEstimate, plan: SwitchbackPlan) -> str:
    counts = est.kept_per_block
    n_a = sum(c for c, b in zip(counts, plan.blocks, strict=True) if b.setting == SETTING_A)
    n_b = sum(counts) - n_a
    return label_line(
        "kept",
        f"{SETTING_A} {n_a}{SEP}{SETTING_B} {n_b}{SEP}per block {min(counts)} to {max(counts)}",
    )


def _estimate_lines(
    title: str, note: str, est: SwitchbackEstimate, a: SwitchbackAnalysis
) -> list[str]:
    lines = rule(title, note)
    if est.covariates:
        lines.extend(wrapped(", ".join(est.covariates)))
    if est.refused:
        return (
            lines
            + wrapped(f"refused   {est.reason}: {est.detail}")
            + indent([_kept_line(est, a.plan)])
        )
    body = [
        label_line(
            "estimate",
            f"{_with_unit(_signed(est.estimate or 0.0), a.unit)}{SEP}p {_p(est.p_value or 1.0)}",
        ),
        label_line("95%", _interval(est)),
        _kept_line(est, a.plan),
    ]
    return lines + indent(body)


def _headline(a: SwitchbackAnalysis) -> str:
    est = a.direct
    if est.refused:
        return f"{a.tag}{SEP}refused {est.reason}"
    return (
        f"{a.tag}{SEP}B - A {_with_unit(_signed(est.estimate or 0.0), a.unit)}{SEP}"
        f"p {_p(est.p_value or 1.0)}"
    )


def _units_value(a: SwitchbackAnalysis) -> str:
    unit = a.target.physics.unit
    if unit.resolved:
        return f"{unit.raw} -> {unit.canonical}"
    if unit.raw:
        return f"{unit.raw!r} unresolved"
    return "none declared"


def analysis_lines(a: SwitchbackAnalysis) -> list[str]:
    """The ``tsdive switchback analyze`` report for one analysis."""
    plan = a.plan
    good = a.good_share
    lines = [
        _headline(a),
        "",
        label_line("plan", f"digest {plan.digest[:DIGEST_SHOWN]} verified{SEP}seed {plan.seed}"),
        label_line(
            "schedule",
            f"{fmt_span(plan.start, plan.schedule_end)}{SEP}"
            f"({fmt_duration((plan.schedule_end - plan.start).total_seconds())})",
        ),
        label_line(
            "blocks",
            f"{plan.k} of {fmt_duration(plan.block_s)}{SEP}{_counts(plan)}{SEP}"
            f"washout {_washout(plan)}",
        ),
        _design_line(plan),
        label_line("units", _units_value(a)),
        label_line(
            "quality",
            f"GOOD {'n/a' if good is None else f'{good:.3f}'}{SEP}"
            f"censored {yes_no_unknown(a.censored)}",
        ),
    ]
    if a.unused:
        lines.append(label_line("unused", ", ".join(a.unused)))
    lines.extend(
        _estimate_lines("Difference in means", "(B - A over the kept samples)", a.direct, a)
    )
    if a.adjusted is not None:
        n_cov = len(a.adjusted.covariates)
        note = f"(OLS on {n_cov} covariate{'' if n_cov == 1 else 's'})"
        lines.extend(_estimate_lines("Adjusted", note, a.adjusted, a))
    lines.extend(rule("Assumptions"))
    lines.extend(wrapped(a.assumptions))
    return lines


def _estimate_json(est: SwitchbackEstimate, unit: str | None) -> dict[str, object]:
    return {
        "covariates": list(est.covariates),
        "estimate": est.estimate,
        "unit": unit,
        "p_value": est.p_value,
        "interval": None
        if est.refused
        else {
            "lo": None if est.lo_unbounded else est.lo,
            "hi": None if est.hi_unbounded else est.hi,
            "lo_unbounded": est.lo_unbounded,
            "hi_unbounded": est.hi_unbounded,
        },
        "n_kept": est.n_kept,
        "kept_per_block": list(est.kept_per_block),
        "reason": est.reason,
        "detail": est.detail,
    }


def analysis_json(a: SwitchbackAnalysis) -> dict[str, object]:
    """The ``tsdive switchback analyze --json`` payload for one analysis."""
    plan = a.plan
    unit = a.target.physics.unit
    return to_jsonable(
        {
            "tag": a.tag,
            "units": {"raw": unit.raw, "canonical": unit.canonical, "resolved": unit.resolved},
            "quality": {
                "good_fraction": a.good_share,
                "censored": a.censored,
                "range_known": a.target.physics.clipping.range_known,
                "clipped_fraction": a.target.physics.clipping.fraction,
            },
            "plan": {
                "digest": plan.digest,
                "verified": True,
                "seed": plan.seed,
                "blocks": plan.k,
                "block_s": plan.block_s,
                "washout_s": plan.washout_s,
                "schedule": {
                    "start": plan.start,
                    "end": plan.schedule_end,
                    "duration_s": (plan.schedule_end - plan.start).total_seconds(),
                },
                "assignments": json_count(plan.n_assignments),
                "reference": "enumerated" if plan.enumerated else "sampled",
                "reference_size": plan.reference_size,
                "smallest_p": plan.min_p,
            },
            "settings": [b.setting for b in plan.blocks],
            "direct": _estimate_json(a.direct, a.unit),
            "adjusted": None if a.adjusted is None else _estimate_json(a.adjusted, a.unit),
            "unused": list(a.unused),
            "assumptions": a.assumptions,
        }
    )

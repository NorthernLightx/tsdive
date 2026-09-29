"""``tsdive run``: one plan, several archives, one evidence ledger.

A plan is a TOML file naming archives, a window, a baseline, a before and
after period for ``compare``, and the steps to walk; ``switchback`` takes
its schedule file, target and covariates from its options. Every step is one of
the CLI's own analyses, parsed by that
command's own parser and rendered by its own ``render()``, so a plan can
say nothing a command line cannot and the same validators refuse the
same values.

Three kinds of failure, kept apart:

- the plan itself (an unknown step, a glob matching no archive) fails
  before any step runs, and nothing is written;
- a step raising a typed tsdive error for one archive becomes a refusal
  row and the run carries on, because a refusal is evidence about that
  tag, not a reason to abandon the other nine;
- a step raising any other error (a rejected option, overlapping windows,
  a file the OS cannot read) becomes an error row, the same input that
  exits 2 on the command line.
"""

from __future__ import annotations

import argparse
import glob
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NoReturn

import pandas as pd

from tsdive.analyses import ScreenAnalysis, SpcAnalysis
from tsdive.analyses_render import render_lines
from tsdive.api import Profile
from tsdive.cli import (
    ANALYSES,
    BASELINED,
    EXTENT_DEFAULTED,
    MULTI_TAG_STEPS,
    SCHEDULED_STEPS,
    STEPS,
    TWO_PERIOD_STEPS,
)
from tsdive.errors import TSDiveError
from tsdive.narrate import EvidenceLedger
from tsdive.narrate.ledger import row_text, tag_table
from tsdive.report import SEP, continued, label_line, plural, wrapped
from tsdive.store.tagstore import Window, meta_from_parquet
from tsdive.ui.jsonout import error_fields, to_jsonable
from tsdive.ui.static_report import render_static_report, write_static_report
from tsdive.ui.svg import window_figure

PLAN_KEYS = frozenset(
    {"archives", "window", "baseline", "before", "after", "steps", "options"}
)


@dataclass(frozen=True)
class Plan:
    """A plan whose steps are known and whose archives exist.

    Construct it through :func:`load_plan`; every field here has already
    been through the refusals that make a half-run impossible. ``before``
    and ``after`` are both set or both ``None``.
    """

    path: Path
    archives: tuple[str, ...]
    steps: tuple[str, ...]
    window: str | None = None
    baseline: str | None = None
    before: str | None = None
    after: str | None = None
    options: dict[str, dict[str, object]] = field(default_factory=dict)


def _steps(listed: object) -> tuple[str, ...]:
    """Known, distinct step names, returned in pipeline order.

    An unknown name is refused by name rather than skipped, because a
    plan that silently drops a step reports fewer findings than it was
    asked for and says nothing about why.
    """
    if not isinstance(listed, list) or not listed:
        raise ValueError("plan needs a non-empty 'steps' list")
    unknown = [str(s) for s in listed if s not in STEPS]
    if unknown:
        raise ValueError(
            f"unknown step(s): {', '.join(unknown)}; known: {', '.join(STEPS)}"
        )
    twice = sorted({str(s) for s in listed if listed.count(s) > 1})
    if twice:
        raise ValueError(f"step(s) listed twice: {', '.join(twice)}")
    return tuple(name for name in STEPS if name in listed)


def _archives(patterns: object, base: Path) -> tuple[str, ...]:
    """Every archive the globs match, resolved against the plan's directory.

    A glob matching nothing refuses the plan: an empty match is a typo or
    a missing export, and running the pipeline over the archives that did
    match would publish a partial answer as a whole one.
    """
    if not isinstance(patterns, list) or not patterns:
        raise ValueError("plan needs a non-empty 'archives' list")
    found: list[str] = []
    for pattern in patterns:
        matched = sorted(glob.glob(str(pattern), root_dir=base))
        if not matched:
            raise ValueError(f"no archive matches {str(pattern)!r}")
        found.extend(str(base / m) for m in matched)
    return tuple(found)


def _text(value: object, key: str) -> str | None:
    """A window key is a quoted ``START/END`` string, or absent.

    Unquoted, TOML reads ``2024-03-31T01:00:00Z`` as a datetime, which is
    one instant and not a range; refusing it here beats handing argparse
    an object it cannot read.
    """
    if value is None or isinstance(value, str):
        return value
    raise ValueError(
        f"'{key}' must be a quoted START/END string, not a {type(value).__name__}"
    )


def load_plan(path: Path) -> Plan:
    """Read and validate a plan file. Everything it refuses, it refuses here."""
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    unknown = sorted(set(raw) - PLAN_KEYS)
    if unknown:
        raise ValueError(f"unknown plan key(s): {', '.join(unknown)}")
    options = raw.get("options", {})
    if not isinstance(options, dict):
        raise ValueError("'options' must be a table of step names")
    stray = sorted(set(options) - set(STEPS))
    if stray:
        raise ValueError(f"options for unknown step(s): {', '.join(stray)}")
    before = _text(raw.get("before"), "before")
    after = _text(raw.get("after"), "after")
    if (before is None) != (after is None):
        given, missing = ("before", "after") if after is None else ("after", "before")
        raise ValueError(
            f"the plan sets '{given}' without '{missing}'; compare needs both periods"
        )
    return Plan(
        path=path,
        archives=_archives(raw.get("archives"), path.parent),
        steps=_steps(raw.get("steps")),
        window=_text(raw.get("window"), "window"),
        baseline=_text(raw.get("baseline"), "baseline"),
        before=before,
        after=after,
        options=options,
    )


def _option_argv(options: dict[str, object]) -> list[str]:
    """Plan options as the flags the step's own parser will validate.

    Dashes are underscores in TOML keys, a list repeats its flag, and
    ``true`` is the bare flag - so ``k = 2.5`` and ``--k 2.5`` reach the
    same validator and refuse the same values.
    """
    argv: list[str] = []
    for key, value in options.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                argv.append(flag)
        elif isinstance(value, list):
            for item in value:
                argv.extend([flag, str(item)])
        else:
            argv.extend([flag, str(value)])
    return argv


def _step_argv(plan: Plan, step: str, paths: Sequence[str]) -> list[str]:
    """The command line this plan step would have been typed as.

    A relative switchback ``plan`` option resolves against the plan
    file's directory, as the archive globs do.
    """
    argv = list(paths)
    if step in SCHEDULED_STEPS:
        options = dict(plan.options.get(step, {}))
        if isinstance(options.get("plan"), str):
            options["plan"] = str(plan.path.parent / str(options["plan"]))
        return argv + _option_argv(options)
    if step in TWO_PERIOD_STEPS:
        if plan.before is None or plan.after is None:
            raise ValueError(
                f"the plan sets no 'before' and 'after', which {step} requires"
            )
        argv += ["--before", plan.before, "--after", plan.after]
        return argv + _option_argv(plan.options.get(step, {}))
    if plan.window is not None:
        argv += ["--window", plan.window]
    elif step not in EXTENT_DEFAULTED:
        raise ValueError(f"the plan sets no 'window', which {step} requires")
    if step in BASELINED:
        if plan.baseline is None:
            raise ValueError(f"the plan sets no 'baseline', which {step} requires")
        argv += ["--baseline", plan.baseline]
    return argv + _option_argv(plan.options.get(step, {}))


def _parse_step(parser: argparse.ArgumentParser, argv: list[str]) -> argparse.Namespace:
    """Parse a step's argv, turning argparse's exit into a refusal.

    ``exit_on_error=False`` still exits on an unrecognised flag, so the
    error hook is replaced instead: a bad option refuses that one step
    with argparse's own message rather than killing the run.
    """

    def refuse(message: str) -> NoReturn:
        raise ValueError(message)

    parser.error = refuse  # type: ignore[method-assign]
    return parser.parse_args(argv)


def _identity(path: str) -> str:
    """The tag a row is about, or its path when the archive will not open."""
    try:
        return str(meta_from_parquet(Path(path)).identity)
    except (TSDiveError, OSError):
        return path


def _row_lines(marker: str, row: Mapping[str, str]) -> list[str]:
    """A refusal or error on stdout: what failed, then why, folded to the page."""
    return [
        f"{marker:<7}  {row['step']}{SEP}{row['tags']}",
        *wrapped(f"[{row['error_type']}] {row['cause']}"),
    ]


def _tag_row(tag: str, doc: Mapping[str, Any] | None, refused: list[str]) -> dict[str, object]:
    """One row of the per-tag table, read off ``Profile.to_dict()``.

    A tag without a profile document carries None in every profile field.
    """
    if doc is None:
        return {
            "tag": tag,
            "coverage": None,
            "good_share": None,
            "censored": None,
            "gaps": None,
            "longest_gap_s": None,
            "flatline": None,
            "refused": refused,
        }
    values = doc["values"]
    coverage = doc["coverage"]
    return {
        "tag": tag,
        "coverage": coverage["coverage"],
        "good_share": (
            values["n_good"] / values["n_samples"] if values["n_samples"] else None
        ),
        "censored": doc["range"]["censored"],
        "gaps": coverage["n_gaps"],
        "longest_gap_s": coverage["longest_gap_s"],
        "flatline": None if doc["flatline"] is None else doc["flatline"]["verdict"],
        "refused": refused,
    }


@dataclass
class Walk:
    """What walking a plan produced, each list in step order.

    ``profiles`` holds the rendered profile of each archive the profile
    step read; ``tags`` one row per archive in plan order; ``figures`` the
    report's plots.
    """

    profiles: list[str] = field(default_factory=list)
    findings: list[dict[str, object]] = field(default_factory=list)
    refusals: list[dict[str, str]] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)
    tags: list[dict[str, object]] = field(default_factory=list)
    figures: list[str] = field(default_factory=list)


def _figures(
    archives: Sequence[str],
    windows: Mapping[str, Window],
    screens: Mapping[str, ScreenAnalysis],
    spcs: Mapping[str, SpcAnalysis],
) -> list[str]:
    """One plot per archive the profile step read, in plan order.

    A screen adds its limits and flagged samples, an spc its rule hits;
    a step that refused for an archive left no entry and adds nothing.
    A regime-keyed screen has no single set of limits, so only its flags
    are drawn.
    """
    figures: list[str] = []
    for path in archives:
        window = windows.get(path)
        if window is None:
            continue
        limits: tuple[float, float, float] | None = None
        flagged: Sequence[pd.Timestamp] = ()
        hits: Sequence[pd.Timestamp] = ()
        screen = screens.get(path)
        if screen is not None:
            flagged = screen.result.flagged_timestamps
            if screen.provisional is not None:
                lcl, ucl = screen.provisional.limits(screen.k)
                limits = (screen.provisional.center, lcl, ucl)
        spc = spcs.get(path)
        if spc is not None:
            hits = [hit.timestamp for hit in spc.hits]
        figures.append(window_figure(window, limits=limits, flagged=flagged, hits=hits))
    return figures


def execute(plan: Plan) -> Walk:
    """Walk the plan into profiles, findings, refusals, errors, tag rows and figures.

    ``profile`` fills the ledger's profiles because a profile describes a
    window; every other step files a finding, which is a verdict about
    one, with the result's ``to_dict()`` document as its ``data``.
    Nothing raises past here. A typed tsdive error becomes a refusal row;
    any other error (a rejected option, overlapping windows, a file the OS
    cannot read) becomes an error row. Both keep step, tags, error type
    and cause apart, so the report lays them out without parsing a
    sentence back into fields. The figures are the report's plots, one per
    profiled archive, with the screen and spc results of that archive
    drawn over it.
    """
    labels = {path: _identity(path) for path in plan.archives}
    walk = Walk()
    documents: dict[str, dict[str, Any]] = {}
    refused: dict[str, list[str]] = {path: [] for path in plan.archives}
    windows: dict[str, Window] = {}
    screens: dict[str, ScreenAnalysis] = {}
    spcs: dict[str, SpcAnalysis] = {}
    for step in plan.steps:
        groups: list[tuple[str, ...]] = (
            [tuple(plan.archives)]
            if step in MULTI_TAG_STEPS
            else [(path,) for path in plan.archives]
        )
        make_parser, _ = STEPS[step]
        analyse = ANALYSES[step]
        for paths in groups:
            tags = ", ".join(labels[path] for path in paths)
            try:
                args = _parse_step(make_parser(), _step_argv(plan, step, paths))
                result = analyse(args)
                lines = render_lines(result.render())
                data = to_jsonable(result.to_dict())
            except TSDiveError as e:
                walk.refusals.append({"step": step, "tags": tags, **error_fields(e)})
                for path in paths:
                    refused[path].append(step)
                continue
            except (ValueError, OSError) as e:
                walk.errors.append({"step": step, "tags": tags, **error_fields(e)})
                continue
            if isinstance(result, Profile):
                windows[paths[0]] = result.window
                documents[paths[0]] = data
            elif isinstance(result, ScreenAnalysis):
                screens[paths[0]] = result
            elif isinstance(result, SpcAnalysis):
                spcs[paths[0]] = result
            if step == "profile":
                walk.profiles.append("\n".join(lines))
            else:
                walk.findings.append(
                    {"step": step, "tags": tags, "text": "\n".join(lines), "data": data}
                )
    walk.tags = [
        _tag_row(labels[path], documents.get(path), refused[path]) for path in plan.archives
    ]
    walk.figures = _figures(plan.archives, windows, screens, spcs)
    return walk


def render_run(plan: Plan, ledger: EvidenceLedger, written: tuple[Path, ...]) -> list[str]:
    """What ``tsdive run`` prints: the counts, the refusals, the errors, the files.

    Findings are not listed here; they are in the files this names, and
    ledger.txt is this text followed by the tag table, every profile and
    every finding. The error count shows only when an error was filed.
    """
    headline = (
        f"run {plan.path.name}{SEP}{plural(len(plan.archives), 'archive')}{SEP}"
        f"{plural(len(plan.steps), 'step')}{SEP}profiles {len(ledger.profiles)}{SEP}"
        f"findings {len(ledger.findings)}{SEP}refusals {len(ledger.refusals)}"
    )
    if ledger.errors:
        headline += f"{SEP}errors {len(ledger.errors)}"
    lines = [headline]
    for row in ledger.refusals:
        lines.extend(["", *_row_lines("REFUSAL", row)])
    for row in ledger.errors:
        lines.extend(["", *_row_lines("ERROR", row)])
    # as_posix, so the same run prints the same paths on every platform and
    # ledger.txt stays byte-identical between them.
    lines.extend(["", label_line("wrote", written[0].as_posix())])
    lines.extend(continued(path.as_posix()) for path in written[1:])
    return lines


def write_run(
    plan: Plan, out_dir: Path, walk: Walk
) -> tuple[EvidenceLedger, tuple[Path, Path, Path], list[str]]:
    """Write ledger.json, ledger.txt and report.html into ``out_dir``.

    Returns the ledger, the files written, and the lines the command
    prints. ledger.txt opens with those same lines, so the file and the
    terminal say one thing. The figures go into report.html only; the
    ledger files carry text and never change with a plot.

    The ledger is titled after the plan file, never after the clock, so
    two runs of the same plan over the same archives write the same bytes.
    """
    ledger = EvidenceLedger(
        title=f"run {plan.path.name}",
        profiles=walk.profiles,
        findings=walk.findings,
        refusals=walk.refusals,
        errors=walk.errors,
        tags=walk.tags,
    )
    html_text = render_static_report(
        title=f"tsdive run {plan.path.name}",
        profiles=walk.profiles,
        benchmarks_markdown_rows=[
            ("profiles rendered", str(len(walk.profiles))),
            ("findings recorded", str(len(walk.findings))),
            ("refusals recorded", str(len(walk.refusals))),
            ("errors recorded", str(len(walk.errors))),
        ],
        refusal_log=[row_text(row) for row in walk.refusals],
        error_log=[row_text(row) for row in walk.errors],
        findings=[(str(f["step"]), str(f["tags"]), str(f["text"])) for f in walk.findings],
        figures=walk.figures,
        tag_table=tag_table(walk.tags) if walk.tags else None,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    written = (
        out_dir / "ledger.json",
        out_dir / "ledger.txt",
        out_dir / "report.html",
    )
    lines = render_run(plan, ledger, written)
    written[0].write_text(ledger.to_json(), encoding="utf-8", newline="\n")
    written[1].write_text(ledger.to_text(lines), encoding="utf-8", newline="\n")
    write_static_report(html_text, written[2])
    return ledger, written, lines

"""MkDocs hooks that build the documentation site from the repository sources.

``mkdocs.yml`` loads this file. On every build it generates the usage page
from README.md, the CLI reference from the command parsers, the API
reference (an index, one page per group and one page per public object)
from ``__all__``, and the changelog and benchmarks pages from the root
markdown files. The CLI pages carry example runs on the demo archives,
executed in-process at build time. None of them is committed. A relative link whose
target lies outside ``docs/`` is rewritten to the file on GitHub, so the
pages under ``docs/`` keep working both on GitHub and on the site.

The module imports neither mkdocs nor griffe at the top, so the page
generators run without the ``docs`` dependency group.
"""

from __future__ import annotations

import argparse
import ast
import atexit
import contextlib
import functools
import glob
import inspect
import io
import json
import logging
import os
import posixpath
import re
import shlex
import shutil
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mkdocs.config.defaults import MkDocsConfig
    from mkdocs.structure.files import Files
    from mkdocs.structure.pages import Page

ROOT = Path(__file__).resolve().parents[1]
REPO_URL = "https://github.com/NorthernLightx/tsdive"
BLOB_URL = f"{REPO_URL}/blob/main/"
TREE_URL = f"{REPO_URL}/tree/main/"

# Root files published as site pages: repository path -> page path.
ROOT_PAGES = {"CHANGELOG.md": "changelog.md", "BENCHMARKS.md": "benchmarks.md"}

USAGE_PAGE = "usage.md"
CLI_DIR = "reference/cli"
CLI_INDEX = "reference/cli.md"
API_DIR = "reference/api"
API_INDEX = f"{API_DIR}/index.md"

# Groups of the API index, in order.
API_GROUPS = [
    "Reading and ingest",
    "Analyses",
    "Switchback",
    "Evaluation",
    "Archive, sampling contract and types",
    "Errors",
    "Package",
]

# Every name in tsdive.__all__ goes under one group. A name missing here,
# or listed here and not exported, stops the build with the names.
TOP_LEVEL_GROUPS: dict[str, list[str]] = {
    "Reading and ingest": [
        "ingest",
        "ingest_wide",
        "init_meta",
        "init_tag_meta",
        "read_meta_json",
        "write_tag",
        "write_demo_data",
    ],
    "Analyses": [
        "profile",
        "Profile",
        "segment",
        "SegmentAnalysis",
        "screen",
        "ScreenAnalysis",
        "spc",
        "SpcAnalysis",
        "mspc",
        "MspcAnalysis",
        "compare",
        "CompareAnalysis",
    ],
    "Switchback": [
        "switchback_plan",
        "SwitchbackPlan",
        "switchback_analyze",
        "SwitchbackAnalysis",
        "SwitchbackEstimate",
    ],
    "Archive, sampling contract and types": [
        "TagStore",
        "SingleFileStore",
        "Window",
        "Source",
        "TagMeta",
        "TagIdentity",
        "Role",
        "EngRange",
        "SamplingContract",
        "AggregateType",
        "CalculationBasis",
        "RetrievalMode",
    ],
    "Errors": [
        "TSDiveError",
        "DesignTooSmall",
        "IncomparableSamplingError",
        "IncomparableUnitsError",
        "InsufficientQuality",
        "MspcAlignmentError",
        "NarratorUnavailable",
        "NonMonotonicIndex",
        "RegimeTooSparse",
        "ScheduleMismatch",
        "SchemaError",
        "UnresolvedUnitError",
    ],
    "Package": ["__version__"],
}

# Every name in a subpackage's __all__ goes under its group, functions
# first, then classes, then constants.
SUBPACKAGE_GROUPS = {"tsdive.switchback": "Switchback", "tsdive.eval": "Evaluation"}

# Group -> (page slug, intro). An intro that names a subpackage shows its
# module docstring. The Package group has no page of its own.
GROUP_PAGES = {
    "Reading and ingest": (
        "reading-and-ingest",
        "These functions write archives from CSV or parquet exports and read tag "
        "metadata files.",
    ),
    "Analyses": (
        "analyses",
        "Each function reads archives over a window and returns a result object. "
        "`render()` on the result returns the text the command of the same name prints.",
    ),
    "Switchback": ("switchback", "tsdive.switchback"),
    "Evaluation": ("eval", "tsdive.eval"),
    "Archive, sampling contract and types": (
        "types",
        "The store that reads archives, the window a read returns, and the metadata "
        "and sampling contract types an archive carries.",
    ),
    "Errors": ("errors", "The typed errors tsdive raises. Each one derives from `TSDiveError`."),
}

# README sections the usage page carries after "## Use". A link from any
# page to one of them resolves to the usage page.
USAGE_EXTRA_SECTIONS = ("What it checks",)

# Stand-ins in docs/index.md for README sections.
README_MARKERS = {
    "<!-- readme: intro -->": "intro",
    "<!-- readme: install -->": "Install",
}

log = logging.getLogger("mkdocs.hooks.tsdive")

_FENCE = re.compile(r"^\s*(```|~~~)")
_LINK_TARGET = re.compile(r"\]\(([^)\s]+)\)")
_SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")


# ------------------------------------------------------------------ links


@dataclass
class Rewrite:
    """A markdown text with its link targets resolved for one site page."""

    text: str
    to_github: list[tuple[str, str]] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)


def _outside_fences(markdown: str) -> list[tuple[str, bool]]:
    """Each line of ``markdown`` paired with True when it is outside a fenced block."""
    out: list[tuple[str, bool]] = []
    fence: str | None = None
    for line in markdown.splitlines(keepends=True):
        m = _FENCE.match(line)
        if fence is None and m:
            fence = m.group(1)
            out.append((line, False))
        elif fence is not None:
            if m and m.group(1) == fence:
                fence = None
            out.append((line, False))
        else:
            out.append((line, True))
    return out


def rewrite_links(markdown: str, source: str, page: str) -> Rewrite:
    """Resolve the relative link targets of ``markdown`` for the site page ``page``.

    ``source`` is the repository path the text was read from and ``page``
    the page path under ``docs/``. A target under ``docs/``, a root file in
    ``ROOT_PAGES``, or a README anchor the usage page carries becomes a path
    relative to ``page``. Any other repository file becomes its GitHub URL
    on main. Targets with a scheme, bare anchors and absolute paths are
    kept. Text inside fenced code blocks is kept verbatim.
    """
    result = Rewrite(text="")
    page_dir = posixpath.dirname(page) or "."
    known = generated_paths()

    def replace(m: re.Match[str]) -> str:
        target = m.group(1)
        if _SCHEME.match(target) or target.startswith(("#", "/")):
            return m.group(0)
        path, sep, anchor = target.partition("#")
        resolved = posixpath.normpath(posixpath.join(posixpath.dirname(source), path))
        generated = resolved.startswith("docs/") and resolved[5:] in known
        if resolved.startswith("../") or not (generated or (ROOT / resolved).exists()):
            result.missing.append(target)
            return m.group(0)
        if resolved == "README.md" and anchor in usage_anchors():
            new = posixpath.relpath(USAGE_PAGE, page_dir) + sep + anchor
        elif resolved in ROOT_PAGES or resolved.startswith("docs/"):
            dest = ROOT_PAGES.get(resolved, resolved.removeprefix("docs/"))
            new = posixpath.relpath(dest, page_dir) + sep + anchor
        else:
            base = TREE_URL if (ROOT / resolved).is_dir() else BLOB_URL
            new = base + resolved + sep + anchor
            result.to_github.append((target, new))
        return f"]({new})"

    parts = [
        _LINK_TARGET.sub(replace, line) if outside else line
        for line, outside in _outside_fences(markdown)
    ]
    result.text = "".join(parts)
    return result


# ------------------------------------------------------------------ README


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def readme_section(heading: str) -> str:
    """The body of the README ``## heading`` section, without its heading line.

    ``intro`` is the text between the badge line and the first ``##``
    heading.
    """
    lines = _read("README.md").splitlines()
    if heading == "intro":
        start = next(i for i, line in enumerate(lines) if line.startswith("[![")) + 1
    else:
        start = lines.index(f"## {heading}") + 1
    end = len(lines)
    for i, (line, outside) in enumerate(_outside_fences("\n".join(lines[start:]))):
        if outside and line.startswith("## "):
            end = start + i
            break
    return "\n".join(lines[start:end]).strip() + "\n"


def _anchor(heading: str) -> str:
    """The id MkDocs gives a heading: punctuation dropped, lower case, hyphens for spaces."""
    text = re.sub(r"[^\w\s-]", "", heading).strip().lower()
    return re.sub(r"[-\s]+", "-", text)


def _usage_body() -> str:
    body = [
        line[1:] if outside and line.startswith("###") else line
        for line, outside in _outside_fences(readme_section("Use"))
    ]
    text = "# Usage\n\n" + "".join(body)
    for heading in USAGE_EXTRA_SECTIONS:
        text += f"\n## {heading}\n\n{readme_section(heading)}"
    return text


def usage_anchors() -> frozenset[str]:
    """The ids of the usage page's second-level headings."""
    return frozenset(
        _anchor(line[3:])
        for line, outside in _outside_fences(_usage_body())
        if outside and line.startswith("## ")
    )


def usage_markdown() -> Rewrite:
    """The README ``## Use`` section as a page, headings raised one level.

    The sections in ``USAGE_EXTRA_SECTIONS`` follow it, at their README level.
    """
    return rewrite_links(_usage_body(), "README.md", USAGE_PAGE)


def index_markdown(markdown: str) -> Rewrite:
    """docs/index.md with each README marker replaced by that README section."""
    for marker, heading in README_MARKERS.items():
        if marker not in markdown:
            raise ValueError(f"docs/index.md has no {marker} line")
        section = rewrite_links(readme_section(heading), "README.md", "index.md")
        if section.missing:
            raise ValueError(f"README {heading} links to missing files: {section.missing}")
        markdown = markdown.replace(marker, section.text.strip())
    return rewrite_links(markdown, "docs/index.md", "index.md")


# ------------------------------------------------------------------ CLI

# Column where MAIN_DOC prints a command's summary.
SUMMARY_COL = 42
# Lines of example output a command page shows before the cut.
EXAMPLE_LINES = 30
# Pages whose ``$ tsdive`` console lines are the examples, in this order.
EXAMPLE_SOURCES = ("README.md", "docs/SWITCHBACK.md")
# Commands the pages above show without a ``$`` prompt: argv, and the
# sentence that says what the example prepared.
EXTRA_EXAMPLES = {
    "demo": (
        ["demo"],
        "The command writes into `tsdive-demo/` under the current directory.",
    ),
    "ingest": (
        ["ingest", "fic101.csv", "--out", "archive/plant1/FIC101.PV.parquet", "--meta",
         "fic101.json"],
        "The example exports the FIC-101 demo archive to `fic101.csv` and its metadata "
        "to `fic101.json` first, then ingests them again.",
    ),
    "report-html": (
        ["report-html", "data/demo/fic101_demo.parquet", "data/demo/tic101_demo.parquet",
         "-o", "report.html"],
        "The command writes `report.html`; the block shows the lines it prints.",
    ),
}
# Guides that walk a command on real output, beside the README usage page.
GUIDES = {"switchback plan": "SWITCHBACK.md", "switchback analyze": "SWITCHBACK.md"}


def cli_entries() -> dict[str, str]:
    """Command -> summary, for every command ``tsdive --help`` lists, in its order.

    A command listed twice (``ingest``, once with ``--wide``) joins its
    summaries with a semicolon.
    """
    from tsdive.cli import MAIN_DOC

    block = MAIN_DOC.split("\ncommands", 1)[1].split("\n\n", 1)[0]
    chunks: dict[str, list[list[str]]] = {}
    current: list[str] = []
    for line in block.splitlines()[1:]:
        m = re.match(r"^  ([a-z][a-z-]*)(?: ([a-z]+)(?= |$))?", line)
        if m:
            current = []
            chunks.setdefault(" ".join(g for g in m.groups() if g), []).append(current)
            if line[SUMMARY_COL - 2 : SUMMARY_COL] == "  " and len(line) > SUMMARY_COL:
                current.append(line[SUMMARY_COL:].strip())
        elif line.startswith(" " * SUMMARY_COL):
            current.append(line.strip())
    entries = {
        name: "; ".join(" ".join(chunk) for chunk in parts if chunk)
        for name, parts in chunks.items()
    }
    empty = [name for name, text in entries.items() if not text]
    if empty:
        raise ValueError(f"MAIN_DOC gives no summary for {empty}")
    return entries


def cli_commands() -> list[str]:
    """The commands ``tsdive --help`` lists, in its order, subcommands spelled out."""
    return list(cli_entries())


def cli_page(command: str) -> str:
    """The page path of ``command``: ``switchback plan`` is ``switchback-plan``."""
    return f"{CLI_DIR}/{command.replace(' ', '-')}.md"


def _cli_attr(prefix: str, command: str) -> Any:
    from tsdive import cli

    return getattr(cli, prefix + command.replace(" ", "_").replace("-", "_"), None)


def cli_parser(command: str) -> argparse.ArgumentParser:
    """The argparse parser of ``command``, from its ``_parser_<command>`` builder."""
    build = _cli_attr("_parser_", command)
    if build is None:
        raise ValueError(f"tsdive.cli has no parser builder for {command!r}")
    return build()


def cli_description(command: str) -> str:
    """The docstring of the function that runs ``command``."""
    for prefix in ("run_", "cmd_"):
        handler = _cli_attr(prefix, command)
        if handler is not None and handler.__doc__:
            return inspect.getdoc(handler) or ""
    raise ValueError(f"tsdive.cli has no documented handler for {command!r}")


@contextlib.contextmanager
def _plain_terminal() -> Iterator[None]:
    """80 columns and no colour, the terminal the pages show."""
    saved = {k: os.environ.get(k) for k in ("COLUMNS", "NO_COLOR")}
    os.environ.update(COLUMNS="80", NO_COLOR="1")
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _run_cli(argv: list[str]) -> tuple[int | str | None, str]:
    """Exit code and combined output of ``tsdive <argv>``, run in-process."""
    from tsdive.cli import main

    buffer = io.StringIO()
    with (
        _plain_terminal(),
        contextlib.redirect_stdout(buffer),
        contextlib.redirect_stderr(buffer),
    ):
        try:
            code: int | str | None = main(argv)
        except SystemExit as e:
            code = e.code
    return code, buffer.getvalue().rstrip()


def cli_help(argv: list[str]) -> str:
    """What ``tsdive <argv> --help`` prints, captured in-process at 80 columns."""
    code, text = _run_cli([*argv, "--help"])
    if code not in (0, None):
        raise RuntimeError(f"tsdive {' '.join(argv)} --help exited {code}")
    return text


def _cell(text: str) -> str:
    return " ".join(text.split()).replace("|", "\\|")


def options_table(parser: argparse.ArgumentParser) -> str:
    """One row per argument of ``parser``: name, value, default, description."""
    formatter = parser._get_formatter()
    rows = ["| Argument | Value | Default | Description |", "|---|---|---|---|"]
    for action in parser._actions:
        if isinstance(action, argparse._HelpAction):
            continue
        takes_value = action.nargs != 0
        if action.option_strings:
            name = ", ".join(f"`{s}`" for s in action.option_strings)
            value = (action.metavar or action.dest.upper()) if takes_value else ""
        else:
            many = " ..." if action.nargs in ("+", "*") else ""
            name = f"`{action.metavar or action.dest}{many}`"
            value = ""
        if action.choices is not None:
            value = "{" + ",".join(map(str, action.choices)) + "}"
        if value and action.nargs in ("+", "*"):
            value = f"{value} ..."
        optional_positional = action.nargs in ("?", "*")
        required = action.required or not (action.option_strings or optional_positional)
        if required:
            default = "required"
        elif takes_value and action.default not in (None, argparse.SUPPRESS):
            default = f"`{action.default}`"
        else:
            default = ""
        help_text = formatter._expand_help(action) if action.help else ""
        rows.append(
            f"| {name} | {f'`{value}`' if value else ''} | {default} | {_cell(help_text)} |"
        )
    return "\n".join(rows) + "\n"


def _console_lines(markdown: str) -> list[str]:
    """Each ``$ tsdive`` command of the console blocks in ``markdown``, continuations joined."""
    found: list[str] = []
    current: list[str] = []
    for line, outside in _outside_fences(markdown):
        if outside:
            continue
        text = line.rstrip("\n")
        if current:
            current.append(text)
        elif text.startswith("$ tsdive "):
            current = [text]
        if current and not text.endswith("\\"):
            found.append("\n".join(current))
            current = []
    return found


def documented_invocations() -> dict[str, tuple[list[str], str]]:
    """Command -> (argv, console text) of its first ``$ tsdive`` line in the example sources."""
    found: dict[str, tuple[list[str], str]] = {}
    for source in EXAMPLE_SOURCES:
        for text in _console_lines(_read(source)):
            argv = shlex.split(text.replace("\\\n", " "))[2:]
            command = " ".join(argv[:2]) if argv[0] == "switchback" else argv[0]
            found.setdefault(command, (argv, text))
    return found


def _wrap_console(argv: list[str]) -> str:
    """``$ tsdive <argv>`` on lines of at most 78 characters, continued with ``\\``."""
    lines, line = [], "$ tsdive"
    for word in (shlex.quote(a) for a in argv):
        if len(line) + len(word) + 3 > 78 and word.startswith("-"):
            lines.append(line + " \\")
            line = "   "
        line += " " + word
    return "\n".join([*lines, line])


@functools.cache
def _demo_data() -> Path:
    """The ``data/`` directory ``tsdive demo data`` writes, built once per build."""
    from tsdive.demo import write_demo_data

    data = Path(tempfile.mkdtemp(prefix="tsdive-docs-")) / "data"
    write_demo_data(data)
    atexit.register(shutil.rmtree, data.parent, ignore_errors=True)
    return data


@contextlib.contextmanager
def demo_workdir() -> Iterator[Path]:
    """A fresh current directory holding ``data/`` and ``examples/plans/demo.toml``.

    It is the directory a reader works in after ``tsdive demo data`` in a
    clone: every example command of the docs runs in one of these.
    """
    before = Path.cwd()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        work = Path(tmp).resolve()
        shutil.copytree(_demo_data(), work / "data")
        plans = work / "examples" / "plans"
        plans.mkdir(parents=True)
        shutil.copy(ROOT / "examples" / "plans" / "demo.toml", plans / "demo.toml")
        os.chdir(work)
        try:
            yield work
        finally:
            os.chdir(before)


def run_in(work: Path, argv: list[str]) -> tuple[int | str | None, str]:
    """``tsdive <argv>`` in ``work``: globs expanded as a shell would, paths relative."""
    expanded = [
        p.replace(os.sep, "/")
        for a in argv
        for p in (sorted(glob.glob(a)) if "*" in a else [a])
    ]
    code, output = _run_cli(expanded)
    return code, output.replace(str(work) + os.sep, "").replace(str(work), ".")


def _export_fic101(work: Path) -> None:
    """``fic101.csv`` and ``fic101.json``: the FIC-101 demo archive as an export."""
    import pandas as pd

    from tsdive.store.tagstore import meta_from_parquet, meta_to_dict

    demo = work / "data" / "demo" / "fic101_demo.parquet"
    pd.read_parquet(demo).to_csv(work / "fic101.csv", index=False)
    meta = meta_to_dict(meta_from_parquet(demo))
    (work / "fic101.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def cli_examples() -> dict[str, tuple[str, str]]:
    """Command -> (console text with its output, note on what it read), run on the demo data.

    Every command runs in-process, in the order ``tsdive --help`` lists
    them, in a temporary directory holding the files the README and the
    guides assume. An exit code other than 0 stops the build.
    """
    documented = documented_invocations()
    examples: dict[str, tuple[str, str]] = {}
    with demo_workdir() as work:
        _export_fic101(work)
        for command in cli_commands():
            if command in documented:
                argv, text = documented[command]
                note = ""
            elif command in EXTRA_EXAMPLES:
                argv, note = EXTRA_EXAMPLES[command]
                text = _wrap_console(argv)
            else:
                raise ValueError(f"no example invocation for tsdive {command}")
            code, output = run_in(work, argv)
            if code not in (0, None):
                raise RuntimeError(f"tsdive {command} example exited {code}:\n{output}")
            examples[command] = ("\n".join([text, *_cut(output.splitlines())]), note)
    return examples


def _cut(lines: list[str], keep: int = EXAMPLE_LINES) -> list[str]:
    """The first ``keep`` lines, and a marked cut when there were more."""
    if len(lines) <= keep:
        return lines
    return [*lines[:keep], f"[{len(lines) - keep} more lines not shown]"]


def _cli_index(entries: dict[str, str]) -> str:
    rows = ["| Command | Summary |", "|---|---|"]
    for command, text in entries.items():
        link = posixpath.relpath(cli_page(command), posixpath.dirname(CLI_INDEX))
        rows.append(f"| [`tsdive {command}`]({link}) | {_cell(text)} |")
    return "\n".join(
        [
            "# CLI reference\n",
            "Every command, with the summary `tsdive --help` prints. Each command links to "
            "its usage line, its options and an example run on the demo archives.\n",
            "The examples run in a directory where `tsdive demo data` wrote those archives, "
            "with the repository's `examples/plans/demo.toml`.\n",
            "\n".join(rows) + "\n",
            "## tsdive --help\n",
            f"```text\n{cli_help([])}\n```\n",
        ]
    )


def _cli_command_page(command: str, example: tuple[str, str], usage_sections: set[str]) -> str:
    here = cli_page(command)
    usage = cli_help(command.split()).split("\n\n", 1)[0]
    parts = [
        f"---\ntitle: tsdive {command}\n---\n",
        f"[CLI reference]({posixpath.relpath(CLI_INDEX, posixpath.dirname(here))})\n",
        f"# tsdive {command}\n",
        cli_description(command) + "\n",
    ]
    if command in usage_sections:
        link = posixpath.relpath(USAGE_PAGE, posixpath.dirname(here))
        parts.append(f"The [usage walkthrough]({link}#{command}) has a section on it.\n")
    elif command in GUIDES:
        link = posixpath.relpath(GUIDES[command], posixpath.dirname(here))
        parts.append(f"The [guide]({link}) walks a trial with it.\n")
    text, note = example
    parts += [
        "## Usage\n",
        f"```text\n{usage}\n```\n",
        "## Options\n",
        options_table(cli_parser(command)),
        "## Example\n",
    ]
    if note:
        parts.append(note + "\n")
    parts.append(f"```console\n{text}\n```\n")
    return "\n".join(parts)


def cli_markdown() -> dict[str, str]:
    """Page path -> markdown of the CLI index and one page per command."""
    entries = cli_entries()
    examples = cli_examples()
    usage_sections = {
        line[4:].strip() for line in readme_section("Use").splitlines() if line.startswith("### ")
    }
    pages = {CLI_INDEX: _cli_index(entries)}
    for command in entries:
        pages[cli_page(command)] = _cli_command_page(command, examples[command], usage_sections)
    return pages


# ------------------------------------------------------------------ API


@functools.cache
def _tree(source: Path) -> ast.Module:
    return ast.parse(source.read_text(encoding="utf-8"))


def _module_source(module: str) -> Path:
    base = ROOT / "src" / Path(*module.split("."))
    return base / "__init__.py" if base.is_dir() else base.with_suffix(".py")


def exports(module: str) -> list[str]:
    """``__all__`` of ``module``, read from its source without importing it."""
    source = _module_source(module)
    for node in _tree(source).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
        ):
            return list(ast.literal_eval(node.value))
    raise ValueError(f"{source} defines no __all__")


@dataclass(frozen=True)
class Definition:
    """Where a public name is defined, what it is, and its docstring."""

    module: str
    kind: str  # "function", "class" or "attribute"
    docstring: str
    value: str = ""  # source of an attribute's value


def definition(module: str, name: str) -> Definition:
    """The definition ``module.name`` resolves to, following imports statically."""
    source = _module_source(module)
    body = _tree(source).body
    for i, node in enumerate(body):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            if node.name == name:
                kind = "class" if isinstance(node, ast.ClassDef) else "function"
                return Definition(module, kind, ast.get_docstring(node) or "")
            continue
        targets = (
            node.targets
            if isinstance(node, ast.Assign)
            else [node.target]
            if isinstance(node, ast.AnnAssign)
            else []
        )
        if any(isinstance(t, ast.Name) and t.id == name for t in targets):
            after = body[i + 1] if i + 1 < len(body) else None
            doc = (
                after.value.value
                if isinstance(after, ast.Expr)
                and isinstance(after.value, ast.Constant)
                and isinstance(after.value.value, str)
                else ""
            )
            value = ast.unparse(node.value) if node.value is not None else ""
            return Definition(module, "attribute", inspect.cleandoc(doc), value)
    package = module if source.name == "__init__.py" else module.rpartition(".")[0]
    for node in body:
        if not isinstance(node, ast.ImportFrom):
            continue
        for alias in node.names:
            if (alias.asname or alias.name) == name:
                base = package
                for _ in range(max(node.level - 1, 0)):
                    base = base.rpartition(".")[0]
                origin = node.module if not node.level else ".".join(
                    p for p in (base, node.module) if p
                )
                return definition(origin, alias.name)
    raise LookupError(f"{module}.{name} is neither defined nor imported in {source.name}")


def object_path(module: str, name: str) -> str:
    """The dotted path mkdocstrings renders for ``module.name``.

    Static analysis resolves ``module.name`` to the submodule when the
    package also has a submodule called ``name`` (``tsdive.compare`` is
    both). Such a name is rendered from the module that defines it.
    """
    pkg = ROOT / "src" / Path(*module.split("."))
    # Listed names, so the match is case-sensitive on every file system.
    submodules = {c.stem for c in pkg.glob("*.py")} | {
        c.name for c in pkg.iterdir() if (c / "__init__.py").exists()
    }
    if name not in submodules:
        return f"{module}.{name}"
    return f"{definition(module, name).module}.{name}"


def summary(docstring: str) -> str:
    """The first paragraph of ``docstring`` on one line, ready for a table cell."""
    first = " ".join(docstring.strip().split("\n\n", 1)[0].split())
    return first.replace("|", "\\|")


@dataclass(frozen=True)
class ApiEntry:
    """One public name: its group, its page, and what the page renders."""

    public: str  # dotted path a caller imports, tsdive.profile
    group: str
    page: str  # page path under docs/; one page per object
    directive: str  # path given to mkdocstrings
    kind: str
    summary: str


_KIND_ORDER = {"function": 0, "class": 1, "attribute": 2}


def api_entries() -> list[ApiEntry]:
    """Every public name in index order: groups in order, then the listed order.

    A name reached under two public paths (``tsdive.SwitchbackPlan`` and
    ``tsdive.switchback.SwitchbackPlan``) gets one page, the first path's.
    """
    names = exports("tsdive")
    grouped = [n for group in TOP_LEVEL_GROUPS.values() for n in group]
    ungrouped = sorted(set(names) - set(grouped))
    stale = sorted(set(grouped) - set(names))
    if ungrouped or stale or len(grouped) != len(set(grouped)):
        raise ValueError(
            f"TOP_LEVEL_GROUPS out of step with tsdive.__all__: add {ungrouped}, "
            f"remove {stale}, and list each name once"
        )
    ordered: list[tuple[str, str, str]] = []  # (group, module, name)
    for group in API_GROUPS:
        ordered += [(group, "tsdive", n) for n in TOP_LEVEL_GROUPS.get(group, [])]
        for module, owner in SUBPACKAGE_GROUPS.items():
            if owner == group:
                found = [(n, definition(module, n)) for n in exports(module)]
                found.sort(key=lambda item: _KIND_ORDER[item[1].kind])
                ordered += [(group, module, n) for n, _ in found]
    resolved = [(group, module, name, definition(module, name)) for group, module, name in ordered]
    # Page paths that differ only in case (tsdive.profile and tsdive.Profile)
    # are one file on a case-insensitive file system; the non-function one
    # takes its kind as a suffix.
    functions = {
        f"{module}.{name}".casefold()
        for _, module, name, found in resolved
        if found.kind == "function"
    }
    entries: list[ApiEntry] = []
    pages: dict[tuple[str, str], str] = {}
    for group, module, name, found in resolved:
        public = f"{module}.{name}"
        stem = public
        if found.kind != "function" and public.casefold() in functions:
            stem = f"{public}-{found.kind}"
        page = pages.setdefault((found.module, name), f"{API_DIR}/{stem}.md")
        text = summary(found.docstring) if found.docstring else f"`{name} = {found.value}`"
        entries.append(
            ApiEntry(public, group, page, object_path(module, name), found.kind, text)
        )
    folded = [page.casefold() for page in set(pages.values())]
    if len(folded) != len(set(folded)):
        raise ValueError("two API pages differ only in case; extend the suffix rule")
    return entries


def _directive(path: str, **options: object) -> str:
    """A mkdocstrings block rendering ``path`` with ``options``."""
    lines = [f"::: {path}", "    options:"] if options else [f"::: {path}"]
    for key, value in options.items():
        lines.append(f"      {key}: {str(value).lower() if isinstance(value, bool) else value}")
    return "\n".join(lines) + "\n"


def _slug(group: str) -> str:
    return GROUP_PAGES[group][0]


def _table(entries: list[ApiEntry], here: str) -> str:
    rows = ["| Name | Summary |", "|---|---|"]
    for e in entries:
        link = posixpath.relpath(e.page, posixpath.dirname(here))
        rows.append(f"| [`{e.public}`]({link}) | {e.summary} |")
    return "\n".join(rows) + "\n"


def _api_index(entries: list[ApiEntry]) -> str:
    parts = [
        "# API reference\n",
        "Every public name of `tsdive`, `tsdive.eval` and `tsdive.switchback`, grouped by "
        "task, with the first line of its docstring. Each name links to its own page.\n",
        "The examples on those pages read the demo archives that "
        "`tsdive demo data` writes under `data/`.\n",
    ]
    for group in API_GROUPS:
        members = [e for e in entries if e.group == group]
        heading = f"[{group}]({_slug(group)}.md)" if group in GROUP_PAGES else group
        parts += [f"## {heading}\n", _table(members, API_INDEX)]
    return "\n".join(parts)


def _group_page(group: str, entries: list[ApiEntry]) -> str:
    slug, intro = GROUP_PAGES[group]
    page = f"{API_DIR}/{slug}.md"
    parts = [f"# {group}\n", "[API reference](index.md)\n"]
    if intro in SUBPACKAGE_GROUPS:
        parts.append(_directive(intro, show_root_heading=False, members=False))
    else:
        parts.append(intro + "\n")
    parts.append(_table([e for e in entries if e.group == group], page))
    return "\n".join(parts)


def _object_page(entry: ApiEntry) -> str:
    back = "[API reference](index.md)"
    if entry.group in GROUP_PAGES:
        back += f" / [{entry.group}]({_slug(entry.group)}.md)"
    block = _directive(
        entry.directive,
        heading_level=1,
        heading=entry.public,
        toc_label=entry.public,
        summary=True,
    )
    return f"---\ntitle: {entry.public}\n---\n\n{back}\n\n{block}"


def api_markdown() -> dict[str, tuple[str, bool]]:
    """Page path -> (markdown, listed in the nav) for the API index, groups and objects."""
    entries = api_entries()
    pages = {API_INDEX: (_api_index(entries), True)}
    for group in GROUP_PAGES:
        pages[f"{API_DIR}/{_slug(group)}.md"] = (_group_page(group, entries), True)
    for entry in entries:
        pages.setdefault(entry.page, (_object_page(entry), False))
    return pages


# ------------------------------------------------------------------ pages


def root_page(source: str) -> Rewrite:
    """A root markdown file with its links resolved for its site page."""
    return rewrite_links(_read(source), source, ROOT_PAGES[source])


@functools.cache
def generated_paths() -> frozenset[str]:
    """Every page path the build generates, from the sources alone."""
    commands = {CLI_INDEX, *(cli_page(c) for c in cli_commands())}
    groups = {f"{API_DIR}/{_slug(g)}.md" for g in GROUP_PAGES}
    objects = {e.page for e in api_entries()}
    return frozenset({USAGE_PAGE, *ROOT_PAGES.values(), *commands, API_INDEX, *groups, *objects})


@dataclass(frozen=True)
class Generated:
    """One generated page: its text, the file its edit link opens, and its nav entry."""

    result: Rewrite
    edit_source: str | None = None
    in_nav: bool = True


def generated_pages() -> dict[str, Generated]:
    """Page path -> the generated page."""
    pages = {USAGE_PAGE: Generated(usage_markdown(), "README.md")}
    for source, page in ROOT_PAGES.items():
        pages[page] = Generated(root_page(source), source)
    for page, text in cli_markdown().items():
        pages[page] = Generated(Rewrite(text), "src/tsdive/cli.py", in_nav=page == CLI_INDEX)
    for page, (text, in_nav) in api_markdown().items():
        pages[page] = Generated(Rewrite(text), in_nav=in_nav)
    return pages


# ------------------------------------------------------------------ hooks


def on_files(files: Files, config: MkDocsConfig) -> Files:
    """Add the generated pages to the build; the per-object pages stay out of the nav."""
    from mkdocs.structure.files import File, InclusionLevel

    _tree.cache_clear()
    generated_paths.cache_clear()
    docs = Path(config.docs_dir).resolve().relative_to(ROOT).as_posix()
    for page, generated in generated_pages().items():
        for target in generated.result.missing:
            log.warning("%s links to %s, which is not in the repository", page, target)
        inclusion = InclusionLevel.INCLUDED if generated.in_nav else InclusionLevel.NOT_IN_NAV
        f = File.generated(config, page, content=generated.result.text, inclusion=inclusion)
        if generated.edit_source is not None:
            f.edit_uri = posixpath.relpath(generated.edit_source, docs)
        files.append(f)
    return files


def on_page_markdown(markdown: str, page: Page, config: MkDocsConfig, **_: Any) -> str:
    """Fill the README markers of the index and resolve links that leave docs/."""
    src = page.file.src_uri
    if page.file.generated_by is not None:
        return markdown
    if src == "index.md":
        result = index_markdown(markdown)
    else:
        result = rewrite_links(markdown, f"docs/{src}", src)
    for target in result.missing:
        log.warning("%s links to %s, which is not in the repository", src, target)
    if result.to_github:
        log.debug("%s: %d links point at GitHub", src, len(result.to_github))
    return result.text

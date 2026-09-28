"""MkDocs hooks that build the documentation site from the repository sources.

``mkdocs.yml`` loads this file. On every build it generates the usage page
from README.md, the CLI reference from the command parsers, the API
reference (an index, one page per group and one page per public object)
from ``__all__``, and the changelog and benchmarks pages from the root
markdown files. None of them is committed. A relative link whose
target lies outside ``docs/`` is rewritten to the file on GitHub, so the
pages under ``docs/`` keep working both on GitHub and on the site.

The module imports neither mkdocs nor griffe at the top, so the page
generators run without the ``docs`` dependency group.
"""

from __future__ import annotations

import ast
import contextlib
import functools
import inspect
import io
import logging
import os
import posixpath
import re
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
CLI_PAGE = "reference/cli.md"
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
    "Reading and ingest": ["ingest", "ingest_wide", "init_meta", "read_meta_json", "write_tag"],
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
    the page path under ``docs/``. A target under ``docs/``, or a root file
    in ``ROOT_PAGES``, becomes a path relative to ``page``. Any other
    repository file becomes its GitHub URL on main. Targets with a scheme,
    bare anchors and absolute paths are kept. Text inside fenced code
    blocks is kept verbatim.
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
        if resolved in ROOT_PAGES or resolved.startswith("docs/"):
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


def usage_markdown() -> Rewrite:
    """The README ``## Use`` section as a page, headings raised one level."""
    body = []
    for line, outside in _outside_fences(readme_section("Use")):
        body.append(line[1:] if outside and line.startswith("###") else line)
    return rewrite_links("# Usage\n\n" + "".join(body), "README.md", USAGE_PAGE)


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


def cli_commands() -> list[str]:
    """The commands ``tsdive --help`` lists, in its order, subcommands spelled out."""
    from tsdive.cli import MAIN_DOC

    block = MAIN_DOC.split("\ncommands", 1)[1].split("\n\n", 1)[0]
    found: list[str] = []
    for line in block.splitlines():
        m = re.match(r"^  ([a-z][a-z-]*)(?: ([a-z]+)(?= |$))?", line)
        if m:
            name = " ".join(g for g in m.groups() if g)
            if name not in found:
                found.append(name)
    return found


def cli_help(argv: list[str]) -> str:
    """What ``tsdive <argv> --help`` prints, captured in-process at 80 columns."""
    from tsdive.cli import main

    saved = {k: os.environ.get(k) for k in ("COLUMNS", "NO_COLOR")}
    os.environ.update(COLUMNS="80", NO_COLOR="1")
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            try:
                code = main([*argv, "--help"])
            except SystemExit as e:
                code = e.code
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    if code not in (0, None):
        raise RuntimeError(f"tsdive {' '.join(argv)} --help exited {code}")
    return buffer.getvalue().rstrip()


def cli_markdown() -> str:
    """One section per command holding its ``--help`` output."""
    from tsdive import __version__

    parts = [
        "# CLI reference\n",
        f"Each section holds what `tsdive <command> --help` prints in tsdive {__version__}.\n",
    ]
    sections: list[tuple[str, list[str]]] = [("tsdive", [])]
    for name in cli_commands():
        words = name.split()
        if len(words) > 1 and (words[0], words[:1]) not in sections:
            sections.append((words[0], words[:1]))
        sections.append((name, words))
    for title, argv in sections:
        parts.append(f"## {title}\n\n```text\n{cli_help(argv)}\n```\n")
    return "\n".join(parts)


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
        "`python scripts/make_demo_archive.py` and "
        "`python examples/switchback/make_trial.py` write under `data/`.\n",
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
    groups = {f"{API_DIR}/{_slug(g)}.md" for g in GROUP_PAGES}
    objects = {e.page for e in api_entries()}
    return frozenset({USAGE_PAGE, CLI_PAGE, *ROOT_PAGES.values(), API_INDEX, *groups, *objects})


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
    pages[CLI_PAGE] = Generated(Rewrite(cli_markdown()), "src/tsdive/cli.py")
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

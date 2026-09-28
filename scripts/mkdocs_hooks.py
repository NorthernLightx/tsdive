"""MkDocs hooks that build the documentation site from the repository sources.

``mkdocs.yml`` loads this file. On every build it generates the usage page
from README.md, the CLI reference from the command parsers, the Python API
pages from ``__all__``, and the changelog and benchmarks pages from the
root markdown files. None of them is committed. A relative link whose
target lies outside ``docs/`` is rewritten to the file on GitHub, so the
pages under ``docs/`` keep working both on GitHub and on the site.

The module imports neither mkdocs nor griffe at the top, so the page
generators run without the ``docs`` dependency group.
"""

from __future__ import annotations

import ast
import contextlib
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
API_PAGES = {
    "tsdive": "reference/api/tsdive.md",
    "tsdive.eval": "reference/api/eval.md",
    "tsdive.switchback": "reference/api/switchback.md",
}
GENERATED = {USAGE_PAGE, CLI_PAGE, *ROOT_PAGES.values(), *API_PAGES.values()}

# The top-level API page lists every name in tsdive.__all__ under one of
# these headings. A name missing here, or listed here and not exported,
# stops the build with the names.
TOP_LEVEL_GROUPS: dict[str, list[str]] = {
    "Reading and ingest": [
        "ingest",
        "ingest_wide",
        "init_meta",
        "read_meta_json",
        "write_tag",
        "TagStore",
        "SingleFileStore",
        "Window",
        "Source",
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
    "Types": [
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

    def replace(m: re.Match[str]) -> str:
        target = m.group(1)
        if _SCHEME.match(target) or target.startswith(("#", "/")):
            return m.group(0)
        path, sep, anchor = target.partition("#")
        resolved = posixpath.normpath(posixpath.join(posixpath.dirname(source), path))
        generated = resolved.startswith("docs/") and resolved[5:] in GENERATED
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


def _package_dir(module: str) -> Path:
    return ROOT / "src" / Path(*module.split("."))


def exports(module: str) -> list[str]:
    """``__all__`` of ``module``, read from its ``__init__.py`` without importing it."""
    init = _package_dir(module) / "__init__.py"
    for node in ast.parse(init.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
        ):
            return list(ast.literal_eval(node.value))
    raise ValueError(f"{init} defines no __all__")


def object_path(module: str, name: str) -> str:
    """The dotted path mkdocstrings renders for ``module.name``.

    Static analysis resolves ``module.name`` to the submodule when the
    package also has a submodule called ``name`` (``tsdive.compare`` is
    both). Such a name is rendered from the module its ``__init__.py``
    imports it from.
    """
    pkg = _package_dir(module)
    # Listed names, so the match is case-sensitive on every file system.
    submodules = {c.stem for c in pkg.glob("*.py")} | {
        c.name for c in pkg.iterdir() if (c / "__init__.py").exists()
    }
    if name not in submodules:
        return f"{module}.{name}"
    for node in ast.parse((pkg / "__init__.py").read_text(encoding="utf-8")).body:
        if isinstance(node, ast.ImportFrom) and any(a.name == name for a in node.names):
            return f"{node.module}.{name}"
    raise ValueError(f"{module}.{name} names a submodule and is not imported in __init__.py")


def _directive(path: str, **options: object) -> str:
    lines = [f"::: {path}"]
    if options:
        lines.append("    options:")
        lines += [f"      {k}: {str(v).lower()}" for k, v in options.items()]
    return "\n".join(lines) + "\n"


def _top_level_page() -> str:
    names = exports("tsdive")
    grouped = [n for group in TOP_LEVEL_GROUPS.values() for n in group]
    ungrouped = sorted(set(names) - set(grouped))
    stale = sorted(set(grouped) - set(names))
    if ungrouped or stale or len(grouped) != len(set(grouped)):
        raise ValueError(
            f"TOP_LEVEL_GROUPS out of step with tsdive.__all__: add {ungrouped}, "
            f"remove {stale}, and list each name once"
        )
    parts = [
        "# tsdive\n",
        "The names in `tsdive.__all__`, grouped by task. "
        "Each one imports from `tsdive` directly.\n",
    ]
    for group, members in TOP_LEVEL_GROUPS.items():
        parts.append(f"## {group}\n")
        parts += [_directive(object_path("tsdive", n)) for n in members]
    return "\n".join(parts)


def _kind(name: str) -> str:
    """The section a subpackage export goes under, read off its PEP 8 name."""
    if name.isupper():
        return "Constants"
    return "Classes" if name[0].isupper() else "Functions"


def _subpackage_page(module: str) -> str:
    names = exports(module)
    parts = [
        f"# {module}\n",
        _directive(module, show_root_heading=False, members=False),
    ]
    for kind in ("Classes", "Functions", "Constants"):
        members = [n for n in names if _kind(n) == kind]
        if members:
            parts.append(f"## {kind}\n")
            parts += [_directive(object_path(module, n)) for n in members]
    return "\n".join(parts)


def api_markdown() -> dict[str, str]:
    """Page path -> markdown of the mkdocstrings directives for each public module."""
    return {
        page: _top_level_page() if module == "tsdive" else _subpackage_page(module)
        for module, page in API_PAGES.items()
    }


# ------------------------------------------------------------------ pages


def root_page(source: str) -> Rewrite:
    """A root markdown file with its links resolved for its site page."""
    return rewrite_links(_read(source), source, ROOT_PAGES[source])


def generated_pages() -> dict[str, tuple[Rewrite, str | None]]:
    """Page path -> (markdown, repository path of the edit link or None)."""
    pages: dict[str, tuple[Rewrite, str | None]] = {USAGE_PAGE: (usage_markdown(), "README.md")}
    for source, page in ROOT_PAGES.items():
        pages[page] = (root_page(source), source)
    pages[CLI_PAGE] = (Rewrite(cli_markdown()), "src/tsdive/cli.py")
    for page, text in api_markdown().items():
        pages[page] = (Rewrite(text), None)
    return pages


# ------------------------------------------------------------------ hooks


def on_files(files: Files, config: MkDocsConfig) -> Files:
    """Add the generated pages to the build."""
    from mkdocs.structure.files import File

    for page, (result, edit_source) in generated_pages().items():
        for target in result.missing:
            log.warning("%s links to %s, which is not in the repository", page, target)
        f = File.generated(config, page, content=result.text)
        if edit_source is not None:
            docs = Path(config.docs_dir).resolve().relative_to(ROOT).as_posix()
            f.edit_uri = posixpath.relpath(edit_source, docs)
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

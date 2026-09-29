"""The documentation site covers every CLI command and every public name.

The page generators in ``scripts/mkdocs_hooks.py`` run without the
``docs`` dependency group. The docstring check reads the package with
griffe, the static analyser the site renders from, and is skipped when
the group is not installed.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import importlib.util
import inspect
import posixpath
import re
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from tsdive import cli

ROOT = Path(__file__).resolve().parents[1]
HOOKS = ROOT / "scripts" / "mkdocs_hooks.py"
MODULES = ("tsdive", "tsdive.eval", "tsdive.switchback")
DIRECTIVE = re.compile(r"^::: (\S+)$", re.MULTILINE)
TARGET = re.compile(r"\]\(([^)\s]+)\)")
ROW = re.compile(r"^\| \[`([\w.]+)`\]\(([^)]+)\) \| (.+) \|$", re.MULTILINE)
ROLE = re.compile(r":(func|class|meth|attr|mod|data|exc|obj):`")

pytestmark = pytest.mark.docs


@pytest.fixture(scope="module")
def hooks() -> ModuleType:
    spec = importlib.util.spec_from_file_location("mkdocs_hooks", HOOKS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # The dataclass decorator looks its module up in sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(spec.name, None)


def _compared_strings(function: str) -> list[str]:
    """String constants that ``function`` in cli.py tests a value against with ``==``."""
    tree = ast.parse(Path(cli.__file__).read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function)
    return [
        node.comparators[0].value
        for node in ast.walk(fn)
        if isinstance(node, ast.Compare)
        and isinstance(node.ops[0], ast.Eq)
        and isinstance(node.comparators[0], ast.Constant)
        and isinstance(node.comparators[0].value, str)
    ]


def _api_pages(hooks: ModuleType) -> dict[str, str]:
    return {page: text for page, (text, _) in hooks.api_markdown().items()}


def _directives(hooks: ModuleType) -> list[str]:
    return [p for text in _api_pages(hooks).values() for p in DIRECTIVE.findall(text)]


def _runtime(path: str) -> object:
    """The object a dotted path names at runtime."""
    module, _, name = path.rpartition(".")
    try:
        return importlib.import_module(path)
    except ModuleNotFoundError:
        return getattr(importlib.import_module(module), name)


@pytest.fixture(scope="module")
def cli_pages(hooks: ModuleType) -> dict[str, str]:
    """The CLI index and command pages, examples run once for the module."""
    return hooks.cli_markdown()


def test_every_command_has_a_page_with_options_and_an_example(
    hooks: ModuleType, cli_pages: dict[str, str]
) -> None:
    dispatched = [c for c in _compared_strings("main") if c != "switchback"]
    commands = dispatched + [f"switchback {s}" for s in _compared_strings("cmd_switchback")]
    assert {"profile", "run", "switchback plan", "switchback analyze"} <= set(commands)
    assert set(commands) == set(hooks.cli_commands())
    index = cli_pages[hooks.CLI_INDEX]
    assert f"```text\n{cli.MAIN_DOC.strip()}\n```" in index
    for command in commands:
        page = hooks.cli_page(command)
        link = posixpath.relpath(page, posixpath.dirname(hooks.CLI_INDEX))
        assert f"[`tsdive {command}`]({link})" in index
        text = cli_pages[page]
        assert f"```text\nusage: tsdive {command} " in text, f"{page} has no usage line"
        for action in hooks.cli_parser(command)._actions:
            if not isinstance(action, argparse._HelpAction):
                positional = action.metavar or action.dest
                name = action.option_strings[0] if action.option_strings else positional
                assert f"| `{name}" in text, f"{page} has no options row for {name}"
                assert action.help, f"tsdive {command} {name} has no help text"
        example = re.search(r"## Example\n.*?```console\n(.*?)\n```", text, re.DOTALL)
        assert example, f"{page} has no example"
        lines = example.group(1).splitlines()
        assert lines[0].startswith(f"$ tsdive {command}"), f"{page} runs {lines[0]}"
        ran = next(i for i, line in enumerate(lines) if not line.endswith("\\"))
        assert lines[ran + 1 :], f"{page} shows no output"


def test_api_index_links_every_export_to_its_own_page(hooks: ModuleType) -> None:
    pages = _api_pages(hooks)
    rows = {name: (link, text) for name, link, text in ROW.findall(pages[hooks.API_INDEX])}
    exported = 0
    for name in MODULES:
        module = importlib.import_module(name)
        for export in module.__all__:
            exported += 1
            public = f"{name}.{export}"
            assert public in rows, f"{public} is not in the API index"
            link, text = rows[public]
            assert text.strip(), f"{public} has no summary"
            page = posixpath.normpath(posixpath.join(hooks.API_DIR, link))
            assert page in pages, f"{public} links to {link}, which is not generated"
            (directive,) = DIRECTIVE.findall(pages[page])
            assert _runtime(directive) is getattr(module, export), f"{page} renders {directive}"
            assert "[API reference](index.md)" in pages[page]
    assert len(rows) == exported


def test_group_pages_list_their_group(hooks: ModuleType) -> None:
    pages = _api_pages(hooks)
    entries = hooks.api_entries()
    for group, (slug, _) in hooks.GROUP_PAGES.items():
        listed = [name for name, _, _ in ROW.findall(pages[f"{hooks.API_DIR}/{slug}.md"])]
        assert listed == [e.public for e in entries if e.group == group]


def test_every_public_function_has_examples() -> None:
    for name in MODULES:
        module = importlib.import_module(name)
        for export in module.__all__:
            obj = getattr(module, export)
            if inspect.isfunction(obj):
                doc = inspect.getdoc(obj) or ""
                assert "\nExamples:\n" in doc, f"{name}.{export} has no Examples section"


@pytest.fixture(scope="module")
def package() -> Any:
    """The package as griffe, the static analyser the site renders from, reads it."""
    griffe = pytest.importorskip("griffe")
    return griffe.load("tsdive", search_paths=[str(ROOT / "src")], docstring_parser="google")


def _docstrings(obj: Any, members: bool) -> Iterator[tuple[str, str]]:
    """(path, docstring) of ``obj`` and, for a class, of its public members."""
    target = obj.final_target if obj.is_alias else obj
    if target.docstring:
        yield target.path, target.docstring.value
    if members and target.kind.value == "class":
        for name, member in target.members.items():
            if not name.startswith("_"):
                yield from _docstrings(member, members=False)


def test_no_rendered_docstring_carries_a_sphinx_role(hooks: ModuleType, package: Any) -> None:
    for path in _directives(hooks):
        obj = package[path.removeprefix("tsdive.")]
        for where, text in _docstrings(obj, members=path not in MODULES):
            assert not ROLE.search(text), f"{where} renders a Sphinx role: {ROLE.search(text)}"


def test_every_rendered_export_has_a_docstring(hooks: ModuleType, package: Any) -> None:
    for path in _directives(hooks):
        if path in MODULES:
            continue
        obj = package[path.removeprefix("tsdive.")]
        target = obj.final_target if obj.is_alias else obj
        runtime = _runtime(path)
        expected = (
            "class"
            if inspect.isclass(runtime)
            else "function"
            if inspect.isfunction(runtime)
            else "attribute"
        )
        assert target.kind.value == expected, f"{path} renders a {target.kind.value}"
        if path != "tsdive.__version__":
            assert target.docstring and target.docstring.value.strip(), f"{path} has no docstring"


def _static_pages() -> dict[str, str]:
    """Page path under docs/ -> text, for every markdown file in docs/."""
    docs = ROOT / "docs"
    return {
        p.relative_to(docs).as_posix(): p.read_text(encoding="utf-8")
        for p in sorted(docs.rglob("*.md"))
    }


def _assert_links_stay_on_site(page: str, text: str, site_pages: set[str]) -> None:
    for target in TARGET.findall(text):
        if re.match(r"^[a-z]+:", target) or target.startswith("#"):
            continue
        path = target.partition("#")[0]
        resolved = posixpath.normpath(posixpath.join(posixpath.dirname(page), path))
        assert resolved in site_pages, f"{page} links to {target}"


def test_no_link_on_a_docs_page_leaves_the_site(hooks: ModuleType) -> None:
    pages = _static_pages()
    site_pages = set(pages) | hooks.generated_paths()
    rewritten = 0
    for page, text in pages.items():
        if page == "index.md":
            result = hooks.index_markdown(text)
        else:
            result = hooks.rewrite_links(text, f"docs/{page}", page)
        assert not result.missing, f"{page}: {result.missing}"
        _assert_links_stay_on_site(page, result.text, site_pages)
        rewritten += len(result.to_github)
    assert rewritten > 0


def test_a_readme_section_the_usage_page_carries_resolves_to_it(hooks: ModuleType) -> None:
    roadmap = (ROOT / "docs" / "ROADMAP.md").read_text(encoding="utf-8")
    assert "](../README.md#what-it-checks)" in roadmap
    result = hooks.rewrite_links(roadmap, "docs/ROADMAP.md", "ROADMAP.md")
    assert "](usage.md#what-it-checks)" in result.text
    assert "\n## What it checks\n" in hooks.usage_markdown().text


def test_generated_pages_link_only_to_site_pages_or_urls(hooks: ModuleType) -> None:
    pages = hooks.generated_pages()
    assert set(pages) == hooks.generated_paths()
    site_pages = set(_static_pages()) | hooks.generated_paths()
    for page, generated in pages.items():
        assert not generated.result.missing, f"{page}: {generated.result.missing}"
        _assert_links_stay_on_site(page, generated.result.text, site_pages)


def test_index_carries_the_readme_introduction(hooks: ModuleType) -> None:
    text = hooks.index_markdown((ROOT / "docs" / "index.md").read_text(encoding="utf-8")).text
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    first = hooks.readme_section("intro").splitlines()[0]
    assert first in readme
    assert first in text
    assert "<!-- readme:" not in text


# The user guide: each page must exist and say something.
REQUIRED_PAGES = [
    "index.md",
    "getting-started.md",
    *(
        f"guide/concepts/{name}.md"
        for name in (
            "windows-and-time",
            "archives-and-metadata",
            "sampling-contract",
            "quality-and-severity",
            "coverage-and-gaps",
            "clipping-and-censoring",
            "baselines",
            "spc-rules",
            "mspc",
            "compare",
            "refusals",
        )
    ),
    *(
        f"guide/output/{name}.md"
        for name in ("profile", "screen", "spc", "mspc", "compare", "switchback")
    ),
    *(
        f"guide/howto/{name}.md"
        for name in (
            "historian-export",
            "baseline-window",
            "many-tags",
            "python",
            "mcp",
            "switchback-trial",
        )
    ),
    "reference/errors.md",
    "reference/glossary.md",
]
GLOSSARY_LINK = re.compile(r"\]\(([^)#\s]*glossary\.md)#([\w-]+)\)")


def test_every_required_guide_page_exists_and_has_content() -> None:
    pages = _static_pages()
    for page in REQUIRED_PAGES:
        assert page in pages, f"docs/{page} is missing"
        body = [line for line in pages[page].splitlines() if line and not line.startswith("#")]
        assert len(body) >= 5, f"docs/{page} is empty"


def test_errors_page_covers_every_typed_error() -> None:
    import tsdive.errors

    text = _static_pages()["reference/errors.md"]
    classes = [
        name
        for name, obj in vars(tsdive.errors).items()
        if inspect.isclass(obj) and issubclass(obj, tsdive.errors.TSDiveError)
    ]
    assert "SchemaError" in classes
    for name in classes:
        assert f"\n### {name}\n" in text, f"docs/reference/errors.md has no section for {name}"


def test_glossary_defines_every_term_a_page_links_to_it() -> None:
    pages = _static_pages()
    defined = set(re.findall(r"^## .+ \{#([\w-]+)\}$", pages["reference/glossary.md"], re.M))
    assert {"tag", "censored", "sampling-contract"} <= defined
    for page, text in pages.items():
        for target, anchor in GLOSSARY_LINK.findall(text):
            resolved = posixpath.normpath(posixpath.join(posixpath.dirname(page), target))
            if resolved == "reference/glossary.md":
                assert anchor in defined, f"{page} links to an undefined glossary term {anchor!r}"


def test_every_guide_block_parses(hooks: ModuleType) -> None:
    for page, text in _static_pages().items():
        for _, block in hooks.guide_blocks(text):
            if block.runs:
                assert hooks.guide_commands(block.body), f"{page} has an empty tsdive block"
                assert set(block.options) <= {"exit", "lines", "tail"}, f"{page}: {block.options}"
            else:
                assert set(block.options) <= {"file", "show", "lines"}, f"{page}: {block.options}"

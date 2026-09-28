"""The documentation site covers every CLI command and every public name.

The page generators in ``scripts/mkdocs_hooks.py`` run without the
``docs`` dependency group. The docstring check reads the package with
griffe, the static analyser the site renders from, and is skipped when
the group is not installed.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
import posixpath
import re
import sys
from pathlib import Path
from types import ModuleType

import pytest

from tsdive import cli

ROOT = Path(__file__).resolve().parents[1]
HOOKS = ROOT / "scripts" / "mkdocs_hooks.py"
MODULES = ("tsdive", "tsdive.eval", "tsdive.switchback")
DIRECTIVE = re.compile(r"^::: (\S+)$", re.MULTILINE)
SECTION = re.compile(r"^## (.+?)\n\n```text\n(.*?)\n```$", re.MULTILINE | re.DOTALL)
TARGET = re.compile(r"\]\(([^)\s]+)\)")

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


def _directives(hooks: ModuleType) -> list[str]:
    return [p for text in hooks.api_markdown().values() for p in DIRECTIVE.findall(text)]


def _runtime(path: str) -> object:
    """The object a dotted path names at runtime."""
    module, _, name = path.rpartition(".")
    try:
        return importlib.import_module(path)
    except ModuleNotFoundError:
        return getattr(importlib.import_module(module), name)


def test_cli_page_holds_every_command(hooks: ModuleType) -> None:
    commands = _compared_strings("main")
    subcommands = [f"switchback {s}" for s in _compared_strings("cmd_switchback")]
    assert {"profile", "switchback", "run"} <= set(commands)
    assert subcommands == ["switchback plan", "switchback analyze"]
    sections = dict(SECTION.findall(hooks.cli_markdown()))
    assert sections["tsdive"] == cli.MAIN_DOC.strip()
    assert sections["switchback"] == cli.SWITCHBACK_DOC.strip()
    for command in [c for c in commands if c != "switchback"] + subcommands:
        assert command in sections, f"no CLI reference section for {command!r}"
        assert sections[command].startswith(f"usage: tsdive {command} ")


def test_api_pages_hold_every_export(hooks: ModuleType) -> None:
    directives = _directives(hooks)
    assert len(directives) == len(set(directives))
    for name in MODULES:
        module = importlib.import_module(name)
        for export in module.__all__:
            hits = [
                d
                for d in directives
                if d.rpartition(".")[2] == export and _runtime(d) is getattr(module, export)
            ]
            assert hits, f"{name}.{export} is on no API page"


def test_every_rendered_export_has_a_docstring(hooks: ModuleType) -> None:
    griffe = pytest.importorskip("griffe")
    package = griffe.load("tsdive", search_paths=[str(ROOT / "src")], docstring_parser="google")
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


def test_no_link_on_a_docs_page_leaves_the_site(hooks: ModuleType) -> None:
    rewritten = 0
    for page in sorted((ROOT / "docs").glob("*.md")):
        text = page.read_text(encoding="utf-8")
        if page.name == "index.md":
            result = hooks.index_markdown(text)
        else:
            result = hooks.rewrite_links(text, f"docs/{page.name}", page.name)
        assert not result.missing, f"{page.name}: {result.missing}"
        assert len(result.to_github) == text.count("](../"), page.name
        assert "](../" not in result.text, page.name
        rewritten += len(result.to_github)
    assert rewritten > 0


def test_generated_pages_link_only_to_site_pages_or_urls(hooks: ModuleType) -> None:
    pages = hooks.generated_pages()
    assert set(pages) == hooks.GENERATED
    site_pages = {p.name for p in (ROOT / "docs").glob("*.md")} | hooks.GENERATED
    for page, (result, _) in pages.items():
        assert not result.missing, f"{page}: {result.missing}"
        for target in TARGET.findall(result.text):
            if re.match(r"^[a-z]+:", target) or target.startswith("#"):
                continue
            path = target.partition("#")[0]
            resolved = posixpath.normpath(posixpath.join(posixpath.dirname(page), path))
            assert resolved in site_pages, f"{page} links to {target}"


def test_index_carries_the_readme_introduction(hooks: ModuleType) -> None:
    text = hooks.index_markdown((ROOT / "docs" / "index.md").read_text(encoding="utf-8")).text
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    first = hooks.readme_section("intro").splitlines()[0]
    assert first in readme
    assert first in text
    assert "<!-- readme:" not in text

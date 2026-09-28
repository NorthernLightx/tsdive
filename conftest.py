"""Working directory for the docstring examples in ``src/tsdive``.

Each doctest runs in a fresh directory holding ``data/demo`` (written by
``scripts/make_demo_archive.py``) and ``data/switchback_demo`` (written by
``examples/switchback/make_trial.py``), so an example reads the paths the
README uses and a file one example writes is not there for the next.
Other tests do not use it.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import shutil
import sys
from pathlib import Path

import pytest
from _pytest.doctest import DoctestItem

ROOT = Path(__file__).resolve().parent
DATA_SCRIPTS = (
    ROOT / "scripts" / "make_demo_archive.py",
    ROOT / "examples" / "switchback" / "make_trial.py",
)


def _run_script(script: Path) -> None:
    """Run ``main()`` of ``script`` in the current directory, discarding its output."""
    spec = importlib.util.spec_from_file_location(f"doctest_data_{script.stem}", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # The dataclass decorator looks its module up in sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    with contextlib.redirect_stdout(io.StringIO()):
        module.main()


@pytest.fixture(scope="session")
def doctest_data(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The ``data/`` directory the scripts write, built once per session."""
    root = tmp_path_factory.mktemp("doctest-data")
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(root)
        for script in DATA_SCRIPTS:
            _run_script(script)
    return root / "data"


@pytest.fixture(autouse=True)
def _doctest_directory(request: pytest.FixtureRequest) -> None:
    if not isinstance(request.node, DoctestItem):
        return
    work: Path = request.getfixturevalue("tmp_path")
    shutil.copytree(request.getfixturevalue("doctest_data"), work / "data")
    request.getfixturevalue("monkeypatch").chdir(work)

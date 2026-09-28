"""Working directory for the docstring and docs page examples.

Each doctest runs in a fresh directory holding ``data/demo`` and
``data/switchback_demo``, the archives ``tsdive demo data`` writes, so an
example reads the paths the README uses and a file one example writes is
not there for the next. Other tests do not use it.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from _pytest.doctest import DoctestItem

from tsdive.demo import write_demo_data


@pytest.fixture(scope="session")
def doctest_data(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The ``data/`` directory ``tsdive demo data`` writes, built once per session."""
    data = tmp_path_factory.mktemp("doctest-data") / "data"
    write_demo_data(data)
    return data


@pytest.fixture(autouse=True)
def _doctest_directory(request: pytest.FixtureRequest) -> None:
    if not isinstance(request.node, DoctestItem):
        return
    work: Path = request.getfixturevalue("tmp_path")
    shutil.copytree(request.getfixturevalue("doctest_data"), work / "data")
    request.getfixturevalue("monkeypatch").chdir(work)

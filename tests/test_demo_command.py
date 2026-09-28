"""`tsdive demo` writes the archives the docs examples read, and never overwrites."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

import tsdive
from tsdive.cli import main
from tsdive.demo import DEFAULT_DIR, demo_archives, write_demo_data
from tsdive.store.tagstore import write_tag

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    "demo/fic101_demo.parquet",
    "demo/tic101_demo.parquet",
    "switchback_demo/ti201.parquet",
    "switchback_demo/fi200.parquet",
    "switchback_demo/tt001.parquet",
]


def test_demo_writes_every_archive_and_names_the_next_command(tmp_path, capsys):
    target = tmp_path / "trial"
    assert main(["demo", str(target)]) == 0
    out = capsys.readouterr().out.splitlines()
    assert all((target / name).exists() for name in FILES)
    assert out[0].startswith("wrote     ") and out[0].endswith("   562 samples")
    assert out[-1] == f"next      tsdive profile {(target / FILES[0]).as_posix()}"


def test_demo_defaults_to_a_directory_under_the_current_one(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["demo"]) == 0
    assert (tmp_path / DEFAULT_DIR / FILES[0]).exists()
    assert capsys.readouterr().out.splitlines()[-1] == (
        f"next      tsdive profile {DEFAULT_DIR}/{FILES[0]}"
    )


def test_demo_refuses_a_directory_that_holds_an_archive(tmp_path, capsys):
    (tmp_path / "switchback_demo").mkdir()
    (tmp_path / "switchback_demo" / "tt001.parquet").write_bytes(b"kept")
    assert main(["demo", str(tmp_path)]) == 2
    assert "tt001.parquet already exists" in capsys.readouterr().err
    assert (tmp_path / "switchback_demo" / "tt001.parquet").read_bytes() == b"kept"
    assert not (tmp_path / FILES[0]).exists()


def test_demo_data_is_what_the_repository_scripts_write(tmp_path):
    """The README and the scripts under scripts/ and examples/ read the same bytes."""
    written = write_demo_data(tmp_path / "package")
    for name, script in (
        ("make_demo_archive", ROOT / "scripts" / "make_demo_archive.py"),
        ("make_trial", ROOT / "examples" / "switchback" / "make_trial.py"),
    ):
        spec = importlib.util.spec_from_file_location(f"test_demo_{name}", script)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    reference = tmp_path / "reference"
    for name, (frame, meta) in demo_archives().items():
        write_tag(reference / name, frame, meta)
    for path in written:
        name = path.relative_to(tmp_path / "package").as_posix()
        assert path.read_bytes() == (reference / name).read_bytes(), name


def test_demo_profile_reads_the_readme_numbers(tmp_path):
    (path, *_) = write_demo_data(tmp_path)
    p = tsdive.profile(path, "2024-03-30T20:00:00Z/2024-03-31T06:00:00Z")
    assert round(p.physics.coverage.coverage, 3) == 0.933
    with pytest.raises(FileExistsError):
        write_demo_data(tmp_path)

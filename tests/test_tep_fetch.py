"""The TEP fetcher, exercised offline against a monkeypatched Dataverse.

No network. What is asserted is that the script states an honest byte
total before asking, treats a closed stdin as a refusal, and throws away
bytes whose MD5 does not match the one the API published.
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPTS = Path(__file__).parents[1] / "scripts"


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fetch_tep = _load("fetch_tep")

FAULT_FREE_TRAIN = b"fault-free training payload"
FAULTY_TRAIN = b"faulty training payload"
FAULT_FREE_TEST = b"fault-free testing payload"

BODIES = {
    3031241: FAULT_FREE_TRAIN,
    3031242: FAULTY_TRAIN,
    3031240: FAULT_FREE_TEST,
}


def _payload(*, md5_of: dict[int, str] | None = None, drop_md5: bool = False) -> dict:
    """A Dataverse dataset payload shaped like the real one."""
    import hashlib

    entries = []
    for datafile_id, name in (
        (3031241, "TEP_FaultFree_Training.RData"),
        (3031242, "TEP_Faulty_Training.RData"),
        (3031240, "TEP_FaultFree_Testing.RData"),
    ):
        body = BODIES[datafile_id]
        digest = (md5_of or {}).get(datafile_id, hashlib.md5(body).hexdigest())
        data = {
            "id": datafile_id,
            "filename": name,
            "filesize": len(body),
        }
        if not drop_md5:
            data["md5"] = digest
            data["checksum"] = {"type": "MD5", "value": digest}
        entries.append({"dataFile": data})
    return {
        "status": "OK",
        "data": {
            "latestVersion": {
                "versionNumber": 1,
                "versionMinorNumber": 0,
                "license": None,
                "files": entries,
            }
        },
    }


class _Response:
    def __init__(self, body: bytes) -> None:
        self._buf = io.BytesIO(body)

    def read(self, n: int = -1) -> bytes:
        return self._buf.read(n)

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


def _router(payload: dict, seen: list[str] | None = None):
    """urlopen stand-in: dataset metadata on one URL, file bytes on the rest."""

    def _open(req, *args, **kwargs):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        if seen is not None:
            seen.append(url)
        if "/api/datasets/" in url:
            return _Response(json.dumps(payload).encode("utf-8"))
        datafile_id = int(url.rsplit("/", maxsplit=1)[-1])
        return _Response(BODIES[datafile_id])

    return _open


def test_default_selection_is_the_training_pair_and_its_total_is_stated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = _payload()
    monkeypatch.setattr(fetch_tep.urllib.request, "urlopen", _router(payload))
    monkeypatch.setattr("builtins.input", lambda prompt="": "n")

    assert fetch_tep.main(["--dest", str(tmp_path / "tep")]) == 1
    out = capsys.readouterr().out
    expected = len(FAULT_FREE_TRAIN) + len(FAULTY_TRAIN)
    assert f"selected 2 file(s): {expected:,d} bytes total" in out
    # The testing file is in the dataset listing but not in the selection.
    assert "dataset holds 3 file(s)" in out
    assert "TEP_FaultFree_Testing.RData  id=" not in out


def test_all_selects_every_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    payload = _payload()
    monkeypatch.setattr(fetch_tep.urllib.request, "urlopen", _router(payload))
    files = fetch_tep.dataset_files(payload)
    assert len(fetch_tep.select(files, names=None)) == 3
    assert [f.filename for f in fetch_tep.select(files, names=("TEP_Faulty_Training.RData",))] == [
        "TEP_Faulty_Training.RData"
    ]


def test_eof_on_the_prompt_aborts_without_a_traceback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(fetch_tep.urllib.request, "urlopen", _router(_payload()))

    def _eof(prompt: str = "") -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", _eof)
    dest = tmp_path / "tep"
    assert fetch_tep.main(["--dest", str(dest)]) == 1
    assert "aborted" in capsys.readouterr().out
    assert not dest.exists()


def test_md5_mismatch_discards_the_bytes_and_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A file that hashes to something else is deleted, not kept and reported."""
    payload = _payload(md5_of={3031241: "0" * 32})
    monkeypatch.setattr(fetch_tep.urllib.request, "urlopen", _router(payload))
    dest = tmp_path / "tep"

    assert fetch_tep.main(["--dest", str(dest), "--yes"]) == 1
    out = capsys.readouterr().out
    assert "MD5 mismatch" in out
    assert "0" * 32 in out
    assert list(dest.glob("*.RData")) == []
    assert list(dest.glob("*.part")) == []
    assert not (dest / "MANIFEST.sha256.json").exists()


def test_a_file_the_api_does_not_checksum_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        fetch_tep.urllib.request, "urlopen", _router(_payload(drop_md5=True))
    )
    assert fetch_tep.main(["--dest", str(tmp_path / "tep"), "--yes"]) == 1
    assert "states no MD5" in capsys.readouterr().out


def test_a_verified_fetch_writes_both_sidecars(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import hashlib

    monkeypatch.setattr(fetch_tep.urllib.request, "urlopen", _router(_payload()))
    dest = tmp_path / "tep"
    assert fetch_tep.main(["--dest", str(dest), "--yes"]) == 0

    assert (dest / "TEP_FaultFree_Training.RData").read_bytes() == FAULT_FREE_TRAIN
    manifest = json.loads((dest / "MANIFEST.sha256.json").read_text("utf-8"))
    assert set(manifest) == {"TEP_FaultFree_Training.RData", "TEP_Faulty_Training.RData"}
    record = manifest["TEP_Faulty_Training.RData"]
    assert record["sha256"] == hashlib.sha256(FAULTY_TRAIN).hexdigest()
    assert record["md5"] == hashlib.md5(FAULTY_TRAIN).hexdigest()
    assert record["datafile_id"] == 3031242

    fetch = json.loads((dest / "FETCH.json").read_text("utf-8"))
    assert fetch["doi"] == "doi:10.7910/DVN/6C3JR1"
    assert fetch["version"] == "1.0"
    assert fetch["n_files"] == 2
    assert fetch["bytes"] == len(FAULT_FREE_TRAIN) + len(FAULTY_TRAIN)
    assert fetch["datafile_ids"]["TEP_FaultFree_Training.RData"] == 3031241


def test_a_second_run_reuses_verified_bytes_instead_of_refetching(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = _payload()
    dest = tmp_path / "tep"
    monkeypatch.setattr(fetch_tep.urllib.request, "urlopen", _router(payload))
    assert fetch_tep.main(["--dest", str(dest), "--yes"]) == 0

    seen: list[str] = []
    monkeypatch.setattr(fetch_tep.urllib.request, "urlopen", _router(payload, seen))
    assert fetch_tep.main(["--dest", str(dest), "--yes"]) == 0
    assert [u for u in seen if "/api/access/" in u] == []

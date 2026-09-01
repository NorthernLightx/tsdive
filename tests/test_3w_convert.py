"""3W fetch/convert scripts, exercised offline against a synthetic fixture.

No network: the 3W-shaped parquet is built here, and the fetcher's
network calls are monkeypatched. What is asserted is the shape of the
decisions the scripts make on 3W's real quirks - a tz-naive index, an
entirely-NaN "absent" variable, per-row Int16 labels with an unlabelled
prefix, and a frozen sensor that carries no quality code at all.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

import tsdive
from tsdive.detectors.flatline import NotAssessed
from tsdive.store.identity import Role
from tsdive.store.quality import Severity
from tsdive.store.tagstore import meta_from_parquet

SCRIPTS = Path(__file__).parents[1] / "scripts"

DESCRIPTIONS = {
    "P-PDG": ("Downhole pressure at the PDG (permanent downhole gauge) [Pa]", "Pa"),
    "P-TPT": ("Subsea Xmas-tree pressure at the TPT [Pa]", "Pa"),
    "QGL": ("Gas lift flow rate [m3/s]", "m3/s"),
    "ESTADO-M1": ("State of the PMV (production master valve) [0, 0.5, or 1]", None),
}


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


convert_3w = _load("convert_3w")
fetch_3w = _load("fetch_3w")

N_ROWS = 1801  # 1800 s at 1 Hz, so three 600 s windows fit
UNLABELLED = 300
FREEZE_AT = 1200


def write_source(tmp_path: Path) -> Path:
    """A 3W-shaped instance: naive index, an absent column, a tag that freezes."""
    index = pd.DatetimeIndex(
        pd.date_range("2020-01-01 00:00:00", periods=N_ROWS, freq="1s"), name="timestamp"
    )
    moving = [100.0 + i for i in range(N_ROWS)]
    freezing = [10.0 + i for i in range(FREEZE_AT)] + [float(10 + FREEZE_AT)] * (
        N_ROWS - FREEZE_AT
    )
    labels = [pd.NA] * UNLABELLED + [0] * (N_ROWS - UNLABELLED)
    frame = pd.DataFrame(
        {
            "P-TPT": moving,
            "P-PDG": freezing,
            "QGL": [float("nan")] * N_ROWS,  # absent variable: entirely NaN
            "ESTADO-M1": [0.0] * (N_ROWS // 2) + [1.0] * (N_ROWS - N_ROWS // 2),
            "class": pd.array(labels, dtype="Int16"),
            "state": pd.array(labels, dtype="Int16"),
        },
        index=index,
    )
    src = tmp_path / "dataset" / "0" / "WELL-00099_20200101000000.parquet"
    src.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(src)
    return src


@pytest.fixture()
def converted(tmp_path: Path) -> tuple[Path, dict]:
    src = write_source(tmp_path)
    out_root = tmp_path / "archives"
    record = convert_3w.convert_instance(
        src,
        out_root,
        tz="UTC",
        descriptions=DESCRIPTIONS,
        source_sha256="0" * 64,
        force=False,
    )
    return out_root / src.stem, record


def test_absent_variable_gets_no_archive(converted: tuple[Path, dict]) -> None:
    out_dir, record = converted
    assert record["absent_variables"] == ["QGL"]
    assert not (out_dir / "QGL.parquet").exists()
    written = sorted(p.stem for p in out_dir.glob("*.parquet"))
    assert written == ["ESTADO-M1", "LABEL_class", "LABEL_state", "P-PDG", "P-TPT"]


def test_instance_json_records_provenance_and_the_tz_assumption(
    converted: tuple[Path, dict],
) -> None:
    out_dir, record = converted
    on_disk = json.loads((out_dir / "INSTANCE.json").read_text("utf-8"))
    assert on_disk["timestamp_tz_assumed"] == "UTC"
    assert on_disk["instance_source"] == "REAL"
    assert on_disk["well"] == "WELL-00099"
    assert on_disk["folder_label"] == "0"
    assert on_disk["n_rows"] == N_ROWS
    assert on_disk["unlabelled_prefix_rows"] == UNLABELLED
    assert on_disk["row_label_histogram"]["class"] == {
        "0": N_ROWS - UNLABELLED,
        "NA": UNLABELLED,
    }
    assert on_disk["source_sha256"] == "0" * 64
    assert record["absent_variables"] == on_disk["absent_variables"]


def test_quality_is_assumed_and_its_codes_are_declared(converted: tuple[Path, dict]) -> None:
    out_dir, _ = converted
    meta = meta_from_parquet(out_dir / "P-PDG.parquet")
    assert meta.quality_assumed is True
    assert meta.quality_codes == {"GOOD_ASSUMED": "GOOD", "NO_DATA": "BAD"}
    assert meta.unit_raw == "Pa"
    assert meta.eng_range is None
    assert meta.sample_rate_s == 1.0
    assert meta.role is Role.PV
    assert meta.asset == "WELL-00099"
    assert meta.loop_id == "WELL-00099_20200101000000"
    profile = tsdive.profile(out_dir / "P-PDG.parquet")
    assert "quality ASSUMED at ingest" in profile.render()


def test_label_columns_become_mode_archives(converted: tuple[Path, dict]) -> None:
    out_dir, _ = converted
    meta = meta_from_parquet(out_dir / "LABEL_class.parquet")
    assert meta.role is Role.MODE
    frame = pd.read_parquet(out_dir / "LABEL_class.parquet")
    assert frame["value"].iloc[0] is None
    assert frame["quality"].iloc[0] == "NO_DATA"
    assert frame["value"].iloc[-1] == "0"
    assert frame["quality"].iloc[-1] == "GOOD_ASSUMED"
    profile = tsdive.profile(out_dir / "LABEL_class.parquet")
    counts = profile.physics.severity_counts
    assert counts[Severity.BAD] == UNLABELLED


def test_estado_tags_keep_their_float_states_and_carry_no_unit(
    converted: tuple[Path, dict],
) -> None:
    out_dir, _ = converted
    meta = meta_from_parquet(out_dir / "ESTADO-M1.parquet")
    assert meta.role is Role.MODE
    assert meta.unit_raw is None
    profile = tsdive.profile(out_dir / "ESTADO-M1.parquet")
    assert profile.physics.severity_counts[Severity.GOOD] == N_ROWS


def test_frozen_tag_fires_a_flatline_signal(converted: tuple[Path, dict]) -> None:
    """The last 600 s window, referenced against the earlier ones."""
    out_dir, _ = converted
    start = pd.Timestamp("2020-01-01 00:20:00+00:00")
    end = pd.Timestamp("2020-01-01 00:30:00+00:00")
    profile = tsdive.profile(
        out_dir / "P-PDG.parquet", f"{start.isoformat()}/{end.isoformat()}", flatline=True
    )
    assert profile.flatline is not None
    assert profile.flatline.fired is True
    fired = {s.signal for s in profile.flatline.signals if s.fired}
    assert "time_since_last_actual_change" in fired
    rendered = profile.render()
    assert "FLATLINE SUSPECTED" in rendered
    assert "[FIRED]" in rendered


def test_estado_and_label_tags_are_not_assessed_for_flatline(
    converted: tuple[Path, dict],
) -> None:
    """A valve state held for half the record is the valve, not the sensor."""
    out_dir, _ = converted
    window = "2020-01-01T00:20:00+00:00/2020-01-01T00:30:00+00:00"
    for point in ("ESTADO-M1", "LABEL_class"):
        verdict = tsdive.profile(out_dir / f"{point}.parquet", window, flatline=True).flatline
        assert verdict is not None, point
        assert verdict.fired is False, point
        assert verdict.not_assessed is NotAssessed.STATE_VALUED_TAG, point


def test_the_3w_units_resolve_off_the_converted_archive(converted: tuple[Path, dict]) -> None:
    """`dataset.ini` spells pressure `Pa` and flow `m3/s`; both must resolve."""
    out_dir, _ = converted
    physics = tsdive.profile(out_dir / "P-PDG.parquet").physics
    assert physics.unit.raw == "Pa"
    assert physics.unit.canonical == "pascals"
    assert "units     Pa -> pascals" in tsdive.profile(
        out_dir / "P-PDG.parquet"
    ).render()


def test_moving_tag_over_the_same_window_does_not_fire(converted: tuple[Path, dict]) -> None:
    out_dir, _ = converted
    profile = tsdive.profile(
        out_dir / "P-TPT.parquet",
        "2020-01-01T00:20:00+00:00/2020-01-01T00:30:00+00:00",
        flatline=True,
    )
    assert profile.flatline is not None
    assert profile.flatline.fired is False


def test_conversion_is_idempotent(tmp_path: Path) -> None:
    src = write_source(tmp_path)
    out_root = tmp_path / "archives"
    kwargs = {"tz": "UTC", "descriptions": DESCRIPTIONS, "source_sha256": None, "force": False}
    first = convert_3w.convert_instance(src, out_root, **kwargs)
    again = convert_3w.convert_instance(src, out_root, **kwargs)
    assert first["skipped"] is False
    assert again["skipped"] is True
    assert again["archives_written"] == first["archives_written"]


def test_estado_bracket_is_not_read_as_a_unit(tmp_path: Path) -> None:
    ini = tmp_path / "dataset.ini"
    ini.write_text(
        "[PARQUET_FILE_PROPERTIES]\n"
        "P-PDG = Downhole pressure at the PDG [Pa]\n"
        "ESTADO-M1 = State of the PMV [0, 0.5, or 1]\n"
        "class = Label of the observation\n",
        encoding="utf-8",
    )
    descriptions = convert_3w.read_variable_descriptions(ini)
    assert descriptions["P-PDG"][1] == "Pa"
    assert descriptions["ESTADO-M1"][1] is None
    assert descriptions["class"][1] is None


class _Response:
    def __init__(self, headers: dict[str, str]) -> None:
        self.headers = headers

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


def test_content_length_is_none_when_the_header_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No header means unknown, not zero: '0 bytes' would be a made-up number."""
    monkeypatch.setattr(fetch_3w.urllib.request, "urlopen", lambda *a, **k: _Response({}))
    assert fetch_3w.content_length("https://example.invalid/x.zip") is None
    monkeypatch.setattr(
        fetch_3w.urllib.request, "urlopen", lambda *a, **k: _Response({"Content-Length": "17"})
    )
    assert fetch_3w.content_length("https://example.invalid/x.zip") == 17


def test_eof_on_the_prompt_aborts_without_a_traceback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(fetch_3w, "head_commit", lambda ref="main": "a" * 40)
    monkeypatch.setattr(
        fetch_3w,
        "dataset_blobs",
        lambda sha: [fetch_3w.Blob("dataset/0/WELL-00099_1.parquet", 5)],
    )

    def _eof(prompt: str = "") -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", _eof)
    assert fetch_3w.main(["--dest", str(tmp_path / "3w")]) == 1
    assert "aborted" in capsys.readouterr().out
    assert not (tmp_path / "3w").exists()


def test_only_real_and_class_filters_never_drop_the_aux_files() -> None:
    blobs = [
        fetch_3w.Blob("dataset/dataset.ini", 1),
        fetch_3w.Blob("dataset/LICENSE-CC-BY", 2),
        fetch_3w.Blob("dataset/README.md", 3),
        fetch_3w.Blob("dataset/0/WELL-00099_1.parquet", 4),
        fetch_3w.Blob("dataset/0/SIMULATED_00099_1.parquet", 5),
        fetch_3w.Blob("dataset/1/WELL-00098_1.parquet", 6),
    ]
    real = fetch_3w.select(blobs, only_real=True, classes=None)
    assert [b.path for b in real] == [
        "dataset/dataset.ini",
        "dataset/LICENSE-CC-BY",
        "dataset/README.md",
        "dataset/0/WELL-00099_1.parquet",
        "dataset/1/WELL-00098_1.parquet",
    ]
    folder0 = fetch_3w.select(blobs, only_real=True, classes={"0"})
    assert [b.path for b in folder0 if b.path.endswith(".parquet")] == [
        "dataset/0/WELL-00099_1.parquet"
    ]

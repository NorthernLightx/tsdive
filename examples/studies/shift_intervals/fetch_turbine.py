"""Opt-in fetcher for the Turbine Upgrade Dataset (Zenodo record 5516556).

Not run by CI, not run by default. Downloads ``Turbine_Upgrade_Dataset.zip``
from the pinned Zenodo record into ``<dest>``, checks its size and sha256
against the pinned values before keeping it, extracts the CSVs into
``<dest>/raw/``, hashes every file into ``MANIFEST.sha256.json`` and records
the record, the URL and the fetch time in ``FETCH.json``. A zip whose sha256
already matches is not downloaded again.

Data licence upstream: CC BY 4.0. See docs/DATA.md.

Usage:
    python examples/studies/shift_intervals/fetch_turbine.py --dest data/turbine_upgrade
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

RECORD = "5516556"
ZIP_NAME = "Turbine_Upgrade_Dataset.zip"
URL = f"https://zenodo.org/records/{RECORD}/files/{ZIP_NAME}?download=1"
ZIP_BYTES = 4_891_261
ZIP_SHA256 = "c302fbdc5e68989dc0792d7125277dd76a3ca20f27a96959d1238a294146511e"
LICENCE = "CC-BY-4.0"


class FetchError(RuntimeError):
    """The download failed or does not match the pinned size and sha256."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, target: Path, attempts: int = 3) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    last: Exception | None = None
    for _ in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=120) as resp:
                part.write_bytes(resp.read())
            break
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise FetchError(f"{url} returned HTTP 404") from e
            last = e
        except OSError as e:
            last = e
    else:
        raise FetchError(f"failed to download {url}: {last}")
    size, digest = part.stat().st_size, sha256_file(part)
    if size != ZIP_BYTES or digest != ZIP_SHA256:
        part.unlink()
        raise FetchError(
            f"{url} gave {size} bytes with sha256 {digest}; "
            f"expected {ZIP_BYTES} bytes with sha256 {ZIP_SHA256}"
        )
    part.replace(target)


def extract(archive: Path, dest: Path) -> list[Path]:
    """Every CSV member of the zip, written flat under ``dest/raw``."""
    out: list[Path] = []
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            name = Path(member.filename).name
            if member.is_dir() or not name.lower().endswith(".csv"):
                continue
            target = dest / "raw" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zf.read(member))
            out.append(target)
    return sorted(out)


def write_records(dest: Path, files: list[Path], started: datetime, downloaded: bool) -> None:
    manifest = {
        path.relative_to(dest).as_posix(): {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in [dest / ZIP_NAME, *files]
    }
    (dest / "MANIFEST.sha256.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    (dest / "FETCH.json").write_text(
        json.dumps(
            {
                "source": "Zenodo",
                "record": RECORD,
                "url": URL,
                "licence": LICENCE,
                "zip_bytes": ZIP_BYTES,
                "zip_sha256": ZIP_SHA256,
                "downloaded": downloaded,
                "fetched_utc": started.isoformat(),
                "completed_utc": datetime.now(UTC).isoformat(),
                "n_files": len(manifest),
                "files": manifest,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dest", default="data/turbine_upgrade")
    args = parser.parse_args(argv)

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    started = datetime.now(UTC)
    archive = dest / ZIP_NAME
    print(f"Zenodo record {RECORD}: {ZIP_NAME}, {ZIP_BYTES:,d} bytes")
    downloaded = not (archive.exists() and sha256_file(archive) == ZIP_SHA256)
    try:
        if downloaded:
            download(URL, archive)
    except FetchError as e:
        print(e)
        return 1
    files = extract(archive, dest)
    write_records(dest, files, started, downloaded)
    manifest_sha = sha256_file(dest / "MANIFEST.sha256.json")
    state = "downloaded" if downloaded else "unchanged"
    print(
        f"zip {state}, {len(files)} CSVs extracted; "
        f"manifest sha256 {manifest_sha[:12]} at {dest / 'MANIFEST.sha256.json'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

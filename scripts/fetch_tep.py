"""Opt-in fetcher for the Tennessee Eastman Process simulation data.

Not run by CI, not run by default. Resolves the file list through the
Harvard Dataverse dataset API first, prints the exact byte total of what
the filters select, asks, and only then downloads. Every file is checked
against the MD5 the API publishes *and* hashed locally into
``MANIFEST.sha256.json``; ``FETCH.json`` records the DOI, the dataset
version, the datafile ids and the fetch time.

TEP is tsdive's validation backbone: a simulation, so every variable's
role is documented and there is ground truth for a fault's start. The
Rieth 2017 re-simulation on Dataverse is the distribution used here
because it is 500 runs per fault rather than one, which is what makes a
held-out split possible at all.

Why the default is two of the four files: the training pair is 519 MB and
the testing pair a further 884 MB. Profiling needs neither test
split, so the download the caller is asked to consent to is the smaller
one unless ``--all`` says otherwise.

Terms: the Dataverse ``license`` field is null; the dataset carries
Harvard's *User Agreement, Public Domain Dedication, and Disclaimer of
Liability*, which waives copyright and requests (not requires)
attribution. See docs/DATA.md.

Usage:
    python scripts/fetch_tep.py --dest data/tep [--all] [--files NAME,NAME] [--yes]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

DOI = "doi:10.7910/DVN/6C3JR1"
DATASET_API = (
    "https://dataverse.harvard.edu/api/datasets/:persistentId/?persistentId={doi}"
)
ACCESS_URL = "https://dataverse.harvard.edu/api/access/datafile/{id}"

# Dataverse answers 403 to a request that states no User-Agent, so the
# fetcher names itself. Not a courtesy: without this every call fails.
HEADERS = {"User-Agent": "tsdive-fetch-tep/0.1", "Accept": "application/json"}

# Profiling reads archives; it does not score a detector, so the two
# testing files buy nothing and cost 884 MB. --all overrides.
DEFAULT_FILES = ("TEP_FaultFree_Training.RData", "TEP_Faulty_Training.RData")

CHUNK = 1 << 20


class FetchError(RuntimeError):
    """Upstream would not answer, or answered with the wrong bytes."""


@dataclass(frozen=True)
class Datafile:
    id: int
    filename: str
    size: int
    md5: str


def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers=dict(HEADERS))
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise FetchError(f"Dataverse returned HTTP {e.code} for {url}") from e
    except OSError as e:
        raise FetchError(f"could not reach {url}: {e}") from e


def dataset_version(payload: dict) -> str:
    version = payload["data"]["latestVersion"]
    return f"{version.get('versionNumber')}.{version.get('versionMinorNumber')}"


def dataset_files(payload: dict) -> list[Datafile]:
    """Every datafile the dataset publishes, with the MD5 it publishes too.

    A file whose checksum the API does not state is refused rather than
    downloaded unverified: the point of fetching through the metadata API
    instead of a hardcoded URL is that the checksum travels with the id.
    """
    files: list[Datafile] = []
    for entry in payload["data"]["latestVersion"]["files"]:
        data = entry["dataFile"]
        checksum = data.get("checksum") or {}
        digest = data.get("md5") or (
            checksum.get("value") if checksum.get("type") == "MD5" else None
        )
        if not digest:
            raise FetchError(
                f"{data.get('filename')}: Dataverse states no MD5, so a download "
                "could not be verified; nothing fetched"
            )
        files.append(
            Datafile(
                id=int(data["id"]),
                filename=str(data["filename"]),
                size=int(data["filesize"]),
                md5=str(digest),
            )
        )
    return sorted(files, key=lambda f: f.filename)


def select(files: list[Datafile], *, names: tuple[str, ...] | None) -> list[Datafile]:
    """Keep the named files; ``None`` keeps everything."""
    if names is None:
        return list(files)
    wanted = {n.strip() for n in names if n.strip()}
    return [f for f in files if f.filename in wanted]


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def download(datafile: Datafile, dest: Path) -> dict:
    """Stream one datafile to ``dest`` and verify it against the API MD5.

    The bytes land in a ``.part`` file and are only renamed once the MD5
    matches, so an interrupted or corrupted download can never be
    mistaken for a complete one on the next run. A mismatch removes the
    partial file and raises: a wrong file is worse than no file.
    """
    target = dest / datafile.filename
    if target.exists() and target.stat().st_size == datafile.size:
        digest = md5_file(target)
        if digest == datafile.md5:
            return {
                "bytes": target.stat().st_size,
                "md5": digest,
                "sha256": sha256_file(target),
                "reused": True,
            }
    partial = target.with_suffix(target.suffix + ".part")
    url = ACCESS_URL.format(id=datafile.id)
    req = urllib.request.Request(url, headers=dict(HEADERS))
    h = hashlib.md5()
    written = 0
    try:
        with urllib.request.urlopen(req, timeout=600) as resp, open(partial, "wb") as out:
            while True:
                chunk = resp.read(CHUNK)
                if not chunk:
                    break
                h.update(chunk)
                out.write(chunk)
                written += len(chunk)
    except OSError as e:
        partial.unlink(missing_ok=True)
        raise FetchError(f"{datafile.filename}: download failed ({e})") from e
    digest = h.hexdigest()
    if digest != datafile.md5:
        partial.unlink(missing_ok=True)
        raise FetchError(
            f"{datafile.filename}: MD5 mismatch - Dataverse states {datafile.md5}, "
            f"the {written:,d} downloaded bytes hash to {digest}; file discarded"
        )
    partial.replace(target)
    return {
        "bytes": written,
        "md5": digest,
        "sha256": sha256_file(target),
        "reused": False,
    }


def confirm(prompt: str) -> bool:
    """Ask, treating EOF as 'no' rather than as a traceback.

    A piped or closed stdin is not consent.
    """
    try:
        return input(prompt).strip().lower() == "y"
    except EOFError:
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", default="data/tep")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument(
        "--all",
        action="store_true",
        help="fetch the testing splits too (adds ~884 MB)",
    )
    parser.add_argument(
        "--files",
        default=None,
        help=(
            "comma-separated filenames to fetch; default is "
            + ",".join(DEFAULT_FILES)
        ),
    )
    args = parser.parse_args(argv)

    try:
        payload = _get_json(DATASET_API.format(doi=DOI))
        files = dataset_files(payload)
        version = dataset_version(payload)
    except FetchError as e:
        print(e)
        return 1
    except (KeyError, TypeError, ValueError) as e:
        print(f"unexpected dataset payload from Dataverse: {e}")
        return 1

    if args.files:
        names: tuple[str, ...] | None = tuple(args.files.split(","))
    elif args.all:
        names = None
    else:
        names = DEFAULT_FILES
    chosen = select(files, names=names)

    print(f"TEP {DOI} version {version}")
    print(
        f"dataset holds {len(files)} file(s), {sum(f.size for f in files):,d} bytes"
    )
    for f in chosen:
        print(f"  {f.filename}  id={f.id}  {f.size:,d} bytes  md5={f.md5}")
    total = sum(f.size for f in chosen)
    print(f"selected {len(chosen)} file(s): {total:,d} bytes total")
    if not chosen:
        print("filters selected no files; nothing to download")
        return 1
    if not args.yes and not confirm("download? [yN] "):
        print("aborted")
        return 1

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    started = datetime.now(UTC)
    manifest: dict[str, dict] = {}
    for f in chosen:
        try:
            record = download(f, dest)
        except FetchError as e:
            print(e)
            return 1
        manifest[f.filename] = {
            "bytes": record["bytes"],
            "datafile_id": f.id,
            "md5": record["md5"],
            "md5_source": "dataverse api",
            "sha256": record["sha256"],
        }
        state = "reused" if record["reused"] else "downloaded"
        print(f"  {state} {f.filename}: {record['bytes']:,d} bytes, md5 verified", flush=True)

    (dest / "MANIFEST.sha256.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    fetched_bytes = sum(r["bytes"] for r in manifest.values())
    (dest / "FETCH.json").write_text(
        json.dumps(
            {
                "bytes": fetched_bytes,
                "completed_utc": datetime.now(UTC).isoformat(),
                "datafile_ids": {name: r["datafile_id"] for name, r in manifest.items()},
                "doi": DOI,
                "fetched_utc": started.isoformat(),
                "files": sorted(manifest),
                "licence": (
                    "Dataverse license field is null; Harvard User Agreement, Public "
                    "Domain Dedication, and Disclaimer of Liability applies"
                ),
                "n_files": len(manifest),
                "source": "Harvard Dataverse",
                "version": version,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(
        f"fetched {len(manifest)} file(s), {fetched_bytes:,d} bytes; "
        f"manifest at {dest / 'MANIFEST.sha256.json'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

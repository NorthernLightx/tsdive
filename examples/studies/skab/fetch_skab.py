"""Opt-in fetcher for SKAB, the Skoltech Anomaly Benchmark (github.com/waico/SKAB).

Not run by CI, not run by default. Downloads the 35 experiment files of
the pinned upstream commit through raw GitHub URLs into
``<dest>/raw/<folder>/<name>.csv``, hashes them locally into
``MANIFEST.sha256.json`` and records the commit, the URLs and the fetch
time in ``FETCH.json``. A file whose sha256 already matches the manifest
from an earlier run is not downloaded again.

Data and code licence upstream: GPL-3.0. See docs/DATA.md.

Usage:
    python examples/studies/skab/fetch_skab.py --dest data/skab
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

REPO = "waico/SKAB"
COMMIT = "b2c0d46c2971dcbfe71e26087b6d231998bb91c2"
RAW_URL = f"https://raw.githubusercontent.com/{REPO}/{{sha}}/data/{{folder}}/{{name}}"
LICENCE = "GPL-3.0"

# Every file under data/ at COMMIT, listed by folder. The counts are
# fixed by the pin, so a different count on disk means a different
# upstream revision.
FILES: dict[str, tuple[str, ...]] = {
    "valve1": tuple(f"{i}.csv" for i in range(16)),
    "valve2": tuple(f"{i}.csv" for i in range(4)),
    "other": tuple(f"{i}.csv" for i in range(1, 15)),
    "anomaly-free": ("anomaly-free.csv",),
}


class FetchError(RuntimeError):
    """Upstream did not answer; the message names the URL."""


def relative_path(folder: str, name: str) -> str:
    return f"raw/{folder}/{name}"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(dest: Path) -> dict[str, dict]:
    path = dest / "MANIFEST.sha256.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def download(url: str, target: Path, attempts: int = 3) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    last: Exception | None = None
    for _ in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=120) as resp:
                payload = resp.read()
            target.write_bytes(payload)
            return
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise FetchError(f"{url} returned HTTP 404") from e
            last = e
        except OSError as e:
            last = e
    raise FetchError(f"failed to download {url}: {last}")


def fetch(dest: Path, sha: str = COMMIT) -> tuple[dict[str, dict], int, int]:
    """Download every file into ``dest``; return (manifest, n_downloaded, n_skipped)."""
    previous = load_manifest(dest)
    manifest: dict[str, dict] = {}
    downloaded = skipped = 0
    for folder, names in FILES.items():
        for name in names:
            rel = relative_path(folder, name)
            target = dest / rel
            known = previous.get(rel, {}).get("sha256")
            if target.exists() and known is not None and sha256_file(target) == known:
                skipped += 1
            else:
                download(RAW_URL.format(sha=sha, folder=folder, name=name), target)
                downloaded += 1
            manifest[rel] = {
                "sha256": sha256_file(target),
                "bytes": target.stat().st_size,
                "url": RAW_URL.format(sha=sha, folder=folder, name=name),
            }
    return manifest, downloaded, skipped


def write_records(dest: Path, manifest: dict[str, dict], started: datetime, sha: str) -> None:
    hashes = {
        rel: {"sha256": rec["sha256"], "bytes": rec["bytes"]}
        for rel, rec in sorted(manifest.items())
    }
    (dest / "MANIFEST.sha256.json").write_text(
        json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    (dest / "FETCH.json").write_text(
        json.dumps(
            {
                "repo": REPO,
                "commit": sha,
                "licence": LICENCE,
                "fetched_utc": started.isoformat(),
                "completed_utc": datetime.now(UTC).isoformat(),
                "n_files": len(manifest),
                "bytes": sum(rec["bytes"] for rec in manifest.values()),
                "files": {rel: rec for rel, rec in sorted(manifest.items())},
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
    parser.add_argument("--dest", default="data/skab")
    parser.add_argument("--sha", default=COMMIT, help="upstream commit to download from")
    args = parser.parse_args(argv)

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    started = datetime.now(UTC)
    n_files = sum(len(names) for names in FILES.values())
    print(f"SKAB {REPO} @ {args.sha}: {n_files} files")
    try:
        manifest, downloaded, skipped = fetch(dest, args.sha)
    except FetchError as e:
        print(e)
        return 1
    write_records(dest, manifest, started, args.sha)
    total = sum(rec["bytes"] for rec in manifest.values())
    manifest_sha = sha256_file(dest / "MANIFEST.sha256.json")
    print(
        f"{downloaded} downloaded, {skipped} unchanged, {total:,d} bytes; "
        f"manifest sha256 {manifest_sha[:12]} at {dest / 'MANIFEST.sha256.json'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

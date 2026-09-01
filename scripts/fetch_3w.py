"""Opt-in fetcher for the 3W dataset (Petrobras, github.com/petrobras/3W).

Not run by CI, not run by default. Enumerates ``dataset/**`` through the
GitHub git-trees API first, prints the exact byte total of what the
filters select, asks, and only then downloads. Files are hashed locally
into ``MANIFEST.sha256.json`` (never git blob shas), and the upstream
commit the bytes came from is recorded in ``FETCH.json``.

Why not the repository zip: the 3W dataset is ~1.87 GB of parquet in the
same repository as its notebooks and toolkit, and ``main.zip`` is
all-or-nothing. Enumerating blobs lets ``--only-real`` and ``--classes``
cut the download instead of the extraction.

Data licence CC BY 4.0 (``dataset/LICENSE-CC-BY``); repository code
Apache-2.0. See docs/DATA.md.

Usage:
    python scripts/fetch_3w.py --dest data/3w [--only-real] [--classes 0,1] [--yes]
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

REPO = "petrobras/3W"
BRANCH_API = f"https://api.github.com/repos/{REPO}/branches/{{ref}}"
TREE_API = f"https://api.github.com/repos/{REPO}/git/trees/{{sha}}?recursive=1"
RAW_URL = f"https://raw.githubusercontent.com/{REPO}/{{sha}}/{{path}}"

# Non-parquet files that describe the dataset itself. Always fetched: the
# units live in dataset.ini and the licence has to travel with the bytes.
AUX_PATHS = ("dataset/dataset.ini", "dataset/LICENSE-CC-BY", "dataset/README.md")

REAL_PREFIX = "WELL-"


class FetchError(RuntimeError):
    """Upstream would not answer; message is already human-readable."""


@dataclass(frozen=True)
class Blob:
    path: str
    size: int

    @property
    def folder_label(self) -> str | None:
        parts = self.path.split("/")
        if len(parts) >= 3 and parts[0] == "dataset" and parts[1].isdigit():
            return parts[1]
        return None

    @property
    def is_real(self) -> bool:
        return Path(self.path).name.startswith(REAL_PREFIX)


def content_length(url: str) -> int | None:
    """Size from a HEAD request, or None when the server does not state one.

    ``None`` means unknown, not zero. GitHub's archive endpoints answer a
    HEAD with no ``Content-Length`` at all, and reporting that as
    ``0 bytes`` would be a fabricated number in the one place the caller is
    being asked to consent to a download.
    """
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.headers.get("Content-Length")
    except OSError:
        return None
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code in (403, 429):
            remaining = e.headers.get("X-RateLimit-Remaining")
            reset = e.headers.get("X-RateLimit-Reset")
            when = ""
            if reset is not None and reset.isdigit():
                when = f"; resets at {datetime.fromtimestamp(int(reset), UTC).isoformat()}"
            raise FetchError(
                f"GitHub API refused {url} with HTTP {e.code} "
                f"(rate limit remaining={remaining}{when}). Wait for the reset, or "
                "export GITHUB_TOKEN and retry - this script does not authenticate."
            ) from e
        raise FetchError(f"GitHub API returned HTTP {e.code} for {url}") from e
    except OSError as e:
        raise FetchError(f"could not reach {url}: {e}") from e


def head_commit(ref: str = "main") -> str:
    """Commit sha of ``ref``, so the download is pinned to one revision."""
    payload = _get_json(BRANCH_API.format(ref=ref))
    try:
        return str(payload["commit"]["sha"])
    except (KeyError, TypeError) as e:
        raise FetchError(f"unexpected branch payload from {BRANCH_API.format(ref=ref)}") from e


def dataset_blobs(sha: str) -> list[Blob]:
    """Every ``dataset/`` blob at ``sha`` that this script knows how to use."""
    payload = _get_json(TREE_API.format(sha=sha))
    if payload.get("truncated"):
        raise FetchError(
            "GitHub truncated the recursive tree listing, so the byte total "
            "is unknown; nothing was downloaded"
        )
    blobs: list[Blob] = []
    for entry in payload.get("tree", []):
        if entry.get("type") != "blob":
            continue
        path = str(entry["path"])
        if path in AUX_PATHS:
            blobs.append(Blob(path, int(entry["size"])))
            continue
        if not path.startswith("dataset/") or not path.endswith(".parquet"):
            continue
        blob = Blob(path, int(entry["size"]))
        if blob.folder_label is None:
            continue
        blobs.append(blob)
    return sorted(blobs, key=lambda b: b.path)


def select(blobs: list[Blob], *, only_real: bool, classes: set[str] | None) -> list[Blob]:
    """Apply the instance filters; aux files are never filtered out."""
    chosen: list[Blob] = []
    for b in blobs:
        if b.path in AUX_PATHS:
            chosen.append(b)
            continue
        if only_real and not b.is_real:
            continue
        if classes is not None and b.folder_label not in classes:
            continue
        chosen.append(b)
    return chosen


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_one(sha: str, blob: Blob, dest: Path, attempts: int = 3) -> tuple[str, dict]:
    target = dest / blob.path
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size == blob.size:
        return blob.path, {"sha256": sha256_file(target), "bytes": target.stat().st_size}
    url = RAW_URL.format(sha=sha, path=blob.path)
    last: Exception | None = None
    for _ in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=300) as resp:
                payload = resp.read()
            target.write_bytes(payload)
            return blob.path, {"sha256": sha256_file(target), "bytes": target.stat().st_size}
        except OSError as e:  # transient network / 5xx
            last = e
    raise FetchError(f"failed to download {blob.path}: {last}")


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
    parser.add_argument("--dest", default="data/3w")
    parser.add_argument("--ref", default="main", help="branch to pin the commit sha from")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument(
        "--only-real",
        action="store_true",
        help=f"keep only real instances (files named {REAL_PREFIX}*)",
    )
    parser.add_argument(
        "--classes",
        default=None,
        help="comma-separated event folders to keep, for example 0,1,5",
    )
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args(argv)

    classes = (
        {c.strip() for c in args.classes.split(",") if c.strip()} if args.classes else None
    )

    try:
        sha = head_commit(args.ref)
        blobs = dataset_blobs(sha)
    except FetchError as e:
        print(e)
        return 1

    chosen = select(blobs, only_real=args.only_real, classes=classes)
    parquet = [b for b in chosen if b.path.endswith(".parquet")]
    total = sum(b.size for b in chosen)
    print(f"3W {REPO} @ {sha}")
    print(f"dataset/ holds {len(blobs)} usable file(s), {sum(b.size for b in blobs):,d} bytes")
    print(
        f"selected {len(chosen)} file(s) ({len(parquet)} instance parquet): {total:,d} bytes"
        f"  [only_real={args.only_real} classes={args.classes or 'all'}]"
    )
    if not parquet:
        print("filters selected no instances; nothing to download")
        return 1
    if not args.yes and not confirm("download? [yN] "):
        print("aborted")
        return 1

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict] = {}
    started = datetime.now(UTC)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = [pool.submit(_download_one, sha, b, dest) for b in chosen]
        for done, fut in enumerate(concurrent.futures.as_completed(futures), start=1):
            try:
                path, record = fut.result()
            except FetchError as e:
                print(e)
                return 1
            manifest[path] = record
            if done % 100 == 0 or done == len(chosen):
                print(f"  {done}/{len(chosen)} files", flush=True)

    (dest / "MANIFEST.sha256.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    fetched_bytes = sum(r["bytes"] for r in manifest.values())
    (dest / "FETCH.json").write_text(
        json.dumps(
            {
                "repo": REPO,
                "ref": args.ref,
                "commit": sha,
                "fetched_utc": started.isoformat(),
                "completed_utc": datetime.now(UTC).isoformat(),
                "n_files": len(manifest),
                "bytes": fetched_bytes,
                "filters": {
                    "only_real": args.only_real,
                    "classes": sorted(classes) if classes else None,
                },
                "data_licence": "CC BY 4.0 (dataset/LICENSE-CC-BY)",
                "code_licence": "Apache-2.0",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(
        f"imported {len(manifest)} file(s), {fetched_bytes:,d} bytes; "
        f"manifest at {dest / 'MANIFEST.sha256.json'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

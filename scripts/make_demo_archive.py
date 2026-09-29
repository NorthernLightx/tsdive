"""Build deterministic synthetic demo archives for the README transcripts.

    uv run python scripts/make_demo_archive.py

Writes data/demo/fic101_demo.parquet and data/demo/tic101_demo.parquet
(git-ignored; regenerate freely). The builders live in ``tsdive.demo``,
which ``tsdive demo`` also calls; this script writes the same bytes into
the repository's ``data/`` and replaces files that exist.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from tsdive.demo import build_fic101, build_tic101
from tsdive.store.tagstore import write_tag

__all__ = ["build_fic101", "build_tic101", "main"]


def main() -> int:
    flow, flow_meta = build_fic101()
    out = write_tag(Path("data/demo/fic101_demo.parquet"), flow, flow_meta, overwrite=True)
    print(f"wrote {out} ({len(flow)} samples)")

    temp, temp_meta = build_tic101(flow)
    out = write_tag(Path("data/demo/tic101_demo.parquet"), temp, temp_meta, overwrite=True)
    print(f"wrote {out} ({len(temp)} samples)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

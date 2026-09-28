"""Build a synthetic switchback trial for the docs/SWITCHBACK.md transcripts.

    uv run python examples/switchback/make_trial.py

Writes data/switchback_demo/ti201.parquet, fi200.parquet and
tt001.parquet (git-ignored; regenerate freely): three tags of one unit at
60 s from 2024-06-01 00:00Z to 2024-06-04 00:00Z. The builder lives in
``tsdive.demo``, which ``tsdive demo`` also calls; this script writes the
same bytes into the repository's ``data/`` and replaces files that exist.

The last day is the trial. The plant follows the schedule that
``tsdive switchback plan`` draws for the window below with seed 7:
setting B raises the outlet temperature by 0.25 degC through a
first-order lag with a 5-minute time constant. The first two days are
history, with no switching.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from tsdive.demo import (
    TRIAL,
    TRIAL_BLOCK,
    TRIAL_SEED,
    TRIAL_SHIFT,
    TRIAL_WASHOUT,
    build_trial,
)
from tsdive.store.tagstore import write_tag

__all__ = ["BLOCK", "OUT", "SEED", "SHIFT", "TRIAL", "WASHOUT", "build", "main"]

OUT = Path("data/switchback_demo")
BLOCK = TRIAL_BLOCK
WASHOUT = TRIAL_WASHOUT
SEED = TRIAL_SEED
SHIFT = TRIAL_SHIFT
build = build_trial


def main() -> int:
    for name, (frame, meta) in build().items():
        out = write_tag(OUT / f"{name}.parquet", frame, meta, overwrite=True)
        print(f"wrote {out.as_posix()} ({len(frame)} samples)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

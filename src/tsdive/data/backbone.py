"""Deterministic SYNTHETIC validation backbone.

Everything here is seeded, replayable, and labelled synthetic, and
numbers published from this backbone are labelled the same way.
Real-data numbers come from the fetched datasets (docs/DATA.md).

Structure: ``n_loops`` control loops, each with SP/PV/MODE tags on one
shared 60 s grid. PV tracks SP through a first-order lag with noise; SP
steps between regimes. Five fault archetypes are injected into specific
PVs at scheduled intervals; the ground truth is written to truth.csv.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from tsdive.store.identity import EngRange, Role, TagIdentity, TagMeta
from tsdive.store.tagstore import write_tag

FAULT_TYPES = ("level_shift", "drift", "stuck_sensor", "variance_burst", "spike")


@dataclass(frozen=True)
class Backbone:
    root: Path
    truth: pd.DataFrame  # fault_type, loop, start, end, magnitude

    def tag_path(self, source_id: str, point_id: str) -> Path:
        return self.root / source_id / f"{point_id}.parquet"


def _write_tag(root: Path, df: pd.DataFrame, meta: TagMeta) -> None:
    path = root / meta.identity.source_id / f"{meta.identity.point_id}.parquet"
    write_tag(path, df, meta, overwrite=True)


def generate_backbone(
    root: str | Path,
    *,
    seed: int = 42,
    n_loops: int = 4,
    hours: int = 48,
    rate_s: int = 60,
) -> Backbone:
    """Generate the synthetic backbone deterministically."""
    rng = np.random.default_rng(seed)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    start = pd.Timestamp("2025-01-01 00:00:00+00:00")
    stamps = [start + pd.Timedelta(rate_s * i, unit="s") for i in range(hours * 3600 // rate_s)]
    n = len(stamps)

    regimes = [40.0, 55.0, 70.0]
    # Fixed 4 h blocks cycling through regimes: every regime appears many
    # times in both halves, which baselining across splits requires.
    block_len = max(4 * 3600 // rate_s, 1)
    regime_ids = (np.arange(n) // block_len) % len(regimes)
    regime_blocks = regime_ids

    truth_rows: list[dict] = []
    # One scheduled fault per loop, non-overlapping in time, in fixed order.
    fault_plan = list(zip(range(n_loops), FAULT_TYPES[:n_loops], strict=True))
    day2_start = int(n / 2)

    for loop in range(n_loops):
        sp = np.array([regimes[r] for r in regime_blocks], dtype=float)
        mode_vals = np.array([f"R{r}" for r in regime_blocks])
        pv = np.zeros(n)
        val = sp[0]
        noise_scale = 0.6
        for i in range(n):
            target = sp[i]
            val += 0.15 * (target - val)  # first-order lag toward SP
            pv[i] = val + float(rng.normal(0, noise_scale))

        fault_type = None
        for loop_id, ft in fault_plan:
            if loop_id == loop:
                fault_type = ft
        if fault_type is not None:
            f0, f1 = day2_start, min(day2_start + 240, n - 1)
            mag = 8.0 if fault_type != "spike" else 25.0
            if fault_type == "level_shift":
                pv[f0:f1] += mag
            elif fault_type == "drift":
                ramp = np.linspace(0.0, mag, f1 - f0)
                pv[f0:f1] += ramp
            elif fault_type == "stuck_sensor":
                stuck_value = float(pv[f0])
                pv[f0:f1] = stuck_value
            elif fault_type == "variance_burst":
                pv[f0:f1] += rng.normal(0, 4.0, size=f1 - f0)
                mag = 4.0
            elif fault_type == "spike":
                spikes = rng.random(f1 - f0) < 0.05
                pv[f0:f1][spikes] += mag
            truth_rows.append(
                {
                    "fault_type": fault_type,
                    "loop": loop,
                    "start": stamps[f0].isoformat(),
                    "end": stamps[f1].isoformat(),
                    "magnitude": mag,
                    "instance_source": f"sim-unit-{loop}",
                }
            )

        qualities = ["GOOD"] * n
        comm_idx = rng.choice(n, size=n // 500, replace=False)
        for i in comm_idx:
            qualities[i] = "COMM FAILURE"
        drop = {int(i) for i in comm_idx}

        src = f"sim-unit-{loop}"
        frames = {
            "SP": (
                TagIdentity(src, f"FIC{loop:03d}.SP"),
                sp,
                ["GOOD"] * n,
                None,
            ),
            "MODE": (
                TagIdentity(src, f"FIC{loop:03d}.MODE"),
                mode_vals,
                ["GOOD"] * n,
                None,
            ),
            "PV": (
                TagIdentity(src, f"FIC{loop:03d}.PV"),
                pv,
                qualities,
                EngRange(zero=0.0, span=100.0),
            ),
        }
        for role, (ident, vals, quals, erange) in frames.items():
            keep = [i for i in range(n) if not (role == "PV" and i in drop)]
            df = pd.DataFrame(
                {
                    "timestamp": [stamps[i] for i in keep],
                    "value": [vals[i] for i in keep],
                    "quality": [quals[i] for i in keep],
                }
            )
            meta = TagMeta(
                identity=ident,
                name=f"FIC{loop:03d} {role}",
                unit_raw="m3/h",
                eng_range=erange,
                sample_rate_s=float(rate_s),
                asset=src,
                loop_id=f"FIC{loop:03d}",
                role=Role(role),
            )
            _write_tag(root, df, meta)

    truth = pd.DataFrame(truth_rows)
    truth.to_csv(root / "truth.csv", index=False)
    manifest = {"kind": "synthetic-backbone", "seed": seed, "n_loops": n_loops}
    (root / "MANIFEST.json").write_text(
        json.dumps(manifest, sort_keys=True),
        encoding="utf-8",
    )
    return Backbone(root=root, truth=truth)


def load_truth(backbone_root: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(backbone_root) / "truth.csv")

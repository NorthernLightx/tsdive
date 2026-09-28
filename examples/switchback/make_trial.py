"""Build a synthetic switchback trial for the docs/SWITCHBACK.md transcripts.

    uv run python examples/switchback/make_trial.py

Writes data/switchback_demo/ti201.parquet, fi200.parquet and
tt001.parquet (git-ignored; regenerate freely): three tags of one unit at
60 s from 2024-06-01 00:00Z to 2024-06-04 00:00Z.

- FI200.PV, feed flow in m3/h: a slow random walk around 40.
- TT001.PV, ambient temperature in degC: a daily cycle around 15.
- TI201.PV, outlet temperature in degC: 180 + 0.8 per m3/h of feed above
  40 + 0.1 per degC of ambient above 15, plus AR(1) noise.

The last day is the trial. The plant follows the schedule that
``tsdive switchback plan`` draws for the window below with seed 7:
setting B raises the outlet temperature by 0.25 degC through a
first-order lag with a 5-minute time constant. The first two days are
history, with no switching.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from tsdive.api import switchback_plan
from tsdive.store.identity import Role, TagIdentity, TagMeta
from tsdive.store.tagstore import write_tag
from tsdive.switchback.inference import lag_response

OUT = Path("data/switchback_demo")
START = pd.Timestamp("2024-06-01 00:00:00+00:00")
TRIAL = ("2024-06-03T00:00:00Z", "2024-06-04T00:00:00Z")
BLOCK = "PT1H"
WASHOUT = "PT15M"
SEED = 7
SHIFT = 0.25
TAU_MIN = 5.0


def build() -> dict[str, tuple[pd.DataFrame, TagMeta]]:
    """The three tags as archive frames and metadata, the trial following its plan."""
    plan = switchback_plan(TRIAL[0], TRIAL[1], BLOCK, WASHOUT, SEED)
    n = 3 * 24 * 60
    stamps = START + pd.to_timedelta(np.arange(n) * 60, unit="s")
    rng = np.random.default_rng(2024)
    feed = 40.0 + np.cumsum(rng.normal(0.0, 0.05, n))
    hours = np.arange(n) / 60.0
    ambient = 15.0 + 6.0 * np.sin(2 * np.pi * (hours - 9.0) / 24.0) + rng.normal(0.0, 0.2, n)
    noise = np.empty(n)
    noise[0] = 0.0
    shocks = rng.normal(0.0, 0.05, n)
    for i in range(1, n):
        noise[i] = 0.95 * noise[i - 1] + shocks[i]

    setting = np.zeros(n)
    trial_start = pd.Timestamp(TRIAL[0])
    for block in plan.blocks:
        inside = (stamps >= block.start) & (stamps < block.end)
        setting[inside] = 1.0 if block.setting == "B" else 0.0
    minutes = np.arange(n, dtype=float)
    response = lag_response(setting, minutes, TAU_MIN)
    response[stamps < trial_start] = 0.0
    outlet = 180.0 + 0.8 * (feed - 40.0) + 0.1 * (ambient - 15.0) + SHIFT * response + noise

    def tag(point: str, name: str, unit: str, values: np.ndarray):
        frame = pd.DataFrame(
            {"timestamp": stamps, "value": np.round(values, 4), "quality": ["GOOD"] * n}
        )
        meta = TagMeta(
            identity=TagIdentity(source_id="demo", point_id=point),
            name=name,
            unit_raw=unit,
            sample_rate_s=60.0,
            asset="demo-unit",
            role=Role.PV,
        )
        return frame, meta

    return {
        "ti201": tag("TI201.PV", "TI-201 outlet temperature", "degC", outlet),
        "fi200": tag("FI200.PV", "FI-200 feed flow", "m3/h", feed),
        "tt001": tag("TT001.PV", "TT-001 ambient temperature", "degC", ambient),
    }


def main() -> int:
    for name, (frame, meta) in build().items():
        out = write_tag(OUT / f"{name}.parquet", frame, meta, overwrite=True)
        print(f"wrote {out.as_posix()} ({len(frame)} samples)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

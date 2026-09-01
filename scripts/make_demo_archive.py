"""Build deterministic synthetic demo archives for the README transcripts.

    uv run python scripts/make_demo_archive.py

Writes data/demo/fic101_demo.parquet and data/demo/tic101_demo.parquet
(git-ignored; regenerate freely). The flow loop is a fictitious tag with:
- 3 s sampling, seeded noise, stepped contract
- a 40-minute collection outage on 2024-03-30 23:00-23:40Z
- a saturated stretch pinned at full scale (eng range 0..100)
- occasional unmapped quality codes ("SENSOR DRIFT")

The temperature tracks it linearly, on the same timestamps, with a
residual burst in the last two hours: the correlation the two tags hold
all night breaks there, which MSPC sees mostly in SPE and partly in T2.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from tsdive.store.identity import EngRange, Role, TagIdentity, TagMeta
from tsdive.store.tagstore import write_tag


def build_fic101() -> tuple[pd.DataFrame, TagMeta]:
    rng = np.random.default_rng(42)
    start = pd.Timestamp("2024-03-30 20:00:00+00:00")
    end = pd.Timestamp("2024-03-31 06:00:00+00:00")
    stamps = []
    cursor = start
    while cursor <= end:
        stamps.append(cursor)
        cursor += pd.Timedelta(60, unit="s")

    values, qualities, kept = [], [], []
    outage = (
        pd.Timestamp("2024-03-30 23:00:00+00:00"),
        pd.Timestamp("2024-03-30 23:40:00+00:00"),
    )
    saturation = (
        pd.Timestamp("2024-03-31 02:00:00+00:00"),
        pd.Timestamp("2024-03-31 02:30:00+00:00"),
    )
    for i, ts in enumerate(stamps):
        if outage[0] < ts < outage[1]:
            continue  # historian stored nothing: collection down
        drift = 0.8 * np.sin(2 * np.pi * i / 240) + float(rng.normal(0, 0.15))
        v = 62.0 + drift
        q = "GOOD"
        if saturation[0] < ts < saturation[1]:
            v = 100.0  # transmitter pinned at full scale
        elif rng.random() < 0.002:
            q = "SENSOR DRIFT"  # unmapped at this site
        kept.append(ts)
        values.append(v)
        qualities.append(q)

    df = pd.DataFrame(
        {
            "timestamp": kept,
            "value": values,
            "quality": qualities,
        }
    )

    meta = TagMeta(
        identity=TagIdentity(source_id="demo", point_id="FIC101.PV"),
        name="FIC-101 flow",
        unit_raw="m3/h",
        eng_range=EngRange(zero=0.0, span=100.0),
        sample_rate_s=60.0,
        asset="demo-unit",
        loop_id="FIC101",
        role=Role.PV,
    )
    return df, meta


def build_tic101(flow: pd.DataFrame) -> tuple[pd.DataFrame, TagMeta]:
    """A temperature on the flow's own timestamps, linear in it until 04:00Z.

    Its own seed, so the flow archive comes out byte-identical whether or
    not this tag is built.
    """
    rng = np.random.default_rng(7)
    burst_from = pd.Timestamp("2024-03-31 04:00:00+00:00")
    values = []
    for ts, flow_value in zip(flow["timestamp"], flow["value"], strict=True):
        v = 120.0 + 0.5 * (float(flow_value) - 62.0) + float(rng.normal(0, 0.1))
        if ts >= burst_from:
            # Residual burst: the temperature stops tracking the flow. Most
            # of it lands in SPE (108 of the 121 monitored rows in the
            # README window); T2 fires on 22 of them, so it is not quiet.
            v += float(rng.normal(0, 2.0))
        values.append(v)

    df = pd.DataFrame(
        {
            "timestamp": list(flow["timestamp"]),
            "value": values,
            "quality": ["GOOD"] * len(values),
        }
    )
    meta = TagMeta(
        identity=TagIdentity(source_id="demo", point_id="TIC101.PV"),
        name="TIC-101 outlet temperature",
        unit_raw="degC",
        eng_range=EngRange(zero=0.0, span=200.0),
        sample_rate_s=60.0,
        asset="demo-unit",
        loop_id="TIC101",
        role=Role.PV,
    )
    return df, meta


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

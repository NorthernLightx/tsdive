"""Deterministic synthetic demo archives, the data every example in the docs reads.

``write_demo_data`` writes two groups of single-tag archives into one
directory, and ``tsdive demo`` calls it:

- ``demo/fic101_demo.parquet`` and ``demo/tic101_demo.parquet``: a flow
  loop at 60 s from 2024-03-30 20:00Z to 2024-03-31 06:00Z, with a
  40-minute collection outage from 23:00Z, a stretch pinned at full scale
  (eng range 0 to 100) from 02:00Z to 02:30Z and a few unmapped quality
  codes (``SENSOR DRIFT``), and a temperature that tracks the flow
  linearly until 04:00Z and carries a residual burst after it.
- ``switchback_demo/ti201.parquet``, ``fi200.parquet`` and
  ``tt001.parquet``: three tags of one unit at 60 s from 2024-06-01 00:00Z
  to 2024-06-04 00:00Z. The last day is a switchback trial that follows
  the schedule ``switchback_plan`` draws for it with seed 7: setting B
  raises the outlet temperature by 0.25 degC through a first-order lag
  with a 5-minute time constant.

Every value comes from seeded generators, so the same call writes the
same bytes on every platform.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tsdive.store.identity import EngRange, Role, TagIdentity, TagMeta
from tsdive.store.tagstore import write_tag

DEFAULT_DIR = "tsdive-demo"
"""Directory ``tsdive demo`` writes into when none is given."""

PROFILE_DIR = "demo"
TRIAL_DIR = "switchback_demo"

TRIAL = ("2024-06-03T00:00:00Z", "2024-06-04T00:00:00Z")
TRIAL_BLOCK = "PT1H"
TRIAL_WASHOUT = "PT15M"
TRIAL_SEED = 7
TRIAL_SHIFT = 0.25
TRIAL_TAU_MIN = 5.0


def build_fic101() -> tuple[pd.DataFrame, TagMeta]:
    """The FIC-101 flow archive frame and metadata.

    One sample a minute, seeded noise on a slow sine, no rows during the
    outage, 100.0 through the saturated stretch.
    """
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


def build_trial() -> dict[str, tuple[pd.DataFrame, TagMeta]]:
    """The three switchback tags as archive frames and metadata, the trial following its plan.

    - FI200.PV, feed flow in m3/h: a slow random walk around 40.
    - TT001.PV, ambient temperature in degC: a daily cycle around 15.
    - TI201.PV, outlet temperature in degC: 180 + 0.8 per m3/h of feed
      above 40 + 0.1 per degC of ambient above 15, plus AR(1) noise and the
      lagged response to setting B.
    """
    from tsdive.api import switchback_plan
    from tsdive.switchback.inference import lag_response

    plan = switchback_plan(TRIAL[0], TRIAL[1], TRIAL_BLOCK, TRIAL_WASHOUT, TRIAL_SEED)
    start = pd.Timestamp("2024-06-01 00:00:00+00:00")
    n = 3 * 24 * 60
    stamps = start + pd.to_timedelta(np.arange(n) * 60, unit="s")
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
    response = lag_response(setting, minutes, TRIAL_TAU_MIN)
    response[stamps < trial_start] = 0.0
    outlet = (
        180.0
        + 0.8 * (feed - 40.0)
        + 0.1 * (ambient - 15.0)
        + TRIAL_SHIFT * response
        + noise
    )

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


def demo_archives() -> dict[str, tuple[pd.DataFrame, TagMeta]]:
    """Path under the demo directory -> archive frame and metadata, in writing order."""
    flow, flow_meta = build_fic101()
    temp, temp_meta = build_tic101(flow)
    archives = {
        f"{PROFILE_DIR}/fic101_demo.parquet": (flow, flow_meta),
        f"{PROFILE_DIR}/tic101_demo.parquet": (temp, temp_meta),
    }
    for name, archive in build_trial().items():
        archives[f"{TRIAL_DIR}/{name}.parquet"] = archive
    return archives


def write_demo_data(directory: str | Path = DEFAULT_DIR) -> list[Path]:
    """Write the demo archives under ``directory`` and return their paths.

    The two profile archives go to ``directory/demo/`` and the three
    switchback tags to ``directory/switchback_demo/``. ``write_demo_data("data")``
    writes the paths the README and the docs examples read.

    Raises:
        FileExistsError: one of the archives exists already; nothing is
            written.

    Examples:
        >>> import tsdive
        >>> for path in tsdive.write_demo_data("tsdive-demo"):
        ...     print(path.as_posix())
        tsdive-demo/demo/fic101_demo.parquet
        tsdive-demo/demo/tic101_demo.parquet
        tsdive-demo/switchback_demo/ti201.parquet
        tsdive-demo/switchback_demo/fi200.parquet
        tsdive-demo/switchback_demo/tt001.parquet
        >>> str(tsdive.profile("tsdive-demo/demo/fic101_demo.parquet").identity)
        'demo:FIC101.PV'
    """
    root = Path(directory)
    archives = demo_archives()
    existing = [root / name for name in archives if (root / name).exists()]
    if existing:
        raise FileExistsError(
            f"{existing[0].as_posix()} already exists; pass a directory that holds "
            "no demo archives"
        )
    return [
        write_tag(root / name, frame, meta) for name, (frame, meta) in archives.items()
    ]

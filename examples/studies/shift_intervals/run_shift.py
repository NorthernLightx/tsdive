"""Before/after shift intervals scored on synthetic series with a known shift.

The study asks which interval method for a before/after level or spread
shift keeps its coverage on autocorrelated, drifting process data, and
whether covariate adjustment narrows the interval without biasing it.
Estimators and rules live in ``shift.py``.

``synthetic`` draws ``--replicates`` series per cell from the generator in
``shift.py``; each cell's seed comes from its parameters alone. Grids:

- ``level``: phi x rho x drift x delta x n per period.
- ``affected_covariate``: rho 0.9, the recorded covariate steps by half the
  target's shift after the change date, drift 0.
- ``spread``: phi x n x SD ratio, delta 0, rho 0, drift 0.

Outputs in ``--out``:

- ``synthetic.csv``: one row per (cell, quantity, method, adjustment,
  refusal set) with coverage of the true shift and its Monte Carlo SE, the
  claim rate at a zero shift, the detection rate otherwise, the median width
  in before-period SDs, the bias, the median variance reduction and the
  firing rate of each rule.
- ``run.json``: provenance, parameters, seeds, counts and wall seconds per bed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from itertools import product
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

import shift as sh  # noqa: E402

import tsdive  # noqa: E402

DEFAULT_REPLICATES = 200
SYNTHETIC_BUDGET_S = 1200

BED_SYNTHETIC = "synthetic"
BEDS = (BED_SYNTHETIC,)

GRID_LEVEL = "level"
GRID_AFFECTED = "affected_covariate"
GRID_SPREAD = "spread"

PHIS = (0.0, 0.5, 0.9, 0.98)
RHOS = (0.0, 0.5, 0.9)
DRIFTS = (0.0, 1.0)
DELTAS = (0.0, 0.25, 0.5, 1.0)
NS = (120, 480)
SD_RATIOS = (1.0, 0.7, 0.5)
AFFECTED_RHO = 0.9
AFFECTED_DELTAS = (0.25, 0.5, 1.0)
COVARIATE_STEP_FRACTION = 0.5

CELL_KEYS = ("grid", "phi", "rho", "drift", "delta", "n", "sd_ratio", "covariate_step")
FLAG_RULES = (sh.BEFORE_TREND, sh.COVARIATE_OUTSIDE, sh.COVARIATE_SHIFTED)


# ---------------------------------------------------------------- helpers


def git_head() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover
        return "unknown"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def r4(value) -> float | None:
    if value is None:
        return None
    v = float(value)
    return None if not math.isfinite(v) else round(v, 4)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n")


def code_version() -> dict:
    return {"git_head": git_head(), "tsdive_version": tsdive.__version__}


def round_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        out[column] = [r4(v) for v in out[column]]
    return out


# ---------------------------------------------------------------- synthetic


def synthetic_cells() -> list[dict]:
    """Every synthetic cell, in a fixed order that the seeds do not depend on."""
    cells = []
    for phi, rho, drift, delta, n in product(PHIS, RHOS, DRIFTS, DELTAS, NS):
        cells.append(
            {
                "grid": GRID_LEVEL,
                "phi": phi,
                "rho": rho,
                "drift": drift,
                "delta": delta,
                "n": n,
                "sd_ratio": 1.0,
                "covariate_step": 0.0,
            }
        )
    for phi, delta, n in product(PHIS, AFFECTED_DELTAS, NS):
        cells.append(
            {
                "grid": GRID_AFFECTED,
                "phi": phi,
                "rho": AFFECTED_RHO,
                "drift": 0.0,
                "delta": delta,
                "n": n,
                "sd_ratio": 1.0,
                "covariate_step": COVARIATE_STEP_FRACTION * delta,
            }
        )
    for phi, n, ratio in product(PHIS, NS, SD_RATIOS):
        cells.append(
            {
                "grid": GRID_SPREAD,
                "phi": phi,
                "rho": 0.0,
                "drift": 0.0,
                "delta": 0.0,
                "n": n,
                "sd_ratio": ratio,
                "covariate_step": 0.0,
            }
        )
    return cells


def cell_quantity(cell: dict) -> str:
    return sh.SPREAD if cell["grid"] == GRID_SPREAD else sh.LEVEL


def true_value(cell: dict) -> float:
    if cell["grid"] == GRID_SPREAD:
        return math.log(cell["sd_ratio"])
    return float(cell["delta"])


def score_cell(cell: dict, replicates: int) -> tuple[list[dict], int]:
    """Aggregated rows of one cell and the seed it drew from."""
    seed = sh.cell_seed(cell)
    n = cell["n"]
    y, x = sh.simulate(
        phi=cell["phi"],
        rho=cell["rho"],
        drift=cell["drift"],
        delta=cell["delta"],
        n=n,
        replicates=replicates,
        sd_ratio=cell["sd_ratio"],
        covariate_step=cell["covariate_step"],
        seed=seed,
    )
    quantity = cell_quantity(cell)
    rows: list[dict] = []
    for r in range(replicates):
        rows += sh.score_target(
            {"y": y[r, :n], "x": x[r, :n]},
            {"y": y[r, n:], "x": x[r, n:]},
            "y",
            ["x"],
            quantities=(quantity,),
        )
    return aggregate_cell(cell, seed, pd.DataFrame(rows), replicates), seed


def _rate(mask: pd.Series) -> float | None:
    return float(mask.mean()) if len(mask) else None


def aggregate_cell(cell: dict, seed: int, frame: pd.DataFrame, replicates: int) -> list[dict]:
    truth = true_value(cell)
    out = []
    for (method, adjustment), group in frame.groupby(["method", "adjustment"], sort=False):
        sets = sh.RAW_REFUSAL_SETS if adjustment == sh.RAW else tuple(sh.REFUSAL_SETS)
        hard = group["refused"].astype(bool)
        flag_rates = {
            f"rate_{rule}": (
                _rate(group[rule].eq(True))
                if rule == sh.BEFORE_TREND or adjustment == sh.ADJUSTED
                else None
            )
            for rule in FLAG_RULES
        }
        for refusal_set in sets:
            refused = hard.copy()
            for rule in sh.REFUSAL_SETS[refusal_set]:
                refused |= group[rule].eq(True)
            answered = group[~refused]
            covered = (answered["lo"] <= truth) & (answered["hi"] >= truth)
            coverage = _rate(covered)
            n_answered = len(answered)
            excludes_zero = _rate(answered["clears"].astype(bool))
            if cell_quantity(cell) == sh.LEVEL:
                width = (answered["hi"] - answered["lo"]) / answered["scale"]
            else:
                width = answered["hi"] - answered["lo"]
            out.append(
                {
                    **{key: cell[key] for key in CELL_KEYS},
                    "seed": seed,
                    "quantity": cell_quantity(cell),
                    "method": method,
                    "adjustment": adjustment,
                    "refusal_set": refusal_set,
                    "true_value": truth,
                    "n_replicates": replicates,
                    "n_answered": n_answered,
                    "coverage": coverage,
                    "coverage_mcse": (
                        math.sqrt(coverage * (1 - coverage) / n_answered)
                        if n_answered
                        else None
                    ),
                    "claim_rate": excludes_zero if truth == 0 else None,
                    "detect_rate": excludes_zero if truth != 0 else None,
                    "bias": (
                        float(answered["estimate"].mean()) - truth if n_answered else None
                    ),
                    "median_width": float(width.median()) if n_answered else None,
                    "median_variance_reduction": (
                        float(group["variance_reduction"].median())
                        if adjustment == sh.ADJUSTED
                        else None
                    ),
                    "rate_refused": _rate(refused),
                    **flag_rates,
                }
            )
    return out


SYNTHETIC_FLOATS = [
    "phi",
    "rho",
    "drift",
    "delta",
    "sd_ratio",
    "covariate_step",
    "true_value",
    "coverage",
    "coverage_mcse",
    "claim_rate",
    "detect_rate",
    "bias",
    "median_width",
    "median_variance_reduction",
    "rate_refused",
    *[f"rate_{rule}" for rule in FLAG_RULES],
]


def run_synthetic(out: Path, replicates: int) -> dict:
    started = time.perf_counter()
    rows: list[dict] = []
    cells = synthetic_cells()
    for cell in cells:
        rows += score_cell(cell, replicates)[0]
    frame = round_frame(pd.DataFrame(rows), SYNTHETIC_FLOATS)
    write_csv(frame, out / "synthetic.csv")
    wall = round(time.perf_counter() - started, 1)
    return {
        "code": code_version(),
        "replicates": replicates,
        "budget_seconds": SYNTHETIC_BUDGET_S,
        "over_budget": wall > SYNTHETIC_BUDGET_S,
        "grid": {
            "phi": list(PHIS),
            "rho": list(RHOS),
            "drift": list(DRIFTS),
            "delta": list(DELTAS),
            "n_per_period": list(NS),
            "sd_ratio": list(SD_RATIOS),
            "affected_rho": AFFECTED_RHO,
            "affected_delta": list(AFFECTED_DELTAS),
            "covariate_step_fraction": COVARIATE_STEP_FRACTION,
        },
        "n_cells": {
            grid: sum(1 for c in cells if c["grid"] == grid)
            for grid in (GRID_LEVEL, GRID_AFFECTED, GRID_SPREAD)
        },
        "seed_rule": (
            "first 8 bytes, little endian, of sha256 over the cell's sorted "
            "name=repr(value) pairs joined by '|'; the seed column of synthetic.csv"
        ),
        "n_rows": len(frame),
        "wall_seconds": wall,
    }


def cell_label(cell: dict) -> str:
    return ",".join(f"{key}={cell[key]}" for key in CELL_KEYS)


# ---------------------------------------------------------------- main


def parameters() -> dict:
    return {
        "z95": sh.Z95,
        "alpha": sh.ALPHA,
        "min_samples": sh.MIN_SAMPLES,
        "trend_t": sh.TREND_T,
        "bootstrap_replicates": sh.BOOT_REPLICATES,
        "bootstrap_seed": sh.BOOT_SEED,
        "bootstrap_min_block": sh.BOOT_MIN_BLOCK,
        "andrews_bartlett_constant": sh.ANDREWS_BARTLETT,
        "covariate_outside_percentiles": [sh.OUTSIDE_LO, sh.OUTSIDE_HI],
        "refusal_sets": {name: list(rules) for name, rules in sh.REFUSAL_SETS.items()},
    }


def update_run_json(out: Path, sections: dict) -> dict:
    """Merge the beds this call ran into ``run.json``; beds not run keep their entry."""
    path = out / "run.json"
    info = json.loads(path.read_text("utf-8")) if path.exists() else {}
    info["study"] = "shift_intervals"
    info["parameters"] = parameters()
    beds = info.get("beds", {})
    beds.update(sections)
    info["beds"] = beds
    text = json.dumps(info, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")
    return info


def run(out: Path, beds: tuple[str, ...] = BEDS, replicates: int = DEFAULT_REPLICATES) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    sections = {}
    if BED_SYNTHETIC in beds:
        sections[BED_SYNTHETIC] = run_synthetic(out, replicates)
    return update_run_json(out, sections)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default=str(HERE / "results"))
    parser.add_argument("--beds", nargs="+", choices=BEDS, default=list(BEDS))
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    args = parser.parse_args(argv)
    info = run(Path(args.out), tuple(args.beds), args.replicates)
    for bed in args.beds:
        print(f"{bed}: {info['beds'][bed]['wall_seconds']} s, {info['beds'][bed]['n_rows']} rows")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

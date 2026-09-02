"""Stage 3-7 detectors on 3W, under two study designs.

**pooled-cross-well.** Five folds of :func:`tsdive.data.group_holdout`
over the 40 real wells; every model is fitted on the training folds'
windows pooled across wells and scored on the fold under test.

**own-history.** Per-tag limits come from the tag's own earlier windows:
for each instance and variable the baseline is that instance's first
``--baseline-windows`` windows, taken label-blind, and every later window
of the same instance is scored against it. Nothing is fitted across
instances, so a group holdout has nothing to hold out; the split is the
whole corpus and is labelled as such.

The pooled design is a study-design error for the two tools whose object
is a *per-tag* limit. `P-PDG` on one well and `P-PDG` on another differ
by orders of magnitude, so a median/MAD fitted on other wells' windows
measures how far this well sits from those wells, not whether this window
is abnormal for this tag. Both designs are run and both are reported.

===========  =====  ==============  ===================================
key          stage  design          what is fitted, and what the score is
===========  =====  ==============  ===================================
mad          3      both            median/MAD per variable; score = max
                                    robust z across the window's variables
regime       4      pooled          the same, keyed by the window's modal
                                    ``LABEL_state``
population   4      pooled          per-(well, variable) window
                                    percentiles; score = did any frozen
                                    variable get indicted
spc          5      both            individuals limits from sub-group
                                    medians; score = rule hits in the window
mspc         6      both            PCA over the 60x6 sub-group grid;
                                    scores = mean T2 and mean SPE
iforest      7      both            IsolationForest over the honest feature
                                    rows; score = max across the window
===========  =====  ==============  ===================================

Stages 6 and 7 keep the 5-fold well holdout in both designs - a model is
fitted across instances either way - and differ in what the columns mean:
pooled autoscaling on training statistics, or a z-score against the same
instance's own first-k-window mean and standard deviation.

Two departures from the library's happy path, both stated because they
change what the numbers mean:

- **SPC and MSPC read the cached sub-group grid, not the 1 Hz archives.**
  Each window is 3,600 samples; scoring five folds against raw archives
  would walk the whole corpus five times. ``build_windows.py`` caches one
  median per minute per (window, variable), and that 60-point series is
  what the individuals chart and the PCA see. It is a subgrouped chart,
  not a chart of every sample, and it cannot see anything that lives
  inside a single minute.
- **MSPC columns are autoscaled on training statistics.** The six
  variables are pressures in Pa and a temperature in degC, spanning seven
  orders of magnitude. ``fit_pca`` mean-centres but does not scale, so an
  unscaled fit would be a model of ``P-ANULAR`` alone. Each column is
  divided by its training standard deviation before the matrix is built.

    uv run python examples/studies/3w_detectors/run_detectors.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[3] / "src"))

import tsdive
from tsdive.baselines import (
    mad_baseline,
    match_population,
    match_regime,
    population_baseline,
    regime_baselines,
    screen_population,
)
from tsdive.data import group_holdout
from tsdive.detectors.isolation import (
    FEATURE_COLUMNS,
    prepare_rows,
    score_isolation,
    train_isolation,
)
from tsdive.errors import (
    InsufficientQuality,
    MspcAlignmentError,
    PopulationTooSparse,
    RegimeTooSparse,
)
from tsdive.eval import clock_control, fires, ranking_metrics, worst_baseline_threshold
from tsdive.mspc.pca import AlignedMatrix, PcaModel, detect, fit_pca
from tsdive.spc import apply_rules, individuals_limits

HERE = Path(__file__).parent
ROOT = Path(__file__).parents[3]

TOOLS = ("mad", "regime", "population", "spc", "mspc_t2", "mspc_spe", "iforest")
# "clock" is not a detector. It reads no sensor value at all - the score
# how far into its instance's record the window sits. Within a 3W
# instance the fault arrives and stays, so any own-history number the
# clock beats is measuring when the window happened.
OWN_HISTORY_TOOLS = ("mad", "spc", "clock")
STANDARDISED_TOOLS = ("mspc_t2", "mspc_spe", "iforest")
ALL_TOOLS = (*TOOLS, "clock")
VARIANTS = ("all", "no_folder9", "no_transients", "no_folder9_no_transients")
# Folders 3 and 4 are all-positive, so an all-positive baseline is
# possible and the metric excluding those instances is reported too.
OWN_HISTORY_VARIANTS = (*VARIANTS, "no_positive_baseline")

DESIGN_POOLED = "pooled-cross-well"
# The pooled tools' own scores, read back over exactly the windows the
# own-history design scores. Without it the two designs' numbers differ
# in population as well as in method and neither difference is isolated.
DESIGN_POOLED_MATCHED = "pooled-cross-well, own-history windows"
DESIGN_OWN = "own-history"
DESIGN_STANDARDISED = "per-instance-standardised"

# Refusals the design imposes rather than the data: a window inside its
# own instance's baseline, or an instance too short to have one. Counted
# separately so the substantive refusals stay readable.
DESIGN_REFUSAL = "design: "


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
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------- scoring


@dataclass
class ToolScores:
    """One tool's answer for one fold: a score or a refusal, per window."""

    scores: dict[int, float]
    refusals: dict[int, str]

    @classmethod
    def empty(cls) -> ToolScores:
        return cls(scores={}, refusals={})

    def refuse(self, key: int, reason: str) -> None:
        self.refusals[key] = reason

    def record(self, key: int, value: float) -> None:
        if np.isfinite(value):
            self.scores[key] = float(value)
        else:
            self.refusals[key] = "score not finite"


def _quality_frame(values: pd.Series, stamps: pd.Series) -> pd.DataFrame:
    """A value+quality frame the baseliners accept.

    The observations are window medians, already filtered to GOOD samples
    by the feature extractor, so every row here is GOOD by construction.
    """
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(stamps, utc=True).reset_index(drop=True),
            "value": values.astype(float).reset_index(drop=True),
            "quality": ["GOOD"] * len(values),
        }
    )


def _robust_z(value: float, center: float, scale: float) -> float | None:
    if not np.isfinite(scale) or scale <= 0:
        return None
    return abs(float(value) - center) / scale


def _synthetic_history(values: np.ndarray) -> pd.DataFrame:
    """A history frame for the MAD baseliner. Stamps are placeholders.

    ``mad_baseline`` reads value and quality only; the timestamps exist
    because the frame schema has a timestamp column, and nothing here
    depends on them.
    """
    stamps = pd.date_range("2000-01-01", periods=len(values), freq="min", tz="UTC")
    return _quality_frame(pd.Series(values, dtype=float), pd.Series(stamps))


# ------------------------------------------------------- own-history design


@dataclass(frozen=True)
class OwnHistory:
    """Which windows are an instance's own baseline, and which get scored.

    ``score_baseline`` is off for the unaligned design, where a baseline
    window is a refusal the design imposes, and on for the onset-aligned
    one, where the threshold a tool is judged at is the worst score it
    gave that instance's own baseline. Those scores are in-sample by
    construction; that is what an alarm limit set during a learning
    period is, and it is what makes the false-alarm rate readable.
    """

    k: int
    baseline: dict[str, list[int]]
    scored: dict[str, list[int]]
    short: dict[str, str]
    score_baseline: bool = False

    @property
    def baseline_keys(self) -> set[int]:
        return {key for keys in self.baseline.values() for key in keys}

    @property
    def scored_keys(self) -> set[int]:
        return {key for keys in self.scored.values() for key in keys}


def own_history_layout(windows: pd.DataFrame, k: int) -> OwnHistory:
    """First ``k`` windows of each instance are its baseline, label-blind.

    Label-blind is the point: the baseline is chosen by position in the
    instance's own record, which is what a deployed screen has, and never
    by whether the window is labelled normal, which is what it does not.
    """
    baseline: dict[str, list[int]] = {}
    scored: dict[str, list[int]] = {}
    short: dict[str, str] = {}
    ordered = windows.sort_values(["instance", "window_index"])
    for instance, group in ordered.groupby("instance", sort=True):
        keys = [int(v) for v in group["window_key"]]
        if len(keys) <= k:
            short[str(instance)] = (
                f"{DESIGN_REFUSAL}instance has {len(keys)} window(s); the first {k} "
                "are its own-history baseline, leaving none to score"
            )
            continue
        baseline[str(instance)] = keys[:k]
        scored[str(instance)] = keys[k:]
    return OwnHistory(k=k, baseline=baseline, scored=scored, short=short)


def _design_refusals(out: ToolScores, layout: OwnHistory, windows: pd.DataFrame) -> None:
    """Refuse, by name, every window this design structurally cannot score."""
    by_instance = windows.groupby("instance")["window_key"]
    for instance, reason in layout.short.items():
        for key in by_instance.get_group(instance):
            out.refuse(int(key), reason)
    if layout.score_baseline:
        return
    baseline_reason = (
        f"{DESIGN_REFUSAL}window is one of this instance's first {layout.k} windows, "
        "which are its label-blind own-history baseline"
    )
    for keys in layout.baseline.values():
        for key in keys:
            out.refuse(int(key), baseline_reason)


def own_history_baselines(
    grid: pd.DataFrame, layout: OwnHistory, subgroup_cols: list[str]
) -> tuple[dict[tuple[str, str], object], dict[tuple[str, str], str]]:
    """median/MAD per (instance, variable) over that instance's first k windows.

    The history is the sub-group median series, not the k window medians:
    ``mad_baseline`` requires 30 GOOD history samples and k=3 window
    medians is three. Three windows of 60 sub-group medians is 180, and
    those medians are the same quantity the SPC chart reads.
    """
    base = grid[grid["window_key"].isin(layout.baseline_keys)]
    baselines: dict[tuple[str, str], object] = {}
    degenerate: dict[tuple[str, str], str] = {}
    for (instance, variable), group in base.groupby(["instance", "variable"], sort=True):
        pair = (str(instance), str(variable))
        values = group[subgroup_cols].to_numpy(dtype=float).ravel()
        values = values[np.isfinite(values)]
        try:
            baseline = mad_baseline(_synthetic_history(values))
        except InsufficientQuality as e:
            degenerate[pair] = f"{variable}: InsufficientQuality: {e}"
            continue
        if baseline.scale <= 0:
            degenerate[pair] = (
                f"{variable}: the tag's own first-{layout.k}-window MAD is zero, "
                "so no robust z exists"
            )
            continue
        baselines[pair] = baseline
    return baselines, degenerate


def _close_out(out: ToolScores, keys: set[int]) -> None:
    """Every key ends up scored or refused, never neither."""
    for key in sorted(keys):
        if key not in out.scores and key not in out.refusals:
            out.refuse(int(key), "no row for this window in the cached tables")


def score_mad_own_history(
    features: pd.DataFrame,
    layout: OwnHistory,
    baselines: dict,
    degenerate: dict,
    windows: pd.DataFrame,
) -> ToolScores:
    """Own history: robust z of the window median against this tag's."""
    out = ToolScores.empty()
    _design_refusals(out, layout, windows)
    sub = features[features["window_key"].isin(layout.scored_keys)]
    for key, group in sub.groupby("window_key", sort=True):
        zs, reasons = [], []
        for _, row in group.iterrows():
            pair = (str(row["instance"]), str(row["variable"]))
            baseline = baselines.get(pair)
            if baseline is None:
                reasons.append(
                    degenerate.get(pair, f"{row['variable']}: no own-history baseline")
                )
                continue
            if pd.isna(row["median"]):
                reasons.append(f"{row['variable']}: window has no median")
                continue
            z = _robust_z(row["median"], baseline.center, baseline.scale)
            if z is None:  # pragma: no cover - zero scale is filtered above
                reasons.append(f"{row['variable']}: own-history scale is zero")
            else:
                zs.append(z)
        if zs:
            out.record(int(key), max(zs))
        else:
            out.refuse(int(key), "; ".join(sorted(set(reasons))) or "no variable present")
    _close_out(out, layout.scored_keys)
    return out


def score_spc_own_history(
    grid: pd.DataFrame,
    layout: OwnHistory,
    baselines: dict,
    degenerate: dict,
    windows: pd.DataFrame,
    subgroup_cols: list[str],
    window_s: float,
) -> ToolScores:
    """Own history: individuals limits from this tag's own baseline."""
    out = ToolScores.empty()
    _design_refusals(out, layout, windows)
    limits = {
        pair: individuals_limits(baseline.center, baseline.scale)
        for pair, baseline in baselines.items()
    }
    step_s = window_s / len(subgroup_cols)
    sub = grid[grid["window_key"].isin(layout.scored_keys)]
    for key, group in sub.groupby("window_key", sort=True):
        hits, scored, reasons = 0, 0, []
        for _, row in group.iterrows():
            pair = (str(row["instance"]), str(row["variable"]))
            if pair not in limits:
                reasons.append(
                    degenerate.get(pair, f"{row['variable']}: no own-history limits")
                )
                continue
            series = row[subgroup_cols].to_numpy(dtype=float)
            finite = np.isfinite(series)
            if finite.sum() < 2:
                reasons.append(f"{row['variable']}: fewer than two sub-group medians")
                continue
            start = pd.Timestamp(row["window_start"])
            stamps = pd.Series(
                [start + pd.Timedelta(step_s * i, unit="s") for i in np.flatnonzero(finite)]
            )
            hits += len(apply_rules(stamps, pd.Series(series[finite]), limits[pair]))
            scored += 1
        if scored:
            out.record(int(key), float(hits))
        else:
            out.refuse(int(key), "; ".join(sorted(set(reasons))) or "no variable present")
    _close_out(out, layout.scored_keys)
    return out


def score_clock_own_history(windows: pd.DataFrame, layout: OwnHistory) -> ToolScores:
    """Control: the window's position in its own instance's record.

    Reads no sensor value. A 3W instance opens normal and the fault, once
    it arrives, stays, so position alone separates the classes; publishing
    it next to the detectors is the only way to see how much of an
    own-history AUC is the detector and how much is the clock.
    """
    out = ToolScores.empty()
    _design_refusals(out, layout, windows)
    positions = {
        int(key): position
        for keys in layout.scored.values()
        for position, key in enumerate(keys)
    }
    for key, score in clock_control(positions).items():
        out.record(key, score)
    return out


def _instance_zscore(
    frame: pd.DataFrame, layout: OwnHistory, columns: list[str]
) -> tuple[pd.DataFrame, dict]:
    """z-score every column per (instance, variable) on its own first k windows.

    The model downstream then reads deviation from this tag's own recent
    normal rather than absolute level. A column whose baseline standard
    deviation is zero is centred and left unscaled; the count is reported
    rather than hidden, because on those columns the row is a raw offset.
    """
    base = frame[frame["window_key"].isin(layout.baseline_keys)]
    grouped = base.groupby(["instance", "variable"], sort=True)[columns]
    mean = grouped.mean()
    sd = grouped.std(ddof=0)
    zero = ~np.isfinite(sd.to_numpy(dtype=float)) | (sd.to_numpy(dtype=float) <= 0)
    sd = sd.mask(pd.DataFrame(zero, index=sd.index, columns=sd.columns), 1.0)

    out = frame[frame["window_key"].isin(layout.scored_keys)].copy()
    index = pd.MultiIndex.from_frame(out[["instance", "variable"]])
    m = mean.reindex(index).to_numpy(dtype=float)
    s = sd.reindex(index).to_numpy(dtype=float)
    out[columns] = (out[columns].to_numpy(dtype=float) - m) / s
    return out, {
        "n_pairs_with_baseline_stats": len(mean),
        "n_zero_scale_columns": int(zero.sum()),
        "n_scored_rows_without_baseline_stats": int(np.isnan(m).any(axis=1).sum()),
    }


def score_mad(train: pd.DataFrame, test: pd.DataFrame) -> ToolScores:
    """Regime-blind median/MAD per variable, pooled over training wells."""
    out = ToolScores.empty()
    baselines = {}
    degenerate: dict[str, str] = {}
    for variable, group in train.groupby("variable", sort=True):
        usable = group[group["median"].notna()]
        try:
            base = mad_baseline(_quality_frame(usable["median"], usable["window_start"]))
        except InsufficientQuality as e:
            degenerate[str(variable)] = f"InsufficientQuality: {e}"
            continue
        if base.scale <= 0:
            degenerate[str(variable)] = "MAD scale is zero across the training folds"
            continue
        baselines[str(variable)] = base

    for key, group in test.groupby("window_key", sort=True):
        zs = []
        reasons = []
        for _, row in group.iterrows():
            variable = str(row["variable"])
            if variable not in baselines:
                reasons.append(degenerate.get(variable, "no training baseline"))
                continue
            if pd.isna(row["median"]):
                reasons.append(f"{variable}: window has no median")
                continue
            base = baselines[variable]
            z = _robust_z(row["median"], base.center, base.scale)
            if z is None:
                reasons.append(f"{variable}: baseline scale is zero")
            else:
                zs.append(z)
        if zs:
            out.record(int(key), max(zs))
        else:
            out.refuse(int(key), "; ".join(sorted(set(reasons))) or "no variable present")
    return out


def score_regime(train: pd.DataFrame, test: pd.DataFrame, *, min_samples: int = 20) -> ToolScores:
    """The MAD screen conditioned on the window's modal state."""
    out = ToolScores.empty()
    per_variable: dict[str, dict] = {}
    for variable, group in train.groupby("variable", sort=True):
        usable = group[group["median"].notna()]
        # regime_baselines refuses the whole call on the first sparse
        # regime, so the sparse ones are dropped here and counted as
        # refusals when a test window carries them.
        counts = usable["state_mode"].value_counts()
        keep = set(counts[counts >= min_samples].index)
        subset = usable[usable["state_mode"].isin(keep)]
        if subset.empty:
            continue
        try:
            per_variable[str(variable)] = regime_baselines(
                _quality_frame(subset["median"], subset["window_start"]),
                subset["state_mode"].astype(str).reset_index(drop=True),
                min_samples=min_samples,
            )
        except (RegimeTooSparse, InsufficientQuality):  # pragma: no cover - defensive
            continue

    for key, group in test.groupby("window_key", sort=True):
        zs = []
        reasons = []
        for _, row in group.iterrows():
            variable = str(row["variable"])
            bases = per_variable.get(variable)
            if bases is None:
                reasons.append(f"{variable}: no regime baseline at all")
                continue
            if pd.isna(row["median"]):
                reasons.append(f"{variable}: window has no median")
                continue
            try:
                base = match_regime(bases, str(row["state_mode"]))
            except RegimeTooSparse as e:
                reasons.append(f"RegimeTooSparse: {e}")
                continue
            z = _robust_z(row["median"], base.center, base.scale)
            if z is None:
                reasons.append(f"{variable}: regime scale is zero")
            else:
                zs.append(z)
        if zs:
            out.record(int(key), max(zs))
        else:
            out.refuse(int(key), "; ".join(sorted(set(reasons))) or "no variable present")
    return out


def score_population(
    train: pd.DataFrame, test: pd.DataFrame, *, min_windows: int
) -> tuple[ToolScores, dict]:
    """Indict a frozen window when its (well, variable) normally moves."""
    out = ToolScores.empty()
    baselines = population_baseline(
        train,
        asset_col="well",
        variable_col="variable",
        min_windows=min_windows,
    )
    detail = {
        "n_frozen_rows": 0,
        "n_frozen_flagged": 0,
        "n_frozen_refused": 0,
        "flagged_pairs": {},
        "flagged_tags": [],
    }
    for key, group in test.groupby("window_key", sort=True):
        fired = 0
        scored = 0
        reasons = []
        for _, row in group.iterrows():
            frozen = bool(row["distinct_count"] == 1)
            if frozen:
                detail["n_frozen_rows"] += 1
            try:
                base = match_population(baselines, str(row["well"]), str(row["variable"]))
                result = screen_population(row, base)
            except PopulationTooSparse as e:
                reasons.append(str(e))
                if frozen:
                    detail["n_frozen_refused"] += 1
                continue
            scored += 1
            if result.fired:
                fired += 1
                detail["n_frozen_flagged"] += 1
                pair = f"{row['well']}|{row['variable']}"
                detail["flagged_pairs"][pair] = detail["flagged_pairs"].get(pair, 0) + 1
                detail["flagged_tags"].append(
                    {
                        "instance": str(row["instance"]),
                        "variable": str(row["variable"]),
                        "window_index": int(row["window_index"]),
                        "evidence": result.evidence,
                    }
                )
        if scored:
            out.record(int(key), float(fired))
        else:
            out.refuse(int(key), "; ".join(sorted(set(reasons))) or "no variable present")
    return out, detail


def score_spc(
    train_grid: pd.DataFrame,
    test_grid: pd.DataFrame,
    subgroup_cols: list[str],
    window_s: float,
) -> ToolScores:
    """Individuals rules over the window's sub-group median series."""
    out = ToolScores.empty()
    limits = {}
    for variable, group in train_grid.groupby("variable", sort=True):
        values = group[subgroup_cols].to_numpy(dtype=float).ravel()
        values = values[np.isfinite(values)]
        if len(values) < 30:
            continue
        stamps = pd.date_range("2000-01-01", periods=len(values), freq="min", tz="UTC")
        try:
            base = mad_baseline(_quality_frame(pd.Series(values), pd.Series(stamps)))
        except InsufficientQuality:  # pragma: no cover - guarded by the length check
            continue
        if base.scale <= 0:
            continue
        limits[str(variable)] = individuals_limits(base.center, base.scale)

    step_s = window_s / len(subgroup_cols)
    for key, group in test_grid.groupby("window_key", sort=True):
        hits = 0
        scored = 0
        reasons = []
        for _, row in group.iterrows():
            variable = str(row["variable"])
            if variable not in limits:
                reasons.append(f"{variable}: no training limits")
                continue
            series = row[subgroup_cols].to_numpy(dtype=float)
            finite = np.isfinite(series)
            if finite.sum() < 2:
                reasons.append(f"{variable}: fewer than two sub-group medians")
                continue
            start = pd.Timestamp(row["window_start"])
            stamps = pd.Series(
                [start + pd.Timedelta(step_s * i, unit="s") for i in np.flatnonzero(finite)]
            )
            hits += len(apply_rules(stamps, pd.Series(series[finite]), limits[variable]))
            scored += 1
        if scored:
            out.record(int(key), float(hits))
        else:
            out.refuse(int(key), "; ".join(sorted(set(reasons))) or "no variable present")
    return out


def _window_matrix(
    rows: pd.DataFrame, variables: list[str], subgroup_cols: list[str]
) -> np.ndarray:
    """A (n_subgroups, n_variables) block, or a refusal naming what is missing."""
    by_variable = {str(r["variable"]): r for _, r in rows.iterrows()}
    missing = [v for v in variables if v not in by_variable]
    if missing:
        raise MspcAlignmentError(
            f"window lacks {', '.join(missing)}; refusing to align a matrix with a hole in it"
        )
    block = np.column_stack(
        [by_variable[v][subgroup_cols].to_numpy(dtype=float) for v in variables]
    )
    if not np.isfinite(block).all():
        raise MspcAlignmentError(
            "sub-group grid has non-finite cells; refusing to interpolate across tags"
        )
    return block


def _aligned(block: np.ndarray, variables: list[str], start: pd.Timestamp, step_s: float):
    index = pd.DatetimeIndex(
        [start + pd.Timedelta(step_s * i, unit="s") for i in range(len(block))]
    )
    return AlignedMatrix(index=index, columns=list(variables), matrix=block, coverage=1.0)


def score_mspc(
    train_grid: pd.DataFrame,
    test_grid: pd.DataFrame,
    variables: list[str],
    subgroup_cols: list[str],
    window_s: float,
    *,
    autoscale: bool = True,
) -> tuple[ToolScores, ToolScores, dict]:
    """PCA T2 and SPE over the sub-group grid.

    ``autoscale`` divides each column by its training standard deviation.
    It is off when the caller has already z-scored the grid per instance:
    scaling twice would put a second, cross-well scale on top of the
    per-tag one and undo the point of the design.
    """
    t2_out, spe_out = ToolScores.empty(), ToolScores.empty()
    step_s = window_s / len(subgroup_cols)
    blocks = []
    n_train_refused = 0
    for _, rows in train_grid.groupby("window_key", sort=True):
        try:
            blocks.append(_window_matrix(rows, variables, subgroup_cols))
        except MspcAlignmentError:
            n_train_refused += 1
    if not blocks:
        raise MspcAlignmentError("no training window carries every common variable")
    stacked = np.vstack(blocks)
    # Autoscale on training statistics: seven orders of magnitude between
    # a pressure in Pa and a temperature in degC, and fit_pca centres but
    # does not scale.
    if autoscale:
        scale = stacked.std(axis=0)
        scale = np.where(scale > 0, scale, 1.0)
    else:
        scale = np.ones(stacked.shape[1])
    start0 = pd.Timestamp("2000-01-01T00:00:00+00:00")
    model: PcaModel = fit_pca(_aligned(stacked / scale, variables, start0, step_s))

    n_test_refused = 0
    for key, rows in test_grid.groupby("window_key", sort=True):
        try:
            block = _window_matrix(rows, variables, subgroup_cols)
        except MspcAlignmentError as e:
            t2_out.refuse(int(key), str(e))
            spe_out.refuse(int(key), str(e))
            n_test_refused += 1
            continue
        start = pd.Timestamp(rows.iloc[0]["window_start"])
        found = detect(model, _aligned(block / scale, variables, start, step_s))
        t2_out.record(int(key), float(np.mean(found.t2)))
        spe_out.record(int(key), float(np.mean(found.spe)))
    return (
        t2_out,
        spe_out,
        {
            "n_train_windows_aligned": len(blocks),
            "n_train_windows_refused": n_train_refused,
            "n_test_windows_refused": n_test_refused,
            "n_components": int(model.components.shape[0]),
            "explained_variance": [round(v, 6) for v in model.explained_variance],
        },
    )


def score_iforest(train: pd.DataFrame, test: pd.DataFrame, seed: int) -> ToolScores:
    """IsolationForest over the honest per-variable feature rows."""
    out = ToolScores.empty()
    tagged_train = train.assign(tag=train["variable"], window_start=train["window_start"])
    rows, _ = prepare_rows(tagged_train)
    rows, n_train_overflow = _float32_representable(rows)
    if len(rows) < 2:
        raise InsufficientQuality("fewer than two complete training feature rows")
    model = train_isolation(rows, seed=seed)

    tagged_test = test.assign(tag=test["variable"], window_start=test["window_start"])
    complete = tagged_test.dropna(subset=list(FEATURE_COLUMNS))
    complete, n_test_overflow = _float32_representable(complete)
    n_excluded = len(tagged_test) - len(complete)
    if len(complete):
        scores = score_isolation(
            model, complete, n_excluded_incomplete=n_excluded
        ).scores
        frame = pd.DataFrame(
            {"window_key": complete["window_key"].to_numpy(), "score": scores.to_numpy()}
        )
        for key, group in frame.groupby("window_key", sort=True):
            out.record(int(key), float(group["score"].max()))
    scored = set(out.scores)
    for key in sorted(test["window_key"].unique()):
        if int(key) not in scored:
            out.refuse(
                int(key),
                "every feature row for this window was incomplete or outside "
                "float32 range, so none reached the forest",
            )
    return out, {"n_train_overflow": n_train_overflow, "n_test_overflow": n_test_overflow}


def mask_beyond_float32(frame: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, dict]:
    """Blank the cells no float32 can hold, and count them.

    3W ships a historian null sentinel as a number: three ``WELL-00006``
    instances hold ``P-PDG`` at -1.180116e+42 for their whole life, and
    3W publishes no engineering ranges, so nothing downstream can call it
    out of range. ``DataPhysics.implausible_magnitude_count`` counts it in
    the profile; the study removes it once, here, for every design, because
    every detector below either casts to float32 - IsolationForest does,
    silently, where the value becomes -inf - or divides by a scale that
    one value dictates.
    """
    block = frame[columns].to_numpy(dtype=float)
    bad = np.abs(block) > float(np.finfo(np.float32).max)
    detail = {"n_cells": int(bad.sum()), "n_rows": int(bad.any(axis=1).sum())}
    if not bad.any():
        return frame, detail
    block[bad] = np.nan
    out = frame.copy()
    out[columns] = block
    return out, detail


def _float32_representable(rows: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop feature rows sklearn would silently overflow, and count them.

    ``IsolationForest`` casts its input to float32 internally. Three
    ``WELL-00006`` instances hold ``P-PDG`` pinned at -1.180116e+42 for
    their whole life - a historian null sentinel that 3W ships as a
    number, and that tsdive cannot call out of range because 3W
    publishes no engineering ranges. Cast to float32 it becomes -inf, so
    the forest would be fitted and scored on a matrix containing
    infinities. Excluded and counted instead.
    """
    limit = float(np.finfo(np.float32).max)
    values = rows[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    keep = (np.abs(values) <= limit).all(axis=1)
    return rows[keep], int((~keep).sum())


# ---------------------------------------------------------------- metrics


def metrics(y: np.ndarray, scores: np.ndarray) -> dict:
    return ranking_metrics(y, scores).to_dict()


# --------------------------------------------------------- onset-aligned

ALIGNED_TOOLS = ("mad", "spc", "mspc_t2", "mspc_spe", "iforest", "clock", "clock_flat")
ALIGNED_VARIANTS = ("all", "no_folder9", "transient_onset", "steady_onset")
DESIGN_ALIGNED = "onset-aligned"
# Baseline and pre are the windows the design calls normal, so they are
# what a one-class model is fitted on. post0 and post1 are never seen by
# any fit.
TRAIN_ROLES = ("baseline", "pre")


def aligned_layout(windows: pd.DataFrame) -> OwnHistory:
    """Baseline = this instance's ``role == "baseline"`` windows; all six scored."""
    baseline: dict[str, list[int]] = {}
    scored: dict[str, list[int]] = {}
    ordered = windows.sort_values(["instance", "onset_offset"])
    k = 0
    for instance, group in ordered.groupby("instance", sort=True):
        base = [int(v) for v in group.loc[group["role"] == "baseline", "window_key"]]
        baseline[str(instance)] = base
        scored[str(instance)] = [int(v) for v in group["window_key"]]
        k = max(k, len(base))
    return OwnHistory(k=k, baseline=baseline, scored=scored, short={}, score_baseline=True)


def score_clock_aligned(windows: pd.DataFrame, layout: OwnHistory) -> ToolScores:
    """Control: how many windows into its own record the window sits.

    Not the offset from onset - that is the design's own coordinate and no
    deployed detector knows it before the fault has happened. This is the
    same quantity the unaligned study's clock control read, and it is the
    only time signal left once every instance contributes the identical
    six relative positions.
    """
    out = ToolScores.empty()
    index = windows.set_index("window_key")["window_index"]
    for key, score in clock_control(index.loc[sorted(layout.scored_keys)]).items():
        out.record(int(key), score)
    return out


def score_clock_flat(windows: pd.DataFrame, layout: OwnHistory) -> ToolScores:
    """The clock with the one thing the design hands it taken back out.

    ``window_index - onset_offset`` is where this instance's onset sits in
    its own record, and it is the same number for all six of its windows.
    Every pre-versus-post comparison is therefore a tie and the AUC is
    exactly 0.5 - which is the point: whatever the clock control scores
    above 0.5 under this design is the fixed two-to-three window spacing
    the design itself puts between the pre window and the post windows,
    and nothing else.
    """
    out = ToolScores.empty()
    frame = windows.set_index("window_key")[["window_index", "onset_offset"]]
    keys = sorted(layout.scored_keys)
    position = frame.loc[keys, "window_index"] - frame.loc[keys, "onset_offset"]
    for key, score in clock_control(position).items():
        out.record(int(key), score)
    return out


def aligned_per_instance(
    windows: pd.DataFrame, answer: ToolScores, tool: str
) -> pd.DataFrame:
    """One row per instance: its baseline threshold and its three test scores.

    The threshold is the largest score the tool gave that instance's own
    baseline windows - the worst thing it saw while it was learning. A
    test window is a detection when it scores strictly above it.
    """
    rows = []
    for instance, group in windows.groupby("instance", sort=True):
        by_role = {
            str(role): [int(v) for v in sub["window_key"]]
            for role, sub in group.groupby("role")
        }
        first = group.iloc[0]
        base = [answer.scores[k] for k in by_role.get("baseline", []) if k in answer.scores]
        row = {
            "tool": tool,
            "instance": str(instance),
            "well": str(first["well"]),
            "folder_label": str(first["folder_label"]),
            "onset_kind": str(first["onset_kind"]),
            "fold": int(first["fold"]),
            "n_baseline_scored": len(base),
            "baseline_max": worst_baseline_threshold(base) if base else None,
        }
        reasons = []
        for role in ("pre", "post0", "post1"):
            keys = by_role.get(role, [])
            key = keys[0] if keys else None
            row[role] = answer.scores.get(key) if key is not None else None
            if key is not None and key not in answer.scores:
                reasons.append(answer.refusals.get(key, "no score and no refusal"))
        row["refusal"] = "; ".join(sorted(set(reasons))) or None
        rows.append(row)
    return pd.DataFrame(rows)


def aligned_variant_mask(frame: pd.DataFrame, variant: str) -> pd.Series:
    keep = pd.Series(True, index=frame.index)
    if variant == "no_folder9":
        keep &= frame["folder_label"].astype(str) != "9"
    if variant == "transient_onset":
        keep &= frame["onset_kind"].astype(str) == "transient"
    if variant == "steady_onset":
        keep &= frame["onset_kind"].astype(str) == "steady"
    return keep


def _aligned_ranking(usable: pd.DataFrame) -> dict:
    """Pre windows as negatives, post windows as positives, pooled."""
    y, scores = [], []
    for _, row in usable.iterrows():
        if row["pre"] is not None and not pd.isna(row["pre"]):
            y.append(0)
            scores.append(float(row["pre"]))
        for role in ("post0", "post1"):
            if row[role] is not None and not pd.isna(row[role]):
                y.append(1)
                scores.append(float(row[role]))
    return metrics(np.array(y, dtype=int), np.array(scores, dtype=float))


def aligned_metrics(frame: pd.DataFrame) -> dict:
    """AUC, paired hit rate and detection at each instance's own threshold.

    Three questions, and they are not the same question:

    - **AUC** pools every scored *pre* window as a negative and every
      scored *post* window as a positive. The clock cannot win it,
      because the design gives every instance the same relative layout.
    - **Paired hit rate** asks only whether the post window outscores the
      pre window *of the same instance*. Any score that rises with time
      wins it outright, so the clock control's row is the number to read
      it against.
    - **Detection at own threshold** is the deployable question: with the
      alarm set at the worst baseline score, how often does it fire
      before the fault (false alarm) and how soon after it.
    """
    usable = frame[frame["baseline_max"].notna()]
    ranking = _aligned_ranking(usable)

    def paired(role: str) -> tuple[int, float | None]:
        sub = usable[usable["pre"].notna() & usable[role].notna()]
        if not len(sub):
            return 0, None
        wins = int((sub[role].to_numpy(dtype=float) > sub["pre"].to_numpy(dtype=float)).sum())
        return len(sub), round(wins / len(sub), 4)

    def fired(role: str) -> tuple[int, float | None]:
        sub = usable[usable[role].notna()]
        if not len(sub):
            return 0, None
        hits = sum(
            fires(score, limit)
            for score, limit in zip(
                sub[role].to_numpy(dtype=float),
                sub["baseline_max"].to_numpy(dtype=float),
                strict=True,
            )
        )
        return len(sub), round(hits / len(sub), 4)

    delays: list[int] = []
    n_never = 0
    for _, row in usable.iterrows():
        limit = float(row["baseline_max"])
        delay = None
        for offset, role in enumerate(("post0", "post1")):
            value = row[role]
            if value is not None and not pd.isna(value) and fires(float(value), limit):
                delay = offset
                break
        if delay is None:
            n_never += 1
        else:
            delays.append(delay)

    n_pre_paired, hit0 = paired("post0")
    _, hit1 = paired("post1")
    n_pre_fired, far = fired("pre")
    n_post0, det0 = fired("post0")
    n_post1, det1 = fired("post1")
    # Stages 6 and 7 fit one model per fold, so their pooled AUC ranks
    # scores five separate models produced. The per-fold range says how
    # much of the pooled number that mixing could be worth.
    per_fold = [
        _aligned_ranking(part)["roc_auc"] for _, part in usable.groupby("fold", sort=True)
    ]
    per_fold = [v for v in per_fold if v is not None]
    return {
        "n_instances": len(frame),
        "n_instances_with_threshold": len(usable),
        "n_instances_refused": int(len(frame) - len(usable)),
        "roc_auc": ranking["roc_auc"],
        "pr_auc": ranking["pr_auc"],
        "n_pre_windows": ranking["n_neg"],
        "n_post_windows": ranking["n_pos"],
        "ranking_refusal": ranking["refusal"],
        "n_paired": n_pre_paired,
        "paired_hit_post0": hit0,
        "paired_hit_post1": hit1,
        "n_pre_at_threshold": n_pre_fired,
        "false_alarm_rate_pre": far,
        "n_post0_at_threshold": n_post0,
        "detection_rate_post0": det0,
        "n_post1_at_threshold": n_post1,
        "detection_rate_post1": det1,
        "n_detected": len(delays),
        "n_never_detected": n_never,
        "median_delay_windows": (float(np.median(delays)) if delays else None),
        "n_folds_scored": len(per_fold),
        "auc_fold_min": min(per_fold) if per_fold else None,
        "auc_fold_max": max(per_fold) if per_fold else None,
    }


def variant_mask(windows: pd.DataFrame, variant: str) -> pd.Series:
    keep = pd.Series(True, index=windows.index)
    if "no_folder9" in variant:
        keep &= windows["folder_label"].astype(str) != "9"
    if "no_transients" in variant:
        keep &= ~windows["transient_only"].astype(bool)
    if variant == "no_positive_baseline":
        # Folders 3 and 4 are all-positive, so some instances' baseline
        # windows are themselves fault windows. Those instances are
        # dropped here, not silently kept.
        keep &= ~windows["baseline_all_positive"].astype(bool)
    return keep


# ------------------------------------------------------ onset-aligned driver


def run_aligned(args) -> int:
    """Score the aligned cache and write the aligned result tables.

    Same tools and the same thresholds as the unaligned run - nothing is
    retuned for this design, because a tool that had to be retuned would
    be answering a different question and the comparison would be empty.
    """
    cache = Path(args.windows)
    manifest = json.loads((cache / "MANIFEST.json").read_text("utf-8"))
    windows = pd.read_parquet(cache / "windows.parquet").reset_index(drop=True)
    features = pd.read_parquet(cache / "features.parquet")
    grid = pd.read_parquet(cache / "subgroup_medians.parquet")
    variables = list(manifest["common_variables"])
    window_s = float(manifest["window_s"])
    subgroup_cols = [c for c in grid.columns if c.startswith("s") and c[1:].isdigit()]

    windows["window_key"] = np.arange(len(windows))
    keys = windows.set_index(["instance", "window_index"])["window_key"]
    for frame in (features, grid):
        frame["window_key"] = keys.loc[
            pd.MultiIndex.from_frame(frame[["instance", "window_index"]])
        ].to_numpy()
    carry = windows.set_index("window_key")[
        ["well", "folder_label", "window_start", "state_mode", "label", "role"]
    ]
    features = features.join(carry, on="window_key")
    grid = grid.join(carry[["window_start"]], on="window_key")
    features, feature_beyond = mask_beyond_float32(features, list(FEATURE_COLUMNS))
    grid, grid_beyond = mask_beyond_float32(grid, subgroup_cols)

    onsets = pd.read_csv(args.onsets)
    kind_of = onsets.set_index("instance")["onset_kind"]
    windows["onset_kind"] = windows["instance"].map(kind_of).astype(str)

    layout = aligned_layout(windows)
    started = time.perf_counter()
    baselines, degenerate = own_history_baselines(grid, layout, subgroup_cols)
    answers: dict[str, ToolScores] = {
        "mad": score_mad_own_history(features, layout, baselines, degenerate, windows),
        "spc": score_spc_own_history(
            grid, layout, baselines, degenerate, windows, subgroup_cols, window_s
        ),
        "clock": score_clock_aligned(windows, layout),
        "clock_flat": score_clock_flat(windows, layout),
    }

    # Stages 6 and 7 fit across instances, so the well holdout stays: a
    # model is scored only on wells it never saw. Each column is first
    # z-scored against the same instance's own baseline windows, which is
    # the per-instance-standardised design of the unaligned run.
    std_grid, grid_std_detail = _instance_zscore(grid, layout, subgroup_cols)
    std_features, feature_std_detail = _instance_zscore(
        features, layout, list(FEATURE_COLUMNS)
    )
    split = group_holdout(
        windows,
        group_col="well",
        stratum_col="folder_label",
        n_folds=args.n_folds,
        seed=args.seed,
    )
    split.leakage_check()
    windows["fold"] = [split.fold_of_group[str(w)] for w in windows["well"]]
    train_roles = set(windows.loc[windows["role"].isin(TRAIN_ROLES), "window_key"])
    for tool in STANDARDISED_TOOLS:
        answers[tool] = ToolScores.empty()
    mspc_detail: list[dict] = []
    iforest_detail: list[dict] = []
    for fold in range(args.n_folds):
        test_keys = set(windows.loc[split.test_indices(windows, fold), "window_key"])
        train_keys = set(windows.loc[split.train_indices(windows, fold), "window_key"])
        fit_keys = train_keys & train_roles
        try:
            t2, spe, info = score_mspc(
                std_grid[std_grid["window_key"].isin(fit_keys)],
                std_grid[std_grid["window_key"].isin(test_keys)],
                variables,
                subgroup_cols,
                window_s,
                autoscale=False,
            )
        except MspcAlignmentError as e:
            refused = ToolScores(
                scores={}, refusals={int(k): str(e) for k in sorted(test_keys)}
            )
            t2, spe, info = refused, refused, {"refusal": str(e)}
        info["fold"] = fold
        info["n_fit_windows"] = len(fit_keys)
        mspc_detail.append(info)
        try:
            forest, f_info = score_iforest(
                std_features[std_features["window_key"].isin(fit_keys)],
                std_features[std_features["window_key"].isin(test_keys)],
                args.seed,
            )
        except InsufficientQuality as e:  # pragma: no cover - defensive
            forest = ToolScores(
                scores={}, refusals={int(k): str(e) for k in sorted(test_keys)}
            )
            f_info = {"refusal": str(e)}
        f_info["fold"] = fold
        iforest_detail.append(f_info)
        for tool, part in (("mspc_t2", t2), ("mspc_spe", spe), ("iforest", forest)):
            answers[tool].scores.update(part.scores)
            answers[tool].refusals.update(part.refusals)
        print(
            f"  aligned fold {fold}: {len(test_keys)} windows, "
            f"{len(split.groups(fold))} well(s) held out, {len(fit_keys)} fitted on",
            flush=True,
        )
    for tool in STANDARDISED_TOOLS:
        _close_out(answers[tool], layout.scored_keys)
    elapsed = time.perf_counter() - started

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_instance = pd.concat(
        [aligned_per_instance(windows, answers[t], t) for t in ALIGNED_TOOLS],
        ignore_index=True,
    )
    per_instance.to_csv(
        out_dir / "aligned_per_instance.csv", index=False, lineterminator="\n"
    )
    score_rows = [
        {
            "tool": tool,
            "design": DESIGN_ALIGNED,
            "window_key": int(key),
            "score": value,
        }
        for tool in ALIGNED_TOOLS
        for key, value in sorted(answers[tool].scores.items())
    ]
    scores_df = pd.DataFrame(score_rows)
    scores_df["score"] = scores_df["score"].map(lambda v: f"{v:.6g}")
    scores_df.to_csv(out_dir / "aligned_scores.csv", index=False, lineterminator="\n")
    windows[
        [
            "window_key",
            "instance",
            "well",
            "folder_label",
            "window_index",
            "onset_offset",
            "role",
            "design_label",
            "label",
            "transient_only",
            "onset_kind",
            "classes",
            "window_start",
            "fold",
        ]
    ].to_csv(out_dir / "aligned_windows.csv", index=False, lineterminator="\n")

    summary_rows = []
    for tool in ALIGNED_TOOLS:
        frame = per_instance[per_instance["tool"] == tool]
        for variant in ALIGNED_VARIANTS:
            kept = frame[aligned_variant_mask(frame, variant)]
            row = {"tool": tool, "design": DESIGN_ALIGNED, "variant": variant}
            row.update(aligned_metrics(kept))
            summary_rows.append(row)
        for folder, part in frame.groupby("folder_label", sort=True):
            row = {"tool": tool, "design": DESIGN_ALIGNED, "variant": f"folder_{folder}"}
            row.update(aligned_metrics(part))
            summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "aligned_summary.csv", index=False, lineterminator="\n")

    onset_summary = json.loads(Path(args.onset_summary).read_text("utf-8"))
    n_all = int(onset_summary["n_instances"])
    n_evaluable = int(onset_summary["n_evaluable"])
    provenance = {
        "design": DESIGN_ALIGNED,
        "dataset_manifest_sha256": sha256_file(Path(args.source) / "MANIFEST.sha256.json"),
        "aligned_manifest": manifest,
        "onsets": onset_summary,
        "code": {"git_head": git_head(), "tsdive_version": tsdive.__version__},
        "split": {
            "kind": "5-fold group holdout by well over the evaluable instances",
            "group_col": "well",
            "stratum_col": "folder_label",
            "n_folds": args.n_folds,
            "seed": args.seed,
            "folds": [list(split.groups(f)) for f in range(args.n_folds)],
            "fold_sizes": list(split.fold_sizes),
            "applies_to": list(STANDARDISED_TOOLS),
            "why_not_the_others": (
                "the own-history tools fit nothing across instances: each tag's "
                "limits come from that instance's own baseline windows, so a "
                "group holdout has no cross-instance parameter to protect"
            ),
        },
        "refusals": {
            "n_instances_in_corpus": n_all,
            "n_instances_evaluable": n_evaluable,
            "n_instances_refused_by_design": n_all - n_evaluable,
            "design_refusal_reason": (
                f"fewer than {onset_summary['pre_onset_windows_required']} whole "
                f"windows before onset or fewer than "
                f"{onset_summary['post_onset_windows_required']} after it, or no "
                "fault row at all"
            ),
        },
        "threshold_rule": (
            "each instance's own alarm limit is the largest score the tool gave "
            "that instance's baseline windows; a test window fires when it scores "
            "strictly above it"
        ),
        "beyond_float32": {"features": feature_beyond, "subgroup_grid": grid_beyond},
        "standardisation": {
            "grid": grid_std_detail,
            "features": feature_std_detail,
        },
        "n_pairs_with_a_usable_baseline": len(baselines),
        "n_pairs_refused_degenerate_baseline": len(degenerate),
        "wall_seconds": round(elapsed, 1),
    }
    (out_dir / "aligned_run.json").write_text(
        json.dumps(
            {
                "provenance": provenance,
                "mspc": mspc_detail,
                "iforest": iforest_detail,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    shown = [
        "tool",
        "variant",
        "n_instances_with_threshold",
        "roc_auc",
        "paired_hit_post0",
        "paired_hit_post1",
        "false_alarm_rate_pre",
        "detection_rate_post0",
        "detection_rate_post1",
        "median_delay_windows",
    ]
    print(summary_df[summary_df["variant"] == "all"][shown].to_string(index=False))
    print(f"{len(windows)} aligned windows, {elapsed:.1f}s")
    return 0


# ---------------------------------------------------------------- driver


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", default=None)
    parser.add_argument("--source", default="data/3w")
    parser.add_argument("--out", default=str(HERE / "results"))
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-baseline-windows", type=int, default=20)
    parser.add_argument(
        "--baseline-windows",
        type=int,
        default=3,
        help="own-history design: how many of an instance's first windows are "
        "its label-blind baseline (default 3 = 3 h)",
    )
    parser.add_argument(
        "--aligned",
        action="store_true",
        help="score the onset-aligned cache instead, where every instance "
        "contributes the same six windows relative to its own fault onset",
    )
    parser.add_argument("--onsets", default="data/3w_windows/onsets.csv")
    parser.add_argument("--onset-summary", default=str(HERE / "results/onsets.json"))
    args = parser.parse_args(argv)
    if args.windows is None:
        args.windows = "data/3w_windows_aligned" if args.aligned else "data/3w_windows"
    if args.aligned:
        return run_aligned(args)

    cache = Path(args.windows)
    manifest = json.loads((cache / "MANIFEST.json").read_text("utf-8"))
    windows = pd.read_parquet(cache / "windows.parquet").reset_index(drop=True)
    features = pd.read_parquet(cache / "features.parquet")
    grid = pd.read_parquet(cache / "subgroup_medians.parquet")
    variables = list(manifest["common_variables"])
    window_s = float(manifest["window_s"])
    subgroup_cols = [c for c in grid.columns if c.startswith("s") and c[1:].isdigit()]

    windows["window_key"] = np.arange(len(windows))
    keys = windows.set_index(["instance", "window_index"])["window_key"]
    for frame in (features, grid):
        frame["window_key"] = keys.loc[
            pd.MultiIndex.from_frame(frame[["instance", "window_index"]])
        ].to_numpy()
    carry = windows.set_index("window_key")[
        ["well", "folder_label", "window_start", "state_mode", "label", "transient_only"]
    ]
    features = features.join(carry, on="window_key")
    grid = grid.join(carry[["window_start"]], on="window_key")
    features, feature_beyond = mask_beyond_float32(features, list(FEATURE_COLUMNS))
    grid, grid_beyond = mask_beyond_float32(grid, subgroup_cols)

    layout = own_history_layout(windows, args.baseline_windows)
    label_by_key = windows.set_index("window_key")["label"]
    positive_baseline = {
        instance: bool(label_by_key.loc[keys].astype(int).eq(1).all())
        for instance, keys in layout.baseline.items()
    }
    windows["baseline_all_positive"] = [
        bool(positive_baseline.get(str(i), False)) for i in windows["instance"]
    ]
    all_keys = [int(k) for k in windows["window_key"]]

    split = group_holdout(
        windows,
        group_col="well",
        stratum_col="folder_label",
        n_folds=args.n_folds,
        seed=args.seed,
    )
    split.leakage_check()
    # The population screen asks "does this sensor normally move ON THIS
    # WELL", so its baseline is the well's other instances - which the
    # well-level split holds out along with the window under test. Under
    # that gate it can only refuse, and that refusal is published as the
    # headline result. It is also run under a second split that holds out
    # whole instances instead, which is the split its question actually
    # has, and every number from it is labelled with that split.
    instance_split = group_holdout(
        windows,
        group_col="instance",
        stratum_col="folder_label",
        n_folds=args.n_folds,
        seed=args.seed,
    )
    instance_split.leakage_check()

    started = time.perf_counter()
    per_fold: list[dict] = []
    score_rows: list[dict] = []
    population_detail: list[dict] = []
    mspc_detail: list[dict] = []
    iforest_detail: list[dict] = []

    indexed = windows.set_index("window_key")

    def emit(
        design: str,
        split_name: str,
        fold: int,
        tool: str,
        answer: ToolScores,
        keys: list[int],
        variants: tuple[str, ...] = VARIANTS,
    ):
        fold_windows = indexed.loc[keys]
        # Every scored window is written out so the report's ROC curves
        # are drawn from the numbers the run produced, not recomputed.
        score_rows.extend(
            {
                "tool": tool,
                "design": design,
                "split": split_name,
                "fold": fold,
                "window_key": int(key),
                "score": answer.scores[int(key)],
            }
            for key in fold_windows.index
            if int(key) in answer.scores
        )
        for variant in variants:
            kept = fold_windows[variant_mask(fold_windows, variant)]
            scored_keys = [k for k in kept.index if int(k) in answer.scores]
            refused_keys = [k for k in kept.index if int(k) in answer.refusals]
            n_design = sum(
                1 for k in refused_keys if answer.refusals[int(k)].startswith(DESIGN_REFUSAL)
            )
            y = kept.loc[scored_keys, "label"].to_numpy(dtype=int)
            s = np.array([answer.scores[int(k)] for k in scored_keys], dtype=float)
            row = {
                "tool": tool,
                "design": design,
                "split": split_name,
                "fold": fold,
                "variant": variant,
                "n_windows": len(kept),
                "n_scored": len(scored_keys),
                "n_refused": len(refused_keys),
                "n_refused_by_design": n_design,
                "refused_frac": round(len(refused_keys) / max(len(kept), 1), 4),
            }
            row.update(
                metrics(y, s)
                if len(scored_keys)
                else {
                    "roc_auc": None,
                    "pr_auc": None,
                    "precision_at_recall_50": None,
                    "n_pos": 0,
                    "n_neg": 0,
                    "refusal": "every window in this fold was refused",
                }
            )
            per_fold.append(row)

    for fold in range(args.n_folds):
        test_idx = split.test_indices(windows, fold)
        train_idx = split.train_indices(windows, fold)
        fold_keys = [int(k) for k in windows.loc[test_idx, "window_key"]]
        test_keys = set(fold_keys)
        train_keys = set(windows.loc[train_idx, "window_key"])
        f_test = features[features["window_key"].isin(test_keys)]
        f_train = features[features["window_key"].isin(train_keys)]

        answers: dict[str, ToolScores] = {}
        answers["mad"] = score_mad(f_train, f_test)
        answers["regime"] = score_regime(f_train, f_test)
        answers["population"], pop_detail = score_population(
            f_train, f_test, min_windows=args.min_baseline_windows
        )
        pop_detail.update(fold=fold, split="well")
        population_detail.append(pop_detail)
        g_train = grid[grid["window_key"].isin(train_keys)]
        g_test = grid[grid["window_key"].isin(test_keys)]
        answers["spc"] = score_spc(g_train, g_test, subgroup_cols, window_s)
        try:
            t2, spe, mspc_info = score_mspc(
                g_train, g_test, variables, subgroup_cols, window_s
            )
        except MspcAlignmentError as e:
            refused = ToolScores(
                scores={}, refusals={int(k): str(e) for k in sorted(test_keys)}
            )
            t2, spe, mspc_info = refused, refused, {"refusal": str(e)}
        answers["mspc_t2"], answers["mspc_spe"] = t2, spe
        mspc_info["fold"] = fold
        mspc_detail.append(mspc_info)
        answers["iforest"], iforest_info = score_iforest(f_train, f_test, args.seed)
        iforest_info["fold"] = fold
        iforest_detail.append(iforest_info)

        for tool in TOOLS:
            emit(DESIGN_POOLED, "well", fold, tool, answers[tool], fold_keys)
        matched = [k for k in fold_keys if k in layout.scored_keys]
        for tool in TOOLS:
            emit(
                DESIGN_POOLED_MATCHED,
                "well",
                fold,
                tool,
                answers[tool],
                matched,
                OWN_HISTORY_VARIANTS,
            )
        print(
            f"  well fold {fold}: {len(test_idx)} windows, "
            f"{len(split.groups(fold))} well(s) held out",
            flush=True,
        )

    for fold in range(args.n_folds):
        test_idx = instance_split.test_indices(windows, fold)
        train_idx = instance_split.train_indices(windows, fold)
        test_keys = set(windows.loc[test_idx, "window_key"])
        train_keys = set(windows.loc[train_idx, "window_key"])
        answer, pop_detail = score_population(
            features[features["window_key"].isin(train_keys)],
            features[features["window_key"].isin(test_keys)],
            min_windows=args.min_baseline_windows,
        )
        pop_detail.update(fold=fold, split="instance")
        population_detail.append(pop_detail)
        emit(
            DESIGN_POOLED,
            "instance",
            fold,
            "population",
            answer,
            [int(k) for k in windows.loc[test_idx, "window_key"]],
        )
        print(
            f"  instance fold {fold}: {len(test_idx)} windows, "
            f"{len(instance_split.groups(fold))} instance(s) held out",
            flush=True,
        )

    # --- own-history: nothing is fitted across instances, so there is
    # nothing for a group holdout to hold out. The split is every window
    # of every instance long enough to have a baseline.
    baselines, degenerate = own_history_baselines(grid, layout, subgroup_cols)
    own_answers = {
        "mad": score_mad_own_history(features, layout, baselines, degenerate, windows),
        "spc": score_spc_own_history(
            grid, layout, baselines, degenerate, windows, subgroup_cols, window_s
        ),
        "clock": score_clock_own_history(windows, layout),
    }
    for tool in OWN_HISTORY_TOOLS:
        emit(
            DESIGN_OWN,
            "own-history",
            0,
            tool,
            own_answers[tool],
            all_keys,
            OWN_HISTORY_VARIANTS,
        )
    print(
        f"  own-history: {len(layout.scored_keys)} scored windows over "
        f"{len(layout.scored)} instances, {len(layout.short)} instance(s) too short",
        flush=True,
    )

    # --- per-instance standardisation: the well holdout stays (a model
    # IS fitted across instances here), but each column is a z-score
    # against that instance's own first-k-window statistics.
    std_grid, grid_std_detail = _instance_zscore(grid, layout, subgroup_cols)
    std_features, feature_std_detail = _instance_zscore(
        features, layout, list(FEATURE_COLUMNS)
    )
    std_mspc_detail: list[dict] = []
    std_iforest_detail: list[dict] = []
    for fold in range(args.n_folds):
        test_idx = split.test_indices(windows, fold)
        train_idx = split.train_indices(windows, fold)
        fold_keys = [int(k) for k in windows.loc[test_idx, "window_key"]]
        test_keys = set(fold_keys)
        train_keys = set(windows.loc[train_idx, "window_key"])
        answers = {}
        try:
            t2, spe, info = score_mspc(
                std_grid[std_grid["window_key"].isin(train_keys)],
                std_grid[std_grid["window_key"].isin(test_keys)],
                variables,
                subgroup_cols,
                window_s,
                autoscale=False,
            )
        except MspcAlignmentError as e:  # pragma: no cover - defensive
            refused = ToolScores(scores={}, refusals={int(k): str(e) for k in fold_keys})
            t2, spe, info = refused, refused, {"refusal": str(e)}
        answers["mspc_t2"], answers["mspc_spe"] = t2, spe
        info["fold"] = fold
        std_mspc_detail.append(info)
        answers["iforest"], info = score_iforest(
            std_features[std_features["window_key"].isin(train_keys)],
            std_features[std_features["window_key"].isin(test_keys)],
            args.seed,
        )
        info["fold"] = fold
        std_iforest_detail.append(info)
        for answer in answers.values():
            _design_refusals(answer, layout, windows)
        for tool in STANDARDISED_TOOLS:
            emit(
                DESIGN_STANDARDISED,
                "well",
                fold,
                tool,
                answers[tool],
                fold_keys,
                OWN_HISTORY_VARIANTS,
            )
        print(
            f"  standardised fold {fold}: {len(test_idx)} windows, "
            f"{len(split.groups(fold))} well(s) held out",
            flush=True,
        )
    elapsed = time.perf_counter() - started

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    fold_df = pd.DataFrame(per_fold)
    fold_df.to_csv(out_dir / "per_fold.csv", index=False, lineterminator="\n")
    scores_df = pd.DataFrame(score_rows).sort_values(
        ["tool", "design", "split", "window_key"]
    )
    scores_df["score"] = scores_df["score"].map(lambda v: f"{v:.6g}")
    scores_df.to_csv(out_dir / "scores.csv", index=False, lineterminator="\n")
    windows[
        [
            "window_key",
            "instance",
            "well",
            "folder_label",
            "window_index",
            "window_start",
            "label",
            "transient_only",
            "state_mode",
            "n_variables_present",
        ]
    ].to_csv(out_dir / "windows.csv", index=False, lineterminator="\n")

    summary_rows = []
    combos = sorted({(r["design"], r["split"], r["tool"]) for r in per_fold})
    for design, split_name, tool in combos:
        for variant in OWN_HISTORY_VARIANTS:
            sub = fold_df[
                (fold_df["tool"] == tool)
                & (fold_df["design"] == design)
                & (fold_df["split"] == split_name)
                & (fold_df["variant"] == variant)
            ]
            if not len(sub):
                continue
            aucs = sub["roc_auc"].dropna().to_numpy(dtype=float)
            summary_rows.append(
                {
                    "tool": tool,
                    "design": design,
                    "split": split_name,
                    "variant": variant,
                    "n_folds_scored": len(aucs),
                    "auc_mean": round(float(aucs.mean()), 4) if len(aucs) else None,
                    "auc_min": round(float(aucs.min()), 4) if len(aucs) else None,
                    "auc_max": round(float(aucs.max()), 4) if len(aucs) else None,
                    "pr_auc_mean": (
                        round(float(sub["pr_auc"].dropna().mean()), 4)
                        if sub["pr_auc"].notna().any()
                        else None
                    ),
                    "precision_at_recall_50_mean": (
                        round(float(sub["precision_at_recall_50"].dropna().mean()), 4)
                        if sub["precision_at_recall_50"].notna().any()
                        else None
                    ),
                    "refused_frac_mean": round(float(sub["refused_frac"].mean()), 4),
                    "n_windows": int(sub["n_windows"].sum()),
                    "n_scored": int(sub["n_scored"].sum()),
                    "n_refused": int(sub["n_refused"].sum()),
                    "n_refused_by_design": int(sub["n_refused_by_design"].sum()),
                }
            )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "summary.csv", index=False, lineterminator="\n")

    profile_csv = ROOT / "examples/studies/3w_profile/results/per_tag.csv"
    r1_pairs: set[tuple[str, str]] = set()
    reachable: set[tuple[str, str]] = set()
    if profile_csv.exists():
        r1 = pd.read_csv(profile_csv)
        r1 = r1[(r1["role"] != "MODE") & (r1["whole_life_distinct"] == 1)]
        r1_pairs = set(zip(r1["instance"], r1["variable"], strict=True))
        reachable = {p for p in r1_pairs if p[1] in variables}

    population_report = {}
    for split_name in ("well", "instance"):
        parts = [d for d in population_detail if d["split"] == split_name]
        flagged_tags = [t for d in parts for t in d["flagged_tags"]]
        flagged_pairs: dict[str, int] = {}
        for d in parts:
            for pair, n in d["flagged_pairs"].items():
                flagged_pairs[pair] = flagged_pairs.get(pair, 0) + n
        n_frozen = sum(d["n_frozen_rows"] for d in parts)
        n_flag = sum(d["n_frozen_flagged"] for d in parts)
        n_ref = sum(d["n_frozen_refused"] for d in parts)
        indicted = {
            (t["instance"], t["variable"])
            for t in flagged_tags
            if (t["instance"], t["variable"]) in r1_pairs
        }
        population_report[split_name] = {
            "frozen_windows": {
                "n_frozen_feature_rows": n_frozen,
                "n_frozen_flagged": n_flag,
                "n_frozen_refused": n_ref,
                "flagged_frac": round(n_flag / max(n_frozen, 1), 6),
                "refused_frac": round(n_ref / max(n_frozen, 1), 6),
            },
            "flagged_pairs": dict(sorted(flagged_pairs.items())),
            "whole_life_frozen": {
                "n_whole_life_frozen_measurement_tags": len(r1_pairs),
                "n_reachable_in_common_variable_set": len(reachable),
                "n_indicted": len(indicted),
                "per_variable_indicted": {
                    v: sum(1 for p in indicted if p[1] == v) for v in sorted(variables)
                },
                "per_variable_reachable": {
                    v: sum(1 for p in reachable if p[1] == v) for v in sorted(variables)
                },
            },
            "example_evidence": [t["evidence"] for t in flagged_tags[:5]],
        }
    frozen = population_report["well"]["frozen_windows"]

    scored_windows = windows[windows["window_key"].isin(layout.scored_keys)]
    positive_baseline_instances = sorted(i for i, v in positive_baseline.items() if v)
    own_history_report = {
        "baseline_windows": args.baseline_windows,
        "n_instances_with_windows": int(windows["instance"].nunique()),
        "n_instances_scored": len(layout.scored),
        "n_instances_too_short": len(layout.short),
        "n_baseline_windows": len(layout.baseline_keys),
        "n_scored_windows": len(layout.scored_keys),
        "scored_positive_rate": (
            round(float(scored_windows["label"].mean()), 6) if len(scored_windows) else None
        ),
        "scored_per_folder": {
            str(k): {"windows": int(v["size"]), "positive": int(v["sum"])}
            for k, v in scored_windows.groupby("folder_label")["label"]
            .agg(["size", "sum"])
            .to_dict("index")
            .items()
        },
        "n_instances_with_fully_positive_baseline": len(positive_baseline_instances),
        "n_scored_windows_in_those_instances": int(
            scored_windows["instance"].isin(positive_baseline_instances).sum()
        ),
        "n_pairs_with_a_usable_baseline": len(baselines),
        "n_pairs_refused_degenerate_baseline": len(degenerate),
        "grid_standardisation": grid_std_detail,
        "feature_standardisation": feature_std_detail,
    }

    provenance = {
        "dataset_manifest_sha256": sha256_file(Path(args.source) / "MANIFEST.sha256.json"),
        "windows_manifest": manifest,
        "code": {"git_head": git_head(), "tsdive_version": tsdive.__version__},
        "splits": {
            "well": {
                "kind": "5-fold group holdout by well",
                "group_col": "well",
                "stratum_col": "folder_label",
                "n_folds": args.n_folds,
                "seed": args.seed,
                "folds": [list(split.groups(f)) for f in range(args.n_folds)],
                "fold_sizes": list(split.fold_sizes),
                "strata_counts": [dict(c) for c in split.strata_counts],
            },
            "instance": {
                "kind": "5-fold group holdout by instance (population screen only)",
                "group_col": "instance",
                "stratum_col": "folder_label",
                "n_folds": args.n_folds,
                "seed": args.seed,
                "fold_sizes": list(instance_split.fold_sizes),
                "strata_counts": [dict(c) for c in instance_split.strata_counts],
            },
            "own-history": {
                "kind": (
                    f"own history, first {args.baseline_windows} windows per instance, "
                    "label-blind; all instances, no group holdout"
                ),
                "why_no_holdout": (
                    "nothing is fitted across instances: each tag's limits come from "
                    "that tag's own earlier windows, so there is no cross-instance "
                    "parameter for a group holdout to protect"
                ),
                "baseline_windows": args.baseline_windows,
            },
        },
        "own_history": own_history_report,
        "n_windows": len(windows),
        "n_positive": int(windows["label"].sum()),
        "positive_rate": round(float(windows["label"].mean()), 6),
        "n_transient_only": int(windows["transient_only"].sum()),
        "common_variables": variables,
        "beyond_float32": {"features": feature_beyond, "subgroup_grid": grid_beyond},
        "min_baseline_windows": args.min_baseline_windows,
        "wall_seconds": round(elapsed, 1),
    }
    (out_dir / "run.json").write_text(
        json.dumps(
            {
                "provenance": provenance,
                "population": population_report,
                "mspc": mspc_detail,
                "mspc_standardised": std_mspc_detail,
                "iforest": iforest_detail,
                "iforest_standardised": std_iforest_detail,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(summary_df[summary_df["variant"] == "all"].to_string(index=False))
    print(f"population, well holdout: {frozen}")
    print(f"population, instance holdout: {population_report['instance']['frozen_windows']}")
    print(f"own-history: {own_history_report}")
    print(f"{len(windows)} windows, {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

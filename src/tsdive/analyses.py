"""The analyses after ``profile``, as library calls returning result objects.

Every function reads its windows from parquet archives, runs the
computation and returns a result holding objects, not text. On each
result ``render()`` returns the plain text ``tsdive <command>`` prints
and ``to_dict()`` the payload ``tsdive <command> --json`` prints; the CLI
calls those two methods, so the library and the CLI cannot disagree.

Windows are ``START/END`` in ISO 8601 UTC, as on the command line. A
malformed window raises ``ValueError``; a read the archive cannot support
raises the typed :class:`tsdive.errors.TSDiveError` the CLI reports.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import cast

import pandas as pd

from tsdive.analyses_render import (
    CONTRIBUTORS_KEPT,
    compare_json,
    compare_lines,
    mspc_json,
    mspc_lines,
    screen_json,
    screen_lines,
    segment_json,
    segment_lines,
    spc_json,
    spc_lines,
)
from tsdive.api import parse_window
from tsdive.baselines.provisional import (
    ProvisionalBaseline,
    ScreenResult,
    mad_baseline,
    moving_range_baseline,
)
from tsdive.baselines.provisional import screen as screen_frame
from tsdive.baselines.regime import RegimeBaseline, regime_baselines, screen_regime
from tsdive.changepoints import segment_window
from tsdive.changepoints.pelt import Segmentation
from tsdive.compare import CompareResult
from tsdive.compare import compare as compare_periods
from tsdive.errors import InsufficientQuality, SchemaError
from tsdive.mspc.pca import (
    AlignedMatrix,
    MspcDetection,
    PcaModel,
    align_windows,
    common_rate,
    detect,
    fit_pca,
)
from tsdive.spc.charts import ControlLimits, RuleHit, apply_rules, individuals_limits
from tsdive.store.clipping import assert_usable_baseline
from tsdive.store.identity import TagIdentity
from tsdive.store.sampling_contract import (
    AggregateType,
    CalculationBasis,
    RetrievalMode,
    SamplingContract,
)
from tsdive.store.tagstore import (
    SingleFileStore,
    Window,
    archive_extent,
    meta_from_parquet,
)

SCREEN_METHODS = ("mad", "moving-range")


def _contract(basis: str | CalculationBasis, stepped: bool) -> SamplingContract:
    return SamplingContract(
        calculation_basis=CalculationBasis(basis),
        retrieval_mode=RetrievalMode.RECORDED,
        aggregate_type=AggregateType.NONE,
        stepped=stepped,
    )


def _baseline_and_monitor(
    path: str | Path,
    baseline: str,
    window: str,
    *,
    basis: str | CalculationBasis,
    stepped: bool,
) -> tuple[Window, Window]:
    """Read a baseline window and a monitored window from one archive.

    Two refusals live here rather than in the baseline, SPC and MSPC
    computations, which are handed frames and never see where those
    frames came from: a baseline overlapping the monitored window grades
    those samples against themselves, and a censored baseline carries no
    information about normal operation (:func:`assert_usable_baseline`).
    """
    b_start, b_end = parse_window(baseline)
    m_start, m_end = parse_window(window)
    # Touching bounds are not an overlap, but reads are inclusive at both
    # ends, so an adjacent pair would share the sample on the boundary.
    # It is given to the monitored window and dropped from the baseline
    # frame below.
    if b_start < m_end and m_start < b_end:
        raise ValueError("baseline and window overlap")
    store = SingleFileStore(Path(path))
    meta = meta_from_parquet(store.path)
    contract = _contract(basis, stepped)
    reads = [
        store.read_window(meta.identity, s.to_pydatetime(), e.to_pydatetime(), contract)
        for s, e in ((b_start, b_end), (m_start, m_end))
    ]
    base, monitor = reads
    assert_usable_baseline(meta, base.physics.clipping)
    if b_end == m_start:
        # The frame loses the boundary sample; the physics block keeps
        # it, so the censoring verdict the baseline was admitted on is
        # still the one taken over the whole read - the conservative side.
        base = replace(base, frame=base.frame[base.frame["timestamp"] < m_start])
    return base, monitor


def _aligned_modes(
    frame: pd.DataFrame, modes: Window, label: str
) -> tuple[pd.DataFrame, pd.Series, int]:
    """Inner-join a frame to its MODE rows on the exact timestamp.

    Exact only: a regime label carried over from a neighbouring second is
    a guess about what the plant was doing. Rows with no mode sample of
    their own are dropped and counted, and a window that loses more than
    half of them is refused rather than screened against a fragment.
    """
    usable = modes.frame.loc[modes.frame["valid"], ["timestamp", "value"]]
    if bool(usable["timestamp"].duplicated().any()):
        raise SchemaError(
            f"{modes.identity}: duplicate timestamps in the mode archive; joining "
            f"them would multiply {label} rows"
        )
    joined = frame.merge(
        usable.rename(columns={"value": "mode"}), on="timestamp", how="inner"
    ).reset_index(drop=True)
    dropped = len(frame) - len(joined)
    if dropped * 2 > len(frame):
        raise InsufficientQuality(
            f"{dropped} of {len(frame)} {label} rows carry no mode sample at their "
            "own timestamp; more than half the window would be screened blind"
        )
    return joined, cast(pd.Series, joined["mode"].astype(str)), dropped


@dataclass(frozen=True)
class Alignment:
    """Rows that carried a mode sample of their own, over rows read."""

    kept: int
    total: int


# --- segment ----------------------------------------------------------------


@dataclass(frozen=True)
class SegmentAnalysis:
    """One window cut into piecewise-constant segments.

    ``frame`` holds one row per segment with the columns ``to_dict()``
    lists under ``segments``.
    """

    window: Window
    found: Segmentation
    penalty_is_default: bool

    @property
    def identity(self) -> TagIdentity:
        return self.window.identity

    @property
    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(segment_json(self)["segments"])

    def render(self) -> str:
        return "\n".join(segment_lines(self))

    def to_dict(self) -> dict[str, object]:
        return segment_json(self)


def segment(
    archive: str | Path,
    window: str | None = None,
    *,
    penalty: float | None = None,
    min_size: int = 10,
    basis: str | CalculationBasis = CalculationBasis.TIME_WEIGHTED,
    stepped: bool = False,
) -> SegmentAnalysis:
    """Segment one window of a single-tag archive into regimes.

    Args:
        archive: parquet archive carrying ``tsdive.meta``.
        window: ``START/END`` in ISO 8601 UTC. Omitted, the archive's own
            extent.
        penalty: cost one more breakpoint has to buy. Omitted, the
            default multiplier times ``log(n)`` on the scaled series.
        min_size: samples a segment must hold, at least.
        basis: calculation basis declared on the read.
        stepped: stepped interpolation between samples.

    Raises:
        ValueError: malformed or naive window.
        TSDiveError: any typed refusal from the read path.
    """
    store = SingleFileStore(Path(archive))
    meta = meta_from_parquet(store.path)
    start, end = parse_window(window) if window else archive_extent(store.path)
    read = store.read_window(
        meta.identity, start.to_pydatetime(), end.to_pydatetime(), _contract(basis, stepped)
    )
    return SegmentAnalysis(
        window=read,
        found=segment_window(read, penalty=penalty, min_size=min_size),
        penalty_is_default=penalty is None,
    )


# --- screen -----------------------------------------------------------------


@dataclass(frozen=True)
class ScreenAnalysis:
    """One screen, whether it used one baseline or one per regime.

    ``provisional`` and ``regimes`` are the two ways a baseline is built;
    exactly one is populated, and both the text and the JSON read the same
    object rather than screening the window twice. ``frame`` holds the
    monitored rows the screen flagged.
    """

    baseline: Window
    monitor: Window
    result: ScreenResult
    k: float
    mode_path: str | None = None
    provisional: ProvisionalBaseline | None = None
    regimes: dict[str, RegimeBaseline] | None = None
    alignment: tuple[Alignment, Alignment] | None = None

    @property
    def identity(self) -> TagIdentity:
        return self.baseline.identity

    @property
    def frame(self) -> pd.DataFrame:
        rows = self.monitor.frame
        flagged = rows["timestamp"].isin(list(self.result.flagged_timestamps))
        return rows[flagged].reset_index(drop=True)

    def render(self) -> str:
        return "\n".join(screen_lines(self))

    def to_dict(self) -> dict[str, object]:
        return screen_json(self)


def screen(
    archive: str | Path,
    baseline: str,
    window: str,
    *,
    method: str = "mad",
    k: float = 3.0,
    mode: str | Path | None = None,
    basis: str | CalculationBasis = CalculationBasis.TIME_WEIGHTED,
    stepped: bool = False,
) -> ScreenAnalysis:
    """Screen one window of a single-tag archive against a baseline.

    Args:
        archive: parquet archive carrying ``tsdive.meta``.
        baseline: ``START/END`` in ISO 8601 UTC for the history.
        window: ``START/END`` in ISO 8601 UTC to screen.
        method: ``"mad"`` or ``"moving-range"``; ignored with ``mode``.
        k: flag beyond ``k`` times the scale from the center.
        mode: MODE archive; one baseline per regime instead of one for
            the window.
        basis: calculation basis declared on the read.
        stepped: stepped interpolation between samples.

    Raises:
        ValueError: malformed window, overlapping windows, unknown method.
        TSDiveError: a censored baseline, or any typed refusal from the
            read path.
    """
    if method not in SCREEN_METHODS:
        raise ValueError(f"method must be one of {SCREEN_METHODS}, not {method!r}")
    base_w, monitor_w = _baseline_and_monitor(
        archive, baseline, window, basis=basis, stepped=stepped
    )
    if mode is None:
        base = mad_baseline(base_w.frame) if method == "mad" else moving_range_baseline(
            base_w.frame
        )
        return ScreenAnalysis(
            baseline=base_w,
            monitor=monitor_w,
            result=screen_frame(base, monitor_w.frame, k=k),
            k=k,
            provisional=base,
        )
    mode_base, mode_monitor = _baseline_and_monitor(
        mode, baseline, window, basis=basis, stepped=stepped
    )
    b_frame, b_modes, b_dropped = _aligned_modes(base_w.frame, mode_base, "baseline")
    m_frame, m_modes, m_dropped = _aligned_modes(monitor_w.frame, mode_monitor, "window")
    regimes = regime_baselines(b_frame, b_modes)
    return ScreenAnalysis(
        baseline=base_w,
        monitor=monitor_w,
        result=screen_regime(regimes, m_frame, m_modes, k=k),
        k=k,
        mode_path=os.fspath(mode),
        regimes=regimes,
        alignment=(
            Alignment(len(b_frame), len(b_frame) + b_dropped),
            Alignment(len(m_frame), len(m_frame) + m_dropped),
        ),
    )


# --- spc --------------------------------------------------------------------


@dataclass(frozen=True)
class SpcAnalysis:
    """An individuals chart: limits from the baseline, rule hits in the window.

    ``frame`` holds one row per rule hit: ``timestamp``, ``rule``,
    ``detail``.
    """

    baseline: Window
    monitor: Window
    limits: ControlLimits
    sigma: float
    n_monitored: int
    hits: list[RuleHit]

    @property
    def identity(self) -> TagIdentity:
        return self.baseline.identity

    @property
    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [(h.timestamp, h.rule, h.detail) for h in self.hits],
            columns=["timestamp", "rule", "detail"],
        )

    def render(self) -> str:
        return "\n".join(spc_lines(self))

    def to_dict(self) -> dict[str, object]:
        return spc_json(self)


def spc(
    archive: str | Path,
    baseline: str,
    window: str,
    *,
    basis: str | CalculationBasis = CalculationBasis.TIME_WEIGHTED,
    stepped: bool = False,
) -> SpcAnalysis:
    """Chart one window of a single-tag archive against baseline limits.

    Args:
        archive: parquet archive carrying ``tsdive.meta``.
        baseline: ``START/END`` in ISO 8601 UTC the limits are computed
            from.
        window: ``START/END`` in ISO 8601 UTC to chart.
        basis: calculation basis declared on the read.
        stepped: stepped interpolation between samples.

    Raises:
        ValueError: malformed window, overlapping windows.
        TSDiveError: a censored baseline, or any typed refusal from the
            read path.
    """
    base_w, monitor_w = _baseline_and_monitor(
        archive, baseline, window, basis=basis, stepped=stepped
    )
    base = mad_baseline(base_w.frame)
    limits = individuals_limits(base.center, base.scale)
    good = monitor_w.frame[monitor_w.frame["valid"]]
    return SpcAnalysis(
        baseline=base_w,
        monitor=monitor_w,
        limits=limits,
        sigma=base.scale,
        n_monitored=len(good),
        hits=apply_rules(
            good["timestamp"], cast(pd.Series, good["value"].astype(float)), limits
        ),
    )


# --- mspc -------------------------------------------------------------------


@dataclass(frozen=True)
class MspcAnalysis:
    """PCA T2 and SPE over aligned tags: the model, and the window it graded.

    ``baseline`` and ``monitor`` are the first archive's reads; ``tags``
    names every archive in order. ``frame`` holds one row per aligned
    timestamp of the window: ``timestamp``, ``t2``, ``spe``,
    ``t2_breach``, ``spe_breach``.
    """

    baseline: Window
    monitor: Window
    tags: list[str]
    rate_s: int
    rate_source: str
    quantile: float
    train: AlignedMatrix
    test: AlignedMatrix
    model: PcaModel
    found: MspcDetection

    @property
    def ranked(self) -> bool:
        """True when the model holds more tags than a top-N list would name."""
        return len(self.model.columns) > CONTRIBUTORS_KEPT

    @property
    def frame(self) -> pd.DataFrame:
        stamps = self.found.timestamps
        return pd.DataFrame(
            {
                "timestamp": stamps,
                "t2": self.found.t2,
                "spe": self.found.spe,
                "t2_breach": stamps.isin(self.found.t2_breaches),
                "spe_breach": stamps.isin(self.found.spe_breaches),
            }
        )

    def render(self) -> str:
        return "\n".join(mspc_lines(self))

    def to_dict(self) -> dict[str, object]:
        return mspc_json(self)


def mspc(
    archives: Sequence[str | Path],
    baseline: str,
    window: str,
    *,
    rate_s: int | None = None,
    variance: float = 0.95,
    quantile: float = 0.99,
    min_coverage: float = 0.95,
) -> MspcAnalysis:
    """Detect multivariate departures over two or more single-tag archives.

    Args:
        archives: parquet archives carrying ``tsdive.meta``.
        baseline: ``START/END`` in ISO 8601 UTC the model is fitted on.
        window: ``START/END`` in ISO 8601 UTC to monitor.
        rate_s: grid rate in seconds. Omitted, the rate every archive
            declares.
        variance: cumulative variance kept, in (0, 1].
        quantile: empirical quantile for the limits, in (0, 1].
        min_coverage: grid cells that must be filled before alignment is
            accepted, in (0, 1].

    Raises:
        ValueError: malformed window, overlapping windows.
        TSDiveError: a censored baseline, archives that do not align, or
            any typed refusal from the read path.
    """
    pairs = [
        _baseline_and_monitor(
            path, baseline, window, basis=CalculationBasis.TIME_WEIGHTED, stepped=False
        )
        for path in archives
    ]
    bases = [b for b, _ in pairs]
    monitors = [m for _, m in pairs]
    rate = rate_s if rate_s is not None else common_rate(bases)
    train = align_windows(bases, rate_s=rate, min_coverage=min_coverage)
    model = fit_pca(train, variance_threshold=variance, limit_quantile=quantile)
    test = align_windows(monitors, rate_s=rate, min_coverage=min_coverage)
    return MspcAnalysis(
        baseline=bases[0],
        monitor=monitors[0],
        tags=[str(b.identity) for b in bases],
        rate_s=rate,
        rate_source="declared" if rate_s is None else "--rate-s",
        quantile=quantile,
        train=train,
        test=test,
        model=model,
        found=detect(model, test),
    )


# --- compare ----------------------------------------------------------------


@dataclass(frozen=True)
class CompareAnalysis(CompareResult):
    """What changed between two periods of one unit, with its report depth.

    ``top`` is how many rows each rendered table prints before it counts
    the rest; ``to_dict()`` carries every row. ``frame`` holds one row
    per tag with the columns ``to_dict()`` lists under ``tags``.
    """

    top: int = 10

    @property
    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(compare_json(self)["tags"])

    def render(self) -> str:
        return "\n".join(compare_lines(self))

    def to_dict(self) -> dict[str, object]:
        return compare_json(self)


def compare(
    archives: Sequence[str | Path],
    before: str,
    after: str,
    *,
    top: int = 10,
    rate_s: int | None = None,
) -> CompareAnalysis:
    """Compare two periods over every single-tag archive of one unit.

    Args:
        archives: parquet archives carrying ``tsdive.meta``.
        before: ``START/END`` in ISO 8601 UTC for the earlier period.
        after: ``START/END`` in ISO 8601 UTC for the later period.
        top: rows each rendered table prints before it counts the rest.
        rate_s: grid rate in seconds. Omitted, the rate every archive
            declares.

    Raises:
        ValueError: malformed periods, overlapping or swapped periods.
        TSDiveError: any typed refusal that stops the whole comparison;
            a refusal on one archive stays a row in ``tags``.
    """
    result = compare_periods(
        [os.fspath(p) for p in archives], before, after, rate_s=rate_s
    )
    return CompareAnalysis(
        **{f.name: getattr(result, f.name) for f in fields(result)}, top=top
    )

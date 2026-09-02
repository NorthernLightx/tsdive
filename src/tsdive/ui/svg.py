"""One inline SVG figure per window for the static HTML report.

Pure string assembly: no JavaScript, no external CSS or fonts, no random
ids and no clock, so the same window draws the same bytes. Every text the
figure shows passes through html.escape.
"""

from __future__ import annotations

import html
import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

from tsdive.store.quality import Severity
from tsdive.store.tagstore import Window

# Four chromatic colours, categorical slots 1, 2, 3 and 7 of the report
# palette: the set passes the colour-vision and normal-vision separation
# checks as a whole. Gaps, limits and the engineering range stay grey.
_STYLE = (
    "svg.tsd text{font:10px system-ui,sans-serif;fill:#1a1a1a}"
    "svg.tsd .muted{fill:#666}"
    "svg.tsd .lbl{paint-order:stroke;stroke:#fcfcfb;stroke-width:3px;stroke-linejoin:round}"
    "svg.tsd .plot{fill:#fcfcfb;stroke:#ccc;stroke-width:1}"
    "svg.tsd .line{fill:none;stroke:#2a78d6;stroke-width:1.5;stroke-linejoin:round}"
    "svg.tsd .below{fill:#eb6834;stroke:none}"
    "svg.tsd .gap{fill:#9a9a9a;fill-opacity:.18;stroke:none}"
    "svg.tsd .eng{stroke:#9a9a9a;stroke-width:1;stroke-dasharray:4 3}"
    "svg.tsd .center{stroke:#666;stroke-width:1}"
    "svg.tsd .limit{stroke:#666;stroke-width:1;stroke-dasharray:6 3}"
    "svg.tsd .leader{stroke:#666;stroke-width:1;fill:none}"
    "svg.tsd .flag{fill:#1baf7a;stroke:#fcfcfb;stroke-width:1}"
    "svg.tsd .hit{fill:#4a3aa7;stroke:#fcfcfb;stroke-width:1}"
)

_MARGIN_X = 6
_MARGIN_TOP = 16
_MARGIN_BOTTOM = 14
# A column right of the plot for the limit labels, one per line, so
# tight limits never write over each other.
_LABEL_COLUMN = 90
_LINE_HEIGHT = 11.0


def _esc(text: object) -> str:
    return html.escape(str(text))


def _num(value: float) -> str:
    return f"{value:.5g}"


def _utc(stamp: object) -> pd.Timestamp:
    t = pd.Timestamp(stamp)  # type: ignore[arg-type]
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _clock(stamp: pd.Timestamp) -> str:
    return _utc(stamp).strftime("%Y-%m-%d %H:%MZ")


def _thin(x: np.ndarray, v: np.ndarray, x0: float) -> np.ndarray:
    """Indices keeping one minimum and one maximum per pixel column.

    The rows arrive in time order, so the sorted union of both index sets
    is a time-ordered subsequence of at most two points per column.
    """
    column = np.floor(x - x0).astype(int)
    grouped = pd.DataFrame({"c": column, "v": v}).groupby("c")["v"]
    keep = np.union1d(grouped.idxmin().to_numpy(), grouped.idxmax().to_numpy())
    return np.sort(keep)


def _y_range(
    values: np.ndarray, limits: tuple[float, float, float] | None
) -> tuple[float, float]:
    """The data range, widened to hold the limits, then padded 5 %."""
    finite = values[np.isfinite(values)]
    lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
    if limits is not None:
        lo, hi = min(lo, *limits), max(hi, *limits)
    pad = (hi - lo) * 0.05 or abs(hi) * 0.05 or 1.0
    return lo - pad, hi + pad


def _spread(wanted: Sequence[float], top: float, bottom: float) -> list[float]:
    """Text baselines in the given order, one line apart, inside the plot.

    Each label starts at its own line's height and moves down until it
    clears the one above; the column then moves up as a block when the
    last one would fall below the plot.
    """
    placed: list[float] = []
    floor = top
    for y in wanted:
        y = max(y, floor)
        placed.append(y)
        floor = y + _LINE_HEIGHT
    excess = max(0.0, placed[-1] - bottom) if placed else 0.0
    return [y - excess for y in placed]


def window_figure(
    window: Window,
    *,
    limits: tuple[float, float, float] | None = None,
    flagged: Sequence[pd.Timestamp] = (),
    hits: Sequence[pd.Timestamp] = (),
    width: int = 720,
    height: int = 180,
) -> str:
    """Draw one window as a self-contained ``<svg>`` element.

    GOOD rows form the value line; rows below GOOD are unjoined markers.
    The value axis spans the plotted values, widened to hold the limits
    and padded 5 %. Each gap in ``window.physics.coverage`` is a shaded
    span; each end of the engineering range is a dashed line when it
    falls inside the axis. ``limits`` is ``(center, lcl, ucl)``, drawn as
    one solid and two dashed lines named in a column right of the plot.
    ``flagged`` and ``hits`` mark those timestamps on the value line with
    ``class="flag"`` and ``class="hit"``.

    A window with more than ``2 * width`` rows is thinned to one minimum
    and one maximum per pixel column; flagged and hit markers keep their
    own coordinates whatever the thinning drops.
    """
    x0, x1 = float(_MARGIN_X), float(width - _LABEL_COLUMN)
    y0, y1 = float(_MARGIN_TOP), float(height - _MARGIN_BOTTOM)
    frame = window.frame
    stamps = pd.DatetimeIndex(pd.to_datetime(frame["timestamp"]))
    stamps = stamps.tz_localize("UTC") if stamps.tz is None else stamps.tz_convert("UTC")
    order = np.argsort(stamps.asi8, kind="stable")
    stamps = stamps[order]
    values = pd.to_numeric(frame["value"].iloc[order], errors="coerce").to_numpy(dtype=float)
    good = (frame["severity"].iloc[order] == Severity.GOOD).to_numpy() & np.isfinite(values)
    below = ~good & np.isfinite(values)

    start, end = _utc(window.start), _utc(window.end)
    span_ns = max(end.value - start.value, 1)

    def sx(ns: np.ndarray | int) -> np.ndarray | float:
        return x0 + (ns - start.value) / span_ns * (x1 - x0)

    lo, hi = _y_range(values, limits)

    def sy(v: np.ndarray | float) -> np.ndarray | float:
        return y1 - (v - lo) / (hi - lo) * (y1 - y0)

    xs = np.asarray(sx(stamps.asi8), dtype=float)
    ys = np.asarray(sy(values), dtype=float)
    thin = len(frame) > 2 * width

    parts: list[str] = [
        f'<svg class="tsd" xmlns="http://www.w3.org/2000/svg" role="img" '
        f'viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'style="max-width:100%;height:auto">',
        f"<title>{_esc(window.identity)}  {_esc(window.meta.name)}</title>",
        f"<style>{_STYLE}</style>",
        f'<rect class="plot" x="{x0:.1f}" y="{y0:.1f}" '
        f'width="{x1 - x0:.1f}" height="{y1 - y0:.1f}"/>',
    ]

    for gap in window.physics.coverage.gaps:
        gx0 = float(sx(_utc(gap.start).value))
        gx1 = float(sx(_utc(gap.end).value))
        gx0, gx1 = max(gx0, x0), min(gx1, x1)
        if gx1 < gx0:
            continue
        parts.append(
            f'<rect class="gap" x="{gx0:.1f}" y="{y0:.1f}" '
            f'width="{max(gx1 - gx0, 1.0):.1f}" height="{y1 - y0:.1f}">'
            f"<title>{_esc(gap.classification.cls)} gap "
            f"{_clock(gap.start)} to {_clock(gap.end)}</title></rect>"
        )

    eng = window.meta.eng_range
    if eng is not None:
        for edge, label in ((eng.zero, "zero"), (eng.high, "high")):
            if lo <= edge <= hi:
                ey = float(sy(edge))
                parts.append(
                    f'<line class="eng" x1="{x0:.1f}" y1="{ey:.1f}" x2="{x1:.1f}" y2="{ey:.1f}">'
                    f"<title>engineering range {label} {_num(edge)}</title></line>"
                )

    if limits is not None:
        center, lcl, ucl = limits
        rows = [(ucl, "limit", "ucl"), (center, "center", "center"), (lcl, "limit", "lcl")]
        lys = [float(sy(value)) for value, _, _ in rows]
        baselines = _spread([ly + 3.5 for ly in lys], y0 + 8.0, y1 + 2.0)
        for (value, cls, name), ly, ty in zip(rows, lys, baselines, strict=True):
            parts.append(
                f'<line class="{cls}" x1="{x0:.1f}" y1="{ly:.1f}" x2="{x1:.1f}" y2="{ly:.1f}"/>'
                f'<path class="leader" d="M{x1:.1f},{ly:.1f} L{x1 + 6:.1f},{ty - 3.5:.1f}"/>'
                f'<text class="muted" x="{x1 + 8:.1f}" y="{ty:.1f}">{name} {_num(value)}</text>'
            )

    idx_good = np.flatnonzero(good)
    if thin and idx_good.size:
        idx_good = idx_good[_thin(xs[idx_good], values[idx_good], x0)]
    if idx_good.size:
        points = " ".join(f"{xs[i]:.1f},{ys[i]:.1f}" for i in idx_good)
        parts.append(f'<polyline class="line" points="{points}"/>')

    idx_below = np.flatnonzero(below)
    if thin and idx_below.size:
        idx_below = idx_below[_thin(xs[idx_below], values[idx_below], x0)]
    for i in idx_below:
        parts.append(f'<circle class="below" cx="{xs[i]:.1f}" cy="{ys[i]:.1f}" r="1.5"/>')

    unique = ~stamps.duplicated(keep="first")
    lookup = pd.Series(values[unique], index=stamps[unique])
    n_flag = n_hit = 0
    for when in flagged:
        v = lookup.get(_utc(when))
        if isinstance(v, float) and math.isfinite(v):
            n_flag += 1
            parts.append(
                f'<circle class="flag" cx="{float(sx(_utc(when).value)):.1f}" '
                f'cy="{float(sy(v)):.1f}" r="3.5">'
                f"<title>flagged {_clock(when)} value {_num(v)}</title></circle>"
            )
    for when in hits:
        v = lookup.get(_utc(when))
        if isinstance(v, float) and math.isfinite(v):
            n_hit += 1
            cx, cy = float(sx(_utc(when).value)), float(sy(v))
            parts.append(
                f'<path class="hit" d="M{cx:.1f},{cy - 4.5:.1f} L{cx + 4.5:.1f},{cy:.1f} '
                f'L{cx:.1f},{cy + 4.5:.1f} L{cx - 4.5:.1f},{cy:.1f} Z">'
                f"<title>rule hit {_clock(when)} value {_num(v)}</title></path>"
            )

    unit = f" {_esc(window.meta.unit_raw)}" if window.meta.unit_raw else ""
    parts.append(
        f'<text class="lbl muted" x="{x0 + 3:.1f}" y="{y0 + 10:.1f}">{_num(hi)}{unit}</text>'
        f'<text class="lbl muted" x="{x0 + 3:.1f}" y="{y1 - 3:.1f}">{_num(lo)}{unit}</text>'
    )
    mid = start + (end - start) / 2
    ty = float(height - 3)
    parts.append(
        f'<text class="muted" x="{x0:.1f}" y="{ty:.1f}">{_clock(start)}</text>'
        f'<text class="muted" x="{(x0 + x1) / 2:.1f}" y="{ty:.1f}" text-anchor="middle">'
        f"{_clock(mid)}</text>"
        f'<text class="muted" x="{x1:.1f}" y="{ty:.1f}" text-anchor="end">{_clock(end)}</text>'
    )

    legend: list[tuple[str, str]] = [("line", "value")]
    if idx_below.size:
        legend.append(("below", "below GOOD"))
    if n_flag:
        legend.append(("flag", "flagged"))
    if n_hit:
        legend.append(("hit", "rule hit"))
    if window.physics.coverage.gaps:
        legend.append(("gap", "gap"))
    lx = x0
    for cls, label in legend:
        if cls == "line":
            swatch = f'<line class="line key" x1="{lx:.1f}" y1="7.0" x2="{lx + 10:.1f}" y2="7.0"/>'
        elif cls == "gap":
            swatch = f'<rect class="gap key" x="{lx:.1f}" y="2.0" width="10.0" height="10.0"/>'
        elif cls == "hit":
            swatch = (
                f'<path class="hit key" d="M{lx + 5:.1f},2.5 L{lx + 9.5:.1f},7.0 '
                f'L{lx + 5:.1f},11.5 L{lx + 0.5:.1f},7.0 Z"/>'
            )
        else:
            r = "3.5" if cls == "flag" else "2.0"
            swatch = f'<circle class="{cls} key" cx="{lx + 5:.1f}" cy="7.0" r="{r}"/>'
        parts.append(f'{swatch}<text x="{lx + 14:.1f}" y="10.5">{_esc(label)}</text>')
        lx += 14 + len(label) * 5.6 + 12
    parts.append(
        f'<text class="muted" x="{width - _MARGIN_X:.1f}" y="10.5" text-anchor="end">'
        f"{_esc(window.identity)}  {_esc(window.meta.name)}</text>"
    )

    parts.append("</svg>")
    return "".join(parts)

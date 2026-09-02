"""The first UI, a self-contained static HTML evidence viewer.

No server, no JavaScript dependencies, no network: one HTML file with
inline CSS rendering profile reports, benchmark rows and refusals.
Everything user-visible passes through html.escape.
"""

from __future__ import annotations

import html
from collections.abc import Sequence
from pathlib import Path

from tsdive import __version__

_CSS = """body{font-family:Georgia,serif;margin:2rem auto;max-width:60rem;
color:#1a1a1a;background:#faf9f6}h1{border-bottom:3px solid #8b0000}
pre{background:#f0efe9;padding:.8rem;overflow-x:auto;border-left:4px solid #999}
table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;
padding:.3rem .5rem;text-align:left}th{background:#eae7de}
.refused{color:#8b0000;font-weight:bold}.ok{color:#1d5c1d}footer{margin-top:2rem;
font-size:.85em;color:#666}"""


def _esc(text: object) -> str:
    return html.escape(str(text))


def render_static_report(
    *,
    title: str,
    profiles: list[str],
    benchmarks_markdown_rows: list[tuple[str, ...]],
    refusal_log: list[str],
    benchmarks_header: tuple[str, ...] | None = None,
    findings: list[tuple[str, str, str]] | None = None,
    figures: Sequence[str] | None = None,
) -> str:
    """Render one self-contained HTML document.

    Every entry in ``benchmarks_markdown_rows`` is data. Column headings
    come from ``benchmarks_header``; without it the table has no ``<th>``
    row, because a caller's first row is not a header by assumption.

    ``findings`` holds ``(step, tags, text)`` triples from a plan run.
    ``figures`` holds rendered ``<svg>`` elements, one per window, and
    is the one input written unescaped. Omitted or empty, neither emits
    a section, so a report built without them is byte-identical to one
    from before they existed.
    """
    parts: list[str] = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        f"<title>{_esc(title)}</title><style>{_CSS}</style></head><body>",
        f"<h1>{_esc(title)}</h1>",
    ]
    parts.append(
        f"<p>tsdive {html.escape(__version__)}: "
        "static evidence snapshot; regenerate it after the archive changes.</p>"
    )

    if figures:
        parts.append("<h2>Plots</h2>")
        parts.extend(figures)

    if profiles:
        parts.append("<h2>Window profiles</h2>")
        for p in profiles:
            parts.append(f"<pre>{_esc(p)}</pre>")

    if findings:
        parts.append("<h2>Findings</h2>")
        for step, tags, text in findings:
            parts.append(f"<h3>{_esc(step)} - {_esc(tags)}</h3><pre>{_esc(text)}</pre>")

    if benchmarks_markdown_rows:
        parts.append("<h2>Benchmark rows</h2><table>")
        if benchmarks_header:
            parts.append(
                "<tr>" + "".join(f"<th>{_esc(c)}</th>" for c in benchmarks_header) + "</tr>"
            )
        for row in benchmarks_markdown_rows:
            parts.append("<tr>" + "".join(f"<td>{_esc(c)}</td>" for c in row) + "</tr>")
        parts.append("</table>")

    if refusal_log:
        parts.append("<h2>Refusal log</h2><ul>")
        for r in refusal_log:
            parts.append(f"<li class='refused'>{_esc(r)}</li>")
        parts.append("</ul>")

    parts.append(
        "<footer>Every number above traces to a window with its sampling "
        "contract, and refusals are listed beside the profiles. "
        "Generated offline.</footer>"
    )
    parts.append("</body></html>")
    return "\n".join(parts)


def write_static_report(html_text: str, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_text, encoding="utf-8", newline="\n")
    return out_path

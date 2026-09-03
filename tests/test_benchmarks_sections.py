"""The benchmark generator must carry every REAL section, not just one."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "examples" / "benchmarks"))

import run_benchmarks  # noqa: E402

BENCH_MD = ROOT / "BENCHMARKS.md"


def test_split_carries_every_real_section():
    text = (
        "# Benchmarks\n\n| a |\n|---|\n| 1 |\n"
        "\n## REAL: study one\n\nrows\n"
        "\n## REAL: study two\n\nmore rows\n"
        "\n## REAL: study three\n\nmore rows still\n"
    )
    generated, appended = run_benchmarks.split_appended(text)
    assert generated == "# Benchmarks\n\n| a |\n|---|\n| 1 |\n"
    assert appended.count("## REAL") == 3
    assert appended.endswith("more rows still\n")


def test_no_real_section_is_not_an_error():
    generated, appended = run_benchmarks.split_appended("# Benchmarks\n\nnothing real\n")
    assert appended == ""
    assert generated.endswith("nothing real\n")


DESIGNS = (
    "pooled-cross-well (artefact, see study)",
    "pooled-cross-well",
    "own-history",
    "per-instance-standardised",
    "onset-aligned",
)
SPLITS = ("group holdout by", "no group holdout")


def _detector_rows(appended: str) -> list[str]:
    return [
        line for line in appended.splitlines() if line.startswith("|") and "AUC " in line
    ]


def test_committed_file_has_every_study_section():
    text = BENCH_MD.read_text(encoding="utf-8")
    _, appended = run_benchmarks.split_appended(text)
    assert "## REAL: 3W v2.0.0 (real WELL-* instances)" in appended
    assert "## REAL: 3W v2.0.0, detectors (three study designs)" in appended
    assert "## REAL: TEP (Rieth 2017 simulation, subset)" in appended
    assert "## REAL: SKAB" in appended
    # A detector row without its design and split named is a number nobody
    # can check: the same tool scores differently under each design.
    detector_rows = _detector_rows(appended)
    assert detector_rows
    assert all(any(d in line for d in DESIGNS) for line in detector_rows)
    assert all(any(s in line for s in SPLITS) for line in detector_rows)
    # Every detector row names the manifest of the data it was scored on.
    assert all(re.search(r"`[0-9a-f]{12}`", line) for line in detector_rows)
    assert all("`14bc1397d3ee`" in line for line in detector_rows if "3W" in line)


def test_pooled_per_tag_rows_are_labelled_an_artefact():
    """The pooled tools are kept, and kept marked as an artefact of the design."""
    _, appended = run_benchmarks.split_appended(BENCH_MD.read_text(encoding="utf-8"))
    rows = _detector_rows(appended)
    artefact = "pooled-cross-well (artefact, see study)"
    for tool in ("MAD screen", "SPC individuals", "MSPC Hotelling T2", "MSPC SPE",
                 "IsolationForest"):
        pooled = [r for r in rows if tool in r and "pooled-cross-well" in r]
        assert pooled, tool
        assert all(artefact in r for r in pooled), tool
        # Each of them also carries the design that replaced it.
        assert [r for r in rows if tool in r and artefact not in r], tool
    # The population screen is pooled by construction and is not marked.
    population = [r for r in rows if "population screen" in r]
    assert population
    assert all(artefact not in r for r in population)
    # The control that reads no sensor value is published beside them.
    assert [r for r in rows if "clock control" in r and "own-history" in r]


def test_onset_aligned_rows_carry_the_metrics_that_read_them():
    """An aligned AUC without its paired hit rate and false-alarm rate misleads.

    The clock control takes the paired hit rate outright and fires on
    every pre window, so an aligned row that published only its AUC would
    look like a detector where the design says it is a timer.
    """
    _, appended = run_benchmarks.split_appended(BENCH_MD.read_text(encoding="utf-8"))
    aligned = [r for r in _detector_rows(appended) if "onset-aligned" in r]
    assert aligned
    for row in aligned:
        assert "paired hit " in row, row
        assert "FAR " in row, row
        assert "detect post1 " in row, row
    # Both controls are published: the clock the design leaves, and the
    # clock with the design's own fixed spacing taken back out.
    assert [r for r in aligned if "clock control" in r and "AUC 0.626" in r]
    assert [r for r in aligned if "design offset removed" in r and "AUC 0.500" in r]

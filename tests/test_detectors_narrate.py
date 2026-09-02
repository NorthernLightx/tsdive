"""MSPC, IsolationForest, static UI, narrator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tsdive.detectors.isolation import (
    FEATURE_COLUMNS,
    prepare_rows,
    score_isolation,
    train_isolation,
)
from tsdive.errors import MspcAlignmentError, NarratorUnavailable
from tsdive.mspc import align_windows, detect, fit_pca
from tsdive.narrate import (
    EvidenceLedger,
    OfflineScriptedNarrator,
    RemoteNarrator,
    default_narrator,
)
from tsdive.ui.static_report import render_static_report


def _matrix(n: int = 300, anomaly: bool = False, seed: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 1.0, size=(n, 3))
    third = -0.5 * base[:, 0] + 0.1 * rng.normal(size=n)
    x = np.column_stack([base[:, 0] + base[:, 1], base[:, 1], third])
    if anomaly:
        x[:20] += 6.0  # joint shift in first block
    return x


def _aligned(x: np.ndarray, start="2025-01-01"):
    idx = pd.DatetimeIndex(
        [pd.Timestamp(start) + pd.Timedelta(i, unit="min") for i in range(len(x))]
    )
    return idx, x


def test_pca_detects_joint_shift_and_replays():
    from tsdive.mspc.pca import AlignedMatrix

    train_x = _matrix(300, seed=5)
    mon_x = _matrix(120, anomaly=True, seed=9)
    cols = ["A", "B", "C"]

    def am(arr):
        idx, arr2 = _aligned(arr)
        return AlignedMatrix(index=idx, columns=list(cols), matrix=arr2, coverage=1.0)

    model = fit_pca(am(train_x))
    det = detect(model, am(mon_x))
    assert len(det.t2_breaches) >= 10
    # Determinism: same inputs -> identical limits and scores.
    model2 = fit_pca(am(train_x))
    det2 = detect(model2, am(mon_x))
    assert model.t2_limit == model2.t2_limit
    assert np.array_equal(det.t2, det2.t2)


def _shell_window(point: str, stamps, values: list[float]):
    from typing import cast

    from tsdive.store.identity import TagIdentity
    from tsdive.store.sampling_contract import CalculationBasis, RetrievalMode, SamplingContract
    from tsdive.store.tagstore import Window

    frame = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp(t) for t in stamps],
            "value": values,
            "quality": ["GOOD"] * len(stamps),
            "severity": ["GOOD"] * len(stamps),
            "valid": [True] * len(stamps),
        }
    )
    return Window(
        identity=TagIdentity("p", point),
        meta=None,  # type: ignore[arg-type]
        contract=SamplingContract(CalculationBasis.TIME_WEIGHTED, RetrievalMode.RECORDED),
        start=cast("pd.Timestamp", frame["timestamp"].min()),
        end=cast("pd.Timestamp", frame["timestamp"].max()),
        frame=frame,
        physics=None,  # type: ignore[arg-type]
    )


def test_alignment_refuses_low_coverage():
    base = pd.Timestamp("2025-01-01")
    on_grid = [base + pd.Timedelta(i, unit="min") for i in range(30)]
    off_grid = [base + pd.Timedelta(i * 60 + 31, unit="s") for i in range(29)]
    w_on = _shell_window("A.PV", on_grid, list(range(30)))
    w_off = _shell_window("B.PV", off_grid, list(range(29)))
    with pytest.raises(MspcAlignmentError):
        align_windows([w_on, w_off], rate_s=60, min_coverage=0.95)


def test_alignment_requires_two_windows():
    base = pd.Timestamp("2025-01-01")
    w = _shell_window("A.PV", [base + pd.Timedelta(i, unit="min") for i in range(5)], [0.0] * 5)
    with pytest.raises(MspcAlignmentError):
        align_windows([w], rate_s=60)


def test_isolation_scores_anomalies_higher():
    rng = np.random.default_rng(2)
    normal = pd.DataFrame(
        {
            "tag": ["t"] * 200,
            "window_start": [str(i) for i in range(200)],
            **{c: rng.normal(0, 1, 200) for c in FEATURE_COLUMNS},
        }
    )
    rows, excluded = prepare_rows(normal)
    assert excluded == 0
    model = train_isolation(rows)

    anomalous = pd.DataFrame(
        {
            "tag": ["t"] * 5,
            "window_start": [f"x{i}" for i in range(5)],
            **{c: rng.normal(12, 1, 5) for c in FEATURE_COLUMNS},
        }
    )
    combined = pd.concat([rows, prepare_rows(anomalous)[0]], ignore_index=True)
    scores = score_isolation(model, combined).scores
    assert scores.iloc[-5:].mean() > scores.iloc[:-5].mean()


def test_prepare_rows_counts_incomplete():
    frame = pd.DataFrame(
        {
            "tag": ["a"] * 3,
            "window_start": ["0", "1", "2"],
            **{c: [1.0, 2.0, None] for c in FEATURE_COLUMNS},
        }
    )
    rows, excluded = prepare_rows(frame)
    assert len(rows) == 2 and excluded == 1

    model = train_isolation(rows)
    result = score_isolation(model, rows, n_excluded_incomplete=excluded)
    assert result.n_excluded_incomplete == 1
    assert len(result.scores) == 2


def test_static_report_escapes_and_contains_sections():
    html_text = render_static_report(
        title="evidence <snapshot>",
        profiles=["tag: demo:x\nunits: <Nm3/h>"],
        benchmarks_markdown_rows=[("auc", "0.99")],
        benchmarks_header=("metric", "value"),
        refusal_log=["[IncomparableUnitsError] <bad>"],
    )
    assert "&lt;Nm3/h&gt;" in html_text
    assert "[IncomparableUnitsError]" in html_text
    assert "<script" not in html_text.lower().replace("javascript", "")
    assert "<th>metric</th><th>value</th>" in html_text
    assert "<td>auc</td><td>0.99</td>" in html_text


def test_static_report_without_header_emits_no_th():
    """A caller's first row is data unless a header is declared."""
    html_text = render_static_report(
        title="t",
        profiles=[],
        benchmarks_markdown_rows=[("profiles rendered", "2"), ("refusals recorded", "1")],
        refusal_log=[],
    )
    assert "<th>" not in html_text
    assert "<td>profiles rendered</td><td>2</td>" in html_text


def test_static_report_without_figures_is_unchanged():
    """``figures=None`` and ``figures=[]`` both draw the report of before."""
    kwargs: dict = {
        "title": "t",
        "profiles": ["demo:x  a tag"],
        "benchmarks_markdown_rows": [("profiles rendered", "1")],
        "refusal_log": [],
    }
    plain = render_static_report(**kwargs)
    assert render_static_report(**kwargs, figures=None) == plain
    assert render_static_report(**kwargs, figures=[]) == plain
    assert "<h2>Plots</h2>" not in plain
    drawn = render_static_report(**kwargs, figures=["<svg><title>x</title></svg>"])
    assert drawn.index("<h2>Plots</h2>") < drawn.index("<h2>Window profiles</h2>")


def test_ledger_text_opens_with_the_header_it_is_given():
    """``tsdive run`` hands its own stdout in, so both say one thing."""
    ledger = EvidenceLedger(
        title="run demo.toml",
        profiles=["demo:x  a tag"],
        findings=[{"step": "spc", "tags": "demo:x", "text": "demo:x  0 rule hits"}],
        refusals=["[MspcAlignmentError] mspc demo:x: aligned coverage 0.867"],
    )
    printed = ["run demo.toml   1 archive   1 step", "", "wrote     out/ledger.json"]
    assert ledger.to_text(printed).splitlines() == [
        *printed,
        "",
        "FINDING  spc   demo:x",
        "demo:x  0 rule hits",
    ]


def test_ledger_text_falls_back_to_its_own_counts():
    ledger = EvidenceLedger(title="snapshot", refusals=["[SchemaError] no meta"])
    assert ledger.to_text().splitlines() == [
        "snapshot   profiles 0   findings 0   benchmark rows 0   refusals 1",
        "REFUSAL  [SchemaError] no meta",
    ]


def test_narrator_offline_and_gated():
    ledger = EvidenceLedger(title="demo", refusals=["[InsufficientQuality] no GOOD samples"])
    text = OfflineScriptedNarrator().narrate(ledger)
    assert "1 typed refusal(s)" in text
    # Remote without env var refuses (provoked in test_errors too):
    saved = None
    import os

    had = os.environ.pop("TSDIVE_LLM_ENDPOINT", None)
    try:
        with pytest.raises(NarratorUnavailable):
            RemoteNarrator().narrate(ledger)
        assert isinstance(default_narrator(), OfflineScriptedNarrator)
    finally:
        if had:
            os.environ["TSDIVE_LLM_ENDPOINT"] = had
            saved = had
    del saved

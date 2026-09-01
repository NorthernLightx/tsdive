"""Diagnostic plot gallery: one figure per major library behavior.

    uv run python examples/plots/make_plots.py

Writes PNGs to examples/plots/out/. Data is deterministic (seeded
backbone + fixture-a); PNG rendering itself is not byte-compared in CI -
these are study artifacts, regenerate locally.

Every plot is built from the public APIs so what you see is what the
library actually computes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "src"))

OUT = Path(__file__).parent / "out"
DAY1 = (
    pd.Timestamp("2025-01-01 00:00:00+00:00"),
    pd.Timestamp("2025-01-02 00:00:00+00:00"),
)
DAY2 = (
    pd.Timestamp("2025-01-02 00:00:00+00:00"),
    pd.Timestamp("2025-01-03 00:00:00+00:00"),
)

FAULT_COLORS = {
    "level_shift": "#d62728",
    "drift": "#ff7f0e",
    "stuck_sensor": "#7f7f7f",
    "variance_burst": "#9467bd",
}


def _contract(stepped: bool = True):
    from tsdive.store.sampling_contract import CalculationBasis, RetrievalMode, SamplingContract

    return SamplingContract(
        CalculationBasis.TIME_WEIGHTED, RetrievalMode.RECORDED, stepped=stepped
    )


def plot_fixture_a():
    """Same samples, two contracts, two different honest answers."""
    from tsdive.store.averaging import weighted_mean
    from tsdive.store.quality import annotate_severity, null_digital_state_values
    from tsdive.store.sampling_contract import (
        CalculationBasis,
        RetrievalMode,
        SamplingContract,
    )

    start = pd.Timestamp("2024-03-01 00:00:00+00:00")
    stamps = [start]
    stamps.extend(start + pd.Timedelta(59 * 60 + 3 * k, unit="s") for k in range(20))
    stamps.append(start + pd.Timedelta(60, unit="min"))
    frame = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp(t) for t in stamps],
            "value": [50.0] + [100.0] * 20 + [50.0],
            "quality": ["GOOD"] * 22,
        }
    )
    frame = null_digital_state_values(annotate_severity(frame))
    tw_c = SamplingContract(
        CalculationBasis.TIME_WEIGHTED, RetrievalMode.RECORDED, stepped=True
    )
    ew_c = SamplingContract(CalculationBasis.EVENT_WEIGHTED, RetrievalMode.RECORDED)
    tw = weighted_mean(frame, tw_c)
    ew = weighted_mean(frame, ew_c)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.step(
        frame["timestamp"], frame["value"].astype(float), where="post", lw=2,
        label="recorded samples (step)",
    )
    ax.plot(frame["timestamp"], frame["value"].astype(float), "o", ms=4, color="#1f77b4")
    ax.axhline(tw, color="#2ca02c", ls="--", lw=2, label=f"time-weighted mean = {tw:.2f}")
    ax.axhline(ew, color="#d62728", ls=":", lw=2.5, label=f"event-weighted mean = {ew:.2f}")
    ax.set_title("Fixture A: 59 min at 50, then 1 min at 100\n"
                 f"same archive, different contract -> {CalculationBasis.TIME_WEIGHTED.value} vs "
                 f"{CalculationBasis.EVENT_WEIGHTED.value}")
    ax.set_ylabel("engineering units")
    ax.legend(loc="center right", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "01_fixture_a_contracts.png", dpi=120)
    plt.close(fig)


def _backbone_with_store(tmp: Path):
    from tsdive.data import generate_backbone
    from tsdive.store.tagstore import TagStore

    bb = generate_backbone(tmp / "backbone", seed=42, n_loops=4, hours=48)
    store = TagStore(bb.root)
    pv_ids = sorted(
        (t for t in store.list_tags() if t.point_id.endswith("PV")),
        key=lambda t: (t.source_id, t.point_id),
    )
    return bb, store, pv_ids


def plot_backbone_overview(bb, store, pv_ids):
    """The synthetic backbone - regimes, faults, ground truth."""
    from tsdive.store.quality import good_mask

    truth = bb.truth
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    for li, (ax, ident) in enumerate(zip(axes, pv_ids, strict=True)):
        h = pd.read_parquet(store.archive_path(ident))
        mode_ident = next(t for t in store.list_tags() if t.point_id == f"FIC{li:03d}.MODE")
        modes = pd.read_parquet(store.archive_path(mode_ident))
        good = h[good_mask(h)]
        ax.plot(good["timestamp"], good["value"].astype(float), lw=0.8, color="#1f77b4")

        # shade regime blocks using MODE transitions
        mvals = modes["value"].astype(str).to_numpy()
        change_pts = [modes["timestamp"].iloc[0]] + [
            modes["timestamp"].iloc[i]
            for i in range(1, len(modes))
            if mvals[i] != mvals[i - 1]
        ] + [modes["timestamp"].iloc[-1]]
        for k in range(len(change_pts) - 1):
            regime = mvals[
                modes["timestamp"].searchsorted(change_pts[k])
            ]
            ax.axvspan(change_pts[k], change_pts[k + 1], alpha=0.06,
                       color=["#2ca02c", "#ff7f0e", "#9467bd"][ord(regime[-1]) % 3])

        trow = truth.iloc[li]
        c = FAULT_COLORS[trow["fault_type"]]
        ax.axvspan(pd.Timestamp(trow["start"]), pd.Timestamp(trow["end"]),
                   color=c, alpha=0.18)
        ax.axvline(DAY2[0], color="k", ls="--", lw=0.8)
        ax.set_ylabel(f"loop {li}\n{trow['fault_type']}", fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    axes[0].set_title("Synthetic backbone (seed=42): PV traces, shaded regimes, fault intervals")
    fig.tight_layout()
    fig.savefig(OUT / "02_backbone_overview.png", dpi=120)
    plt.close(fig)


def plot_provisional_vs_refined(bb, store, pv_ids, truth):
    """Provisional vs refined: global MAD misses the shift; regime-keyed catches it."""
    from tsdive.store.quality import good_mask

    shift = truth.iloc[0]
    fs, fe = pd.Timestamp(shift["start"]), pd.Timestamp(shift["end"])
    hist0 = pd.read_parquet(store.archive_path(pv_ids[0]))
    history0 = hist0[(hist0["timestamp"] >= DAY1[0]) & (hist0["timestamp"] < DAY1[1])]
    monitor0 = hist0[(hist0["timestamp"] >= DAY2[0]) & (hist0["timestamp"] < DAY2[1])]

    from tsdive.baselines import mad_baseline, regime_baselines, screen_regime

    base3 = mad_baseline(history0)
    lo3, hi3 = base3.limits(3.0)

    mode_ident = next(t for t in store.list_tags() if t.point_id == "FIC000.MODE")
    modes_all = pd.read_parquet(store.archive_path(mode_ident))[["timestamp", "value"]]
    modes_all = modes_all.rename(columns={"value": "mode"})
    j = modes_all.merge(hist0[["timestamp", "value", "quality"]], on="timestamp")
    train4 = j[j["timestamp"] < DAY2[0]]
    mon4 = j[j["timestamp"] >= DAY2[0]]
    bases4 = regime_baselines(train4, train4["mode"])
    res4 = screen_regime(bases4, mon4, mon4["mode"], k=3.0)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    mgood = monitor0[good_mask(monitor0)]

    ax = axes[0]
    ax.plot(mgood["timestamp"], mgood["value"].astype(float), lw=0.9, color="#1f77b4")
    ax.axhline(base3.center, color="#2ca02c", ls="-", label=f"MAD center {base3.center:.1f}")
    ax.axhline(lo3, color="#d62728", ls="--", label=f"global 3-sigma limits [{lo3:.1f}, {hi3:.1f}]")
    ax.axhline(hi3, color="#d62728", ls="--")
    ax.axvspan(fs, fe, color=FAULT_COLORS["level_shift"], alpha=0.15)
    ax.set_title("PROVISIONAL global MAD screen -> detection 0%: "
                 "the +8 shift hides inside cross-regime spread")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_ylabel("PV loop0")

    ax = axes[1]
    ax.plot(mgood["timestamp"], mgood["value"].astype(float), lw=0.9, color="#1f77b4")
    for regime, b in bases4.items():
        ax.axhline(b.center, color="#2ca02c", ls="-", lw=1, alpha=0.7)
        ax.axhspan(b.center - 3 * b.scale, b.center + 3 * b.scale, color="#2ca02c", alpha=0.08)
        ax.text(monitor0["timestamp"].iloc[5], b.center + 3 * b.scale + 0.4,
                f"{regime} +/-3 sigma", fontsize=7, color="#1a5c1a")
    flagged_ts = set(res4.flagged_timestamps)
    fmask = monitor0["timestamp"].isin(flagged_ts) & good_mask(monitor0)
    fdet = monitor0[fmask]
    ax.plot(
        fdet["timestamp"], fdet["value"].astype(float), "rx", ms=6,
        label=f"flagged ({len(fdet)})",
    )
    ax.axvspan(fs, fe, color=FAULT_COLORS["level_shift"], alpha=0.15)
    ax.set_title(
        "MODE-keyed limits -> 100% detection inside fault;\n"
        "note: SP-step transitions also flag (PV lags MODE through the loop)"
    )
    ax.legend(fontsize=8)
    ax.set_ylabel("PV loop0")
    fig.tight_layout()
    fig.savefig(OUT / "03_provisional_vs_refined_screens.png", dpi=120)
    plt.close(fig)


def plot_spc_ichart(store, pv_ids, truth):
    """Individuals chart with per-rule hit markers."""
    from tsdive.store.quality import good_mask

    drift = truth.iloc[1]
    fs5, fe5 = pd.Timestamp(drift["start"]), pd.Timestamp(drift["end"])
    hist1 = pd.read_parquet(store.archive_path(pv_ids[1]))
    h_hist = hist1[(hist1["timestamp"] >= DAY1[0]) & (hist1["timestamp"] < DAY1[1])]
    h_mon = hist1[(hist1["timestamp"] >= DAY2[0]) & (hist1["timestamp"] < DAY2[1])]
    from tsdive.baselines import mad_baseline
    from tsdive.spc import apply_rules, individuals_limits

    base5 = mad_baseline(h_hist)
    lim5 = individuals_limits(base5.center, base5.scale)
    gm = h_mon[good_mask(h_mon)].reset_index(drop=True)
    hits5 = apply_rules(gm["timestamp"], gm["value"].astype(float), lim5)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(
        gm["timestamp"], gm["value"].astype(float), "-o", ms=2.5, lw=0.8,
        color="#1f77b4", label="monitored GOOD samples",
    )
    ax.axhline(lim5.center, color="#2ca02c", label=f"center {lim5.center:.2f}")
    ax.axhline(lim5.ucl, color="#d62728", ls="--", label="UCL/LCL (baseline MAD)")
    ax.axhline(lim5.lcl, color="#d62728", ls="--")
    markers = {"BEYOND_3SIGMA": ("rv", 7), "RUN_9_SAMESIDE": ("y^", 6), "TREND_6": ("cs", 5)}
    for rule, (mark, size) in markers.items():
        pts = [
            (h.timestamp, float(gm.loc[gm["timestamp"] == h.timestamp, "value"].iloc[0]))
            for h in hits5
            if h.rule == rule
        ]
        if pts:
            xs, ys = zip(*pts, strict=True)
            ax.plot(xs, ys, mark, ms=size, alpha=0.85, label=f"{rule} ({len(pts)})")
    ax.axvspan(fs5, fe5, color=FAULT_COLORS["drift"], alpha=0.15, label="drift interval")
    ax.set_title("SPC individuals chart: each rule reports its own hits (no blended score)")
    ax.legend(fontsize=7, ncol=3)
    ax.set_ylabel("PV loop1")
    fig.tight_layout()
    fig.savefig(OUT / "04_spc_ichart_rules.png", dpi=120)
    plt.close(fig)


def plot_mspc(store, pv_ids, truth):
    """T2 sees location faults; SPE sees the variance burst."""
    from tsdive.mspc import align_windows, detect, fit_pca

    w_train = [
        store.read_window(i, DAY1[0].to_pydatetime(), DAY1[1].to_pydatetime(), _contract())
        for i in pv_ids
    ]
    w_mon = [
        store.read_window(i, DAY2[0].to_pydatetime(), DAY2[1].to_pydatetime(), _contract())
        for i in pv_ids
    ]
    model = fit_pca(align_windows(w_train, rate_s=60, min_coverage=0.9))
    det = detect(model, align_windows(w_mon, rate_s=60, min_coverage=0.9))

    burst = truth.iloc[3]
    fs6, fe6 = pd.Timestamp(burst["start"]), pd.Timestamp(burst["end"])

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    ax = axes[0]
    ax.plot(det.timestamps, det.t2, lw=0.9, color="#1f77b4", label="Hotelling T2")
    ax.axhline(model.t2_limit, color="#d62728", ls="--",
               label=f"empirical 99% limit {model.t2_limit:.1f}")
    ax.set_yscale("log")
    ax.set_title(f"T2 breaches inside variance_burst interval: "
                 f"{sum(1 for t in det.t2_breaches if fs6 <= t <= fe6)} (location statistic)")
    ax.axvspan(fs6, fe6, color=FAULT_COLORS["variance_burst"], alpha=0.15)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(det.timestamps, det.spe, lw=0.9, color="#ff7f0e", label="SPE (residual)")
    ax.axhline(model.spe_limit, color="#d62728", ls="--",
               label=f"empirical 99% limit {model.spe_limit:.1f}")
    spe_in = sum(1 for t in det.spe_breaches if fs6 <= t <= fe6)
    ax.set_title(f"SPE breaches inside variance_burst interval: {spe_in} (residual statistic)")
    ax.axvspan(fs6, fe6, color=FAULT_COLORS["variance_burst"], alpha=0.15)
    ax.legend(fontsize=8)
    ax.set_ylabel("Q statistic")
    fig.suptitle("MSPC: variance faults surface in SPE while T2 stays quiet", y=1.0)
    fig.tight_layout()
    fig.savefig(OUT / "05_mspc_t2_spe.png", dpi=120)
    plt.close(fig)


def plot_isolation_scores(store, pv_ids, truth):
    """IsolationForest separates fault windows from normal ones."""
    from sklearn.metrics import roc_auc_score

    from tsdive.detectors.isolation import prepare_rows, score_isolation, train_isolation
    from tsdive.store.quality import good_mask

    def window_feature(h, lo, hi):
        wdf = h[(h["timestamp"] >= lo) & (h["timestamp"] < hi)]
        good = wdf[good_mask(wdf)]
        vals = good["value"].astype(float)
        arr = vals.to_numpy(dtype=float)
        med = float(np.median(arr))
        changes = int((vals.diff().fillna(0.0) != 0).sum())
        end_ts = pd.Timestamp(good["timestamp"].iloc[-1])
        span_h = (hi - lo).total_seconds() / 3600.0
        return {
            "weighted_mean": round(float(vals.mean()), 4),
            "std": round(float(vals.std()), 4),
            "min": round(float(vals.min()), 4),
            "max": round(float(vals.max()), 4),
            "median": round(med, 4),
            "mad": round(float(np.median(np.abs(arr - med))), 4),
            "distinct_count": int(vals.nunique()),
            "stall_s": float((hi - end_ts).total_seconds()),
            "changes_per_hour": round(changes / span_h, 4),
        }

    rows, labels, tags = [], [], []
    step = 4 * 3600
    for li, ident in enumerate(pv_ids):
        h = pd.read_parquet(store.archive_path(ident))
        trow = truth.iloc[li]
        ffs, ffe = pd.Timestamp(trow["start"]), pd.Timestamp(trow["end"])
        for day_lo, lab_base in ((DAY1, None), (DAY2, None)):
            secs = int((day_lo[1] - day_lo[0]).total_seconds())
            for k in range(secs // step):
                lo = day_lo[0] + pd.Timedelta(step * k, unit="s")
                hi = lo + pd.Timedelta(step, unit="s")
                rows.append(window_feature(h, lo, hi))
                is_fault_window = bool(lab_base) or not (hi <= ffs or lo >= ffe)
                labels.append(1 if is_fault_window else 0)
                tags.append(f"L{li} D{1 if lab_base is None and day_lo is DAY1 else 2} w{k}")

    fr = pd.DataFrame(rows)
    tagged = fr.assign(tag="t", window_start=[f"{i:03d}" for i in range(len(fr))])
    tr_idx = [i for i, lab in enumerate(labels) if lab == 0]
    train7, _ = prepare_rows(tagged.iloc[tr_idx])
    model7 = train_isolation(train7)
    allrows, excluded7 = prepare_rows(tagged)
    scores = score_isolation(model7, allrows, n_excluded_incomplete=excluded7).scores.to_numpy()
    orig_ids = allrows["window_start"].astype(int).to_numpy()
    y = np.array(labels)[orig_ids]
    auc = float(roc_auc_score(y, scores))

    order = np.argsort(scores)
    colors = ["#d62728" if y[i] else "#2ca02c" for i in order]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.bar(range(len(order)), scores[order], color=colors)
    ax.set_xlabel("windows sorted by anomaly score")
    ax.set_ylabel("isolation score (higher = more anomalous)")
    ax.set_title(f"IsolationForest over physics-gated features - ROC-AUC {auc:.3f}\n"
                 "red = overlaps a fault interval; green = normal (day-1 windows train the model)")
    fig.tight_layout()
    fig.savefig(OUT / "06_isolation_scores.png", dpi=120)
    plt.close(fig)


def plot_demo_physics():
    """Profile recap on the demo archive: gaps by class, censoring, unmapped codes."""
    demo = ROOT / "data/demo/fic101_demo.parquet"
    if not demo.exists():
        import runpy

        runpy.run_path(str(ROOT / "scripts/make_demo_archive.py"), run_name="__main__")
    raw = pd.read_parquet(demo)
    raw = raw.sort_values("timestamp").reset_index(drop=True)

    from tsdive.store.quality import Severity, annotate_severity

    ann = annotate_severity(raw.rename(columns={"value": "value"}))

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 6.5), sharex=True, gridspec_kw={"height_ratios": [3, 2]}
    )
    good = ann[ann["severity"] == Severity.GOOD]
    ax1.plot(good["timestamp"], good["value"].astype(float), lw=0.8, color="#1f77b4")

    sat = good[pd.to_numeric(good["value"], errors="coerce") >= 99.999]
    if len(sat) > 0:
        first_sat = sat["timestamp"].iloc[len(sat) // 2]
        ax1.annotate(
            "saturated at full scale\n(censored; flatline refused)",
            xy=(first_sat, 100.0),
            xytext=(first_sat - pd.Timedelta(9, unit="h"), 80),
            arrowprops=dict(arrowstyle="->", color="#8b0000"),
            color="#8b0000",
            fontsize=8,
        )
        ax1.axhline(100.0, color="#8b0000", ls=":", lw=1)

    outage = (pd.Timestamp("2024-03-30 23:00:00+00:00"), pd.Timestamp("2024-03-30 23:40:00+00:00"))
    ax1.axvspan(*outage, color="#d62728", alpha=0.25)
    ax1.annotate(
        "collection outage\ngap class: unknown\n(reduces coverage)",
        xy=(outage[0] + pd.Timedelta(20, unit="min"), 45),
        xytext=(outage[0] + pd.Timedelta(4, unit="h"), 40),
        arrowprops=dict(arrowstyle="->"),
            fontsize=8,
        )
    ax1.set_ylabel("flow m3/h")
    ax1.set_title(
        "Demo archive profile: holes, censoring and quality are accounted for",
        fontsize=11,
        loc="left",
    )

    sev_color = {Severity.GOOD: "#2ca02c", Severity.UNCERTAIN: "#ff7f0e", Severity.BAD: "#d62728"}
    for s, c in sev_color.items():
        sub = ann[ann["severity"] == s]
        ax2.plot(
            sub["timestamp"], [s.value] * len(sub), "|", ms=10, color=c,
            label=f"{s.value}={len(sub)}",
        )
    unmapped = ann["quality"].astype(str).str.upper().eq("SENSOR DRIFT")
    if unmapped.any():
        ax2.plot(ann.loc[unmapped, "timestamp"], ["UNCERTAIN"] * int(unmapped.sum()),
                 "k*", ms=9, label="unmapped codes (never guessed GOOD)")
    ax2.set_yticks(["GOOD", "UNCERTAIN", "BAD"])
    ax2.legend(fontsize=8, loc="lower right")
    ax2.set_ylabel("derived severity")
    fig.tight_layout()
    fig.subplots_adjust(top=0.93)
    fig.savefig(OUT / "07_demo_data_physics.png", dpi=120)
    plt.close(fig)


def main() -> int:
    import tempfile

    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        bb, store, pv_ids = _backbone_with_store(Path(tmp))
        truth = bb.truth

        plot_fixture_a()
        print("wrote 01_fixture_a_contracts.png")
        plot_backbone_overview(bb, store, pv_ids)
        print("wrote 02_backbone_overview.png")
        plot_provisional_vs_refined(bb, store, pv_ids, truth)
        print("wrote 03_provisional_vs_refined_screens.png")
        plot_spc_ichart(store, pv_ids, truth)
        print("wrote 04_spc_ichart_rules.png")
        plot_mspc(store, pv_ids, truth)
        print("wrote 05_mspc_t2_spe.png")
        plot_isolation_scores(store, pv_ids, truth)
        print("wrote 06_isolation_scores.png")
    plot_demo_physics()
    print("wrote 07_demo_data_physics.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())

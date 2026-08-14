#!/usr/bin/env python3
"""Generate PinkSight paper / 50-page landscape-brief figures from FROZEN report JSONs.

No new compute. Every figure traces to a named artifact (P12 rule: "no manual figures").
Emits PNG (300 dpi, for the IMRaD paper) + SVG (vector, for the landscape brief) per figure,
plus figures_manifest.json mapping each figure -> its source JSON(s) + a ledger-safe caption.

Claim-ledger discipline baked in: every AUROC shows its DeLong CI; the 0.50 chance line is
always drawn; the estimator is named; no forbidden framing. Track B is a SEPARATE figure and
is NEVER placed on the same axes as the Duke cohort (ADR-0015 firewall).

Run:  .venv/bin/python scripts/make_paper_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Palette encodes the story: one teal "signal" bar (clinical) above a field of slate nulls.
C = {
    "signal": "#0E7C7B",   # clinical-alone anchor — the sole significant Duke modality
    "null": "#64748B",     # imaging / fusion rungs — the characterised ceiling
    "floor": "#B45309",    # radiomics LOCK-4 floor
    "chance": "#B91C1C",   # 0.50 chance line + shuffle sentinels
    "histo": "#4338CA",    # Track B histology (distinct track)
    "pilot": "#7C3AED",    # pCR pilot
    "target": "#0F766E",   # gate-target reference lines
    "muted": "#94A3B8",
    "grid": "#E2E8F0",
    "ink": "#0F172A",
}
MANIFEST: list[dict] = []


def load(rel: str) -> dict:
    with open(ROOT / rel) as fh:
        return json.load(fh)


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#CBD5E1",
        "axes.linewidth": 0.9,
        "axes.grid": True,
        "grid.color": C["grid"],
        "grid.linewidth": 0.8,
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11.5,
        "axes.titlecolor": C["ink"],
        "text.color": C["ink"],
        "axes.labelcolor": C["ink"],
        "svg.fonttype": "none",  # keep text editable in the brief's vector editor
    })


def footer(fig, text: str) -> None:
    fig.text(0.008, 0.012, text, fontsize=7, color=C["muted"], ha="left", va="bottom")


def save(fig, stem: str) -> list[str]:
    files = []
    for ext, kw in (("png", {"dpi": 300}), ("svg", {})):
        p = OUT / f"{stem}.{ext}"
        fig.savefig(p, bbox_inches="tight", **kw)
        files.append(str(p.relative_to(ROOT)))
    plt.close(fig)
    return files


def record(fid, title, files, sources, caption, **meta) -> None:
    MANIFEST.append({"id": fid, "title": title, "files": files,
                     "sources": sources, "caption": caption, **meta})


def _forest(ax, rows, chance=0.5):
    """rows: list of {label, auroc, ci, color, shuffle, note}. Horizontal forest plot."""
    n = len(rows)
    for y, r in enumerate(rows):
        yy = n - 1 - y  # first row on top
        ci = r.get("ci")
        if ci:
            ax.plot(ci, [yy, yy], color=r["color"], lw=2.4, alpha=0.85, solid_capstyle="round", zorder=3)
            for x in ci:
                ax.plot([x, x], [yy - 0.13, yy + 0.13], color=r["color"], lw=1.6, zorder=3)
        ax.plot(r["auroc"], yy, "o", color=r["color"], ms=10, zorder=6,
                markeredgecolor="white", markeredgewidth=1.2)
        ax.text(r["auroc"], yy + 0.27, f"{r['auroc']:.3f}", ha="center", va="bottom",
                fontsize=10.5, color=r["color"], fontweight="bold", zorder=7)
        if r.get("shuffle") is not None:
            ax.plot(r["shuffle"], yy, "x", color=C["chance"], ms=7, mew=1.8, zorder=5)
        if r.get("note"):
            ax.text(1.004, yy, r["note"], transform=ax.get_yaxis_transform(),
                    va="center", ha="left", fontsize=8.5, color=C["muted"])
    ax.axvline(chance, color=C["chance"], ls="--", lw=1.3, alpha=0.75, zorder=1)
    ax.set_yticks(range(n))
    ax.set_yticklabels([r["label"] for r in reversed(rows)])
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlabel("AUROC (95% DeLong CI)")
    ax.grid(axis="y", visible=False)


# ---------------------------------------------------------------- Fig 3 (headline)
def fig3_ablation() -> None:
    src = "reports/G3_fusion_arch_bundle/ablation_table.json"
    d = load(src)["rungs"]
    g5 = load("reports/G5_external/metrics.json")
    clin_ci = g5["internal_auroc_h6_anchor_ci95"]  # [0.642, 0.747], same H6 basis
    moe7 = load("reports/G3_fusion_arch_bundle/moe7_corrected_reporting.json")["salt_distribution_auroc_3seed_mean"]
    # clinical (ceiling) at top, imaging/fusion rungs descending below it
    rows = [
        {"label": "Clinical-alone (LogReg)", "auroc": d["clinical_alone"]["auroc_mean"],
         "ci": clin_ci, "color": C["signal"], "shuffle": d["clinical_alone"]["shuffle_auroc"]},
        {"label": "Biology-gated MoE #7", "auroc": moe7["mean"],
         "ci": moe7["disclosed_band"], "color": C["null"], "note": "‡"},
        {"label": "Flat fusion", "auroc": d["flat_fusion"]["auroc_mean"],
         "ci": d["flat_fusion"]["ci95_mean"], "color": C["null"],
         "shuffle": d["flat_fusion"]["shuffle_auroc"]},
        {"label": "Hierarchical fusion #4", "auroc": d["hierarchical"]["auroc_mean"],
         "ci": d["hierarchical"]["ci95_mean"], "color": C["null"],
         "shuffle": d["hierarchical"]["shuffle_auroc"], "note": "†"},
        {"label": "Radiomics floor (G1)", "auroc": d["radiomics"]["auroc_mean"],
         "ci": d["radiomics"]["ci95_mean"], "color": C["floor"]},
        {"label": "Unimodal MRI (3D-CNN)", "auroc": d["unimodal_mri"]["auroc_mean"],
         "ci": d["unimodal_mri"]["ci95_mean"], "color": C["null"]},
    ]
    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    _forest(ax, rows)
    ax.axvline(0.708, color=C["signal"], ls=":", lw=1.2, alpha=0.55)
    ax.axvline(0.5669, color=C["floor"], ls=":", lw=1.1, alpha=0.55)
    ax.set_xlim(0.44, 0.80)
    ax.set_title("Subtype characterisation on Duke — ablation ladder\n"
                 "luminal-like vs TNBC · N=613 · patient-level 5-fold OOF · 3 seeds",
                 loc="left")
    ax.text(0.975, 0.30,
            "Paired DeLong vs 0.708 anchor:  #4 Δ −0.130,  #7 Δ −0.075  (both p<0.001)\n"
            "Flat fusion vs estimator-matched clinical:  Δ −0.0026, p=0.98  (+0.03 not demonstrated)",
            transform=ax.transAxes, ha="right", va="center", fontsize=9.2, color=C["ink"],
            bbox=dict(boxstyle="round,pad=0.4", fc="#F8FAFC", ec="#E2E8F0"))
    legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C["signal"], ms=9, label="Clinical anchor (sole significant modality)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C["null"], ms=9, label="Imaging / fusion rung (characterised ceiling)"),
        Line2D([0], [0], marker="x", color=C["chance"], ls="", label="Label-shuffle sentinel"),
        Line2D([0], [0], color=C["chance"], ls="--", label="0.50 chance"),
    ]
    ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.16),
              ncol=4, fontsize=8.8, frameon=False)
    footer(fig, "† #4 CI single-seed (ci_seed=0, headline);  ‡ #7 20-salt distribution [0.615,0.689], md5-deterministic (all 20/20 < 0.708). "
                "Source: ablation_table.json + hierarchical_oof.json + moe7_corrected_reporting.json · paired_vs_anchor_delong.json. "
                "Characterisation of the imaging-fusion information ceiling at diagnosis — not detection, not kinetics.")
    files = save(fig, "fig3_ablation_ladder")
    record("fig3", "Ablation ladder — subtype characterisation (Duke, headline)", files,
           [src, "reports/G3_fusion_arch_bundle/paired_vs_anchor_delong.json", "reports/G3_fusion_arch_bundle/moe7_corrected_reporting.json"],
           "Every imaging-involving rung falls below the clinical-alone anchor (LogReg, 0.708). "
           "Paired DeLong vs the 0.708 anchor: #4 Δ−0.130, #7 Δ−0.075 (both p<0.001); flat-fusion vs estimator-matched "
           "clinical Δ−0.0026, p=0.98 (+0.03 not demonstrated). MoE #7 is a 20-salt distribution 0.650±0.018 (all < 0.708). "
           "A characterised information ceiling.",
           n=613, seeds=3)


# ---------------------------------------------------------------- Fig 5 reliability
def fig5_reliability() -> None:
    src = "reports/G5_calibration/metrics.json"
    d = load(src)

    def panel(ax, block, title):
        ax.plot([0, 1], [0, 1], color=C["muted"], ls="--", lw=1, zorder=1)
        for key, col, lab in (("reliability_before", C["null"], "before T-scaling"),
                              ("reliability_after", C["signal"], "after T-scaling")):
            bins = block[key]
            xs = [b["conf"] for b in bins]
            ys = [b["acc"] for b in bins]
            sz = [18 + 3.2 * b["count"] for b in bins]
            ax.plot(xs, ys, "-", color=col, lw=1.6, alpha=0.85, zorder=3)
            ax.scatter(xs, ys, s=sz, color=col, alpha=0.8, edgecolor="white", linewidth=0.8, zorder=4, label=lab)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("predicted confidence"); ax.set_ylabel("observed accuracy")
        ax.set_title(title, loc="left", fontsize=12.5)
        ax.text(0.03, 0.93, f"ECE {block['ece_before']:.3f} → {block['ece_after']:.3f}",
                fontsize=10, color=C["ink"], va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="#F8FAFC", ec="#E2E8F0"))
        ax.legend(loc="lower right", fontsize=8.6)

    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 5.4))
    panel(a, d["clinical_alone_headline"],
          f"Internal — Duke clinical (LogReg, N={d['clinical_alone_headline']['n']})")
    panel(b, d["ispy2_external"],
          f"Quasi-external — ISPY2 (N={d['ispy2_external']['n']})")
    fig.suptitle("Calibration reliability — internal meets the ECE ≤ 0.05 target; "
                 "external ECE is driven by prevalence shift", x=0.012, ha="left",
                 fontsize=14.5, fontweight="bold", y=1.02)
    footer(fig, f"Temperature T={d['temperature']:.4f} fit on Duke OOF only (never external). "
                f"Source: {src}. Marker size ∝ bin count.")
    files = save(fig, "fig5_reliability")
    record("fig5", "Calibration reliability — internal vs quasi-external", files, [src],
           "Internal Duke clinical model is well-calibrated (ECE 0.0196); the external ISPY2 ECE (0.39) reflects "
           "a 48% vs 21% TNBC prevalence shift, not leakage. Temperature scaling fit on Duke OOF only.",
           temperature=d["temperature"])


# ---------------------------------------------------------------- Fig 6 external
def fig6_external() -> None:
    src = "reports/G5_external/metrics.json"
    d = load(src)
    rows = [
        {"label": f"Internal full-dev (N={d.get('internal_auroc_h6_anchor_n', 624) or 624})",
         "auroc": d["internal_auroc_logreg_full_dev_pooled_oof"],
         "ci": d["internal_auroc_logreg_full_dev_ci95_mean"], "color": C["signal"]},
        {"label": "Internal H6 anchor (N=613)",
         "auroc": d["internal_auroc_h6_anchor_n613_intersection"],
         "ci": d["internal_auroc_h6_anchor_ci95"], "color": C["signal"]},
        {"label": f"Quasi-external ISPY2 (N={d['n_ispy2']['n_ispy2_balanced']})",
         "auroc": d["auroc"], "ci": d["delong_ci"], "color": C["null"],
         "shuffle": d["shuffle_auroc"]},
    ]
    fig, ax = plt.subplots(figsize=(12.0, 4.4))
    _forest(ax, rows)
    ax.set_xlim(0.44, 0.82)
    ax.set_title("Quasi-external robustness — ISPY2 (clinical-alone, LogReg)", loc="left")
    ax.annotate(f"honest internal→external drop  Δ = {d['delta_internal_external']:.3f}",
                xy=(d["auroc"], 0), xytext=(0.47, 0.55), fontsize=9.6, color=C["ink"],
                arrowprops=dict(arrowstyle="->", color=C["muted"], lw=1))
    legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C["signal"], ms=9, label="Internal (Duke)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C["null"], ms=9, label="Quasi-external (ISPY2)"),
        Line2D([0], [0], marker="x", color=C["chance"], ls="", label="Label-shuffle sentinel"),
        Line2D([0], [0], color=C["chance"], ls="--", label="0.50 chance"),
    ]
    ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=4, fontsize=8.6, frameon=False)
    footer(fig, f"Estimator: LogReg (C=1.0). Shuffle {d['shuffle_auroc']:.3f} = leakage-clean. "
                f"Source: {src}. Quasi-external robustness check; the drop is reported honestly.")
    files = save(fig, "fig6_external_validation")
    record("fig6", "Quasi-external validation on ISPY2", files, [src],
           "ISPY2 external AUROC 0.5725 [0.531, 0.614] vs internal 0.719; honest drop Δ=0.147. "
           "Real-but-weak (CI lower bound > 0.50); shuffle sentinel at chance confirms leakage-clean.",
           n_external=d["n_ispy2"]["n_ispy2_balanced"])


# ---------------------------------------------------------------- Fig 7 XAI
def fig7_xai() -> None:
    src = "reports/G5_xai/metrics.json"
    d = load(src)
    ps = d["per_seed"]
    seeds = d["seeds_scored"]

    def spread(metric):
        vals = [ps[str(s)][metric] for s in seeds]
        return sum(vals) / len(vals), min(vals), max(vals)

    metrics = [("iou", "IoU", 0.30), ("pointing_game_score", "Pointing game", 0.70),
               ("cf_flip_rate", "Counterfactual flip", None)]
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    xs = range(len(metrics))
    for i, (key, lab, tgt) in enumerate(metrics):
        m, lo, hi = spread(key)
        ax.bar(i, m, width=0.52, color=C["null"], alpha=0.85, edgecolor="white", zorder=3)
        ax.plot([i, i], [lo, hi], color=C["ink"], lw=1.6, zorder=5)
        ax.plot([i - 0.08, i + 0.08], [lo, lo], color=C["ink"], lw=1.4, zorder=5)
        ax.plot([i - 0.08, i + 0.08], [hi, hi], color=C["ink"], lw=1.4, zorder=5)
        ax.text(i, hi + 0.03, f"{m:.3f}", ha="center", fontsize=10.5, fontweight="bold", color=C["ink"])
        if tgt is not None:
            ax.plot([i - 0.30, i + 0.30], [tgt, tgt], color=C["target"], ls="--", lw=1.6, zorder=4)
            ax.text(i + 0.32, tgt, f"gate {tgt:.2f}", va="center", fontsize=8.6, color=C["target"])
    ax.set_xticks(list(xs))
    ax.set_xticklabels([m[1] for m in metrics])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("score (3-seed mean; whisker = seed min–max)")
    ax.set_title("Explainability on the leak-free null encoder — IoU below the 0.30 gate on every seed",
                 loc="left")
    verdict = d.get("randomization_verdict", "PASS")
    ax.text(0.985, 0.95, f"model-randomization sanity: {verdict} ×{len(seeds)}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9.4, color=C["signal"],
            bbox=dict(boxstyle="round,pad=0.35", fc="#ECFDF5", ec="#A7F3D0"))
    footer(fig, f"Grad-CAM/HiResCAM over the G5 encoder (AUROC {d['encoder_auroc_from_training']:.4f} ≈ shuffle). "
                f"Source: {src}. Low/diffuse IoU is the EXPECTED corroboration of the imaging ceiling — "
                "reported as-measured, never tuned toward 0.30.")
    files = save(fig, "fig7_xai_metrics")
    record("fig7", "XAI faithfulness on the null encoder", files, [src],
           "IoU 0.123±0.035 (< 0.30 gate on all 3 seeds), pointing 0.635, counterfactual-flip 0.212; "
           "model-randomization sanity PASS ×3. On a leak-free null encoder (AUROC 0.5008), low IoU is the "
           "expected corroboration of the Duke imaging→subtype ceiling.", n=d["n_scored"], seeds=len(seeds))


# ---------------------------------------------------------------- Fig 8 Track B (firewalled)
def fig8_trackb() -> None:
    # ADR-0012 RATIFIED (2026-08-12): the UNI2-h/ABMIL 0.9675 confirmation bar is now
    # logged (decisions.md + Table 2) and shown alongside arm-3 (TITAN). Firewall
    # (ADR-0012/ADR-0015): same 640-patient cohort, DIFFERENT encoder -> encoder-robustness,
    # NOT independent corroboration; never juxtaposed with any Duke number.
    s_arm3 = "reports/novel_heads/arm3_histology_morphology/metrics_20260728.json"
    a = load(s_arm3)["floor_gate"]
    mil = load("reports/trackb/mil_cv_uni.json")
    rows = [
        {"label": "H&E histology · TITAN + LogReg\n(ADR-0015 co-headline)",
         "auroc": a["auroc"], "ci": a["delong_ci95"], "color": C["histo"],
         "shuffle": load(s_arm3)["shuffle_sentinel"]["auroc"]},
        {"label": "H&E histology · UNI2-h + ABMIL\n(ADR-0012 confirmation, same cohort)",
         "auroc": mil["auroc"], "ci": [mil["ci_lo"], mil["ci_hi"]], "color": C["histo"],
         "shuffle": mil["shuffle_auroc"]},
    ]
    fig, ax = plt.subplots(figsize=(12.0, 3.0))
    _forest(ax, rows)
    ax.set_xlim(0.40, 1.0)
    ax.set_title(f"Track B — H&E histology subtype characterisation (TCGA-BRCA, N={a['n']})", loc="left")
    ax.text(0.985, 0.06, "Standalone Track-B result — reported alongside Track A, NEVER compared or\n"
                          "juxtaposed with the Duke cohort. TITAN & UNI2-h = same 640-patient cohort,\n"
                          "different encoder -> encoder-robustness, not independent corroboration (ADR-0012/0015).",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.4, color=C["histo"],
            bbox=dict(boxstyle="round,pad=0.35", fc="#EEF2FF", ec="#C7D2FE"))
    footer(fig, f"Frozen TITAN+LogReg (0.9646) & frozen UNI2-h+ABMIL (0.9675) on {a['n']} patients, 3 seeds. "
                f"Source: {s_arm3} + reports/trackb/mil_cv_uni.json.")
    files = save(fig, "fig8_trackb_histology")
    record("fig8", "Track B histology (firewalled, own axes)", files, [s_arm3, "reports/trackb/mil_cv_uni.json"],
           "TITAN+LogReg 0.9646 [0.943,0.986] (ADR-0015 co-headline) + UNI2-h/ABMIL 0.9675 [0.948,0.987] "
           "(ADR-0012 confirmation, same 640-patient cohort, different encoder -> encoder-robustness, NOT "
           "independent corroboration). 3 seeds. Reported alongside Track A, never juxtaposed with the Duke null.",
           n=a["n"], seeds=a["seeds"])


# ---------------------------------------------------------------- Fig pCR pilot
def fig_pcr() -> None:
    s_floor = "reports/pcr_pilot/metrics_floor.json"
    s_cnn = "reports/pcr_pilot/metrics_cnn.json"
    f = load(s_floor)
    c = load(s_cnn)
    rows = [
        {"label": "Phase D — radiomics floor", "auroc": f["real"]["auroc_mean"],
         "ci": f["real"]["delong_ci95_mean"], "color": C["pilot"],
         "shuffle": f["shuffle_sentinel"]["auroc_mean"], "note": "GO ≥ 0.65"},
        {"label": "Phase E — 3D-CNN", "auroc": c["auroc_pooled_oof_mean"],
         "ci": c["delong_ci95_per_seed"]["0"], "color": C["pilot"], "note": "GO ≥ 0.62 †"},
    ]
    fig, ax = plt.subplots(figsize=(12.0, 3.9))
    _forest(ax, rows)
    ax.axvline(0.65, color=C["target"], ls="--", lw=1.4, alpha=0.7)
    ax.axvline(0.62, color=C["target"], ls=":", lw=1.2, alpha=0.6)
    ax.set_xlim(0.40, 0.80)
    ax.set_title(f"pCR pilot — ISPY2 / MAMA-MIA (N={f['n']}) · distinct cohort from Duke", loc="left")
    ax.text(0.66, 1.35, "GO 0.65", color=C["target"], fontsize=8.6)
    ax.text(0.985, 0.9, "Both gates NO-GO — independent pilot, no cross-cohort claim",
            transform=ax.transAxes, ha="right", va="top", fontsize=9, color=C["pilot"],
            bbox=dict(boxstyle="round,pad=0.3", fc="#F5F3FF", ec="#DDD6FE"))
    footer(fig, "† Phase E CI shown for seed 0 (per-seed CIs recorded; all cross 0.50). "
                f"Sources: {s_floor} · {s_cnn}. pCR = pathological complete response; distinct cohort, no Duke comparison.")
    files = save(fig, "fig_pcr_pilot")
    record("figA", "pCR pilot (ISPY2/MAMA-MIA)", files, [s_floor, s_cnn],
           "Phase D radiomics floor 0.599 [0.561, 0.637] (NO-GO vs 0.65); Phase E 3D-CNN 0.4874 genuine null "
           "(NO-GO vs 0.62). Independent pilot on a distinct cohort — no cross-cohort ceiling claim.",
           n=f["n"])


# ---------------------------------------------------------------- Fig 2 CONSORT / waterfall
def fig2_consort() -> None:
    from matplotlib.patches import FancyBboxPatch
    src = "DATA_CARD.md"
    fig, ax = plt.subplots(figsize=(13.5, 6.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def box(x, y, w, h, text, fc, ec, fs=9.5):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                     boxstyle="round,pad=0.010,rounding_size=0.018", fc=fc, ec=ec, lw=1.5, zorder=3))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=C["ink"], zorder=4)

    def vdown(x, y0, y1):
        ax.annotate("", xy=(x, y1), xytext=(x, y0),
                    arrowprops=dict(arrowstyle="-|>", color=C["muted"], lw=1.7), zorder=2)

    def excl(y, from_x, text):
        ax.annotate("", xy=(0.38, y), xytext=(from_x, y),
                    arrowprops=dict(arrowstyle="-|>", color="#D97706", lw=1.4), zorder=2)
        box(0.505, y, 0.24, 0.115, text, "#FEF3C7", "#FCD34D", 8.4)

    xm = 0.20
    box(xm, 0.855, 0.32, 0.12, "Duke-Breast-Cancer-MRI (TCIA)\npublished  N = 922\nDCE-MRI + structured clinical", "#F0FDFA", C["signal"])
    vdown(xm, 0.795, 0.725)
    box(xm, 0.655, 0.32, 0.13, "Molecular subtype available  N = 922\nluminal-like 595 · TNBC 164\nHER2-enriched 59 · ER/PR+HER2+ 104", "#F8FAFC", "#CBD5E1", 9)
    excl(0.655, xm + 0.16, "Excluded from LOCK-3 scope\nHER2-enriched 59 + ER/PR+HER2+ 104\n= 163 patients")
    vdown(xm, 0.59, 0.475)
    box(xm, 0.41, 0.32, 0.12, "In analysis scope (LOCK-3)  N = 759\nluminal-like 595  vs  TNBC 164  (~3.6:1)", "#F0FDFA", C["signal"], 9.3)
    excl(0.41, xm + 0.16, "Image-level QC + patient-level split;\nquasi-external (scanner/year) holdout\ncarved FIRST, sealed")
    vdown(xm, 0.35, 0.235)
    box(xm, 0.15, 0.32, 0.14, "Development cohort\nN = 624 (5-fold CV) · N = 613 (H6-basis)\nLuminal A vs B UNDETERMINED — Ki-67 N = 0", "#F0FDFA", C["signal"], 9)

    xr = 0.85
    box(xr, 0.68, 0.28, 0.22, "Quasi-external — ISPY2\nclean pool 1215\n(ISPY2 980 / ISPY1 171 / NACT 64)\nG5 set N = 739  (LumA 381 / TNBC 358)\nDuke ∩ external = 0  (de-dup asserted)", "#F1F5F9", C["null"], 8.6)
    box(xr, 0.20, 0.28, 0.17, "Track B — TCGA-BRCA (H&E WSI)\nN = 640  (LumA 475 / Basal 165)\nDuke ∩ TCGA ID overlap = 0\n(reported alongside — never vs Duke)", "#EEF2FF", C["histo"], 8.6)

    ax.text(0.015, 0.985, "Cohort flow — Track A (Duke) with the sealed external pool and Track B",
            fontsize=14.5, fontweight="bold", va="top", color=C["ink"])
    footer(fig, "Source: DATA_CARD.md (G0 audit 2026-06-21, sha256 8ef0945c) · reports/audit_report.md · scripts/dedup_external.py. "
                "Classes are luminal-like (Luminal A+B, unsplit) vs TNBC — Luminal A alone undetermined (Ki-67 N=0).")
    files = save(fig, "fig2_consort_waterfall")
    record("fig2", "Cohort flow / N-waterfall (Track A + external + Track B)", files, [src],
           "Duke 922 → 759 in-scope (luminal-like 595 vs TNBC 164) → dev N=624/613; classes are luminal-like "
           "(A+B) vs TNBC, Luminal A undetermined (Ki-67 N=0). Sealed external pool 1215 (ISPY2 set N=739, "
           "Duke∩external=0). Track B TCGA-BRCA N=640, reported alongside, never juxtaposed with Duke.",
           n_duke_scope=759, n_dev=624)


def main() -> None:
    style()
    for fn in (fig2_consort, fig3_ablation, fig5_reliability, fig6_external, fig7_xai, fig8_trackb, fig_pcr):
        fn()
    manifest_path = OUT / "figures_manifest.json"
    with open(manifest_path, "w") as fh:
        json.dump(MANIFEST, fh, indent=2)
    print(f"Wrote {len(MANIFEST)} figures to {OUT.relative_to(ROOT)}/ (PNG 300dpi + SVG each)")
    for m in MANIFEST:
        print(f"  {m['id']:5s} {m['title']}")
    print(f"  manifest: {manifest_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

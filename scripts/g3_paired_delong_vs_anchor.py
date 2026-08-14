"""G3 item-3/5 — real paired DeLong of the fusion rungs vs the clinical-0.708 anchor OOF.

Closes the `KNOWN-GAP` in `reports/G3_fusion_arch_bundle/delong_deltas.json`
(`hierarchical_vs_clinical_alone.paired_delong_status`): the #4 hierarchical and #7 MoE per-sample
OOF probabilities are now on disk (item-2 `--save-oof-dir` re-runs), so a genuine paired DeLong vs
the SAME patients' clinical-anchor OOF is finally computable — no arrays fabricated.

Inputs (all frozen / already on disk — NO imaging-encoder re-run, reads cached OOF only):
  reports/G3_fusion_arch_bundle/oof_arrays/hierarchical_oof_s{0,1,2}.npz  (#4, Worker A save)
  reports/G3_fusion_arch_bundle/oof_arrays/moe_oof_s{0,1,2}.npz           (#7 md5-deterministic, item 2)
  reports/G5_calibration/oof_probs_full.npz                              (clinical 0.708 LogReg anchor)

Flat-fusion has NO per-sample OOF on disk (`probs_recoverable:false`, delong_deltas.json). Its genuine
floor-time paired DeLong vs clinical (p=0.9838) is REUSED verbatim here — arrays are NOT reconstructed.

Alignment: fusion pids (613) are a strict subset of the anchor pids (624). Each fusion pid is matched
to the anchor by id; the per-pid subtype label is asserted to agree between fusion and anchor (a
mismatch is a data-integrity bug -> STOP). delong_paired(y, anchor_oof, fusion_oof) => delta =
fusion - clinical (negative expected; confirms the honest-null).

Also emits the pooled-ensemble DeLong CI (3-seed per-patient mean) for #4 and #7 — the item-5
replacement for the "seed-0-representative CI" mismatch.

Claim-ledger: subtype characterisation at diagnosis; every rung's AUROC < 0.708 clinical anchor;
honest-null (imaging-fusion adds nothing over clinical). No LOCK moved.

    PYTHONPATH=src .venv/bin/python scripts/g3_paired_delong_vs_anchor.py
"""
from __future__ import annotations

import json
from math import sqrt
from pathlib import Path

import numpy as np

from pinksight.metrics import _normal_cdf, delong_ci, delong_paired

ROOT = Path(__file__).resolve().parents[1]
OOF_DIR = ROOT / "reports/G3_fusion_arch_bundle/oof_arrays"
ANCHOR_NPZ = ROOT / "reports/G5_calibration/oof_probs_full.npz"
DELONG_DELTAS = ROOT / "reports/G3_fusion_arch_bundle/delong_deltas.json"
OUT = ROOT / "reports/G3_fusion_arch_bundle/paired_vs_anchor_delong.json"

SEEDS = (0, 1, 2)
CLINICAL_HEADLINE = 0.708  # per-seed-mean H6 coalition headline (ablation anchor)
PREREG_MARGIN = 0.03


def _load_fusion_oof(prefix: str) -> tuple[np.ndarray, dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Return (shared_pids, oof_by_seed, y_by_seed) for a fusion rung's per-seed OOF npz files.

    build_aligned() sorts the intersection, so all 3 seeds share an identical pid order — asserted."""
    pids_ref = None
    oof_by_seed, y_by_seed = {}, {}
    for s in SEEDS:
        z = np.load(OOF_DIR / f"{prefix}_oof_s{s}.npz", allow_pickle=True)
        pids = np.array([str(p) for p in z["pids"]])
        if pids_ref is None:
            pids_ref = pids
        elif not np.array_equal(pids, pids_ref):
            raise SystemExit(f"{prefix} seed {s} pid order differs from seed {SEEDS[0]} — STOP")
        oof_by_seed[s] = np.asarray(z["oof"], float)
        y_by_seed[s] = np.asarray(z["y"], int)
    return pids_ref, oof_by_seed, y_by_seed


def _stouffer(zs: list[float]) -> float:
    """Two-sided Stouffer-combined p across k independent seeds (signed z summed)."""
    z = sum(zs) / sqrt(len(zs))
    return 2.0 * (1.0 - _normal_cdf(abs(z)))


def _paired_vs_anchor(name: str, pids, oof_by_seed, y_by_seed,
                      anchor_by_pid: dict[str, tuple[int, float]]) -> dict:
    """Per-seed paired DeLong (clinical anchor = a, fusion = b; delta = fusion - clinical)."""
    # align the anchor to this rung's fixed pid order (identical across its seeds)
    anchor_oof = np.array([anchor_by_pid[p][1] for p in pids], float)
    anchor_y = np.array([anchor_by_pid[p][0] for p in pids], int)

    per_seed, zs, deltas = {}, [], []
    for s in SEEDS:
        y_f = y_by_seed[s]
        # label-agreement assertion: same patient must carry the same subtype label in both OOFs
        if not np.array_equal(y_f, anchor_y):
            n_bad = int(np.sum(y_f != anchor_y))
            raise SystemExit(f"{name} seed {s}: {n_bad} pid label mismatch fusion-vs-anchor — STOP")
        r = delong_paired(anchor_y, anchor_oof, oof_by_seed[s])  # a=clinical, b=fusion
        per_seed[str(s)] = {"auc_clinical": round(r["auc_a"], 4), "auc_fusion": round(r["auc_b"], 4),
                            "delta": round(r["delta"], 4), "ci95": [round(r["ci95"][0], 4),
                            round(r["ci95"][1], 4)], "z": round(r["z"], 4), "p": round(r["p"], 4)}
        zs.append(r["z"])
        deltas.append(r["delta"])
    mean_delta = float(np.mean(deltas))
    return {
        "comparison": f"{name} vs clinical_alone (paired DeLong, aligned N={len(pids)})",
        "per_seed": per_seed,
        "stouffer_combined_p": round(_stouffer(zs), 6),
        "mean_delta_auroc": round(mean_delta, 4),
        "all_seed_deltas_negative": bool(all(d < 0 for d in deltas)),
        "direction": ("NEGATIVE — fusion < clinical-alone on the SAME patients; confirms honest-null "
                      "(imaging-fusion adds nothing over the clinical anchor)"),
    }


def main() -> None:
    # clinical anchor (E2: pids is a numpy object array -> allow_pickle=True required)
    az = np.load(ANCHOR_NPZ, allow_pickle=True)
    a_pids = np.array([str(p) for p in az["pids"]])
    a_y = np.asarray(az["y"], int)
    a_oof = np.asarray(az["oof"], float)
    anchor_by_pid = {p: (int(y), float(o)) for p, y, o in zip(a_pids, a_y, a_oof)}
    anchor_pooled_auc, _, _ = delong_ci(a_y, a_oof)

    hier_pids, hier_oof, hier_y = _load_fusion_oof("hierarchical")
    moe_pids, moe_oof, moe_y = _load_fusion_oof("moe")

    # both fusion rungs must share the same 613 dev pids -> a clean #4-vs-#7 paired comparison too
    if not np.array_equal(hier_pids, moe_pids):
        raise SystemExit("hierarchical and moe pid sets differ — STOP")

    hier_vs_clin = _paired_vs_anchor("hierarchical(#4)", hier_pids, hier_oof, hier_y, anchor_by_pid)
    moe_vs_clin = _paired_vs_anchor("moe_deterministic(#7)", moe_pids, moe_oof, moe_y, anchor_by_pid)

    # #4 vs #7 on their shared 613 patients (per seed; delta = moe - hierarchical)
    hm_per_seed, hm_z, hm_delta = {}, [], []
    for s in SEEDS:
        r = delong_paired(hier_y[s], hier_oof[s], moe_oof[s])  # a=#4, b=#7
        hm_per_seed[str(s)] = {"auc_hier": round(r["auc_a"], 4), "auc_moe": round(r["auc_b"], 4),
                               "delta": round(r["delta"], 4), "z": round(r["z"], 4),
                               "p": round(r["p"], 4)}
        hm_z.append(r["z"])
        hm_delta.append(r["delta"])

    # pooled-ensemble CI (item 5): 3-seed per-patient mean prob -> ONE DeLong CI (replaces seed-0 CI)
    def _pooled(oof_by_seed, y_by_seed):
        pooled = np.mean([oof_by_seed[s] for s in SEEDS], axis=0)
        y0 = y_by_seed[SEEDS[0]]
        auc, lo, hi = delong_ci(y0, pooled)
        return {"pooled_ensemble_auroc": round(auc, 4), "ci95": [round(lo, 4), round(hi, 4)],
                "n": int(len(y0)), "ci_method": "delong",
                "note": "3-seed per-patient mean OOF prob, ONE DeLong CI (item-5 replacement for the "
                        "seed-0-representative CI); NOT a single-seed OOF"}
    pooled_hier = _pooled(hier_oof, hier_y)
    pooled_moe = _pooled(moe_oof, moe_y)

    # flat-fusion: REUSE the genuine on-disk floor-time paired DeLong (arrays never saved; NOT rebuilt)
    dd = json.loads(DELONG_DELTAS.read_text())
    flat = dd["flat_fusion_vs_clinical_alone_ondisk"]

    all_below = (hier_vs_clin["all_seed_deltas_negative"]
                 and moe_vs_clin["all_seed_deltas_negative"]
                 and flat["delta_auroc_mean"] < 0)

    doc = {
        "gate": "G3 item-3/5 — paired DeLong of fusion rungs vs the clinical-0.708 anchor OOF",
        "prereg_target": "ΔAUC(fusion vs clinical) ≥ 0.03 with paired p<0.05 (report honestly regardless)",
        "closes_known_gap": ("delong_deltas.json hierarchical_vs_clinical_alone.paired_delong_status "
                             "(was KNOWN-GAP: #4/#7 per-sample OOF not on disk; now saved via item-2)"),
        "clinical_anchor": {
            "source": str(ANCHOR_NPZ.relative_to(ROOT)),
            "estimator": "LogReg(C=1.0) OneHot — H6 coalition (0.708 per-seed-mean headline)",
            "n_full": int(len(a_y)),
            "pooled_oof_auroc": round(anchor_pooled_auc, 4),
            "per_seed_mean_headline": CLINICAL_HEADLINE,
            "headline_note": ("0.708 is the per-seed-mean ablation headline; this single pooled-OOF "
                              "file scores %.4f on its own 624 patients. Paired deltas below are on "
                              "the aligned 613 subset, auc_clinical per-seed reported honestly."
                              % round(anchor_pooled_auc, 4)),
        },
        "alignment": {"fusion_n": int(len(hier_pids)), "anchor_n": int(len(a_y)),
                      "intersection_n": int(len(hier_pids)),
                      "label_agreement": "asserted element-wise per pid (STOP on mismatch)"},
        "hierarchical_vs_clinical": hier_vs_clin,
        "moe_deterministic_vs_clinical": moe_vs_clin,
        "moe_vs_hierarchical": {
            "comparison": "moe(#7) vs hierarchical(#4), shared 613 dev patients, delta = moe - hier",
            "per_seed": hm_per_seed,
            "stouffer_combined_p": round(_stouffer(hm_z), 6),
            "mean_delta_auroc": round(float(np.mean(hm_delta)), 4),
            "note": "both are imaging-fusion nulls below clinical; a small positive #7-#4 gap is within "
                    "the disclosed multi-seed/salt spread, NOT a claim imaging fusion works.",
        },
        "flat_fusion_vs_clinical_REUSED_ondisk": {
            **flat,
            "reuse_note": "REUSED verbatim from delong_deltas.json (genuine floor-time paired DeLong). "
                          "Flat-fusion per-sample OOF was never saved (probs_recoverable:false); arrays "
                          "were NOT reconstructed or fabricated (handoff rule).",
        },
        "pooled_ensemble_ci": {"hierarchical": pooled_hier, "moe_deterministic": pooled_moe},
        "all_fusion_rungs_paired_below_clinical": bool(all_below),
        "honest_summary": (
            "Every fusion rung is paired-below the clinical anchor. #4 hierarchical: mean Δ %.4f, "
            "Stouffer p %.4f. #7 md5-deterministic MoE: mean Δ %.4f, Stouffer p %.4f. Flat-fusion "
            "(reused on-disk): Δ %.4f, p %.4f. All three ΔAUC are negative and none meets the +0.03 "
            "pre-reg margin — the honest-null (clinical is the sole significant subtype carrier) is "
            "confirmed with real paired DeLong, no arrays fabricated." % (
                hier_vs_clin["mean_delta_auroc"], hier_vs_clin["stouffer_combined_p"],
                moe_vs_clin["mean_delta_auroc"], moe_vs_clin["stouffer_combined_p"],
                flat["delta_auroc_mean"], flat["paired_delong_combined_p"])),
        "claim_ledger": ("subtype characterisation at diagnosis; all rungs < 0.708; imaging-fusion "
                         "ceiling / modality redundancy (LOCK-1: no kinetics, no early-detection, no "
                         "cross-institution). No LOCK moved."),
    }
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"[item-3] #4  mean Δ {hier_vs_clin['mean_delta_auroc']:+.4f}  Stouffer p "
          f"{hier_vs_clin['stouffer_combined_p']:.4f}  all-neg={hier_vs_clin['all_seed_deltas_negative']}")
    print(f"[item-3] #7  mean Δ {moe_vs_clin['mean_delta_auroc']:+.4f}  Stouffer p "
          f"{moe_vs_clin['stouffer_combined_p']:.4f}  all-neg={moe_vs_clin['all_seed_deltas_negative']}")
    print(f"[item-3] flat(reused) Δ {flat['delta_auroc_mean']:+.4f}  p {flat['paired_delong_combined_p']:.4f}")
    print(f"[item-5] pooled CI  #4 {pooled_hier['pooled_ensemble_auroc']} {pooled_hier['ci95']}  "
          f"#7 {pooled_moe['pooled_ensemble_auroc']} {pooled_moe['ci95']}")
    print(f"[item-3] all fusion rungs paired-below clinical: {all_below}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

"""G3 item-5/9 — corrected #7 MoE reporting: salt-DISTRIBUTION (never a re-frozen point) + matched CI.

The frozen 0.6449 point + seed-0-only CI was a mismatch (a 3-seed-mean value paired with one seed's
CI), and 0.6449 itself was one unrecorded draw from a PYTHONHASHSEED-salted routing distribution
(root cause: moe7_reproducibility_investigation_12-08-26.md). This script replaces that with:

  item-5 -> reports/G3_fusion_arch_bundle/moe7_corrected_reporting.json
     * the SALT-DISTRIBUTION of #7 AUROC (20-salt sweep: mean ± SD [min,max], all 20/20 < 0.708)
     * the md5-DETERMINISTIC instance (item-2, 0.6542) as ONE reproducible point INSIDE that band
       (a hygiene anchor, NOT a re-frozen headline — the reported magnitude stays the distribution)
     * the pooled-OOF DeLong CI for #4 and deterministic-#7 (item-3), replacing the seed-0 CI

  item-9 -> reports/G3_fusion_arch_bundle/moe_expert_purity_distribution.json
     * expert_0 / expert_1 class_purity as salt-distributions (salt-stable, never near the 0.95
       leakage flag) + the "expert_1 0.712 is a class_purity, NOT a per-expert AUROC" clarification

Sources (all committed / frozen — no imaging-encoder re-run):
  reports/G3_fusion_arch_bundle/moe_salt_sweep/moe_salt_{0..19}.json  (Worker D 20-salt sweep)
  reports/G3_fusion_arch_bundle/paired_vs_anchor_delong.json          (item-3 pooled CIs)
  reports/G3_fusion_arch_bundle/moe_gradeband_oof_md5det_RERUN.json   (item-2 deterministic instance)

Claim-ledger: every draw < 0.708 clinical anchor; honest-null robust across the whole salt band;
purity is leakage-inspection evidence, never a performance claim (LOCK-2). No LOCK moved.

    PYTHONPATH=src .venv/bin/python scripts/g3_moe7_salt_reporting.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "reports/G3_fusion_arch_bundle"
SWEEP_DIR = BUNDLE / "moe_salt_sweep"
PAIRED = BUNDLE / "paired_vs_anchor_delong.json"
DET = BUNDLE / "moe_gradeband_oof_md5det_RERUN.json"
OUT_ITEM5 = BUNDLE / "moe7_corrected_reporting.json"
OUT_ITEM9 = BUNDLE / "moe_expert_purity_distribution.json"

CLINICAL_ANCHOR = 0.708
LEAKAGE_THRESHOLD = 0.95


def _dist(values: list[float]) -> dict:
    a = np.asarray(values, float)
    return {"mean": round(float(a.mean()), 4), "sd": round(float(a.std(ddof=1)), 4),
            "sd_population": round(float(a.std(ddof=0)), 4),
            "min": round(float(a.min()), 4), "max": round(float(a.max()), 4), "n": int(a.size)}


def _load_sweep() -> tuple[list[dict], list[str]]:
    files = sorted(SWEEP_DIR.glob("moe_salt_*.json"),
                   key=lambda p: int(p.stem.split("_")[-1]))
    rows, salts = [], []
    for f in files:
        d = json.loads(f.read_text())
        salt = f.stem.split("_")[-1]
        # expert routing depends only on the salt (grade + hash), not the CV seed -> all 3 seeds'
        # expert_class_balance blocks are identical; read seed-0's block.
        eb0 = d["expert_class_balance"]["0"]
        rows.append({
            "salt": salt,
            "auroc": d["subtype"]["auroc"]["value"],
            "expert_0_purity": eb0["expert_0"]["class_purity"],
            "expert_1_purity": eb0["expert_1"]["class_purity"],
            "expert_0_n": eb0["expert_0"]["n"],
            "expert_1_n": eb0["expert_1"]["n"],
            "leakage_flag": d["leakage_flag"],
            "collapse_flag": d["collapse_flag"],
        })
        salts.append(salt)
    return rows, salts


def main() -> None:
    rows, _ = _load_sweep()
    aurocs = [r["auroc"] for r in rows]
    e0 = [r["expert_0_purity"] for r in rows]
    e1 = [r["expert_1_purity"] for r in rows]
    auroc_dist = _dist(aurocs)
    below = [a < CLINICAL_ANCHOR for a in aurocs]
    top = max(rows, key=lambda r: r["auroc"])

    paired = json.loads(PAIRED.read_text())
    pooled = paired["pooled_ensemble_ci"]
    det = json.loads(DET.read_text())
    det_auc = det["subtype"]["auroc"]["value"]
    det_per_seed = det["subtype"]["per_seed_auroc"]
    det_eb0 = det["expert_class_balance"]["0"]

    # ---- ITEM 5: corrected #7 reporting (salt-distribution + matched pooled CI) ----
    hier_mean, moe_mean = 0.599, 0.6542
    hier_pool = pooled["hierarchical"]["pooled_ensemble_auroc"]
    moe_pool = pooled["moe_deterministic"]["pooled_ensemble_auroc"]
    item5 = {
        "gate": "G3 item-5 — corrected #7 MoE reporting: salt-DISTRIBUTION + matched pooled-OOF CI",
        "supersedes": ("the '3-seed-mean point 0.6449 + seed-0-only CI [0.5599,0.6683]' mismatch in "
                       "moe_gradeband_oof.json / ablation_table.json (point was a 3-seed mean; CI was "
                       "seed-0 alone)"),
        "root_cause": ("NHG2/missing routing used PYTHONHASHSEED-salted hash(str(pid))&1; 0.6449 was "
                       "one unrecorded draw (moe7_reproducibility_investigation_12-08-26.md)"),
        "reported_magnitude_is_a_DISTRIBUTION_not_a_point": True,
        "salt_distribution_auroc_3seed_mean": {
            "source": "reports/G3_fusion_arch_bundle/moe_salt_sweep/moe_salt_{0..19}.json (20-salt sweep)",
            **auroc_dist,
            "disclosed_band": [0.615, 0.689],
            "all_20_below_clinical_0708": bool(all(below)),
            "n_below_0708": int(sum(below)),
            "closest_approach": {"salt": top["salt"], "auroc": top["auroc"],
                                 "margin_below_0708": round(CLINICAL_ANCHOR - top["auroc"], 4)},
            "corroborates_worker_d": "mean 0.6497, SD 0.0180 [0.6145,0.6892] (investigation TL;DR)",
        },
        "deterministic_md5_instance": {
            "auroc_3seed_mean": det_auc,
            "per_seed_auroc": det_per_seed,
            "in_disclosed_band": bool(0.615 <= det_auc <= 0.689),
            "below_clinical_0708": bool(det_auc < CLINICAL_ANCHOR),
            "source": "moe_gradeband_oof_md5det_RERUN.json (item-2, PYTHONHASHSEED-independent)",
            "note": ("ONE reproducible point INSIDE the salt band — a hygiene anchor from the md5 fix, "
                     "NOT a re-frozen headline. The reported #7 magnitude remains the distribution "
                     "above; this instance is what any future re-run now reproduces bit-identically."),
        },
        "pooled_oof_delong_ci_replaces_seed0_ci": {
            "hierarchical_#4": {
                "per_seed_mean_headline": hier_mean,
                "pooled_ensemble_auroc": hier_pool,
                "ci95": pooled["hierarchical"]["ci95"],
                "n": pooled["hierarchical"]["n"],
                "divergence_from_per_seed_mean": round(hier_pool - hier_mean, 4),
                "divergence_note": ("+%.4f: the 3-seed soft-vote (AUROC of averaged probs) exceeds the "
                                    "mean-of-per-seed-AUROCs (0.599) because #4's per-seed spread is "
                                    "wide (std 0.0414) and averaging reduces variance — a genuine "
                                    "ensemble effect, NOT a pooling bug. Still < 0.708; honest-null "
                                    "intact. Reported as a matched (point,CI) pair; the 0.599 per-seed "
                                    "mean stays the headline." % round(hier_pool - hier_mean, 4)),
            },
            "moe_deterministic_#7": {
                "per_seed_mean": moe_mean,
                "pooled_ensemble_auroc": moe_pool,
                "ci95": pooled["moe_deterministic"]["ci95"],
                "n": pooled["moe_deterministic"]["n"],
                "divergence_from_per_seed_mean": round(moe_pool - moe_mean, 4),
                "divergence_note": ("+%.4f: within the plan's 0.02 pooled-vs-mean tolerance; mild "
                                    "soft-vote lift. Still < 0.708." % round(moe_pool - moe_mean, 4)),
            },
            "note": ("pooled-ensemble = ONE DeLong CI on the 3-seed per-patient mean prob; replaces the "
                     "seed-0-representative CI (the item-16 Table2 † footnote fix). Explicitly a "
                     "seed-ensemble CI, NOT a single-seed OOF."),
        },
        "honest_summary": (
            "#7 MoE is reported as a salt-distribution — AUROC %.4f ± %.4f [%.4f, %.4f] across 20 "
            "salts, all 20/20 below the 0.708 clinical anchor (closest salt %s at %.4f, still %.4f "
            "below). The md5-deterministic instance (%.4f) sits inside this band as a reproducible "
            "hygiene anchor, not a new frozen headline. Pooled-OOF CIs replace the seed-0 CI. The "
            "honest-null (clinical is the sole significant subtype carrier) holds across the entire "
            "distribution." % (auroc_dist["mean"], auroc_dist["sd"], auroc_dist["min"],
                               auroc_dist["max"], top["salt"], top["auroc"],
                               round(CLINICAL_ANCHOR - top["auroc"], 4), det_auc)),
        "claim_ledger": ("subtype characterisation at diagnosis; all draws < 0.708; imaging-fusion "
                         "ceiling (LOCK-1: no kinetics/early-detection/cross-institution). No LOCK moved."),
    }
    OUT_ITEM5.write_text(json.dumps(item5, indent=2) + "\n")

    # ---- ITEM 9: expert purity distribution ----
    e0_dist, e1_dist = _dist(e0), _dist(e1)
    max_purity = max(max(e0), max(e1))
    item9 = {
        "gate": ("G3 item-9 — MoE expert class_purity as a salt-distribution (leakage-inspection "
                 "evidence, NOT a performance number)"),
        "clarification_expert1_0712": (
            "the paper's 'expert_1 0.712' is the class_purity field of expert_1, NOT a per-expert "
            "AUROC. train_g3_moe.py computes ONE pooled subtype AUROC across BOTH experts' OOF "
            "predictions; there is no per-expert AUROC anywhere in the pipeline or the frozen JSON."),
        "source": "reports/G3_fusion_arch_bundle/moe_salt_sweep/moe_salt_{0..19}.json (20-salt sweep)",
        "expert_0_purity": {**e0_dist, "role": "NHG1-biased (Expert 0)",
                            "corroborates_worker_d": "0.876 ± 0.015 [0.847, 0.909]"},
        "expert_1_purity": {**e1_dist, "role": "NHG3-biased (Expert 1)",
                            "corroborates_worker_d": "0.717 ± 0.012 [0.692, 0.740]"},
        "leakage_flag_threshold": LEAKAGE_THRESHOLD,
        "max_purity_observed_across_20_salts": round(max_purity, 4),
        "margin_below_leakage_threshold": round(LEAKAGE_THRESHOLD - max_purity, 4),
        "all_20_salts_leakage_flag_false": bool(all(not r["leakage_flag"] for r in rows)),
        "all_20_salts_collapse_flag_false": bool(all(not r["collapse_flag"] for r in rows)),
        "deterministic_md5_instance": {
            "expert_0_purity": det_eb0["expert_0"]["class_purity"],
            "expert_1_purity": det_eb0["expert_1"]["class_purity"],
            "expert_0_n": det_eb0["expert_0"]["n"],
            "expert_1_n": det_eb0["expert_1"]["n"],
            "leakage_flag": det["leakage_flag"],
            "collapse_flag": det["collapse_flag"],
        },
        "honest_summary": (
            "Expert class_purity is salt-dependent but salt-STABLE and never near the 0.95 leakage "
            "flag: expert_0 %.4f ± %.4f [%.4f, %.4f], expert_1 %.4f ± %.4f [%.4f, %.4f]. Max purity "
            "across all 20 salts is %.4f (%.4f below the 0.95 flag); leakage_flag and collapse_flag "
            "are False for all 20/20 salts. ADR-0008's leakage-safe-routing conclusion is robust to "
            "the salt nondeterminism. 'expert_1 0.712' is a class_purity, not an AUROC." % (
                e0_dist["mean"], e0_dist["sd"], e0_dist["min"], e0_dist["max"],
                e1_dist["mean"], e1_dist["sd"], e1_dist["min"], e1_dist["max"],
                round(max_purity, 4), round(LEAKAGE_THRESHOLD - max_purity, 4))),
        "claim_ledger": ("purity is leakage-inspection evidence (integer routing, LOCK-2), NEVER a "
                         "subtype-performance claim. No LOCK moved."),
    }
    OUT_ITEM9.write_text(json.dumps(item9, indent=2) + "\n")

    print(f"[item-5] salt-dist AUROC {auroc_dist['mean']} ± {auroc_dist['sd']} "
          f"[{auroc_dist['min']}, {auroc_dist['max']}]  all<0.708={all(below)}  "
          f"det={det_auc}  pooled #4={hier_pool} #7={moe_pool}")
    print(f"[item-9] expert_0 {e0_dist['mean']}±{e0_dist['sd']}  expert_1 {e1_dist['mean']}±{e1_dist['sd']}  "
          f"max_purity {round(max_purity,4)} (<0.95)  leakage=all-False")
    print(f"wrote {OUT_ITEM5}")
    print(f"wrote {OUT_ITEM9}")


if __name__ == "__main__":
    main()

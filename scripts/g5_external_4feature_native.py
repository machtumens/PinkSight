from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import g5_external_eval as gee  
from pinksight.metrics import delong_ci  
from pinksight.models import clinical_encoder as ce  

CONFIG = ROOT / "configs" / "g5_external_ispy2.yaml"
NINE_FEATURE_METRICS = ROOT / "reports" / "G5_external" / "metrics.json"
RESULTS_JSON = ROOT / "reports" / "G5_external" / "external_4feature_native_metrics.json"
RESULTS_JSON_ALT = ROOT / "reports" / "G5_external" / "ispy2_4feature_matched.json"

NATIVE_NUM_IDX = [3]        
NATIVE_CAT_IDX = [0, 1, 2]  
EXCLUDED_NUM_IDX = [0, 1, 2]  
EXCLUDED_CAT_IDX = [3, 4]     

_SEEDS = (0, 1, 2)
_SHUFFLE_SEED = 0


def _native_names() -> dict:
    return {
        "numeric": [ce.FEATURES_NUM[i] for i in NATIVE_NUM_IDX],
        "categorical": [ce.FEATURES_CAT[i] for i in NATIVE_CAT_IDX],
        "excluded_numeric": [ce.FEATURES_NUM[i] for i in EXCLUDED_NUM_IDX],
        "excluded_categorical": [ce.FEATURES_CAT[i] for i in EXCLUDED_CAT_IDX],
    }


def _internal_pooled(x_num, x_cat, y, groups, cards, seeds=_SEEDS) -> dict:
    y = np.asarray(list(y), int)
    aucs, cis = [], []
    for seed in seeds:
        oof = ce.logreg_oof(x_num, x_cat, y, groups, cards, seed)
        auc, lo, hi = delong_ci(y, oof)
        aucs.append(auc)
        cis.append((lo, hi))
    return {
        "auroc": round(float(np.mean(aucs)), 4),
        "auroc_per_seed": {str(s): round(a, 4) for s, a in zip(seeds, aucs)},
        "ci95": [round(float(np.mean([c[0] for c in cis])), 4), round(float(np.mean([c[1] for c in cis])), 4)],
        "n": int(len(y)),
    }


def _external_pooled(x_num_d, x_cat_d, y_d, cards, x_num_e, x_cat_e, y_e, seeds=_SEEDS) -> dict:
    y_e = np.asarray(y_e, int)
    aucs, cis, shufs = [], [], []
    rng = np.random.default_rng(_SHUFFLE_SEED)
    for seed in seeds:
        state = ce.fit_logreg_on_full_dev(x_num_d, x_cat_d, list(y_d), cards, seed)
        probs = ce.predict_logreg_with_state(state, x_num_e, x_cat_e)
        auc, lo, hi = delong_ci(y_e, probs)
        aucs.append(auc)
        cis.append((lo, hi))
        s_auc, _, _ = delong_ci(rng.permutation(y_e), probs)
        shufs.append(s_auc)
    return {
        "auroc": round(float(np.mean(aucs)), 4),
        "auroc_per_seed": {str(s): round(a, 4) for s, a in zip(seeds, aucs)},
        "ci95": [round(float(np.mean([c[0] for c in cis])), 4), round(float(np.mean([c[1] for c in cis])), 4)],
        "shuffle_auroc": round(float(np.mean(shufs)), 4),
        "n": int(len(y_e)),
    }


def run() -> dict:
    ce._assert_leak_free()  

    import yaml

    cfg = yaml.safe_load(CONFIG.read_text())

    x_num_d, x_cat_d, y_d, groups_d, cards = ce.load_xy(
        ROOT / cfg["duke_clinical_table"], ROOT / cfg["duke_split"]
    )
    duke_maps = gee._duke_factorize_maps(ROOT / cfg["duke_clinical_table"], ROOT / cfg["duke_split"])
    x_num_e, x_cat_e, y_e, meta = gee.build_external_matrix(cfg, duke_maps, cards)
    parity = gee.feature_parity_audit(x_num_e, x_cat_e, y_e, cfg)

    ext_npz = ROOT / "reports/G5_external/ispy2_external_probs.npz"
    cohort_match = None
    if ext_npz.exists():
        dz = np.load(ext_npz, allow_pickle=True)  
        npz_pids = {str(p) for p in dz["pids"]}
        built_pids = set(meta["pids"])
        cohort_match = {
            "npz_n": len(npz_pids),
            "built_n": len(built_pids),
            "pid_set_identical": bool(npz_pids == built_pids),
        }

    xn_d4, xc_d4 = x_num_d[:, NATIVE_NUM_IDX], x_cat_d[:, NATIVE_CAT_IDX]
    xn_e4, xc_e4 = x_num_e[:, NATIVE_NUM_IDX], x_cat_e[:, NATIVE_CAT_IDX]
    cards4 = [cards[i] for i in NATIVE_CAT_IDX]

    internal_4 = _internal_pooled(xn_d4, xc_d4, y_d, groups_d, cards4)
    external_4 = _external_pooled(xn_d4, xc_d4, y_d, cards4, xn_e4, xc_e4, y_e)

    internal_9 = _internal_pooled(x_num_d, x_cat_d, y_d, groups_d, cards)
    external_9 = _external_pooled(x_num_d, x_cat_d, y_d, cards, x_num_e, x_cat_e, y_e)

    published_9_external = None
    if NINE_FEATURE_METRICS.exists():
        published_9_external = json.loads(NINE_FEATURE_METRICS.read_text()).get("auroc")

    drop_4 = round(internal_4["auroc"] - external_4["auroc"], 4)
    drop_9 = round(internal_9["auroc"] - external_9["auroc"], 4)
    ref_9_ext = published_9_external if published_9_external is not None else external_9["auroc"]
    delta_vs_9feature_external = round(float(ref_9_ext) - external_4["auroc"], 4)

    confound_note = (
        f"4-feature-native internal->external drop = {drop_4} (internal {internal_4['auroc']} -> external "
        f"{external_4['auroc']}); 9-feature drop = {drop_9} (internal {internal_9['auroc']} -> external "
        f"{external_9['auroc']}). "
        + (
            "Removing the 5 imputed/absent features (incl. grade) SHRINKS the internal->external gap -> "
            "some of the external signal is honest, carried by the 4 natively-present features, not "
            "imputation noise."
            if drop_4 < drop_9
            else "Removing the 5 imputed/absent features does NOT shrink the internal->external gap -> "
            "the confound removal does not recover external signal; reported as-is."
        )
        + " Per-cohort robustness only (LOCK-1); no cross-institution generalisation claim. Grade + "
        "staging/metastatic/lymphadenopathy excluded (imputed/absent -- the isolated confound). LOCK-2: "
        "no IHC/receptor/Ki-67 feature entered any input. No LOCK moved."
    )

    out = {
        "gate": "G5-external-4feature-native (plan item 7)",
        "framing": "isolate the grade-imputation confound: 4 ISPY2-native features only; per-cohort robustness (LOCK-1 characterisation)",
        "estimator": "LogReg(C=1.0, max_iter=1000) + OneHot(handle_unknown='ignore') + median-impute + standardize (Duke-dev-fit ONLY) -- H6 coalition estimator",
        "features_4_native": _native_names(),
        "internal_auroc": internal_4["auroc"],
        "internal_ci95": internal_4["ci95"],
        "external_auroc": external_4["auroc"],
        "external_ci95": external_4["ci95"],
        "external_shuffle": external_4["shuffle_auroc"],
        "delta_vs_9feature_external": delta_vs_9feature_external,
        "internal_4feature_per_seed": internal_4["auroc_per_seed"],
        "external_4feature_per_seed": external_4["auroc_per_seed"],
        "internal_9feature_auroc_fresh": internal_9["auroc"],
        "external_9feature_auroc_fresh": external_9["auroc"],
        "published_9feature_external_auroc_ondisk": published_9_external,
        "drop_4feature_internal_to_external": drop_4,
        "drop_9feature_internal_to_external": drop_9,
        "confound_shrinks_external_gap": bool(drop_4 < drop_9),
        "n_duke_dev": internal_4["n"],
        "n_ispy2_external": external_4["n"],
        "ispy2_cohort_meta": {k: meta[k] for k in ("n_ispy2_balanced", "n_luma", "n_tnbc", "tnbc_prevalence")},
        "ispy2_cohort_matches_frozen_npz": cohort_match,
        "feature_parity_audit_9feature": parity,
        "seeds": list(_SEEDS),
        "shuffle_seed": _SHUFFLE_SEED,
        "compute": "cpu-zero-dollar",
        "lock2_note": "ce._assert_leak_free() passed; FORBIDDEN IHC/receptor/Ki-67/subtype fields excluded from all inputs; grade excluded as the isolated imputation confound.",
        "confound_note": confound_note,
    }

    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON.write_text(json.dumps(out, indent=2) + "\n")
    RESULTS_JSON_ALT.write_text(json.dumps(out, indent=2) + "\n")
    return out


def main() -> int:
    if not (ROOT / "data/mamma_mia/clinical_and_imaging_info.xlsx").exists():
        print("[g5-4feature] AWAITING DATA — data/mamma_mia/ not present (gitignored).")  
        return 0
    out = run()
    print(json.dumps(out, indent=2))  
    print(  
        f"\n[g5-4feature] 4-native internal {out['internal_auroc']} {out['internal_ci95']} -> external "
        f"{out['external_auroc']} {out['external_ci95']} (shuffle {out['external_shuffle']}); "
        f"drop_4={out['drop_4feature_internal_to_external']} vs drop_9={out['drop_9feature_internal_to_external']}; "
        f"delta_vs_9feature_external={out['delta_vs_9feature_external']}"
    )
    print(f"[g5-4feature] wrote {RESULTS_JSON.relative_to(ROOT)} + {RESULTS_JSON_ALT.name}")  
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

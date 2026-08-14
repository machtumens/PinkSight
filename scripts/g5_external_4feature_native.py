#!/usr/bin/env python3
"""G5 external — 4-feature ISPY2-NATIVE clinical LogReg (isolates the grade-imputation confound).

Plan item 7 (model-integrity-remediation_12-08-26). The published 9-feature external number (ISPY2
AUROC 0.5725) leans on Duke-train imputation for the 5 features ABSENT in ISPY2 (Nottingham grade + 2
staging + metastatic + lymphadenopathy). This script re-fits the SAME H6 coalition estimator
(LogReg(C=1.0, max_iter=1000) + OneHot(handle_unknown='ignore') + median-impute + standardize, all fit
on Duke-dev ONLY) restricted to the 4 features that are NATIVELY present in ISPY2 -- age, menopausal
status, race/ethnicity, multifocality -- so the internal->external drop carries NO imputation confound.

EXCLUDED on purpose (this is the point of the ablation):
  * Nottingham grade  (FEATURES_NUM[2]) -- imputed/absent in ISPY2; the confound being isolated.
  * T-staging, N-nodes (FEATURES_NUM[0,1]) -- absent in ISPY2, Duke-median-imputed in the 9-feat model.
  * Metastatic, Lymphadenopathy (FEATURES_CAT[3,4]) -- absent in ISPY2, OOV in the 9-feat model.
LOCK-2: NO IHC/receptor/Ki-67/subtype field is ever a feature (ce._assert_leak_free()); grade is also
excluded here (it is imputed/absent -- excluding it is the whole ablation). LOCK-1: per-cohort external
robustness / honesty pass -- report the drop; NEVER a cross-institution generalisation claim, never
kinetics/early-detection. $0-local, CPU only (LOCK-5).

Reuses scripts/g5_external_eval.py's data-loading (build_external_matrix / _duke_factorize_maps /
feature_parity_audit) and src/pinksight/models/clinical_encoder.py's LogReg helpers (logreg_oof /
fit_logreg_on_full_dev / predict_logreg_with_state) -- the 4-feature model is those helpers on the
column subset, so the estimator + the Duke-train-only discipline are byte-identical to the 9-feat model.

Seeded (LAW L-3): seeds [0,1,2] (LogReg is effectively deterministic; near-zero spread reported).

Run:  uv run python scripts/g5_external_4feature_native.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import g5_external_eval as gee  # noqa: E402  (build_external_matrix / _duke_factorize_maps / parity)
from pinksight.metrics import delong_ci  # noqa: E402
from pinksight.models import clinical_encoder as ce  # noqa: E402

CONFIG = ROOT / "configs" / "g5_external_ispy2.yaml"
NINE_FEATURE_METRICS = ROOT / "reports" / "G5_external" / "metrics.json"
# Plan checklist names external_4feature_native_metrics.json; the EXECUTE handoff also names
# ispy2_4feature_matched.json. Write BOTH (identical content).
RESULTS_JSON = ROOT / "reports" / "G5_external" / "external_4feature_native_metrics.json"
RESULTS_JSON_ALT = ROOT / "reports" / "G5_external" / "ispy2_4feature_matched.json"

# The 4 ISPY2-native features, as column indices into ce.FEATURES_NUM / ce.FEATURES_CAT.
NATIVE_NUM_IDX = [3]        # Age at diagnosis (years)
NATIVE_CAT_IDX = [0, 1, 2]  # Menopause, Race and Ethnicity, Multicentric/Multifocal
EXCLUDED_NUM_IDX = [0, 1, 2]  # T-staging, N-nodes, Nottingham grade (grade = the isolated confound)
EXCLUDED_CAT_IDX = [3, 4]     # Metastatic, Lymphadenopathy

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
    """Multi-seed pooled-OOF internal AUROC + mean DeLong CI for the given feature columns."""
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
    """Fit on FULL Duke-dev per seed, apply once to ISPY2, pooled external AUROC + CI + shuffle."""
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
    ce._assert_leak_free()  # LOCK-2: forbidden IHC/receptor fields never in FEATURES

    import yaml

    cfg = yaml.safe_load(CONFIG.read_text())

    # 1) Duke-dev full 9-feature matrix (leak-guarded on load).
    x_num_d, x_cat_d, y_d, groups_d, cards = ce.load_xy(
        ROOT / cfg["duke_clinical_table"], ROOT / cfg["duke_split"]
    )
    # 2) ISPY2 external 9-feature matrix via the parity crosswalk (Duke code space).
    duke_maps = gee._duke_factorize_maps(ROOT / cfg["duke_clinical_table"], ROOT / cfg["duke_split"])
    x_num_e, x_cat_e, y_e, meta = gee.build_external_matrix(cfg, duke_maps, cards)
    # 3) Read-only feature-parity audit of the 9-feature external matrix (E1 confirmation: the 5
    #    ISPY2-absent features are Duke-median-imputed (numeric) / OOV (categorical), not silent 0).
    parity = gee.feature_parity_audit(x_num_e, x_cat_e, y_e, cfg)

    # 4) Cross-check the reconstructed ISPY2 cohort against the frozen external probs npz.
    ext_npz = ROOT / "reports/G5_external/ispy2_external_probs.npz"
    cohort_match = None
    if ext_npz.exists():
        dz = np.load(ext_npz, allow_pickle=True)  # pids is an object array
        npz_pids = {str(p) for p in dz["pids"]}
        built_pids = set(meta["pids"])
        cohort_match = {
            "npz_n": len(npz_pids),
            "built_n": len(built_pids),
            "pid_set_identical": bool(npz_pids == built_pids),
        }

    # 5) Feature slices.
    xn_d4, xc_d4 = x_num_d[:, NATIVE_NUM_IDX], x_cat_d[:, NATIVE_CAT_IDX]
    xn_e4, xc_e4 = x_num_e[:, NATIVE_NUM_IDX], x_cat_e[:, NATIVE_CAT_IDX]
    cards4 = [cards[i] for i in NATIVE_CAT_IDX]

    # 6) 4-feature-native internal (Duke dev pooled OOF) + external (ISPY2).
    internal_4 = _internal_pooled(xn_d4, xc_d4, y_d, groups_d, cards4)
    external_4 = _external_pooled(xn_d4, xc_d4, y_d, cards4, xn_e4, xc_e4, y_e)

    # 7) Fresh 9-feature internal + external in the SAME run (apples-to-apples comparison basis), and
    #    cross-check against the on-disk published 9-feature number.
    internal_9 = _internal_pooled(x_num_d, x_cat_d, y_d, groups_d, cards)
    external_9 = _external_pooled(x_num_d, x_cat_d, y_d, cards, x_num_e, x_cat_e, y_e)

    published_9_external = None
    if NINE_FEATURE_METRICS.exists():
        published_9_external = json.loads(NINE_FEATURE_METRICS.read_text()).get("auroc")

    drop_4 = round(internal_4["auroc"] - external_4["auroc"], 4)
    drop_9 = round(internal_9["auroc"] - external_9["auroc"], 4)
    # delta_vs_9feature_external per the plan: published 9-feature external AUROC minus this 4-feature.
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
        # ---- the plan's required output fields ----
        "internal_auroc": internal_4["auroc"],
        "internal_ci95": internal_4["ci95"],
        "external_auroc": external_4["auroc"],
        "external_ci95": external_4["ci95"],
        "external_shuffle": external_4["shuffle_auroc"],
        "delta_vs_9feature_external": delta_vs_9feature_external,
        # ---- honest context (LAW L-2 multi-seed + confound isolation) ----
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
        print("[g5-4feature] AWAITING DATA — data/mamma_mia/ not present (gitignored).")  # noqa: T201
        return 0
    out = run()
    print(json.dumps(out, indent=2))  # noqa: T201
    print(  # noqa: T201
        f"\n[g5-4feature] 4-native internal {out['internal_auroc']} {out['internal_ci95']} -> external "
        f"{out['external_auroc']} {out['external_ci95']} (shuffle {out['external_shuffle']}); "
        f"drop_4={out['drop_4feature_internal_to_external']} vs drop_9={out['drop_9feature_internal_to_external']}; "
        f"delta_vs_9feature_external={out['delta_vs_9feature_external']}"
    )
    print(f"[g5-4feature] wrote {RESULTS_JSON.relative_to(ROOT)} + {RESULTS_JSON_ALT.name}")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

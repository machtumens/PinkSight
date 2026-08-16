from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from pinksight import FORBIDDEN_FEATURES  
from pinksight.metrics import delong_ci  
from pinksight.models import clinical_encoder as ce  


_MENO_EXT_TO_DUKE = {
    "pre": 0,
    "post": 1,
    "peri": 2,
}
_ETH_EXT_TO_DUKE = {
    "caucasian": 1,
    "african american": 2,
    "asian": 3,
    "american indian/alaskan native": 4,
    "hispanic": 5,
    "multiple race": 6,
    "hawaiian/pacific islander": 7,
}


def _norm_meno(v) -> float:
    if not isinstance(v, str):
        return np.nan
    s = v.strip().lower()
    for pref, code in _MENO_EXT_TO_DUKE.items():
        if s.startswith(pref):
            return float(code)
    return np.nan


def _norm_eth(v) -> float:
    if not isinstance(v, str):
        return np.nan
    return float(_ETH_EXT_TO_DUKE.get(v.strip().lower(), np.nan))


def _duke_factorize_maps(clin_path: Path, split_yaml: Path) -> dict[str, dict[float, int]]:
    from audit_ki67 import load as load_clinical

    df = load_clinical(clin_path)
    df.columns = [str(c) for c in df.columns]
    dev = set(yaml.safe_load(split_yaml.read_text())["dev"])

    def col(name):
        s = df[name]
        return s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s

    pid = col("Patient ID").astype(str).str.strip()
    subtype = pd.to_numeric(col("Mol Subtype"), errors="coerce").map(ce._LABEL)
    keep = pid.isin(dev) & subtype.notna()  

    maps: dict[str, dict[float, int]] = {}
    for c in ce.FEATURES_CAT:
        series = pd.to_numeric(col(c), errors="coerce")[keep]
        _codes, uniq = pd.factorize(series, use_na_sentinel=False)
        maps[c] = {float(level): (k + 1) for k, level in enumerate(uniq) if pd.notna(level)}
    return maps


def _read_h6_clinical_anchor() -> dict:
    p = ROOT / "reports/G2_imaging/MODALITY_AUDIT/metrics.json"
    if not p.exists():
        return {"auroc": None, "ci95": None, "n": None, "source": "H6 metrics absent"}
    try:
        d = json.loads(p.read_text())
        clin = d.get("coalitions", {}).get("clinical", {})
        return {
            "auroc": clin.get("auroc_mean"),
            "ci95": clin.get("delong_ci95_seed0"),
            "n": d.get("n_intersection"),
            "source": "reports/G2_imaging/MODALITY_AUDIT/metrics.json :: coalitions.clinical (H6 ablation-ladder LogReg)",
        }
    except (json.JSONDecodeError, KeyError):  
        return {"auroc": None, "ci95": None, "n": None, "source": "H6 metrics unreadable"}


def build_external_matrix(cfg: dict, duke_maps: dict, cards: list[int]):
    m = pd.read_excel(ROOT / cfg["external_table"])
    m["_pid"] = m["patient_id"].astype(str)
    clean = {ln.strip() for ln in (ROOT / cfg["clean_ids"]).read_text().splitlines() if ln.strip()}
    coh = m[(m["dataset"] == cfg["cohort"]) & (m["_pid"].isin(clean))].copy()

    label_map = cfg["label_map"]
    sub = coh[cfg["subtype_col"]].astype(str).str.strip().str.lower()
    in_contrast = sub.isin(label_map.keys())
    coh, sub = coh[in_contrast], sub[in_contrast]
    y = sub.map(label_map).astype(int).to_numpy()

    n = len(coh)
    def get_num(col_name):
        return pd.to_numeric(coh[col_name], errors="coerce").to_numpy(float) if col_name in coh else np.full(n, np.nan)

    x_num = np.column_stack([
        np.full(n, np.nan),                         
        np.full(n, np.nan),                         
        get_num("nottingham_grade"),                
        get_num("age"),                             
    ])

    meno_duke = coh["menopause"].map(_norm_meno) if "menopause" in coh else pd.Series(np.nan, index=coh.index)
    eth_duke = coh["ethnicity"].map(_norm_eth) if "ethnicity" in coh else pd.Series(np.nan, index=coh.index)
    multi_duke = pd.to_numeric(coh["multifocal_cancer"], errors="coerce") if "multifocal_cancer" in coh else pd.Series(np.nan, index=coh.index)

    def to_code(duke_level_series, feat_name):
        mp = duke_maps[feat_name]
        return np.array([mp.get(float(v), 0) if pd.notna(v) else 0 for v in duke_level_series], dtype=int)

    x_cat = np.column_stack([
        to_code(meno_duke, "Menopause (at diagnosis)"),
        to_code(eth_duke, "Race and Ethnicity"),
        to_code(multi_duke, "Multicentric/Multifocal"),
        np.zeros(n, dtype=int),   
        np.zeros(n, dtype=int),   
    ])

    meta = {
        "n_ispy2_balanced": int(n),
        "n_luma": int((y == 0).sum()),
        "n_tnbc": int((y == 1).sum()),
        "tnbc_prevalence": round(float(y.mean()), 4),
        "pids": [str(p) for p in coh["_pid"].tolist()],
    }
    return x_num, x_cat, y, meta


def feature_parity_audit(x_num, x_cat, y, cfg) -> dict:
    num_names = list(ce.FEATURES_NUM)
    cat_names = list(ce.FEATURES_CAT)
    audit = {"numeric": {}, "categorical": {}, "forbidden_features_excluded": sorted(FORBIDDEN_FEATURES)}
    for j, name in enumerate(num_names):
        present = int(np.isfinite(x_num[:, j]).sum())
        audit["numeric"][name] = {
            "external_non_null": present,
            "of": int(len(y)),
            "imputed_with": "duke_train_median" if present < len(y) else "none",
        }
    for j, name in enumerate(cat_names):
        mapped = int((x_cat[:, j] > 0).sum())
        audit["categorical"][name] = {
            "external_mapped_to_duke_code": mapped,
            "of": int(len(y)),
            "oov_or_absent": int((x_cat[:, j] == 0).sum()),
        }
    audit["features_mappable"] = sum(
        1 for j, _ in enumerate(num_names) if np.isfinite(x_num[:, j]).any()
    ) + sum(1 for j, _ in enumerate(cat_names) if (x_cat[:, j] > 0).any())
    audit["features_total"] = len(num_names) + len(cat_names)
    return audit


def run(config_path: Path) -> dict:
    cfg = yaml.safe_load(config_path.read_text())

    ce._assert_leak_free()  
    x_num_d, x_cat_d, y_d, groups_d, cards = ce.load_xy(
        ROOT / cfg["duke_clinical_table"], ROOT / cfg["duke_split"]
    )
    internal = ce.logreg_cross_val_auroc(x_num_d, x_cat_d, y_d, groups_d, cards)
    internal_auroc = internal["auroc_pooled_oof_mean"]        
    h6_anchor = _read_h6_clinical_anchor()

    stats_path = ROOT / cfg["impute_stats_artifact"]
    if not stats_path.exists():
        ce.save_impute_stats(x_num_d, x_cat_d, groups_d, cards, stats_path)

    duke_maps = _duke_factorize_maps(ROOT / cfg["duke_clinical_table"], ROOT / cfg["duke_split"])
    x_num_e, x_cat_e, y_e, meta = build_external_matrix(cfg, duke_maps, cards)
    parity = feature_parity_audit(x_num_e, x_cat_e, y_e, cfg)

    seeds = cfg.get("seeds", [0, 1, 2])
    ext_aucs, ext_cis, shuffle_aucs = {}, {}, {}
    rng = np.random.default_rng(cfg.get("shuffle_seed", 0))
    probs_by_seed = {}
    for seed in seeds:
        state = ce.fit_logreg_on_full_dev(x_num_d, x_cat_d, y_d, cards, seed)
        probs = ce.predict_logreg_with_state(state, x_num_e, x_cat_e)
        probs_by_seed[str(seed)] = probs.tolist()
        auc, lo, hi = delong_ci(y_e, probs)
        ext_aucs[str(seed)], ext_cis[str(seed)] = round(auc, 4), [round(lo, 4), round(hi, 4)]
        y_shuf = rng.permutation(y_e)
        s_auc, _, _ = delong_ci(y_shuf, probs)
        shuffle_aucs[str(seed)] = round(s_auc, 4)

    ext_mean = float(np.mean(list(ext_aucs.values())))
    shuffle_mean = float(np.mean(list(shuffle_aucs.values())))
    ci_lo = round(float(np.mean([c[0] for c in ext_cis.values()])), 4)
    ci_hi = round(float(np.mean([c[1] for c in ext_cis.values()])), 4)

    sanity = _tnbc_only_sanity(cfg, duke_maps, cards, seeds)

    out = {
        "gate": "G5-leg1",
        "cohort": cfg["cohort"],
        "framing": "per-cohort external robustness / honesty pass — report the drop; NOT a cross-institution generalisation claim (LOCK-1)",
        "auroc": round(ext_mean, 4),                       
        "auroc_per_seed": ext_aucs,
        "delong_ci": [ci_lo, ci_hi],                       
        "delong_ci_per_seed": ext_cis,
        "internal_auroc_logreg_full_dev_pooled_oof": round(float(internal_auroc), 4),  
        "internal_auroc_logreg_full_dev_ci95_mean": internal["delong_ci95_mean"],
        "internal_auroc_logreg_full_dev_per_seed": internal["auroc_pooled_oof_per_seed"],
        "internal_auroc_h6_anchor_n613_intersection": h6_anchor["auroc"],  
        "internal_auroc_h6_anchor_ci95": h6_anchor["ci95"],
        "internal_auroc_h6_anchor_n": h6_anchor["n"],
        "delta_internal_external": round(float(internal_auroc) - ext_mean, 4),  
        "delta_internal_external_h6_anchor_basis": round(float(h6_anchor["auroc"]) - ext_mean, 4) if h6_anchor["auroc"] is not None else None,
        "internal_estimator_note": (
            "ESTIMATOR = H6 coalition LogReg(C=1.0, max_iter=1000) + OneHot(handle_unknown=ignore) — "
            "the ablation-ladder estimator that produces the 0.708 clinical-subtype headline. PRIMARY Δ "
            "= LogReg pooled-OOF internal on FULL Duke-dev (N=624, estimator-consistent with the single "
            "pooled external DeLong AUROC over 739). N nuance (plan requirement): the 0.708 clinical-subtype "
            "anchor is the SAME LogReg on the N=613 3-way (clinical∩radiomics∩MRI) imaging intersection "
            "(reports/G2_imaging/MODALITY_AUDIT/metrics.json); the full-dev value differs only by cohort "
            "N. Both Δ bases are reported. This REPLACES the prior FTT G5 leg (internal ~0.634 -> ext "
            "0.525) which mismatched the LogReg headline (apples-to-oranges)."
        ),
        "shuffle_auroc": round(shuffle_mean, 4),           
        "shuffle_auroc_per_seed": shuffle_aucs,
        "seeds": list(seeds),
        "n_ispy2": meta,
        "feature_parity_audit": parity,
        "tnbc_only_sanity_separate": sanity,
        "model": "clinical LogReg(C=1.0, max_iter=1000) + OneHot(handle_unknown=ignore) [H6 coalition estimator], re-fit on full Duke-dev (E2: no saved checkpoint; deterministic re-fit)",
        "prior_leg_estimator": "FT-Transformer (n_blocks=2) — SUPERSEDED by this LogReg re-run (25-07-26); the FTT internal ~0.634 mismatched the 0.708 LogReg headline",
        "impute_stats_artifact": str(stats_path.relative_to(ROOT)),
        "impute_stats_source": "duke_dev_full (LOCK-2: no external stats fit)",
        "integrity_note": (
            f"{parity['features_mappable']}/{parity['features_total']} Duke features have any ISPY2 "
            "signal; the remaining features are Duke-train-imputed (numeric) or OOV (categorical). "
            "The external number therefore leans on the mappable features (age, menopause, ethnicity, "
            "multifocal); Nottingham grade + staging/nodal/metastatic fields are absent in ISPY2. "
            "This is honest per-cohort robustness, not a generalisation claim."
        ),
    }

    out_path = ROOT / cfg["out_metrics"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    np.savez(out_path.parent / "ispy2_external_probs.npz",
             pids=np.array(meta["pids"], dtype=object), y=y_e,
             **{f"probs_seed{s}": np.asarray(p) for s, p in probs_by_seed.items()})
    return out


def _tnbc_only_sanity(cfg, duke_maps, cards, seeds) -> dict:
    m = pd.read_excel(ROOT / cfg["external_table"])
    m["_pid"] = m["patient_id"].astype(str)
    clean = {ln.strip() for ln in (ROOT / cfg["clean_ids"]).read_text().splitlines() if ln.strip()}
    x_num_d, x_cat_d, y_d, _g, _c = ce.load_xy(ROOT / cfg["duke_clinical_table"], ROOT / cfg["duke_split"])
    state = ce.fit_logreg_on_full_dev(x_num_d, x_cat_d, y_d, cards, 0)
    notes = {}
    for ds in cfg.get("sanity_cohorts_tnbc_only", []):
        d = m[(m["dataset"] == ds) & (m["_pid"].isin(clean))].copy()
        sub = d[cfg["subtype_col"]].astype(str).str.strip().str.lower()
        d = d[sub == "triple_negative"]
        if len(d) == 0:
            notes[ds] = {"n_tnbc": 0}
            continue
        cfg_local = dict(cfg, cohort=ds)
        xn, xc, yy, meta = build_external_matrix(cfg_local, duke_maps, cards)
        probs = ce.predict_logreg_with_state(state, xn, xc)
        notes[ds] = {
            "n_tnbc": int(len(yy)),
            "mean_p_tnbc": round(float(np.mean(probs)), 4),
            "note": "TNBC-only (no LumA) — directional mean P(TNBC), NOT an AUROC; NEVER pooled (LOCK-2 cohort-shortcut guard).",
        }
    return notes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=ROOT / "configs/g5_external_ispy2.yaml")
    args = ap.parse_args()
    if not (ROOT / "data/mamma_mia/clinical_and_imaging_info.xlsx").exists():
        print("[g5-external] AWAITING DATA — data/mamma_mia/ not present (gitignored).")  
        return 0
    out = run(args.config)
    print(json.dumps(out, indent=2))  
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

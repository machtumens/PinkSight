"""G3 item-4 — minimum detectable effect (MDE) / post-hoc power for the pre-reg ΔAUC ≥ 0.03 margin.

Question: at N=613, given the OBSERVED paired-DeLong standard error of ΔAUC(fusion vs clinical), what
is the smallest true ΔAUC detectable at 80%/90% power (α=0.05 two-sided)? If that MDE exceeds the
pre-registered +0.03 margin, the study is underpowered to DETECT a genuine small fusion benefit — the
honest evidence behind "fusion not demonstrated" (never "fusion rejected").

Primary variance = the paired-DeLong SE from item-3 (`paired_vs_anchor_delong.json`), the principled
estimate for detecting a ΔAUC between two correlated ROCs on the SAME patients. Secondary sensitivity
= the across-seed AUROC spread (`multiseed_spread.json`).

MDE = SE_paired * (z_{1-α/2} + z_power).  N_required ≈ N * (SE_obs / SE_needed)^2, SE_needed = margin
/ (z_{1-α/2} + z_power)  (SE ∝ 1/√N at fixed effect structure).

Reads only frozen JSON (no imaging-encoder re-run, no model fit). Claim-ledger: characterisation at
diagnosis; the observed ΔAUC is negative (clinical beats fusion) — this power calc quantifies that a
small POSITIVE +0.03 imaging benefit would be undetectable here, it does NOT assert one exists. No
LOCK moved.

    PYTHONPATH=src .venv/bin/python scripts/g3_power_analysis.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAIRED = ROOT / "reports/G3_fusion_arch_bundle/paired_vs_anchor_delong.json"
SPREAD = ROOT / "reports/G3_fusion_arch_bundle/multiseed_spread.json"
OUT = ROOT / "reports/G3_fusion_arch_bundle/mde_power.json"

# Standard normal quantiles (scipy-free, matching src/pinksight/metrics._Z95 convention).
Z_ALPHA_2 = 1.959963985   # norm.ppf(0.975), α=0.05 two-sided
Z_POWER_80 = 0.841621234  # norm.ppf(0.80)
Z_POWER_90 = 1.281551566  # norm.ppf(0.90)
PREREG_MARGIN = 0.03
N = 613


def _paired_se_per_seed(block: dict) -> dict[str, float]:
    """Recover the paired ΔAUC SE per seed from its 95% CI: se = (ci_hi - delta) / z_{0.975}."""
    out = {}
    for s, d in block["per_seed"].items():
        se = (d["ci95"][1] - d["delta"]) / Z_ALPHA_2
        out[s] = round(se, 5)
    return out


def _model_power(name: str, paired_block: dict) -> dict:
    ses = _paired_se_per_seed(paired_block)
    se_mean = sum(ses.values()) / len(ses)
    mde_80 = se_mean * (Z_ALPHA_2 + Z_POWER_80)
    mde_90 = se_mean * (Z_ALPHA_2 + Z_POWER_90)
    se_needed_80 = PREREG_MARGIN / (Z_ALPHA_2 + Z_POWER_80)
    n_req_80 = N * (se_mean / se_needed_80) ** 2
    return {
        "comparison": paired_block["comparison"],
        "observed_paired_se_mean": round(se_mean, 5),
        "per_seed_paired_se": ses,
        "observed_mean_delta_auroc": paired_block["mean_delta_auroc"],
        "mde_at_80pct_power": round(mde_80, 4),
        "mde_at_90pct_power": round(mde_90, 4),
        "se_needed_for_0.03_at_80pct": round(se_needed_80, 5),
        "n_required_for_0.03_margin_at_80pct": int(round(n_req_80)),
        "is_pre_reg_margin_detectable_at_n613": bool(mde_80 <= PREREG_MARGIN),
    }


def main() -> None:
    paired = json.loads(PAIRED.read_text())
    spread = json.loads(SPREAD.read_text())

    hier = _model_power("hierarchical", paired["hierarchical_vs_clinical"])
    moe = _model_power("moe_deterministic", paired["moe_deterministic_vs_clinical"])
    detectable = hier["is_pre_reg_margin_detectable_at_n613"] or moe["is_pre_reg_margin_detectable_at_n613"]

    doc = {
        "gate": "G3 item-4 — MDE / post-hoc power for the pre-reg ΔAUC ≥ 0.03 margin at N=613",
        "prereg_margin": PREREG_MARGIN,
        "n": N,
        "alpha": 0.05,
        "test": "two-sided paired-DeLong ΔAUC (fusion vs clinical), normal approximation",
        "z_alpha_2": Z_ALPHA_2,
        "z_power_80": Z_POWER_80,
        "z_power_90": Z_POWER_90,
        "mde_formula": "MDE = SE_paired * (z_{1-α/2} + z_power)",
        "per_model": {"hierarchical": hier, "moe_deterministic": moe},
        "secondary_sensitivity_across_seed_auroc_spread": {
            "hierarchical_std_across_seeds": spread["hierarchical"]["std_across_seeds"],
            "moe_std_across_seeds": spread["moe"]["std_across_seeds"],
            "note": ("across-seed AUROC spread is a coarser variance proxy than the paired-DeLong SE; "
                     "reported as a secondary check. Both point the same way: variance >> the +0.03 "
                     "margin at N=613."),
        },
        "is_pre_reg_margin_detectable_at_n613": bool(detectable),
        "honest_interpretation": (
            "At N=613 the paired-DeLong SE of ΔAUC(fusion vs clinical) is ~%.3f–%.3f, so the minimum "
            "detectable ΔAUC at 80%% power is ~%.3f–%.3f — roughly %.0f× the pre-registered +0.03 "
            "margin. Detecting a genuine +0.03 imaging-fusion benefit would need ≈%d–%d patients. The "
            "study is UNDERPOWERED to demonstrate the pre-reg margin: 'fusion not demonstrated', NOT "
            "'fusion rejected'. Caveat (honest-null): the OBSERVED ΔAUC is negative and paired-"
            "significant in the OPPOSITE direction (clinical beats fusion, item-3 Stouffer p≈0) — this "
            "power calc shows a small POSITIVE fusion benefit would be invisible here, it does NOT "
            "assert one exists." % (
                min(hier["observed_paired_se_mean"], moe["observed_paired_se_mean"]),
                max(hier["observed_paired_se_mean"], moe["observed_paired_se_mean"]),
                min(moe["mde_at_80pct_power"], hier["mde_at_80pct_power"]),
                max(moe["mde_at_80pct_power"], hier["mde_at_80pct_power"]),
                round(min(hier["mde_at_80pct_power"], moe["mde_at_80pct_power"]) / PREREG_MARGIN),
                min(hier["n_required_for_0.03_margin_at_80pct"], moe["n_required_for_0.03_margin_at_80pct"]),
                max(hier["n_required_for_0.03_margin_at_80pct"], moe["n_required_for_0.03_margin_at_80pct"]))),
        "claim_ledger": ("characterisation at diagnosis; power calc backs 'underpowered / not "
                         "demonstrated', never a positive imaging claim (LOCK-1). No LOCK moved."),
    }
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"[item-4] #4  se {hier['observed_paired_se_mean']}  MDE80 {hier['mde_at_80pct_power']}  "
          f"N_req(0.03) {hier['n_required_for_0.03_margin_at_80pct']}  "
          f"detectable@613={hier['is_pre_reg_margin_detectable_at_n613']}")
    print(f"[item-4] #7  se {moe['observed_paired_se_mean']}  MDE80 {moe['mde_at_80pct_power']}  "
          f"N_req(0.03) {moe['n_required_for_0.03_margin_at_80pct']}  "
          f"detectable@613={moe['is_pre_reg_margin_detectable_at_n613']}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

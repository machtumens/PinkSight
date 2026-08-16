from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pinksight.eval.calibration import (  
    apply_temperature,
    fit_temperature,
    reliability_curve,
    smooth_ece,
)
from pinksight.metrics import ece  
from pinksight.models import clinical_encoder as ce  

OUT_DIR = ROOT / "reports" / "G5_calibration"
OOF_PROBS_NPY = OUT_DIR / "oof_probs.npy"       
RELIABILITY_PNG = OUT_DIR / "reliability.png"
RESULTS_JSON = OUT_DIR / "metrics.json"

_EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, float), _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


def clinical_oof_probs(clin_path: Path, split_yaml: Path):
    x_num, x_cat, y, groups, cards = ce.load_xy(clin_path, split_yaml)
    seed = 0  
    oof = ce.logreg_oof(x_num, x_cat, y, groups, cards, seed)
    return [str(g) for g in groups], np.asarray(list(y), int), oof


def _cal_block(y, prob, temperature) -> dict:
    p_after = apply_temperature(_logit(prob), temperature)
    return {
        "n": int(len(y)),
        "prevalence": round(float(np.mean(y)), 4),
        "ece_before": round(ece(y, prob), 4),
        "ece_after": round(ece(y, p_after), 4),
        "smooth_ece_before": round(smooth_ece(y, prob), 4),
        "smooth_ece_after": round(smooth_ece(y, p_after), 4),
        "reliability_before": reliability_curve(y, prob, 10),
        "reliability_after": reliability_curve(y, p_after, 10),
    }


def _plot(clin_block, ext_block, temperature) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  
        return False

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, block, title in (
        (axes[0], clin_block, "Clinical-alone (Duke OOF)"),
        (axes[1], ext_block, "ISPY2 external"),
    ):
        ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="perfect")
        for rows, style, lbl in (
            (block["reliability_before"], "o-", f"pre-T (ECE {block['ece_before']})"),
            (block["reliability_after"], "s-", f"post-T (ECE {block['ece_after']})"),
        ):
            if rows:
                conf = [r["conf"] for r in rows]
                acc = [r["acc"] for r in rows]
                ax.plot(conf, acc, style, lw=1.5, ms=5, label=lbl)
        ax.set_title(title)
        ax.set_xlabel("mean predicted P(TNBC)")
        ax.set_ylabel("empirical TNBC rate")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.3)
    fig.suptitle(
        f"G5 calibration — temperature T={temperature:.3f} (fit on Duke OOF / val-equivalent only)",
        fontsize=11,
    )
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(RELIABILITY_PNG, dpi=120)
    plt.close(fig)
    return True


def run() -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pids, y_clin, oof = clinical_oof_probs(
        ROOT / "data/raw/Clinical_and_Other_Features.xlsx", ROOT / "configs/split_v2.yaml"
    )
    np.save(OOF_PROBS_NPY, oof)
    np.savez(OUT_DIR / "oof_probs_full.npz", pids=np.array(pids, dtype=object), y=y_clin, oof=oof)

    temperature = fit_temperature(_logit(oof), y_clin.astype(float))

    clin_block = _cal_block(y_clin, oof, temperature)

    ext_block = None
    ext_npz = ROOT / "reports/G5_external/ispy2_external_probs.npz"
    if ext_npz.exists():
        d = np.load(ext_npz, allow_pickle=True)
        y_ext = np.asarray(d["y"], int)
        key = "probs_seed0" if "probs_seed0" in d.files else next(k for k in d.files if k.startswith("probs_seed"))
        p_ext = np.asarray(d[key], float)
        ext_block = _cal_block(y_ext, p_ext, temperature)

    plotted = _plot(clin_block, ext_block or {"reliability_before": [], "reliability_after": [], "ece_before": None, "ece_after": None}, temperature)

    out = {
        "gate": "G5-leg2",
        "framing": "calibration (ECE + reliability) of the characterised clinical-subtype model; LOCK-1 characterisation only",
        "temperature": round(float(temperature), 4),
        "temperature_fit_on": "duke_oof_val_equivalent_only (E4/LOCK-2: never test/external)",
        "targets": {"good": 0.05, "acceptable": 0.10},
        "clinical_alone_headline": clin_block,
        "ispy2_external": ext_block if ext_block is not None else {"status": "external probs npz absent — run g5_external_eval first"},
        "oof_probs_saved": str(OOF_PROBS_NPY.relative_to(ROOT)),
        "reliability_png": str(RELIABILITY_PNG.relative_to(ROOT)) if plotted else "matplotlib absent — PNG skipped",
        "estimator": "LogReg(C=1.0, max_iter=1000) + OneHot(handle_unknown=ignore) — H6 coalition estimator (0.708 headline)",
        "prior_leg_estimator": "FT-Transformer (n_blocks=2) — SUPERSEDED by this LogReg re-run (25-07-26)",
        "note": (
            "G3 known-gap closed: per-patient OOF probabilities saved to disk this time. ESTIMATOR is "
            "the H6 coalition LogReg (0.708 headline), NOT the FTT (prior leg). Temperature scaling fit "
            "on the Duke CV held-out (out-of-fold = val-equivalent) LogReg predictions only; the ISPY2 "
            "external block applies the SAME T (never re-fit on external). ECE reported as actual — not "
            "forced to the target. Characterisation framing only (LOCK-1)."
        ),
    }
    RESULTS_JSON.write_text(json.dumps(out, indent=2) + "\n")
    return out


def main() -> int:
    if not (ROOT / "data/raw/Clinical_and_Other_Features.xlsx").exists():
        print("[g5-calibration] AWAITING DATA — data/raw/ clinical table not present (gitignored).")  
        return 0
    out = run()
    print(json.dumps(out, indent=2))  
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

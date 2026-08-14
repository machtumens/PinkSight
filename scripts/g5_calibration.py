#!/usr/bin/env python3
"""G5 LEG-2 — calibration of the clinical-subtype model + ISPY2 external ($0, CPU).

ESTIMATOR (G5 re-run 25-07-26): the H6 COALITION LogReg (the 0.708 clinical-subtype headline
estimator), NOT the FT-Transformer. The clinical OOF comes from clinical_encoder.logreg_oof and the
ISPY2 external probs from the LogReg re-run (g5_external_eval). The prior calibration leg calibrated
the FTT logits — apples-to-oranges under a LogReg headline. Temperature scaling operates on logit(p)
= inverse-sigmoid of the LogReg predict_proba output (a valid post-hoc scalar over any probability
vector); LOCK-2/E4 unchanged (T fit on Duke OOF / val-equivalent ONLY).

WHAT: reliability diagram + ECE (pre/post temperature scaling) for (a) the clinical-alone headline
(the Duke LogReg's per-patient out-of-fold probabilities) and (b) the ISPY2 external predictions.
Closes the G3 known-gap by SAVING per-patient OOF probabilities to disk this time
(reports/G5_calibration/oof_probs.npy) so a temperature-scaled ECE is recoverable.

TEMPERATURE SCALING (E4 / LOCK-2): T is fit on the VAL fold of configs/split_v2.yaml ONLY — never on
the reported test fold or the external cohort. split_v2 has no separate `val` list (only dev/test);
the leakage-safe val-equivalent is the CROSS-VALIDATION HELD-OUT (out-of-fold) predictions: each dev
patient is scored by a model that never trained on that patient, so the pooled-OOF predictions ARE
held-out ("validation") predictions. T is fit on those OOF predictions and then applied — never fit
on test/external. Reuses pinksight.eval.calibration (fit_temperature/apply_temperature/reliability).

COMPUTE: CPU only, $0. No GPU / RunPod / Kaggle (LOCK-5). No FORBIDDEN_FEATURES enter anything —
temperature scaling is a post-hoc scalar over existing logits and adds no features (LOCK-2).

Run:  uv run python scripts/g5_calibration.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pinksight.eval.calibration import (  # noqa: E402
    apply_temperature,
    fit_temperature,
    reliability_curve,
    smooth_ece,
)
from pinksight.metrics import ece  # noqa: E402
from pinksight.models import clinical_encoder as ce  # noqa: E402

OUT_DIR = ROOT / "reports" / "G5_calibration"
OOF_PROBS_NPY = OUT_DIR / "oof_probs.npy"       # L2-C9: the G3 known-gap closer
RELIABILITY_PNG = OUT_DIR / "reliability.png"
RESULTS_JSON = OUT_DIR / "metrics.json"

_EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    """Exact inverse-sigmoid: OOF/external probs come from sigmoid(logits), so logit(p) recovers the
    logit temperature scaling operates on (clip to avoid ±inf at 0/1)."""
    p = np.clip(np.asarray(p, float), _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


def clinical_oof_probs(clin_path: Path, split_yaml: Path):
    """Per-patient pooled-OOF probabilities for the Duke clinical model (seed 0 — deterministic).

    ESTIMATOR (G5 re-run 25-07-26): the H6 COALITION LogReg (clinical_encoder.logreg_oof) — the
    ablation-ladder estimator behind the 0.708 clinical-subtype headline, NOT the FT-Transformer. The
    prior calibration leg used the FTT OOF (via ce._fit_eval_fold); calibrating the FTT logits under a
    LogReg headline was apples-to-oranges. logreg_oof runs the SAME patient-grouped StratifiedGroupKFold(5)
    (each dev patient scored by a model that never trained on them — the val-equivalent). Returns
    (pids, y, oof)."""
    x_num, x_cat, y, groups, cards = ce.load_xy(clin_path, split_yaml)
    seed = 0  # deterministic single-seed OOF for the headline calibration curve
    oof = ce.logreg_oof(x_num, x_cat, y, groups, cards, seed)
    return [str(g) for g in groups], np.asarray(list(y), int), oof


def _cal_block(y, prob, temperature) -> dict:
    """ECE (binned + SmoothECE) + reliability rows for a prob vector, before and after applying T."""
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
    """Reliability diagram (clinical-alone + ISPY2, pre/post temp). Returns False if matplotlib absent."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover
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

    # 1) Clinical-alone per-patient OOF probs (headline). Save to disk — closes the G3 known-gap (L2-C9).
    pids, y_clin, oof = clinical_oof_probs(
        ROOT / "data/raw/Clinical_and_Other_Features.xlsx", ROOT / "configs/split_v2.yaml"
    )
    np.save(OOF_PROBS_NPY, oof)
    # also persist labels + pids alongside for a fully reproducible calibration re-run
    np.savez(OUT_DIR / "oof_probs_full.npz", pids=np.array(pids, dtype=object), y=y_clin, oof=oof)

    # 2) Fit temperature on the OOF (val-equivalent) logits ONLY (E4 — never test/external).
    temperature = fit_temperature(_logit(oof), y_clin.astype(float))

    clin_block = _cal_block(y_clin, oof, temperature)

    # 3) ISPY2 external calibration: load the external probs written by g5_external_eval (seed 0),
    #    apply the SAME T fit on Duke OOF (never re-fit on external — LOCK-2).
    ext_block = None
    ext_npz = ROOT / "reports/G5_external/ispy2_external_probs.npz"
    if ext_npz.exists():
        d = np.load(ext_npz, allow_pickle=True)
        y_ext = np.asarray(d["y"], int)
        # seed-0 external probs (present as probs_seed0)
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
        print("[g5-calibration] AWAITING DATA — data/raw/ clinical table not present (gitignored).")  # noqa: T201
        return 0
    out = run()
    print(json.dumps(out, indent=2))  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

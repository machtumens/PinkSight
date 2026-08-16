from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pinksight.eval.calibration import apply_temperature, fit_temperature, smooth_ece  
from pinksight.metrics import ece  

OUT_DIR = ROOT / "reports" / "G5_calibration"
OOF_NPZ = OUT_DIR / "oof_probs_full.npz"
RESULTS_JSON = OUT_DIR / "calibration_refit.json"
RESULTS_JSON_ALT = OUT_DIR / "calibration_compare.json"

_EPS = 1e-6
_KFOLDS = 5
_SEED = 0  

_TABLE2_ECE_RAW = 0.0196
_TABLE2_ECE_TEMP = 0.0244
_TABLE2_T = 1.0953
_REPRO_TOL = 0.005


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, float), _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


def _held_out_isotonic(oof: np.ndarray, y: np.ndarray, n_splits: int = _KFOLDS, seed: int = _SEED) -> np.ndarray:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.model_selection import StratifiedKFold

    oof = np.asarray(oof, float)
    y = np.asarray(y, int)
    out = np.full(len(y), np.nan)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in skf.split(oof.reshape(-1, 1), y):
        ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(oof[tr], y[tr])
        out[te] = ir.predict(oof[te])
    if np.isnan(out).any():
        raise RuntimeError("held-out isotonic incomplete — a row was never in a test fold")
    return out


def _in_sample_isotonic(oof: np.ndarray, y: np.ndarray) -> np.ndarray:
    from sklearn.isotonic import IsotonicRegression

    oof = np.asarray(oof, float)
    ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(oof, np.asarray(y, int))
    return ir.predict(oof)


def _block(y: np.ndarray, prob: np.ndarray) -> dict:
    return {"ece": round(float(ece(y, prob)), 4), "smooth_ece": round(float(smooth_ece(y, prob)), 4)}


def run() -> dict:
    if not OOF_NPZ.exists():
        raise FileNotFoundError(f"internal OOF probs absent: {OOF_NPZ} (run scripts/g5_calibration.py first)")

    d = np.load(OOF_NPZ, allow_pickle=True)  
    y = np.asarray(d["y"], int)
    oof = np.asarray(d["oof"], float)
    n = int(len(y))

    none_block = _block(y, oof)

    temperature = float(fit_temperature(_logit(oof), y.astype(float)))
    p_temp = apply_temperature(_logit(oof), temperature)
    temp_block = _block(y, p_temp)
    temp_block["T"] = round(temperature, 4)

    p_iso_ho = _held_out_isotonic(oof, y)
    iso_block = _block(y, p_iso_ho)
    iso_block["fit"] = "held_out_kfold5_seed0_leakage_free"
    p_iso_is = _in_sample_isotonic(oof, y)
    iso_in_sample = _block(y, p_iso_is)
    iso_in_sample["fit"] = "in_sample_LEAKY_optimistic_lower_bound_not_used_for_best"

    candidates = {"none": none_block["ece"], "temperature": temp_block["ece"], "isotonic": iso_block["ece"]}
    simpler_rank = {"none": 0, "temperature": 1, "isotonic": 2}
    best_method = min(candidates, key=lambda m: (candidates[m], simpler_rank[m]))

    reproduces_table2_temp = (
        abs(none_block["ece"] - _TABLE2_ECE_RAW) <= _REPRO_TOL
        and abs(temp_block["ece"] - _TABLE2_ECE_TEMP) <= _REPRO_TOL
        and abs(temperature - _TABLE2_T) <= _REPRO_TOL
    )

    temp_worsens = temp_block["ece"] > none_block["ece"]
    iso_beats_raw = iso_block["ece"] < none_block["ece"]

    honest_note = (
        f"Best method = {best_method!r} by lowest held-out ECE (ties -> simpler). "
        f"Temperature scaling {'WORSENS' if temp_worsens else 'does not worsen'} internal ECE "
        f"({none_block['ece']} raw -> {temp_block['ece']} temp, T={temperature:.4f}). "
        f"Held-out (leakage-free) isotonic {'beats' if iso_beats_raw else 'does NOT beat'} raw "
        f"({none_block['ece']} raw vs {iso_block['ece']} isotonic-held-out). "
        f"The in-sample isotonic ECE ({iso_in_sample['ece']}) is a LEAKY optimistic lower bound "
        "(fit+scored on the same OOF, the demo_coimbra in-sample calibration-leak class) and is NOT "
        "used to select best_method. Characterisation framing only (LOCK-1); post-hoc calibration adds "
        "no features (LOCK-2). No LOCK moved."
    )

    out = {
        "gate": "G5-calibration-refit (plan item 6)",
        "framing": "internal calibration re-fit of the characterised clinical-subtype (0.708) OOF; LOCK-1 characterisation only",
        "source_oof": str(OOF_NPZ.relative_to(ROOT)),
        "n": n,
        "prevalence": round(float(np.mean(y)), 4),
        "none": none_block,
        "temperature": temp_block,
        "isotonic": iso_block,
        "isotonic_in_sample_leaky": iso_in_sample,
        "best_method": best_method,
        "best_method_selection": "lowest binned ECE among {none, temperature, isotonic(held-out)}; ties -> simpler (none>temperature>isotonic)",
        "temperature_worsens_ece": bool(temp_worsens),
        "isotonic_held_out_beats_raw": bool(iso_beats_raw),
        "reproduces_table2_temperature": bool(reproduces_table2_temp),
        "table2_reference": {"ece_raw": _TABLE2_ECE_RAW, "ece_temp": _TABLE2_ECE_TEMP, "T": _TABLE2_T, "tol": _REPRO_TOL},
        "isotonic_fit_discipline": (
            "held-out K-fold(5, seed=0): IsotonicRegression fit on train folds only, scored out-of-fold "
            "(leakage-free). In-sample isotonic recorded separately as a labelled leaky lower bound."
        ),
        "seed": _SEED,
        "compute": "cpu-zero-dollar",
        "honest_note": honest_note,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON.write_text(json.dumps(out, indent=2) + "\n")
    RESULTS_JSON_ALT.write_text(json.dumps(out, indent=2) + "\n")
    return out


def main() -> int:
    out = run()
    print(json.dumps(out, indent=2))  
    print(  
        f"\n[calibration-refit] best={out['best_method']} | "
        f"none {out['none']['ece']} | temp {out['temperature']['ece']} (T={out['temperature']['T']}) | "
        f"isotonic-held-out {out['isotonic']['ece']} (in-sample-leaky {out['isotonic_in_sample_leaky']['ece']}) | "
        f"reproduces_table2_temp={out['reproduces_table2_temperature']}"
    )
    print(f"[calibration-refit] wrote {RESULTS_JSON.relative_to(ROOT)} + {RESULTS_JSON_ALT.name}")  
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

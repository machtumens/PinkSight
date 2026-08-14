#!/usr/bin/env python3
"""G5 calibration RE-FIT — none vs temperature vs isotonic on the internal Duke clinical-anchor OOF.

Plan item 6 (model-integrity-remediation_12-08-26). Closes the "temperature-scaling worsens internal
ECE" claim with a proper 3-way comparison and names the HONEST best method (lowest ECE), regardless of
outcome. $0-local, CPU only (LOCK-5). Characterisation framing only (LOCK-1); no FORBIDDEN feature ever
enters anything — this is a post-hoc scalar/monotone map over EXISTING probabilities (LOCK-2 unchanged).

INPUT: reports/G5_calibration/oof_probs_full.npz — the internal Duke clinical-subtype (0.708) headline
per-patient OUT-OF-FOLD probabilities (pids [object], y, oof). This is the only rung with recoverable
per-sample probs pre-dating this plan. `pids` is a numpy object array -> np.load(..., allow_pickle=True)
(Execute-Agent Instruction E2).

THREE calibration methods, ECE (binned) + SmoothECE (bin-free) for each:
  * none          — the raw OOF probabilities (the OOF is CV-held-out, so this is already a held-out
                    calibration state, nothing fit).
  * temperature   — ONE scalar T fit on the OOF logits (val-equivalent), applied to the same OOF.
                    Reuses pinksight.eval.calibration.fit_temperature/apply_temperature EXACTLY as
                    scripts/g5_calibration.py (reproduces the Table2 `ECE 0.0196 -> 0.0244, T=1.095`).
                    T is a 1-parameter map, so in-sample optimism on N=624 is negligible.
  * isotonic      — a non-parametric monotone map. Fit+scored IN-SAMPLE it overfits to a fake ~0 ECE
                    (the exact in-sample calibration leak documented for demo_coimbra: ECE ~3.4e-17).
                    That is NOT a real improvement, so the HONEST isotonic number is a leakage-free
                    K-fold HELD-OUT isotonic (fit on train folds, scored out-of-fold). The in-sample
                    isotonic ECE is also recorded, clearly labelled as an optimistic LEAKY lower bound,
                    and is NEVER used to pick `best_method`.

BEST METHOD is chosen by lowest HELD-OUT/honest ECE among {none, temperature, isotonic(held-out)}, ties
broken toward the SIMPLER method (none > temperature > isotonic — fewer fit parameters). The expected,
honest finding is that no post-hoc method improves on the already-good raw calibration at this small N
(temperature WORSENS ECE; held-out isotonic does not beat raw) -> best = none. Reported as-is either way.

Seeded for reproducibility (LAW L-3): the held-out isotonic uses StratifiedKFold(random_state=0).

Run:  uv run python scripts/g5_calibration_refit.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pinksight.eval.calibration import apply_temperature, fit_temperature, smooth_ece  # noqa: E402
from pinksight.metrics import ece  # noqa: E402

OUT_DIR = ROOT / "reports" / "G5_calibration"
OOF_NPZ = OUT_DIR / "oof_probs_full.npz"
# Plan checklist names calibration_refit.json; the EXECUTE handoff also names calibration_compare.json.
# Write BOTH (identical content) so the plan gate and the handoff instruction are both satisfied.
RESULTS_JSON = OUT_DIR / "calibration_refit.json"
RESULTS_JSON_ALT = OUT_DIR / "calibration_compare.json"

_EPS = 1e-6
_KFOLDS = 5
_SEED = 0  # LAW L-3 — deterministic held-out isotonic

# The frozen Table2 / metrics.json temperature reference this item must reproduce.
_TABLE2_ECE_RAW = 0.0196
_TABLE2_ECE_TEMP = 0.0244
_TABLE2_T = 1.0953
_REPRO_TOL = 0.005


def _logit(p: np.ndarray) -> np.ndarray:
    """Exact inverse-sigmoid (probs come from sigmoid(logits)); clip to avoid +/-inf at 0/1."""
    p = np.clip(np.asarray(p, float), _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


def _held_out_isotonic(oof: np.ndarray, y: np.ndarray, n_splits: int = _KFOLDS, seed: int = _SEED) -> np.ndarray:
    """Leakage-free isotonic: K-fold, fit IsotonicRegression on train folds, score the held-out fold.

    Each patient's calibrated prob comes from a monotone map that never saw that patient -> no in-sample
    calibration leak. Mirrors the nested-CV recalibration discipline used in demo_coimbra_calibration.
    """
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
    """In-sample isotonic (fit+score on the SAME OOF) — an optimistic LEAKY lower bound, reported for
    transparency only, NEVER used to select best_method (it fakes a ~0 ECE)."""
    from sklearn.isotonic import IsotonicRegression

    oof = np.asarray(oof, float)
    ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(oof, np.asarray(y, int))
    return ir.predict(oof)


def _block(y: np.ndarray, prob: np.ndarray) -> dict:
    return {"ece": round(float(ece(y, prob)), 4), "smooth_ece": round(float(smooth_ece(y, prob)), 4)}


def run() -> dict:
    if not OOF_NPZ.exists():
        raise FileNotFoundError(f"internal OOF probs absent: {OOF_NPZ} (run scripts/g5_calibration.py first)")

    d = np.load(OOF_NPZ, allow_pickle=True)  # E2: pids is an object array
    y = np.asarray(d["y"], int)
    oof = np.asarray(d["oof"], float)
    n = int(len(y))

    # --- none (raw OOF, already held-out) ---
    none_block = _block(y, oof)

    # --- temperature (1-param, fit on OOF logits, applied to OOF) — reproduces the Table2 number ---
    temperature = float(fit_temperature(_logit(oof), y.astype(float)))
    p_temp = apply_temperature(_logit(oof), temperature)
    temp_block = _block(y, p_temp)
    temp_block["T"] = round(temperature, 4)

    # --- isotonic: honest held-out + labelled leaky in-sample ---
    p_iso_ho = _held_out_isotonic(oof, y)
    iso_block = _block(y, p_iso_ho)
    iso_block["fit"] = "held_out_kfold5_seed0_leakage_free"
    p_iso_is = _in_sample_isotonic(oof, y)
    iso_in_sample = _block(y, p_iso_is)
    iso_in_sample["fit"] = "in_sample_LEAKY_optimistic_lower_bound_not_used_for_best"

    # --- pick best by honest ECE, ties -> simpler ---
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
    print(json.dumps(out, indent=2))  # noqa: T201
    print(  # noqa: T201
        f"\n[calibration-refit] best={out['best_method']} | "
        f"none {out['none']['ece']} | temp {out['temperature']['ece']} (T={out['temperature']['T']}) | "
        f"isotonic-held-out {out['isotonic']['ece']} (in-sample-leaky {out['isotonic_in_sample_leaky']['ece']}) | "
        f"reproduces_table2_temp={out['reproduces_table2_temperature']}"
    )
    print(f"[calibration-refit] wrote {RESULTS_JSON.relative_to(ROOT)} + {RESULTS_JSON_ALT.name}")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""G3 #3 — counterfactual saliency for the GRADE head (HiResCAM + minimal-perturbation flip).

Counterfactual input-sensitivity for grade CHARACTERISATION — NOT a growth-rate / kinetics claim,
NOT a causal intervention (decisions.md LOCK-1). Grade head ONLY: the subtype head has negligible
imaging signal (H6 Shapley −0.025), so counterfactuals over it are ill-defined. Allowed wording:
"model grade characterisation is sensitive to enhancement pattern in region Z".

Two modes:
  --smoke-only : run on the synthetic tiny-CNN grade fixture (E-5). $0 CPU. Step 3.3.
                 Gate: flip_achieved on >= 1/3 synthetic patients.
  (default)    : full run on a trained G3 grade-head encoder checkpoint (Step 3.6, GPU) — computes
                 HiResCAM IoU / pointing vs nnU-Net PREDICTED masks + 3-seed randomization sanity
                 (Step 3.5 hardening) + counterfactual flip rate. OUT OF SCOPE for the $0 leg; the
                 checkpoint path raises until a trained G3 grade encoder exists.

    PYTHONPATH=src .venv/bin/python scripts/xai_counterfactual_grade.py --smoke-only \
        --out reports/G3_fusion_arch_bundle/smoke_xai_cf.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from fixtures.synthetic import tiny_cnn_grade_fixture  # noqa: E402

from pinksight.xai.faithfulness import randomization_test  # noqa: E402
from pinksight.xai.saliency import (  # noqa: E402
    counterfactual_grade_map,
    randomize_weights,
)


def mean_randomization_rel_drop(model, vol, cam_trained, grade_head, target_layer,
                                n_seeds: int = 3) -> dict:
    """Step 3.5 hardening (TICKET-001 promotion): 3-seed AVERAGE randomization sanity.

    A faithful HiResCAM map concentrates in its ROI; a weight-randomized map should not. Single-seed
    rel_drop is noisy (faithfulness.py line ~79 caveat), so we average over `n_seeds` re-init seeds
    and report mean_rel_drop. Returns {per_seed_rel_drop, mean_rel_drop, passed}."""
    rels = []
    for seed in range(n_seeds):
        rnd = randomize_weights(model, seed=seed)
        # HiResCAM on the randomized trunk via the grade-head wrapper the counterfactual util uses.
        res = counterfactual_grade_map(rnd, vol, grade_head, _find_layer(rnd, target_layer),
                                       n_steps=1)  # n_steps=1: we only need the randomized cam here
        r = randomization_test(cam_trained, res["cam_original"])
        rels.append(r["rel_drop"])
    mean_rel = float(np.mean(rels))
    return {"per_seed_rel_drop": [round(x, 4) for x in rels],
            "mean_rel_drop": round(mean_rel, 4), "passed": bool(mean_rel > 0.50)}


def _find_layer(randomized_model, orig_layer):
    """Map the original target layer to the same position in a deep-copied randomized model."""
    # randomize_weights deep-copies the module; index 2 of the trunk Sequential is the target conv.
    try:
        return randomized_model[2]
    except (TypeError, KeyError, IndexError):
        return orig_layer


def run_smoke(out_path: Path) -> dict:
    """$0 CPU smoke on the synthetic grade fixture — architecture validity + flip check (Step 3.3)."""
    fx = tiny_cnn_grade_fixture()
    model, grade_head, target_layer, vol = (
        fx["model"], fx["grade_head"], fx["target_layer"], fx["volume"])

    # Run the counterfactual on 3 synthetic "patients" (jittered copies of the fixture volume).
    rng = np.random.default_rng(0)
    import torch
    flips, deltas, rhos = [], [], []
    cam0 = None
    for i in range(3):
        v = vol + torch.from_numpy(rng.normal(0, 0.01, vol.shape)).float()
        res = counterfactual_grade_map(model, v, grade_head, target_layer,
                                       target_flip=i % 2, n_steps=150, lr=0.05)
        flips.append(res["flip_achieved"])
        deltas.append(res["delta_logit"])
        rhos.append(res["spearman_rho"])
        if cam0 is None:
            cam0 = res["cam_original"]

    flip_rate = float(np.mean(flips))
    sanity = mean_randomization_rel_drop(model, vol, cam0, grade_head, target_layer, n_seeds=3)

    doc = {
        "gate": "G3 #3 counterfactual XAI (grade head) — $0 CPU smoke",
        "smoke_only": True,
        "cf_flip_rate": round(flip_rate, 4),
        "cf_flip_achieved_count": int(sum(flips)),
        "n_synthetic_patients": 3,
        "cf_flip_gate_met": bool(sum(flips) >= 1),   # Step 3.3 gate: >= 1/3 patients flip
        "delta_logit_mean": round(float(np.mean(deltas)), 4),
        "spearman_rho_mean": round(float(np.mean(rhos)), 4),
        "sanity_3seed": sanity,
        "note": ("Synthetic tiny-CNN grade fixture (E-5). Architecture validity + flip check only — "
                 "NOT the real IoU/pointing gate (that needs a trained G3 grade encoder, Step 3.6, "
                 "GPU). 3-seed randomization sanity is the Step 3.5 hardening on the fixture."),
        "claim_ledger": ("counterfactual input-sensitivity for grade characterisation — NOT a "
                         "growth-rate / kinetics claim; NOT a causal intervention; grade head only."),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"[G3 #3] cf_flip_rate {flip_rate:.4f} ({sum(flips)}/3)  "
          f"delta_logit_mean {doc['delta_logit_mean']}  "
          f"sanity mean_rel_drop {sanity['mean_rel_drop']} passed={sanity['passed']}")
    print(f"wrote {out_path}")
    return doc


def run_full(encoder_ckpt, target_layer, n_sanity_seeds, sanity_threshold, split_yaml,
             masks_dir, out_path):
    """Full XAI run on a trained G3 grade encoder (Step 3.6). OUT OF SCOPE for the $0 leg."""
    raise SystemExit(
        "xai_counterfactual_grade.py full run (Step 3.6) needs a trained G3 grade-head encoder "
        "checkpoint, which does not exist in the $0 local leg. Run --smoke-only for the CPU gate. "
        "The full IoU/pointing/3-seed-sanity run is the GPU checkpoint the orchestrator gates with "
        "the user."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke-only", action="store_true")
    ap.add_argument("--encoder-ckpt", type=Path, default=None)
    ap.add_argument("--target-layer", default="layer4")
    ap.add_argument("--n-sanity-seeds", type=int, default=3)
    ap.add_argument("--sanity-threshold", type=float, default=0.50)
    ap.add_argument("--split", type=Path, default=ROOT / "configs/split_v2.yaml")
    ap.add_argument("--masks-dir", type=Path, default=ROOT / "data/processed_masks")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if args.smoke_only:
        run_smoke(args.out)
    else:
        run_full(args.encoder_ckpt, args.target_layer, args.n_sanity_seeds, args.sanity_threshold,
                 args.split, args.masks_dir, args.out)


if __name__ == "__main__":
    # No CLI args (e.g. the jury notebook run top-to-bottom): run the built-in fixture
    # self-check instead of erroring on the required --out. main() still serves the full CLI.
    if len(sys.argv) > 1:
        main()
    else:
        run_smoke(ROOT / "reports/EXP-xai-cf-grade")

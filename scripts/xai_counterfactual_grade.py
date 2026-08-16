from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from fixtures.synthetic import tiny_cnn_grade_fixture  

from pinksight.xai.faithfulness import randomization_test  
from pinksight.xai.saliency import (  
    counterfactual_grade_map,
    randomize_weights,
)


def mean_randomization_rel_drop(model, vol, cam_trained, grade_head, target_layer,
                                n_seeds: int = 3) -> dict:
    rels = []
    for seed in range(n_seeds):
        rnd = randomize_weights(model, seed=seed)
        res = counterfactual_grade_map(rnd, vol, grade_head, _find_layer(rnd, target_layer),
                                       n_steps=1)  
        r = randomization_test(cam_trained, res["cam_original"])
        rels.append(r["rel_drop"])
    mean_rel = float(np.mean(rels))
    return {"per_seed_rel_drop": [round(x, 4) for x in rels],
            "mean_rel_drop": round(mean_rel, 4), "passed": bool(mean_rel > 0.50)}


def _find_layer(randomized_model, orig_layer):
    try:
        return randomized_model[2]
    except (TypeError, KeyError, IndexError):
        return orig_layer


def run_smoke(out_path: Path) -> dict:
    fx = tiny_cnn_grade_fixture()
    model, grade_head, target_layer, vol = (
        fx["model"], fx["grade_head"], fx["target_layer"], fx["volume"])

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
        "cf_flip_gate_met": bool(sum(flips) >= 1),   
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
    if len(sys.argv) > 1:
        main()
    else:
        run_smoke(ROOT / "reports/EXP-xai-cf-grade")

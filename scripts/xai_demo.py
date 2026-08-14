"""P10 demo: write reports/EXP-fixture/xai/scores.json (IoU, pointing, randomization, SHAP).

  PYTHONPATH=src .venv/bin/python scripts/xai_demo.py
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from fixtures.synthetic import (  # noqa: E402
    clinical_model_fixture,
    saliency_mask_fixture,
    tiny_cnn_fixture,
)

from pinksight.xai.faithfulness import box_hit, iou, pointing_game, randomization_test  # noqa: E402
from pinksight.xai.saliency import clinical_shap, grad_cam_3d, randomize_weights  # noqa: E402

if __name__ == "__main__":
    fx = saliency_mask_fixture()
    model, layer, vol = tiny_cnn_fixture()
    cam = grad_cam_3d(model, vol, layer)
    rnd = randomize_weights(model, 1)
    cam_rnd = grad_cam_3d(rnd, vol, rnd[2])  # hook the randomized model's OWN layer
    clin = clinical_model_fixture()
    values, base = clinical_shap(clin["model"], clin["X"], clin["background"])

    scores = {
        "localisation_vs_independent_reference": {
            "iou": round(iou(fx["saliency"], fx["reference_mask"]), 4),
            "pointing_game_hit": pointing_game(fx["saliency"], fx["reference_mask"]),
            "box_hit": box_hit(fx["saliency"], fx["box"]),
            "iou_bar": 0.30, "pointing_bar": 0.70,
            "note": "reference held OUT of training — NOT the model's conditioning mask (Rule 6).",
        },
        "randomization_sanity": randomization_test(fx["saliency"], fx["random_saliency"]),
        "grad_cam_3d": {
            "cam_shape": list(cam.shape),
            "orig_vs_random_weights": randomization_test(cam, cam_rnd),
            "note": "Grad-CAM runs on the tiny 3D CNN today; the >50% randomization flip is only "
                    "expected once the encoder is TRAINED (G2). On an untrained net the map is "
                    "input-driven, so the flip is reported but not gated here.",
        },
        "clinical_shap": {"mean_abs_shap": [round(float(v), 4) for v in np.abs(values).mean(0)],
                          "base_value": round(base, 4)},
        "source": "synthetic fixture — wired to saliency tensors, not a trained model",
    }
    out = ROOT / "reports" / "EXP-fixture" / "xai"
    out.mkdir(parents=True, exist_ok=True)
    (out / "scores.json").write_text(json.dumps(scores, indent=2, sort_keys=True) + "\n")
    np.save(out / "grad_cam_seed0.npy", cam)
    print(f"wrote {out/'scores.json'} + grad_cam_seed0.npy")  # noqa: T201
    print(f"IoU={scores['localisation_vs_independent_reference']['iou']}  "  # noqa: T201
          f"randomization drop={scores['randomization_sanity']['rel_drop']} "
          f"passed={scores['randomization_sanity']['passed']}")

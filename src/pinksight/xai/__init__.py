"""P10 explainability harness — saliency generation + faithfulness gauntlet (non-circular IoU)."""

from pinksight.xai.faithfulness import box_hit, iou, pointing_game, randomization_test
from pinksight.xai.saliency import attention_map, clinical_shap, grad_cam_3d, randomize_weights

__all__ = ["box_hit", "iou", "pointing_game", "randomization_test",
           "attention_map", "clinical_shap", "grad_cam_3d", "randomize_weights"]

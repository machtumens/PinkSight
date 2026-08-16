
from __future__ import annotations

import numpy as np

SEEDS = (0, 1, 2)
N_PATIENTS = 200
KI67_THRESH = 14.0  

RUNGS = ("radiomics", "unimodal", "late_fusion", "cross_attn")
_SEP = {"radiomics": 0.55, "unimodal": 0.75, "late_fusion": 0.95, "cross_attn": 1.20}


def _labels(rng: np.random.Generator, n: int, prev: float = 0.21) -> np.ndarray:
    y = (rng.random(n) < prev).astype(int)
    if y.sum() == 0:
        y[0] = 1
    if y.sum() == n:
        y[0] = 0
    return y


def classification_fixture(seeds=SEEDS, n=N_PATIENTS) -> dict:
    y_true: dict[int, np.ndarray] = {}
    probs: dict[str, dict[int, np.ndarray]] = {r: {} for r in RUNGS}
    for seed in seeds:
        rng = np.random.default_rng(1000 + seed)
        y = _labels(rng, n)
        y_true[seed] = y
        latent = rng.normal(0, 1, n)  
        for r in RUNGS:
            logit = _SEP[r] * (y * 2 - 1) + 0.5 * latent + rng.normal(0, 0.8, n)
            probs[r][seed] = 1.0 / (1.0 + np.exp(-logit))
    return {"y_true": y_true, "probs": probs}


def ki67_fixture(seed=7, n=N_PATIENTS) -> dict:
    rng = np.random.default_rng(seed)
    y = rng.uniform(0.0, 60.0, n)
    pred = np.clip(y + rng.normal(0, 12, n), 0.0, 100.0)
    return {"y_true": y, "pred": pred, "thresh": KI67_THRESH}


def logits_fixture(seed=11, n=N_PATIENTS) -> dict:
    rng = np.random.default_rng(seed)
    y = _labels(rng, n)
    logits = 3.0 * (y * 2 - 1) + rng.normal(0, 1.5, n)  
    half = n // 2
    return {
        "val": {"logits": logits[:half], "labels": y[:half]},
        "test": {"logits": logits[half:], "labels": y[half:]},
    }


def _ball(shape, cz, cy, cx, r) -> np.ndarray:
    zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]
    return ((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2) <= r * r


def saliency_mask_fixture(seed=3, shape=(24, 24, 24)) -> dict:
    rng = np.random.default_rng(seed)
    sal = _ball(shape, 12, 12, 12, 6).astype(float) + rng.random(shape) * 0.05
    ref = _ball(shape, 14, 11, 13, 6).astype(np.uint8)  
    rand_sal = rng.random(shape) * 0.05  
    box = (6, 18, 5, 19, 7, 19)  
    return {"saliency": sal, "reference_mask": ref, "random_saliency": rand_sal, "box": box}


def tiny_cnn_fixture(seed=0):
    import torch
    from torch import nn

    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Conv3d(1, 4, 3, padding=1),
        nn.ReLU(),
        nn.Conv3d(4, 8, 3, padding=1),  
        nn.ReLU(),
        nn.AdaptiveAvgPool3d(1),
        nn.Flatten(),
        nn.Linear(8, 1),
    )
    vol = torch.zeros(1, 1, 16, 16, 16)
    vol[0, 0, 6:11, 6:11, 6:11] = 1.0  
    return model, model[2], vol


def tiny_cnn_grade_fixture(seed=0):
    import torch
    from torch import nn

    torch.manual_seed(seed)
    trunk = nn.Sequential(
        nn.Conv3d(1, 4, 3, padding=1),
        nn.ReLU(),
        nn.Conv3d(4, 8, 3, padding=1),  
        nn.ReLU(),
    )
    grade_head = nn.Linear(8, 1)          
    subtype_head = nn.Linear(8, 1)
    subtype_head.debug_name = "subtype_head"  
    vol = torch.zeros(1, 1, 16, 16, 16)
    vol[0, 0, 6:11, 6:11, 6:11] = 1.0
    return {"model": trunk, "grade_head": grade_head, "subtype_head": subtype_head,
            "target_layer": trunk[2], "volume": vol}


def clinical_model_fixture(seed=5, n=300, d=6):
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, (n, d))
    coef = np.array([1.4, -0.9, 0.6, 0.0, 0.3, -0.5])
    y = (1.0 / (1.0 + np.exp(-(x @ coef))) > rng.random(n)).astype(int)
    model = LogisticRegression(max_iter=500).fit(x, y)
    return {"model": model, "X": x[:20], "background": x}

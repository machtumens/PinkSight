"""Synthetic fixtures for the eval/stats/xai harnesses — runnable today, no trained model.

All generators are seeded so metrics.json / stats.json regenerate byte-stable. They emit only
prediction arrays, logits, and saliency/mask tensors — the three harnesses are model-agnostic,
so a synthetic draw exercises every code path the real G2/G3 predictions will.

Claim-ledger: labels are "subtype characterisation" + "Ki-67 stratification at diagnosis".
"""

from __future__ import annotations

import numpy as np

SEEDS = (0, 1, 2)
N_PATIENTS = 200
KI67_THRESH = 14.0  # decisions.md: Ki-67 binary @ 14%

# ablation ladder rungs, weakest → strongest (cross-attn fusion must beat unimodal)
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
    """Per-seed, patient-aligned probs for each ablation rung on a shared label vector.

    Returns {"y_true": {seed: y}, "probs": {rung: {seed: p}}}. Within a seed the rungs score
    the SAME patients (a shared latent makes their ROCs correlated, like real fusion vs unimodal),
    so paired ΔAUC tests are valid. Higher rungs carry more class signal.
    """
    y_true: dict[int, np.ndarray] = {}
    probs: dict[str, dict[int, np.ndarray]] = {r: {} for r in RUNGS}
    for seed in seeds:
        rng = np.random.default_rng(1000 + seed)
        y = _labels(rng, n)
        y_true[seed] = y
        latent = rng.normal(0, 1, n)  # shared nuisance → correlated rungs
        for r in RUNGS:
            logit = _SEP[r] * (y * 2 - 1) + 0.5 * latent + rng.normal(0, 0.8, n)
            probs[r][seed] = 1.0 / (1.0 + np.exp(-logit))
    return {"y_true": y_true, "probs": probs}


def ki67_fixture(seed=7, n=N_PATIENTS) -> dict:
    """Continuous Ki-67 index (%) + a correlated-but-noisy prediction. A snapshot, not kinetics."""
    rng = np.random.default_rng(seed)
    y = rng.uniform(0.0, 60.0, n)
    pred = np.clip(y + rng.normal(0, 12, n), 0.0, 100.0)
    return {"y_true": y, "pred": pred, "thresh": KI67_THRESH}


def logits_fixture(seed=11, n=N_PATIENTS) -> dict:
    """Miscalibrated logits split into val/test — for temperature scaling (fit on val only)."""
    rng = np.random.default_rng(seed)
    y = _labels(rng, n)
    logits = 3.0 * (y * 2 - 1) + rng.normal(0, 1.5, n)  # over-confident scale → T>1 expected
    half = n // 2
    return {
        "val": {"logits": logits[:half], "labels": y[:half]},
        "test": {"logits": logits[half:], "labels": y[half:]},
    }


def _ball(shape, cz, cy, cx, r) -> np.ndarray:
    zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]
    return ((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2) <= r * r


def saliency_mask_fixture(seed=3, shape=(24, 24, 24)) -> dict:
    """A saliency volume + an INDEPENDENT reference mask that overlaps it.

    The reference is a separate radiologist-style blob (offset a few voxels) — NOT the region the
    saliency was derived from — so IoU / pointing are honest, non-circular scores (Rule 6).
    Also returns a near-uniform "randomized-model" saliency so the sanity test is runnable.
    """
    rng = np.random.default_rng(seed)
    sal = _ball(shape, 12, 12, 12, 6).astype(float) + rng.random(shape) * 0.05
    ref = _ball(shape, 14, 11, 13, 6).astype(np.uint8)  # independent, slightly offset
    rand_sal = rng.random(shape) * 0.05  # weight-randomized model → no concentration
    box = (6, 18, 5, 19, 7, 19)  # fallback box enclosing the reference (Duke boxes-only case)
    return {"saliency": sal, "reference_mask": ref, "random_saliency": rand_sal, "box": box}


def tiny_cnn_fixture(seed=0):
    """A minimal 3D CNN so Grad-CAM is exercised on a real (if untrained) model."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Conv3d(1, 4, 3, padding=1),
        nn.ReLU(),
        nn.Conv3d(4, 8, 3, padding=1),  # <- Grad-CAM target layer (index 2)
        nn.ReLU(),
        nn.AdaptiveAvgPool3d(1),
        nn.Flatten(),
        nn.Linear(8, 1),
    )
    vol = torch.zeros(1, 1, 16, 16, 16)
    vol[0, 0, 6:11, 6:11, 6:11] = 1.0  # a bright blob to give the CAM something to latch onto
    return model, model[2], vol


def tiny_cnn_grade_fixture(seed=0):
    """A minimal 3D CNN trunk + a named grade_head, for counterfactual_grade_map() smoke (E-5).

    Returns a dict (NOT the tuple that tiny_cnn_fixture returns — that signature is left untouched):
      {"model": trunk, "grade_head": nn.Linear, "target_layer": conv, "volume": (1,1,16,16,16),
       "subtype_head": nn.Linear w/ debug_name='subtype_head' (for the grade-head-only guard test)}.
    The trunk outputs a feature map at `target_layer`; the counterfactual util GAP-pools it before
    the grade_head. A bright blob gives the CAM something to latch onto. Claim-ledger: grade
    characterisation at diagnosis only — the counterfactual is input-sensitivity, not kinetics."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    trunk = nn.Sequential(
        nn.Conv3d(1, 4, 3, padding=1),
        nn.ReLU(),
        nn.Conv3d(4, 8, 3, padding=1),  # <- target layer for HiResCAM (index 2), 8-channel feature map
        nn.ReLU(),
    )
    grade_head = nn.Linear(8, 1)          # consumes GAP-pooled 8-d features -> grade logit (NHG1 v NHG3)
    subtype_head = nn.Linear(8, 1)
    subtype_head.debug_name = "subtype_head"  # sentinel so the E-6 grade-head-only guard can reject it
    vol = torch.zeros(1, 1, 16, 16, 16)
    vol[0, 0, 6:11, 6:11, 6:11] = 1.0
    return {"model": trunk, "grade_head": grade_head, "subtype_head": subtype_head,
            "target_layer": trunk[2], "volume": vol}


def clinical_model_fixture(seed=5, n=300, d=6):
    """A fitted linear clinical model + a small feature matrix — exact SHAP target (additive)."""
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, (n, d))
    coef = np.array([1.4, -0.9, 0.6, 0.0, 0.3, -0.5])
    y = (1.0 / (1.0 + np.exp(-(x @ coef))) > rng.random(n)).astype(int)
    model = LogisticRegression(max_iter=500).fit(x, y)
    return {"model": model, "X": x[:20], "background": x}

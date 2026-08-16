
from __future__ import annotations

import numpy as np

N_PATIENTS = 120
TILE_DIM = 1024   
GENO_DIM = 40     
PREV = 0.30       
SEEDS = (0, 1, 2)


def _labels(rng: np.random.Generator, n: int, prev: float = PREV) -> np.ndarray:
    y = (rng.random(n) < prev).astype(int)
    if y.sum() == 0:
        y[0] = 1
    if y.sum() == n:
        y[0] = 0
    return y


def bag_fixture(seed: int = 0, n_tiles: int = 32, tile_dim: int = TILE_DIM,
                label: int = 1) -> np.ndarray:
    rng = np.random.default_rng(1000 + seed)
    shift = 0.6 if label == 1 else -0.6
    return (rng.standard_normal((n_tiles, tile_dim)) * 0.2 + shift * 0.05).astype(np.float32)


def genomics_fixture(seed: int = 0, n: int = N_PATIENTS, d: int = GENO_DIM) -> dict:
    rng = np.random.default_rng(2000 + seed)
    y = _labels(rng, n)
    coef = rng.standard_normal(d) * 0.3
    x = rng.standard_normal((n, d)).astype(np.float32)
    x += (y[:, None] * 2 - 1) * 0.15 * np.sign(coef)[None, :].astype(np.float32)
    return {"X": x, "y": y}


RUNG_MEANING = {
    "radiomics": "genomics-only (label-safe expression/CNA MLP)",
    "unimodal": "WSI-only (frozen foundation tiles -> attention-MIL)",
    "late_fusion": "WSI + genomics, late concat",
    "cross_attn": "WSI + genomics, modality-dropout fusion (strongest Track B rung)",
}


def cohort_fixture(seeds=SEEDS, n=N_PATIENTS) -> dict:
    rungs = ("radiomics", "unimodal", "late_fusion", "cross_attn")
    sep = {"radiomics": 0.45, "unimodal": 0.70, "late_fusion": 0.90, "cross_attn": 1.05}
    y_true: dict[int, np.ndarray] = {}
    probs: dict[str, dict[int, np.ndarray]] = {r: {} for r in rungs}
    for seed in seeds:
        rng = np.random.default_rng(3000 + seed)
        y = _labels(rng, n)
        y_true[seed] = y
        latent = rng.normal(0, 1, n)  
        for r in rungs:
            logit = sep[r] * (y * 2 - 1) + 0.5 * latent + rng.normal(0, 0.8, n)
            probs[r][seed] = 1.0 / (1.0 + np.exp(-logit))
    return {"y_true": y_true, "probs": probs, "rung_meaning": RUNG_MEANING}

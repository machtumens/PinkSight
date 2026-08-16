
from __future__ import annotations

import numpy as np

ENHANCEMENT_PERCENTILE_DEFAULT = 90.0  
_METHODS = ("percentile",)


def enhancement_mask(
    pre: np.ndarray,
    post: np.ndarray,
    method: str = "percentile",
    percentile: float = ENHANCEMENT_PERCENTILE_DEFAULT,
) -> np.ndarray:
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}")
    if not (0.0 <= percentile <= 100.0):
        raise ValueError(f"percentile must be in [0, 100], got {percentile}")
    pre_arr = np.asarray(pre, dtype=np.float64)
    post_arr = np.asarray(post, dtype=np.float64)
    if pre_arr.shape != post_arr.shape:
        raise ValueError(f"pre/post shape mismatch: {pre_arr.shape} != {post_arr.shape}")
    enhancement = post_arr - pre_arr  
    threshold = float(np.percentile(enhancement, percentile))
    return enhancement > threshold  

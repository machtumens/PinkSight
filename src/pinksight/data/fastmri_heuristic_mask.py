"""fastMRI-NYU label-blind heuristic enhancement mask (lesion-crop plan, Rung 0 / Rung 1).

No nnU-Net weights are provisioned for NYU: ``h0_localizer.predict_mask`` raises
``WeightsNotProvisioned`` by design (LOCK-2 — never fabricate a segmentation). This module builds a
materially DIFFERENT, label-blind mask SOURCE keyed purely on DCE enhancement intensity, whose only
job is to drive the SAME already-tested ``lesion_crop`` / ``derive_lesion_box`` box-derive → crop →
resample chain the Duke imaging arms use. It reuses the downstream box/crop/rim machinery, never a
fabricated tumour segmentation.

INTEGRITY (pre-registered, load-bearing — see the plan Section 2):
  * LABEL-BLIND BY CONSTRUCTION: the signature takes ONLY image arrays. No parameter named
    label / y / target / malignant / class_id exists, and there is no label-dependent branch
    anywhere in the computation. Asserted structurally in
    ``tests/test_fastmri_heuristic_mask.py::test_signature_has_no_label_parameter``.
  * DETERMINISTIC: identical ``(pre, post)`` inputs always yield a byte-identical mask (pure numpy,
    no RNG, no global state, inputs never mutated). Asserted in
    ``test_identical_images_yield_identical_masks_regardless_of_label``.

The mask is a RAW percentile threshold on the enhancement image ``post - pre``. It stays a pure
threshold — ``derive_lesion_box`` already applies ``largest_cc`` + rim dilation + boxing downstream,
so no connected-component reduction is done here. A degenerate (all-equal) input yields a well-formed
EMPTY mask, so the tested ``[1.6]`` Duke-box fallback fires cleanly downstream instead of raising.

numpy-only (no scikit-image / Otsu — not a project dependency; YAGNI).
"""

from __future__ import annotations

import numpy as np

ENHANCEMENT_PERCENTILE_DEFAULT = 90.0  # keep the most-enhancing ~10% of voxels as tumour candidates
_METHODS = ("percentile",)


def enhancement_mask(
    pre: np.ndarray,
    post: np.ndarray,
    method: str = "percentile",
    percentile: float = ENHANCEMENT_PERCENTILE_DEFAULT,
) -> np.ndarray:
    """Label-blind heuristic tumour-candidate mask from a (pre, post) DCE pair.

    Enhancement is ``post - pre``; the mask keeps voxels whose enhancement STRICTLY exceeds the
    ``percentile``-th percentile of the enhancement image (i.e. the most-enhancing
    ``100 - percentile`` percent of voxels). Returns a boolean ``np.ndarray`` with the SAME shape as
    ``pre`` / ``post``.

    Strict ``>`` (not ``>=``) is deliberate: on a degenerate all-equal input the enhancement image is
    constant, its percentile equals that constant, and ``> constant`` yields an all-False (EMPTY)
    mask — so the downstream ``[1.6]`` Duke-box fallback fires cleanly rather than selecting the whole
    volume.

    Label-blind + deterministic by construction (plan Section 2 rule 3): no label parameter exists and
    the computation is pure numpy with no RNG or global state. ``pre`` / ``post`` are never mutated
    (cast to a fresh float64 working copy before any arithmetic).
    """
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}")
    if not (0.0 <= percentile <= 100.0):
        raise ValueError(f"percentile must be in [0, 100], got {percentile}")
    pre_arr = np.asarray(pre, dtype=np.float64)
    post_arr = np.asarray(post, dtype=np.float64)
    if pre_arr.shape != post_arr.shape:
        raise ValueError(f"pre/post shape mismatch: {pre_arr.shape} != {post_arr.shape}")
    enhancement = post_arr - pre_arr  # fresh array; inputs untouched (float64 dodges int wrap-around)
    threshold = float(np.percentile(enhancement, percentile))
    return enhancement > threshold  # boolean ndarray, same shape as the inputs

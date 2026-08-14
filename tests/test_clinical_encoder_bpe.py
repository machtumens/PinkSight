"""BP1 clinical-encoder BPE join/leakage guard (LOCK-2) — the IN-scope half of the old test_bp1_bpe.

Phase-4 package split: the BPE-math self-checks that exercised ``scripts/extract_bpe_features.py``
(an OUT-of-release preprocessing script, not shipped in this package) were dropped, since they test
code that no longer lives in the tree. This file keeps the load-bearing LOCK-2 guard on
``pinksight.models.clinical_encoder.load_bpe`` — that a FORBIDDEN (label-defining) field name in a BPE
npz raises ``LeakageError`` — running in its natural IN-scope home with no dependency on the OUT
script. Torch-free / scipy-free: ``clinical_encoder`` lazy-imports the ml stack, so this collects and
runs under a base install.
"""

from __future__ import annotations

import numpy as np
import pytest

# The BPE feature-name schema (written by the OUT-of-release scripts/extract_bpe_features.py,
# FEATURE_NAMES there). Hard-coded here as the fixture's expected names so this IN-scope test needs
# no import of the OUT script.
BPE_FEATURE_NAMES = ["bpe_magnitude", "bpe_entropy", "tumour_residual"]


def test_load_bpe_join_and_leakage(tmp_path):
    from pinksight.models.clinical_encoder import LeakageError, load_bpe

    good = tmp_path / "bpe.npz"
    np.savez(good, patient_ids=np.array(["Breast_MRI_001"], dtype=object),
             features=np.array([[0.5, 1.2, 1.0]]), feature_names=np.array(BPE_FEATURE_NAMES, dtype=object))
    m, names = load_bpe(good)
    assert names == BPE_FEATURE_NAMES
    assert m["Breast_MRI_001"].tolist() == [0.5, 1.2, 1.0]

    bad = tmp_path / "leak.npz"
    np.savez(bad, patient_ids=np.array(["x"], dtype=object), features=np.array([[1.0]]),
             feature_names=np.array(["Ki-67"], dtype=object))  # a forbidden field name
    with pytest.raises(LeakageError):
        load_bpe(bad)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


from __future__ import annotations

import numpy as np
import pytest

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
             feature_names=np.array(["Ki-67"], dtype=object))  
    with pytest.raises(LeakageError):
        load_bpe(bad)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

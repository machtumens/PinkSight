
from pathlib import Path

import pytest

FIXED_CASE = "Breast_MRI_042"
_FIXTURE = Path("data/raw/duke_breast_cancer_mri") / FIXED_CASE
_LANDMARKS = Path("configs/nyul_standard_v1.npy")
GOLDEN_SHA256 = "df81e48753554458e7d9138c6d6bc06dfdc7090cfb65921fa652be664bbbeece"


@pytest.mark.leakage
@pytest.mark.skipif(not _FIXTURE.exists(), reason="Duke fixture not present (data/ gitignored)")
@pytest.mark.skipif(not _LANDMARKS.exists(), reason="frozen Nyul landmarks not present")
def test_preprocessing_golden_output():
    pytest.importorskip("SimpleITK")
    pytest.importorskip("intensity_normalization")
    from intensity_normalization.normalizers.population.nyul import NyulNormalizer

    from pinksight.data.annotation_boxes import load_boxes
    from pinksight.data.preprocess import golden_digest, preprocess_patient

    nyul = NyulNormalizer()
    nyul.load_standard_histogram(str(_LANDMARKS))  
    out = preprocess_patient(_FIXTURE, load_boxes()[FIXED_CASE], nyul)

    assert out.shape == (4, 73, 46, 62)  
    assert golden_digest(out) == GOLDEN_SHA256, "preprocessing output drifted from the golden hash"

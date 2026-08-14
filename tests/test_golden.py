"""Golden-output test (manual §10.2 testing pyramid): re-run the P03 preprocessing chain on a
fixed case and assert it matches a stored hash, so a silent change to N4 / resample / Nyul / crop
fails CI loudly.

The hash is pinned to the chain + the frozen Nyul landmarks (configs/nyul_standard_v1.npy, fit on
the FULL dev split — 613 patients, LOCK-2 train-only, via scripts/fit_nyul_dev.py) +
SimpleITK 2.5.5 / intensity-normalization 3.0.1. Regenerate with:
    uv run python scripts/build_golden_fixture.py   # prints the new GOLDEN_SHA256
data/ is gitignored, so this skips on a fresh clone (like the real-manifest split checks).

Golden regenerated 2026-07-09: commit ff41271 (2026-06-29) intentionally refit the Nyul landmarks
from the earlier degenerate 2-patient fixture to the full 613-patient dev split and rewrote series
classification in phase_stack.py — both legitimately changed the deterministic preprocessing output.
Verified deterministic (hash stable across repeated builds), no dependency drift. See the
[P03-GOLDEN] entry in decisions.md.
"""

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
    nyul.load_standard_histogram(str(_LANDMARKS))  # frozen landmarks, not a re-fit
    out = preprocess_patient(_FIXTURE, load_boxes()[FIXED_CASE], nyul)

    assert out.shape == (4, 73, 46, 62)  # pre + 3 post phases as channels, ROI+margin crop
    assert golden_digest(out) == GOLDEN_SHA256, "preprocessing output drifted from the golden hash"

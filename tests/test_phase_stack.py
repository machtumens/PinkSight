"""Phase-stack selector checks — synthetic (CI) + real fixture (skip-on-clone).

data/ is gitignored, so the real-fixture test skips on a fresh clone; the synthetic
test exercises the classification + ordering logic in CI today.
"""

from pathlib import Path

import pytest

from pinksight.data.phase_stack import select_phase_stack

# real Duke series names are non-uniform across patients — the selector must tolerate all
SYNTHETIC = {
    "Breast_MRI_042": ["2-ax 3d t1 bilateral-24558", "6-ax dyn pre-12763",
                       "7-ax dyn 1st pass-45961", "10-ax dyn 2nd pass-46620",
                       "13-ax dyn 3rd pass-88122"],
    "Breast_MRI_043": ["3-ax 3d dyn pre-69878", "5-ax 3d dyn 1st pass-93190",
                       "6-ax 3d dyn 2nd pass-36242", "7-ax 3d dyn 3rd pass-30956",
                       "8-ax 3d dyn 4th pass-17280", "11-ax t1 +c-69949"],
    "Breast_MRI_044": ["2-ax t1 tse-72943", "4-ax dyn pre-14564",
                       "5-ax dyn 1st pass-50525", "8-ax dyn 2nd pass-88806",
                       "13-ax dyn 3rd pass-19814", "16-ax dyn 4th pass-57432"],
}
EXPECTED_POSTS = {"Breast_MRI_042": 3, "Breast_MRI_043": 4, "Breast_MRI_044": 4}


def _build(tmp_path: Path, patient: str, series: list[str]) -> Path:
    """Mimic TCIA layout: patient/study/{num-desc-uid}/*.dcm + a duplicate numeric tree."""
    study = tmp_path / patient / "1990-01-01-STUDY-99999"
    numeric = tmp_path / patient / "99999"  # duplicate UID-named tree — must be ignored
    for name in series:
        uid = name.rsplit("-", 1)[1]
        (study / name).mkdir(parents=True)
        (study / name / "a.dcm").write_bytes(b"")
        (numeric / uid).mkdir(parents=True)
        (numeric / uid / "a.dcm").write_bytes(b"")
    return tmp_path / patient


@pytest.mark.parametrize("patient", list(SYNTHETIC))
def test_synthetic_phase_stack(tmp_path, patient):
    ps = select_phase_stack(_build(tmp_path, patient, SYNTHETIC[patient]))
    assert "pre" in ps.pre.description.lower()
    assert len(ps.posts) == EXPECTED_POSTS[patient]
    # posts are temporally ordered, and the stack leads with the pre-contrast baseline
    pass_nums = [int(s.description.split()[-2][:-2]) for s in ps.posts]  # "...1st pass" -> 1
    assert pass_nums == sorted(pass_nums)
    assert ps.stack[0] is ps.pre and len(ps.stack) == 1 + len(ps.posts)


def test_pre_contrast_required(tmp_path):
    """No pre-contrast series -> loud failure, never a silent wrong baseline."""
    with pytest.raises(ValueError, match="pre-contrast"):
        select_phase_stack(_build(tmp_path, "Breast_MRI_999", ["7-ax dyn 1st pass-1"]))


_FIXTURE = Path("data/raw/duke_breast_cancer_mri")


@pytest.mark.skipif(not _FIXTURE.exists(), reason="Duke fixture not present (data/ gitignored)")
@pytest.mark.parametrize("patient", list(EXPECTED_POSTS))
def test_real_fixture_phase_stack(patient):
    ps = select_phase_stack(_FIXTURE / patient)
    assert len(ps.posts) == EXPECTED_POSTS[patient]
    assert "pre" in ps.pre.description.lower()

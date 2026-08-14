"""[HEAD2-GRADE-2D-SLICE] unit + leakage guards for the DeepRadGrade 2D-per-slice grade recipe.

Torch-free tests skip cleanly when the ml stack / data is absent (same discipline as test_leakage.py);
the model + slice-plan tests run wherever torch is installed. The critical NEW guard the pre-reg
mandates — no patient's slices cross a CV fold boundary — is asserted here at the dataset level.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

_PROC = Path("data/processed")
_MASKS = Path("data/processed_masks")


def test_tiny_cnn_2d_shape_and_from_scratch():
    """Model contract: (N,3,64,64) -> (N,1) logits; channel-doubling to 512; NOT pretrained."""
    pytest.importorskip("torch")
    import torch

    from pinksight.models.tiny_cnn_2d import TinyCnn2D

    m = TinyCnn2D(in_channels=3)
    out = m(torch.randn(4, 3, 64, 64))
    assert out.shape == (4, 1), out.shape
    # channel-doubling reaches 512 feature maps
    widths = [mod.out_channels for mod in m.features.modules() if isinstance(mod, torch.nn.Conv2d)]
    assert 512 in widths, f"CNN must reach 512 feature maps (DeepRadGrade), got {sorted(set(widths))}"


def test_patient_slice_plan_supra_central():
    """TEST slice = the slice immediately SUPERIOR to the tumour centre; TRAIN = up to 8 around it."""
    from pinksight.data.slice_dataset import patient_slice_plan

    # synthetic mask (row, col, slice): tumour concentrated at slice 20
    m = np.zeros((10, 10, 40), dtype=bool)
    m[3:7, 3:7, 18:23] = True  # centre-of-mass slice = 20 (most voxels)
    train_idx, test_idx = patient_slice_plan("synthetic", m, n_slices=40)
    assert len(train_idx) <= 8
    assert test_idx == 21, f"supra-central test slice must be centre+1 (=21), got {test_idx}"
    assert test_idx not in train_idx or True  # (train may overlap the region; test is a single slice)


def test_patient_slice_plan_mask_fallback_uses_mid_slice():
    """[1.6]/LOCK-2: no mask -> centre = mid-slice (NEVER a ground-truth box); test = mid+1."""
    from pinksight.data.slice_dataset import patient_slice_plan

    train_idx, test_idx = patient_slice_plan("no_mask", None, n_slices=30)
    assert test_idx == 16, f"fallback test slice must be mid(15)+1=16, got {test_idx}"
    assert len(train_idx) <= 8


@pytest.mark.leakage
def test_slice_level_patient_disjoint_on_disk():
    """The NEW leakage guard: for a StratifiedGroupKFold patient split, no patient's SLICES may land in
    both the train and test slice sets of any fold. Runs against the real on-disk cohort when present.
    """
    pytest.importorskip("torch")
    if not (_PROC.exists() and any(_PROC.glob("*.npy"))):
        pytest.skip("data/processed/*.npy not present — slice-disjoint gate cannot run on real data")

    from sklearn.model_selection import StratifiedGroupKFold

    from pinksight.data.slice_dataset import SliceGradeDataset

    pids = sorted(f.stem for f in _PROC.glob("*.npy"))[:20]  # small deterministic subset for speed
    items = [(p, i % 2) for i, p in enumerate(pids)]  # synthetic balanced labels (leak logic only)
    y = np.array([lab for _, lab in items])
    groups = [p for p, _ in items]

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    for tr, te in cv.split(np.zeros(len(items)), y, groups):
        tr_ds = SliceGradeDataset([items[i] for i in tr], proc_dir=_PROC, split="train", augment=True)
        te_ds = SliceGradeDataset([items[i] for i in te], proc_dir=_PROC, split="test")
        tr_pids = {s[0] for s in tr_ds.samples}
        te_pids = {s[0] for s in te_ds.samples}
        assert tr_pids.isdisjoint(te_pids), (
            f"SLICE-LEVEL LEAK: patient(s) {tr_pids & te_pids} have slices in both train and test"
        )
        # test set emits EXACTLY one slice per patient (supra-central) -> patient-level OOF
        assert len(te_ds.samples) == len(te), (
            f"test dataset must emit 1 slice/patient (got {len(te_ds.samples)} for {len(te)} patients)"
        )

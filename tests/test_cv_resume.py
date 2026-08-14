"""Fold-level resume for cross_val_imaging (the Kaggle-survival contract).

GPU-free: monkeypatches train_model with a deterministic stub so we test the CHECKPOINT/RESUME
plumbing, not the CNN. Proves: (1) a second run with the same ckpt_dir retrains ZERO folds and
returns identical metrics; (2) ckpt_dir=None is unaffected; (3) a config change refuses to reuse
stale preds; (4) weights_dir persists a loadable per-fold state_dict, and refuses to resume a run
that was trained without one. This is the thing that must hold, or a Kaggle session death loses
hours — or a paid GPU run finishes with nothing for Grad-CAM to attribute over.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest

pytest.importorskip("torch")

import torch
from torch import nn

from pinksight.train import cv as cvmod
from pinksight.train.loop import TrainCfg

CALLS: list[str] = []


class _TinyNet(nn.Module):
    """Stand-in for the 3D-ResNet: real parameters, so a saved state_dict round-trips for real."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 1)


def _stub_train_model(model, train_loader, val_loader, cfg, pos_weight, log_tag, oof_loader=None):
    """Deterministic stand-in for the real 3D-CNN fit: derive pids from the loader's dataset, emit
    reproducible pseudo-probs. Records each call so the test can count retrains. Mirrors the honest-OOF
    contract: when `oof_loader` is given, return ITS fold's pids/probs (not the val fold's), and the
    best-on-val `state_dict` as its first return value (what weights_dir persists)."""
    CALLS.append(log_tag)
    src = oof_loader if oof_loader is not None else val_loader
    out_pids = [pid for pid, _ in src.dataset.items]
    # deterministic, label-correlated so AUROC is well-defined and stable across runs
    lbl = {pid: lab for pid, lab in src.dataset.items}
    probs = np.array([0.25 + 0.5 * lbl[p] for p in out_pids], dtype=float)
    best_state = copy.deepcopy(model.state_dict()) if model is not None else None
    return best_state, probs, out_pids


@pytest.fixture(autouse=True)
def _patch(monkeypatch):
    CALLS.clear()
    monkeypatch.setattr(cvmod, "train_model", _stub_train_model)
    # set_seed / model_factory are cheap no-ops here
    monkeypatch.setattr(cvmod, "set_seed", lambda s: None)


def _items(n=40):
    # balanced, patient-disjoint (unique pids) so StratifiedGroupKFold is happy
    return [(f"p{i:03d}", i % 2) for i in range(n)]


def _run(ckpt_dir):
    return cvmod.cross_val_imaging(
        _items(), TrainCfg(epochs=1, batch_size=4, device="cpu"),
        model_factory=lambda: None, seeds=(0, 1, 2), ckpt_dir=ckpt_dir,
    )


def test_resume_retrains_zero_folds_and_matches(tmp_path):
    first = _run(tmp_path)
    n_first = len(CALLS)
    assert n_first == 15, f"expected 3 seeds x 5 folds = 15 trains, got {n_first}"

    CALLS.clear()
    second = _run(tmp_path)  # same ckpt_dir → every fold loads from disk
    assert CALLS == [], f"resume retrained folds it should have loaded: {CALLS}"
    assert second["auroc_pooled_oof_mean"] == first["auroc_pooled_oof_mean"]
    assert second["delong_ci95_mean"] == first["delong_ci95_mean"]


def test_none_ckpt_is_unaffected():
    m = _run(None)
    assert len(CALLS) == 15 and "auroc_pooled_oof_mean" in m


def test_weights_dir_writes_loadable_state_dicts(tmp_path):
    """G5-LEG3: every fold leaves a .pt that load_state_dict accepts strictly — no missing/unexpected
    keys. Without this the paid GPU run ends with probs but no weights, and Grad-CAM has nothing."""
    cvmod.cross_val_imaging(
        _items(), TrainCfg(epochs=1, batch_size=4, device="cpu"),
        model_factory=_TinyNet, seeds=(0, 1, 2), weights_dir=tmp_path,
    )
    pts = sorted(tmp_path.glob("model_s*f*.pt"))
    assert len(pts) == 15, f"expected 3 seeds x 5 folds = 15 weight files, got {len(pts)}"
    state = torch.load(pts[0], map_location="cpu", weights_only=True)
    missing, unexpected = _TinyNet().load_state_dict(state, strict=True)
    assert not missing and not unexpected


def test_weights_dir_refuses_partial_resume(tmp_path):
    """Resuming a probs-only run with weights_dir set would silently attribute over whichever folds
    happened to retrain. Refuse instead of back-filling (same rule as embed_dir)."""
    ck = tmp_path / "ck"
    cvmod.cross_val_imaging(  # first run: probs only, no weights
        _items(), TrainCfg(epochs=1, batch_size=4, device="cpu"),
        model_factory=_TinyNet, seeds=(0, 1, 2), ckpt_dir=ck,
    )
    with pytest.raises(RuntimeError, match="weights cannot be back-filled|back-filled from probs"):
        cvmod.cross_val_imaging(
            _items(), TrainCfg(epochs=1, batch_size=4, device="cpu"),
            model_factory=_TinyNet, seeds=(0, 1, 2), ckpt_dir=ck, weights_dir=tmp_path / "w",
        )


def test_config_change_refuses_stale_ckpt(tmp_path):
    _run(tmp_path)
    with pytest.raises(RuntimeError, match="config mismatch"):
        # different batch size → different training → must not pool with the old preds
        cvmod.cross_val_imaging(
            _items(), TrainCfg(epochs=1, batch_size=8, device="cpu"),
            model_factory=lambda: None, seeds=(0, 1, 2), ckpt_dir=tmp_path,
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

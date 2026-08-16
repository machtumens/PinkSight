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

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 1)


def _stub_train_model(model, train_loader, val_loader, cfg, pos_weight, log_tag, oof_loader=None):
    CALLS.append(log_tag)
    src = oof_loader if oof_loader is not None else val_loader
    out_pids = [pid for pid, _ in src.dataset.items]
    lbl = {pid: lab for pid, lab in src.dataset.items}
    probs = np.array([0.25 + 0.5 * lbl[p] for p in out_pids], dtype=float)
    best_state = copy.deepcopy(model.state_dict()) if model is not None else None
    return best_state, probs, out_pids


@pytest.fixture(autouse=True)
def _patch(monkeypatch):
    CALLS.clear()
    monkeypatch.setattr(cvmod, "train_model", _stub_train_model)
    monkeypatch.setattr(cvmod, "set_seed", lambda s: None)


def _items(n=40):
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
    second = _run(tmp_path)  
    assert CALLS == [], f"resume retrained folds it should have loaded: {CALLS}"
    assert second["auroc_pooled_oof_mean"] == first["auroc_pooled_oof_mean"]
    assert second["delong_ci95_mean"] == first["delong_ci95_mean"]


def test_none_ckpt_is_unaffected():
    m = _run(None)
    assert len(CALLS) == 15 and "auroc_pooled_oof_mean" in m


def test_weights_dir_writes_loadable_state_dicts(tmp_path):
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
    ck = tmp_path / "ck"
    cvmod.cross_val_imaging(  
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
        cvmod.cross_val_imaging(
            _items(), TrainCfg(epochs=1, batch_size=8, device="cpu"),
            model_factory=lambda: None, seeds=(0, 1, 2), ckpt_dir=tmp_path,
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

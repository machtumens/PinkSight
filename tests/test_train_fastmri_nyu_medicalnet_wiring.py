
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"


def _write_fixtures(proc: Path, n_per_class: int, spatial: int, start: int = 0) -> list[tuple[str, int]]:
    import numpy as np

    items: list[tuple[str, int]] = []
    idx = start
    for label in (1, 0):  
        for _ in range(n_per_class):
            pid = f"fastMRI_breast_{idx:03d}"
            vol = np.random.default_rng(idx).standard_normal((4, spatial, spatial, spatial)).astype("float32")
            np.save(proc / f"{pid}.npy", vol)
            items.append((pid, label))
            idx += 1
    return items


def test_run_hchar_seed_threads_medicalnet_and_lr_flags(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("monai")
    import numpy as np
    from torch import nn

    sys.path.insert(0, str(_ROOT / "scripts"))
    sys.path.insert(0, str(_SRC))
    import train_fastmri_nyu as mod

    spatial = 8
    proc = tmp_path / "proc"
    proc.mkdir()
    train_items = _write_fixtures(proc, n_per_class=10, spatial=spatial, start=0)   
    test_items = _write_fixtures(proc, n_per_class=3, spatial=spatial, start=100)   

    monkeypatch.setattr(mod, "PROC", proc)
    monkeypatch.setattr(mod, "CROP_MODE", "none")

    enc_kwargs: list[dict] = []

    class _SpyEncoder(nn.Module):
        def __init__(self, in_channels, depth=18, medicalnet_weights=None, freeze_bn=False):
            super().__init__()
            self.embed_dim = 512
            enc_kwargs.append(dict(in_channels=in_channels, depth=depth,
                                   medicalnet_weights=medicalnet_weights, freeze_bn=freeze_bn))
            self._p = nn.Linear(1, 1)  

        def forward(self, x):  
            return x

    monkeypatch.setattr(mod, "MriEncoder", _SpyEncoder)

    captured: dict = {}

    def _fake_train_model(model, train_loader, val_loader, cfg, pos_weight, log_tag="", oof_loader=None):
        captured["cfg"] = cfg
        captured["model"] = model
        pids = [pid for pid, _ in test_items]
        return None, np.zeros(len(pids), dtype=float), pids

    monkeypatch.setattr(mod, "train_model", _fake_train_model)

    sentinel = str(tmp_path / "fake_medicalnet.pth")  
    out = mod._run_hchar_seed(
        0, train_items, test_items, "kinetic", spatial, 4, 1, "cpu",
        medicalnet=sentinel, backbone_lr=1e-4, head_lr=1e-3, scheduler="cosine",
    )

    assert set(out) == {pid for pid, _ in test_items}

    assert enc_kwargs, "MriEncoder was never constructed"
    hchar_enc = enc_kwargs[0]
    assert hchar_enc["medicalnet_weights"] == sentinel
    assert hchar_enc["in_channels"] == 4          
    assert hchar_enc["depth"] == 18
    assert hchar_enc["freeze_bn"] is False         

    cfg = captured["cfg"]
    assert cfg.backbone_lr == 1e-4
    assert cfg.head_lr == 1e-3
    assert cfg.scheduler == "cosine"
    assert cfg.loss == "focal"
    assert cfg.amp is False
    assert cfg.grad_clip == 1.0


from __future__ import annotations

import copy
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class TrainCfg:
    epochs: int = 60
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 4
    patience: int = 10
    device: str = "cpu"
    log_dir: Path | None = None
    loss: str = "bce"
    focal_gamma: float = 2.0
    accum_steps: int = 1  
    amp: bool = True
    grad_clip: float = 0.0
    backbone_lr: float | None = None
    head_lr: float | None = None
    scheduler: str = "none"


def _focal_loss_with_logits(
    logits: torch.Tensor, target: torch.Tensor, pos_weight: float, gamma: float
) -> torch.Tensor:
    bce = nn.functional.binary_cross_entropy_with_logits(
        logits, target, pos_weight=torch.tensor([pos_weight], device=logits.device), reduction="none"
    )
    if gamma == 0.0:
        return bce.mean()
    with torch.autocast(device_type="cuda", enabled=False):
        logits_f = logits.float()
        target_f = target.float()
        p = torch.sigmoid(logits_f)
        p_t = p * target_f + (1.0 - p) * (1.0 - target_f)  
        modulator = (1.0 - p_t) ** gamma
    return (modulator * bce).mean()


def _eval(
    model: nn.Module, loader: DataLoader, device: str, amp: bool = True
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    model.eval()
    use_amp = amp and device.startswith("cuda")
    probs, ys, pids = [], [], []
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
        for x, y, pid in loader:
            p = torch.sigmoid(model(x.to(device))).float().cpu().numpy().ravel()
            probs.append(p)
            ys.append(y.numpy().ravel())
            pids.extend(pid)
    return np.concatenate(probs), np.concatenate(ys), pids


def _score(y: np.ndarray, probs: np.ndarray) -> float:
    if not np.all(np.isfinite(probs)):
        return -np.inf
    if len(set(y.tolist())) > 1:
        return float(roc_auc_score(y, probs))
    return -float(np.mean((probs - y) ** 2))  


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainCfg,
    pos_weight: float,
    log_tag: str = "",
    oof_loader: DataLoader | None = None,
) -> tuple[dict, np.ndarray, list[str]]:
    device = cfg.device
    model = model.to(device)
    if cfg.backbone_lr is not None:
        head_params, backbone_params = [], []
        for name, p in model.named_parameters():
            (head_params if ".head." in f".{name}." else backbone_params).append(p)
        param_groups = [
            {"params": backbone_params, "lr": cfg.backbone_lr},
            {"params": head_params, "lr": cfg.head_lr if cfg.head_lr is not None else cfg.lr},
        ]
        opt = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, cfg.epochs))
        if cfg.scheduler == "cosine" else None
    )
    if cfg.scheduler not in ("none", "cosine"):
        raise ValueError(f"TrainCfg.scheduler must be 'none' or 'cosine', got {cfg.scheduler!r}")
    if cfg.loss == "focal":
        def loss_fn(out, tgt):
            return _focal_loss_with_logits(out, tgt, pos_weight, cfg.focal_gamma)
    elif cfg.loss == "bce":
        _bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
        loss_fn = _bce
    else:
        raise ValueError(f"TrainCfg.loss must be 'bce' or 'focal', got {cfg.loss!r}")
    accum_steps = max(1, int(cfg.accum_steps))
    use_amp = cfg.amp and device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    grad_clip = float(cfg.grad_clip)

    best_score, best_state, best_probs, best_pids = -np.inf, None, None, None
    stale, rows = 0, []
    for epoch in range(cfg.epochs):
        model.train()
        opt.zero_grad()
        n_batches = len(train_loader)
        for i, (x, y, _) in enumerate(train_loader):
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(x.to(device))
                loss = loss_fn(out, y.to(device).unsqueeze(1))
                if accum_steps > 1:
                    loss = loss / accum_steps
            scaler.scale(loss).backward()
            is_step = ((i + 1) % accum_steps == 0) or ((i + 1) == n_batches)
            if is_step:
                if grad_clip > 0.0:
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()

        if scheduler is not None:  
            scheduler.step()

        probs, ys, pids = _eval(model, val_loader, device, amp=cfg.amp)
        score = _score(ys, probs)
        rows.append({"epoch": epoch, "tag": log_tag, "val_score": round(float(score), 4)})

        if score > best_score:
            best_score, stale = score, 0
            best_state = copy.deepcopy(model.state_dict())
            best_probs, best_pids = probs, pids
        else:
            stale += 1
            if stale >= cfg.patience:
                break

    if cfg.log_dir is not None and rows:
        cfg.log_dir.mkdir(parents=True, exist_ok=True)
        f = cfg.log_dir / f"epochs_{log_tag or 'run'}.csv"
        with f.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    if oof_loader is not None:
        if best_state is not None:
            model.load_state_dict(best_state)
        oof_probs, _, oof_pids = _eval(model, oof_loader, device, amp=cfg.amp)
        return best_state, oof_probs, oof_pids
    return best_state, best_probs, best_pids


if __name__ == "__main__":  
    torch.manual_seed(0)
    _logits = torch.randn(64, 1)
    _target = (torch.rand(64, 1) > 0.5).float()

    for _pw in (1.0, 3.0):
        _bce_ref = nn.functional.binary_cross_entropy_with_logits(
            _logits, _target, pos_weight=torch.tensor([_pw])
        )
        _focal0 = _focal_loss_with_logits(_logits, _target, pos_weight=_pw, gamma=0.0)
        assert torch.allclose(_bce_ref, _focal0, atol=1e-6), (_pw, _bce_ref.item(), _focal0.item())

    _focal2 = _focal_loss_with_logits(_logits, _target, pos_weight=1.0, gamma=2.0)
    _bce1 = nn.functional.binary_cross_entropy_with_logits(_logits, _target)
    assert _focal2.item() < _bce1.item(), (_focal2.item(), _bce1.item())

    def _grad_of(accum: int) -> torch.Tensor:
        m = nn.Linear(4, 1)
        torch.manual_seed(1)
        with torch.no_grad():
            m.weight.copy_(torch.zeros_like(m.weight))
            m.bias.copy_(torch.zeros_like(m.bias))
        x = torch.arange(8 * 4, dtype=torch.float32).reshape(8, 4) / 32.0
        y = (torch.arange(8, dtype=torch.float32) % 2).reshape(8, 1)
        m.zero_grad()
        if accum == 1:
            loss = nn.functional.binary_cross_entropy_with_logits(m(x), y)
            loss.backward()
        else:
            for xb, yb in ((x[:4], y[:4]), (x[4:], y[4:])):
                (nn.functional.binary_cross_entropy_with_logits(m(xb), yb) / 2).backward()
        return m.weight.grad.clone()

    assert torch.allclose(_grad_of(1), _grad_of(2), atol=1e-6)

    _y = np.array([0, 1, 0, 1])
    assert _score(_y, np.array([np.nan, 0.6, 0.4, 0.7])) == -np.inf
    assert _score(_y, np.array([0.1, np.inf, 0.4, 0.7])) == -np.inf
    assert np.isfinite(_score(_y, np.array([0.1, 0.6, 0.4, 0.7])))  

    print("OK: focal@gamma0==BCE, focal@gamma2<BCE, grad-accum(2)==full-batch, _score(NaN)=-inf")  

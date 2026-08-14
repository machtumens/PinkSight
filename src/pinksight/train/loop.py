"""P06: the reusable PyTorch training loop — trains ANY nn.Module producing (N,1) logits.

ponytail: a plain AdamW + BCEWithLogitsLoss loop with early-stop on val AUROC and best-state kept in
memory. No Lightning, no scheduler, no AMP/DDP, no disk checkpoint — add those only when a real run
shows the need. The recipe (AdamW lr=1e-3 wd=1e-4, pos_weight=neg/pos, sigmoid at eval) mirrors the
proven clinical baseline (`models/clinical_encoder.py:153-168`) so the imaging arm is comparable.
"""

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
    # FIX-2 (flag-gated, defaults preserve byte-identical prior behavior):
    #   loss="bce"  -> BCEWithLogitsLoss + pos_weight (unchanged default path)
    #   loss="focal"-> pre-registered focal loss ("Head 1 subtype (focal)"), pos_weight becomes alpha
    loss: str = "bce"
    focal_gamma: float = 2.0
    accum_steps: int = 1  # grad-accumulation: 1 == current single-step behavior
    # FIX-3 (fp16 NaN hardening; defaults preserve byte-identical prior behavior):
    #   amp=True   -> fp16 autocast + GradScaler on CUDA (unchanged default; ~2x throughput, half VRAM)
    #   amp=False  -> pure fp32 for BOTH train and _eval (autocast off, scaler no-op) — kills the fp16
    #                 overflow -> BatchNorm-corruption -> NaN-logits crash for unattended local runs.
    amp: bool = True
    #   grad_clip>0 -> max-norm gradient clipping each optimizer step (post-unscale under AMP). Off (0.0)
    #                 == byte-identical; a modest clip (e.g. 1.0) caps the exploding grads that seed NaNs.
    grad_clip: float = 0.0
    # [G2-SUBTRACTION-REOPEN] recipe correction (flag-gated; ALL defaults preserve byte-identical prior
    # behavior — the primary fault was a single 1e-3 LR on the 33M-param pretrained backbone).
    #   backbone_lr=None -> single AdamW param group at `lr` (UNCHANGED default path).
    #   backbone_lr=1e-4 -> two param groups: backbone (all params NOT under `.head.`) at backbone_lr,
    #                       head (params under `.head.`) at head_lr (or `lr` if head_lr is None).
    backbone_lr: float | None = None
    head_lr: float | None = None
    #   scheduler="none"  -> no LR schedule (UNCHANGED default). "cosine" -> CosineAnnealingLR(T_max=epochs),
    #                        stepped once per epoch (prevents late-epoch LR-induced instability, no warmup).
    scheduler: str = "none"


def _focal_loss_with_logits(
    logits: torch.Tensor, target: torch.Tensor, pos_weight: float, gamma: float
) -> torch.Tensor:
    """Binary focal loss on logits. At gamma=0 this reduces EXACTLY to pos_weight-BCEWithLogits.

    focal = alpha_t * (1 - p_t)**gamma * BCE, with p_t = p if y==1 else 1-p. The alpha term reuses
    `pos_weight` (weight applied to the positive class) so the class-balancing knob is preserved.
    """
    bce = nn.functional.binary_cross_entropy_with_logits(
        logits, target, pos_weight=torch.tensor([pos_weight], device=logits.device), reduction="none"
    )
    if gamma == 0.0:
        return bce.mean()
    # TICKET-002: compute the modulating factor in fp32 OUTSIDE autocast. Under AMP fp16, a saturated
    # logit rounds p_t to exactly 1.0 -> (1-p_t)**gamma = 0 and fp16 intermediates overflow -> NaN.
    # BCEWithLogits above is autocast-safe (log-sum-exp internally); only the sigmoid/p_t math needs fp32.
    with torch.autocast(device_type="cuda", enabled=False):
        logits_f = logits.float()
        target_f = target.float()
        p = torch.sigmoid(logits_f)
        p_t = p * target_f + (1.0 - p) * (1.0 - target_f)  # prob of the true class
        modulator = (1.0 - p_t) ** gamma
    return (modulator * bce).mean()


def _eval(
    model: nn.Module, loader: DataLoader, device: str, amp: bool = True
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (probs, y, pids) over a loader. One pass, no grad. AMP autocast on CUDA when `amp` (no-op
    on CPU). `amp=False` forces pure fp32 eval — needed for local runs where fp16 overflow corrupts
    BatchNorm running-stats and yields NaN probs that crash roc_auc_score downstream."""
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
    """Early-stop metric: val AUROC when both classes present, else -Brier (tiny single-class fold).

    FIX-3 (ALWAYS ON, no flag): a non-finite prob (NaN/inf) from an fp16-overflowed epoch returns -inf
    instead of crashing roc_auc_score. Early-stop then skips the bad epoch and KEEPS the last good
    best_state, so one blown epoch/fold degrades gracefully instead of killing an unattended run."""
    if not np.all(np.isfinite(probs)):
        return -np.inf
    if len(set(y.tolist())) > 1:
        return float(roc_auc_score(y, probs))
    return -float(np.mean((probs - y) ** 2))  # negative Brier — higher is better, no logit gymnastics


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainCfg,
    pos_weight: float,
    log_tag: str = "",
    oof_loader: DataLoader | None = None,
) -> tuple[dict, np.ndarray, list[str]]:
    """Train to best val score; return (best_state_dict, probs, pids).

    Early stopping selects the best epoch on `val_loader`. If `oof_loader` is given, the returned
    probs/pids are that best model's predictions on `oof_loader` — a fold the loop NEVER selected
    against, so they're honest out-of-fold preds (no test-set peeking). `oof_loader=None` keeps the
    old contract (returns the peeked val preds) — used only by the tiny-N smoke path where there is
    no room to carve an inner val split.
    """
    device = cfg.device
    model = model.to(device)
    # [G2-SUBTRACTION-REOPEN] optimizer param groups (flag-gated). Default (backbone_lr is None) is the
    # UNCHANGED single-group AdamW at cfg.lr. When backbone_lr is set, split into backbone vs head groups
    # so the pretrained backbone fine-tunes at a low LR (e.g. 1e-4) while the fresh head trains faster
    # (e.g. 1e-3). FILTER on ".head." ONLY (contract C2): SubtypeClassifier's head is `self.head`
    # (param names "head.weight"/"head.bias"); MriEncoder has NO ".fc" (feed_forward=False). A ".fc."
    # filter would silently push every param into the backbone group.
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
    # Cosine annealing LR schedule (flag-gated; "none" is a no-op, byte-identical). Stepped per EPOCH.
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, cfg.epochs))
        if cfg.scheduler == "cosine" else None
    )
    if cfg.scheduler not in ("none", "cosine"):
        raise ValueError(f"TrainCfg.scheduler must be 'none' or 'cosine', got {cfg.scheduler!r}")
    # Loss selection (flag-gated). Default "bce" is the unchanged pos_weight BCE path.
    if cfg.loss == "focal":
        def loss_fn(out, tgt):
            return _focal_loss_with_logits(out, tgt, pos_weight, cfg.focal_gamma)
    elif cfg.loss == "bce":
        _bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
        loss_fn = _bce
    else:
        raise ValueError(f"TrainCfg.loss must be 'bce' or 'focal', got {cfg.loss!r}")
    accum_steps = max(1, int(cfg.accum_steps))
    # AMP: fp16 autocast + loss scaling on CUDA (T4 tensor cores) — ~2x throughput, half VRAM.
    # enabled=False on CPU makes scaler/autocast no-ops, so the CPU path is byte-identical to before.
    # FIX-3: cfg.amp=False forces pure fp32 (scaler/autocast off) — the local-run NaN escape hatch.
    use_amp = cfg.amp and device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    grad_clip = float(cfg.grad_clip)

    best_score, best_state, best_probs, best_pids = -np.inf, None, None, None
    stale, rows = 0, []
    for epoch in range(cfg.epochs):
        model.train()
        # Grad-accumulation (accum_steps==1 => byte-identical to prior per-batch step). We divide the
        # loss by accum_steps so the accumulated gradient magnitude matches a single larger batch, then
        # scaler.step/update/zero_grad fire once per accum window (correct AMP ordering).
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
                # FIX-3: max-norm grad clip (off when grad_clip==0.0). Under AMP the grads must be
                # unscaled FIRST (scaler.unscale_ is idempotent per step) so the clip norm is measured
                # in real units, not fp16-scaled units; then scaler.step. In fp32 use_amp is False so
                # unscale_ is a cheap no-op and clip_grad_norm_ runs on the raw grads.
                if grad_clip > 0.0:
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()

        if scheduler is not None:  # [G2-SUBTRACTION-REOPEN] cosine anneal, stepped once per epoch
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

    # Honest OOF: predict the held-out test fold ONCE with the best-on-val model. The loop never
    # early-stopped on this fold, so these preds are unpeeked (fixes the real≈shuffle bug).
    if oof_loader is not None:
        if best_state is not None:
            model.load_state_dict(best_state)
        oof_probs, _, oof_pids = _eval(model, oof_loader, device, amp=cfg.amp)
        return best_state, oof_probs, oof_pids
    return best_state, best_probs, best_pids


if __name__ == "__main__":  # runnable check: `python -m pinksight.train.loop`
    torch.manual_seed(0)
    _logits = torch.randn(64, 1)
    _target = (torch.rand(64, 1) > 0.5).float()

    # (1) focal @ gamma=0 reduces EXACTLY to pos_weight BCEWithLogits.
    for _pw in (1.0, 3.0):
        _bce_ref = nn.functional.binary_cross_entropy_with_logits(
            _logits, _target, pos_weight=torch.tensor([_pw])
        )
        _focal0 = _focal_loss_with_logits(_logits, _target, pos_weight=_pw, gamma=0.0)
        assert torch.allclose(_bce_ref, _focal0, atol=1e-6), (_pw, _bce_ref.item(), _focal0.item())

    # (2) gamma>0 down-weights easy (confident-correct) examples => strictly smaller than BCE.
    _focal2 = _focal_loss_with_logits(_logits, _target, pos_weight=1.0, gamma=2.0)
    _bce1 = nn.functional.binary_cross_entropy_with_logits(_logits, _target)
    assert _focal2.item() < _bce1.item(), (_focal2.item(), _bce1.item())

    # (3) grad-accum: accum_steps=2 over 2 half-batches gives ~the same gradient as one full batch.
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

    # (4) FIX-3: _score returns -inf (never crashes) when probs are non-finite — the fp16-NaN escape.
    _y = np.array([0, 1, 0, 1])
    assert _score(_y, np.array([np.nan, 0.6, 0.4, 0.7])) == -np.inf
    assert _score(_y, np.array([0.1, np.inf, 0.4, 0.7])) == -np.inf
    assert np.isfinite(_score(_y, np.array([0.1, 0.6, 0.4, 0.7])))  # finite probs unaffected

    print("OK: focal@gamma0==BCE, focal@gamma2<BCE, grad-accum(2)==full-batch, _score(NaN)=-inf")  # noqa: T201

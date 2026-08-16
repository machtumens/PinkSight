
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class TensorDict(dict):

    def to(self, device) -> "TensorDict":
        return TensorDict({k: v.to(device) for k, v in self.items()})


class FocalLoss(nn.Module):

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = logits.reshape(-1)
        target = target.reshape(-1).float()
        p = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        p_t = p * target + (1 - p) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        loss = alpha_t * (1 - p_t).pow(self.gamma) * ce
        return loss.mean()


class Ki67HuberLoss(nn.Module):

    def __init__(self, delta: float = 10.0) -> None:
        super().__init__()
        self.delta = delta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.reshape(-1)
        target = target.reshape(-1).float()
        mask = ~torch.isnan(target)
        if mask.sum() == 0:  
            return pred.sum() * 0.0
        return F.huber_loss(pred[mask], target[mask], delta=self.delta, reduction="mean")


class KendallUncertaintyWeighting(nn.Module):

    def __init__(self, n_tasks: int = 2) -> None:
        super().__init__()
        self.log_var = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, losses: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        if len(losses) != self.log_var.numel():
            raise ValueError(f"expected {self.log_var.numel()} task losses, got {len(losses)}")
        stacked = torch.stack([loss.reshape(()) for loss in losses])
        active = stacked.detach() != 0.0  
        if not bool(active.any()):
            active = torch.ones_like(active)  
        log_var = self.log_var.clamp(-7.0, 7.0)
        precision = torch.exp(-log_var)
        per_task = precision * stacked + log_var
        total = per_task[active].sum()
        return total, precision.detach()


class CrossAttentionFusion(nn.Module):

    def __init__(self, modality_dims: dict[str, int], fused_dim: int = 128,
                 n_heads: int = 4, attn_dropout: float = 0.0) -> None:
        super().__init__()
        if not modality_dims:
            raise ValueError("need at least one modality")
        if fused_dim % n_heads != 0:
            raise ValueError(f"fused_dim {fused_dim} must be divisible by n_heads {n_heads}")
        self.fused_dim = fused_dim
        self.embed_dim = fused_dim  
        self.modality_names = tuple(modality_dims)
        self.projections = nn.ModuleDict(
            {name: nn.Linear(dim, fused_dim) for name, dim in modality_dims.items()}
        )
        self.attn = nn.MultiheadAttention(
            fused_dim, num_heads=n_heads, dropout=attn_dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(fused_dim)

    def forward(self, feats: dict[str, torch.Tensor], drop: set[str] | None = None,
                return_attn: bool = False):
        drop = drop or set()
        present = [name for name in self.modality_names if name in feats and name not in drop]
        if not present:
            raise ValueError("all modalities dropped/absent — cannot fuse an empty set")

        tokens = []
        for name in present:
            x = feats[name]
            if x.dim() == 1:
                x = x.unsqueeze(0)
            tokens.append(torch.relu(self.projections[name](x)))  
        seq = torch.stack(tokens, dim=1)  

        attended, attn_w = self.attn(seq, seq, seq, need_weights=return_attn,
                                     average_attn_weights=True)
        fused_seq = self.norm(seq + attended)
        fused = fused_seq.mean(dim=1)  
        if return_attn:
            return fused, attn_w  
        return fused


class FusionModel(nn.Module):

    def __init__(self, modality_dims: dict[str, int], fused_dim: int = 128, n_heads: int = 4,
                 p_modality_dropout: float = 0.25, focal_alpha: float = 0.75,
                 focal_gamma: float = 2.0, ki67_delta: float = 10.0) -> None:
        super().__init__()
        self.fusion = CrossAttentionFusion(modality_dims, fused_dim, n_heads)
        self.embed_dim = fused_dim
        self.subtype_head = nn.Linear(fused_dim, 1)
        self.ki67_head = nn.Linear(fused_dim, 1)
        self.p_modality_dropout = p_modality_dropout
        self.modality_names = self.fusion.modality_names
        self.subtype_loss = FocalLoss(focal_alpha, focal_gamma)
        self.ki67_loss = Ki67HuberLoss(ki67_delta)
        self.uncertainty = KendallUncertaintyWeighting(n_tasks=2)

    def _sample_drop(self, feats: dict[str, torch.Tensor]) -> set[str]:
        present = [n for n in self.modality_names if n in feats]
        if len(present) <= 1 or self.p_modality_dropout <= 0:
            return set()
        keep_min = 1
        drop = {n for n in present if torch.rand(()) < self.p_modality_dropout}
        if len(present) - len(drop) < keep_min:  
            spare = present[int(torch.randint(len(present), ()).item())]
            drop.discard(spare)
        return drop

    def forward(self, feats: dict[str, torch.Tensor], drop: set[str] | None = None,
                return_attn: bool = False) -> dict:
        if drop is None and self.training:
            drop = self._sample_drop(feats)
        out = self.fusion(feats, drop=drop, return_attn=return_attn)
        fused, attn = (out if return_attn else (out, None))
        return {
            "subtype_logit": self.subtype_head(fused),
            "ki67": self.ki67_head(fused),
            "fused": fused,
            "attn": attn,
        }

    def joint_loss(self, out: dict, y_subtype: torch.Tensor,
                   y_ki67: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if y_ki67 is None:
            y_ki67 = torch.full_like(out["ki67"].reshape(-1), float("nan"))
        l_subtype = self.subtype_loss(out["subtype_logit"], y_subtype)
        l_ki67 = self.ki67_loss(out["ki67"], y_ki67)
        return self.uncertainty([l_subtype, l_ki67])


class _SubtypeLogitAdapter(nn.Module):

    def __init__(self, fusion_model: FusionModel) -> None:
        super().__init__()
        self.fusion_model = fusion_model
        self.embed_dim = fusion_model.embed_dim

    def forward(self, feats: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.fusion_model(feats)["subtype_logit"]


def subtype_only(fusion_model: FusionModel) -> nn.Module:
    return _SubtypeLogitAdapter(fusion_model)


class HierarchicalStagedFusion(nn.Module):

    def __init__(self, modality_dims: dict[str, int], stage1_dim: int = 256, fused_dim: int = 128,
                 n_heads: int = 4, p_modality_dropout: float = 0.25, focal_alpha: float = 0.75,
                 focal_gamma: float = 2.0, ki67_delta: float = 10.0,
                 imaging_key: str = "mri", clinical_key: str = "clinical",
                 radiomics_key: str = "radiomics") -> None:
        super().__init__()
        if not modality_dims:
            raise ValueError("need at least one modality")
        if imaging_key not in modality_dims:
            raise ValueError(f"imaging_key '{imaging_key}' absent from modality_dims {list(modality_dims)}")
        self.fused_dim = fused_dim
        self.embed_dim = fused_dim  
        self.modality_names = tuple(modality_dims)
        self.imaging_key = imaging_key
        self.clinical_key = clinical_key
        self.radiomics_key = radiomics_key
        self.p_modality_dropout = p_modality_dropout

        self.stage1_proj = nn.Sequential(
            nn.Linear(modality_dims[imaging_key], stage1_dim), nn.ReLU(),
        )
        self.stage1_to_fused = nn.Linear(stage1_dim, fused_dim)

        self._has_radiomics = radiomics_key in modality_dims
        if self._has_radiomics:
            self.stage2_fusion = CrossAttentionFusion(
                {"stage1": fused_dim, radiomics_key: modality_dims[radiomics_key]},
                fused_dim=fused_dim, n_heads=n_heads,
            )

        self._has_clinical = clinical_key in modality_dims
        if self._has_clinical:
            self.clinical_proj = nn.Sequential(
                nn.Linear(modality_dims[clinical_key], fused_dim), nn.ReLU(),
            )
        self.stage3_norm = nn.LayerNorm(fused_dim)

        self.subtype_head = nn.Linear(fused_dim, 1)
        self.grade_head = nn.Linear(fused_dim, 1)   
        self.ki67_head = nn.Linear(fused_dim, 1)    

        self.subtype_loss = FocalLoss(focal_alpha, focal_gamma)
        self.grade_loss = FocalLoss(focal_alpha, focal_gamma)
        self.ki67_loss = Ki67HuberLoss(ki67_delta)
        self.uncertainty = KendallUncertaintyWeighting(n_tasks=2)  

    def _assert_no_forbidden(self, feats: dict[str, torch.Tensor]) -> None:
        from pinksight import FORBIDDEN_FEATURES
        leaked = set(feats) & FORBIDDEN_FEATURES
        if leaked:
            raise ValueError(
                f"LEAKAGE: forbidden feature(s) {sorted(leaked)} passed as a fusion modality — "
                "ER/PR/HER2/Ki-67/Mol-Subtype/Oncotype must never reach the classifier (LOCK-2)."
            )

    def _sample_drop(self, feats: dict[str, torch.Tensor]) -> set[str]:
        present = [n for n in self.modality_names if n in feats]
        if len(present) <= 1 or self.p_modality_dropout <= 0:
            return set()
        drop = {n for n in present if torch.rand(()) < self.p_modality_dropout}
        if len(present) - len(drop) < 1:  
            spare = present[int(torch.randint(len(present), ()).item())]
            drop.discard(spare)
        return drop

    def forward(self, feats: dict[str, torch.Tensor], drop: set[str] | None = None,
                return_attn: bool = False) -> dict:
        self._assert_no_forbidden(feats)
        if drop is None and self.training:
            drop = self._sample_drop(feats)
        drop = drop or set()
        present = {n for n in self.modality_names if n in feats and n not in drop}
        if not present:
            raise ValueError("all modalities dropped/absent — cannot fuse an empty set")

        attn = None
        if self.imaging_key in present:
            x_img = feats[self.imaging_key]
            if x_img.dim() == 1:
                x_img = x_img.unsqueeze(0)
            stage1_rep = self.stage1_to_fused(self.stage1_proj(x_img))  
            ref = x_img
        else:
            ref = feats[self.clinical_key if self.clinical_key in present else next(iter(present))]
            if ref.dim() == 1:
                ref = ref.unsqueeze(0)
            stage1_rep = torch.zeros(ref.shape[0], self.fused_dim, device=ref.device, dtype=ref.dtype)

        if self._has_radiomics and self.radiomics_key in present:
            stage2_in = {"stage1": stage1_rep, self.radiomics_key: feats[self.radiomics_key]}
            out2 = self.stage2_fusion(stage2_in, return_attn=return_attn)
            stage2_rep, attn = (out2 if return_attn else (out2, None))
        else:
            stage2_rep = stage1_rep  

        if self._has_clinical and self.clinical_key in present:
            x_clin = feats[self.clinical_key]
            if x_clin.dim() == 1:
                x_clin = x_clin.unsqueeze(0)
            clinical_proj = self.clinical_proj(x_clin)          
            stage3_rep = self.stage3_norm(stage2_rep + clinical_proj)
        else:
            stage3_rep = self.stage3_norm(stage2_rep)           

        return {
            "subtype_logit": self.subtype_head(stage3_rep),
            "grade_logit": self.grade_head(stage3_rep),
            "ki67": self.ki67_head(stage3_rep),
            "fused": stage3_rep,
            "attn": attn,
        }

    def joint_loss(self, out: dict, y_subtype: torch.Tensor,
                   y_grade: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        l_subtype = self.subtype_loss(out["subtype_logit"], y_subtype)
        if y_grade is None:
            l_grade = out["grade_logit"].sum() * 0.0
        else:
            l_grade = self.grade_loss(out["grade_logit"], y_grade)
        return self.uncertainty([l_subtype, l_grade])


class _HierarchicalSubtypeAdapter(nn.Module):

    def __init__(self, model: "HierarchicalStagedFusion") -> None:
        super().__init__()
        self.model = model
        self.embed_dim = model.embed_dim

    def forward(self, feats: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.model(feats)["subtype_logit"]


def hierarchical_subtype_only(model: "HierarchicalStagedFusion") -> nn.Module:
    return _HierarchicalSubtypeAdapter(model)


class BiologyGatedMoE(nn.Module):

    _ALLOWED_STRATA = ("hr_status", "grade_band")

    def __init__(self, fused_dim: int, n_experts: int = 2, expert_hidden: int | None = None,
                 strata_init_method: str = "hr_status", gate_min: float = 0.8) -> None:
        super().__init__()
        if n_experts < 2:
            raise ValueError(f"n_experts must be >= 2, got {n_experts}")
        if strata_init_method not in self._ALLOWED_STRATA:
            raise ValueError(
                f"strata_init_method '{strata_init_method}' not allowed — MoE gating must be a KNOWN "
                f"biological stratum {self._ALLOWED_STRATA} (learned router banned at G3 scale, #7)."
            )
        if not 0.0 <= gate_min <= 1.0:
            raise ValueError(f"gate_min must be in [0,1], got {gate_min}")
        self.fused_dim = fused_dim
        self.embed_dim = fused_dim
        self.n_experts = n_experts
        self.strata_init_method = strata_init_method
        self.gate_min = gate_min

        hidden = expert_hidden or fused_dim
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(fused_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))
            for _ in range(n_experts)
        ])
        self.gate = nn.Linear(fused_dim, n_experts)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def _soft_gate(self, stage3_rep: torch.Tensor, strata_labels: torch.Tensor) -> torch.Tensor:
        n = stage3_rep.shape[0]
        logits = self.gate(stage3_rep)                      
        soft = torch.softmax(logits, dim=1)                 
        assigned = torch.zeros_like(soft)
        assigned[torch.arange(n, device=soft.device), strata_labels] = 1.0
        other = soft * (1.0 - assigned)
        other_sum = other.sum(dim=1, keepdim=True)
        n_other = max(self.n_experts - 1, 1)
        uniform_other = (1.0 - assigned) / n_other
        other_frac = torch.where(other_sum > 1e-8, other / other_sum.clamp_min(1e-8), uniform_other)
        return assigned * self.gate_min + other_frac * (1.0 - self.gate_min)

    def forward(self, stage3_rep: torch.Tensor, strata_labels: torch.Tensor) -> dict:
        if strata_labels.dtype not in (torch.long, torch.int, torch.int8, torch.int16, torch.int32,
                                       torch.int64):
            raise ValueError(
                f"strata_labels must be an integer routing tensor (got dtype {strata_labels.dtype}) — "
                "HR-status/grade-band route the gradient as an INTEGER index, never an ER/PR float "
                "feature (LOCK-2 leakage guard)."
            )
        if stage3_rep.dim() == 1:
            stage3_rep = stage3_rep.unsqueeze(0)
        if int(strata_labels.max()) >= self.n_experts or int(strata_labels.min()) < 0:
            raise ValueError(
                f"strata_labels out of range [0, {self.n_experts - 1}]: "
                f"min={int(strata_labels.min())}, max={int(strata_labels.max())}"
            )
        weights = self._soft_gate(stage3_rep, strata_labels)            
        expert_logits = torch.cat([e(stage3_rep) for e in self.experts], dim=1)  
        subtype_logit = (weights * expert_logits).sum(dim=1, keepdim=True)       
        return {"subtype_logit": subtype_logit, "expert_weights": weights}

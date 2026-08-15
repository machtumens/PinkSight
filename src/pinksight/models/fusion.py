"""P06 (Track A): cross-attention FUSION over the FROZEN MRI embedding + FROZEN clinical embedding.

This is the Track-A fusion head that decisions.md's architecture spine calls for:
`3D-CNN MRI encoder + FT-Transformer (clinical) -> modality-dropout cross-attention fusion ->
Head 1 subtype (focal) + Head 2 Ki-67 (Huber, binary @14%) with Kendall uncertainty weighting`.

DESIGN (composes with the existing stack, does NOT rewrite it):
  * Inputs are PRE-COMPUTED, FROZEN per-modality embeddings — a dict {name: [N, in_dim]} exactly like
    `trackb.modality_dropout.ModalityDropoutFusion.forward` consumes. The MRI encoder
    (`models.mri_encoder.MriEncoder`, 512-d) and clinical encoder (FT-Transformer) are trained/frozen
    UPSTREAM; only this fusion module + its heads are trained. That keeps the 8 GB-VRAM 4060 budget
    tiny (no 3D conv in the training graph) and makes the fusion delta honest (same frozen unimodal
    reps feed both the unimodal rung and the fused rung).
  * CROSS-ATTENTION: each modality is projected to a shared width, then a lightweight multi-head
    cross-attention lets each modality attend over the *other present* modalities before a residual
    mean-pool. With one present modality it degrades to a self-projection (identical to the unimodal
    rep) — so the ablation's unimodal rung and the fused rung share one code path.
  * MODALITY DROPOUT (train-time, [5.2]/[0.1]): a present modality can be randomly dropped so the
    model learns to predict from any subset; at inference a missing modality is simply omitted. This
    mirrors `ModalityDropoutFusion` — reused verbatim as the graceful-degradation contract.
  * TWO HEADS with Kendall (2018) homoscedastic uncertainty weighting:
      Head 1 subtype  -> Linear(embed_dim, 1) logit, trained with FOCAL loss (class imbalance:
                         TNBC prev ~0.21). This is the CHARACTERISATION head (Luminal-like vs TNBC).
      Head 2 Ki-67    -> Linear(embed_dim, 1), Huber regression toward the Ki-67 index AT DIAGNOSIS
                         (binary decision @14% is downstream, not here). This head is a PLACEHOLDER:
                         Ki-67 labels are N=0 in the dev cohort (see docs/DATA_CARD.md, G0
                         Ki-67 N=0), so it stays DESCRIPTIVE / UNTRAINED — the loss is defined and
                         wired but contributes nothing until Ki-67 labels exist. It is NOT growth
                         rate / kinetics (claim ledger): Ki-67 is a proliferation snapshot at
                         diagnosis.

CLAIM LEDGER: subtype CHARACTERISATION + Ki-67 aggressiveness AT DIAGNOSIS only. No pre-detection,
no growth-rate/kinetics, no cross-institution generalisation.

ponytail: one small cross-attention block + two Linear heads + three loss classes. No transformer
stack, no gating network, no learned pooling until a real fused run shows they help.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class TensorDict(dict):
    """A dict of per-modality tensors that also answers `.to(device)` — so a batched multi-modal
    input can flow through `train.loop.train_model` UNCHANGED (that loop calls `x.to(device)` on the
    batch, assuming a single tensor). Returning a `TensorDict` from a DataLoader's collate_fn is what
    lets the fusion head reuse the proven AMP / inner-val / honest-OOF loop with zero edits to it.

    ponytail: subclass dict, add one method — no new container type, no loop rewrite.
    """

    def to(self, device) -> "TensorDict":
        return TensorDict({k: v.to(device) for k, v in self.items()})


# --------------------------------------------------------------------------------------------------
# Losses (none of these existed in the repo — defined here, asserted by tests/test_fusion.py)
# --------------------------------------------------------------------------------------------------
class FocalLoss(nn.Module):
    """Binary focal loss (Lin et al. 2017) for the subtype head — down-weights easy negatives so the
    ~21% TNBC signal is not swamped. `alpha` weights the positive (TNBC) class. Reduces to weighted
    BCE at gamma=0, so it is a strict generalisation of the loop's BCEWithLogitsLoss recipe."""

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
    """Huber regression toward the Ki-67 index AT DIAGNOSIS (proliferation snapshot, NOT kinetics).

    PLACEHOLDER: Ki-67 labels are N=0 in the dev cohort, so this head is descriptive/untrained. When
    a target is missing (NaN), the per-sample loss is masked to zero — the head contributes nothing
    until real Ki-67 labels exist. `delta` is on the raw index scale (0-100). Never converts to a
    growth rate (claim ledger)."""

    def __init__(self, delta: float = 10.0) -> None:
        super().__init__()
        self.delta = delta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.reshape(-1)
        target = target.reshape(-1).float()
        mask = ~torch.isnan(target)
        if mask.sum() == 0:  # N=0 Ki-67 labels -> descriptive, no gradient (placeholder contract)
            return pred.sum() * 0.0
        return F.huber_loss(pred[mask], target[mask], delta=self.delta, reduction="mean")


class KendallUncertaintyWeighting(nn.Module):
    """Homoscedastic (task) uncertainty weighting (Kendall, Gal & Cipolla 2018).

    Learns one log-variance per task and combines losses as sum_i exp(-s_i) * L_i + s_i. A
    classification task (subtype) and a regression task (Ki-67) are placed on a comparable scale
    without a hand-tuned lambda. `s_i` = log(sigma_i^2) is learned. Returns (total, weights) where
    weights = exp(-s_i) so a run card can log how the two heads were balanced."""

    def __init__(self, n_tasks: int = 2) -> None:
        super().__init__()
        self.log_var = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, losses: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        if len(losses) != self.log_var.numel():
            raise ValueError(f"expected {self.log_var.numel()} task losses, got {len(losses)}")
        stacked = torch.stack([loss.reshape(()) for loss in losses])
        # TICKET-023 [MED]: two guards against an unbounded/NaN Kendall total.
        # (1) A task can be a ZERO PLACEHOLDER (e.g. Ki67HuberLoss returns pred.sum()*0.0 while Ki-67
        #     labels are N=0). Its log-var then has no real signal and drifts to -inf under gradient,
        #     making exp(-log_var) overflow. SKIP any task whose loss value is an exact zero placeholder
        #     so a descriptive/untrained head never destabilises the weighting.
        # (2) CLAMP log_var to [-7, 7] so exp(-log_var) stays bounded (~[9e-4, 1097]) even if a real
        #     task's log-var wanders — the total can no longer diverge to -inf / overflow.
        active = stacked.detach() != 0.0  # zero placeholder -> value is exactly 0.0
        if not bool(active.any()):
            active = torch.ones_like(active)  # all-zero (degenerate) -> keep all so grad still flows
        log_var = self.log_var.clamp(-7.0, 7.0)
        precision = torch.exp(-log_var)
        per_task = precision * stacked + log_var
        total = per_task[active].sum()
        return total, precision.detach()


# --------------------------------------------------------------------------------------------------
# Cross-attention fusion over frozen per-modality embeddings
# --------------------------------------------------------------------------------------------------
class CrossAttentionFusion(nn.Module):
    """Project each named FROZEN modality embedding to `fused_dim`, let present modalities cross-attend,
    then residual mean-pool the PRESENT ones. Same dict-in / graceful-degradation contract as
    `trackb.modality_dropout.ModalityDropoutFusion`, plus a cross-attention interaction term and
    train-time modality dropout.

    modality_dims: {name: in_dim} (e.g. {"mri": 512, "clinical": <ftt d_out>}).
    forward(feats, drop=..., return_attn=...): feats is {name: [N, in_dim]} for PRESENT modalities
    only. Absent modalities are left out of the dict. `drop` additionally removes named modalities at
    random-caller control (train-time). Requires >=1 present modality.

    Exposes `.embed_dim == fused_dim` so it drops straight into `models.heads.SubtypeClassifier` or
    `trackb.head.TrackBSubtypeHead`.
    """

    def __init__(self, modality_dims: dict[str, int], fused_dim: int = 128,
                 n_heads: int = 4, attn_dropout: float = 0.0) -> None:
        super().__init__()
        if not modality_dims:
            raise ValueError("need at least one modality")
        if fused_dim % n_heads != 0:
            raise ValueError(f"fused_dim {fused_dim} must be divisible by n_heads {n_heads}")
        self.fused_dim = fused_dim
        self.embed_dim = fused_dim  # head-agnostic contract (matches ModalityDropoutFusion)
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
            tokens.append(torch.relu(self.projections[name](x)))  # [N, fused_dim]
        seq = torch.stack(tokens, dim=1)  # [N, T_present, fused_dim]

        # each modality-token attends over the present modalities; residual + norm (transformer block
        # without the FFN). One present token -> attends to itself -> ~identity (unimodal path).
        attended, attn_w = self.attn(seq, seq, seq, need_weights=return_attn,
                                     average_attn_weights=True)
        fused_seq = self.norm(seq + attended)
        fused = fused_seq.mean(dim=1)  # [N, fused_dim] mean-pool the present modalities
        if return_attn:
            return fused, attn_w  # attn_w: [N, T_present, T_present] for XAI (P06 saliency)
        return fused


class FusionModel(nn.Module):
    """End-to-end fusion head: CrossAttentionFusion -> {subtype logit, Ki-67 regression}.

    Trainable part ONLY (encoders are frozen upstream). forward(feats, drop=...) returns a dict
    {"subtype_logit": [N,1], "ki67": [N,1], "fused": [N, fused_dim], "attn": ...}. `train_step`
    computes the Kendall-weighted joint loss; the Ki-67 term is a masked no-op while labels are N=0.

    modality_dims: {name: in_dim}. p_modality_dropout: per-modality independent drop prob at train
    time (0 disables). The subtype head is the one the ablation / stats machinery scores; the Ki-67
    head is the descriptive placeholder.
    """

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
        """Train-time: independently drop present modalities, but NEVER drop the last one (>=1 must
        survive so the fused set is non-empty — the ModalityDropoutFusion invariant)."""
        present = [n for n in self.modality_names if n in feats]
        if len(present) <= 1 or self.p_modality_dropout <= 0:
            return set()
        keep_min = 1
        drop = {n for n in present if torch.rand(()) < self.p_modality_dropout}
        if len(present) - len(drop) < keep_min:  # would empty the set -> spare a random modality
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
        """Kendall-weighted (focal subtype + Huber Ki-67). y_ki67=None -> all-NaN placeholder target
        (N=0 Ki-67 labels) so the Ki-67 term is a masked no-op. Returns (total_loss, task_weights)."""
        if y_ki67 is None:
            y_ki67 = torch.full_like(out["ki67"].reshape(-1), float("nan"))
        l_subtype = self.subtype_loss(out["subtype_logit"], y_subtype)
        l_ki67 = self.ki67_loss(out["ki67"], y_ki67)
        return self.uncertainty([l_subtype, l_ki67])


class _SubtypeLogitAdapter(nn.Module):
    """Wrap a FusionModel so it presents the (N,1)-logit contract that `train.loop.train_model`
    expects (`forward(x) -> (N,1)`). `x` is the batched dict of frozen modality embeddings; this
    forwards the subtype logit only. Lets the fusion head reuse the PROVEN train loop (AMP, inner-val
    early-stop, honest OOF) with zero changes to that loop — the same loop the G2 encoder used."""

    def __init__(self, fusion_model: FusionModel) -> None:
        super().__init__()
        self.fusion_model = fusion_model
        self.embed_dim = fusion_model.embed_dim

    def forward(self, feats: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.fusion_model(feats)["subtype_logit"]


def subtype_only(fusion_model: FusionModel) -> nn.Module:
    """Public helper: adapt a FusionModel to the loop's (N,1)-logit contract (subtype head only)."""
    return _SubtypeLogitAdapter(fusion_model)


# --------------------------------------------------------------------------------------------------
# G3 #4 — Hierarchical staged fusion (clinical enters LATE and STRONG)
# --------------------------------------------------------------------------------------------------
class HierarchicalStagedFusion(nn.Module):
    """Staged fusion where clinical enters LATE and STRONG so MRI cannot dilute it in early layers.

    Design rationale (decisions.md H6-MODALITY-AUDIT): clinical is the SOLE significant subtype
    modality (Shapley +0.135; MRI Shapley −0.025; naive fully-fused 0.621 < clinical-alone 0.708).
    A flat concat lets the empty MRI stream dilute the clinical signal. This module structurally
    prevents that dilution with a three-stage pipeline:

      Stage 1 — imaging features (frozen MRI encoder output) -> Linear -> ReLU -> stage1_rep.
      Stage 2 — OPTIONAL biological/radiomics stream, residual-fused with stage1_rep via
                `CrossAttentionFusion` on the present modalities -> stage2_rep. When no radiomics
                key is present in `feats`, Stage 2 is SKIPPED entirely and stage1_rep is projected
                straight to fused_dim (execute-agent instruction E-2). The G2 embeddings dir ships
                no radiomics stream, so this is the default path.
      Stage 3 — clinical trunk enters LATE and STRONG via an additive LayerNorm skip
                (`LayerNorm(stage2_rep + clinical_proj)`). The additive skip guarantees the clinical
                signal is preserved regardless of imaging quality — a structural firewall, not a
                weighting knob. When clinical is absent, stage3_rep == stage2_rep (graceful
                degradation, same contract as CrossAttentionFusion / ModalityDropoutFusion).

    Contract (identical to `FusionModel`): exposes `.embed_dim == fused_dim`, and
    `forward(feats, drop=..., return_attn=...)` returns a dict with keys
    {"subtype_logit": [N,1], "grade_logit": [N,1], "ki67": [N,1], "fused": [N, fused_dim], "attn"}.
    The extra `grade_logit` head is the biological-prior stage (G3 plan #4) and the counterfactual
    XAI target (#3); it is trained with focal loss like subtype. Kendall weighting is n_tasks=2
    (subtype + grade) by default.

    Leakage guard (LOCK-2): asserts at forward time that no key in `feats` is a FORBIDDEN feature —
    the imaging/clinical embeddings must never smuggle ER/PR/HER2/Ki-67/etc. in as a modality.

    modality_dims: {name: in_dim}, e.g. {"mri": 512, "clinical": 128} (+ optional {"radiomics": 107}).
    """

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
        self.embed_dim = fused_dim  # head-agnostic contract (matches FusionModel)
        self.modality_names = tuple(modality_dims)
        self.imaging_key = imaging_key
        self.clinical_key = clinical_key
        self.radiomics_key = radiomics_key
        self.p_modality_dropout = p_modality_dropout

        # Stage 1 — imaging projection to stage1_dim, then to fused_dim.
        self.stage1_proj = nn.Sequential(
            nn.Linear(modality_dims[imaging_key], stage1_dim), nn.ReLU(),
        )
        self.stage1_to_fused = nn.Linear(stage1_dim, fused_dim)

        # Stage 2 — optional radiomics cross-attention. Built only if a radiomics dim is declared.
        # The cross-attention fuses {stage1(as fused_dim), radiomics(projected)} on present modalities.
        self._has_radiomics = radiomics_key in modality_dims
        if self._has_radiomics:
            # stage1 already lives in fused_dim; radiomics is projected from its raw dim.
            self.stage2_fusion = CrossAttentionFusion(
                {"stage1": fused_dim, radiomics_key: modality_dims[radiomics_key]},
                fused_dim=fused_dim, n_heads=n_heads,
            )

        # Stage 3 — clinical trunk LATE and STRONG (additive LayerNorm skip).
        self._has_clinical = clinical_key in modality_dims
        if self._has_clinical:
            self.clinical_proj = nn.Sequential(
                nn.Linear(modality_dims[clinical_key], fused_dim), nn.ReLU(),
            )
        self.stage3_norm = nn.LayerNorm(fused_dim)

        # Diagnosis heads — both operate on stage3_rep.
        self.subtype_head = nn.Linear(fused_dim, 1)
        self.grade_head = nn.Linear(fused_dim, 1)   # NHG1 vs NHG3 (biological prior + XAI target)
        self.ki67_head = nn.Linear(fused_dim, 1)    # descriptive placeholder (Ki-67 N=0)

        self.subtype_loss = FocalLoss(focal_alpha, focal_gamma)
        self.grade_loss = FocalLoss(focal_alpha, focal_gamma)
        self.ki67_loss = Ki67HuberLoss(ki67_delta)
        self.uncertainty = KendallUncertaintyWeighting(n_tasks=2)  # subtype + grade

    def _assert_no_forbidden(self, feats: dict[str, torch.Tensor]) -> None:
        """LOCK-2 leakage guard: no modality key may be a FORBIDDEN feature."""
        from pinksight import FORBIDDEN_FEATURES
        leaked = set(feats) & FORBIDDEN_FEATURES
        if leaked:
            raise ValueError(
                f"LEAKAGE: forbidden feature(s) {sorted(leaked)} passed as a fusion modality — "
                "ER/PR/HER2/Ki-67/Mol-Subtype/Oncotype must never reach the classifier (LOCK-2)."
            )

    def _sample_drop(self, feats: dict[str, torch.Tensor]) -> set[str]:
        """Train-time modality dropout — never empties the present set (>=1 survives)."""
        present = [n for n in self.modality_names if n in feats]
        if len(present) <= 1 or self.p_modality_dropout <= 0:
            return set()
        drop = {n for n in present if torch.rand(()) < self.p_modality_dropout}
        if len(present) - len(drop) < 1:  # would empty -> spare a random modality
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
        # --- Stage 1: imaging -------------------------------------------------------------------
        if self.imaging_key in present:
            x_img = feats[self.imaging_key]
            if x_img.dim() == 1:
                x_img = x_img.unsqueeze(0)
            stage1_rep = self.stage1_to_fused(self.stage1_proj(x_img))  # [N, fused_dim]
            ref = x_img
        else:
            # No imaging present — seed stage1 from clinical batch size (graceful degradation).
            ref = feats[self.clinical_key if self.clinical_key in present else next(iter(present))]
            if ref.dim() == 1:
                ref = ref.unsqueeze(0)
            stage1_rep = torch.zeros(ref.shape[0], self.fused_dim, device=ref.device, dtype=ref.dtype)

        # --- Stage 2: optional radiomics cross-attention ----------------------------------------
        if self._has_radiomics and self.radiomics_key in present:
            stage2_in = {"stage1": stage1_rep, self.radiomics_key: feats[self.radiomics_key]}
            out2 = self.stage2_fusion(stage2_in, return_attn=return_attn)
            stage2_rep, attn = (out2 if return_attn else (out2, None))
        else:
            stage2_rep = stage1_rep  # E-2: radiomics absent -> pass stage1_rep straight through

        # --- Stage 3: clinical LATE and STRONG (additive skip) ----------------------------------
        if self._has_clinical and self.clinical_key in present:
            x_clin = feats[self.clinical_key]
            if x_clin.dim() == 1:
                x_clin = x_clin.unsqueeze(0)
            clinical_proj = self.clinical_proj(x_clin)          # [N, fused_dim]
            stage3_rep = self.stage3_norm(stage2_rep + clinical_proj)
        else:
            stage3_rep = self.stage3_norm(stage2_rep)           # clinical absent -> norm(stage2)

        return {
            "subtype_logit": self.subtype_head(stage3_rep),
            "grade_logit": self.grade_head(stage3_rep),
            "ki67": self.ki67_head(stage3_rep),
            "fused": stage3_rep,
            "attn": attn,
        }

    def joint_loss(self, out: dict, y_subtype: torch.Tensor,
                   y_grade: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Kendall-weighted (focal subtype + focal grade). y_grade=None -> grade term is a masked
        no-op via an all-NaN target-free path (uses subtype logits' batch for the zero placeholder),
        so the subtype head trains alone when grade labels are absent. Returns (total, task_weights)."""
        l_subtype = self.subtype_loss(out["subtype_logit"], y_subtype)
        if y_grade is None:
            # No grade labels -> contribute an exact-zero placeholder (Kendall skips zero tasks).
            l_grade = out["grade_logit"].sum() * 0.0
        else:
            l_grade = self.grade_loss(out["grade_logit"], y_grade)
        return self.uncertainty([l_subtype, l_grade])


class _HierarchicalSubtypeAdapter(nn.Module):
    """Adapt a HierarchicalStagedFusion to the loop's (N,1)-logit contract (subtype head only)."""

    def __init__(self, model: "HierarchicalStagedFusion") -> None:
        super().__init__()
        self.model = model
        self.embed_dim = model.embed_dim

    def forward(self, feats: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.model(feats)["subtype_logit"]


def hierarchical_subtype_only(model: "HierarchicalStagedFusion") -> nn.Module:
    """Public helper: adapt a HierarchicalStagedFusion to the (N,1)-logit contract (subtype head)."""
    return _HierarchicalSubtypeAdapter(model)


# --------------------------------------------------------------------------------------------------
# G3 #7 — Biology-gated Mixture-of-Experts (deterministic strata gating, NOT a learned router)
# --------------------------------------------------------------------------------------------------
class BiologyGatedMoE(nn.Module):
    """Mixture-of-experts on top of a fused representation, gated by KNOWN biological strata.

    Design constraint (G3 plan #7): a full learned router starves/overfits at N~320–900, so gating
    is DETERMINISTIC on known biology (HR-status or grade-band), NOT learned. The soft gate is
    INITIALISED to the hard strata assignment and may adapt only slightly during fine-tuning — the
    pre-assigned expert's weight is CLAMPED to >= gate_min (default 0.8, execute-agent instruction
    E-3) so it can never collapse into a free router.

    Leakage guard (LOCK-2): `strata_labels` is an INTEGER expert-index tensor (dtype long/int), used
    ONLY to route the gradient — it is NOT a feature fed to any head. HR-status is derived from the
    ER/PR LABEL columns upstream (a routing decision), never passed as an ER/PR float feature. The
    forward asserts `strata_labels` is integer-typed so a float ER/PR value can never be mistaken
    for a routing label.

    forward(stage3_rep, strata_labels) -> {"subtype_logit": [N,1], "expert_weights": [N, n_experts]}.
    `expert_weights` rows sum to 1 (soft gate after the gate_min clamp + renormalise).
    """

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
        # Each expert: a small MLP trunk + its own subtype logit head.
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(fused_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))
            for _ in range(n_experts)
        ])
        # Soft gate: learns a per-patient adjustment, but is dominated by the hard strata assignment
        # via the gate_min clamp below. Initialised near-zero so the initial gate ≈ hard assignment.
        self.gate = nn.Linear(fused_dim, n_experts)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def _soft_gate(self, stage3_rep: torch.Tensor, strata_labels: torch.Tensor) -> torch.Tensor:
        """Soft gate that GUARANTEES the pre-assigned expert carries >= gate_min AFTER normalisation.

        The assigned (biological strata) expert is given exactly `gate_min`; the remaining
        `(1 - gate_min)` mass is distributed across the OTHER experts proportional to their learned
        softmax weights. Rows sum to 1 and the assigned expert's final weight is a hard floor —
        preventing expert collapse / full learned-router behaviour at small N (execute-agent E-3).
        A naive `max(soft, gate_min)`-then-renormalise does NOT hold the floor (renormalisation
        pulls the assigned weight back below gate_min), which is why the mass is redistributed instead."""
        n = stage3_rep.shape[0]
        logits = self.gate(stage3_rep)                      # [N, n_experts]
        soft = torch.softmax(logits, dim=1)                 # learned per-patient preference
        assigned = torch.zeros_like(soft)
        assigned[torch.arange(n, device=soft.device), strata_labels] = 1.0
        # Zero out the assigned expert's soft weight; renormalise the rest to the residual budget.
        other = soft * (1.0 - assigned)
        other_sum = other.sum(dim=1, keepdim=True)
        # When the other experts have ~0 mass, spread the residual uniformly over them.
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
        weights = self._soft_gate(stage3_rep, strata_labels)            # [N, n_experts]
        expert_logits = torch.cat([e(stage3_rep) for e in self.experts], dim=1)  # [N, n_experts]
        subtype_logit = (weights * expert_logits).sum(dim=1, keepdim=True)       # [N, 1]
        return {"subtype_logit": subtype_logit, "expert_weights": weights}

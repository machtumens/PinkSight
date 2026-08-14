"""Track B subtype head (decisions.md [5.2]) — SUBTYPE CHARACTERISATION ONLY.

Track B is HARD-GATED (LOCK-6, dormant until Track A clears G5) and NOT patient-matched to MRI; the
~84-patient Duke∩TCGA overlap MUST be de-duplicated before any "external" claim.

CLAIM LEDGER: Track B outputs subtype characterisation ONLY. NO survival, NO growth/change-rate, NO
"early detection", NO H4/H5/H6/T4 semantics. This is a single Linear on the fused/MIL representation,
mirroring src/pinksight/models/heads.py::SubtypeClassifier so both tracks share one head pattern.
"""

from __future__ import annotations

from torch import nn


class TrackBSubtypeHead(nn.Module):
    """rep-producing module (MIL / fusion, exposing .embed_dim) -> Linear(embed_dim, 1) logits.

    `rep_module` is anything whose forward returns [N, embed_dim] (GenomicsEncoder,
    ModalityDropoutFusion) OR a (bag_embedding, attention) tuple (GatedAttentionMIL) — the tuple's
    first element is used and the attention is passed through for XAI.
    """

    def __init__(self, rep_module: nn.Module, embed_dim: int | None = None) -> None:
        super().__init__()
        self.rep_module = rep_module
        dim = embed_dim if embed_dim is not None else getattr(rep_module, "embed_dim", 128)
        self.head = nn.Linear(dim, 1)

    def forward(self, *args, **kwargs):
        rep = self.rep_module(*args, **kwargs)
        attn = None
        if isinstance(rep, tuple):
            rep, attn = rep
        if rep.dim() == 1:
            rep = rep.unsqueeze(0)
        logits = self.head(rep)  # [N, 1] — subtype logit (Luminal-like vs TNBC)
        return (logits, attn) if attn is not None else logits

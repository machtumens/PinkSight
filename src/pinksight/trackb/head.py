
from __future__ import annotations

from torch import nn


class TrackBSubtypeHead(nn.Module):

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
        logits = self.head(rep)  
        return (logits, attn) if attn is not None else logits

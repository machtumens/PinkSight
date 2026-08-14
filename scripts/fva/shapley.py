"""Exact n-player Shapley attribution over a coalition-AUROC dict (ported verbatim from h6).

φ_i = Σ_{S ⊆ N\{i}} [|S|!(n-|S|-1)!/n!] (v(S∪{i}) − v(S)), where v = coalition pooled-OOF AUROC.

Extraction only (C2-2): the algorithm is the exact one from ``h6_modality_audit.exact_shapley`` —
no algorithmic change. The efficiency axiom (Σφ_i == v(N) − v(∅)) is asserted by the caller via the
returned ``efficiency_ok`` flag. Project use is n=3 (clinical/radiomics/mri); the code is n-general.
"""

from __future__ import annotations

from itertools import combinations
from math import factorial


def exact_shapley(coalition_auc: dict, players: list[str], tol: float = 1e-6) -> dict:
    """Exact Shapley value per player over all 2^n coalition AUROCs. Efficiency gap reported.

    Ported verbatim from h6_modality_audit.exact_shapley — the only addition is the ``tol``
    parameter (default 1e-6, the frozen pre-reg efficiency tolerance) so the efficiency check is
    configurable via FVAConfig without changing the default behaviour.

    Args:
        coalition_auc: dict mapping ``tuple(sorted(coalition))`` -> pooled-OOF AUROC mean.
            MUST contain every subset of ``players`` including ``tuple()`` (the ∅ coalition).
        players: the ordered player list (the streams being attributed).
        tol: efficiency-axiom tolerance (frozen at 1e-6 in the pre-reg).

    Returns:
        dict with keys: ``phi`` (per-player value, 4dp), ``sum_phi`` (6dp),
        ``v_all_minus_empty`` (6dp), ``efficiency_gap`` (float), ``efficiency_ok`` (bool).
    """
    n = len(players)
    phi = {}
    for i in players:
        others = [p for p in players if p != i]
        total = 0.0
        for r in range(len(others) + 1):
            for S in combinations(others, r):
                w = factorial(len(S)) * factorial(n - len(S) - 1) / factorial(n)
                v_S = coalition_auc[tuple(sorted(S))]
                v_Si = coalition_auc[tuple(sorted(S + (i,)))]
                total += w * (v_Si - v_S)
        phi[i] = total
    v_all = coalition_auc[tuple(sorted(players))]
    v_empty = coalition_auc[tuple()]
    efficiency_gap = abs(sum(phi.values()) - (v_all - v_empty))
    return {
        "phi": {k: round(v, 4) for k, v in phi.items()},
        "sum_phi": round(sum(phi.values()), 6),
        "v_all_minus_empty": round(v_all - v_empty, 6),
        "efficiency_gap": float(efficiency_gap),
        "efficiency_ok": bool(efficiency_gap < tol),
    }

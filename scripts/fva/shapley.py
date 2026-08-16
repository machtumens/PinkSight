
from __future__ import annotations

from itertools import combinations
from math import factorial


def exact_shapley(coalition_auc: dict, players: list[str], tol: float = 1e-6) -> dict:
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

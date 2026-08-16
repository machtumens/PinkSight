
from __future__ import annotations

from typing import Any

OUTCOME_GATE = 0.005

NYU_FIREWALL_KEY = "nyu_internal_firewalled"

VERDICT_REPORTABLE = "BOUNDED_NULL_REPORTABLE"
VERDICT_HALT = "NON_REPORTABLE_HALT"

_HALT_REASON = (
    "|Δ| > 0.005 — a frozen NYU representation carrying separable Duke subtype signal is a "
    "CROSS-INSTITUTION TRANSFER FINDING. Under ADR-0016 this is NON-REPORTABLE: it must NOT enter any "
    "artifact as a result/claim and requires a NEW ADR + fresh /red-team. No decisions.md claim, no "
    "LOCK moved."
)
_OK_REASON = (
    "|Δ| ≤ 0.005 — bounded null ('no separable effect'). REPORTABLE under ADR-0016 as a Duke-only "
    "frozen-feature slot-attachment ablation. No cross-institution claim; no LOCK moved."
)


def gate_verdict(delta: float, gate: float = OUTCOME_GATE) -> dict[str, Any]:
    d = float(delta)
    reportable = abs(d) <= float(gate)
    return {
        "reportable": bool(reportable),
        "verdict": VERDICT_REPORTABLE if reportable else VERDICT_HALT,
        "delta": d,
        "gate": float(gate),
        "reason": _OK_REASON if reportable else _HALT_REASON,
    }


def assert_frozen(module: Any) -> None:
    params = list(module.parameters())
    if not params:
        raise AssertionError("assert_frozen: module has no parameters — not a real encoder")
    thawed = sum(1 for p in params if p.requires_grad)
    if thawed:
        raise AssertionError(
            f"FROZEN-ONLY INVARIANT VIOLATED (ADR-0016): {thawed} parameter tensor(s) still have "
            "requires_grad=True — the NYU encoder must be fully frozen before it touches Duke."
        )
    if getattr(module, "training", False):
        raise AssertionError(
            "FROZEN-ONLY INVARIANT VIOLATED (ADR-0016): encoder is in training() mode — call .eval() "
            "so BatchNorm/dropout are fixed before extracting Duke embeddings."
        )


def assert_no_juxtaposition(duke_block: dict, nyu_block: dict | None = None) -> None:
    banned_in_duke = ("nyu", "fastmri", "pooled", "juxtapos")
    for k in duke_block:
        kl = str(k).lower()
        if kl == NYU_FIREWALL_KEY:
            raise AssertionError(
                "FIREWALL VIOLATED (ADR-0016 Fix #4): the NYU-internal block is nested INSIDE the Duke "
                "ablation block — NYU numbers must live in a SEPARATE top-level firewalled block."
            )
        if any(b in kl for b in banned_in_duke):
            raise AssertionError(
                f"FIREWALL VIOLATED (ADR-0016 Fix #4): Duke ablation block key {k!r} names an NYU/"
                "fastMRI/pooled metric — no NYU number may sit inside the Duke ablation block."
            )
    if nyu_block is not None:
        for k in nyu_block:
            kl = str(k).lower()
            if any(b in kl for b in ("duke", "slot_delta", "ablation_delta", "pooled", "juxtapos")):
                raise AssertionError(
                    f"FIREWALL VIOLATED (ADR-0016 Fix #4): NYU-internal block key {k!r} names a Duke/"
                    "pooled metric — the NYU block is NYU-INTERNAL only, never a comparison."
                )


def aggregate_paired_delta(per_seed_delta: list[float], per_seed_p: list[float],
                           gate: float = OUTCOME_GATE) -> dict[str, Any]:
    if not per_seed_delta:
        raise ValueError("aggregate_paired_delta needs at least one seed's Δ")
    import statistics

    mean_delta = statistics.fmean(per_seed_delta)
    std_delta = statistics.pstdev(per_seed_delta) if len(per_seed_delta) > 1 else 0.0
    mean_p = statistics.fmean(per_seed_p) if per_seed_p else float("nan")
    v = gate_verdict(mean_delta, gate)
    v.update({
        "delta_mean": round(mean_delta, 6),
        "delta_std_across_seeds": round(std_delta, 6),
        "delta_per_seed": [round(float(x), 6) for x in per_seed_delta],
        "paired_p_mean": round(mean_p, 6) if per_seed_p else None,
        "paired_p_per_seed": [round(float(x), 6) for x in per_seed_p] if per_seed_p else None,
        "direction": "with_slot - without_slot (Δ>0 => frozen NYU slot adds Duke subtype separability)",
    })
    return v


def selfcheck() -> int:
    assert gate_verdict(0.0)["reportable"] and gate_verdict(0.0)["verdict"] == VERDICT_REPORTABLE
    assert gate_verdict(0.005)["reportable"], "boundary |Δ|==gate must be reportable (≤)"
    assert gate_verdict(-0.005)["reportable"], "negative boundary must be reportable (symmetric)"
    assert not gate_verdict(0.02)["reportable"], "a large positive Δ must HALT"
    assert not gate_verdict(-0.02)["reportable"], "a large negative Δ must HALT (symmetric)"
    assert gate_verdict(0.02)["verdict"] == VERDICT_HALT

    import torch
    from torch import nn

    m = nn.Linear(4, 2)
    for p in m.parameters():
        p.requires_grad_(True)
    try:
        assert_frozen(m)
    except AssertionError:
        pass
    else:
        raise AssertionError("assert_frozen failed to fire on a thawed module")
    for p in m.parameters():
        p.requires_grad_(False)
    m.eval()
    assert_frozen(m)  
    m.train()
    try:
        assert_frozen(m)
    except AssertionError:
        pass
    else:
        raise AssertionError("assert_frozen failed to fire on a training-mode module")

    try:
        assert_no_juxtaposition({"auroc_with_slot": 0.71, "nyu_hchar_auroc": 0.599})
    except AssertionError:
        pass
    else:
        raise AssertionError("firewall failed to fire on an NYU number inside the Duke block")
    try:
        assert_no_juxtaposition({"auroc_with_slot": 0.71, NYU_FIREWALL_KEY: {}})
    except AssertionError:
        pass
    else:
        raise AssertionError("firewall failed to fire on a nested NYU block")
    assert_no_juxtaposition(
        {"auroc_without_slot": 0.708, "auroc_with_slot": 0.707, "delta_mean": -0.001},
        {"hchar_auroc": 0.599, "hchar_delong_ci95": [0.4303, 0.7676]},
    )

    agg = aggregate_paired_delta([0.001, -0.002, 0.0], [0.9, 0.8, 1.0])
    assert agg["reportable"] and abs(agg["delta_mean"]) <= OUTCOME_GATE
    big = aggregate_paired_delta([0.03, 0.02, 0.025], [0.01, 0.02, 0.01])
    assert not big["reportable"] and big["verdict"] == VERDICT_HALT

    _ = torch  
    print("nyu_duke_slot selfcheck OK — ±0.005 gate branches (symmetric), frozen invariant fires, "  
          "firewall blocks NYU/Duke juxtaposition, paired-Δ aggregation correct.")
    return 0


if __name__ == "__main__":
    raise SystemExit(selfcheck())

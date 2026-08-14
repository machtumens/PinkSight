"""ADR-0016 deferred frozen-feature Duke-only fusion-slot — PURE gate/invariant/firewall logic.

The heavy training + extraction lives in scripts/fastmri_nyu_duke_slot_{extract,ablation}.py; this
module holds the load-bearing INVARIANTS so a fast, GPU-free test can prove they cannot silently break:

  * ``gate_verdict``          — the ±0.005 OUTCOME-GATE branch (ADR-0016 red-team Fix #1). ONLY a
                                bounded null |Δ| ≤ 0.005 is REPORTABLE; a Δ beyond the band HALTS as a
                                NON-REPORTABLE cross-institution finding (NEW ADR + fresh /red-team).
  * ``assert_frozen``         — the frozen-encoder invariant: every parameter requires_grad==False AND
                                the module is in eval() mode. NYU weights never unfreeze into Duke.
  * ``assert_no_juxtaposition`` — the reporting firewall (Fix #4): no NYU number is placed IN
                                COMPARISON with a Duke number. NYU-internal metrics must be quarantined
                                under the reserved ``NYU_FIREWALL_KEY`` sub-block, never in the Duke
                                ablation block. Co-locating the ADR-0008 Duke null as a reminder is OK.
  * ``aggregate_paired_delta`` — collapse per-seed paired-DeLong Δ/p into a mean Δ + the gate verdict.

Framing (ADR-0016): "frozen-feature transfer", "Duke-only slot-attachment ablation", "bounded null".
NEVER "imaging works", "generalises across institutions", early-detection, or kinetics.
"""

from __future__ import annotations

from typing import Any

# The ADR-0016 outcome-gate half-width. |Δ(Duke subtype AUROC)| at or below this is a bounded null and
# is the ONLY reportable outcome under this ADR. A Δ beyond it (positive OR negative) is a
# cross-institution transfer finding → NON-REPORTABLE without a NEW ADR + fresh /red-team.
OUTCOME_GATE = 0.005

# Results-dict key under which (and ONLY under which) an NYU-internal number may appear. Keeping NYU
# metrics behind this reserved key is the mechanical half of the firewall: the Duke ablation block must
# never carry an NYU number, and this block must never carry a Duke ablation number.
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
    """The ±0.005 OUTCOME-GATE branch (ADR-0016 Fix #1). Symmetric: a large NEGATIVE Δ halts too.

    Returns a dict: ``reportable`` (bool), ``verdict`` (BOUNDED_NULL_REPORTABLE | NON_REPORTABLE_HALT),
    ``delta``, ``gate``, and a human ``reason``. The caller MUST refuse to emit Δ as a result when
    ``reportable`` is False — it writes the HALT flag instead.
    """
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
    """RAISE unless `module` is a genuinely frozen encoder: every parameter requires_grad==False AND
    the module is in eval() mode. This is the mechanical proof the NYU encoder never takes a Duke
    gradient (ADR-0016 'no cross-cohort gradient' / frozen-transfer clause)."""
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
    """Reporting firewall (Fix #4): a Duke ablation block must carry NO NYU number and vice-versa.

    ``duke_block`` = the Duke slot-ablation metrics; ``nyu_block`` = the NYU-internal metrics (if any),
    which the caller must place under the reserved ``NYU_FIREWALL_KEY``. RAISE if a Duke block key
    smuggles an NYU metric, or an NYU block key smuggles a Duke ablation metric, or a pooled/juxtaposed
    key exists. This is the mechanical fingerprint of a forbidden comparison.
    """
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
    """Collapse per-seed paired-DeLong Δ (with-slot − without-slot) and p into the gate decision.

    Δ is averaged across seeds (the ADR reports one Δ under the gate); the per-seed spread + mean
    two-sided p are carried for honesty. The gate branch is applied to the MEAN Δ.
    """
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
    """Runnable check (ponytail): the gate branch, frozen invariant, and firewall all fire correctly."""
    # ---- gate branch: bounded null reportable; both large signs HALT ----
    assert gate_verdict(0.0)["reportable"] and gate_verdict(0.0)["verdict"] == VERDICT_REPORTABLE
    assert gate_verdict(0.005)["reportable"], "boundary |Δ|==gate must be reportable (≤)"
    assert gate_verdict(-0.005)["reportable"], "negative boundary must be reportable (symmetric)"
    assert not gate_verdict(0.02)["reportable"], "a large positive Δ must HALT"
    assert not gate_verdict(-0.02)["reportable"], "a large negative Δ must HALT (symmetric)"
    assert gate_verdict(0.02)["verdict"] == VERDICT_HALT

    # ---- frozen invariant: a thawed / training module must RAISE; a frozen eval one must PASS ----
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
    assert_frozen(m)  # frozen + eval → passes
    m.train()
    try:
        assert_frozen(m)
    except AssertionError:
        pass
    else:
        raise AssertionError("assert_frozen failed to fire on a training-mode module")

    # ---- firewall: an NYU key inside the Duke block must RAISE; separated blocks PASS ----
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
    # clean, firewall-separated structure passes:
    assert_no_juxtaposition(
        {"auroc_without_slot": 0.708, "auroc_with_slot": 0.707, "delta_mean": -0.001},
        {"hchar_auroc": 0.599, "hchar_delong_ci95": [0.4303, 0.7676]},
    )

    # ---- aggregation ----
    agg = aggregate_paired_delta([0.001, -0.002, 0.0], [0.9, 0.8, 1.0])
    assert agg["reportable"] and abs(agg["delta_mean"]) <= OUTCOME_GATE
    big = aggregate_paired_delta([0.03, 0.02, 0.025], [0.01, 0.02, 0.01])
    assert not big["reportable"] and big["verdict"] == VERDICT_HALT

    _ = torch  # silence unused if torch import is elided by a linter
    print("nyu_duke_slot selfcheck OK — ±0.005 gate branches (symmetric), frozen invariant fires, "  # noqa: T201
          "firewall blocks NYU/Duke juxtaposition, paired-Δ aggregation correct.")
    return 0


if __name__ == "__main__":
    raise SystemExit(selfcheck())

"""ADR-0016 frozen-slot — the ONE runnable check that FAILS if the gate logic or the frozen-only
invariant (or the reporting firewall) silently breaks.

This is deliberately GPU-free and data-free: it exercises pinksight.eval.nyu_duke_slot, the module that
holds the load-bearing invariants the heavy extract/ablation scripts import. If any of these regress,
the ±0.005 outcome-gate could pass a cross-institution finding as a null, an unfrozen NYU encoder could
take a Duke gradient, or an NYU number could be juxtaposed with a Duke number — all ADR-0016 violations.
"""

from __future__ import annotations

import pytest

pytest.importorskip("scipy", reason="scipy is an ml/arms/trackb-extra dep — skip cleanly under a base install; pinksight.eval.nyu_duke_slot imports scipy")


from pinksight.eval.nyu_duke_slot import (
    OUTCOME_GATE,
    VERDICT_HALT,
    VERDICT_REPORTABLE,
    aggregate_paired_delta,
    assert_frozen,
    assert_no_juxtaposition,
    gate_verdict,
    selfcheck,
)


def test_gate_is_the_ratified_half_width():
    assert OUTCOME_GATE == 0.005, "the ADR-0016 outcome-gate half-width must be exactly 0.005"


@pytest.mark.parametrize("delta,reportable", [
    (0.0, True), (0.005, True), (-0.005, True), (0.004999, True), (-0.004999, True),
    (0.0051, False), (-0.0051, False), (0.02, False), (-0.02, False), (0.5, False),
])
def test_gate_branch_symmetric(delta, reportable):
    v = gate_verdict(delta)
    assert v["reportable"] is reportable
    assert v["verdict"] == (VERDICT_REPORTABLE if reportable else VERDICT_HALT)


def test_gate_halt_reason_forbids_reporting():
    v = gate_verdict(0.03)
    assert not v["reportable"]
    assert "NON-REPORTABLE" in v["reason"] and "NEW ADR" in v["reason"]


def test_frozen_invariant_fires_on_thawed_and_training():
    torch = pytest.importorskip("torch")
    m = torch.nn.Linear(4, 2)
    for p in m.parameters():          # thawed -> must raise
        p.requires_grad_(True)
    with pytest.raises(AssertionError):
        assert_frozen(m)
    for p in m.parameters():          # frozen + eval -> passes
        p.requires_grad_(False)
    m.eval()
    assert_frozen(m)
    m.train()                          # training mode -> must raise even when params frozen
    with pytest.raises(AssertionError):
        assert_frozen(m)


def test_frozen_invariant_rejects_empty_module():
    torch = pytest.importorskip("torch")
    with pytest.raises(AssertionError):
        assert_frozen(torch.nn.Identity())  # no parameters — not a real encoder


def test_firewall_blocks_nyu_number_in_duke_block():
    with pytest.raises(AssertionError):
        assert_no_juxtaposition({"auroc_with_slot": 0.71, "nyu_hchar_auroc": 0.599})
    with pytest.raises(AssertionError):
        assert_no_juxtaposition({"auroc_with_slot": 0.71, "pooled_nyu_duke_auroc": 0.65})


def test_firewall_blocks_duke_number_in_nyu_block():
    with pytest.raises(AssertionError):
        assert_no_juxtaposition({"auroc_with_slot": 0.71},
                                {"hchar_auroc": 0.599, "duke_slot_delta": 0.001})


def test_firewall_passes_on_separated_blocks():
    assert_no_juxtaposition(
        {"auroc_without_slot": 0.708, "auroc_with_slot": 0.707, "delta_auroc": -0.001},
        {"hchar_auroc": 0.599, "hchar_delong_ci95": [0.4303, 0.7676]},
    )


def test_aggregate_bounded_null_is_reportable():
    agg = aggregate_paired_delta([0.001, -0.002, 0.0], [0.9, 0.8, 1.0])
    assert agg["reportable"] and abs(agg["delta_mean"]) <= OUTCOME_GATE


def test_aggregate_large_delta_halts():
    agg = aggregate_paired_delta([0.03, 0.02, 0.025], [0.01, 0.02, 0.01])
    assert not agg["reportable"] and agg["verdict"] == VERDICT_HALT


def test_module_selfcheck_passes():
    assert selfcheck() == 0

"""Shared JSON report contract for the E2E synthetic plumbing harness (Stages 0-6 <-> Stage 7 seam).

This module is the ONE coordination seam between the Python harness (Stages 0-6) and the desktop demo
(Stage 7). Both sides build against the single schema assembled by ``build_report`` here, so a field
name can never drift between the two surfaces without breaking this module's ``selfcheck``/tests.

INTEGRITY (constitutional — these are the load-bearing invariants, mirrored from
``eval/nyu_duke_slot.py``'s raise-on-violation style):
  * NON-REPORTABLE BY CONSTRUCTION. Every report carries ``provenance.datasetTag == SYNTHETIC_TAG``
    and a ``provenance.manifestSha256`` matching the synthetic-generation config.
    ``assert_synthetic_provenance`` RAISES on a missing/mismatched tag or hash, so a synthetic
    plumbing run can never be silently mistaken for a real-cohort characterisation result.
  * CHARACTERISATION FRAMING ONLY. Field names and note strings describe at-diagnosis subtype
    characterisation, Ki-67 stratification, and explainable fusion — never detection, cross-institution
    transfer, or proliferation dynamics. New strings here are scanned by ``ci/ledger_lint.py``.
  * FORWARD-ONLY PLUMBING. No metric assembled here is a scientific result. The deep stack that feeds
    these numbers runs frozen/forward-only on fabricated data; the numbers prove tensor wiring, not
    biology (the ``controlVerdict`` block proves only that the wiring is not silently dead).

``build_report`` / ``assert_synthetic_provenance`` / ``selfcheck`` depend only on stdlib + numpy so
the desktop side and the module selfcheck stay importable without the heavy ml extra. ``control_verdict``
lazily imports sklearn + ``pinksight.metrics`` (same lazy-import discipline as ``clinical_encoder.py``).
"""

from __future__ import annotations

from typing import Any

# The constant, non-reportable provenance tag. assert_synthetic_provenance checks this verbatim.
SYNTHETIC_TAG = "SYNTHETIC — NOT A RESULT"
REPORT_VERSION = "e2e-synthetic-v1"

# Control-sentinel PASS/FAIL bars — factored here so the driver script AND the test suite call the
# SAME thresholds (never duplicate them). Mirror tests/test_fva_sentinel.py's asserted bars.
POSITIVE_REAL_MIN = 0.75   # positive control real-signal AUROC must clear this
POSITIVE_SHUFFLE_MAX = 0.55  # positive control shuffle companion must collapse at/below this
CHANCE = 0.50
# The label-attribution leak test: a score is leak-free iff the REAL-label AUROC does not materially
# exceed the permutation-null (shuffle) AUROC — i.e. the labels buy ~no separable signal. This is the
# criterion that actually attributes signal to LABELS (vs the CI-crosses-0.50 test, which assumes the
# null floor is exactly 0.50 — false for high-dim coalitions whose pooled-OOF AUROC floats slightly
# above 0.50 for ANY labels). ~2σ of the real-vs-shuffle Δ noise at n≈10k. ponytail: absolute margin
# calibrated for the n≈10k deliverable regime; tighten with a per-n SE if small-N reports ever gate on it.
SHUFFLE_LEAK_MARGIN = 0.02

# Default synthetic Ki-67 descriptor (characterisation framing; scanned by ledger_lint).
KI67_DESCRIPTOR_DEFAULT = (
    "Descriptive companion — imaging correlate of Ki-67 index at diagnosis (aggressiveness snapshot). "
    "SYNTHETIC-ONLY plumbing: proves the head's tensor wiring end-to-end; does not evaluate the real "
    "Ki-67 head (N=0 real labels, unchanged)."
)

# Uninformative synthetic Nottingham-grade block. A random-init forward-only stack carries no grade
# characterisation signal, so probability 0.5 (uninformative) is the honest plumbing value — this is
# NOT a grade characterisation result (datasetTag/SYNTHETIC covers non-reportability).
_SYNTHETIC_NOTTINGHAM: dict[str, Any] = {
    "label": "NHG1",
    "probability": 0.5,
    "uncertainty": [0.4, 0.6],
}


class SyntheticProvenanceError(AssertionError):
    """A report failed the non-reportable-by-construction provenance gate (missing/mismatched tag)."""


def build_report(
    stream_name: str,
    manifest: dict[str, Any],
    subtype_out: dict[str, Any],
    ki67_out: dict[str, Any] | None,
    calibration_out: dict[str, Any] | None,
    modality_contribution_out: list[dict[str, Any]],
    xai_out: dict[str, Any] | None,
    control_verdict_out: dict[str, Any] | None,
    *,
    study_id: str | None = None,
    nottingham_out: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the shared E2E-synthetic report dict from the per-stage outputs.

    ``manifest`` is a ``synthetic_cohort.build_manifest`` dict (carries ``manifest_sha256`` + ``config``
    + ``generated_at``). ``stream_name`` is ``"negative_control"`` or ``"positive_control"``. The
    per-study blocks (subtype/nottingham/xai) represent ONE representative study from the stream; the
    cohort-level blocks (provenance/calibration/controlVerdict) summarise all ``config.n`` patients.

    Every block is additive-optional except the ones the desktop ``CharacterisationReport`` TS type
    requires (``studyId``, ``subtype``, ``ki67Descriptor``, ``nottinghamGrade``, ``calibration``,
    ``modalities``, ``audit``) — those are always present so the desktop side never sees a missing key.
    """
    config = manifest.get("config", {})
    n_patients = config.get("n")
    seed = config.get("seed", 0)
    if study_id is None:
        prefix = "SYN-NEG" if stream_name == "negative_control" else "SYN-POS"
        study_id = f"{prefix}-00000"

    ki67 = ki67_out or {}
    calibration = calibration_out or {}
    nottingham = nottingham_out or dict(_SYNTHETIC_NOTTINGHAM)
    generated_at = manifest.get("generated_at", "unknown")

    report: dict[str, Any] = {
        "reportVersion": REPORT_VERSION,
        "studyId": study_id,
        "provenance": {
            "datasetTag": SYNTHETIC_TAG,
            "manifestSha256": manifest.get("manifest_sha256", ""),
            "seed": seed,
            "gitCommit": config.get("git_commit", "unknown"),
            "generatedAt": generated_at,
            "cohortStream": stream_name,
            "nPatients": n_patients,
        },
        "subtype": {
            "label": subtype_out["label"],
            "probability": subtype_out["probability"],
            "uncertainty": list(subtype_out["uncertainty"]),
            "abstained": bool(subtype_out.get("abstained", False)),
        },
        "ki67Descriptor": ki67.get("descriptor", KI67_DESCRIPTOR_DEFAULT),
        "ki67Stratum": ki67.get("stratum", "not_assessed"),
        "nottinghamGrade": {
            "label": nottingham["label"],
            "probability": nottingham["probability"],
            "uncertainty": list(nottingham["uncertainty"]),
        },
        "calibration": {
            "ece": calibration.get("ece"),
            "smoothEce": calibration.get("smoothEce"),
            "band": calibration.get("band", "not_assessed"),
        },
        "modalities": [
            {
                "modality": m["modality"],
                "present": bool(m["present"]),
                "contribution": m["contribution"],
            }
            for m in modality_contribution_out
        ],
        "audit": {
            "modelHash": "sha256:e2e-synthetic-forward-only",
            "seed": seed,
            "split": "synthetic-plumbing-only",
            "generatedAt": generated_at,
        },
    }
    if xai_out is not None:
        report["xai"] = xai_out
    if control_verdict_out is not None:
        report["controlVerdict"] = control_verdict_out
    return report


def assert_synthetic_provenance(report: dict[str, Any], expected_manifest_sha256: str) -> None:
    """RAISE (``SyntheticProvenanceError``) unless the report is honestly tagged non-reportable.

    The manifest-hash gate (firewall #2): a report is only admissible when its ``provenance.datasetTag``
    is exactly ``SYNTHETIC_TAG`` AND its ``provenance.manifestSha256`` matches the recorded
    synthetic-generation config hash. A stripped tag, a mismatched hash, or a missing provenance block
    all raise — so a synthetic plumbing report can never masquerade as a real-cohort result. Mirrors
    ``nyu_duke_slot.assert_frozen`` / ``assert_no_juxtaposition``'s raise-on-violation contract.
    """
    prov = report.get("provenance")
    if not isinstance(prov, dict):
        raise SyntheticProvenanceError(
            "provenance block missing — a synthetic report MUST carry a provenance block with the "
            f"non-reportable datasetTag {SYNTHETIC_TAG!r} (firewall #2)."
        )
    tag = prov.get("datasetTag")
    if tag != SYNTHETIC_TAG:
        raise SyntheticProvenanceError(
            f"datasetTag {tag!r} != required {SYNTHETIC_TAG!r} — refusing to treat this report as a "
            "valid synthetic-plumbing artifact (a synthetic number is never a result; firewall #2)."
        )
    got = prov.get("manifestSha256")
    if got != expected_manifest_sha256:
        raise SyntheticProvenanceError(
            f"manifestSha256 mismatch: report carries {got!r} but the generation config hashes to "
            f"{expected_manifest_sha256!r} — a stale/tampered report cannot be trusted (firewall #2)."
        )


# Banned substrings that fingerprint a forbidden cross-organ pooling / comparison / ranking in a
# run-of-runs manifest. A ``manifest_of_runs`` must list each organ's own controlVerdict SEPARATELY —
# never a pooled number, a cross-organ delta, or a ranking. Checked against every current
# ``control_verdict()`` field name (auroc / delongCi95 / shuffleAuroc / shuffleAtChance /
# labelAttributableSignal / leakFreeByShuffle / expected / verdict / stream) — none collide, so a clean
# manifest built from real verdict blocks never false-positives.
_POOLING_BANNED_SUBSTRINGS = (
    "pooled", "combined", "overall", "aggregate", "delta", "ranking", "comparison", "juxtapos",
)


def assert_no_cross_organ_pooling(manifest_of_runs: dict[str, dict]) -> None:
    """RAISE (``AssertionError``) if a run-of-runs manifest pools, combines, or compares organs (LOCK-1).

    A ``manifest_of_runs`` is a run-of-runs index: ``{organ_name: {"negative_control": <controlVerdict>,
    "positive_control": <controlVerdict>}, ...}`` plus reserved ``_generatedAt`` / ``_note`` metadata.
    Each organ's own control-sentinel verdict is listed SEPARATELY — never a pooled AUROC, a cross-organ
    delta/ranking, or an organ-vs-organ comparison. This is the mechanical half of the LOCK-1
    no-cross-institution / no-pooling discipline (mirrors ``nyu_duke_slot.assert_no_juxtaposition``'s
    raise-on-violation contract; plain ``AssertionError``, no new exception subclass).

    Two levels are checked:
      * every TOP-LEVEL key (organ names + reserved metadata) — a banned substring here fingerprints a
        pooled/combined/overall/aggregate/delta/ranking/comparison/juxtaposition key.
      * every ORGAN BLOCK's own keys (the dict value under each organ key) — catches a delta/ranking key
        smuggled one level down (e.g. ``{"trackb-...": {"negative_control": {...}, "vs_trackc_delta": ...}}``).

    PASS on a clean manifest whose organ blocks carry only ``negative_control`` / ``positive_control``
    (each holding that organ's own unmodified ``controlVerdict`` dict) plus reserved metadata keys.
    """
    if not isinstance(manifest_of_runs, dict):
        raise AssertionError(
            "manifest_of_runs must be a dict of {organ: {stream: controlVerdict}}, got "
            f"{type(manifest_of_runs).__name__} — cannot assert the no-pooling firewall on a non-dict."
        )
    for top_key, block in manifest_of_runs.items():
        top_hit = next((b for b in _POOLING_BANNED_SUBSTRINGS if b in str(top_key).lower()), None)
        if top_hit is not None:
            raise AssertionError(
                f"NO-CROSS-ORGAN-POOLING FIREWALL VIOLATED (LOCK-1): top-level key {top_key!r} names a "
                f"{top_hit!r} metric — a run-of-runs index lists each organ SEPARATELY, never a pooled/"
                "combined/aggregate number, a cross-organ delta, or a ranking/comparison."
            )
        if isinstance(block, dict):
            for organ_key in block:
                nested_hit = next(
                    (b for b in _POOLING_BANNED_SUBSTRINGS if b in str(organ_key).lower()), None
                )
                if nested_hit is not None:
                    raise AssertionError(
                        f"NO-CROSS-ORGAN-POOLING FIREWALL VIOLATED (LOCK-1): organ block {top_key!r} "
                        f"carries key {organ_key!r} naming a {nested_hit!r} metric — an organ block holds "
                        "only its own negative_control/positive_control controlVerdict, never a "
                        "cross-organ delta/ranking/comparison smuggled one level down."
                    )


def control_verdict(
    stream_name: str,
    y: Any = None,
    real_oof: Any = None,
    shuffle_oof: Any = None,
) -> dict[str, Any]:
    """Derive the ``controlVerdict`` block from a stream's real + shuffle pooled-OOF predictions.

    Owns the ONE copy of the control PASS/FAIL logic (both the driver and the test call this):
      * ``negative_control`` PASS iff the DeLong 95% CI on the real-signal OOF CROSSES 0.50 (lower <=
        0.50 <= upper) — no separable path exists, chance is the correct expectation and any tight
        departure is a genuine leak signature.
      * ``positive_control`` PASS iff the real-signal AUROC > POSITIVE_REAL_MIN AND the shuffle
        companion AUROC <= POSITIVE_SHUFFLE_MAX (the pipeline recovers a known injected signal and
        collapses to chance when labels are permuted — proof the chain is not silently dead).

    When ``real_oof``/``y`` is None or the labels are single-class (e.g. the n=1 desktop adapter path,
    too small for CV), returns a ``NOT_ASSESSED`` block instead of raising — graceful degradation
    mirroring ``ablation._aggregate``'s NaN-tolerance.
    """
    import numpy as np

    if stream_name not in ("negative_control", "positive_control"):
        raise ValueError(
            f"stream_name must be 'negative_control' or 'positive_control', got {stream_name!r}"
        )
    if real_oof is None or y is None:
        return {
            "stream": stream_name,
            "verdict": "NOT_ASSESSED",
            "expected": "control sentinel needs a patient-grouped CV cohort (skipped at tiny N)",
            "note": "n too small for the coalition_oof sentinel (e.g. single-study desktop path)",
        }
    y_arr = np.asarray(y, int)
    if len(np.unique(y_arr)) < 2:
        return {
            "stream": stream_name,
            "verdict": "NOT_ASSESSED",
            "expected": "control sentinel needs both classes present",
            "note": "single-class cohort — the CV sentinel is undefined",
        }

    from sklearn.metrics import roc_auc_score

    from pinksight.metrics import delong_ci

    real_auc = float(roc_auc_score(y_arr, np.asarray(real_oof, float)))
    shuffle_auc = (
        float(roc_auc_score(y_arr, np.asarray(shuffle_oof, float)))
        if shuffle_oof is not None
        else float("nan")
    )
    auc, lo, hi = delong_ci(y_arr, np.asarray(real_oof, float))

    if stream_name == "negative_control":
        passed = bool(lo <= CHANCE <= hi)
        expected = "chance (~0.50, CI crosses 0.50)"
    else:
        passed = bool(real_auc > POSITIVE_REAL_MIN and shuffle_auc <= POSITIVE_SHUFFLE_MAX)
        expected = f"real > {POSITIVE_REAL_MIN} AND shuffle <= {POSITIVE_SHUFFLE_MAX}"

    # Label-attribution leak test (additive, non-breaking): does the REAL-label AUROC exceed the
    # permutation-null? For a negative control, leakFreeByShuffle=True is the leak-free reading (real ~=
    # shuffle -> labels buy nothing) even when the tight n=10k CI floats just above 0.50; for a positive
    # control it is False by design (real >> shuffle -> the injected signal IS label-attributable). None
    # when no shuffle companion was supplied. See SHUFFLE_LEAK_MARGIN.
    has_shuffle = not bool(np.isnan(shuffle_auc))
    label_signal = real_auc - shuffle_auc
    leak_free_by_shuffle = bool(real_auc <= shuffle_auc + SHUFFLE_LEAK_MARGIN) if has_shuffle else None

    return {
        "stream": stream_name,
        "auroc": round(real_auc, 4),
        "delongCi95": [round(float(lo), 4), round(float(hi), 4)],
        "shuffleAuroc": round(shuffle_auc, 4),
        "shuffleAtChance": bool(0.45 <= shuffle_auc <= 0.55),
        "labelAttributableSignal": round(label_signal, 4) if has_shuffle else None,
        "leakFreeByShuffle": leak_free_by_shuffle,
        "expected": expected,
        "verdict": "PASS" if passed else "FAIL",
    }


def _fixture_manifest() -> dict[str, Any]:
    """A tiny in-memory manifest for the selfcheck (no cohort generation, no torch)."""
    return {
        "manifest_sha256": "0" * 64,
        "generated_at": "1970-01-01T00:00:00+00:00",
        "config": {
            "n": 4,
            "seed": 0,
            "cube_size": 16,
            "channels": "pre_post",
            "effect_size": 1.2,
            "stream_name": "negative_control",
            "git_commit": "unknown",
        },
    }


def selfcheck() -> int:
    """Runnable check (mirrors ispy2_dataset/pcr_head/nyu_duke_slot convention): build_report assembles
    a shape-complete report; assert_synthetic_provenance passes on a correct tag and RAISES on both a
    stripped tag and a mismatched hash; control_verdict returns NOT_ASSESSED at tiny N. Exits 0."""
    manifest = _fixture_manifest()
    subtype_out = {"label": "Luminal A", "probability": 0.5, "uncertainty": [0.4, 0.6], "abstained": False}
    modalities = [
        {"modality": "clinical", "present": True, "contribution": 0.0},
        {"modality": "mri", "present": True, "contribution": 0.0},
        {"modality": "path", "present": False, "contribution": 0.0},
        {"modality": "genomic", "present": False, "contribution": 0.0},
    ]
    report = build_report(
        "negative_control", manifest, subtype_out, None, None, modalities, None,
        control_verdict("negative_control"),  # NOT_ASSESSED path (no oofs)
    )

    # Every field the desktop CharacterisationReport TS type requires must be present.
    required = ("studyId", "subtype", "ki67Descriptor", "nottinghamGrade", "calibration",
                "modalities", "audit", "provenance")
    missing = [k for k in required if k not in report]
    assert not missing, f"report missing desktop-required fields: {missing}"
    assert report["provenance"]["datasetTag"] == SYNTHETIC_TAG, "tag not stamped"
    assert report["controlVerdict"]["verdict"] == "NOT_ASSESSED", "tiny-N control should be NOT_ASSESSED"

    # (1) Pass path: a correct tag + hash is accepted.
    assert_synthetic_provenance(report, manifest["manifest_sha256"])

    # (2) Deliberate-violation path A: a stripped tag must RAISE.
    tampered = {**report, "provenance": {**report["provenance"], "datasetTag": "not synthetic"}}
    try:
        assert_synthetic_provenance(tampered, manifest["manifest_sha256"])
    except SyntheticProvenanceError:
        pass
    else:
        raise AssertionError("assert_synthetic_provenance failed to fire on a stripped datasetTag")

    # (3) Deliberate-violation path B: a mismatched manifest hash must RAISE.
    try:
        assert_synthetic_provenance(report, "f" * 64)
    except SyntheticProvenanceError:
        pass
    else:
        raise AssertionError("assert_synthetic_provenance failed to fire on a mismatched hash")

    # (4) No-cross-organ-pooling firewall: a clean run-of-runs manifest passes; a banned top-level key
    #     and a banned key nested one level inside an organ block both RAISE (LOCK-1, mechanical).
    clean_manifest = {
        "fused-track-a-realistic": {
            "negative_control": control_verdict("negative_control"),
            "positive_control": control_verdict("positive_control"),
        },
        "trackb-wsi-genomics": {
            "negative_control": control_verdict("negative_control"),
            "positive_control": control_verdict("positive_control"),
        },
        "_generatedAt": "1970-01-01T00:00:00+00:00",
        "_note": "run-of-runs index — SYNTHETIC — NOT A RESULT; each organ listed separately, never a "
                 "pooled or cross-organ number.",
    }
    assert_no_cross_organ_pooling(clean_manifest)  # clean manifest passes without raising

    try:
        assert_no_cross_organ_pooling({**clean_manifest, "pooled_auroc": 0.9})
    except AssertionError:
        pass
    else:
        raise AssertionError("assert_no_cross_organ_pooling failed to fire on a banned top-level key")

    try:
        assert_no_cross_organ_pooling(
            {"trackb-wsi-genomics": {
                "negative_control": control_verdict("negative_control"),
                "vs_trackc_delta": 0.02,
            }}
        )
    except AssertionError:
        pass
    else:
        raise AssertionError("assert_no_cross_organ_pooling failed to fire on a nested banned key")

    print(  # noqa: T201
        "e2e_report_contract selfcheck OK — build_report shape-complete; provenance gate passes on a "
        "correct tag and RAISES on both a stripped tag and a mismatched hash; control NOT_ASSESSED at "
        "N=0; no-cross-organ-pooling firewall passes clean and RAISES on a banned top-level + nested key."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(selfcheck())

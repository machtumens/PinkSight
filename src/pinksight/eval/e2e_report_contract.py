
from __future__ import annotations

from typing import Any

SYNTHETIC_TAG = "SYNTHETIC — NOT A RESULT"
REPORT_VERSION = "e2e-synthetic-v1"

POSITIVE_REAL_MIN = 0.75   
POSITIVE_SHUFFLE_MAX = 0.55  
CHANCE = 0.50
SHUFFLE_LEAK_MARGIN = 0.02

KI67_DESCRIPTOR_DEFAULT = (
    "Descriptive companion — imaging correlate of Ki-67 index at diagnosis (aggressiveness snapshot). "
    "SYNTHETIC-ONLY plumbing: proves the head's tensor wiring end-to-end; does not evaluate the real "
    "Ki-67 head (N=0 real labels, unchanged)."
)

_SYNTHETIC_NOTTINGHAM: dict[str, Any] = {
    "label": "NHG1",
    "probability": 0.5,
    "uncertainty": [0.4, 0.6],
}


class SyntheticProvenanceError(AssertionError):
    pass


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


_POOLING_BANNED_SUBSTRINGS = (
    "pooled", "combined", "overall", "aggregate", "delta", "ranking", "comparison", "juxtapos",
)


def assert_no_cross_organ_pooling(manifest_of_runs: dict[str, dict]) -> None:
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
        control_verdict("negative_control"),  
    )

    required = ("studyId", "subtype", "ki67Descriptor", "nottinghamGrade", "calibration",
                "modalities", "audit", "provenance")
    missing = [k for k in required if k not in report]
    assert not missing, f"report missing desktop-required fields: {missing}"
    assert report["provenance"]["datasetTag"] == SYNTHETIC_TAG, "tag not stamped"
    assert report["controlVerdict"]["verdict"] == "NOT_ASSESSED", "tiny-N control should be NOT_ASSESSED"

    assert_synthetic_provenance(report, manifest["manifest_sha256"])

    tampered = {**report, "provenance": {**report["provenance"], "datasetTag": "not synthetic"}}
    try:
        assert_synthetic_provenance(tampered, manifest["manifest_sha256"])
    except SyntheticProvenanceError:
        pass
    else:
        raise AssertionError("assert_synthetic_provenance failed to fire on a stripped datasetTag")

    try:
        assert_synthetic_provenance(report, "f" * 64)
    except SyntheticProvenanceError:
        pass
    else:
        raise AssertionError("assert_synthetic_provenance failed to fire on a mismatched hash")

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
    assert_no_cross_organ_pooling(clean_manifest)  

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

    print(  
        "e2e_report_contract selfcheck OK — build_report shape-complete; provenance gate passes on a "
        "correct tag and RAISES on both a stripped tag and a mismatched hash; control NOT_ASSESSED at "
        "N=0; no-cross-organ-pooling firewall passes clean and RAISES on a banned top-level + nested key."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(selfcheck())

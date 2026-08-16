import os

from main import build_report, assert_ledger_safe, FORBIDDEN_TERMS

GOLDEN_MOCK_REPORT = {
    "studyId": "GOLDEN-0001",
    "subtype": {
        "label": "Luminal A",
        "probability": 0.71,
        "uncertainty": [0.58, 0.83],
        "abstained": False,
    },
    "ki67Descriptor": (
        "Descriptive companion — imaging correlate of Ki-67 index at "
        "diagnosis (aggressiveness snapshot, not kinetics)."
    ),
    "nottinghamGrade": {"label": "NHG1", "probability": 0.68, "uncertainty": [0.55, 0.80]},
    "calibration": {"ece": 0.041, "band": "good"},
    "modalities": [
        {"modality": "clinical", "present": True, "contribution": 0.135},
        {"modality": "mri", "present": True, "contribution": -0.025},
        {"modality": "path", "present": False, "contribution": 0.0},
        {"modality": "genomic", "present": False, "contribution": 0.0},
    ],
    "audit": {
        "modelHash": "sha256:demo0000",
        "seed": 42,
        "split": "split_v2 (scanner holdout)",
        "generatedAt": "mocked",
    },
}


def test_report_shape_and_ledger():
    r = build_report("TEST-0001")
    assert r["studyId"] == "TEST-0001"
    assert r["subtype"]["label"] in ("Luminal A", "Triple-Negative")
    lo, hi = r["subtype"]["uncertainty"]
    assert 0.0 <= lo <= r["subtype"]["probability"] <= hi <= 1.0, "point must sit inside band"
    assert r["calibration"]["band"] in ("good", "acceptable", "poor")
    assert_ledger_safe(r)


def test_guard_actually_catches_violations():
    for term in FORBIDDEN_TERMS:
        bad = {"note": f"this model does {term} of tumours"}
        try:
            assert_ledger_safe(bad)
        except ValueError:
            continue
        raise AssertionError(f"guard failed to catch forbidden term: {term}")


def test_synthetic_live_report_is_ledger_safe_and_shape_compatible():
    from pinksight_report_adapter import get_synthetic_report, SYNTHETIC_TAG

    r = get_synthetic_report("LIVE-0001", stream="positive_control")

    for key in ("studyId", "subtype", "ki67Descriptor", "nottinghamGrade",
                "calibration", "modalities", "audit"):
        assert key in r, f"synthetic report missing desktop-required field: {key}"

    assert r["studyId"] == "LIVE-0001", "caller studyId must label the report"
    assert r["provenance"]["datasetTag"] == SYNTHETIC_TAG, "live report must stay SYNTHETIC-tagged"

    assert r["subtype"]["label"] in ("Luminal A", "Triple-Negative")
    lo, hi = r["subtype"]["uncertainty"]
    assert 0.0 <= lo <= r["subtype"]["probability"] <= hi <= 1.0, "point must sit inside band"

    assert_ledger_safe(r)


def test_existing_mock_path_unchanged():
    prev = os.environ.pop("PINKSIGHT_SIDECAR_LIVE", None)
    try:
        r = build_report("GOLDEN-0001")
        assert r == GOLDEN_MOCK_REPORT, "default mock path regressed — byte/shape mismatch vs golden"
    finally:
        if prev is not None:
            os.environ["PINKSIGHT_SIDECAR_LIVE"] = prev


def test_dispatch_block_echoed_into_infer_report():
    r = build_report("T-1", cohort="duke", modalities=["mri", "clinical"])
    assert r["dispatch"]["status"] == "WIRED"
    assert r["dispatch"]["crossCohortGradient"] is False
    assert r["dispatch"]["harnessScript"] == "scripts/train_g3_hierarchical.py"
    assert_ledger_safe(r)


def test_live_override_forces_mock_even_with_env_var_set():
    prev = os.environ.get("PINKSIGHT_SIDECAR_LIVE")
    os.environ["PINKSIGHT_SIDECAR_LIVE"] = "1"
    try:
        r = build_report("T-2", live_override=False)
        assert r["audit"]["generatedAt"] == "mocked", "live_override=False must force the mock path"
    finally:
        if prev is None:
            os.environ.pop("PINKSIGHT_SIDECAR_LIVE", None)
        else:
            os.environ["PINKSIGHT_SIDECAR_LIVE"] = prev


if __name__ == "__main__":
    test_report_shape_and_ledger()
    test_guard_actually_catches_violations()
    test_existing_mock_path_unchanged()
    test_dispatch_block_echoed_into_infer_report()
    test_live_override_forces_mock_even_with_env_var_set()
    try:
        test_synthetic_live_report_is_ledger_safe_and_shape_compatible()
        _live = "synthetic-live report ledger-safe + shape-compatible"
    except ImportError:
        _live = "synthetic-live test SKIPPED (pinksight/ml stack unavailable in bare-python mode)"
    print(  
        "ok — report shape valid, ledger guard catches all forbidden terms, "
        "mock path byte-unchanged, dispatch block echoed, live_override narrows env gate, " + _live
    )

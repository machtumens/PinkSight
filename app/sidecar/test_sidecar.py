"""
One runnable check for the sidecar's real logic: the claim-ledger guard, plus the additive
synthetic-live wiring (default-OFF ``PINKSIGHT_SIDECAR_LIVE`` env gate) and a golden-diff regression
guard proving the default mock path is byte/shape-unchanged.
Run: `python sidecar/test_sidecar.py`  (no framework needed; the mock-path tests need no ml stack —
the synthetic-live test additionally needs the `pinksight` package + ml stack, imported lazily inside
that test so the other tests still run on a bare box).
"""
import os

from main import build_report, assert_ledger_safe, FORBIDDEN_TERMS

# Recorded golden for the DEFAULT (mock) path — proves zero behavioural regression once the
# synthetic-live branch is added to build_report. Byte/shape-identical to the original mock report
# (only studyId varies with the argument).
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
    # every valid report is ledger-safe by construction
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
    # Call the adapter directly (not via HTTP — uvicorn/FastAPI may be absent). Imported lazily so the
    # mock-path tests above still run on a box without the pinksight/ml stack.
    from pinksight_report_adapter import get_synthetic_report, SYNTHETIC_TAG

    r = get_synthetic_report("LIVE-0001", stream="positive_control")

    # every field the desktop CharacterisationReport TS type requires is present
    for key in ("studyId", "subtype", "ki67Descriptor", "nottinghamGrade",
                "calibration", "modalities", "audit"):
        assert key in r, f"synthetic report missing desktop-required field: {key}"

    # the caller's studyId labels the report, but provenance stays honestly non-reportable
    assert r["studyId"] == "LIVE-0001", "caller studyId must label the report"
    assert r["provenance"]["datasetTag"] == SYNTHETIC_TAG, "live report must stay SYNTHETIC-tagged"

    # subtype block is shape-compatible with the existing mock contract
    assert r["subtype"]["label"] in ("Luminal A", "Triple-Negative")
    lo, hi = r["subtype"]["uncertainty"]
    assert 0.0 <= lo <= r["subtype"]["probability"] <= hi <= 1.0, "point must sit inside band"

    # a live synthetic report is ledger-safe by construction (the same guard the sidecar enforces)
    assert_ledger_safe(r)


def test_existing_mock_path_unchanged():
    # Default (env gate OFF) must be byte/shape-identical to the recorded golden — zero regression.
    prev = os.environ.pop("PINKSIGHT_SIDECAR_LIVE", None)
    try:
        r = build_report("GOLDEN-0001")
        assert r == GOLDEN_MOCK_REPORT, "default mock path regressed — byte/shape mismatch vs golden"
    finally:
        if prev is not None:
            os.environ["PINKSIGHT_SIDECAR_LIVE"] = prev


def test_dispatch_block_echoed_into_infer_report():
    # When cohort+modalities are supplied, build_report echoes the pure dispatch routing block (a
    # routing decision, never a per-organ number). One WIRED row (duke, {mri, clinical}) -> G3 harness.
    r = build_report("T-1", cohort="duke", modalities=["mri", "clinical"])
    assert r["dispatch"]["status"] == "WIRED"
    assert r["dispatch"]["crossCohortGradient"] is False
    assert r["dispatch"]["harnessScript"] == "scripts/train_g3_hierarchical.py"
    # double-prove ledger safety on the augmented payload (build_report already ran the guard once)
    assert_ledger_safe(r)


def test_live_override_forces_mock_even_with_env_var_set():
    # Per-request live_override=False must NARROW the env gate: even with PINKSIGHT_SIDECAR_LIVE=1 set,
    # the mock path is taken. audit.generatedAt == "mocked" is the fixed-mock branch's tell (the live
    # synthetic report has no such marker — it carries a top-level provenance block instead).
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
        # bare-python (no uv) can't import the pinksight package — the mock-path tests above still
        # prove the ledger guard + zero-regression; run `uv run pytest` for the synthetic-live test.
        _live = "synthetic-live test SKIPPED (pinksight/ml stack unavailable in bare-python mode)"
    print(  # noqa: T201
        "ok — report shape valid, ledger guard catches all forbidden terms, "
        "mock path byte-unchanged, dispatch block echoed, live_override narrows env gate, " + _live
    )

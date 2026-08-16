
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("monai")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import e2e_synthetic_harness_run as harness

from pinksight import FORBIDDEN_FEATURES
from pinksight.data.synthetic_cohort import (
    generate_negative_control,
    generate_positive_control,
)
from pinksight.eval.e2e_report_contract import (
    SYNTHETIC_TAG,
    SyntheticProvenanceError,
    assert_synthetic_provenance,
)
from pinksight.models.clinical_encoder import FEATURES_CAT, FEATURES_NUM

SMOKE_N = 40
SMOKE_CUBE = 8
SMOKE_SEED = 8
SMOKE_EFFECT = 1.5


@pytest.fixture(scope="module")
def streams(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("e2e_synthetic")
    neg = harness.run_stream(
        "negative_control", n=SMOKE_N, seed=SMOKE_SEED, cube_size=SMOKE_CUBE,
        xai_subsample=0, batch_size=32, out_dir=out_dir,
    )
    pos = harness.run_stream(
        "positive_control", n=SMOKE_N, seed=SMOKE_SEED, cube_size=SMOKE_CUBE,
        xai_subsample=3, batch_size=32, out_dir=out_dir, effect_size=SMOKE_EFFECT,
    )
    return {"negative": neg, "positive": pos, "out_dir": out_dir}


def test_full_chain_runs_without_error(streams):
    for key in ("negative", "positive"):
        res = streams[key]
        assert len(res["patient_ids"]) == SMOKE_N
        assert res["mri_embeddings"].shape == (SMOKE_N, 512)
        assert np.isfinite(res["mri_embeddings"]).all()
        assert "reportVersion" in res["report"]
        assert res["report"]["provenance"]["datasetTag"] == SYNTHETIC_TAG


def test_negative_control_auroc_near_chance(streams):
    cv = streams["negative"]["control_verdict"]
    lo, hi = cv["delongCi95"]
    assert lo <= 0.50 <= hi, f"negative control CI {cv['delongCi95']} must cross 0.50 (leakage sentinel)"
    assert cv["verdict"] == "PASS"


def test_positive_control_recovers_signal_and_shuffle_collapses(streams):
    cv = streams["positive"]["control_verdict"]
    assert cv["auroc"] > 0.75, f"positive control real AUROC {cv['auroc']} should clear 0.75"
    assert cv["shuffleAuroc"] < cv["auroc"] - 0.15, (
        f"shuffle {cv['shuffleAuroc']} did not collapse below real {cv['auroc']} — the chain may be "
        "leaking the label or is silently dead"
    )


def _collect_keys(obj) -> set:
    keys: set = set()
    if isinstance(obj, dict):
        keys |= set(obj.keys())
        for v in obj.values():
            keys |= _collect_keys(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            keys |= _collect_keys(v)
    return keys


def test_no_forbidden_features_in_synthetic_inputs_or_report(streams):
    for _pid, _mri, row, _label in generate_negative_control(6, seed=0, cube_size=4):
        assert set(row).isdisjoint(FORBIDDEN_FEATURES), f"forbidden key in synthetic row: {set(row)}"
    report_keys = _collect_keys(streams["positive"]["report"])
    leaked = report_keys & set(FORBIDDEN_FEATURES)
    assert not leaked, f"forbidden feature name leaked as a report key: {sorted(leaked)}"


def test_modality_dropout_clinical_only_path_runs(streams):
    assert streams["negative"]["dropout_clinical_only_ran"] is True
    assert streams["positive"]["dropout_clinical_only_ran"] is True
    modalities = {m["modality"]: m for m in streams["positive"]["report"]["modalities"]}
    assert modalities["clinical"]["present"] is True


def test_manifest_hash_gate_raises_on_stripped_tag(streams):
    res = streams["negative"]
    report, manifest = res["report"], res["manifest"]
    assert_synthetic_provenance(report, manifest["manifest_sha256"])
    stripped = {**report, "provenance": {**report["provenance"], "datasetTag": "REAL"}}
    with pytest.raises(SyntheticProvenanceError):
        assert_synthetic_provenance(stripped, manifest["manifest_sha256"])
    with pytest.raises(SyntheticProvenanceError):
        assert_synthetic_provenance(report, "0" * 64)


def test_negative_and_positive_streams_never_cross_id_namespace():
    neg_ids = {pid for pid, *_ in generate_negative_control(50, seed=0, cube_size=4)}
    pos_ids = {pid for pid, *_ in generate_positive_control(50, seed=0, cube_size=4)}
    assert neg_ids.isdisjoint(pos_ids), "SYN-NEG and SYN-POS namespaces overlap (DD-2 violated)"
    assert all(pid.startswith("SYN-NEG-") for pid in neg_ids)
    assert all(pid.startswith("SYN-POS-") for pid in pos_ids)


def test_report_contract_has_every_desktop_expected_field(streams):
    report = streams["positive"]["report"]
    for field in ("studyId", "subtype", "ki67Descriptor", "nottinghamGrade", "calibration",
                  "modalities", "audit"):
        assert field in report, f"desktop-required field {field!r} missing"
    for field in ("label", "probability", "uncertainty", "abstained"):
        assert field in report["subtype"], f"subtype.{field} missing"
    for field in ("label", "probability", "uncertainty"):
        assert field in report["nottinghamGrade"], f"nottinghamGrade.{field} missing"
    for field in ("ece", "band"):
        assert field in report["calibration"], f"calibration.{field} missing"
    for mod in report["modalities"]:
        assert {"modality", "present", "contribution"} <= set(mod)
    for field in ("modelHash", "seed", "split", "generatedAt"):
        assert field in report["audit"], f"audit.{field} missing"


def test_xai_stage_produces_finite_shaped_maps(streams):
    xai = streams["positive"]["xai"]
    assert xai is not None, "positive stream must produce an XAI block"
    assert xai["mapRef"], "XAI mapRef path must be set"
    assert isinstance(xai["iou"], float) and np.isfinite(xai["iou"])
    assert isinstance(xai["pointingGame"], bool)
    assert isinstance(xai["randomizationPassed"], bool)
    saved = streams["out_dir"] / xai["mapRef"]
    cam = np.load(saved)
    assert cam.shape == (SMOKE_CUBE, SMOKE_CUBE, SMOKE_CUBE)
    assert np.isfinite(cam).all()


def test_h0_fallback_fires_on_degenerate_mask():
    degenerate_pair = np.ones((2, 16, 16, 16), dtype=np.float32)
    crop, used_fallback, mask_cube = harness.stage1_lesion_crop(degenerate_pair, cube_size=8)
    assert used_fallback is True, "an all-False mask must trigger the [1.6] Duke-box fallback"
    assert crop.shape == (2, 8, 8, 8)
    assert np.isfinite(crop).all()
    assert mask_cube.shape == (8, 8, 8)


def test_synthetic_rows_use_the_real_clinical_feature_schema():
    _pid, _mri, row, _label = next(iter(generate_negative_control(2, seed=0, cube_size=4)))
    assert set(row) == set(FEATURES_NUM) | set(FEATURES_CAT)
    assert len(row) == 9


from __future__ import annotations

import pytest

pytest.importorskip("scipy", reason="scipy is an ml/arms/trackb-extra dep — skip cleanly under a base install; kit imports the scipy-backed eval/stats stack")

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from pinksight.eval.e2e_report_contract import (
    SYNTHETIC_TAG,
    assert_no_cross_organ_pooling,
    control_verdict,
)

EXPECTED_ORGANS = {
    "fused-track-a-realistic",
    "trackb-wsi-genomics",
    "trackc-coimbra",
    "fastmri-nyu-standalone",
}
_BANNED = ("pooled", "combined", "overall", "aggregate", "delta", "ranking", "comparison", "juxtapos")


def _full_verdict(stream: str) -> dict:
    rng = np.random.default_rng(0)
    n = 200
    y = np.array([0, 1] * (n // 2))
    noise = rng.normal(size=n)
    return control_verdict(stream, y=y, real_oof=noise, shuffle_oof=noise)


def _clean_manifest() -> dict:
    return {
        organ: {"negative_control": _full_verdict("negative_control"),
                "positive_control": _full_verdict("positive_control")}
        for organ in EXPECTED_ORGANS
    } | {
        "_generatedAt": "1970-01-01T00:00:00+00:00",
        "_note": "run-of-runs index — SYNTHETIC — NOT A RESULT; each organ listed separately.",
    }


@pytest.mark.parametrize("banned", _BANNED)
def test_guard_raises_top_level(banned):
    manifest = _clean_manifest()
    manifest[f"{banned}_auroc"] = 0.9  
    with pytest.raises(AssertionError):
        assert_no_cross_organ_pooling(manifest)


def test_guard_raises_nested():
    manifest = {
        "trackb-wsi-genomics": {
            "negative_control": control_verdict("negative_control"),
            "vs_trackc_delta": 0.02,
        }
    }
    with pytest.raises(AssertionError):
        assert_no_cross_organ_pooling(manifest)


def test_guard_passes_clean_manifest():
    manifest = _clean_manifest()
    assert set(k for k, v in manifest.items() if isinstance(v, dict)) == EXPECTED_ORGANS
    assert_no_cross_organ_pooling(manifest)  
    fields = set(_full_verdict("negative_control"))
    assert not any(b in f.lower() for f in fields for b in _BANNED), fields


def test_guard_rejects_non_dict():
    with pytest.raises(AssertionError):
        assert_no_cross_organ_pooling(["not", "a", "dict"])  


def test_run_stream_per_patient_keys(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("monai")
    import e2e_synthetic_harness_run as harness

    n = 40
    res = harness.run_stream(
        "negative_control", n=n, seed=0, cube_size=8, xai_subsample=0, batch_size=16, out_dir=tmp_path,
    )
    sp = np.asarray(res["per_patient_subtype_prob"], dtype=float)
    kr = np.asarray(res["per_patient_ki67_raw"], dtype=float)
    assert sp.shape == (n,), sp.shape
    assert kr.shape == (n,), kr.shape
    assert np.isfinite(sp).all() and np.isfinite(kr).all()
    assert (sp >= 0.0).all() and (sp <= 1.0).all(), (sp.min(), sp.max())
    assert len(res["patient_ids"]) == n


@pytest.fixture(scope="module")
def deliverable_a_smallN(tmp_path_factory):
    pytest.importorskip("torch")
    pytest.importorskip("monai")
    import e2e_synthetic_fused_10k_validation_run as fused_kit

    out = tmp_path_factory.mktemp("fused10k_smoke")
    rc = fused_kit.main(
        ["--n-patients", "60", "--cube-size", "8", "--xai-subsample", "0", "--seed", "0",
         "--out-dir", str(out)]
    )
    assert rc == 0
    return out


def _read_jsonl(out_dir: Path, stream: str) -> tuple[dict, list[dict]]:
    lines = (out_dir / f"e2e_synthetic_fused_10k_{stream}_per_sample.jsonl").read_text().splitlines()
    return json.loads(lines[0]), [json.loads(x) for x in lines[1:]]


@pytest.mark.parametrize("stream", ["negative_control", "positive_control"])
def test_deliverable_a_smallN_jsonl_shape(deliverable_a_smallN, stream):
    header, rows = _read_jsonl(deliverable_a_smallN, stream)
    n = header["n"]
    assert header.get("_manifest") is True
    assert {"manifestSha256", "n", "stream", "datasetTag"} <= set(header)
    assert header["stream"] == stream
    assert header["datasetTag"] == SYNTHETIC_TAG
    assert len(rows) == n
    assert all(r["datasetTag"] == SYNTHETIC_TAG for r in rows)
    probs = np.array([r["subtypeProbability"] for r in rows], dtype=float)
    assert probs.std() > 0.0, "subtypeProbability is constant — per-patient signal not wired through"
    assert ((probs >= 0.0) & (probs <= 1.0)).all()


@pytest.mark.parametrize("stream", ["negative_control", "positive_control"])
def test_deliverable_a_smallN_constants(deliverable_a_smallN, stream):
    header, rows = _read_jsonl(deliverable_a_smallN, stream)
    assert {r["ki67Stratum"] for r in rows} == {"not_assessed"}
    assert {r["nottinghamGradeLabel"] for r in rows} == {"NHG1"}
    assert {r["nottinghamGradeProbability"] for r in rows} == {0.5}
    note = header["_note"]
    assert isinstance(note, str) and note
    assert "NHG1" in note and "not_assessed" in note
    assert "stratification" in note.lower()  


def test_deliverable_b_smallN_manifest():
    pytest.importorskip("torch")
    pytest.importorskip("monai")
    import e2e_synthetic_companion_index_run as companion

    fused_neg = control_verdict("negative_control")
    fused_pos = control_verdict("positive_control")
    manifest = companion.assemble_manifest_of_runs(
        60, 0, fused_neg, fused_pos, cube_size=8, batch_size=32,
    )
    organ_keys = {k for k, v in manifest.items() if isinstance(v, dict)}
    assert organ_keys == EXPECTED_ORGANS, organ_keys
    assert_no_cross_organ_pooling(manifest)
    for organ in organ_keys:
        assert set(manifest[organ]) == {"negative_control", "positive_control"}, organ
        assert "verdict" in manifest[organ]["negative_control"]
        assert "verdict" in manifest[organ]["positive_control"]

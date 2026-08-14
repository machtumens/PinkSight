"""Fast small-N smoke gate for the fused-model + companion-index n=10,000 SYNTHETIC validation kit.

Covers the two new deliverables and the new LOCK-1 guard at a small deterministic scale so the whole
suite stays part of the default ``pytest -q`` pass. The wall-clock-heavy n=10,000 runs are standalone
deliverables (DD-3), NOT part of this fast suite.

  * ``assert_no_cross_organ_pooling`` — the new mechanical LOCK-1 firewall: raises on a banned top-level
    key, raises on a banned key nested one level inside an organ block, passes on a clean 4-organ
    manifest built from real ``control_verdict()`` output. (Torch-free — always runs.)
  * ``run_stream`` per-patient keys — the additive ``per_patient_subtype_prob`` / ``per_patient_ki67_raw``
    return-dict keys have the right shape / finiteness / range. (torch+monai ``importorskip``.)
  * Deliverable A JSONL — row count == n+1, every data row is SYNTHETIC-tagged, ``subtypeProbability``
    varies per patient, and the grade / ki67-stratum placeholder columns are constant + documented.
  * Deliverable B manifest — exactly the 4 expected organ keys, the guard passes, each stream carries a
    verdict field.

Nothing here is a scientific result: every number is control-sentinel plumbing on fabricated data, no
LOCK is moved.
"""

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


# ==================================================================================================
# Guard — assert_no_cross_organ_pooling (torch-free, always runs)
# ==================================================================================================
def _full_verdict(stream: str) -> dict:
    """A full (non-NOT_ASSESSED) controlVerdict from fixed OOF arrays — carries every real verdict field
    name (auroc/delongCi95/shuffleAuroc/... ) so a clean manifest built from it proves no field-name in
    the real verdict schema collides with the banned-substring list. Deterministic, torch-free."""
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
    """A banned substring in ANY top-level key (organ name or metadata) must RAISE (LOCK-1)."""
    manifest = _clean_manifest()
    manifest[f"{banned}_auroc"] = 0.9  # smuggle a pooled/combined/delta/... top-level number
    with pytest.raises(AssertionError):
        assert_no_cross_organ_pooling(manifest)


def test_guard_raises_nested():
    """A banned key smuggled one level down inside an organ block must RAISE (the example from the
    Public Contract: {"trackb-...": {"negative_control": {...}, "vs_trackc_delta": 0.02}})."""
    manifest = {
        "trackb-wsi-genomics": {
            "negative_control": control_verdict("negative_control"),
            "vs_trackc_delta": 0.02,
        }
    }
    with pytest.raises(AssertionError):
        assert_no_cross_organ_pooling(manifest)


def test_guard_passes_clean_manifest():
    """A clean 4-organ manifest built from real control_verdict() output passes without raising, and no
    real verdict field name collides with the banned-substring list."""
    manifest = _clean_manifest()
    assert set(k for k, v in manifest.items() if isinstance(v, dict)) == EXPECTED_ORGANS
    assert_no_cross_organ_pooling(manifest)  # must not raise
    # sanity: the real verdict field names really are collision-free with the banned list
    fields = set(_full_verdict("negative_control"))
    assert not any(b in f.lower() for f in fields for b in _BANNED), fields


def test_guard_rejects_non_dict():
    """A non-dict manifest is a structural error — the guard RAISES rather than silently passing."""
    with pytest.raises(AssertionError):
        assert_no_cross_organ_pooling(["not", "a", "dict"])  # type: ignore[arg-type]


# ==================================================================================================
# Deliverable A — run_stream additive per-patient keys (torch + monai)
# ==================================================================================================
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


# ==================================================================================================
# Deliverable A — per-sample JSONL (torch + monai; kit run once, shared across tests)
# ==================================================================================================
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
    # header + n data rows == n + 1 total lines
    assert len(rows) == n
    assert all(r["datasetTag"] == SYNTHETIC_TAG for r in rows)
    # subtypeProbability is a genuine per-patient forward-pass output → not all identical
    probs = np.array([r["subtypeProbability"] for r in rows], dtype=float)
    assert probs.std() > 0.0, "subtypeProbability is constant — per-patient signal not wired through"
    assert ((probs >= 0.0) & (probs <= 1.0)).all()


@pytest.mark.parametrize("stream", ["negative_control", "positive_control"])
def test_deliverable_a_smallN_constants(deliverable_a_smallN, stream):
    header, rows = _read_jsonl(deliverable_a_smallN, stream)
    # D2/D3 placeholder columns are constant across every row (never a fabricated per-patient head).
    assert {r["ki67Stratum"] for r in rows} == {"not_assessed"}
    assert {r["nottinghamGradeLabel"] for r in rows} == {"NHG1"}
    assert {r["nottinghamGradeProbability"] for r in rows} == {0.5}
    # ...and the header documents WHY they are constant (self-describing file).
    note = header["_note"]
    assert isinstance(note, str) and note
    assert "NHG1" in note and "not_assessed" in note
    assert "stratification" in note.lower()  # explicitly flags ki67RawValue is NOT a stratification


# ==================================================================================================
# Deliverable B — companion index manifest_of_runs (torch + monai; runs 3 organs fresh)
# ==================================================================================================
def test_deliverable_b_smallN_manifest():
    pytest.importorskip("torch")
    pytest.importorskip("monai")
    import e2e_synthetic_companion_index_run as companion

    # Fused Track-A verdicts are READ (never re-run); a real control_verdict dict stands in for the
    # loaded scorecard here (this test proves the aggregate WIRING + guard, not the fused 10k numbers).
    fused_neg = control_verdict("negative_control")
    fused_pos = control_verdict("positive_control")
    manifest = companion.assemble_manifest_of_runs(
        60, 0, fused_neg, fused_pos, cube_size=8, batch_size=32,
    )
    organ_keys = {k for k, v in manifest.items() if isinstance(v, dict)}
    assert organ_keys == EXPECTED_ORGANS, organ_keys
    # the guard passes on the freshly-assembled manifest (no pooled/compared key smuggled in)
    assert_no_cross_organ_pooling(manifest)
    for organ in organ_keys:
        assert set(manifest[organ]) == {"negative_control", "positive_control"}, organ
        assert "verdict" in manifest[organ]["negative_control"]
        assert "verdict" in manifest[organ]["positive_control"]

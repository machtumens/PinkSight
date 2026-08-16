
from __future__ import annotations

import pytest

pytest.importorskip("scipy", reason="scipy is an ml/arms/trackb-extra dep — skip cleanly under a base install; stream harness imports the scipy-backed stack")

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from pinksight import FORBIDDEN_FEATURES
from pinksight.data.synthetic_cohort import (
    A_NEG_PREFIX,
    A_POS_PREFIX,
    generate_negative_control,
    generate_positive_control,
    generate_realistic_negative_control,
    generate_realistic_positive_control,
)
from pinksight.data.synthetic_streams import (
    FASTMRI_CHANNELS,
    FASTMRI_IMAGE_FEATURES,
    MODALITY_C_GENES,
    UNI_DIM,
    build_stream_manifest,
    build_stream_report,
    generate_fastmri_stream,
    generate_tabular_stream,
    generate_wsi_genomics_stream,
)
from pinksight.eval.e2e_report_contract import (
    SyntheticProvenanceError,
    assert_synthetic_provenance,
    control_verdict,
)
from pinksight.models.clinical_encoder import FEATURES_CAT, FEATURES_NUM

SMOKE_N = 60


def test_all_stream_id_prefixes_pairwise_disjoint():
    a_neg = {pid for pid, *_ in generate_realistic_negative_control(SMOKE_N, seed=0, cube_size=4)}
    a_pos = {pid for pid, *_ in generate_realistic_positive_control(SMOKE_N, seed=0, cube_size=4)}
    plain_neg = {pid for pid, *_ in generate_negative_control(SMOKE_N, seed=0, cube_size=4)}
    plain_pos = {pid for pid, *_ in generate_positive_control(SMOKE_N, seed=0, cube_size=4)}
    c_ids, _cx, _cy = generate_tabular_stream(SMOKE_N, seed=0)
    c_ids = set(map(str, c_ids))
    f_ids = {pid for pid, *_ in generate_fastmri_stream(SMOKE_N, seed=0, cube_size=4)}
    b_ids = {pid for pid, *_ in generate_wsi_genomics_stream(SMOKE_N, seed=0, n_tiles_range=(4, 8))}

    namespaces = {
        "A-NEG": a_neg, "A-POS": a_pos, "plain-NEG": plain_neg, "plain-POS": plain_pos,
        "C": c_ids, "F": f_ids, "B": b_ids,
    }
    names = list(namespaces)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            assert namespaces[names[i]].isdisjoint(namespaces[names[j]]), (
                f"ID namespaces {names[i]} and {names[j]} overlap (DD-2 violated)"
            )
    assert all(pid.startswith(A_NEG_PREFIX) for pid in a_neg)
    assert all(pid.startswith(A_POS_PREFIX) for pid in a_pos)
    assert all(pid.startswith("SYN-F-") for pid in f_ids)
    assert all(pid.startswith("SYN-B-") for pid in b_ids)
    assert all(pid.startswith("SYN-C-") for pid in c_ids)


def test_stream_a_realistic_rows_forbidden_free_and_schema_matched():
    for _pid, _mri, row, _label in generate_realistic_negative_control(SMOKE_N, seed=0, cube_size=4):
        assert set(row) == set(FEATURES_NUM) | set(FEATURES_CAT), "realistic row keys != feature schema"
        assert set(row).isdisjoint(FORBIDDEN_FEATURES), f"forbidden key in realistic row: {set(row)}"


def test_stream_a_realistic_marginals_are_duke_like():
    neg = list(generate_realistic_negative_control(400, seed=0, cube_size=4))
    ages = np.array([row[FEATURES_NUM[3]] for _, _, row, _ in neg])  
    assert 25.0 <= ages.min() and ages.max() <= 90.0, (ages.min(), ages.max())
    assert 45.0 <= ages.mean() <= 67.0, ages.mean()  
    prev = float(np.mean([y for *_, y in neg]))  
    assert 0.10 <= prev <= 0.35, prev


def test_stream_a_positive_injects_clinical_shift_negative_lacks():
    def _tstage_gap(stream):
        c1 = [r[FEATURES_NUM[0]] for _, _, r, y in stream if y == 1]
        c0 = [r[FEATURES_NUM[0]] for _, _, r, y in stream if y == 0]
        return abs(float(np.mean(c1) - np.mean(c0)))

    neg = list(generate_realistic_negative_control(200, seed=1, cube_size=4))
    pos = list(generate_realistic_positive_control(200, seed=1, cube_size=4))
    assert _tstage_gap(pos) > _tstage_gap(neg), "realism-variant positive shift not injected"


def test_stream_f_is_images_only_and_has_no_normal_analog():
    stream = list(generate_fastmri_stream(SMOKE_N, seed=0, cube_size=6))
    _pid, cube, _label = stream[0]
    assert cube.shape == (FASTMRI_CHANNELS, 6, 6, 6), cube.shape
    assert set(FASTMRI_IMAGE_FEATURES).isdisjoint(FORBIDDEN_FEATURES)
    assert {y for *_, y in stream} <= {0, 1}, "Stream-F carries a non-{benign,malignant} label"


def test_stream_f_positive_injects_enhancing_blob_negative_lacks():
    def _blob_gap(stream, cube_size):
        q = max(cube_size // 4, 1)
        c1 = [float(cube[1, q:3 * q, q:3 * q, q:3 * q].mean()) for _, cube, y in stream if y == 1]
        c0 = [float(cube[1, q:3 * q, q:3 * q, q:3 * q].mean()) for _, cube, y in stream if y == 0]
        return abs(float(np.mean(c1) - np.mean(c0)))

    neg = list(generate_fastmri_stream(120, seed=2, cube_size=8, effect_size=0.0))
    pos = list(generate_fastmri_stream(120, seed=2, cube_size=8, effect_size=1.5))
    assert _blob_gap(pos, 8) > _blob_gap(neg, 8), "Stream-F positive enhancing blob not injected"


def test_stream_b_panel_is_canonical_and_forbidden_free():
    assert MODALITY_C_GENES == ("TP53", "PIK3CA", "GATA3", "CDH1", "MAP3K1", "PTEN"), "panel drifted"
    assert set(MODALITY_C_GENES).isdisjoint(FORBIDDEN_FEATURES), "a Stream-B gene is a forbidden mirror"
    for mirror in ("ESR1", "PGR", "ERBB2", "MKI67"):
        assert mirror in FORBIDDEN_FEATURES and mirror not in MODALITY_C_GENES


def test_stream_b_bag_and_gene_shapes():
    _pid, bag, genes, _label = next(iter(generate_wsi_genomics_stream(4, seed=0, n_tiles_range=(4, 8))))
    assert bag.ndim == 2 and bag.shape[1] == UNI_DIM, bag.shape  
    assert bag.dtype == np.float32
    assert genes.shape == (len(MODALITY_C_GENES),), genes.shape


def test_stream_b_positive_injects_gene_shift_negative_lacks():
    def _gene0_gap(stream):
        c1 = [g[0] for _, _, g, y in stream if y == 1]
        c0 = [g[0] for _, _, g, y in stream if y == 0]
        return abs(float(np.mean(c1) - np.mean(c0)))

    neg = list(generate_wsi_genomics_stream(200, seed=3, effect_size=0.0, n_tiles_range=(4, 8)))
    pos = list(generate_wsi_genomics_stream(200, seed=3, effect_size=1.5, n_tiles_range=(4, 8)))
    assert _gene0_gap(pos) > _gene0_gap(neg), "Stream-B positive gene-axis shift not injected"


def _lean_report(organ, features):
    config = {"organ": organ, "stream_name": "negative_control", "n": 4, "seed": 0,
              "effect_size": 0.0, "git_commit": "test"}
    manifest = build_stream_manifest(config)
    report = build_stream_report(organ, "negative_control", manifest, features, control_verdict("negative_control"))
    return report, manifest


@pytest.mark.parametrize(
    ("organ", "features"),
    [
        ("trackb-wsi-genomics", list(MODALITY_C_GENES) + ["wsi_bag_meanpool_1536d"]),
        ("fastmri-nyu-standalone", list(FASTMRI_IMAGE_FEATURES)),
    ],
)
def test_provenance_gate_raises_on_tampered_tag(organ, features):
    report, manifest = _lean_report(organ, features)
    assert_synthetic_provenance(report, manifest["manifest_sha256"])  
    stripped = {**report, "provenance": {**report["provenance"], "datasetTag": "REAL"}}
    with pytest.raises(SyntheticProvenanceError):
        assert_synthetic_provenance(stripped, manifest["manifest_sha256"])  
    with pytest.raises(SyntheticProvenanceError):
        assert_synthetic_provenance(report, "0" * 64)  


def test_lean_report_is_forbidden_free_and_non_reportable():
    report, _manifest = _lean_report("trackb-wsi-genomics", list(MODALITY_C_GENES) + ["wsi_bag_meanpool_1536d"])
    assert report["provenance"]["datasetTag"] == "SYNTHETIC — NOT A RESULT"
    assert set(report["features"]).isdisjoint(FORBIDDEN_FEATURES)


def test_control_verdict_leak_free_by_shuffle_field():
    import numpy as np

    from pinksight.eval.e2e_report_contract import control_verdict

    rng = np.random.default_rng(0)
    n = 400
    y = np.array([0, 1] * (n // 2))
    noise = rng.normal(size=n)

    neg = control_verdict("negative_control", y=y, real_oof=noise, shuffle_oof=noise)
    assert neg["leakFreeByShuffle"] is True
    assert neg["labelAttributableSignal"] == 0.0

    real_sig = y + rng.normal(0, 0.3, n)
    pos = control_verdict("positive_control", y=y, real_oof=real_sig, shuffle_oof=noise)
    assert pos["leakFreeByShuffle"] is False
    assert pos["labelAttributableSignal"] > 0.15

    none_v = control_verdict("negative_control", y=y, real_oof=noise, shuffle_oof=None)
    assert none_v["leakFreeByShuffle"] is None
    assert none_v["labelAttributableSignal"] is None


def test_stream_b_runner_end_to_end_small_n():
    import e2e_synthetic_trackb_run as trackb

    neg = trackb.run_control("negative_control", 80, 0, "test", n_tiles_range=(4, 8))
    pos = trackb.run_control("positive_control", 80, 0, "test", effect_size=1.5, n_tiles_range=(4, 8))
    lo, hi = neg["controlVerdict"]["delongCi95"]
    assert lo <= 0.50 <= hi, f"Stream-B negative CI {neg['controlVerdict']['delongCi95']} must cross 0.50"
    pv = pos["controlVerdict"]
    assert pv["auroc"] > 0.55, pv["auroc"]                                       
    assert pv["shuffleAuroc"] < pv["auroc"], (pv["shuffleAuroc"], pv["auroc"])   


def test_stream_f_runner_end_to_end_small_n():
    pytest.importorskip("torch")
    pytest.importorskip("monai")
    import e2e_synthetic_fastmri_run as fastmri

    enc = fastmri.build_encoder()
    neg = fastmri.run_control(enc, "negative_control", 60, 0, 8, 32, "test")
    pos = fastmri.run_control(enc, "positive_control", 60, 0, 8, 32, "test", effect_size=2.0)
    lo, hi = neg["controlVerdict"]["delongCi95"]
    assert lo <= 0.50 <= hi, f"Stream-F negative CI {neg['controlVerdict']['delongCi95']} must cross 0.50"
    pv = pos["controlVerdict"]
    assert pv["shuffleAuroc"] < pv["auroc"] - 0.15, (pv["shuffleAuroc"], pv["auroc"])


def test_stream_a_realistic_runner_end_to_end_small_n():
    pytest.importorskip("torch")
    pytest.importorskip("monai")
    import e2e_synthetic_harness_run as harness

    neg = harness.run_stream("negative_control", n=40, seed=8, cube_size=8, xai_subsample=0,
                             batch_size=32, realistic=True)
    assert neg["patient_ids"][0].startswith(A_NEG_PREFIX)
    assert neg["report"]["provenance"]["datasetTag"] == "SYNTHETIC — NOT A RESULT"
    lo, hi = neg["control_verdict"]["delongCi95"]
    assert lo <= 0.50 <= hi, f"Stream-A negative CI {neg['control_verdict']['delongCi95']} must cross 0.50"

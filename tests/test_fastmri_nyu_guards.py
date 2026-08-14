"""Runnable claim guards for the fastMRI-NYU standalone encoder (ADR-0016, RATIFIED).

The smallest tests that FAIL if a ratified claim guard breaks (ponytail). No heavy frameworks —
pure pandas/numpy over the real labels xlsx + the quarantine manifest. Skips gracefully if the xlsx
is absent (CI without the data).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pinksight import FORBIDDEN_FEATURES
from pinksight.data import fastmri_nyu as fx

_HAS_XLSX = fx.LABELS_XLSX.exists()
skip_no_data = pytest.mark.skipif(not _HAS_XLSX, reason="fastMRI labels xlsx not on disk")


# --- static guards (no data needed) --------------------------------------------------------------
def test_nyu_biomarkers_in_forbidden_features():
    """ADR-0016 Fix #6: every NYU biomarker column name is quarantined in FORBIDDEN_FEATURES."""
    assert fx.FORBIDDEN_NYU_BIOMARKERS <= FORBIDDEN_FEATURES


def test_images_only_guard_fires_on_wrong_channel_count():
    """A tensor whose channel count != declared image channels (leaked non-image feature) must RAISE."""
    fx.assert_images_only(np.zeros((4, 8, 8, 8)), expected_channels=4)          # clean per-item
    fx.assert_images_only(np.zeros((2, 4, 8, 8, 8)), expected_channels=4)       # clean batched
    with pytest.raises(AssertionError):
        fx.assert_images_only(np.zeros((5, 8, 8, 8)), expected_channels=4)      # extra channel leaked
    with pytest.raises(AssertionError):
        fx.assert_images_only(np.zeros((8, 8)), expected_channels=4)            # not an image tensor


def test_patient_disjoint_guard_fires_on_overlap():
    fx.assert_patient_disjoint(["a", "b"], ["c", "d"])                          # clean
    with pytest.raises(AssertionError):
        fx.assert_patient_disjoint(["a", "b"], ["b", "c"])                      # overlap


def test_normals_guard_fires_on_leak():
    q = {"fastMRI_breast_012", "fastMRI_breast_017"}
    fx.assert_no_normals(["fastMRI_breast_002"], quarantine=q)                  # clean
    with pytest.raises(AssertionError):
        fx.assert_no_normals(["fastMRI_breast_017"], quarantine=q)             # a normal leaked


# --- fast fixture guard (tiny synthetic cohort; no xlsx, no full-cohort collection) --------------
@pytest.fixture
def tiny_cohort() -> pd.DataFrame:
    """8-patient synthetic frame (id/repeat/split/age/lesion ONLY) — the smallest cohort that lets the
    LOADER-level interlock (normals-dropped + patient-disjoint) fire without the 300-row xlsx or any
    full-cohort scan. syn_03/syn_07 are verified-normal (lesion==negative) and MUST NOT survive into an
    H-char or H5 item list. Deterministic + millisecond-fast under CPU contention."""
    rows = [
        # (pid, repeat, split, age, lesion)
        ("syn_01", 0, fx.SPLIT_TRAIN, 40, fx.LESION_MALIG),
        ("syn_02", 0, fx.SPLIT_TRAIN, 55, fx.LESION_BENIGN),
        ("syn_03", 0, fx.SPLIT_TRAIN, 33, fx.LESION_NEG),     # normal -> dropped
        ("syn_04", 0, fx.SPLIT_TRAIN, 61, fx.LESION_MALIG),
        ("syn_05", 0, fx.SPLIT_TEST, 47, fx.LESION_BENIGN),
        ("syn_06", 0, fx.SPLIT_TEST, 52, fx.LESION_MALIG),
        ("syn_07", 0, fx.SPLIT_TEST, 29, fx.LESION_NEG),      # normal -> dropped
        ("syn_08", 0, fx.SPLIT_TEST, 70, fx.LESION_BENIGN),
    ]
    return pd.DataFrame(rows, columns=[fx.COL_PID, fx.COL_REPEAT, fx.COL_SPLIT, fx.COL_AGE, fx.COL_LESION])


def test_loader_interlock_on_tiny_fixture(tiny_cohort):
    """H-char AND H5 loaders drop the verified-normals and keep train/test patient-disjoint — proven on
    a tiny synthetic frame (the three claim guards, xlsx-free and deterministic)."""
    normals = {"syn_03", "syn_07"}
    for cohort in (fx.hchar_items(tiny_cohort), fx.age_items(tiny_cohort)):
        train_pids = {p for p, _ in cohort["train"]}
        test_pids = {p for p, _ in cohort["test"]}
        assert not ((train_pids | test_pids) & normals)   # (b) normals absent from every item list
        assert not (train_pids & test_pids)               # (c) patient-disjoint split
    hc = fx.hchar_items(tiny_cohort)
    assert len(hc["train"]) == 3 and len(hc["test"]) == 3  # 2 normals removed from the 8
    with pytest.raises(AssertionError):                    # a forced leak still trips the interlock
        fx.assert_no_normals(["syn_03"], quarantine=normals)


# --- data-backed guards --------------------------------------------------------------------------
@skip_no_data
def test_cohort_counts_match_ratified_plan():
    df = fx.load_labels()
    assert len(df) == 300
    assert len(fx.normal_ids(df)) == 51
    hc = fx.hchar_items(df)
    assert (len(hc["train"]), len(hc["test"])) == (199, 50)
    assert sum(y for _, y in hc["train"]) == 72   # train malignant
    assert sum(y for _, y in hc["test"]) == 18    # test malignant


@skip_no_data
def test_load_labels_drops_biomarker_columns():
    """load_labels keeps ONLY admissible columns — biomarkers cannot leak downstream (static)."""
    df = fx.load_labels()
    fx.assert_admissible_columns(df)
    assert not (set(df.columns) & fx.FORBIDDEN_NYU_BIOMARKERS)


@skip_no_data
def test_normals_absent_from_hchar_and_h5_items():
    """The ratified HARD interlock: no verified-normal appears in any H-char OR H5 item list."""
    df = fx.load_labels()
    q = set(fx.normal_ids(df))
    for cohort in (fx.hchar_items(df), fx.age_items(df)):
        pids = {p for p, _ in cohort["train"] + cohort["test"]}
        assert not (pids & q), "a quarantined normal reached an H-char/H5 item list"


@skip_no_data
def test_quarantine_manifest_roundtrip(tmp_path: Path):
    m = fx.write_quarantine_manifest(tmp_path / "q.json")
    assert m["n_quarantined"] == 51
    assert set(fx.load_quarantine(tmp_path / "q.json")) == set(m["patient_ids"])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

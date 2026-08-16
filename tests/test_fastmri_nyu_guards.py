
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pinksight import FORBIDDEN_FEATURES
from pinksight.data import fastmri_nyu as fx

_HAS_XLSX = fx.LABELS_XLSX.exists()
skip_no_data = pytest.mark.skipif(not _HAS_XLSX, reason="fastMRI labels xlsx not on disk")


def test_nyu_biomarkers_in_forbidden_features():
    assert fx.FORBIDDEN_NYU_BIOMARKERS <= FORBIDDEN_FEATURES


def test_images_only_guard_fires_on_wrong_channel_count():
    fx.assert_images_only(np.zeros((4, 8, 8, 8)), expected_channels=4)          
    fx.assert_images_only(np.zeros((2, 4, 8, 8, 8)), expected_channels=4)       
    with pytest.raises(AssertionError):
        fx.assert_images_only(np.zeros((5, 8, 8, 8)), expected_channels=4)      
    with pytest.raises(AssertionError):
        fx.assert_images_only(np.zeros((8, 8)), expected_channels=4)            


def test_patient_disjoint_guard_fires_on_overlap():
    fx.assert_patient_disjoint(["a", "b"], ["c", "d"])                          
    with pytest.raises(AssertionError):
        fx.assert_patient_disjoint(["a", "b"], ["b", "c"])                      


def test_normals_guard_fires_on_leak():
    q = {"fastMRI_breast_012", "fastMRI_breast_017"}
    fx.assert_no_normals(["fastMRI_breast_002"], quarantine=q)                  
    with pytest.raises(AssertionError):
        fx.assert_no_normals(["fastMRI_breast_017"], quarantine=q)             


@pytest.fixture
def tiny_cohort() -> pd.DataFrame:
    rows = [
        ("syn_01", 0, fx.SPLIT_TRAIN, 40, fx.LESION_MALIG),
        ("syn_02", 0, fx.SPLIT_TRAIN, 55, fx.LESION_BENIGN),
        ("syn_03", 0, fx.SPLIT_TRAIN, 33, fx.LESION_NEG),     
        ("syn_04", 0, fx.SPLIT_TRAIN, 61, fx.LESION_MALIG),
        ("syn_05", 0, fx.SPLIT_TEST, 47, fx.LESION_BENIGN),
        ("syn_06", 0, fx.SPLIT_TEST, 52, fx.LESION_MALIG),
        ("syn_07", 0, fx.SPLIT_TEST, 29, fx.LESION_NEG),      
        ("syn_08", 0, fx.SPLIT_TEST, 70, fx.LESION_BENIGN),
    ]
    return pd.DataFrame(rows, columns=[fx.COL_PID, fx.COL_REPEAT, fx.COL_SPLIT, fx.COL_AGE, fx.COL_LESION])


def test_loader_interlock_on_tiny_fixture(tiny_cohort):
    normals = {"syn_03", "syn_07"}
    for cohort in (fx.hchar_items(tiny_cohort), fx.age_items(tiny_cohort)):
        train_pids = {p for p, _ in cohort["train"]}
        test_pids = {p for p, _ in cohort["test"]}
        assert not ((train_pids | test_pids) & normals)   
        assert not (train_pids & test_pids)               
    hc = fx.hchar_items(tiny_cohort)
    assert len(hc["train"]) == 3 and len(hc["test"]) == 3  
    with pytest.raises(AssertionError):                    
        fx.assert_no_normals(["syn_03"], quarantine=normals)


@skip_no_data
def test_cohort_counts_match_ratified_plan():
    df = fx.load_labels()
    assert len(df) == 300
    assert len(fx.normal_ids(df)) == 51
    hc = fx.hchar_items(df)
    assert (len(hc["train"]), len(hc["test"])) == (199, 50)
    assert sum(y for _, y in hc["train"]) == 72   
    assert sum(y for _, y in hc["test"]) == 18    


@skip_no_data
def test_load_labels_drops_biomarker_columns():
    df = fx.load_labels()
    fx.assert_admissible_columns(df)
    assert not (set(df.columns) & fx.FORBIDDEN_NYU_BIOMARKERS)


@skip_no_data
def test_normals_absent_from_hchar_and_h5_items():
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

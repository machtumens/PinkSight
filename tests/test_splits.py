import pytest


def assert_patient_disjoint(train_ids, val_ids, test_ids):
    """Fail if any patient ID appears in more than one split (the LOCK-2 rule)."""
    train, val, test = set(train_ids), set(val_ids), set(test_ids)
    assert not (train & val), f"patient overlap train∩val: {sorted(train & val)}"
    assert not (train & test), f"patient overlap train∩test: {sorted(train & test)}"
    assert not (val & test), f"patient overlap val∩test: {sorted(val & test)}"


@pytest.mark.leakage
def test_patient_disjoint_logic_on_synthetic():
    """The disjointness assertion is exercised NOW on synthetic IDs.

    Honesty note: leakage-FEATURE exclusion is asserted in CI today (tests/test_leakage.py);
    the patient-DISJOINT split check below is real but runs on synthetic data until the
    frozen manifest exists (G1) — see test_patient_level_split_is_disjoint_real_manifest.
    """
    assert_patient_disjoint(["p1", "p2", "p3"], ["p4"], ["p5", "p6"])  # clean
    with pytest.raises(AssertionError):
        assert_patient_disjoint(["p1", "p2"], ["p2"], ["p3"])  # p2 leaks train→val


@pytest.mark.leakage
def test_patient_level_split_is_disjoint_real_manifest():
    """Real LOCK-2 check on the frozen scanner-holdout split (configs/split_v2.yaml).

    dev (the 5-fold CV pool) and test (quasi-external holdout) must be patient-disjoint.
    Skips gracefully on a fresh clone where the carve has not been run yet.
    """
    from pathlib import Path

    import yaml

    split = Path("configs/split_v2.yaml")
    if not split.exists():
        pytest.skip("configs/split_v2.yaml not carved yet — run scripts/carve_split.py")
    s = yaml.safe_load(split.read_text())
    dev, test = s["dev"], s["test"]
    assert dev and test, "split has an empty side"
    assert_patient_disjoint(dev, [], test)  # dev ∩ test must be empty


def assert_external_excludes_duke(patient_ids, datasets):
    """LOCK-2 dedup: fail if a row tagged non-DUKE still carries a DUKE_* patient ID.

    MAMA-MIA re-IDs the Duke-Breast-Cancer-MRI cases as DUKE_NNN (== Breast_MRI_NNN),
    so they overlap our anchor cohort and must be dropped before any "external"/H0 claim.
    Returns the clean external pool (datasets != DUKE).
    """
    external = [pid for pid, ds in zip(patient_ids, datasets) if ds != "DUKE"]
    leaked = [pid for pid in external if str(pid).upper().startswith("DUKE_")]
    assert not leaked, f"DUKE patient leaked into external pool: {leaked}"
    return external


@pytest.mark.leakage
def test_external_dedup_logic_on_synthetic():
    """The Duke-exclusion assertion runs in CI on synthetic IDs (data/ is gitignored)."""
    ids = ["DUKE_001", "ISPY1_005", "NACT_009", "DUKE_002"]
    dss = ["DUKE", "ISPY1", "NACT", "DUKE"]
    assert assert_external_excludes_duke(ids, dss) == ["ISPY1_005", "NACT_009"]  # clean
    with pytest.raises(AssertionError):
        assert_external_excludes_duke(["DUKE_001"], ["ISPY1"])  # mislabeled DUKE leaks


@pytest.mark.leakage
def test_mama_mia_duke_overlap_excluded_real():
    """LOCK-2: Duke ∩ MAMA-MIA = 291 (all DUKE_*); dropping dataset==DUKE leaves 1215 clean external.

    Numbers audited 2026-06-22 vs data/mamma_mia/clinical_and_imaging_info.xlsx;
    DUKE_NNN↔Breast_MRI_NNN maps 291/291. Skips on a fresh clone (data/ is gitignored).
    """
    from pathlib import Path

    f = Path("data/mamma_mia/clinical_and_imaging_info.xlsx")
    if not f.exists():
        pytest.skip("data/mamma_mia/ not present — local-only (gitignored)")
    import pandas as pd

    m = pd.read_excel(f)
    external = assert_external_excludes_duke(m["patient_id"], m["dataset"])
    duke = m.loc[m["dataset"] == "DUKE", "patient_id"].astype(str)
    assert len(duke) == 291, f"expected 291 Duke overlap, got {len(duke)} — re-audit DATA_CARD"
    assert len(external) == 1215, f"expected 1215 clean external, got {len(external)}"
    idx = duke.str.extract(r"DUKE_(\d+)")[0].astype(int)
    assert idx.max() <= 922, f"DUKE_* index {idx.max()} exceeds Duke cohort (922)"

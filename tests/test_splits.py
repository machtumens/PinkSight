import pytest


def assert_patient_disjoint(train_ids, val_ids, test_ids):
    train, val, test = set(train_ids), set(val_ids), set(test_ids)
    assert not (train & val), f"patient overlap train∩val: {sorted(train & val)}"
    assert not (train & test), f"patient overlap train∩test: {sorted(train & test)}"
    assert not (val & test), f"patient overlap val∩test: {sorted(val & test)}"


@pytest.mark.leakage
def test_patient_disjoint_logic_on_synthetic():
    assert_patient_disjoint(["p1", "p2", "p3"], ["p4"], ["p5", "p6"])  
    with pytest.raises(AssertionError):
        assert_patient_disjoint(["p1", "p2"], ["p2"], ["p3"])  


@pytest.mark.leakage
def test_patient_level_split_is_disjoint_real_manifest():
    from pathlib import Path

    import yaml

    split = Path("configs/split_v2.yaml")
    if not split.exists():
        pytest.skip("configs/split_v2.yaml not carved yet — run scripts/carve_split.py")
    s = yaml.safe_load(split.read_text())
    dev, test = s["dev"], s["test"]
    assert dev and test, "split has an empty side"
    assert_patient_disjoint(dev, [], test)  


def assert_external_excludes_duke(patient_ids, datasets):
    external = [pid for pid, ds in zip(patient_ids, datasets) if ds != "DUKE"]
    leaked = [pid for pid in external if str(pid).upper().startswith("DUKE_")]
    assert not leaked, f"DUKE patient leaked into external pool: {leaked}"
    return external


@pytest.mark.leakage
def test_external_dedup_logic_on_synthetic():
    ids = ["DUKE_001", "ISPY1_005", "NACT_009", "DUKE_002"]
    dss = ["DUKE", "ISPY1", "NACT", "DUKE"]
    assert assert_external_excludes_duke(ids, dss) == ["ISPY1_005", "NACT_009"]  
    with pytest.raises(AssertionError):
        assert_external_excludes_duke(["DUKE_001"], ["ISPY1"])  


@pytest.mark.leakage
def test_mama_mia_duke_overlap_excluded_real():
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


from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pinksight import FORBIDDEN_FEATURES

LABELS_XLSX = Path("data/fastmri_breast/fastMRI_breast_labels.xlsx")
TASK_FOLDER = Path("process/general-plans/active/fastmri-encoder-deferred-fusion_04-08-26")
NORMALS_QUARANTINE_PATH = TASK_FOLDER / "normals_quarantine_manifest.json"

COL_PID = "Patient Coded Name"
COL_REPEAT = "Repeated scans (0 = not repeat, 1:16 = repeated)"
COL_SPLIT = "Data split (0=training, 1=testing)"
COL_AGE = "Age"
COL_LESION = "Lesion status (0 = negative, 1= malignancy, 2= benign)"

LESION_NEG, LESION_MALIG, LESION_BENIGN = 0, 1, 2
SPLIT_TRAIN, SPLIT_TEST = 0, 1

FORBIDDEN_NYU_BIOMARKERS = frozenset({
    "Grade (1-3)", "ER+ (yes=1, no=0)", "PR+ (yes=1, no=0)", "HER2+ (yes=1, no=0)",
    "Biomarkers (Raw Text)", "Malignant lesion type",
})
ADMISSIBLE_COLUMNS = frozenset({COL_PID, COL_REPEAT, COL_SPLIT, COL_AGE, COL_LESION})


def load_labels(xlsx: str | Path = LABELS_XLSX) -> pd.DataFrame:
    df = pd.read_excel(xlsx)
    missing = ADMISSIBLE_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"labels xlsx missing admissible columns: {sorted(missing)}")
    df = df[list(ADMISSIBLE_COLUMNS)].copy()
    df[COL_PID] = df[COL_PID].astype(str)
    for c in (COL_REPEAT, COL_SPLIT, COL_AGE, COL_LESION):
        df[c] = df[c].astype(int)
    if df[COL_PID].duplicated().any():
        raise ValueError("duplicate patient IDs in labels xlsx — expected one row per patient")
    return df


def normal_ids(df: pd.DataFrame | None = None) -> list[str]:
    df = load_labels() if df is None else df
    return sorted(df.loc[df[COL_LESION] == LESION_NEG, COL_PID].tolist())


def hchar_items(df: pd.DataFrame | None = None) -> dict[str, list[tuple[str, int]]]:
    df = load_labels() if df is None else df
    hc = df[df[COL_LESION].isin([LESION_MALIG, LESION_BENIGN])].copy()
    hc["label"] = (hc[COL_LESION] == LESION_MALIG).astype(int)  
    out = {
        "train": [(r[COL_PID], int(r["label"])) for _, r in hc[hc[COL_SPLIT] == SPLIT_TRAIN].iterrows()],
        "test": [(r[COL_PID], int(r["label"])) for _, r in hc[hc[COL_SPLIT] == SPLIT_TEST].iterrows()],
    }
    assert_no_normals([p for p, _ in out["train"]] + [p for p, _ in out["test"]])
    assert_patient_disjoint([p for p, _ in out["train"]], [p for p, _ in out["test"]])
    return out


def age_items(df: pd.DataFrame | None = None) -> dict[str, list[tuple[str, float]]]:
    df = load_labels() if df is None else df
    hc = df[df[COL_LESION].isin([LESION_MALIG, LESION_BENIGN])].copy()
    out = {
        "train": [(r[COL_PID], float(r[COL_AGE])) for _, r in hc[hc[COL_SPLIT] == SPLIT_TRAIN].iterrows()],
        "test": [(r[COL_PID], float(r[COL_AGE])) for _, r in hc[hc[COL_SPLIT] == SPLIT_TEST].iterrows()],
    }
    assert_no_normals([p for p, _ in out["train"]] + [p for p, _ in out["test"]])
    assert_patient_disjoint([p for p, _ in out["train"]], [p for p, _ in out["test"]])
    return out


def write_quarantine_manifest(path: str | Path = NORMALS_QUARANTINE_PATH) -> dict:
    df = load_labels()
    ids = normal_ids(df)
    manifest = {
        "purpose": "ADR-0016 Fix #2 HARD interlock — the 51 verified-normal (H6/anomaly) patient IDs. "
                   "The H-char/H5 loader CI-asserts these are ABSENT from every batch. H6 is NOT "
                   "trained; unlocking this manifest requires H6's own early-detection /red-team.",
        "cohort": "fastMRI-NYU breast (NYU-only)",
        "lesion_status_quarantined": "negative (0) == verified-normal",
        "n_quarantined": len(ids),
        "patient_ids": ids,
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_quarantine(path: str | Path = NORMALS_QUARANTINE_PATH) -> set[str]:
    p = Path(path)
    if p.exists():
        return set(json.loads(p.read_text())["patient_ids"])
    return set(normal_ids())


def assert_no_normals(pids, quarantine: set[str] | None = None) -> None:
    q = load_quarantine() if quarantine is None else set(quarantine)
    leaked = sorted(set(map(str, pids)) & q)
    if leaked:
        raise AssertionError(
            f"NORMALS QUARANTINE VIOLATED (ADR-0016 Fix #2): {len(leaked)} verified-normal ID(s) "
            f"reached an H-char/H5 batch: {leaked[:5]}{'...' if len(leaked) > 5 else ''}"
        )


def assert_patient_disjoint(train_pids, test_pids) -> None:
    overlap = sorted(set(map(str, train_pids)) & set(map(str, test_pids)))
    if overlap:
        raise AssertionError(
            f"PATIENT-DISJOINT VIOLATED (LOCK-2): {len(overlap)} patient(s) in BOTH train and test: "
            f"{overlap[:5]}{'...' if len(overlap) > 5 else ''}"
        )


def assert_images_only(x, expected_channels: int) -> None:
    shape = tuple(getattr(x, "shape", ()))
    if len(shape) not in (4, 5):  
        raise AssertionError(f"images-only guard: expected a 4D/5D image tensor, got shape {shape}")
    c = shape[0] if len(shape) == 4 else shape[1]
    if int(c) != int(expected_channels):
        raise AssertionError(
            f"IMAGES-ONLY VIOLATED (ADR-0016 Fix #6): input has {c} channels but the images-only "
            f"encoder declares {expected_channels} DCE channels — a non-image feature may have leaked in."
        )


def assert_admissible_columns(df: pd.DataFrame) -> None:
    leaked = sorted(set(df.columns) & (FORBIDDEN_NYU_BIOMARKERS | set(FORBIDDEN_FEATURES)))
    if leaked:
        raise AssertionError(
            f"FORBIDDEN biomarker column(s) present in the pipeline frame (LOCK-2): {leaked}. "
            "load_labels() must keep only ADMISSIBLE_COLUMNS."
        )


def selfcheck() -> int:
    df = load_labels()
    assert len(df) == 300, f"expected 300 patients, got {len(df)}"
    assert_admissible_columns(df)
    assert FORBIDDEN_NYU_BIOMARKERS <= FORBIDDEN_FEATURES, "NYU biomarker cols must be in FORBIDDEN_FEATURES"

    normals = normal_ids(df)
    assert len(normals) == 51, f"expected 51 verified-normals, got {len(normals)}"

    hc = hchar_items(df)
    assert len(hc["train"]) == 199 and len(hc["test"]) == 50, (len(hc["train"]), len(hc["test"]))
    ntrain_malig = sum(lab for _, lab in hc["train"])
    ntest_malig = sum(lab for _, lab in hc["test"])
    assert ntrain_malig == 72 and ntest_malig == 18, (ntrain_malig, ntest_malig)

    ag = age_items(df)
    assert len(ag["train"]) == 199 and len(ag["test"]) == 50
    ages = [a for _, a in ag["train"] + ag["test"]]
    assert 25 <= min(ages) and max(ages) <= 82, (min(ages), max(ages))

    for bad in (
        lambda: assert_no_normals([normals[0]]),                       
        lambda: assert_patient_disjoint(["fastMRI_breast_002"], ["fastMRI_breast_002"]),
        lambda: assert_images_only(np.zeros((2, 8, 8, 8)), expected_channels=4),  
    ):
        try:
            bad()
        except AssertionError:
            pass
        else:
            raise AssertionError("a claim guard failed to fire on a violation")
    assert_no_normals([p for p, _ in hc["train"]])
    assert_images_only(np.zeros((4, 8, 8, 8)), expected_channels=4)
    assert_images_only(np.zeros((3, 4, 8, 8, 8)), expected_channels=4)

    print("fastmri_nyu selfcheck OK — 300 pts, 51 normals quarantined, H-char 199/50 (72/18 malig), "  
          "H5 249, biomarkers in FORBIDDEN_FEATURES, all claim guards fire.")
    return 0


if __name__ == "__main__":
    raise SystemExit(selfcheck())

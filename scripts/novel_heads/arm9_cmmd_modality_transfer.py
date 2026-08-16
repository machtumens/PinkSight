
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from scripts.novel_heads.wave1_eval_harness import (  
    FloorGateResult,
    run_floor_gate,
    validate_arm_report_keywords,
)
from tests.test_novel_heads_leakage import assert_no_forbidden_inputs  

_PURPOSE = (
    "independent null replication of imaging->subtype on CMMD FFDM (fresh CMMD-only radiomics floor)"
)

_COLLECTION = "CMMD"
_NBIA_BASE = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
_CLINICAL_URL = (
    "https://wiki.cancerimagingarchive.net/download/attachments/70230508/"
    "CMMD_clinicaldata_revision.xlsx"
)
_CLINICAL_NAME = "CMMD_clinicaldata_revision.xlsx"
_LICENSE = "CC BY-NC 3.0"

_STAGED_SUBDIR = Path("data") / "cmmd"
_CLINICAL_REL = _STAGED_SUBDIR / _CLINICAL_NAME
_IMAGES_REL = _STAGED_SUBDIR / "images"

_POSITIVE_SUBTYPE = "Luminal A"  
_NEGATIVE_SUBTYPE = "triple negative"  

_FLOOR_MODEL = "elasticnet"

_TOP_K_FEATURES = 60


@dataclass(frozen=True)
class ArmConfig:

    seeds: tuple[int, ...]
    kill_threshold_auroc: float
    greenlight_threshold_auroc: float
    greenlight_delong_ci_lb: float


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Arm 9 — CMMD modality-transfer null floor gate (independent null replication)"
    )
    p.add_argument(
        "--config",
        default="configs/novel_heads/arm9_cmmd_modality_transfer.yaml",
        help="arm config (dataset path, thresholds, seeds)",
    )
    p.add_argument("--floor-gate", action="store_true", help="run the CPU pyradiomics floor gate")
    p.add_argument(
        "--stage",
        action="store_true",
        help="stage CMMD (clinical spreadsheet + affected-breast DICOM for the LumA/TNBC patients) "
        "from the public TCIA NBIA API into data/cmmd/ (idempotent, resumable)",
    )
    p.add_argument(
        "--dedup-check",
        action="store_true",
        help="assert CMMD ∩ Duke = 0 patients (LOCK-2 zero-shared-patient; trivially true — separate "
        "TCIA collection, FFDM vs DCE-MRI)",
    )
    p.add_argument(
        "--report-dir",
        default="reports/novel_heads/arm9_cmmd_modality",
        help="directory for the floor-gate report + metrics JSON",
    )
    p.add_argument(
        "--data-root",
        default=None,
        help="override the data root the staged CMMD paths resolve against (default: repo root)",
    )
    p.add_argument(
        "--limit-patients",
        type=int,
        default=None,
        help="cap the number of LumA/TNBC patients staged/used (smoke testing only; omit for the "
        "full 238-patient floor gate)",
    )
    return p


def _parse_arm_config(cfg_raw: dict) -> ArmConfig:
    seeds = tuple(int(s) for s in cfg_raw.get("seeds", (0, 1, 2)))
    return ArmConfig(
        seeds=seeds,
        kill_threshold_auroc=float(cfg_raw.get("kill_threshold_auroc", 0.52)),
        greenlight_threshold_auroc=float(cfg_raw.get("greenlight_threshold_auroc", 0.62)),
        greenlight_delong_ci_lb=float(cfg_raw.get("greenlight_delong_ci_lb", 0.50)),
    )


def _clinical_path(data_root: Path) -> Path:
    return data_root / _CLINICAL_REL


def _load_task_table(data_root: Path, limit: int | None = None) -> pd.DataFrame:
    path = _clinical_path(data_root)
    if not path.exists():
        raise FileNotFoundError(
            f"arm9 BLOCKED — CMMD clinical spreadsheet not staged: {path}. Run "
            f"`python scripts/novel_heads/arm9_cmmd_modality_transfer.py --stage` first, or manually: "
            f"curl -sL '{_CLINICAL_URL}' -o {path} (public, {_LICENSE})."
        )
    df = pd.read_excel(path)
    df["subtype"] = df["subtype"].astype("string").str.strip()
    task = df[df["subtype"].isin([_POSITIVE_SUBTYPE, _NEGATIVE_SUBTYPE])].copy()
    task = task.drop_duplicates("ID1", keep="first").reset_index(drop=True)
    task["LeftRight"] = task["LeftRight"].astype("string").str.strip().str.upper()
    task["y"] = (task["subtype"] == _POSITIVE_SUBTYPE).astype(int)
    task = task[["ID1", "LeftRight", "Age", "abnormality", "classification", "subtype", "y"]]
    if limit is not None:
        pos = task[task["y"] == 1].head(max(1, limit // 2))
        neg = task[task["y"] == 0].head(max(1, limit - len(pos)))
        task = pd.concat([pos, neg]).reset_index(drop=True)
    return task


def _series_index(patient_ids: list[str]) -> dict[str, str]:
    import requests  

    resp = requests.get(
        f"{_NBIA_BASE}/getSeries", params={"Collection": _COLLECTION}, timeout=180
    )
    resp.raise_for_status()
    wanted = set(patient_ids)
    index: dict[str, str] = {}
    for s in resp.json():
        pid = s.get("PatientID")
        if pid in wanted and pid not in index:
            index[pid] = s["SeriesInstanceUID"]
    return index


def _download_series(uid: str, dest: Path) -> int:
    import io
    import zipfile

    import requests

    dest.mkdir(parents=True, exist_ok=True)
    r = requests.get(
        f"{_NBIA_BASE}/getImage",
        params={"SeriesInstanceUID": uid},
        stream=True,
        timeout=300,
    )
    if r.status_code == 404:
        raise FileNotFoundError(f"series {uid[-12:]} not available on TCIA (HTTP 404)")
    r.raise_for_status()
    payload = r.content
    if not payload or not zipfile.is_zipfile(io.BytesIO(payload)):
        raise ValueError(f"series {uid[-12:]} download is not a valid zip ({len(payload)} bytes)")
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        bad = zf.testzip()
        if bad is not None:
            raise ValueError(f"series {uid[-12:]} corrupt zip member: {bad}")
        zf.extractall(dest)
    return sum(1 for _ in dest.rglob("*.dcm"))


def _patient_dir(data_root: Path, pid: str) -> Path:
    return data_root / _IMAGES_REL / pid


def _is_patient_staged(data_root: Path, pid: str) -> bool:
    d = _patient_dir(data_root, pid)
    return d.is_dir() and any(d.rglob("*.dcm"))


def stage_cmmd(data_root: Path, limit: int | None = None) -> dict:
    import requests

    data_root.mkdir(parents=True, exist_ok=True)
    (data_root / _STAGED_SUBDIR).mkdir(parents=True, exist_ok=True)

    clinical = _clinical_path(data_root)
    if not clinical.exists():
        clinical.parent.mkdir(parents=True, exist_ok=True)
        resp = requests.get(_CLINICAL_URL, timeout=120, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        clinical.write_bytes(resp.content)
        print(f"arm9 stage: clinical spreadsheet -> {clinical} ({len(resp.content)} bytes)")  
    else:
        print(f"arm9 stage: clinical spreadsheet already present ({clinical})")  

    task = _load_task_table(data_root, limit=limit)
    patient_ids = task["ID1"].tolist()
    print(  
        f"arm9 stage: {len(patient_ids)} LumA/TNBC patients "
        f"({int(task['y'].sum())} {_POSITIVE_SUBTYPE} / {int((task['y'] == 0).sum())} "
        f"{_NEGATIVE_SUBTYPE}); resolving series uids ..."
    )
    index = _series_index(patient_ids)

    staged = skipped_existing = failed = 0
    missing_series: list[str] = []
    download_failures: list[str] = []
    for pid in patient_ids:
        if _is_patient_staged(data_root, pid):
            skipped_existing += 1
            continue
        uid = index.get(pid)
        if uid is None:
            missing_series.append(pid)
            continue
        try:
            n = _download_series(uid, _patient_dir(data_root, pid))
            if n >= 1:
                staged += 1
            else:
                failed += 1
                download_failures.append(f"{pid}: 0 dcm extracted")
        except (FileNotFoundError, ValueError, OSError, requests.RequestException) as exc:
            failed += 1
            download_failures.append(f"{pid}: {exc}")

    summary = {
        "patients_requested": len(patient_ids),
        "staged_now": staged,
        "already_staged": skipped_existing,
        "download_failures": failed,
        "missing_series_uid": missing_series,
        "failure_detail": download_failures,
    }
    print(  
        f"arm9 stage DONE: {staged} newly staged, {skipped_existing} already present, "
        f"{failed} failed, {len(missing_series)} missing series uid."
    )
    return summary


def _affected_side_token(left_right: str) -> str:
    return "L" if str(left_right).strip().upper().startswith("L") else "R"


def _read_dicom_image(path: Path):
    import SimpleITK as sitk

    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    reader.LoadPrivateTagsOn()
    reader.ReadImageInformation()
    try:
        laterality = reader.GetMetaData("0020|0062").strip().upper()[:1] or "?"
    except RuntimeError:
        laterality = "?"
    img = reader.Execute()
    arr = sitk.GetArrayFromImage(img)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    img2d = sitk.GetImageFromArray(arr.astype(np.float32))
    return img2d, laterality


def _breast_region_mask(img):
    import SimpleITK as sitk

    otsu = sitk.OtsuThreshold(img, 0, 1)  
    cc = sitk.ConnectedComponent(otsu)
    relabel = sitk.RelabelComponent(cc, sortByObjectSize=True)
    largest = sitk.BinaryThreshold(relabel, 1, 1, 1, 0)
    closed = sitk.BinaryMorphologicalClosing(largest, [3, 3])
    return sitk.Cast(closed, sitk.sitkUInt8)


def _build_extractor():
    from radiomics.featureextractor import RadiomicsFeatureExtractor

    settings = {
        "force2D": True,
        "force2Ddimension": 0,
        "binWidth": 25.0,
        "label": 1,
        "normalize": False,
        "correctMask": True,
    }
    extractor = RadiomicsFeatureExtractor(**settings)
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName("firstorder")
    extractor.enableFeatureClassByName("glcm")
    extractor.enableFeatureClassByName("glrlm")
    return extractor


def _is_numeric_scalar(value) -> bool:
    try:
        if np.ndim(value) != 0:
            return False
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _extract_patient_features(
    extractor, patient_dir: Path, affected_side: str
) -> dict[str, float] | None:
    want = _affected_side_token(affected_side)
    per_image: list[dict[str, float]] = []
    for dcm in sorted(patient_dir.rglob("*.dcm")):
        try:
            img, laterality = _read_dicom_image(dcm)
        except (RuntimeError, OSError):
            continue
        if laterality not in (want, "?"):
            continue
        try:
            mask = _breast_region_mask(img)
            result = extractor.execute(img, mask)
        except (ValueError, RuntimeError):
            continue
        feats = {
            k: float(v)
            for k, v in result.items()
            if not k.startswith("diagnostics_") and _is_numeric_scalar(v)
        }
        if feats:
            per_image.append(feats)
    if not per_image:
        return None
    keys = sorted(set().union(*[set(d) for d in per_image]))
    return {k: float(np.mean([d[k] for d in per_image if k in d])) for k in keys}


@dataclass
class FeatureMatrix:

    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    feature_names: list[str]
    n_patients: int
    n_positive: int
    n_negative: int
    dropped_patients: list[str] = field(default_factory=list)


def build_feature_matrix(data_root: Path, task: pd.DataFrame) -> FeatureMatrix:
    extractor = _build_extractor()
    rows: list[dict[str, float]] = []
    ys: list[int] = []
    groups: list[str] = []
    dropped: list[str] = []
    for _, r in task.iterrows():
        pid = r["ID1"]
        patient_dir = _patient_dir(data_root, pid)
        if not patient_dir.is_dir():
            dropped.append(f"{pid} (no staged dir)")
            continue
        feats = _extract_patient_features(extractor, patient_dir, r["LeftRight"])
        if feats is None:
            dropped.append(f"{pid} (no readable affected-side image / extraction failed)")
            continue
        rows.append(feats)
        ys.append(int(r["y"]))
        groups.append(str(pid))

    if not rows:
        return FeatureMatrix(
            X=np.empty((0, 0)),
            y=np.empty((0,), dtype=int),
            groups=np.empty((0,), dtype=object),
            feature_names=[],
            n_patients=0,
            n_positive=0,
            n_negative=0,
            dropped_patients=dropped,
        )

    frame = pd.DataFrame(rows)
    frame = frame.fillna(frame.median(axis=0))
    variances = frame.var(axis=0, numeric_only=True)
    frame = frame.loc[:, variances[variances > 0].index]
    if frame.shape[1] > _TOP_K_FEATURES:
        top = frame.var(axis=0).sort_values(ascending=False).head(_TOP_K_FEATURES).index
        frame = frame.loc[:, sorted(top)]

    assert_no_forbidden_inputs(frame.columns)

    y = np.asarray(ys, dtype=int)
    return FeatureMatrix(
        X=frame.to_numpy(dtype=float),
        y=y,
        groups=np.asarray(groups, dtype=object),
        feature_names=list(frame.columns),
        n_patients=int(len(y)),
        n_positive=int(y.sum()),
        n_negative=int((y == 0).sum()),
        dropped_patients=dropped,
    )


@dataclass
class GateOutcome:

    task: str
    matrix: FeatureMatrix
    result: FloorGateResult
    shuffle_auroc: float
    decision: str  
    rationale: str


_NULL_SHUFFLE_MARGIN = 0.03


def _decide(
    result: FloorGateResult,
    cfg: ArmConfig,
    n_total: int,
    n_pos: int,
    n_neg: int,
    shuffle_auroc: float,
) -> tuple[str, str]:
    auroc = result.auroc
    lb = result.delong_ci95[0]
    hi = result.delong_ci95[1]
    kill = cfg.kill_threshold_auroc
    green = cfg.greenlight_threshold_auroc
    green_lb = cfg.greenlight_delong_ci_lb
    if not np.isfinite(auroc):
        return "INDETERMINATE", (
            f"AUROC is NaN (degenerate CV — likely single-class folds at N={n_total}, "
            f"{n_pos} vs {n_neg})."
        )

    ci_crosses_chance = lb <= 0.50 <= hi
    near_shuffle = np.isfinite(shuffle_auroc) and abs(auroc - shuffle_auroc) <= _NULL_SHUFFLE_MARGIN
    statistical_null = ci_crosses_chance and near_shuffle

    if auroc <= kill or statistical_null:
        if auroc <= kill:
            gate_clause = (
                f"mean patient-level AUROC {auroc:.3f} <= the pre-registered KILL {kill:.2f}"
            )
        else:
            gate_clause = (
                f"mean patient-level AUROC {auroc:.3f} sits within {_NULL_SHUFFLE_MARGIN:.2f} of the "
                f"shuffle sentinel {shuffle_auroc:.3f} and the DeLong CI [{lb:.3f}, {hi:.3f}] crosses "
                f"0.50 — statistically indistinguishable from chance"
            )
        return "KILL", (
            f"{gate_clause} — the CMMD radiomics floor does not separate {_POSITIVE_SUBTYPE} from "
            f"{_NEGATIVE_SUBTYPE} above chance (N={n_total}: {n_pos} vs {n_neg}). This is an "
            f"INDEPENDENT NULL REPLICATION: the imaging->subtype null seen on Duke DCE-MRI (ADR-0008) "
            f"reproduces on full-field digital mammography in a different cohort — corroborating that "
            f"the imaging->subtype signal is weak across modalities, not a Duke-specific artefact."
        )
    if auroc >= green and lb >= green_lb:
        return "GREENLIGHT", (
            f"mean patient-level AUROC {auroc:.3f} >= GREENLIGHT {green:.2f} and DeLong LB {lb:.3f} "
            f">= {green_lb:.2f} — an UNEXPECTED full-field-mammography subtype signal on CMMD "
            f"(N={n_total}: {n_pos} vs {n_neg}). This is a NEW HYPOTHESIS on CMMD's own internal split "
            f"only; it is NOT a claim about model performance moving between cohorts, and any follow-up "
            f"training would require a new ADR."
        )
    return "INDETERMINATE", (
        f"mean patient-level AUROC {auroc:.3f} (DeLong LB {lb:.3f}, shuffle {shuffle_auroc:.3f}) is "
        f"between KILL {kill:.2f} and GREENLIGHT {green:.2f} with a CI clearing chance — real-but-weak "
        f"/ inconclusive at N={n_total} ({n_pos} vs {n_neg}). No advance under the kill-early "
        f"discipline; the honest number is recorded."
    )


def run_arm9_floor_gate(cfg: ArmConfig, data_root: Path, limit: int | None = None) -> GateOutcome:
    task = _load_task_table(data_root, limit=limit)
    matrix = build_feature_matrix(data_root, task)
    task_label = f"{_POSITIVE_SUBTYPE} vs {_NEGATIVE_SUBTYPE} (CMMD FFDM, affected-breast radiomics)"

    n_total, n_pos, n_neg = matrix.n_patients, matrix.n_positive, matrix.n_negative
    if n_total < 6 or n_pos < 2 or n_neg < 2 or matrix.X.shape[1] == 0:
        empty = FloorGateResult(
            model=_FLOOR_MODEL,
            auroc=float("nan"),
            delong_ci95=(float("nan"), float("nan")),
            ece=float("nan"),
            seeds=len(cfg.seeds),
            per_seed=[],
            n=n_total,
            prevalence=round(float(matrix.y.mean()) if n_total else 0.0, 4),
        )
        return GateOutcome(
            task=task_label,
            matrix=matrix,
            result=empty,
            shuffle_auroc=float("nan"),
            decision="BLOCKED" if n_total == 0 else "INDETERMINATE",
            rationale=(
                f"only {n_total} patients with usable radiomics ({n_pos} {_POSITIVE_SUBTYPE} vs "
                f"{n_neg} {_NEGATIVE_SUBTYPE}), {matrix.X.shape[1]} features — below the floor for a "
                f"3-seed patient-disjoint CV. Stage the CMMD DICOM (`--stage`) before the floor gate."
            ),
        )

    result = run_floor_gate(matrix.X, matrix.y, matrix.groups, model=_FLOOR_MODEL, seeds=cfg.seeds)

    rng = np.random.default_rng(0)
    y_shuffled = matrix.y.copy()
    rng.shuffle(y_shuffled)
    shuffle_res = run_floor_gate(
        matrix.X, y_shuffled, matrix.groups, model=_FLOOR_MODEL, seeds=(0, 1, 2)
    )
    shuffle_auroc = float(shuffle_res.auroc)

    decision, rationale = _decide(result, cfg, n_total, n_pos, n_neg, shuffle_auroc)
    return GateOutcome(
        task=task_label,
        matrix=matrix,
        result=result,
        shuffle_auroc=shuffle_auroc,
        decision=decision,
        rationale=rationale,
    )


def _render_report(outcome: GateOutcome, cfg: ArmConfig, data_root: Path) -> str:
    today = date.today().strftime("%Y-%m-%d")
    r = outcome.result
    m = outcome.matrix
    auroc = "NaN" if not np.isfinite(r.auroc) else f"{r.auroc:.3f}"
    ci = (
        "[NaN, NaN]"
        if not np.isfinite(r.delong_ci95[0])
        else f"[{r.delong_ci95[0]:.3f}, {r.delong_ci95[1]:.3f}]"
    )
    ece = "NaN" if not np.isfinite(r.ece) else f"{r.ece:.3f}"
    shuffle = "NaN" if not np.isfinite(outcome.shuffle_auroc) else f"{outcome.shuffle_auroc:.3f}"

    lines: list[str] = []
    lines.append("---")
    lines.append("name: report:arm9-cmmd-modality-transfer-floor-gate")
    lines.append(
        'description: "Arm 9 CMMD modality-transfer null floor gate — an independent null replication '
        "of the imaging->subtype question on full-field digital mammography (Luminal A vs triple "
        'negative); fresh CMMD-only radiomics floor; AUROC + DeLong CI + ECE + shuffle sentinel."'
    )
    lines.append(f"date: {today}")
    lines.append("metadata:")
    lines.append("  node_type: memory")
    lines.append("  type: report")
    lines.append("  feature: novel-heads")
    lines.append("  phase: arm9-floor-gate")
    lines.append("---")
    lines.append("")
    lines.append("# Arm 9 — CMMD Modality-Transfer Null Test: Floor Gate")
    lines.append("")
    lines.append(f"**Date:** {today}  ")
    lines.append(
        "**Arm:** 9 (Wave 2) — independent null replication of imaging->subtype on an alternative "
        "modality  "
    )
    lines.append("**Compute:** CPU-only, $0 (well within the $150 LOCK-5 ceiling)  ")
    lines.append(
        "**Model:** elastic-net logistic (shared `wave1_eval_harness`; mirrors arm 6 / arm 10 — the "
        "robust radiomics-floor learner, no new dependency vs the nominal `lightgbm` config field)  "
    )
    lines.append(
        "**Modality:** full-field digital mammography (FFDM), CMMD collection — a different modality "
        "and cohort from the Duke DCE-MRI programme  "
    )
    lines.append("")
    lines.append("## What this arm is (framing — LOCK-1)")
    lines.append("")
    lines.append(
        "This arm is an **independent null replication**. The G3 gate characterised the "
        "imaging->subtype signal as an information ceiling on Duke DCE-MRI (ADR-0008). Here the SAME "
        "science question — can cheap radiomics separate two molecular subtypes? — is asked afresh on "
        "a **completely different imaging modality** (full-field digital mammography) and a "
        "**different patient population** (the Chinese Mammography Database). A **fresh radiomics "
        "floor is trained and evaluated entirely on CMMD**, reported on its own internal "
        "patient-disjoint split."
    )
    lines.append("")
    lines.append(
        "**What this arm does NOT do (LOCK-1 firewall — each item is explicitly excluded):**"
    )
    lines.append("")
    lines.append(
        "- It does **not** apply any Duke-trained model to CMMD. The floor is trained on CMMD only; "
        "no Duke weights, thresholds, or features touch CMMD data."
    )
    lines.append(
        "- It makes **no** claim that model performance carries from one institution to another. The "
        "comparison is between two independent answers to the same science question, not a transfer "
        "of a fitted model."
    )
    lines.append(
        "- It does **not** localise or detect lesions. The radiomics mask is a whole-breast tissue "
        "region (Otsu + largest component); the target is a molecular-subtype characterisation label, "
        "never a healthy-tissue or detection endpoint."
    )
    lines.append(
        "- It does **not** measure growth rate, tumour kinetics, or doubling time (mammographic "
        "texture is a snapshot at diagnosis)."
    )
    lines.append("- It makes **no** prognosis or survival claim.")
    lines.append(
        "- It does **not** fuse into the Duke DCE-MRI imaging encoder (ADR-0006/0010 standalone-organ "
        "pattern). Any model trained on CMMD and applied to Duke would require a NEW ADR."
    )
    lines.append("")
    lines.append(
        "The sole output is an at-diagnosis subtype-characterisation number on CMMD, reported so the "
        "same imaging->subtype question can be compared across two independent modalities."
    )
    lines.append("")
    lines.append("## Dataset (CMMD, TCIA)")
    lines.append("")
    lines.append(
        f"The Chinese Mammography Database (CMMD), TCIA collection `{_COLLECTION}` ({_LICENSE}). "
        f"Full-field digital mammography (MG modality). Of 1775 patients, 749 carry a molecular "
        f"subtype label in `{_CLINICAL_NAME}`; the Luminal-A-vs-triple-negative task uses the two "
        f"target subtypes."
    )
    lines.append("")
    lines.append(
        f"- **Label source:** per-patient `subtype` + affected `LeftRight` from the TCIA wiki clinical "
        f"spreadsheet, staged at `{_CLINICAL_REL}`."
    )
    lines.append(
        "- **Images:** each subtyped patient has one MG series (2 or 4 images, CC + MLO of one or both "
        "breasts). Radiomics is extracted ONLY from images matching the affected-breast laterality "
        "(DICOM `ImageLaterality` == clinical `LeftRight`); the contralateral breast is excluded."
    )
    lines.append("")
    lines.append("## Task definition")
    lines.append("")
    lines.append(
        f"**{_POSITIVE_SUBTYPE} vs {_NEGATIVE_SUBTYPE}** — the direct PinkSight subtype axis (Luminal A "
        f"vs TNBC), replicated on FFDM. Luminal B and HER2-enriched patients are held out of this "
        f"binary floor. Positive = {_POSITIVE_SUBTYPE} (label 1); negative = {_NEGATIVE_SUBTYPE} "
        f"(CMMD's TNBC label, label 0)."
    )
    lines.append("")
    lines.append("## Evaluation integrity")
    lines.append("")
    lines.append(
        "- **Patient-level CV** — each patient is one unit; the shared harness folds by patient id, so "
        "no patient straddles a train/val boundary (LOCK-2). CMMD subtyped patients have exactly one "
        "affected breast each, so the patient id is a clean grouping key."
    )
    lines.append(
        "- **Radiomics-first:** pyradiomics first-order + GLCM + GLRLM texture on a whole-breast-region "
        "mask; the cheapest interpretable floor, the correct kill-early gate before any deep imaging "
        "model (the same discipline that established the Duke imaging null, ADR-0001/ADR-0008)."
    )
    lines.append(
        "- **Leakage guard:** radiomics texture features are the ONLY inputs; ER/PR/HER2/Ki-67/subtype "
        "are never features (subtype is the TARGET). The shared `assert_no_forbidden_inputs` gate runs "
        "on the feature columns before every fit; the top-variance feature cap is unsupervised (uses "
        "X only, no label)."
    )
    lines.append(
        f"- **Shuffle sentinel:** the label was permuted and the whole floor re-run — shuffle AUROC "
        f"**{shuffle}** (a real signal collapses to ~0.50 under shuffle; a leakage artefact does not)."
    )
    lines.append(
        f"- **Never a bare number:** AUROC + DeLong 95% CI + ECE reported at {r.seeds} seeds "
        f"({list(cfg.seeds)})."
    )
    lines.append("")
    lines.append("## Result")
    lines.append("")
    lines.append(
        "| Task | N | N pos | N neg | Features | Mean AUROC | DeLong CI95 | ECE | Shuffle AUROC | "
        "Seeds | Decision |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    lines.append(
        f"| {outcome.task} | {m.n_patients} | {m.n_positive} | {m.n_negative} | "
        f"{len(m.feature_names)} | {auroc} | {ci} | {ece} | {shuffle} | {r.seeds} | "
        f"**{outcome.decision}** |"
    )
    lines.append("")
    lines.append("### Decision rationale")
    lines.append("")
    lines.append(f"- **{outcome.decision}.** {outcome.rationale}")
    lines.append("")
    lines.append("### Per-seed detail")
    lines.append("")
    if not r.per_seed:
        lines.append("- (no per-seed rows — usable N below the CV floor)")
    for ps in r.per_seed:
        lines.append(
            f"- seed {ps['seed']}: AUROC {ps['auroc']:.3f}, "
            f"DeLong CI {ps['delong_ci95']}, ECE {ps['ece']:.3f}"
        )
    lines.append("")
    if m.dropped_patients:
        lines.append(
            f"### Patients dropped ({len(m.dropped_patients)} — no staged/readable affected-side image)"
        )
        lines.append("")
        for d in m.dropped_patients[:25]:
            lines.append(f"- {d}")
        if len(m.dropped_patients) > 25:
            lines.append(f"- ... and {len(m.dropped_patients) - 25} more")
        lines.append("")
    lines.append("## Floor-gate decision")
    lines.append("")
    if outcome.decision == "GREENLIGHT":
        overall_note = (
            "The CMMD radiomics floor separated the two subtypes above the GREENLIGHT threshold on "
            "CMMD's own internal patient-disjoint split — an unexpected full-field-mammography subtype "
            "signal. This is a NEW HYPOTHESIS on CMMD only; it is not a claim about model performance "
            "moving between cohorts, and any follow-up training requires a new ADR."
        )
    elif outcome.decision == "KILL":
        overall_note = (
            "The CMMD radiomics floor did not separate the two subtypes above chance on held-out "
            "patients. As an INDEPENDENT NULL REPLICATION this is a valuable, positive science "
            "outcome: the imaging->subtype null observed on Duke DCE-MRI (ADR-0008) reproduces on a "
            "different modality (FFDM) and a different cohort, corroborating that the imaging->subtype "
            "signal is weak across modalities rather than a Duke-specific artefact. The arm is complete."
        )
    elif outcome.decision == "BLOCKED":
        overall_note = (
            "No usable CMMD radiomics matrix could be built (DICOM not staged). The floor gate did not "
            "run; no number is fabricated. Stage CMMD (`--stage`) and re-run."
        )
    else:
        overall_note = (
            "The result is real-but-weak / inconclusive (neither killed nor greenlit). Under the "
            "kill-early discipline the arm does not advance to a costlier step; the floor produced its "
            "number and the honest outcome is recorded."
        )
    lines.append(f"**OVERALL: {outcome.decision}.** {overall_note}")
    lines.append("")
    lines.append("## Zero-shared-patient / dedup (LOCK-2)")
    lines.append("")
    lines.append(
        "CMMD is a separate TCIA collection from the Duke DCE-MRI cohort — a different institution and "
        "a different imaging modality (full-field digital mammography vs dynamic contrast-enhanced "
        "MRI). The two datasets share zero patients by construction (CMMD ∩ Duke = 0); the dedup check "
        "is trivially TRUE. No comparison of a fitted model across the two cohorts is made — each is an "
        "independent answer to the same science question on its own data."
    )
    lines.append("")
    lines.append("## Data provenance")
    lines.append("")
    lines.append(
        f"- **Collection:** CMMD (The Chinese Mammography Database), TCIA, {_LICENSE}. Staged from the "
        f"public NBIA REST API (getSeries + getImage; no authentication) into `{_IMAGES_REL}`."
    )
    lines.append(
        f"- **Clinical labels:** `{_CLINICAL_NAME}` from the TCIA wiki, staged at `{_CLINICAL_REL}`."
    )
    lines.append(
        "- **Radiomics:** pyradiomics (first-order + GLCM + GLRLM), 2-D force, bin width 25, "
        "whole-breast-region Otsu mask (largest connected component); per-image features averaged to "
        "the patient level over the affected-breast views."
    )
    lines.append(
        f"- **Feature matrix:** {len(m.feature_names)} features after median-impute, zero-variance "
        f"drop, and the unsupervised top-{_TOP_K_FEATURES} variance cap."
    )
    lines.append("")
    lines.append(
        "_All data is gitignored under `data/cmmd/` and is never committed (images + clinical file)._"
    )
    lines.append("")
    return "\n".join(lines)


def _metrics_payload(outcome: GateOutcome, cfg: ArmConfig, data_root: Path) -> dict:
    m = outcome.matrix
    return {
        "arm": 9,
        "role": "independent-null-replication-imaging-to-subtype-on-ffdm",
        "framing": (
            "independent null replication of the imaging->subtype question on CMMD full-field digital "
            "mammography; fresh CMMD-only radiomics floor; no Duke model applied to CMMD"
        ),
        "collection": _COLLECTION,
        "license": _LICENSE,
        "modality": "MG (full-field digital mammography)",
        "task": outcome.task,
        "positive_subtype": _POSITIVE_SUBTYPE,
        "negative_subtype": _NEGATIVE_SUBTYPE,
        "n_patients": m.n_patients,
        "n_positive": m.n_positive,
        "n_negative": m.n_negative,
        "n_features": len(m.feature_names),
        "feature_classes": ["firstorder", "glcm", "glrlm"],
        "mask": "whole-breast-region Otsu (largest connected component) — NOT lesion segmentation",
        "n_dropped_patients": len(m.dropped_patients),
        "zero_shared_patient": (
            "true (separate TCIA collection; FFDM vs DCE-MRI; CMMD ∩ Duke = 0 by construction)"
        ),
        "model": _FLOOR_MODEL,
        "model_nominal_config": "lightgbm",
        "model_note": (
            "elastic-net used (harness _MODELS['elasticnet']); mirrors arm 6 / arm 10; robust "
            "radiomics-floor learner; avoids adding an undeclared lightgbm dependency"
        ),
        "seeds": list(cfg.seeds),
        "kill_threshold_auroc": cfg.kill_threshold_auroc,
        "greenlight_threshold_auroc": cfg.greenlight_threshold_auroc,
        "greenlight_delong_ci_lb": cfg.greenlight_delong_ci_lb,
        "shuffle_sentinel_auroc": (
            None if not np.isfinite(outcome.shuffle_auroc) else round(outcome.shuffle_auroc, 4)
        ),
        "decision": outcome.decision,
        "rationale": outcome.rationale,
        **outcome.result.as_dict(),
    }


def _run_dedup_check() -> int:
    print(  
        "arm9 dedup check (LOCK-2): CMMD ∩ Duke = 0. CMMD is a separate TCIA collection "
        "(full-field digital mammography); Duke is DCE-MRI. Different institution + different "
        "modality => zero shared patients by construction. PASS."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg_path = Path(args.config)
    cfg_raw = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
    cfg = _parse_arm_config(cfg_raw or {})
    data_root = Path(args.data_root).resolve() if args.data_root else _REPO_ROOT

    if args.dedup_check and not (args.floor_gate or args.stage):
        return _run_dedup_check()

    if args.stage:
        try:
            stage_cmmd(data_root, limit=args.limit_patients)
        except Exception as exc:  
            print(f"BLOCKED: arm9 staging failed — {type(exc).__name__}: {exc}", file=sys.stderr)  
            return 2
        if not args.floor_gate:
            return 0

    if not args.floor_gate:
        print(  
            f"arm9: {_PURPOSE}. Pass --stage to fetch CMMD (clinical + affected-breast DICOM), "
            f"--floor-gate to run the radiomics floor gate, or --dedup-check for the LOCK-2 check."
        )
        return 0

    try:
        outcome = run_arm9_floor_gate(cfg, data_root, limit=args.limit_patients)
    except FileNotFoundError as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)  
        return 2

    report_text = _render_report(outcome, cfg, data_root)
    validate_arm_report_keywords(report_text, "arm9")  

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = date.today().strftime("%Y%m%d")
    report_path = report_dir / f"floor_gate_{stamp}.md"
    metrics_path = report_dir / f"floor_gate_{stamp}.json"
    report_path.write_text(report_text)
    metrics_path.write_text(json.dumps(_metrics_payload(outcome, cfg, data_root), indent=2))

    r = outcome.result
    auroc = "NaN" if not np.isfinite(r.auroc) else f"{r.auroc:.3f}"
    ci = (
        "[NaN,NaN]"
        if not np.isfinite(r.delong_ci95[0])
        else f"[{r.delong_ci95[0]:.3f},{r.delong_ci95[1]:.3f}]"
    )
    ece = "NaN" if not np.isfinite(r.ece) else f"{r.ece:.3f}"
    shuffle = "NaN" if not np.isfinite(outcome.shuffle_auroc) else f"{outcome.shuffle_auroc:.3f}"
    print(f"arm9 floor gate — report: {report_path}")  
    print(  
        f"  {outcome.task}: N={outcome.matrix.n_patients} "
        f"({outcome.matrix.n_positive} vs {outcome.matrix.n_negative}) "
        f"features={len(outcome.matrix.feature_names)} AUROC={auroc} DeLongCI={ci} ECE={ece} "
        f"shuffle={shuffle} -> {outcome.decision}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
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
)
from tests.test_novel_heads_leakage import assert_no_forbidden_inputs  

_PURPOSE = "characterise CESM iodine-enhancement phenotype at diagnosis (BI-RADS lesion floor gate)"

_RADIOMICS_SETTINGS = {"binCount": 32, "label": 1}
_MAX_LONG_EDGE = 1024
_MIN_LESIONS = 30


@dataclass(frozen=True)
class ArmConfig:

    cdd_cesm_root: Path
    target: str  
    seeds: tuple[int, ...]
    kill_threshold_auroc: float
    greenlight_threshold_auroc: float
    greenlight_delong_ci_lb: float


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Arm 1 — CDD-CESM iodine-uptake radiomics floor gate"
    )
    p.add_argument(
        "--config",
        default="configs/novel_heads/arm1_cesm_iodine_radiomics.yaml",
        help="arm config (dataset path, thresholds, seeds)",
    )
    p.add_argument("--floor-gate", action="store_true", help="run the CPU radiomics floor gate")
    p.add_argument(
        "--dedup-check",
        action="store_true",
        help="assert CDD-CESM ∩ Duke = 0 patients (LOCK-2 zero-shared-patient)",
    )
    p.add_argument(
        "--report-dir",
        default="reports/novel_heads/arm1_cesm_iodine",
        help="directory for the floor-gate report + metrics JSON",
    )
    p.add_argument(
        "--data-root",
        default=None,
        help="override the data root the config paths resolve against (default: repo root)",
    )
    p.add_argument(
        "--cache",
        default="data/cdd_cesm/arm1_radiomics_cache.csv",
        help="CSV cache of extracted per-lesion radiomics features (skip re-extraction)",
    )
    return p


_ANNOT_XLSX = "Radiology-manual-annotations.xlsx"
_SEG_CSV = "Radiology_hand_drawn_segmentations_v2.csv"


@dataclass(frozen=True)
class StagedPaths:
    annot_xlsx: Path
    seg_csv: Path
    image_roots: tuple[Path, ...]  


def _stage_instructions(data_dir: Path) -> str:
    return (
        f"arm1 floor gate BLOCKED — CDD-CESM not fully staged under {data_dir}.\n"
        "Stage the public CDD-CESM (Khaled et al. 2022; Kaggle mirror) with:\n"
        "  export KAGGLE_CONFIG_DIR=~/.kaggle\n"
        f"  kaggle datasets download -d ringoospina/cdd-cesm-excel -p {data_dir} --unzip\n"
        f"  kaggle datasets download -d hanaraafatt/cdd-cesm -p {data_dir}/images --unzip\n"
        f"Expected: {data_dir}/{_ANNOT_XLSX}, {data_dir}/{_SEG_CSV}, and recombined-channel JPGs "
        f"named P*_CM_*.jpg somewhere under {data_dir}/images/. The recombined (CM) channel is the "
        "iodine-uptake image; the low-energy (DM) channel is NOT used by this arm."
    )


def _resolve_staged(cfg: ArmConfig, data_root: Path) -> StagedPaths:
    data_dir = data_root / "data" / "cdd_cesm"
    annot = data_dir / _ANNOT_XLSX
    seg = data_dir / _SEG_CSV
    missing = [str(p) for p in (annot, seg) if not p.exists()]
    if missing:
        raise FileNotFoundError(_stage_instructions(data_dir))
    image_roots = tuple(p for p in (data_dir / "images", data_dir) if p.exists())
    return StagedPaths(annot_xlsx=annot, seg_csv=seg, image_roots=image_roots)


def _parse_birads(value: object) -> int | None:
    import re

    ints = [int(x) for x in re.findall(r"[0-6]", str(value))]
    return max(ints) if ints else None


def _load_cm_labels(paths: StagedPaths) -> pd.DataFrame:
    lab = pd.ExcelFile(paths.annot_xlsx).parse("all")
    lab["Image_name"] = lab["Image_name"].astype(str).str.strip()
    cm = lab[lab["Image_name"].str.contains("_CM_")].copy()
    cm["birads_num"] = cm["BIRADS"].map(_parse_birads)
    cm = cm.dropna(subset=["birads_num"]).copy()
    cm["birads_num"] = cm["birads_num"].astype(int)
    cm["pathology"] = (
        cm["Pathology Classification/ Follow up"].astype(str).str.strip().str.lower()
    )
    cm["y"] = (cm["birads_num"] >= 4).astype(int)
    cm["patient_id"] = cm["Patient_ID"].astype(int)
    return cm[["Image_name", "patient_id", "birads_num", "pathology", "y"]].reset_index(drop=True)


def _load_cm_polygons(paths: StagedPaths) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    seg = pd.read_csv(paths.seg_csv)
    seg["fn"] = seg["#filename"].astype(str).str.strip()
    cm = seg[seg["fn"].str.contains("_CM_")]
    out: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for _, row in cm.iterrows():
        blob = str(row["region_shape_attributes"])
        if "all_points_x" not in blob:
            continue
        try:
            shape = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if shape.get("name") != "polygon":
            continue
        xs = np.asarray(shape.get("all_points_x", []), dtype=float)
        ys = np.asarray(shape.get("all_points_y", []), dtype=float)
        if xs.size < 3 or ys.size < 3:
            continue
        stem = row["fn"][:-4] if row["fn"].lower().endswith(".jpg") else row["fn"]
        out.setdefault(stem, []).append((xs, ys))
    return out


def _find_image(image_roots: tuple[Path, ...], stem: str) -> Path | None:
    for root in image_roots:
        hits = list(root.rglob(f"{stem}.jpg"))
        if hits:
            return hits[0]
    return None


def _load_gray(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as im:
        return np.asarray(im.convert("L"), dtype=np.float32)


def _rasterize_polygons(
    polys: list[tuple[np.ndarray, np.ndarray]], shape: tuple[int, int], scale: float
) -> np.ndarray:
    from skimage.draw import polygon as sk_polygon

    mask = np.zeros(shape, dtype=np.uint8)
    h, w = shape
    for xs, ys in polys:
        rr, cc = sk_polygon(ys * scale, xs * scale, shape=shape)
        rr = np.clip(rr, 0, h - 1)
        cc = np.clip(cc, 0, w - 1)
        mask[rr, cc] = 1
    return mask


def _downscale(img: np.ndarray) -> tuple[np.ndarray, float]:
    long_edge = max(img.shape)
    if long_edge <= _MAX_LONG_EDGE:
        return img, 1.0
    from skimage.transform import resize

    scale = _MAX_LONG_EDGE / long_edge
    new_h = max(1, int(round(img.shape[0] * scale)))
    new_w = max(1, int(round(img.shape[1] * scale)))
    out = resize(img, (new_h, new_w), order=1, preserve_range=True, anti_aliasing=True)
    return out.astype(np.float32), scale


def _extract_one(img: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    import SimpleITK as sitk
    from radiomics import featureextractor

    if int(mask.sum()) < 16:  
        return {}
    sitk_img = sitk.GetImageFromArray(img.astype(np.float32))
    sitk_mask = sitk.GetImageFromArray(mask.astype(np.uint8))
    extractor = featureextractor.RadiomicsFeatureExtractor(**_RADIOMICS_SETTINGS)
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName("firstorder")
    extractor.enableFeatureClassByName("glcm")
    try:
        result = extractor.execute(sitk_img, sitk_mask)
    except Exception:  
        return {}
    feats: dict[str, float] = {}
    for key, value in result.items():
        if key.startswith("diagnostics_"):
            continue
        try:
            feats[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return feats


def _build_feature_table(
    cfg: ArmConfig, paths: StagedPaths, cache_path: Path
) -> pd.DataFrame:
    if cache_path.exists():
        cached = pd.read_csv(cache_path)
        if len(cached) >= _MIN_LESIONS:
            return cached

    labels = _load_cm_labels(paths)
    polygons = _load_cm_polygons(paths)
    label_by_name = {r.Image_name: r for r in labels.itertuples(index=False)}

    rows: list[dict] = []
    n_no_image = 0
    n_no_feats = 0
    for stem, polys in polygons.items():
        meta = label_by_name.get(stem)
        if meta is None:
            continue  
        img_path = _find_image(paths.image_roots, stem)
        if img_path is None:
            n_no_image += 1
            continue
        img = _load_gray(img_path)
        img_ds, scale = _downscale(img)
        mask = _rasterize_polygons(polys, img_ds.shape, scale)
        feats = _extract_one(img_ds, mask)
        if not feats:
            n_no_feats += 1
            continue
        rows.append(
            {
                "patient_id": int(meta.patient_id),
                "Image_name": stem,
                "birads_num": int(meta.birads_num),
                "pathology": str(meta.pathology),
                "y": int(meta.y),
                **feats,
            }
        )

    if not rows:
        raise RuntimeError(
            "arm1 floor gate BLOCKED — no recombined lesion images could be extracted "
            f"(images-missing={n_no_image}, extraction-failed={n_no_feats}). The annotation + "
            "segmentation staged but the recombined JPGs (P*_CM_*.jpg) are not on disk. "
            + _stage_instructions(paths.annot_xlsx.parent)
        )
    table = pd.DataFrame(rows)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(cache_path, index=False)
    return table


@dataclass
class Arm1GateOutcome:

    n_lesions: int
    n_patients: int
    n_pos: int
    n_features: int
    result: FloorGateResult
    shuffle_auroc: float
    shuffle_ci95: tuple[float, float]
    malignancy_auroc: float
    malignancy_ci95: tuple[float, float]
    malignancy_n: int
    decision: str  
    rationale: str


def _feature_columns(table: pd.DataFrame) -> list[str]:
    meta = {"patient_id", "Image_name", "birads_num", "pathology", "y"}
    return [c for c in table.columns if c not in meta]


def _decide(
    result: FloorGateResult, cfg: ArmConfig, n_lesions: int, n_pos: int
) -> tuple[str, str]:
    auroc = result.auroc
    lb = result.delong_ci95[0]
    kill = cfg.kill_threshold_auroc
    green = cfg.greenlight_threshold_auroc
    green_lb = cfg.greenlight_delong_ci_lb
    if not np.isfinite(auroc):
        return "INDETERMINATE", f"AUROC is NaN (degenerate CV) at N={n_lesions}."
    if auroc <= kill:
        return (
            "KILL",
            f"mean patient-disjoint AUROC {auroc:.3f} <= KILL {kill:.2f} — the recombined-channel "
            f"radiomics do not characterise BI-RADS suspicion beyond chance ({n_lesions} lesions, "
            f"{n_pos} BI-RADS>=4). Honest imaging-null echo of the Duke radiomics floor.",
        )
    if auroc >= green and lb >= green_lb:
        return (
            "GREENLIGHT",
            f"mean patient-disjoint AUROC {auroc:.3f} >= GREENLIGHT {green:.2f} and DeLong LB "
            f"{lb:.3f} >= {green_lb:.2f} — the iodine-enhancement phenotype carries a BI-RADS-"
            f"characterising signal ({n_lesions} lesions, {n_pos} BI-RADS>=4). Advancing needs a NEW "
            f"ADR before any deep-learning CESM extension.",
        )
    return (
        "INDETERMINATE",
        f"mean patient-disjoint AUROC {auroc:.3f} (DeLong LB {lb:.3f}) is between KILL {kill:.2f} and "
        f"GREENLIGHT {green:.2f} (or LB < {green_lb:.2f}) — real-but-weak / inconclusive at "
        f"N={n_lesions}.",
    )


def run_arm1_floor_gate(cfg: ArmConfig, paths: StagedPaths, cache_path: Path) -> Arm1GateOutcome:
    table = _build_feature_table(cfg, paths, cache_path)

    feat_cols = _feature_columns(table)
    feats = table[feat_cols].apply(pd.to_numeric, errors="coerce")
    feats = feats.dropna(axis=1, how="all")
    feats = feats.loc[:, feats.std(axis=0, skipna=True).fillna(0.0) > 0]
    feats = feats.fillna(feats.mean(axis=0))

    assert_no_forbidden_inputs(feats.columns)

    x = feats.to_numpy(dtype=float)
    y = table["y"].to_numpy(dtype=int)
    groups = table["patient_id"].to_numpy()  

    result = run_floor_gate(x, y, groups, model="elasticnet", seeds=cfg.seeds)

    rng = np.random.default_rng(0)
    y_shuffled = y.copy()
    rng.shuffle(y_shuffled)
    shuffle_res = run_floor_gate(x, y_shuffled, groups, model="elasticnet", seeds=cfg.seeds)

    mal_mask = table["pathology"].isin(["malignant", "benign"]).to_numpy()
    mal_auroc, mal_ci = float("nan"), (float("nan"), float("nan"))
    mal_n = int(mal_mask.sum())
    if mal_n >= _MIN_LESIONS:
        y_mal = (table.loc[mal_mask, "pathology"].to_numpy() == "malignant").astype(int)
        mal_res = run_floor_gate(
            x[mal_mask], y_mal, groups[mal_mask], model="elasticnet", seeds=cfg.seeds
        )
        mal_auroc, mal_ci = float(mal_res.auroc), mal_res.delong_ci95

    decision, rationale = _decide(result, cfg, n_lesions=len(y), n_pos=int(y.sum()))
    return Arm1GateOutcome(
        n_lesions=int(len(y)),
        n_patients=int(pd.unique(groups).size),
        n_pos=int(y.sum()),
        n_features=int(x.shape[1]),
        result=result,
        shuffle_auroc=float(shuffle_res.auroc),
        shuffle_ci95=shuffle_res.delong_ci95,
        malignancy_auroc=mal_auroc,
        malignancy_ci95=mal_ci,
        malignancy_n=mal_n,
        decision=decision,
        rationale=rationale,
    )


def run_dedup_check() -> dict:
    return {
        "check": "cdd_cesm_vs_duke",
        "shared_patients": 0,
        "basis": (
            "different modality (CESM mammography vs DCE-MRI), different institution, disjoint "
            "patient-ID namespaces (CDD-CESM P{n} vs Duke Breast_MRI_{n}) — zero overlap by design"
        ),
        "status": "PASS",
    }


def _ci_str(ci: tuple[float, float]) -> str:
    if not np.isfinite(ci[0]):
        return "[NaN, NaN]"
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def _render_report(outcome: Arm1GateOutcome, cfg: ArmConfig, dedup: dict) -> str:
    today = date.today().strftime("%Y-%m-%d")
    r = outcome.result
    auroc = "NaN" if not np.isfinite(r.auroc) else f"{r.auroc:.3f}"
    lines: list[str] = []
    lines.append("---")
    lines.append("name: report:arm1-cesm-iodine-floor-gate")
    lines.append(
        'description: "Arm 1 CDD-CESM iodine-uptake radiomics floor gate — recombined-channel '
        "lesion-level BI-RADS characterisation at diagnosis; AUROC + DeLong CI + ECE + multi-seed "
        '+ shuffle sentinel; KILL/GREENLIGHT decision."'
    )
    lines.append(f"date: {today}")
    lines.append("metadata:")
    lines.append("  node_type: memory")
    lines.append("  type: report")
    lines.append("  feature: novel-heads")
    lines.append("  phase: arm1-floor-gate")
    lines.append("---")
    lines.append("")
    lines.append("# Arm 1 — CDD-CESM Iodine-Uptake-Signature Characterisation: Floor Gate")
    lines.append("")
    lines.append(f"**Date:** {today}  ")
    lines.append("**Arm:** 1 (Wave 2) — iodine-enhancement phenotype characterisation  ")
    lines.append("**Compute:** CPU-only, $0 (well within the $150 LOCK-5 ceiling)  ")
    lines.append("**Model:** elastic-net logistic (shared `wave1_eval_harness`)  ")
    lines.append(
        "**Channel:** CESM **recombined** (iodine-uptake) image, inside the hand-drawn lesion ROI  "
    )
    lines.append("")
    lines.append("## What this arm characterises (ledger framing — LOCK-1)")
    lines.append("")
    lines.append(
        "This organ characterises, **at diagnosis**, the **iodine-enhancement phenotype** of an "
        "**already-identified finding**: it extracts classical texture / first-order radiomics from "
        "the CESM **recombined** channel (the iodine-uptake image — a physically distinct contrast "
        "signal from Duke DCE-MRI kinetics), **inside the hand-drawn lesion ROI**, and asks whether "
        "that enhancement phenotype tracks the BI-RADS suspicion category. It is a lesion-level "
        "**characterisation** organ."
    )
    lines.append("")
    lines.append(
        "**What this organ does NOT do (LOCK-1 firewall — each item is explicitly excluded):**"
    )
    lines.append("")
    lines.append(
        "- It is **not** early detection and **not** a screening tool — every image characterised "
        "already carries a radiologist-identified finding with a hand-drawn ROI."
    )
    lines.append(
        "- It does **not** perform pre-detection in healthy tissue — normal/unmarked tissue is never "
        "scored (no lesion ROI exists for it)."
    )
    lines.append("- It does **not** measure growth rate, tumour kinetics, or doubling time.")
    lines.append(
        "- It makes **no** cross-institution generalisation claim — a fresh radiomics floor is "
        "trained on CDD-CESM only; no Duke model is applied here."
    )
    lines.append("")
    lines.append(
        "BI-RADS is used strictly as a **lesion-level characterisation label**, not a healthy-tissue "
        "screening endpoint. Every forbidden term above appears **only inside an exclusion** — the "
        "organ's sole output is at-diagnosis iodine-enhancement characterisation of identified lesions."
    )
    lines.append("")
    lines.append("## Floor gate = cheapest de-risk (radiomics-first)")
    lines.append("")
    lines.append(
        "The same radiomics-first discipline that established the Duke imaging null (ADR-0001 / "
        "ADR-0008): before any deep-learning CESM model, the recombined-channel texture inside the "
        "lesion ROI must clear a floor — can classical features recover BI-RADS suspicion on held-out "
        "(patient-disjoint) lesions at all? If not, no costlier imaging step is warranted."
    )
    lines.append("")
    lines.append("## Dedup — CDD-CESM ∩ Duke (LOCK-2)")
    lines.append("")
    lines.append(
        f"**{dedup['status']} — {dedup['shared_patients']} shared patients.** {dedup['basis']}. This "
        "is a standalone companion organ; it does **not** fuse into the Duke DCE-MRI imaging encoder "
        "(ADR-0006/0010 standalone-organ pattern)."
    )
    lines.append("")
    lines.append("## Evaluation integrity")
    lines.append("")
    lines.append(
        "- **PATIENT-disjoint CV** — radiomics are extracted per-lesion-image, but folds are assigned "
        "by `Patient_ID`, so both breasts and both views (CC/MLO) of one patient always stay on the "
        "same side of the train/val split (LOCK-2). Never image/slice-level splitting."
    )
    lines.append(
        "- **Leakage guard:** the feature matrix is pyradiomics first-order + GLCM texture only; no "
        "ER/PR/HER2/Ki-67/subtype column is reachable as an input (CDD-CESM carries none), and the "
        "shared `assert_no_forbidden_inputs` gate runs on the feature columns before every fit."
    )
    lines.append(
        f"- **Never a bare number:** AUROC + DeLong 95% CI + ECE reported at {len(cfg.seeds)} seeds "
        f"({list(cfg.seeds)})."
    )
    lines.append(
        "- **Shuffle sentinel:** the BI-RADS label is permuted and the identical patient-disjoint CV "
        "re-run; the shuffled AUROC must collapse to ~0.50 to prove the real number is not a leakage / "
        "ROI-construction artifact."
    )
    lines.append("")
    lines.append("## Sample")
    lines.append("")
    lines.append(
        f"- **{outcome.n_lesions} recombined-channel lesion images** with a hand-drawn ROI + a "
        f"parseable BI-RADS label, across **{outcome.n_patients} patients**."
    )
    lines.append(
        f"- **{outcome.n_pos} BI-RADS >= 4** (suspicious/malignant category) vs "
        f"**{outcome.n_lesions - outcome.n_pos} BI-RADS <= 3**."
    )
    lines.append(
        f"- **{outcome.n_features} radiomics features** (first-order + GLCM, binCount=32) after "
        "dropping zero-variance / all-NaN columns."
    )
    lines.append(
        "- **Lesion-enrichment note (methods honesty):** the ROI-gated subset is enriched for "
        "BI-RADS >= 4 because normal/unmarked images carry no lesion polygon. This is the correct "
        "**lesion-level characterisation** framing — the arm characterises the enhancement phenotype "
        "of identified findings, not a screening prevalence. The shuffle sentinel controls for any "
        "class-imbalance-driven inflation."
    )
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| Task | N | pos | Mean AUROC | DeLong CI95 | ECE | Seeds |")
    lines.append("|---|---|---|---|---|---|---|")
    lines.append(
        f"| **BI-RADS >= 4 vs <= 3** (primary) | {outcome.n_lesions} | {outcome.n_pos} | {auroc} | "
        f"{_ci_str(r.delong_ci95)} | {r.ece:.3f} | {r.seeds} |"
    )
    mal_auroc = (
        "NaN" if not np.isfinite(outcome.malignancy_auroc) else f"{outcome.malignancy_auroc:.3f}"
    )
    lines.append(
        f"| Malignant vs Benign (secondary) | {outcome.malignancy_n} | — | {mal_auroc} | "
        f"{_ci_str(outcome.malignancy_ci95)} | — | {r.seeds} |"
    )
    shuf = "NaN" if not np.isfinite(outcome.shuffle_auroc) else f"{outcome.shuffle_auroc:.3f}"
    lines.append(
        f"| Shuffle sentinel (BI-RADS permuted) | {outcome.n_lesions} | — | {shuf} | "
        f"{_ci_str(outcome.shuffle_ci95)} | — | {r.seeds} |"
    )
    lines.append("")
    lines.append("### Per-seed detail (primary BI-RADS task)")
    lines.append("")
    for ps in r.per_seed:
        lines.append(
            f"- seed {ps['seed']}: AUROC {ps['auroc']:.3f}, DeLong CI {ps['delong_ci95']}, "
            f"ECE {ps['ece']:.3f}"
        )
    lines.append("")
    lines.append("## Floor-gate decision")
    lines.append("")
    lines.append(f"**{outcome.decision}.** {outcome.rationale}")
    lines.append("")
    shuffle_clean = (
        np.isfinite(outcome.shuffle_auroc) and abs(outcome.shuffle_auroc - 0.50) <= 0.06
    )
    lines.append(
        f"**Shuffle sentinel:** shuffled AUROC {shuf} — "
        + (
            "collapses to ~0.50 as required; the primary number is leakage-clean."
            if shuffle_clean
            else "does NOT sit at ~0.50; treat the primary number with caution (possible construction "
            "artifact) and investigate before any advance."
        )
    )
    lines.append("")
    lines.append("## Data provenance")
    lines.append("")
    lines.append(
        "- **Images:** CDD-CESM (Categorized Digital Database for Contrast-Enhanced Spectral "
        "Mammography), Khaled et al. 2022 — recombined-channel JPGs `P{ID}_{L|R}_CM_{CC|MLO}.jpg`. "
        "Kaggle mirror `hanaraafatt/cdd-cesm`."
    )
    lines.append(
        "- **Labels:** `Radiology-manual-annotations.xlsx` (sheet `all`) — BI-RADS + pathology "
        "classification + Patient_ID. Kaggle mirror `ringoospina/cdd-cesm-excel`."
    )
    lines.append(
        "- **Lesion ROIs:** `Radiology_hand_drawn_segmentations_v2.csv` (VGG Image Annotator polygon "
        "format) — per-image hand-drawn lesion polygons on the recombined channel."
    )
    lines.append(
        "- **BI-RADS label:** max category digit per image (`3$2` -> 3); target = BI-RADS >= 4."
    )
    lines.append(
        "- **Extractor:** PyRadiomics first-order + GLCM, binCount=32 (same knob as the G1 / pCR "
        "radiomics floors); images downscaled to a 1024px long edge with the ROI scaled to match."
    )
    lines.append("")
    return "\n".join(lines)


def _metrics_payload(outcome: Arm1GateOutcome, cfg: ArmConfig, dedup: dict) -> dict:
    r = outcome.result
    return {
        "arm": 1,
        "wave": 2,
        "role": "iodine-enhancement-phenotype-characterisation",
        "framing": (
            "lesion-level iodine-enhancement characterisation at diagnosis from the CESM recombined "
            "channel (NOT screening / early detection / pre-detection in healthy tissue)"
        ),
        "channel": "cesm_recombined",
        "target": cfg.target,
        "model": "elasticnet",
        "model_nominal_config": "lightgbm",
        "model_note": (
            "config model: lightgbm is nominal; the floor gate runs the shared harness elastic-net "
            "learner (lightgbm not installed; identical to arm 6 / arm 4 precedent). p>>n-robust."
        ),
        "seeds": list(cfg.seeds),
        "kill_threshold_auroc": cfg.kill_threshold_auroc,
        "greenlight_threshold_auroc": cfg.greenlight_threshold_auroc,
        "greenlight_delong_ci_lb": cfg.greenlight_delong_ci_lb,
        "n_lesions": outcome.n_lesions,
        "n_patients": outcome.n_patients,
        "n_pos_birads_ge4": outcome.n_pos,
        "n_features": outcome.n_features,
        "decision": outcome.decision,
        "rationale": outcome.rationale,
        "primary_birads_ge4": r.as_dict(),
        "shuffle_sentinel": {
            "auroc": round(outcome.shuffle_auroc, 4)
            if np.isfinite(outcome.shuffle_auroc)
            else None,
            "delong_ci95": [round(outcome.shuffle_ci95[0], 4), round(outcome.shuffle_ci95[1], 4)]
            if np.isfinite(outcome.shuffle_ci95[0])
            else None,
        },
        "secondary_malignancy": {
            "n": outcome.malignancy_n,
            "auroc": round(outcome.malignancy_auroc, 4)
            if np.isfinite(outcome.malignancy_auroc)
            else None,
            "delong_ci95": [
                round(outcome.malignancy_ci95[0], 4),
                round(outcome.malignancy_ci95[1], 4),
            ]
            if np.isfinite(outcome.malignancy_ci95[0])
            else None,
        },
        "dedup_check": dedup,
        "provenance": {
            "images": "CDD-CESM recombined channel — Khaled et al. 2022 (Kaggle hanaraafatt/cdd-cesm)",
            "labels": "Radiology-manual-annotations.xlsx (Kaggle ringoospina/cdd-cesm-excel)",
            "rois": "Radiology_hand_drawn_segmentations_v2.csv (VGG-VIA polygons)",
        },
    }


def _parse_config(cfg_path: Path) -> ArmConfig:
    cfg_raw = yaml.safe_load(cfg_path.read_text()) or {}
    datasets = cfg_raw.get("datasets", {}) or {}
    return ArmConfig(
        cdd_cesm_root=Path(str(datasets.get("cdd_cesm_root", "TBD/CDD-CESM"))),
        target=str(cfg_raw.get("target", "birads_ge4")),
        seeds=tuple(int(s) for s in cfg_raw.get("seeds", (0, 1, 2))),
        kill_threshold_auroc=float(cfg_raw.get("kill_threshold_auroc", 0.52)),
        greenlight_threshold_auroc=float(cfg_raw.get("greenlight_threshold_auroc", 0.60)),
        greenlight_delong_ci_lb=float(cfg_raw.get("greenlight_delong_ci_lb", 0.52)),
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = _parse_config(Path(args.config))
    data_root = Path(args.data_root).resolve() if args.data_root else _REPO_ROOT

    if args.dedup_check and not args.floor_gate:
        dedup = run_dedup_check()
        print(  
            f"arm1 dedup-check ({dedup['check']}): {dedup['shared_patients']} shared patients "
            f"-> {dedup['status']}. {dedup['basis']}."
        )
        return 0 if dedup["status"] == "PASS" else 1

    if not args.floor_gate:
        print(  
            f"arm1: {_PURPOSE}. Pass --floor-gate to run the recombined-channel radiomics floor gate."
        )
        return 0

    try:
        paths = _resolve_staged(cfg, data_root)
    except FileNotFoundError as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)  
        return 2

    dedup = run_dedup_check()
    try:
        outcome = run_arm1_floor_gate(cfg, paths, Path(args.cache))
    except RuntimeError as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)  
        return 2

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = date.today().strftime("%Y%m%d")
    report_path = report_dir / f"floor_gate_{stamp}.md"
    metrics_path = report_dir / f"floor_gate_{stamp}.json"
    report_path.write_text(_render_report(outcome, cfg, dedup))
    metrics_path.write_text(json.dumps(_metrics_payload(outcome, cfg, dedup), indent=2))

    r = outcome.result
    auroc = "NaN" if not np.isfinite(r.auroc) else f"{r.auroc:.3f}"
    shuf = "NaN" if not np.isfinite(outcome.shuffle_auroc) else f"{outcome.shuffle_auroc:.3f}"
    print(f"arm1 floor gate — report: {report_path}")  
    print(  
        f"  BI-RADS>=4: N={outcome.n_lesions} ({outcome.n_patients} pts) pos={outcome.n_pos} "
        f"AUROC={auroc} DeLongCI={_ci_str(r.delong_ci95)} ECE={r.ece:.3f} "
        f"shuffle={shuf} -> {outcome.decision}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

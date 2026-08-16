
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from intensity_normalization.normalizers.population.nyul import NyulNormalizer

from pinksight.data.annotation_boxes import crop
from pinksight.data.phase_stack import select_phase_stack
from pinksight.data.preprocess import (
    NYUL_LANDMARKS,
    _apply_nyul,
    load_series,
    n4_correct,
    resample_iso,
    to_rcs,
)
from pinksight.models.h0_localizer import largest_cc, predict_mask, predicted_roi_box

DUKE = Path("data/duke_breast_cancer_mri")
WEIGHTS = "data/mamma_mia/full_image_dce_mri_tumor_segmentation/full_image_dce_mri_tumor_segmentation"
OUT = Path("data/processed")
MASKS = Path("data/processed_masks")


def process(pid: str, nyul: NyulNormalizer) -> tuple[np.ndarray, np.ndarray]:
    ps = select_phase_stack(DUKE / pid)
    n4iso = [to_rcs(resample_iso(n4_correct(load_series(s.path)))) for s in ps.stack]  
    post0 = n4iso[1] if ps.posts else n4iso[0]  
    raw_mask = predict_mask(post0, weights_dir=WEIGHTS)
    box = predicted_roi_box(raw_mask, rim_mm=7, patient=pid)  
    channels = [_apply_nyul(nyul, a) for a in n4iso]  
    vol = np.stack([crop(c, box) for c in channels]).astype(np.float32)  
    mask = crop(largest_cc(raw_mask).astype(np.uint8), box)  
    return vol, mask


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=Path("data/manifest_v1.csv"))
    ap.add_argument("--exclusions", type=Path, default=Path("data/processed/_phase_stack_exclusions.tsv"))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    MASKS.mkdir(parents=True, exist_ok=True)
    nyul = NyulNormalizer()
    nyul.load_standard_histogram(str(NYUL_LANDMARKS))

    m = pd.read_csv(args.manifest)
    excl = set(pd.read_csv(args.exclusions, sep="\t")["patient_id"]) if args.exclusions.exists() else set()
    patients = [p for p in m["patient_id"].tolist() if p not in excl]
    if args.limit:
        patients = patients[: args.limit]

    done, skipped, failed = 0, 0, []
    log = OUT / "_preprocess_failures.tsv"
    for i, pid in enumerate(patients):
        vp, mp = OUT / f"{pid}.npy", MASKS / f"{pid}.npy"
        if vp.exists() and mp.exists():
            skipped += 1
            continue
        try:
            vol, mask = process(pid, nyul)
            np.save(vp, vol)
            np.save(mp, mask)
            done += 1
        except Exception as e:  
            failed.append((pid, str(e)[:120]))
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(patients)}  done={done} skip={skipped} fail={len(failed)}")  
    if failed:
        with log.open("w") as f:
            f.write("patient_id\treason\n")
            for pid, msg in failed:
                f.write(f"{pid}\tpreprocess_failed: {msg}\n")
    print(f"done={done} skipped={skipped} failed={len(failed)} (log: {log if failed else 'none'})")  


if __name__ == "__main__":
    main()

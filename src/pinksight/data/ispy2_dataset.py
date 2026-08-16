
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

PROC_DIR = Path("data/ispy2/processed")
MASK_DIR = Path("data/ispy2/processed_masks")
NIFTI_DIR = Path("data/ispy2/nifti")

IMAGES_DIR = Path("data/mamma_mia/images")
NNUNET_MASK_DIR = Path("data/ispy2/nnunet_masks")
LABELS_TSV = Path("data/ispy2/labels.tsv")

PRE_POST_ORDER = ("pre", "post1", "post2", "post3")  

PHASE_SUFFIXES = ("_0000", "_0001", "_0002", "_0003", "_0004", "_0005")


def _load_nifti(path: Path) -> np.ndarray:
    import nibabel as nib  

    img = nib.load(str(path))
    return np.asarray(img.get_fdata(), dtype=np.float32)


def stack_phases(phase_paths: list[Path]) -> np.ndarray:
    if not phase_paths:
        raise ValueError("phase_paths is empty — need at least the pre-contrast phase")
    vols = [_load_nifti(p) for p in phase_paths]
    ref = vols[0].shape
    for p, v in zip(phase_paths, vols):
        if v.shape != ref:
            raise ValueError(
                f"phase {p} shape {v.shape} != reference {ref} — cannot stack channels"
            )
    return np.stack(vols, axis=0).astype(np.float32)  


def cache_patient(
    pid: str,
    phase_paths: list[Path],
    mask_path: Path | None = None,
    proc_dir: Path = PROC_DIR,
    mask_dir: Path = MASK_DIR,
) -> Path:
    vol = stack_phases(phase_paths)
    proc_dir.mkdir(parents=True, exist_ok=True)
    out = proc_dir / f"{pid}.npy"
    np.save(out, vol)
    if mask_path is not None:
        mask = _load_nifti(mask_path)
        if mask.shape != vol.shape[1:]:
            raise ValueError(f"mask shape {mask.shape} != volume spatial {vol.shape[1:]} for {pid}")
        mask_dir.mkdir(parents=True, exist_ok=True)
        np.save(mask_dir / f"{pid}.npy", (mask > 0).astype(np.uint8))
    return out


def resolve_phase_paths(pid: str, images_dir: Path = IMAGES_DIR) -> list[Path]:
    case_dir = images_dir / pid
    paths = [
        case_dir / f"{pid}{suf}.nii.gz"
        for suf in PHASE_SUFFIXES
        if (case_dir / f"{pid}{suf}.nii.gz").exists()
    ]
    if not paths:
        raise FileNotFoundError(f"no phase NIfTI found for {pid} under {case_dir}")
    return paths


def cache_ispy2_from_labels(
    labels_tsv: Path = LABELS_TSV,
    images_dir: Path = IMAGES_DIR,
    nnunet_mask_dir: Path = NNUNET_MASK_DIR,
    proc_dir: Path = PROC_DIR,
    mask_dir: Path = MASK_DIR,
    require_mask: bool = True,
) -> dict:
    import pandas as pd

    if not labels_tsv.exists():
        return {"cached": 0, "skipped": 0, "failures": [], "note": f"AWAITING DATA — {labels_tsv} absent"}

    df = pd.read_csv(labels_tsv, sep="\t" if labels_tsv.suffix == ".tsv" else ",")
    if "patient_id" not in df.columns:
        raise ValueError(f"no patient_id column in {labels_tsv} (cols: {list(df.columns)[:8]})")

    cached, skipped, failures = 0, 0, []
    for pid in (str(p) for p in df["patient_id"].tolist()):
        mask_path = nnunet_mask_dir / f"{pid}.nii.gz"
        if require_mask and not mask_path.exists():
            skipped += 1
            failures.append((pid, "no nnU-Net mask yet"))
            continue
        try:
            phase_paths = resolve_phase_paths(pid, images_dir)
            cache_patient(
                pid,
                phase_paths,
                mask_path=mask_path if mask_path.exists() else None,
                proc_dir=proc_dir,
                mask_dir=mask_dir,
            )
            cached += 1
        except Exception as e:  
            failures.append((pid, str(e)[:120]))
    return {"cached": cached, "skipped": skipped, "failures": failures}


def kinetic_phase_paths(pid: str, images_dir: Path = IMAGES_DIR) -> list[Path]:
    ordered = resolve_phase_paths(pid, images_dir)  
    if len(ordered) == 1:  
        return ordered
    if len(ordered) == 2:  
        return ordered
    return [ordered[0], ordered[1], ordered[-1]]


def _build_nifti_kinetic_dataset_cls():
    from .dataset import NpyVolumeDataset, _znorm, select_channels
    from .lesion_crop import lesion_crop

    class _NiftiKineticDataset(NpyVolumeDataset):

        def __init__(
            self,
            items: list[tuple[str, int]],
            images_dir: Path = IMAGES_DIR,
            nnunet_mask_dir: Path = NNUNET_MASK_DIR,
            channels: str = "kinetic",
            spatial_size: tuple[int, int, int] = (96, 96, 96),
            augment: bool = False,
            crop_size: int = 96,
        ) -> None:
            if channels != "kinetic":
                raise ValueError(
                    f"NiftiKineticDataset is disk-safe-locked to channels='kinetic' (4ch), got {channels!r}"
                )
            super().__init__(
                items,
                proc_dir=IMAGES_DIR,
                channels=channels,
                spatial_size=spatial_size,
                augment=augment,
                crop_mode="lesion",
                crop_size=crop_size,
            )
            self.images_dir = Path(images_dir)
            self.nnunet_mask_dir = Path(nnunet_mask_dir)

        def _raw_volume(self, pid: str) -> np.ndarray:
            return stack_phases(kinetic_phase_paths(pid, self.images_dir))

        def _raw_mask(self, pid: str, spatial: tuple[int, int, int]) -> np.ndarray | None:
            mp = self.nnunet_mask_dir / f"{pid}.nii.gz"
            if not mp.exists():
                return None
            m = _load_nifti(mp)
            if m.shape != tuple(spatial):
                return None  
            return (m > 0).astype(np.uint8)

        def __getitem__(self, i: int):  
            import torch

            pid, label = self.items[i]
            vol = self._raw_volume(pid)  
            vol = select_channels(vol, self.channels)  
            spatial = (vol.shape[1], vol.shape[2], vol.shape[3])
            mask = self._raw_mask(pid, spatial)  
            vol_f = np.ascontiguousarray(vol, dtype=np.float32)
            x = lesion_crop(vol_f, mask, out_size=self.crop_size)  
            x = _znorm(np.asarray(x, dtype=np.float32))  
            if self._aug is not None:
                x = np.asarray(self._aug(x), dtype=np.float32)  
            x = torch.as_tensor(x, dtype=torch.float32)
            y = torch.tensor(float(label), dtype=torch.float32)
            return x, y, pid

    return _NiftiKineticDataset


_NIFTI_KINETIC_CLS = None  


def NiftiKineticDataset(*args, **kwargs):  
    global _NIFTI_KINETIC_CLS
    if _NIFTI_KINETIC_CLS is None:
        _NIFTI_KINETIC_CLS = _build_nifti_kinetic_dataset_cls()
    return _NIFTI_KINETIC_CLS(*args, **kwargs)


def assert_patient_disjoint_split(
    items: list[tuple[str, int]], n_splits: int = 5, seed: int = 0
) -> None:
    from sklearn.model_selection import StratifiedGroupKFold  

    pids = [p for p, _ in items]
    y = np.array([lab for _, lab in items])
    groups = np.asarray(pids)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in cv.split(np.zeros(len(items)), y, groups):
        tr_pids, te_pids = set(groups[tr]), set(groups[te])
        leaked = tr_pids & te_pids
        assert not leaked, (
            f"patient(s) {sorted(leaked)[:5]} span a CV fold boundary — LOCK-2 violation"
        )


def selfcheck() -> int:
    items = [(f"ISPY2_{i:03d}", i % 2) for i in range(20)]  
    assert_patient_disjoint_split(items, n_splits=5, seed=0)
    assert_patient_disjoint_split(items + [("ISPY2_000", 0)], n_splits=5, seed=1)
    print("[ispy2] split-integrity OK — 20 synthetic patients, patient-disjoint across 5 folds.")  

    try:
        import nibabel as nib  
    except ImportError:
        print(  
            "[ispy2] nibabel absent — skipping NIfTI round-trip legs (split-integrity leg passed)."
        )
        return 0

    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        tmpd = Path(tmp)
        nifti_dir = tmpd / "nifti"
        nifti_dir.mkdir()
        import nibabel as nib

        shape = (16, 16, 12)
        phase_paths = []
        for name in ("pre", "post1", "post2"):
            arr = rng.random(shape, dtype=np.float32) * 50.0
            p = nifti_dir / f"{name}.nii.gz"
            nib.save(nib.Nifti1Image(arr, affine=np.eye(4)), str(p))
            phase_paths.append(p)
        mask_arr = np.zeros(shape, dtype=np.float32)
        mask_arr[6:10, 6:10, 4:8] = 1.0
        mask_p = nifti_dir / "mask.nii.gz"
        nib.save(nib.Nifti1Image(mask_arr, affine=np.eye(4)), str(mask_p))

        vol = stack_phases(phase_paths)
        assert vol.shape == (3, 16, 16, 12), vol.shape
        bad = nifti_dir / "bad.nii.gz"
        nib.save(nib.Nifti1Image(rng.random((8, 8, 8), dtype=np.float32), np.eye(4)), str(bad))
        try:
            stack_phases([phase_paths[0], bad])
        except ValueError:
            pass
        else:  
            raise AssertionError("mismatched-shape phase must raise")

        proc = tmpd / "processed"
        masks = tmpd / "processed_masks"
        out = cache_patient("ISPY2_TEST", phase_paths, mask_p, proc_dir=proc, mask_dir=masks)
        cached = np.load(out)
        assert cached.shape == (3, 16, 16, 12) and cached.dtype == np.float32, (
            cached.shape,
            cached.dtype,
        )
        cached_mask = np.load(masks / "ISPY2_TEST.npy")
        assert cached_mask.shape == (16, 16, 12) and cached_mask.dtype == np.uint8, (
            cached_mask.shape
        )
        assert set(np.unique(cached_mask)).issubset({0, 1}), "mask must be binary 0/1"

        try:
            import torch  
            from pinksight.data.dataset import NpyVolumeDataset

            ds = NpyVolumeDataset(
                [("ISPY2_TEST", 1)], proc_dir=proc, channels="first_post", spatial_size=(8, 8, 8)
            )
            x, y, pid = ds[0]
            assert tuple(x.shape) == (1, 8, 8, 8), x.shape  
            assert float(y) == 1.0 and pid == "ISPY2_TEST", (float(y), pid)
            print(  
                "[ispy2] NIfTI round-trip OK — cache contract (C,r,c,s), binary mask, "
                "NpyVolumeDataset read at fixed grid."
            )
        except ImportError:
            print(  
                "[ispy2] torch/monai absent — cache written+verified; skipped NpyVolumeDataset read."
            )

    return 0


def selfcheck_driver() -> int:
    try:
        import nibabel as nib
    except ImportError:
        print(  
            "[ispy2] nibabel absent — skipping driver round-trip (import nibabel to run this leg)."
        )
        print("[ispy2] driver-plumbing OK — skipped (nibabel absent).")  
        return 0

    import pandas as pd

    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        tmpd = Path(tmp)
        images = tmpd / "images"
        masks_in = tmpd / "nnunet_masks"
        images.mkdir()
        masks_in.mkdir()

        shape = (12, 12, 8)
        spec = {"ISPY2_A": (3, True), "ISPY2_B": (4, True), "ISPY2_C": (3, False)}
        for pid, (n_phase, has_mask) in spec.items():
            cdir = images / pid
            cdir.mkdir()
            for ch in range(n_phase):
                arr = rng.random(shape, dtype=np.float32) * 40.0
                nib.save(nib.Nifti1Image(arr, np.eye(4)), str(cdir / f"{pid}_{ch:04d}.nii.gz"))
            if has_mask:
                m = np.zeros(shape, dtype=np.float32)
                m[4:8, 4:8, 3:6] = 1.0
                nib.save(nib.Nifti1Image(m, np.eye(4)), str(masks_in / f"{pid}.nii.gz"))

        labels = tmpd / "labels.tsv"
        pd.DataFrame(
            {"patient_id": list(spec), "pcr": [1, 0, 1], "age": [44, 51, 38]}
        ).to_csv(labels, sep="\t", index=False)

        assert [p.name for p in resolve_phase_paths("ISPY2_B", images)] == [
            f"ISPY2_B_{i:04d}.nii.gz" for i in range(4)
        ], "phase paths must be in canonical _0000.._000N order"

        proc = tmpd / "processed"
        mask_out = tmpd / "processed_masks"
        summary = cache_ispy2_from_labels(
            labels_tsv=labels,
            images_dir=images,
            nnunet_mask_dir=masks_in,
            proc_dir=proc,
            mask_dir=mask_out,
            require_mask=True,
        )
        assert summary["cached"] == 2, summary
        assert summary["skipped"] == 1, summary
        assert any(pid == "ISPY2_C" for pid, _ in summary["failures"]), summary["failures"]
        assert not (proc / "ISPY2_C.npy").exists(), "mask-less case must NOT be cached (lesion-ROI floor)"

        for pid, n_phase in (("ISPY2_A", 3), ("ISPY2_B", 4)):
            vol = np.load(proc / f"{pid}.npy")
            assert vol.shape == (n_phase, *shape) and vol.dtype == np.float32, (pid, vol.shape)
            mk = np.load(mask_out / f"{pid}.npy")
            assert mk.shape == shape and mk.dtype == np.uint8, (pid, mk.shape)
            assert set(np.unique(mk)).issubset({0, 1}), "mask must be binary 0/1"

        empty = cache_ispy2_from_labels(labels_tsv=tmpd / "nope.tsv")
        assert empty["cached"] == 0 and "AWAITING DATA" in empty.get("note", ""), empty

    print(  
        "[ispy2] driver loop OK — labels TSV → phase-ordered stack + nnU-Net mask → cache_patient; "
        "2 cached, 1 mask-less case skipped (lesion-ROI floor), channel count == phases present."
    )
    print("[ispy2] driver-plumbing OK — Phase C1 cache driver validated on synthetic data.")  
    return 0


if __name__ == "__main__":
    import argparse

    _ap = argparse.ArgumentParser(description="ISPY2 pCR dataset loader + Phase C1 cache driver.")
    _ap.add_argument(
        "--selfcheck-driver",
        action="store_true",
        help="Phase C1 driver plumbing check (labels TSV → cache) on synthetic data, no download",
    )
    _args = _ap.parse_args()
    raise SystemExit(selfcheck_driver() if _args.selfcheck_driver else selfcheck())

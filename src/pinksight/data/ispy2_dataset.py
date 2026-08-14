"""pCR pilot (Phase C1/E1): ISPY2 NIfTI -> cached NPY volume loader for the pCR CNN/radiomics arms.

BreastDCEDL_ISPY2 ships per-patient NIfTI volumes (pre-contrast + post-contrast phases) + a lesion
segmentation mask. This module converts them to the SAME on-disk contract the existing imaging harness
eats: `(C, row, col, slice)` float32 at `data/ispy2/processed/{pid}.npy`, with C = the pre-contrast +
post-contrast phases stacked as channels ([3.3] phases-as-channels, matching Duke's `preprocess.py`).
Once cached, `pinksight.data.dataset.NpyVolumeDataset` + `train.cv.cross_val_imaging` are reused
verbatim — this file only handles the NIfTI ingest that Duke's SimpleITK DICOM path doesn't cover.

Mask policy ([plan C3], LOCK-2): the ISPY2 seg mask is used for the BOUNDING BOX only (crop geometry),
never as a pixel-level gating input, and consistently for both train and test folds. The predicted-mask
path in `NpyVolumeDataset` (crop_mode="lesion"/"box") consumes a mask on the crop grid at
`data/ispy2/processed_masks/{pid}.npy`; caching writes the resampled mask there.

DRY SURFACE: `selfcheck()` runs WITHOUT any real NIfTI on disk — it fabricates a tiny synthetic
volume+mask, exercises the NIfTI->NPY conversion end to end (via nibabel round-trip in a temp dir),
and asserts (1) the cached tensor shape/contract, (2) `NpyVolumeDataset` reads it, (3) a patient-level
StratifiedGroupKFold split keeps every patient's rows on ONE side of each fold. Real ISPY2 caching is
gated behind the 84GB download + Phase B4 pre-reg approval and is NOT run here.

ponytail: the ingest is `nibabel.load -> get_fdata -> stack phases -> (C, r, c, s)`. No N4/Nyul here
(ISPY2 is a fresh cohort; harmonisation is a Phase-C decision, not this loader's job) — the loader's
one responsibility is the on-disk contract. Resampling to a fixed grid stays in NpyVolumeDataset.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

PROC_DIR = Path("data/ispy2/processed")
MASK_DIR = Path("data/ispy2/processed_masks")
NIFTI_DIR = Path("data/ispy2/nifti")

# Phase C1 driver on-disk contract: the MAMA-MIA ISPY2 images + the nnU-Net-inferred masks.
# Images ship as data/mamma_mia/images/{pid}/{pid}_0000..000N.nii.gz (pre + post-contrast phases);
# nnU-Net masks land at data/ispy2/nnunet_masks/{pid}.nii.gz (scripts/pcr_nnunet_infer_ispy2.py).
IMAGES_DIR = Path("data/mamma_mia/images")
NNUNET_MASK_DIR = Path("data/ispy2/nnunet_masks")
LABELS_TSV = Path("data/ispy2/labels.tsv")

# ISPY2 phase file naming (documented; the real names are confirmed at Phase A3 from the download).
# The loader is naming-agnostic: it takes an EXPLICIT ordered list of phase paths so a channel-order
# mismatch (the Phase F0 risk) is impossible to introduce silently — the caller states the order.
PRE_POST_ORDER = ("pre", "post1", "post2", "post3")  # canonical channel order (pre + up to 3 post)

# MAMA-MIA phase file suffixes, in canonical channel order (pre first, then post-contrast). The driver
# stacks the phases PRESENT for a case in this order — a case with 3 phases uses _0000.._0002, a case
# with 6 uses _0000.._0005. Channel order is fixed here so it can never be silently permuted per-case.
PHASE_SUFFIXES = ("_0000", "_0001", "_0002", "_0003", "_0004", "_0005")


def _load_nifti(path: Path) -> np.ndarray:
    """Load one NIfTI volume -> (row, col, slice) float32. nibabel is a locked dep (requirements.lock)."""
    import nibabel as nib  # import-local so the module imports without the ml stack (torch-free selfcheck)

    img = nib.load(str(path))
    return np.asarray(img.get_fdata(), dtype=np.float32)


def stack_phases(phase_paths: list[Path]) -> np.ndarray:
    """Stack an ORDERED list of single-phase NIfTI volumes into (C, row, col, slice) float32.

    The caller passes phases in the canonical PRE_POST_ORDER (pre first, then post-contrast). Every
    phase must share spatial dims (assert) — a mismatched phase would silently corrupt the channel
    stack. Returns the (C, r, c, s) contract the imaging harness expects.
    """
    if not phase_paths:
        raise ValueError("phase_paths is empty — need at least the pre-contrast phase")
    vols = [_load_nifti(p) for p in phase_paths]
    ref = vols[0].shape
    for p, v in zip(phase_paths, vols):
        if v.shape != ref:
            raise ValueError(
                f"phase {p} shape {v.shape} != reference {ref} — cannot stack channels"
            )
    return np.stack(vols, axis=0).astype(np.float32)  # (C, r, c, s)


def cache_patient(
    pid: str,
    phase_paths: list[Path],
    mask_path: Path | None = None,
    proc_dir: Path = PROC_DIR,
    mask_dir: Path = MASK_DIR,
) -> Path:
    """Convert one patient's ordered NIfTI phases (+ optional mask) to the cached NPY contract.

    Writes `proc_dir/{pid}.npy` = (C, r, c, s) float32. If `mask_path` is given, writes
    `mask_dir/{pid}.npy` = (r, c, s) uint8 on the SAME grid (bounding-box source only, [plan C3]).
    Returns the volume cache path. Real ISPY2 ingest calls this per patient at Phase C1; the selfcheck
    calls it on synthetic NIfTI.
    """
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
    """Resolve one ISPY2 case's phase NIfTIs in canonical channel order (pre first, then post-contrast).

    MAMA-MIA ships `{pid}/{pid}_0000.nii.gz .. {pid}_000N.nii.gz`. Returns the phases PRESENT for this
    case in PHASE_SUFFIXES order (a case with 3 phases → 3 paths; a case with 6 → 6). Raises if no
    phase exists. Fixed order here means the (C, r, c, s) channel stack can never be silently permuted.
    """
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
    """Phase C1 DRIVER: iterate the labels TSV → per case, stack its phase NIfTIs + the nnU-Net mask →
    call `cache_patient` → write `proc_dir/{pid}.npy` + `mask_dir/{pid}.npy`.

    This is the previously-unwired driver loop. It REUSES `resolve_phase_paths` + `cache_patient` +
    `stack_phases` verbatim (no re-implementation). Masks come from `scripts/pcr_nnunet_infer_ispy2.py`
    (data/ispy2/nnunet_masks/{pid}.nii.gz); with `require_mask=True` a case with no inferred mask yet is
    CONSORT-skipped rather than cached mask-free (the floor is a lesion-ROI floor, plan C3). Returns a
    summary dict {cached, skipped, failures}. CONSORT-logs every skip/failure; never aborts the whole
    run on one bad case.

    Real run is gated behind: the 84GB image download (done), the nnU-Net mask inference pass (GPU), and
    Phase B4 pre-reg approval (granted 2026-07-16). The caching RUN itself happens once masks land.
    """
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
        except Exception as e:  # noqa: BLE001 — CONSORT-log, never fabricate a cache
            failures.append((pid, str(e)[:120]))
    return {"cached": cached, "skipped": skipped, "failures": failures}


def kinetic_phase_paths(pid: str, images_dir: Path = IMAGES_DIR) -> list[Path]:
    """Resolve the 3 NIfTI phases the `kinetic` 4ch policy needs, in [pre, first-post, LAST-post] order.

    Disk-safe (Phase E) path — the full 6-phase processed-NPY cache is ~126 MB/case (~123 GB for 980)
    and DOES NOT FIT the 152 GB free on this machine, so the CNN reads the kinetic channels straight
    from raw NIfTI on-the-fly instead of caching. `select_channels("kinetic")` builds
    (pre, first_post, first_post-pre, second_post-pre) from vol[0], vol[1], vol[2] — so to make its
    "second_post" channel the LAST available post-contrast phase (task spec: "late-post = last available
    _000N"), we hand it exactly a 3-phase stack: _0000 (pre), _0001 (first-post), _000N (last-post).
    A case with only pre+one-post degrades exactly as select_channels already handles (padded diff).
    """
    ordered = resolve_phase_paths(pid, images_dir)  # canonical _0000.._000N order, present phases only
    if len(ordered) == 1:  # pre only — no post-contrast phase at all (never expected in-cohort)
        return ordered
    if len(ordered) == 2:  # pre + one post — kinetic pads the 2nd diff (first_post-pre) itself
        return ordered
    # >=3 phases: [pre, first-post, LAST-post] so kinetic's late channel is (last_post - pre).
    return [ordered[0], ordered[1], ordered[-1]]


def _build_nifti_kinetic_dataset_cls():
    """Lazily build the disk-safe NIfTI dataset CLASS, importing NpyVolumeDataset locally.

    Deferring the base-class import keeps THIS module torch-free at import time (the selfcheck /
    selfcheck_driver dry-surface gates import ispy2_dataset with only numpy+nibabel; NpyVolumeDataset
    pulls torch+monai). The class is built once on first use and memoised on the module. `stack_phases`
    + `kinetic_phase_paths` (this module) and `select_channels`/`_znorm`/`lesion_crop` (the inherited
    pipeline) are all reused verbatim — the subclass overrides ONLY the raw-volume+mask source.
    """
    from .dataset import NpyVolumeDataset, _znorm, select_channels
    from .lesion_crop import lesion_crop

    class _NiftiKineticDataset(NpyVolumeDataset):
        """Disk-safe (pid, label) -> (x:(4,*spatial) float32, y, pid), reading kinetic 4ch from NIfTI.

        Overrides ONLY the raw-volume source of `NpyVolumeDataset`: instead of `np.load(proc/{pid}.npy)`
        it stacks [pre, first-post, last-post] NIfTI phases on-the-fly (`kinetic_phase_paths` +
        `stack_phases`) and reads the nnU-Net mask from `nnunet_masks/{pid}.nii.gz` onto the SAME native
        grid (mask inferred on the _0000 grid; all phases share it). Everything downstream —
        `select_channels("kinetic")` (4ch), lesion bounding-box crop, `_znorm`, augment — is the EXACT
        inherited, already-tested pipeline (no re-implementation, no new eval knob).

        ponytail: the whole disk-safe trick — swap the cache read for a NIfTI read, inherit the rest. No
        processed/{pid}.npy is ever written; per-case memory is one 3-phase crop, not a 6ch volume.
        Channels are FIXED to "kinetic" (task spec E3); passing anything else raises, so a caller can't
        silently drift off the disk-safe contract.
        """

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
            # proc_dir is a placeholder (we override the volume source); crop_mode='lesion' routes the
            # inherited pipeline through the nnU-Net bounding-box crop.
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
            """[pre, first-post, last-post] NIfTI phases -> (3, r, c, s) float32, on-the-fly (no cache)."""
            return stack_phases(kinetic_phase_paths(pid, self.images_dir))

        def _raw_mask(self, pid: str, spatial: tuple[int, int, int]) -> np.ndarray | None:
            """nnU-Net lesion mask (native _0000 grid) -> (r, c, s) uint8, or None -> [1.6] Duke-box fallback.

            The mask NIfTI shares the phase grid, so it matches `spatial` directly (no resample). A grid
            mismatch (unexpected) or absent mask falls back to the whole crop rather than crashing —
            same policy as the base loader's predicted-mask read (LOCK-2: predicted mask only, never GT).
            """
            mp = self.nnunet_mask_dir / f"{pid}.nii.gz"
            if not mp.exists():
                return None
            m = _load_nifti(mp)
            if m.shape != tuple(spatial):
                return None  # unreliable grid -> [1.6] fallback (not a crash)
            return (m > 0).astype(np.uint8)

        def __getitem__(self, i: int):  # type: ignore[override]
            import torch

            pid, label = self.items[i]
            vol = self._raw_volume(pid)  # (3, r, c, s) — [pre, first-post, last-post]
            vol = select_channels(vol, self.channels)  # kinetic -> (4, r, c, s)
            spatial = (vol.shape[1], vol.shape[2], vol.shape[3])
            mask = self._raw_mask(pid, spatial)  # lesion mask on the SAME grid, or None
            vol_f = np.ascontiguousarray(vol, dtype=np.float32)
            x = lesion_crop(vol_f, mask, out_size=self.crop_size)  # bounding-box crop -> fixed cube
            x = _znorm(np.asarray(x, dtype=np.float32))  # per-channel z-norm (inherited preprocessing)
            if self._aug is not None:
                x = np.asarray(self._aug(x), dtype=np.float32)  # train-only regularizer
            x = torch.as_tensor(x, dtype=torch.float32)
            y = torch.tensor(float(label), dtype=torch.float32)
            return x, y, pid

    return _NiftiKineticDataset


_NIFTI_KINETIC_CLS = None  # memoised lazy class (built on first NiftiKineticDataset(...) call)


def NiftiKineticDataset(*args, **kwargs):  # noqa: N802 — callable shim named like a class on purpose
    """Disk-safe kinetic-4ch NIfTI Dataset — instantiate on demand (torch-free module import preserved).

    Behaves exactly like a class constructor: `NiftiKineticDataset(items, ...)` returns a dataset
    instance. The real class is built lazily by `_build_nifti_kinetic_dataset_cls()` so importing this
    module (for the torch-free selfcheck gates) never pulls torch/monai. See that builder for the full
    contract. Signature: (items, images_dir=, nnunet_mask_dir=, channels='kinetic', spatial_size=,
    augment=, crop_size=).
    """
    global _NIFTI_KINETIC_CLS
    if _NIFTI_KINETIC_CLS is None:
        _NIFTI_KINETIC_CLS = _build_nifti_kinetic_dataset_cls()
    return _NIFTI_KINETIC_CLS(*args, **kwargs)


def assert_patient_disjoint_split(
    items: list[tuple[str, int]], n_splits: int = 5, seed: int = 0
) -> None:
    """LOCK-2 guard: verify a patient-level StratifiedGroupKFold keeps each patient on ONE side of
    every fold. Reused as a standalone check so Phase C4/E3 CV construction is provably patient-disjoint
    before any training. Raises AssertionError on any patient that spans a fold boundary.
    """
    from sklearn.model_selection import StratifiedGroupKFold  # import-local (torch-free base)

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
    """Runnable check WITHOUT real data: synthetic NIfTI -> cache -> NpyVolumeDataset -> disjoint split.

    Proves the loader contract on the dry surface:
      1. stack_phases yields (C, r, c, s) and rejects a mismatched-shape phase;
      2. cache_patient writes the (C, r, c, s) NPY + a matching (r, c, s) mask via a real nibabel round-trip;
      3. NpyVolumeDataset reads the cache to a (C, *spatial) tensor at the requested grid;
      4. a patient-level StratifiedGroupKFold split is patient-disjoint (assert_patient_disjoint_split).
    Skips the torch/nibabel-dependent legs cleanly if those libs are absent (same discipline as
    test_grade_2d_slice.py), but ALWAYS runs the pure-numpy split-integrity check.
    """
    # --- (4) split integrity is pure numpy+sklearn — always run it first, no data needed.
    items = [(f"ISPY2_{i:03d}", i % 2) for i in range(20)]  # 20 synthetic patients, balanced pCR
    assert_patient_disjoint_split(items, n_splits=5, seed=0)
    # a duplicated patient id across "two rows" must still land on one side (group semantics) — sanity:
    assert_patient_disjoint_split(items + [("ISPY2_000", 0)], n_splits=5, seed=1)
    print("[ispy2] split-integrity OK — 20 synthetic patients, patient-disjoint across 5 folds.")  # noqa: T201

    # --- (1)(2)(3) NIfTI round-trip needs nibabel; the dataset read needs torch+monai. Skip cleanly.
    try:
        import nibabel as nib  # noqa: F401
    except ImportError:
        print(  # noqa: T201
            "[ispy2] nibabel absent — skipping NIfTI round-trip legs (split-integrity leg passed)."
        )
        return 0

    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        tmpd = Path(tmp)
        nifti_dir = tmpd / "nifti"
        nifti_dir.mkdir()
        # write a synthetic pre + 2 post-contrast phases (16^3 each) + a lesion mask as real NIfTI.
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

        # (1) stack + mismatch rejection
        vol = stack_phases(phase_paths)
        assert vol.shape == (3, 16, 16, 12), vol.shape
        bad = nifti_dir / "bad.nii.gz"
        nib.save(nib.Nifti1Image(rng.random((8, 8, 8), dtype=np.float32), np.eye(4)), str(bad))
        try:
            stack_phases([phase_paths[0], bad])
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError("mismatched-shape phase must raise")

        # (2) cache_patient writes the (C,r,c,s) NPY + (r,c,s) mask
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

        # (3) NpyVolumeDataset reads the cache to a fixed grid (needs torch+monai)
        try:
            import torch  # noqa: F401
            from pinksight.data.dataset import NpyVolumeDataset

            ds = NpyVolumeDataset(
                [("ISPY2_TEST", 1)], proc_dir=proc, channels="first_post", spatial_size=(8, 8, 8)
            )
            x, y, pid = ds[0]
            assert tuple(x.shape) == (1, 8, 8, 8), x.shape  # first_post -> 1 channel, resized cube
            assert float(y) == 1.0 and pid == "ISPY2_TEST", (float(y), pid)
            print(  # noqa: T201
                "[ispy2] NIfTI round-trip OK — cache contract (C,r,c,s), binary mask, "
                "NpyVolumeDataset read at fixed grid."
            )
        except ImportError:
            print(  # noqa: T201
                "[ispy2] torch/monai absent — cache written+verified; skipped NpyVolumeDataset read."
            )

    return 0


def selfcheck_driver() -> int:
    """Phase C1 DRIVER plumbing check WITHOUT the 84GB download: synthetic labels TSV + fake multi-phase
    image dirs + fake nnU-Net masks → cache_ispy2_from_labels → assert per-case NPY cache + mask NPY.

    Proves the driver loop (label TSV → phase resolution → cache_patient) end to end on synthetic data:
      1. resolve_phase_paths returns the case's phases in canonical channel order (pre..postN);
      2. cache_ispy2_from_labels writes proc/{pid}.npy = (C, r, c, s) + processed_masks/{pid}.npy;
      3. a case with NO nnU-Net mask is CONSORT-skipped (require_mask=True), not cached mask-free;
      4. the number of channels equals the number of phases present for each case.
    Needs nibabel (a locked dep). Skips cleanly if nibabel is absent (same discipline as selfcheck()).
    """
    try:
        import nibabel as nib
    except ImportError:
        print(  # noqa: T201
            "[ispy2] nibabel absent — skipping driver round-trip (import nibabel to run this leg)."
        )
        print("[ispy2] driver-plumbing OK — skipped (nibabel absent).")  # noqa: T201
        return 0

    import pandas as pd

    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        tmpd = Path(tmp)
        images = tmpd / "images"
        masks_in = tmpd / "nnunet_masks"
        images.mkdir()
        masks_in.mkdir()

        # 3 cases: A has 3 phases + mask, B has 4 phases + mask, C has 3 phases but NO mask (skip).
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

        # (1) phase resolution honours channel order + count.
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
        # (2)+(3) A,B cached (have masks); C skipped (no mask), not cached mask-free.
        assert summary["cached"] == 2, summary
        assert summary["skipped"] == 1, summary
        assert any(pid == "ISPY2_C" for pid, _ in summary["failures"]), summary["failures"]
        assert not (proc / "ISPY2_C.npy").exists(), "mask-less case must NOT be cached (lesion-ROI floor)"

        # (4) channel count == phases present; mask NPY is binary on the volume grid.
        for pid, n_phase in (("ISPY2_A", 3), ("ISPY2_B", 4)):
            vol = np.load(proc / f"{pid}.npy")
            assert vol.shape == (n_phase, *shape) and vol.dtype == np.float32, (pid, vol.shape)
            mk = np.load(mask_out / f"{pid}.npy")
            assert mk.shape == shape and mk.dtype == np.uint8, (pid, mk.shape)
            assert set(np.unique(mk)).issubset({0, 1}), "mask must be binary 0/1"

        # empty/absent labels TSV → clean AWAITING-DATA summary, no crash.
        empty = cache_ispy2_from_labels(labels_tsv=tmpd / "nope.tsv")
        assert empty["cached"] == 0 and "AWAITING DATA" in empty.get("note", ""), empty

    print(  # noqa: T201
        "[ispy2] driver loop OK — labels TSV → phase-ordered stack + nnU-Net mask → cache_patient; "
        "2 cached, 1 mask-less case skipped (lesion-ROI floor), channel count == phases present."
    )
    print("[ispy2] driver-plumbing OK — Phase C1 cache driver validated on synthetic data.")  # noqa: T201
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

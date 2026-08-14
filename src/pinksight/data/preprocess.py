"""P03 sub-step (1): DCE-MRI preprocessing chain (LOCK-3) — N4 -> 1mm-iso -> Nyul,
all phases as channels, lesion ROI+margin crop, composed over the phase stack.

LOCK-3 substance: N4 bias correction + Nyul histogram-landmark normalisation + 1mm
isotropic resample; all DCE phases as channels; lesion crop + 5-10mm peritumoral margin.
LOCK-2: the Nyul standard histogram is fit on TRAIN (dev-split) patients ONLY.

MONAI has neither an N4 nor a Nyul transform, so the heavy steps use SimpleITK (N4,
DICOM read, resample) and intensity-normalization (Nyul). They are ordinary callables
assembled left-to-right by `compose` below.

ponytail: `compose` is a 3-line stand-in for monai.transforms.Compose (same call shape:
compose([f, g])(x) == g(f(x))) so sub-step (1) runs without pulling torch/MONAI. The P04
encoder loader, which needs MONAI anyway, swaps in monai.transforms.Compose unchanged.
N4 uses shrink=4 + capped iterations (the standard fast+deterministic config) — raise
N4_ITERS if QC shows bias residual. Frozen Nyul landmarks live in configs/ (LAW L-3),
so the chain reproduces without re-fitting.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from intensity_normalization.adapters.images import NumpyImageAdapter
from intensity_normalization.normalizers.population.nyul import NyulNormalizer

from .annotation_boxes import Box, crop
from .phase_stack import select_phase_stack

ISO_MM = 1.0
N4_SHRINK = 4
N4_ITERS = [50, 50, 50]  # per-resolution-level fit iterations (deterministic)
MARGIN_MM = 7  # peritumoral margin, mid of LOCK-3's 5-10mm; == voxels at 1mm iso
NYUL_LANDMARKS = Path("configs/nyul_standard_v1.npy")


def compose(steps: list[Callable]) -> Callable:
    """Left-to-right call chain — drop-in shape of monai.transforms.Compose."""

    def run(x):
        for f in steps:
            x = f(x)
        return x

    return run


def load_series(series_dir: str | Path) -> sitk.Image:
    """Read one DICOM series directory into a float32 SimpleITK volume."""
    reader = sitk.ImageSeriesReader()
    files = reader.GetGDCMSeriesFileNames(str(series_dir))
    if not files:
        raise ValueError(f"{series_dir}: no DICOM series found")
    reader.SetFileNames(files)
    return sitk.Cast(reader.Execute(), sitk.sitkFloat32)


def n4_correct(img: sitk.Image) -> sitk.Image:
    """N4 bias-field correction; bias fit on a shrunk image, applied at full resolution."""
    dim = img.GetDimension()
    mask = sitk.OtsuThreshold(img, 0, 1, 200)
    shrunk = sitk.Shrink(img, [N4_SHRINK] * dim)
    mask_shrunk = sitk.Shrink(mask, [N4_SHRINK] * dim)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations(N4_ITERS)
    corrector.Execute(shrunk, mask_shrunk)
    log_bias = corrector.GetLogBiasFieldAsImage(img)  # full-res bias field (float64)
    return img / sitk.Cast(sitk.Exp(log_bias), img.GetPixelID())


def resample_iso(img: sitk.Image) -> sitk.Image:
    """Resample to ISO_MM isotropic spacing (linear), preserving physical extent."""
    sp, sz = img.GetSpacing(), img.GetSize()
    new_size = [int(round(sz[i] * sp[i] / ISO_MM)) for i in range(img.GetDimension())]
    f = sitk.ResampleImageFilter()
    f.SetOutputSpacing([ISO_MM] * img.GetDimension())
    f.SetSize(new_size)
    f.SetOutputOrigin(img.GetOrigin())
    f.SetOutputDirection(img.GetDirection())
    f.SetInterpolator(sitk.sitkLinear)
    return f.Execute(img)


def to_rcs(img: sitk.Image) -> np.ndarray:
    """SimpleITK array is (slice, row, col); return (row, col, slice) to match Box order."""
    return np.transpose(sitk.GetArrayFromImage(img), (1, 2, 0))


def rescale_box(box: Box, native_spacing_xyz: tuple[float, float, float]) -> Box:
    """Map a native-grid Duke box into the 1mm-iso grid. SimpleITK spacing is (x=col,y=row,z=slice)."""
    sx, sy, sz = native_spacing_xyz

    def scale(lo: int, hi: int, s: float) -> tuple[int, int]:
        return int(round(lo * s / ISO_MM)), int(round(hi * s / ISO_MM))

    return Box(box.patient, scale(*box.row, sy), scale(*box.col, sx), scale(*box.slice, sz))


def fit_nyul(train_volumes: list[np.ndarray]) -> NyulNormalizer:
    """Fit a Nyul standard histogram on TRAIN volumes only (LOCK-2)."""
    nyul = NyulNormalizer()
    nyul.fit_population([NumpyImageAdapter(v.astype(np.float32)) for v in train_volumes])
    return nyul


def _apply_nyul(nyul: NyulNormalizer, arr: np.ndarray) -> np.ndarray:
    return np.asarray(nyul.transform(NumpyImageAdapter(arr.astype(np.float32))).get_data())


def preprocess_patient(
    patient_dir: str | Path, box: Box, nyul: NyulNormalizer, margin_mm: int = MARGIN_MM
) -> np.ndarray:
    """One patient -> (C, row, col, slice) float32: N4 -> 1mm-iso -> Nyul per phase, then ROI crop.

    Channels are the phase stack (pre-contrast first, then ordered post passes). The lesion
    box is rescaled to the iso grid once (all phases share the patient's native geometry).
    """
    ps = select_phase_stack(patient_dir)
    image_chain = compose([n4_correct, resample_iso])
    native_spacing = load_series(ps.pre.path).GetSpacing()
    channels = [_apply_nyul(nyul, to_rcs(image_chain(load_series(s.path)))) for s in ps.stack]
    rbox = rescale_box(box, native_spacing)
    return np.stack([crop(c, rbox, margin_mm) for c in channels], axis=0).astype(np.float32)


def golden_digest(arr: np.ndarray) -> str:
    """Stable sha256 of a processed volume — rounded to 3dp so float jitter doesn't trip CI."""
    import hashlib

    return hashlib.sha256(np.round(arr, 3).astype(np.float32).tobytes()).hexdigest()

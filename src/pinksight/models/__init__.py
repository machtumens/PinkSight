"""P04 models: H0 localiser hand-off (`h0_localizer`) + DCE-MRI encoder (`mri_encoder`).

[G2-RADIMAGENET-FROZEN-2P5D] also exports `SliceEncoder` — the frozen 2.5D slice feature extractor
for the RadImageNet-frozen imaging arm (additive; the 3D `MriEncoder` path is untouched).
"""

__all__ = ["PhysioEncoder", "SliceEncoder"]


def __getattr__(name: str):
    # Lazy re-export (PEP 562): `physio_encoder`/`slice_encoder` pull in torch at import, so defer
    # loading them until the attribute is actually accessed. This keeps torch-free submodules
    # importable without torch installed — e.g. `from pinksight.models.clinical_encoder import
    # AGE_FEATURE` (the Tier-1 `make demo` / synthetic-stream path, and clinical_encoder's own
    # deliberately torch-free design) no longer triggers an eager torch import via this package
    # __init__. `from pinksight.models import PhysioEncoder|SliceEncoder` still works unchanged when
    # the `ml` extra is present.
    if name == "PhysioEncoder":
        from .physio_encoder import PhysioEncoder

        return PhysioEncoder
    if name == "SliceEncoder":
        from .slice_encoder import SliceEncoder

        return SliceEncoder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

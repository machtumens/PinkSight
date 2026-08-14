"""PinkSight — multimodal breast-cancer subtype + Ki-67 fusion (Team TestEin, OPSI 2026).

ponytail: package root only. data/models/train/eval/viz subpackages grow per-gate (G1+),
per the manual's "don't pre-build" rule.
"""

__version__ = "0.0.0"

# Leakage ledger — fields that MUST NEVER reach the classifier inputs (manual B2 / decisions.md LOCK-2).
# Single source of truth, imported by tests/test_leakage.py and any future feature pipeline.
FORBIDDEN_FEATURES = frozenset(
    {
        "ER", "PR", "HER2", "Ki-67", "Ki67",
        "Mol Subtype", "Molecular Subtype", "Oncotype",
        # Gene-symbol level (Track B / [EXT-3] — mRNA proxy, LOCK-2 mirror): any mRNA/genomic
        # feature matrix must exclude these four IHC-defining transcripts. MKI67 is the
        # non-negotiable exclusion flagged by [EXT-3].
        "ESR1", "PGR", "ERBB2", "MKI67",
        # fastMRI-NYU (ADR-0016 / red-team Fix #6) — the labels xlsx carries these biomarker/metadata
        # columns; they are METADATA ONLY and must NEVER become encoder inputs (the H-char encoder is
        # images-only). Exact shipped column names quarantined here so the leak guard covers NYU too.
        "Grade (1-3)", "ER+ (yes=1, no=0)", "PR+ (yes=1, no=0)", "HER2+ (yes=1, no=0)",
        "Biomarkers (Raw Text)", "Malignant lesion type",
    }
)

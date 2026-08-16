
__version__ = "0.0.0"

FORBIDDEN_FEATURES = frozenset(
    {
        "ER", "PR", "HER2", "Ki-67", "Ki67",
        "Mol Subtype", "Molecular Subtype", "Oncotype",
        "ESR1", "PGR", "ERBB2", "MKI67",
        "Grade (1-3)", "ER+ (yes=1, no=0)", "PR+ (yes=1, no=0)", "HER2+ (yes=1, no=0)",
        "Biomarkers (Raw Text)", "Malignant lesion type",
    }
)

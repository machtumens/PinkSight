# Model weights license — CC-BY-NC-4.0

This file governs the **model weight files** distributed with PinkSight. It does **not** cover the
source code in this repository — the code is licensed under **Apache-2.0** (see [`LICENSE`](LICENSE)
and [`NOTICE`](NOTICE)). PinkSight is **dual-licensed**: code = Apache-2.0, model weights =
CC-BY-NC-4.0.

## What this covers

The 15 **G5 imaging-encoder** weight files, distributed as **GitHub Release assets** (they are not
committed to the git tree):

```
reports/G5_xai/weights/model_s{0,1,2}f{0-4}.pt      # 3 seeds × 5 folds — 15 files, ~1.9 GB
```

Their SHA-256 digests are recorded in [`results/TRAINED_ARTIFACTS.md`](results/TRAINED_ARTIFACTS.md)
(human-readable) and [`scripts/g5_weights.sha256`](scripts/g5_weights.sha256) (machine-readable);
fetch and verify them with [`scripts/fetch_weights.py`](scripts/fetch_weights.py).

## License: Creative Commons Attribution-NonCommercial 4.0 International (CC-BY-NC-4.0)

The weight files are licensed under **CC-BY-NC-4.0**.

- Human-readable summary (deed): <https://creativecommons.org/licenses/by-nc/4.0/>
- Full legal code: <https://creativecommons.org/licenses/by-nc/4.0/legalcode>

Under this license you may **share** (copy and redistribute) and **adapt** (remix, transform, build
upon) the weight files, under these terms:

- **Attribution (BY)** — you must give appropriate credit (see "Required attribution" below),
  provide a link to the license, and indicate if changes were made.
- **NonCommercial (NC)** — you may **not** use the weight files for commercial purposes
  (CC-BY-NC-4.0 §2(a)). This term is inherited from the training data (see "Why NonCommercial").
- **No additional restrictions** — you may **not** apply legal terms or technological measures that
  legally restrict others from doing anything the license permits.

This is a plain-language summary, not a substitute for the full legal code linked above.

## Why NonCommercial (provenance)

The weight files are **derived from the Duke-Breast-Cancer-MRI collection**, hosted on The Cancer
Imaging Archive (TCIA) under **CC-BY-NC-4.0**. The NonCommercial term of the source data propagates
to derived material, so the weights are released under the same CC-BY-NC-4.0 terms. The Apache-2.0
**code** license is unaffected and continues to cover only the source code.

These weight files are provided for reproducibility and provenance of the research pipeline; see
[`results/Table2_results.md`](results/Table2_results.md) and
[`docs/adr/0008-g3-fusion-architecture-reframe.md`](docs/adr/0008-g3-fusion-architecture-reframe.md)
for the scientific characterisation of what they do and do not demonstrate.

## Required attribution

Cite the source dataset:

> Saha, A., Harowicz, M., Grimm, L., et al. (2021). *Duke-Breast-Cancer-MRI.* The Cancer Imaging
> Archive (TCIA). DOI: [10.7937/TCIA.e3sv-re93](https://doi.org/10.7937/TCIA.e3sv-re93)

Observe the **TCIA Data Usage Policy**:
<https://www.cancerimagingarchive.net/data-usage-policies-and-restrictions/>

## Prohibited use

- **No commercial use** of the weight files (per the NonCommercial term above).
- **No re-identification.** The weight files must **not** be used to re-identify subjects or to
  attempt to recover identifiable patient information.
- **No DUA-restricted patient data** is granted by this file. Raw cohort data is never redistributed
  here and is obtained separately, under each cohort's own Data-Use Agreement — see
  [`data/README.md`](data/README.md).

## Summary

| Artifact | License | Where |
|---|---|---|
| Source code | Apache-2.0 | [`LICENSE`](LICENSE) |
| The 15 G5 model-weight files | CC-BY-NC-4.0 | this file + GitHub Release assets |
| Patient / cohort data | not redistributed (each cohort's own DUA) | [`data/README.md`](data/README.md) |

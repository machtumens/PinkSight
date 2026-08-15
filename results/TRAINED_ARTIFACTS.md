# Trained artifacts — real-data manifest

The real trained artifacts on disk, per cohort. Large weights are **not bundled** (1.9 GB); this
manifest is how you locate and verify them. Paths are relative to the repository root.

**Verify any file:** `sha256sum <path>` and compare to the hash below (or
`sha256sum -c` against a checksum file built from this table).

**Distribution:** the 15 G5 imaging-encoder weights below are distributed as **GitHub Release
assets** — fetch and verify them with `scripts/fetch_weights.py` (or `sha256sum -c
scripts/g5_weights.sha256`, the machine-readable checksum file built from this table). They are
licensed CC-BY-NC-4.0 (see `LICENSE-WEIGHTS.md`), separately from the Apache-2.0 code license. Every
other artifact listed here stays hash-manifest-only — obtain or reproduce it yourself, then verify it
against its row.

**Firewall (LOCK-1):** Track A (Duke) and Track B (TCGA-BRCA) artifacts are **separate organs on
patient-disjoint cohorts**. Never pool them, never compare their numbers. See `Table2_results.md`
for each artifact's verified result within its own cohort.

---

## Track A — Duke (MRI + clinical)

### Imaging encoder — G5 (real, but an **honest null**)
3D-ResNet-18 imaging encoder, 3 seeds × 5 folds. This is the one large artifact trained end-to-end
on real Duke MRI. Its result is a **characterised null** (encoder AUROC **0.5008**; XAI IoU
**0.123 < 0.30** on all seeds) — it is included for provenance and reproducibility, **not** as a
positive imaging result. The subtype signal on Duke comes from the clinical stream, not this encoder.

Path: `reports/G5_xai/weights/` · 15 files · 132 MB each (~1.9 GB total)

| File | SHA-256 |
|---|---|
| `model_s0f0.pt` | `fbf610c1a4522a14d6f3d3ef6af98bb1c044ecb0c0b045531e907caad56f5c67` |
| `model_s0f1.pt` | `0a2e318dd95dd9c7a5e933fbf1961ede3d6df01b373f0d4b93d26b3a590ee6f3` |
| `model_s0f2.pt` | `d0fb89cac3d56f21498e0eae75006a42a443e4ea3421f7a6f83ac67c9cd297b8` |
| `model_s0f3.pt` | `c590ec11931a99ffbb67e40582d712a1cf4bbd5296931f09afebfc0608b557d3` |
| `model_s0f4.pt` | `d41c3df8c085d1d60ac1bc2ca9bf00594fe3d8bf8b122297b6e1e389a2341139` |
| `model_s1f0.pt` | `c0be4f80fdcbf9b259f92f1aa0cce909b36e601a7bf0c9d600419b7e732fe3ac` |
| `model_s1f1.pt` | `abc2bfc2a2c66d3fe13878b10ca55721eb4b9a3c9a80526da59aeea681c5d139` |
| `model_s1f2.pt` | `0f0ce8d87a3e301b8b78a2d145d556fa01106fd5c9815a3d56d9e6e1264f6838` |
| `model_s1f3.pt` | `ba7ee592bbe07983ebd55e94b28e1cbc5fa82645e133fd6691d833dc6261c022` |
| `model_s1f4.pt` | `bad95af7eab33f997d72883347ec5d3c37c8278eb049b5dc8154e259305a53b3` |
| `model_s2f0.pt` | `50e65da6c7dd9eb1921442b4fd52560dc2f7bac2e4d03d1c294e30254914f2ca` |
| `model_s2f1.pt` | `a200d402b7ead30a53dd3a09fd365b43cf40bda1fef2fc78404cf4042a99777b` |
| `model_s2f2.pt` | `e5a623d2fcfb20db5828157dc6824e0737c78d14dd698d208c1835f43b985001` |
| `model_s2f3.pt` | `608bb4925b49051b5b6e2efe572854c91cbd2a21bc5a4c6d73b7f233f785d087` |
| `model_s2f4.pt` | `c637b1d546fd8195abffb6b0a534bc051f97434f3d936755f4dea4711c01cdc2` |

### Clinical anchor pipeline (the real signal, AUROC 0.708)
The clinical anchor is a LogReg — a small model; its train-fold-fit preprocessing state is on disk.
The fitted coefficients / OOF predictions live under `reports/G2_imaging/` and `reports/G5_external/`
(cited per row in `Table2_results.md`).

| File | Size | SHA-256 |
|---|---|---|
| `reports/G5_external/clinical_impute_stats_train.pkl` | 4 KB | `863828e97647908ac55df31ce128bac6e60573bfda6a25b6079ffe607c569ca5` |

---

## Track B — TCGA-BRCA (H&E histology) — reported alongside, firewalled

Track B models are **small heads on a frozen foundation encoder** (TITAN / UNI2-h). The heavy part
is the frozen encoder; the trained part is a LogReg / ABMIL head. What is on disk:

| File | Size | SHA-256 | Note |
|---|---|---|---|
| `data/pathology/features/TCGA_TITAN_features.pkl` | 35 MB | `af92e22a324cf164d4d1a57d1df27b2046fd57be64e1fc4788182d87bcfe6256` | Frozen TITAN slide embeddings (N=640) behind arm-3 (0.9646, ADR-0015). |
| `reports/trackb/mil_cv_uni.json` | — | `b8c98d386cf8e93cdc8bb668bc21034581184bc18605194fd06bfa1dde60e995` | UNI2-h + ABMIL cross-val result (0.9675, ADR-0012). Same 640-patient cohort as arm-3, different encoder → encoder-robustness, **not** independent corroboration. |

---

## Not a trained PinkSight artifact (upstream pretrained backbones)

For completeness — these are third-party pretrained weights the pipeline *uses*, not models PinkSight
trained: `data/pretrained/resnet50_radimagenet.pth`, `data/medicalnet/…/pretrain/resnet_*.pth`.
They carry no PinkSight result and are outside the firewall.

_Every result number for these artifacts is in `Table2_results.md`, verified against its source JSON
and `decisions.md`._

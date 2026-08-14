# CODEBASE_MAP.md — orientation digest for the PinkSight release package

> **Read this instead of re-scanning.** One file = oriented to the whole release tree. This maps the
> **repo and code**; it does not repeat the claim boundary, gates, or targets — those live in
> `docs/CLAIM_LEDGER.md`. This is the public release package (Team TestEin, OPSI 2026), not the
> internal working tree: there is no `JOURNAL.md`, `decisions.md`, `process/`, or `prompts/` here.

## 30-second frame
Breast **DCE-MRI + clinical** multimodal-fusion research artifact: subtype characterisation
(Luminal-A vs TNBC) + Ki-67 stratification at diagnosis + explainable cross-attention fusion.
Research / OPSI 2026, **not** a clinical product. The headline finding is an honest null (see
`results/Table2_results.md` and `README.md`). Every number is a within-cohort result on its own
patient-disjoint cohort; numbers are never pooled or juxtaposed across cohorts (LOCK-1).

## Where the frozen results are (don't re-derive — open these)
- **The reportable numbers** → `results/Table2_results.md` (the full results matrix, every rung with
  its DeLong CI + ECE + shuffle + source JSON). `results/Table1_cohorts.md` = cohort table;
  `results/TRAINED_ARTIFACTS.md` = SHA-256 manifest for the (un-bundled) weights + frozen features.
- **Locked decisions / rationale** → `docs/adr/` (16 ADRs, `0001`…`0016`). These decision records are
  the authority for why each architectural / scoping choice was made.
- **The claim boundary (allowed vs forbidden framing)** → `docs/CLAIM_LEDGER.md`, enforced in CI by
  `ci/ledger_lint.py`. **Cohort acquisition + DUAs** → `data/README.md`. **N-waterfall / splits /
  leakage** → `docs/DATA_CARD.md`.
- **System architecture (routing, framing guard)** → `docs/architecture/pinksight_system_architecture.md`.

## Repo map (every top-level path, one line)
| Path | What's there |
|---|---|
| `README.md` | Public front page: honest-null thesis, results table, architecture, how-to-run, claim ledger, cohorts/DUA, citation. Read first. |
| `LICENSE` / `NOTICE` | Apache-2.0 (code only). `NOTICE` states: data not redistributed (DUA-bound), weights hash-manifest-only. |
| `CITATION.cff` | Machine-readable citation metadata (Team TestEin, OPSI 2026). |
| `pyproject.toml` | Package + tiered extras (`ml` / `arms` / `trackb` / `dev`). Base install = pandas/numpy only. |
| `Makefile` | The clone-and-run surface: `make demo` (zero-data synthetic), `make reproduce` (real-data, needs your cohorts). Plain `python3`/`pip`, no `uv`. |
| `pinksight_cli.py` | Stdlib-only terminal "jury workstation" — renders MOCK reports by construction; `--selfcheck` runs the ledger-guard wiring check. |
| `requirements.lock.txt` | Pinned dependency lock for reproducible installs. |
| `src/pinksight/` | The importable package (see Package surface below). |
| `scripts/` | Runnable entrypoints: preprocessing, G3/G5 training + eval, Track-B MIL, Track-C panel, fastMRI-NYU, the synthetic demo/reproduce runners, and `pinksight_dispatch.py` (cohort/modality → harness router). Subdirs `fva/`, `novel_heads/`. |
| `configs/` | Hydra configs + the frozen patient-level split `split_v2.yaml` (scanner/year quasi-external holdout, shipped) + Nyúl `.npy` landmarks + `novel_heads/` arm configs. |
| `results/` | Frozen result tables + the trained-artifact SHA-256 manifest (see above). |
| `paper/` | `pinksight_paper.md`, `figures/` (7 figures × png+svg + manifest), `tables/`. |
| `notebooks/` | Presented-evidence notebooks. `SYNTHETIC — NOT A RESULT` = synthetic control-sentinel render; `MOCK` = mock workstation payload. Neither is a real result. |
| `docs/` | `adr/` (16 ADRs), `architecture/`, `CLAIM_LEDGER.md`, `DATA_CARD.md`, this `CODEBASE_MAP.md`. |
| `app/` | Desktop "workstation" demo (Tauri v2 + React front end in `src/`, `src-tauri/`; Python inference `sidecar/`; `design/` mockups). OPSI demo surface, off the result spine; MOCK by default. |
| `ci/` | Lint gates. `ledger_lint.py` is wired into `.github/workflows/ci.yml`. `consistency_lint.py` + `map_lint.py` are present but **not** wired (their targets `decisions.md` / `docs/map/MASTER.md` are intentionally not shipped) — see the note below. |
| `tests/` | pytest integrity suite (~40 files) + `fixtures/`. LOCK-2 leakage/split firewall lives in `test_leakage.py` + `test_splits.py`. |
| `data/` | `README.md` only — **no patient data ships** (`.gitignore` walls `data/*` except the README). |
| `.github/workflows/ci.yml` | 3 jobs: `ledger-lint`, `leakage` (LOCK-2 firewall, unfiltered), `demo-smoke` (documented consumer path). |

## Package surface (`src/pinksight/`, what runs)
- `seed.py` — `set_seed()` (default 42); reproducibility. `metrics.py` — top-level metric helpers.
- `baseline/` — `radiomics_baseline.py` (the G1 radiomics floor).
- `data/` — cohort loaders + preprocessing: `dataset.py`, `ispy2_dataset.py`, `fastmri_nyu.py`,
  `preprocess.py`, `lesion_crop.py`, `phase_stack.py`, `annotation_boxes.py`, and the demo fixtures
  `synthetic_cohort.py` / `synthetic_streams.py` (the zero-data control-sentinel generators).
- `models/` — encoders + heads: `clinical_encoder.py` (FT-Transformer, LOCK-3), `mri_encoder.py`
  (3D-ResNet), `fusion.py` (modality-dropout cross-attention), `heads.py` (Head-1 subtype / Head-2
  Ki-67), `h0_localizer.py` (predicted-ROI localizer, scipy-only), `pcr_head.py` (ADR-0011 deferred
  slot), plus `physio_encoder.py` / `slice_encoder.py` / `tiny_cnn_2d.py`.
- `eval/` — `ablation.py`, `calibration.py`, `metrics.py`, `selective.py`, `leakage_probe.py`,
  `e2e_report_contract.py`, `nyu_duke_slot.py`.
- `stats/` — `compare.py` (DeLong / paired-DeLong), `temperature.py` (calibration scaling).
- `trackb/` — `mil.py` (ABMIL over frozen WSI features), `gate.py` (`assert_gate_open`, LOCK-6),
  `genomics.py`, `head.py`, `modality_dropout.py`, `tiles.py`.
- `train/` — patient-level cross-validation + training loops: `cv.py`, `cv_slices.py`, `loop.py`.
- `xai/` — `saliency.py` (Grad-CAM / HiResCAM), `faithfulness.py`.

## Integrity gates (== CI)
`.github/workflows/ci.yml` runs three jobs on every push/PR:
1. **`ledger-lint`** — `ci/ledger_lint.py --selfcheck` then `ci/ledger_lint.py .` (claim-boundary guard
   over `.md` / `.ipynb` / `.tsx` / `.html`; `docs/adr/` and the by-name ledger-defining files are
   exempt because they quote the bans in order to forbid them).
2. **`leakage`** — the LOCK-2 leakage/split firewall (`test_leakage.py` + `test_splits.py`), run
   unfiltered; a small subset needs DUA-bound real-data files and shows `SKIPPED` by design (that
   real-data attestation ran once locally, EVL-confirmed; CI never sees the data).
3. **`demo-smoke`** — `pip install -e . && make demo` (the documented consumer path).

> **Orphaned lint scripts (known gap, non-blocking):** `ci/consistency_lint.py` and `ci/map_lint.py`
> ship in `ci/` but have no wired CI job here — their targets (`decisions.md` and `docs/map/MASTER.md`)
> are internal working-tree docs, correctly not part of the public release. A future maintenance pass
> may either port their targets and wire the jobs, or remove the two scripts.

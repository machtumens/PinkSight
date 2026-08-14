# ADR-0007: pCR-to-NAC characterisation pilot on BreastDCEDL_ISPY2 (+ ISPY1 honesty holdout)

Date: 2026-07-14 (proposed) · 2026-07-16 (B4 sign-off)   Status: **ACCEPTED** (Richard's B4 pre-reg approval granted 2026-07-16; does NOT move any LOCK — still flips to final-number ratification at Phase H2)

> **This ADR records a NEW additive pilot; it does NOT propose changing any LOCKED invariant.**
> Unlike ADR-0006 (which narrowed LOCK-1 to admit a recurrence organ), the pCR pilot fits the
> **existing** LOCK-1 ALLOWED framing ("characterisation … AT diagnosis") with no amendment — see
> §Claim-ledger compatibility. Per **LAW L-1**, the operative go/no-go is the dated `decisions.md`
> `[PCR-PILOT-ISPY2]` entry; the PROPOSED entry (2026-07-14) is now joined by an ACCEPTED entry
> (2026-07-16). **B4 sign-off is GRANTED (Richard, 2026-07-16)** — the pre-reg hard gate is cleared;
> feature extraction / model runs may proceed under the plan's remaining gates (radiomics floor →
> GREENLIGHT at D3 → CNN). Two facts recorded at sign-off (data-source substitution + mask policy)
> are in §B4 sign-off provenance below.

## Context

The two flagship imaging targets are sealed imaging-nulls on our anchor cohort:

- **Subtype (Luminal-vs-TNBC):** best recoverable pooled-OOF AUROC **0.567**, DeLong UB ≤ 0.624
  (< LOCK-4's 0.75), sealed 6-axis-independent across H4 (info-ceiling) + H6 (MRI non-additive).
- **Grade (NHG1-vs-NHG3):** imaging-NULL across both the 3D-cube (0.502) and DeepRadGrade's own
  2D-per-slice recipe (0.564-but-shuffle-noise). Head-2 dropped (`[HEAD2-GRADE-PIVOT]` REJECTED).

Both nulls are *honest and defensible*, and the FVA (Fusion-Value-Attribution) standard packages them
as the submission spine. But the project would be stronger with **one ledger-clean binary imaging
target that the imaging can actually predict** — to demonstrate the pipeline recovers signal where
signal exists, not only that it correctly reports its absence.

**pCR (pathologic complete response to neoadjuvant chemotherapy) is that target.** It is a binary
outcome read from a POST-treatment pathology report; the model input is the **PRE-treatment** DCE-MRI.
The literature (Syed 2022 et al.) reports recoverable pre-treatment imaging signal for pCR
(radiomics floor ~0.65–0.72; deep ~0.78–0.83). This is a genuinely different question from
subtype/grade — response phenotype, not molecular class — so the prior nulls do not predict this one.

The data exists publicly and cleanly: **BreastDCEDL_ISPY2** (982 pts, CC-BY-4.0, TCIA) for internal
patient-level CV, and **ISPY1 / ACRIN-6657** (222 pts, TCIA) as a quasi-external honesty holdout. Duke
is not used; the sealed G2 subtype architecture is not touched.

## Decision (proposed)

Stand up a **pCR prediction head on BreastDCEDL_ISPY2** as an additive pilot, cheapest-first:

1. **Radiomics floor** (PyRadiomics, first post-contrast inside lesion mask) → patient-level
   StratifiedGroupKFold pooled-OOF AUROC + DeLong CI + shuffle sentinel. **$0 local.**
2. **GREENLIGHT gate (pre-committed, FROZEN):** floor pooled-OOF AUROC **≥ 0.65** AND shuffle ≤ 0.55
   → proceed to CNN. **< 0.65 → record informative floor/null, close pilot, log to decisions.md.**
3. **3D-CNN arm** (3D-ResNet-18 `MriEncoder` per ADR-0001 + pCR head, focal loss, **amp=False**) —
   3 seeds × 5 folds, pooled-OOF AUROC [DeLong CI] + SmoothECE. **$0 smoke / ~$5–10 RunPod.**
4. **ISPY1 external honesty pass** — inference only; report internal→external AUROC + Δ. **Honesty,
   not a generalisation claim.**
5. **Handoff** — DeLong + SmoothECE + multi-seed reporting → dated decisions.md follow-up entry.

Backbone rides ADR-0001 (3D-ResNet-18/34 + MedicalNet, the G2-ratified default) — this pilot does NOT
re-open ADR-0001. Mask policy: provided ISPY2 seg masks used for BOUNDING BOX only, consistently for
train and test folds (LOCK-2; no ground-truth-mask gating at test).

## Claim-ledger compatibility (LOCK-1 — NO amendment needed)

pCR sits inside LOCK-1's **existing** ALLOWED clause because the characterisation is anchored at
diagnosis on the pre-treatment scan:

- **ALLOWED framing (used verbatim in all reports):** "characterisation of a treatment-relevant
  imaging phenotype AT DIAGNOSIS", "pCR prediction from pre-treatment DCE-MRI imaging biomarkers",
  "recoverable signal at this cohort's information ceiling."
- **FORBIDDEN framing (barred from every report/JOURNAL line):** "early detection", "screening",
  "growth rate", "kinetics over time", "doubling time", "clinical-trial-grade FP/FN reduction", and
  any cross-institution generalisation CLAIM. The ISPY1 pass reports the drop (Δ) — it does not claim
  generalisation.

Why no LOCK-1 amendment (contrast with ADR-0006): the recurrence organ read naturally as
disease-course/kinetics and needed a carve-out; pCR is a single-timepoint response phenotype
characterised from a single pre-treatment snapshot, which is already ALLOWED. This ADR therefore
**records a design decision** rather than proposing a LOCK change.

## Evaluation integrity (LOCK-2 — unchanged, code-enforced)

- **Patient-level splits only** — StratifiedGroupKFold, patient as group; no slice/image-level splits.
  Enforced by `src.pinksight.data.ispy2_dataset.assert_patient_disjoint_split` +
  `train.cv.cross_val_imaging`'s per-fold patient-disjoint assert.
- **ISPY1 holdout carved FIRST** — `configs/ispy1_holdout.yaml` is the machine-readable external
  contract; every listed patient is excluded from all ISPY2 training folds. Carved before any training.
- **Leakage audit** (`scripts/pcr_leakage_audit.py`) — dual-guard: the shared `FORBIDDEN_FEATURES`
  ledger (ER/PR/HER2/Ki-67/…) + a pCR-specific outcome-pattern audit (pcr label / RCB / residual /
  response / survival / on-treatment / post-treatment). HR/HER2 permitted as stratification metadata
  ONLY, never a model input.
- **De-dup** (`scripts/pcr_dedup_ispy.py`) — assert 0 patient overlap ISPY2 ∩ ISPY1 and ISPY2 ∩ Duke.
- **Reporting** — pooled-OOF AUROC + DeLong CI + multi-seed spread + shuffle sentinel (≤ 0.55) +
  SmoothECE (bin-free, small-N-robust; `src.pinksight.eval.calibration.smooth_ece`).
- **fp16-NaN lesson** — `amp=False` hard-set in the pilot's `TrainCfg` (G2 fold-5 crash).

## ISPY1-holdout-carved-first attestation

At Phase A7, `configs/ispy1_holdout.yaml` is created and committed BEFORE any ISPY2 CV fold is
constructed (plan acceptance criterion 3). On the current $0 dry surface the file exists as a
well-formed DRY STUB (`populated: false`, empty `patient_ids`) because the ISPY1 label TSV is not yet
downloaded; Phase A2 populates the patient IDs and flips `populated: true`. No ISPY2 training may run
against an unpopulated holdout — Phase F1 asserts no ISPY1 ID appears in any ISPY2 training fold.

## B4 sign-off provenance (recorded 2026-07-16)

Two facts recorded at Richard's B4 sign-off. Both are within the pilot's declared scope; neither moves
a LOCK; ALLOWED framing is preserved throughout.

1. **Data-source substitution — MAMA-MIA's ISPY2 (980 pts) for BreastDCEDL_ISPY2 (982 pts).** The
   internal cohort is delivered via the **MAMA-MIA** Synapse release (`syn60868042`), the SAME
   underlying TCIA I-SPY2 source but a different curation/harmonisation than the BreastDCEDL packaging
   this ADR's Context/Datasets sections name. N reconciliation: **980** labelled ISPY2 (`patient_id` +
   `age` inputs only) vs 982 named; ISPY1 honesty holdout **167** labelled (4 null-pCR dropped) vs 222
   named. De-dup + leakage audits already GREEN on the delivered data (ISPY2∩ISPY1 = 0, ISPY2∩Duke =
   0). Read "BreastDCEDL_ISPY2" as "MAMA-MIA ISPY2 (980)" throughout. The GREENLIGHT rule, shuffle
   sentinel, patient-level CV, and honest-null commitment are unchanged.

2. **Mask policy — real nnU-Net-inferred lesion masks.** The lesion ROI uses masks **predicted by the
   nnU-Net segmenter shipped inside MAMA-MIA** (`full_image_dce_mri_tumor_segmentation/`, an nnU-Net v2
   model folder), produced by a one-time ~$2 GPU inference pass
   (`scripts/pcr_nnunet_infer_ispy2.py`, RunPod playbook). This resolves Open Question 4 to a **true
   lesion-ROI floor** rather than the interim "expert-mask-aided bounding box" — masks used
   consistently for BOTH train and test folds, ROI geometry only (never a pixel-level gating input,
   LOCK-2). Cost inside the ~$5–10 pilot envelope, well under the $150 cap. Masks feed the existing
   cache contract (`data/ispy2/processed_masks/{pid}.npy`) unchanged.

## Consequences

- **Positive:** a ledger-clean binary imaging target the pipeline can (per literature) actually
  predict, complementing the honest-null spine; a new public dataset (no Duke overlap); a clean
  internal→external honesty pass. Reuses the entire existing imaging harness (CV loop, shuffle
  sentinel, DeLong, radiomics extractor) — minimal new surface.
- **Negative / risks:** (1) GREENLIGHT may not fire (floor < 0.65) — pre-committed as an acceptable
  informative-null outcome. (2) Internal→external collapse (Δ > 0.15–0.20) — reported transparently,
  never suppressed; investigate scanner/year confound. (3) 84 GB download logistics (Aspera/TCIA
  HTTPS). (4) ISPY2 TSV column schema + ISPY1 channel/phase mapping unverified until download —
  resolved at Phase A/F0.
- **Compute:** ~$5–10 total, well inside LOCK-5 ($150 cap).

## Follow-up levers (not in the pilot)

- BreastDCEDL 2,070-pt scale-up (Nat. Sci. Data 2026) if the CNN clears ~0.78.
- ACRIN-6698 DWI/ADC channel as an additional modality if the pilot succeeds.

## Status / ratification

**PROPOSED — pending Richard's B4 pre-reg sign-off.** On approval, this ADR stays PROPOSED through
Phases C–G and flips to **ACCEPTED** at Phase H2 with the final headline number (or NO-GO null)
recorded, alongside the dated `decisions.md` `[PCR-PILOT-ISPY2]` follow-up entry. No LOCK moves at any
point — this is an additive pilot, not an invariant change.

References: plan `process/general-plans/active/pcr-pilot-ispy2_14-07-26/pcr-pilot-ispy2_PLAN_14-07-26.md`;
pre-reg `reports/pcr_pilot/PCR-PILOT-ISPY2_PREREG_14-07-26.md`; decisions.md `[PCR-PILOT-ISPY2]`
(2026-07-14, PROPOSED); backbone ADR-0001; contrast ADR-0006 (recurrence organ, LOCK-1 amendment).

---

## Outcome & Ratification — 2026-07-16

**Status:** ACCEPTED (Richard, 2026-07-16) — ratified as an official PinkSight pilot, architecture-forward framing with the integrity firewall below in force.

### The floor result (IMMUTABLE — never soften, round up, or call a pass)

Radiomics floor run completed 2026-07-16 on the full MAMA-MIA ISPY2 cohort (N = 980; 0 empty masks, 0 CONSORT skips). nnU-Net single-fold lesion masks inferred on a RunPod A5000 (~$0.24). pCR prevalence 0.322.

| Metric | Value |
|---|---|
| Pooled-OOF AUROC | **0.599** |
| DeLong 95% CI | **[0.561, 0.637]** |
| Per-seed AUROCs | 0.596 / 0.586 / 0.615 |
| Shuffle sentinel | **0.498** (≤ 0.55 — leakage-clean) |
| GREENLIGHT gate (≥ 0.65) | **NO-GO (computed)** |

The DeLong CI is **entirely below 0.65**. The pre-registered GREENLIGHT threshold did not fire. Phase E (3D-CNN arm) is **NOT run and is NOT claimed.** ISPY1 external pass was **NOT run** and is **NOT claimed.** Reports on disk: `reports/pcr_pilot/metrics_floor.json` and `reports/pcr_pilot/PCR-PILOT-ISPY2_FLOOR_REPORT_16-07-26.md`.

### Architecture-forward ratification (positioning only — mirrors ADR-0008 discipline)

The pilot is ratified as a **methods/pipeline contribution** and as an **independent pre-registered pCR characterisation on a distinct cohort (ISPY2 / MAMA-MIA)** — not as a passing model, not as a detection claim, and NOT as a cross-cohort or cross-institution generalisation of any imaging ceiling.

Three concrete contributions recorded:

1. **Validated end-to-end pilot pipeline** — label adapter → dedup/leakage audits → nnU-Net mask inference → lesion-ROI radiomics → DeLong/shuffle floor. Every harness component is reusable for the broader project.
2. **Independent ISPY2 pCR pilot: weak-but-above-chance floor on its own cohort and target.** The pCR imaging→treatment-response signal is *real but weak* (CI entirely above 0.50 — signal exists; CI entirely below 0.65 — GREENLIGHT threshold not met). This pilot is conducted on **ISPY2 (MAMA-MIA, N=980)** — a **distinct cohort from the Duke subtype/grade work** (NOT Duke-adjacent; NOT the same cohort family). The prior project imaging characterisations are: subtype imaging-null on the **Duke cohort** (best floor 0.567) and grade imaging-null on the **Duke cohort** (0.502–0.564). The resemblance in floor magnitudes across the three targets is a within-project observation and **must NOT be framed as a single imaging ceiling that generalises across cohorts or institutions — subtype and grade are Duke; pCR is ISPY2; these are distinct cohorts and no cross-cohort law is asserted.** Cross-institution generalisation remains FORBIDDEN under LOCK-1, unchanged by this ratification.
3. **New ledger-clean target class on the official record.** pCR-at-diagnosis (characterisation of a treatment-relevant imaging phenotype) is now a ratified, pre-registered PinkSight pilot with a full integrity record.

**ALLOWED wording for every reference to this result:** "characterisation of a treatment-relevant imaging phenotype at diagnosis", "recoverable-but-weak signal at this cohort's information ceiling", "informative floor", "pre-registered radiomics floor pilot", "independent ISPY2 pCR pilot (distinct cohort from Duke subtype/grade work)".

### "Does NOT do" firewall (mandatory — read before citing this ADR)

Ratifying this ADR as architecture-forward does **NOT** license:

- ❌ Any claim that **GREENLIGHT fired** or that **the model passes** the pre-registered threshold. It does NOT. AUROC 0.599, CI [0.561, 0.637] — NO-GO.
- ❌ Any claim that **"imaging predicts pCR"** or that pCR signal is strong / clinically useful. The floor is informative; it is not a passing model.
- ❌ Any reference to the **3D-CNN arm (Phase E)** as run or as yielding a number. Phase E was **NOT run**. No CNN number exists.
- ❌ The **ISPY1 external honesty pass** as run or as yielding a number. It was **NOT run**.
- ❌ **"Early detection"**, **"screening"**, **"growth rate"**, **"kinetics over time"**, **"doubling time"**, **"clinical-trial-grade FP/FN reduction"**, or **any cross-institution generalisation** claim — these remain FORBIDDEN under LOCK-1 verbatim, unchanged by this ratification.
- ❌ Any claim that **a single "imaging ceiling" generalises across cohorts or institutions.** Subtype and grade characterisations are **Duke cohort** work; pCR is **ISPY2 (MAMA-MIA)** — these are distinct cohorts. The floor magnitudes may be noted as a within-project observation, but no cross-cohort ceiling law is asserted. Framing ISPY2 as "Duke-adjacent" or as the "same cohort family" is FORBIDDEN.
- ❌ Presenting 0.599 as a "strong", "respectable", or "near-threshold" result. It is an informative floor with a CI that excludes 0.50 but does not reach the pre-registered gate.
- ❌ Any implication that the result is **"under-powered"** or that "more data would pass the gate." N = 980 is the full available cohort. The floor is a real, N-adequate result. "Under-powered" framing is forbidden-framing drift.
- ❌ Moving any LOCK. No LOCK is amended by this ratification.

### No LOCK moved

LOCK-1 (claim discipline), LOCK-2 (leakage & evaluation integrity), LOCK-3 (cohort/label scope), LOCK-4 (baseline→MVP gate), LOCK-5 (compute ceiling), and LOCK-6 (scope protection) are all **unchanged**.

### Ratification block

- [x] Richard — 2026-07-16 — **ratified: pCR pilot is an official PinkSight pilot; architecture-forward framing per this Outcome section; "does NOT do" firewall in force; no LOCK moved; GREENLIGHT NO-GO recorded honestly and immutably.**

---

### Phase D3 — gate override (2026-07-16)

**This is an OVERRIDE, NOT a pass. Read before citing.**

The pre-registered GREENLIGHT threshold of **≥ 0.65** remains **frozen and unchanged**. The radiomics floor AUROC of **0.599** (DeLong CI [0.561, 0.637]) **did NOT clear it**. The CI is entirely below 0.65. Per the pre-committed rule in §Decision (proposed) item 2, the mandatory outcome of a floor < 0.65 is: "record informative floor/null, close pilot, log to decisions.md."

**Richard has explicitly elected to override this rule and proceed to Phase E (the 3D-CNN arm).** This decision is recorded here as a conscious, on-record gate override — not a reinterpretation of the result, not a retroactive lowering of the gate, not a claim that GREENLIGHT fired.

**Stated rationale for the override (two parts):**

1. **The floor signal is real.** DeLong CI [0.561, 0.637] excludes 0.50 entirely; the shuffle sentinel is 0.498 (leakage-clean). The floor reflects a genuine above-chance imaging phenotype on N=980.
2. **The radiomics floor is a linear lower bound.** The floor is produced by a logistic regression on handcrafted PyRadiomics features. A 3D-CNN operating on raw voxel data can extract nonlinear spatial features that handcrafted radiomics cannot represent. The floor may understate recoverable nonlinear signal; Phase E is the empirical test of whether that headroom exists above the 0.65 threshold.

**What the override authorises:**

Phase E — 3D-ResNet-18 + pCR head, `amp=False`, 3 seeds × 5 folds, patient-level OOF AUROC, DeLong CI, SmoothECE, shuffle sentinel — is authorised to proceed. The result will be reported with the same rigour as the radiomics floor and framed honestly. A CNN AUROC < 0.65 is a second NO-GO; a CNN AUROC ≥ 0.65 is a GREENLIGHT; no other interpretation is admissible.

**What the override does NOT do:**

- Does NOT lower, modify, or retroactively reframe the pre-registered threshold of 0.65.
- Does NOT constitute a GREENLIGHT or a pass of any gate.
- Does NOT pre-judge the CNN result or predict it will pass.
- Does NOT amend the "does NOT do" firewall in §Outcome & Ratification; all forbidden framings remain barred.
- Does NOT move any LOCK (LOCK-1 through LOCK-6 unchanged).
- Does NOT license calling 0.599 a "near-threshold", "strong", or "respectable" result. It is an informative NO-GO floor.

**Reference:** `decisions.md` entry `[PCR-PILOT-ISPY2] PHASE-D3 GATE OVERRIDE` (2026-07-16); floor report `reports/pcr_pilot/PCR-PILOT-ISPY2_FLOOR_REPORT_16-07-26.md`.

---

### Phase E — freeze_bn correction + relaunch (2026-07-20)

**This subsection records a training-artifact correction and a relaunch. It does NOT reframe any result, does NOT move any LOCK, and does NOT touch the "does NOT do" firewall above (all forbidden framings remain barred).**

**What went wrong (factual).** The first from-scratch Phase E CNN attempt reported pooled-OOF AUROC **0.4986** (at-chance). That number is now known to be a **training artifact, NOT a data null.** The run used `freeze_bn=True` in the model factory, which — on this batch=1 config — has been proven to make the 3D-ResNet-18 emit **constant output** (a degenerate network): aligned overfit probes on the exact production config give a probability spread of only **0.0003** and AUROC **~0.46** under `freeze_bn=True`. A network that cannot separate even a deliberately-overfit 20-patient training set cannot produce a meaningful CV number; the 0.4986 reflects the frozen-BN degeneracy, not the imaging→pCR signal.

**The fix (factual).** `model_factory_nifti()` in `scripts/pcr_cnn_ispy2.py` was changed `freeze_bn=True → freeze_bn=False` — plain `BatchNorm3d` at batch=1, MedicalNet transfer (`resnet_18_23dataset.pth`) retained. Aligned overfit-probe evidence on the corrected path:

| Config | Overfit-probe AUROC | Prob spread |
|---|---|---|
| `freeze_bn=True` (degenerate) | ~0.46 | 0.0003 |
| plain `BatchNorm3d` (freeze_bn=False) | **1.0000** | 0.068 → 0.909 |
| GroupNorm (alt) | 1.0000 | — |
| **Fixed production config** (MedicalNet + freeze_bn=False + BN @ batch=1) | **1.0000** | 0.068 → 0.909 |

The fixed production-config sanity probe is on disk: `reports/pcr_pilot/sanity_overfit_fixed.log` (`SANITY_OVERFIT_AUROC=1.0000`, MedicalNet load warning fired). The loop is now proven functional; a fresh 3-seed × 5-fold OOF run is relaunched on the corrected config under a fresh pre-registration (`reports/pcr_pilot/PCR-PILOT-ISPY2_PREREG-CORRECTED_20-07-26.md`), with the **frozen GREENLIGHT gate ≥ 0.62 pooled-OOF AUROC** (Richard, immutable) unchanged. The 13 stale `freeze_bn=True` checkpoints were namespaced to `reports/pcr_pilot/oof_ckpt_freezebn_FAILED/` so the resume logic cannot silently pool them.

**Blast-radius cross-reference (no reframing).** The `freeze_bn=True` degeneracy affects **only the two freeze-BN G2 arms** (`run_r18_mn_corrected_smoke`, `run_smoke_lesion`). The plain-BN G2/G3 runs used a **functional** loop, so the **ADR-0001 imaging→subtype null and the ADR-0008 fusion-ceiling nulls STAND** — they were produced by a non-degenerate network and are unaffected by this correction. This cross-reference is recorded per the session's blast-radius audit; it changes no G2/G3 number and moves no LOCK.

**Firewall intact.** Nothing in this correction licenses "imaging predicts pCR", early-detection, kinetics, cross-institution generalisation, or any GREENLIGHT/pass claim. Phase E remains a pre-registered one-shot test against the frozen ≥ 0.62 gate; a result < 0.62 is a NO-GO, honestly reported. No LOCK (LOCK-1 through LOCK-6) is moved by this subsection.

**Reference:** pre-reg `reports/pcr_pilot/PCR-PILOT-ISPY2_PREREG-CORRECTED_20-07-26.md`; sanity evidence `reports/pcr_pilot/sanity_overfit_fixed.log`; backbone ADR-0001; fusion ADR-0008.

---

### Phase E — CNN outcome (2026-07-22)

**Status: PHASE E COMPLETE — NO-GO. Genuine null from a proven-functional loop.**

**This subsection records the final Phase E CNN result. It does NOT move any LOCK, does NOT reframe the "does NOT do" firewall above (all forbidden framings remain barred), and does NOT claim that "imaging predicts pCR."**

#### The result (IMMUTABLE)

3D-ResNet-18 + pCR head, `amp=False`, `freeze_bn=False` (corrected config), MedicalNet transfer, 3 seeds × 5 folds, local RTX 4060, $0, pid-aligned OOF. N = 980.

| Metric | Seed 0 | Seed 1 | Seed 2 | 3-seed mean |
|---|---|---|---|---|
| Pooled OOF AUROC | 0.4979 | 0.4839 | 0.4803 | **0.4874** |
| DeLong 95% CI | [0.459, 0.537] | [0.445, 0.523] | [0.441, 0.520] | — |
| ECE | — | — | — | 0.174 |

All three DeLong CIs cross 0.50. The pooled-OOF mean AUROC is **0.4874**, well below both the pre-registered radiomics floor (0.599) and the frozen GREENLIGHT gate (≥ 0.62). **The gate did not fire. Phase E is a NO-GO.**

#### Genuine null — the loop is proven functional

This is NOT a repeat of the earlier `freeze_bn=True` training artifact. The loop was proven functional before this run:

- Corrected sanity overfit probe (`freeze_bn=False`, MedicalNet, 20-pt set): AUROC **1.000**, prob spread 0.068 → 0.909. A degenerate network cannot do this.
- Predictions show real spread and calibration (ECE 0.174) — not a dead constant.
- The earlier `freeze_bn=True` artifact (0.4986) has been separated and namespaced.

The CNN could learn from the training data. It simply did not recover separable pCR signal in the 5-fold patient-level OOF regime on N = 980 against this target. This is a **characterised information ceiling for this cohort, this target, and this architecture** — the same honest-null discipline applied in G3 (ADR-0008) and in the G2 subtype/grade arms.

#### G2/G3 blast-radius audit — ADR-0001 and ADR-0008 stand

The `freeze_bn=True` degeneracy affected only two G2 corroborating arms:

| Run | freeze_bn | Loop status | AUROC | Classification |
|---|---|---|---|---|
| run_r18_mn_corrected_smoke | True (dead) | Degenerate (constant output) | 0.491 | Artifact — corroborating arm only |
| run_smoke_lesion | True (dead) | Degenerate (constant output) | 0.504 | Artifact — corroborating arm only |
| run_r18_scratch | False | Functional | 0.489 | Genuine null |
| run_r18_medicalnet | False | Functional | 0.518 | Genuine null |
| run_r18_mn_focal | False | Functional | 0.509 | Genuine null |
| run_r18_mn_bn_fixed4 | False | Functional | 0.530 | Genuine null |
| HEAD2-GRADE_SMOKE | False | Functional | 0.502 | Genuine null |
| G3 flat-fusion | — | Functional (over plain-BN embeddings) | 0.636 | Genuine |
| G3 hierarchical #4 | — | Functional | 0.599 | Genuine |
| G3 MoE #7 | — | Functional | 0.645 | Genuine |
| Grade 2D-slice | — | Functional (batch=32) | 0.564 | Genuine null |
| Clinical-alone | — | n/a | 0.708 | Genuine ceiling |

**ADR-0001 (imaging→subtype null) and ADR-0008 (G3 fusion ceiling) are LARGELY VINDICATED.** The imaging nulls were produced by functional loops. Only two freeze-BN corroborating arms carry an asterisk and must be cited only as confirmatory evidence, never as the primary basis of the imaging-null claim. The primary plain-BN arms (run_r18_medicalnet at 0.518, run_r18_mn_bn_fixed4 at 0.530) stand. One optional follow-up exists: a GroupNorm re-run of a single G2 arm would confirm norm-robustness of the plain-BN nulls — but it is not required to sustain the existing ADR claims.

#### Shuffle sentinel

PID 172328 (`metrics_cnn_shuffle.json`) completed 2026-07-23. Pooled-OOF AUROC **0.5019** [DeLong CI 0.4634, 0.5404], ECE 0.0987, N = 980, 1 seed. **PASS — leakage-clean.** CI straddles 0.50; result is at chance; well below the pre-registered SHUFFLE_MAX 0.55 threshold. Real ≈ shuffle ≈ chance → the Phase E CNN null is genuine and leak-free. Full picture: radiomics floor 0.599 (ratified) · CNN real 0.4874 (genuine null) · CNN shuffle 0.5019 (leakage-clean).

#### "Does NOT do" firewall — unchanged, Phase E additions

In addition to the existing firewall items:

- ❌ Any claim that Phase E CNN recovers pCR signal. It does NOT. AUROC 0.4874, all CIs cross 0.50.
- ❌ Framing 0.4874 as "under-powered" or "would improve with more data." N = 980 is the full available ISPY2 cohort; this result is N-adequate. "Under-powered" framing is forbidden drift.
- ❌ Any claim that the radiomics floor (0.599) implies CNN headroom exists. Phase E tested that hypothesis empirically and it did not hold on this cohort.
- ❌ Any cross-institution generalisation. The pCR null is on ISPY2 / MAMA-MIA — distinct from the Duke subtype/grade work; no cross-cohort ceiling law is asserted.

The existing "does NOT do" items from §Outcome & Ratification remain in force, unchanged.

#### No LOCK moved

LOCK-1 through LOCK-6 are all unchanged. No gate target is modified. The frozen GREENLIGHT threshold of **≥ 0.62** (Phase E, per `PCR-PILOT-ISPY2_PREREG-CORRECTED_20-07-26.md`) is preserved as written — the result of 0.4874 is a NO-GO against it.

**Reference:** Phase E pre-reg `reports/pcr_pilot/PCR-PILOT-ISPY2_PREREG-CORRECTED_20-07-26.md`; CNN metrics `reports/pcr_pilot/metrics_cnn.json`; shuffle sentinel `reports/pcr_pilot/metrics_cnn_shuffle.json` (COMPLETE — 0.5019, leakage-clean PASS, 2026-07-23); handoff `reports/pcr_pilot/PCR-PILOT-HANDOFF_17-07-26.md`; backbone ADR-0001; fusion ADR-0008; freeze_bn correction §Phase E — freeze_bn correction + relaunch (2026-07-20) above.

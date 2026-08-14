# ADR-0001: Imaging encoder — resolve the DenseNet121-3D ↔ MedicalNet inconsistency
Date: 2026-06-20   Status: proposed (records OPEN-LOCK **O-1**; final choice deferred to G2)

## Context
decisions.md OPEN-LOCK **O-1** marks the combo "DenseNet121-3D ([3.1]) + MedicalNet-pretrained ([3.2])"
INVALID as written: MedicalNet publishes pretrained weights only for **3D-ResNet** backbones
(ResNet-10/18/34/50…), never for DenseNet. So "DenseNet121-3D, MedicalNet-pretrained" cannot both
be true — one half has to give. The binding constraints: (a) the imaging branch is a 3D DCE-MRI
encoder over a small single-institution cohort, so transfer from medical pretraining is worth more
than raw capacity; (b) compute is capped (~$150, **LOCK-5**), so a large from-scratch 3D search is
unaffordable. This is a real, judge-visible inconsistency — record it, do not silently patch it.

## Options considered
1. **3D-ResNet-18/34 + MedicalNet pretrained weights (recommended)** — pro: genuine medical transfer
   on the exact backbone MedicalNet ships; small, fits the **LOCK-5** compute cap; widely reproduced.
   Con: less raw capacity than DenseNet121.
2. **DenseNet121-3D + Models Genesis (or from scratch)** — pro: keeps DenseNet capacity if it helps;
   Models Genesis gives self-supervised medical pretraining. Con: heavier; no MedicalNet transfer;
   more compute risk against the **LOCK-5** cap.
3. **Run both as a small ablation, decide on the G2 number** — pro: the choice becomes evidence-backed
   instead of a guess, and yields a clean ablation row for the paper. Con: spends part of the budget
   before G3 (the committed floor, **LOCK-6**).

## Decision
<PENDING — to be ratified at **G2** against the unimodal-encoder number (**LOCK-4**, the G2 row).
Recommended default: Option 1 (3D-ResNet-18 + MedicalNet), which is also the encoder named in
`CLAUDE.md`. Escalate to Option 3 only if the G2 number fails to beat the radiomics baseline and the
remaining budget against the **LOCK-5** cap allows.>

## Consequences
- Every "DenseNet121-3D (MedicalNet)" mention must carry the **O-1** OPEN caveat until this ADR is
  accepted — enforced by the consistency-lint (`ci/consistency_lint.py`), the leak/leak-rule assertion
  ([1.16]), and the reopened [3.1] note in decisions.md.
- Whatever backbone is chosen, the encoder is trained AND evaluated on the **same realistic input
  available at test time** (**LOCK-2**). Under decisions.md v1 that input is the lesion crop from
  Duke's **provided annotations** ([1.6]); if the architecture-v2 detection front-end (forward register
  **H0**, tracked as **O-4**) is later adopted, the same invariant requires **PREDICTED, never
  ground-truth, masks** at test. The integrity constraint is independent of the backbone choice.
- Accepting a backbone unblocks the G1/G2 encoder build (prompt-library register **P04**, planned) to
  fix its O-1 placeholder.

---

## Status update — 2026-07-08 (recipe-rescue ablation running)

The `znorm+aug-v1` ratification sweep returned r18-scratch 0.489 and r18+MedicalNet 0.518 — both below the 0.567 radiomics floor (LOCK-4 not cleared). **O-1 ratification is therefore still PENDING.**

A norm×channel factorial rescue ablation has been **pre-registered in decisions.md** ([G2-RESCUE], 2026-07-08) before runs start. Four arms test whether the null is recipe-level (freeze-BN + multi-phase channels fix it) or architecture-level (all arms stay at chance):

- `r18_mn_fbn_first` / `r18_mn_fbn_prepost` / `r18_mn_bn_prepost` / `r18_mn_fbn_fixed4`

Success criterion: any arm clears pooled-OOF AUROC 0.567 with DeLong 95% CI + shuffle sentinel + 3-seed spread. A null result (all arms ≤ 0.567) is an acceptable, honest outcome and is itself evidence that the imaging null is architecture-level — this strengthens (rather than blocks) ADR-0001 closure: if recipe cannot save it, Option 1 (3D-ResNet-18 + MedicalNet) is ratified on the null, and the project routes to the honest-null branch [4.7] (clinical + radiomics anchor the fusion stream).

This ADR will be closed once the rescue results land (expected: 2026-07-09).

---
*Citations resolve against `decisions.md` → Invariant & Reference Register (v1.1). Template: `docs/templates/ADR_TEMPLATE.md`.*

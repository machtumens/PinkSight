# PinkSight — System Architecture: one modality-dropout skeleton, many separately-trained organs

**Date:** 2026-08-04 · **Status:** methods design-note (Piece A of the arm-integration plan) · **Moves no LOCK.**

> **Doc-location note (out of scope for this plan):** `docs/architecture/` is a NEW folder — none of
> `docs/{adr,lit,map,ref,source-pdfs,templates}` hosts this kind of doc. A future `make map` /
> `docs/map/MASTER.md` refresh should link this file from E4 (models & training) or a new E-node. That
> refresh is explicitly **not** performed by this plan; this doc is discoverable via `scripts/pinksight_dispatch.py`
> and the arm-integration task folder until then.

---

## 1. North star — "one system," honestly framed

PinkSight is presented for submission as **one modality-dropout skeleton hosting many separately-trained
organs**, each fit on its own cohort with its own labels. There is **no single network** that ingests
every modality, and there is **no shared parameter trained across cohorts**. What makes it "one system"
is (a) a shared modality-dropout *design pattern* — any organ can be present or absent at inference — and
(b) a single **cohort/modality → harness routing table** (`scripts/pinksight_dispatch.py`) that selects
which own-cohort harness owns a given input.

This is the same **"no >1-cohort gradient"** contract ADR-0011 established for the pCR task-head slot
(frozen-trunk / separate-weights), generalised into a lookup table instead of restated per organ. The
dispatcher *selects* an organ; it never *fuses* organs and never trains anything across cohorts.

---

## 2. Topology (routing only — zero gradient across every arrow)

```
patient / cohort record  --(cohort tag + available modality set)-->  dispatch()
    |
    |   ── Track A (Duke MRI+clinical — the headline) ─────────────────────────────────
    +-- ("duke", {mri, clinical}) ----------> scripts/train_g3_hierarchical.py        [BUILT]
    +-- ("duke", {clinical})       ---------> NOT WIRED (E9: no clinical-only training entrypoint)
    +-- ("duke", {clinical, recurrence}) ---> NOT WIRED (ADR-0006 organ — no scripts/ entrypoint)
    |
    |   ── Track B (TCGA-BRCA WSI / genomics — gated stretch upside) ──────────────────
    +-- ("tcga_brca", {wsi})          ------> scripts/trackb_mil_cv.py                 [BUILT]
    +-- ("tcga_brca", {wsi, genomics}) -----> scripts/trackb_fusion_wsi_genomics.py    [Piece B — NEW]
    +-- ("tcga_brca", {hrd})          ------> scripts/novel_heads/arm4_hrd_brcaness.py [GREENLIGHT]
    |
    |   ── Companion organs (other cohorts — standalone, own-cohort) ──────────────────
    +-- ("metabric",   {purity})    --------> scripts/novel_heads/arm2_purity_admixture.py       [GREENLIGHT]
    +-- ("cptac_brca", {proteomic}) --------> scripts/novel_heads/arm8_proteogenomic_discordance.py [GREENLIGHT-PILOT, tension]
    +-- ("cdd_cesm",   {cesm})      --------> scripts/novel_heads/arm1_cesm_iodine_radiomics.py    [GREENLIGHT]
    +-- ("cmmd",       {ffdm})      --------> scripts/novel_heads/arm9_cmmd_modality_transfer.py   [KILL]
    +-- ("track_c",    {tabular})   --------> scripts/track_c_tabular_panel.py                     [ADR-0010 ensemble companion — NOT fusion]
    +-- ("fastmri_nyu",{dce})       --------> scripts/train_fastmri_nyu.py                         [ADR-0016 NYU-INTERNAL NO-GO null; NYU-only]
    |
    +-- unknown combo               --------> NOT WIRED — confirm path or backlog (never guesses)

Every arrow is INFERENCE-TIME ROUTING ONLY. No arrow trains a shared parameter across cohorts.
```

---

## 3. Dispatcher registry — status by track (Track A and Track B in SEPARATE blocks, LOCK-1)

Statuses/decisions are **cited from `docs/map/e5-novel-heads-arms.md`'s scoreboard** (not re-derived here).
Per **LOCK-1**, Track A (Duke) and Track B (TCGA-BRCA) entries live in separate blocks and are **never
placed in side-by-side columns**; no cross-institution juxtaposition or transfer is implied anywhere.

### 3A. Track A — Duke MRI + clinical (the headline)

| cohort | modalities | harness | dispatcher status | note |
|---|---|---|---|---|
| duke | mri + clinical | `scripts/train_g3_hierarchical.py` | WIRED | G3 hierarchical fusion + MoE (BUILT; see decisions.md / docs/CLAIM_LEDGER.md gate spine) |
| duke | clinical | — | NOT WIRED | E9 (04-08-26): `train_imaging_mvp.py` is pure-imaging with no `--clinical-only-path` flag; no clinical-only Duke *training* entrypoint exists — backlog |
| duke | clinical + recurrence | — | NOT WIRED | ADR-0006 clinical-companion recurrence-stratification organ has no standalone `scripts/` entrypoint — backlog |

### 3B. Track B — TCGA-BRCA WSI / genomics (gated stretch)

| cohort | modalities | harness | dispatcher status | e5 decision |
|---|---|---|---|---|
| tcga_brca | wsi | `scripts/trackb_mil_cv.py` | WIRED | BUILT (UNI2-h ABMIL; methods-rigour only) |
| tcga_brca | wsi + genomics | `scripts/trackb_fusion_wsi_genomics.py` | WIRED | **Piece B — NEW** (this plan; late-fusion, not cross-attention) |
| tcga_brca | hrd | `scripts/novel_heads/arm4_hrd_brcaness.py` | WIRED | GREENLIGHT (arm 4) |

### 3C. Companion organs — other cohorts (standalone, own-cohort)

| cohort | modalities | harness | dispatcher status | e5 decision |
|---|---|---|---|---|
| metabric | purity | `scripts/novel_heads/arm2_purity_admixture.py` | WIRED | GREENLIGHT (arm 2) |
| cptac_brca | proteomic | `scripts/novel_heads/arm8_proteogenomic_discordance.py` | WIRED | GREENLIGHT-PILOT — **replication tension, PI-gated** (arm 8); every reference must co-locate the Mertins-2016 replication failure with the discovery number |
| cdd_cesm | cesm | `scripts/novel_heads/arm1_cesm_iodine_radiomics.py` | WIRED | GREENLIGHT (arm 1) |
| cmmd | ffdm | `scripts/novel_heads/arm9_cmmd_modality_transfer.py` | WIRED | KILL — corroborates the ADR-0008 imaging null on a second modality/cohort (arm 9) |
| track_c | tabular | `scripts/track_c_tabular_panel.py` | WIRED | ADR-0010 ensemble companion (Coimbra/BCSC/METABRIC); ensemble NOT fusion; path-scope-amended LOCK-1 for Track-C artifacts only; zero shared patients |
| fastmri_nyu | dce | `scripts/train_fastmri_nyu.py` | WIRED | ADR-0016 standalone NYU encoder — NYU-INTERNAL NO-GO honest null (AUROC 0.599 DeLong [0.4303,0.7676], LB<0.60); own-cohort, NYU-only. **Guard: NEVER juxtapose an NYU number with a Duke number — the 0.599 ≡ ADR-0008 Duke hierarchical-#4 figure is PURE COINCIDENCE (different institution + task), never placed side-by-side.** |

> **Registry-freshness caveat (known gap):** `scripts/pinksight_dispatch.py --selfcheck` checks that each
> WIRED script *path exists*, NOT that its CLI contract is callable nor that these GREENLIGHT/KILL/PILOT
> status annotations stay in sync with the e5 scoreboard as arms change. Treat the decision column as a
> point-in-time citation of e5, not a live mirror.

---

## 4. Framing guard restated inline (LOCK-1 / LOCK-2 / LOCK-6)

- **LOCK-1 (claim discipline).** Every organ here does **subtype characterisation / molecular-snapshot
  characterisation AT DIAGNOSIS** only — strictly inside the CLAIM LEDGER's ALLOWED framings, and none of
  its FORBIDDEN framings (see `decisions.md` LOCK-1 for the exact banned list). Track A (Duke) numbers and
  Track B (TCGA-BRCA) numbers are never juxtaposed — routing a Duke and a TCGA-BRCA cohort through one
  table implies no transfer between them, and no organ claims performance that generalises beyond its own
  cohort.
- **LOCK-2 (leakage / evaluation integrity).** No organ takes a label-defining field (ER/PR/HER2/Ki-67/
  Mol-Subtype/Oncotype or their gene-symbol mirrors ESR1/PGR/ERBB2/MKI67) as an input. Piece B adds a
  structurally-separate PAM50 proliferation co-target whose zero-gradient firewall is proven by
  `tests/test_leakage.py::test_pam50_proliferation_never_feeds_subtype_head` (structural + bit-for-bit
  empirical equivalence).
- **LOCK-6 (two-track scope).** Track A (Duke MRI+clinical) is the headline; Track B (TCGA-BRCA WSI/
  genomics) is gated, time-boxed upside behind `assert_gate_open()`. The dispatcher documents both without
  promoting Track B to headline status.

---

## 5. Non-independence caveat — arm 3, arm 5, AND Piece B share one embedding set

From `docs/map/e5-novel-heads-arms.md` (verbatim): *"Both [arm 3 and arm 5] read the same 640 TCGA-BRCA
patients through the same frozen TITAN matrix. They are two targets on one feature set — not two
independent GREENLIGHTs of one claim."*

**Piece B (`trackb_fusion_wsi_genomics.py`) is a THIRD readout on that same 640-patient cohort and the same
frozen TITAN embeddings.** Its WSI logit is exactly arm 3's TITAN signal. Piece B is therefore **not
independent evidence** of a WSI subtype signal either — arm 3, arm 5, and Piece B are three targets read
off one embedding set on one cohort. Any submission summary must state this rather than count them as
independent corroboration. And per e5: arm 3 is **TCGA-BRCA H&E histopathology, a different modality and
cohort from Duke DCE-MRI — it does NOT reverse or reopen the Duke imaging ceiling** (ADR-0008 unchanged).

---

## 6. Governance pointer (not executed by this doc)

Piece B produces one new intra-TCGA-BRCA ΔAUC number. Per the arm-integration plan, that number does **not**
reach `decisions.md` / `JOURNAL.md` until a **new ADR** (extending ADR-0008 / 0011 / 0014 / 0015) is drafted
and a **`/red-team` pass** clears it — a separate, later session, not part of this doc or this plan's EXECUTE
phase. This architecture doc records the routing and framing only; it ratifies nothing.

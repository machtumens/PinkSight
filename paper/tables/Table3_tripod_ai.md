# Table 3 — Completed TRIPOD+AI reporting checklist

> Paper supplement. Checklist source: **Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement:
> updated guidance for reporting clinical prediction models that use regression or machine learning
> methods. *BMJ* 2024;385:e078378** — official checklist PDF, version **11-January-2024**
> (tripod-statement.org). All 27 items / 52 sub-items are reproduced (requirement text condensed from the
> official wording; the authoritative phrasing is the source checklist). "Applies": **D** = development
> only, **E** = evaluation only, **D;E** = both. Status maps against `reports/paper/pinksight_paper.md`.
> Cohort contrast throughout is **luminal-like vs TNBC** (characterisation at diagnosis). Estimators are
> named beside each result; Track B (TCGA-BRCA H&E) is reported **alongside** Track A and never compared
> with it. Item 2 defers to the separate *TRIPOD+AI for Abstracts* checklist (not itself completed here).

**Summary:** Complete **26** · Partial **19** · Missing **7** (of 52 sub-items). Ranked gaps at the foot.

## Title

| Item | Topic | Applies | Requirement (condensed) | Status | Manuscript location / gap note |
|---|---|---|---|---|---|
| 1 | Title | D;E | Identify the study as developing/evaluating a multivariable prediction model, the target population, and the outcome. | Complete | Title identifies a multimodal model, its development + evaluation, and the outcome (subtype characterisation, luminal-like vs TNBC, at diagnosis); target population is implicit. |

## Abstract

| Item | Topic | Applies | Requirement (condensed) | Status | Manuscript location / gap note |
|---|---|---|---|---|---|
| 2 | Abstract | D;E | See the *TRIPOD+AI for Abstracts* checklist. | Partial | Structured abstract (Background/Methods/Results/Conclusions) with DeLong CIs, calibration and the external result is present; the separate *TRIPOD+AI for Abstracts* sub-checklist is not itself completed. |

## Introduction

| Item | Topic | Applies | Requirement (condensed) | Status | Manuscript location / gap note |
|---|---|---|---|---|---|
| 3a | Background | D;E | Healthcare context (diagnostic/prognostic) and rationale, with references to existing models. | Complete | §1 gives the diagnostic-characterisation context and fusion rationale; cites existing models (Lipkova'22, Waqas'24, Witowski'24, Li'16, Couture'18). |
| 3b | Background | D;E | Target population and intended purpose in the care pathway, incl. intended users. | Partial | §1/§2 describe the at-diagnosis characterisation purpose and research-only posture; intended users (clinicians/patients/public) are not named. |
| 3c | Background | D;E | Describe any known health inequalities between sociodemographic groups. | Partial | Regional late-diagnosis burden (Indonesia/ASEAN) motivates the study; known inequalities *between sociodemographic groups* (e.g. race/ethnicity) are not stated. |
| 4 | Objectives | D;E | Specify objectives, incl. whether development, validation, or both. | Complete | §1 states the pre-registered primary objective (fusion ΔAUROC ≥ 0.03); §2 frames the study as development (Duke) plus quasi-external evaluation (ISPY2). |

## Methods

| Item | Topic | Applies | Requirement (condensed) | Status | Manuscript location / gap note |
|---|---|---|---|---|---|
| 5a | Data | D;E | Sources of data separately for development and evaluation, rationale, representativeness. | Complete | §2.2 + Table 1 separate development (Duke DCE-MRI+clinical) from evaluation (ISPY2) and Track B (TCGA-BRCA); representativeness (single-institution, skewed) noted in §4. |
| 5b | Data | D;E | Dates of collected participant data (accrual start/end; end of follow-up). | **Missing** | No data-collection / accrual date ranges are given for any cohort. |
| 6a | Participants | D;E | Key elements of study setting; number and location of centres. | Partial | §2.2 notes Duke is single-institution and Track B uses leave-one-site-out; explicit centre counts/locations per cohort are not given. |
| 6b | Participants | D;E | Eligibility criteria for study participants. | Complete | §2.3 + Fig 2 give eligibility and the N-waterfall (922→759 in scope; HER2-enriched and ER/PR+HER2+ excluded). |
| 6c | Participants | D;E | Treatments received and how handled during development/evaluation, if relevant. | Partial | Track A uses pre-treatment diagnostic imaging (treatment handling not applicable) but this is not stated; the pCR companion's neoadjuvant context is not detailed. |
| 7 | Data preparation | D;E | Data pre-processing and quality checking, incl. whether similar across sociodemographic groups. | Partial | §2.4/§2.13 describe MONAI preprocessing (N4, 1 mm resample, Nyúl) and image QC, fit on the training fold; uniformity of QC across sociodemographic groups is not addressed. |
| 8a | Outcome | D;E | Define the predicted outcome and time horizon, how/when assessed, rationale, cross-group consistency. | Complete | §2.1/§2.3 define the outcome (molecular subtype, luminal-like vs TNBC) at diagnosis, IHC-derived label used only as ground truth, with rationale; cross-group consistency clause not addressed. |
| 8b | Outcome | D;E | If outcome assessment is subjective, describe assessor qualifications/demographics. | Partial | Labels are registry/IHC-derived (objective assay), so subjective-assessor description is effectively N/A; not stated. |
| 8c | Outcome | D;E | Report any actions to blind assessment of the outcome. | Partial | Pre-existing retrospective subtype labels; outcome-assessment blinding is effectively N/A; not stated. |
| 9a | Predictors | D | Choice of initial predictors and any pre-selection before model building. | Complete | §2.4 gives the nine leakage-safe clinical predictors (common to Duke and ISPY2) and DCE-MRI phase volumes, with rationale. |
| 9b | Predictors | D;E | Define all predictors, how/when measured (and any blinding). | Complete | §2.4 defines all predictors and their at-diagnosis measurement; the forbidden leakage set is excluded (LOCK-2, CI-asserted). |
| 9c | Predictors | D;E | If predictor measurement is subjective, describe assessor qualifications/demographics. | Partial | Nottingham grade and the provided ROI bounding box involve subjective reads; assessor qualifications/demographics are not described. |
| 10 | Sample size | D;E | How study size was arrived at, justified as sufficient, incl. any calculation. | Complete | §2.5 gives the pre-registered power calculation (80% power, α=0.05, ΔAUROC=0.05) and reports the minimum detectable effect where N is limiting. |
| 11 | Missing data | D;E | How missing data were handled; reasons for omitting any data. | Partial | §2.4 notes imputers fit on the training fold; missing-data amounts and the omission rationale per cohort are not detailed. |
| 12a | Analytical methods | D | How data were used/partitioned in analysis, considering sample-size requirements. | Complete | §2.5/§2.7 describe patient-level 5-fold CV over three seeds with a quasi-external holdout carved first, across the six-rung ablation ladder. |
| 12b | Analytical methods | D | How predictors were handled (functional form, rescaling, transformation, standardisation). | Complete | §2.4 describes tabular standardisation (scalers on train fold), Nyúl image normalization, and FT-Transformer tabular encoding. |
| 12c | Analytical methods | D | Model type, rationale, all model-building steps incl. hyperparameter tuning and internal validation. | Complete | §2.6/§2.7/§2.9 specify each estimator (LogReg, FT-Transformer, 3D-ResNet-18/MedicalNet, hierarchical cross-attention, grade-band MoE) and internal validation; per-estimator hyperparameter-tuning detail is light (points to frozen configs). |
| 12d | Analytical methods | D;E | Heterogeneity in parameters/performance across clusters (per TRIPOD-Cluster). | Partial | Track B reports leave-one-site-out (0.9679); Track A is single-institution so within-development cluster heterogeneity is not applicable; TRIPOD-Cluster framing is not explicit. |
| 12e | Analytical methods | D;E | All measures/plots (and rationale) to evaluate performance and compare models. | Complete | §2.8/§2.9 specify AUROC+DeLong CI, paired DeLong, ECE, label-shuffle sentinels, saliency box-IoU/pointing-game, and reliability diagrams. |
| 12f | Analytical methods | E | Any model updating (e.g. recalibration) from evaluation. | Complete | §2.9/§2.10/§3.6 describe temperature-scaling recalibration fit on Duke out-of-fold data only. |
| 12g | Analytical methods | E | For evaluation, how predictions were calculated (formula/code/object/API). | Complete | §2.10: the frozen Duke LogReg is applied once to the sealed ISPY2 set; scoring code is in `submission/`. |
| 13 | Class imbalance | D;E | If class-imbalance methods used, why/how, and any recalibration. | Partial | Prevalence is reported (luminal-like 595 / TNBC 164) with stratified CV; explicit imbalance-handling methods (weighting/resampling/focal) and any imbalance recalibration are not detailed. |
| 14 | Fairness | D;E | Approaches used to address model fairness, and their rationale. | **Missing** | No fairness analysis is reported; race/ethnicity, age and menopausal status are inputs, but no subgroup/bias assessment or explicit justification for omitting one is given. |
| 15 | Model output | D | Output of the model (probabilities/classification); any thresholds and how identified. | Complete | Output is a calibrated subtype probability, evaluated threshold-free (AUROC+ECE); no operating classification threshold is set (research-only). |
| 16 | Training vs evaluation | D;E | Differences between development and evaluation data (setting, eligibility, outcome, predictors). | Complete | §2.4/§3.5 identify the shared nine-feature set and the TNBC prevalence difference (21% Duke vs 48% ISPY2); de-duplication in §2.5. |
| 17 | Ethical approval | D;E | Name IRB/ethics committee; describe informed consent or waiver. | **Missing** | No ethics-approval or consent/waiver statement; public de-identified datasets are used but their data-use terms / IRB-exemption are not declared. |

## Open science

| Item | Topic | Applies | Requirement (condensed) | Status | Manuscript location / gap note |
|---|---|---|---|---|---|
| 18a | Funding | D;E | Source of funding and role of funders. | Partial | Open Science notes the fixed compute budget; a formal funding source + funder-role statement is absent. |
| 18b | Conflicts of interest | D;E | Declare conflicts of interest and financial disclosures for all authors. | **Missing** | No conflict-of-interest / financial-disclosure statement for the authors. |
| 18c | Protocol | D;E | Where the study protocol can be accessed, or state none prepared. | Complete | §2.1 + Open Science cite the frozen pre-registration / analysis plan (`docs/pre_registration.md`) as the protocol location. |
| 18d | Registration | D;E | Registration info (register name + number), or state not registered. | Partial | Internal pre-registration is cited, but no public register name/number is given and non-registration is not explicitly stated. |
| 18e | Data sharing | D;E | Details of the availability of the study data. | Partial | Source cohorts are public and named in Methods; a consolidated data-availability statement with access routes (TCIA / GDC / ISPY2) is not given. |
| 18f | Code sharing | D;E | Details of the availability of the analytical code. | Complete | Open Science provides analytical code + a synthetic-data demo (`submission/`) and figure/table regeneration via `scripts/make_paper_figures.py`. |

## Patient & public involvement

| Item | Topic | Applies | Requirement (condensed) | Status | Manuscript location / gap note |
|---|---|---|---|---|---|
| 19 | Patient & public involvement | D;E | Details of any PPI across the study, or state no involvement. | **Missing** | No PPI statement; add an explicit "no patient or public involvement" declaration. |

## Results

| Item | Topic | Applies | Requirement (condensed) | Status | Manuscript location / gap note |
|---|---|---|---|---|---|
| 20a | Participants | D;E | Flow of participants (numbers with/without outcome; follow-up); a diagram may help. | Complete | §2.3 + Fig 2 give the participant flow / N-waterfall with outcome counts. |
| 20b | Participants | D;E | Characteristics overall and per source: key dates, key predictors (demographics), treatments, N, events, follow-up, missing data; group differences. | Partial | Table 1 reports characteristics, sample size and prevalence per cohort; key dates, missing-data amounts and between-demographic-group differences are not fully tabulated. |
| 20c | Participants | E | For evaluation, compare distribution of important predictors vs development data. | Partial | §3.5 compares outcome prevalence (Duke vs ISPY2); a full development-vs-evaluation predictor-distribution comparison is not shown. |
| 21 | Model development | D;E | Number of participants and outcome events in each analysis. | Complete | §2.3/§3.1 give per-analysis participant and TNBC-event counts (Duke N=624, N=613 intersection; ISPY2 N=739; TCGA N=640). |
| 22 | Model specification | D | Full model (formula/code/object/API) enabling new-individual prediction and third-party use; access restrictions. | Partial | Frozen configs/code are shared (`submission/`, synthetic); the deployable clinical-anchor LogReg object/coefficients and reuse restrictions for new-individual prediction are not fully specified. |
| 23a | Model performance | D;E | Performance estimates with CIs, incl. key subgroups; consider plots. | Partial | §3 + Table 2 report every estimate with DeLong CI, ECE, shuffle and multi-seed spread; sociodemographic subgroup performance is not reported. |
| 23b | Model performance | D;E | If examined, results of any heterogeneity across clusters. | Complete | Track B leave-one-site-out heterogeneity is reported (0.9679); Track A is single-institution. |
| 24 | Model updating | E | Results from any model updating, incl. the updated model and performance. | Complete | §3.6 reports recalibration results (internal ECE 0.0196→0.0244; external 0.391→0.373 under temperature scaling). |

## Discussion

| Item | Topic | Applies | Requirement (condensed) | Status | Manuscript location / gap note |
|---|---|---|---|---|---|
| 25 | Interpretation | D;E | Overall interpretation, incl. fairness, in context of objectives and prior studies. | Complete | §4 interprets the characterised imaging-information ceiling and the leakage-safe architecture contribution against prior work (Li'16 as prior work, not a transfer comparison; Tafavvoghi'24 validation gap); fairness is not part of the interpretation. |
| 26 | Limitations | D;E | Limitations (sample, size, overfitting, missing data) and effects on bias, uncertainty, generalizability. | Complete | §4 discusses single-institution scope, demographically skewed corpus, bounding-box ROI, absent Ki-67, single-seed fusion CIs, prevalence-driven external calibration, and the descoped clinician pretest. |
| 27a | Usability in current care | D | How poor-quality/unavailable input data should be assessed/handled at implementation. | **Missing** | Not addressed (research-only, no deployment); state as N/A + future work. |
| 27b | Usability in current care | D | Whether users interact with the input/model, and the expertise required. | **Missing** | Not addressed (research-only, no deployment); state as N/A + future work. |
| 27c | Usability in current care | D;E | Next steps for future research (applicability, generalizability). | Complete | §4 gives next steps: learned H0 segmentation front-end, matched multimodal cohorts pairing imaging with histology, reader studies, and the v2.0 forward-hypothesis. |

---

## Gaps to close before submission (ranked)

**Tier 1 — mandatory reporting statements (reviewers will flag; each is a short paragraph).**
1. **Item 17 — Ethics/data-use.** Add an IRB-exemption + public de-identified data-use-terms statement (Duke/TCIA, ISPY2, TCGA-BRCA). *(Missing)*
2. **Item 18b — Conflicts of interest.** Add a COI / financial-disclosure declaration for both authors. *(Missing)*
3. **Item 18a — Funding.** Add a funding statement (competition/self-funded; funder role in design/analysis). *(Partial)*
4. **Item 19 — Patient & public involvement.** Add an explicit "no PPI" statement. *(Missing)*
5. **Item 18d — Registration.** State the internal pre-registration and either give a public register (e.g. OSF) name/number or state "not registered in a public registry." *(Partial)*
6. **Item 18e — Data availability.** Add a consolidated data-availability statement naming access routes (TCIA / GDC / ISPY2). *(Partial)*

**Tier 2 — TRIPOD+AI fairness cluster (the flagship AI additions; inputs already collected).**
7. **Item 14 — Fairness.** Report a subgroup/fairness assessment or explicitly justify omitting one (race/ethnicity, age, menopausal status are inputs). *(Missing)*
8. **Item 23a — Subgroup performance.** Add per-subgroup AUROC + DeLong CI, or state the analysis is not powered. *(Partial)*
9. **Item 3c — Health inequalities.** Add a sentence on documented sociodemographic breast-cancer disparities. *(Partial)*
10. **Item 7 — QC across groups.** State whether preprocessing/QC was uniform across sociodemographic groups. *(Partial)*

**Tier 3 — reproducibility / reporting completeness.**
11. **Item 5b — Data dates.** Add per-cohort accrual/collection date ranges. *(Missing)*
12. **Item 11 — Missing data.** Report missing-data amounts + imputation per cohort. *(Partial)*
13. **Item 13 — Class imbalance.** State imbalance handling used (stratification / class weights / focal) beyond prevalence reporting. *(Partial)*
14. **Item 22 — Model specification.** Publish the frozen clinical-anchor LogReg coefficients/object + reuse restrictions. *(Partial)*
15. **Items 20b/20c — Table 1 completeness.** Add key dates, missing-data amounts, and a development-vs-evaluation predictor-distribution comparison. *(Partial)*
16. **Item 6a — Centres.** Name centre counts/locations per cohort. *(Partial)*

**Tier 4 — state-N/A clarifications (research-only posture; needed for a fully ticked sheet).**
17. **Items 27a/27b — Usability.** State N/A (research-only, not for clinical use) and defer to future work. *(Missing)*
18. **Items 8b/8c — Outcome assessor / blinding.** State labels are registry/IHC-derived (objective; blinding N/A). *(Partial)*
19. **Item 9c — Predictor interpretation.** Note Nottingham grade + ROI box are subjective reads with assessor qualifications unavailable. *(Partial)*
20. **Item 6c — Treatments.** State Track A uses pre-treatment diagnostic imaging (N/A); note neoadjuvant context for the pCR companion. *(Partial)*
21. **Item 3b — Intended users.** Name intended users or state research-only (no clinical users). *(Partial)*
22. **Item 12d — Cluster heterogeneity.** Note Track A is single-institution; leave-one-site-out covers Track B. *(Partial)*
23. **Item 2 — Abstract.** Complete the separate *TRIPOD+AI for Abstracts* checklist as its own supplement. *(Partial)*

*Checklist reproduced for author completion under its intended use. Every mapping traces to
`reports/paper/pinksight_paper.md`; verify wording against the official source before submission.*

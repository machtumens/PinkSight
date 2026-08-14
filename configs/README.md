# configs/

Hydra/OmegaConf configs. **No magic numbers in code — everything that defines a run lives here**
(manual §11, decisions.md LAW).

## The frozen split (do NOT regenerate)

`split_v2.yaml` will hold the **patient-level, quasi-external** train/val/test split, frozen once at
G0/G1 and **never regenerated** — regenerating it silently re-randomises the holdout and breaks every
downstream "external" number (manual §9 / LOCK-2).

It does **not exist yet**: the G0 audit (`scripts/audit_ki67.py`) produces the N-waterfall first,
then the split is carved (quasi-external scanner/year slice FIRST) and frozen here. Until then this
directory is intentionally a placeholder — no fabricated split, no invented N.

## Layout (grows per-gate)

```
configs/
  split_v2.yaml        # frozen patient-level split            (G1)
  data/                # cohort, modality, ROI                 (G1)
  model/               # encoder, fusion, heads                (G2)
  train/<run>.yaml     # one run = one reproducible config     (G2+)
```

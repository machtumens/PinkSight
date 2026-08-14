# PinkSight Desktop — Characterisation Workstation

Native cross-platform desktop scaffold for the PinkSight explainable multimodal
model. **Research / OPSI 2026. NOT a clinical product.** Every screen enforces the
claim ledger: characterisation & localisation only — never early detection, never
growth-rate, Ki-67 as a diagnosis-time descriptor. <!-- # allow-ledger -->

Stack: **Tauri v2 (Rust shell) · React + TypeScript · Tailwind + shadcn/ui · Python
(FastAPI) inference sidecar.** Packages to `.exe/.msi/.app/.dmg/.deb/.rpm/.AppImage`.

---

## Run it

```bash
# 1. Frontend deps
npm install

# 2. Inference sidecar (separate terminal) — PyTorch/MONAI lives here
pip install -r sidecar/requirements.txt
npm run sidecar            # serves 127.0.0.1:8756  (UI degrades to mock if offline)

# 3a. Browser dev (fastest to see UI)
npm run dev                # http://localhost:1420

# 3b. Native window (needs Rust toolchain + app icons — see below)
npm run tauri dev
```

**Before `tauri dev/build`:** generate icons once —
`npm run tauri icon path/to/logo.png` (Tauri refuses to build without them).

Verify the ledger guard: `python3 sidecar/test_sidecar.py`.

---

## Architecture — the UI embodies the pipeline

```
DICOM in ─▶ [sidecar: segment → 3D-CNN MRI enc + FT-Transformer clinical
                     → cross-attention fusion → subtype + grade heads → XAI]
         ◀─ CharacterisationReport (calibrated + uncertainty + provenance + audit)
UI (Tauri/React) renders it — never a bare number.
```

- **`src/lib/inference.ts`** — the only seam to the model. Swap the sidecar, UI unchanged.
- **`src/components/UncertaintyBar.tsx`** — the trust primitive; no output without its band/abstention.
- **`sidecar/main.py`** — `build_report()` is mocked; replace with a real forward pass. `assert_ledger_safe()` blocks forbidden framing at the source.

## What's real vs. mocked (ponytail — honest ceilings)

| Real | Mocked / stubbed |
|---|---|
| Tauri v2 shell, routing, token layer, async job lifecycle, sidecar client + fallback, ledger guard + test | MRI viewport (placeholder — mount Cornerstone3D/VTK.js), the model itself (fixed report), report signing/export, sidecar auto-spawn, Track B (path/genomics) shown as honest empty slots |

## Phases 2–5, condensed

- **Vision** — a workstation that makes honesty legible: calibration, uncertainty, and provenance are foreground, not footnotes. "A trustworthy 0.80 beats a leaky 0.92."
- **Workflow** — study-centric: Patients ▶ Study Viewer ▶ Run characterisation ▶ Reliability ▶ signed Report ▶ Audit. Every click has a purpose.
- **IA** — Dashboard · Patients · Study/MRI Viewer (fusion + XAI + uncertainty) · Report · Audit · Settings. Experiments/Research-mode/Dev-tools are future slots.
- **Design system** — `src/index.css` is the source of truth: clinical calm, one rose signal color, semantic-only color (uncertainty/abstain/calibration/modality-provenance), tabular numerals, dark-first.

## Next steps (not built — YAGNI until needed)

1. Mount a real DICOM viewer in `StudyViewer` (Cornerstone3D).
2. Wire the G2 encoder into `sidecar/build_report`.
3. Bundle the sidecar as a Tauri sidecar binary + auto-spawn in `main.rs`.
4. `npx shadcn add dialog table` etc. — `components.json` is already configured.

/*
  Inference seam. The UI never imports a model — it talks to a local Python
  sidecar (FastAPI on 127.0.0.1) that owns PyTorch/MONAI. In a packaged build,
  Tauri spawns the sidecar as a bundled binary; in dev you run `npm run sidecar`.

  Every field here is characterisation, never detection. See sidecar/main.py for
  the enforced FORBIDDEN_TERMS guard.
*/

const SIDECAR = "http://127.0.0.1:8756";

export type JobState = "queued" | "running" | "done" | "failed";

export type ModalityContribution = {
  modality: "mri" | "clinical" | "path" | "genomic";
  present: boolean;      // false => greyed slot (modality dropout / Track B future)
  contribution: number;  // signed Shapley-style value; 0 when absent
};

export type SubtypeCharacterisation = {
  label: "Luminal A" | "Triple-Negative";
  probability: number;        // calibrated
  uncertainty: [number, number]; // 90% band
  abstained: boolean;         // model declined below confidence floor
};

// Non-reportable provenance stamp. `datasetTag` lets a reader always tell a real synthetic-harness
// payload ("SYNTHETIC — NOT A RESULT") apart from a hardcoded MOCK UI fallback — never a bare number.
export type Provenance = {
  datasetTag: string;
  manifestSha256?: string;
  seed?: number;
  gitCommit?: string;
  generatedAt?: string;
  cohortStream?: string;   // "negative_control" | "positive_control" | "mock"
  nPatients?: number;
};

// Ki-67 stratum at diagnosis (aggressiveness snapshot) — characterisation companion only.
export type Ki67Stratum = "high" | "low" | "not_assessed";

// Explainable-fusion saliency summary (reported-not-gated on a random-init encoder).
export type XaiBlock = {
  mapRef?: string;
  iou?: number | null;
  pointingGame?: boolean | null;
  randomizationPassed?: boolean | null;
  note?: string;
};

// Control-sentinel verdict — evidence the pipeline wiring is not silently dead (synthetic plumbing).
export type ControlVerdict = {
  stream?: string;
  auroc?: number;
  delongCi95?: [number, number];
  shuffleAuroc?: number;
  shuffleAtChance?: boolean;
  expected?: string;
  verdict?: string;   // "PASS" | "FAIL" | "NOT_ASSESSED"
  note?: string;
};

// Dispatcher routing decision — a pure (cohort, modalities) -> harness lookup from
// scripts/pinksight_dispatch.py. A routing decision ONLY, never a per-organ prediction number.
// `crossCohortGradient` is always false (inference-time routing, no >1-cohort gradient).
export type DispatchBlock = {
  cohort: string;
  modalities: string[];
  harnessScript: string | null;
  status: string;
  crossCohortGradient: boolean;
  note: string;
};

export type CharacterisationReport = {
  studyId: string;
  subtype: SubtypeCharacterisation;
  ki67Descriptor: string;     // descriptive companion, NOT a growth-rate claim
  ki67Stratum?: Ki67Stratum;  // optional at-diagnosis stratum companion to the descriptor
  nottinghamGrade: { label: "NHG1" | "NHG2" | "NHG3"; probability: number; uncertainty: [number, number] };
  calibration: { ece: number; band: "good" | "acceptable" | "poor" };
  modalities: ModalityContribution[];
  provenance?: Provenance;          // optional provenance stamp (present on synthetic-harness payloads)
  xai?: XaiBlock;                   // optional explainable-fusion saliency summary
  controlVerdict?: ControlVerdict;  // optional control-sentinel verdict (synthetic plumbing runs)
  dispatch?: DispatchBlock;         // optional routing decision (present when cohort+modalities sent)
  audit: { modelHash: string; seed: number; split: string; generatedAt: string };
};

/** Fallback so the workstation is demoable with the sidecar offline. */
export const MOCK_REPORT: CharacterisationReport = {
  studyId: "DUKE-0421",
  subtype: {
    label: "Luminal A",
    probability: 0.71,
    uncertainty: [0.58, 0.83],
    abstained: false,
  },
  ki67Descriptor: "Descriptive companion — imaging correlate of Ki-67 index at diagnosis (not kinetics).",
  ki67Stratum: "not_assessed",
  nottinghamGrade: { label: "NHG1", probability: 0.68, uncertainty: [0.55, 0.80] },
  calibration: { ece: 0.041, band: "good" },
  modalities: [
    { modality: "clinical", present: true, contribution: 0.135 },
    { modality: "mri", present: true, contribution: -0.025 },
    { modality: "path", present: false, contribution: 0 },
    { modality: "genomic", present: false, contribution: 0 },
  ],
  // Distinct MOCK tag (NOT the harness's "SYNTHETIC — NOT A RESULT") so a reader can always tell a
  // hardcoded UI fallback apart from a real synthetic-harness payload.
  provenance: {
    datasetTag: "MOCK — NOT SYNTHETIC, NOT A RESULT",
    seed: 42,
    generatedAt: "mocked",
    cohortStream: "mock",
    nPatients: 1,
  },
  xai: {
    mapRef: "",
    iou: null,
    pointingGame: null,
    randomizationPassed: null,
    note: "mock fallback — no saliency computed offline",
  },
  controlVerdict: {
    stream: "mock",
    verdict: "NOT_ASSESSED",
    expected: "hardcoded UI fallback — no control sentinel run",
    note: "offline MOCK, not a synthetic-harness payload",
  },
  audit: {
    modelHash: "sha256:demo0000",
    seed: 42,
    split: "split_v2 (scanner holdout)",
    generatedAt: "mocked",
  },
};

export type InferenceResult =
  | { state: "done"; report: CharacterisationReport }
  | { state: "failed"; error: string };

/**
 * Run inference against the sidecar. Falls back to MOCK_REPORT when the sidecar
 * is unreachable so the UI still demos — but flags it as mocked in the audit.
 *
 * `opts` is optional and backward compatible: when omitted, the POST body is byte-identical to the
 * original `{ studyId }`. Only defined optional fields are spread onto the body. `live` narrows the
 * server-side `PINKSIGHT_SIDECAR_LIVE` gate (it can never widen it — see sidecar/main.py).
 */
export async function runInference(
  studyId: string,
  opts?: { live?: boolean; cohort?: string; modalities?: string[] }
): Promise<InferenceResult> {
  try {
    const body = {
      studyId,
      ...(opts?.live !== undefined && { live: opts.live }),
      ...(opts?.cohort !== undefined && { cohort: opts.cohort }),
      ...(opts?.modalities !== undefined && { modalities: opts.modalities }),
    };
    const res = await fetch(`${SIDECAR}/infer`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) return { state: "failed", error: `sidecar ${res.status}` };
    const report = (await res.json()) as CharacterisationReport;
    return { state: "done", report };
  } catch {
    // Sidecar offline: degrade to mock, but never silently — audit says "mocked".
    return { state: "done", report: { ...MOCK_REPORT, studyId } };
  }
}

/**
 * Fetch the dispatcher's routing decision for a (cohort, modalities) selection. This is a routing
 * decision ONLY — never a per-organ prediction number. There is deliberately NO mock fallback: a
 * fabricated WIRED/NOT-WIRED verdict while offline would violate the dispatcher's "never silently
 * guessed" contract, so an unreachable sidecar surfaces `{ state: "failed" }` and the picker renders
 * that honestly instead.
 */
export async function fetchDispatch(
  cohort: string,
  modalities: string[]
): Promise<{ state: "done"; dispatch: DispatchBlock } | { state: "failed"; error: string }> {
  try {
    const res = await fetch(
      `${SIDECAR}/dispatch?cohort=${encodeURIComponent(cohort)}&modalities=${encodeURIComponent(
        modalities.join(",")
      )}`
    );
    if (!res.ok) return { state: "failed", error: `sidecar ${res.status}` };
    const dispatch = (await res.json()) as DispatchBlock;
    return { state: "done", dispatch };
  } catch {
    return { state: "failed", error: "sidecar unreachable" };
  }
}

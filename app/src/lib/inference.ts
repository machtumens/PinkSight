const SIDECAR = "http://127.0.0.1:8756";

export type JobState = "queued" | "running" | "done" | "failed";

export type ModalityContribution = {
  modality: "mri" | "clinical" | "path" | "genomic";
  present: boolean;
  contribution: number;
};

export type SubtypeCharacterisation = {
  label: "Luminal A" | "Triple-Negative";
  probability: number;
  uncertainty: [number, number];
  abstained: boolean;
};

export type Provenance = {
  datasetTag: string;
  manifestSha256?: string;
  seed?: number;
  gitCommit?: string;
  generatedAt?: string;
  cohortStream?: string;
  nPatients?: number;
};

export type Ki67Stratum = "high" | "low" | "not_assessed";

export type XaiBlock = {
  mapRef?: string;
  iou?: number | null;
  pointingGame?: boolean | null;
  randomizationPassed?: boolean | null;
  note?: string;
};

export type ControlVerdict = {
  stream?: string;
  auroc?: number;
  delongCi95?: [number, number];
  shuffleAuroc?: number;
  shuffleAtChance?: boolean;
  expected?: string;
  verdict?: string;
  note?: string;
};

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
  ki67Descriptor: string;
  ki67Stratum?: Ki67Stratum;
  nottinghamGrade: { label: "NHG1" | "NHG2" | "NHG3"; probability: number; uncertainty: [number, number] };
  calibration: { ece: number; band: "good" | "acceptable" | "poor" };
  modalities: ModalityContribution[];
  provenance?: Provenance;
  xai?: XaiBlock;
  controlVerdict?: ControlVerdict;
  dispatch?: DispatchBlock;
  audit: { modelHash: string; seed: number; split: string; generatedAt: string };
};

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
    return { state: "done", report: { ...MOCK_REPORT, studyId } };
  }
}

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

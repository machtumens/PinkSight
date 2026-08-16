import { useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { Play, Layers, FileText, Loader2 } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { UncertaintyBar } from "@/components/UncertaintyBar";
import { ModalityBars } from "@/components/ModalityBars";
import { DispatchPicker } from "@/components/DispatchPicker";
import { MriPhantom } from "@/components/MriPhantom";
import { useInferenceMode } from "@/lib/inference-mode-context";
import {
  runInference,
  MOCK_REPORT,
  type CharacterisationReport,
  type JobState,
} from "@/lib/inference";

const COHORT_DEFAULTS: Record<string, string[]> = {
  duke: ["mri", "clinical"],
  tcga_brca: ["wsi", "genomics"],
  metabric: ["clinical", "genomics"],
  cptac_brca: ["proteomic", "genomics"],
  cdd_cesm: ["cesm", "ffdm"],
  cmmd: ["ffdm"],
  fastmri_nyu: ["dce"],
  track_c: ["tabular", "recurrence"],
};

export default function StudyViewer() {
  const { id = "DUKE-0421" } = useParams();
  const nav = useNavigate();
  const { mode } = useInferenceMode();
  const [phase, setPhase] = useState(2);
  const [xai, setXai] = useState(true);
  const [job, setJob] = useState<JobState>("queued");
  const [report, setReport] = useState<CharacterisationReport | null>(null);
  const [cohort, setCohort] = useState<string>("");
  const [modalities, setModalities] = useState<string[]>([]);

  async function characterise() {
    setJob("running");
    const res = await runInference(id, {
      live: mode === "live",
      ...(cohort && { cohort, modalities }),
    });
    if (res.state === "failed") { setJob("failed"); return; }
    setReport(res.report);
    setJob("done");
  }

  function handleCohort(next: string) {
    setCohort(next);
    if (next && modalities.length === 0 && COHORT_DEFAULTS[next]) setModalities(COHORT_DEFAULTS[next]);
  }

  return (
    <div className="grid h-full grid-cols-[1fr_380px]">
      <section className="flex flex-col bg-black/90">
        <div className="flex items-center justify-between border-b border-border/40 px-4 py-2 text-sm text-white/80">
          <span className="tnum">{id} · DCE-MRI · phase {phase}/5</span>
          <Button
            size="sm"
            variant={xai ? "default" : "outline"}
            onClick={() => setXai((v) => !v)}
          >
            <Layers /> XAI overlay
          </Button>
        </div>

        <div className="relative flex flex-1 items-center justify-center p-6">
          <MriPhantom phase={phase} xai={xai} />
          <span className="absolute bottom-4 text-xs text-white/40">
            synthetic phantom — not patient data
          </span>
        </div>

        <div className="border-t border-border/40 px-4 py-3">
          <input
            type="range" min={0} max={5} value={phase}
            onChange={(e) => setPhase(Number(e.target.value))}
            className="w-full accent-primary"
          />
        </div>
      </section>

      <section className="flex flex-col gap-4 overflow-y-auto border-l border-border p-4">
        <DispatchPicker
          cohort={cohort}
          modalities={modalities}
          onCohortChange={handleCohort}
          onModalitiesChange={setModalities}
        />
        {job !== "done" && (
          <Button onClick={characterise} disabled={job === "running"}>
            {job === "running" ? <Loader2 className="animate-spin" /> : <Play />}
            {job === "running" ? "Characterising…" : "Run characterisation"}
          </Button>
        )}
        {job === "failed" && (
          <Badge tone="bad">sidecar unreachable — start it with `npm run sidecar`</Badge>
        )}

        {job === "done" && report && (
          <>
            <Badge tone={report.provenance?.datasetTag?.startsWith("SYNTHETIC") ? "warn" : "muted"}>
              {report.provenance?.datasetTag ?? "MOCK — NOT SYNTHETIC, NOT A RESULT"}
            </Badge>
            <Card>
              <CardHeader><CardTitle>Subtype characterisation</CardTitle></CardHeader>
              <CardContent>
                <UncertaintyBar
                  label={report.subtype.label}
                  probability={report.subtype.probability}
                  band={report.subtype.uncertainty}
                  abstained={report.subtype.abstained}
                />
                <UncertaintyBar
                  label={`Nottingham ${report.nottinghamGrade.label}`}
                  probability={report.nottinghamGrade.probability}
                  band={report.nottinghamGrade.uncertainty}
                />
              </CardContent>
            </Card>

            <Card>
              <CardHeader><CardTitle>Cross-attention fusion</CardTitle></CardHeader>
              <CardContent className="space-y-2">
                <ModalityBars modalities={report.modalities} />
                <p className="pt-1 text-xs text-muted-foreground">
                  Modality-dropout gating: absent streams contribute nothing, shown honestly.
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader><CardTitle>Reliability</CardTitle></CardHeader>
              <CardContent className="flex items-center justify-between text-sm">
                <span>Calibration (ECE)</span>
                <Badge tone={report.calibration.band === "good" ? "good" : report.calibration.band === "acceptable" ? "warn" : "bad"}>
                  {report.calibration.ece.toFixed(3)} · {report.calibration.band}
                </Badge>
              </CardContent>
            </Card>

            <Button variant="outline" onClick={() => nav(`/report/${id}`, { state: { report } })}>
              <FileText /> Open signed report
            </Button>
          </>
        )}

        <p className="mt-auto text-[10px] text-muted-foreground">
          Characterisation &amp; localisation only. Not early detection. Not growth rate. Ki-67 shown as a
          diagnosis-time descriptor, never kinetics. Model hash {MOCK_REPORT.audit.modelHash}.
        </p>
      </section>
    </div>
  );
}

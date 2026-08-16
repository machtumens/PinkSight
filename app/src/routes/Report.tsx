import { useParams, useNavigate, useLocation } from "react-router-dom";
import { ArrowLeft, ShieldCheck } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { UncertaintyBar } from "@/components/UncertaintyBar";
import { ModalityBars } from "@/components/ModalityBars";
import { MOCK_REPORT, type CharacterisationReport } from "@/lib/inference";

export default function Report() {
  const { id = "DUKE-0421" } = useParams();
  const nav = useNavigate();
  const location = useLocation();
  const r = (location.state as { report?: CharacterisationReport } | null)?.report ?? MOCK_REPORT;
  const synthetic = r.provenance?.datasetTag?.startsWith("SYNTHETIC") ?? false;

  return (
    <div className="mx-auto max-w-2xl p-6">
      <Button variant="ghost" size="sm" onClick={() => nav(-1)} className="mb-4">
        <ArrowLeft /> Back to viewer
      </Button>

      <div className="mb-3 flex items-baseline justify-between">
        <div>
          <h1 className="text-lg font-semibold">Characterisation Report</h1>
          <p className="text-xs text-muted-foreground">
            Subtype &amp; grade at diagnosis · explainable fusion
          </p>
        </div>
        <span className="tnum text-sm text-muted-foreground">{id}</span>
      </div>

      <div className="mb-4">
        <Badge tone={synthetic ? "warn" : "muted"}>
          {r.provenance?.datasetTag ?? "MOCK — NOT SYNTHETIC, NOT A RESULT"}
        </Badge>
      </div>

      <div className="space-y-4">
        <Card>
          <CardHeader><CardTitle>Characterisation</CardTitle></CardHeader>
          <CardContent>
            <UncertaintyBar
              label={r.subtype.label}
              probability={r.subtype.probability}
              band={r.subtype.uncertainty}
              abstained={r.subtype.abstained}
            />
            <UncertaintyBar
              label={`Nottingham ${r.nottinghamGrade.label}`}
              probability={r.nottinghamGrade.probability}
              band={r.nottinghamGrade.uncertainty}
            />
            <p className="pt-2 text-xs text-muted-foreground">
              <span className="font-medium text-foreground">Ki-67:</span> {r.ki67Descriptor}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader><CardTitle>Cross-attention fusion</CardTitle></CardHeader>
          <CardContent className="space-y-2">
            <ModalityBars modalities={r.modalities} />
            <p className="pt-1 text-xs text-muted-foreground">
              Modality-dropout gating: absent streams contribute nothing, shown honestly.
            </p>
          </CardContent>
        </Card>

        {r.xai && (
          <Card>
            <CardHeader><CardTitle>Explainability (XAI)</CardTitle></CardHeader>
            <CardContent className="space-y-2 text-sm">
              <XaiRow k="Saliency IoU" v={r.xai.iou} fmt={(n) => n.toFixed(3)} good={(n) => n >= 0.3} />
              <XaiRow k="Pointing game" v={r.xai.pointingGame} />
              <XaiRow k="Randomization sanity" v={r.xai.randomizationPassed} />
              {r.xai.note && <p className="pt-1 text-xs text-muted-foreground">{r.xai.note}</p>}
              <p className="text-xs text-muted-foreground">
                XAI is reported, not gated, until a trained encoder exists.
              </p>
            </CardContent>
          </Card>
        )}

        <Card>
          <CardHeader><CardTitle>Reliability</CardTitle></CardHeader>
          <CardContent>
            <CalibrationMeter ece={r.calibration.ece} band={r.calibration.band} />
          </CardContent>
        </Card>

        {r.dispatch && (
          <Card>
            <CardHeader><CardTitle>Routing</CardTitle></CardHeader>
            <CardContent className="space-y-1 text-xs text-muted-foreground">
              <Row k="Harness">{r.dispatch.harnessScript ?? "—"}</Row>
              <Row k="Status">{r.dispatch.status}</Row>
              <div className="tnum">cross_cohort_gradient: false</div>
              <p>{r.dispatch.note}</p>
              <p>No live prediction available for this organ yet — routing decision only.</p>
            </CardContent>
          </Card>
        )}

        <Card>
          <CardHeader><CardTitle>Reproducibility</CardTitle></CardHeader>
          <CardContent className="space-y-1 text-xs tnum text-muted-foreground">
            <Row k="Model hash">{r.audit.modelHash}</Row>
            <Row k="Seed">{r.audit.seed}</Row>
            <Row k="Split">{r.audit.split}</Row>
          </CardContent>
        </Card>

        <p className="rounded-md bg-muted p-3 text-xs text-muted-foreground">
          Research output (OPSI 2026). Characterisation &amp; localisation of subtype and grade at
          diagnosis. Not a diagnostic device. Not early detection.
        </p>

        <Button className="w-full" disabled>
          <ShieldCheck /> Sign &amp; export (PDF / DICOM-SR) — stub
        </Button>
      </div>
    </div>
  );
}

function Row({ k, children }: { k: string; children: React.ReactNode }) {
  return (
    <div className="flex justify-between gap-4">
      <dt className="shrink-0 text-muted-foreground">{k}</dt>
      <dd className="text-right">{children}</dd>
    </div>
  );
}

function XaiRow({
  k,
  v,
  fmt,
  good,
}: {
  k: string;
  v: number | boolean | null | undefined;
  fmt?: (n: number) => string;
  good?: (n: number) => boolean;
}) {
  let content: React.ReactNode;
  if (v === null || v === undefined) {
    content = <Badge tone="muted">n/a</Badge>;
  } else if (typeof v === "boolean") {
    content = <Badge tone={v ? "good" : "bad"}>{v ? "pass" : "fail"}</Badge>;
  } else {
    content = <Badge tone={good ? (good(v) ? "good" : "warn") : "muted"}>{fmt ? fmt(v) : String(v)}</Badge>;
  }
  return (
    <div className="flex items-center justify-between">
      <span className="text-muted-foreground">{k}</span>
      {content}
    </div>
  );
}

function CalibrationMeter({ ece, band }: { ece: number; band: "good" | "acceptable" | "poor" }) {
  const SCALE = 0.15;
  const pos = Math.min(1, ece / SCALE);
  const tone = band === "good" ? "bg-good" : band === "acceptable" ? "bg-warn" : "bg-bad";
  return (
    <div>
      <div className="mb-1 flex items-center justify-between text-sm">
        <span>Calibration (ECE)</span>
        <span className="tnum font-semibold">
          {ece.toFixed(3)} <span className="font-normal text-muted-foreground">· {band}</span>
        </span>
      </div>
      <div className="relative h-2 w-full rounded-full bg-muted">
        <div className={`absolute inset-y-0 left-0 rounded-full ${tone}`} style={{ width: `${pos * 100}%` }} />
        <div className="absolute inset-y-0 w-px bg-foreground/40" style={{ left: `${(0.05 / SCALE) * 100}%` }} />
        <div className="absolute inset-y-0 w-px bg-foreground/40" style={{ left: `${(0.1 / SCALE) * 100}%` }} />
      </div>
      <div className="mt-1 flex justify-between text-[10px] text-muted-foreground">
        <span>0</span>
        <span>good ≤.05</span>
        <span>acc ≤.10</span>
        <span>.15</span>
      </div>
    </div>
  );
}

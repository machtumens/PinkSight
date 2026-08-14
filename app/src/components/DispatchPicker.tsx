import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Select } from "@/components/ui/select";
import { Checkbox } from "@/components/ui/checkbox";
import { fetchDispatch, type DispatchBlock } from "@/lib/inference";

// The 8 cohorts + 12 modalities appearing across COHORT_HARNESS_REGISTRY / NOT_WIRED_COMBOS in
// scripts/pinksight_dispatch.py — surfaced verbatim as the registry keys (technical abbreviations).
const COHORTS = [
  "duke", "tcga_brca", "metabric", "cptac_brca", "cdd_cesm", "cmmd", "fastmri_nyu", "track_c",
] as const;
const MODALITIES = [
  "mri", "clinical", "wsi", "genomics", "purity", "hrd", "proteomic", "cesm", "ffdm", "dce",
  "tabular", "recurrence",
] as const;

type Props = {
  cohort: string;
  modalities: string[];
  onCohortChange: (cohort: string) => void;
  onModalitiesChange: (modalities: string[]) => void;
};

type RoutingState =
  | { state: "idle" }
  | { state: "loading" }
  | { state: "done"; dispatch: DispatchBlock }
  | { state: "failed"; error: string };

// Routing-decision-only picker. It shows which research harness (organ) owns a (cohort, modalities)
// selection and whether it is WIRED — and NOTHING that looks like a per-organ prediction. There is
// NO run affordance in this component, ever: zero of the wired organs has a single-patient forward
// path today (the honest ceiling), so a "run" here would be a lie. The existing generic synthetic
// "Run characterisation" button lives in StudyViewer and is independent of this selection.
export function DispatchPicker({ cohort, modalities, onCohortChange, onModalitiesChange }: Props) {
  const [routing, setRouting] = useState<RoutingState>({ state: "idle" });

  function toggleModality(m: string, on: boolean) {
    onModalitiesChange(on ? [...modalities, m] : modalities.filter((x) => x !== m));
  }

  async function checkRouting() {
    setRouting({ state: "loading" });
    const res = await fetchDispatch(cohort, modalities);
    if (res.state === "failed") setRouting({ state: "failed", error: res.error });
    else setRouting({ state: "done", dispatch: res.dispatch });
  }

  return (
    <Card>
      <CardHeader><CardTitle>Cohort / modality routing</CardTitle></CardHeader>
      <CardContent className="space-y-3">
        <div className="flex flex-col gap-1">
          <label className="text-xs text-muted-foreground">Cohort</label>
          <Select value={cohort} onChange={(e) => onCohortChange(e.target.value)}>
            <option value="">— select cohort —</option>
            {COHORTS.map((c) => (
              <option key={c} value={c}>{c}</option>
            ))}
          </Select>
        </div>

        <div className="flex flex-col gap-1">
          <label className="text-xs text-muted-foreground">Modalities</label>
          <div className="grid grid-cols-3 gap-1.5">
            {MODALITIES.map((m) => (
              <label key={m} className="flex items-center gap-1.5 text-xs">
                <Checkbox
                  checked={modalities.includes(m)}
                  onChange={(e) => toggleModality(m, e.target.checked)}
                />
                {m}
              </label>
            ))}
          </div>
        </div>

        <Button
          variant="outline"
          size="sm"
          onClick={checkRouting}
          disabled={!cohort || modalities.length === 0 || routing.state === "loading"}
        >
          {routing.state === "loading" ? "Checking…" : "Check routing"}
        </Button>

        {routing.state === "failed" && <Badge tone="bad">sidecar unreachable</Badge>}

        {routing.state === "done" && (
          <div className="space-y-1.5 rounded-md border border-border p-3 text-sm">
            <div className="flex items-center justify-between gap-2">
              <span className="text-muted-foreground">Harness</span>
              <span className="tnum text-right">{routing.dispatch.harnessScript ?? "—"}</span>
            </div>
            <div className="flex items-center justify-between gap-2">
              <span className="text-muted-foreground">Status</span>
              <Badge tone={routing.dispatch.status === "WIRED" ? "good" : "warn"}>
                {routing.dispatch.status}
              </Badge>
            </div>
            {/* surfaced verbatim, never summarized away — the constitutional routing-only contract */}
            <div className="tnum text-xs text-muted-foreground">cross_cohort_gradient: false</div>
            <p className="text-xs text-muted-foreground">{routing.dispatch.note}</p>
            <p className="text-xs text-muted-foreground">
              No live prediction available for this organ yet — routing decision only.
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

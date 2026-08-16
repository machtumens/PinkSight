import type { ModalityContribution } from "@/lib/inference";

const MOD_COLOR: Record<string, string> = {
  mri: "bg-mod-mri",
  clinical: "bg-mod-clinical",
  path: "bg-mod-path",
  genomic: "bg-mod-genomic",
};

export function ModalityBars({ modalities }: { modalities: ModalityContribution[] }) {
  const max = Math.max(
    0.05,
    ...modalities.filter((m) => m.present).map((m) => Math.abs(m.contribution)),
  );

  return (
    <div className="space-y-2">
      {modalities.map((m) => {
        const frac = m.present ? Math.abs(m.contribution) / max : 0;
        const pos = m.contribution >= 0;
        const width = `${(frac * 50).toFixed(1)}%`;
        return (
          <div key={m.modality} className="flex items-center gap-2 text-sm">
            <span
              className={`size-2.5 shrink-0 rounded-full ${MOD_COLOR[m.modality] ?? "bg-muted"} ${
                m.present ? "" : "opacity-25"
              }`}
            />
            <span className={`w-16 shrink-0 capitalize ${m.present ? "" : "text-muted-foreground"}`}>
              {m.modality}
            </span>
            <div className="relative h-3 flex-1 rounded bg-muted/60">
              <div className="absolute inset-y-0 left-1/2 w-px bg-border" />
              {m.present && (
                <div
                  className={`absolute inset-y-0 rounded transition-[width] duration-500 ${
                    pos ? "bg-good" : "bg-bad"
                  }`}
                  style={pos ? { left: "50%", width } : { right: "50%", width }}
                />
              )}
            </div>
            <span
              className={`tnum w-14 shrink-0 text-right ${
                m.present ? "font-medium" : "text-xs text-muted-foreground"
              }`}
            >
              {m.present ? `${pos ? "+" : ""}${m.contribution.toFixed(3)}` : "absent"}
            </span>
          </div>
        );
      })}
    </div>
  );
}

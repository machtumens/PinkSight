import { Badge } from "@/components/ui/badge";

export function UncertaintyBar({
  label,
  probability,
  band,
  abstained = false,
}: {
  label: string;
  probability: number;
  band: [number, number];
  abstained?: boolean;
}) {
  const pct = (n: number) => `${(n * 100).toFixed(0)}%`;
  const [lo, hi] = band;

  if (abstained) {
    return (
      <div className="flex items-center justify-between gap-3 py-1.5">
        <span className="text-sm">{label}</span>
        <Badge tone="abstain">model declines — below confidence floor</Badge>
      </div>
    );
  }

  return (
    <div className="py-1.5">
      <div className="mb-1 flex items-center justify-between text-sm">
        <span>{label}</span>
        <span className="tnum font-semibold">
          {pct(probability)}{" "}
          <span className="text-muted-foreground font-normal">
            [{pct(lo)}–{pct(hi)}]
          </span>
        </span>
      </div>
      <div className="relative h-2 w-full rounded-full bg-muted">
        <div
          className="absolute h-2 rounded-full bg-uncertain/40"
          style={{ left: `${lo * 100}%`, width: `${(hi - lo) * 100}%` }}
        />
        <div
          className="absolute top-1/2 h-3 w-[2px] -translate-y-1/2 rounded bg-foreground"
          style={{ left: `${probability * 100}%` }}
        />
      </div>
    </div>
  );
}

import { Badge } from "@/components/ui/badge";

// Every inference is logged: model hash, seed, split, provenance. Integrity as a feature.
const LOG = [
  { t: "2026-07-13 15:42", study: "DUKE-0421", event: "characterise", hash: "sha256:demo0000", by: "researcher" },
  { t: "2026-07-13 15:41", study: "DUKE-0421", event: "import DICOM", hash: "—", by: "researcher" },
  { t: "2026-07-13 15:30", study: "DUKE-0588", event: "characterise", hash: "sha256:demo0000", by: "researcher" },
];

export default function Audit() {
  return (
    <div className="p-6">
      <header className="mb-4">
        <h1 className="text-xl font-semibold">Audit Trail</h1>
        <p className="text-sm text-muted-foreground">Append-only. Reproducible by construction.</p>
      </header>
      <div className="overflow-hidden rounded-lg border border-border">
        <table className="w-full text-sm">
          <thead className="bg-muted text-muted-foreground">
            <tr className="[&>th]:px-4 [&>th]:py-2 [&>th]:text-left [&>th]:font-medium">
              <th>Time</th><th>Study</th><th>Event</th><th>Model hash</th><th>User</th>
            </tr>
          </thead>
          <tbody>
            {LOG.map((l, i) => (
              <tr key={i} className="border-t border-border [&>td]:px-4 [&>td]:py-2.5">
                <td className="tnum text-muted-foreground">{l.t}</td>
                <td className="tnum">{l.study}</td>
                <td><Badge tone="muted">{l.event}</Badge></td>
                <td className="tnum text-muted-foreground">{l.hash}</td>
                <td>{l.by}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

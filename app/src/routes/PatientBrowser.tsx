import { useNavigate } from "react-router-dom";
import { Badge } from "@/components/ui/badge";

const STUDIES = [
  { id: "DUKE-0421", patient: "P-0421", scanner: "GE 1.5T", year: 2019, series: "DCE 6-phase", status: "ready" },
  { id: "DUKE-0588", patient: "P-0588", scanner: "Siemens 3T", year: 2021, series: "DCE 5-phase", status: "ready" },
  { id: "DUKE-0733", patient: "P-0733", scanner: "GE 3T", year: 2020, series: "DCE 6-phase", status: "held-out" },
];

export default function PatientBrowser() {
  const nav = useNavigate();
  return (
    <div className="p-6">
      <header className="mb-4">
        <h1 className="text-xl font-semibold">Patients</h1>
        <p className="text-sm text-muted-foreground">Double-click a study to open the viewer.</p>
      </header>

      <div className="overflow-hidden rounded-lg border border-border">
        <table className="w-full text-sm">
          <thead className="bg-muted text-muted-foreground">
            <tr className="[&>th]:px-4 [&>th]:py-2 [&>th]:text-left [&>th]:font-medium">
              <th>Study</th><th>Patient</th><th>Scanner</th><th>Year</th><th>Series</th><th>Split</th>
            </tr>
          </thead>
          <tbody>
            {STUDIES.map((s) => (
              <tr
                key={s.id}
                onDoubleClick={() => nav(`/study/${s.id}`)}
                className="cursor-pointer border-t border-border hover:bg-muted/50 [&>td]:px-4 [&>td]:py-2.5"
              >
                <td className="font-medium tnum">{s.id}</td>
                <td className="tnum text-muted-foreground">{s.patient}</td>
                <td>{s.scanner}</td>
                <td className="tnum">{s.year}</td>
                <td className="text-muted-foreground">{s.series}</td>
                <td>
                  {s.status === "held-out"
                    ? <Badge tone="warn">held-out</Badge>
                    : <Badge tone="muted">train/val</Badge>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

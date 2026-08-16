import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

const SIDECAR = "http://127.0.0.1:8756";
type Health = "checking" | "online" | "offline";

function useSidecarHealth(): Health {
  const [health, setHealth] = useState<Health>("checking");
  useEffect(() => {
    let alive = true;
    const ping = async () => {
      try {
        const res = await fetch(`${SIDECAR}/health`);
        if (alive) setHealth(res.ok ? "online" : "offline");
      } catch {
        if (alive) setHealth("offline");
      }
    };
    ping();
    const id = setInterval(ping, 3000);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);
  return health;
}

export default function Dashboard() {
  const health = useSidecarHealth();
  const dot =
    health === "online" ? "bg-good" : health === "offline" ? "bg-bad" : "bg-muted-foreground animate-pulse";
  const statusText =
    health === "online" ? "online" : health === "offline" ? "offline · npm run sidecar" : "checking…";

  return (
    <div className="p-6">
      <header className="mb-6">
        <h1 className="text-xl font-semibold">Workstation</h1>
        <p className="text-sm text-muted-foreground">
          Explainable multimodal characterisation — subtype &amp; grade at diagnosis.
        </p>
      </header>

      <div className="grid grid-cols-3 gap-4">
        <Card>
          <CardHeader><CardTitle>Active Model</CardTitle></CardHeader>
          <CardContent className="space-y-2 text-sm">
            <div className="flex justify-between"><span>Backbone</span><span className="text-muted-foreground">3D-ResNet-18 (G2)</span></div>
            <div className="flex justify-between"><span>Split</span><span className="text-muted-foreground tnum">split_v2 · scanner holdout</span></div>
            <div className="flex justify-between"><span>Calibration</span><Badge tone="good">ECE 0.041</Badge></div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader><CardTitle>Honest Performance</CardTitle></CardHeader>
          <CardContent className="space-y-2 text-sm">
            <div className="flex justify-between"><span>Clinical anchor AUROC</span><span className="tnum font-semibold">0.708</span></div>
            <div className="flex justify-between"><span>Imaging-only AUROC</span><span className="tnum text-muted-foreground">0.567</span></div>
            <p className="pt-1 text-xs text-muted-foreground">
              Characterisation performance is reported with its ceiling, never inflated.
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader><CardTitle>Inference Sidecar</CardTitle></CardHeader>
          <CardContent className="space-y-2 text-sm">
            <div className="flex justify-between"><span>Endpoint</span><span className="text-muted-foreground tnum">127.0.0.1:8756</span></div>
            <div className="flex items-center justify-between">
              <span>Status</span>
              <span className="flex items-center gap-1.5">
                <span className={`size-2 rounded-full ${dot}`} />
                <span className="text-muted-foreground tnum">{statusText}</span>
              </span>
            </div>
            <p className="pt-1 text-xs text-muted-foreground">
              PyTorch/MONAI runs out-of-process; UI degrades to mock if offline.
            </p>
          </CardContent>
        </Card>
      </div>

      <p className="mt-6 text-xs text-muted-foreground">
        Open <span className="text-primary">Patients</span> to load a study and run characterisation.
      </p>
    </div>
  );
}

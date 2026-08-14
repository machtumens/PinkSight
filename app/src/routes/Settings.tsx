import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Switch } from "@/components/ui/switch";
import { useInferenceMode } from "@/lib/inference-mode-context";

export default function Settings() {
  const dark = document.documentElement.classList.contains("dark");
  const { mode, setMode } = useInferenceMode();
  return (
    <div className="p-6">
      <h1 className="mb-4 text-xl font-semibold">Settings</h1>
      <div className="grid max-w-xl gap-4">
        <Card>
          <CardHeader><CardTitle>Appearance</CardTitle></CardHeader>
          <CardContent className="flex items-center justify-between text-sm">
            <span>Theme</span>
            <button
              className="rounded-md border border-border px-3 py-1"
              onClick={() => document.documentElement.classList.toggle("dark")}
            >
              {dark ? "Dark" : "Light"} — toggle
            </button>
          </CardContent>
        </Card>
        <Card>
          <CardHeader><CardTitle>Inference sidecar</CardTitle></CardHeader>
          <CardContent className="space-y-3 text-sm text-muted-foreground">
            <div className="flex items-center justify-between">
              <span className="text-foreground">
                Mode: {mode === "live" ? "Synthetic (live)" : "Mock"}
              </span>
              <Switch
                checked={mode === "live"}
                onCheckedChange={(on) => setMode(on ? "live" : "mock")}
                aria-label="Toggle inference mode between Mock and Synthetic (live)"
              />
            </div>
            <div>Endpoint: <span className="tnum">http://127.0.0.1:8756</span></div>
            <div>Start: <span className="tnum">npm run sidecar</span></div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

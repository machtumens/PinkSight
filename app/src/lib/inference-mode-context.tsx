/*
  Inference mode toggle state — "mock" (canned report) vs "live" (one forward-only SYNTHETIC pipeline
  pass). Low-frequency app-level state read by Settings (writes it), StudyViewer + Report (read it).
  React Context is the right fit: a single boolean-ish value, rarely updated, read across routes.
*/
import { createContext, useContext, useState, type ReactNode } from "react";

export type InferenceMode = "mock" | "live";

type InferenceModeValue = {
  mode: InferenceMode;
  setMode: (m: InferenceMode) => void;
};

const InferenceModeContext = createContext<InferenceModeValue | null>(null);

// Safe default = "mock": the app never defaults into a state that could be read as a real result.
export function InferenceModeProvider({ children }: { children: ReactNode }) {
  const [mode, setMode] = useState<InferenceMode>("mock");
  return (
    <InferenceModeContext.Provider value={{ mode, setMode }}>
      {children}
    </InferenceModeContext.Provider>
  );
}

export function useInferenceMode(): InferenceModeValue {
  const ctx = useContext(InferenceModeContext);
  if (ctx === null) {
    throw new Error("useInferenceMode must be used within InferenceModeProvider");
  }
  return ctx;
}

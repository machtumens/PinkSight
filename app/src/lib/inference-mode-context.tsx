import { createContext, useContext, useState, type ReactNode } from "react";

export type InferenceMode = "mock" | "live";

type InferenceModeValue = {
  mode: InferenceMode;
  setMode: (m: InferenceMode) => void;
};

const InferenceModeContext = createContext<InferenceModeValue | null>(null);

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

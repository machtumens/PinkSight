import * as React from "react";
import { cn } from "@/lib/utils";

type Tone = "default" | "good" | "warn" | "bad" | "abstain" | "muted";

const tones: Record<Tone, string> = {
  default: "bg-primary/15 text-primary",
  good: "bg-good/15 text-good",
  warn: "bg-warn/15 text-warn",
  bad: "bg-bad/15 text-bad",
  abstain: "bg-abstain/15 text-abstain",
  muted: "bg-muted text-muted-foreground",
};

export function Badge({
  tone = "default",
  className,
  ...props
}: React.HTMLAttributes<HTMLSpanElement> & { tone?: Tone }) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium tnum",
        tones[tone],
        className
      )}
      {...props}
    />
  );
}

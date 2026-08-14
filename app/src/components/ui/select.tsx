import * as React from "react";
import { cn } from "@/lib/utils";

// Minimal native <select> wrapper (dependency-free, matching button.tsx/badge.tsx house style — the
// project's shadcn primitives are hand-rolled on native elements, no @radix-ui). Consumers pass
// <option> children directly.
export type SelectProps = React.SelectHTMLAttributes<HTMLSelectElement>;

export const Select = React.forwardRef<HTMLSelectElement, SelectProps>(
  ({ className, children, ...props }, ref) => (
    <select
      ref={ref}
      className={cn(
        "h-9 rounded-md border border-border bg-transparent px-3 py-1 text-sm transition-colors",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
        "disabled:pointer-events-none disabled:opacity-50",
        className
      )}
      {...props}
    >
      {children}
    </select>
  )
);
Select.displayName = "Select";

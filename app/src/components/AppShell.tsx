import { NavLink, Outlet } from "react-router-dom";
import {
  LayoutDashboard,
  Users,
  ScrollText,
  Settings as SettingsIcon,
  Microscope,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { InferenceModeProvider } from "@/lib/inference-mode-context";

const nav = [
  { to: "/", label: "Dashboard", icon: LayoutDashboard, end: true },
  { to: "/patients", label: "Patients", icon: Users, end: false },
  { to: "/audit", label: "Audit Trail", icon: ScrollText, end: false },
  { to: "/settings", label: "Settings", icon: SettingsIcon, end: false },
];

export default function AppShell() {
  return (
    <div className="flex h-screen w-screen overflow-hidden">
      <aside className="flex w-56 shrink-0 flex-col border-r border-border bg-card">
        <div className="flex items-center gap-2 px-4 py-4">
          <div className="flex size-8 items-center justify-center rounded-md bg-primary text-primary-foreground">
            <Microscope className="size-4" />
          </div>
          <div className="leading-tight">
            <div className="text-sm font-semibold">PinkSight</div>
            <div className="text-[10px] uppercase tracking-wide text-muted-foreground">
              Research · not clinical
            </div>
          </div>
        </div>
        <nav className="flex flex-1 flex-col gap-0.5 px-2">
          {nav.map(({ to, label, icon: Icon, end }) => (
            <NavLink
              key={to}
              to={to}
              end={end}
              className={({ isActive }) =>
                cn(
                  "flex items-center gap-3 rounded-md px-3 py-2 text-sm transition-colors",
                  isActive ? "bg-primary/10 text-primary" : "text-muted-foreground hover:bg-muted"
                )
              }
            >
              <Icon className="size-4" />
              {label}
            </NavLink>
          ))}
        </nav>
        <div className="border-t border-border px-4 py-3 text-[10px] text-muted-foreground">
          Integrity &gt; Performance &gt; Impact
        </div>
      </aside>

      <main className="flex-1 overflow-y-auto bg-background">
        <InferenceModeProvider>
          <Outlet />
        </InferenceModeProvider>
      </main>
    </div>
  );
}

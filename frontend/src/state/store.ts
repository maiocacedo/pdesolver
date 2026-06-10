/**
 * Zustand store. Single global state for the app.
 *
 * Three slices:
 *   - system     — the problem definition (always a system, length ≥ 1)
 *   - ui         — view state (active tab, mode, drawers, …)
 *   - run        — solve state and result history
 *
 * Keep them as separate slices in the same store so subscriptions stay narrow:
 *   useStore((s) => s.run.field)  — re-renders only when the field changes.
 */

import { create } from "zustand";
import type {
  BCType, DiscMethod, FieldOut, PDEPayload, PDESPayload,
  SolveResult, TimeMethod,
} from "../types";
import { bridge } from "../api/pywebview";
import { heatPreset } from "../gallery/examples";

// ── per-PDE config (UI shape — slightly richer than PDEPayload) ─────
export interface PDEConfig {
  id: string;
  name: string;
  func: string;
  eq: string;                               // pdesolver equation string, e.g. "du/dt = d2u/dx2"
  ic: string;
  west: { type: BCType; expr: string };
  east: { type: BCType; expr: string };
  // 2-D only
  north?: { type: BCType; expr: string };
  south?: { type: BCType; expr: string };
}

export interface SystemConfig {
  pdes: PDEConfig[];                        // length ≥ 1
  activePdeId: string;
  domain: { xmin: string; xmax: string; t0: string; tf: string; ymin?: string; ymax?: string };
  mesh: { nx: number; nt: number; ny?: number };
  scheme: { disc: DiscMethod; time: TimeMethod };
}

export interface UIState {
  mode: "simple" | "advanced";
  vizTab: "plot1d" | "heatmap" | "animation" | "plot3d";
  showInspector: boolean;
  drawer: "gallery" | "history" | null;
  sidebarWidth: number;
  dirty: boolean;
  layoutMode: "grid" | "tabs";
  maximizedPanel: "plot1d" | "heatmap" | "plot3d" | "console" | null;
}

export interface HistoryEntry {
  name: string;
  time: string;
  nx: number;
  nt: number;
  method: TimeMethod;
  snapshot: SystemConfig;
}

export interface RunState {
  status: "pristine" | "solving" | "solved" | "error";
  fields: FieldOut[] | null;                // length matches system.pdes
  activeFieldIndex: number;                 // which field the viz is showing
  lastRunMs: number;
  history: HistoryEntry[];
  error: string | null;
  meta?: SolveResult["meta"] | null;
}

interface Store {
  system: SystemConfig;
  ui: UIState;
  run: RunState;

  // System mutations — always go through these so we can mark dirty + revalidate
  setPdeField(id: string, patch: Partial<PDEConfig>): void;
  addPde(): void;
  removePde(id: string): void;
  setActivePde(id: string): void;
  setMesh(patch: Partial<SystemConfig["mesh"]>): void;
  setDomain(patch: Partial<SystemConfig["domain"]>): void;
  setScheme(patch: Partial<SystemConfig["scheme"]>): void;
  loadPreset(preset: SystemConfig): void;

  // UI
  setUI(patch: Partial<UIState>): void;
  toggleLayoutMode(): void;
  toggleMaximizedPanel(panel: "plot1d" | "heatmap" | "plot3d" | "console"): void;

  // Run
  solve(): Promise<void>;
  resetRun(): void;
}

export const useStore = create<Store>((set, get) => ({
  system: heatPreset(),
  ui: {
    mode: "simple",
    vizTab: "plot1d",
    showInspector: true,
    drawer: null,
    sidebarWidth: 380,
    dirty: false,
    layoutMode: "tabs",
    maximizedPanel: null,
  },
  run: {
    status: "pristine",
    fields: null,
    activeFieldIndex: 0,
    lastRunMs: 0,
    history: [],
    error: null,
    meta: null,
  },

  setPdeField: (id, patch) => set((s) => ({
    system: {
      ...s.system,
      pdes: s.system.pdes.map((p) => (p.id === id ? { ...p, ...patch } : p)),
    },
    ui: { ...s.ui, dirty: true },
  })),

  addPde: () => set((s) => {
    const id = `pde-${Date.now()}-${Math.random().toString(36).substring(2, 6)}`;
    const n = s.system.pdes.length + 1;
    const next: PDEConfig = {
      ...s.system.pdes[0],
      id,
      name: `u${n}`,
      func: `u${n}`,
      eq: `du${n}/dt = d2u${n}/dx2`,
    };
    return {
      system: { ...s.system, pdes: [...s.system.pdes, next], activePdeId: id },
      ui: { ...s.ui, dirty: true },
    };
  }),

  removePde: (id) => set((s) => {
    if (s.system.pdes.length <= 1) return s;
    const pdes = s.system.pdes.filter((p) => p.id !== id);
    const activePdeId =
      s.system.activePdeId === id ? pdes[0].id : s.system.activePdeId;
    return {
      system: { ...s.system, pdes, activePdeId },
      ui: { ...s.ui, dirty: true },
    };
  }),

  setActivePde: (id) => set((s) => ({
    system: { ...s.system, activePdeId: id },
  })),

  setMesh: (patch) => set((s) => ({
    system: { ...s.system, mesh: { ...s.system.mesh, ...patch } },
    ui: { ...s.ui, dirty: true },
  })),

  setDomain: (patch) => set((s) => ({
    system: { ...s.system, domain: { ...s.system.domain, ...patch } },
    ui: { ...s.ui, dirty: true },
  })),

  setScheme: (patch) => set((s) => ({
    system: { ...s.system, scheme: { ...s.system.scheme, ...patch } },
    ui: { ...s.ui, dirty: true },
  })),

  loadPreset: (preset) => set((s) => ({
    system: preset,
    ui: { ...s.ui, dirty: false },
    run: {
      ...s.run,
      status: "pristine",
      fields: null,
      error: null,
      meta: null,
      lastRunMs: 0
    },
  })),

  setUI: (patch) => set((s) => ({ ui: { ...s.ui, ...patch } })),

  toggleLayoutMode: () => set((s) => ({
    ui: {
      ...s.ui,
      layoutMode: s.ui.layoutMode === "tabs" ? "grid" : "tabs",
      maximizedPanel: null,
    },
  })),

  toggleMaximizedPanel: (panel) => set((s) => ({
    ui: {
      ...s.ui,
      maximizedPanel: s.ui.maximizedPanel === panel ? null : panel,
    },
  })),

  solve: async () => {
    set((s) => ({ run: { ...s.run, status: "solving", error: null } }));
    const t0 = performance.now();
    try {
      const payload = toPayload(get().system);
      const result: SolveResult = await bridge.solve(payload);
      set((s) => ({
        run: {
          ...s.run,
          status: "solved",
          fields: result.fields,
          activeFieldIndex: 0,
          lastRunMs: performance.now() - t0,
          meta: result.meta,
          history: [
            historyEntry(get().system),
            ...s.run.history,
          ].slice(0, 10),
        },
      }));
    } catch (err) {
      set((s) => ({
        run: { ...s.run, status: "error", error: String(err), meta: null },
      }));
    }
  },

  resetRun: () => set((s) => ({
    run: { ...s.run, status: "pristine", fields: null, error: null, meta: null },
  })),
}));

// ── Helpers ────────────────────────────────────────────────────────
function toPayload(sys: SystemConfig): PDESPayload {
  const is2D = !!sys.domain.ymin && sys.mesh.ny !== undefined;
  const ivar_boundary: Array<[number, number]> = [
    [parseFloat(sys.domain.xmin) || 0, parseFloat(sys.domain.xmax) || 1],
    ...(is2D ? [[parseFloat(sys.domain.ymin!) || 0, parseFloat(sys.domain.ymax!) || 1] as [number, number]] : []),
  ];
  const pdes: PDEPayload[] = sys.pdes.map((p) => ({
    id: p.id,
    name: p.name,
    eq: p.eq,
    func: p.func,
    sp_var: is2D ? ["x", "y"] : ["x"],
    ivar: ["t"],
    ivar_boundary,
    expr_ic: p.ic,
    west_bd: p.west.type,
    west_func_bd: p.west.expr,
    east_bd: p.east.type,
    east_func_bd: p.east.expr,
    ...(is2D && {
      north_bd: p.north?.type ?? "Dirichlet",
      north_func_bd: p.north?.expr ?? "0",
      south_bd: p.south?.type ?? "Dirichlet",
      south_func_bd: p.south?.expr ?? "0",
    }),
  }));
  return {
    pdes,
    disc_n: is2D ? [sys.mesh.nx, sys.mesh.ny!] : [sys.mesh.nx],
    discretize: { method: sys.scheme.disc },
    solve: { method: sys.scheme.time, tf: parseFloat(sys.domain.tf) || 0.1, nt: sys.mesh.nt },
  };
}

function historyEntry(sys: SystemConfig): HistoryEntry {
  return {
    name: sys.pdes.map((p) => p.name).join(" + "),
    time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    nx: sys.mesh.nx,
    nt: sys.mesh.nt,
    method: sys.scheme.time,
    snapshot: sys,
  };
}

# Curve Overlay & Coupled Variables Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement global store-driven visualization state for overlaying multiple coupled variables in the 1D Profile view, updating active variables, and showing a unified double legend.

**Architecture:** We will extend the global Zustand store to manage selection of visible variables and sync the active field index for other visualizers. We will update `Plot1D.tsx` to render a double legend when overlaying multiple time steps of multiple variables, and clean up local state in `VizPanel.tsx`.

**Tech Stack:** React, TypeScript, Zustand, SVG.

---

## 1. Mapeamento de Arquivos

### A. Frontend (Modificados)
* [frontend/src/state/store.ts](file:///c:/Projetos/PDESsolver/frontend/src/state/store.ts) — Add `visibleFieldIndices` state and `toggleVisibleField` action.
* [frontend/src/viz/VizPanel.tsx](file:///c:/Projetos/PDESsolver/frontend/src/viz/VizPanel.tsx) — Connect checkboxes to the global store instead of local state.
* [frontend/src/viz/Plot1D.tsx](file:///c:/Projetos/PDESsolver/frontend/src/viz/Plot1D.tsx) — Implement Double Legend for variables and time steps.

---

## 2. Tarefas e Passos de Implementação

### Task 1: Update the Zustand State Store

**Files:**
* Modify: [frontend/src/state/store.ts](file:///c:/Projetos/PDESsolver/frontend/src/state/store.ts)

- [ ] **Step 1: Add visibleFieldIndices to RunState and action to Store**
  Modify the `RunState` and `Store` interface declarations to include the new fields.
  ```typescript
  // In RunState:
  visibleFieldIndices: number[];

  // In Store:
  toggleVisibleField(index: number): void;
  ```
- [ ] **Step 2: Initialize state and implement toggleVisibleField action**
  Initialize `visibleFieldIndices: [0]` in the store config. Add the implementation of `toggleVisibleField` in the `create` body.
  ```typescript
  toggleVisibleField: (index) => set((s) => {
    const isVisible = s.run.visibleFieldIndices.includes(index);
    let nextVisible: number[];
    let nextActive = s.run.activeFieldIndex;

    if (isVisible) {
      if (s.run.visibleFieldIndices.length > 1) {
        nextVisible = s.run.visibleFieldIndices.filter((i) => i !== index);
        if (s.run.activeFieldIndex === index) {
          nextActive = nextVisible[0];
        }
      } else {
        nextVisible = s.run.visibleFieldIndices;
      }
    } else {
      nextVisible = [...s.run.visibleFieldIndices, index].sort((a, b) => a - b);
      nextActive = index;
    }

    return {
      run: {
        ...s.run,
        visibleFieldIndices: nextVisible,
        activeFieldIndex: nextActive,
      },
    };
  }),
  ```
- [ ] **Step 3: Reset selections on simulation runs and resets**
  Update the `solve` and `resetRun` functions to reset `visibleFieldIndices` to `[0]` and `activeFieldIndex` to `0` when a simulation is solved or reset.
  In `solve` success path:
  ```typescript
  fields: result.fields,
  activeFieldIndex: 0,
  visibleFieldIndices: [0],
  ```
  In `resetRun`:
  ```typescript
  fields: null,
  activeFieldIndex: 0,
  visibleFieldIndices: [0],
  ```
- [ ] **Step 4: Verify frontend build**
  Run: `npm run build` in directory `c:\Projetos\PDESsolver\frontend`
  Expected: Build succeeds without typescript errors.

---

### Task 2: Sync Visualization Panel Checkboxes

**Files:**
* Modify: [frontend/src/viz/VizPanel.tsx](file:///c:/Projetos/PDESsolver/frontend/src/viz/VizPanel.tsx)

- [ ] **Step 1: Replace local visibleFieldIndices state with global store state**
  Remove local `visibleFieldIndices` state and its `useEffect` reset, and pull the state and action from `useStore`:
  ```typescript
  const visibleFieldIndices = useStore((s) => s.run.visibleFieldIndices);
  const toggleVisibleField = useStore((s) => s.toggleVisibleField);
  ```
- [ ] **Step 2: Update renderFieldSelectors to trigger toggleVisibleField**
  Update the checkbox `onChange` in `renderFieldSelectors` to call `toggleVisibleField(idx)`.
  ```typescript
  onChange={() => toggleVisibleField(idx)}
  ```
- [ ] **Step 3: Verify frontend build**
  Run: `npm run build` in directory `c:\Projetos\PDESsolver\frontend`
  Expected: Build succeeds without typescript errors.

---

### Task 3: Render Double Legend in Plot1D

**Files:**
* Modify: [frontend/src/viz/Plot1D.tsx](file:///c:/Projetos/PDESsolver/frontend/src/viz/Plot1D.tsx)

- [ ] **Step 1: Replace multi-field legend block with a Double Legend**
  Locate the condition `{visibleFieldIndices.length > 1 && (` and update the legend markup to display side-by-side columns: one for variables, and one for time progress when `mode === "all"`.
  ```tsx
  {visibleFieldIndices.length > 1 && (
    <g transform={`translate(${W - padR - 220}, ${padT + 6})`}>
      <rect x="0" y="0" width="210" height={Math.max(visibleFieldIndices.length, mode === "all" ? 3 : 0) * 14 + 10} fill="var(--surface)" stroke="var(--border)" rx="6" />
      <g transform="translate(8, 0)">
        {visibleFieldIndices.map((fieldIdx, idx) => {
          const f = fields[fieldIdx];
          if (!f) return null;
          const label = f.meta?.fieldName || `Field ${fieldIdx + 1}`;
          return (
            <g key={fieldIdx} transform={`translate(0, ${10 + idx * 14})`}>
              <line x1="0" x2="16" y1="2" y2="2" stroke={FIELD_COLORS[fieldIdx % FIELD_COLORS.length]} strokeWidth="2.4" />
              <text x="22" y="5" fontSize="10" fontFamily="var(--font-mono)" fill="var(--text-muted)">
                {label}
              </text>
            </g>
          );
        })}
      </g>
      {mode === "all" && (
        <g transform="translate(110, 0)">
          <text x="0" y="15" fontSize="9" fontWeight="bold" fontFamily="var(--font-mono)" fill="var(--text-faint)">
            Time:
          </text>
          <g transform="translate(0, 20)">
            <line x1="0" x2="16" y1="2" y2="2" stroke="var(--text)" strokeWidth="2.4" opacity="0.25" />
            <text x="22" y="5" fontSize="9" fontFamily="var(--font-mono)" fill="var(--text-faint)">t₀ (faded)</text>
          </g>
          <g transform="translate(0, 34)">
            <line x1="0" x2="16" y1="2" y2="2" stroke="var(--text)" strokeWidth="2.4" opacity="0.92" />
            <text x="22" y="5" fontSize="9" fontFamily="var(--font-mono)" fill="var(--text-faint)">t_f (solid)</text>
          </g>
        </g>
      )}
    </g>
  )}
  ```
- [ ] **Step 2: Verify frontend build**
  Run: `npm run build` in directory `c:\Projetos\PDESsolver\frontend`
  Expected: Build succeeds without typescript errors.

# Curve Overlay & Coupled Variables Design Spec

Design specification for implementing the multi-variable overlay visualization, active variable synchronization, and double legend in the PDESolver Studio 1D Profile view.

## Goal
To allow users to visualize multiple coupled PDE variables (e.g., $u$ and $v$ in a system of PDEs) simultaneously on the 1D Profile plot with color-coded variables, opacity-coded time progression, a double legend, and automatic synchronization of the active variable for single-variable panels (Heatmap and Surface 3D).

---

## Proposed Changes

### 1. Global State updates (`frontend/src/state/store.ts`)
* Add `visibleFieldIndices: number[]` to the store's `RunState` (initialized to `[0]`).
* Add the `toggleVisibleField(index: number)` action to the store:
  * **If already visible (and length > 1):** Remove from `visibleFieldIndices`. If it was the `activeFieldIndex`, set `activeFieldIndex` to the new first visible index.
  * **If not visible:** Add to `visibleFieldIndices`, sort numerically, and set `activeFieldIndex = index`.
* Reset `visibleFieldIndices` to `[0]` and `activeFieldIndex` to `0` inside the successful execution of `solve()`.

### 2. Visualization Panel (`frontend/src/viz/VizPanel.tsx`)
* Replace the local component state `visibleFieldIndices` with the global Zustand state `run.visibleFieldIndices`.
* Wire up the checkboxes in the panel headers to trigger the store's `toggleVisibleField` action.
* In the tab control header, render the checkboxes for selecting variables.

### 3. 1D Profile Plotter (`frontend/src/viz/Plot1D.tsx`)
* Maintain multi-curve rendering where each field uses `FIELD_COLORS[fieldIdx % FIELD_COLORS.length]`.
* If `mode === "all"`, map time profile opacity using `0.25 + 0.67 * tNorm`.
* If `visibleFieldIndices.length > 1` and `mode === "all"`, render the **Double Legend** in the top right:
  * **Variables:** Showing colored line indicators for each selected variable.
  * **Time Progress:** Showing an opacity scale from $t_0$ to $t_f$.
* If `visibleFieldIndices.length > 1` but `mode === "snapshots"`, render only the variables legend.

---

## Verification Plan

### Manual Verification
1. Load a coupled system preset (e.g. system of PDEs with 2 variables).
2. Click "Run" to solve.
3. Verify that both variables are listed as checkboxes in the header of the "1D Profile" visualizer.
4. Toggle checkboxes:
   * Checking both variables overlays both curves on the 1D Plot.
   * Switching to "All Profiles" mode displays multiple time steps for both variables.
   * Verify the double legend appears at the top right of the plot.
5. Verify that checking a variable updates the Heatmap and Surface 3D panels to show that variable (most recently selected variable becomes active).
6. Verify you cannot uncheck the last remaining checkbox.

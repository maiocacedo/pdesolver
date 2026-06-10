import { useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";
import { useStore } from "../state/store";
import type { FieldOut } from "../types";
import { Icon } from "../components/Icon";
import { Plot1D } from "./Plot1D";
import { Heatmap } from "./Heatmap";
import { Heatmap2D } from "./Heatmap2D";
import { Surface3D } from "./Surface3D";
import { Surface3DWebGL } from "./Surface3DWebGL";
import type { Palette } from "./colormap";

type VizTab = "plot1d" | "heatmap" | "plot3d";

interface Props {
  palette?: Palette;
  tab?: VizTab;
  onTabChange?: (t: VizTab) => void;
  engine3D?: "auto" | "webgl" | "compat";
}

function checkWebGLSupport(): boolean {
  try {
    const canvas = document.createElement("canvas");
    return !!(
      window.WebGLRenderingContext &&
      (canvas.getContext("webgl") || canvas.getContext("experimental-webgl"))
    );
  } catch (e) {
    return false;
  }
}

const isWebGLSupported = typeof window !== "undefined" ? checkWebGLSupport() : false;

function EmptyState({ solving }: { solving: boolean }) {
  if (solving) {
    return (
      <div style={{ textAlign: "center", color: "var(--text-muted)" }}>
        <div style={{
          width: 36, height: 36, borderRadius: "50%", margin: "0 auto 14px",
          border: "2.5px solid var(--accent-faint)", borderTopColor: "var(--accent)",
          animation: "spin 0.7s linear infinite",
        }} />
        <div style={{ fontSize: 13, fontWeight: 500, color: "var(--text)" }}>Solving…</div>
        <div style={{ fontSize: 12, marginTop: 4, fontFamily: "var(--font-mono)" }}>
          discretize → step → assemble
        </div>
      </div>
    );
  }
  return (
    <div style={{ textAlign: "center", color: "var(--text-faint)", maxWidth: 360 }}>
      <div style={{
        width: 56, height: 56, margin: "0 auto 16px",
        borderRadius: 14, background: "var(--surface-alt)",
        display: "grid", placeItems: "center", color: "var(--accent)",
      }}>
        <Icon.Plot />
      </div>
      <div style={{ fontSize: 14, fontWeight: 500, color: "var(--text)", marginBottom: 6 }}>
        Run the solver to see results
      </div>
      <div style={{ fontSize: 12.5, lineHeight: 1.55 }}>
        Press <span className="kbd">F5</span> or click{" "}
        <span className="kbd">▶ Run</span> to discretize and integrate.
      </div>
    </div>
  );
}

const TABS: Array<{ id: VizTab; label: string; glyph: ReactNode }> = [
  { id: "plot1d", label: "1D profile", glyph: <Icon.Plot /> },
  { id: "heatmap", label: "Heatmap", glyph: <Icon.Heatmap /> },
  { id: "plot3d", label: "Surface 3D", glyph: <Icon.Cube /> },
];

interface ConsoleProps {
  status: "pristine" | "solving" | "solved" | "error";
  lastRunMs: number;
  error: string | null;
  meta?: { converged: boolean; elapsed_ms: number; backend: string } | null;
  system: any;
}

function SolverConsole({ status, lastRunMs, error, meta, system }: ConsoleProps) {
  const is2D = !!system.domain.ymin && system.mesh.ny !== undefined;
  
  const statStatus = status.toUpperCase();
  const statBackend = meta?.backend ? meta.backend.toUpperCase() : "N/A";
  const statTime = status === "solved" ? `${lastRunMs.toFixed(1)} ms` : "N/A";
  const statConverged = meta?.converged !== undefined ? (meta.converged ? "YES" : "NO") : "N/A";
  const statGrid = is2D 
    ? `${system.mesh.nx} × ${system.mesh.ny} × ${system.mesh.nt}`
    : `${system.mesh.nx} × ${system.mesh.nt}`;

  const lines: Array<{ text: string; type: "info" | "success" | "error" | "default" }> = [];
  lines.push({ text: `[SYSTEM] Initialized PDESolver Studio.`, type: "default" });
  
  if (status === "pristine") {
    lines.push({ text: `[INFO] Ready to solve. Click "Run" or press F5 to start.`, type: "info" });
  } else {
    lines.push({ text: `[INFO] Launching solver backend...`, type: "info" });
    lines.push({ 
      text: `[INFO] Domain boundaries: X=[${system.domain.xmin}, ${system.domain.xmax}]${is2D ? ` Y=[${system.domain.ymin}, ${system.domain.ymax}]` : ""}`,
      type: "default" 
    });
    lines.push({ 
      text: `[INFO] Discretization scheme: ${system.scheme.disc.toUpperCase()} (mesh: ${statGrid})`, 
      type: "default" 
    });
    lines.push({ 
      text: `[INFO] Time integration scheme: ${system.scheme.time.toUpperCase()} (t0=${system.domain.t0}, tf=${system.domain.tf})`, 
      type: "default" 
    });
    
    if (status === "solving") {
      lines.push({ text: `[INFO] Solving system equations...`, type: "info" });
    } else if (status === "solved") {
      lines.push({ text: `[SUCCESS] Simulation finished successfully.`, type: "success" });
      if (meta) {
        lines.push({ text: `[SUCCESS] Backend: ${meta.backend} | Solver elapsed: ${meta.elapsed_ms.toFixed(2)} ms`, type: "success" });
        lines.push({ text: `[SUCCESS] Converged: ${meta.converged ? "Yes" : "No"}`, type: "success" });
      }
      lines.push({ text: `[SUCCESS] Total client execution: ${lastRunMs.toFixed(1)} ms. Ready for visualization.`, type: "success" });
    } else if (status === "error") {
      lines.push({ text: `[ERROR] Simulation failed!`, type: "error" });
      lines.push({ text: `[ERROR] Details: ${error}`, type: "error" });
    }
  }

  return (
    <div className="console-container">
      <div className="console-stats">
        <div className="console-stat-card">
          <div className="console-stat-label">Status</div>
          <div className="console-stat-value" style={{ 
            color: status === "solved" ? "var(--success)" : status === "error" ? "oklch(0.65 0.2 20)" : "var(--text)"
          }}>{statStatus}</div>
        </div>
        <div className="console-stat-card">
          <div className="console-stat-label">Grid Size</div>
          <div className="console-stat-value">{statGrid}</div>
        </div>
        <div className="console-stat-card">
          <div className="console-stat-label">Execution Time</div>
          <div className="console-stat-value">{statTime}</div>
        </div>
        <div className="console-stat-card">
          <div className="console-stat-label">Converged</div>
          <div className="console-stat-value">{statConverged}</div>
        </div>
        <div className="console-stat-card">
          <div className="console-stat-label">Backend</div>
          <div className="console-stat-value">{statBackend}</div>
        </div>
      </div>
      <div className="console-log">
        {lines.map((line, idx) => (
          <div key={idx} className="console-log-line" data-type={line.type}>
            {line.text}
          </div>
        ))}
      </div>
    </div>
  );
}

export function VizPanel({ palette = "viridis", tab: tabProp, onTabChange, engine3D = "auto" }: Props) {
  const storeTab = useStore((s) => s.ui.vizTab);
  const setUI = useStore((s) => s.setUI);
  const status = useStore((s) => s.run.status);
  const fields = useStore((s) => s.run.fields);
  const activeFieldIndex = useStore((s) => s.run.activeFieldIndex);

  const layoutMode = useStore((s) => s.ui.layoutMode);
  const maximizedPanel = useStore((s) => s.ui.maximizedPanel);
  const toggleLayoutMode = useStore((s) => s.toggleLayoutMode);
  const toggleMaximizedPanel = useStore((s) => s.toggleMaximizedPanel);
  const runMeta = useStore((s) => s.run.meta);
  const system = useStore((s) => s.system);
  const runError = useStore((s) => s.run.error);
  const lastRunMs = useStore((s) => s.run.lastRunMs);
  const visibleFieldIndices = useStore((s) => s.run.visibleFieldIndices);
  const toggleVisibleField = useStore((s) => s.toggleVisibleField);

  // Allow controlled tab from parent (DesktopShell menu) or fall back to store
  const tab: VizTab = tabProp ?? storeTab;
  const setTab = (t: VizTab) => {
    setUI({ vizTab: t });
    onTabChange?.(t);
  };

  const field: FieldOut | null = fields?.[activeFieldIndex] ?? null;
  const is2D = !!field?.ys;

  const visiblePanels: Array<"plot1d" | "heatmap" | "plot3d" | "console"> = [];
  if (!is2D) {
    visiblePanels.push("plot1d", "heatmap", "plot3d", "console");
  } else {
    visiblePanels.push("plot3d", "console");
  }

  const [tIndex, setTIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1.0);
  const [plotMode, setPlotMode] = useState<"snapshots" | "all">("snapshots");

  const useWebGL = engine3D === "webgl" || (engine3D === "auto" && isWebGLSupported);

  const render3DPlot = () => {
    if (useWebGL) {
      return <Surface3DWebGL field={field!} palette={palette} tIndex={tIndex} />;
    }
    return <Surface3D field={field!} palette={palette} tIndex={tIndex} />;
  };

  const [recording, setRecording] = useState(false);

  const startRecording = async () => {
    let canvas: HTMLCanvasElement | null = null;
    if (layoutMode === "tabs") {
      canvas = document.querySelector(".viz-frame canvas") as HTMLCanvasElement;
    } else {
      if (maximizedPanel) {
        canvas = document.querySelector(`.grid-panel[data-panel="${maximizedPanel}"] canvas`) as HTMLCanvasElement;
      }
      if (!canvas) {
        canvas = document.querySelector(".grid-panel[data-panel=\"plot3d\"] canvas") as HTMLCanvasElement 
          || document.querySelector(".grid-panel[data-panel=\"heatmap\"] canvas") as HTMLCanvasElement;
      }
    }

    if (!canvas) {
      alert("No canvas found to record! Video recording is optimized for canvas-based views (Heatmaps, 2D simulation, and 3D surface). Please switch tab/view to record.");
      return;
    }

    setPlaying(false);

    try {
      const stream = (canvas as any).captureStream ? (canvas as any).captureStream(30) : null;
      if (!stream) {
        alert("Canvas recording is not supported in this browser.");
        return;
      }

      const chunks: Blob[] = [];
      const recorder = new MediaRecorder(stream, { mimeType: "video/webm; codecs=vp9" });

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunks.push(e.data);
      };

      recorder.onstop = () => {
        const blob = new Blob(chunks, { type: "video/webm" });
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = "simulation_video.webm";
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
      };

      setRecording(true);
      recorder.start();

      const totalSteps = field!.ts.length;
      for (let i = 0; i < totalSteps; i++) {
        setTIndex(i);
        await new Promise((resolve) => setTimeout(resolve, 80));
      }

      recorder.stop();
    } catch (err) {
      console.error("Recording failed", err);
      alert("Recording failed: " + err);
    } finally {
      setRecording(false);
    }
  };



  const timeRef = useRef(0);
  const requestRef = useRef<number | null>(null);
  const previousTimeRef = useRef<number | null>(null);

  useEffect(() => {
    if (field) {
      timeRef.current = field.ts[tIndex] ?? field.ts[0];
    }
  }, [tIndex, field]);

  useEffect(() => {
    if (!playing || !field) {
      if (requestRef.current !== null) {
        cancelAnimationFrame(requestRef.current);
        requestRef.current = null;
      }
      previousTimeRef.current = null;
      return;
    }

    const t0 = field.ts[0];
    const tf = field.ts[field.ts.length - 1];
    const physicalDuration = tf - t0;
    const playbackDurationSeconds = 6.0 / speed;
    const speedFactor = physicalDuration / playbackDurationSeconds;

    const animate = (realTimeMs: number) => {
      if (previousTimeRef.current !== null) {
        const deltaRealTimeSeconds = (realTimeMs - previousTimeRef.current) / 1000;
        const deltaPhysicalTime = deltaRealTimeSeconds * speedFactor;
        
        let targetPhysicalTime = timeRef.current + deltaPhysicalTime;
        if (targetPhysicalTime >= tf) {
          targetPhysicalTime = t0;
        }
        timeRef.current = targetPhysicalTime;

        const ts = field.ts;
        let closestIdx = 0;
        let minDiff = Infinity;
        for (let i = 0; i < ts.length; i++) {
          const diff = Math.abs(ts[i] - targetPhysicalTime);
          if (diff < minDiff) {
            minDiff = diff;
            closestIdx = i;
          }
        }
        setTIndex(closestIdx);
      }
      previousTimeRef.current = realTimeMs;
      requestRef.current = requestAnimationFrame(animate);
    };

    requestRef.current = requestAnimationFrame(animate);
    return () => {
      if (requestRef.current !== null) {
        cancelAnimationFrame(requestRef.current);
      }
    };
  }, [playing, field, speed]);

  useEffect(() => { setTIndex(0); }, [field]);

  const empty = !field;
  const solving = status === "solving";
  const showSlider = !empty && (
    (layoutMode === "tabs" && (
      (tab === "plot1d" && plotMode !== "all") ||
      tab === "plot3d" ||
      (tab === "heatmap" && is2D)
    )) ||
    (layoutMode === "grid" && (
      maximizedPanel === null ||
      maximizedPanel === "plot1d" ||
      maximizedPanel === "plot3d"
    ))
  );

  const fn = field?.meta?.fieldName ?? "u";
  const frameTitle = empty
    ? "Nothing to plot yet"
    : is2D
      ? tab === "heatmap"
        ? `${fn}(x, y) — t = ${field.ts[tIndex].toFixed(4)}`
        : tab === "plot3d"
          ? `Surface ${fn}(x, y) — t = ${field.ts[tIndex].toFixed(4)}`
          : "Not available for 2D fields"
      : tab === "plot1d"
        ? plotMode === "all" ? `Profiles for ${fn}(x, t)` : `${fn}(x, t = ${field.ts[tIndex].toFixed(4)})`
        : tab === "heatmap" ? `${fn}(x, t) over the (x, t) plane`
        : `Surface ${fn}(x, t) — t = ${field.ts[tIndex].toFixed(4)}`;

  const frameSub = empty
    ? "—"
    : is2D
      ? tab === "heatmap" ? "mode='heatmap2d'" : tab === "plot3d" ? "mode='plot3d'" : "—"
      : tab === "plot1d" ? `mode='plot1d${plotMode === "all" ? "_all" : ""}'`
        : tab === "heatmap" ? "mode='heatmap1d'"
        : "mode='plot3d'";

  const getPanelTitle = (panelId: "plot1d" | "heatmap" | "plot3d" | "console") => {
    switch (panelId) {
      case "plot1d":
        return "1D Profile";
      case "heatmap":
        return "Heatmap 1D";
      case "plot3d":
        return is2D ? "Heatmap 2D" : "Surface 3D";
      case "console":
        return "Solver Statistics & Console";
    }
  };

  const renderFieldSelectors = () => {
    if (!fields || fields.length <= 1) return null;
    return (
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginLeft: 12 }}>
        {fields.map((f, idx) => {
          const name = f.meta?.fieldName || `Field ${idx + 1}`;
          const isChecked = visibleFieldIndices.includes(idx);
          return (
            <label key={idx} style={{
              display: "flex", alignItems: "center", gap: 6, fontSize: 12,
              fontWeight: 500, color: "var(--text-muted)", cursor: "pointer",
              userSelect: "none"
            }}>
              <input
                type="checkbox"
                checked={isChecked}
                onChange={() => toggleVisibleField(idx)}
                style={{
                  accentColor: "var(--accent)",
                  cursor: "pointer",
                }}
              />
              <span style={{ color: isChecked ? "var(--text)" : "var(--text-faint)" }}>{name}</span>
            </label>
          );
        })}
      </div>
    );
  };

  const renderPanelHeader = (panelId: "plot1d" | "heatmap" | "plot3d" | "console") => {
    const title = getPanelTitle(panelId);
    const isMax = maximizedPanel === panelId;
    return (
      <div className="panel-header">
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <span className="panel-title">{title}</span>
          {panelId === "plot1d" && renderFieldSelectors()}
        </div>
        <button
          className="panel-action"
          onClick={() => toggleMaximizedPanel(panelId)}
          title={isMax ? "Restore grid layout" : "Maximize panel"}
        >
          {isMax ? "↙ Restore" : "↗ Maximize"}
        </button>
      </div>
    );
  };

  const renderPanelBody = (panelId: "plot1d" | "heatmap" | "plot3d" | "console") => {
    if (solving && panelId !== "console") {
      return <EmptyState solving={true} />;
    }
    if (empty && panelId !== "console") {
      return <EmptyState solving={false} />;
    }

    switch (panelId) {
      case "plot1d":
        if (is2D) {
          return (
            <div style={{ textAlign: "center", color: "var(--text-faint)", fontSize: 13 }}>
              Not available for 2D fields
            </div>
          );
        }
        return <Plot1D fields={fields!} visibleFieldIndices={visibleFieldIndices} mode={plotMode} tIndex={tIndex} palette={palette} />;
      case "heatmap":
        if (is2D) {
          return (
            <div style={{ textAlign: "center", color: "var(--text-faint)", fontSize: 13, padding: 16 }}>
              Heatmap 1D is not applicable for 2D fields.<br />See Heatmap 2D in Panel 3.
            </div>
          );
        }
        return <Heatmap field={field!} palette={palette} />;
      case "plot3d":
        return render3DPlot();
      case "console":
        return (
          <SolverConsole
            status={status}
            lastRunMs={lastRunMs}
            error={runError}
            meta={runMeta}
            system={system}
          />
        );
    }
  };

  return (
    <>
      <div className="tabs">
        {layoutMode === "tabs" ? (
          TABS.map((tb) => (
            <button key={tb.id} className="tab" data-active={tab === tb.id ? "1" : "0"}
                    onClick={() => setTab(tb.id)}>
              <span className="tab-glyph">{tb.glyph}</span>
              {tb.label}
            </button>
          ))
        ) : (
          <div className="tab" data-active="1" style={{ cursor: "default" }}>
            <span className="tab-glyph"><Icon.Gallery /></span>
            Dashboard Grid
          </div>
        )}
        <div style={{ flex: 1 }} />
        <div style={{ display: "flex", alignItems: "center", gap: 8, paddingRight: 12 }}>
          {layoutMode === "tabs" && tab === "plot1d" && !empty && renderFieldSelectors()}
          {layoutMode === "tabs" && tab === "plot1d" && !empty && (
            <div className="seg" style={{ marginRight: 8, marginLeft: 8 }}>
              <button data-active={plotMode === "snapshots" ? "1" : "0"}
                      onClick={() => setPlotMode("snapshots")}>Snapshot</button>
              <button data-active={plotMode === "all" ? "1" : "0"}
                      onClick={() => setPlotMode("all")}>All profiles</button>
            </div>
          )}
          
          <div className="seg">
            <button data-active={layoutMode === "tabs" ? "1" : "0"}
                    onClick={() => {
                      if (layoutMode === "grid") toggleLayoutMode();
                    }}>Tabs View</button>
            <button data-active={layoutMode === "grid" ? "1" : "0"}
                    onClick={() => {
                      if (layoutMode === "tabs") toggleLayoutMode();
                    }}>Grid View</button>
          </div>
        </div>
      </div>

      <div className="viz-stage">
        {layoutMode === "grid" ? (
          maximizedPanel ? (
            <div className="grid-panel" data-panel={maximizedPanel} style={{ width: "100%", height: "100%" }}>
              {renderPanelHeader(maximizedPanel)}
              <div className="panel-body">
                {renderPanelBody(maximizedPanel)}
              </div>
            </div>
          ) : (
            <div className="viz-grid-layout" style={{
              gridTemplateColumns: visiblePanels.length <= 2 ? "1fr 1fr" : "repeat(2, 1fr)",
              gridTemplateRows: visiblePanels.length <= 2 ? "1fr" : "repeat(2, 1fr)"
            }}>
              {visiblePanels.map((panelId) => (
                <div key={panelId} className="grid-panel" data-panel={panelId}>
                  {renderPanelHeader(panelId)}
                  <div className="panel-body">
                    {renderPanelBody(panelId)}
                  </div>
                </div>
              ))}
            </div>
          )
        ) : (
          <div className="viz-frame">
            <div className="viz-frame-head">
              <span className="viz-frame-title">{frameTitle}</span>
              <span className="viz-frame-sub">{frameSub}</span>
            </div>
            <div style={{ flex: 1, minHeight: 0, display: "flex", alignItems: "center", justifyContent: "center" }}>
              {empty ? (
                <EmptyState solving={solving} />
              ) : is2D ? (
                tab === "heatmap" ? (
                  <Heatmap2D field={field} tIndex={tIndex} palette={palette} />
                ) : tab === "plot3d" ? (
                  render3DPlot()
                ) : (
                  <div style={{ textAlign: "center", color: "var(--text-faint)", fontSize: 13 }}>
                    Not available for 2D fields — switch to Heatmap or Surface 3D
                  </div>
                )
              ) : tab === "plot1d" ? (
                <Plot1D fields={fields!} visibleFieldIndices={visibleFieldIndices} mode={plotMode} tIndex={tIndex} palette={palette} />
              ) : tab === "heatmap" ? (
                <Heatmap field={field} palette={palette} />
              ) : (
                render3DPlot()
              )}
            </div>
          </div>
        )}

        {showSlider && field && (
          <div className="bottom-rail" style={{ padding: 0 }}>
            <div className="time-slider">
              <button className="play" aria-label={playing ? "Pause" : "Play"}
                      disabled={recording}
                      onClick={() => setPlaying(!playing)}>
                {playing ? <Icon.Pause /> : <Icon.Run />}
              </button>
              <select
                value={speed}
                disabled={recording}
                onChange={(e) => setSpeed(parseFloat(e.target.value))}
                className="speed-select"
                aria-label="Playback speed"
              >
                <option value="0.25">0.25x</option>
                <option value="0.5">0.5x</option>
                <option value="1">1.0x</option>
                <option value="1.5">1.5x</option>
                <option value="2">2.0x</option>
                <option value="4">4.0x</option>
              </select>
              <div className="label">t = {field.ts[tIndex].toFixed(4)}</div>
              <div className="time-track"
                   onClick={(e) => {
                     if (recording) return; // Prevent clicks during recording
                     const r = e.currentTarget.getBoundingClientRect();
                     const f = (e.clientX - r.left) / r.width;
                     const t0 = field.ts[0];
                     const tf = field.ts[field.ts.length - 1];
                     const targetTime = t0 + f * (tf - t0);
                     
                     let closestIdx = 0;
                     let minDiff = Infinity;
                     for (let i = 0; i < field.ts.length; i++) {
                       const diff = Math.abs(field.ts[i] - targetTime);
                       if (diff < minDiff) {
                         minDiff = diff;
                         closestIdx = i;
                       }
                     }
                     setTIndex(closestIdx);
                   }}>
                {(() => {
                  const t0 = field.ts[0];
                  const tf = field.ts[field.ts.length - 1];
                  const pct = ((field.ts[tIndex] - t0) / Math.max(1e-9, tf - t0)) * 100;
                  return (
                    <>
                      <div className="time-track-fill" style={{ width: `${pct}%` }} />
                      <div className="time-track-thumb" style={{ left: `${pct}%` }} />
                    </>
                  );
                })()}
              </div>
              <div className="label" style={{ textAlign: "left" }}>
                step {tIndex + 1}/{field.ts.length}
              </div>
              <button className="play" aria-label="Reset"
                      disabled={recording}
                      onClick={() => { setTIndex(0); setPlaying(false); }}>
                <Icon.Reset />
              </button>
              <button className="play" aria-label="Record simulation video"
                      onClick={startRecording}
                      disabled={recording}
                      style={{
                        marginLeft: 6,
                        color: recording ? "oklch(0.65 0.25 20)" : "inherit",
                        position: "relative"
                      }}>
                {recording ? (
                  <span style={{
                    display: "inline-block",
                    width: 8,
                    height: 8,
                    borderRadius: "50%",
                    background: "oklch(0.65 0.25 20)",
                    boxShadow: "0 0 8px oklch(0.65 0.25 20)"
                  }} />
                ) : (
                  <span style={{ fontSize: 11 }}>📹 Rec</span>
                )}
              </button>
            </div>
          </div>
        )}
      </div>
    </>
  );
}

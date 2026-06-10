import type { FieldOut } from "../types";
import { colormap, type Palette } from "./colormap";

interface Props {
  fields: FieldOut[];
  visibleFieldIndices: number[];
  mode?: "snapshots" | "all";
  tIndex?: number;
  palette?: Palette;
}

const FIELD_COLORS = [
  "var(--accent)",        // Blue/Teal (Field 0)
  "oklch(0.65 0.22 140)", // Vibrant Emerald (Field 1)
  "oklch(0.60 0.22 30)",  // Warm Coral (Field 2)
  "oklch(0.55 0.20 280)", // Royal Violet (Field 3)
  "oklch(0.62 0.20 80)",  // Amber Gold (Field 4)
];

export function Plot1D({ fields, visibleFieldIndices, mode = "snapshots", tIndex = 0, palette = "viridis" }: Props) {
  const activeField = fields[visibleFieldIndices[0]] || fields[0];
  if (!activeField) {
    return (
      <div style={{ textAlign: "center", color: "var(--text-faint)", fontSize: 13, padding: 24 }}>
        No field data available
      </div>
    );
  }

  const { xs, ts } = activeField;
  const W = 700, H = 360, padL = 50, padR = 20, padT = 22, padB = 36;
  const innerW = W - padL - padR;
  const innerH = H - padT - padB;

  const xRange: [number, number] = [xs[0], xs[xs.length - 1]];
  
  let globalMin = Infinity;
  let globalMax = -Infinity;
  visibleFieldIndices.forEach(idx => {
    const f = fields[idx];
    if (f) {
      if (f.min < globalMin) globalMin = f.min;
      if (f.max > globalMax) globalMax = f.max;
    }
  });
  if (globalMin === Infinity) {
    globalMin = 0;
    globalMax = 1;
  }
  const min = globalMin;
  const max = globalMax;

  const yPad = (max - min) * 0.08 || 0.05;
  const yRange: [number, number] = [min - yPad, max + yPad];

  const mapX = (x: number) => padL + ((x - xRange[0]) / (xRange[1] - xRange[0])) * innerW;
  const mapY = (y: number) => padT + (1 - (y - yRange[0]) / (yRange[1] - yRange[0])) * innerH;

  const profiles =
    mode === "all"
      ? Array.from({ length: Math.min(8, activeField.grid.length) }, (_, i) =>
          Math.floor((i / 7) * (activeField.grid.length - 1))
        )
      : [Math.max(0, Math.min(activeField.grid.length - 1, tIndex))];

  const xTicks = 6, yTicks = 5;
  const xT = Array.from({ length: xTicks }, (_, i) => xRange[0] + (i / (xTicks - 1)) * (xRange[1] - xRange[0]));
  const yT = Array.from({ length: yTicks }, (_, i) => yRange[0] + (i / (yTicks - 1)) * (yRange[1] - yRange[0]));

  const yLabel = visibleFieldIndices.length > 0
    ? `${visibleFieldIndices.map(idx => fields[idx]?.meta?.fieldName || `Field ${idx}`).join(", ")}(x, t)`
    : "u(x, t)";

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "100%", display: "block" }}>
      {xT.map((x, i) => (
        <line key={`vx${i}`} x1={mapX(x)} x2={mapX(x)} y1={padT} y2={H - padB} stroke="var(--border-soft)" strokeWidth="1" />
      ))}
      {yT.map((y, i) => (
        <line key={`hy${i}`} x1={padL} x2={W - padR} y1={mapY(y)} y2={mapY(y)} stroke="var(--border-soft)" strokeWidth="1" />
      ))}
      <line x1={padL} x2={W - padR} y1={H - padB} y2={H - padB} stroke="var(--border-strong)" strokeWidth="1" />
      <line x1={padL} x2={padL} y1={padT} y2={H - padB} stroke="var(--border-strong)" strokeWidth="1" />
      {yRange[0] < 0 && yRange[1] > 0 && (
        <line x1={padL} x2={W - padR} y1={mapY(0)} y2={mapY(0)} stroke="var(--text-faint)" strokeDasharray="3 3" />
      )}
      {xT.map((x, i) => (
        <text key={`xl${i}`} x={mapX(x)} y={H - padB + 16} textAnchor="middle" fontSize="10" fontFamily="var(--font-mono)" fill="var(--text-faint)">
          {x.toFixed(2)}
        </text>
      ))}
      {yT.map((y, i) => (
        <text key={`yl${i}`} x={padL - 8} y={mapY(y) + 3} textAnchor="end" fontSize="10" fontFamily="var(--font-mono)" fill="var(--text-faint)">
          {y.toFixed(2)}
        </text>
      ))}
      <text x={(padL + W - padR) / 2} y={H - 6} textAnchor="middle" fontSize="11" fontFamily="var(--font-math)" fontStyle="italic" fill="var(--text-muted)">x</text>
      <text x={14} y={(padT + H - padB) / 2} textAnchor="middle" fontSize="11" fontFamily="var(--font-math)" fontStyle="italic" fill="var(--text-muted)"
            transform={`rotate(-90 14 ${(padT + H - padB) / 2})`}>{yLabel}</text>
      {visibleFieldIndices.map((fieldIdx) => {
        const f = fields[fieldIdx];
        if (!f) return null;
        
        return profiles.map((ti, k) => {
          const row = f.grid[ti];
          if (!row) return null;
          const path = row.map((y, i) => `${i === 0 ? "M" : "L"}${mapX(xs[i])},${mapY(y)}`).join(" ");
          
          const tNorm = profiles.length === 1 ? 0.6 : k / Math.max(1, profiles.length - 1);
          
          let stroke = "var(--accent)";
          let opacity = "0.92";
          
          if (visibleFieldIndices.length > 1) {
            stroke = FIELD_COLORS[fieldIdx % FIELD_COLORS.length];
            if (profiles.length > 1) {
              opacity = (0.25 + 0.67 * tNorm).toFixed(2);
            }
          } else {
            stroke = profiles.length === 1 ? FIELD_COLORS[fieldIdx % FIELD_COLORS.length] : colormap(tNorm, palette);
          }
          
          return (
            <path key={`${fieldIdx}-${ti}`} d={path} stroke={stroke}
                  strokeWidth={profiles.length === 1 ? 2.4 : 1.8}
                  fill="none" strokeLinecap="round" strokeLinejoin="round" opacity={opacity} />
          );
        });
      })}
      {visibleFieldIndices.length > 1 && (
        <g transform={`translate(${W - padR - 110}, ${padT + 6})`}>
          <rect x="0" y="0" width="100" height={visibleFieldIndices.length * 14 + 10} fill="var(--surface)" stroke="var(--border)" rx="6" />
          {visibleFieldIndices.map((fieldIdx, idx) => {
            const f = fields[fieldIdx];
            if (!f) return null;
            const label = f.meta?.fieldName || `Field ${fieldIdx + 1}`;
            return (
              <g key={fieldIdx} transform={`translate(8, ${10 + idx * 14})`}>
                <line x1="0" x2="16" y1="2" y2="2" stroke={FIELD_COLORS[fieldIdx % FIELD_COLORS.length]} strokeWidth="2.4" />
                <text x="22" y="5" fontSize="10" fontFamily="var(--font-mono)" fill="var(--text-muted)">
                  {label}
                </text>
              </g>
            );
          })}
        </g>
      )}
      {visibleFieldIndices.length === 1 && mode === "all" && (
        <g transform={`translate(${W - padR - 110}, ${padT + 6})`}>
          <rect x="0" y="0" width="100" height={profiles.length * 14 + 10} fill="var(--surface)" stroke="var(--border)" rx="6" />
          {profiles.map((ti, k) => {
            const tNorm = k / Math.max(1, profiles.length - 1);
            return (
              <g key={ti} transform={`translate(8, ${10 + k * 14})`}>
                <line x1="0" x2="16" y1="2" y2="2" stroke={colormap(tNorm, palette)} strokeWidth="2.4" />
                <text x="22" y="5" fontSize="10" fontFamily="var(--font-mono)" fill="var(--text-muted)">
                  t = {ts[ti].toFixed(3)}
                </text>
              </g>
            );
          })}
        </g>
      )}
    </svg>
  );
}

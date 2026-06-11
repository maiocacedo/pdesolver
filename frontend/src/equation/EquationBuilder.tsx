import { Latex, pdeToLatex, validateEquation } from "./Latex";

interface EqConfig {
  func: string;
  eq: string;
}

interface Props {
  config: EqConfig;
  setConfig: (c: EqConfig) => void;
}

export function EquationBuilder({ config, setConfig }: Props) {
  const { func, eq } = config;
  const error = validateEquation(eq, func);

  return (
    <>
      <div className="field">
        <span className="field-label">Function name</span>
        <div className="input-group" style={{ width: 120 }}>
          <input
            className="input"
            value={func}
            maxLength={8}
            onChange={(e) => {
              const f = e.target.value || "u";
              setConfig({ func: f, eq });
            }}
          />
          <span className="addon">(x, t)</span>
        </div>
      </div>

      <div className="field">
        <span className="field-label">Equation</span>
        <input
          className="input input-mono"
          value={eq}
          onChange={(e) => setConfig({ func, eq: e.target.value })}
          placeholder="du/dt = d2u/dx2"
          style={{ fontFamily: "var(--font-mono)", fontSize: 13 }}
        />
        <span className="field-hint">
          pdesolver format — e.g. <code>du/dt = d2u/dx2 + sin(x)</code>
        </span>
      </div>

      {error ? (
        <div style={{
          marginTop: 12,
          padding: "8px 12px",
          background: "oklch(0.2 0.05 50)",
          border: "1px solid oklch(0.4 0.1 50)",
          borderRadius: 8,
          color: "oklch(0.85 0.08 60)",
          fontSize: "11px",
          display: "flex",
          alignItems: "center",
          gap: 6,
          lineHeight: "1.4"
        }}>
          <span>⚠️</span> {error}
        </div>
      ) : (
        <div style={{
          marginTop: 12,
          padding: "10px 14px",
          background: "var(--surface-alt)",
          border: "1px solid var(--border)",
          borderRadius: 8,
          minHeight: 44,
          display: "block",
          textAlign: "center",
          overflowX: "auto",
          fontSize: "13.5px",
          color: "var(--text)"
        }}>
          <Latex math={pdeToLatex(eq || " ")} block />
        </div>
      )}
    </>
  );
}

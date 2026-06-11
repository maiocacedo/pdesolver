import { useEffect, useRef } from "react";
import katex from "katex";
import "katex/dist/katex.min.css";

// Translates pdesolver equations and formulas to LaTeX format
export function pdeToLatex(expr: string): string {
  if (!expr) return "";

  let s = expr.trim();

  // Normalize spaces around operators
  s = s.replace(/\s+/g, " ");

  // Replace second order spatial derivatives: d2{V}/d{x}2 -> \frac{\partial^2 V}{\partial x^2}
  s = s.replace(/d2([a-zA-Z_]\w*)\/d([a-zA-Z_]\w*)2/g, "\\frac{\\partial^2 $1}{\\partial $2^2}");

  // Replace first order spatial/temporal derivatives: d{V}/d{x} -> \frac{\partial V}{\partial x}
  s = s.replace(/d([a-zA-Z_]\w*)\/d([a-zA-Z_]\w*)/g, "\\frac{\\partial $1}{\\partial $2}");

  // Replace powers: ** -> ^
  s = s.replace(/\*\*/g, "^");

  // Replace multiplication: * -> \cdot
  s = s.replace(/\*/g, " \\cdot ");

  // Trig and exponential functions
  s = s.replace(/\bsin\b/g, "\\sin");
  s = s.replace(/\bcos\b/g, "\\cos");
  s = s.replace(/\btan\b/g, "\\tan");
  s = s.replace(/\bexp\b/g, "\\exp");
  s = s.replace(/\bsqrt\b/g, "\\sqrt");

  // Pi constant
  s = s.replace(/\bpi\b/g, "\\pi");

  return s;
}

// Basic equation validation
export function validateEquation(eq: string, funcName: string): string | null {
  const trimmed = eq.trim();
  if (!trimmed) return "Equation cannot be empty";
  if (!trimmed.includes("=")) return "Missing '=' sign";

  const parts = trimmed.split("=");
  if (parts.length > 2) return "Multiple '=' signs found";

  const lhs = parts[0].trim();
  const rhs = parts[1].trim();

  if (!lhs) return "Left-hand side is empty";
  if (!rhs) return "Right-hand side is empty";

  // The LHS must be a time derivative like du/dt or d{funcName}/dt or d2{funcName}/dt2
  const lhsRegex = new RegExp(`^d(2)?${funcName}/dt(2)?$`);
  if (!lhsRegex.test(lhs)) {
    return `Left-hand side must be a time derivative of '${funcName}', e.g., d${funcName}/dt or d2${funcName}/dt2`;
  }

  return null;
}

// Basic initial condition validation
export function validateIc(ic: string): string | null {
  const trimmed = ic.trim();
  if (!trimmed) return "Initial condition cannot be empty";

  let balance = 0;
  for (const char of trimmed) {
    if (char === "(") balance++;
    if (char === ")") balance--;
    if (balance < 0) return "Mismatched parentheses";
  }
  if (balance !== 0) return "Mismatched parentheses";

  return null;
}

interface LatexProps {
  math: string;
  block?: boolean;
}

export function Latex({ math, block = false }: LatexProps) {
  const containerRef = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    if (!containerRef.current) return;
    try {
      katex.render(math, containerRef.current, {
        displayMode: block,
        throwOnError: false,
      });
    } catch (e) {
      containerRef.current.textContent = math;
    }
  }, [math, block]);

  const Component = block ? "div" : "span";
  return (
    <Component
      ref={containerRef as any}
      style={block ? { width: "100%", overflowX: "auto", display: "block" } : undefined}
    />
  );
}

from math import factorial
from typing import List, Tuple

import numpy as np
import sympy as sp

H = sp.Symbol("h", positive=True)

_DERIV_LABEL = {
    ("x", 1): "du/dx",
    ("x", 2): "d2u/dx2",
    ("y", 1): "du/dy",
    ("y", 2): "d2u/dy2",
}


def _uniform_stencil(order: int, method: str):
    if order == 2:
        return (1 / H ** 2, -2 / H ** 2, 1 / H ** 2), (-H, sp.Integer(0), H)
    if method == "central":
        return (-1 / (2 * H), sp.Integer(0), 1 / (2 * H)), (-H, sp.Integer(0), H)
    if method == "forward":
        return (sp.Integer(0), -1 / H, 1 / H), (-H, sp.Integer(0), H)
    if method == "backward":
        return (-1 / H, 1 / H, sp.Integer(0)), (-H, sp.Integer(0), H)
    raise ValueError(
        f"Método inválido: '{method}'. Use 'forward', 'central' ou 'backward'."
    )


def symbolic_moments(weights, offsets, kmax: int = 6) -> List:
    out = []
    for k in range(kmax + 1):
        c = sum(w * d ** k for w, d in zip(weights, offsets))
        out.append(sp.simplify(sp.expand(c) / sp.factorial(k)))
    return out


def numeric_moments(weights, offsets, kmax: int = 6) -> np.ndarray:
    return np.array([
        sum(w * d ** k for w, d in zip(weights, offsets)) / factorial(k)
        for k in range(kmax + 1)
    ])


def leading_term(order: int, method: str, kmax: int = 6) -> Tuple:
    weights, offsets = _uniform_stencil(order, method)
    moments = symbolic_moments(weights, offsets, kmax)
    for k, c in enumerate(moments):
        if k == order:
            continue
        c = sp.simplify(c)
        if c != 0:
            return k, sp.simplify(c)
    return None, sp.Integer(0)


def error_order(order: int, method: str, kmax: int = 6) -> int:
    k, coeff = leading_term(order, method, kmax)
    if k is None or coeff == 0:
        return kmax
    return int(sp.Poly(sp.expand(coeff), H).degree())


def axis_moments(axis, order: int, method: str, kmax: int = 6) -> np.ndarray:
    weights = axis.w2() if order == 2 else axis.w1(method)
    offsets = (-axis.hm, np.zeros_like(axis.hm), axis.hp)
    return numeric_moments(weights, offsets, kmax)


def mesh_leading_coefficient(axis, order: int, method: str,
                             kmax: int = 6) -> Tuple[int, float]:
    moments = axis_moments(axis, order, method, kmax)
    interior = slice(1, -1) if not axis.periodic else slice(None)
    for k in range(kmax + 1):
        if k == order:
            continue
        peak = float(np.max(np.abs(moments[k][interior])))
        if peak > 1e-12:
            return k, peak
    return -1, 0.0


def operator_terms(operator) -> List[dict]:
    terms = []
    axis_of = {"_x": 0, "_xx": 0, "_y": 1, "_yy": 1}
    order_of = {"_x": 1, "_xx": 2, "_y": 1, "_yy": 2}
    names = "xy"

    for j, expr in enumerate(operator.exprs):
        for ki, v, suf in operator._fields:
            if suf not in axis_of:
                continue
            sym = sp.Symbol(f"{v}{suf}")
            if sym not in expr.free_symbols:
                continue
            coeff = sp.diff(expr, sym)
            field_syms = {
                s for s in coeff.free_symbols
                if s.name.startswith("XX")
            }
            axis_idx = axis_of[suf]
            order = order_of[suf]
            axis = operator.grid.axes[axis_idx]
            k_sym, c_sym = leading_term(order, operator.method)
            k_num, c_num = mesh_leading_coefficient(
                axis, order, operator.method
            )
            terms.append({
                "eq": j,
                "func": ki,
                "label": _DERIV_LABEL[(names[axis_idx], order)],
                "axis": names[axis_idx],
                "order": order,
                "coeff": coeff,
                "linear": not field_syms,
                "sym_k": k_sym,
                "sym_coeff": c_sym,
                "sym_order": error_order(order, operator.method),
                "mesh_k": k_num,
                "mesh_coeff": c_num,
            })
    return terms


def modified_equation(operator) -> List[dict]:
    extra = []
    for term in operator_terms(operator):
        if not term["linear"] or term["sym_k"] is None:
            continue
        signed = None
        if term["coeff"].is_number:
            sinal = sp.sign(sp.expand(term["coeff"] * term["sym_coeff"]) / H
                            ** term["sym_order"])
            signed = float(sp.re(sp.N(sinal))) * abs(
                float(sp.re(sp.N(term["coeff"]))) * term["mesh_coeff"]
            )
        extra.append({
            "eq": term["eq"],
            "from": term["label"],
            "derivative_order": term["sym_k"],
            "coefficient": sp.simplify(term["coeff"] * term["sym_coeff"]),
            "magnitude": abs(signed) if signed is not None else None,
            "signed": signed,
        })
    return extra


def physical_coefficients(operator) -> List[dict]:
    out = []
    for j, expr in enumerate(operator.exprs):
        for ki, v, suf in operator._fields:
            if suf not in ("_x", "_xx", "_y", "_yy"):
                continue
            sym = sp.Symbol(f"{v}{suf}")
            if sym not in expr.free_symbols:
                continue
            coeff = sp.diff(expr, sym)
            if not coeff.is_number:
                continue
            out.append({
                "eq": j, "func": ki, "suffix": suf,
                "value": float(sp.re(sp.N(coeff))),
            })
    return out

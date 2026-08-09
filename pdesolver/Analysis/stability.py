from typing import List, Tuple

import numpy as np
import sympy as sp

from ..Solvers.RKF import RKF45_A, RKF45_B4, RKF45_B5

_A_STABLE = ("bdf2", "cn")


def butcher_matrix(rows) -> np.ndarray:
    s = len(rows)
    A = np.zeros((s, s))
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            A[i, j] = val
    return A


def stability_polynomial(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    s = len(b)
    one = np.ones(s)
    coeffs = [1.0]
    v = one.copy()
    for _ in range(s):
        coeffs.append(float(b @ v))
        v = A @ v
    return np.array(coeffs)


def stability_function(coeffs: np.ndarray):
    descending = coeffs[::-1]

    def R(z):
        return np.polyval(descending, z)

    return R


def real_axis_limit(coeffs: np.ndarray, lo: float = -20.0) -> float:
    R = stability_function(coeffs)
    hi = 0.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if abs(R(mid)) <= 1.0:
            hi = mid
        else:
            lo = mid
    return hi


def rkf45_polynomial(embedded: bool = False) -> np.ndarray:
    A = butcher_matrix(RKF45_A)
    b = np.array(RKF45_B4 if embedded else RKF45_B5)
    return stability_polynomial(A, b)


def _worst_node(axis) -> int:
    return int(np.argmin(axis.hm + axis.hp))


def _axis_symbol(axis, order: int, method: str, k: np.ndarray) -> np.ndarray:
    weights = axis.w2() if order == 2 else axis.w1(method)
    i = _worst_node(axis)
    wm, wc, wp = (float(w[i]) for w in weights)
    hm, hp = float(axis.hm[i]), float(axis.hp[i])
    return wm * np.exp(-1j * k * hm) + wc + wp * np.exp(1j * k * hp)


def symbol_eigenvalues(operator, nk: int = 48) -> Tuple[np.ndarray, bool]:
    grid = operator.grid
    ax = grid.axes[0]
    hx = float(ax.hm[_worst_node(ax)])
    kx = np.linspace(0.0, np.pi / hx, nk)

    if operator.ndim == 2:
        ay = grid.axes[1]
        hy = float(ay.hm[_worst_node(ay)])
        ky = np.linspace(0.0, np.pi / hy, nk)
    else:
        ay, ky = None, np.zeros(1)

    sx1 = _axis_symbol(ax, 1, operator.method, kx)
    sx2 = _axis_symbol(ax, 2, operator.method, kx)
    if ay is not None:
        sy1 = _axis_symbol(ay, 1, operator.method, ky)
        sy2 = _axis_symbol(ay, 2, operator.method, ky)

    n = operator.n_funcs
    linear = True
    entries = []
    for j, expr in enumerate(operator.exprs):
        for ki, v, suf in operator._fields:
            sym = sp.Symbol(f"{v}{suf}")
            if sym not in expr.free_symbols:
                continue
            coeff = sp.diff(expr, sym)
            if not coeff.is_number:
                linear = False
                continue
            entries.append((j, ki, suf, complex(sp.N(coeff))))

    lams = []
    for a in range(len(kx)):
        for b_ in range(len(ky)):
            M = np.zeros((n, n), dtype=complex)
            for j, ki, suf, c in entries:
                if suf == "":
                    s = 1.0
                elif suf == "_x":
                    s = sx1[a]
                elif suf == "_xx":
                    s = sx2[a]
                elif suf == "_y":
                    s = sy1[b_]
                elif suf == "_yy":
                    s = sy2[b_]
                elif suf == "_xy":
                    s = sx1[a] * sy1[b_]
                else:
                    continue
                M[j, ki] += c * s
            lams.extend(np.linalg.eigvals(M).tolist())

    return np.array(lams), linear


def max_stable_dt(lams: np.ndarray, coeffs: np.ndarray) -> float:
    R = stability_function(coeffs)
    scale = np.max(np.abs(lams))
    if scale <= 0.0:
        return float("inf")

    def ok(dt):
        return bool(np.all(np.abs(R(dt * lams)) <= 1.0 + 1e-12))

    lo, hi = 0.0, 10.0 / scale
    while ok(hi):
        hi *= 2.0
        if hi > 1e12 / scale:
            return float("inf")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if ok(mid):
            lo = mid
        else:
            hi = mid
    return lo


def spectral_sample(operator, t_val: float = 0.0, k: int = 6):
    import scipy.sparse.linalg as sla

    from ..Solvers.solver_base import ColoredJacobian

    jac = ColoredJacobian(*operator.sparsity())
    L, _ = jac.build(operator, np.zeros(operator.size), t_val)
    L = L.astype(np.float64)
    kk = min(k, max(1, L.shape[0] - 2))
    vals = []
    for which in ("LM", "LR"):
        try:
            vals.extend(sla.eigs(
                L, k=kk, which=which, return_eigenvectors=False,
                tol=1e-6, maxiter=5000,
            ).tolist())
        except Exception:
            continue
    if not vals:
        return None
    return np.array(vals, dtype=complex)


def stability_limit(operator, method: str = "RKF", nk: int = 48) -> dict:
    key = method.lower()
    if key in _A_STABLE:
        return {
            "method": method,
            "unconditional": True,
            "dt_max": float("inf"),
            "linear": True,
            "note": "método A-estável: sem restrição de estabilidade em dt",
        }

    coeffs = rkf45_polynomial()
    lams, linear = symbol_eigenvalues(operator, nk=nk)
    dt_sym = max_stable_dt(lams, coeffs)

    out = {
        "method": method,
        "unconditional": False,
        "dt_max": dt_sym,
        "linear": linear,
        "real_axis_limit": real_axis_limit(coeffs),
        "lambda_max": float(np.max(np.abs(lams))) if lams.size else 0.0,
    }

    growth = float(np.max(lams.real)) if lams.size else 0.0
    out["growth_rate"] = growth
    out["unstable_mode"] = growth > 1e-8 * max(1.0, out["lambda_max"])

    lams_num = spectral_sample(operator)
    if lams_num is not None:
        out["dt_max_spectral"] = max_stable_dt(lams_num, coeffs)
        out["lambda_spectral"] = complex(
            lams_num[int(np.argmax(np.abs(lams_num)))]
        )
        out["growth_spectral"] = float(np.max(lams_num.real))
    return out


def cell_peclet(operator) -> List[dict]:
    from .truncation import physical_coefficients

    coeffs = physical_coefficients(operator)
    per_eq: dict = {}
    for c in coeffs:
        per_eq.setdefault((c["eq"], c["func"]), {})[c["suffix"]] = c["value"]

    out = []
    names = {"_x": ("_xx", 0, "x"), "_y": ("_yy", 1, "y")}
    for (eq, func), terms in per_eq.items():
        for adv, (dif, axis_idx, name) in names.items():
            if adv not in terms:
                continue
            a = abs(terms[adv])
            d = abs(terms.get(dif, 0.0))
            if a == 0.0:
                continue
            axis = operator.grid.axes[axis_idx]
            h = float(np.max(axis.hm + axis.hp) / 2.0)
            pe = float("inf") if d == 0.0 else a * h / (2.0 * d)
            out.append({
                "eq": eq, "func": func, "axis": name,
                "peclet": pe, "advection": a, "diffusion": d, "h": h,
            })
    return out

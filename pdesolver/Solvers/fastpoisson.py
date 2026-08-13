import numpy as np
import sympy as sp
from scipy.fft import dstn, idstn


def _constant_laplacian_coeffs(operator):
    """Return per-equation (axx, ayy) if the implicit part is a plain,
    constant-coefficient Laplacian acting on the equation's own function.

    Returns None when the structure does not qualify.
    """
    coeffs = []
    for j, expr in enumerate(operator.exprs):
        expr = sp.expand(expr)
        if expr == 0:
            coeffs.append((0.0, 0.0))
            continue

        v = operator._xd_var[j]
        sxx = sp.Symbol(f"{v}_xx")
        syy = sp.Symbol(f"{v}_yy")
        permitidos = {sxx} | ({syy} if operator.ndim == 2 else set())

        usados = {s for s in expr.free_symbols
                  if s.name.startswith("XX") or s.name in ("CX", "CY", "TT")}
        if not usados <= permitidos:
            return None

        axx = sp.diff(expr, sxx)
        ayy = sp.diff(expr, syy) if operator.ndim == 2 else sp.Integer(0)
        if not (axx.is_number and ayy.is_number):
            return None
        resto = sp.simplify(expr - axx * sxx - (ayy * syy if operator.ndim == 2
                                                else 0))
        if resto != 0:
            return None
        coeffs.append((float(axx), float(ayy)))
    return coeffs


def applicable(operator, pdes) -> bool:
    if operator.ndim != 2:
        return False
    if not operator.grid.uniform:
        return False
    if any(ax.periodic for ax in operator.grid.axes):
        return False

    lados = ("west", "east") + (("south", "north") if operator.ndim == 2 else ())
    for pde in pdes.pdes:
        for lado in lados:
            if str(getattr(pde, f"{lado}_bd", "")).lower() != "dirichlet":
                return False

    return _constant_laplacian_coeffs(operator) is not None


class FastDiffusionSolver:
    """Direct solver for ``(I - c*L)u = b`` via the discrete sine transform.

    Valid when ``L`` is a constant-coefficient Laplacian on a uniform
    tensor-product grid with Dirichlet boundaries, in which case the discrete
    sine basis diagonalizes it and the solve costs ``O(N log N)``.
    """

    def __init__(self, operator, coeffs, c):
        self.shape = tuple(operator.grid.shape)
        self.n_funcs = operator.n_funcs
        self.ndim = operator.ndim
        self.c = float(c)
        self.coeffs = coeffs

        eixos = operator.grid.axes
        self._h = [float(ax.hp[0]) for ax in eixos]
        self._interior = [ax.n - 2 for ax in eixos]
        if any(m < 1 for m in self._interior):
            raise ValueError("Solver DST exige ao menos 3 nós por eixo.")

        autov = []
        for m, h, ax in zip(self._interior, self._h, eixos):
            k = np.arange(1, m + 1)
            autov.append(-(4.0 / h ** 2)
                         * np.sin(k * np.pi / (2.0 * (ax.n - 1))) ** 2)

        self._denom = []
        for axx, ayy in coeffs:
            if self.ndim == 2:
                lam = axx * autov[0][:, None] + ayy * autov[1][None, :]
            else:
                lam = axx * autov[0]
            self._denom.append(1.0 - self.c * lam)

        self._eixos_t = tuple(range(1, self.ndim + 1))

    def solve(self, rhs):
        r = np.asarray(rhs, dtype=np.float64).reshape(
            (self.n_funcs,) + self.shape
        )
        u = r.copy()

        if self.ndim == 2:
            b = r[:, 1:-1, 1:-1].copy()
            for f, (axx, ayy) in enumerate(self.coeffs):
                cx = self.c * axx / self._h[0] ** 2
                cy = self.c * ayy / self._h[1] ** 2
                b[f, 0, :] += cx * r[f, 0, 1:-1]
                b[f, -1, :] += cx * r[f, -1, 1:-1]
                b[f, :, 0] += cy * r[f, 1:-1, 0]
                b[f, :, -1] += cy * r[f, 1:-1, -1]
        else:
            b = r[:, 1:-1].copy()
            for f, (axx, _) in enumerate(self.coeffs):
                cx = self.c * axx / self._h[0] ** 2
                b[f, 0] += cx * r[f, 0]
                b[f, -1] += cx * r[f, -1]

        bh = dstn(b, type=1, axes=self._eixos_t, norm='ortho')
        for f in range(self.n_funcs):
            bh[f] /= self._denom[f]
        sol = idstn(bh, type=1, axes=self._eixos_t, norm='ortho')

        if self.ndim == 2:
            u[:, 1:-1, 1:-1] = sol
        else:
            u[:, 1:-1] = sol
        return u.reshape(-1)


def build(operator, pdes, c):
    """Return a FastDiffusionSolver, or None when the structure disqualifies."""
    if not applicable(operator, pdes):
        return None
    coeffs = _constant_laplacian_coeffs(operator)
    if coeffs is None:
        return None
    try:
        return FastDiffusionSolver(operator, coeffs, c)
    except ValueError:
        return None

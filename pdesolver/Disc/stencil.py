from typing import List, Tuple

import numpy as np
import scipy.sparse as sp_sparse
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr

from ..Auxs.FuncAux import build_func_map
from ..Auxs.FuncAux import repl_symbol as _repl_symbol
from .Disc import bc_coeffs, dirichlet_map

_SUFFIXES_1D = ("", "_x", "_xx")
_SUFFIXES_2D = ("", "_x", "_xx", "_y", "_yy", "_xy")

_OFFSETS = {
    "":    ((0, 0),),
    "_x":  ((-1, 0), (0, 0), (1, 0)),
    "_xx": ((-1, 0), (0, 0), (1, 0)),
    "_y":  ((0, -1), (0, 0), (0, 1)),
    "_yy": ((0, -1), (0, 0), (0, 1)),
    "_xy": tuple((dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)),
}


def _to_field_form(pdes, str_sp_vars: str) -> Tuple[List[str], List[str]]:
    xd_var = pdes.xs(pdes.funcs)
    eqrs = [eq.split("=")[1] for eq in pdes.eqs]

    for j in range(len(eqrs)):
        for i, func in enumerate(pdes.funcs):
            eqrs[j] = eqrs[j].replace(str(func), f"{xd_var[i]}{str_sp_vars}")

    sx = str_sp_vars[0]
    sy = str_sp_vars[1] if len(str_sp_vars) == 2 else None

    for j in range(len(eqrs)):
        for v in xd_var:
            if sy is not None:
                eqrs[j] = eqrs[j].replace(
                    f"d2{v}{str_sp_vars}/d{sx}d{sy}", f"{v}_xy"
                )
                eqrs[j] = eqrs[j].replace(
                    f"d2{v}{str_sp_vars}/d{sy}d{sx}", f"{v}_xy"
                )
                eqrs[j] = eqrs[j].replace(
                    f"d2{v}{str_sp_vars}/d{sy}2", f"{v}_yy"
                )
                eqrs[j] = eqrs[j].replace(f"d{v}{str_sp_vars}/d{sy}", f"{v}_y")
            eqrs[j] = eqrs[j].replace(f"d2{v}{str_sp_vars}/d{sx}2", f"{v}_xx")
            eqrs[j] = eqrs[j].replace(f"d{v}{str_sp_vars}/d{sx}", f"{v}_x")
            eqrs[j] = eqrs[j].replace(f"{v}{str_sp_vars}", v)
        eqrs[j] = _repl_symbol(eqrs[j], sx, "CX")
        if sy is not None:
            eqrs[j] = _repl_symbol(eqrs[j], sy, "CY")
        eqrs[j] = _repl_symbol(eqrs[j], pdes.ivars[0], "TT")

    return eqrs, xd_var


def _bc_lambda(expr_str: str):
    t_sym, x_sym, y_sym = sp.symbols("t x y")
    return sp.lambdify((t_sym, x_sym, y_sym), parse_expr(str(expr_str)),
                       modules="numpy")


def group_constraints(constraints: dict) -> List[tuple]:
    groups: dict = {}
    for idx, info in constraints.items():
        groups.setdefault(str(info["expr"]), []).append(
            (idx, info["x"], info["y"])
        )
    out = []
    for expr, items in groups.items():
        out.append((
            _bc_lambda(expr),
            np.array([i for i, _, _ in items], dtype=np.int64),
            np.array([x for _, x, _ in items], dtype=np.float64),
            np.array([y for _, _, y in items], dtype=np.float64),
        ))
    return out


def _axis_colors(n: int, periodic: bool) -> np.ndarray:
    idx = np.arange(n)
    colors = idx % 3
    if periodic and n % 3 != 0:
        colors = np.where(idx >= 3 * (n // 3), 3 + colors, colors)
    return colors


def _axis_targets(idx: np.ndarray, n: int, periodic: bool, d: int) -> List:
    if d == 0:
        return [idx]
    t = idx + d
    inside = (t >= 0) & (t < n)
    cands = [np.where(inside, t, idx)]
    if not bool(inside.all()):
        if periodic:
            cands.append(t % n)
        else:
            cands.append(np.clip(idx - d, 0, n - 1))
            cands.append(np.clip(t, 0, n - 1))
    return cands


class StencilOperator:

    def __init__(self, pdes, grid, method="central", xp=None, exprs=None):
        self.grid = grid
        self.ndim = grid.ndim
        self.shape = grid.shape
        self.method = method
        self.xp = xp if xp is not None else np
        self.n_funcs = len(pdes.funcs)
        self.block = int(np.prod(grid.shape))
        self.size = self.block * self.n_funcs

        str_sp_vars = "".join(pdes.sp_vars)
        if self.ndim != len(str_sp_vars):
            raise ValueError(
                f"A malha tem {self.ndim} eixo(s), mas a EDP declara "
                f"{len(str_sp_vars)} variável(is) espacial(is)."
            )

        eqrs, xd_var = _to_field_form(pdes, str_sp_vars)
        self._xd_var = xd_var

        suffixes = _SUFFIXES_2D if self.ndim == 2 else _SUFFIXES_1D
        sym_map = {}
        for v in xd_var:
            for suf in suffixes:
                sym_map[f"{v}{suf}"] = sp.Symbol(f"{v}{suf}")
        for name in ("CX", "CY", "TT"):
            sym_map[name] = sp.Symbol(name)

        if exprs is None:
            self.exprs = [parse_expr(e, local_dict=sym_map) for e in eqrs]
        else:
            self.exprs = list(exprs)
        self._sym_map = sym_map

        free = set()
        for e in self.exprs:
            free |= {s.name for s in e.free_symbols}
        for v in xd_var:
            for suf in _SUFFIXES_2D:
                if suf not in suffixes and f"{v}{suf}" in free:
                    raise ValueError(
                        f"O termo '{suf.strip('_')}' exige duas variáveis "
                        f"espaciais, mas a EDP declara apenas {self.ndim}."
                    )

        self._fields = [
            (ki, v, suf)
            for ki, v in enumerate(xd_var)
            for suf in suffixes
            if f"{v}{suf}" in free
        ]

        args = [sym_map["TT"], sym_map["CX"]]
        if self.ndim == 2:
            args.append(sym_map["CY"])
        args += [sym_map[f"{v}{suf}"] for _, v, suf in self._fields]

        self._f = sp.lambdify(
            tuple(args), tuple(self.exprs),
            modules=[build_func_map(self.xp), self.xp],
        )

        self._pdes = pdes
        self._setup_weights()
        self._setup_coords()
        self._setup_bcs(pdes)
        self._setup_buffers()

    def _setup_buffers(self):
        xp = self.xp
        pad = tuple(s + 2 for s in self.shape)
        self._pbuf = [xp.empty(pad, dtype=xp.float64)
                      for _ in range(self.n_funcs)]

    def to_device(self, xp):
        if xp is self.xp:
            return self
        return StencilOperator(
            self._pdes, self.grid, method=self.method, xp=xp,
            exprs=self.exprs,
        )

    def _field_symbols(self):
        suffixes = _SUFFIXES_2D if self.ndim == 2 else _SUFFIXES_1D
        return {sp.Symbol(f"{v}{suf}")
                for v in self._xd_var for suf in suffixes}

    def _term_is_linear(self, termo, campos) -> bool:
        presentes = [s for s in campos if s in termo.free_symbols]
        if not presentes:
            return True
        for s in presentes:
            if sp.diff(termo, s).free_symbols & campos:
                return False
        return True

    def _term_order(self, termo, campos) -> int:
        presentes = {s.name for s in termo.free_symbols} & {
            s.name for s in campos
        }
        ordem = 0
        for nome in presentes:
            if nome.endswith(("_xx", "_yy", "_xy")):
                ordem = max(ordem, 2)
            elif nome.endswith(("_x", "_y")):
                ordem = max(ordem, 1)
        return ordem

    def term_stiffness(self, nk: int = 24) -> List[dict]:
        from ..Analysis.stability import symbol_eigenvalues

        campos = self._field_symbols()
        registros = []
        for j, expr in enumerate(self.exprs):
            for termo in sp.Add.make_args(sp.expand(expr)):
                se = [sp.Integer(0)] * len(self.exprs)
                se[j] = termo
                sonda = StencilOperator(
                    self._pdes, self.grid, method=self.method, exprs=se
                )
                lams, _ = symbol_eigenvalues(sonda, nk=nk)
                registros.append({
                    'eq': j,
                    'termo': termo,
                    'ordem': self._term_order(termo, campos),
                    'lambda_max': float(np.max(np.abs(lams))) if lams.size else 0.0,
                    'linear': self._term_is_linear(termo, campos),
                })
        return registros

    def split_stiff(self, ordem_min: int = 2, nk: int = 24):
        registros = self.term_stiffness(nk=nk)
        if not registros:
            zero = [sp.Integer(0)] * len(self.exprs)
            return self, StencilOperator(
                self._pdes, self.grid, method=self.method, xp=self.xp,
                exprs=zero,
            ), registros

        rigidos = [[] for _ in self.exprs]
        brandos = [[] for _ in self.exprs]
        for r in registros:
            r['rigido'] = bool(r['linear'] and r['ordem'] >= ordem_min)
            (rigidos if r['rigido'] else brandos)[r['eq']].append(r['termo'])

        def _monta(partes):
            return [sp.Add(*p) if p else sp.Integer(0) for p in partes]

        op_rigido = StencilOperator(
            self._pdes, self.grid, method=self.method, xp=self.xp,
            exprs=_monta(rigidos),
        )
        op_brando = StencilOperator(
            self._pdes, self.grid, method=self.method, xp=self.xp,
            exprs=_monta(brandos),
        )
        return op_rigido, op_brando, registros

    def _setup_weights(self):
        xp = self.xp
        ax = self.grid.axes[0]
        sx = (-1, 1) if self.ndim == 2 else (-1,)
        self._vxm, self._vxc, self._vxp = (
            xp.asarray(w.reshape(sx)) for w in ax.w1(self.method)
        )
        self._wxm, self._wxc, self._wxp = (
            xp.asarray(w.reshape(sx)) for w in ax.w2()
        )
        if self.ndim == 2:
            ay = self.grid.axes[1]
            self._vym, self._vyc, self._vyp = (
                xp.asarray(w.reshape(1, -1)) for w in ay.w1(self.method)
            )
            self._wym, self._wyc, self._wyp = (
                xp.asarray(w.reshape(1, -1)) for w in ay.w2()
            )

    def _setup_coords(self):
        xp = self.xp
        ax = self.grid.axes[0]
        if self.ndim == 2:
            ay = self.grid.axes[1]
            self._cx = xp.asarray(ax.nodes.reshape(-1, 1))
            self._cy = xp.asarray(ay.nodes.reshape(1, -1))
        else:
            self._cx = xp.asarray(ax.nodes)
            self._cy = None

    def _setup_bcs(self, pdes):
        ax = self.grid.axes[0]
        sides = ["west", "east"]
        if self.ndim == 2:
            sides += ["south", "north"]

        edges = {
            "west": (float(ax.nodes[0]), None, float(ax.hp[0])),
            "east": (float(ax.nodes[-1]), None, float(ax.hm[-1])),
        }
        if self.ndim == 2:
            ay = self.grid.axes[1]
            edges["west"] = (float(ax.nodes[0]), ay.nodes, float(ax.hp[0]))
            edges["east"] = (float(ax.nodes[-1]), ay.nodes, float(ax.hm[-1]))
            edges["south"] = (ax.nodes, float(ay.nodes[0]), float(ay.hp[0]))
            edges["north"] = (ax.nodes, float(ay.nodes[-1]), float(ay.hm[-1]))

        self._bc = []
        for pde in pdes.pdes:
            entry = {}
            for side in sides:
                kind = str(getattr(pde, f"{side}_bd", "Neumann")).lower()
                raw = getattr(pde, f"{side}_func_bd", "0")
                a, b, g = bc_coeffs(kind, raw)
                ex, ey, h = edges[side]
                entry[side] = (
                    kind,
                    _bc_lambda(a), _bc_lambda(b), _bc_lambda(g),
                    h, ex, ey if ey is not None else 0.0,
                )
            self._bc.append(entry)

        bd_kind = {
            s: [str(getattr(p, f"{s}_bd", "Neumann")) for p in pdes.pdes]
            for s in ("west", "east", "south", "north")
        }
        bd_func = {
            s: [str(getattr(p, f"{s}_func_bd", "0")) for p in pdes.pdes]
            for s in ("west", "east", "south", "north")
        }
        self.dirichlet_constraints = dirichlet_map(
            self.grid, bd_kind, bd_func, self.n_funcs
        )
        self.neumann_constraints: dict = {}
        self._dirichlet_groups = group_constraints(self.dirichlet_constraints)

    def _ghost(self, k, side, inward, bnd, t):
        kind, fa, fb, fg, h, ex, ey = self._bc[k][side]
        if kind == "dirichlet":
            return inward
        gv = self.xp.asarray(fg(t, ex, ey))
        if kind == "neumann":
            return inward + 2.0 * h * gv
        av = self.xp.asarray(fa(t, ex, ey))
        bv = self.xp.asarray(fb(t, ex, ey))
        return inward + (2.0 * h / bv) * (gv - av * bnd)

    def _pad(self, U, k, t):
        ax = self.grid.axes[0]
        P = self._pbuf[k]

        if self.ndim == 1:
            P[1:-1] = U
            if ax.periodic:
                P[0] = U[-1]
                P[-1] = U[0]
            else:
                P[0] = self._ghost(k, "west", U[1], U[0], t)
                P[-1] = self._ghost(k, "east", U[-2], U[-1], t)
            return P

        ay = self.grid.axes[1]
        P[1:-1, 1:-1] = U

        if ay.periodic:
            P[1:-1, 0] = U[:, -1]
            P[1:-1, -1] = U[:, 0]
        else:
            P[1:-1, 0] = self._ghost(k, "south", U[:, 1], U[:, 0], t)
            P[1:-1, -1] = self._ghost(k, "north", U[:, -2], U[:, -1], t)

        if ax.periodic:
            P[0, :] = P[-2, :]
            P[-1, :] = P[1, :]
        else:
            P[0, 1:-1] = self._ghost(k, "west", U[1, :], U[0, :], t)
            P[-1, 1:-1] = self._ghost(k, "east", U[-2, :], U[-1, :], t)
            P[0, 0] = P[0, 1]
            P[0, -1] = P[0, -2]
            P[-1, 0] = P[-1, 1]
            P[-1, -1] = P[-1, -2]
        return P

    def _derivs(self, P, needed):
        out = {}
        if self.ndim == 1:
            C, Xm, Xp = P[1:-1], P[:-2], P[2:]
            if "" in needed:
                out[""] = C
            if "_x" in needed:
                out["_x"] = self._vxm * Xm + self._vxc * C + self._vxp * Xp
            if "_xx" in needed:
                out["_xx"] = self._wxm * Xm + self._wxc * C + self._wxp * Xp
            return out

        C = P[1:-1, 1:-1]
        Xm, Xp = P[:-2, 1:-1], P[2:, 1:-1]
        Ym, Yp = P[1:-1, :-2], P[1:-1, 2:]
        if "" in needed:
            out[""] = C
        if "_x" in needed:
            out["_x"] = self._vxm * Xm + self._vxc * C + self._vxp * Xp
        if "_xx" in needed:
            out["_xx"] = self._wxm * Xm + self._wxc * C + self._wxp * Xp
        if "_y" in needed:
            out["_y"] = self._vym * Ym + self._vyc * C + self._vyp * Yp
        if "_yy" in needed:
            out["_yy"] = self._wym * Ym + self._wyc * C + self._wyp * Yp
        if "_xy" in needed:
            dx = self._vxm * P[:-2, :] + self._vxc * P[1:-1, :] + self._vxp * P[2:, :]
            out["_xy"] = (
                self._vym * dx[:, :-2]
                + self._vyc * dx[:, 1:-1]
                + self._vyp * dx[:, 2:]
            )
        return out

    def _apply_dirichlet(self, flat, t):
        for f, idxs, xs, ys in self._dirichlet_groups:
            vals = f(t, xs, ys)
            flat[idxs] = self.xp.asarray(
                np.broadcast_to(np.asarray(vals, dtype=np.float64), idxs.shape)
            )

    def __call__(self, t, u):
        xp = self.xp
        state = xp.asarray(u, dtype=xp.float64).reshape(
            (self.n_funcs,) + tuple(self.shape)
        )

        fields = {}
        for k in range(self.n_funcs):
            needed = {suf for ki, _, suf in self._fields if ki == k}
            if not needed:
                continue
            P = self._pad(state[k], k, float(t))
            for suf, arr in self._derivs(P, needed).items():
                fields[(k, suf)] = arr

        args = [float(t), self._cx]
        if self.ndim == 2:
            args.append(self._cy)
        args += [fields[(ki, suf)] for ki, _, suf in self._fields]

        raw = self._f(*args)
        out = xp.empty((self.n_funcs,) + tuple(self.shape), dtype=xp.float64)
        for j in range(self.n_funcs):
            out[j] = raw[j]

        flat = out.reshape(-1)
        self._apply_dirichlet(flat, float(t))
        return flat

    def sparsity(self) -> Tuple[sp_sparse.csr_matrix, np.ndarray, int]:
        Nx = self.shape[0]
        Ny = self.shape[1] if self.ndim == 2 else 1
        ax = self.grid.axes[0]
        py = self.grid.axes[1].periodic if self.ndim == 2 else False

        if self.ndim == 2:
            gi, gj = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing="ij")
            gi, gj = gi.ravel(), gj.ravel()
        else:
            gi = np.arange(Nx)
            gj = np.zeros(Nx, dtype=np.int64)

        rows, cols = [], []
        for j_eq, expr in enumerate(self.exprs):
            names = {s.name for s in expr.free_symbols}
            base = j_eq * self.block + gi * Ny + gj
            for k, v in enumerate(self._xd_var):
                offs = set()
                for suf in (_SUFFIXES_2D if self.ndim == 2 else _SUFFIXES_1D):
                    if f"{v}{suf}" in names:
                        offs |= set(_OFFSETS[suf])
                for dx, dy in offs:
                    for tx in _axis_targets(gi, Nx, ax.periodic, dx):
                        for ty in _axis_targets(gj, Ny, py, dy):
                            rows.append(base)
                            cols.append(k * self.block + tx * Ny + ty)

        if not rows:
            pattern = sp_sparse.csr_matrix(
                (self.size, self.size), dtype=bool
            )
            return pattern, np.zeros(self.size, dtype=int), 1

        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        pattern = sp_sparse.coo_matrix(
            (np.ones(rows.size, dtype=bool), (rows, cols)),
            shape=(self.size, self.size),
        ).tocsr()
        pattern.data[:] = True

        cx = _axis_colors(Nx, ax.periodic)
        nx_c = int(cx.max()) + 1
        if self.ndim == 2:
            cy = _axis_colors(Ny, py)
            ny_c = int(cy.max()) + 1
            node = (cx[gi] * ny_c + cy[gj])
            per_node = nx_c * ny_c
        else:
            node = cx[gi]
            per_node = nx_c

        colors = np.empty(self.size, dtype=int)
        for k in range(self.n_funcs):
            colors[k * self.block:(k + 1) * self.block] = node + k * per_node
        return pattern, colors, per_node * self.n_funcs

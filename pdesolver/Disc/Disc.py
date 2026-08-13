import re
from typing import List, Tuple

from ..Auxs.FuncAux import repl_symbol as _repl_symbol
from .boundaries import get_boundary
from .grid import build_grid

_STENCILS = {
    "d2x": "(WXm_ii*{v}_i-1_j + WXc_ii*{v}_ii_j + WXp_ii*{v}_i+1_j)",
    "d1x": "(VXm_ii*{v}_i-1_j + VXc_ii*{v}_ii_j + VXp_ii*{v}_i+1_j)",
    "d2y": "(WYm_j*{v}_ii_j-1 + WYc_j*{v}_ii_j + WYp_j*{v}_ii_j+1)",
    "d1y": "(VYm_j*{v}_ii_j-1 + VYc_j*{v}_ii_j + VYp_j*{v}_ii_j+1)",
    "d2xy": (
        "(VXm_ii*(VYm_j*{v}_i-1_j-1 + VYc_j*{v}_i-1_j + VYp_j*{v}_i-1_j+1)"
        " + VXc_ii*(VYm_j*{v}_ii_j-1 + VYc_j*{v}_ii_j + VYp_j*{v}_ii_j+1)"
        " + VXp_ii*(VYm_j*{v}_i+1_j-1 + VYc_j*{v}_i+1_j + VYp_j*{v}_i+1_j+1))"
    ),
}

_TOKEN_PATTERN = re.compile(r"\b([WV][XY][mcp]|C[XY])_(\d+)\b")

_FLUX = ("neumann", "robin")


def _build_discretized_eqs(
    eqrs: List[str],
    xd_var: List[str],
    str_sp_vars: str,
) -> List[str]:

    s = _STENCILS

    if len(str_sp_vars) == 1:
        for eq in eqrs:
            if f"/d{str_sp_vars[0]}d" in eq:
                raise ValueError(
                    "O termo misto exige duas variáveis espaciais, mas a EDP "
                    "declara apenas uma."
                )

    for j in range(len(eqrs)):
        if len(str_sp_vars) == 2:
            sx, sy = str_sp_vars[0], str_sp_vars[1]
            for v in xd_var:
                eqrs[j] = eqrs[j].replace(
                    f"d2{v}{str_sp_vars}/d{sx}d{sy}", s["d2xy"].format(v=v)
                )
                eqrs[j] = eqrs[j].replace(
                    f"d2{v}{str_sp_vars}/d{sy}d{sx}", s["d2xy"].format(v=v)
                )

        for k, sp_var in enumerate(str_sp_vars):
            for v in xd_var:
                if k == 0:
                    eqrs[j] = eqrs[j].replace(
                        f"d2{v}{str_sp_vars}/d{sp_var}2", s["d2x"].format(v=v)
                    )
                    eqrs[j] = eqrs[j].replace(
                        f"d{v}{str_sp_vars}/d{sp_var}", s["d1x"].format(v=v)
                    )
                elif k == 1:
                    eqrs[j] = eqrs[j].replace(
                        f"d2{v}{str_sp_vars}/d{sp_var}2", s["d2y"].format(v=v)
                    )
                    eqrs[j] = eqrs[j].replace(
                        f"d{v}{str_sp_vars}/d{sp_var}", s["d1y"].format(v=v)
                    )

        for v in xd_var:
            eqrs[j] = eqrs[j].replace(f"{v}{str_sp_vars}", f"{v}_ii_j")

    for j in range(len(eqrs)):
        eqrs[j] = _repl_symbol(eqrs[j], str_sp_vars[0], "CX_ii")
        if len(str_sp_vars) == 2:
            eqrs[j] = _repl_symbol(eqrs[j], str_sp_vars[1], "CY_j")

    return eqrs


def _active_sides(i: int, j: int, ax, ay) -> List[str]:
    asides = []
    if not ax.periodic:
        if i == 0:
            asides.append("west")
        if i == ax.n - 1:
            asides.append("east")
    if not ay.periodic:
        if j == 0:
            asides.append("south")
        if j == ay.n - 1:
            asides.append("north")
    return asides


def dirichlet_map(grid, bd_kind: dict, bd_func: dict, n_funcs: int) -> dict:
    constraints: dict = {}

    if grid.ndim == 2:
        ax, ay = grid.axes[0], grid.axes[1]
        Nx, Ny = ax.n, ay.n
        any_flux = any(
            bd_kind[s][f].lower() in _FLUX
            for s in bd_kind for f in range(n_funcs)
        )
        if any_flux and (Nx < 3 or Ny < 3):
            raise ValueError(
                f"Neumann/Robin 2D exige min(Nx, Ny) >= 3. Recebido: "
                f"Nx={Nx}, Ny={Ny}."
            )
        for func in range(n_funcs):
            for i in range(Nx):
                for j in range(Ny):
                    asides = _active_sides(i, j, ax, ay)
                    if not asides:
                        continue
                    kinds = [bd_kind[s][func].lower() for s in asides]
                    if "dirichlet" in kinds:
                        sd = asides[kinds.index("dirichlet")]
                        constraints[func * Nx * Ny + i * Ny + j] = {
                            "expr": bd_func[sd][func],
                            "x": float(ax.nodes[i]),
                            "y": float(ay.nodes[j]),
                        }
        return constraints

    ax = grid.axes[0]
    Nx = ax.n
    any_flux_1d = any(
        bd_kind["west"][f].lower() in _FLUX or bd_kind["east"][f].lower() in _FLUX
        for f in range(n_funcs)
    )
    if any_flux_1d and Nx < 3:
        raise ValueError("Neumann/Robin 1D exige Nx >= 3.")
    for func in range(n_funcs):
        offset = func * Nx
        if ax.periodic:
            continue
        if bd_kind["west"][func].lower() == "dirichlet":
            constraints[offset] = {
                "expr": bd_func["west"][func],
                "x": float(ax.nodes[0]),
                "y": 0.0,
            }
        if bd_kind["east"][func].lower() == "dirichlet":
            constraints[offset + Nx - 1] = {
                "expr": bd_func["east"][func],
                "x": float(ax.nodes[Nx - 1]),
                "y": 0.0,
            }
    return constraints


def _wrap(idx: int, n: int, periodic: bool) -> int:
    if periodic:
        return idx % n
    return idx


def _axis_range(n: int, periodic: bool) -> List[int]:
    if periodic:
        return list(range(n))
    return list(range(1, n - 1))


def _expand_x(eq: str, i: int, n: int, periodic: bool) -> str:
    return (
        eq
        .replace("i+1", str(_wrap(i + 1, n, periodic)))
        .replace("i-1", str(_wrap(i - 1, n, periodic)))
        .replace("i-2", str(_wrap(i - 2, n, periodic)))
        .replace("i+2", str(_wrap(i + 2, n, periodic)))
        .replace("ii",  str(i))
    )


def _expand_y(eq: str, j: int, n: int, periodic: bool) -> str:
    return (
        eq
        .replace("j+1", str(_wrap(j + 1, n, periodic)))
        .replace("j-1", str(_wrap(j - 1, n, periodic)))
        .replace("j-2", str(_wrap(j - 2, n, periodic)))
        .replace("j+2", str(_wrap(j + 2, n, periodic)))
        .replace("j",   str(j))
    )


def _expand_indices(
    eqrs: List[str],
    grid,
    str_sp_vars: str,
) -> List[List[str]]:

    Nx = grid.shape[0]
    px = grid.axes[0].periodic

    partial = []
    for eq in eqrs:
        partial.append([_expand_x(eq, i, Nx, px) for i in _axis_range(Nx, px)])

    list_eq: List[List[str]] = [[] for _ in partial]

    if len(str_sp_vars) == 2:
        Ny = grid.shape[1]
        py = grid.axes[1].periodic
        for j_eq, row in enumerate(partial):
            for eq_i in row:
                for k in _axis_range(Ny, py):
                    list_eq[j_eq].append(_expand_y(eq_i, k, Ny, py))
    else:
        for j_eq, row in enumerate(partial):
            for eq_i in row:
                list_eq[j_eq].append(eq_i.replace("j", "0"))

    return list_eq


def _weight_tables(grid, method: str, str_sp_vars: str) -> dict:
    tables = {}
    ax = grid.axes[0]
    tables["WXm"], tables["WXc"], tables["WXp"] = ax.w2()
    tables["VXm"], tables["VXc"], tables["VXp"] = ax.w1(method)
    tables["CX"] = ax.nodes
    if len(str_sp_vars) == 2:
        ay = grid.axes[1]
        tables["WYm"], tables["WYc"], tables["WYp"] = ay.w2()
        tables["VYm"], tables["VYc"], tables["VYp"] = ay.w1(method)
        tables["CY"] = ay.nodes
    return tables


def _substitute_tokens(eq: str, tables: dict) -> str:
    def _repl(match):
        return f"({float(tables[match.group(1)][int(match.group(2))])!r})"

    return _TOKEN_PATTERN.sub(_repl, eq)


def bc_coeffs(kind: str, func_bd_str: str) -> Tuple[str, str, str]:
    k = (kind or "").lower()
    if k == "robin":
        parts = [p.strip() for p in str(func_bd_str).split(";")]
        if len(parts) == 3:
            return parts[0], parts[1], parts[2]
        if len(parts) == 2:
            return parts[0], parts[1], "0"
        return "0", "1", str(func_bd_str)
    return "0", "1", str(func_bd_str)


def _ghost_repl(inward: str, bnd: str, h: float, a: str, b: str, g: str) -> str:
    if a.strip() == "0" and b.strip() == "1":
        return f"({inward} + 2*{h}*({g}))"
    return f"({inward} + (2*{h}/({b}))*(({g}) - ({a})*{bnd}))"


def _expand_node_2d(template: str, i: int, j: int, grid) -> str:
    s = _expand_x(template, i, grid.shape[0], grid.axes[0].periodic)
    return _expand_y(s, j, grid.shape[1], grid.axes[1].periodic)


def _bd_func_xy(expr: str, x_val, y_val, str_sp_vars: str) -> str:
    out = _repl_symbol(expr, str_sp_vars[0], str(x_val))
    out = _repl_symbol(out, str_sp_vars[1], str(y_val))
    return out


def _ghost_eq_1d(
    template: str,
    side: str,
    grid,
    n_funcs: int,
    west_bd: List[str],
    east_bd: List[str],
    west_func_bd: List[str],
    east_func_bd: List[str],
    str_sp_vars: str,
) -> str:

    ax = grid.axes[0]
    Nx = ax.n
    i = 0 if side == "west" else Nx - 1
    hx = float(ax.hp[0]) if side == "west" else float(ax.hm[Nx - 1])

    eq = _expand_x(template, i, Nx, ax.periodic).replace("j", "0")

    for k in range(n_funcs):
        if side == "west":
            a, b, g = bc_coeffs(west_bd[k], west_func_bd[k])
            xc = str(float(ax.nodes[0]))
            tok = f"XX{k}_-1_0"
            inward = f"XX{k}_1_0"
            bnd = f"XX{k}_0_0"
        else:
            a, b, g = bc_coeffs(east_bd[k], east_func_bd[k])
            xc = str(float(ax.nodes[Nx - 1]))
            tok = f"XX{k}_{Nx}_0"
            inward = f"XX{k}_{Nx - 2}_0"
            bnd = f"XX{k}_{Nx - 1}_0"
        a = _repl_symbol(a, str_sp_vars[0], xc)
        b = _repl_symbol(b, str_sp_vars[0], xc)
        g = _repl_symbol(g, str_sp_vars[0], xc)
        eq = eq.replace(tok, _ghost_repl(inward, bnd, hx, a, b, g))
    return eq


def _ghost_eq_2d(
    template: str,
    i: int,
    j: int,
    grid,
    n_funcs: int,
    bd_kind: dict,
    bd_func: dict,
    str_sp_vars: str,
) -> str:

    ax, ay = grid.axes[0], grid.axes[1]
    Nx, Ny = ax.n, ay.n
    x_val = float(ax.nodes[i])
    y_val = float(ay.nodes[j])

    eq = _expand_node_2d(template, i, j, grid)

    sides = []
    if i == 0 and not ax.periodic:
        sides.append(("west", float(ax.hp[0]), f"XX{{k}}_-1_{j}",
                      f"XX{{k}}_1_{j}", f"XX{{k}}_0_{j}"))
    if i == Nx - 1 and not ax.periodic:
        sides.append(("east", float(ax.hm[Nx - 1]), f"XX{{k}}_{Nx}_{j}",
                      f"XX{{k}}_{Nx - 2}_{j}", f"XX{{k}}_{Nx - 1}_{j}"))
    if j == 0 and not ay.periodic:
        sides.append(("south", float(ay.hp[0]), f"XX{{k}}_{i}_-1",
                      f"XX{{k}}_{i}_1", f"XX{{k}}_{i}_0"))
    if j == Ny - 1 and not ay.periodic:
        sides.append(("north", float(ay.hm[Ny - 1]), f"XX{{k}}_{i}_{Ny}",
                      f"XX{{k}}_{i}_{Ny - 2}", f"XX{{k}}_{i}_{Ny - 1}"))

    for side, h, tok, inward, bnd in sides:
        for k in range(n_funcs):
            a, b, g = bc_coeffs(bd_kind[side][k], bd_func[side][k])
            a = _bd_func_xy(a, x_val, y_val, str_sp_vars)
            b = _bd_func_xy(b, x_val, y_val, str_sp_vars)
            g = _bd_func_xy(g, x_val, y_val, str_sp_vars)
            eq = eq.replace(
                tok.format(k=k),
                _ghost_repl(inward.format(k=k), bnd.format(k=k), h, a, b, g),
            )
    return eq


def _func_from_lhs(lhs: str, funcs: List[str]) -> str:
    alvo = lhs.strip()
    for func in funcs:
        if alvo in (f"d{func}/dt", f"d{func}/d t", func):
            return func
    for func in funcs:
        if func in alvo:
            return func
    raise ValueError(
        f"Não foi possível identificar a função em '{lhs.strip()}'. "
        f"Funções do sistema: {funcs}."
    )


def _normalize_where(where, grid) -> "np.ndarray":
    import numpy as np

    if callable(where):
        mask = where(*grid.coords())
    else:
        mask = where
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != tuple(grid.shape):
        raise ValueError(
            f"A máscara da região tem forma {mask.shape}, mas a malha é "
            f"{tuple(grid.shape)}."
        )
    return mask


def _region_templates(pdes, regions, xd_var, str_sp_vars, grid):
    specs = []
    for r in regions:
        if "eq" not in r or "where" not in r:
            raise ValueError(
                "Cada região exige as chaves 'where' (máscara) e 'eq' "
                "(equação na mesma notação do PDE)."
            )
        lhs, rhs = str(r["eq"]).split("=", 1)
        fname = _func_from_lhs(lhs, pdes.funcs)
        fidx = pdes.funcs.index(fname)

        tmpl = rhs
        for i, func in enumerate(pdes.funcs):
            tmpl = tmpl.replace(str(func), f"{xd_var[i]}{str_sp_vars}")
        tmpl = _build_discretized_eqs([tmpl], xd_var, str_sp_vars)[0]

        specs.append((fidx, _normalize_where(r["where"], grid), tmpl))
    return specs


def _region_equation(specs, func: int, i: int, j: int, grid, str_sp_vars):
    for fidx, mask, tmpl in specs:
        if fidx != func:
            continue
        hit = mask[i, j] if len(str_sp_vars) == 2 else mask[i]
        if not hit:
            continue
        if len(str_sp_vars) == 2:
            return _expand_node_2d(tmpl, i, j, grid)
        return _expand_x(
            tmpl, i, grid.shape[0], grid.axes[0].periodic
        ).replace("j", "0")
    return None


def _build_position_labels(n_part, str_sp_vars, n_funcs):
    positions = []

    if len(str_sp_vars) == 2:
        for func in range(n_funcs):
            aux = []
            for i in range(n_part[0]):
                for j in range(n_part[1]):
                    if i == 0:                   aux.append(f"W{func}_{i}_{j}")
                    elif i == n_part[0] - 1:     aux.append(f"E{func}_{i}_{j}")
                    elif j == 0:                 aux.append(f"S{func}_{i}_{j}")
                    elif j == n_part[1] - 1:     aux.append(f"N{func}_{i}_{j}")
                    else:                        aux.append(f"Ce{func}_{i}_{j}")
            positions.append(aux)
    else:
        positions = [[""] * n_part[0] for _ in range(n_funcs)]

    return positions


def periodic_axes(pdes) -> List[bool]:
    n_sp = len(pdes.sp_vars)
    flags = [False] * n_sp
    pairs = [("west_bd", "east_bd", 0), ("south_bd", "north_bd", 1)]
    for lo, hi, axis in pairs:
        if axis >= n_sp:
            continue
        for pde in pdes.pdes:
            a = str(getattr(pde, lo, "")).lower() == "periodic"
            b = str(getattr(pde, hi, "")).lower() == "periodic"
            if a != b:
                raise ValueError(
                    f"Contorno periódico exige o par completo: '{lo}' e '{hi}' "
                    f"devem ser 'Periodic' juntos."
                )
            if a:
                flags[axis] = True
    for axis in range(n_sp):
        kinds = set()
        for pde in pdes.pdes:
            lo, hi, _ = pairs[axis]
            kinds.add(str(getattr(pde, lo, "")).lower() == "periodic")
        if len(kinds) > 1:
            raise ValueError(
                f"Todas as equações devem concordar sobre a periodicidade do "
                f"eixo '{pdes.sp_vars[axis]}'."
            )
    return flags


def df(
    pdes,
    n_part: List[int],
    west_bd=None,
    method:   str = "forward",
    north_bd=None,
    south_bd=None,
    east_bd=None,
    north_func_bd=None,
    south_func_bd=None,
    west_func_bd=None,
    east_func_bd=None,
    north_alpha_bd: str = "0",
    south_alpha_bd: str = "0",
    east_alpha_bd:  str = "0",
    north_beta_bd:  str = "1",
    south_beta_bd:  str = "1",
    east_beta_bd:   str = "1",
    grid=None,
    regions=None,
) -> Tuple[List[str], List[str], dict, dict]:

    n_funcs_total = len(pdes.funcs)

    def _as_list(val, default="neumann"):
        if val is None:
            return [default] * n_funcs_total
        if isinstance(val, list):
            return val
        return [val] * n_funcs_total

    west_bd       = _as_list(west_bd,  "neumann")
    east_bd       = _as_list(east_bd,  "neumann")
    north_bd      = _as_list(north_bd, "neumann")
    south_bd      = _as_list(south_bd, "neumann")
    west_func_bd  = _as_list(west_func_bd,  "0")
    east_func_bd  = _as_list(east_func_bd,  "0")
    north_func_bd = _as_list(north_func_bd, "0")
    south_func_bd = _as_list(south_func_bd, "0")

    xd_var = pdes.xs(pdes.funcs)
    eqrs = [eq.split("=")[1] for eq in pdes.eqs]
    str_sp_vars = "".join(pdes.sp_vars)

    if grid is None:
        grid = build_grid(
            pdes.pdes[0].ivar_boundary, n_part, periodic=periodic_axes(pdes)
        )

    for j in range(len(eqrs)):
        for i, func in enumerate(pdes.funcs):
            eqrs[j] = eqrs[j].replace(str(func), f"{xd_var[i]}{str_sp_vars}")

    eqrs = _build_discretized_eqs(eqrs, xd_var, str_sp_vars)

    region_specs = _region_templates(
        pdes, regions or [], xd_var, str_sp_vars, grid
    )

    list_eq = _expand_indices(eqrs, grid, str_sp_vars)

    n_funcs = len(pdes.funcs)

    list_positions = _build_position_labels(n_part, str_sp_vars, n_funcs)

    if len(str_sp_vars) == 1:
        ax = grid.axes[0]
        Nx = ax.n
        for func in range(n_funcs):
            C = 0
            for i in range(Nx):
                if i == 0 and not ax.periodic:
                    if west_bd[func].lower() in _FLUX:
                        list_positions[func][i] = _ghost_eq_1d(
                            eqrs[func], "west", grid, n_funcs,
                            west_bd, east_bd, west_func_bd, east_func_bd,
                            str_sp_vars
                        )
                    else:
                        bc = get_boundary(west_bd[func], west_func_bd[func])
                        list_positions[func][i] = bc.apply(
                            "west", list_eq, n_part, xd_var, str_sp_vars
                        )[func][0]
                elif i == Nx - 1 and not ax.periodic:
                    if east_bd[func].lower() in _FLUX:
                        list_positions[func][i] = _ghost_eq_1d(
                            eqrs[func], "east", grid, n_funcs,
                            west_bd, east_bd, west_func_bd, east_func_bd,
                            str_sp_vars
                        )
                    else:
                        bc = get_boundary(
                            east_bd[func], east_func_bd[func],
                            east_alpha_bd, east_beta_bd
                        )
                        list_positions[func][i] = bc.apply(
                            "east", list_eq, n_part, xd_var, str_sp_vars
                        )[func][0]
                else:
                    list_positions[func][i] = list_eq[func][C]
                    C += 1
                    sub = _region_equation(
                        region_specs, func, i, 0, grid, str_sp_vars
                    )
                    if sub is not None:
                        list_positions[func][i] = sub

    elif len(str_sp_vars) == 2:
        ax, ay = grid.axes[0], grid.axes[1]
        Nx, Ny = ax.n, ay.n
        bd_kind = {
            "west": west_bd, "east": east_bd,
            "south": south_bd, "north": north_bd,
        }
        bd_func = {
            "west": west_func_bd, "east": east_func_bd,
            "south": south_func_bd, "north": north_func_bd,
        }
        for func in range(n_funcs):
            C = 0
            for i in range(Nx):
                for j in range(Ny):
                    pos = i * Ny + j
                    asides = _active_sides(i, j, ax, ay)
                    if not asides:
                        list_positions[func][pos] = list_eq[func][C]
                        C += 1
                        sub = _region_equation(
                            region_specs, func, i, j, grid, str_sp_vars
                        )
                        if sub is not None:
                            list_positions[func][pos] = sub
                        continue
                    kinds = [bd_kind[s][func].lower() for s in asides]
                    if "dirichlet" in kinds:
                        sd = asides[kinds.index("dirichlet")]
                        list_positions[func][pos] = _bd_func_xy(
                            bd_func[sd][func],
                            float(ax.nodes[i]), float(ay.nodes[j]),
                            str_sp_vars,
                        )
                    else:
                        list_positions[func][pos] = _ghost_eq_2d(
                            eqrs[func], i, j, grid, n_funcs,
                            bd_kind, bd_func, str_sp_vars
                        )

    d_vars: List[str] = []
    if len(str_sp_vars) == 2:
        for func in range(n_funcs):
            for i in range(grid.shape[0]):
                for j in range(grid.shape[1]):
                    d_vars.append(f"XX{func}_{i}_{j}")
    else:
        for func in range(n_funcs):
            for i in range(grid.shape[0]):
                d_vars.append(f"XX{func}_{i}_0")

    flat_list_positions: List[str] = []
    for L in list_positions:
        flat_list_positions.extend(L)

    tables = _weight_tables(grid, method, str_sp_vars)
    for i in range(len(flat_list_positions)):
        flat_list_positions[i] = _substitute_tokens(
            flat_list_positions[i], tables
        )

    dirichlet_constraints = dirichlet_map(
        grid,
        {
            "west": west_bd, "east": east_bd,
            "south": south_bd, "north": north_bd,
        },
        {
            "west": west_func_bd, "east": east_func_bd,
            "south": south_func_bd, "north": north_func_bd,
        },
        n_funcs,
    )
    neumann_constraints: dict = {}

    return flat_list_positions, d_vars, dirichlet_constraints, neumann_constraints

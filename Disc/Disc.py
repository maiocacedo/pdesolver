from typing import List, Tuple
import re
from Auxs.FuncAux import repl_symbol as _repl_symbol
from .boundaries import get_boundary

def _build_discretized_eqs(
    eqrs: List[str],
    xd_var: List[str],
    str_sp_vars: str,
    method: str,
) -> List[str]:

    SCHEMES = {
        "forward": {
            "d2x": "({v}_i+1_j - 2*{v}_ii_j + {v}_i-1_j)/ h{v}_ ** 2",
            "d1x": "({v}_i+1_j - {v}_ii_j)/ h{v}_",
            "d2y": "({v}_ii_j+1 - 2*{v}_ii_j + {v}_ii_j-1)/ h{v}_ ** 2",
            "d1y": "({v}_ii_j+1 - {v}_ii_j)/ h{v}_",
        },
        "central": {
            "d2x": "({v}_i+1_j - 2*{v}_ii_j + {v}_i-1_j)/ h{v}_ ** 2",
            "d1x": "({v}_i+1_j - {v}_i-1_j)/(2* h{v}_)",
            "d2y": "({v}_ii_j+1 - 2*{v}_ii_j + {v}_ii_j-1)/ h{v}_ ** 2",
            "d1y": "({v}_ii_j+1 - {v}_ii_j-1)/(2* h{v}_)",
        },
        "backward": {
            "d2x": "({v}_i+1_j - 2*{v}_ii_j + {v}_i-1_j)/ h{v}_ ** 2",
            "d1x": "({v}_ii_j - {v}_i-1_j)/ h{v}_",
            "d2y": "({v}_ii_j+1 - 2*{v}_ii_j + {v}_ii_j-1)/ h{v}_ ** 2",
            "d1y": "({v}_ii_j - {v}_ii_j-1)/ h{v}_",
        },
    }

    if method not in SCHEMES:
        raise ValueError(
            f"Método inválido: '{method}'. Use 'forward', 'central' ou 'backward'."
        )

    s = SCHEMES[method]

    for j in range(len(eqrs)):
        for k, sp_var in enumerate(str_sp_vars):
            for v in xd_var:
                if k == 0:
                    eqrs[j] = eqrs[j].replace(f"d2{v}{str_sp_vars}/d{sp_var}2", s["d2x"].format(v=v))
                    eqrs[j] = eqrs[j].replace(f"d{v}{str_sp_vars}/d{sp_var}",   s["d1x"].format(v=v))
                elif k == 1:
                    eqrs[j] = eqrs[j].replace(f"d2{v}{str_sp_vars}/d{sp_var}2", s["d2y"].format(v=v))
                    eqrs[j] = eqrs[j].replace(f"d{v}{str_sp_vars}/d{sp_var}",   s["d1y"].format(v=v))

        for v in xd_var:
            eqrs[j] = eqrs[j].replace(f"{v}{str_sp_vars}", f"{v}_ii_j")

    for j in range(len(eqrs)):
        eqrs[j] = _repl_symbol(eqrs[j], str_sp_vars[0], f"ii * h{xd_var[0]}_")
        if len(str_sp_vars) == 2:
            eqrs[j] = _repl_symbol(eqrs[j], str_sp_vars[1], f"j * h{xd_var[0]}_")

    return eqrs


def _expand_indices(
    eqrs: List[str],
    n_part: List[int],
    str_sp_vars: str,
) -> List[List[str]]:

    partial = []
    for eq in eqrs:
        row = [
            eq
            .replace("i+1", str(i + 1))
            .replace("i-1", str(i - 1))
            .replace("i-2", str(i - 2))
            .replace("i+2", str(i + 2))
            .replace("ii",  str(i))
            for i in range(1, n_part[0] - 1)
        ]
        partial.append(row)

    list_eq: List[List[str]] = [[] for _ in partial]

    if len(str_sp_vars) == 2:
        for j_eq, row in enumerate(partial):
            for eq_i in row:
                for k in range(1, n_part[1] - 1):
                    list_eq[j_eq].append(
                        eq_i
                        .replace("j+1", str(k + 1))
                        .replace("j-1", str(k - 1))
                        .replace("j-2", str(k - 2))
                        .replace("j+2", str(k + 2))
                        .replace("j",   str(k))
                    )
    else:
        for j_eq, row in enumerate(partial):
            for eq_i in row:
                list_eq[j_eq].append(eq_i.replace("j", "0"))

    return list_eq


def _bc_coeffs(kind: str, func_bd_str: str) -> Tuple[str, str, str]:
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


def _expand_node_2d(template: str, i: int, j: int) -> str:
    s = (
        template
        .replace("i+1", str(i + 1))
        .replace("i-1", str(i - 1))
        .replace("i-2", str(i - 2))
        .replace("i+2", str(i + 2))
        .replace("ii",  str(i))
    )
    s = (
        s
        .replace("j+1", str(j + 1))
        .replace("j-1", str(j - 1))
        .replace("j-2", str(j - 2))
        .replace("j+2", str(j + 2))
        .replace("j",   str(j))
    )
    return s


def _bd_func_xy(expr: str, x_val, y_val, str_sp_vars: str) -> str:
    out = _repl_symbol(expr, str_sp_vars[0], str(x_val))
    out = _repl_symbol(out, str_sp_vars[1], str(y_val))
    return out


def _ghost_eq_1d(
    template: str,
    side: str,
    n_part: List[int],
    n_funcs: int,
    west_bd: List[str],
    east_bd: List[str],
    west_func_bd: List[str],
    east_func_bd: List[str],
    str_sp_vars: str,
) -> str:

    Nx = n_part[0]
    hx = 1.0 / (Nx - 1)
    i = 0 if side == "west" else Nx - 1

    eq = (
        template
        .replace("i+1", str(i + 1))
        .replace("i-1", str(i - 1))
        .replace("i-2", str(i - 2))
        .replace("i+2", str(i + 2))
        .replace("ii",  str(i))
        .replace("j", "0")
    )

    for k in range(n_funcs):
        if side == "west":
            a, b, g = _bc_coeffs(west_bd[k], west_func_bd[k])
            xc = "0"
            tok = f"XX{k}_-1_0"
            inward = f"XX{k}_1_0"
            bnd = f"XX{k}_0_0"
        else:
            a, b, g = _bc_coeffs(east_bd[k], east_func_bd[k])
            xc = str((Nx - 1) * hx)
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
    n_part: List[int],
    n_funcs: int,
    bd_kind: dict,
    bd_func: dict,
    str_sp_vars: str,
) -> str:

    Nx, Ny = n_part[0], n_part[1]
    h = 1.0 / (Nx - 1)
    hx = 1.0 / (Nx - 1)
    hy = 1.0 / (Ny - 1)
    x_val = i * hx
    y_val = j * hy

    eq = _expand_node_2d(template, i, j)

    if i == 0:
        for k in range(n_funcs):
            a, b, g = _bc_coeffs(bd_kind["west"][k], bd_func["west"][k])
            a = _bd_func_xy(a, x_val, y_val, str_sp_vars)
            b = _bd_func_xy(b, x_val, y_val, str_sp_vars)
            g = _bd_func_xy(g, x_val, y_val, str_sp_vars)
            eq = eq.replace(
                f"XX{k}_-1_{j}",
                _ghost_repl(f"XX{k}_1_{j}", f"XX{k}_0_{j}", h, a, b, g),
            )
    if i == Nx - 1:
        for k in range(n_funcs):
            a, b, g = _bc_coeffs(bd_kind["east"][k], bd_func["east"][k])
            a = _bd_func_xy(a, x_val, y_val, str_sp_vars)
            b = _bd_func_xy(b, x_val, y_val, str_sp_vars)
            g = _bd_func_xy(g, x_val, y_val, str_sp_vars)
            eq = eq.replace(
                f"XX{k}_{Nx}_{j}",
                _ghost_repl(f"XX{k}_{Nx - 2}_{j}", f"XX{k}_{Nx - 1}_{j}", h, a, b, g),
            )
    if j == 0:
        for k in range(n_funcs):
            a, b, g = _bc_coeffs(bd_kind["south"][k], bd_func["south"][k])
            a = _bd_func_xy(a, x_val, y_val, str_sp_vars)
            b = _bd_func_xy(b, x_val, y_val, str_sp_vars)
            g = _bd_func_xy(g, x_val, y_val, str_sp_vars)
            eq = eq.replace(
                f"XX{k}_{i}_-1",
                _ghost_repl(f"XX{k}_{i}_1", f"XX{k}_{i}_0", h, a, b, g),
            )
    if j == Ny - 1:
        for k in range(n_funcs):
            a, b, g = _bc_coeffs(bd_kind["north"][k], bd_func["north"][k])
            a = _bd_func_xy(a, x_val, y_val, str_sp_vars)
            b = _bd_func_xy(b, x_val, y_val, str_sp_vars)
            g = _bd_func_xy(g, x_val, y_val, str_sp_vars)
            eq = eq.replace(
                f"XX{k}_{i}_{Ny}",
                _ghost_repl(f"XX{k}_{i}_{Ny - 2}", f"XX{k}_{i}_{Ny - 1}", h, a, b, g),
            )
    return eq


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

    for j in range(len(eqrs)):
        for i, func in enumerate(pdes.funcs):
            eqrs[j] = eqrs[j].replace(str(func), f"{xd_var[i]}{str_sp_vars}")


    eqrs = _build_discretized_eqs(eqrs, xd_var, str_sp_vars, method)

    list_eq = _expand_indices(eqrs, n_part, str_sp_vars)

    n_funcs = len(pdes.funcs)

    _FLUX = ("neumann", "robin")

    list_positions = _build_position_labels(n_part, str_sp_vars, n_funcs)

    if len(str_sp_vars) == 1:
        for func in range(n_funcs):
            C = 0
            for i in range(n_part[0]):
                if i == 0:
                    if west_bd[func].lower() in _FLUX:
                        list_positions[func][i] = _ghost_eq_1d(
                            eqrs[func], "west", n_part, n_funcs,
                            west_bd, east_bd, west_func_bd, east_func_bd, str_sp_vars
                        )
                    else:
                        bc = get_boundary(west_bd[func], west_func_bd[func])
                        list_positions[func][i] = bc.apply(
                            "west", list_eq, n_part, xd_var, str_sp_vars
                        )[func][0]
                elif i == n_part[0] - 1:
                    if east_bd[func].lower() in _FLUX:
                        list_positions[func][i] = _ghost_eq_1d(
                            eqrs[func], "east", n_part, n_funcs,
                            west_bd, east_bd, west_func_bd, east_func_bd, str_sp_vars
                        )
                    else:
                        bc = get_boundary(east_bd[func], east_func_bd[func], east_alpha_bd, east_beta_bd)
                        list_positions[func][i] = bc.apply(
                            "east", list_eq, n_part, xd_var, str_sp_vars
                        )[func][0]
                else:
                    list_positions[func][i] = list_eq[func][C]
                    C += 1

    elif len(str_sp_vars) == 2:
        Nx, Ny = n_part[0], n_part[1]
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
                    asides = []
                    if i == 0:      asides.append("west")
                    if i == Nx - 1: asides.append("east")
                    if j == 0:      asides.append("south")
                    if j == Ny - 1: asides.append("north")
                    if not asides:
                        list_positions[func][pos] = list_eq[func][C]
                        C += 1
                        continue
                    kinds = [bd_kind[s][func].lower() for s in asides]
                    if "dirichlet" in kinds:
                        sd = asides[kinds.index("dirichlet")]
                        hx = 1.0 / (Nx - 1)
                        hy = 1.0 / (Ny - 1)
                        list_positions[func][pos] = _bd_func_xy(
                            bd_func[sd][func], i * hx, j * hy, str_sp_vars
                        )
                    else:
                        list_positions[func][pos] = _ghost_eq_2d(
                            eqrs[func], i, j, n_part, n_funcs,
                            bd_kind, bd_func, str_sp_vars
                        )

    d_vars: List[str] = []
    if len(str_sp_vars) == 2:
        for func in range(n_funcs):
            for i in range(n_part[0]):
                for j in range(n_part[1]):
                    name = f"XX{func}_{i}_{j}"
                    if name not in d_vars:
                        d_vars.append(name)
    else:
        for func in range(n_funcs):
            for i in range(n_part[0]):
                name = f"XX{func}_{i}_0"
                if name not in d_vars:
                    d_vars.append(name)

    flat_list_positions: List[str] = []
    for L in list_positions:
        flat_list_positions.extend(L)

    hx_val = str(1.0 / (n_part[0] - 1))
    for i in range(len(flat_list_positions)):
        for func_name in xd_var:
            flat_list_positions[i] = flat_list_positions[i].replace(
                    f"h{func_name}_", hx_val
            )

    _h_pattern = re.compile(r'\bh[A-Za-z0-9]+_\b')
    for i in range(len(flat_list_positions)):
        flat_list_positions[i] = _h_pattern.sub(hx_val, flat_list_positions[i])

    dirichlet_constraints: dict = {}
    neumann_constraints: dict = {}
    hx = 1.0 / (n_part[0] - 1)
    hy = 1.0 / (n_part[1] - 1) if len(str_sp_vars) == 2 else 1.0

    if len(str_sp_vars) == 2:
        Nx, Ny = n_part[0], n_part[1]
        bd_kind = {
            "west": west_bd, "east": east_bd,
            "south": south_bd, "north": north_bd,
        }
        bd_func = {
            "west": west_func_bd, "east": east_func_bd,
            "south": south_func_bd, "north": north_func_bd,
        }
        any_flux = any(
            bd_kind[s][f].lower() in _FLUX
            for s in bd_kind for f in range(n_funcs)
        )
        if any_flux and (Nx < 3 or Ny < 3):
            raise ValueError(
                f"Neumann/Robin 2D exige min(Nx, Ny) >= 3. Recebido: Nx={Nx}, Ny={Ny}."
            )

        def _idx(func, i, j):
            return func * Nx * Ny + i * Ny + j

        for func in range(n_funcs):
            for i in range(Nx):
                for j in range(Ny):
                    asides = []
                    if i == 0:      asides.append("west")
                    if i == Nx - 1: asides.append("east")
                    if j == 0:      asides.append("south")
                    if j == Ny - 1: asides.append("north")
                    if not asides:
                        continue
                    kinds = [bd_kind[s][func].lower() for s in asides]
                    if "dirichlet" in kinds:
                        sd = asides[kinds.index("dirichlet")]
                        dirichlet_constraints[_idx(func, i, j)] = {
                            "expr": bd_func[sd][func],
                            "x": i * hx,
                            "y": j * hy,
                        }
    else:
        Nx = n_part[0]
        any_flux_1d = any(
            west_bd[f].lower() in _FLUX or east_bd[f].lower() in _FLUX
            for f in range(n_funcs)
        )
        if any_flux_1d and Nx < 3:
            raise ValueError("Neumann/Robin 1D exige Nx >= 3.")
        for func in range(n_funcs):
            offset = func * Nx
            if west_bd[func].lower() == 'dirichlet':
                dirichlet_constraints[offset] = {
                    'expr': west_func_bd[func],
                    'x': 0.0,
                    'y': 0.0,
                }
            if east_bd[func].lower() == 'dirichlet':
                dirichlet_constraints[offset + Nx - 1] = {
                    'expr': east_func_bd[func],
                    'x': (Nx - 1) * hx,
                    'y': 0.0,
                }

    return flat_list_positions, d_vars, dirichlet_constraints, neumann_constraints

from typing import List, Tuple
import re
from ..Auxs.FuncAux import repl_symbol as _repl_symbol
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

    if len(str_sp_vars) == 2:
        list_south = [[] for _ in range(n_funcs)]
        list_north = [[] for _ in range(n_funcs)]
        list_west  = [[] for _ in range(n_funcs)]
        list_east  = [[] for _ in range(n_funcs)]

        for func in range(n_funcs):
            s_bc = get_boundary(south_bd[func], south_func_bd[func], south_alpha_bd, south_beta_bd)
            list_south[func] = s_bc.apply("south", list_eq, n_part, xd_var, str_sp_vars)[func]

            n_bc = get_boundary(north_bd[func], north_func_bd[func], north_alpha_bd, north_beta_bd)
            list_north[func] = n_bc.apply("north", list_eq, n_part, xd_var, str_sp_vars)[func]

            w_bc = get_boundary(west_bd[func], west_func_bd[func])
            list_west[func] = w_bc.apply("west", list_eq, n_part, xd_var, str_sp_vars)[func] \
                              if west_bd[func].lower() in ("dirichlet", "neumann", "robin") else []

            e_bc = get_boundary(east_bd[func], east_func_bd[func], east_alpha_bd, east_beta_bd)
            east_col = e_bc.apply("east", list_eq, n_part, xd_var, str_sp_vars)[func]
            list_east[func] = [list_south[func][-1]] + east_col + [list_north[func][-1]]

    list_positions = _build_position_labels(n_part, str_sp_vars, n_funcs)

    if len(str_sp_vars) == 1:
        for func in range(n_funcs):
            C = 0
            for i in range(n_part[0]):
                if i == 0:
                    bc = get_boundary(west_bd[func], west_func_bd[func])
                    list_positions[func][i] = bc.apply(
                        "west", list_eq, n_part, xd_var, str_sp_vars
                    )[func][0]
                elif i == n_part[0] - 1:
                    bc = get_boundary(east_bd[func], east_func_bd[func], east_alpha_bd, east_beta_bd)
                    list_positions[func][i] = bc.apply(
                        "east", list_eq, n_part, xd_var, str_sp_vars
                    )[func][0]
                else:
                    list_positions[func][i] = list_eq[func][C]
                    C += 1

    elif len(str_sp_vars) == 2:
        for func in range(n_funcs):
            C = 0
            for idx in range(len(list_positions[func])):
                label = list_positions[func][idx]
                if   "S" in label: list_positions[func][idx] = list_south[func].pop(0) if list_south[func] else label
                elif "N" in label: list_positions[func][idx] = list_north[func].pop(0) if list_north[func] else label
                elif "E" in label: list_positions[func][idx] = list_east[func].pop(0)  if list_east[func]  else label
                elif "W" in label: list_positions[func][idx] = list_west[func].pop(0)  if list_west[func]  else label
                elif "C" in label:
                    list_positions[func][idx] = list_eq[func][C]
                    C += 1

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
        sides = {
            'west':  west_bd,
            'east':  east_bd,
            'north': north_bd,
            'south': south_bd,
        }
        func_exprs = {
            'west':  west_func_bd,
            'east':  east_func_bd,
            'north': north_func_bd,
            'south': south_func_bd,
        }
        Nx, Ny = n_part[0], n_part[1]

        any_neumann = any(
            sides[s][f].lower() == 'neumann'
            for s in sides for f in range(n_funcs)
        )
        if any_neumann and (Nx < 3 or Ny < 3):
            raise ValueError(
                f"Neumann com discretizacao one-sided de 2a ordem exige "
                f"min(Nx, Ny) >= 3. Recebido: Nx={Nx}, Ny={Ny}."
            )

        def _idx(func, i, j):
            return func * Nx * Ny + i * Ny + j

        for func in range(n_funcs):
            for i in range(Nx):
                for j in range(Ny):
                    idx = _idx(func, i, j)
                    side = None
                    if   i == 0:      side = 'west'
                    elif i == Nx - 1: side = 'east'
                    elif j == 0:      side = 'south'
                    elif j == Ny - 1: side = 'north'
                    if side is None:
                        continue
                    bc_kind = sides[side][func].lower()

                    if bc_kind == 'dirichlet':
                        dirichlet_constraints[idx] = {
                            'expr': func_exprs[side][func],
                            'x': i * hx,
                            'y': j * hy,
                        }
                    elif bc_kind == 'neumann':
                        if side == 'south':
                            n1 = _idx(func, i, 1)
                            n2 = _idx(func, i, 2)
                        elif side == 'north':
                            n1 = _idx(func, i, Ny - 2)
                            n2 = _idx(func, i, Ny - 3)
                        elif side == 'west':
                            n1 = _idx(func, 1, j)
                            n2 = _idx(func, 2, j)
                        else:
                            n1 = _idx(func, Nx - 2, j)
                            n2 = _idx(func, Nx - 3, j)

                        neumann_constraints[idx] = {
                            'expr': func_exprs[side][func],
                            'x': i * hx,
                            'y': j * hy,
                            'side': side,
                            'idx_n1': n1,
                            'idx_n2': n2,
                        }
    else:
        Nx = n_part[0]
        for func in range(n_funcs):
            offset = func * Nx
            if west_bd[func].lower() == 'dirichlet':
                dirichlet_constraints[offset] = {
                    'expr': west_func_bd[func],
                    'x': 0.0,
                    'y': 0.0,
                }
            elif west_bd[func].lower() == 'neumann':
                if Nx < 3:
                    raise ValueError("Neumann 1D exige Nx >= 3.")
                neumann_constraints[offset] = {
                    'expr': west_func_bd[func],
                    'x': 0.0,
                    'y': 0.0,
                    'side': 'west',
                    'idx_n1': offset + 1,
                    'idx_n2': offset + 2,
                }
            if east_bd[func].lower() == 'dirichlet':
                dirichlet_constraints[offset + Nx - 1] = {
                    'expr': east_func_bd[func],
                    'x': (Nx - 1) * hx,
                    'y': 0.0,
                }
            elif east_bd[func].lower() == 'neumann':
                if Nx < 3:
                    raise ValueError("Neumann 1D exige Nx >= 3.")
                neumann_constraints[offset + Nx - 1] = {
                    'expr': east_func_bd[func],
                    'x': (Nx - 1) * hx,
                    'y': 0.0,
                    'side': 'east',
                    'idx_n1': offset + Nx - 2,
                    'idx_n2': offset + Nx - 3,
                }

    return flat_list_positions, d_vars, dirichlet_constraints, neumann_constraints

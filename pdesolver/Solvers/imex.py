import time

import numpy as np
import scipy.sparse as sp_sparse
from scipy.sparse.linalg import splu

from ..Disc.stencil import group_constraints
from . import fastpoisson
from .solver_base import (
    ColoredJacobian, impose_dirichlet, make_history, save_to_history,
)


def _make_bc_lambda(expr_str: str):
    import sympy as sp
    from sympy.parsing.sympy_parser import parse_expr

    t_sym, x_sym, y_sym = sp.symbols('t x y')
    return sp.lambdify((t_sym, x_sym, y_sym), parse_expr(expr_str),
                       modules='numpy')


def stiffness_report(operator, ordem_min=2, nk=24):
    """Classify each term and estimate the stiffness removed."""
    op_rig, op_bra, regs = operator.split_stiff(ordem_min=ordem_min, nk=nk)

    from ..Analysis.stability import symbol_eigenvalues

    lam_tot, _ = symbol_eigenvalues(operator, nk=nk)
    lam_bra, _ = symbol_eigenvalues(op_bra, nk=nk)
    pico_tot = float(np.max(np.abs(lam_tot))) if lam_tot.size else 0.0
    pico_bra = float(np.max(np.abs(lam_bra))) if lam_bra.size else 0.0
    if pico_bra <= 0.0:
        pico_bra = max(
            (r['lambda_max'] for r in regs if not r['rigido']), default=0.0
        )
    ganho = (pico_tot / pico_bra) if pico_bra > 0 else float('inf')
    return op_rig, op_bra, {
        'termos': regs,
        'lambda_total': pico_tot,
        'lambda_explicito': pico_bra,
        'ganho_de_passo': ganho,
    }


def imex(operator, tf, nt, ic, n_funcs=None,
         dirichlet_constraints=None, neumann_constraints=None,
         verbose=False, ordem_min=2):
    """Semi-implicit BDF2: stiff linear terms implicit, the rest explicit."""
    dt = tf / nt
    n = operator.size
    u = np.array(ic, dtype=np.float64).flatten()

    dirichlet_constraints = dirichlet_constraints or {}
    neumann_constraints = neumann_constraints or {}
    dirichlet_groups = group_constraints(dirichlet_constraints)
    neumann_lambdas = {
        idx: _make_bc_lambda(info['expr'])
        for idx, info in neumann_constraints.items()
    }

    h_neumann = None
    if neumann_constraints:
        eixos = {round(info['x'], 12) for info in neumann_constraints.values()}
        if len(eixos) >= 2:
            h_neumann = 1.0 / (len(eixos) - 1)

    def _apply_bcs(vec, t_val):
        impose_dirichlet(vec, t_val, dirichlet_groups)
        if h_neumann is not None:
            two_h = 2.0 * h_neumann
            for idx, info in neumann_constraints.items():
                f_val = float(neumann_lambdas[idx](t_val, info['x'], info['y']))
                vec[idx] = (4.0 * vec[info['idx_n1']]
                            - vec[info['idx_n2']] + two_h * f_val) / 3.0
        return vec

    t0 = time.time()
    op_rig, op_bra, info = stiffness_report(operator, ordem_min=ordem_min)
    if verbose:
        n_imp = sum(1 for r in info['termos'] if r['rigido'])
        print(f"  [IMEX] Separação simbólica: {n_imp}/{len(info['termos'])} "
              f"termos implícitos ({time.time()-t0:.3f}s)")
        for r in sorted(info['termos'], key=lambda z: -z['lambda_max']):
            tag = 'implícito' if r['rigido'] else 'explícito'
            extra = '' if r['linear'] else ' [não linear]'
            print(f"    {tag:<10} |λ|={r['lambda_max']:11.2f}{extra}  "
                  f"{r['termo']}")
        print(f"  [IMEX] |λ| total={info['lambda_total']:.4g}, "
              f"explícito={info['lambda_explicito']:.4g} "
              f"→ passo ~{info['ganho_de_passo']:.1f}x maior")

    zeros = np.zeros(n)
    jac = ColoredJacobian(*op_rig.sparsity())
    L, _ = jac.build(op_rig, zeros, 0.0)

    def fonte(t_val):
        return np.asarray(op_rig(t_val, zeros))

    def explicito(t_val, vec):
        return np.asarray(op_bra(t_val, vec))

    t_fat = time.time()
    lu_e = fastpoisson.build(op_rig, operator._pdes, dt)
    lu_2 = fastpoisson.build(op_rig, operator._pdes, 2.0 * dt / 3.0)
    if lu_e is not None and lu_2 is not None:
        estagio = "DST"
    else:
        ident = sp_sparse.eye(n, format='csr')
        lu_e = splu((ident - dt * L).tocsc())
        lu_2 = splu((ident - (2.0 * dt / 3.0) * L).tocsc())
        estagio = "LU esparsa"
    if verbose:
        print(f"  [IMEX] Estágio implícito por {estagio}: "
              f"{time.time()-t_fat:.3f}s")

    final_list, use_groups, n_elements = make_history(n_funcs, n)
    u = _apply_bcs(u, 0.0)
    save_to_history(u, final_list, use_groups, n_funcs, n_elements)

    t_loop = time.time()
    n_ant = explicito(0.0, u)
    rhs = u + dt * (fonte(dt) + n_ant)
    impose_dirichlet(rhs, dt, dirichlet_groups)
    u_prev = u.copy()
    u = _apply_bcs(lu_e.solve(rhs), dt)
    save_to_history(u, final_list, use_groups, n_funcs, n_elements)

    for passo in range(1, nt):
        t_novo = (passo + 1) * dt
        n_atual = explicito(passo * dt, u)
        rhs = ((4.0 * u - u_prev) / 3.0
               + (2.0 * dt / 3.0) * (fonte(t_novo)
                                     + 2.0 * n_atual - n_ant))
        impose_dirichlet(rhs, t_novo, dirichlet_groups)
        u_prev = u.copy()
        u = _apply_bcs(lu_2.solve(rhs), t_novo)
        n_ant = n_atual
        save_to_history(u, final_list, use_groups, n_funcs, n_elements)

    if verbose:
        print(f"  [IMEX] Loop de tempo: {time.time()-t_loop:.3f}s")

    return u, final_list

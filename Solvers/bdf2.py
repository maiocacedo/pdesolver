import time
import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
import scipy.sparse as sp_sparse
from scipy.sparse.linalg import spsolve, splu

from .solver_base import (
    compile_equations, extract_linear_structure, detect_linearity,
    eval_F, picard_step, newton_step,
    make_history, save_to_history,
)


def _make_bc_lambda(expr_str: str):
    t_sym, x_sym, y_sym = sp.symbols('t x y')
    expr = parse_expr(expr_str)
    return sp.lambdify((t_sym, x_sym, y_sym), expr, modules='numpy')


def bdf2(flat_list, d_vars, tf, nt, ic, n_funcs=None,
         nonlinear_method='newton', tol_nl=1e-8, max_iter_nl=20,
         dirichlet_constraints=None,
         neumann_constraints=None,
         verbose=False,
         is_linear=None):
    dt = tf / nt
    n  = len(d_vars)
    u  = np.array(ic, dtype=np.float64).flatten()

    dirichlet_constraints = dirichlet_constraints or {}
    neumann_constraints   = neumann_constraints   or {}

    dirichlet_lambdas = {
        idx: _make_bc_lambda(info['expr'])
        for idx, info in dirichlet_constraints.items()
    }
    neumann_lambdas = {
        idx: _make_bc_lambda(info['expr'])
        for idx, info in neumann_constraints.items()
    }

    h_neumann = None
    for info in neumann_constraints.values():
        side = info['side']
        if side in ('south', 'north'):
            pass
    if neumann_constraints:
        all_x = set()
        all_y = set()
        for info in dirichlet_constraints.values():
            all_x.add(round(info['x'], 12))
            all_y.add(round(info['y'], 12))
        for info in neumann_constraints.values():
            all_x.add(round(info['x'], 12))
            all_y.add(round(info['y'], 12))
        Nx_est = len(all_x)
        Ny_est = len(all_y)
        if Nx_est >= 2:
            h_neumann = 1.0 / (Nx_est - 1)
        elif Ny_est >= 2:
            h_neumann = 1.0 / (Ny_est - 1)
        else:
            raise RuntimeError(
                "Nao foi possivel inferir o passo h da malha para Neumann."
            )

    def _apply_dirichlet(u, t_val):
        for idx, info in dirichlet_constraints.items():
            f = dirichlet_lambdas[idx]
            u[idx] = float(f(t_val, info['x'], info['y']))
        return u

    def _apply_neumann(u, t_val):
        if h_neumann is None:
            return u
        two_h = 2.0 * h_neumann
        for idx, info in neumann_constraints.items():
            f = neumann_lambdas[idx]
            f_val = float(f(t_val, info['x'], info['y']))
            u[idx] = (4.0 * u[info['idx_n1']]
                      - u[info['idx_n2']]
                      + two_h * f_val) / 3.0
        return u

    def _apply_bcs(u, t_val):
        u = _apply_dirichlet(u, t_val)
        u = _apply_neumann(u, t_val)
        return u

    u = _apply_bcs(u, 0.0)

    final_list, use_groups, n_elements = make_history(n_funcs, n)

    funcs = compile_equations(flat_list, d_vars, verbose=verbose)

    overwrite_indices = list(dirichlet_constraints.keys()) + list(neumann_constraints.keys())
    if is_linear is None:
        is_linear, L = detect_linearity(funcs, n, verbose=verbose,
                                        dirichlet_indices=overwrite_indices)
    else:
        L = None

    I = sp_sparse.eye(n, format='csr')

    if is_linear:
        L, fonte_func = extract_linear_structure(funcs, n, verbose=verbose, L=L)
        A_bdf1 = I - dt * L
        A_bdf2 = I - (2.0 * dt / 3.0) * L
        t_lu = time.time()
        lu_bdf1 = splu(A_bdf1.tocsc())
        lu_bdf2 = splu(A_bdf2.tocsc())
        if verbose:
            print(f"  [BDF2] Pré-fatoração LU (A_bdf1, A_bdf2): {time.time()-t_lu:.3f}s")
    else:
        if verbose:
            print(f"  [BDF2] EDP nao-linear detectada - usando {nonlinear_method.upper()} "
                  f"(tol={tol_nl:.0e}, max_iter={max_iter_nl})")
        _, fonte_func = extract_linear_structure(funcs, n, verbose=verbose)

    save_to_history(u, final_list, use_groups, n_funcs, n_elements)

    t0_loop = time.time()
    total_iters = 0
    tempo_1 = dt

    if is_linear:
        rhs_1  = u + dt * fonte_func(tempo_1)
        u_prev = u.copy()
        u      = lu_bdf1.solve(rhs_1)
        u      = _apply_bcs(u, tempo_1)
    else:
        if nonlinear_method == 'newton':
            u_new, n_iter = newton_step(
                funcs, u, tempo_1, dt, n, u,
                alpha=1.0, max_iter=max_iter_nl,
                tol_nl=tol_nl, verbose=verbose
            )
        else:
            u_new, n_iter = picard_step(
                funcs, u, tempo_1, dt, n, u,
                alpha=1.0, max_iter=max_iter_nl,
                tol_nl=tol_nl, verbose=verbose
            )
        total_iters += n_iter
        u_prev = u.copy()
        u      = _apply_bcs(u_new, tempo_1)

    save_to_history(u, final_list, use_groups, n_funcs, n_elements)

    for passo in range(1, nt):
        tempo_n1 = (passo + 1) * dt
        rhs_hist = (4.0 * u - u_prev) / 3.0

        if is_linear:
            rhs_vec = rhs_hist + (2.0 * dt / 3.0) * fonte_func(tempo_n1)
            u_prev  = u.copy()
            u       = lu_bdf2.solve(rhs_vec)
            u       = _apply_bcs(u, tempo_n1)
        else:
            if nonlinear_method == 'newton':
                u_new, n_iter = newton_step(
                    funcs, u, tempo_n1, dt, n, rhs_hist,
                    alpha=2.0/3.0, max_iter=max_iter_nl,
                    tol_nl=tol_nl, verbose=verbose
                )
            else:
                u_new, n_iter = picard_step(
                    funcs, u, tempo_n1, dt, n, rhs_hist,
                    alpha=2.0/3.0, max_iter=max_iter_nl,
                    tol_nl=tol_nl, verbose=verbose
                )
            total_iters += n_iter
            u_prev = u.copy()
            u      = _apply_bcs(u_new, tempo_n1)

        save_to_history(u, final_list, use_groups, n_funcs, n_elements)

    elapsed = time.time() - t0_loop
    if verbose:
        print(f"  [BDF2] Loop de tempo: {elapsed:.3f}s", end="")
        if not is_linear:
            print(f" | Media iteracoes/passo: {total_iters/nt:.1f}", end="")
        print()

    return u, final_list

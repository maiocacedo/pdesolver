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


def cn(flat_list, d_vars, tf, nt, ic, n_funcs=None,
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
    if neumann_constraints:
        all_x = set(); all_y = set()
        for info in dirichlet_constraints.values():
            all_x.add(round(info['x'], 12))
            all_y.add(round(info['y'], 12))
        for info in neumann_constraints.values():
            all_x.add(round(info['x'], 12))
            all_y.add(round(info['y'], 12))
        Nx_est = len(all_x); Ny_est = len(all_y)
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
        A_impl = I - (dt / 2.0) * L
        A_expl = I + (dt / 2.0) * L
        t_lu = time.time()
        lu_impl = splu(A_impl.tocsc())
        if verbose:
            print(f"  [CN] Pré-fatoração LU (A_impl): {time.time()-t_lu:.3f}s")
    else:
        if verbose:
            print(f"  [CN] EDP nao-linear detectada - usando {nonlinear_method.upper()} "
                  f"(tol={tol_nl:.0e}, max_iter={max_iter_nl})")
        _, fonte_func = extract_linear_structure(funcs, n, verbose=verbose)

    save_to_history(u, final_list, use_groups, n_funcs, n_elements)

    t0 = time.time()
    total_iters = 0

    for passo in range(nt):
        tempo_atual   = passo * dt
        tempo_proximo = (passo + 1) * dt

        if is_linear:
            f_n  = fonte_func(tempo_atual)
            f_n1 = fonte_func(tempo_proximo)
            rhs  = A_expl.dot(u) + (dt / 2.0) * (f_n + f_n1)
            u    = lu_impl.solve(rhs)
            u    = _apply_bcs(u, tempo_proximo)
        else:
            F_n      = eval_F(funcs, tempo_atual, u)
            rhs_hist = u + (dt / 2.0) * F_n

            if nonlinear_method == 'newton':
                u, n_iter = newton_step(
                    funcs, u, tempo_proximo, dt, n, rhs_hist,
                    alpha=0.5, max_iter=max_iter_nl,
                    tol_nl=tol_nl, verbose=verbose
                )
            else:
                u, n_iter = picard_step(
                    funcs, u, tempo_proximo, dt, n, rhs_hist,
                    alpha=0.5, max_iter=max_iter_nl,
                    tol_nl=tol_nl, verbose=verbose
                )
            u = _apply_bcs(u, tempo_proximo)
            total_iters += n_iter

        save_to_history(u, final_list, use_groups, n_funcs, n_elements)

    elapsed = time.time() - t0
    if verbose:
        print(f"  [CN] Loop de tempo: {elapsed:.3f}s", end="")
        if not is_linear:
            print(f" | Media iteracoes/passo: {total_iters/nt:.1f}", end="")
        print()

    return u, final_list

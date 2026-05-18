import re
import time
import warnings
import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
import scipy.sparse as sp_sparse
from scipy.sparse.linalg import spsolve

def compile_equations(flat_list, d_vars, verbose=False, use_cse=True):
    t0 = time.time()
    t_sym = sp.Symbol('t')
    u_sym = sp.IndexedBase('u')
    subs = {sp.Symbol(v): u_sym[i] for i, v in enumerate(d_vars)}
    exprs = [parse_expr(eq).xreplace(subs) for eq in flat_list]
    matrix_expr = sp.Matrix(exprs)
    F_raw = sp.lambdify((t_sym, u_sym), matrix_expr,
                        modules='numpy', cse=use_cse)
    n = len(flat_list)
    class _VectorF:
        __slots__ = ('_raw', '_n', '_scalar_cache')
        def __init__(self, raw, n):
            self._raw = raw
            self._n = n
            self._scalar_cache = {}
        def __call__(self, t, u):
            out = self._raw(float(t), np.asarray(u, dtype=np.float64))
            return np.asarray(out, dtype=np.float64).reshape(-1)
        def __len__(self):
            return self._n
        def __getitem__(self, i):
            if i in self._scalar_cache:
                return self._scalar_cache[i]
            def _scalar(t, *u_vals):
                u_arr = np.asarray(u_vals, dtype=np.float64)
                return float(self._raw(float(t), u_arr)[i])
            self._scalar_cache[i] = _scalar
            return _scalar
        def __iter__(self):
            for i in range(self._n):
                yield self[i]
    F_callable = _VectorF(F_raw, n)
    if verbose:
        cse_tag = " + CSE" if use_cse else ""
        print(f"  Compilação lambdify vetorizada{cse_tag}: {time.time()-t0:.3f}s")
    return F_callable

def detect_linearity_symbolic(pde_eqs, func_names, sp_vars, verbose=False):
    t0 = time.time()
    try:
        derivative_syms = []
        patterns = []
        for fi, fname in enumerate(func_names):
            for sv in sp_vars:
                s2 = sp.Symbol(f"_LIN_{fi}_d2{sv}")
                s1 = sp.Symbol(f"_LIN_{fi}_d1{sv}")
                derivative_syms.extend([s2, s1])
                patterns.append((
                    re.compile(rf"d2{re.escape(fname)}/d{re.escape(sv)}2"),
                    str(s2),
                ))
                patterns.append((
                    re.compile(rf"d{re.escape(fname)}/d{re.escape(sv)}"),
                    str(s1),
                ))
            s0 = sp.Symbol(f"_LIN_{fi}_val")
            derivative_syms.append(s0)
            patterns.append((
                re.compile(rf"\b{re.escape(fname)}\b"),
                str(s0),
            ))
        for eq_str in pde_eqs:
            rhs_str = eq_str.split("=", 1)[1]
            for pat, repl in patterns:
                rhs_str = pat.sub(repl, rhs_str)
            expr = parse_expr(rhs_str)
            relevant = [s for s in derivative_syms if s in expr.free_symbols]
            if not relevant:
                continue
            try:
                poly = sp.Poly(expr, *relevant)
            except sp.PolynomialError:
                if verbose:
                    print(f"  Detecção simbólica: expressão não-polinomial "
                          f"→ NÃO-LINEAR ({time.time()-t0:.3f}s)")
                return False
            if poly.total_degree() > 1:
                if verbose:
                    print(f"  Detecção simbólica: grau {poly.total_degree()} "
                          f"em U → NÃO-LINEAR ({time.time()-t0:.3f}s)")
                return False
        if verbose:
            print(f"  Detecção simbólica: LINEAR ({time.time()-t0:.3f}s)")
        return True
    except Exception as e:
        if verbose:
            print(f"  Detecção simbólica falhou ({e}) — caindo para numérica.")
        return None

def detect_linearity(funcs, n, t0_val=0.0, verbose=False,
                     dirichlet_indices=None):
    t0 = time.time()
    zeros = np.zeros(n)
    L0, fonte = _extract_L(funcs, n, zeros, t0_val)
    F0 = fonte
    rng = np.random.default_rng(42)
    is_linear = True
    for _ in range(3):
        u_rand = rng.standard_normal(n) * 0.5
        if dirichlet_indices:
            for idx in dirichlet_indices:
                u_rand[idx] = 0.0
        F_rand = eval_F(funcs, t0_val, u_rand)
        F_pred = L0 @ u_rand + F0
        err = np.max(np.abs(F_rand - F_pred))
        if err > 1e-6:
            is_linear = False
            break
    if verbose:
        status = "LINEAR" if is_linear else "NÃO-LINEAR"
        print(f"  Detecção de linearidade: {status} ({time.time()-t0:.3f}s)")
    return is_linear, L0


def _extract_L(funcs, n, u_ref, t_val, eps=1e-6):
    F_ref = eval_F(funcs, t_val, u_ref)
    fonte = eval_F(funcs, t_val, np.zeros(n))
    rows_all, cols_all, vals_all = [], [], []
    for j in range(n):
        u_pert = u_ref.copy()
        u_pert[j] += eps
        F_pert = eval_F(funcs, t_val, u_pert)
        dF_j = (F_pert - F_ref) / eps
        nz = np.nonzero(np.abs(dF_j) > 1e-15)[0]
        rows_all.extend(nz.tolist())
        cols_all.extend([j] * len(nz))
        vals_all.extend(dF_j[nz].tolist())
    L = sp_sparse.csr_matrix(
        (vals_all, (rows_all, cols_all)), shape=(n, n)
    )
    return L, fonte


def extract_linear_structure(funcs, n, t0_val=0.0, verbose=False, L=None):
    t0 = time.time()
    if L is None:
        L, _ = _extract_L(funcs, n, np.zeros(n), t0_val)
    z = np.zeros(n)
    F0 = eval_F(funcs, 0.0, z)
    F1 = eval_F(funcs, 1.0, z)
    t_dependent = np.where(np.abs(F1 - F0) > 1e-14)[0]
    if len(t_dependent) == 0:
        _fonte_const = F0.copy()
        def fonte_func(t_val):
            return _fonte_const
        if verbose:
            print(f"  Extração estrutura A: {time.time()-t0:.3f}s  "
                  f"[fonte constante — pré-computada]")
    else:
        _fonte_const = F0.copy()
        _t_dep_idx = t_dependent
        def fonte_func(t_val):
            out = _fonte_const.copy()
            F_t = eval_F(funcs, t_val, z)
            out[_t_dep_idx] = F_t[_t_dep_idx]
            return out
        if verbose:
            print(f"  Extração estrutura A: {time.time()-t0:.3f}s  "
                  f"[fonte parcial: {len(t_dependent)}/{n} eqs dependem de t]")
    return L, fonte_func

def _detect_sparsity_pattern(funcs, n, t_val=0.0, eps=1e-6):
    u_ref = np.zeros(n)
    F_ref = eval_F(funcs, t_val, u_ref)
    rows_sp, cols_sp = [], []
    for j in range(n):
        u_pert = u_ref.copy()
        u_pert[j] += eps
        F_pert = eval_F(funcs, t_val, u_pert)
        dF_j = (F_pert - F_ref) / eps
        nz = np.nonzero(np.abs(dF_j) > 1e-15)[0]
        rows_sp.extend(nz.tolist())
        cols_sp.extend([j] * len(nz))
    data_sp = np.ones(len(rows_sp), dtype=bool)
    sparsity = sp_sparse.csr_matrix(
        (data_sp, (rows_sp, cols_sp)), shape=(n, n)
    )
    conflict = (sparsity.T @ sparsity).tocsr()
    colors = np.full(n, -1, dtype=int)
    for j in range(n):
        _, neighbors = conflict[j].nonzero()
        used = set(colors[neighbors[colors[neighbors] >= 0]])
        c = 0
        while c in used:
            c += 1
        colors[j] = c
    n_colors = int(colors.max()) + 1
    return sparsity, colors, n_colors


def _jacobian_sparse_colored(funcs, n, u_k, t_val, sparsity, colors, n_colors,
                              eps=1e-6, _csc_cache={}):
    F_k = eval_F(funcs, t_val, u_k)
    sp_id = id(sparsity)
    if sp_id not in _csc_cache:
        sp_csc = sparsity.tocsc()
        col_rows = [sp_csc.indices[sp_csc.indptr[j]:sp_csc.indptr[j+1]]
                    for j in range(n)]
        _csc_cache[sp_id] = col_rows
    else:
        col_rows = _csc_cache[sp_id]
    all_rows = np.empty(sparsity.nnz, dtype=np.int32)
    all_cols = np.empty(sparsity.nnz, dtype=np.int32)
    all_vals = np.empty(sparsity.nnz, dtype=np.float64)
    ptr = 0
    for c in range(n_colors):
        cols_c = np.where(colors == c)[0]
        u_pert = u_k.copy()
        u_pert[cols_c] += eps
        F_pert = eval_F(funcs, t_val, u_pert)
        dF = (F_pert - F_k) / eps
        for j in cols_c:
            rows_j = col_rows[j]
            k = len(rows_j)
            if k == 0:
                continue
            all_rows[ptr:ptr+k] = rows_j
            all_cols[ptr:ptr+k] = j
            all_vals[ptr:ptr+k] = dF[rows_j]
            ptr += k
    J_F = sp_sparse.csr_matrix(
        (all_vals[:ptr], (all_rows[:ptr], all_cols[:ptr])), shape=(n, n)
    )
    return J_F, F_k


def eval_F(funcs, t_val, u):
    if callable(funcs) and not isinstance(funcs, list):
        return funcs(t_val, u)
    return np.array([f(float(t_val), *u) for f in funcs], dtype=np.float64)

def picard_step(funcs, u, t_new, dt, n, rhs_hist,
                alpha, max_iter=50, tol_nl=1e-8, verbose=False):
    I = sp_sparse.eye(n, format='csr')
    u_k = u.copy()
    for k in range(max_iter):
        L_k, _ = _extract_L(funcs, n, u_k, t_new)
        fonte_k = eval_F(funcs, t_new, np.zeros(n))
        A = I - alpha * dt * L_k
        b = rhs_hist + alpha * dt * fonte_k
        u_new = spsolve(A, b)
        res = np.linalg.norm(u_new - u_k)
        if verbose:
            print(f"    Picard iter {k+1}: ||res|| = {res:.2e}")
        if res < tol_nl:
            return u_new, k + 1
        u_k = u_new
    warnings.warn(f"[Picard] Não convergiu em {max_iter} iterações "
                  f"(||res||={res:.2e}). Usando último iterate.")
    return u_k, max_iter

def newton_step(funcs, u, t_new, dt, n, rhs_hist,
                alpha, max_iter=20, tol_nl=1e-8, eps=1e-6, verbose=False,
                _cache={}):
    if n not in _cache:
        t_cache = time.time()
        sparsity, colors, n_colors = _detect_sparsity_pattern(funcs, n, eps=eps)
        _cache[n] = (sparsity, colors, n_colors)
        if verbose:
            print(f"  [Newton] Esparsidade detectada: {sparsity.nnz} entradas não-nulas, "
                  f"{n_colors} cores (vs {n} colunas) — {time.time()-t_cache:.3f}s")
    else:
        sparsity, colors, n_colors = _cache[n]
    I = sp_sparse.eye(n, format='csr')
    u_k = u.copy()
    for k in range(max_iter):
        J_F, F_k = _jacobian_sparse_colored(
            funcs, n, u_k, t_new, sparsity, colors, n_colors, eps=eps
        )
        G_k = u_k - alpha * dt * F_k - rhs_hist
        res = np.linalg.norm(G_k)
        if verbose:
            print(f"    Newton iter {k+1}: ||G|| = {res:.2e}")
        if res < tol_nl:
            return u_k, k
        J_G = I - alpha * dt * J_F
        delta = spsolve(J_G, G_k)
        u_k = u_k - delta
    warnings.warn(f"[Newton] Não convergiu em {max_iter} iterações "
                  f"(||G||={res:.2e}). Usando último iterate.")
    return u_k, max_iter

def make_history(n_funcs, n):
    use_groups = n_funcs is not None and (n % n_funcs == 0)
    n_elements = (n // n_funcs) if use_groups else n
    final_list = [[] for _ in range(n_funcs)] if use_groups else []
    return final_list, use_groups, n_elements


def save_to_history(u, final_list, use_groups, n_funcs, n_elements):
    if use_groups:
        u_r = u.reshape((n_funcs, n_elements))
        for j in range(n_funcs):
            final_list[j].append(u_r[j].tolist())

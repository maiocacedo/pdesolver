import ctypes
import os
import re
import sys
import warnings

import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr

from Auxs.FuncAux import symbol_references


def _fix_cupy_path():
    if sys.platform != "win32":
        return
    try:
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("ignore")
            import cupy as cp
            from cupy.cuda import compiler as _compiler

        cupy_dir = os.path.dirname(cp.__file__)
        try:
            cupy_dir.encode("ascii")
            return
        except UnicodeEncodeError:
            pass

        short_dir = ctypes.create_unicode_buffer(512)
        ctypes.windll.kernel32.GetShortPathNameW(cupy_dir, short_dir, 512)
        short_dir = short_dir.value.replace("\\", "/")

        _original = _compiler.compile_using_nvrtc

        def _patched(
            source,
            options=(),
            arch=None,
            filename="kern.cu",
            name_expressions=None,
            log_stream=None,
            cache_in_memory=False,
            jitify=False,
        ):
            options = tuple(
                o.replace(cupy_dir, short_dir).replace(
                    cupy_dir.replace("\\", "/"), short_dir
                )
                for o in options
            )
            return _original(
                source,
                options,
                arch,
                filename,
                name_expressions,
                log_stream,
                cache_in_memory,
                jitify,
            )

        _compiler.compile_using_nvrtc = _patched

    except Exception:
        pass


_fix_cupy_path()

_GPU_THRESHOLD = 10000

try:
    import warnings as _w

    with _w.catch_warnings():
        _w.simplefilter("ignore")
        import cupy as cp

        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("nenhuma GPU CUDA disponivel")
        _test = cp.array([1.0])
        _test[0] = 2.0
        del _test
    _CUPY_AVAILABLE = True
except Exception:
    cp = None
    _CUPY_AVAILABLE = False


def _build_func_map(xp):
    return {
        "sin": xp.sin,
        "cos": xp.cos,
        "tan": xp.tan,
        "asin": xp.arcsin,
        "acos": xp.arccos,
        "atan": xp.arctan,
        "atan2": xp.arctan2,
        "sinh": xp.sinh,
        "cosh": xp.cosh,
        "tanh": xp.tanh,
        "exp": xp.exp,
        "log": xp.log,
        "sqrt": xp.sqrt,
        "Abs": xp.abs,
        "sign": xp.sign,
        "Max": xp.maximum,
        "Min": xp.minimum,
        "mod": xp.mod,
        "floor": xp.floor,
        "ceil": xp.ceil,
        "sech": lambda x: 1.0 / xp.cosh(x),
    }


def natural_sort_key(s):
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split("([0-9]+)", s)
    ]


def get_short_path(path):
    if sys.platform != "win32":
        return path
    buf = ctypes.create_unicode_buffer(512)
    ctypes.windll.kernel32.GetShortPathNameW(path, buf, 512)
    return buf.value or path


def _make_bc_lambda(expr_str):
    t_sym, x_sym, y_sym = sp.symbols("t x y")
    expr = parse_expr(expr_str)
    return sp.lambdify((t_sym, x_sym, y_sym), expr, modules="numpy")


def SERKF45_cuda(
    oldexpr,
    ivar,
    funcs,
    yn,
    x0,
    xn,
    n,
    n_funcs,
    sp_vars,
    dt_max=None,
    tol=1e-5,
    dt_init=None,
    atol=1e-6,
    rtol=None,
    max_steps=10_000_000,
    dirichlet_constraints=None,
    neumann_constraints=None,
    save_every=None,
    verbose=False,
):
    if rtol is None:
        rtol = tol

    dirichlet_constraints = dirichlet_constraints or {}
    neumann_constraints = neumann_constraints or {}

    dirichlet_lambdas = {
        idx: _make_bc_lambda(info["expr"])
        for idx, info in dirichlet_constraints.items()
    }
    neumann_lambdas = {
        idx: _make_bc_lambda(info["expr"]) for idx, info in neumann_constraints.items()
    }

    h_neumann = None
    if neumann_constraints:
        all_x = set()
        all_y = set()
        for info in dirichlet_constraints.values():
            all_x.add(round(info["x"], 12))
            all_y.add(round(info["y"], 12))
        for info in neumann_constraints.values():
            all_x.add(round(info["x"], 12))
            all_y.add(round(info["y"], 12))
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

    def _apply_dirichlet_gpu(y_gpu, t_val):
        for idx, info in dirichlet_constraints.items():
            y_gpu[idx] = float(dirichlet_lambdas[idx](t_val, info["x"], info["y"]))

    def _apply_neumann_gpu(y_gpu, t_val):
        if h_neumann is None:
            return
        two_h = 2.0 * h_neumann
        for idx, info in neumann_constraints.items():
            f_val = float(neumann_lambdas[idx](t_val, info["x"], info["y"]))
            u_n1 = float(y_gpu[info["idx_n1"]])
            u_n2 = float(y_gpu[info["idx_n2"]])
            y_gpu[idx] = (4.0 * u_n1 - u_n2 + two_h * f_val) / 3.0

    def _apply_bcs_gpu(y_gpu, t_val):
        _apply_dirichlet_gpu(y_gpu, t_val)
        _apply_neumann_gpu(y_gpu, t_val)

    olddvar = sorted(symbol_references(funcs), key=natural_sort_key)
    oldivar = symbol_references(ivar)
    sym_map = {name: sp.Symbol(name) for name in (oldivar + olddvar)}
    t_sym = sym_map[oldivar[0]]
    y_syms = [sym_map[name] for name in olddvar]

    exprs = [parse_expr(e, local_dict=sym_map, evaluate=False) for e in oldexpr]
    m = len(exprs)
    if m == 0:
        raise ValueError("Lista de EDOs vazia.")

    use_history = n_funcs and (m % n_funcs == 0)
    n_elements = (m // n_funcs) if use_history else m
    final_list = [[] for _ in range(n_funcs)] if use_history else []

    if _CUPY_AVAILABLE and m >= _GPU_THRESHOLD:
        _xp = cp
        _mode = "GPU"
    else:
        _xp = np
        _mode = "CPU"

    dtype = _xp.float64

    F_tuple = sp.lambdify(
        (t_sym, *y_syms),
        tuple(exprs),
        modules=[_build_func_map(_xp), _xp],
    )

    _out_buf = _xp.empty(m, dtype=dtype)

    def F_all(t_scalar, y_vec):
        out = F_tuple(t_scalar, *[y_vec[k] for k in range(m)])
        for k in range(m):
            _out_buf[k] = out[k]
        return _out_buf

    def _to_numpy(arr):
        if _mode == "GPU":
            return arr.get()
        return np.asarray(arr)

    _n_frames = max(int(n), 1)
    if save_every is None:
        _use_time_trigger = True
        _dt_save = (float(xn) - float(x0)) / _n_frames
        _t_next_save = float(x0) + _dt_save
    else:
        _use_time_trigger = False

    def save_history(y_arr):
        if use_history:
            y_host = _to_numpy(y_arr).reshape((n_funcs, n_elements))
            for jgrp in range(n_funcs):
                final_list[jgrp].append(y_host[jgrp].tolist())

    print(f"Integrando {m} EDOs com SERKF45 ({_mode})...")

    y = _xp.asarray(yn, dtype=dtype).reshape(
        m,
    )
    if y.size != m:
        raise ValueError(f"len(yn) ({y.size}) != numero de EDOs ({m}).")

    _apply_bcs_gpu(y, float(x0))
    save_history(y)

    h = dtype(float(dt_init) if dt_init is not None else (xn - x0) / _n_frames)

    def clamp_h(h_, t_, t1_, dt_max_):
        if dt_max_ is not None:
            h_ = min(float(h_), float(dt_max_))
        h_ = min(float(h_), float(t1_) - float(t_))
        h_ = max(float(h_), 1e-14)
        return dtype(h_)

    t = dtype(float(x0))
    t1 = dtype(float(xn))
    h = clamp_h(h, t, t1, dt_max)

    c2, c3, c4, c5, c6 = 1 / 4, 3 / 8, 12 / 13, 1.0, 1 / 2
    a21 = 1 / 4
    a31, a32 = 3 / 32, 9 / 32
    a41, a42, a43 = 1932 / 2197, -7200 / 2197, 7296 / 2197
    a51, a52, a53, a54 = 439 / 216, -8.0, 3680 / 513, -845 / 4104
    a61, a62, a63, a64, a65 = -8 / 27, 2.0, -3544 / 2565, 1859 / 4104, -11 / 40
    b1, b3, b4, b5, b6 = 16 / 135, 6656 / 12825, 28561 / 56430, -9 / 50, 2 / 55
    b1s, b3s, b4s, b5s = 25 / 216, 1408 / 2565, 2197 / 4104, -1 / 5

    k1 = _xp.empty(m, dtype=dtype)
    k2 = _xp.empty(m, dtype=dtype)
    k3 = _xp.empty(m, dtype=dtype)
    k4 = _xp.empty(m, dtype=dtype)
    k5 = _xp.empty(m, dtype=dtype)
    k6 = _xp.empty(m, dtype=dtype)
    y4 = _xp.empty(m, dtype=dtype)
    y5 = _xp.empty(m, dtype=dtype)
    sc = _xp.empty(m, dtype=dtype)

    err_prev = 1.0
    n_steps = 0
    n_reject = 0

    while float(t) < float(t1) - 1e-14:
        if n_steps >= max_steps:
            warnings.warn(
                f"[SERKF45] Limite de {max_steps} passos atingido em "
                f"t={float(t):.4f}. Verifique a tolerancia ou aumente "
                f"max_steps."
            )
            break

        _xp.copyto(k1, F_all(t, y))
        _xp.copyto(k2, F_all(t + c2 * h, y + h * (a21 * k1)))
        _xp.copyto(k3, F_all(t + c3 * h, y + h * (a31 * k1 + a32 * k2)))
        _xp.copyto(k4, F_all(t + c4 * h, y + h * (a41 * k1 + a42 * k2 + a43 * k3)))
        _xp.copyto(
            k5, F_all(t + c5 * h, y + h * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4))
        )
        _xp.copyto(
            k6,
            F_all(
                t + c6 * h,
                y + h * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5),
            ),
        )

        _xp.copyto(y4, y + h * (b1s * k1 + b3s * k3 + b4s * k4 + b5s * k5))
        _xp.copyto(y5, y + h * (b1 * k1 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6))

        _xp.maximum(_xp.abs(y), _xp.abs(y5), out=sc)
        sc *= rtol
        sc += atol
        diff = (y5 - y4) / sc
        err = float(_to_numpy(_xp.sqrt(_xp.mean(diff * diff))))

        if err <= 1.0:
            _xp.copyto(y, y5)
            t = t + h
            n_steps += 1
            _apply_bcs_gpu(y, float(t))

            if save_every == 0:
                pass
            elif _use_time_trigger:
                if float(t) >= _t_next_save - 1e-14:
                    save_history(y)
                    _t_next_save += _dt_save
            elif save_every == 1:
                save_history(y)
            else:
                if n_steps % save_every == 0:
                    save_history(y)

            if err > 1e-10:
                factor = 0.9 * (1.0 / err) ** 0.4 * err_prev**0.1
            else:
                factor = 5.0
            factor = min(5.0, max(0.2, factor))
            err_prev = err
            h = clamp_h(dtype(factor) * h, t, t1, dt_max)

        else:
            n_reject += 1
            factor = max(0.1, 0.9 * (1.0 / err) ** 0.25)
            h = clamp_h(dtype(factor) * h, t, t1, dt_max)

    if use_history and save_every != 0:
        if abs(float(t) - float(t1)) < 1e-10:
            if _use_time_trigger and float(t1) > _t_next_save - _dt_save - 1e-10:
                save_history(y)

    if verbose:
        print(
            f"  [RKF] Passos aceitos: {n_steps} | "
            f"Rejeitados: {n_reject} | "
            f"Frames salvos: {len(final_list[0]) if use_history else 0}"
            f" | Device: {_mode}"
        )

    return _to_numpy(y), final_list

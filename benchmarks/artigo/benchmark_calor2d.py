"""
benchmark_calor2d.py
Equacao do Calor 2D:  dF/dt = a*(d2F/dx2 + d2F/dy2),  Dirichlet 0.
"""
import platform
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

NX    = 21
NY    = 21
TF    = 1.0
NT    = 200
ALPHA = 0.1
N_RUNS = int(__import__("os").environ.get("N_RUNS", "100"))


def rmse(a, b):
    a = np.asarray(a).flatten(); b = np.asarray(b).flatten()
    return float(np.sqrt(np.mean((a - b) ** 2)))


def bench(fn, runs=N_RUNS, warmup=1):
    for _ in range(warmup):
        fn()
    times, result = [], None
    for _ in range(runs):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    t = np.array(times)
    st = {"n": int(len(t)), "mean": float(t.mean()),
          "std": float(t.std(ddof=1)) if len(t) > 1 else 0.0,
          "min": float(t.min()), "median": float(np.median(t))}
    st["cv"] = (st["std"] / st["mean"] * 100.0) if st["mean"] > 0 else 0.0
    return result, st


def fmt(label, rms, st):
    return (f"  {label:<20} {rms:12.4e}  {st['mean']:9.4f} {st['std']:8.4f} "
            f"{st['min']:9.4f} {st['cv']:6.1f}")


import os as _os

sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
import fipy
import pde as pypde
from fipy import CellVariable, DiffusionTerm, Grid2D, TransientTerm
from fipy.tools import numerix

from pdesolver import PDE as PDEOBJ
from pdesolver import PDES


def analitica2d(X, Y, t):
    return (np.sin(np.pi * X) * np.sin(np.pi * Y)
            * np.exp(-2.0 * np.pi**2 * ALPHA * t))


def proprio(method, backend="symbolic"):
    def run():
        eq = PDEOBJ(f"dF/dt = {ALPHA}*d2F/dx2 + {ALPHA}*d2F/dy2", "F",
                    ["x", "y"], ["t"], ivar_boundary=[(0, 1), (0, 1)],
                    expr_ic="sin(pi*x)*sin(pi*y)",
                    west_bd="Dirichlet", west_func_bd="0",
                    east_bd="Dirichlet", east_func_bd="0",
                    north_bd="Dirichlet", north_func_bd="0",
                    south_bd="Dirichlet", south_func_bd="0")
        sim = PDES([eq], [NX, NY], backend=backend)
        sim.discretize(method="central")
        sim.solve(method=method, tf=TF, nt=NT)
        _, hist = sim.results
        return np.asarray(hist[0][-1]).reshape(NX, NY)
    sol, st = bench(run)
    x = np.linspace(0, 1, NX); y = np.linspace(0, 1, NY)
    X, Y = np.meshgrid(x, y, indexing="ij")
    return rmse(sol, analitica2d(X, Y, TF)), st


def fipy_run():
    dx, dy = 1.0 / (NX - 1), 1.0 / (NY - 1)
    def run():
        mesh = Grid2D(dx=dx, dy=dy, nx=NX - 1, ny=NY - 1)
        phi  = CellVariable(name="F", mesh=mesh)
        xc, yc = mesh.cellCenters
        phi.setValue(numerix.sin(numerix.pi * xc) * numerix.sin(numerix.pi * yc))
        for face in [mesh.facesLeft, mesh.facesRight, mesh.facesTop, mesh.facesBottom]:
            phi.constrain(0., face)
        eq = TransientTerm() == DiffusionTerm(coeff=ALPHA)
        dt = TF / NT
        for _ in range(NT):
            eq.solve(var=phi, dt=dt)
        return np.asarray(xc), np.asarray(yc), np.asarray(phi)
    (xc, yc, phi_v), st = bench(run)
    return rmse(phi_v, analitica2d(xc, yc, TF)), st


def pypde_run():
    def run():
        grid  = pypde.CartesianGrid([[0, 1], [0, 1]], [NX, NY],
                                    periodic=[False, False])
        field = pypde.ScalarField.from_expression(grid, "sin(pi*x) * sin(pi*y)")
        eq    = pypde.PDE({"F": f"{ALPHA}*laplace(F)"},
                          bc=[{"value": 0}, {"value": 0}])
        ctrl  = pypde.Controller(pypde.ScipySolver(eq, method="LSODA"),
                                 t_range=TF, tracker=None)
        return ctrl.run(field, dt=TF / NT)
    res, st = bench(run)
    coords = res.grid.cell_coords
    X, Y = coords[..., 0], coords[..., 1]
    return rmse(res.data, analitica2d(X, Y, TF)), st


def main():
    print("=" * 70)
    print(f"BENCHMARK - Calor 2D | NX={NX} NY={NY} NT={NT} N_RUNS={N_RUNS}")
    print("=" * 70)
    HDR = (f"  {'Metodo':<20} {'RMSE':>12}  {'t_med(s)':>9} {'dp(s)':>8} "
           f"{'t_min(s)':>9} {'CV(%)':>6}")
    print(HDR); print("  " + "-" * (len(HDR) - 2))
    linhas = []
    for lab, call in [
        ("PDESolver/BDF2",  lambda: proprio("bdf2")),
        ("PDESolver/CN",    lambda: proprio("CN")),
        ("PDESolver/RKF45", lambda: proprio("RKF")),
        ("  ^ stencil BDF2", lambda: proprio("bdf2", "stencil")),
        ("  ^ stencil RKF45", lambda: proprio("RKF", "stencil")),
        ("FiPy/CN",         fipy_run),
        ("py-pde/LSODA",    pypde_run),
    ]:
        r, st = call()
        print(fmt(lab, r, st)); linhas.append((lab, r, st))
    print("  " + "-" * (len(HDR) - 2))
    br = min(linhas, key=lambda z: z[1]); bt = min(linhas, key=lambda z: z[2]["mean"])
    print(f"\n  Menor RMSE : {br[0].strip()} ({br[1]:.4e})")
    print(f"  Menor tempo: {bt[0].strip()} ({bt[2]['mean']:.4f}s)")
    print(f"\n  Plataforma: {platform.platform()}")


if __name__ == "__main__":
    main()

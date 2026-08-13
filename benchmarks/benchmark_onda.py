"""
benchmark_onda.py
Onda 1D como sistema de 1a ordem:  dU/dt = V ;  dV/dt = c^2*d2U/dx2.
Dirichlet 0. Solucao analitica: U=sin(pi x)*cos(pi c t).
Compara PDESolver (BDF2, CN, RKF45), FiPy e py-pde.
Reporta RMSE e estatistica de tempo (media, desvio, minimo, CV) sobre N_RUNS.
"""
import numpy as np
import time
import warnings
import platform
import sys
from scipy.interpolate import interp1d
warnings.filterwarnings("ignore")

C     = 1.0
NX    = 41
TF    = 1.0
NT    = 200
x_ref = np.linspace(0, 1, NX)
N_RUNS = 100   # repeticoes para a estatistica de tempo (metodologia do artigo)


def rmse(a, b):
    a = np.asarray(a).flatten(); b = np.asarray(b).flatten()
    return float(np.sqrt(np.mean((a - b) ** 2)))


def bench(fn, runs=N_RUNS, warmup=1):
    for _ in range(warmup):   # descarta o custo unico de compilacao/cache
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


def _env():
    print("\nAmbiente:")
    print(f"  Python    : {sys.version.split()[0]}")
    for nm, mod in [("pdesolver", "pdesolver"), ("FiPy", "fipy"), ("py-pde", "pde")]:
        try:
            print(f"  {nm:<9} : {__import__(mod).__version__}")
        except Exception:
            pass
    print(f"  Plataforma: {platform.platform()}")


# -- PDESolver -----------------------------------------------------------------
try:
    from pdesolver import PDE as PDEOBJ, PDES
    HAS_PROPRIO = True
except ImportError:
    try:
        _src = str(__import__("pathlib").Path(__file__).resolve().parent.parent)
        if _src not in sys.path:
            sys.path.insert(0, _src)
        from PDE import PDE as PDEOBJ
        from PDES import PDES
        HAS_PROPRIO = True
    except ImportError:
        HAS_PROPRIO = False

# -- FiPy ----------------------------------------------------------------------
try:
    import fipy
    from fipy import CellVariable, Grid1D, TransientTerm, DiffusionTerm, ImplicitSourceTerm
    from fipy.tools import numerix
    HAS_FIPY = True
except ImportError:
    HAS_FIPY = False

# -- py-pde --------------------------------------------------------------------
try:
    import pde as pypde
    HAS_PYPDE = True
except ImportError:
    HAS_PYPDE = False


def analitica(x, t):
    return np.sin(np.pi * x) * np.cos(np.pi * C * t)


def proprio(method):
    def run():
        eq_u = PDEOBJ(eq="dU/dt = V", func="U", sp_var=["x"], ivar=["t"],
                      ivar_boundary=[(0, 1)], expr_ic="sin(pi*x)",
                      west_bd="Dirichlet", west_func_bd="0",
                      east_bd="Dirichlet", east_func_bd="0")
        eq_v = PDEOBJ(eq=f"dV/dt = {C**2}*d2U/dx2", func="V", sp_var=["x"], ivar=["t"],
                      ivar_boundary=[(0, 1)], expr_ic="0",
                      west_bd="Dirichlet", west_func_bd="0",
                      east_bd="Dirichlet", east_func_bd="0")
        sim = PDES([eq_u, eq_v], disc_n=[NX])
        sim.discretize(method="central")
        sim.solve(method=method, tf=TF, nt=NT)
        return np.array(sim.results[0][:NX])
    sol, st = bench(run)
    return rmse(sol, analitica(x_ref, TF)), st


def fipy_run():
    def run():
        mesh = Grid1D(nx=NX, dx=1.0 / NX)
        x_fp = np.array(mesh.cellCenters[0])
        U = CellVariable(name="U", mesh=mesh, value=numerix.sin(numerix.pi * x_fp))
        V = CellVariable(name="V", mesh=mesh, value=0.0)
        U.constrain(0., mesh.facesLeft); U.constrain(0., mesh.facesRight)
        V.constrain(0., mesh.facesLeft); V.constrain(0., mesh.facesRight)
        eq_U = TransientTerm(var=U) == ImplicitSourceTerm(coeff=1.0, var=V)
        eq_V = TransientTerm(var=V) == C**2 * DiffusionTerm(coeff=1.0, var=U)
        dt = TF / NT
        for _ in range(NT):
            eq_U.solve(var=U, dt=dt)
            eq_V.solve(var=V, dt=dt)
        x_c = np.array(mesh.cellCenters[0])
        f = interp1d(x_c, np.array(U.value), kind="linear", fill_value="extrapolate")
        return f(x_ref)
    sol, st = bench(run)
    return rmse(sol, analitica(x_ref, TF)), st


def pypde_run():
    def run():
        grid  = pypde.CartesianGrid([[0, 1]], NX)
        state = pypde.FieldCollection([
            pypde.ScalarField.from_expression(grid, "sin(pi*x)"),
            pypde.ScalarField.from_expression(grid, "0"),
        ])
        eq = pypde.PDE({"u": "v", "v": f"{C**2} * laplace(u)"}, bc={"value": 0})
        sol = eq.solve(state, t_range=TF, dt=TF / NT, tracker=None)
        return np.array(sol[0].data)
    sol, st = bench(run)
    return rmse(sol, analitica(x_ref, TF)), st


def main():
    print("=" * 70)
    print("BENCHMARK - Equacao da Onda 1D")
    print(f"c={C}, TF={TF}, NX={NX}, NT={NT}, N_RUNS={N_RUNS}")
    print("=" * 70)

    HDR = (f"  {'Metodo':<20} {'RMSE':>12}  {'t_med(s)':>9} {'dp(s)':>8} "
           f"{'t_min(s)':>9} {'CV(%)':>6}")
    print()
    print(HDR)
    print("  " + "-" * (len(HDR) - 2))

    linhas = []
    pdesolver_runs = [
        ("PDESolver/BDF2",  lambda: proprio("bdf2")),
        ("PDESolver/CN",    lambda: proprio("CN")),
        ("PDESolver/RKF45", lambda: proprio("RKF")),
    ]
    if HAS_PROPRIO:
        for lab, call in pdesolver_runs:
            try:
                r, st = call()
                print(fmt(lab, r, st)); linhas.append((lab, r, st))
            except Exception as e:
                print(f"  {lab:<20} {'(erro)':>12}   ({e})")
    else:
        print("  [pdesolver nao encontrado]")

    if HAS_FIPY:
        try:
            r, st = fipy_run()
            print(fmt("FiPy/splitting", r, st)); linhas.append(("FiPy/splitting", r, st))
        except Exception as e:
            print(f"  {'FiPy/splitting':<20} {'(erro)':>12}   ({e})")
    else:
        print("  [fipy nao encontrado]")

    if HAS_PYPDE:
        try:
            r, st = pypde_run()
            print(fmt("py-pde/LSODA", r, st)); linhas.append(("py-pde/LSODA", r, st))
        except Exception as e:
            print(f"  {'py-pde/LSODA':<20} {'(erro)':>12}   ({e})")
    else:
        print("  [py-pde nao encontrado]")

    print("  " + "-" * (len(HDR) - 2))
    if linhas:
        br = min(linhas, key=lambda z: z[1])
        bt = min(linhas, key=lambda z: z[2]["mean"])
        print(f"\n  Menor RMSE : {br[0]} ({br[1]:.4e})")
        print(f"  Menor tempo: {bt[0]} ({bt[2]['mean']:.4f}s +/- {bt[2]['std']:.4f}, "
              f"n={bt[2]['n']})")
    _env()


if __name__ == "__main__":
    main()

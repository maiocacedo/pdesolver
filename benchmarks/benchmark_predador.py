"""
benchmark_predador.py
Presa-Predador 1D:  dU/dt = Du*d2U/dx2 - U*V ;  dV/dt = Dv*d2V/dx2 + U*V.
Neumann homogeneo. Referencia: Radau em malha fina.
Compara PDESolver (BDF2, CN, RKF45), FiPy e py-pde.
Reporta RMSE e estatistica de tempo (media, desvio, minimo, CV) sobre N_RUNS.
"""
import numpy as np
import time
import warnings
import platform
import sys
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
warnings.filterwarnings("ignore")

NX = 41
TF = 1.0
NT = 200      # dt = 5e-3, config declarada (Tabela 2 do artigo)
DU = 0.1
DV = 0.05
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
    from fipy import CellVariable, Grid1D, TransientTerm, DiffusionTerm
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


def _lap_neu(f, h):
    d = np.empty_like(f)
    d[1:-1] = (f[2:] - 2*f[1:-1] + f[:-2]) / h**2
    d[0]    = 2*(f[1]  - f[0])  / h**2
    d[-1]   = 2*(f[-2] - f[-1]) / h**2
    return d


def referencia(NX_ref=401):
    h = 1.0 / (NX_ref - 1)
    x_r = np.linspace(0, 1, NX_ref)
    U0  = np.ones(NX_ref); V0 = np.sin(np.pi * x_r) ** 2
    def rhs(t, y):
        U, V = y[:NX_ref], y[NX_ref:]
        dU = DU * _lap_neu(U, h) - U * V
        dV = DV * _lap_neu(V, h) + U * V
        return np.concatenate([dU, dV])
    sol = solve_ivp(rhs, [0, TF], np.concatenate([U0, V0]),
                    method="Radau", rtol=1e-8, atol=1e-10)
    return x_r, sol.y[:NX_ref, -1], sol.y[NX_ref:, -1]


def proprio(method, x_ref, u_ref, v_ref):
    iu = interp1d(x_ref, u_ref, kind="cubic", fill_value="extrapolate")
    iv = interp1d(x_ref, v_ref, kind="cubic", fill_value="extrapolate")
    def run():
        eq_u = PDEOBJ(f"dU/dt = {DU}*d2U/dx2 - U*V", "U", ["x"], ["t"],
                      ivar_boundary=[(0, 1)], expr_ic="1",
                      west_bd="Neumann", west_func_bd="0",
                      east_bd="Neumann", east_func_bd="0")
        eq_v = PDEOBJ(f"dV/dt = {DV}*d2V/dx2 + U*V", "V", ["x"], ["t"],
                      ivar_boundary=[(0, 1)], expr_ic="sin(pi*x)**2",
                      west_bd="Neumann", west_func_bd="0",
                      east_bd="Neumann", east_func_bd="0")
        sim = PDES([eq_u, eq_v], [NX])
        sim.discretize(method="central")
        sim.solve(method=method, tf=TF, nt=NT)
        try:
            _, hist = sim.results
            return np.asarray(hist[0][-1]), np.asarray(hist[1][-1])
        except Exception:
            arr = np.asarray(sim.results[0]).flatten()
            return arr[:NX], arr[NX:2*NX]
    (u_f, v_f), st = bench(run)
    x = np.linspace(0, 1, NX)
    return (rmse(u_f, iu(x)) + rmse(v_f, iv(x))) / 2.0, st


def fipy_run(x_ref, u_ref, v_ref):
    dx = 1.0 / (NX - 1)
    iu = interp1d(x_ref, u_ref, kind="cubic", fill_value="extrapolate")
    iv = interp1d(x_ref, v_ref, kind="cubic", fill_value="extrapolate")
    def run():
        mesh = Grid1D(dx=dx, nx=NX - 1)
        U    = CellVariable(mesh=mesh, value=1.0)
        xc   = mesh.cellCenters[0]
        V    = CellVariable(mesh=mesh, value=numerix.sin(numerix.pi * xc) ** 2)
        eq_u = (TransientTerm(var=U) == DiffusionTerm(DU, var=U)
                - fipy.ImplicitSourceTerm(coeff=V, var=U))
        eq_v = (TransientTerm(var=V) == DiffusionTerm(DV, var=V)
                + fipy.ImplicitSourceTerm(coeff=U, var=V))
        dt = TF / NT
        for _ in range(NT):
            (eq_u & eq_v).solve(dt=dt)
        return np.asarray(xc), np.asarray(U), np.asarray(V)
    (xc, U_f, V_f), st = bench(run)
    return (rmse(U_f, iu(xc)) + rmse(V_f, iv(xc))) / 2.0, st


def pypde_run(x_ref, u_ref, v_ref):
    iu = interp1d(x_ref, u_ref, kind="cubic", fill_value="extrapolate")
    iv = interp1d(x_ref, v_ref, kind="cubic", fill_value="extrapolate")
    def run():
        grid  = pypde.CartesianGrid([[0, 1]], NX, periodic=False)
        U     = pypde.ScalarField(grid, data=1.0)
        V     = pypde.ScalarField.from_expression(grid, "sin(pi*x)**2")
        state = pypde.FieldCollection([U, V])
        eq    = pypde.PDE({"u": f"{DU}*laplace(u) - u*v",
                           "v": f"{DV}*laplace(v) + u*v"},
                          bc=[{"derivative": 0}, {"derivative": 0}])
        ctrl  = pypde.Controller(pypde.ScipySolver(eq, method="LSODA"),
                                 t_range=TF, tracker=None)
        return ctrl.run(state, dt=TF / NT)
    res, st = bench(run)
    x = res.grid.axes_coords[0]
    return (rmse(np.asarray(res[0].data), iu(x))
            + rmse(np.asarray(res[1].data), iv(x))) / 2.0, st


def main():
    print("=" * 70)
    print("BENCHMARK - Presa-Predador 1D")
    print(f"Du={DU}, Dv={DV}, TF={TF}, NX={NX}, NT={NT}, N_RUNS={N_RUNS}")
    print("=" * 70)
    print("Computando referencia (Radau)...", end=" ", flush=True)
    x_ref_pp, u_ref_pp, v_ref_pp = referencia()
    print("ok")
    HDR = (f"  {'Metodo':<20} {'RMSE':>12}  {'t_med(s)':>9} {'dp(s)':>8} "
           f"{'t_min(s)':>9} {'CV(%)':>6}")
    print()
    print(HDR)
    print("  " + "-" * (len(HDR) - 2))

    linhas = []
    pdesolver_runs = [
        ("PDESolver/BDF2",  lambda: proprio("bdf2", x_ref_pp, u_ref_pp, v_ref_pp)),
        ("PDESolver/CN",    lambda: proprio("CN",   x_ref_pp, u_ref_pp, v_ref_pp)),
        ("PDESolver/RKF45", lambda: proprio("RKF",  x_ref_pp, u_ref_pp, v_ref_pp)),
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
            r, st = fipy_run(x_ref_pp, u_ref_pp, v_ref_pp)
            print(fmt("FiPy/Picard-CN", r, st)); linhas.append(("FiPy/Picard-CN", r, st))
        except Exception as e:
            print(f"  {'FiPy/Picard-CN':<20} {'(erro)':>12}   ({e})")
    else:
        print("  [fipy nao encontrado]")

    if HAS_PYPDE:
        try:
            r, st = pypde_run(x_ref_pp, u_ref_pp, v_ref_pp)
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

"""Complete benchmark suite for pdesolver against FiPy and py-pde.

Three parts, selectable with --parte:

  casos        the six reference problems, accuracy and wall time
  capacidades  capability matrix: what each library can express, and whether
               the result is correct
  escala       how setup and solve cost grow with the mesh

Timing notes
------------
Wall-clock benchmarking is only meaningful on an idle machine. This script
reports the median and the minimum rather than the mean, together with the
coefficient of variation, and prints a warning whenever CV exceeds 5%. A run
with a high CV should not be used for publication figures.

FiPy is run under two time schemes, because they are not equivalent:
  'FiPy/Euler'  TransientTerm() == DiffusionTerm(...)   -- fully implicit,
                first order in time
  'FiPy/CN'     half implicit + half explicit           -- second order
Comparing a second-order method against 'FiPy/Euler' overstates the accuracy
difference; the second-order pairing is the fair one.

Usage
-----
    python benchmarks/benchmark_completo.py
    python benchmarks/benchmark_completo.py --parte casos --runs 50
    python benchmarks/benchmark_completo.py --json resultados.json
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import warnings

import matplotlib

matplotlib.use('Agg')
import numpy as np

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pdesolver import PDE, PDES

try:
    import fipy
    from fipy import (
        CellVariable,
        CentralDifferenceConvectionTerm,
        DiffusionTerm,
        Grid1D,
        Grid2D,
        ImplicitSourceTerm,
        TransientTerm,
    )
    from fipy.tools import numerix
    TEM_FIPY = True
except ImportError:
    TEM_FIPY = False

try:
    import pde as pypde
    TEM_PYPDE = True
except ImportError:
    TEM_PYPDE = False

CV_ALERTA = 5.0


# ----------------------------------------------------------------- ambiente

def _cpu_nome():
    try:
        if sys.platform == 'win32':
            out = subprocess.run(
                ['powershell', '-NoProfile', '-Command',
                 '(Get-CimInstance Win32_Processor).Name'],
                capture_output=True, text=True, timeout=20)
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip().splitlines()[0]
        elif sys.platform == 'darwin':
            out = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'],
                                 capture_output=True, text=True, timeout=20)
            if out.returncode == 0:
                return out.stdout.strip()
        else:
            with open('/proc/cpuinfo') as fh:
                for linha in fh:
                    if 'model name' in linha:
                        return linha.split(':', 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or 'desconhecido'


def calibracao(repeticoes=7):
    a = np.random.default_rng(0).standard_normal((600, 600))
    tempos = []
    for _ in range(repeticoes):
        t0 = time.perf_counter()
        a @ a
        tempos.append(time.perf_counter() - t0)
    return float(np.median(tempos))


def ambiente():
    versoes = {'python': sys.version.split()[0], 'numpy': np.__version__}
    for nome, mod in (('pdesolver', 'pdesolver'), ('scipy', 'scipy'),
                      ('sympy', 'sympy'), ('fipy', 'fipy'), ('py-pde', 'pde')):
        try:
            versoes[nome] = __import__(mod).__version__
        except Exception:
            versoes[nome] = 'ausente'
    try:
        from pdesolver.Solvers.RKF import _CUPY_AVAILABLE
        gpu = 'disponivel' if _CUPY_AVAILABLE else 'ausente'
    except Exception:
        gpu = 'desconhecido'
    return {
        'cpu': _cpu_nome(),
        'nucleos_logicos': os.cpu_count(),
        'plataforma': platform.platform(),
        'versoes': versoes,
        'gpu_cupy': gpu,
        'calibracao_dgemm_600_s': round(calibracao(), 5),
    }


def imprime_ambiente(env):
    print('\nAmbiente')
    print(f"  CPU           : {env['cpu']}  ({env['nucleos_logicos']} nucleos logicos)")
    print(f"  Plataforma    : {env['plataforma']}")
    print(f"  GPU (CuPy)    : {env['gpu_cupy']}")
    print(f"  Calibracao    : {env['calibracao_dgemm_600_s']:.5f}s "
          f"(produto 600x600, mediana) -- use para comparar entre maquinas")
    print('  Versoes       : ' + ', '.join(
        f'{k}={v}' for k, v in env['versoes'].items()))


# ------------------------------------------------------------------ medicao

def bench(fn, runs, warmup=2):
    for _ in range(warmup):
        resultado = fn()
    tempos = []
    for _ in range(runs):
        t0 = time.perf_counter()
        resultado = fn()
        tempos.append(time.perf_counter() - t0)
    t = np.array(tempos)
    mediana = float(np.median(t))
    st = {
        'n': int(t.size),
        'mediana': mediana,
        'min': float(t.min()),
        'media': float(t.mean()),
        'mad': float(np.median(np.abs(t - mediana))),
        'cv': float(t.std(ddof=1) / t.mean() * 100.0) if t.size > 1 else 0.0,
    }
    return resultado, st


def rmse(a, b):
    a = np.asarray(a, dtype=float).flatten()
    b = np.asarray(b, dtype=float).flatten()
    return float(np.sqrt(np.mean((a - b) ** 2)))


def linha(rotulo, err, st):
    marca = ' !' if st['cv'] > CV_ALERTA else '  '
    return (f"  {rotulo:<24}{err:12.4e}{st['mediana']:10.4f}"
            f"{st['min']:10.4f}{st['cv']:8.1f}{marca}")


CABECALHO = (f"  {'Metodo':<24}{'RMSE':>12}{'mediana(s)':>10}"
             f"{'min(s)':>10}{'CV(%)':>8}")


# -------------------------------------------------------------------- casos

def _pdesolver(pdes, disc_n, metodo, backend, tf, nt, disc='central'):
    def executa():
        sim = PDES([PDE(**p) for p in pdes], disc_n, backend=backend)
        sim.discretize(method=disc)
        sim.solve(method=metodo, tf=tf, nt=nt)
        _, hist = sim.results
        return [np.asarray(h[-1]) for h in hist]
    return executa


def caso_calor1d(cfg):
    nx, tf, nt, al = cfg['nx'], cfg['tf'], cfg['nt'], 0.1
    x = np.linspace(0, 1, nx)
    ref = np.sin(np.pi * x) * np.exp(-np.pi ** 2 * al * tf)

    pdes = [dict(eq=f'dU/dt = {al}*d2U/dx2', func='U', sp_var=['x'],
                 ivar=['t'], ivar_boundary=[(0, 1)], expr_ic='sin(pi*x)',
                 west_bd='Dirichlet', west_func_bd='0',
                 east_bd='Dirichlet', east_func_bd='0')]

    def fipy_fn(esquema):
        def executa():
            malha = Grid1D(dx=1.0 / (nx - 1), nx=nx - 1)
            phi = CellVariable(mesh=malha)
            xc = malha.cellCenters[0]
            phi.setValue(numerix.sin(numerix.pi * xc))
            phi.constrain(0., malha.facesLeft)
            phi.constrain(0., malha.facesRight)
            if esquema == 'euler':
                eq = TransientTerm() == DiffusionTerm(coeff=al)
            else:
                eq = (TransientTerm() == DiffusionTerm(coeff=0.5 * al)
                      + 0.5 * al * phi.faceGrad.divergence)
            for _ in range(nt):
                eq.solve(var=phi, dt=tf / nt)
            return np.asarray(xc), np.asarray(phi)
        return executa

    def fipy_err(saida):
        xc, v = saida
        return rmse(v, np.sin(np.pi * xc) * np.exp(-np.pi ** 2 * al * tf))

    def pypde_fn():
        grade = pypde.CartesianGrid([[0, 1]], nx, periodic=False)
        campo = pypde.ScalarField.from_expression(grade, 'sin(pi*x)')
        eq = pypde.PDE({'U': f'{al}*laplace(U)'},
                       bc=[{'value': 0}, {'value': 0}])
        ctrl = pypde.Controller(pypde.ScipySolver(eq, method='LSODA'),
                                t_range=tf, tracker=None)
        return ctrl.run(campo, dt=tf / nt)

    def pypde_err(res):
        xs = res.grid.axes_coords[0]
        return rmse(res.data, np.sin(np.pi * xs) * np.exp(-np.pi ** 2 * al * tf))

    return dict(nome='Calor 1D', pdes=pdes, disc_n=[nx], disc='central',
                ref=lambda s: rmse(s[0], ref), fipy=fipy_fn, fipy_err=fipy_err,
                pypde=pypde_fn, pypde_err=pypde_err, tf=tf, nt=nt)


def caso_calor2d(cfg):
    n, tf, nt, al = cfg['nx2d'], cfg['tf'], cfg['nt'], 0.1
    x = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, x, indexing='ij')
    ref = (np.sin(np.pi * X) * np.sin(np.pi * Y)
           * np.exp(-2 * np.pi ** 2 * al * tf)).flatten()

    pdes = [dict(eq=f'dF/dt = {al}*d2F/dx2 + {al}*d2F/dy2', func='F',
                 sp_var=['x', 'y'], ivar=['t'],
                 ivar_boundary=[(0, 1), (0, 1)],
                 expr_ic='sin(pi*x)*sin(pi*y)',
                 west_bd='Dirichlet', west_func_bd='0',
                 east_bd='Dirichlet', east_func_bd='0',
                 north_bd='Dirichlet', north_func_bd='0',
                 south_bd='Dirichlet', south_func_bd='0')]

    def fipy_fn(esquema):
        def executa():
            malha = Grid2D(dx=1.0 / (n - 1), dy=1.0 / (n - 1),
                           nx=n - 1, ny=n - 1)
            phi = CellVariable(mesh=malha)
            xc, yc = malha.cellCenters
            phi.setValue(numerix.sin(numerix.pi * xc)
                         * numerix.sin(numerix.pi * yc))
            for face in (malha.facesLeft, malha.facesRight,
                         malha.facesTop, malha.facesBottom):
                phi.constrain(0., face)
            if esquema == 'euler':
                eq = TransientTerm() == DiffusionTerm(coeff=al)
            else:
                eq = (TransientTerm() == DiffusionTerm(coeff=0.5 * al)
                      + 0.5 * al * phi.faceGrad.divergence)
            for _ in range(nt):
                eq.solve(var=phi, dt=tf / nt)
            return np.asarray(xc), np.asarray(yc), np.asarray(phi)
        return executa

    def fipy_err(saida):
        xc, yc, v = saida
        exato = (np.sin(np.pi * xc) * np.sin(np.pi * yc)
                 * np.exp(-2 * np.pi ** 2 * al * tf))
        return rmse(v, exato)

    def pypde_fn():
        grade = pypde.CartesianGrid([[0, 1], [0, 1]], [n, n],
                                    periodic=[False, False])
        campo = pypde.ScalarField.from_expression(grade,
                                                  'sin(pi*x) * sin(pi*y)')
        eq = pypde.PDE({'F': f'{al}*laplace(F)'},
                       bc=[{'value': 0}, {'value': 0}])
        ctrl = pypde.Controller(pypde.ScipySolver(eq, method='LSODA'),
                                t_range=tf, tracker=None)
        return ctrl.run(campo, dt=tf / nt)

    def pypde_err(res):
        c = res.grid.cell_coords
        exato = (np.sin(np.pi * c[..., 0]) * np.sin(np.pi * c[..., 1])
                 * np.exp(-2 * np.pi ** 2 * al * tf))
        return rmse(res.data, exato)

    return dict(nome='Calor 2D', pdes=pdes, disc_n=[n, n], disc='central',
                ref=lambda s: rmse(s[0], ref), fipy=fipy_fn, fipy_err=fipy_err,
                pypde=pypde_fn, pypde_err=pypde_err, tf=tf, nt=nt)


def caso_adveccao(cfg):
    from scipy.integrate import solve_ivp
    from scipy.interpolate import interp1d
    nx, tf, nt, c, d = cfg['nx'], cfg['tf'], cfg['nt'], 1.0, 0.05

    nr = 401
    hr = 1.0 / (nr - 1)
    xr = np.linspace(0, 1, nr)

    def rhs(t, u):
        du = np.zeros_like(u)
        du[1:-1] = (d * (u[2:] - 2 * u[1:-1] + u[:-2]) / hr ** 2
                    - c * (u[2:] - u[:-2]) / (2.0 * hr))
        return du

    sol = solve_ivp(rhs, [0, tf], np.sin(np.pi * xr), method='Radau',
                    rtol=1e-10, atol=1e-12)
    interp = interp1d(xr, sol.y[:, -1], kind='cubic',
                      fill_value='extrapolate')
    ref = interp(np.linspace(0, 1, nx))

    pdes = [dict(eq=f'dU/dt = -{c}*dU/dx + {d}*d2U/dx2', func='U',
                 sp_var=['x'], ivar=['t'], ivar_boundary=[(0, 1)],
                 expr_ic='sin(pi*x)',
                 west_bd='Dirichlet', west_func_bd='0',
                 east_bd='Dirichlet', east_func_bd='0')]

    def fipy_fn(esquema):
        def executa():
            malha = Grid1D(dx=1.0 / (nx - 1), nx=nx - 1)
            phi = CellVariable(mesh=malha)
            xc = malha.cellCenters[0]
            phi.setValue(numerix.sin(numerix.pi * xc))
            phi.constrain(0., malha.facesLeft)
            phi.constrain(0., malha.facesRight)
            if esquema == 'euler':
                eq = (TransientTerm() == DiffusionTerm(coeff=d)
                      - CentralDifferenceConvectionTerm(coeff=[[c]]))
            else:
                eq = (TransientTerm()
                      == DiffusionTerm(coeff=0.5 * d)
                      + 0.5 * d * phi.faceGrad.divergence
                      - CentralDifferenceConvectionTerm(coeff=[[c]]))
            for _ in range(nt):
                eq.solve(var=phi, dt=tf / nt)
            return np.asarray(xc), np.asarray(phi)
        return executa

    def fipy_err(saida):
        xc, v = saida
        return rmse(v, interp(xc))

    def pypde_fn():
        grade = pypde.CartesianGrid([[0, 1]], nx, periodic=False)
        campo = pypde.ScalarField.from_expression(grade, 'sin(pi*x)')
        eq = pypde.PDE({'U': f'{d}*laplace(U) - {c}*d_dx(U)'},
                       bc=[{'value': 0}, {'value': 0}])
        ctrl = pypde.Controller(pypde.ScipySolver(eq, method='LSODA'),
                                t_range=tf, tracker=None)
        return ctrl.run(campo, dt=tf / nt)

    def pypde_err(res):
        return rmse(res.data, interp(res.grid.axes_coords[0]))

    return dict(nome='Adveccao-Difusao', pdes=pdes, disc_n=[nx],
                disc='central', ref=lambda s: rmse(s[0], ref), fipy=fipy_fn,
                fipy_err=fipy_err, pypde=pypde_fn, pypde_err=pypde_err,
                tf=tf, nt=nt)


def caso_burgers(cfg):
    nx, tf, nt, nu = cfg['nx'], cfg['tf'], cfg['nt'], 0.05

    def analitica(x, t, termos=80):
        xi = np.linspace(0, 1, 4000)
        phi0 = np.exp((np.cos(np.pi * xi) - 1.0) / (2.0 * np.pi * nu))
        a = np.zeros(termos + 1)
        for k in range(termos + 1):
            val = np.trapezoid(phi0 * np.cos(k * np.pi * xi), xi)
            a[k] = val if k == 0 else 2.0 * val
        narr = np.arange(termos + 1, dtype=float)
        decai = np.exp(-nu * (narr * np.pi) ** 2 * t)
        u = np.empty_like(x, dtype=float)
        for i, xv in enumerate(x):
            ph = np.dot(a * decai, np.cos(narr * np.pi * xv))
            dph = -np.dot(a * decai * narr * np.pi, np.sin(narr * np.pi * xv))
            u[i] = (-2.0 * nu * dph / ph) if abs(ph) > 1e-15 else 0.0
        return u

    xg = np.linspace(0, 1, nx)
    ref = analitica(xg, tf)

    pdes = [dict(eq=f'dU/dt = -U*dU/dx + {nu}*d2U/dx2', func='U',
                 sp_var=['x'], ivar=['t'], ivar_boundary=[(0, 1)],
                 expr_ic='sin(pi*x)',
                 west_bd='Dirichlet', west_func_bd='0',
                 east_bd='Dirichlet', east_func_bd='0')]

    def fipy_fn(esquema):
        def executa():
            dx = 1.0 / (nx - 1)
            malha = Grid1D(dx=dx, nx=nx - 1)
            phi = CellVariable(mesh=malha)
            xc = malha.cellCenters[0]
            phi.setValue(numerix.sin(numerix.pi * xc))
            phi.constrain(0., malha.facesLeft)
            phi.constrain(0., malha.facesRight)
            if esquema == 'euler':
                eq = TransientTerm() == DiffusionTerm(coeff=nu)
            else:
                eq = (TransientTerm() == DiffusionTerm(coeff=0.5 * nu)
                      + 0.5 * nu * phi.faceGrad.divergence)
            for _ in range(nt):
                p = np.asarray(phi.value)
                adv = np.zeros_like(p)
                adv[1:-1] = p[1:-1] * (p[2:] - p[:-2]) / (2.0 * dx)
                phi.setValue(p - (tf / nt) * adv)
                eq.solve(var=phi, dt=tf / nt)
            return np.asarray(xc), np.asarray(phi)
        return executa

    def fipy_err(saida):
        xc, v = saida
        return rmse(v, analitica(np.asarray(xc), tf))

    def pypde_fn():
        grade = pypde.CartesianGrid([[0, 1]], nx, periodic=False)
        campo = pypde.ScalarField.from_expression(grade, 'sin(pi*x)')
        eq = pypde.PDE({'U': f'{nu}*laplace(U) - U*d_dx(U)'},
                       bc=[{'value': 0}, {'value': 0}])
        ctrl = pypde.Controller(pypde.ScipySolver(eq, method='LSODA'),
                                t_range=tf, tracker=None)
        return ctrl.run(campo, dt=tf / nt)

    def pypde_err(res):
        return rmse(res.data, analitica(res.grid.axes_coords[0], tf))

    return dict(nome='Burgers Viscosa', pdes=pdes, disc_n=[nx],
                disc='central', ref=lambda s: rmse(s[0], ref), fipy=fipy_fn,
                fipy_err=fipy_err, pypde=pypde_fn, pypde_err=pypde_err,
                tf=tf, nt=nt)


CONSTRUTORES = [caso_calor1d, caso_calor2d, caso_adveccao, caso_burgers]


def parte_casos(cfg, runs, saida):
    print('\n' + '=' * 72)
    print('PARTE 1 - problemas de referencia')
    print('=' * 72)
    print('  FiPy/Euler  = totalmente implicito, 1a ordem no tempo')
    print('  FiPy/CN     = Crank-Nicolson real, 2a ordem  <- comparacao justa')

    for construtor in CONSTRUTORES:
        caso = construtor(cfg)
        print(f"\n{caso['nome']}  (nt={caso['nt']}, tf={caso['tf']})")
        print(CABECALHO)
        print('  ' + '-' * (len(CABECALHO) - 2))
        registros = []

        for metodo in ('bdf2', 'CN', 'RKF'):
            for backend in ('symbolic', 'stencil'):
                fn = _pdesolver(caso['pdes'], caso['disc_n'], metodo, backend,
                                caso['tf'], caso['nt'], caso['disc'])
                sol, st = bench(fn, runs)
                err = caso['ref'](sol)
                rotulo = f"pdesolver/{metodo}/{backend[:4]}"
                print(linha(rotulo, err, st))
                registros.append((rotulo, err, st))

        if TEM_FIPY:
            for esquema, rot in (('euler', 'FiPy/Euler'), ('cn', 'FiPy/CN')):
                try:
                    sol, st = bench(caso['fipy'](esquema), runs)
                    err = caso['fipy_err'](sol)
                    print(linha(rot, err, st))
                    registros.append((rot, err, st))
                except Exception as exc:
                    print(f'  {rot:<24}(erro) {type(exc).__name__}')

        if TEM_PYPDE:
            try:
                sol, st = bench(caso['pypde'], runs)
                err = caso['pypde_err'](sol)
                print(linha('py-pde/LSODA', err, st))
                registros.append(('py-pde/LSODA', err, st))
            except Exception as exc:
                print(f'  {"py-pde/LSODA":<24}(erro) {type(exc).__name__}')

        print('  ' + '-' * (len(CABECALHO) - 2))
        melhor_err = min(registros, key=lambda r: r[1])
        melhor_t = min(registros, key=lambda r: r[2]['mediana'])
        print(f"  menor RMSE : {melhor_err[0]} ({melhor_err[1]:.4e})")
        print(f"  menor tempo: {melhor_t[0]} ({melhor_t[2]['mediana']:.4f}s)")
        saida.setdefault('casos', {})[caso['nome']] = [
            {'metodo': r[0], 'rmse': r[1], **r[2]} for r in registros
        ]


# -------------------------------------------------------------- capacidades

def _cap_pdesolver_mista():
    n = 33
    k = 2 * np.pi
    lam = 3.0 * k ** 2
    tf = 0.005
    p = PDE('du/dt = d2u/dx2 + 1.0*d2u/dxdy + d2u/dy2', 'u', ['x', 'y'],
            ['t'], ivar_boundary=[(0, 1), (0, 1)],
            expr_ic='sin(2*pi*x + 2*pi*y)',
            west_bd='Periodic', east_bd='Periodic',
            north_bd='Periodic', south_bd='Periodic')
    sim = PDES([p], [n, n])
    sim.discretize(method='central')
    sim.solve(method='CN', tf=tf, nt=200)
    X, Y = sim.grid.coords()
    exato = (np.sin(k * X + k * Y) * np.exp(-lam * tf)).flatten()
    return rmse(np.asarray(sim.results[0]), exato)


def _cap_pypde_mista():
    n = 33
    k = 2 * np.pi
    lam = 3.0 * k ** 2
    tf = 0.005
    grade = pypde.CartesianGrid([[0, 1], [0, 1]], [n, n], periodic=True)
    campo = pypde.ScalarField.from_expression(grade, 'sin(2*pi*x + 2*pi*y)')
    eq = pypde.PDE({'u': 'laplace(u) + 1.0*d_dx(d_dy(u))'})
    res = eq.solve(campo, t_range=tf, dt=tf / 200, tracker=None)
    c = res.grid.cell_coords
    exato = np.sin(k * c[..., 0] + k * c[..., 1]) * np.exp(-lam * tf)
    return rmse(res.data, exato)


def _cap_pdesolver_naolinear():
    p = PDE('dU/dt = 0.05*d2U/dx2 + 0.5*dU/dx*dU/dx*d2U/dx2', 'U', ['x'],
            ['t'], ivar_boundary=[(0, 1)], expr_ic='sin(pi*x)',
            west_bd='Dirichlet', west_func_bd='0',
            east_bd='Dirichlet', east_func_bd='0')
    sim = PDES([p], [41])
    sim.discretize(method='central')
    sim.solve(method='bdf2', tf=0.2, nt=400)
    u = np.asarray(sim.results[0])
    return float(u[20])


def _cap_pypde_naolinear():
    grade = pypde.CartesianGrid([[0, 1]], 41, periodic=False)
    campo = pypde.ScalarField.from_expression(grade, 'sin(pi*x)')
    eq = pypde.PDE({'U': '0.05*laplace(U) + 0.5*d_dx(U)*d_dx(U)*laplace(U)'},
                   bc=[{'value': 0}, {'value': 0}])
    ctrl = pypde.Controller(pypde.ScipySolver(eq, method='LSODA'),
                            t_range=0.2, tracker=None)
    res = ctrl.run(campo, dt=5e-4)
    return float(res.data[20])


def _cap_pdesolver_naouniforme():
    p = PDE('du/dt = -du/dx + 0.005*d2u/dx2', 'u', ['x'], ['t'],
            ivar_boundary=[(0, 1)], expr_ic='0',
            west_bd='Dirichlet', west_func_bd='0',
            east_bd='Dirichlet', east_func_bd='1')
    ref = lambda x: (np.exp(x / 0.005) - 1) / (np.exp(1 / 0.005) - 1)
    erros = {}
    for rotulo, malha in (('uniforme', 'uniform'),
                          ('tanh_right', {'type': 'tanh_right', 'beta': 5.0})):
        sim = PDES([p], [41], mesh=malha)
        sim.discretize(method='central')
        sim.solve(method='bdf2', tf=3.0, nt=600)
        x = sim.grid.axes[0].nodes
        erros[rotulo] = round(rmse(np.asarray(sim.results[0]), ref(x)), 6)
    erros['ganho'] = round(erros['uniforme'] / erros['tanh_right'], 1)
    return erros


def _cap_pdesolver_regiao():
    n = 31
    p = PDE('dU/dt = 0.2*d2U/dx2 + 0.2*d2U/dy2', 'U', ['x', 'y'], ['t'],
            ivar_boundary=[(0, 1), (0, 1)], expr_ic='1',
            west_bd='Dirichlet', west_func_bd='0',
            east_bd='Dirichlet', east_func_bd='0',
            north_bd='Dirichlet', north_func_bd='0',
            south_bd='Dirichlet', south_func_bd='0')
    sim = PDES([p], [n, n])
    X, Y = sim.grid.coords()
    bloco = (np.abs(X - 0.5) < 0.12) & (np.abs(Y - 0.5) < 0.12)
    sim.discretize(method='central',
                   regions=[{'where': bloco, 'eq': 'dU/dt = 0'}])
    sim.solve(method='bdf2', tf=0.2, nt=100)
    u = np.asarray(sim.results[0]).reshape(n, n)
    return bool(np.allclose(u[bloco], 1.0)) and u[3, 3] < 0.5


def _cap_pdesolver_analise():
    p = PDE('du/dt = -1.0*du/dx + 0.01*d2u/dx2', 'u', ['x'], ['t'],
            ivar_boundary=[(0, 1)], expr_ic='0')
    sim = PDES([p], [51], backend='stencil')
    sim.discretize(method='backward')
    dados = sim.analyze(method='RKF', verbose=False)
    dif = dados['numerical_diffusion'][0]
    return {'difusao_numerica': round(float(dif['numerical']), 6),
            'difusao_fisica': round(float(dif['physical']), 6),
            'dt_max': round(float(dados['stability']['dt_max']), 6)}


def parte_capacidades(saida):
    print('\n' + '=' * 72)
    print('PARTE 2 - matriz de capacidades')
    print('=' * 72)
    print('  declarativo = expressavel na notacao da propria biblioteca')
    print('  ausente     = sem sintaxe/termo; exigiria codigo numerico manual')

    testes = [
        ('Derivada mista d2u/dxdy', [
            ('pdesolver', _cap_pdesolver_mista, 'RMSE'),
            ('py-pde', _cap_pypde_mista if TEM_PYPDE else None, 'RMSE'),
            ('FiPy', None, 'sem termo de derivada cruzada'),
        ]),
        ('Produto nao linear de derivadas', [
            ('pdesolver', _cap_pdesolver_naolinear, 'max|u|'),
            ('py-pde', _cap_pypde_naolinear if TEM_PYPDE else None, 'max|u|'),
            ('FiPy', None, 'exige decomposicao manual em Terms'),
        ]),
        ('Malha nao uniforme', [
            ('pdesolver', _cap_pdesolver_naouniforme, 'RMSE por malha'),
            ('py-pde', None, 'CartesianGrid e uniforme'),
            ('FiPy', 'nativo', 'Grid1D(dx=array)'),
        ]),
        ('Equacao propria por no (regiao)', [
            ('pdesolver', _cap_pdesolver_regiao, 'obstaculo preservado'),
            ('py-pde', None, 'sem sintaxe declarativa para mascara'),
            ('FiPy', 'manual', 'possivel via mascaras, nao declarativo'),
        ]),
        ('Analise da discretizacao', [
            ('pdesolver', _cap_pdesolver_analise, 'erro/estabilidade'),
            ('py-pde', None, 'nao disponivel'),
            ('FiPy', None, 'nao disponivel'),
        ]),
    ]

    for titulo, entradas in testes:
        print(f'\n  {titulo}')
        for lib, alvo, nota in entradas:
            if alvo is None:
                print(f'    {lib:<12} ausente        ({nota})')
            elif isinstance(alvo, str):
                print(f'    {lib:<12} {alvo:<14} ({nota})')
            else:
                try:
                    valor = alvo()
                    print(f'    {lib:<12} declarativo    {nota} = {valor}')
                    saida.setdefault('capacidades', {})[f'{titulo}/{lib}'] = str(valor)
                except Exception as exc:
                    print(f'    {lib:<12} FALHOU         {type(exc).__name__}: '
                          f'{str(exc)[:45]}')


# -------------------------------------------------------------------- escala

def parte_escala(saida, tamanhos=(21, 41, 81, 161, 321)):
    print('\n' + '=' * 72)
    print('PARTE 3 - escalabilidade (calor 2D, montagem + 20 passos)')
    print('=' * 72)
    print(f"  {'N':>5}{'DOF':>9}{'simbolico':>12}{'stencil':>10}{'ganho':>9}")
    print('  ' + '-' * 45)
    for n in tamanhos:
        tempos = {}
        for backend in ('symbolic', 'stencil'):
            if backend == 'symbolic' and n > 81:
                tempos[backend] = float('nan')
                continue
            t0 = time.perf_counter()
            p = PDE('dF/dt = 0.1*d2F/dx2 + 0.1*d2F/dy2', 'F', ['x', 'y'],
                    ['t'], ivar_boundary=[(0, 1), (0, 1)],
                    expr_ic='sin(pi*x)*sin(pi*y)',
                    west_bd='Dirichlet', west_func_bd='0',
                    east_bd='Dirichlet', east_func_bd='0',
                    north_bd='Dirichlet', north_func_bd='0',
                    south_bd='Dirichlet', south_func_bd='0')
            sim = PDES([p], [n, n], backend=backend)
            sim.discretize(method='central')
            sim.solve(method='bdf2', tf=0.01, nt=20)
            tempos[backend] = time.perf_counter() - t0
        ts, tv = tempos['symbolic'], tempos['stencil']
        if ts == ts:
            print(f'  {n:5d}{n*n:9d}{ts:11.3f}s{tv:9.3f}s{ts/tv:8.0f}x')
        else:
            print(f'  {n:5d}{n*n:9d}{"inviavel":>12}{tv:9.3f}s')
        saida.setdefault('escala', {})[n] = tempos


# ---------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--parte', default='tudo',
                    choices=['tudo', 'casos', 'capacidades', 'escala'])
    ap.add_argument('--runs', type=int, default=30)
    ap.add_argument('--nx', type=int, default=41)
    ap.add_argument('--nx2d', type=int, default=21)
    ap.add_argument('--nt', type=int, default=200)
    ap.add_argument('--tf', type=float, default=1.0)
    ap.add_argument('--json', default=None)
    args = ap.parse_args()

    cfg = {'nx': args.nx, 'nx2d': args.nx2d, 'nt': args.nt, 'tf': args.tf}
    env = ambiente()

    print('=' * 72)
    print('pdesolver - benchmark completo')
    print('=' * 72)
    imprime_ambiente(env)
    print(f'\n  repeticoes por medida: {args.runs} (mais 2 de aquecimento)')
    print(f'  alerta de ruido: CV > {CV_ALERTA}% marcado com "!"')

    saida = {'ambiente': env, 'config': vars(args)}
    if args.parte in ('tudo', 'casos'):
        parte_casos(cfg, args.runs, saida)
    if args.parte in ('tudo', 'capacidades'):
        parte_capacidades(saida)
    if args.parte in ('tudo', 'escala'):
        parte_escala(saida)

    if args.json:
        with open(args.json, 'w', encoding='utf-8') as fh:
            json.dump(saida, fh, indent=2, ensure_ascii=False, default=str)
        print(f'\n  resultados gravados em {args.json}')


if __name__ == '__main__':
    main()

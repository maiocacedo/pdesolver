
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import pytest
from pdesolver import PDE, PDES

DISC_N   = [15, 15]
TF       = 1.0
NT       = 200
TOL_MAE  = 1e-3
ALPHA_X  = 0.1
ALPHA_Y  = 0.2

x = np.linspace(0, 1, DISC_N[0])
y = np.linspace(0, 1, DISC_N[1])
X, Y = np.meshgrid(x, y, indexing='ij')


def solucao_analitica(X, Y, t):
    return np.sin(np.pi * X) * np.sin(np.pi * Y) * np.exp(-(ALPHA_X + ALPHA_Y) * np.pi**2 * t)


def mae(numerica, analitica):
    return np.mean(np.abs(np.array(numerica) - analitica.flatten()))


def montar_sistema():
    pde = PDE(
        f'dF/dt = {ALPHA_X}*d2F/dx2 + {ALPHA_Y}*d2F/dy2',
        'F', ['x', 'y'], ['t'],
        ivar_boundary=[(0, 1), (0, 1)],
        expr_ic='sin(pi * x) * sin(pi * y)',
        west_bd='Dirichlet',  west_func_bd='0',
        east_bd='Dirichlet',  east_func_bd='0',
        north_bd='Dirichlet', north_func_bd='0',
        south_bd='Dirichlet', south_func_bd='0',
    )
    sim = PDES([pde], DISC_N)
    sim.discretize(method='central')
    return sim


ref = solucao_analitica(X, Y, TF).flatten()

def test_calor_bdf2():
    sim = montar_sistema()
    sim.solve(method='bdf2', tf=TF, nt=NT, tol=1e-8)
    erro = mae(sim.results[0], ref.reshape(DISC_N[0], DISC_N[1]))
    assert erro < TOL_MAE, f"BDF2 — MAE={erro:.2e} > {TOL_MAE:.0e}"


def test_calor_cn():
    sim = montar_sistema()
    sim.solve(method='CN', tf=TF, nt=NT, tol=1e-8)
    erro = mae(sim.results[0], ref.reshape(DISC_N[0], DISC_N[1]))
    assert erro < TOL_MAE, f"CN — MAE={erro:.2e} > {TOL_MAE:.0e}"


def test_calor_rkf():
    sim = montar_sistema()
    sim.solve(method='RKF', tf=TF, nt=NT, tol=1e-6)
    erro = mae(sim.results[0], ref.reshape(DISC_N[0], DISC_N[1]))
    assert erro < TOL_MAE, f"RKF — MAE={erro:.2e} > {TOL_MAE:.0e}"


def test_calor_ordem_bdf2():
    sim1 = montar_sistema()
    sim1.solve(method='bdf2', tf=TF, nt=NT, tol=1e-8)
    erro1 = mae(sim1.results[0], ref.reshape(DISC_N[0], DISC_N[1]))

    sim2 = montar_sistema()
    sim2.solve(method='bdf2', tf=TF, nt=NT * 2, tol=1e-8)
    erro2 = mae(sim2.results[0], ref.reshape(DISC_N[0], DISC_N[1]))

    assert erro2 < erro1 * 1.05, (
        f"BDF2 piorou mais de 5% com refinamento de dt — possível instabilidade: "
        f"erro(nt={NT})={erro1:.2e}, erro(nt={NT*2})={erro2:.2e}"
    )
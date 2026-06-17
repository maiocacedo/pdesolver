import numpy as np
import matplotlib
matplotlib.use('Agg')
import pytest
from pdesolver import PDE, PDES

DISC_N  = [15, 15]
TF      = 1.0
NT      = 500
TOL_MAE = 1e-2 
NU      = 0.1

x = np.linspace(0, 1, DISC_N[0])
y = np.linspace(0, 1, DISC_N[1])
X, Y = np.meshgrid(x, y, indexing='ij')


def solucao_analitica(X, Y, t):
    return 1.0 / (1.0 + np.exp((X + Y - t) / (2 * NU)))


def mae(numerica, analitica):
    return np.mean(np.abs(np.array(numerica) - analitica.flatten()))


def montar_sistema():
    bc_expr = f"1 / (1 + exp((x + y - t) / (2*{NU})))"
    ic_expr = f"1 / (1 + exp((x + y) / (2*{NU})))"

    pde = PDE(
        f'dU/dt = -U*dU/dx - U*dU/dy + {NU}*d2U/dx2 + {NU}*d2U/dy2',
        'U', ['x', 'y'], ['t'],
        ivar_boundary=[(0, 1), (0, 1)],
        expr_ic=ic_expr,
        west_bd='Dirichlet',  west_func_bd=bc_expr,
        east_bd='Dirichlet',  east_func_bd=bc_expr,
        north_bd='Dirichlet', north_func_bd=bc_expr,
        south_bd='Dirichlet', south_func_bd=bc_expr,
    )
    sim = PDES([pde], DISC_N)
    sim.discretize(method='backward')
    return sim


ref = solucao_analitica(X, Y, TF).flatten()

def test_burgers_bdf2():
    sim = montar_sistema()
    sim.solve(method='bdf2', tf=TF, nt=NT, tol=1e-8)
    erro = mae(sim.results[0], ref)
    assert erro < TOL_MAE, f"BDF2 — MAE={erro:.2e} > {TOL_MAE:.0e}"


def test_burgers_cn():
    sim = montar_sistema()
    sim.solve(method='CN', tf=TF, nt=NT, tol=1e-8)
    erro = mae(sim.results[0], ref)
    assert erro < TOL_MAE, f"CN — MAE={erro:.2e} > {TOL_MAE:.0e}"


def test_burgers_rkf():
    sim = montar_sistema()
    sim.solve(method='RKF', tf=TF, nt=NT, tol=1e-6)
    erro = mae(sim.results[0], ref)
    assert erro < TOL_MAE, f"RKF — MAE={erro:.2e} > {TOL_MAE:.0e}"
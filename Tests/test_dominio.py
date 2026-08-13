import matplotlib
import numpy as np

matplotlib.use('Agg')
import pytest

from pdesolver import PDE, PDES

TF = 0.1
NT = 200


def mae(numerica, analitica):
    return np.mean(np.abs(np.array(numerica) - analitica))


def test_dominio_nao_unitario_1d():
    L = 2.0
    pde = PDE(
        'du/dt = d2u/dx2',
        'u', ['x'], ['t'],
        ivar_boundary=[(0, L)],
        expr_ic=f'sin(pi*x/{L})',
        west_bd='Dirichlet', west_func_bd='0',
        east_bd='Dirichlet', east_func_bd='0',
    )
    sim = PDES([pde], [41])
    sim.discretize(method='central')
    sim.solve(method='bdf2', tf=TF, nt=NT)
    x = sim.grid.axes[0].nodes
    ref = np.sin(np.pi * x / L) * np.exp(-(np.pi / L) ** 2 * TF)
    erro = mae(sim.results[0], ref)
    assert erro < 1e-4, f"Domínio [0, {L}] — MAE={erro:.2e}"


def test_dominio_deslocado_1d():
    a, b = 1.0, 3.0
    pde = PDE(
        'du/dt = d2u/dx2',
        'u', ['x'], ['t'],
        ivar_boundary=[(a, b)],
        expr_ic=f'sin(pi*(x-{a})/{b-a})',
        west_bd='Dirichlet', west_func_bd='0',
        east_bd='Dirichlet', east_func_bd='0',
    )
    sim = PDES([pde], [41])
    sim.discretize(method='central')
    sim.solve(method='bdf2', tf=TF, nt=NT)
    x = sim.grid.axes[0].nodes
    ref = np.sin(np.pi * (x - a) / (b - a)) * np.exp(
        -(np.pi / (b - a)) ** 2 * TF
    )
    erro = mae(sim.results[0], ref)
    assert erro < 1e-4, f"Domínio [{a}, {b}] — MAE={erro:.2e}"


def test_malha_retangular_2d():
    nx, ny = 21, 41
    pde = PDE(
        'du/dt = d2u/dx2 + d2u/dy2',
        'u', ['x', 'y'], ['t'],
        ivar_boundary=[(0, 1), (0, 1)],
        expr_ic='sin(pi*x)*sin(pi*y)',
        west_bd='Dirichlet',  west_func_bd='0',
        east_bd='Dirichlet',  east_func_bd='0',
        north_bd='Dirichlet', north_func_bd='0',
        south_bd='Dirichlet', south_func_bd='0',
    )
    sim = PDES([pde], [nx, ny])
    sim.discretize(method='central')
    sim.solve(method='bdf2', tf=TF, nt=NT)
    X, Y = sim.grid.coords()
    ref = (
        np.sin(np.pi * X) * np.sin(np.pi * Y) * np.exp(-2 * np.pi ** 2 * TF)
    ).flatten()
    erro = mae(sim.results[0], ref)
    assert erro < 1e-3, f"Malha {nx}x{ny} — MAE={erro:.2e}"


def test_dominio_retangular_2d():
    pde = PDE(
        'du/dt = d2u/dx2 + d2u/dy2',
        'u', ['x', 'y'], ['t'],
        ivar_boundary=[(0, 2), (0, 3)],
        expr_ic='sin(pi*x/2)*sin(pi*y/3)',
        west_bd='Dirichlet',  west_func_bd='0',
        east_bd='Dirichlet',  east_func_bd='0',
        north_bd='Dirichlet', north_func_bd='0',
        south_bd='Dirichlet', south_func_bd='0',
    )
    sim = PDES([pde], [25, 25])
    sim.discretize(method='central')
    sim.solve(method='bdf2', tf=TF, nt=NT)
    X, Y = sim.grid.coords()
    lam = (np.pi / 2) ** 2 + (np.pi / 3) ** 2
    ref = (
        np.sin(np.pi * X / 2) * np.sin(np.pi * Y / 3) * np.exp(-lam * TF)
    ).flatten()
    erro = mae(sim.results[0], ref)
    assert erro < 1e-3, f"Domínio (0,2)x(0,3) — MAE={erro:.2e}"


def test_passo_reflete_o_dominio():
    sim = PDES([PDE(
        'du/dt = d2u/dx2', 'u', ['x'], ['t'],
        ivar_boundary=[(0, 2)], expr_ic='0',
    )], [5])
    assert np.allclose(np.diff(sim.grid.axes[0].nodes), 0.5)
    m, c, p = sim.grid.axes[0].w2()
    assert np.allclose(c, -2.0 / 0.5 ** 2)


def test_dominio_invertido_rejeitado():
    with pytest.raises(ValueError, match='Domínio inválido'):
        PDES([PDE(
            'du/dt = d2u/dx2', 'u', ['x'], ['t'],
            ivar_boundary=[(1, 0)], expr_ic='0',
        )], [11])

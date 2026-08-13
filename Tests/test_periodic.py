import numpy as np
import pytest
from _helpers import mae

from pdesolver import PDE, PDES

DISC_N  = [64]
C       = 1.0
TF      = 1.0
NT      = 800
TOL_MAE = 2e-2

ALPHA   = 0.05
DISC_2D = [32, 32]
TF_2D   = 0.5
NT_2D   = 200
TOL_2D  = 1e-3


def montar_adveccao():
    pde = PDE(
        f'du/dt = -{C}*du/dx',
        'u', ['x'], ['t'],
        ivar_boundary=[(0, 1)],
        expr_ic='sin(2*pi*x)',
        west_bd='Periodic', east_bd='Periodic',
    )
    sim = PDES([pde], DISC_N)
    sim.discretize(method='central')
    return sim


def montar_calor_2d(backend='symbolic'):
    pde = PDE(
        f'du/dt = {ALPHA}*d2u/dx2 + {ALPHA}*d2u/dy2',
        'u', ['x', 'y'], ['t'],
        ivar_boundary=[(0, 1), (0, 1)],
        expr_ic='sin(2*pi*x)*sin(2*pi*y)',
        west_bd='Periodic',  east_bd='Periodic',
        north_bd='Periodic', south_bd='Periodic',
    )
    sim = PDES([pde], DISC_2D, backend=backend)
    sim.discretize(method='central')
    return sim


def test_malha_periodica_nao_duplica_extremo():
    sim = montar_adveccao()
    eixo = sim.grid.axes[0]
    assert eixo.n == DISC_N[0]
    assert eixo.periodic
    assert eixo.nodes[-1] < 1.0, (
        "Eixo periódico não deve conter o nó final duplicado do domínio."
    )
    assert np.allclose(np.diff(eixo.nodes), 1.0 / DISC_N[0])


@pytest.mark.parametrize('metodo', ['bdf2', 'CN', 'RKF'])
def test_adveccao_periodica_volta_completa(metodo):
    sim = montar_adveccao()
    sim.solve(method=metodo, tf=TF, nt=NT, tol=1e-8)
    x = sim.grid.axes[0].nodes
    ref = np.sin(2 * np.pi * (x - C * TF))
    erro = mae(sim.results[0], ref)
    assert erro < TOL_MAE, f"{metodo} — MAE={erro:.2e} > {TOL_MAE:.0e}"


def test_adveccao_periodica_sem_contorno_dirichlet():
    sim = montar_adveccao()
    assert sim.dirichlet_constraints == {}, (
        "Eixo periódico não deve gerar restrições de Dirichlet."
    )


def test_calor_2d_periodico():
    sim = montar_calor_2d()
    sim.solve(method='CN', tf=TF_2D, nt=NT_2D)
    X, Y = sim.grid.coords()
    ref = (
        np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y)
        * np.exp(-2 * ALPHA * (2 * np.pi) ** 2 * TF_2D)
    ).flatten()
    erro = mae(sim.results[0], ref)
    assert erro < TOL_2D, f"Calor 2D periódico — MAE={erro:.2e} > {TOL_2D:.0e}"


def test_calor_2d_periodico_stencil():
    sim = montar_calor_2d(backend='stencil')
    sim.solve(method='CN', tf=TF_2D, nt=NT_2D)
    X, Y = sim.grid.coords()
    ref = (
        np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y)
        * np.exp(-2 * ALPHA * (2 * np.pi) ** 2 * TF_2D)
    ).flatten()
    erro = mae(sim.results[0], ref)
    assert erro < TOL_2D, f"Stencil periódico — MAE={erro:.2e} > {TOL_2D:.0e}"


def test_par_periodico_incompleto_rejeitado():
    pde = PDE(
        'du/dt = -du/dx',
        'u', ['x'], ['t'],
        ivar_boundary=[(0, 1)],
        expr_ic='sin(2*pi*x)',
        west_bd='Periodic', east_bd='Dirichlet',
    )
    with pytest.raises(ValueError, match='periódico'):
        PDES([pde], DISC_N)

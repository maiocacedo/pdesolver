import numpy as np
import matplotlib
matplotlib.use('Agg')
import pytest
from pdesolver import PDE, PDES

A_XX = 1.0
B_XY = 0.5
C_YY = 1.0
TF   = 0.01
NT   = 200

K = 2 * np.pi
LAMBDA = (A_XX + 2 * B_XY + C_YY) * K ** 2


def mae(numerica, analitica):
    return np.mean(np.abs(np.array(numerica) - analitica))


def montar(n, backend='symbolic'):
    pde = PDE(
        f'du/dt = {A_XX}*d2u/dx2 + {2*B_XY}*d2u/dxdy + {C_YY}*d2u/dy2',
        'u', ['x', 'y'], ['t'],
        ivar_boundary=[(0, 1), (0, 1)],
        expr_ic='sin(2*pi*x + 2*pi*y)',
        west_bd='Periodic',  east_bd='Periodic',
        north_bd='Periodic', south_bd='Periodic',
    )
    sim = PDES([pde], [n, n], backend=backend)
    sim.discretize(method='central')
    return sim


def erro_em(n, backend='symbolic'):
    sim = montar(n, backend=backend)
    sim.solve(method='CN', tf=TF, nt=NT)
    X, Y = sim.grid.coords()
    ref = (np.sin(K * X + K * Y) * np.exp(-LAMBDA * TF)).flatten()
    return mae(sim.results[0], ref)


def test_difusao_anisotropica():
    erro = erro_em(32)
    assert erro < 5e-3, f"Termo misto — MAE={erro:.2e}"


def test_difusao_anisotropica_stencil():
    erro = erro_em(32, backend='stencil')
    assert erro < 5e-3, f"Termo misto (stencil) — MAE={erro:.2e}"


def test_termo_misto_ordem_dois():
    e1 = erro_em(16)
    e2 = erro_em(32)
    ordem = np.log2(e1 / e2)
    assert ordem > 1.8, (
        f"Ordem observada {ordem:.2f} abaixo de 2: "
        f"erro(16)={e1:.2e}, erro(32)={e2:.2e}"
    )


def test_ordem_das_derivadas_cruzadas_equivalente():
    pde_xy = PDE(
        'du/dt = d2u/dxdy', 'u', ['x', 'y'], ['t'],
        ivar_boundary=[(0, 1), (0, 1)], expr_ic='sin(2*pi*x + 2*pi*y)',
        west_bd='Periodic', east_bd='Periodic',
        north_bd='Periodic', south_bd='Periodic',
    )
    pde_yx = PDE(
        'du/dt = d2u/dydx', 'u', ['x', 'y'], ['t'],
        ivar_boundary=[(0, 1), (0, 1)], expr_ic='sin(2*pi*x + 2*pi*y)',
        west_bd='Periodic', east_bd='Periodic',
        north_bd='Periodic', south_bd='Periodic',
    )
    a = PDES([pde_xy], [16, 16])
    a.discretize(method='central')
    b = PDES([pde_yx], [16, 16])
    b.discretize(method='central')
    assert a.disc_results[0] == b.disc_results[0]


def test_termo_misto_exige_duas_dimensoes():
    pde = PDE(
        'du/dt = d2u/dxdy', 'u', ['x'], ['t'],
        ivar_boundary=[(0, 1)], expr_ic='0',
    )
    sim = PDES([pde], [11])
    with pytest.raises(ValueError, match='duas variáveis espaciais'):
        sim.discretize(method='central')

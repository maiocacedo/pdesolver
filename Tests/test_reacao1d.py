import numpy as np
import matplotlib
matplotlib.use('Agg')
import pytest
from pdesolver import PDE, PDES

DISC_N = [15]
TF     = 5.0
NT     = 500
V      = 0.5
DAX    = 0.001
K      = 0.1


def montar_sistema():
    pde_c = PDE(
        f'dC/dt = -{V}*(dC/dx) + {DAX}*(d2C/dx2) - {K}*C',
        'C', ['x'], ['t'],
        ivar_boundary=[(0, 1)],
        expr_ic='0',
        west_bd='Dirichlet', west_func_bd='1',
        east_bd='Neumann',   east_func_bd='0',
    )
    pde_d = PDE(
        f'dD/dt = -{V}*(dD/dx) + {DAX}*(d2D/dx2) + {K}*C',
        'D', ['x'], ['t'],
        ivar_boundary=[(0, 1)],
        expr_ic='0',
        west_bd='Dirichlet', west_func_bd='0',
        east_bd='Neumann',   east_func_bd='0',
    )
    sim = PDES([pde_c, pde_d], DISC_N)
    sim.discretize(method='backward')
    sim.solve(method='bdf2', tf=TF, nt=NT, tol=1e-8)
    return sim


@pytest.fixture(scope='module')
def resultado():
    return montar_sistema()


def test_valores_positivos(resultado):
    _, hist = resultado.results
    C_final = np.array(hist[0][-1])
    D_final = np.array(hist[1][-1])
    assert np.all(C_final >= -1e-10), f"C contém valores negativos: min={C_final.min():.2e}"
    assert np.all(D_final >= -1e-10), f"D contém valores negativos: min={D_final.min():.2e}"


def test_conservacao_massa(resultado):
    _, hist = resultado.results
    dx = 1.0 / (DISC_N[0] - 1)

    massa_inicial = (np.trapezoid(np.array(hist[0][0]), dx=dx) +
                     np.trapezoid(np.array(hist[1][0]), dx=dx))
    massa_final   = (np.trapezoid(np.array(hist[0][-1]), dx=dx) +
                     np.trapezoid(np.array(hist[1][-1]), dx=dx))

    assert massa_final >= 0, "Massa total negativa — instabilidade numérica"
    assert massa_final > massa_inicial, "Massa deveria crescer (entrada Dirichlet C=1)"


def test_d_cresce(resultado):
    _, hist = resultado.results
    dx = 1.0 / (DISC_N[0] - 1)
    massa_d_inicial = np.trapezoid(np.array(hist[1][0]),  dx=dx)
    massa_d_final   = np.trapezoid(np.array(hist[1][-1]), dx=dx)
    assert massa_d_final > massa_d_inicial, "D deveria aumentar ao longo do tempo"


def test_estado_estacionario(resultado):
    _, hist = resultado.results
    C_pen = np.array(hist[0][-2])
    C_fin = np.array(hist[0][-1])
    variacao = np.max(np.abs(C_fin - C_pen))
    assert variacao < 1e-3, f"Sistema não convergiu: variação={variacao:.2e}"


def test_c_limitado(resultado):
    _, hist = resultado.results
    C_final = np.array(hist[0][-1])
    assert np.all(C_final <= 1.0 + 1e-10), f"C > 1: max={C_final.max():.4f}"
import numpy as np
import matplotlib
matplotlib.use('Agg')
import warnings
import pytest
from pdesolver import PDE, PDES

DISC_N = [15, 15]
TF     = 0.2
NT     = 50
DU     = 1.0
EPS    = 0.08
GAMMA  = 0.5
BETA   = 0.7


def montar_sistema():
    pde_u = PDE(
        f'dU/dt = {DU}*d2U/dx2 + {DU}*d2U/dy2 + U - U**3/3 - V',
        'U', ['x', 'y'], ['t'],
        ivar_boundary=[(0, 1), (0, 1)],
        expr_ic='2*Heaviside(0.5 - x)*Heaviside(y - 0.5) - 1',
        west_bd='Neumann', west_func_bd='0',
        east_bd='Neumann', east_func_bd='0',
        north_bd='Neumann', north_func_bd='0',
        south_bd='Neumann', south_func_bd='0',
    )
    pde_v = PDE(
        f'dV/dt = {EPS}*(U - {GAMMA}*V + {BETA})',
        'V', ['x', 'y'], ['t'],
        ivar_boundary=[(0, 1), (0, 1)],
        expr_ic=f'{BETA} - {GAMMA}*y',
        west_bd='Neumann', west_func_bd='0',
        east_bd='Neumann', east_func_bd='0',
        north_bd='Neumann', north_func_bd='0',
        south_bd='Neumann', south_func_bd='0',
    )
    sim = PDES([pde_u, pde_v], DISC_N)
    sim.discretize(method='central')
    return sim


@pytest.fixture(scope='module')
def resultado():
    sim = montar_sistema()
    sim.solve(method='bdf2', tf=TF, nt=NT, tol=1e-6)
    return sim


def test_estabilidade_u(resultado):
    _, hist = resultado.results
    U_final = np.array(hist[0][-1])
    assert np.all(U_final >= -3.0), f"u < -3: min={U_final.min():.2f}"
    assert np.all(U_final <=  3.0), f"u >  3: max={U_final.max():.2f}"


def test_estabilidade_v(resultado):
    _, hist = resultado.results
    V_final = np.array(hist[1][-1])
    assert np.all(V_final >= -2.0), f"v < -2: min={V_final.min():.2f}"
    assert np.all(V_final <=  2.0), f"v >  2: max={V_final.max():.2f}"


def test_ci_assimetrica(resultado):
    _, hist = resultado.results
    U_ini = np.array(hist[0][0])
    assert U_ini.max() > 0.5,  "CI de u deveria ter valores positivos (Heaviside)"
    assert U_ini.min() < -0.5, "CI de u deveria ter valores negativos (Heaviside)"


def test_historico_completo(resultado):
    _, hist = resultado.results
    assert len(hist[0]) == NT + 1, (
        f"Histórico de u tem {len(hist[0])} entradas, esperado {NT + 1}"
    )
    assert len(hist[1]) == NT + 1, (
        f"Histórico de v tem {len(hist[1])} entradas, esperado {NT + 1}"
    )


def test_sem_nan(resultado):
    _, hist = resultado.results
    U_final = np.array(hist[0][-1])
    V_final = np.array(hist[1][-1])
    assert np.all(np.isfinite(U_final)), "u contém NaN ou Inf"
    assert np.all(np.isfinite(V_final)), "v contém NaN ou Inf"
import matplotlib
import numpy as np

matplotlib.use('Agg')
import pytest

from pdesolver import PDE, PDES

EPS      = 0.005
N_LAYER  = 41
BETA     = 5.0
TF_LAYER = 3.0
NT_LAYER = 600

TF   = 0.1
NT   = 200
ALFA = 1.0


def mae(numerica, analitica):
    return np.mean(np.abs(np.array(numerica) - analitica))


def camada_limite(mesh):
    pde = PDE(
        f'du/dt = -du/dx + {EPS}*d2u/dx2',
        'u', ['x'], ['t'],
        ivar_boundary=[(0, 1)],
        expr_ic='0',
        west_bd='Dirichlet', west_func_bd='0',
        east_bd='Dirichlet', east_func_bd='1',
    )
    sim = PDES([pde], [N_LAYER], mesh=mesh)
    sim.discretize(method='central')
    sim.solve(method='bdf2', tf=TF_LAYER, nt=NT_LAYER)
    x = sim.grid.axes[0].nodes
    ref = (np.exp(x / EPS) - 1.0) / (np.exp(1.0 / EPS) - 1.0)
    return mae(sim.results[0], ref)


def calor_1d(mesh, n):
    pde = PDE(
        f'du/dt = {ALFA}*d2u/dx2',
        'u', ['x'], ['t'],
        ivar_boundary=[(0, 1)],
        expr_ic='sin(pi*x)',
        west_bd='Dirichlet', west_func_bd='0',
        east_bd='Dirichlet', east_func_bd='0',
    )
    sim = PDES([pde], [n], mesh=mesh)
    sim.discretize(method='central')
    sim.solve(method='bdf2', tf=TF, nt=NT)
    x = sim.grid.axes[0].nodes
    ref = np.sin(np.pi * x) * np.exp(-ALFA * np.pi ** 2 * TF)
    return mae(sim.results[0], ref)


def test_malha_concentrada_vence_uniforme_na_camada_limite():
    erro_unif = camada_limite('uniform')
    erro_tanh = camada_limite({'type': 'tanh_right', 'beta': BETA})
    assert erro_tanh < erro_unif, (
        f"Malha concentrada deveria resolver melhor a camada limite: "
        f"uniforme={erro_unif:.2e}, tanh_right={erro_tanh:.2e}"
    )


@pytest.mark.parametrize('mesh', [
    'uniform',
    'chebyshev',
    {'type': 'tanh', 'beta': 2.0},
    {'type': 'tanh_left', 'beta': 2.0},
])
def test_ordem_dois_em_malha_estirada(mesh):
    e1 = calor_1d(mesh, 21)
    e2 = calor_1d(mesh, 41)
    ordem = np.log2(e1 / e2)
    assert ordem > 1.7, (
        f"Ordem observada {ordem:.2f} abaixo de 2 para mesh={mesh}: "
        f"erro(21)={e1:.2e}, erro(41)={e2:.2e}"
    )


def test_nos_explicitos():
    nodes = np.linspace(0, 1, 25) ** 2
    pde = PDE(
        'du/dt = d2u/dx2',
        'u', ['x'], ['t'],
        ivar_boundary=[(0, 1)],
        expr_ic='sin(pi*x)',
        west_bd='Dirichlet', west_func_bd='0',
        east_bd='Dirichlet', east_func_bd='0',
    )
    sim = PDES([pde], [25], mesh={'nodes': nodes})
    assert np.allclose(sim.grid.axes[0].nodes, nodes)
    sim.discretize(method='central')
    sim.solve(method='bdf2', tf=TF, nt=NT)
    ref = np.sin(np.pi * nodes) * np.exp(-np.pi ** 2 * TF)
    erro = mae(sim.results[0], ref)
    assert erro < 5e-3, f"Nós explícitos — MAE={erro:.2e}"


def test_malha_uniforme_reduz_a_pesos_classicos():
    sim = PDES([PDE(
        'du/dt = d2u/dx2', 'u', ['x'], ['t'],
        ivar_boundary=[(0, 1)], expr_ic='0',
    )], [5])
    eixo = sim.grid.axes[0]
    h = 0.25
    m, c, p = eixo.w2()
    assert np.allclose(m, 1.0 / h ** 2)
    assert np.allclose(c, -2.0 / h ** 2)
    assert np.allclose(p, 1.0 / h ** 2)
    m, c, p = eixo.w1('central')
    assert np.allclose(m, -1.0 / (2 * h))
    assert np.allclose(c, 0.0)
    assert np.allclose(p, 1.0 / (2 * h))


def test_espacamento_desconhecido_rejeitado():
    with pytest.raises(ValueError, match='Espaçamento inválido'):
        PDES([PDE(
            'du/dt = d2u/dx2', 'u', ['x'], ['t'],
            ivar_boundary=[(0, 1)], expr_ic='0',
        )], [11], mesh='exponencial')

import numpy as np
import pytest

from pdesolver import PDE, PDES

K1, K2, A = 1.0, 0.25, 0.4
U0, UL = 1.0, 0.0
U_INT = (K1 * U0 / A + K2 * UL / (1 - A)) / (K1 / A + K2 / (1 - A))


def pde_calor2d():
    return PDE(
        'dU/dt = 0.2*d2U/dx2 + 0.2*d2U/dy2',
        'U', ['x', 'y'], ['t'],
        ivar_boundary=[(0, 1), (0, 1)],
        expr_ic='1',
        west_bd='Dirichlet',  west_func_bd='0',
        east_bd='Dirichlet',  east_func_bd='0',
        north_bd='Dirichlet', north_func_bd='0',
        south_bd='Dirichlet', south_func_bd='0',
    )


def test_obstaculo_permanece_congelado():
    n = 31
    sim = PDES([pde_calor2d()], [n, n])
    X, Y = sim.grid.coords()
    bloco = (np.abs(X - 0.5) < 0.12) & (np.abs(Y - 0.5) < 0.12)
    sim.discretize(method='central',
                   regions=[{'where': bloco, 'eq': 'dU/dt = 0'}])
    sim.solve(method='bdf2', tf=0.2, nt=100)
    u = np.asarray(sim.results[0]).reshape(n, n)
    assert np.allclose(u[bloco], 1.0), "o obstáculo deveria manter o valor inicial"
    assert u[3, 3] < 0.5, "fora do obstáculo a solução deveria difundir"


def test_mascara_por_callable_equivale_a_array():
    n = 31
    a = PDES([pde_calor2d()], [n, n])
    X, Y = a.grid.coords()
    bloco = (np.abs(X - 0.5) < 0.12) & (np.abs(Y - 0.5) < 0.12)
    a.discretize(method='central',
                 regions=[{'where': bloco, 'eq': 'dU/dt = 0'}])
    a.solve(method='bdf2', tf=0.1, nt=50)

    b = PDES([pde_calor2d()], [n, n])
    b.discretize(method='central', regions=[{
        'where': lambda X, Y: (np.abs(X - 0.5) < 0.12) & (np.abs(Y - 0.5) < 0.12),
        'eq': 'dU/dt = 0',
    }])
    b.solve(method='bdf2', tf=0.1, nt=50)

    assert np.allclose(np.asarray(a.results[0]), np.asarray(b.results[0]))


@pytest.mark.parametrize('n', [41, 81])
def test_interface_de_material(n):
    h = 1.0 / (n - 1)
    i = int(round(A / h))
    pde = PDE(
        f'dU/dt = {K1}*d2U/dx2',
        'U', ['x'], ['t'],
        ivar_boundary=[(0, 1)],
        expr_ic='1-x',
        west_bd='Dirichlet', west_func_bd=str(U0),
        east_bd='Dirichlet', east_func_bd=str(UL),
    )
    sim = PDES([pde], [n])
    x = sim.grid.axes[0].nodes
    sim.discretize(
        method='central',
        regions=[{'where': x > A + h / 2, 'eq': f'dU/dt = {K2}*d2U/dx2'}],
    )
    flat, d_vars = sim.disc_results
    flat[i] = (f'({K2}*XX0_{i+1}_0 - {K1 + K2}*XX0_{i}_0 '
               f'+ {K1}*XX0_{i-1}_0)/{h ** 2}')
    sim.disc_results = (flat, d_vars)
    sim.solve(method='bdf2', tf=5.0, nt=1000)

    u = np.asarray(sim.results[0])
    assert u[i] == pytest.approx(U_INT, abs=1e-9)

    fluxo_esq = K1 * (u[i] - u[i - 1]) / h
    fluxo_dir = K2 * (u[i + 1] - u[i]) / h
    assert fluxo_esq == pytest.approx(fluxo_dir, abs=1e-9), (
        "o fluxo deve ser contínuo através da interface"
    )


def test_fonte_pontual():
    n, k, s_val = 81, 1.0, 10.0
    h = 1.0 / (n - 1)
    i = (n - 1) // 2
    pde = PDE(
        f'dU/dt = {k}*d2U/dx2', 'U', ['x'], ['t'],
        ivar_boundary=[(0, 1)], expr_ic='0',
        west_bd='Dirichlet', west_func_bd='0',
        east_bd='Dirichlet', east_func_bd='0',
    )
    sim = PDES([pde], [n])
    x = sim.grid.axes[0].nodes
    fonte = np.zeros(n, dtype=bool)
    fonte[i] = True
    sim.discretize(
        method='central',
        regions=[{'where': fonte, 'eq': f'dU/dt = {k}*d2U/dx2 + {s_val}'}],
    )
    sim.solve(method='bdf2', tf=2.0, nt=400)
    u = np.asarray(sim.results[0])

    assert u[i] == pytest.approx(s_val * h * x[i] * (1 - x[i]) / k, rel=1e-6)
    salto = (u[i + 1] - u[i]) / h - (u[i] - u[i - 1]) / h
    assert salto == pytest.approx(-s_val * h / k, rel=1e-6)


def test_regiao_nao_altera_nos_fora_dela():
    n = 31
    base = PDES([pde_calor2d()], [n, n])
    base.discretize(method='central')
    base.solve(method='bdf2', tf=0.1, nt=50)

    vazia = PDES([pde_calor2d()], [n, n])
    X, Y = vazia.grid.coords()
    vazia.discretize(method='central', regions=[
        {'where': np.zeros_like(X, dtype=bool), 'eq': 'dU/dt = 0'},
    ])
    vazia.solve(method='bdf2', tf=0.1, nt=50)

    assert np.allclose(
        np.asarray(base.results[0]), np.asarray(vazia.results[0])
    ), "região vazia não deveria alterar nada"


def test_stencil_recusa_regioes():
    n = 21
    sim = PDES([pde_calor2d()], [n, n], backend='stencil')
    X, Y = sim.grid.coords()
    with pytest.raises(ValueError, match="backend='symbolic'"):
        sim.discretize(method='central', regions=[
            {'where': np.abs(X - 0.5) < 0.1, 'eq': 'dU/dt = 0'},
        ])


def test_regiao_exige_chaves():
    sim = PDES([pde_calor2d()], [21, 21])
    with pytest.raises(ValueError, match="'where'"):
        sim.discretize(method='central', regions=[{'eq': 'dU/dt = 0'}])


def test_mascara_com_forma_errada_rejeitada():
    sim = PDES([pde_calor2d()], [21, 21])
    with pytest.raises(ValueError, match='forma'):
        sim.discretize(method='central', regions=[
            {'where': np.zeros((5, 5), dtype=bool), 'eq': 'dU/dt = 0'},
        ])


@pytest.mark.parametrize('metodo,nt', [('bdf2', 250), ('CN', 250),
                                       ('bdf2', 1000), ('CN', 1000)])
def test_dirichlet_nao_homogeneo_e_exato_no_estacionario(metodo, nt):
    pde = PDE(
        'dU/dt = 1.0*d2U/dx2', 'U', ['x'], ['t'],
        ivar_boundary=[(0, 1)], expr_ic='1-x',
        west_bd='Dirichlet', west_func_bd='1',
        east_bd='Dirichlet', east_func_bd='0',
    )
    sim = PDES([pde], [21])
    sim.discretize(method='central')
    sim.solve(method=metodo, tf=5.0, nt=nt)
    x = sim.grid.axes[0].nodes
    erro = np.max(np.abs(np.asarray(sim.results[0]) - (1 - x)))
    assert erro < 1e-12, (
        f"{metodo}: estacionário deveria ser exatamente 1-x, erro={erro:.2e} "
        f"(contaminação O(dt) na fronteira de Dirichlet)"
    )

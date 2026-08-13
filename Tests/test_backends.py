import matplotlib
import numpy as np

matplotlib.use('Agg')
import pytest

from pdesolver import PDE, PDES
from pdesolver.Solvers.solver_base import compile_equations

TOL_RHS = 1e-10
SEED    = 20260801


def calor_2d():
    return [PDE(
        'dF/dt = 0.1*d2F/dx2 + 0.2*d2F/dy2',
        'F', ['x', 'y'], ['t'],
        ivar_boundary=[(0, 1), (0, 1)],
        expr_ic='sin(pi*x)*sin(pi*y)',
        west_bd='Dirichlet',  west_func_bd='0',
        east_bd='Dirichlet',  east_func_bd='0',
        north_bd='Dirichlet', north_func_bd='0',
        south_bd='Dirichlet', south_func_bd='0',
    )]


def burgers_2d():
    return [PDE(
        'dU/dt = -U*dU/dx - U*dU/dy + 0.1*d2U/dx2 + 0.1*d2U/dy2',
        'U', ['x', 'y'], ['t'],
        ivar_boundary=[(0, 1), (0, 1)],
        expr_ic='0',
        west_bd='Dirichlet',  west_func_bd='sin(t) + x*y',
        east_bd='Dirichlet',  east_func_bd='x',
        north_bd='Dirichlet', north_func_bd='y',
        south_bd='Dirichlet', south_func_bd='1',
    )]


def neumann_2d():
    return [PDE(
        'dF/dt = d2F/dx2 + d2F/dy2',
        'F', ['x', 'y'], ['t'],
        ivar_boundary=[(0, 1), (0, 1)],
        expr_ic='0',
        west_bd='Neumann',  west_func_bd='0',
        east_bd='Neumann',  east_func_bd='0',
        north_bd='Neumann', north_func_bd='0',
        south_bd='Neumann', south_func_bd='0',
    )]


def toro_misto():
    return [PDE(
        'du/dt = d2u/dx2 + d2u/dxdy + d2u/dy2',
        'u', ['x', 'y'], ['t'],
        ivar_boundary=[(0, 1), (0, 1)],
        expr_ic='0',
        west_bd='Periodic',  east_bd='Periodic',
        north_bd='Periodic', south_bd='Periodic',
    )]


def acoplado_1d():
    return [
        PDE('dC/dt = -0.5*dC/dx + 0.001*d2C/dx2 - 0.1*C',
            'C', ['x'], ['t'], ivar_boundary=[(0, 1)], expr_ic='0',
            west_bd='Dirichlet', west_func_bd='1',
            east_bd='Neumann',   east_func_bd='0'),
        PDE('dD/dt = -0.5*dD/dx + 0.001*d2D/dx2 + 0.1*C',
            'D', ['x'], ['t'], ivar_boundary=[(0, 1)], expr_ic='0',
            west_bd='Dirichlet', west_func_bd='0',
            east_bd='Neumann',   east_func_bd='0'),
    ]


CASOS = [
    ('calor 2D dirichlet',   calor_2d,    [9, 9],   'uniform', 'central'),
    ('burgers 2D',           burgers_2d,  [9, 9],   'uniform', 'backward'),
    ('calor 2D neumann',     neumann_2d,  [9, 9],   'uniform', 'central'),
    ('toro 2D termo misto',  toro_misto,  [9, 9],   'uniform', 'central'),
    ('acoplado 1D',          acoplado_1d, [15],     'uniform', 'backward'),
    ('nao uniforme 2D',      calor_2d,    [7, 11],  'tanh',    'central'),
    ('chebyshev 2D',         calor_2d,    [9, 9],   'chebyshev', 'central'),
]


@pytest.mark.parametrize(
    'nome,fabrica,disc_n,mesh,metodo',
    CASOS, ids=[c[0] for c in CASOS],
)
def test_backends_concordam_no_rhs(nome, fabrica, disc_n, mesh, metodo):
    sim_sym = PDES(fabrica(), disc_n, mesh=mesh)
    sim_sym.discretize(method=metodo)
    F = compile_equations(*sim_sym.disc_results)

    sim_st = PDES(fabrica(), disc_n, mesh=mesh, backend='stencil')
    sim_st.discretize(method=metodo)

    rng = np.random.default_rng(SEED)
    u = rng.standard_normal(sim_st.operator.size)
    dif = np.max(np.abs(F(0.3, u) - np.asarray(sim_st.operator(0.3, u))))
    assert dif < TOL_RHS, f"{nome} — max|simbólico - stencil| = {dif:.2e}"


@pytest.mark.parametrize('metodo', ['bdf2', 'CN'])
def test_backends_concordam_na_solucao(metodo):
    sim_sym = PDES(calor_2d(), [15, 15])
    sim_sym.discretize(method='central')
    sim_sym.solve(method=metodo, tf=1.0, nt=100)

    sim_st = PDES(calor_2d(), [15, 15], backend='stencil')
    sim_st.discretize(method='central')
    sim_st.solve(method=metodo, tf=1.0, nt=100)

    dif = np.max(np.abs(
        np.asarray(sim_sym.results[0]) - np.asarray(sim_st.results[0])
    ))
    assert dif < 1e-10, f"{metodo} — max|simbólico - stencil| = {dif:.2e}"


@pytest.mark.parametrize('metodo', ['bdf2', 'CN'])
def test_backends_concordam_em_problema_nao_linear(metodo):
    sim_sym = PDES(burgers_2d(), [11, 11])
    sim_sym.discretize(method='backward')
    sim_sym.solve(method=metodo, tf=0.2, nt=40)

    sim_st = PDES(burgers_2d(), [11, 11], backend='stencil')
    sim_st.discretize(method='backward')
    sim_st.solve(method=metodo, tf=0.2, nt=40)

    dif = np.max(np.abs(
        np.asarray(sim_sym.results[0]) - np.asarray(sim_st.results[0])
    ))
    assert dif < 1e-8, (
        f"{metodo} não-linear — max|simbólico - stencil| = {dif:.2e}"
    )


def test_esparsidade_analitica_cobre_o_jacobiano_numerico():
    sim = PDES(burgers_2d(), [9, 9], backend='stencil')
    sim.discretize(method='central')
    op = sim.operator
    padrao, cores, n_cores = op.sparsity()

    rng = np.random.default_rng(SEED)
    u = rng.standard_normal(op.size)
    base = np.asarray(op(0.0, u))
    eps = 1e-6
    faltando = 0
    for j in range(op.size):
        v = u.copy()
        v[j] += eps
        dF = (np.asarray(op(0.0, v)) - base) / eps
        linhas = np.nonzero(np.abs(dF) > 1e-6)[0]
        previstas = set(padrao[:, j].nonzero()[0].tolist())
        faltando += sum(1 for r in linhas if r not in previstas)
    assert faltando == 0, (
        f"{faltando} entradas do Jacobiano numérico fora do padrão analítico."
    )


def test_coloracao_e_valida():
    sim = PDES(toro_misto(), [11, 11], backend='stencil')
    sim.discretize(method='central')
    padrao, cores, n_cores = sim.operator.sparsity()
    conflito = (padrao.T @ padrao).tocoo()
    for i, j in zip(conflito.row, conflito.col):
        if i != j:
            assert cores[i] != cores[j], (
                f"Colunas {i} e {j} conflitam mas têm a mesma cor."
            )
    assert n_cores < sim.operator.size


def test_backend_invalido_rejeitado():
    sim = PDES(calor_2d(), [9, 9], backend='vetorial')
    with pytest.raises(ValueError, match='Backend inválido'):
        sim.discretize(method='central')

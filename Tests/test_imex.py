import math

import matplotlib
import numpy as np

matplotlib.use('Agg')
import pytest
import sympy as sp

from pdesolver import PDE, PDES

NU = 0.02
C = 1.0
TF = 0.2


def sistema(n=31, eq=None, backend='stencil'):
    eq = eq or (f'dU/dt = {NU}*d2U/dx2 + {NU}*d2U/dy2 '
                f'- {C}*dU/dx + U - U**3')
    pde = PDE(
        eq, 'U', ['x', 'y'], ['t'],
        ivar_boundary=[(0, 1), (0, 1)],
        expr_ic='sin(pi*x)*sin(pi*y)',
        west_bd='Dirichlet',  west_func_bd='0',
        east_bd='Dirichlet',  east_func_bd='0',
        north_bd='Dirichlet', north_func_bd='0',
        south_bd='Dirichlet', south_func_bd='0',
    )
    sim = PDES([pde], [n, n], backend=backend)
    sim.discretize(method='central')
    return sim


def test_separacao_preserva_o_operador():
    op = sistema().operator
    rig, bra, _ = op.split_stiff()
    rng = np.random.default_rng(7)
    u = rng.standard_normal(op.size)
    total = np.asarray(op(0.3, u))
    partes = np.asarray(rig(0.3, u)) + np.asarray(bra(0.3, u))
    dif = np.max(np.abs(total - partes))
    assert dif < 1e-10, f"rígido + brando deveria reconstruir o operador: {dif:.2e}"


def test_classificacao_dos_termos():
    op = sistema().operator
    _, _, regs = op.split_stiff()
    por_termo = {str(r['termo']): r for r in regs}

    difusivos = [r for k, r in por_termo.items() if '_xx' in k or '_yy' in k]
    assert difusivos, "esperados termos difusivos"
    assert all(r['rigido'] for r in difusivos), (
        "termos de segunda derivada deveriam ir para o lado implícito"
    )

    advectivos = [r for k, r in por_termo.items()
                  if '_x' in k and '_xx' not in k]
    assert advectivos and not any(r['rigido'] for r in advectivos), (
        "advecção deveria permanecer explícita"
    )

    nao_lineares = [r for r in regs if not r['linear']]
    assert nao_lineares, "esperado ao menos um termo não linear"
    assert not any(r['rigido'] for r in nao_lineares), (
        "termos não lineares nunca vão para o lado implícito"
    )


def test_termo_nao_linear_rigido_fica_explicito():
    sim = sistema(eq=f'dU/dt = U*d2U/dx2 + {NU}*d2U/dy2')
    _, _, regs = sim.operator.split_stiff()
    por_termo = {str(r['termo']): r for r in regs}
    nl = [r for k, r in por_termo.items() if 'U*' in k or '_xx*' in k
          or not r['linear']]
    assert nl, "esperado termo não linear com segunda derivada"
    assert not any(r['rigido'] for r in nl)


def test_ordem_temporal_dois():
    n = 31
    ref_sim = sistema(n)
    ref_sim.solve(method='imex', tf=TF, nt=6400)
    ref = np.asarray(ref_sim.results[0])

    erros = []
    for nt in (100, 200, 400):
        sim = sistema(n)
        sim.solve(method='imex', tf=TF, nt=nt)
        erros.append(
            float(np.sqrt(np.mean((np.asarray(sim.results[0]) - ref) ** 2)))
        )

    ordens = [math.log2(erros[i] / erros[i + 1]) for i in range(len(erros) - 1)]
    assert all(o > 1.8 for o in ordens), (
        f"ordem temporal observada {[round(o, 2) for o in ordens]} abaixo de 2"
    )


def test_difusao_pura_concorda_com_bdf2():
    eq = f'dU/dt = {NU}*d2U/dx2 + {NU}*d2U/dy2'
    a = sistema(31, eq=eq)
    a.solve(method='imex', tf=TF, nt=400)
    b = sistema(31, eq=eq)
    b.solve(method='bdf2', tf=TF, nt=400)
    dif = np.max(np.abs(np.asarray(a.results[0]) - np.asarray(b.results[0])))
    assert dif < 1e-8, (
        f"sem termos brandos o IMEX reduz ao BDF2; diferença = {dif:.2e}"
    )


def test_difusao_pura_leva_tudo_para_implicito():
    sim = sistema(21, eq=f'dU/dt = {NU}*d2U/dx2 + {NU}*d2U/dy2')
    _, bra, regs = sim.operator.split_stiff()
    assert all(r['rigido'] for r in regs)
    assert all(e == sp.Integer(0) for e in bra.exprs), (
        "a parte explícita deveria ser identicamente nula"
    )


def test_acuracia_contra_analitica():
    al = 0.1
    n = 41
    eq = f'dU/dt = {al}*d2U/dx2 + {al}*d2U/dy2'
    sim = sistema(n, eq=eq)
    sim.solve(method='imex', tf=1.0, nt=200)
    X, Y = sim.grid.coords()
    exato = (np.sin(np.pi * X) * np.sin(np.pi * Y)
             * np.exp(-2 * al * np.pi ** 2)).flatten()
    rmse = float(np.sqrt(np.mean((np.asarray(sim.results[0]) - exato) ** 2)))
    assert rmse < 1e-3, f"IMEX — RMSE={rmse:.2e}"


def test_relatorio_de_rigidez():
    from pdesolver.Solvers.imex import stiffness_report
    _, _, info = stiffness_report(sistema().operator)
    assert info['lambda_total'] > info['lambda_explicito'] > 0
    assert info['ganho_de_passo'] > 1.0, (
        "remover a difusão deveria reduzir o |lambda| explícito"
    )


def test_imex_exige_backend_stencil():
    sim = sistema(backend='symbolic')
    with pytest.raises(ValueError, match="backend='stencil'"):
        sim.solve(method='imex', tf=TF, nt=10)


def test_dst_concorda_com_lu():
    import scipy.sparse as sps
    from scipy.sparse.linalg import splu

    from pdesolver.Solvers import fastpoisson
    from pdesolver.Solvers.solver_base import ColoredJacobian

    sim = sistema(41)
    rig, _, _ = sim.operator.split_stiff()
    n = sim.operator.size
    c = 0.7
    jac = ColoredJacobian(*rig.sparsity())
    L, _ = jac.build(rig, np.zeros(n), 0.0)
    lu = splu((sps.eye(n, format='csr') - c * L).tocsc())

    fp = fastpoisson.build(rig, sim, c)
    assert fp is not None, "o solver DST deveria se aplicar a este problema"

    rng = np.random.default_rng(11)
    rhs = rng.standard_normal(n)
    dif = np.max(np.abs(lu.solve(rhs) - fp.solve(rhs)))
    assert dif < 1e-10, f"DST deveria reproduzir a LU esparsa: {dif:.2e}"


def test_dst_recusa_estrutura_inelegivel():
    from pdesolver.Solvers import fastpoisson

    estirada = PDES(
        [PDE(f'dU/dt = {NU}*d2U/dx2 + {NU}*d2U/dy2', 'U', ['x', 'y'], ['t'],
             ivar_boundary=[(0, 1), (0, 1)], expr_ic='sin(pi*x)*sin(pi*y)',
             west_bd='Dirichlet',  west_func_bd='0',
             east_bd='Dirichlet',  east_func_bd='0',
             north_bd='Dirichlet', north_func_bd='0',
             south_bd='Dirichlet', south_func_bd='0')],
        [31, 31], mesh={'type': 'tanh', 'beta': 2.0}, backend='stencil')
    estirada.discretize(method='central')
    rig, _, _ = estirada.operator.split_stiff()
    assert fastpoisson.build(rig, estirada, 0.5) is None, (
        "malha estirada não é diagonalizável pela DST"
    )

    variavel = sistema(31, eq=f'dU/dt = (1+x)*d2U/dx2 + {NU}*d2U/dy2')
    rig, _, _ = variavel.operator.split_stiff()
    assert fastpoisson.build(rig, variavel, 0.5) is None, (
        "coeficiente variável não é diagonalizável pela DST"
    )


def test_imex_com_dst_bate_com_analitica():
    al = 0.1
    sim = sistema(81, eq=f'dU/dt = {al}*d2U/dx2 + {al}*d2U/dy2')
    sim.solve(method='imex', tf=1.0, nt=200)
    X, Y = sim.grid.coords()
    exato = (np.sin(np.pi * X) * np.sin(np.pi * Y)
             * np.exp(-2 * al * np.pi ** 2)).flatten()
    rmse = float(np.sqrt(np.mean((np.asarray(sim.results[0]) - exato) ** 2)))
    assert rmse < 1e-4, f"IMEX+DST — RMSE={rmse:.2e}"

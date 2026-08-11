import math

import matplotlib
import numpy as np

matplotlib.use('Agg')
import pytest
import sympy as sp

from pdesolver import PDE, PDES
from pdesolver.Analysis.stability import (
    max_stable_dt,
    real_axis_limit,
    rkf45_polynomial,
    symbol_eigenvalues,
)
from pdesolver.Analysis.truncation import H, error_order, leading_term

C_ADV = 1.0
NU    = 0.01
N_1D  = 51


def calor_2d(n=64, mesh='uniform'):
    pde = PDE(
        'dF/dt = 0.1*d2F/dx2 + 0.2*d2F/dy2',
        'F', ['x', 'y'], ['t'],
        ivar_boundary=[(0, 1), (0, 1)],
        expr_ic='sin(pi*x)*sin(pi*y)',
        west_bd='Dirichlet',  west_func_bd='0',
        east_bd='Dirichlet',  east_func_bd='0',
        north_bd='Dirichlet', north_func_bd='0',
        south_bd='Dirichlet', south_func_bd='0',
    )
    sim = PDES([pde], [n, n], mesh=mesh, backend='stencil')
    sim.discretize(method='central')
    return sim


def adveccao_1d(metodo, nu=NU, n=N_1D):
    pde = PDE(
        f'du/dt = -{C_ADV}*du/dx + {nu}*d2u/dx2',
        'u', ['x'], ['t'],
        ivar_boundary=[(0, 1)],
        expr_ic='0',
    )
    sim = PDES([pde], [n], backend='stencil')
    sim.discretize(method=metodo)
    return sim


@pytest.mark.parametrize('ordem,metodo,k,coef', [
    (1, 'central',  3, H ** 2 / 6),
    (1, 'backward', 2, -H / 2),
    (1, 'forward',  2, H / 2),
    (2, 'central',  4, H ** 2 / 12),
])
def test_termo_lider_bate_com_livro_texto(ordem, metodo, k, coef):
    k_obt, c_obt = leading_term(ordem, metodo)
    assert k_obt == k
    assert sp.simplify(c_obt - coef) == 0


@pytest.mark.parametrize('ordem,metodo,esperado', [
    (1, 'central', 2), (1, 'backward', 1), (1, 'forward', 1),
    (2, 'central', 2), (2, 'backward', 2), (2, 'forward', 2),
])
def test_ordem_formal(ordem, metodo, esperado):
    assert error_order(ordem, metodo) == esperado


def test_polinomio_de_estabilidade_reproduz_exponencial():
    coeffs = rkf45_polynomial()
    for k in range(6):
        assert coeffs[k] == pytest.approx(1.0 / math.factorial(k), rel=1e-12), (
            f"coeficiente z^{k} deveria ser 1/{k}! para um método de ordem 5"
        )


def test_limite_no_eixo_real():
    z = real_axis_limit(rkf45_polynomial())
    assert -4.0 < z < -3.0, f"limite fora do esperado para RKF45: {z}"
    R = np.polyval(rkf45_polynomial()[::-1], z)
    assert abs(abs(R) - 1.0) < 1e-6


def test_simbolico_e_espectral_concordam():
    sim = calor_2d(n=48)
    est = sim.analyze(method='RKF', verbose=False)['stability']
    razao = est['dt_max'] / est['dt_max_spectral']
    assert 0.9 < razao < 1.1, (
        f"Fourier e espectro discordam: {est['dt_max']:.3e} vs "
        f"{est['dt_max_spectral']:.3e}"
    )


def test_dt_previsto_tem_a_ordem_do_passo_real():
    sim = calor_2d(n=64)
    previsto = sim.stability_limit(method='RKF')

    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        sim.solve(method='RKF', tf=0.05, nt=50, save_every=0, verbose=True)
    linha = [ln for ln in buf.getvalue().splitlines() if 'Passos aceitos' in ln]
    passos = int(linha[0].split('aceitos:')[1].split('|')[0])
    real = 0.05 / passos

    assert 0.4 < real / previsto < 2.5, (
        f"dt previsto {previsto:.3e} destoa do observado {real:.3e}"
    )


def test_metodos_implicitos_sao_incondicionais():
    sim = calor_2d(n=32)
    for metodo in ('bdf2', 'CN'):
        est = sim.analyze(method=metodo, verbose=False)['stability']
        assert est['unconditional']
        assert est['dt_max'] == float('inf')


def test_upwind_adiciona_difusao_numerica():
    sim = adveccao_1d('backward')
    extras = sim.modified_equation()
    difusivos = [e for e in extras if e['derivative_order'] == 2]
    assert difusivos, "esperado um termo difusivo vindo de du/dx"
    h = 1.0 / (N_1D - 1)
    assert difusivos[0]['signed'] == pytest.approx(C_ADV * h / 2, rel=1e-9)
    assert difusivos[0]['signed'] > 0


def test_downwind_e_antidifusivo():
    sim = adveccao_1d('forward')
    extras = sim.modified_equation()
    difusivos = [e for e in extras if e['derivative_order'] == 2]
    assert difusivos[0]['signed'] < 0, (
        "forward em advecção positiva deveria subtrair difusão"
    )


def test_downwind_com_peclet_alto_tem_modo_crescente():
    sim = adveccao_1d('forward', nu=0.002)
    est = sim.analyze(method='RKF', verbose=False)['stability']
    assert est['unstable_mode'], (
        f"esperado modo crescente; Re(lambda) máx = {est['growth_rate']:.3e}"
    )


def test_peclet_de_celula():
    sim = adveccao_1d('central')
    pe = sim.analyze(method='RKF', verbose=False)['peclet']
    assert pe, "esperado um número de Péclet"
    h = 1.0 / (N_1D - 1)
    assert pe[0]['peclet'] == pytest.approx(C_ADV * h / (2 * NU), rel=1e-9)


def test_relatorio_menciona_as_secoes():
    sim = adveccao_1d('backward')
    from pdesolver.Analysis import report_text
    texto = report_text(sim.operator, method='RKF')
    for esperado in ('Discretização', 'Difusão numérica', 'Estabilidade'):
        assert esperado in texto


def test_malha_estirada_preserva_segunda_ordem():
    pde = PDE(
        'du/dt = du/dx', 'u', ['x'], ['t'],
        ivar_boundary=[(0, 1)], expr_ic='0',
    )
    uniforme = PDES([pde], [41], backend='stencil')
    uniforme.discretize(method='central')
    estirada = PDES([pde], [41], mesh={'type': 'tanh', 'beta': 3.0},
                    backend='stencil')
    estirada.discretize(method='central')

    for sim in (uniforme, estirada):
        assert sim.truncation_error()[0]['mesh_k'] == 3, (
            "os pesos interpolatórios exatos anulam o termo em u'' mesmo "
            "em malha não uniforme"
        )

    eixo = estirada.grid.axes[0]
    esperado = float(np.max((eixo.hm * eixo.hp / 6.0)[1:-1]))
    obtido = estirada.truncation_error()[0]['mesh_coeff']
    assert obtido == pytest.approx(esperado, rel=1e-9), (
        "coeficiente líder em malha não uniforme deveria ser hm*hp/6"
    )


def test_analise_funciona_sem_discretizar():
    pde = PDE(
        'du/dt = d2u/dx2', 'u', ['x'], ['t'],
        ivar_boundary=[(0, 1)], expr_ic='0',
    )
    sim = PDES([pde], [21])
    dados = sim.analyze(verbose=False)
    assert dados['stability']['dt_max'] > 0


def test_espectro_do_simbolo_e_estavel_para_difusao():
    sim = calor_2d(n=32)
    lams, linear = symbol_eigenvalues(sim.operator, nk=24)
    assert linear
    assert np.max(lams.real) <= 1e-9, "difusão pura não deve ter modo crescente"
    dt = max_stable_dt(lams, rkf45_polynomial())
    assert dt > 0 and np.isfinite(dt)

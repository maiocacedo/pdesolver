"""Teste de superfície/contrato: exercita cada método público da pdesolver
ao menos uma vez e cobre lacunas (Robin, visualize, FuncAux, __repr__).

Os testes numéricos profundos vivem nos arquivos por-feature; aqui o foco é
contrato de tipo/shape, casos de borda, tratamento de erro e robustez.
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from _helpers import mae

from pdesolver import PDE, PDES


def sistema_calor_1d(n=21, backend="symbolic", mesh="uniform"):
    pde = PDE(
        "du/dt = d2u/dx2", "u", ["x"], ["t"],
        ivar_boundary=[(0, 1)], expr_ic="sin(pi*x)",
        west_bd="Dirichlet", west_func_bd="0",
        east_bd="Dirichlet", east_func_bd="0",
    )
    return PDES([pde], [n], backend=backend, mesh=mesh)


def sistema_calor_2d(n=9, backend="symbolic", mesh="uniform"):
    pde = PDE(
        "dF/dt = 0.1*d2F/dx2 + 0.1*d2F/dy2", "F", ["x", "y"], ["t"],
        ivar_boundary=[(0, 1), (0, 1)], expr_ic="sin(pi*x)*sin(pi*y)",
        west_bd="Dirichlet", west_func_bd="0",
        east_bd="Dirichlet", east_func_bd="0",
        north_bd="Dirichlet", north_func_bd="0",
        south_bd="Dirichlet", south_func_bd="0",
    )
    return PDES([pde], [n, n], backend=backend, mesh=mesh)


# --------------------------------------------------------------------------- #
# Modelo PDE / PDES
# --------------------------------------------------------------------------- #

def test_pde_guarda_atributos():
    pde = PDE(
        "du/dt = d2u/dx2", "u", ["x"], ["t"],
        ivar_boundary=[(0, 1)], expr_ic="sin(pi*x)",
    )
    assert pde.func == "u"
    assert pde.sp_var == ["x"]
    assert pde.ivar == ["t"]
    assert pde.west_bd == "Dirichlet"  # default


def test_pdes_expoe_funcs_e_sp_vars():
    sim = sistema_calor_2d()
    assert sim.funcs == ["F"]
    assert sim.sp_vars == ["x", "y"]
    assert sim.disc_n == [9, 9]


def test_disc_n_setter_recomputa_grid_e_ic():
    sim = sistema_calor_1d(n=11)
    ic_antes = list(sim.ic)
    sim.disc_n = [21]
    assert sim.disc_n == [21]
    assert len(sim.ic) == 21          # ic recomputado no novo tamanho
    assert len(sim.ic) != len(ic_antes)
    assert sim.grid.axes[0].n == 21   # grid recomputado


def test_repr_reflete_estado():
    sim = sistema_calor_1d()
    assert "not solved" in repr(sim)
    sim.discretize(method="central")
    sim.solve(method="bdf2", tf=0.05, nt=5)
    assert "solved" in repr(sim)
    assert "not solved" not in repr(sim)


# --------------------------------------------------------------------------- #
# discretize
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("metodo", ["central", "forward", "backward"])
def test_discretize_metodos_simbolico(metodo):
    sim = sistema_calor_1d()
    sim.discretize(method=metodo)
    assert sim.disc_results is not None
    flat, dvars = sim.disc_results
    assert len(flat) == sim.disc_n[0]


def test_discretize_stencil_constroi_operador():
    sim = sistema_calor_2d(backend="stencil")
    sim.discretize(method="central")
    assert sim.operator is not None
    assert sim.operator.size == 9 * 9


def test_backend_invalido_levanta():
    sim = sistema_calor_2d(backend="vetorial")
    with pytest.raises(ValueError, match="Backend inválido"):
        sim.discretize(method="central")


def test_solve_sem_discretizar_levanta():
    sim = sistema_calor_1d()
    with pytest.raises(RuntimeError, match="discretize"):
        sim.solve(method="bdf2", tf=0.05, nt=5)


# --------------------------------------------------------------------------- #
# solve — contrato de results e todos os métodos
# --------------------------------------------------------------------------- #

def _assert_estrutura_results(results):
    """Estrutura comum a todos os métodos: `(finais, historico)`."""
    assert isinstance(results, tuple) and len(results) == 2
    finais, historico = results
    assert isinstance(historico, list) and len(historico) >= 1
    ultimo = np.asarray(historico[0][-1])
    assert np.all(np.isfinite(ultimo))          # sem NaN/Inf
    assert np.asarray(finais).size == ultimo.size
    return historico


@pytest.mark.parametrize("metodo", ["bdf2", "CN"])
def test_solve_passo_fixo_historico_completo(metodo):
    # Métodos de passo fixo guardam exatamente nt+1 snapshots.
    nt = 10
    sim = sistema_calor_2d()
    sim.discretize(method="central")
    r = sim.solve(method=metodo, tf=0.05, nt=nt, tol=1e-6)
    hist = _assert_estrutura_results(r)
    assert len(hist[0]) == nt + 1


def test_solve_rkf_estrutura():
    # RKF é adaptativo: o nº de saídas depende dos passos aceitos, não de nt.
    sim = sistema_calor_2d()
    sim.discretize(method="central")
    r = sim.solve(method="RKF", tf=0.05, nt=10, tol=1e-6)
    hist = _assert_estrutura_results(r)
    assert len(hist[0]) >= 2


def test_solve_imex_stencil():
    nt = 10
    sim = sistema_calor_2d(backend="stencil")
    sim.discretize(method="central")
    r = sim.solve(method="imex", tf=0.05, nt=nt)
    hist = _assert_estrutura_results(r)
    assert len(hist[0]) == nt + 1     # imex também é de passo fixo


def test_solve_metodo_desconhecido_levanta():
    sim = sistema_calor_1d()
    sim.discretize(method="central")
    with pytest.raises(ValueError, match="[Uu]nknown method"):
        sim.solve(method="euler", tf=0.05, nt=5)


# --------------------------------------------------------------------------- #
# Análise e persistência (smoke)
# --------------------------------------------------------------------------- #

def test_analyze_retorna_chaves_esperadas():
    sim = sistema_calor_2d(backend="stencil")
    sim.discretize(method="central")
    dados = sim.analyze(method="RKF", verbose=False)
    for chave in ("terms", "modified", "numerical_diffusion",
                  "stability", "peclet"):
        assert chave in dados
    assert dados["stability"]["dt_max"] > 0


def test_stability_truncation_modified_tipos():
    sim = sistema_calor_1d(backend="stencil")
    sim.discretize(method="central")
    assert isinstance(sim.stability_limit(method="RKF"), float)
    assert isinstance(sim.truncation_error(), list)
    assert isinstance(sim.modified_equation(), list)


def test_save_load_roundtrip_smoke(tmp_path):
    sim = sistema_calor_2d()
    sim.discretize(method="central")
    sim.solve(method="bdf2", tf=0.05, nt=5)
    caminho = tmp_path / "api.json"
    sim.save_to_json(str(caminho))
    assert caminho.exists()
    carregado = PDES.load_from_json(str(caminho))
    assert carregado.funcs == sim.funcs
    assert carregado.disc_n == sim.disc_n
    assert carregado.results is not None


# --------------------------------------------------------------------------- #
# Condições de contorno
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("bc,func", [
    ("Neumann", "0"),
    ("Dirichlet", "0"),
])
def test_bc_1d_roda(bc, func):
    pde = PDE(
        "du/dt = d2u/dx2", "u", ["x"], ["t"],
        ivar_boundary=[(0, 1)], expr_ic="sin(pi*x)",
        west_bd=bc, west_func_bd=func,
        east_bd=bc, east_func_bd=func,
    )
    sim = PDES([pde], [15])
    sim.discretize(method="central")
    r = sim.solve(method="bdf2", tf=0.05, nt=10)
    assert np.all(np.isfinite(np.asarray(r[0])))


def test_bc_periodica_roda():
    pde = PDE(
        "du/dt = -du/dx", "u", ["x"], ["t"],
        ivar_boundary=[(0, 1)], expr_ic="sin(2*pi*x)",
        west_bd="Periodic", east_bd="Periodic",
    )
    sim = PDES([pde], [32])
    sim.discretize(method="central")
    r = sim.solve(method="bdf2", tf=0.05, nt=20)
    assert np.all(np.isfinite(np.asarray(r[0])))


def test_robin_estado_estacionario():
    # oeste Dirichlet u=1; leste Robin  u + u' = 0  ->  u(x)=1-x/2, u(1)=0.5
    pde = PDE(
        "du/dt = d2u/dx2", "u", ["x"], ["t"],
        ivar_boundary=[(0, 1)], expr_ic="1 - x/2",
        west_bd="Dirichlet", west_func_bd="1",
        east_bd="Robin", east_func_bd="1; 1; 0",
    )
    sim = PDES([pde], [21])
    sim.discretize(method="central")
    sim.solve(method="bdf2", tf=5.0, nt=500)
    x = sim.grid.axes[0].nodes
    u = np.asarray(sim.results[0])
    assert u[-1] == pytest.approx(0.5, abs=5e-2)
    assert mae(u, 1 - x / 2) < 5e-2


def test_bc_desconhecida_levanta():
    from pdesolver.Disc.boundaries import get_boundary
    with pytest.raises(ValueError, match="[Cc]ontorno desconhecido"):
        get_boundary("Mystery", "0")


# --------------------------------------------------------------------------- #
# visualize — smoke de todos os modos sob Agg
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def resolvido_1d():
    pde = PDE(
        "du/dt = d2u/dx2", "u", ["x"], ["t"],
        ivar_boundary=[(0, 1)], expr_ic="sin(pi*x)",
        west_bd="Dirichlet", west_func_bd="0",
        east_bd="Dirichlet", east_func_bd="0",
    )
    sim = PDES([pde], [21])
    sim.discretize(method="central")
    sim.solve(method="bdf2", tf=0.05, nt=10)
    return sim


@pytest.fixture(scope="module")
def resolvido_2d():
    sim = sistema_calor_2d(n=9)
    sim.discretize(method="central")
    sim.solve(method="bdf2", tf=0.05, nt=10)
    return sim


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("modo", ["plot1d", "plot1d_all", "heatmap1d",
                                  "animation1d"])
def test_visualize_1d_nao_quebra(resolvido_1d, modo):
    resolvido_1d.visualize(mode=modo)
    assert plt.get_fignums()          # criou ao menos uma figura
    plt.close("all")


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("modo", ["heatmap", "animation", "plot3d",
                                  "animation3d"])
def test_visualize_2d_nao_quebra(resolvido_2d, modo):
    resolvido_2d.visualize(mode=modo)
    assert plt.get_fignums()
    plt.close("all")


def test_visualize_sem_solve_nao_quebra(capsys):
    sim = sistema_calor_1d()
    sim.discretize(method="central")
    sim.visualize(mode="plot1d")      # results é None -> caminho de guarda
    assert "solve" in capsys.readouterr().out.lower()


def test_visualize_modo_desconhecido_avisa(resolvido_2d, capsys):
    resolvido_2d.visualize(mode="hologram")
    assert "desconhecido" in capsys.readouterr().out.lower()
    plt.close("all")


# --------------------------------------------------------------------------- #
# FuncAux — utilidades puras
# --------------------------------------------------------------------------- #

def test_funcaux_build_func_map():
    import numpy as _np

    from pdesolver.Auxs.FuncAux import build_func_map
    m = build_func_map(_np)
    for chave in ("sin", "cos", "exp", "sqrt", "tanh", "sech"):
        assert chave in m
    assert m["sin"] is _np.sin
    assert m["sech"](0.0) == pytest.approx(1.0)   # 1/cosh(0)


def test_funcaux_d_dt():
    import sympy as sp

    from pdesolver.Auxs.FuncAux import d_dt
    assert d_dt("x*t") == "x"
    # derivada de sin(t) é cos(t) (comparação robusta via sympy)
    assert sp.simplify(sp.sympify(d_dt("sin(t)")) - sp.sympify("cos(t)")) == 0


def test_funcaux_repl_symbol_respeita_fronteira():
    from pdesolver.Auxs.FuncAux import repl_symbol
    assert repl_symbol("x + xx + x2", "x", "Y") == "Y + xx + x2"

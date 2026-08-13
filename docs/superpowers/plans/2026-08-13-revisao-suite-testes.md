# Revisão da Suíte de Testes + Teste Abrangente de API — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Limpar as inconsistências da suíte de testes (colisões `calor_2d`, `mae` duplicado, boilerplate Agg, imports mortos) e adicionar `Tests/test_api.py` cobrindo toda a superfície pública da `pdesolver`.

**Architecture:** Centralizar infraestrutura de teste em `Tests/conftest.py` (backend headless + `sys.path`) e `Tests/_helpers.py` (métricas puras). Refatorar os arquivos existentes para consumir essa infra sem alterar nenhuma asserção numérica. Adicionar um arquivo de smoke/contrato que exercita cada método público ao menos uma vez.

**Tech Stack:** Python 3.14, pytest 9, numpy, sympy, scipy, matplotlib (backend Agg nos testes).

## Global Constraints

- **Somente testes.** Nenhuma alteração em `pdesolver/`. Se um teste expõe bug de produção, marcar `xfail(strict=True)` com motivo e reportar — não corrigir a lib.
- **Zero regressão.** Todo teste hoje verde continua verde; nenhuma tolerância numérica é enfraquecida.
- **Lint limpo.** `ruff check Tests` sem apontamentos (regras E, F, W, I do `pyproject.toml`), em especial F401.
- **Commits:** mensagens em pt-BR, prefixo convencional, **sem `Co-Authored-By`**. Após a sequência de commits, `git push` para `origin`.
- **Contrato `results`:** `sim.results` é uma tupla `(finais, historico)`; `finais` é o campo final achatado; `historico[func]` tem `nt + 1` snapshots; `historico[func][-1]` é o estado final da função `func`.
- **Rodar pytest:** usar `.venv/Scripts/python.exe -m pytest`.

---

### Task 0: Baseline verde

**Files:** nenhum (verificação).

- [ ] **Step 1: Rodar a suíte inteira e registrar o resultado**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: todos passam (a suíte é lenta, ~vários minutos). Anotar o número de testes verdes como linha de base. Se algo já falha em `origin/main`, registrar e tratar como pré-existente (não escopo desta branch).

---

### Task 1: Infraestrutura compartilhada (`conftest.py` + `_helpers.py`)

**Files:**
- Create: `Tests/conftest.py`
- Create: `Tests/_helpers.py`

**Interfaces:**
- Produces: `Tests/_helpers.py` expõe `mae(numerica, analitica) -> float` e `rmse(numerica, analitica) -> float`, ambos aceitando array-like/escalares e achatando via `np.asarray(...).ravel()`. `Tests/conftest.py` insere seu próprio diretório em `sys.path` (habilita `import _helpers`) e garante backend `Agg` em todos os testes.

- [ ] **Step 1: Escrever `Tests/_helpers.py`**

```python
"""Métricas numéricas compartilhadas pelos testes."""

import numpy as np


def mae(numerica, analitica):
    """Erro absoluto médio entre dois campos, achatados."""
    a = np.asarray(numerica, dtype=float).ravel()
    b = np.asarray(analitica, dtype=float).ravel()
    return float(np.mean(np.abs(a - b)))


def rmse(numerica, analitica):
    """Raiz do erro quadrático médio entre dois campos, achatados."""
    a = np.asarray(numerica, dtype=float).ravel()
    b = np.asarray(analitica, dtype=float).ravel()
    return float(np.sqrt(np.mean((a - b) ** 2)))
```

- [ ] **Step 2: Escrever `Tests/conftest.py`**

```python
"""Configuração compartilhada da suíte.

Garante backend matplotlib headless (Agg) e torna `Tests/` importável para
os módulos auxiliares (`import _helpers`), independente do import-mode.
"""

import os
import sys

# Torna helpers do diretório de testes importáveis por nome simples.
sys.path.insert(0, os.path.dirname(__file__))

# Backend headless antes de qualquer import de matplotlib nos testes.
import matplotlib  # noqa: E402

matplotlib.use("Agg", force=True)

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _force_agg():
    """Reforça Agg por teste.

    Importar `pdesolver.Auxs.Visualize` roda `_select_backend()`, que pode
    trocar o backend para TkAgg/QtAgg em máquinas com essas libs. Reforçar
    aqui mantém os testes headless e determinísticos.
    """
    matplotlib.use("Agg", force=True)
    yield
```

- [ ] **Step 3: Verificar que a infra é aditiva e não quebra nada**

Run: `.venv/Scripts/python.exe -m pytest -q Tests/test_calor.py`
Expected: PASS (mesmos testes de antes; a infra ainda não é consumida).

- [ ] **Step 4: Verificar que `import _helpers` funciona sob a config do projeto**

Run: `.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider --collect-only Tests/test_calor.py`
Expected: coleta sem erro (confirma que `conftest.py` carrega e ajusta `sys.path`).

- [ ] **Step 5: Commit**

```bash
git add Tests/conftest.py Tests/_helpers.py
git commit -m "test(infra): conftest com backend Agg e helpers de métrica compartilhados"
```

---

### Task 2: Centralizar Agg e de-duplicar `mae` nos arquivos existentes

Remove o boilerplate `matplotlib.use('Agg')` (agora em `conftest.py`), troca cada `def mae` local por `from _helpers import mae`, e remove `import pytest`/`import matplotlib` não usados.

**Files (modify):**
- `Tests/test_calor.py`, `Tests/test_adveccao.py`, `Tests/test_burgers.py`, `Tests/test_dominio.py`, `Tests/test_mixed.py`, `Tests/test_nonuniform.py`, `Tests/test_periodic.py` (têm `def mae`)
- `Tests/test_fitzhugh.py`, `Tests/test_reacao1d.py`, `Tests/test_load.py`, `Tests/test_imex.py` (só boilerplate/imports)

**Interfaces:**
- Consumes: `mae` de `Tests/_helpers.py` (Task 1).

- [ ] **Step 1: Nos 7 arquivos com `def mae`, remover a definição local e importar do helper**

Em cada um: apagar o bloco
```python
import matplotlib
matplotlib.use('Agg')
```
e a função
```python
def mae(numerica, analitica):
    return np.mean(np.abs(np.array(numerica) - analitica.flatten()))
```
e adicionar, junto aos imports do topo:
```python
from _helpers import mae
```
Remover `import pytest` **apenas** onde não é usado (calor, adveccao, burgers). Manter `import numpy as np` (ainda usado). Manter `import pytest` onde há `@pytest.mark`/`pytest.raises`/`pytest.approx` (dominio, mixed, nonuniform, periodic).

- [ ] **Step 2: Nos 4 arquivos só com boilerplate, remover o bloco Agg**

Em `test_fitzhugh.py`, `test_reacao1d.py`, `test_load.py`, `test_imex.py`: apagar
```python
import matplotlib
matplotlib.use('Agg')
```
(mantendo os demais imports que são usados).

- [ ] **Step 3: Verificar ausência de duplicação por grep**

Run: `grep -rn "def mae\|matplotlib.use" Tests/ --include=test_*.py`
Expected: nenhuma linha (todo `mae`/`use` saiu dos test_*.py; ficam só em `_helpers.py`/`conftest.py`).

- [ ] **Step 4: Rodar os arquivos afetados**

Run: `.venv/Scripts/python.exe -m pytest -q Tests/test_calor.py Tests/test_adveccao.py Tests/test_burgers.py Tests/test_dominio.py Tests/test_mixed.py Tests/test_nonuniform.py Tests/test_periodic.py Tests/test_fitzhugh.py Tests/test_reacao1d.py Tests/test_load.py Tests/test_imex.py`
Expected: PASS (mesmos testes, mesmas tolerâncias).

- [ ] **Step 5: Lint**

Run: `.venv/Scripts/python.exe -m ruff check Tests/`
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add Tests/
git commit -m "test: centraliza backend Agg no conftest e remove mae/import duplicados"
```

---

### Task 3: Desfazer as colisões `calor_2d`

**Files (modify):**
- `Tests/test_analysis.py` — `calor_2d(n=64, mesh='uniform')` → `sim_calor2d(n=64, mesh='uniform')`
- `Tests/test_backends.py` — `calor_2d()` → `pdes_calor2d()`
- `Tests/test_regions.py` — `calor_2d(n=31)` → `pde_calor2d()` (remover o parâmetro `n`, que é morto)

- [ ] **Step 1: `test_analysis.py` — renomear def e todas as chamadas**

Renomear `def calor_2d(` → `def sim_calor2d(`. Atualizar todas as chamadas `calor_2d(` → `sim_calor2d(` (aparecem em `test_simbolico_e_espectral_concordam`, `test_dt_previsto_tem_a_ordem_do_passo_real`, `test_metodos_implicitos_sao_incondicionais`, `test_espectro_do_simbolo_e_estavel_para_difusao`). Também remover o boilerplate `matplotlib.use('Agg')`.

- [ ] **Step 2: `test_backends.py` — renomear def e todas as chamadas**

Renomear `def calor_2d(` → `def pdes_calor2d(`. Atualizar a entrada em `CASOS` (`calor_2d` → `pdes_calor2d`, duas ocorrências: 'calor 2D dirichlet' e 'nao uniforme 2D' e 'chebyshev 2D') e as chamadas em `test_backends_concordam_na_solucao` e `test_backend_invalido_rejeitado`. Remover boilerplate Agg.

- [ ] **Step 3: `test_regions.py` — renomear def, remover `n` morto, atualizar chamadas**

Trocar
```python
def calor_2d(n=31):
    return PDE(
```
por
```python
def pde_calor2d():
    return PDE(
```
Atualizar todas as chamadas `calor_2d()` → `pde_calor2d()` (em `test_obstaculo_permanece_congelado`, `test_mascara_por_callable_equivale_a_array`, `test_regiao_nao_altera_nos_fora_dela`, `test_stencil_recusa_regioes`, `test_regiao_exige_chaves`, `test_mascara_com_forma_errada_rejeitada`). Remover boilerplate Agg.

- [ ] **Step 4: Verificar que nenhum `calor_2d` sobrou**

Run: `grep -rn "calor_2d" Tests/`
Expected: só `sim_calor2d`, `pdes_calor2d`, `pde_calor2d` (nenhum `def calor_2d` ou `calor_2d(` cru).

- [ ] **Step 5: Rodar os três arquivos + lint**

Run: `.venv/Scripts/python.exe -m pytest -q Tests/test_backends.py Tests/test_regions.py && .venv/Scripts/python.exe -m pytest -q Tests/test_analysis.py`
Expected: PASS.
Run: `.venv/Scripts/python.exe -m ruff check Tests/`
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add Tests/
git commit -m "test: desfaz colisão de nome calor_2d (sim_/pdes_/pde_) e remove parâmetro morto"
```

---

### Task 4: `test_api.py` — modelo `PDE`, `PDES`, `discretize`, `__repr__`

**Files:**
- Create: `Tests/test_api.py`

**Interfaces:**
- Consumes: `pdesolver.PDE`, `pdesolver.PDES`; `mae`/`rmse` de `_helpers`.
- Produces: fábrica local `sistema_calor_1d(n=21, **kw)` e `sistema_calor_2d(n=9, **kw)` retornando `PDES` já construído (sem discretizar), reusadas pelas tasks 5–7 deste arquivo.

- [ ] **Step 1: Escrever o cabeçalho e as fábricas + primeiros testes**

```python
"""Teste de superfície/contrato: exercita cada método público da pdesolver
ao menos uma vez e cobre lacunas (Robin, visualize, FuncAux, __repr__).

Os testes numéricos profundos vivem nos arquivos por-feature; aqui o foco é
contrato de tipo/shape, casos de borda, tratamento de erro e robustez.
"""

import numpy as np
import pytest

from pdesolver import PDE, PDES
from _helpers import mae


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
```

- [ ] **Step 2: Rodar e ver passar**

Run: `.venv/Scripts/python.exe -m pytest -q Tests/test_api.py`
Expected: 4 passed.

- [ ] **Step 3: Adicionar testes de `discretize` (métodos, backends, erro)**

```python
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
```

- [ ] **Step 4: Rodar e ver passar**

Run: `.venv/Scripts/python.exe -m pytest -q Tests/test_api.py`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add Tests/test_api.py
git commit -m "test(api): PDE/PDES, disc_n setter, __repr__, discretize e erros"
```

---

### Task 5: `test_api.py` — `solve` (todos os métodos), contrato de `results`, Analysis e persistência (smoke)

**Files (modify):** `Tests/test_api.py`

**Interfaces:**
- Consumes: `sistema_calor_1d`, `sistema_calor_2d` (Task 4).

- [ ] **Step 1: Contrato de `results` + `solve` em todos os métodos**

```python
def _assert_contrato_results(results, nt):
    assert isinstance(results, tuple) and len(results) == 2
    finais, historico = results
    assert isinstance(historico, list) and len(historico) >= 1
    assert len(historico[0]) == nt + 1          # nt+1 snapshots
    ultimo = np.asarray(historico[0][-1])
    assert np.all(np.isfinite(ultimo))          # sem NaN/Inf
    assert np.asarray(finais).size == ultimo.size


@pytest.mark.parametrize("metodo", ["bdf2", "CN", "RKF"])
def test_solve_simbolico_respeita_contrato(metodo):
    nt = 10
    sim = sistema_calor_2d()
    sim.discretize(method="central")
    r = sim.solve(method=metodo, tf=0.05, nt=nt, tol=1e-6)
    _assert_contrato_results(r, nt)


def test_solve_imex_stencil():
    nt = 10
    sim = sistema_calor_2d(backend="stencil")
    sim.discretize(method="central")
    r = sim.solve(method="imex", tf=0.05, nt=nt)
    _assert_contrato_results(r, nt)


def test_solve_metodo_desconhecido_levanta():
    sim = sistema_calor_1d()
    sim.discretize(method="central")
    with pytest.raises(ValueError, match="[Uu]nknown method"):
        sim.solve(method="euler", tf=0.05, nt=5)
```

- [ ] **Step 2: Rodar**

Run: `.venv/Scripts/python.exe -m pytest -q Tests/test_api.py -k "solve or contrato"`
Expected: PASS (5 novos).

- [ ] **Step 3: Analysis (smoke de chaves) + persistência (round-trip leve)**

```python
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
```

- [ ] **Step 4: Rodar**

Run: `.venv/Scripts/python.exe -m pytest -q Tests/test_api.py`
Expected: todos passam.

- [ ] **Step 5: Commit**

```bash
git add Tests/test_api.py
git commit -m "test(api): contrato de results, solve em todos os métodos, analysis e persistência"
```

---

### Task 6: `test_api.py` — condições de contorno (Dirichlet/Neumann/Periodic + Robin real)

**Files (modify):** `Tests/test_api.py`

**Nota de implementação (Robin):** o estado estacionário de `du/dt=d2u/dx2` com oeste Dirichlet `u(0)=1` e leste Robin `1;1;0` (isto é, `u + u' = 0`) é a reta `u(x)=1-x/2`, logo `u(1)=0.5`. **Rodar o teste de fato:** se o valor bater (tol folgada, ex. `abs<5e-2`), manter como asserção. Se destoar por bug de produção em Robin, converter em `@pytest.mark.xfail(strict=True, reason="Robin ... <descrição do desvio observado>")` e reportar o achado ao usuário — não alterar a lib.

- [ ] **Step 1: BCs básicas em smoke + Robin real + BC desconhecida**

```python
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
```

- [ ] **Step 2: Rodar (e adaptar Robin conforme a nota, se preciso)**

Run: `.venv/Scripts/python.exe -m pytest -q Tests/test_api.py -k "bc or robin"`
Expected: PASS. Se `test_robin_estado_estacionario` falhar por desvio numérico de produção, aplicar `xfail(strict=True)` com o motivo observado e registrar para reporte.

- [ ] **Step 3: Commit**

```bash
git add Tests/test_api.py
git commit -m "test(api): contornos Dirichlet/Neumann/Periodic e Robin real"
```

---

### Task 7: `test_api.py` — `visualize()` (smoke de todos os modos + guardas)

**Files (modify):** `Tests/test_api.py`

- [ ] **Step 1: Fixtures de sistemas resolvidos 1D/2D e smoke de todos os modos**

```python
import matplotlib.pyplot as plt


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
```

- [ ] **Step 2: Rodar**

Run: `.venv/Scripts/python.exe -m pytest -q Tests/test_api.py -k visualize`
Expected: PASS (10 combinações + 2 guardas). Se um modo levantar exceção real (bug de produção), aplicar `xfail(strict=True)` com o motivo e reportar.

- [ ] **Step 3: Commit**

```bash
git add Tests/test_api.py
git commit -m "test(api): smoke de visualize em todos os modos 1D/2D e guardas"
```

---

### Task 8: `test_api.py` — utilidades puras de `FuncAux`

**Files (modify):** `Tests/test_api.py`

- [ ] **Step 1: Unidades de `build_func_map`, `d_dt`, `repl_symbol`**

```python
def test_funcaux_build_func_map():
    import numpy as _np
    from pdesolver.Auxs.FuncAux import build_func_map
    m = build_func_map(_np)
    for chave in ("sin", "cos", "exp", "sqrt", "tanh", "sech"):
        assert chave in m
    assert m["sin"] is _np.sin
    assert m["sech"](0.0) == pytest.approx(1.0)   # 1/cosh(0)


def test_funcaux_d_dt():
    from pdesolver.Auxs.FuncAux import d_dt
    assert d_dt("x*t") == "x"
    # derivada de sin(t) é cos(t) (comparação robusta via sympy)
    import sympy as sp
    assert sp.simplify(sp.sympify(d_dt("sin(t)")) - sp.sympify("cos(t)")) == 0


def test_funcaux_repl_symbol_respeita_fronteira():
    from pdesolver.Auxs.FuncAux import repl_symbol
    assert repl_symbol("x + xx + x2", "x", "Y") == "Y + xx + x2"
```

- [ ] **Step 2: Rodar o arquivo inteiro + lint**

Run: `.venv/Scripts/python.exe -m pytest -q Tests/test_api.py`
Expected: todos passam.
Run: `.venv/Scripts/python.exe -m ruff check Tests/`
Expected: `All checks passed!`

- [ ] **Step 3: Commit**

```bash
git add Tests/test_api.py
git commit -m "test(api): unidades puras de FuncAux (func_map, d_dt, repl_symbol)"
```

---

### Task 9: Verificação final + PR description

**Files:**
- Create: `docs/PR_revisao-suite-testes.md`

- [ ] **Step 1: Suíte inteira verde**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: baseline da Task 0 + os novos testes de `test_api.py`, todos verdes; nenhum arquivo previamente verde regrediu.

- [ ] **Step 2: Lint final**

Run: `.venv/Scripts/python.exe -m ruff check Tests/`
Expected: `All checks passed!`

- [ ] **Step 3: Escrever a PR description em `docs/PR_revisao-suite-testes.md`**

Conteúdo: resumo (o quê e por quê), lista de mudanças (Parte A/B/C), o achado do stub `RobinBC` (código morto) e, se houver, qualquer `xfail` aplicado com motivo; seção "Como testar" (`pytest -q`, `ruff check Tests/`); nota de que não há mudança de produção.

- [ ] **Step 4: Commit + push**

```bash
git add docs/PR_revisao-suite-testes.md
git commit -m "docs: descrição da PR da revisão da suíte de testes"
git push -u origin tests/revisao-suite
```

---

## Self-Review

**Spec coverage:**
- Parte A (conftest + helpers) → Task 1. ✓
- Parte B (calor_2d, mae, imports, boilerplate) → Tasks 2–3. ✓
- Parte C (test_api: PDE/PDES/repr, discretize/solve+contrato, analysis, persistência, Robin, visualize, FuncAux) → Tasks 4–8. ✓
- Dimensões de qualidade (contrato, borda, erro, regressão, headless) → distribuídas em 4–8. ✓
- Verificação (baseline, sem regressão, ruff) → Tasks 0 e 9. ✓
- Não-objetivos (sem mudar produção; Robin xfail; RobinBC fora de escopo) → Global Constraints + notas das Tasks 6/9. ✓

**Placeholder scan:** sem TBD/TODO; todo passo de código tem código concreto. ✓

**Type consistency:** `sistema_calor_1d`/`sistema_calor_2d` definidas na Task 4 e consumidas em 5–7; `_assert_contrato_results` definido e usado na Task 5; nomes renomeados (`sim_calor2d`/`pdes_calor2d`/`pde_calor2d`) consistentes na Task 3; `mae` importado de `_helpers` em todas as tasks. ✓

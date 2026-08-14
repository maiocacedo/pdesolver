# Revisão da suíte de testes + teste abrangente de API

**Data:** 2026-08-13
**Branch:** `tests/revisao-suite` (a partir de `origin/main`)
**Autor:** Caio Macedo (com Claude)

## Contexto

A suíte de testes atual (15 arquivos em `Tests/`, presente em `origin/main`)
cresceu de forma orgânica e acumulou inconsistências que dificultam manutenção:

1. **Colisão de nome `calor_2d`.** O mesmo nome é definido em três arquivos com
   três contratos incompatíveis:
   - `test_analysis.py` → `calor_2d(n=64, mesh='uniform')` retorna um `PDES`.
   - `test_backends.py` → `calor_2d()` retorna uma `list[PDE]`.
   - `test_regions.py` → `calor_2d(n=31)` retorna um único `PDE`, e o
     parâmetro `n` nunca é usado.
2. **Duplicação de `mae()`** em 7 arquivos (calor, adveccao, burgers, dominio,
   mixed, nonuniform, periodic).
3. **Boilerplate `matplotlib.use('Agg')`** repetido nos 15 arquivos, e frágil:
   importar `pdesolver.Auxs.Visualize` roda `_select_backend()`, que pode trocar
   o backend para TkAgg em máquinas com tkinter instalado.
4. **`import pytest` não usado** em calor, adveccao e burgers.
5. **Acesso inconsistente a `results`.** A maioria dos arquivos usa
   `sim.results[0]`; `test_fitzhugh.py` e `test_reacao1d.py` fazem
   `_, hist = results; hist[0][-1]`. O contrato real — `results = (finais,
   historico)` com `historico[func][passo]` — não está documentado em lugar
   nenhum.
6. **Lacunas de cobertura** para "tudo o que a lib oferece": condição de
   contorno **Robin**, `PDES.visualize()`, `PDES.__repr__` e o módulo
   `Auxs/FuncAux` não têm teste dedicado.

## Objetivos

- Eliminar as inconsistências acima **sem alterar nenhum comportamento da
  biblioteca** (mudança apenas em `Tests/`).
- Adicionar um arquivo `Tests/test_api.py` que exercita **cada superfície
  pública ao menos uma vez** e fecha as lacunas de cobertura reais.
- Cobrir, de forma explícita, diferentes dimensões de qualidade: correção,
  contrato de tipo/shape, casos de borda, tratamento de erro, guardas de
  regressão e robustez em ambiente headless.

## Não-objetivos

- Não alterar código de produção da `pdesolver`. Se um teste revelar um bug
  real (ex.: a implementação de Robin), o achado é **reportado ao usuário**, não
  corrigido silenciosamente nesta branch.
- Não consolidar nem mesclar arquivos de teste existentes (o "refactor completo"
  foi recusado).
- Não mexer no stub `pdesolver/Disc/boundaries/robin.py` (`RobinBC`), que
  aparenta ser código morto — o caminho real de Robin é `Disc.bc_coeffs`. Fica
  registrado como possível follow-up.
- Sem mudanças em benchmarks ou performance.

## Arquitetura da mudança

### Parte A — `Tests/conftest.py` e helpers compartilhados

Criar `Tests/conftest.py` com:

- **Fixture `autouse` de backend headless:** força
  `matplotlib.use("Agg", force=True)` *após* o import da biblioteca, blindando a
  suíte contra o `_select_backend()` de `Visualize`. Substitui o boilerplate
  `matplotlib.use('Agg')` espalhado.
- **Helpers numéricos `mae` e `rmse`** num único ponto importável, removendo as
  7 cópias de `mae`. A forma de exposição (função importável em
  `Tests/_helpers.py` vs. fixture) será decidida na implementação conforme o que
  o `--import-mode=importlib` do projeto suporta de forma limpa; o critério é:
  um único ponto de definição, importado pelos demais arquivos.

Fábricas de sistema realmente reusadas entre arquivos podem migrar para
`conftest.py`; fábricas usadas em um único arquivo permanecem locais, mas com
nome honesto ao que retornam.

### Parte B — Desfazer colisões e smells pontuais

Renomear os três `calor_2d` para nomes que refletem o tipo de retorno:

| Arquivo | Antes | Depois |
| --- | --- | --- |
| `test_analysis.py` | `calor_2d(n, mesh)` → `PDES` | `sim_calor2d(n, mesh)` |
| `test_backends.py` | `calor_2d()` → `list[PDE]` | `pdes_calor2d()` |
| `test_regions.py` | `calor_2d(n=31)` → `PDE` | `pde_calor2d()` (sem `n`) |

Remover os `import pytest` não usados. Substituir as `mae()` locais pelo helper
compartilhado. Nenhuma asserção ou tolerância numérica é alterada — só a forma.

### Parte C — `Tests/test_api.py` (smoke/contrato de superfície)

Arquivo novo, organizado por área da API pública. Cada teste exercita a
superfície e valida tipo/shape/erro, **sem reprovar a numérica profunda** que já
existe nos arquivos por-feature (esses continuam sendo a autoridade de
acurácia).

Áreas cobertas:

1. **`PDE`** — construção 1D e 2D; atributos preservados.
2. **`PDES`** — `funcs`, `sp_vars`; getter/setter de `disc_n` (o setter
   recomputa `grid` e `ic`); `__repr__` nos estados "not solved" e "solved".
3. **`discretize`** — `central`/`forward`/`backward` no backend simbólico;
   backend `stencil`; backend inválido levanta `ValueError`.
4. **`solve`** — `bdf2`, `CN`, `RKF` (simbólico) e `imex` (stencil) rodam e
   retornam `results`. **Fixa e documenta o contrato**: `results` é uma tupla
   `(finais, historico)` e `len(historico[func]) == nt + 1`.
5. **Analysis** — smoke leve de `analyze()` (chaves do dict), `stability_limit()`
   (float), `truncation_error()` e `modified_equation()` (listas).
6. **Condições de contorno** — Dirichlet, Neumann e Periodic em smoke; **Robin**
   com teste real (`west_func_bd='α; β; g'`, relação `α·u + β·u′ = g`),
   verificando o comportamento de ghost-node / estacionário; BC desconhecida
   levanta erro.
7. **Persistência** — smoke leve de `save_to_json` / `load_from_json` (round-trip
   básico; deep já em `test_load.py`).
8. **`visualize()`** — smoke de todos os modos sob Agg: 1D (`plot1d`,
   `plot1d_all`, `heatmap1d`, `animation1d`) e 2D (`heatmap`, `animation`,
   `plot3d`, `animation3d`), afirmando que a chamada não levanta exceção e cria
   figura; inclui os caminhos de guarda (sem `solve()`, modo desconhecido).
9. **`FuncAux`** — unidades puras: chaves de `build_func_map`, derivada de
   `d_dt`, fronteira de palavra de `repl_symbol`.

### Dimensões de qualidade (mapa)

| Dimensão | Onde |
| --- | --- |
| Correção vs. analítica | Robin, solve (leve); profundos já existentes |
| Contrato de tipo/shape | `results` tupla, chaves de dict, tamanho de histórico |
| Casos de borda | 1D vs 2D, `__repr__` resolvido/não-resolvido |
| Tratamento de erro | backend inválido, domínio inválido, BC/modo desconhecido |
| Regressão | `len(historico[func]) == nt + 1` |
| Robustez headless | `visualize()` não quebra sob Agg |

## Verificação

- **Baseline verde:** rodar `pytest` na suíte inteira antes de qualquer mudança e
  registrar o resultado.
- **Sem regressão:** após cada etapa, `pytest` deve manter todos os testes
  previamente verdes passando; nenhum é enfraquecido.
- **Lint:** `ruff check` (regras E, F, W, I já configuradas em `pyproject.toml`)
  sem apontamentos nos arquivos tocados — em especial F401 (imports não usados).

## Riscos e mitigações

- **Robin pode estar incorreto na lib.** Se o teste real de Robin falhar por bug
  de produção, o teste é marcado `xfail(strict=True)` com motivo explícito e o
  achado é reportado — não se altera a lib nesta branch.
- **Modos de animação sob Agg.** `FuncAnimation` é construído de forma preguiçosa;
  o smoke valida a construção sem exceção, filtrando o warning inócuo de
  "animation deleted without rendering".
- **Import-mode e helpers compartilhados.** Se a importação direta de um módulo
  auxiliar não for limpa sob `importlib`, os helpers são expostos como fixtures
  de `conftest.py`.

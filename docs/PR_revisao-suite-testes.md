# Revisão da suíte de testes + teste abrangente de API

## Resumo

A suíte de testes cresceu de forma orgânica e acumulou inconsistências de
manutenção (nomes colidindo, helpers duplicados, boilerplate espalhado) e
deixava partes da API pública sem cobertura. Esta PR **limpa a suíte** e
**adiciona um teste de superfície/contrato** que exercita tudo o que a
biblioteca oferece.

**Somente testes** — nenhuma linha de `pdesolver/` foi alterada.

## Motivação

- O helper `calor_2d` estava definido em **três arquivos com três contratos
  incompatíveis** (retornando ora `PDES`, ora `list[PDE]`, ora um único `PDE`,
  este último com um parâmetro `n` que nunca era usado).
- A função `mae` estava **copiada em 7 arquivos**; `matplotlib.use('Agg')`
  repetido nos 15; havia `import` mortos (`pytest`, `warnings`, `os`,
  `tempfile`).
- O contrato de `results` (`(finais, historico)`) não estava documentado, e os
  arquivos o acessavam de formas divergentes.
- **Sem cobertura dedicada:** condição de contorno **Robin**, `visualize()`,
  `__repr__` e o módulo `Auxs/FuncAux`.

## Mudanças

### A. Infraestrutura compartilhada
- `Tests/conftest.py` — força backend headless **Agg** (blindando contra o
  `_select_backend()` de `Visualize`, que troca de backend em máquinas com
  Tk/Qt) e torna `Tests/` importável.
- `Tests/_helpers.py` — `mae` e `rmse` num único ponto; remove as 7 cópias.

### B. Limpeza dos arquivos existentes
- Desfeita a colisão `calor_2d` → `sim_calor2d` (analysis), `pdes_calor2d`
  (backends), `pde_calor2d` (regions, sem o parâmetro morto).
- Removidos boilerplate de Agg, `mae` locais e imports não usados.
- Suíte permanece **verde e com as mesmas tolerâncias numéricas** (nenhum teste
  enfraquecido).

### C. Novo `Tests/test_api.py` (36 testes)
Smoke/contrato de superfície cobrindo cada método público ao menos uma vez:
`PDE`/`PDES`, setter de `disc_n`, `__repr__`, `discretize` (central/forward/
backward, stencil, backend inválido), `solve` (bdf2/CN/RKF/imex), Analysis,
persistência, contornos (incl. **Robin real**), `visualize()` (todos os modos
1D/2D) e `FuncAux`.

## Achados durante a revisão

1. **Contrato de `results` documentado.** É a tupla `(finais, historico)`, com
   `historico[func][passo]`. Métodos de **passo fixo** (bdf2/CN/imex) guardam
   `nt + 1` snapshots; **RKF é adaptativo** e o número de saídas varia — o teste
   agora afirma cada caso explicitamente.
2. **`RobinBC` (`pdesolver/Disc/boundaries/robin.py`) é código morto.** O caminho
   real de Robin é `Disc.bc_coeffs` (relação `α·u + β·u′ = g` via string
   `"α; β; g"`), verificado por `test_robin_estado_estacionario`. A classe
   `RobinBC` não é usada por `df` — **sugestão de follow-up:** remover ou ligar.

Nenhum `xfail` foi necessário: Robin e todos os modos de `visualize` passam.

## Cobertura por dimensão de qualidade

| Dimensão | Onde |
| --- | --- |
| Contrato de tipo/shape | `results` tupla, chaves de `analyze`, tamanho do histórico |
| Casos de borda | 1D vs 2D, `__repr__` resolvido/não-resolvido |
| Tratamento de erro | backend inválido, sem discretizar, método/contorno desconhecido |
| Correção | Robin (estado estacionário `1 - x/2`) |
| Regressão | passo fixo → `nt + 1` snapshots |
| Robustez headless | `visualize()` não quebra sob Agg |

## Como testar

```bash
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m ruff check Tests/
```

Esperado: suíte verde (baseline de 113 + 36 novos) e `All checks passed!`.

## Notas

- Sem alteração de comportamento da biblioteca.
- Sem mudança em benchmarks/performance.

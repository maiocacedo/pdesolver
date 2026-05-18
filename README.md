# pdesolver

Solver simbólico para equações diferenciais parciais (PDEs) usando discretização por diferenças finitas.

## Instalação

```bash
pip install pdesolver
```

Para suporte a GPU (CUDA 12):

```bash
pip install pdesolver[gpu]
```

## Uso rápido

```python
from pdesolver import PDE, PDES

pde = PDE(
    eq="du/dt = d2u/dx2",
    func="u",
    sp_var=["x"],
    ivar=["t"],
    ivar_boundary=[(0, 1)],
    expr_ic="sin(pi*x)",
    west_bd="Dirichlet", west_func_bd="0",
    east_bd="Dirichlet", east_func_bd="0",
)

sistema = PDES(pdes=[pde], disc_n=[50])
sistema.discretize(method='central')
sistema.solve(method='bdf2', tf=0.1, nt=100)
sistema.visualize(mode='plot1d_all', tf=0.1)
```

## Métodos disponíveis

### Discretização espacial
- `method='central'` — diferenças centrais (padrão)
- `method='forward'` — diferenças progressivas
- `method='backward'` — diferenças regressivas

### Integração temporal
- `method='bdf2'` — BDF-2 implícito (padrão, recomendado)
- `method='CN'` — Crank-Nicolson
- `method='RKF'` — Runge-Kutta-Fehlberg 4(5) com CUDA *(requer cupy)*

### Condições de contorno
- `Dirichlet` — valor prescrito
- `Neumann` — fluxo prescrito
- `Robin` — combinação linear

### Visualização
- `'plot1d'` / `'plot1d_all'` — perfis 1D
- `'heatmap1d'` — mapa espaciotemporal 1D
- `'animation1d'` — animação 1D
- `'heatmap'` — mapa de calor 2D
- `'plot3d'` / `'animation3d'` — superfície 3D

## Salvar e carregar resultados

```python
sistema.save_to_json("resultado.json")

sistema2 = PDES.load_from_json("resultado.json")
```

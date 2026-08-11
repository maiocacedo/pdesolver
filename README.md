# pdesolver

**Symbolic solver for partial differential equations (PDEs) using finite difference discretization.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![PyPI](https://img.shields.io/pypi/v/pdesolver.svg)](https://pypi.org/project/pdesolver/)
[![CI](https://github.com/maiocacedo/PDESsolver/actions/workflows/ci.yml/badge.svg)](https://github.com/maiocacedo/PDESsolver/actions/workflows/ci.yml)

---

## Overview

**pdesolver** is designed for researchers, students, and educators in computational science who need to solve partial differential equations numerically without the steep learning curve of full-scale FEM frameworks like FEniCS or PETSc. It bridges the gap between symbolic mathematics (SymPy) and production numerical solvers by letting users define PDEs in an intuitive string notation and automatically handling discretization, boundary condition assembly, and time integration.

**Target audience:** Anyone working with 1D or 2D time-dependent PDEs who wants a Pythonic, accessible tool that "just works" from equation to visualization in a few lines of code.

**At a glance:** Dirichlet, Neumann, Robin and periodic boundaries; uniform and
[stretched meshes](#non-uniform-meshes); first, second and
[mixed second derivatives](#equation-notation); coupled systems; implicit
(BDF2, Crank–Nicolson) and explicit adaptive (RKF45) time integration; and a
[vectorized backend](#discretization-backends) whose symbolic cost is
independent of the mesh size, for large 2D grids and GPU execution.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Equation Notation](#equation-notation)
- [The `PDE` Class](#the-pde-class)
- [The `PDES` Class (System)](#the-pdes-class-system)
- [Boundary Conditions](#boundary-conditions)
- [Spatial Discretization](#spatial-discretization)
- [Non-Uniform Meshes](#non-uniform-meshes)
- [Discretization Analysis](#discretization-analysis)
- [Discretization Backends](#discretization-backends)
- [Solvers (Time Integration)](#solvers-time-integration)
- [Visualization](#visualization)
- [Import & Export (JSON)](#import--export-json)
- [Coupled Systems](#coupled-systems)
- [PDE Solver Studio](#pde-solver-studio)

---

## Installation

```bash
pip install pdesolver
```

For GPU support (CUDA 12):

```bash
pip install pdesolver[gpu]
```

### Dependencies

| Package      | Minimum Version |
|:-------------|:----------------|
| numpy        | ≥ 1.24          |
| sympy        | ≥ 1.12          |
| matplotlib   | ≥ 3.7           |
| scipy        | ≥ 1.10          |
| cupy-cuda12x | *(optional, GPU only)* |

---

## Quick Start

```python
from pdesolver import PDE, PDES

# Define a 1D heat equation: ∂u/∂t = ∂²u/∂x²
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

# Create a system, discretize, solve, and visualize
sistema = PDES(pdes=[pde], disc_n=[50])
sistema.discretize(method='central')
sistema.solve(method='bdf2', tf=0.1, nt=100)
sistema.visualize(mode='plot1d_all', tf=0.1)
```

---

## Equation Notation

Equations are written as strings using a human-readable notation. The library parses these strings symbolically using SymPy.

| Notation | Mathematical Meaning | Example |
|:---------|:---------------------|:--------|
| `du/dx` | ∂u/∂x — first partial derivative | `du/dx` |
| `d2u/dx2` | ∂²u/∂x² — second partial derivative | `d2u/dx2` |
| `du/dy` | ∂u/∂y — first derivative in y | `du/dy` |
| `d2u/dy2` | ∂²u/∂y² — second derivative in y | `d2u/dy2` |
| `d2u/dxdy` | ∂²u/∂x∂y — mixed second derivative (2D only) | `d2u/dxdy` |
| `du/dt` | ∂u/∂t — time derivative (always on the **left** side of `=`) | `du/dt = ...` |
| `u` | the unknown function itself | `- k*u` |

**Rules:**
- The **left side** of `=` must always be the time derivative: `du/dt = ...`
- The **right side** contains the spatial derivatives, source terms, and reactions.
- Standard math functions from SymPy are supported: `sin`, `cos`, `exp`, `sqrt`, `pi`, `Heaviside`, `**` (power), etc.
- Constants can be written inline: `0.1*d2u/dx2 + 3*u`

### Examples of valid equations

```
du/dt = d2u/dx2                                         # Heat equation 1D
du/dt = d2u/dx2 + d2u/dy2                               # Heat equation 2D
du/dt = -u*du/dx + 0.01*d2u/dx2                         # Burgers equation 1D
du/dt = -0.5*du/dx + 0.001*d2u/dx2 - 0.1*u             # Advection-diffusion-reaction
dU/dt = 1.0*d2U/dx2 + 1.0*d2U/dy2 + U - U**3/3 - V    # FitzHugh-Nagumo (U component)
du/dt = d2u/dx2 + 1.0*d2u/dxdy + d2u/dy2               # Anisotropic diffusion (mixed term)
```

> **Note:** `d2u/dxdy` and `d2u/dydx` are equivalent and produce the same
> stencil. The mixed term requires two spatial variables.

---

## The `PDE` Class

Each PDE is defined as a `PDE` object that stores the equation and all its properties:

```python
class PDE:
    def __init__(self, eq, func, sp_var, ivar, ivar_boundary, expr_ic,
                 west_bd="Dirichlet", west_func_bd="0",
                 east_bd="Dirichlet", east_func_bd="0",
                 north_bd="Dirichlet", north_func_bd="0",
                 south_bd="Dirichlet", south_func_bd="0"):
```

### Attributes

| Attribute | Type | Description |
|:----------|:-----|:------------|
| `eq` | `str` | The PDE equation as a string (e.g. `"du/dt = d2u/dx2"`). Left side must be the time derivative. |
| `func` | `str` | Name of the unknown function (e.g. `"u"`, `"C"`, `"T"`). This name must match the one used in `eq`. |
| `sp_var` | `list[str]` | Spatial variable names. Use `["x"]` for 1D or `["x", "y"]` for 2D problems. |
| `ivar` | `list[str]` | Independent variable for time integration, typically `["t"]`. |
| `ivar_boundary` | `list[tuple]` | Domain boundaries for each spatial variable. E.g. `[(0, 1)]` for x ∈ [0, 1] or `[(0, 1), (0, 2)]` for a 2D domain. |
| `expr_ic` | `str` | Initial condition as a SymPy expression string. Variables must match `sp_var`. E.g. `"sin(pi*x)"`, `"0"`, `"exp(-100*(x-0.5)**2)"`. |
| `west_bd` | `str` | Boundary condition type on the **west** side (x = x_min). One of: `"Dirichlet"`, `"Neumann"`, `"Robin"`. Default: `"Dirichlet"`. |
| `west_func_bd` | `str` | Boundary value/expression for the west side. Can depend on `x`, `y`, and `t`. Default: `"0"`. |
| `east_bd` | `str` | Boundary condition type on the **east** side (x = x_max). Default: `"Dirichlet"`. |
| `east_func_bd` | `str` | Boundary value/expression for the east side. Default: `"0"`. |
| `north_bd` | `str` | Boundary condition type on the **north** side (y = y_max, 2D only). Default: `"Dirichlet"`. |
| `north_func_bd` | `str` | Boundary value/expression for the north side. Default: `"0"`. |
| `south_bd` | `str` | Boundary condition type on the **south** side (y = y_min, 2D only). Default: `"Dirichlet"`. |
| `south_func_bd` | `str` | Boundary value/expression for the south side. Default: `"0"`. |

### Boundary sides reference

For **1D** problems, only `west` (left) and `east` (right) boundaries are used:

```
west  ----------- domain -----------  east
x = x_min                           x = x_max
```

For **2D** problems, all four sides are used:

```
               north (y = y_max)
          +------------------------+
          |                        |
  west    |        domain          |  east
(x=x_min) |                        | (x=x_max)
          +------------------------+
               south (y = y_min)
```

---

## The `PDES` Class (System)

The `PDES` class manages one or more `PDE` objects as a coupled system. It orchestrates discretization, solving, and visualization.

```python
sistema = PDES(pdes=[pde1, pde2], disc_n=[50])       # 1D with 50 points
sistema = PDES(pdes=[pde], disc_n=[30, 30])           # 2D with 30×30 grid
```

### Constructor

| Parameter | Type | Description |
|:----------|:-----|:------------|
| `pdes` | `list[PDE]` | List of PDE objects. For coupled systems, pass multiple PDEs. |
| `disc_n` | `list[int]` | Number of grid points per spatial dimension. `[50]` for 1D, `[30, 30]` for 2D. |
| `mesh` | `str \| dict \| list` | Node distribution. `'uniform'` (default), `'chebyshev'`, `'tanh'`, `'tanh_left'`, `'tanh_right'`, or explicit nodes. See [Non-Uniform Meshes](#non-uniform-meshes). |
| `backend` | `str` | `'symbolic'` (default) or `'stencil'`. See [Discretization Backends](#discretization-backends). |

### Key methods

| Method | Description |
|:-------|:------------|
| `discretize(method='central')` | Applies finite difference discretization to the spatial derivatives. |
| `solve(method='bdf2', tf=1.0, nt=100, ...)` | Integrates the system in time. |
| `visualize(mode='heatmap', func_idx=0, ...)` | Generates plots and animations. |
| `save_to_json(filepath)` | Exports the system and results to a JSON file. |
| `PDES.load_from_json(filepath)` | Class method — loads a system from a JSON file. |

---

## Boundary Conditions

The library supports four types of boundary conditions. Dirichlet, Neumann, and Robin are applied independently to each side of the domain; Periodic applies to a whole axis and must be declared on both of its sides.

### Dirichlet — prescribed value

Fixes the function value at the boundary. The value can be a constant or a time/space-dependent expression.

**Mathematical form:** u(x_boundary, t) = g(x, y, t)

```python
pde = PDE(
    eq="du/dt = d2u/dx2", func="u",
    sp_var=["x"], ivar=["t"],
    ivar_boundary=[(0, 1)],
    expr_ic="sin(pi*x)",
    west_bd="Dirichlet", west_func_bd="0",       # u(0, t) = 0
    east_bd="Dirichlet", east_func_bd="sin(t)",   # u(1, t) = sin(t)
)
```

### Neumann — prescribed flux (derivative)

Fixes the normal derivative at the boundary. Implemented using a second-order one-sided finite difference stencil.

**Mathematical form:** ∂u/∂n(x_boundary, t) = g(x, y, t)

```python
pde = PDE(
    eq="du/dt = d2u/dx2", func="u",
    sp_var=["x"], ivar=["t"],
    ivar_boundary=[(0, 1)],
    expr_ic="sin(pi*x)",
    west_bd="Neumann", west_func_bd="0",   # ∂u/∂x(0, t) = 0  (insulated)
    east_bd="Neumann", east_func_bd="0",   # ∂u/∂x(1, t) = 0  (insulated)
)
```

> **Note:** Neumann conditions require at least 3 grid points per dimension (`disc_n ≥ 3`).

### Robin — linear combination

Combines Dirichlet and Neumann conditions. Defined by coefficients α and β such that:

**Mathematical form:** α·u + β·∂u/∂n = g(x, y, t)

Robin boundary conditions are configured via the discretization module with `alpha` and `beta` parameters.

### Periodic — wrap-around domain

The solution and its derivatives are continuous across the domain edges, so
`u(a, t) = u(b, t)`. Useful for advection on a ring, spectral-like test cases,
and any problem with no physical boundary.

```python
# 1D advection on a periodic ring
pde = PDE(
    eq="du/dt = -du/dx",
    func="u",
    sp_var=["x"], ivar=["t"],
    ivar_boundary=[(0, 1)],
    expr_ic="sin(2*pi*x)",
    west_bd="Periodic", east_bd="Periodic",
)
```

```python
# 2D torus — both axes periodic
pde = PDE(
    eq="du/dt = 0.05*d2u/dx2 + 0.05*d2u/dy2",
    func="u",
    sp_var=["x", "y"], ivar=["t"],
    ivar_boundary=[(0, 1), (0, 1)],
    expr_ic="sin(2*pi*x)*sin(2*pi*y)",
    west_bd="Periodic",  east_bd="Periodic",
    north_bd="Periodic", south_bd="Periodic",
)
```

> **Periodicity is a property of an axis, not of a side.** Both sides of an axis
> must be declared together — `west_bd="Periodic"` with `east_bd="Dirichlet"`
> raises `ValueError`. Declaring one axis periodic and the other not is fine
> (a cylinder).

> **Node count:** a periodic axis places `disc_n` nodes over `[a, b)` — the
> endpoint is *not* duplicated, since `u(a) = u(b)` would make the system
> singular. A periodic axis therefore produces no Dirichlet constraints.

### Mixed boundaries

You can mix boundary types freely on different sides:

```python
pde = PDE(
    eq="dC/dt = -0.5*dC/dx + 0.001*d2C/dx2 - 0.1*C",
    func="C",
    sp_var=["x"], ivar=["t"],
    ivar_boundary=[(0, 1)],
    expr_ic="0",
    west_bd="Dirichlet", west_func_bd="1",   # fixed inlet concentration
    east_bd="Neumann",   east_func_bd="0",   # zero-flux outlet
)
```

---

## Spatial Discretization

The `discretize()` method replaces spatial derivatives with finite difference approximations.

```python
sistema.discretize(method='central')   # default
```

### Available schemes

| Method | Name | 1st derivative (∂u/∂x) | 2nd derivative (∂²u/∂x²) | Best for |
|:-------|:-----|:------------------------|:--------------------------|:---------|
| `'central'` | Central differences | (u_{i+1} − u_{i−1}) / 2h | (u_{i+1} − 2u_i + u_{i−1}) / h² | Diffusion-dominated problems |
| `'forward'` | Forward differences | (u_{i+1} − u_i) / h | (u_{i+1} − 2u_i + u_{i−1}) / h² | — |
| `'backward'` | Backward differences | (u_i − u_{i−1}) / h | (u_{i+1} − 2u_i + u_{i−1}) / h² | Advection-dominated problems |

> **Note:** The second derivative stencil is the same for all three methods. Only the first derivative approximation changes.

The stencils above are shown for a uniform mesh. On a stretched mesh the same
schemes are built from variable-coefficient weights that reduce exactly to
these formulas when the spacing is constant.

---

## Non-Uniform Meshes

Grid points do not have to be equally spaced. Clustering nodes where the
solution varies fastest — boundary layers, shocks, sharp fronts — resolves the
feature with far fewer points than a uniform grid of the same size.

```python
sistema = PDES(pdes=[pde], disc_n=[41], mesh='chebyshev')
sistema = PDES(pdes=[pde], disc_n=[41], mesh={'type': 'tanh_right', 'beta': 5.0})
sistema = PDES(pdes=[pde], disc_n=[30, 30], mesh=['tanh', 'uniform'])  # per axis
```

### Available distributions

| `type` | Clustering | Parameter | Notes |
|:-------|:-----------|:----------|:------|
| `'uniform'` | none | — | Default; equally spaced. |
| `'chebyshev'` | both ends | — | Gauss–Lobatto nodes. |
| `'tanh'` | both ends | `beta` | Symmetric stretching; larger `beta` = stronger clustering. |
| `'tanh_left'` | near `a` | `beta` | One-sided, towards the start of the domain. |
| `'tanh_right'` | near `b` | `beta` | One-sided, towards the end of the domain. |

Explicit node arrays are also accepted:

```python
import numpy as np
sistema = PDES(pdes=[pde], disc_n=[25], mesh={'nodes': np.linspace(0, 1, 25)**2})
```

### Why it helps

For an advection–diffusion boundary layer of width ε = 0.005 solved on 41 points,
clustering towards the outflow boundary cuts the mean absolute error by ~4×
compared to a uniform grid with the same number of points:

| Mesh | MAE |
|:-----|----:|
| `'uniform'` | 1.85 × 10⁻² |
| `{'type': 'tanh_right', 'beta': 5.0}` | 4.23 × 10⁻³ |

The variable-coefficient stencils remain second-order accurate on smoothly
stretched meshes.

> **Note:** a periodic axis supports `'uniform'` spacing or explicit nodes.

---

## Discretization Analysis

Because the equation is kept in symbolic form, the library can derive what your
discretization is *actually* doing — before you run it, and for the equation
you actually wrote rather than a tabulated special case.

```python
sistema = PDES(pdes=[pde], disc_n=[51])
sistema.discretize(method='backward')
sistema.analyze()
```

```
  [análise] Discretização
    du/dx      esquema 'backward' — erro O(h^1), termo líder -h/2*u^(2)
      na malha atual: |coef| máximo = 1.000e-02
    d2u/dx2    esquema 'backward' — erro O(h^2), termo líder h**2/12*u^(4)
      na malha atual: |coef| máximo = 3.333e-05
  [análise] Difusão numérica introduzida pelo esquema
    de du/dx: adiciona 1.000e-02 vs física 1.000e-02 (100.0%)
  [análise] Estabilidade
    RKF: dt_max = 1.839e-02 (símbolo de Fourier, |z| = 3.6777)
    espectro do operador montado: dt_max = 1.966e-02
```

That report says something a convergence test would not: on this mesh the
upwind scheme contributes as much diffusion as the physical viscosity, so the
simulation is running at roughly twice the intended ν.

### Methods

| Method | Returns |
|:-------|:--------|
| `analyze(method='RKF')` | Full report — truncation, numerical diffusion, stability, Péclet. Prints unless `verbose=False`. |
| `truncation_error()` | Leading truncation term of every discretized derivative, symbolic and as realised on the current mesh. |
| `modified_equation()` | The terms the scheme adds to the PDE that are absent from the continuous equation. |
| `stability_limit(method='RKF')` | Largest stable `dt`. Returns `inf` for the A-stable implicit methods. |

### What it detects

- **Truncation error and formal order**, derived from the stencil moments rather than tabulated — so it stays correct on stretched meshes.
- **Numerical diffusion, with its sign.** A downwind scheme reports `SUBTRAI`, together with a warning that an anti-diffusive scheme tends to be unstable.
- **Growing modes.** If the discrete spatial operator has an eigenvalue with positive real part, no time step makes the scheme stable, and the report says so instead of quoting a `dt_max`.
- **Cell Péclet number**, warning when central differencing will oscillate.

### Closing the loop: automatic IMEX splitting

The stiffness analysis is not only reported — it drives an integrator.
`solve(method='imex')` classifies each additive term by the order of the spatial
derivative it carries and by whether it is linear in the unknowns. Linear
second-order terms scale as `h⁻²` and are what force the step size down, so they
go to the implicit side; first-order and reaction terms stay explicit. A
semi-implicit BDF2 then advances the system, factorizing the implicit operator
once.

```python
sistema = PDES(pdes=[pde], disc_n=[321, 321], backend='stencil')
sistema.discretize(method='central')
sistema.solve(method='imex', tf=0.2, nt=200, verbose=True)
```

```
[IMEX] Separação simbólica: 2/5 termos implícitos
  implícito  |λ|=512.00   0.02*XX0_xx
  implícito  |λ|=512.00   0.02*XX0_yy
  explícito  |λ|= 79.81   -1.0*XX0_x
  explícito  |λ|=  1.00   XX0
  explícito  |λ|=  0.00   -XX0**3  [não linear]
[IMEX] |λ| total=1024, explícito=79.81 → passo ~12.8x maior
```

Nonlinear terms are never moved to the implicit side, which keeps the implicit
operator constant and the factorization reusable.

### The implicit stage: fast Poisson solver

When the implicit part is a constant-coefficient Laplacian on a uniform 2D grid
with Dirichlet boundaries, the discrete sine basis diagonalizes it exactly, and
the stage is solved by a **discrete sine transform** in `O(N log N)` instead of
a sparse triangular solve. The solver detects this structure automatically and
falls back to a sparse LU factorization otherwise — a stretched mesh, a variable
coefficient or a mixed term all disqualify it, and each case is covered by a
test.

The DST result reproduces the sparse LU solve to machine precision
(≈10⁻¹³), and is faster from the smallest 2D meshes upward:

| Grid | Unknowns | sparse LU | DST | Speedup |
|:-----|---------:|----------:|----:|--------:|
| 41²  | 1 681    | 0.07 ms | 0.04 ms | 1.7× |
| 81²  | 6 561    | 0.34 ms | 0.09 ms | 3.5× |
| 161² | 25 921   | 2.14 ms | 0.31 ms | 6.8× |
| 321² | 103 041  | 12.78 ms | 1.74 ms | **7.3×** |

It also removes the factorization itself, which costs 0.32 s at 321².

> **Against the implicit solvers.** This is where the gain is largest. On a
> non-linear problem (`+ U − U³`) with Dirichlet boundaries, at the same step
> count and comparable accuracy:
>
> | Grid | BDF2 | CN | **IMEX** |
> |:-----|-----:|---:|---------:|
> | 128² | 25.5 s | 22.7 s | **0.88 s** |
> | 256² | 67.3 s | 73.6 s | **0.87 s** |
>
> 26× to 85× faster, because BDF2 and CN enter Newton at every step — colored
> Jacobian, assembly, sparse LU — while IMEX puts the non-linearity on the
> explicit side and **removes Newton entirely**, leaving one DST per step. Note
> that the IMEX cost barely moves from 128² to 256².
>
> **Against the explicit solver.** On a 321² advection–diffusion–reaction
> problem the explicit RKF needs 886 steps — it is stability limited — against
> 200 for IMEX, completing in 2.75 s against 12.40 s, a **4.5×** speedup
> (2.3× at 161²). The advantage only appears over long enough horizons: on short
> integrations where stability is not binding, RKF wins outright.
>
> These end-to-end timings were taken on a busy machine and varied by roughly a
> factor of two between runs. The per-solve figures in the table above are
> stable; re-measure the end-to-end ratio on an idle machine before quoting it.

### Stability

The time step limit is obtained two independent ways, which cross-check each
other: from the Fourier symbol of the stencil combined with the stability
polynomial of the Runge–Kutta tableau, and from the eigenvalues of the
assembled operator matrix. On the 2D heat equation the two agree to within
0.1%.

> **Interpretation:** `dt_max` is a worst case over *all* Fourier modes. A run
> starting from smooth data can exceed it while the highest modes carry no
> energy, so treat it as a design bound rather than a hard prediction of the
> step the adaptive solver will choose.

### A note on stretched meshes

The variable-coefficient weights are the exact three-point interpolatory
weights, for which the `u''` moment cancels identically — so the first
derivative stays second order on an *arbitrary* mesh, with leading term
`h⁻h⁺/6·u'''`. The naive centred formula `(u₊ − u₋)/(h⁻ + h⁺)` does not have
this property and drops to first order as soon as the spacing varies. You can
confirm this on your own mesh with `truncation_error()`.

---

## Discretization Backends

The same discretization is available through two independent implementations.
They agree to machine precision, which is what the cross-backend test suite
checks; the choice is purely about performance.

```python
sistema = PDES(pdes=[pde], disc_n=[30, 30])                       # 'symbolic' (default)
sistema = PDES(pdes=[pde], disc_n=[512, 512], backend='stencil')  # large meshes
```

| Backend | How it works | Cost of setup |
|:--------|:-------------|:--------------|
| `'symbolic'` | Emits one equation string per grid node and compiles them all with SymPy. Every node is parsed individually. | Grows with the mesh. |
| `'stencil'` | Parses each PDE **once** into an expression over derivative fields, then evaluates it over whole arrays with NumPy or CuPy. | Independent of the mesh. |

Because the symbolic backend re-derives the same stencil at every node, its
setup cost grows with the grid while the PDE itself does not change. Measured
on the 2D heat equation (time to reach a state ready to integrate):

| Grid | DOF | `'symbolic'` | `'stencil'` | Speedup |
|:-----|----:|-------------:|------------:|--------:|
| 15² | 225 | 0.61 s | 0.022 s | 28× |
| 30² | 900 | 5.06 s | 0.006 s | 877× |
| 60² | 3 600 | 25.6 s | 0.010 s | 2 646× |
| 128² | 16 384 | not feasible | 0.014 s | — |
| 256² | 65 536 | not feasible | 0.025 s | — |
| 512² | 262 144 | not feasible | 0.034 s | — |

The stencil backend also supplies an analytic sparsity pattern and colouring
for the implicit solvers. The number of right-hand side evaluations needed to
assemble the Jacobian becomes **independent of the mesh** — 9 colours whether
the grid has 225 or 65 536 unknowns, instead of one evaluation per column.

### GPU execution

The stencil backend is array-module neutral, so the same operator runs under
NumPy or CuPy. `solve(method='RKF', ...)` moves to the GPU automatically once
the problem is large enough to amortize kernel launch overhead. Right-hand side
evaluation on an RTX 4070 Ti:

| Grid | DOF | CPU | GPU | Speedup |
|:-----|----:|----:|----:|--------:|
| 128² | 16 384 | 0.09 ms | 0.46 ms | 0.2× |
| 256² | 65 536 | 0.36 ms | 0.46 ms | 0.8× |
| 512² | 262 144 | 6.1 ms | 0.45 ms | **13.5×** |
| 1024² | 1 048 576 | 25.4 ms | 1.0–1.5 ms | **17–24×** |

Below roughly 65 000 unknowns the GPU is *slower* — its runtime is dominated by
a fixed ~0.45 ms launch overhead — which is why the automatic switch only
happens above that size. A full RKF solve on a 512² grid takes 9.2 s on CPU and
0.69 s on GPU, producing results that agree bit for bit.

> **Windows note:** when CUDA libraries come from the NVIDIA pip wheels, their
> DLL directories are added to the search path automatically on import. No
> manual `PATH` configuration is needed.

All features — periodic boundaries, non-uniform meshes, mixed derivatives —
are available on both backends. The one thing only `'symbolic'` can do is give
individual nodes their own equation; see below.

### Heterogeneous regions — symbolic backend only

The symbolic backend represents the discretized system as **one independent
algebraic equation per node**, so nodes can carry different equations. The
vectorized backend cannot: assuming the same equation everywhere is precisely
what makes vectorization possible.

```python
X, Y = sistema.grid.coords()
bloco = (abs(X - 0.5) < 0.12) & (abs(Y - 0.5) < 0.12)

sistema.discretize(method='central', regions=[
    {'where': bloco, 'eq': 'dU/dt = 0'},          # frozen internal obstacle
])
```

`'where'` takes a boolean mask over the grid or a callable of the grid
coordinates; `'eq'` is an equation in the usual notation that replaces the
system equation at those nodes. Boundary nodes keep their boundary condition —
regions apply to interior nodes.

This covers internal obstacles and masked domains, different physics per
region, point sources, and material interfaces. For full control, `disc_results`
is a plain list of expression strings that you may edit directly before
solving — needed when a node requires a stencil that the notation cannot
express, such as the conservative flux form across a conductivity jump:

```python
flat, d_vars = sistema.disc_results
flat[i] = f'({k2}*XX0_{i+1}_0 - {k1+k2}*XX0_{i}_0 + {k1}*XX0_{i-1}_0)/{h**2}'
sistema.disc_results = (flat, d_vars)
```

`examples/regioes_heterogeneas.py` runs three such cases and checks each
against a closed-form result: a frozen obstacle, a two-material interface
(interface temperature exact to 6 × 10⁻¹², flux continuous to 10⁻¹³), and a
point source (peak and derivative jump exact).

> Run `python benchmarks/bench_scaling.py` to reproduce these numbers on your
> own machine.

---

## Solvers (Time Integration)

The `solve()` method advances the discretized system through time. The library **automatically detects** whether the PDE is linear or nonlinear and selects the appropriate algorithm.

```python
sistema.solve(method='bdf2', tf=1.0, nt=100, verbose=False)
```

### Common parameters

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `method` | `str` | `'bdf2'` | Solver method: `'bdf2'`, `'CN'`, or `'RKF'`. |
| `tf` | `float` | `1.0` | Final simulation time. |
| `nt` | `int` | `100` | Number of time steps (for BDF2 and CN) or initial steps estimate (for RKF). |
| `tol` | `float` | `1e-6` | Error tolerance. Used by RKF for adaptive step control and by nonlinear solvers for convergence. |
| `verbose` | `bool` | `False` | Print solver performance information (timing, iterations, linearity detection). |

---

### BDF2 — Backward Differentiation Formula of 2nd order

**Implicit, A-stable, 2nd order.** Recommended as the default solver for most problems.

```python
sistema.solve(method='bdf2', tf=1.0, nt=200, verbose=True)
```

**How it works:**
- Uses the BDF-1 (implicit Euler) formula for the first time step, then switches to the BDF-2 formula.
- **Linear PDEs:** Pre-factorizes the system matrix using sparse LU decomposition (`scipy.sparse.linalg.splu`) — the factorization is reused at every step, making it very efficient.
- **Nonlinear PDEs:** Uses Newton's method at each time step with sparse colored Jacobian computation for efficiency.

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `nonlinear_method` | `str` | `'newton'` | Method for nonlinear equations: `'newton'` or `'picard'`. |
| `tol_nl` | `float` | `1e-8` | Convergence tolerance for the nonlinear solver. |
| `max_iter_nl` | `int` | `20` | Maximum nonlinear iterations per time step. |

---

### CN — Crank-Nicolson

**Implicit, A-stable, 2nd order.** A classic method that averages the explicit and implicit evaluations.

```python
sistema.solve(method='CN', tf=1.0, nt=200)
```

**How it works:**
- Evaluates the right-hand side at both the current time (explicit part) and the next time (implicit part), weighting each by 0.5.
- **Linear PDEs:** Solves via sparse LU factorization, similar to BDF2.
- **Nonlinear PDEs:** Uses Newton or Picard iteration.

Accepts the same `nonlinear_method`, `tol_nl`, and `max_iter_nl` parameters as BDF2.

---

### RKF — Runge-Kutta-Fehlberg 4(5)

**Explicit, adaptive step size, 4th/5th order.** Runs on the **GPU** via CuPy.

```python
sistema.solve(method='RKF', tf=1.0, nt=100, tol=1e-6)
```

**How it works:**
- Computes a 4th-order and a 5th-order solution at each step, using the difference to estimate the local error.
- The step size is automatically adjusted to keep the error below the specified tolerance.
- All computations are performed on the GPU using CuPy arrays.

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `tol` | `float` | `1e-5` | Relative error tolerance for step size control. |
| `dt_init` | `float` | `tf/nt` | Initial time step. |
| `dt_max` | `float` | `None` | Maximum allowed time step. |
| `atol` | `float` | `1e-6` | Absolute error tolerance. |
| `rtol` | `float` | `tol` | Relative error tolerance (defaults to `tol`). |
| `max_steps` | `int` | `10,000,000` | Maximum number of integration steps. |

> **Requires:** `pip install pdesolver[gpu]` (CuPy with CUDA 12).

### Solver comparison

| Feature | BDF2 | Crank-Nicolson | RKF 4(5) |
|:--------|:-----|:---------------|:---------|
| Type | Implicit | Implicit | Explicit |
| Order | 2nd | 2nd | 4th/5th |
| Step size | Fixed | Fixed | Adaptive |
| Hardware | CPU | CPU | GPU (CUDA) |
| Linear PDEs | ✅ LU factorization | ✅ LU factorization | ✅ |
| Nonlinear PDEs | ✅ Newton/Picard | ✅ Newton/Picard | ✅ |
| Stiff problems | ✅ A-stable | ✅ A-stable | ⚠️ Conditional |

---

## Visualization

The `visualize()` method provides several built-in plotting modes. The available modes depend on the dimensionality of the problem (1D or 2D).

```python
sistema.visualize(mode='plot1d_all', func_idx=0, tf=0.1)
```

### Common parameters

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `mode` | `str` | `'heatmap'` | Visualization mode (see tables below). |
| `func_idx` | `int` | `0` | Index of the function to plot (for coupled systems with multiple PDEs). |
| `time_step` | `int` | `-1` | Time step index to plot. `-1` means the last step. |

---

### 1D Visualization Modes

| Mode | Description | Key Parameters |
|:-----|:------------|:---------------|
| `'plot1d'` | Plots u(x) at a single time step. | `time_step`, `color='steelblue'`, `lw=2` |
| `'plot1d_all'` | Overlays multiple u(x) profiles at evenly spaced times with a colorbar. | `n_profiles=10`, `cmap='viridis'`, `lw=1.5`, `tf` |
| `'heatmap1d'` | Spatiotemporal heatmap — x (vertical) vs t (horizontal). Shows the full evolution. | `cmap='viridis'`, `tf` |
| `'animation1d'` | Animated plot of u(x) evolving in time. | `frames_step=1`, `interval=50`, `color='steelblue'`, `lw=2`, `tf` |

#### Examples

```python
# Single profile at the last time step
sistema.visualize(mode='plot1d', time_step=-1)

# 10 profiles overlaid, with real-time labels
sistema.visualize(mode='plot1d_all', n_profiles=10, tf=0.1, cmap='plasma')

# Full spatiotemporal heatmap
sistema.visualize(mode='heatmap1d', tf=0.1, cmap='inferno')

# Animation
sistema.visualize(mode='animation1d', tf=0.1, interval=30, frames_step=2)
```

---

### 2D Visualization Modes

| Mode | Description | Key Parameters |
|:-----|:------------|:---------------|
| `'heatmap'` | Filled contour plot (heatmap) at a single time step. | `time_step` |
| `'animation'` | Animated 2D contour plot over time. | `frames_step=1`, `interval=50` |
| `'plot3d'` | 3D surface plot at a single time step. | `time_step`, `cmap='hot'`, `alpha=1.0`, `elev=30`, `azim=-60` |
| `'animation3d'` | Animated 3D surface plot over time. | `frames_step=1`, `interval=50`, `cmap='hot'`, `alpha=1.0`, `elev=30`, `azim=-60` |

#### Examples

```python
# 2D heatmap at the final step
sistema.visualize(mode='heatmap', time_step=-1)

# 3D surface with custom view angle
sistema.visualize(mode='plot3d', time_step=-1, cmap='coolwarm', elev=45, azim=-45)

# Animated 3D surface
sistema.visualize(mode='animation3d', interval=30, cmap='viridis', elev=40)
```

---

## Import & Export (JSON)

The library supports saving and loading the entire system state (PDEs, grid configuration, and results) in JSON format.

### Saving

```python
sistema.save_to_json("my_simulation.json")
```

### Loading

```python
from pdesolver import PDES

loaded = PDES.load_from_json("my_simulation.json")
# Results are restored, visualization is immediately available:
loaded.visualize(mode='heatmap', time_step=-1)

# You can also re-discretize with a different grid:
loaded.disc_n = [30, 30]
loaded.discretize(method='central')
loaded.solve(method='bdf2', tf=0.5, nt=200)
```

### JSON structure

The exported JSON file has the following structure:

```json
{
    "disc_n": [50],
    "pdes": [
        {
            "eq": "du/dt = d2u/dx2",
            "func": "u",
            "expr_ic": "sin(pi*x)",
            "sp_var": ["x"],
            "ivar": ["t"],
            "ivar_boundary": [[0, 1]],
            "west_bd": "Dirichlet",
            "west_func_bd": "0",
            "east_bd": "Dirichlet",
            "east_func_bd": "0",
            "north_bd": "Dirichlet",
            "north_func_bd": "0",
            "south_bd": "Dirichlet",
            "south_func_bd": "0"
        }
    ],
    "results": [
        [
            [0.0, 0.0627, 0.1243, "...array of u values at t=0..."],
            [0.0, 0.0581, 0.1152, "...array of u values at t=dt..."],
            "...one array per time step..."
        ]
    ]
}
```

> **Note:** The `results` field is a nested list: `results[func_idx][time_step][spatial_point]`. Initial conditions (`ic`) are not saved because they are derived from `expr_ic` and `disc_n` on load.

---

## Coupled Systems

To solve systems of coupled PDEs, create multiple `PDE` objects and pass them together:

```python
from pdesolver import PDE, PDES

# Advection-diffusion-reaction: C → D with first-order kinetics
pde_c = PDE(
    eq="dC/dt = -0.5*dC/dx + 0.001*d2C/dx2 - 0.1*C",
    func="C",
    sp_var=["x"], ivar=["t"],
    ivar_boundary=[(0, 1)],
    expr_ic="0",
    west_bd="Dirichlet", west_func_bd="1",
    east_bd="Neumann",   east_func_bd="0",
)

pde_d = PDE(
    eq="dD/dt = -0.5*dD/dx + 0.001*d2D/dx2 + 0.1*C",
    func="D",
    sp_var=["x"], ivar=["t"],
    ivar_boundary=[(0, 1)],
    expr_ic="0",
    west_bd="Dirichlet", west_func_bd="0",
    east_bd="Neumann",   east_func_bd="0",
)

sistema = PDES(pdes=[pde_c, pde_d], disc_n=[50])
sistema.discretize(method='backward')
sistema.solve(method='bdf2', tf=5.0, nt=500)

# Visualize each function by its index
sistema.visualize(mode='plot1d_all', func_idx=0, tf=5.0)  # C
sistema.visualize(mode='plot1d_all', func_idx=1, tf=5.0)  # D
```

> **Important:** In coupled systems, function names used in one equation (e.g. `C` in the equation for `D`) must match the `func` attribute of the corresponding `PDE` object.

## PDE Solver Studio

An interactive desktop graphical application (WebView) is available to build, simulate, and analyze PDE systems visually.

Features:
- Visual Equation builder and system management
- Multi-dimensional visualization (1D plot profiles, 2D heatmaps, and 3D surface rendering)
- Light and Dark theme modes
- JSON configuration import/export to save and restore workspace states

For pre-compiled binaries and installation guides, visit the [Studio Releases Page](https://github.com/maiocacedo/pdesolver-studio/releases/).

---

## Citation

If you use pdesolver in your research, please cite it:

```bibtex
@software{pdesolver,
  author = {Macedo, Caio},
  title = {pdesolver: Symbolic PDE solver using finite difference discretization},
  url = {https://github.com/maiocacedo/PDESsolver},
  version = {0.1.9},
  year = {2026}
}
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to:
- Report bugs and request features
- Set up a development environment
- Submit pull requests

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).

## Support

- **Bug reports:** [GitHub Issues](https://github.com/maiocacedo/PDESsolver/issues)
- **Feature requests:** [GitHub Issues](https://github.com/maiocacedo/PDESsolver/issues)
- **Questions:** Open an issue with the `[QUESTION]` prefix

## License

MIT — see [LICENSE](https://github.com/maiocacedo/PDESsolver/blob/main/LICENSE) for details.

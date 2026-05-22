# pdesolver

**Symbolic solver for partial differential equations (PDEs) using finite difference discretization.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![PyPI](https://img.shields.io/pypi/v/pdesolver.svg)](https://pypi.org/project/pdesolver/)

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Equation Notation](#equation-notation)
- [The `PDE` Class](#the-pde-class)
- [The `PDES` Class (System)](#the-pdes-class-system)
- [Boundary Conditions](#boundary-conditions)
- [Spatial Discretization](#spatial-discretization)
- [Solvers (Time Integration)](#solvers-time-integration)
- [Visualization](#visualization)
- [Import & Export (JSON)](#import--export-json)
- [Coupled Systems](#coupled-systems)

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
```

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
west ←——————— domain ———————→ east
x = x_min                    x = x_max
```

For **2D** problems, all four sides are used:

```
              north (y = y_max)
         ┌─────────────────────┐
         │                     │
  west   │       domain        │  east
(x=x_min)│                     │(x=x_max)
         └─────────────────────┘
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

The library supports three types of boundary conditions, applied independently to each side of the domain.

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

---

## License

MIT — see [LICENSE](LICENSE) for details.

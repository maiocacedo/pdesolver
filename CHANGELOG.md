# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-08-01

### Added
- **Discretization Analysis (`pdesolver.Analysis`):** the symbolic representation of the equation is now used to derive what the discretization actually does. `PDES.analyze()`, `truncation_error()`, `modified_equation()` and `stability_limit()` report:
  - the truncation error and formal order of each stencil, computed from the stencil moments rather than tabulated, so it remains correct on stretched meshes;
  - the **numerical diffusion the scheme introduces, with its sign** — an upwind scheme reports how much diffusion it adds relative to the physical coefficient, a downwind scheme is flagged as anti-diffusive;
  - the **stability limit on `dt`**, obtained two independent ways that cross-check each other (Fourier symbol of the stencil combined with the stability polynomial of the Runge–Kutta tableau, and the eigenvalues of the assembled operator), agreeing to within 0.1% on the 2D heat equation; A-stable implicit methods correctly report no limit;
  - **growing modes** — if the spatial operator has an eigenvalue with positive real part, no time step is stable and the report says so;
  - the **cell Péclet number**, warning when central differencing will oscillate.

- **Periodic Boundary Conditions:** `west_bd='Periodic'` / `east_bd='Periodic'` (and `south`/`north`) for 1D and 2D. Periodicity is an axis property, so both sides of an axis must be declared together; a periodic axis places `disc_n` nodes over `[a, b)` without duplicating the endpoint.
- **Non-Uniform Meshes:** new `mesh=` parameter on `PDES` with `'uniform'`, `'chebyshev'`, `'tanh'`, `'tanh_left'`, `'tanh_right'` distributions, per-axis specification, and explicit node arrays. Stencils use variable-coefficient weights that reduce exactly to the classical formulas on a uniform grid.
- **Mixed Second Derivative:** `d2u/dxdy` (equivalently `d2u/dydx`) for anisotropic diffusion and rotated-coordinate problems.
- **Stencil Backend:** new `backend='stencil'` on `PDES`. Each PDE is parsed once into an expression over derivative fields and evaluated over whole arrays, making the symbolic cost independent of the mesh — 2646× faster setup at 60×60, and grids of 512×512 and beyond become tractable. Runs on NumPy or CuPy.
- **Analytic Sparsity for Implicit Solvers:** the stencil backend supplies the Jacobian sparsity pattern and a structural colouring, so the number of right-hand side evaluations needed to assemble the operator matrix no longer grows with the mesh (9 colours at any grid size, instead of one evaluation per column).
- **Shared Mesh Layer:** new `Grid`/`Axis` objects consumed by both backends, giving identical results across them (verified to machine precision by `Tests/test_backends.py`).
- **Heterogeneous Regions:** `discretize(regions=[...])` gives individual nodes their own equation, selected by a boolean mask or a callable of the grid coordinates. Covers internal obstacles and masked domains, per-region physics, point sources and material interfaces. Available on the `'symbolic'` backend only — the vectorized operator assumes one equation per node, which is what makes it vectorizable, and it raises a clear error pointing at the alternative. `examples/regioes_heterogeneas.py` verifies three cases against closed-form results.
- **Automatic IMEX splitting (`method='imex'`):** each additive term of the equation is classified by the order of the spatial derivative it carries and by whether it is linear in the unknowns. Terms of second order that are linear go to the implicit side, everything else stays explicit; a semi-implicit BDF2 (SBDF2) then advances the system, factorizing the implicit operator once. Second-order temporal convergence verified (observed orders 2.01, 2.01, 2.01) and the split is checked to reconstruct the original operator to 10⁻¹³. Requires `backend='stencil'`.

- **Fast Poisson solver for the implicit stage.** When the implicit part is a constant-coefficient Laplacian on a uniform 2D grid with Dirichlet boundaries, the discrete sine basis diagonalizes it exactly and the stage is solved by DST in `O(N log N)` rather than by sparse triangular solves. The structure is detected automatically, with a sparse LU fallback for stretched meshes, variable coefficients or mixed terms. Verified to reproduce the LU solve to ≈10⁻¹³ and 1.7× to **7.3×** faster per solve (41² to 321²), plus removal of the factorization itself (0.32 s at 321²). End to end on a 321² advection–diffusion–reaction problem, IMEX takes 2.75 s against 12.40 s for the explicit solver — **4.5×**, growing with the mesh. Without this stage the net gain was only about 1.3×, since the sparse solve dominated.

  The largest gain is against the **implicit** solvers, not the explicit one. On a non-linear problem with Dirichlet boundaries, at matched step count and comparable accuracy, IMEX runs in 0.88 s at 128² and 0.87 s at 256², against 25.5 s / 67.3 s for BDF2 and 22.7 s / 73.6 s for Crank–Nicolson — **26× to 85×**. BDF2 and CN enter Newton at every step (colored Jacobian, assembly, sparse LU) while IMEX moves the non-linearity to the explicit side and removes Newton entirely, leaving one DST per step; its cost is then almost independent of the mesh.
- **Benchmarks:** `benchmarks/bench_scaling.py` reproduces the scaling figures reported in the paper.
- The RKF45 Butcher tableau is exposed as module-level constants (`RKF45_A`, `RKF45_B5`, `RKF45_B4`, `RKF45_C`) so the stability analysis derives its stability polynomial from the coefficients the solver actually uses.

### Fixed
- **Non-homogeneous Dirichlet boundaries carried an O(dt) error in the implicit solvers.** The Dirichlet rows entered the implicit system as `u_new[p] = u_old[p] + α·dt·g` instead of `u_new[p] = g`; applying the boundary condition after the solve corrected node `p` itself but not its interior neighbours, which had already been solved coupled to the wrong value. The error was exactly `α·dt·g` propagated inward — verified as `erro/dt = 0.6333` (BDF2, α=2/3) and `0.475` (CN, α=1/2), constant across four step sizes. The boundary values are now imposed in the right-hand side of the linear system and in the Newton residual. On a steady problem with `u(0)=1, u(1)=0` the error drops from `1.3×10⁻²` to `1.5×10⁻¹⁵`.

  This went unnoticed because **every existing test used homogeneous Dirichlet conditions** (`g = 0`), for which the expression is accidentally correct. Problems with non-zero or time-dependent Dirichlet data were affected, including the published benchmark cases that use them.
- **Domain bounds were ignored.** The discretization hard-coded `h = 1/(N-1)`, so every domain other than `[0, 1]` produced silently wrong results (e.g. `ivar_boundary=[(0, 2)]` with `N=5` used `h = 0.25` instead of `0.5`). `ivar_boundary` is now honoured on every axis.
- **`hy` was emitted as `hx`.** Both spatial axes shared a single spacing token, so any non-square grid such as `disc_n=[30, 60]` was silently wrong. Each axis now carries its own spacing.
- Visualization now plots against the actual grid coordinates rather than assuming `[0, 1]`.

### Changed
- `PDES.__init__` accepts `mesh` and `backend`; existing positional usage is unchanged.
- The stale `boundaries/periodic_example.py` stub was replaced by a registered `PeriodicBC`.
- **GPU switch threshold raised from 10 000 to 65 536 unknowns.** Measured on an RTX 4070 Ti, the GPU is ~5× *slower* than the CPU at 16 384 unknowns because its runtime is dominated by a fixed ~0.45 ms kernel launch overhead; the break-even point is near 65 000. Below the threshold the solver now correctly stays on the CPU.
- `mesh` and `backend` are persisted by `save_to_json` and restored by `load_from_json`, so a stretched grid survives a round-trip. Files written by earlier versions still load, defaulting to a uniform mesh and the symbolic backend.

### Fixed (GPU)
- On Windows, CUDA libraries installed as NVIDIA pip wheels (`nvidia-cuda-nvrtc-cu12` and friends) were not discoverable: `os.add_dll_directory` alone is insufficient because NVRTC loads `nvrtc-builtins` through the plain Windows loader, which searches `PATH`. Their directories are now added to both on import, so `pip install pdesolver[gpu]` works without manual environment setup.

## [0.1.9] - 2026-07-31

### Added
- **Numerical Time Solvers:** BDF2 (Backward Differentiation Formula 2nd order), Crank-Nicolson (CN), and Runge-Kutta-Fehlberg (RKF45 / RKF) adaptive time steppers.
- **Spatial Discretization:** High-order finite difference spatial discretization for both 1D and 2D spatial domains.
- **Boundary Conditions:** Full support for Dirichlet, Neumann, and Robin boundary conditions.
- **Coupled Systems:** Multivariable coupled partial differential equation system support.
- **Automatic Linearity Detection:** Intelligent automatic detection of system linearity for selecting optimal linear vs non-linear solver pathways.
- **Serialization & Data Exchange:** JSON import/export functionality for problem definitions and execution parameters.
- **Visualization:** Integrated visualization modes (surface, heatmap, and 1D animation/plots).
- **GPU Acceleration:** Optional GPU backend support via CuPy for high-performance array computations.
- **Packaging:** PyPI publication release configuration (`pdesolver`).

## [0.1.0] - 2026-04-15

### Added
- Initial release of PDESsolver library for numerical PDE solving.

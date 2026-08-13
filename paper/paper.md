---
title: 'pdesolver: A Symbolic PDE Solver Using Finite Difference Discretization in Python'
tags:
  - Python
  - partial differential equations
  - finite differences
  - numerical methods
  - scientific computing
authors:
  - name: Caio Macedo
    orcid: 0009-0005-6840-5148
    affiliation: 1
affiliations:
  - name: Universidade Tecnológica Federal do Paraná (UTFPR)
    index: 1
date: 31 July 2026
bibliography: paper.bib
---

# Summary

`pdesolver` is a Python library for solving partial differential equations (PDEs)
using finite difference discretization. It allows users to define PDEs using an
intuitive string notation (e.g., `du/dt = d2u/dx2`), automatically discretizes
spatial derivatives using central, forward, or backward finite difference schemes,
and integrates in time using implicit (BDF2, Crank-Nicolson) or explicit adaptive
(Runge-Kutta-Fehlberg 4/5) solvers. The library supports 1D and 2D domains,
Dirichlet, Neumann, Robin, and periodic boundary conditions, uniform and
stretched (non-uniform) meshes, first, second, and mixed second spatial
derivatives, coupled multi-equation systems, and GPU acceleration via CuPy.

Because the equation is retained in symbolic form rather than compiled away,
the library can analyse its own discretization. From the stencil moments it
derives the truncation error and formal order of each approximation, and from
the coefficients of the user's equation it reports the numerical diffusion the
scheme introduces, together with its sign: an upwind scheme is quantified
against the physical diffusion coefficient, while a downwind scheme is
identified as anti-diffusive. Combining the Fourier symbol of the stencil with
the stability polynomial of the time integrator's Butcher tableau yields the
stability limit on the time step for the equation as written, rather than for a
tabulated model problem, and a positive growth rate in the spatial operator is
reported as such instead of as a step size. These diagnostics are available
before any time step is taken.

The discretization is available through two interchangeable backends that agree
to machine precision. The default backend emits one equation per grid node and
compiles them symbolically, which keeps the generated system inspectable and is
convenient for teaching and for small problems. A vectorized backend instead
parses each PDE once into an expression over derivative fields and evaluates it
over whole arrays, so its symbolic cost is independent of the mesh size. This
removes the setup bottleneck that otherwise limits node-wise symbolic
discretization to small grids, and makes large two-dimensional problems and GPU
execution practical.

# Statement of Need

<!-- TODO: Author must write this section. Key points to cover:
- Who is the target audience? (researchers, students, educators in computational science)
- What problem does this solve?
- What gap does it fill between symbolic math (SymPy) and heavy FEM frameworks?
-->

# State of the Field

<!-- TODO: Author must write this section. Compare with:
- FEniCS [@fenics] - full FEM framework, steeper learning curve
- py-pde [@py-pde] - similar scope but different design philosophy
- FiPy [@fipy] - finite volume method
- Devito [@devito] - DSL for stencil computations
- Explain why pdesolver exists as a distinct contribution
-->

# Software Design

<!-- TODO: Author must write this section. Cover:
- Architecture: PDE -> PDES -> Disc -> Solvers -> Visualize pipeline
- Trade-offs: symbolic parsing vs. DSL, sparse matrices, auto linearity detection
- Why string-based equation input? (accessibility for non-programmers)
- GPU vs. CPU solver strategy
-->

# Verification

Correctness is checked against closed-form solutions wherever one exists: the
1D and 2D heat equations, advection–diffusion, Burgers' equation, periodic
advection over a full domain traversal, and anisotropic diffusion with a mixed
derivative, for which a plane wave is an exact eigenfunction of the continuous
operator. Convergence tests confirm the expected second-order spatial accuracy,
including the mixed-derivative stencil and the variable-coefficient stencils on
stretched meshes.

Because the two discretization backends are independent implementations of the
same mathematics, they also verify each other. The test suite compares their
right-hand sides across boundary condition types, mesh distributions, coupled
systems, and all three finite difference schemes, and requires agreement to
machine precision.

The analysis layer is verified against closed-form results. The derived leading
truncation terms reproduce the classical values — `h²/6·u'''` for the centred
first derivative, `∓h/2·u''` for the one-sided ones, `h²/12·u''''` for the
centred second derivative — and the stability polynomial built from the Butcher
tableau reproduces the exponential to fifth order, as it must for a fifth-order
method. The two independent routes to the stability limit, the Fourier symbol
and the spectrum of the assembled operator, agree to within 0.1% on the 2D heat
equation.

One property is worth recording because it is easy to get wrong. The
variable-coefficient weights used here are the exact three-point interpolatory
weights, for which the `u''` moment cancels identically: the first derivative
therefore remains second order on an arbitrary non-uniform mesh, with leading
term `h⁻h⁺u'''/6`. The naive centred formula `(u₊ − u₋)/(h⁻ + h⁺)` loses this
cancellation and degrades to first order as soon as the spacing varies. The
analysis layer confirms this directly on the mesh in use.

# Performance

The node-wise symbolic backend re-derives the same stencil at every grid point,
so its setup cost grows with the mesh even though the PDE does not change. The
vectorized backend performs its symbolic work once. Setup times for the 2D heat
equation, measured to the point where the system is ready to integrate:

| Grid | Unknowns | Node-wise symbolic | Vectorized | Speedup |
|:-----|---------:|-------------------:|-----------:|--------:|
| 15²  | 225      | 0.61 s             | 0.022 s    | 28×     |
| 30²  | 900      | 5.06 s             | 0.006 s    | 877×    |
| 60²  | 3 600    | 25.6 s             | 0.010 s    | 2 646×  |
| 128² | 16 384   | not feasible       | 0.014 s    | —       |
| 256² | 65 536   | not feasible       | 0.025 s    | —       |
| 512² | 262 144  | not feasible       | 0.034 s    | —       |

The vectorized backend also yields the Jacobian sparsity pattern and a
structural colouring analytically, so assembling the operator matrix for the
implicit solvers requires a number of right-hand side evaluations that does not
depend on the mesh — nine, whether the grid holds 225 or 65 536 unknowns —
rather than one evaluation per column.

Because the vectorized backend is neutral with respect to the array module, the
same operator executes under NumPy or CuPy without modification. On an RTX 4070
Ti, right-hand side evaluation for the 2D heat equation is 13.5 times faster on
the GPU at 512² and 17 to 24 times faster at 1024², while below roughly 65 000
unknowns the GPU is slower because its runtime is dominated by a fixed kernel
launch overhead of about 0.45 ms. The solver therefore switches devices only
above that size. A complete adaptive Runge–Kutta–Fehlberg solve on a 512² grid
takes 9.2 s on the CPU and 0.69 s on the GPU, and the two produce bitwise
identical results. `benchmarks/bench_scaling.py` reproduces these measurements.

# Limitations

The library targets structured Cartesian grids in one and two spatial
dimensions; unstructured geometries and three-dimensional domains are out of
scope, and problems on complex geometry are better served by a finite element
framework. Spatial discretization is second order. Time integration uses a
fixed step for the implicit solvers and adaptive stepping only in the explicit
Runge–Kutta–Fehlberg solver, so non-uniform refinement is available in space but
not, for the implicit methods, in time. As an explicit method, the adaptive
solver remains stability limited on diffusion-dominated problems, where the
accepted step size scales with the square of the mesh spacing. Non-uniform node
distributions on a periodic axis are restricted to uniform spacing or
explicitly supplied nodes.

The analysis layer inherits the assumptions of the theory it applies. The
stability limit is a linear result: nonlinear terms are excluded from the
symbol and the reported bound applies to the linearized operator, which the
report states explicitly when it applies. The bound is also a worst case over
all Fourier modes, so a computation started from smooth data may legitimately
exceed it while the highest modes carry no energy; it is a design bound rather
than a prediction of the step an adaptive controller will select. Variable
coefficients are evaluated at the most restrictive cell rather than treated as
frozen coefficients throughout.

# Research Impact Statement

<!-- TODO: Author must write this section. Include:
- Reference to the companion article (caio_conexoes_final)
- Any other research use
- The benchmark numbers above come from benchmarks/bench_scaling.py
-->

# AI Usage Disclosure

Generative AI tools were used to assist in the development of this software,
specifically to follow expected coding standards and conventions, and to optimize
the implemented algorithms. Additionally, AI assistants were used extensively 
in a pair-programming approach to help build the companion PDE Solver Studio. 
All AI-generated outputs were reviewed and validated by the author.

# Acknowledgements

<!-- TODO: Author must write this section -->

# References

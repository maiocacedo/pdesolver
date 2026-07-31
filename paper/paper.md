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
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
  - name: TODO_AFFILIATION
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
Dirichlet, Neumann, and Robin boundary conditions, coupled multi-equation systems,
and GPU acceleration via CuPy.

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

# Research Impact Statement

<!-- TODO: Author must write this section. Include:
- Reference to the companion article (caio_conexoes_final)
- Any other research use
- Benchmarks or reproducible examples
-->

# AI Usage Disclosure

<!-- TODO: Author must complete this section honestly. Template:
"[Generative AI tools were / were not] used in the development of this software.
[If used: Tool X was used for Y. All outputs were reviewed and validated by the authors.]"
-->

# Acknowledgements

<!-- TODO: Author must write this section -->

# References

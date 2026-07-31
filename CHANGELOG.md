# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

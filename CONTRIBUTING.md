# Contributing to PDESsolver

Thank you for your interest in contributing to PDESsolver! We welcome contributions from the community, whether reporting bugs, suggesting new features, improving documentation, or submitting pull requests.

## Table of Contents
- [Development Setup](#development-setup)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Features](#suggesting-features)
- [Submitting Pull Requests](#submitting-pull-requests)
- [Code Style](#code-style)
- [Code of Conduct](#code-of-conduct)
- [Getting Help](#getting-help)

---

## Development Setup

To set up a local development environment:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/maiocacedo/PDESsolver.git
   cd PDESsolver
   ```

2. **Create and activate a virtual environment:**
   - **Linux / macOS:**
     ```bash
     python -m venv .venv
     source .venv/bin/activate
     ```
   - **Windows:**
     ```cmd
     python -m venv .venv
     .venv\Scripts\activate
     ```

3. **Install the package in editable mode with development dependencies:**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Run the test suite to verify the setup:**
   ```bash
   pytest Tests/ -v
   ```

---

## Reporting Bugs

If you encounter a bug or unexpected behavior:

1. Check existing [GitHub Issues](https://github.com/maiocacedo/PDESsolver/issues) to ensure it hasn't already been reported.
2. Open a new issue with a clear summary title.
3. Include the following details in your report:
   - **Environment details:** Python version, PDESsolver version, OS, and relevant dependency versions (e.g., NumPy, SciPy, CuPy if using GPU).
   - **Minimal Reproducible Example (MRE):** A self-contained Python snippet demonstrating the bug.
   - **Expected vs Actual Behavior:** Clear description of what should happen versus what actually occurred (including full tracebacks if applicable).

---

## Suggesting Features

We welcome feature requests and enhancements! To suggest a feature:

1. Check open issues to avoid duplicates.
2. Open a new issue titled with the `[FEATURE]` prefix (e.g., `[FEATURE] Add 3D Spatial Discretization Support`).
3. Describe the proposed feature, the problem it solves, potential API design, and any alternative solutions considered.

---

## Submitting Pull Requests

We appreciate PRs! Follow these steps to submit a pull request:

1. **Fork the repository** on GitHub.
2. **Create a feature branch** off `main`:
   ```bash
   git checkout -b feature/my-new-feature
   ```
3. **Write tests** for any new functionality or bug fixes.
4. **Ensure code style adherence** (PEP 8, type hints, NumPy docstrings).
5. **Run all tests** to ensure no regressions:
   ```bash
   pytest Tests/ -v
   ```
6. **Commit changes** using [Conventional Commits](https://www.conventionalcommits.org/) (e.g., `feat: add new solver`, `fix: correct boundary condition calculation`, `docs: update setup instructions`).
7. **Push to your fork** and open a Pull Request against the `main` branch of `maiocacedo/PDESsolver`.

---

## Code Style

To maintain high code quality and consistency:

- **PEP 8:** Follow standard Python PEP 8 formatting guidelines.
- **Naming Conventions:** Use clear English variable and function names (`snake_case` for variables/functions, `PascalCase` for classes).
- **Docstrings:** Use [NumPy-style docstrings](https://numpydoc.readthedocs.io/en/latest/format.html) for all public classes, methods, and functions.
- **Type Hints:** Provide Python type annotations for function signatures and public APIs.

---

## Code of Conduct

This project enforces a Code of Conduct to ensure a welcoming environment for everyone. Please review and abide by our [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) in all interactions.

---

## Getting Help

If you have questions about using PDESsolver or need assistance:

- Search existing issues and discussions.
- Open a new issue with the `[QUESTION]` prefix (e.g., `[QUESTION] How to configure Robin BCs for coupled system?`).

"""Tests for JSON save/load round-trip functionality."""

import os
import tempfile

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from pdesolver import PDE, PDES


@pytest.fixture
def sample_system():
    """Create a simple 2D heat equation system for testing."""
    pde = PDE(
        eq="dF/dt = 0.1*d2F/dx2 + 0.2*d2F/dy2",
        func="F",
        sp_var=["x", "y"],
        ivar=["t"],
        ivar_boundary=[(0, 1), (0, 1)],
        expr_ic="sin(pi * x) * sin(pi * y)",
        west_bd="Dirichlet", west_func_bd="0",
        east_bd="Dirichlet", east_func_bd="0",
        north_bd="Dirichlet", north_func_bd="0",
        south_bd="Dirichlet", south_func_bd="0",
    )
    sistema = PDES([pde], [10, 10])
    sistema.discretize(method="central")
    sistema.solve(method="bdf2", tf=0.1, nt=10)
    return sistema


def test_save_load_roundtrip(sample_system, tmp_path):
    """Test that saving and loading preserves system state."""
    filepath = str(tmp_path / "test_output.json")
    sample_system.save_to_json(filepath)

    loaded = PDES.load_from_json(filepath)

    assert loaded.funcs == sample_system.funcs
    assert loaded.disc_n == sample_system.disc_n
    assert loaded.pdes[0].eq == sample_system.pdes[0].eq
    assert loaded.pdes[0].west_bd == sample_system.pdes[0].west_bd
    assert loaded.results is not None


def test_load_results_shape(sample_system, tmp_path):
    """Test that loaded results have the correct shape."""
    filepath = str(tmp_path / "test_output.json")
    sample_system.save_to_json(filepath)

    loaded = PDES.load_from_json(filepath)
    assert len(loaded.results) == len(sample_system.results)


def test_load_can_rediscretize(sample_system, tmp_path):
    """Test that a loaded system can be re-discretized and re-solved."""
    filepath = str(tmp_path / "test_output.json")
    sample_system.save_to_json(filepath)

    loaded = PDES.load_from_json(filepath)
    loaded.disc_n = [5, 5]
    loaded.discretize(method="central")
    loaded.solve(method="bdf2", tf=0.1, nt=10)
    assert loaded.results is not None


def test_roundtrip_preserva_malha_e_backend(tmp_path):
    pde = PDE(
        eq="dF/dt = 0.1*d2F/dx2 + 0.2*d2F/dy2",
        func="F",
        sp_var=["x", "y"],
        ivar=["t"],
        ivar_boundary=[(0, 2), (0, 3)],
        expr_ic="sin(pi*x/2)*sin(pi*y/3)",
        west_bd="Dirichlet",  west_func_bd="0",
        east_bd="Dirichlet",  east_func_bd="0",
        north_bd="Dirichlet", north_func_bd="0",
        south_bd="Dirichlet", south_func_bd="0",
    )
    sistema = PDES([pde], [9, 11], mesh="chebyshev", backend="stencil")
    sistema.discretize(method="central")
    sistema.solve(method="bdf2", tf=0.1, nt=10)

    caminho = tmp_path / "malha.json"
    sistema.save_to_json(str(caminho))
    carregado = PDES.load_from_json(str(caminho))

    assert carregado.mesh == "chebyshev"
    assert carregado.backend == "stencil"
    assert np.allclose(
        carregado.grid.axes[0].nodes, sistema.grid.axes[0].nodes
    )
    assert np.allclose(
        carregado.grid.axes[1].nodes, sistema.grid.axes[1].nodes
    )

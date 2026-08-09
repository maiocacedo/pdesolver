import json

import matplotlib
if matplotlib.get_backend().lower() == "agg":
    try:
        matplotlib.use("TkAgg")
    except ImportError:
        pass  # TkAgg not available, keep Agg backend
import numpy as np
import sympy as sp

from .Solvers import RKF as SERKF45
from .Auxs.PDESEncoder import PDESEncoder
from .Auxs.Visualize import visualize as _visualize
from .Disc.Disc import df, periodic_axes
from .Disc.grid import build_grid
from .Disc.stencil import StencilOperator
from .PDE import PDE
from .Solvers.bdf2 import bdf2
from .Solvers.CN import cn
from .Solvers.imex import imex
from .Solvers.solver_base import detect_linearity_symbolic
from .Analysis import analyze, modified_equation, operator_terms, report_text
from .Analysis import stability_limit


class PDES:
    """System of coupled partial differential equations.

    Manages one or more :class:`PDE` objects, orchestrating spatial
    discretization, time integration, and visualization.

    Parameters
    ----------
    pdes : list of PDE
        List of PDE objects defining the system.
    disc_n : list of int
        Number of grid points per spatial dimension.

    Examples
    --------
    >>> sistema = PDES(pdes=[pde], disc_n=[50])
    >>> sistema.discretize(method='central')
    >>> sistema.solve(method='bdf2', tf=0.1, nt=100)
    """
    @property
    def disc_n(self):
        return self._disc_n

    @disc_n.setter
    def disc_n(self, value):
        self._disc_n = value
        self.grid = self._grid_calc()
        self.ic = self._ic_calc(self.pdes)
        self.disc_results = None
        self.dirichlet_constraints = {}
        self.neumann_constraints = {}

    def __init__(self, pdes, disc_n, mesh="uniform", backend="symbolic",
                 n_sp=1, n_temp=1):
        self.pdes = pdes
        self.eqs = [pde.eq for pde in pdes]
        self.ivars = pdes[0].ivar
        self.disc_results = None
        self.dirichlet_constraints = {}
        self.neumann_constraints = {}

        self.funcs = [pde.func for pde in pdes]
        self.sp_vars = pdes[0].sp_var
        self.mesh = mesh
        self.backend = backend
        self._disc_n = disc_n
        self.grid = self._grid_calc()
        self.ic = self._ic_calc(pdes)
        self.operator = None
        self.results = None

    def _grid_calc(self):
        return build_grid(
            self.pdes[0].ivar_boundary,
            self.disc_n,
            mesh=self.mesh,
            periodic=periodic_axes(self),
        )

    def _ic_calc(self, pdes):
        all_ics = []
        mesh = self.grid.coords()
        for pde in pdes:
            expr = sp.parse_expr(pde.expr_ic)
            sp_symbols = [sp.Symbol(v) for v in pde.sp_var]
            f_ic = sp.lambdify(sp_symbols, expr, modules="numpy")
            ic_values = f_ic(*mesh)
            if np.isscalar(ic_values):
                ic_values = np.broadcast_to(ic_values, mesh[0].shape)
            all_ics.extend(np.asarray(ic_values).flatten().tolist())
        return all_ics

    def xs(self, vars):
        nvars = vars.copy()
        for i in range(len(nvars)):
            nvars[i] = f"XX{i}"
        return nvars

    def discretize(self, method="central", regions=None):
        """Discretize spatial derivatives using finite differences.

        Parameters
        ----------
        method : str, optional
            Finite difference scheme: ``'central'``, ``'forward'``, or
            ``'backward'``. Default is ``'central'``.
        regions : list of dict, optional
            Per-region equation overrides, only available on the
            ``'symbolic'`` backend. Each entry needs ``'where'`` — a boolean
            mask over the grid, or a callable of the grid coordinates — and
            ``'eq'``, an equation in the usual notation that replaces the
            system equation at those nodes. Boundary nodes keep their boundary
            condition; regions apply to interior nodes.

        Examples
        --------
        >>> X, Y = sistema.grid.coords()
        >>> bloco = (abs(X - 0.5) < 0.1) & (abs(Y - 0.5) < 0.1)
        >>> sistema.discretize(regions=[{'where': bloco, 'eq': 'dU/dt = 0'}])
        """
        if self.backend == "stencil":
            if regions:
                raise ValueError(
                    "Regiões com equação própria exigem backend='symbolic': "
                    "o operador vetorizado pressupõe a mesma equação em todos "
                    "os nós, que é o que permite vetorizar."
                )
            self.operator = StencilOperator(self, self.grid, method=method)
            self.disc_results = None
            self.dirichlet_constraints = self.operator.dirichlet_constraints
            self.neumann_constraints = self.operator.neumann_constraints
            return

        if self.backend != "symbolic":
            raise ValueError(
                f"Backend inválido: '{self.backend}'. Use 'symbolic' ou "
                f"'stencil'."
            )

        flat_list, d_vars, dirichlet_constraints, neumann_constraints = df(
            self,
            self.disc_n,
            method=method,
            west_bd=[pde.west_bd for pde in self.pdes],
            west_func_bd=[pde.west_func_bd for pde in self.pdes],
            east_bd=[pde.east_bd for pde in self.pdes],
            east_func_bd=[pde.east_func_bd for pde in self.pdes],
            north_bd=[pde.north_bd for pde in self.pdes],
            north_func_bd=[pde.north_func_bd for pde in self.pdes],
            south_bd=[pde.south_bd for pde in self.pdes],
            south_func_bd=[pde.south_func_bd for pde in self.pdes],
            grid=self.grid,
            regions=regions,
        )
        self.disc_results = (flat_list, d_vars)
        self.dirichlet_constraints = dirichlet_constraints
        self.neumann_constraints = neumann_constraints

    def solve(self, method="bdf2", tf=1.0, nt=100, tol=1e-6, verbose=False, **kwargs):
        """Integrate the discretized system in time.

        Parameters
        ----------
        method : str, optional
            Time integration method: ``'bdf2'``, ``'CN'``, or ``'RKF'``.
        tf : float, optional
            Final time.
        nt : int, optional
            Number of time steps.
        tol : float, optional
            Tolerance for adaptive methods (RKF).
        verbose : bool, optional
            Print solver progress.

        Returns
        -------
        list of numpy.ndarray
            Solution arrays, one per PDE function.

        Raises
        ------
        ValueError
            If ``method`` is not one of the supported methods.
        """
        dt = tf / nt
        dc = self.dirichlet_constraints
        nc = getattr(self, "neumann_constraints", {})

        if self.disc_results is None and self.operator is None:
            raise RuntimeError("Rode .discretize() antes de .solve().")

        flat = self.disc_results[0] if self.disc_results is not None else None
        dvars = self.disc_results[1] if self.disc_results is not None else None
        op = self.operator

        is_linear_sym = None
        if method in ("bdf2", "CN") and "is_linear" not in kwargs:
            is_linear_sym = detect_linearity_symbolic(
                self.eqs, self.funcs, self.sp_vars, verbose=verbose
            )

        if method.lower() == "bdf2":
            self.results = bdf2(
                flat,
                dvars,
                tf=tf,
                nt=nt,
                ic=self.ic,
                n_funcs=len(self.funcs),
                dirichlet_constraints=dc,
                neumann_constraints=nc,
                verbose=verbose,
                is_linear=is_linear_sym,
                operator=op,
                **kwargs,
            )
        elif method.lower() == "cn":
            self.results = cn(
                flat,
                dvars,
                tf=tf,
                nt=nt,
                ic=self.ic,
                n_funcs=len(self.funcs),
                dirichlet_constraints=dc,
                neumann_constraints=nc,
                verbose=verbose,
                is_linear=is_linear_sym,
                operator=op,
                **kwargs,
            )
        elif method.lower() == "imex":
            if op is None:
                raise ValueError(
                    "O método 'imex' exige backend='stencil': a separação "
                    "simbólica entre termos rígidos e brandos é feita sobre "
                    "os campos de derivada do operador vetorizado."
                )
            self.results = imex(
                op,
                tf=tf,
                nt=nt,
                ic=self.ic,
                n_funcs=len(self.funcs),
                dirichlet_constraints=dc,
                neumann_constraints=nc,
                verbose=verbose,
                **kwargs,
            )
        elif method.lower() == "rkf":
            if op is not None:
                op = op.to_device(SERKF45.select_array_module(op.size))
            self.results = SERKF45.SERKF45_cuda(
                flat,
                ivar=self.ivars,
                funcs=dvars,
                yn=self.ic,
                sp_vars=self.sp_vars,
                n=100,
                n_funcs=len(self.funcs),
                dt_init=dt,
                tol=tol,
                x0=0,
                xn=nt * dt,
                dirichlet_constraints=dc,
                neumann_constraints=nc,
                verbose=verbose,
                operator=op,
                **kwargs,
            )
        else:
            raise ValueError(
                f"Unknown method '{method}'. Use: 'bdf2', 'CN', 'RKF' "
                f"or 'imex'."
            )
        return self.results

    def _analysis_operator(self, method):
        if self.operator is not None:
            return self.operator
        return StencilOperator(self, self.grid, method=method)

    def analyze(self, method="RKF", disc_method="central", verbose=True):
        """Analyse the discretization symbolically.

        Derives the truncation error of each finite difference stencil, the
        numerical diffusion the scheme introduces, the stability limit on the
        time step, and the cell Péclet number.

        Parameters
        ----------
        method : str, optional
            Time integrator the stability limit refers to: ``'RKF'``,
            ``'bdf2'`` or ``'CN'``.
        disc_method : str, optional
            Finite difference scheme, used only when the system has not been
            discretized yet.
        verbose : bool, optional
            Print a readable report in addition to returning the data.

        Returns
        -------
        dict
            Keys ``'terms'``, ``'modified'``, ``'numerical_diffusion'``,
            ``'stability'`` and ``'peclet'``.
        """
        op = self._analysis_operator(disc_method)
        if verbose:
            print(report_text(op, method=method))
        return analyze(op, method=method)

    def stability_limit(self, method="RKF", disc_method="central"):
        """Return the largest stable time step for the chosen integrator.

        Parameters
        ----------
        method : str, optional
            ``'RKF'``, ``'bdf2'`` or ``'CN'``. Implicit methods are A-stable
            and report ``inf``.
        disc_method : str, optional
            Finite difference scheme, used only when the system has not been
            discretized yet.

        Returns
        -------
        float
            Maximum stable ``dt``.
        """
        op = self._analysis_operator(disc_method)
        return stability_limit(op, method=method)["dt_max"]

    def truncation_error(self, disc_method="central"):
        """Return the truncation error of each discretized derivative.

        Parameters
        ----------
        disc_method : str, optional
            Finite difference scheme, used only when the system has not been
            discretized yet.

        Returns
        -------
        list of dict
            One entry per discretized derivative, with the symbolic leading
            term and the largest coefficient realised on the current mesh.
        """
        return operator_terms(self._analysis_operator(disc_method))

    def modified_equation(self, disc_method="central"):
        """Return the extra terms the discretization adds to the PDE.

        Parameters
        ----------
        disc_method : str, optional
            Finite difference scheme, used only when the system has not been
            discretized yet.

        Returns
        -------
        list of dict
            Terms absent from the continuous equation but present in the
            equation the scheme actually solves.
        """
        return modified_equation(self._analysis_operator(disc_method))

    def visualize(self, mode="heatmap", func_idx=0, time_step=-1, **kwargs):
        """Visualize the solution.

        Parameters
        ----------
        mode : str, optional
            Visualization mode: ``'heatmap'``, ``'surface'``, ``'profile'``,
            or ``'animation'``.
        func_idx : int, optional
            Index of the PDE function to visualize.
        time_step : int, optional
            Time step to visualize. ``-1`` for the last step.
        """
        _visualize(self, mode=mode, func_idx=func_idx, time_step=time_step, **kwargs)

    def __repr__(self):
        status = "solved" if self.results is not None else "not solved"
        return (
            f"PDES(funcs={self.funcs}, disc_n={self.disc_n}, "
            f"sp_vars={self.sp_vars}, status='{status}')"
        )

    def save_to_json(self, filepath="pdes1.json"):
        """Save the system state to a JSON file.

        Parameters
        ----------
        filepath : str, optional
            Path to the output JSON file.
        """
        data = {
            "disc_n": list(self.disc_n),
            "mesh": self.mesh,
            "backend": self.backend,
            "pdes": [pde.__dict__ for pde in self.pdes],
            # 'ic' não é salvo — é derivado de expr_ic e disc_n
            "results": self.results,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, cls=PDESEncoder, indent=4, ensure_ascii=False)
        print(f"System saved successfully to: {filepath}")

    @classmethod
    def load_from_json(cls, filepath, pde_class=PDE):
        """Load a system from a previously saved JSON file.

        Parameters
        ----------
        filepath : str
            Path to the JSON file.
        pde_class : type, optional
            PDE class to use for reconstruction.

        Returns
        -------
        PDES
            Reconstructed system.
        """
        import inspect

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        init_params = set(inspect.signature(pde_class.__init__).parameters) - {"self"}

        reconstructed_pdes = []
        for pde_dict in data["pdes"]:
            kwargs = {k: v for k, v in pde_dict.items() if k in init_params}
            reconstructed_pdes.append(pde_class(**kwargs))

        obj = cls(
            pdes=reconstructed_pdes,
            disc_n=data["disc_n"],
            mesh=data.get("mesh", "uniform"),
            backend=data.get("backend", "symbolic"),
        )
        obj.disc_results = None
        obj.dirichlet_constraints = {}
        obj.neumann_constraints = {}

        if data.get("results") is not None:
            raw = data["results"]
            try:
                obj.results = np.array(raw)
            except ValueError:
                obj.results = np.array([np.array(r) for r in raw], dtype=object)

        return obj

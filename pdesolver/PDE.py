"""PDE data model for defining partial differential equations."""


class PDE:
    """Represents a single partial differential equation with boundary conditions.

    The equation is specified as a string in human-readable notation
    (e.g., ``"du/dt = d2u/dx2"``). The left side must always be the
    time derivative.

    Parameters
    ----------
    eq : str
        The PDE equation string. Left side must be the time derivative.
    func : str
        Name of the unknown function (e.g., ``"u"``, ``"T"``).
    sp_var : list of str
        Spatial variable names (e.g., ``["x"]`` or ``["x", "y"]``).
    ivar : list of str
        Independent variable for time integration (e.g., ``["t"]``).
    ivar_boundary : list of tuple
        Domain boundaries per spatial variable (e.g., ``[(0, 1)]``).
    expr_ic : str
        Initial condition as a SymPy expression string.
    west_bd, east_bd, north_bd, south_bd : str
        Boundary condition type: ``"Dirichlet"``, ``"Neumann"``, or ``"Robin"``.
    west_func_bd, east_func_bd, north_func_bd, south_func_bd : str
        Boundary value/expression for the corresponding side.

    Examples
    --------
    >>> pde = PDE(
    ...     eq="du/dt = d2u/dx2",
    ...     func="u",
    ...     sp_var=["x"], ivar=["t"],
    ...     ivar_boundary=[(0, 1)],
    ...     expr_ic="sin(pi*x)",
    ... )
    """

    def __init__(
        self,
        eq: str,
        func: str,
        sp_var: list,
        ivar: list,
        ivar_boundary: list,
        expr_ic: str,
        west_bd: str = "Dirichlet",
        west_func_bd: str = "0",
        east_bd: str = "Dirichlet",
        east_func_bd: str = "0",
        north_bd: str = "Dirichlet",
        north_func_bd: str = "0",
        south_bd: str = "Dirichlet",
        south_func_bd: str = "0",
    ):
        self.eq = eq
        self.func = func
        self.expr_ic = expr_ic
        self.sp_var = sp_var
        self.ivar = ivar
        self.ivar_boundary = ivar_boundary
        self.west_bd = west_bd
        self.west_func_bd = west_func_bd
        self.east_bd = east_bd
        self.east_func_bd = east_func_bd
        self.north_bd = north_bd
        self.north_func_bd = north_func_bd
        self.south_bd = south_bd
        self.south_func_bd = south_func_bd

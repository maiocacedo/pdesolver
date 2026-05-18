class PDE:

    eq = ''
    func = ''
    expr_ic = ''
    sp_var = []
    ivar = []
    ivar_boundary = []
    ic = []

    def __init__(self, eq, func, sp_var, ivar, ivar_boundary, expr_ic,
                 west_bd="Dirichlet", west_func_bd="0",
                 east_bd="Dirichlet", east_func_bd="0",
                 north_bd="Dirichlet", north_func_bd="0",
                 south_bd="Dirichlet", south_func_bd="0"):
        self.eq = eq
        self.func = func
        self.expr_ic = expr_ic
        self.sp_var = sp_var
        self.ivar = ivar
        self.ivar_boundary = ivar_boundary
        self.west_bd       = west_bd
        self.west_func_bd  = west_func_bd
        self.east_bd       = east_bd
        self.east_func_bd  = east_func_bd
        self.north_bd      = north_bd
        self.north_func_bd = north_func_bd
        self.south_bd      = south_bd
        self.south_func_bd = south_func_bd

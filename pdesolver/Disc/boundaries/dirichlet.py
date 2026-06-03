
from typing import List
from ...Auxs.FuncAux import repl_symbol as _repl_symbol
from .boundary_base import BoundaryCondition


class DirichletBC(BoundaryCondition):

    def __init__(self, bd_func: str, use_time_derivative: bool = True):
        super().__init__(bd_func)
        self.use_time_derivative = use_time_derivative

    def _replace_xy(self, expr: str, X: str, Y: str, str_sp_vars: str) -> str:
        out = _repl_symbol(expr, str_sp_vars[0], X)
        if len(str_sp_vars) == 2:
            out = _repl_symbol(out, str_sp_vars[1], Y)
        return out

    def apply(
        self,
        bd: str,
        list_eq: List[List[str]],
        n_part: List[int],
        xd_var: List[str],
        str_sp_vars: str = "",
    ) -> List[List[str]]:

        is_2d = len(str_sp_vars) == 2
        self._check_side(bd, is_2d)
        bd = bd.lower()

        if is_2d:
            return self._apply_2d(bd, list_eq, n_part, xd_var, str_sp_vars)
        return self._apply_1d(bd, list_eq, n_part, xd_var, str_sp_vars)

    def _apply_2d(self, bd, list_eq, n_part, xd_var, str_sp_vars):
        Nx, Ny = n_part[0], n_part[1]
        hx = f"h{xd_var[0]}_"
        hy = f"h{xd_var[0]}_"
        result = [[] for _ in range(len(list_eq))]

        if bd == "north":
            for func in range(len(list_eq)):
                for i in range(Nx):
                    expr = self._replace_xy(self.bd_func, f"{i} * {hx}", f"{Ny-1} * {hy}", str_sp_vars)
                    result[func].append(expr)

        elif bd == "south":
            for func in range(len(list_eq)):
                for i in range(Nx):
                    expr = self._replace_xy(self.bd_func, f"{i} * {hx}", f"0 * {hy}", str_sp_vars)
                    result[func].append(expr)

        elif bd == "east":
            for func in range(len(list_eq)):
                for j in range(Ny):
                    expr = self._replace_xy(self.bd_func, f"{Nx-1} * {hx}", f"{j} * {hy}", str_sp_vars)
                    result[func].append(expr)

        elif bd == "west":
            for func in range(len(list_eq)):
                for j in range(Ny):
                    expr = self._replace_xy(self.bd_func, f"0 * {hx}", f"{j} * {hy}", str_sp_vars)
                    result[func].append(expr)

        return result

    def _apply_1d(self, bd, list_eq, n_part, xd_var, str_sp_vars):
        Nx = n_part[0]
        hx = f"h{xd_var[0]}_"
        result = [[] for _ in range(len(list_eq))]

        if bd == "west":
            for func in range(len(list_eq)):
                expr = self._replace_xy(self.bd_func, f"0 * {hx}", "", str_sp_vars)
                result[func].append(expr)

        elif bd == "east":
            for func in range(len(list_eq)):
                expr = self._replace_xy(self.bd_func, f"{Nx-1} * {hx}", "", str_sp_vars)
                result[func].append(expr)

        return result

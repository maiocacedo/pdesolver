from typing import List
from .boundary_base import BoundaryCondition


class RobinBC(BoundaryCondition):

    def __init__(self, bd_func: str, alpha: str = "0", beta: str = "1"):
        super().__init__(bd_func)
        self.alpha = alpha
        self.beta  = beta

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
        n_funcs = len(list_eq)
        result = [[] for _ in range(n_funcs)]

        if bd in ("west", "east"):
            length = Ny
        elif bd in ("south", "north"):
            length = Nx
        else:
            raise ValueError(f"Lado invalido: {bd}")

        for func in range(n_funcs):
            for _ in range(length):
                result[func].append("0")
        return result

    def _apply_1d(self, bd, list_eq, n_part, xd_var, str_sp_vars):
        n_funcs = len(list_eq)
        result = [[] for _ in range(n_funcs)]
        for func in range(n_funcs):
            result[func].append("0")
        return result

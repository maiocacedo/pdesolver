from typing import List
from .boundary_base import BoundaryCondition


class PeriodicBC(BoundaryCondition):
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

        result = [[] for _ in range(len(list_eq))]

        if not is_2d:
            for func in range(len(list_eq)):
                if bd == "west":
                    result[func].append(list_eq[func][-1])
                elif bd == "east":
                    result[func].append(list_eq[func][0])
        else:
            raise NotImplementedError("Periódico 2D ainda não implementado.")

        return result

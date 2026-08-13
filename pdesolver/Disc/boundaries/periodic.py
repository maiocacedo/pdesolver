from typing import List

from .boundary_base import BoundaryCondition


class PeriodicBC(BoundaryCondition):

    def __init__(self, bd_func: str = "0"):
        super().__init__(bd_func)

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

        raise RuntimeError(
            "Contornos periódicos não passam por apply(): o eixo inteiro é "
            "resolvido pelo envolvimento de índices na expansão da malha, de "
            "modo que os nós das pontas recebem a mesma equação dos nós "
            "interiores."
        )

    @staticmethod
    def axis_of(bd: str) -> int:
        return 0 if bd.lower() in ("west", "east") else 1

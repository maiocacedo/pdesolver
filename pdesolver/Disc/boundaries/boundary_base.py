from abc import ABC, abstractmethod
from typing import List


class BoundaryCondition(ABC):

    def __init__(self, bd_func: str):
        self.bd_func = bd_func

    @abstractmethod
    def apply(
        self,
        bd: str,
        list_eq: List[List[str]],
        n_part: List[int],
        xd_var: List[str],
        str_sp_vars: str = "",
    ) -> List[List[str]]:

        ...

    @staticmethod
    def _valid_sides_2d() -> List[str]:
        return ["north", "south", "east", "west"]

    @staticmethod
    def _valid_sides_1d() -> List[str]:
        return ["east", "west"]

    def _check_side(self, bd: str, is_2d: bool) -> None:
        valid = self._valid_sides_2d() if is_2d else self._valid_sides_1d()
        if bd.lower() not in valid:
            raise ValueError(
                f"Contorno inválido: '{bd}'. Válidos para "
                f"{'2D' if is_2d else '1D'}: {valid}"
            )

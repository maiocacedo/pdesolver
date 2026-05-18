from .boundary_base import BoundaryCondition
from .dirichlet import DirichletBC
from .neumann import NeumannBC
from .robin import RobinBC


BOUNDARY_REGISTRY: dict[str, type[BoundaryCondition]] = {
    "dirichlet": DirichletBC,
    "neumann": NeumannBC,
    "robin": RobinBC,
}


def get_boundary(
    name: str,
    bd_func: str,
    alpha: str = "0",
    beta: str = "1",
    use_time_derivative: bool = True,
) -> BoundaryCondition:

    key = name.lower()
    if key not in BOUNDARY_REGISTRY:
        raise ValueError(
            f"Tipo de contorno desconhecido: '{name}'. "
            f"Disponíveis: {list(BOUNDARY_REGISTRY.keys())}"
        )
    cls = BOUNDARY_REGISTRY[key]

    if key == "dirichlet":
        return cls(bd_func, use_time_derivative=use_time_derivative)
    if key == "robin":
        return cls(bd_func, alpha=alpha, beta=beta)
    return cls(bd_func)


__all__ = [
    "BoundaryCondition",
    "DirichletBC",
    "NeumannBC",
    "RobinBC",
    "BOUNDARY_REGISTRY",
    "get_boundary",
]

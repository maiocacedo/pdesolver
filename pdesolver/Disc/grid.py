from typing import List, Sequence, Tuple

import numpy as np

_SPACINGS = ("uniform", "chebyshev", "tanh", "tanh_left", "tanh_right")


def _nodes_from_spacing(
    a: float,
    b: float,
    n: int,
    spacing: str,
    beta: float,
    periodic: bool,
) -> np.ndarray:

    if periodic:
        if spacing != "uniform":
            raise ValueError(
                f"Eixo periódico aceita apenas spacing='uniform' ou nós "
                f"explícitos. Recebido: '{spacing}'."
            )
        return a + (b - a) * np.arange(n, dtype=np.float64) / n

    if spacing == "uniform":
        return np.linspace(a, b, n)

    if spacing == "chebyshev":
        k = np.arange(n, dtype=np.float64)
        return (a + b) / 2.0 - (b - a) / 2.0 * np.cos(np.pi * k / (n - 1))

    if beta <= 0.0:
        raise ValueError(f"beta deve ser positivo. Recebido: {beta}.")

    s = np.linspace(0.0, 1.0, n)
    if spacing == "tanh":
        return (a + b) / 2.0 + (b - a) / 2.0 * np.tanh(
            beta * (2.0 * s - 1.0)
        ) / np.tanh(beta)
    if spacing == "tanh_left":
        return a + (b - a) * (1.0 - np.tanh(beta * (1.0 - s)) / np.tanh(beta))
    if spacing == "tanh_right":
        return a + (b - a) * np.tanh(beta * s) / np.tanh(beta)

    raise ValueError(
        f"Espaçamento inválido: '{spacing}'. Use um de {list(_SPACINGS)} "
        f"ou forneça 'nodes'."
    )


def _extended_nodes(
    nodes: np.ndarray,
    a: float,
    b: float,
    periodic: bool,
) -> np.ndarray:

    xe = np.empty(nodes.size + 2, dtype=np.float64)
    xe[1:-1] = nodes
    if periodic:
        period = b - a
        xe[0] = nodes[-1] - period
        xe[-1] = nodes[0] + period
    else:
        xe[0] = nodes[0] - (nodes[1] - nodes[0])
        xe[-1] = nodes[-1] + (nodes[-1] - nodes[-2])
    return xe


class Axis:

    def __init__(self, nodes, bounds, periodic: bool = False):
        nodes = np.asarray(nodes, dtype=np.float64).ravel()
        if nodes.size < 3:
            raise ValueError(
                f"Cada eixo exige ao menos 3 nós. Recebido: {nodes.size}."
            )
        if np.any(np.diff(nodes) <= 0.0):
            raise ValueError("Os nós de um eixo devem ser estritamente crescentes.")

        self.nodes = nodes
        self.n = int(nodes.size)
        self.bounds = (float(bounds[0]), float(bounds[1]))
        self.periodic = bool(periodic)

        xe = _extended_nodes(nodes, self.bounds[0], self.bounds[1], self.periodic)
        self.hm = xe[1:-1] - xe[:-2]
        self.hp = xe[2:] - xe[1:-1]

    @property
    def uniform(self) -> bool:
        return bool(np.allclose(self.hm, self.hp))

    def w1(self, method: str = "central") -> Tuple[np.ndarray, ...]:
        hm, hp = self.hm, self.hp
        if method == "central":
            return (
                -hp / (hm * (hm + hp)),
                (hp - hm) / (hm * hp),
                hm / (hp * (hm + hp)),
            )
        if method == "forward":
            return (np.zeros_like(hp), -1.0 / hp, 1.0 / hp)
        if method == "backward":
            return (-1.0 / hm, 1.0 / hm, np.zeros_like(hm))
        raise ValueError(
            f"Método inválido: '{method}'. Use 'forward', 'central' ou 'backward'."
        )

    def w2(self) -> Tuple[np.ndarray, ...]:
        hm, hp = self.hm, self.hp
        return (
            2.0 / (hm * (hm + hp)),
            -2.0 / (hm * hp),
            2.0 / (hp * (hm + hp)),
        )

    def __repr__(self):
        kind = "uniform" if self.uniform else "não uniforme"
        per = ", periódico" if self.periodic else ""
        return f"Axis(n={self.n}, bounds={self.bounds}, {kind}{per})"


class Grid:

    def __init__(self, axes: Sequence[Axis]):
        self.axes = list(axes)
        self.ndim = len(self.axes)
        self.shape = tuple(ax.n for ax in self.axes)
        self.size = int(np.prod(self.shape))

    @property
    def periodic(self) -> List[bool]:
        return [ax.periodic for ax in self.axes]

    @property
    def uniform(self) -> bool:
        return all(ax.uniform for ax in self.axes)

    def coords(self) -> List[np.ndarray]:
        return list(
            np.meshgrid(*[ax.nodes for ax in self.axes], indexing="ij")
        )

    def __repr__(self):
        return f"Grid(shape={self.shape}, ndim={self.ndim})"


def _axis_spec(mesh, k: int, ndim: int) -> dict:
    if isinstance(mesh, (list, tuple)):
        if len(mesh) != ndim:
            raise ValueError(
                f"mesh como lista exige um item por eixo ({ndim}). "
                f"Recebido: {len(mesh)}."
            )
        spec = mesh[k]
    else:
        spec = mesh

    if isinstance(spec, str):
        return {"type": spec}
    if isinstance(spec, dict):
        return dict(spec)
    raise ValueError(
        "Cada item de mesh deve ser uma string ou um dicionário com as "
        "chaves 'type', 'beta' ou 'nodes'."
    )


def build_grid(bounds, disc_n, mesh="uniform", periodic=None) -> Grid:
    ndim = len(disc_n)
    if len(bounds) < ndim:
        raise ValueError(
            f"ivar_boundary define {len(bounds)} domínio(s), mas disc_n pede "
            f"{ndim} eixo(s)."
        )
    if periodic is None:
        periodic = [False] * ndim

    axes = []
    for k in range(ndim):
        spec = _axis_spec(mesh, k, ndim)
        a, b = bounds[k]
        if float(b) <= float(a):
            raise ValueError(
                f"Domínio inválido no eixo {k}: ({a}, {b}). Exige-se a < b."
            )
        nodes = spec.get("nodes")
        if nodes is None:
            nodes = _nodes_from_spacing(
                float(a),
                float(b),
                int(disc_n[k]),
                spec.get("type", "uniform"),
                float(spec.get("beta", 2.0)),
                bool(periodic[k]),
            )
        else:
            nodes = np.asarray(nodes, dtype=np.float64).ravel()
        axes.append(Axis(nodes, (a, b), bool(periodic[k])))

    return Grid(axes)

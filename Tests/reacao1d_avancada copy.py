# import time
import os
import sys

diretorio_atual = os.path.dirname(os.path.abspath(__file__))
diretorio_pai = os.path.abspath(os.path.join(diretorio_atual, ".."))
sys.path.append(diretorio_pai)

import numpy as np

import PDE

# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
from PDES import PDES

# ---------------------------------------------------------------------------
# Parâmetros
# ---------------------------------------------------------------------------
disc_n = [7, 7]
tf = 5.0
nt = 100

x_lin = np.linspace(0, 1, disc_n[0])
y_lin = np.linspace(0, 1, disc_n[1])
X, Y = np.meshgrid(x_lin, y_lin, indexing="ij")

v = 1.0
dax = 0.00001

PDE_A = PDE.PDE(
    f"dA/dt = -{v}*(dA/dx) + {dax}*(d2A/dx2 + d2A/dy2) - A*B",
    "A",
    ["x", "y"],
    ["t"],
    ivar_boundary=[(0, 1), (0, 1)],
    expr_ic="1e-6",
    west_bd="Dirichlet",
    west_func_bd="1",
    east_bd="Neumann",
    east_func_bd="0",
    north_bd="Neumann",
    north_func_bd="0",
    south_bd="Neumann",
    south_func_bd="0",
)

PDE_B = PDE.PDE(
    f"dB/dt = -{v}*(dB/dx) + {dax}*(d2B/dx2 + d2B/dy2) -A*B",
    "B",
    ["x", "y"],
    ["t"],
    ivar_boundary=[(0, 1), (0, 1)],
    expr_ic="1e-6",
    west_bd="Dirichlet",
    west_func_bd="1",
    east_bd="Neumann",
    east_func_bd="0",
    north_bd="Neumann",
    north_func_bd="0",
    south_bd="Neumann",
    south_func_bd="0",
)

PDE_C = PDE.PDE(
    f"dC/dt = -{v}*(dC/dx) + {dax}*(d2C/dx2 + d2C/dy2) + 2*A*B",
    "C",
    ["x", "y"],
    ["t"],
    ivar_boundary=[(0, 1), (0, 1)],
    expr_ic="1e-6",
    west_bd="Dirichlet",
    west_func_bd="0",
    east_bd="Neumann",
    east_func_bd="0",
    north_bd="Neumann",
    north_func_bd="0",
    south_bd="Neumann",
    south_func_bd="0",
)

PDE_D = PDE.PDE(
    f"dD/dt = -{v}*(dD/dx) + {dax}*(d2D/dx2 + d2D/dy2) + 2*A*B",
    "D",
    ["x", "y"],
    ["t"],
    ivar_boundary=[(0, 1), (0, 1)],
    expr_ic="0",
    west_bd="Dirichlet",
    west_func_bd="0",
    east_bd="Neumann",
    east_func_bd="0",
    north_bd="Neumann",
    north_func_bd="0",
    south_bd="Neumann",
    south_func_bd="0",
)

PDES1 = PDES([PDE_A, PDE_B, PDE_C, PDE_D], disc_n)
print("discretizando...")
PDES1.discretize(method="backward")
print("discretizado")
PDES1.solve(method="bdf2", tf=tf, nt=nt, tol=1e-6, verbose=True)
resultado_final_cn = PDES1.results

PDES1.save_to_json()

# Perfil no instante final (ou qualquer time_step)
PDES1.visualize(mode="plot3d", func_idx=0)
PDES1.visualize(mode="plot3d", func_idx=0, time_step=0)  # instante inicial

# Todos os perfis sobrepostos com gradiente de cor temporal
PDES1.visualize(mode="plot3d_all", func_idx=0, n_profiles=10, tf=5.0, cmap="plasma")

# Heatmap x vs t — vê a evolução inteira de uma vez
PDES1.visualize(mode="heatmap", func_idx=0, tf=5.0, cmap="viridis")

# Animação
PDES1.visualize(mode="animation3d", func_idx=0, tf=1.0, interval=5)
PDES1.visualize(mode="animation3d", func_idx=1, tf=1.0, interval=5)
PDES1.visualize(mode="animation3d", func_idx=2, tf=1.0, interval=5)
PDES1.visualize(mode="animation3d", func_idx=3, tf=1.0, interval=5)

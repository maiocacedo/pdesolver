import os
import sys

import numpy as np

diretorio_atual = os.path.dirname(os.path.abspath(__file__))
diretorio_pai = os.path.abspath(os.path.join(diretorio_atual, ".."))
sys.path.append(diretorio_pai)

import time

import PDE
from PDES import PDES

start_time_df = time.time()

# Equacao da onda 2D: d2u/dt2 = c^2 (d2u/dx2 + d2u/dy2)
# Sistema de 1a ordem (v = du/dt):
#   du/dt = v
#   dv/dt = c^2 (d2u/dx2 + d2u/dy2)
#
# Para ter propagacao SO para a direita (sem onda indo para esquerda),
# d'Alembert exige:
#   u(x,y,0) = f(x)           (pulso inicial)
#   du/dt(x,y,0) = -c * f'(x) (velocidade inicial casada com a propagacao)
#
# Pulso gaussiano centrado em x0=0.3, largura ~0.1:
#   f(x)   = exp(-100 (x-x0)^2)
#   f'(x)  = -200 (x-x0) * f(x)
# Logo v0(x) = -c * f'(x) = 200 c (x-x0) * f(x)
c = 10.0
x0 = 0.3

# Mais nos em x (direcao de propagacao) que em y
disc_n = [40, 40]

ic_u = f"exp(-100*(x-{x0})**2)"
ic_v = f"{200 * c}*(x-{x0})*exp(-100*(x-{x0})**2)"

PDE1 = PDE.PDE(
    "dF/dt = G",
    "F",
    ["x", "y"],
    ["t"],
    ivar_boundary=[(0, 1), (0, 1)],
    expr_ic=ic_u,
    west_bd="Dirichlet",
    west_func_bd="0",
    east_bd="Dirichlet",
    east_func_bd="0",
    north_bd="Dirichlet",
    north_func_bd="0",
    south_bd="Dirichlet",
    south_func_bd="0",
)

PDE2 = PDE.PDE(
    f"dG/dt = {c**2}*(d2F/dx2 + d2F/dy2)",
    "G",
    ["x", "y"],
    ["t"],
    ivar_boundary=[(0, 1), (0, 1)],
    expr_ic=ic_v,
    west_bd="Dirichlet",
    west_func_bd="0",
    east_bd="Dirichlet",
    east_func_bd="0",
    north_bd="Dirichlet",
    north_func_bd="0",
    south_bd="Dirichlet",
    south_func_bd="0",
)

# Tempo final: pulso parte de x=0.3 com velocidade c=1, entao em t=0.3
# o pico esta em x=0.6 (ainda nao tocou x=1, evita reflexao).
tf = 1.2
nt = 600


# Solucao analitica enquanto o pulso nao atinge o contorno:
#   u(x, y, t) = exp(-100 (x - x0 - c t)^2)
def analitica_pulso(X, Y, t, c=c, x0=x0):
    return np.exp(-100.0 * (X - x0 - c * t) ** 2).flatten().tolist()


resultado_analitico = analitica_pulso(
    *np.meshgrid(np.linspace(0, 1, disc_n[0]), np.linspace(0, 1, disc_n[1])), t=tf
)

PDES1 = PDES([PDE1, PDE2], disc_n)

PDES1.discretize(method="central")

n_nodes = disc_n[0] * disc_n[1]


PDES1.solve(method="bdf2", tf=tf, nt=nt, tol=1e-6)
resultado_final_bdf2 = PDES1.results[0][:n_nodes]
mae_resultado_final_bdf2 = sum(
    abs(r - p) for r, p in zip(resultado_analitico, resultado_final_bdf2)
) / len(resultado_analitico)
print(mae_resultado_final_bdf2)

PDES1.visualize(mode="plot3d", func_idx=0, cmap="RdYlBu_r")
PDES1.visualize(mode="animation3d", func_idx=0, cmap="RdYlBu_r")

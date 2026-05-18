import numpy as np
import os, sys

diretorio_atual = os.path.dirname(os.path.abspath(__file__))
diretorio_pai = os.path.abspath(os.path.join(diretorio_atual, '..'))
sys.path.append(diretorio_pai)

from PDES import PDES
import PDE
import cmocean


c = 10.0

x_src, y_src = 0.5, 0.5
t0 = 0.01
sigma_s = 0.02
sigma_t = 0.005
A = -5000.0

disc_n = [81, 81]

ic_F = '0'
ic_G = '0'

fonte = (f'{A}*exp(-((x-{x_src})**2 + (y-{y_src})**2)/{sigma_s**2})'
         f'*exp(-(t-{t0})**2/{sigma_t**2})')

PDE1 = PDE.PDE(
    'dF/dt = G',
    'F', ['x', 'y'], ['t'],
    ivar_boundary=[(0, 1), (0, 1)],
    expr_ic=ic_F,
    west_bd="Dirichlet",  west_func_bd="0",
    east_bd="Dirichlet",  east_func_bd="0",
    north_bd="Dirichlet", north_func_bd="0",
    south_bd="Dirichlet", south_func_bd="0",
)

PDE2 = PDE.PDE(
    f'dG/dt = {c**2}*(d2F/dx2 + d2F/dy2) + {fonte}',
    'G', ['x', 'y'], ['t'],
    ivar_boundary=[(0, 1), (0, 1)],
    expr_ic=ic_G,
    west_bd="Dirichlet",  west_func_bd="0",
    east_bd="Dirichlet",  east_func_bd="0",
    north_bd="Dirichlet", north_func_bd="0",
    south_bd="Dirichlet", south_func_bd="0",
)

tf = 2
nt = 2000

PDES1 = PDES([PDE1, PDE2], disc_n)
PDES1.discretize(method="central")
PDES1.solve(method='bdf2', tf=tf, nt=nt, tol=1e-6)

PDES1.visualize(mode='animation3d', func_idx=0, cmap='RdYlBu_r')

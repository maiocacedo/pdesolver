"""Heterogeneous regions: problems the vectorized backend cannot express.

The symbolic backend represents the discretized system as one independent
algebraic equation per node, so individual nodes can carry their own equation.
This script demonstrates three cases that follow from that and verifies each
against a closed-form result.

Run with:
    python examples/regioes_heterogeneas.py
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pdesolver import PDE, PDES


def obstaculo_interno():
    print('\n== Obstaculo interno: bloco congelado dentro do dominio ==')
    n = 41
    pde = PDE(
        'dU/dt = 0.2*d2U/dx2 + 0.2*d2U/dy2',
        'U', ['x', 'y'], ['t'],
        ivar_boundary=[(0, 1), (0, 1)],
        expr_ic='1',
        west_bd='Dirichlet',  west_func_bd='0',
        east_bd='Dirichlet',  east_func_bd='0',
        north_bd='Dirichlet', north_func_bd='0',
        south_bd='Dirichlet', south_func_bd='0',
    )
    sim = PDES([pde], [n, n])
    X, Y = sim.grid.coords()
    bloco = (np.abs(X - 0.5) < 0.12) & (np.abs(Y - 0.5) < 0.12)

    sim.discretize(method='central',
                   regions=[{'where': bloco, 'eq': 'dU/dt = 0'}])
    sim.solve(method='bdf2', tf=0.2, nt=100)

    u = np.asarray(sim.results[0]).reshape(n, n)
    print(f'   nos no obstaculo      : {int(bloco.sum())}')
    print(f'   obstaculo preservado  : {np.allclose(u[bloco], 1.0)}')
    print(f'   campo fora do bloco   : {u[5, 5]:.6f} (difundiu para o contorno)')


def interface_de_material():
    print('\n== Interface entre dois materiais (forma conservativa) ==')
    k1, k2, a = 1.0, 0.25, 0.4
    u0, ul = 1.0, 0.0
    u_int = (k1 * u0 / a + k2 * ul / (1 - a)) / (k1 / a + k2 / (1 - a))
    print(f'   temperatura de interface analitica: {u_int:.9f}')

    for n in (41, 81, 161):
        h = 1.0 / (n - 1)
        i = int(round(a / h))
        pde = PDE(
            f'dU/dt = {k1}*d2U/dx2',
            'U', ['x'], ['t'],
            ivar_boundary=[(0, 1)],
            expr_ic='1-x',
            west_bd='Dirichlet', west_func_bd=str(u0),
            east_bd='Dirichlet', east_func_bd=str(ul),
        )
        sim = PDES([pde], [n])
        x = sim.grid.axes[0].nodes

        sim.discretize(
            method='central',
            regions=[{'where': x > a + h / 2, 'eq': f'dU/dt = {k2}*d2U/dx2'}],
        )

        flat, d_vars = sim.disc_results
        flat[i] = (f'({k2}*XX0_{i+1}_0 - {k1 + k2}*XX0_{i}_0 '
                   f'+ {k1}*XX0_{i-1}_0)/{h ** 2}')
        sim.disc_results = (flat, d_vars)

        sim.solve(method='bdf2', tf=5.0, nt=1000)
        u = np.asarray(sim.results[0])
        exata = np.where(
            x <= a,
            u0 + (u_int - u0) * x / a,
            u_int + (ul - u_int) * (x - a) / (1 - a),
        )
        fluxo_esq = k1 * (u[i] - u[i - 1]) / h
        fluxo_dir = k2 * (u[i + 1] - u[i]) / h
        print(f'   N={n:4d}  u_interface={u[i]:.9f}  '
              f'MAE={np.mean(np.abs(u - exata)):.2e}  '
              f'salto de fluxo={abs(fluxo_esq - fluxo_dir):.2e}')


def fonte_pontual():
    print('\n== Fonte pontual interna ==')
    n, k, s_val = 81, 1.0, 10.0
    h = 1.0 / (n - 1)
    i = (n - 1) // 2

    pde = PDE(
        f'dU/dt = {k}*d2U/dx2',
        'U', ['x'], ['t'],
        ivar_boundary=[(0, 1)],
        expr_ic='0',
        west_bd='Dirichlet', west_func_bd='0',
        east_bd='Dirichlet', east_func_bd='0',
    )
    sim = PDES([pde], [n])
    x = sim.grid.axes[0].nodes
    fonte = np.zeros(n, dtype=bool)
    fonte[i] = True

    sim.discretize(
        method='central',
        regions=[{'where': fonte, 'eq': f'dU/dt = {k}*d2U/dx2 + {s_val}'}],
    )
    sim.solve(method='bdf2', tf=2.0, nt=400)
    u = np.asarray(sim.results[0])

    pico = s_val * h * x[i] * (1 - x[i]) / k
    salto = (u[i + 1] - u[i]) / h - (u[i] - u[i - 1]) / h
    print(f'   pico numerico={u[i]:.6f}  analitico={pico:.6f}')
    print(f'   salto de derivada={salto:.6f}  esperado={-s_val * h / k:.6f}')


if __name__ == '__main__':
    print('pdesolver - regioes heterogeneas (backend simbolico)')
    obstaculo_interno()
    interface_de_material()
    fonte_pontual()

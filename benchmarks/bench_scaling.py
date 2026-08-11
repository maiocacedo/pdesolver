"""Scaling benchmark for the symbolic and stencil discretization backends.

Reproduces the performance figures reported in the accompanying paper.

Usage
-----
    python benchmarks/bench_scaling.py
    python benchmarks/bench_scaling.py --max-symbolic 45 --sizes 64,128,256
"""

import argparse
import os
import sys
import time

import matplotlib
matplotlib.use('Agg')
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pdesolver import PDE, PDES
from pdesolver.Solvers.RKF import select_array_module
from pdesolver.Solvers.solver_base import ColoredJacobian, compile_equations

HEAT_2D = 'dF/dt = 0.1*d2F/dx2 + 0.2*d2F/dy2'
IC_2D = 'sin(pi*x)*sin(pi*y)'


def _system(n, backend):
    pde = PDE(
        HEAT_2D, 'F', ['x', 'y'], ['t'],
        ivar_boundary=[(0, 1), (0, 1)],
        expr_ic=IC_2D,
        west_bd='Dirichlet',  west_func_bd='0',
        east_bd='Dirichlet',  east_func_bd='0',
        north_bd='Dirichlet', north_func_bd='0',
        south_bd='Dirichlet', south_func_bd='0',
    )
    return PDES([pde], [n, n], backend=backend)


def _timed(fn, repeats=1, warmup=0):
    for _ in range(warmup):
        fn()
    best = None
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(repeats):
            out = fn()
        dt = (time.perf_counter() - t0) / repeats
        best = dt if best is None else min(best, dt)
    return best, out


def bench_setup(sizes, max_symbolic):
    print('\n== Setup cost: equations ready to integrate ==')
    print(f'{"N":>6} {"DOF":>9} {"symbolic":>12} {"stencil":>10} {"speedup":>9}')
    for n in sizes:
        sim = _system(n, 'stencil')
        t_sten, _ = _timed(lambda: sim.discretize(method='central'))

        if n <= max_symbolic:
            sym = _system(n, 'symbolic')
            t_disc, _ = _timed(lambda: sym.discretize(method='central'))
            t_comp, _ = _timed(
                lambda: compile_equations(*sym.disc_results)
            )
            t_sym = t_disc + t_comp
            print(f'{n:6d} {n*n:9d} {t_sym:11.3f}s {t_sten:9.4f}s '
                  f'{t_sym/t_sten:8.0f}x')
        else:
            print(f'{n:6d} {n*n:9d} {"not feasible":>12} {t_sten:9.4f}s')


def bench_rhs(sizes):
    print('\n== Right-hand side evaluation (stencil backend) ==')
    xp_gpu = select_array_module(10 ** 9)
    has_gpu = xp_gpu is not np
    header = f'{"N":>6} {"DOF":>9} {"CPU":>12} {"ns/DOF":>9}'
    if has_gpu:
        header += f' {"GPU":>12} {"speedup":>9}'
    print(header)

    for n in sizes:
        sim = _system(n, 'stencil')
        sim.discretize(method='central')
        op = sim.operator
        u = np.array(sim.ic)
        reps = max(10, int(5e7 // (n * n)))
        t_cpu, _ = _timed(lambda: op(0.0, u), reps, warmup=5)
        row = (f'{n:6d} {n*n:9d} {t_cpu*1e6:11.1f}us '
               f'{t_cpu/(n*n)*1e9:8.2f}')
        if has_gpu:
            import cupy as cp
            gop = op.to_device(xp_gpu)
            gu = cp.asarray(u)

            def _gpu_call():
                gop(0.0, gu)
                cp.cuda.Stream.null.synchronize()

            t_gpu, _ = _timed(_gpu_call, reps, warmup=10)
            row += f' {t_gpu*1e6:11.1f}us {t_cpu/t_gpu:8.1f}x'
        print(row)

    if not has_gpu:
        print('  (no CuPy/CUDA device detected - GPU column omitted)')


def bench_implicit(sizes):
    print('\n== Implicit setup: operator matrix assembly ==')
    print(f'{"N":>6} {"DOF":>9} {"colors":>8} {"nnz":>10} {"assembly":>11}')
    for n in sizes:
        sim = _system(n, 'stencil')
        sim.discretize(method='central')
        op = sim.operator
        pattern, colors, n_colors = op.sparsity()
        jac = ColoredJacobian(pattern, colors, n_colors)
        u = np.zeros(op.size)
        t_asm, _ = _timed(lambda: jac.build(op, u, 0.0))
        print(f'{n:6d} {n*n:9d} {n_colors:8d} {jac.nnz:10d} {t_asm:10.3f}s')
    print('  Colour count is independent of N: the number of right-hand side')
    print('  evaluations needed for the Jacobian does not grow with the mesh.')


def bench_solve(sizes, tf, nt):
    print(f'\n== End-to-end solve (RKF, tf={tf}, history off) ==')
    print(f'{"N":>6} {"DOF":>9} {"wall":>11} {"device":>8}')
    for n in sizes:
        sim = _system(n, 'stencil')
        sim.discretize(method='central')
        t0 = time.perf_counter()
        sim.solve(method='RKF', tf=tf, nt=nt, save_every=0)
        wall = time.perf_counter() - t0
        dev = 'GPU' if select_array_module(sim.operator.size) is not np else 'CPU'
        print(f'{n:6d} {n*n:9d} {wall:10.3f}s {dev:>8}')
    print('  Explicit stepping is stability limited: the accepted step size')
    print('  scales as h^2 for diffusion, so step count grows with the mesh.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', default='15,30,60,128,256,512',
                        help='comma-separated grid sizes (NxN)')
    parser.add_argument('--max-symbolic', type=int, default=60,
                        help='largest N still benchmarked on the symbolic path')
    parser.add_argument('--tf', type=float, default=0.005)
    parser.add_argument('--nt', type=int, default=5)
    parser.add_argument('--skip-solve', action='store_true')
    args = parser.parse_args()

    sizes = [int(s) for s in args.sizes.split(',') if s.strip()]

    print('pdesolver - scaling benchmark')
    print(f'grid sizes: {sizes}')

    bench_setup(sizes, args.max_symbolic)
    bench_rhs(sizes)
    bench_implicit([n for n in sizes if n <= 256])
    if not args.skip_solve:
        bench_solve([n for n in sizes if n <= 256], args.tf, args.nt)


if __name__ == '__main__':
    main()

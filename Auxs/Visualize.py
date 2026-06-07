
import numpy as np
import matplotlib


def _select_backend():

    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            return  # nao forcar nada — inline cuida do show()
    except Exception:
        pass

    import importlib.util as _il
    for backend, modulo in (("TkAgg", "tkinter"),
                            ("QtAgg", "PyQt5"),
                            ("QtAgg", "PySide6")):
        if _il.find_spec(modulo) is not None:
            try:
                matplotlib.use(backend, force=True)
                return
            except Exception:
                continue
    matplotlib.use("Agg", force=True)


_select_backend()

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def visualize(pdes_obj, mode='heatmap', func_idx=0, time_step=-1, **kwargs):
    
    if pdes_obj.results is None:
        print("Erro: rode .solve() antes de visualizar.")
        return

    _, historico = pdes_obj.results

    if not historico or not historico[func_idx]:
        print("Erro: histórico vazio.")
        return

    if _is_1d(pdes_obj):
        x = _get_x1d(pdes_obj)
        if mode in ('plot1d', 'plot'):
            plot1d(historico, x, pdes_obj.funcs, func_idx, time_step, **kwargs)
        elif mode == 'plot1d_all':
            plot1d_all(historico, x, pdes_obj.funcs, func_idx, **kwargs)
        elif mode == 'heatmap1d':
            heatmap1d(historico, x, pdes_obj.funcs, pdes_obj.pdes, func_idx, **kwargs)
        elif mode == 'animation1d':
            animate1d(historico, x, pdes_obj.funcs, func_idx, **kwargs)
        else:
            print(f"Modo '{mode}' não reconhecido para 1D. "
                  f"Use: 'plot1d', 'plot1d_all', 'heatmap1d', 'animation1d'.")
        return

    nx, ny = pdes_obj.disc_n
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    if mode == 'heatmap':
        plot_heatmap(historico, X, Y, pdes_obj.funcs, pdes_obj.disc_n, func_idx, time_step)
    elif mode == 'animation':
        animate2d(historico, X, Y, pdes_obj.funcs, pdes_obj.disc_n, func_idx, **kwargs)
    elif mode == 'plot3d':
        plot3d(historico, X, Y, pdes_obj.funcs, pdes_obj.disc_n, func_idx, time_step, **kwargs)
    elif mode == 'animation3d':
        animate3d(historico, X, Y, pdes_obj.funcs, pdes_obj.disc_n, func_idx, **kwargs)
    else:
        print(f"Modo '{mode}' desconhecido para 2D. "
              f"Use: 'heatmap', 'animation', 'plot3d', 'animation3d'.")


def _is_1d(pdes_obj):
    return len(pdes_obj.disc_n) == 1


def _get_x1d(pdes_obj):
    a, b = pdes_obj.pdes[0].ivar_boundary[0]
    return np.linspace(a, b, pdes_obj.disc_n[0])


def plot1d(historico, x, funcs, func_idx, time_step, **kwargs):
    color = kwargs.get('color', 'steelblue')
    lw    = kwargs.get('lw', 2)

    data  = np.array(historico[func_idx][time_step])
    label = time_step if time_step >= 0 else len(historico[func_idx]) + time_step

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, data, color=color, linewidth=lw, label=f't = passo {label}')
    ax.set_xlabel('x')
    ax.set_ylabel(funcs[func_idx])
    ax.set_title(f"Perfil de {funcs[func_idx]} — passo {label}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot1d_all(historico, x, funcs, func_idx, **kwargs):
    n_profiles = kwargs.get('n_profiles', 10)
    cmap       = kwargs.get('cmap', 'viridis')
    lw         = kwargs.get('lw', 1.5)
    tf_real    = kwargs.get('tf', None)

    n_passos = len(historico[func_idx])
    indices  = np.linspace(0, n_passos - 1, n_profiles, dtype=int)
    cm       = plt.get_cmap(cmap)
    colors   = [cm(i / (n_profiles - 1)) for i in range(n_profiles)]

    fig, ax = plt.subplots(figsize=(8, 5))

    for idx_cor, idx_passo in enumerate(indices):
        data = np.array(historico[func_idx][idx_passo])
        if tf_real is not None:
            t_val = tf_real * idx_passo / (n_passos - 1)
            lbl   = f't = {t_val:.2f}'
        else:
            lbl = f'passo {idx_passo}'
        ax.plot(x, data, color=colors[idx_cor], linewidth=lw, label=lbl)

    ax.set_xlabel('x')
    ax.set_ylabel(funcs[func_idx])
    ax.set_title(f"Evolução de {funcs[func_idx]} ao longo do tempo")
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label('t / tf')

    plt.tight_layout()
    plt.show()


def heatmap1d(historico, x, funcs, pdes, func_idx, **kwargs):
    """Heatmap x (eixo y) vs t (eixo x) — evolução espaciotemporal completa."""
    cmap    = kwargs.get('cmap', 'viridis')
    tf_real = kwargs.get('tf', None)

    matriz   = np.array(historico[func_idx])  # (n_passos, nx)
    n_passos = matriz.shape[0]
    t_max    = tf_real if tf_real is not None else n_passos - 1
    a, b     = pdes[0].ivar_boundary[0]

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(
        matriz.T,
        aspect='auto',
        origin='lower',
        extent=[0, t_max, a, b],
        cmap=cmap
    )
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.set_title(f"Evolução espaciotemporal de {funcs[func_idx]}")
    plt.colorbar(im, ax=ax, label=funcs[func_idx])
    plt.tight_layout()
    plt.show()


def animate1d(historico, x, funcs, func_idx, **kwargs):
    frames_step = kwargs.get('frames_step', 1)
    interval    = kwargs.get('interval', 50)
    color       = kwargs.get('color', 'steelblue')
    lw          = kwargs.get('lw', 2)
    tf_real     = kwargs.get('tf', None)

    n_passos = len(historico[func_idx])
    frames   = list(range(0, n_passos, frames_step))

    todos  = np.concatenate([historico[func_idx][f] for f in frames])
    y_min, y_max = todos.min(), todos.max()
    margem = (y_max - y_min) * 0.05 or 0.1

    fig, ax = plt.subplots(figsize=(8, 4))
    linha,  = ax.plot(x, historico[func_idx][0], color=color, linewidth=lw)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(y_min - margem, y_max + margem)
    ax.set_xlabel('x')
    ax.set_ylabel(funcs[func_idx])
    ax.grid(True, alpha=0.3)
    titulo = ax.set_title('')

    def update(frame_idx):
        passo = frames[frame_idx]
        linha.set_ydata(np.array(historico[func_idx][passo]))
        if tf_real is not None:
            t_val = tf_real * passo / (n_passos - 1)
            titulo.set_text(f"{funcs[func_idx]} — t = {t_val:.3f}")
        else:
            titulo.set_text(f"{funcs[func_idx]} — passo {passo}")
        return linha, titulo

    anim = FuncAnimation(fig, update, frames=len(frames),
                         interval=interval, blit=True)
    plt.tight_layout()
    plt.show()

def plot_heatmap(historico, X, Y, funcs, disc_n, func_idx, time_step):
    data = np.array(historico[func_idx][time_step]).reshape(disc_n)
    plt.figure(figsize=(7, 6))
    contorno = plt.contourf(X, Y, data, levels=30, cmap='hot')
    plt.title(f"Distribuição de {funcs[func_idx]} no passo {time_step}")
    plt.colorbar(contorno)
    plt.show()


def animate2d(historico, X, Y, funcs, disc_n, func_idx, **kwargs):
    frames_step = kwargs.get('frames_step', 1)
    interval    = kwargs.get('interval', 50)

    fig, ax = plt.subplots(figsize=(7, 6))
    Z_ini   = np.array(historico[func_idx][0]).reshape(disc_n)
    contorno = ax.contourf(X, Y, Z_ini, levels=30, cmap='hot')
    fig.colorbar(contorno, ax=ax)

    def update(frame):
        ax.clear()
        Z    = np.array(historico[func_idx][frame]).reshape(disc_n)
        cont = ax.contourf(X, Y, Z, levels=30, cmap='hot')
        ax.set_title(f"Evolução de {funcs[func_idx]} - Passo {frame}")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        return cont,

    frames_list = range(0, len(historico[func_idx]), frames_step)
    anim = FuncAnimation(fig, update, frames=frames_list,
                         interval=interval, blit=False)
    plt.show()


def plot3d(historico, X, Y, funcs, disc_n, func_idx, time_step, **kwargs):
    cmap  = kwargs.get('cmap', 'hot')
    alpha = kwargs.get('alpha', 1.0)
    elev  = kwargs.get('elev', 30)
    azim  = kwargs.get('azim', -60)

    data  = np.array(historico[func_idx][time_step]).reshape(disc_n)
    label = time_step if time_step >= 0 else len(historico[func_idx]) + time_step

    fig = plt.figure(figsize=(8, 6))
    ax  = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, data, cmap=cmap, alpha=alpha)
    ax.set_title(f"Superfície 3D de {funcs[func_idx]} - Passo {label}")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel(str(funcs[func_idx]))
    ax.view_init(elev=elev, azim=azim)
    plt.tight_layout()
    plt.show()


def animate3d(historico, X, Y, funcs, disc_n, func_idx, **kwargs):
    frames_step = kwargs.get('frames_step', 1)
    interval    = kwargs.get('interval', 50)
    cmap        = kwargs.get('cmap', 'hot')
    alpha       = kwargs.get('alpha', 1.0)
    elev        = kwargs.get('elev', 30)
    azim        = kwargs.get('azim', -60)

    todos_frames = [
        np.array(historico[func_idx][f]).reshape(disc_n)
        for f in range(0, len(historico[func_idx]), frames_step)
    ]
    z_min = min(Z.min() for Z in todos_frames)
    z_max = max(Z.max() for Z in todos_frames)

    fig = plt.figure(figsize=(8, 6))
    ax  = fig.add_subplot(111, projection='3d')

    def update(frame):
        ax.clear()
        Z = todos_frames[frame]
        ax.plot_surface(X, Y, Z, cmap=cmap, alpha=alpha)
        ax.set_title(
            f"Evolução 3D de {funcs[func_idx]} - Passo {frame * frames_step}"
        )
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel(str(funcs[func_idx]))
        ax.set_zlim(z_min, z_max)
        ax.view_init(elev=elev, azim=azim)

    anim = FuncAnimation(fig, update, frames=len(todos_frames),
                         interval=interval, blit=False)
    plt.show()

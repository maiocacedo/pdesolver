"""Configuração compartilhada da suíte.

Garante backend matplotlib headless (Agg) e torna `Tests/` importável para
os módulos auxiliares (`import _helpers`), independente do import-mode.
"""

import os
import sys

# Torna helpers do diretório de testes importáveis por nome simples.
sys.path.insert(0, os.path.dirname(__file__))

# Backend headless antes de qualquer import de matplotlib nos testes.
import matplotlib  # noqa: E402

matplotlib.use("Agg", force=True)

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _force_agg():
    """Reforça Agg por teste.

    Importar `pdesolver.Auxs.Visualize` roda `_select_backend()`, que pode
    trocar o backend para TkAgg/QtAgg em máquinas com essas libs. Reforçar
    aqui mantém os testes headless e determinísticos.
    """
    matplotlib.use("Agg", force=True)
    yield

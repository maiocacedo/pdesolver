"""Métricas numéricas compartilhadas pelos testes."""

import numpy as np


def mae(numerica, analitica):
    """Erro absoluto médio entre dois campos, achatados."""
    a = np.asarray(numerica, dtype=float).ravel()
    b = np.asarray(analitica, dtype=float).ravel()
    return float(np.mean(np.abs(a - b)))


def rmse(numerica, analitica):
    """Raiz do erro quadrático médio entre dois campos, achatados."""
    a = np.asarray(numerica, dtype=float).ravel()
    b = np.asarray(analitica, dtype=float).ravel()
    return float(np.sqrt(np.mean((a - b) ** 2)))

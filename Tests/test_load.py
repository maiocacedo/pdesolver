import os
import sys
import time

import numpy as np

diretorio_atual = os.path.dirname(os.path.abspath(__file__))
diretorio_pai = os.path.abspath(os.path.join(diretorio_atual, ".."))
sys.path.append(diretorio_pai)

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from PDE import PDE
from PDES import PDES

PDES1 = PDES.load_from_json("pdes1.json")

print(PDES1.funcs)
PDES1.disc_n = [10, 10]
print(PDES1.disc_n)
print(PDES1.ic)
print(PDES1.pdes[0].west_func_bd)

PDES1.discretize(method="central")
PDES1.solve(tf=0.1, nt=10)


PDES1.visualize()

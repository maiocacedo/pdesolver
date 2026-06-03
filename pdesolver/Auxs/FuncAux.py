import re

import sympy as sp
from sympy.parsing.sympy_parser import parse_expr

def symbol_references(in_list):
    slist = []

    for e in in_list:
        globals()[e] = sp.Symbol(e)
        slist.append(e)
    return slist

def d_dt(expr_str: str) -> str:
    t = sp.Symbol('t')
    try:
        e = parse_expr(expr_str, evaluate=False)
        return str(sp.diff(e, t))
    except Exception:
        return expr_str

def repl_symbol(expr: str, sym: str, repl: str) -> str:
    pattern = rf'(?<![A-Za-z0-9_]){sym}(?![A-Za-z0-9_])'
    return re.sub(pattern, repl, expr)

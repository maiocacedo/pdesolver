import re

import sympy as sp
from sympy.parsing.sympy_parser import parse_expr

def build_func_map(xp):
    return {
        "sin": xp.sin,
        "cos": xp.cos,
        "tan": xp.tan,
        "asin": xp.arcsin,
        "acos": xp.arccos,
        "atan": xp.arctan,
        "atan2": xp.arctan2,
        "sinh": xp.sinh,
        "cosh": xp.cosh,
        "tanh": xp.tanh,
        "exp": xp.exp,
        "log": xp.log,
        "sqrt": xp.sqrt,
        "Abs": xp.abs,
        "sign": xp.sign,
        "Max": xp.maximum,
        "Min": xp.minimum,
        "mod": xp.mod,
        "floor": xp.floor,
        "ceil": xp.ceil,
        "sech": lambda x: 1.0 / xp.cosh(x),
    }

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

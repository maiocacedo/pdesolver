from typing import List

from .stability import cell_peclet, stability_limit
from .truncation import modified_equation, operator_terms, physical_coefficients


def _fmt(value) -> str:
    if value is None:
        return "-"
    if value == float("inf"):
        return "infinito"
    return f"{value:.3e}"


def _numerical_diffusion(operator) -> List[dict]:
    fisico = {}
    for c in physical_coefficients(operator):
        if c["suffix"] in ("_xx", "_yy"):
            eixo = "x" if c["suffix"] == "_xx" else "y"
            fisico[(c["eq"], c["func"], eixo)] = abs(c["value"])

    out = []
    for extra in modified_equation(operator):
        if extra["derivative_order"] != 2 or extra["signed"] is None:
            continue
        eixo = "x" if extra["from"].endswith("dx") else "y"
        ref = fisico.get((extra["eq"], 0, eixo), 0.0)
        out.append({
            "eq": extra["eq"],
            "from": extra["from"],
            "numerical": extra["signed"],
            "physical": ref,
            "ratio": (abs(extra["signed"]) / ref) if ref > 0 else None,
        })
    return out


def analyze(operator, method: str = "RKF") -> dict:
    return {
        "terms": operator_terms(operator),
        "modified": modified_equation(operator),
        "numerical_diffusion": _numerical_diffusion(operator),
        "stability": stability_limit(operator, method=method),
        "peclet": cell_peclet(operator),
    }


def report_text(operator, method: str = "RKF") -> str:
    data = analyze(operator, method=method)
    linhas = []
    add = linhas.append

    add("  [análise] Discretização")
    vistos = set()
    for t in data["terms"]:
        chave = (t["label"], t["order"])
        if chave in vistos:
            continue
        vistos.add(chave)
        add(f"    {t['label']:<10s} esquema '{operator.method}' — erro "
            f"O(h^{t['sym_order']}), termo líder "
            f"{t['sym_coeff']}*u^({t['sym_k']})")
        if t["mesh_k"] == t["sym_k"]:
            add(f"      na malha atual: |coef| máximo = "
                f"{_fmt(t['mesh_coeff'])}")
        elif t["mesh_k"] >= 0:
            add(f"      ATENÇÃO: nesta malha o termo líder é u^({t['mesh_k']})"
                f" com |coef| = {_fmt(t['mesh_coeff'])} — o estiramento "
                f"degradou a ordem")

    if data["numerical_diffusion"]:
        add("  [análise] Difusão numérica introduzida pelo esquema")
        for d in data["numerical_diffusion"]:
            sinal = "adiciona" if d["numerical"] > 0 else "SUBTRAI"
            valor = _fmt(abs(d["numerical"]))
            if d["ratio"] is None:
                add(f"    de {d['from']}: {sinal} {valor} "
                    f"(equação sem difusão física neste eixo)")
            else:
                add(f"    de {d['from']}: {sinal} {valor} vs física "
                    f"{_fmt(d['physical'])} ({100*d['ratio']:.1f}%)")
            if d["numerical"] < 0:
                add("      ATENÇÃO: difusão negativa — esquema "
                    "antidifusivo, tende a ser instável")

    est = data["stability"]
    add("  [análise] Estabilidade")
    if est["unconditional"]:
        add(f"    {est['method']}: {est['note']}")
    else:
        if not est["linear"]:
            add("    AVISO: termos não lineares ignorados na análise — "
                "o limite abaixo vale para a parte linear")
        add(f"    {est['method']}: dt_max = {_fmt(est['dt_max'])} "
            f"(símbolo de Fourier, |z| = {abs(est['real_axis_limit']):.4f})")
        if "dt_max_spectral" in est:
            add(f"    espectro do operador montado: dt_max = "
                f"{_fmt(est['dt_max_spectral'])}")
        if est.get("unstable_mode"):
            add(f"    ATENÇÃO: o operador espacial tem modo com "
                f"crescimento Re(λ) = {est['growth_rate']:.3e} > 0 — "
                f"nenhum dt torna o esquema estável")

    avisos = [p for p in data["peclet"] if p["peclet"] > 1.0]
    if avisos and operator.method == "central":
        add("  [análise] Avisos")
        for p in avisos:
            add(f"    Péclet de célula em {p['axis']} = {p['peclet']:.2f} > 1 "
                f"— diferenças centrais tendem a oscilar; considere "
                f"'backward' ou refinar a malha")

    return "\n".join(linhas)

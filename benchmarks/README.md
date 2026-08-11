# Benchmark suite

Reproducibility material for the accuracy, performance and capability claims.

## Contents

| File | Purpose |
|:-----|:--------|
| `benchmark_completo.py` | Full suite: reference problems, capability matrix, scaling |
| `colab_benchmark.ipynb` | Google Colab notebook — accuracy, capabilities and **GPU** |
| `bench_scaling.py` | Focused scaling study (setup cost, RHS cost, CPU vs GPU) |
| `Dockerfile` | Pinned software stack for reproducible **accuracy** figures |
| `artigo/` | The original per-case scripts, adapted to run both backends |

## Where to run what

| Measurement | Colab | Bare metal | Docker |
|:---|:---:|:---:|:---:|
| RMSE / accuracy | yes | yes | yes |
| Capability matrix | yes | yes | yes |
| CPU vs GPU | **yes** (free T4) | only with a CUDA device | no |
| **Absolute CPU timing** | **no** | **yes** (idle) | no |

Colab runs on a shared, preempted 2-vCPU VM whose processor model changes
between sessions, so absolute wall times there are not publishable — but it is
the only way a reviewer without a CUDA device can verify the GPU claims. Quote
CPU timings from an idle local machine and always report the calibration
figure alongside them.

## Running

```bash
python benchmarks/benchmark_completo.py                      # everything
python benchmarks/benchmark_completo.py --parte capacidades  # capability matrix only
python benchmarks/benchmark_completo.py --runs 50 --json r.json
```

## Timing methodology

Wall-clock benchmarking is only meaningful on an **idle machine**. The suite:

- discards two warm-up runs, then times `--runs` repetitions;
- reports the **median** and the **minimum** rather than the mean, since the
  mean is dominated by outliers from background load;
- reports the coefficient of variation and marks any measurement with
  `CV > 5%` using `!`. **Do not publish a figure carrying that mark.**
- prints a **calibration** number — the median time of a 600×600 matrix
  product — so timings can be compared across machines.

A container or VM pins library versions, which makes *accuracy* figures exactly
reproducible. It does **not** make timings comparable: virtualization adds
jitter and does not remove host contention. Use the Dockerfile for RMSE
reproducibility and bare metal for timing.

## A note on the FiPy time scheme

FiPy is run under two schemes, because they are not interchangeable:

```python
# FiPy/Euler — fully implicit, first order in time
eq = TransientTerm() == DiffusionTerm(coeff=alpha)

# FiPy/CN — Crank-Nicolson, second order in time
eq = (TransientTerm() == DiffusionTerm(coeff=0.5*alpha)
      + 0.5*alpha*phi.faceGrad.divergence)
```

`TransientTerm() == DiffusionTerm(...)` solved with `eq.solve()` is **backward
Euler**, not Crank-Nicolson. Comparing a second-order method against it
overstates the accuracy difference by roughly the ratio of temporal orders. On
the 1D heat problem the same FiPy code gives `7.74e-4` under Euler and
`1.33e-4` under a real Crank-Nicolson — the latter matching pdesolver's
`1.35e-4`. Only the second-order pairing is a fair accuracy comparison; the
execution-time gap is large under either scheme.

## Capability matrix

Each entry is classified as:

- **declarativo** — expressible in the library's own notation, and the result
  is checked (against a closed-form solution, or cross-checked against another
  library that can also express it);
- **manual** — achievable, but only by dropping to hand-written numerics;
- **ausente** — no syntax or term exists for it.

Where two libraries can both express a feature, the suite verifies that they
**agree**, rather than asserting that one is better.

## Comparing across machines

Report, alongside any timing table:

1. the `CPU` and `Plataforma` lines the suite prints;
2. the `Calibracao` figure;
3. the library versions block;
4. the CV of every timing quoted.

Two timing tables without those four items are not comparable.

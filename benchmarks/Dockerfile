# Reproducible software stack for the pdesolver benchmark suite.
#
# This pins library versions so that accuracy figures (RMSE) are exactly
# reproducible on any host. It does NOT make wall-clock timings comparable
# across machines: container overhead and host load both perturb timing.
# For publication timings, run on an idle bare-metal machine and report the
# calibration figure the benchmark prints.
#
#   docker build -t pdesolver-bench -f benchmarks/Dockerfile .
#   docker run --rm pdesolver-bench --parte capacidades
#   docker run --rm pdesolver-bench --runs 50 --json /tmp/r.json

FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    MPLBACKEND=Agg \
    PYTHONUNBUFFERED=1 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir \
        "numpy==2.4.2" \
        "scipy==1.16.2" \
        "sympy==1.14.0" \
        "matplotlib==3.10.7" \
        "fipy==4.0.2" \
        "py-pde==0.56.0"

COPY pyproject.toml README.md ./
COPY pdesolver ./pdesolver
COPY benchmarks ./benchmarks

RUN pip install --no-cache-dir --no-deps -e .

ENTRYPOINT ["python", "benchmarks/benchmark_completo.py"]
CMD ["--parte", "tudo"]

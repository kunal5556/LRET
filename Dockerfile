##########################################################
# Multi-Stage Dockerfile for QLRET (Phase 6a)
# Stages: cpp-builder -> python-builder -> tester -> runtime
# Usage:
#   docker build -t qlret:latest .
#   docker run --rm qlret:latest ./quantum_sim -n 6 -d 8 --mode sequential
#   docker run --rm qlret:latest python -c "from qlret import simulate_json; print('OK')"
##########################################################

# --------------------------------------------------------
# Stage 1: C++ builder (build quantum_sim + _qlret_native)
# --------------------------------------------------------
FROM python:3.11-slim AS cpp-builder

ENV DEBIAN_FRONTEND=noninteractive

# Build dependencies (git/ca-certificates for FetchContent)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    ca-certificates \
    libeigen3-dev \
    libomp-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy source with caching-friendly order
COPY CMakeLists.txt .
COPY include/ include/
COPY src/ src/
COPY python/ python/
COPY main.cpp .
COPY tests/ tests/
COPY test_noise_import.cpp .
COPY test_fidelity.cpp .
COPY test_minimal.cpp .

# Configure and build (enable Python bindings)
RUN mkdir -p build && cd build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DUSE_PYTHON=ON && \
    cmake --build . -- -j$(nproc)

# --------------------------------------------------------
# Stage 2: Python builder (install qlret + deps)
# --------------------------------------------------------
FROM python:3.11-slim AS python-builder

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Runtime deps needed by numpy/PennyLane wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    libomp5 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python dependencies
RUN pip install --no-cache-dir --upgrade pip wheel setuptools
RUN pip install --no-cache-dir \
    numpy>=1.20 \
    scipy \
    matplotlib \
    pennylane>=0.30 \
    pytest \
    pytest-cov

# Copy Python package sources
COPY python/ python/

# Bring in native module from cpp-builder
COPY --from=cpp-builder /app/python/qlret/_qlret_native*.so python/qlret/

# Install qlret (includes dev extras for tests)
RUN pip install --no-cache-dir ./python[dev]

# Ensure native module lives inside installed package
RUN python - <<"PY"
import shutil
import pathlib
import qlret

dest = pathlib.Path(qlret.__file__).parent
for so in pathlib.Path('/app/python/qlret').glob('_qlret_native*.so'):
    shutil.copy2(so, dest / so.name)
print(f"Native module installed to: {dest}")
PY

# Quick import check
RUN python - <<"PY"
import qlret
from qlret import simulate_json
print("qlret version", qlret.__version__)
PY

# --------------------------------------------------------
# Stage 3: Tester (run pytest to gate the image)
# --------------------------------------------------------
FROM python-builder AS tester

WORKDIR /app

# Copy C++ executable from builder
COPY --from=cpp-builder /app/build/quantum_sim /usr/local/bin/quantum_sim
ENV PATH="/usr/local/bin:${PATH}"

# Bring tests and samples
COPY python/tests/ tests/
COPY samples/ samples/

# Run integration tests (fail build on errors)
RUN pytest tests/ -v --tb=short

# --------------------------------------------------------
# Stage 4: Runtime (minimal, flexible entrypoint)
# --------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Runtime libs only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libomp5 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy executables and Python environment
COPY --from=cpp-builder /app/build/quantum_sim /usr/local/bin/quantum_sim
COPY --from=cpp-builder /app/build/test_* /usr/local/bin/
COPY --from=cpp-builder /app/python/qlret/_qlret_native*.so /tmp/
COPY --from=python-builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
# Note: Copy python-builder /usr/local/bin EXCEPT we need to preserve quantum_sim
RUN mkdir -p /app/tmp_bin
COPY --from=python-builder /usr/local/bin /app/tmp_bin
RUN cp -rn /app/tmp_bin/* /usr/local/bin/ 2>/dev/null || true && rm -rf /app/tmp_bin

# Place native module with installed package
RUN python - <<"PY"
import pathlib, shutil
import site

site_packages = pathlib.Path(site.getsitepackages()[0])
qlret_dir = site_packages / 'qlret'
qlret_dir.mkdir(parents=True, exist_ok=True)
for so in pathlib.Path('/tmp').glob('_qlret_native*.so'):
    shutil.move(str(so), qlret_dir / so.name)
print(f"Native module moved to {qlret_dir}")
PY

# Copy samples and scripts for convenience
COPY samples/ samples/
COPY scripts/ scripts/

# Workspace for outputs
RUN mkdir -p /app/output

# Default to bash for flexible usage (override in docker run)
CMD ["/bin/bash"]

# Phase 6: Docker Integration & Testing - Strategic Implementation Plan

**Date:** January 3, 2026  
**Phase:** 6 (Final Integration Phase)  
**Duration:** 8-10 hours  
**Complexity:** Medium  
**Risk:** Low (no core code changes, only packaging/testing)

---

## Executive Summary

Phase 6 transforms our Docker container from a single-purpose C++ executor into a comprehensive development and testing environment. This phase adds:
- Multi-stage Docker builds (C++ + Python + Testing)
- Automated integration testing during build
- Performance benchmarking framework
- Documentation and examples
- CI/CD automation via GitHub Actions

**Key Principle:** Zero changes to core simulator code. All work is packaging, testing, and automation.

---

## Table of Contents

1. [Phase 6a: Multi-Stage Dockerfile](#phase-6a-multi-stage-dockerfile)
2. [Phase 6b: Integration Testing](#phase-6b-integration-testing)
3. [Phase 6c: Performance Benchmarking](#phase-6c-performance-benchmarking)
4. [Phase 6d: Documentation](#phase-6d-documentation)
5. [Phase 6e: CI/CD Automation](#phase-6e-cicd-automation)
6. [Implementation Order](#implementation-order)
7. [Risk Mitigation](#risk-mitigation)
8. [Success Criteria](#success-criteria)

---

## Phase 6a: Multi-Stage Dockerfile

### Objective
Create a production-grade multi-stage Dockerfile that supports:
- C++ binary compilation with Python bindings
- Python package installation
- Automated testing
- Multiple execution modes (CLI, Python, Jupyter)

### Current State (Dockerfile.backup-phase5)
```dockerfile
Builder stage (Ubuntu 24.04)
├── Install: cmake, build-essential, eigen, OpenMP
├── Build: quantum_sim binary only
└── Output: Single executable

Runtime stage (Ubuntu 24.04)
├── Copy: quantum_sim binary
├── Python: Basic python3 for scripts only
└── Entry: ./quantum_sim (CLI only)
```

### Target State (New Multi-Stage Dockerfile)
```dockerfile
Stage 1: cpp-builder (Ubuntu 24.04)
├── Install: cmake, build-essential, eigen, OpenMP, Python dev
├── Build: quantum_sim + _qlret_native.so (pybind11 module)
├── CMake flags: -DUSE_PYTHON=ON -DCMAKE_BUILD_TYPE=Release
└── Output: binaries + Python module

Stage 2: python-builder (Python 3.11)
├── Install: pip, wheel, setuptools
├── Install deps: numpy, pennylane, scipy
├── Build: qlret package from python/
└── Output: Installed qlret in site-packages

Stage 3: tester (combines 1+2)
├── Copy: All binaries + Python environment
├── Copy: Test suite from python/tests/
├── Run: pytest python/tests/ -v --tb=short
├── Validation: Exit code 0 = all tests pass
└── Output: Test results (logged)

Stage 4: runtime (minimal final image)
├── Copy: quantum_sim, _qlret_native, qlret package
├── Install: Only runtime deps (no build tools)
├── Optional: Jupyter for interactive mode
├── Entry: Flexible (bash, python, quantum_sim)
└── Size: ~800 MB (vs 500 MB current)
```

### Implementation Steps

#### Step 6a.1: Create cpp-builder stage (30 min)
```dockerfile
FROM ubuntu:24.04 AS cpp-builder

ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    ca-certificates \
    libeigen3-dev \
    libomp-dev \
    python3-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy source
COPY CMakeLists.txt .
COPY include/ include/
COPY src/ src/
COPY python/ python/
COPY main.cpp .

# Build with Python bindings
RUN mkdir -p build && cd build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DUSE_PYTHON=ON \
        -DCMAKE_INSTALL_PREFIX=/app/install && \
    make -j$(nproc) && \
    make install

# Verify outputs
RUN ls -lh build/quantum_sim && \
    ls -lh python/qlret/_qlret_native*.so
```

**Key Points:**
- Add `python3-dev` for pybind11 headers
- Set `USE_PYTHON=ON` to build bindings
- Verify both binary and .so module exist

#### Step 6a.2: Create python-builder stage (30 min)
```dockerfile
FROM python:3.11-slim AS python-builder

WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip wheel setuptools

# Install qlret dependencies
RUN pip install --no-cache-dir \
    numpy>=1.20 \
    pennylane>=0.30 \
    scipy \
    matplotlib \
    pytest \
    pytest-cov

# Copy Python package source
COPY python/ python/

# Copy C++ module from cpp-builder
COPY --from=cpp-builder /app/python/qlret/_qlret_native*.so python/qlret/

# Install qlret package
RUN cd python && pip install -e .[pennylane]

# Verify installation
RUN python -c "import qlret; print(qlret.__version__)" && \
    python -c "from qlret import QLRETDevice; print('Device OK')"
```

**Key Points:**
- Use slim Python image (smaller)
- Copy .so module from cpp-builder
- Verify imports work

#### Step 6a.3: Create tester stage (30 min)
```dockerfile
FROM python-builder AS tester

WORKDIR /app

# Copy quantum_sim binary
COPY --from=cpp-builder /app/build/quantum_sim /app/quantum_sim
ENV PATH="/app:${PATH}"

# Copy test files and samples
COPY python/tests/ tests/
COPY samples/ samples/

# Run integration tests
RUN pytest tests/ -v --tb=short || (echo "Tests failed!" && exit 1)

# Run quick sanity checks
RUN python -c "from qlret import simulate_json, load_json_file; print('API OK')" && \
    ./quantum_sim --version && \
    echo "Sanity checks passed"
```

**Key Points:**
- Combine C++ binary + Python environment
- Run full pytest suite
- Build fails if tests fail (quality gate)

#### Step 6a.4: Create runtime stage (30 min)
```dockerfile
FROM python:3.11-slim AS runtime

ENV DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libomp5 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy binaries from cpp-builder
COPY --from=cpp-builder /app/build/quantum_sim ./quantum_sim
COPY --from=cpp-builder /app/python/qlret/_qlret_native*.so /tmp/

# Copy Python environment from python-builder
COPY --from=python-builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=python-builder /usr/local/bin /usr/local/bin

# Move native module to correct location
RUN mkdir -p /usr/local/lib/python3.11/site-packages/qlret && \
    mv /tmp/_qlret_native*.so /usr/local/lib/python3.11/site-packages/qlret/

# Copy samples and scripts
COPY samples/ samples/
COPY scripts/ scripts/

# Create output directory
RUN mkdir -p /app/output

# Default to bash (flexible entry)
CMD ["/bin/bash"]

# Usage examples in labels
LABEL usage.cli="docker run qlret ./quantum_sim -n 10 -d 20"
LABEL usage.python="docker run qlret python -c 'from qlret import ...'"
LABEL usage.jupyter="docker run -p 8888:8888 qlret jupyter notebook --ip=0.0.0.0"
```

**Key Points:**
- Minimal runtime deps (no build tools)
- Flexible CMD (bash by default)
- Multiple execution modes supported

---

## Phase 6b: Integration Testing

### Objective
Create comprehensive integration tests that validate all execution modes in Docker.

### Test Categories

#### 6b.1: JSON Circuit Tests (30 min)
```python
# tests/integration/test_json_execution.py
import subprocess
import json
import pytest
from pathlib import Path

def test_bell_pair_json_cli():
    """Test quantum_sim with JSON input via CLI."""
    result = subprocess.run(
        ["./quantum_sim", 
         "--input-json", "samples/json/bell_pair.json",
         "--output-json", "/tmp/result.json"],
        capture_output=True,
        text=True,
        timeout=30
    )
    assert result.returncode == 0
    
    # Verify output
    with open("/tmp/result.json") as f:
        data = json.load(f)
    assert data["status"] == "success"
    assert len(data["expectation_values"]) == 2

def test_json_subprocess_backend():
    """Test Python API with subprocess backend."""
    from qlret import simulate_json, load_json_file
    circuit = load_json_file("samples/json/bell_pair.json")
    result = simulate_json(circuit, use_native=False)
    assert result["status"] == "success"

def test_json_native_backend():
    """Test Python API with native bindings."""
    from qlret import simulate_json, load_json_file
    circuit = load_json_file("samples/json/bell_pair.json")
    result = simulate_json(circuit, use_native=True)
    assert result["status"] == "success"
```

#### 6b.2: PennyLane Device Tests (30 min)
```python
# tests/integration/test_pennylane_device.py
import pytest
import numpy as np

pennylane = pytest.importorskip("pennylane")
qml = pennylane

from qlret import QLRETDevice

def test_device_basic_circuit():
    """Test basic circuit execution."""
    dev = QLRETDevice(wires=2, shots=None)
    
    @qml.qnode(dev)
    def circuit():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
    
    result = circuit()
    assert abs(result - 1.0) < 0.1  # Bell state ZZ = 1

def test_device_gradients():
    """Test parameter-shift gradients."""
    dev = QLRETDevice(wires=1, shots=None)
    
    @qml.qnode(dev, diff_method="parameter-shift")
    def circuit(theta):
        qml.RX(theta, wires=0)
        return qml.expval(qml.PauliZ(0))
    
    theta = 0.5
    grad = qml.grad(circuit)(theta)
    expected = -np.sin(theta)
    assert abs(grad - expected) < 0.1

def test_device_sampling():
    """Test shot-based sampling."""
    dev = QLRETDevice(wires=2, shots=100)
    
    @qml.qnode(dev)
    def circuit():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.sample()
    
    samples = circuit()
    assert len(samples) == 100
```

#### 6b.3: CLI Regression Tests (20 min)
```python
# tests/integration/test_cli_regression.py
def test_basic_cli_simulation():
    """Test original CLI mode still works."""
    result = subprocess.run(
        ["./quantum_sim", "-n", "8", "-d", "10", "--mode", "sequential"],
        capture_output=True,
        text=True,
        timeout=60
    )
    assert result.returncode == 0
    assert "Final rank" in result.stdout

def test_parallel_modes():
    """Test all parallel modes."""
    modes = ["sequential", "row", "column", "hybrid"]
    for mode in modes:
        result = subprocess.run(
            ["./quantum_sim", "-n", "6", "-d", "8", "--mode", mode],
            capture_output=True,
            timeout=60
        )
        assert result.returncode == 0
```

#### 6b.4: Test Runner Script (20 min)
```python
# tests/run_integration_tests.py
"""
Master test runner for Docker integration tests.
Usage: python tests/run_integration_tests.py
"""
import pytest
import sys

def main():
    args = [
        "tests/integration/",
        "-v",
        "--tb=short",
        "--durations=10",
        "--color=yes",
        "-ra"  # Show all test results
    ]
    
    exit_code = pytest.main(args)
    
    if exit_code == 0:
        print("\n✅ All integration tests passed!")
    else:
        print(f"\n❌ Tests failed with code {exit_code}")
    
    return exit_code

if __name__ == "__main__":
    sys.exit(main())
```

---

## Phase 6c: Performance Benchmarking

### Objective
Create automated benchmarks comparing QLRET against other simulators.

### Implementation Steps

#### 6c.1: Benchmark Framework (45 min)
```python
# benchmarks/framework.py
import time
import numpy as np
from typing import Dict, List, Callable
import pennylane as qml

class BenchmarkSuite:
    """Framework for running comparative benchmarks."""
    
    def __init__(self):
        self.results = []
    
    def run_benchmark(
        self,
        name: str,
        circuit_fn: Callable,
        devices: List[str],
        num_runs: int = 5
    ) -> Dict:
        """Run circuit on multiple devices and measure performance."""
        results = {}
        
        for device_name in devices:
            times = []
            
            for _ in range(num_runs):
                dev = qml.device(device_name, wires=circuit_fn.__code__.co_argcount)
                qnode = qml.QNode(circuit_fn, dev)
                
                start = time.perf_counter()
                result = qnode()
                elapsed = time.perf_counter() - start
                
                times.append(elapsed)
            
            results[device_name] = {
                "mean": np.mean(times),
                "std": np.std(times),
                "min": np.min(times),
                "max": np.max(times)
            }
        
        self.results.append({"name": name, "devices": results})
        return results
    
    def print_summary(self):
        """Print benchmark results in table format."""
        print("\n" + "="*80)
        print("QLRET Performance Benchmark Results")
        print("="*80)
        
        for bench in self.results:
            print(f"\n{bench['name']}:")
            print("-" * 60)
            print(f"{'Device':<20} {'Mean (ms)':<15} {'Std (ms)':<15}")
            print("-" * 60)
            
            for device, stats in bench["devices"].items():
                print(f"{device:<20} {stats['mean']*1000:<15.3f} {stats['std']*1000:<15.3f}")
```

#### 6c.2: Standard Benchmark Circuits (30 min)
```python
# benchmarks/circuits.py
import pennylane as qml

def ghz_state(n_qubits: int):
    """GHZ state preparation circuit."""
    @qml.qnode(qml.device("default.qubit", wires=n_qubits))
    def circuit():
        qml.Hadamard(wires=0)
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        return qml.expval(qml.PauliZ(0))
    return circuit

def qaoa_circuit(n_qubits: int):
    """QAOA-inspired circuit with parameterization."""
    @qml.qnode(qml.device("default.qubit", wires=n_qubits))
    def circuit(gamma, beta):
        # Initial state
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
        
        # Problem Hamiltonian
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
            qml.RZ(gamma, wires=i + 1)
            qml.CNOT(wires=[i, i + 1])
        
        # Mixer Hamiltonian
        for i in range(n_qubits):
            qml.RX(beta, wires=i)
        
        return qml.expval(qml.PauliZ(0))
    return circuit

def random_circuit(n_qubits: int, depth: int):
    """Random circuit for stress testing."""
    @qml.qnode(qml.device("default.qubit", wires=n_qubits))
    def circuit():
        for layer in range(depth):
            for q in range(n_qubits):
                qml.RX(0.1 * layer, wires=q)
                qml.RY(0.2 * layer, wires=q)
            for q in range(n_qubits - 1):
                qml.CNOT(wires=[q, q + 1])
        return qml.expval(qml.PauliZ(0))
    return circuit
```

#### 6c.3: Benchmark Runner (30 min)
```python
# benchmarks/run_benchmarks.py
from framework import BenchmarkSuite
from circuits import ghz_state, qaoa_circuit, random_circuit

def main():
    suite = BenchmarkSuite()
    
    # Devices to compare
    devices = ["qlret", "default.qubit"]
    
    # Benchmark 1: GHZ states of varying sizes
    for n in [4, 6, 8, 10]:
        suite.run_benchmark(
            name=f"GHZ-{n}",
            circuit_fn=ghz_state(n),
            devices=devices,
            num_runs=5
        )
    
    # Benchmark 2: QAOA circuits
    for n in [4, 6, 8]:
        suite.run_benchmark(
            name=f"QAOA-{n}",
            circuit_fn=qaoa_circuit(n),
            devices=devices,
            num_runs=3
        )
    
    # Benchmark 3: Random circuits (depth sweep)
    n = 8
    for d in [10, 20, 30]:
        suite.run_benchmark(
            name=f"Random-n{n}-d{d}",
            circuit_fn=random_circuit(n, d),
            devices=devices,
            num_runs=3
        )
    
    # Print results
    suite.print_summary()
    
    # Save results
    import json
    with open("/app/output/benchmark_results.json", "w") as f:
        json.dump(suite.results, f, indent=2)

if __name__ == "__main__":
    main()
```

---

## Phase 6d: Documentation

### Objective
Create comprehensive documentation for all usage modes.

### Documentation Files

#### 6d.1: Docker Quick Start (30 min)
```markdown
# DOCKER_QUICKSTART.md
Quick reference for using QLRET Docker image.

## Installation
```bash
docker pull ghcr.io/kunal5556/qlret:latest
# or build from source
docker build -t qlret .
```

## Usage Modes

### 1. CLI Mode (Original)
```bash
docker run qlret ./quantum_sim -n 10 -d 20 --mode compare
```

### 2. JSON Mode
```bash
docker run -v $(pwd)/circuits:/data qlret \
    ./quantum_sim --input-json /data/circuit.json
```

### 3. Python Interactive
```bash
docker run -it qlret python
>>> from qlret import QLRETDevice
>>> import pennylane as qml
>>> ...
```

### 4. Jupyter Notebooks
```bash
docker run -p 8888:8888 qlret \
    jupyter notebook --ip=0.0.0.0 --no-browser --allow-root
```

## Volume Mounts
- `/app/output` - For CSV/JSON outputs
- `/app/samples` - Sample circuits (read-only)
```

#### 6d.2: JSON Schema Reference (30 min)
```json
// docs/JSON_SCHEMA.json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "QLRET Circuit Specification",
  "type": "object",
  "required": ["circuit"],
  "properties": {
    "circuit": {
      "type": "object",
      "required": ["num_qubits", "operations"],
      "properties": {
        "num_qubits": {
          "type": "integer",
          "minimum": 1,
          "description": "Number of qubits in the circuit"
        },
        "operations": {
          "type": "array",
          "items": {
            "type": "object",
            "required": ["name", "wires"],
            "properties": {
              "name": {
                "type": "string",
                "enum": ["H", "X", "Y", "Z", "S", "T", "RX", "RY", "RZ", "CNOT", "CZ", "SWAP"]
              },
              "wires": {
                "type": "array",
                "items": {"type": "integer"}
              },
              "params": {
                "type": "array",
                "items": {"type": "number"}
              }
            }
          }
        },
        "observables": {
          "type": "array",
          "items": {
            "oneOf": [
              {
                "type": "object",
                "required": ["type", "operator", "wires"],
                "properties": {
                  "type": {"const": "PAULI"},
                  "operator": {"enum": ["X", "Y", "Z", "I"]},
                  "wires": {"type": "array", "items": {"type": "integer"}},
                  "coefficient": {"type": "number", "default": 1.0}
                }
              }
            ]
          }
        }
      }
    },
    "config": {
      "type": "object",
      "properties": {
        "epsilon": {"type": "number", "default": 1e-4},
        "initial_rank": {"type": "integer", "default": 1},
        "shots": {"type": "integer", "minimum": 1}
      }
    }
  }
}
```

#### 6d.3: Python API Reference (20 min)
Create: `docs/PYTHON_API.md`

#### 6d.4: Performance Tuning Guide (20 min)
Create: `docs/PERFORMANCE_TUNING.md`

---

## Phase 6e: CI/CD Automation

### Objective
Automate building, testing, and deployment via GitHub Actions.

### Implementation Steps

#### 6e.1: Build & Test Workflow (45 min)
```yaml
# .github/workflows/build-test.yml
name: Build and Test

on:
  push:
    branches: [ main, feature/* ]
  pull_request:
    branches: [ main ]

jobs:
  build-test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2
    
    - name: Build Docker image
      uses: docker/build-push-action@v4
      with:
        context: .
        target: runtime
        tags: qlret:test
        cache-from: type=gha
        cache-to: type=gha,mode=max
    
    - name: Run integration tests
      run: |
        docker run qlret:test pytest tests/integration/ -v
    
    - name: Run benchmarks
      run: |
        docker run qlret:test python benchmarks/run_benchmarks.py
```

#### 6e.2: Docker Publish Workflow (30 min)
```yaml
# .github/workflows/docker-publish.yml
name: Publish Docker Image

on:
  release:
    types: [published]
  push:
    tags:
      - 'v*'

jobs:
  publish:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Log in to GitHub Container Registry
      uses: docker/login-action@v2
      with:
        registry: ghcr.io
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v4
      with:
        images: ghcr.io/${{ github.repository }}
        tags: |
          type=semver,pattern={{version}}
          type=semver,pattern={{major}}.{{minor}}
          type=raw,value=latest
    
    - name: Build and push
      uses: docker/build-push-action@v4
      with:
        context: .
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
```

---

## Implementation Order

### Week 1: Core Docker Infrastructure
1. **Day 1-2:** Phase 6a (Multi-stage Dockerfile)
   - cpp-builder stage
   - python-builder stage
   - tester stage
   - runtime stage
   - Test locally

2. **Day 3:** Phase 6b.1-6b.3 (Integration Tests)
   - JSON circuit tests
   - PennyLane device tests
   - CLI regression tests

### Week 2: Benchmarking & Automation
3. **Day 4:** Phase 6c (Performance Benchmarking)
   - Benchmark framework
   - Standard circuits
   - Benchmark runner

4. **Day 5:** Phase 6d (Documentation)
   - Docker quickstart
   - JSON schema
   - Python API docs
   - Performance tuning

5. **Day 6:** Phase 6e (CI/CD)
   - GitHub Actions workflows
   - Automated testing
   - Docker publishing

---

## Risk Mitigation

### Risk 1: Docker Build Failures
**Mitigation:**
- Test each stage independently
- Use `docker build --target <stage>` for debugging
- Add verbose logging in RUN commands

### Risk 2: Test Failures in Docker
**Mitigation:**
- Run tests locally first
- Use `docker run -it <image> bash` to debug interactively
- Add `|| true` for non-critical tests initially

### Risk 3: Performance Regression
**Mitigation:**
- Baseline current performance before changes
- Track benchmark results over time
- Alert on >10% degradation

### Risk 4: Large Docker Image Size
**Mitigation:**
- Use multi-stage builds (only runtime deps in final)
- Remove build artifacts
- Use .dockerignore file
- Target: <1 GB final image

---

## Success Criteria

### Phase 6a: Dockerfile
- ✅ Docker builds successfully with all 4 stages
- ✅ quantum_sim binary works in runtime stage
- ✅ Python qlret package imports successfully
- ✅ _qlret_native.so loads without errors
- ✅ Final image <1 GB

### Phase 6b: Testing
- ✅ All pytest tests pass in tester stage
- ✅ JSON circuit execution works (CLI + Python)
- ✅ PennyLane device executes circuits
- ✅ Gradients compute correctly
- ✅ CLI regression tests pass

### Phase 6c: Benchmarking
- ✅ Benchmarks run without errors
- ✅ Results saved to JSON
- ✅ Performance within 2x of default.qubit
- ✅ No memory leaks detected

### Phase 6d: Documentation
- ✅ Docker quickstart complete
- ✅ JSON schema validated
- ✅ Python API documented
- ✅ Performance guide written

### Phase 6e: CI/CD
- ✅ GitHub Actions workflows run
- ✅ Tests pass on CI
- ✅ Docker image publishes
- ✅ Version tags work

---

## Model Recommendations for Each Sub-Phase

### Phase 6a: Multi-Stage Dockerfile
**Model:** GPT-5.1 Codex Max (or Claude Sonnet 4.5)  
**Reason:** Standard Docker patterns, well-understood multi-stage builds

**Switch to Opus 4.5 if:**
- Python module linking issues
- Complex dependency conflicts
- Build failures difficult to debug

### Phase 6b: Integration Testing
**Model:** GPT-5.1 Codex Max  
**Reason:** Test scaffolding, pytest fixtures, straightforward

**Switch to Opus 4.5 if:**
- Test fixtures become complex
- Need sophisticated mocking
- Edge cases in device testing

### Phase 6c: Performance Benchmarking
**Model:** Opus 4.5 ⭐⭐⭐ **RECOMMENDED**  
**Reason:**
- Performance analysis requires deep understanding
- Statistical validation needs expertise
- Edge cases in comparative benchmarks
- Potential numerical issues to debug

### Phase 6d: Documentation
**Model:** GPT-5.1 Codex Max  
**Reason:** Straightforward technical writing

### Phase 6e: CI/CD Automation
**Model:** GPT-5.1 Codex Max  
**Reason:** Standard GitHub Actions YAML patterns

**Switch to Opus 4.5 if:**
- Complex workflow orchestration
- Multi-job dependencies
- Tricky caching strategies

---

## Validation Checklist

Before marking Phase 6 complete:

### Local Testing
- [ ] Docker builds successfully: `docker build -t qlret:test .`
- [ ] All tests pass: `docker run qlret:test pytest tests/ -v`
- [ ] CLI works: `docker run qlret:test ./quantum_sim -n 8 -d 10`
- [ ] Python works: `docker run qlret:test python -c "import qlret"`
- [ ] Benchmarks run: `docker run qlret:test python benchmarks/run_benchmarks.py`

### CI/CD Testing
- [ ] Push to branch triggers workflow
- [ ] Tests pass on GitHub Actions
- [ ] Docker image builds successfully
- [ ] Tags publish to registry

### Documentation Testing
- [ ] All examples in docs run successfully
- [ ] JSON schema validates sample files
- [ ] README updated with Phase 6 info

### Performance Validation
- [ ] QLRET matches expected performance
- [ ] No degradation vs Phase 5
- [ ] Memory usage acceptable

---

## Post-Phase 6 Roadmap

After Phase 6 completion, project is ready for:

1. **Public Release**
   - PyPI package publication
   - Docker Hub official images
   - GitHub Releases with binaries

2. **Performance Optimization**
   - Profile and optimize hot paths
   - GPU acceleration testing
   - MPI cluster benchmarks

3. **Advanced Features**
   - Tensor network integration
   - Adaptive truncation
   - Custom observable support

4. **Publications**
   - Technical paper on low-rank method
   - Benchmarking study vs competitors
   - Integration guide for ML researchers

---

## Summary

Phase 6 transforms QLRET from a research tool into a production-grade platform:

**Before Phase 6:**
- Single execution mode (CLI)
- Manual testing required
- No benchmarking data
- Basic documentation

**After Phase 6:**
- 4 execution modes (CLI, JSON, Python, Jupyter)
- Automated testing in CI/CD
- Performance tracking vs competitors
- Comprehensive documentation

**Effort:** 8-10 hours  
**Risk:** Low (no core code changes)  
**Impact:** High (enables wide adoption)

**Next Step:** Begin implementation with Phase 6a (Multi-Stage Dockerfile)

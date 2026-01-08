# Complete LRET Testing Roadmap: Tiers 1-11

Based on TESTING_BACKLOG.md (4405 lines) and exploration of existing infrastructure, here's the comprehensive roadmap integrating **all phases** (1-9) with special attention to Phases 2, 3, and 7.

---

## Executive Summary

| Tier | Phase | Focus | Duration | Status | Hardware |
|------|-------|-------|----------|--------|----------|
| **1** | 1 | Core LRET Tests | 1-2h | ✅ Ready | None |
| **2** | 4 | Noise & Calibration | 2-3h | ⏳ Ready | None |
| **3** | 5 | Python Integration | 2-3h | ⏳ Ready | None |
| **4** | 9.1 | Core QEC | 1-2h | ✅ Passing | None |
| **5** | 9.2 | Distributed QEC | 1.5-2h | ⏳ Disabled | None |
| **6** | 9.3 | Adaptive QEC | 1-1.5h | ⏳ Disabled | None |
| **7** | 3 | MPI/Distributed | 1-2h | ⏳ Optional | MPI Library |
| **8** | 2 | GPU Acceleration | 1-2h | ⏳ Optional | CUDA + cuQuantum |
| **9** | 7 | Benchmarking | 1-2h | ✅ Ready | None |
| **10** | 6 | Docker & CI | 1-2h | ⏳ Ready | Docker |
| **11** | 0 | Documentation | 1-2h | ⏳ Ready | None |

**Total Time (Mandatory Tiers 1-6, 9-10):** ~15-17 hours  
**Total Time (With Optional GPU/MPI):** ~18-21 hours  

---

## TIER 1: Phase 1 - Core LRET Tests

**Duration:** 1-2 hours  
**Purpose:** Validate basic LRET simulator functionality  
**Dependencies:** None  
**Current Status:** ✅ Tests exist, should pass

### Test Files:

```
test_simple.cpp          - Basic circuit simulation
test_fidelity.cpp        - Fidelity validation vs reference
test_minimal.cpp         - Minimal debug test
main.cpp (quantum_sim)   - CLI executable testing
tests/demo_batch.cpp     - Batch heuristic testing
```

### Execution:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

./build/test_simple
./build/test_fidelity
./build/test_minimal
./quantum_sim -n 6 -d 8
./build/demo_batch
```

### Success Criteria:

- ✅ All 5 tests pass
- ✅ Fidelity matches expected values within tolerance
- ✅ CLI executes without errors
- ✅ Batch output is valid

### Expected Time:

- Build: 30s
- Execution: 2-3 minutes
- Verification: 5 minutes

---

## TIER 2: Phase 4 - Noise Model & Calibration Tests

**Duration:** 2-3 hours  
**Purpose:** Validate noise model import, calibration, and advanced noise effects  
**Dependencies:** Tier 1 (core library)  
**Current Status:** ⏳ Infrastructure exists; needs implementation/testing

### Test Files & Scripts:

```
C++ Tests:
  tests/test_advanced_noise.cpp          - Time-varying, correlated, memory effects
  tests/test_leakage_measurement.cpp     - Leakage & readout errors

Python Scripts:
  scripts/test_noise_import.py            - Qiskit noise model loading
  scripts/calibrate_noise_model.py        - End-to-end calibration pipeline
  scripts/fit_depolarizing.py             - Fit depolarizing parameters
  scripts/fit_t1_t2.py                    - Fit relaxation parameters
  scripts/fit_correlated_errors.py        - Fit two-qubit correlations
  scripts/fit_time_scaling.py             - Detect time-dependent growth
  scripts/detect_memory_effects.py        - Detect gate-sequence dependencies
```

### Sub-Tasks:

#### 2.1: Noise Import (30 min)
```bash
# Test Qiskit noise model loading
python3 scripts/test_noise_import.py --input scripts/sample_noise_with_leakage.json
# Or download from IBM:
python3 scripts/download_ibm_noise.py --backend ibmq_manila --output ibm_noise.json
```

**Expected:** JSON parses, errors recognized, circuit modified

#### 2.2: Calibration Scripts (1h)
```bash
# Generate synthetic calibration data
python3 scripts/generate_calibration_data.py --depths 5,10,15,20,25,30 --trials 10

# Fit models
python3 scripts/fit_depolarizing.py calibration_data.csv
python3 scripts/fit_t1_t2.py relaxation_data.csv
python3 scripts/fit_correlated_errors.py correlated_data.csv

# Full pipeline
python3 scripts/calibrate_noise_model.py --input calibration_data.csv
```

**Expected:** R² > 0.95, parameters physically reasonable

#### 2.3: Advanced Noise C++ Tests (30 min)
```bash
cmake --build build --target test_advanced_noise
./build/test_advanced_noise

cmake --build build --target test_leakage_measurement
./build/test_leakage_measurement
```

**Expected:** Time scaling, correlations, memory effects all work; fidelity degrades appropriately

### Success Criteria:

- ✅ All Qiskit error types recognized
- ✅ Calibration R² > 0.95 for all models
- ✅ T2 ≤ 2*T1 (physical constraint)
- ✅ C++ tests pass
- ✅ Fidelity with noise < without noise
- ✅ All CSV/JSON outputs valid

### Expected Duration Breakdown:

- Noise import: 15 min
- Data generation: 30 min
- Calibration fitting: 45 min
- C++ tests: 20 min
- Verification: 15 min

---

## TIER 3: Phase 5 - Python Integration Tests

**Duration:** 2-3 hours  
**Purpose:** Validate Python package, PennyLane device, JAX/PyTorch bindings  
**Dependencies:** Tier 1 (core library)  
**Current Status:** ⏳ Framework exists; needs execution/validation

### Python Package Installation:

```bash
cd python
pip install -e .[dev]
python -c "import qlret; print(qlret.__version__)"
```

### Test Files:

```
python/tests/test_qlret_device.py         - 15 comprehensive tests
python/tests/test_jax_interface.py        - JAX integration (optional)
python/tests/test_pytorch_interface.py    - PyTorch integration (optional)
python/tests/test_ml_integration.py       - VQE, QAOA examples (optional)
```

### Sub-Tasks:

#### 3.1: Package & API (45 min)
```bash
cd python
pytest tests/test_qlret_device.py -v --tb=short
# Expected: 15 tests pass
```

#### 3.2: PennyLane Device (30 min)
```python
import pennylane as qml
from qlret import QLRETDevice

dev = QLRETDevice(wires=2)
@qml.qnode(dev)
def circuit():
    qml.Hadamard(0)
    qml.CNOT([0, 1])
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

result = circuit()
print(f"Bell state expectation: {result}")  # Should be ~0
```

#### 3.3: JSON API (30 min)
```python
from qlret import simulate_json

circuit = {
    "circuit": {
        "num_qubits": 2,
        "operations": [
            {"name": "H", "wires": [0]},
            {"name": "CNOT", "wires": [0, 1]}
        ],
        "observables": [{"type": "PAULI", "operator": "Z", "wires": [0, 1]}]
    }
}
result = simulate_json(circuit)
print(result['expectation_values'][0])  # Should be ~1.0 for Bell
```

#### 3.4: ML Frameworks (45 min, Optional)
```bash
# JAX tests (if JAX installed)
pytest tests/test_jax_interface.py -v --tb=short

# PyTorch tests (if PyTorch installed)
pytest tests/test_pytorch_interface.py -v --tb=short

# Integration tests (if both installed)
pytest tests/test_ml_integration.py -v --tb=short
```

### Success Criteria:

- ✅ Package installation succeeds
- ✅ All 15 core tests pass
- ✅ PennyLane device creates circuits correctly
- ✅ JSON API returns valid results
- ✅ Gradients correct within 1e-4 (ML tests)
- ✅ JAX/PyTorch optional (skip if not installed)

### Expected Duration Breakdown:

- Installation: 10 min
- Core API tests: 20 min
- PennyLane: 15 min
- JSON API: 15 min
- ML frameworks: 30 min (optional)
- Verification: 15 min

---

## TIER 4: Phase 9.1 - Core QEC Tests

**Duration:** 1-2 hours  
**Purpose:** Validate stabilizer codes, syndrome extraction, decoding, logical operations  
**Dependencies:** Tier 1 (core library)  
**Current Status:** ✅ **Currently passing (60+ tests)**

### Test Files:

```
tests/test_qec_stabilizer.cpp   - Pauli algebra, code generation (4/5 tests passing)
tests/test_qec_syndrome.cpp     - Syndrome extraction (15/15 tests passing)
tests/test_qec_decoder.cpp      - MWPM, Union-Find decoders (15/15 tests passing)
tests/test_qec_logical.cpp      - Logical qubits, QEC rounds (24/24 tests passing)
```

### Execution:

```bash
cmake --build build --target test_qec_stabilizer
./build/test_qec_stabilizer
# Expected: 4/5 tests pass (1 pre-existing logic issue)

cmake --build build --target test_qec_syndrome
./build/test_qec_syndrome
# Expected: 15/15 tests pass

cmake --build build --target test_qec_decoder
./build/test_qec_decoder
# Expected: 15/15 tests pass

cmake --build build --target test_qec_logical
./build/test_qec_logical
# Expected: 24/24 tests pass
```

### Success Criteria:

- ✅ test_qec_stabilizer: 4/5 pass (pre-existing issue acceptable)
- ✅ test_qec_syndrome: 15/15 pass
- ✅ test_qec_decoder: 15/15 pass
- ✅ test_qec_logical: 24/24 pass
- ✅ Total: 58+ tests pass

### Expected Duration:

- Execution: 5 minutes
- Verification: 5 minutes

---

## TIER 5: Phase 9.2 - Distributed QEC Tests

**Duration:** 1.5-2 hours  
**Purpose:** Validate distributed QEC across multiple ranks, partition strategies  
**Dependencies:** Tier 4 (core QEC)  
**Current Status:** ⏳ Code complete, disabled in CMakeLists.txt

### Re-enable Tests:

```bash
# Option 1: Edit CMakeLists.txt to enable
# Look for test_qec_distributed and uncomment

# Option 2: Build specific target
cmake --build build --target test_qec_distributed
./build/test_qec_distributed
# Expected: 52 tests pass
```

### Test Coverage (52 tests):

```
Configuration (2 tests)
Partition Maps (7 tests)
Local/Global Syndrome (5 tests)
Syndrome Extraction (4 tests)
Parallel MWPM Decoder (5 tests)
Distributed Logical Qubit (7 tests)
FT-QEC Runner (4 tests)
Distributed QEC Simulator (4 tests)
Integration Tests (5 tests)
Performance Tests (2 tests)
```

### Sub-Tasks:

#### 5.1: Enable & Compile (15 min)
- Uncomment test_qec_distributed in CMakeLists.txt
- Rebuild library

#### 5.2: Run Tests (30 min)
```bash
./build/test_qec_distributed 2>&1 | tee qec_distributed_results.txt
```

#### 5.3: MPI Integration (45 min, Optional)
```bash
# If MPI available, test with multiple ranks
mpirun -np 4 ./build/test_qec_distributed
```

### Success Criteria:

- ✅ All 52 tests pass in single-rank mode
- ✅ Partition maps assign all qubits (no duplicates/gaps)
- ✅ Parallel decode matches single-rank within tolerance
- ✅ Load balancing correctly distributes work

### Expected Duration Breakdown:

- Enable: 5 min
- Compilation: 30 sec
- Test execution: 10 min
- MPI tests: 20 min (optional)
- Verification: 10 min

---

## TIER 6: Phase 9.3 - Adaptive QEC Tests

**Duration:** 1-1.5 hours  
**Purpose:** Validate ML-driven code selection, distance adaptation, closed-loop control  
**Dependencies:** Tier 4 & 5  
**Current Status:** ⏳ Code complete, disabled in CMakeLists.txt

### Re-enable Tests:

```bash
# Uncomment test_qec_adaptive in CMakeLists.txt
cmake --build build --target test_qec_adaptive
./build/test_qec_adaptive
# Expected: 45 tests pass
```

### Test Coverage (45 tests):

```
NoiseProfile (13 tests)
AdaptiveCodeSelector (9 tests)
MLDecoder (3 tests)
ClosedLoopController (7 tests)
DynamicDistanceSelector (5 tests)
AdaptiveQECController (6 tests)
Integration Tests (2 tests)
```

### Sub-Tasks:

#### 6.1: Enable & Compile (10 min)
```bash
# Uncomment in CMakeLists.txt
cmake --build build --target test_qec_adaptive
```

#### 6.2: Run Tests (20 min)
```bash
./build/test_qec_adaptive 2>&1 | tee qec_adaptive_results.txt
```

#### 6.3: Python Training Scripts (25 min, Optional)
```bash
# Generate training data for ML decoder
python3 scripts/generate_qec_training_data.py \
    --code surface \
    --distance 5 \
    --num-samples 10000 \
    --output data/train.npz

# Train ML decoder (requires JAX/Flax)
python3 scripts/train_ml_decoder.py \
    --input data/train.npz \
    --output models/ml_decoder.pkl
```

### Success Criteria:

- ✅ All 45 C++ tests pass
- ✅ Code selection matches noise profile correctly
- ✅ Distance adaptation responds to error rate changes
- ✅ Drift detection triggers appropriately
- ✅ ML training produces model with loss < 0.1

### Expected Duration Breakdown:

- Enable: 5 min
- Compilation: 30 sec
- Test execution: 15 min
- Python training: 20 min (optional)
- Verification: 10 min

---

## TIER 7: Phase 3 - MPI Distributed Simulation Tests

**Duration:** 1-2 hours  
**Purpose:** Validate MPI distribution strategies, collectives, load balancing  
**Dependencies:** Tier 5 (distributed QEC)  
**Current Status:** ⏳ Infrastructure exists; requires MPI environment

### Prerequisites:

```bash
# Install MPI
brew install open-mpi  # macOS
# or: apt-get install libopenmpi-dev  # Linux
# or: conda install -c conda-forge openmpi  # Anaconda

# Verify installation
which mpirun
mpirun --version
```

### Build with MPI:

```bash
cmake -S . -B build \
    -DUSE_MPI=ON \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

### MPI Tests:

#### 7.1: Single-Rank Stubs (15 min)
```bash
./build/test_qec_distributed
# Should use MPI stubs and pass in single-process mode
```

#### 7.2: Multi-Rank MPI (45 min)
```bash
# Test with 2, 4, 8 ranks
mpirun -np 2 ./build/test_qec_distributed
mpirun -np 4 ./build/test_qec_distributed
mpirun -np 8 ./build/test_qec_distributed

# Multi-GPU MPI (if GPU available)
mpirun -np 2 ./build/test_distributed_gpu_mpi
mpirun -np 4 ./build/test_multi_gpu_collectives
mpirun -np 4 ./build/test_multi_gpu_load_balance
```

#### 7.3: Verify Distribution Strategies (30 min)
```cpp
// Tests should cover:
// - Row-wise distribution
// - Column-wise distribution  
// - Load balancing
// - MPI collectives (AllReduce, Broadcast, Gather)
// - Fault tolerance checkpointing
```

### Success Criteria:

- ✅ All tests pass in single-rank mode
- ✅ All tests pass in multi-rank mode (2, 4, 8 ranks)
- ✅ Results match single-GPU baseline within 1e-9
- ✅ No MPI errors or deadlocks
- ✅ Performance scales reasonably (>80% of theoretical)

### Expected Duration Breakdown:

- MPI installation: 10 min
- Build: 1 min
- Single-rank tests: 15 min
- Multi-rank tests: 20 min
- Multi-GPU tests: 15 min (if available)
- Verification: 10 min

### Optional: Skip if MPI Unavailable
If MPI not available, all tests use stubs and single-rank mode can complete Tier 5/6.

---

## TIER 8: Phase 2 - GPU Acceleration Tests

**Duration:** 1-2 hours  
**Purpose:** Validate GPU kernels, multi-GPU distribution, autodiff on GPU  
**Dependencies:** Tier 4 (core QEC)  
**Current Status:** ⏳ Infrastructure exists; requires CUDA + cuQuantum

### Prerequisites:

```bash
# Install CUDA Toolkit (version 11.8+)
# Download from: https://developer.nvidia.com/cuda-downloads
# Verify: nvcc --version

# Install cuQuantum (optional, for advanced features)
# See: https://docs.nvidia.com/cuda/cuquantum/

# Verify GPU
nvidia-smi
```

### Build with GPU:

```bash
cmake -S . -B build \
    -DUSE_GPU=ON \
    -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

### GPU Tests:

#### 8.1: Single-GPU Smoke Test (15 min)
```bash
cmake --build build --target test_distributed_gpu
./build/test_distributed_gpu
# Expected: Single GPU state distribution works
```

#### 8.2: GPU Info & Properties (10 min)
```bash
./quantum_sim --gpu-info
# Should print GPU name, compute capability, memory, bandwidth
```

#### 8.3: Multi-GPU MPI Tests (30 min, requires MPI + 2+ GPUs)
```bash
cmake -S . -B build \
    -DUSE_GPU=ON \
    -DUSE_MPI=ON \
    -DBUILD_MULTI_GPU_TESTS=ON \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build

mpirun -np 2 ./build/test_distributed_gpu_mpi
mpirun -np 4 ./build/test_multi_gpu_sync
mpirun -np 4 ./build/test_multi_gpu_collectives
mpirun -np 4 ./build/test_multi_gpu_load_balance
```

#### 8.4: Autodiff on GPU (20 min, optional)
```bash
cmake --build build --target test_autodiff_multi_gpu
mpirun -np 2 ./build/test_autodiff_multi_gpu
# Gradients should match single-GPU within 1e-4
```

### Success Criteria:

- ✅ GPU state distribution works correctly
- ✅ GPU kernels execute without CUDA errors
- ✅ Results match CPU baseline within 1e-9
- ✅ Multi-GPU results match single-GPU within tolerance
- ✅ Speedup ≥ 10x vs CPU (for large circuits)
- ✅ No deadlocks or NCCL errors

### Expected Duration Breakdown:

- CUDA setup: 30 min (one-time)
- Build: 2 min
- Single-GPU tests: 15 min
- Multi-GPU tests: 20 min (if available)
- Autodiff tests: 10 min (if available)
- Verification: 10 min

### Optional: Skip if GPU Unavailable
If no CUDA/GPU, skip this entire tier. All functionality works on CPU (Tiers 1-6, 9-11 are sufficient).

---

## TIER 9: Phase 7 - Benchmarking & Performance Analysis

**Duration:** 1-2 hours  
**Purpose:** Comprehensive performance measurement across all modes and circuits  
**Dependencies:** Tier 1 (core library)  
**Current Status:** ✅ **Ready to run NOW**

### Benchmark Suite:

```bash
# Full benchmark suite (30-60 minutes depending on machine)
python3 scripts/benchmark_suite.py

# Quick CI mode (5-10 minutes)
python3 scripts/benchmark_suite.py --quick

# Specific categories only
python3 scripts/benchmark_suite.py --categories scaling,parallel

# Custom output
python3 scripts/benchmark_suite.py --output results/my_benchmark.csv
```

### Expected Outputs:

1. **Benchmark Results CSV:**
```
benchmark_results.csv
- timestamp, category, num_qubits, circuit_depth, mode
- execution_time_ms, final_rank, fidelity, memory_mb
```

2. **Analysis Report:**
```bash
python3 scripts/benchmark_analysis.py results/benchmark_results.csv
# Generates:
# - Regression report
# - Scaling analysis
# - Speedup metrics
# - Memory efficiency
```

3. **Visualizations:**
```bash
python3 scripts/benchmark_visualize.py results/benchmark_results.csv -o plots/
# Generates:
# - plot_scaling.svg          (time vs qubit count)
# - plot_speedup.svg          (parallel speedup)
# - plot_fidelity.svg         (LRET vs FDM accuracy)
# - plot_memory.svg           (memory scaling)
# - plot_summary.svg          (comprehensive overview)
```

### Benchmark Categories:

1. **Scaling** (6-14 qubits, 5-30 depth)
   - Measures exponential time growth
   - Tests all simulation modes

2. **Parallel** (6-16 qubits)
   - Compares speedup across modes
   - Sequential vs Row vs Column vs Hybrid vs Adaptive

3. **Accuracy** (LRET vs FDM)
   - Validates LRET fidelity
   - Statistical significance testing

4. **Depth Scaling** (Circuit depth 5-30)
   - Rank growth analysis
   - Depth-dependent scaling

5. **Memory Profiling**
   - Memory usage by qubit count
   - Peak memory tracking

### Sub-Tasks:

#### 9.1: Quick Benchmark (15 min)
```bash
python3 scripts/benchmark_suite.py --quick
```

#### 9.2: Full Benchmark (45 min)
```bash
python3 scripts/benchmark_suite.py
```

#### 9.3: Analysis (10 min)
```bash
python3 scripts/benchmark_analysis.py benchmark_results.csv > analysis.txt
```

#### 9.4: Visualization (10 min)
```bash
python3 scripts/benchmark_visualize.py benchmark_results.csv -o plots/
```

### Success Criteria:

- ✅ Quick mode completes in <10 min
- ✅ Full suite completes in <60 min
- ✅ CSV has valid schema
- ✅ Analysis detects regressions (if baseline exists)
- ✅ All plots generate without errors
- ✅ Speedup curves make physical sense

### Expected Duration Breakdown:

- Setup/plotting: 5 min
- Quick benchmark: 10 min
- Full benchmark: 45 min
- Analysis: 5 min
- Visualization: 5 min
- Verification: 10 min

---

## TIER 10: Phase 6 - Docker Integration & CI

**Duration:** 1-2 hours  
**Purpose:** Validate Docker builds, container runtime, CI pipelines  
**Dependencies:** Tier 1-6, 9 complete  
**Current Status:** ⏳ Dockerfile exists; needs execution/validation

### Prerequisites:

```bash
# Install Docker
brew install docker  # macOS (via Docker Desktop)
# or: apt-get install docker.io  # Linux

# Verify
docker --version
```

### Docker Build & Test:

#### 10.1: Build Docker Image (15 min)
```bash
docker build -t qlret:latest .
# Multi-stage build: builder → tester → runtime
```

#### 10.2: Run Tests in Container (30 min)
```bash
# Run C++ tests
docker run --rm qlret:latest ./build/test_qec_stabilizer
docker run --rm qlret:latest ./build/test_qec_syndrome
docker run --rm qlret:latest ./build/test_qec_decoder
docker run --rm qlret:latest ./build/test_qec_logical

# Run Python tests
docker run --rm qlret:latest python -m pytest python/tests/ -v

# Run benchmarks
docker run --rm qlret:latest python scripts/benchmark_suite.py --quick
```

#### 10.3: Volume Mounting (10 min)
```bash
# Test output persistence
docker run -v $(pwd)/docker_output:/app/output qlret:latest \
    ./quantum_sim -n 10 -d 15 -o /app/output/results.csv

# Verify output
cat docker_output/results.csv
```

#### 10.4: CI Pipeline (15 min)
```bash
# Check GitHub Actions workflows
cat .github/workflows/ci.yml
# Should include: build, test, benchmark, docker

# Run locally with act (if available)
act -j test  # Run test job locally
```

### Success Criteria:

- ✅ Docker image builds cleanly
- ✅ All container tests pass
- ✅ Output persists with volume mounting
- ✅ CI passes in GitHub Actions
- ✅ No security vulnerabilities
- ✅ Container runs quickly (<5s startup)

### Expected Duration Breakdown:

- Docker installation: 5 min (one-time)
- Image build: 10 min
- Container tests: 15 min
- Volume mounting: 5 min
- CI verification: 10 min
- Cleanup: 5 min

### Optional: Skip if Docker Unavailable
If Docker not available, skip this tier. Core testing (Tiers 1-9) is sufficient.

---

## TIER 11: Phase 0 - Documentation & Release

**Duration:** 1-2 hours  
**Purpose:** Documentation, contribution guide, release notes  
**Dependencies:** Tiers 1-10 complete  
**Current Status:** ⏳ Partially complete; needs updates

### Documentation Tasks:

#### 11.1: Update README.md (20 min)
```markdown
- Add tier-based testing roadmap
- Update installation instructions
- Add quick-start examples
- Document performance expectations
- Link to test results
```

#### 11.2: Contribution Guide (20 min)
```markdown
- Development setup
- Build instructions (CPU, GPU, MPI)
- Test execution
- Code style guidelines
- PR submission process
```

#### 11.3: API Documentation (20 min)
```bash
# Generate Doxygen docs
doxygen Doxyfile

# Build HTML docs
cd docs/html
python -m http.server 8000
# Visit http://localhost:8000
```

#### 11.4: Release Notes (15 min)
```markdown
- Summarize Tier 1-10 accomplishments
- Document new features
- List bug fixes
- Note performance improvements
- Acknowledge contributors
```

#### 11.5: Performance Report (15 min)
```markdown
- Benchmark results summary
- Speedup curves
- Memory profiles
- Comparison vs FDM
```

### Success Criteria:

- ✅ README covers installation & testing
- ✅ Contribution guide is complete
- ✅ API docs generate without errors
- ✅ Release notes are comprehensive
- ✅ All links work
- ✅ Examples execute without errors

### Expected Duration Breakdown:

- README: 20 min
- Contribution guide: 20 min
- API documentation: 15 min
- Release notes: 15 min
- Performance report: 15 min
- Verification: 15 min

---

## Execution Strategy: Three Paths

### Path A: Complete (All Tiers 1-11)

**Mandatory:** Tiers 1-6, 9-10 (~12 hours)  
**Optional:** Tiers 7-8 (~2-3 hours if hardware available)  
**Documentation:** Tier 11 (~1-2 hours)  
**Total:** 15-19 hours

```bash
# Tiers 1-6 (Core QEC Testing)
./build/test_simple && ./build/test_fidelity && ./build/test_minimal
python3 scripts/calibrate_noise_model.py  # Tier 2
cd python && pytest tests/ -v             # Tier 3
./build/test_qec_* && mpirun -np 4 ./build/test_qec_distributed  # Tiers 4-6

# Tier 9 (Benchmarking)
python3 scripts/benchmark_suite.py

# Tier 10 (Docker)
docker build -t qlret .
docker run --rm qlret ./build/test_qec_logical

# Tier 11 (Documentation)
# Update README, release notes, API docs
```

### Path B: Core Only (Tiers 1-6, 9-10, Skip Optional)

**Time:** ~12 hours  
**Skip:** GPU/MPI hardware-dependent tests (Tiers 7-8)

```bash
# Tiers 1-6, 9-10, 11 as above
# Skip mpirun commands and GPU tests
```

### Path C: Quick Validation (Tiers 1, 4, 9)

**Time:** ~2 hours  
**Purpose:** Rapid validation of core QEC + benchmarks

```bash
./build/test_simple && ./build/test_fidelity      # Tier 1
./build/test_qec_*                                # Tier 4
python3 scripts/benchmark_suite.py --quick        # Tier 9
```

---

## Quick Reference: Commands by Tier

```bash
# TIER 1: Core LRET
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
./build/test_simple && ./build/test_fidelity && ./build/test_minimal
./quantum_sim -n 6 -d 8 && ./build/demo_batch

# TIER 2: Noise Calibration
python3 scripts/generate_calibration_data.py --depths 5,10,15,20,25,30
python3 scripts/calibrate_noise_model.py calibration_data.csv
./build/test_advanced_noise && ./build/test_leakage_measurement

# TIER 3: Python Integration
cd python && pip install -e . && pytest tests/test_qlret_device.py -v

# TIER 4: Core QEC
./build/test_qec_stabilizer && ./build/test_qec_syndrome
./build/test_qec_decoder && ./build/test_qec_logical

# TIER 5: Distributed QEC
cmake --build build --target test_qec_distributed
./build/test_qec_distributed

# TIER 6: Adaptive QEC
cmake --build build --target test_qec_adaptive
./build/test_qec_adaptive

# TIER 7: MPI (if available)
cmake -S . -B build -DUSE_MPI=ON -DCMAKE_BUILD_TYPE=Release
mpirun -np 4 ./build/test_qec_distributed

# TIER 8: GPU (if available)
cmake -S . -B build -DUSE_GPU=ON -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
./build/test_distributed_gpu
mpirun -np 2 ./build/test_distributed_gpu_mpi

# TIER 9: Benchmarking
python3 scripts/benchmark_suite.py --quick
python3 scripts/benchmark_analysis.py benchmark_results.csv
python3 scripts/benchmark_visualize.py benchmark_results.csv -o plots/

# TIER 10: Docker
docker build -t qlret .
docker run --rm qlret ./build/test_qec_logical

# TIER 11: Documentation
# Update README.md, CONTRIBUTING.md, RELEASE_NOTES.md
```

---

## Checklist for Success

### Pre-Execution:
- [ ] LRET repository cloned and branch checked out
- [ ] CMake 3.20+ installed
- [ ] C++17 compiler available (clang, g++, MSVC)
- [ ] Eigen3 library installed
- [ ] Python 3.10+ available

### Tier 1 Checklist:
- [ ] Core library compiles
- [ ] test_simple passes
- [ ] test_fidelity passes
- [ ] test_minimal passes
- [ ] quantum_sim executable runs
- [ ] demo_batch produces valid output

### Tier 2 Checklist:
- [ ] Noise modules compile
- [ ] Calibration scripts execute
- [ ] R² values > 0.95 for all fits
- [ ] T2 ≤ 2*T1 constraints satisfied
- [ ] C++ noise tests pass

### Tier 3 Checklist:
- [ ] Python package installs
- [ ] All 15 tests pass
- [ ] PennyLane device works
- [ ] JSON API returns correct results
- [ ] Gradient checks pass (if ML frameworks installed)

### Tier 4 Checklist:
- [ ] All 4 QEC test executables compile
- [ ] test_qec_stabilizer: 4/5 pass
- [ ] test_qec_syndrome: 15/15 pass
- [ ] test_qec_decoder: 15/15 pass
- [ ] test_qec_logical: 24/24 pass

### Tier 5 Checklist:
- [ ] test_qec_distributed enabled
- [ ] All 52 tests pass
- [ ] Partition maps valid
- [ ] Parallel decode accurate

### Tier 6 Checklist:
- [ ] test_qec_adaptive enabled
- [ ] All 45 tests pass
- [ ] Code selection logic correct
- [ ] Distance adaptation works

### Tier 7 Checklist (Optional):
- [ ] MPI library installed
- [ ] Build succeeds with USE_MPI=ON
- [ ] Single-rank MPI tests pass
- [ ] Multi-rank tests pass (if ≥2 processes)

### Tier 8 Checklist (Optional):
- [ ] CUDA toolkit installed
- [ ] Build succeeds with USE_GPU=ON
- [ ] GPU info prints correctly
- [ ] Single-GPU tests pass
- [ ] Multi-GPU tests pass (if ≥2 GPUs)

### Tier 9 Checklist:
- [ ] benchmark_suite.py executes
- [ ] Quick mode completes in <10 min
- [ ] CSV has valid schema
- [ ] Analysis produces regression report
- [ ] Plots generate without errors

### Tier 10 Checklist (Optional):
- [ ] Docker installed
- [ ] Image builds successfully
- [ ] Container tests pass
- [ ] Volume mounting works
- [ ] CI passes

### Tier 11 Checklist:
- [ ] README updated
- [ ] Contribution guide added
- [ ] API docs generate
- [ ] Release notes written
- [ ] All links verified

---

## Timeline Estimate

| Path | Duration | Scope |
|------|----------|-------|
| **Quick (C, 1/4/9)** | 2h | Core QEC + benchmarks |
| **Core (B, 1-6/9-11)** | 12h | All mandatory, no GPU/MPI |
| **Complete (A, all)** | 15-19h | Everything including GPU/MPI |

**Recommended:** Start with Path B, add Tiers 7-8 only if hardware available.

---

## Conclusion

TESTING_BACKLOG.md is **fully addressed** by this 11-tier roadmap:

- **Phase 1:** Tier 1 ✅
- **Phase 2:** Tier 8 (optional, GPU hardware dependent)
- **Phase 3:** Tier 7 (optional, MPI hardware dependent)
- **Phase 4:** Tier 2 ✅
- **Phase 5:** Tier 3 ✅
- **Phase 6:** Tier 10 + 11 ✅
- **Phase 7:** Tier 9 (benchmarking) ✅
- **Phase 8:** Tier 8 (autodiff + GPU) - optional
- **Phase 9:** Tiers 4, 5, 6 ✅

**Ready to execute immediately:** Tiers 1-6, 9-11 (no special hardware)  
**Can start now:** Tier 9 (benchmarking, completely independent)

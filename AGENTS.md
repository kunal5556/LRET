# LRET Quantum Simulator - AI Assistant Rules

This is an LRET (Low-Rank Entanglement Tracking) quantum simulator project. A high-performance C++17 quantum computing simulation framework with Python bindings, quantum error correction, and distributed computing support.

## Project Structure

- `src/` - Core C++ implementation (quantum_sim, QEC, autodiff, checkpoints)
- `include/` - C++ header files
- `python/` - Python package (qlret) with PennyLane device and JAX interface
- `tests/` - C++ test binaries
- `scripts/` - Python utility scripts (noise calibration, ML decoder training)
- `build/` - CMake build directory with compiled binaries
- `samples/` - Example JSON configurations and circuits
- `docs/` - Documentation (API reference, user guides, deployment)

## Key Components

### Quantum Simulation
- **FDM Simulator** (`src/fdm_simulator.cpp`) - Finite difference method for open quantum systems
- **SIMD Kernels** (`src/simd_kernels.cpp`) - Optimized vector operations
- **GPU Simulator** (`src/gpu_simulator.cu`) - CUDA-accelerated simulation

### Quantum Error Correction (QEC)
- `src/qec_adaptive.cpp` - Adaptive QEC with ML-driven decoding
- `src/qec_decoder.cpp` - MWPM and union-find decoders
- `src/qec_syndrome.cpp` - Syndrome extraction
- `src/qec_stabilizer.cpp` - Stabilizer measurements
- `src/qec_logical.cpp` - Logical qubit operations
- `src/qec_distributed.cpp` - Distributed QEC across MPI nodes

### Infrastructure
- `src/checkpoint.cpp` - State checkpointing with serialization
- `src/autodiff.cpp` - Automatic differentiation for variational circuits
- `src/mpi_parallel.cpp` - MPI parallelization
- `src/resource_monitor.cpp` - Resource monitoring

### Python Interface
- `python/qlret/pennylane_device.py` - PennyLane quantum device
- `python/qlret/jax_interface.py` - JAX integration for autodiff
- `python/qlret/cirq_compare.py` - Cirq comparison utilities

## Build Instructions

```bash
cd build
cmake ..
make -j$(nproc)  # or make -j$(sysctl -n hw.ncpu) on macOS
```

## Test Binaries

After building, run tests from the `build/` directory:
- `./test_simple` - Basic functionality
- `./test_fidelity` - Fidelity calculations
- `./test_autodiff` - Automatic differentiation
- `./test_checkpoint` - Checkpoint/restore
- `./test_qec_*` - Quantum error correction tests
- `./quantum_sim` - Main simulator with JSON input

## Code Standards

- Use C++17 features
- Follow existing naming conventions (snake_case for functions, CamelCase for classes)
- Add tests for new functionality
- Use Eigen3 for linear algebra
- MPI for distributed computing (optional)
- OpenMP for threading (optional)

## Dependencies

- Required: CMake 3.16+, C++17 compiler, Eigen3
- Optional: MPI, OpenMP, CUDA (for GPU support)
- Python: Python 3.8+, PennyLane, NumPy, JAX (optional)

## Common Tasks

### Run a quantum simulation
```bash
./build/quantum_sim samples/basic_gates.json
```

### Run Python tests
```bash
cd python && pytest tests/
```

### Build and run all C++ tests
```bash
cd build && make -j$(sysctl -n hw.ncpu) && ctest
```

## Important Files

- `CMakeLists.txt` - Build configuration
- `Dockerfile` - Docker container setup
- `README.md` - Project overview
- `agent.md` - Detailed AI agent guide (18,500+ lines)
- `AGENT_GUIDE.md` - User-friendly agent guide

## External Documentation

---

## ✅ PennyLane Benchmarking Instructions

**NEW - CRITICAL FOR AGENT: Complete PennyLane Testing & Benchmarking Suite**

### Overview

This section provides complete instructions for running PennyLane benchmarks comparing LRET plugin performance against standard PennyLane devices (default.mixed, lightning.qubit). The goal is to demonstrate LRET's performance advantages (10-500× memory, 50-200× speed) through rigorous testing.

### PennyLane Documentation Files

**Created**: January 2026  
**Branch**: `pennylane-documentation-benchmarking`

#### Core Strategy Documents

1. **PENNYLANE_ALGORITHM_CATALOG.md** (1438 lines, 80+ pages)
   - 20 quantum algorithms with complete implementations
   - Tier 1 (must test): VQE, QAOA, QNN, QFT, QPE, Grover, Metrology
   - Tier 2 (should test): UCCSD-VQE, Portfolio, QSVM, QAE, VQD, qGAN
   - Tier 3 (optional): VQT, Quantum Walk, Boltzmann, etc.
   - Each algorithm has: mathematical formulation, PennyLane code, complexity analysis

2. **PENNYLANE_BENCHMARKING_STRATEGY.md** (1745+ lines)
   - 8 benchmark categories with detailed test protocols
   - Category 1: Memory Efficiency (peak memory vs qubits)
   - Category 2: Execution Speed (wall-clock time vs qubits)
   - Category 3: Accuracy/Fidelity (vs exact simulation)
   - Category 4: Gradient Computation (parameter-shift performance)
   - Category 5: Scalability (time/memory vs problem size)
   - Category 6: Applications (VQE, QAOA, QNN, QFT, QPE, Grover, Metrology)
   - Category 7: Framework Integration (PyTorch, JAX, TensorFlow)
   - Category 8: Cross-Simulator Comparison (vs Qiskit Aer, Cirq)

3. **BENCHMARKING_EXECUTION_STRATEGY.md** (1086 lines)
   - Phase-by-phase implementation plan (5 phases)
   - Phase 1: Infrastructure Setup (3-4 days)
   - Phase 2: Implementation (10 days, code 8 categories)
   - Phase 3: Execution (7-10 CONTINUOUS DAYS, 150-200+ hours machine time)
   - Phase 4: Analysis & Visualization (4-5 days)
   - Phase 5: Publication & Reproducibility Guide (3-4 days)
   - Complete code models and examples for each phase

4. **BENCHMARKING_METHODOLOGY_CONFIRMATION.md** (590 lines)
   - Scientific validation: Must run tests ourselves on same hardware
   - Breaking point analysis: Test devices to their limits
   - Expected breaking points: default.mixed ~12 qubits, LRET ~22 qubits
   - Fair comparison criteria (same hardware, software, parameters)

5. **BENCHMARKING_QUICK_ANSWER.md** (248 lines)
   - Executive summary answering "how to proceed?"
   - 5-phase pipeline overview with timelines
   - Recommended tools and technologies

### Critical Timing Information

**IMPORTANT - Long Execution Times Are Expected:**

Single trial duration: 30-40 hours (breaking point testing for 14-24 qubits)  
5 complete trials: 150-200+ hours total  
Wall clock time: 7-10 continuous days OR 1-2 weeks of overnight runs

**Why times are so long:**
- LRET at 20 qubits: 100-300 seconds per circuit
- LRET at 22 qubits: 300-1000 seconds per circuit
- With 5 trials × 3 noise levels × 3 depths = 45 runs
- Testing to breaking points takes longer than simple averaging
- This is STANDARD for publication-grade benchmarking

### Expected Breaking Points

| Device | Breaking Point | Reason |
|--------|---------------|--------|
| default.mixed | ~12 qubits | OOM (Out of Memory) |
| lightning.qubit | ~18-20 qubits | OOM (no noise support) |
| LRET | ~22-24 qubits | Timeout/OOM (with noise!) |

**LRET Advantage**: 10 additional qubits = 1024× more quantum states!

### QNN Classifier Benchmarking (Priority Test)

**Qubit Limits for QNN:**

| Qubits | Training Time | Status |
|--------|--------------|--------|
| 8 | ~1-2 hours | ✅ Easy |
| 10 | ~4-6 hours | ✅ Fast |
| 12 | ~15-25 hours | ⚠️ Long but manageable |
| 14 | ~35-50 hours | ✅ Good stopping point |
| 16 | ~80-120 hours | ✅ Maximum - shows clear advantage |
| 18+ | 200h+ | ❌ Not practical |

**Recommendation**: Test up to 16 qubits to show exponential advantage (256× more states).

### How the Agent Should Run Benchmarks

#### Option A: Quick Test (QNN 8-10 qubits)

**Purpose**: Fast validation that LRET plugin works with PennyLane  
**Duration**: 2-6 hours total  
**Requirements**: Standard laptop/workstation (8GB+ RAM)

**Steps:**
1. Ensure Python environment has: `pennylane>=0.30`, `torch`, `numpy`, `scipy`, `psutil`
2. Build LRET Python bindings: `cd python && pip install -e .`
3. Run QNN test for both LRET and default.mixed:
   ```python
   # Test LRET device
   import pennylane as qml
   dev = qml.device("qlret.mixed", wires=8, epsilon=1e-4)
   # Run QNN training circuit
   
   # Test default.mixed device
   dev_baseline = qml.device("default.mixed", wires=8)
   # Run same circuit
   ```
4. Compare: execution time, memory usage, training accuracy
5. Save results to `results/qnn_8q_comparison.json`

#### Option B: Full Breaking Point Test

**Purpose**: Demonstrate LRET's scalability advantage  
**Duration**: 7-10 continuous days  
**Requirements**: Dedicated workstation (32GB+ RAM), background execution

**Steps:**
1. Set up environment (see Phase 1 in BENCHMARKING_EXECUTION_STRATEGY.md)
2. Create `benchmarks/` directory structure
3. Implement benchmark runner scripts (see code models in strategy docs)
4. Run breaking point tests:
   - Test default.mixed: 2, 4, 6, 8, 10, 12 qubits (will fail at 12-14)
   - Test LRET: 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22 qubits
   - Collect data: time, memory, fidelity, final rank
5. Run 5 trials per configuration
6. Generate comparison plots and tables
7. Save complete results to `results/breaking_point_analysis.csv`

**Background execution command:**
```bash
nohup python benchmarks/run_all.py --trials 5 > benchmark.log 2>&1 &
# or
screen -S benchmark
python benchmarks/run_all.py --trials 5
# Detach with Ctrl+A then D
```

### Key Performance Claims to Validate

**Memory Efficiency**: 10-500× reduction  
**Speed**: 50-200× faster for large systems  
**Accuracy**: >99.9% fidelity  
**Scalability**: Works to 20+ qubits vs 12 for default.mixed

### Directory Structure for Benchmarking

```
LRET/
├── python/
│   └── qlret/
│       ├── pennylane_device.py    # LRET PennyLane device
│       └── tests/
│           └── test_pennylane.py  # Unit tests
├── benchmarks/                      # CREATE THIS
│   ├── __init__.py
│   ├── config.py                   # Configuration
│   ├── run_all.py                  # Master runner
│   ├── utils/
│   │   ├── device_factory.py
│   │   ├── circuit_generators.py
│   │   └── metrics.py
│   ├── 01_memory_efficiency/
│   ├── 02_execution_speed/
│   ├── 03_accuracy/
│   ├── 04_gradient_computation/
│   ├── 05_scalability/
│   ├── 06_applications/
│   │   └── qnn_classifier.py       # QNN test
│   ├── 07_framework_integration/
│   └── 08_cross_simulator/
└── results/                         # CREATE THIS
    ├── raw_data/                   # JSON from each run
    ├── processed_data/             # CSV aggregated
    └── plots/                      # PNG/PDF figures
```

### Agent Checklist for Benchmarking

**Before Running:**
- [ ] Confirm Python environment has PennyLane 0.30+
- [ ] Build LRET Python bindings: `cd python && pip install -e .`
- [ ] Test LRET device loads: `python -c "import pennylane as qml; dev = qml.device('qlret.mixed', wires=4); print(dev)"`
- [ ] Create `benchmarks/` and `results/` directories
- [ ] Understand expected execution times (hours to days)

**During Execution:**
- [ ] Monitor progress with timestamps
- [ ] Save intermediate results (don't lose data!)
- [ ] Check system resources (memory, disk space)
- [ ] Log any errors or anomalies

**After Completion:**
- [ ] Aggregate raw data to CSV
- [ ] Generate comparison tables (LRET vs default.mixed)
- [ ] Create plots (time vs qubits, memory vs qubits)
- [ ] Save detailed report to `results/benchmark_summary.md`
- [ ] Confirm breaking points match expectations

### Automated Test Script Template

```python
#!/usr/bin/env python3
"""
Automated QNN Classifier Benchmark - LRET vs PennyLane default.mixed
Run time: 2-6 hours for 8-10 qubits
"""

import pennylane as qml
import numpy as np
import time
import psutil
import json
from datetime import datetime

def run_qnn_benchmark(n_qubits, device_name):
    """Run QNN training on specified device"""
    # Create device
    if device_name == "lret":
        dev = qml.device("qlret.mixed", wires=n_qubits, epsilon=1e-4)
    else:
        dev = qml.device("default.mixed", wires=n_qubits)
    
    # Define QNN circuit
    @qml.qnode(dev)
    def circuit(params, x):
        # Embedding layer
        for i in range(n_qubits):
            qml.RY(x[i], wires=i)
        
        # Variational layer
        for layer in range(2):
            for i in range(n_qubits):
                qml.RY(params[layer, i, 0], wires=i)
                qml.RZ(params[layer, i, 1], wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i+1])
        
        return qml.expval(qml.PauliZ(0))
    
    # Training
    params = np.random.random((2, n_qubits, 2))
    x_train = np.random.random((10, n_qubits))
    y_train = np.random.choice([-1, 1], size=10)
    
    start_time = time.time()
    start_memory = psutil.virtual_memory().used / (1024**2)
    
    # Simple training loop
    for epoch in range(5):
        for x, y in zip(x_train, y_train):
            prediction = circuit(params, x)
            # Gradient step (simplified)
    
    end_time = time.time()
    end_memory = psutil.virtual_memory().used / (1024**2)
    
    return {
        "device": device_name,
        "n_qubits": n_qubits,
        "training_time_seconds": end_time - start_time,
        "memory_used_mb": end_memory - start_memory,
        "timestamp": datetime.now().isoformat()
    }

# Run benchmark
print("Starting QNN Benchmark - LRET vs default.mixed")
results = []

for n_qubits in [8, 10]:
    print(f"\nTesting {n_qubits} qubits...")
    
    # Test LRET
    print(f"  Running on LRET...")
    lret_result = run_qnn_benchmark(n_qubits, "lret")
    results.append(lret_result)
    print(f"    Time: {lret_result['training_time_seconds']:.2f}s")
    
    # Test default.mixed
    print(f"  Running on default.mixed...")
    baseline_result = run_qnn_benchmark(n_qubits, "default.mixed")
    results.append(baseline_result)
    print(f"    Time: {baseline_result['training_time_seconds']:.2f}s")
    
    # Comparison
    speedup = baseline_result['training_time_seconds'] / lret_result['training_time_seconds']
    print(f"  Speedup: {speedup:.2f}x ({'LRET faster' if speedup > 1 else 'baseline faster'})")

# Save results
with open('results/qnn_benchmark.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n✓ Benchmark complete! Results saved to results/qnn_benchmark.json")
```

### What Results Should Look Like

**Success Criteria:**
- LRET completes all tests without errors
- LRET matches or exceeds baseline accuracy (>99% fidelity)
- LRET shows memory advantage for 12+ qubits
- LRET shows speed advantage for 14+ qubits
- Breaking point: LRET reaches 18-22 qubits, baseline fails at 12-14

**Example Output:**
```
Testing 8 qubits...
  LRET: 45.2s, 280 MB
  default.mixed: 38.1s, 320 MB
  Speedup: 0.84x (baseline faster at small scale - expected)

Testing 10 qubits...
  LRET: 156.3s, 680 MB
  default.mixed: 198.7s, 2.4 GB
  Speedup: 1.27x (LRET faster, 3.5× less memory!)

Testing 12 qubits...
  LRET: 580.2s, 1.8 GB
  default.mixed: TIMEOUT (>600s) or OOM
  Speedup: ∞ (LRET still works, baseline failed!)
```

### Agent Decision Making

**If asked to "run PennyLane benchmarks":**
1. Ask: "Quick test (QNN 8-10 qubits, 2-6 hours) or full breaking point analysis (7-10 days)?"
2. If quick test: Run automated script above
3. If full analysis: Explain timeline, get confirmation, run Phase 1-5 pipeline

**If running into errors:**
1. Check LRET device installation: `python -c "import qlret; print(qlret.__version__)"`
2. Check PennyLane recognizes device: `qml.device('qlret.mixed', wires=4)`
3. Check build artifacts exist: `ls python/qlret/build/` should show .so/.dll files
4. Rebuild if needed: `cd python && pip install -e . --force-reinstall`

---

For detailed agent instructions and capabilities: @agent.md
For beginner-friendly guide: @AGENT_GUIDE.md
For manual circuit execution workflows: @MANUAL_CIRCUIT_EXECUTION_GUIDE.md
For Cirq FDM comparison setup: @CIRQ_COMPARISON_GUIDE.md

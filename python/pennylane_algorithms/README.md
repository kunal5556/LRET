# PennyLane Algorithm Benchmarks

Comprehensive benchmarking suite for testing the LRET quantum simulator against standard PennyLane devices across 20 quantum algorithms.

## Overview

This package provides:
- **20 quantum algorithms** across 3 tiers of complexity
- **LRET mode comparison**: sequential, batched, parallel, openmp
- **Device comparison**: qlret.mixed vs default.mixed vs default.qubit
- **Python parallelism comparison**: multiprocessing, threading, joblib

## Quick Start

```bash
# Run all benchmarks (full suite)
python run_all_benchmarks.py

# Run specific tier
python run_all_benchmarks.py --tier 1

# Run specific algorithm
python run_all_benchmarks.py --algorithm vqe qaoa

# Quick test (1 trial each)
python run_all_benchmarks.py --quick

# List available algorithms
python run_all_benchmarks.py --list
```

## Structure

```
pennylane_algorithms/
├── __init__.py
├── run_all_benchmarks.py     # Master runner script
├── README.md
│
├── utils/                     # Shared utilities
│   ├── benchmark_utils.py    # BenchmarkResult, Timer, MemoryTracker
│   ├── device_factory.py     # Device creation helpers
│   ├── parallel_modes.py     # Python parallelism comparison
│   └── plotting.py           # Visualization utilities
│
├── tier1/                     # Must Test (7 algorithms)
│   ├── vqe.py                # VQE for H2/LiH molecules
│   ├── qaoa.py               # QAOA MaxCut
│   ├── qnn.py                # QNN Classifier
│   ├── qft.py                # QFT Fidelity
│   ├── qpe.py                # Quantum Phase Estimation
│   ├── grover.py             # Grover's Search
│   └── metrology.py          # Quantum Metrology (QFI)
│
├── tier2/                     # Should Test (7 algorithms)
│   ├── uccsd_vqe.py          # UCCSD-VQE Chemistry
│   ├── portfolio.py          # Portfolio Optimization
│   ├── qsvm.py               # Quantum SVM
│   ├── qae.py                # Quantum Amplitude Estimation
│   ├── vqd.py                # Variational Quantum Deflation
│   ├── qgan.py               # Quantum GAN
│   └── number_partitioning.py # Number Partitioning
│
├── tier3/                     # Optional (6 algorithms)
│   ├── vqt.py                # Variational Quantum Thermalizer
│   ├── quantum_walk.py       # Quantum Walk
│   ├── kernel_alignment.py   # Quantum Kernel Alignment
│   ├── subsampling_qnn.py    # Sub-sampling QNN
│   ├── hea.py                # Hardware-Efficient Ansatz
│   └── adapt_vqe.py          # ADAPT-VQE
│
└── results/                   # Output directory (created on run)
```

## Algorithms

### Tier 1 - Must Test (Core Algorithms)

| Algorithm | Qubits | Description |
|-----------|--------|-------------|
| VQE | 4 | Variational Quantum Eigensolver for H2 molecule |
| QAOA | 6 | MaxCut optimization on random graphs |
| QNN | 4 | Quantum Neural Network classifier |
| QFT | 4 | Quantum Fourier Transform fidelity |
| QPE | 5 | Quantum Phase Estimation accuracy |
| Grover | 4 | Grover's search algorithm |
| Metrology | 4 | Quantum metrology with GHZ states |

### Tier 2 - Should Test (Application Algorithms)

| Algorithm | Qubits | Description |
|-----------|--------|-------------|
| UCCSD-VQE | 4 | Chemistry-focused VQE with UCCSD ansatz |
| Portfolio | 6 | Quantum portfolio optimization |
| QSVM | 4 | Quantum kernel SVM classification |
| QAE | 6 | Quantum amplitude estimation |
| VQD | 2 | Variational quantum deflation for excited states |
| qGAN | 3 | Quantum generative adversarial network |
| Number Partitioning | 4 | Combinatorial optimization via QAOA |

### Tier 3 - Optional (Advanced Algorithms)

| Algorithm | Qubits | Description |
|-----------|--------|-------------|
| VQT | 3 | Variational quantum thermalizer |
| Quantum Walk | 4 | Continuous-time quantum walk |
| Kernel Alignment | 2 | Trainable quantum kernels |
| Subsampling QNN | 4 | Large-scale QNN with batching |
| HEA | 4 | Hardware-efficient ansatz study |
| ADAPT-VQE | 4 | Adaptive ansatz construction |

## Benchmark Methods

Each algorithm benchmark provides:

1. **`compare_lret_modes()`** - Compare LRET execution modes:
   - `sequential`: Single-threaded execution
   - `batched`: Batched circuit execution
   - `parallel`: Python multiprocessing
   - `openmp`: OpenMP parallelization

2. **`compare_devices()`** - Compare against standard devices:
   - `qlret.mixed`: LRET low-rank density matrix
   - `default.mixed`: PennyLane default mixed-state
   - `default.qubit`: PennyLane default pure-state (where applicable)

3. **`run_full_benchmark()`** - Run complete benchmark suite

## Output Format

Results are saved as JSON with structure:

```json
{
  "metadata": {
    "timestamp": "20240115_143022",
    "n_trials": 3,
    "algorithms_run": ["vqe", "qaoa", ...]
  },
  "results": {
    "vqe": {
      "name": "VQE (H2/LiH)",
      "tier": 1,
      "n_qubits": 4,
      "data": {
        "lret_modes": {...},
        "device_comparison": {...}
      }
    }
  }
}
```

## BenchmarkResult Fields

Each benchmark returns `BenchmarkResult` objects with:

| Field | Type | Description |
|-------|------|-------------|
| `algorithm` | str | Algorithm name |
| `device_name` | str | Device used |
| `mode` | str | Execution mode |
| `n_qubits` | int | Number of qubits |
| `execution_time_seconds` | float | Wall-clock time |
| `peak_memory_mb` | float | Peak memory usage |
| `result_value` | float | Primary result (energy, accuracy, etc.) |
| `secondary_value` | float | Secondary metric (error, fidelity, etc.) |
| `with_noise` | bool | Whether noise was simulated |
| `success` | bool | Whether benchmark completed |
| `error_message` | str | Error details if failed |
| `extra_data` | dict | Algorithm-specific metadata |

## Requirements

```
pennylane>=0.30
numpy
psutil
matplotlib (optional, for plotting)
```

## Example Usage

### Run individual algorithm

```python
from pennylane_algorithms.tier1 import run_vqe_benchmark

results = run_vqe_benchmark(n_qubits=4, n_trials=3)
print(results)
```

### Use benchmark class directly

```python
from pennylane_algorithms.tier1 import VQEBenchmark

benchmark = VQEBenchmark(n_qubits=4, max_iterations=50)

# Compare LRET modes
lret_results = benchmark.compare_lret_modes(n_trials=3)

# Compare against default.mixed
device_results = benchmark.compare_devices(n_trials=3)
```

### Generate plots

```python
from pennylane_algorithms.utils.plotting import plot_device_comparison

results = {...}  # From benchmark run
plot_device_comparison(results, output_file='comparison.png')
```

## Expected Performance

Based on LRET's low-rank density matrix approach:

| Metric | LRET Advantage |
|--------|----------------|
| Memory | 10-500× reduction for 12+ qubits |
| Speed | 50-200× faster for 14+ qubits |
| Accuracy | >99.9% fidelity |
| Scalability | 20+ qubits vs 12 for default.mixed |

## Contributing

To add a new algorithm:

1. Create `tier{N}/algorithm_name.py`
2. Implement the benchmark class following existing patterns
3. Add to tier `__init__.py`
4. Register in `run_all_benchmarks.py` ALGORITHMS dict

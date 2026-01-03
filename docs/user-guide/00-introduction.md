# Introduction to LRET

## What is LRET?

**LRET (Low-Rank Entanglement Tracking)** is a quantum circuit simulator optimized for simulating noisy quantum circuits efficiently. Unlike traditional full-density-matrix (FDM) simulators that require exponential memory ($O(4^n)$ for $n$ qubits), LRET leverages the insight that realistic noise keeps density matrices low-rank, achieving dramatic speedups while maintaining high fidelity.

## Why LRET?

### The Problem: Exponential Scaling

Classical quantum circuit simulators face fundamental scaling challenges:

| Qubits | State Vector Size | Density Matrix Size | Memory Required |
|--------|-------------------|---------------------|-----------------|
| 10     | 1,024            | 1,048,576           | 8 MB            |
| 15     | 32,768           | 1,073,741,824       | 8 GB            |
| 20     | 1,048,576        | 1,099,511,627,776   | 8 TB            |
| 25     | 33,554,432       | 1,125,899,906,842,624 | 8 PB          |

Traditional FDM simulators store the full density matrix, making simulations beyond 15 qubits impractical on consumer hardware.

### The Solution: Low-Rank Decomposition

LRET represents the density matrix $\rho$ in a factorized form:

$$
\rho = LL^\dagger, \quad L \in \mathbb{C}^{2^n \times r}
$$

where $r$ is the **rank** (typically $r \ll 2^n$ for noisy circuits). This reduces:
- **Memory:** $O(2^n \cdot r)$ instead of $O(4^n)$
- **Gate operations:** $O(2^n \cdot r)$ instead of $O(4^n)$

**Key Insight:** Realistic noise (depolarizing, damping, etc.) keeps $r$ small even for deep circuits.

## Real-World Performance

Benchmark results on Intel Xeon 8-core (1% depolarizing noise):

| Qubits | Depth | FDM Time | LRET Time | Speedup | Rank | Fidelity |
|--------|-------|----------|-----------|---------|------|----------|
| 8      | 20    | 234 ms   | 45 ms     | 5.2×    | 11   | 0.9998   |
| 10     | 30    | 1.2 s    | 178 ms    | 6.7×    | 15   | 0.9997   |
| 12     | 40    | 18.4 s   | 712 ms    | 25.8×   | 22   | 0.9996   |
| 14     | 50    | 294 s    | 3.2 s     | 91.9×   | 31   | 0.9995   |

**Observations:**
- Rank grows sub-exponentially with qubit count and depth
- Speedup increases exponentially (up to 90× for 14 qubits)
- Fidelity loss < 0.05% across all tests

## Core Features

### 1. **Efficient Noisy Circuit Simulation**
```python
from qlret import QuantumSimulator

sim = QuantumSimulator(n_qubits=12, noise_level=0.01)
sim.h(0)  # Hadamard gate
sim.cnot(0, 1)  # CNOT gate
result = sim.measure_all()
print(f"Final rank: {sim.current_rank}")  # Rank << 2^12
```

### 2. **Realistic Noise Models**
- **Depolarizing noise:** Uniform errors on all Pauli operators
- **Amplitude damping:** Energy relaxation (T1 processes)
- **Phase damping:** Dephasing (T2 processes)
- **Leakage errors:** Transitions to non-computational states
- **IBM device calibration:** Import real device noise from IBM Quantum

```python
# Import IBM device noise
from qlret import load_noise_model

noise = load_noise_model("ibmq_manila.json")
sim = QuantumSimulator(n_qubits=5, noise_model=noise)
```

### 3. **Parallel Execution Modes**
LRET supports multiple parallelization strategies:

- **Sequential:** Single-threaded baseline
- **Row Parallel:** Parallelize row updates (best for small ranks)
- **Column Parallel:** Parallelize column updates (best for large ranks)
- **Hybrid:** Auto-selects based on rank and qubit count (recommended)

```cpp
// C++ API: Parallel simulation with 8 threads
auto result = run_simulation_optimized(
    L_init, gates, n_qubits,
    batch_size=64, verbose=true, do_truncation=true, threshold=1e-4
);
```

### 4. **PennyLane Integration**
LRET integrates seamlessly with PennyLane for hybrid quantum-classical algorithms:

```python
import pennylane as qml
from qlret import QLRETDevice

dev = QLRETDevice(wires=4, noise_model="ibm_device.json")

@qml.qnode(dev)
def vqe_circuit(params):
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(0))

# Gradient-based optimization works seamlessly
```

### 5. **Comprehensive CLI Tool**
```bash
# Simulate 12 qubits, depth 40, 1% noise, hybrid parallelization
quantum_sim -n 12 -d 40 --noise 0.01 --mode hybrid --output results.csv

# Import noise from JSON
quantum_sim -n 8 -d 20 --noise-file ibm_device.json --fidelity-check

# Benchmarking mode
quantum_sim --benchmark --qubits 8,10,12 --depths 20,30,40
```

## When to Use LRET

### ✅ **LRET Excels At:**
- **Noisy intermediate-scale circuits (10-20 qubits)** with realistic noise
- **VQE, QAOA, and other variational algorithms** with noise modeling
- **Device simulation** using calibrated noise models
- **Noise resilience studies** comparing algorithms under various noise levels
- **Algorithm prototyping** before running on real quantum hardware

### ❌ **LRET Is Not Ideal For:**
- **Noiseless circuits:** Rank grows exponentially without noise (use state vector simulators)
- **Very large circuits (>20 qubits):** Memory still grows exponentially with rank
- **Extremely low noise (< 0.001%):** Low-rank advantage diminishes
- **Fault-tolerant circuits:** Error correction may increase rank

### 🤔 **Alternative Simulators:**

| Use Case | Recommended Simulator |
|----------|----------------------|
| Noiseless < 30 qubits | Qiskit Aer (state vector) |
| Noiseless > 30 qubits | Distributed simulators (e.g., Qulacs) |
| Tensor network circuits | ITensor, TensorNetwork |
| Stabilizer circuits | Stim, CHP |
| Small noisy circuits | LRET ⭐ |

## Architecture Overview

```
User Interface Layer
├── CLI Tool (quantum_sim)
├── Python Module (qlret.*)
└── PennyLane Device (QLRETDevice)
         │
         ▼
Core Simulator (C++)
├── LRET Algorithm
│   ├── Choi matrix representation
│   ├── SVD-based rank truncation
│   └── Efficient gate application
├── Parallelization (OpenMP)
│   ├── Row/Column/Hybrid modes
│   └── SIMD-optimized kernels
└── Noise Models
    ├── Built-in models (depolarizing, damping)
    └── IBM device import (JSON format)
         │
         ▼
External Dependencies
├── Eigen3 (linear algebra)
├── OpenMP (parallelization)
└── pybind11 (Python bindings)
```

## Getting Started

Choose your installation method:

1. **[Docker](01-installation.md#docker-installation)** (recommended for quick start)
2. **[Python Package](01-installation.md#python-installation)** (for Python users)
3. **[Build from Source](01-installation.md#building-from-source)** (for C++ developers)

After installation, proceed to the **[Quick Start Tutorial](02-quick-start.md)** to run your first simulation in 5 minutes.

## Community & Support

- **Documentation:** [https://lret.readthedocs.io](https://lret.readthedocs.io) (coming soon)
- **GitHub Repository:** [https://github.com/kunal5556/LRET](https://github.com/kunal5556/LRET)
- **Issue Tracker:** [GitHub Issues](https://github.com/kunal5556/LRET/issues)
- **Discussions:** [GitHub Discussions](https://github.com/kunal5556/LRET/discussions)

## Next Steps

- **[Installation Guide →](01-installation.md)** - Install LRET on your system
- **[Quick Start Tutorial →](02-quick-start.md)** - Your first simulation in 5 minutes
- **[CLI Reference →](03-cli-reference.md)** - Complete command-line interface
- **[Python Interface →](04-python-interface.md)** - Python API documentation

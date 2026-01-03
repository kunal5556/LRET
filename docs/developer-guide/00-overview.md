# Architecture Overview

High-level overview of LRET's architecture, design decisions, and system components.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interface Layer                      │
├────────────────┬─────────────────┬──────────────────────────────┤
│   CLI Tool     │  Python Module  │    PennyLane Device          │
│  quantum_sim   │    qlret.*      │     QLRETDevice              │
│   (main.cpp)   │  (python_bindings.cpp)  │  (pennylane plugin)  │
└────────┬───────┴────────┬────────┴─────────────┬────────────────┘
         │                │                       │
         │                └───────────┬───────────┘
         │                            │
         ▼                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Core Simulator Layer                        │
│  ┌────────────────────────────────────────────────────────────┐│
│  │            LRET Algorithm (simulator.cpp/h)                ││
│  │  • Low-rank density matrix: ρ = LL†                        ││
│  │  • Choi matrix representation for gates                    ││
│  │  • SVD-based rank truncation                               ││
│  │  • Efficient gate application: L' = apply_gate(L, gate)    ││
│  └────────────────────────────────────────────────────────────┘│
│  ┌────────────────────────────────────────────────────────────┐│
│  │         Gate Operations (gates_and_noise.cpp/h)            ││
│  │  • 1-qubit gates: H, X, Y, Z, RX, RY, RZ, S, T            ││
│  │  • 2-qubit gates: CNOT, CZ, SWAP, CRX, CRY, CRZ           ││
│  │  • 3-qubit gates: Toffoli, Fredkin                        ││
│  │  • Noise channels: Depolarizing, damping, leakage         ││
│  └────────────────────────────────────────────────────────────┘│
│  ┌────────────────────────────────────────────────────────────┐│
│  │      Parallelization (parallel_modes.cpp/h)                ││
│  │  • Sequential: Baseline single-threaded                    ││
│  │  • Row parallel: Parallelize L row updates (small rank)    ││
│  │  • Column parallel: Parallelize L column updates (large)   ││
│  │  • Hybrid: Auto-select based on rank/qubit count          ││
│  └────────────────────────────────────────────────────────────┘│
│  ┌────────────────────────────────────────────────────────────┐│
│  │         Circuit Optimization (circuit_optimizer.cpp/h)     ││
│  │  • Gate fusion: Combine adjacent single-qubit gates        ││
│  │  • Gate cancellation: X-X, H-H, CNOT-CNOT pairs           ││
│  │  • Commutation rules: Reorder gates for efficiency        ││
│  └────────────────────────────────────────────────────────────┘│
│  ┌────────────────────────────────────────────────────────────┐│
│  │            Noise Models (noise_import.cpp/h)               ││
│  │  • Built-in: Depolarizing, amplitude/phase damping        ││
│  │  • IBM device import: JSON-based calibration data         ││
│  │  • Custom models: User-defined noise specifications       ││
│  └────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
         │                            │
         ▼                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Infrastructure Layer                           │
│  ┌──────────────────┐  ┌──────────────────┐  ┌────────────────┐│
│  │ I/O & Serialization│  │  Resource Monitor │  │  Benchmarking  ││
│  │ • JSON interface  │  │  • Memory tracking│  │  • Timing      ││
│  │ • CSV output      │  │  • CPU usage      │  │  • Metrics     ││
│  │ • HDF5 support    │  │  • Rank tracking  │  │  • Reporting   ││
│  └──────────────────┘  └──────────────────┘  └────────────────┘│
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    External Libraries                            │
│  • Eigen3: Linear algebra (matrices, SVD)                       │
│  • OpenMP: Parallelization                                       │
│  • pybind11: Python bindings                                     │
│  • nlohmann/json: JSON parsing                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Design Principles

### 1. Low-Rank Decomposition

**Core Idea:** Represent density matrix $\rho$ in factorized form:

$$\rho = LL^\dagger, \quad L \in \mathbb{C}^{2^n \times r}$$

**Benefits:**
- Memory: $O(2^n \cdot r)$ vs $O(4^n)$ for full density matrix
- Operations: Apply gates to $L$ directly, avoid constructing full $\rho$
- Truncation: SVD-based rank reduction maintains fidelity

**Implementation:** `simulator.cpp`, class `LRETSimulator`

### 2. Choi Matrix Representation

**Gate Application:** Each quantum gate/noise channel is represented by its Choi matrix $\Lambda$:

$$\rho' = \mathcal{E}(\rho) \Leftrightarrow L' = \text{reshape}(\Lambda \cdot \text{vec}(L))$$

**Advantages:**
- Unified framework for unitary gates and noise channels
- Efficient matrix-vector operations
- Easy to compose multiple channels

**Implementation:** `gates_and_noise.cpp`, functions `apply_gate_choi()`

### 3. Adaptive Parallelization

**Strategy:** Choose parallelization mode based on circuit properties:

| Rank | Qubits | Best Mode | Reason |
|------|--------|-----------|--------|
| Small (< 50) | Any | Row parallel | Row updates independent |
| Large (> 50) | < 12 | Column parallel | Column updates independent |
| Any | > 12 | Hybrid | Auto-select per gate |

**Implementation:** `parallel_modes.cpp`, function `auto_select_parallel_mode()`

### 4. Modular Architecture

**Separation of Concerns:**
- **Core:** Algorithm implementation (simulator.cpp)
- **Gates:** Gate and noise definitions (gates_and_noise.cpp)
- **I/O:** Input/output handling (json_interface.cpp, output_formatter.cpp)
- **Bindings:** Language interfaces (python_bindings.cpp)
- **Tools:** CLI and utilities (main.cpp, cli_parser.cpp)

**Benefits:** Easy testing, extensibility, maintenance

---

## Core Components

### Simulator Core (`simulator.cpp/h`)

**Key Classes:**
- `LRETSimulator`: Main simulator class
  - State representation: Eigen matrix `L_`
  - Gate application: `apply_gate()`, `apply_noise()`
  - Rank management: `truncate_rank()`, `compute_svd()`
  - Measurement: `measure()`, `measure_all()`

**Key Algorithms:**
```cpp
// Gate application
MatrixXcd apply_gate(const MatrixXcd& L, const MatrixXcd& gate_choi, 
                     int target_qubit, int n_qubits);

// Rank truncation via SVD
void truncate_rank(MatrixXcd& L, double threshold);

// Parallel gate application
MatrixXcd apply_gate_parallel(const MatrixXcd& L, const MatrixXcd& gate_choi,
                              int target_qubit, int n_qubits, ParallelMode mode);
```

### Gate Library (`gates_and_noise.cpp/h`)

**Gate Definitions:**
- Single-qubit: Hadamard, Pauli-X/Y/Z, rotations (RX, RY, RZ), phase gates (S, T)
- Two-qubit: CNOT, CZ, SWAP, controlled rotations
- Three-qubit: Toffoli, Fredkin

**Noise Channels:**
- Depolarizing: $\mathcal{E}(\rho) = (1-p)\rho + \frac{p}{3}(X\rho X + Y\rho Y + Z\rho Z)$
- Amplitude damping: Energy relaxation (T1)
- Phase damping: Dephasing (T2)
- Leakage: Transitions to non-computational states

**Data Structures:**
```cpp
struct Gate {
    std::string name;
    std::vector<int> targets;
    std::vector<double> params;  // For parametric gates
    MatrixXcd choi_matrix;
};

struct NoiseChannel {
    std::string type;  // "depolarizing", "amplitude_damping", etc.
    double parameter;   // Error probability or rate
    std::vector<int> affected_qubits;
};
```

### Parallelization (`parallel_modes.cpp/h`)

**Parallel Strategies:**

**Row Parallel:**
```cpp
// Parallelize over rows of L
#pragma omp parallel for
for (int i = 0; i < L.rows(); i++) {
    L_new.row(i) = apply_gate_to_row(L.row(i), gate_choi);
}
```

**Column Parallel:**
```cpp
// Parallelize over columns of L
#pragma omp parallel for
for (int j = 0; j < L.cols(); j++) {
    L_new.col(j) = apply_gate_to_col(L.col(j), gate_choi);
}
```

**Hybrid:**
```cpp
ParallelMode select_mode(int rank, int n_qubits) {
    if (rank < 50) return ROW_PARALLEL;
    if (n_qubits < 12) return COLUMN_PARALLEL;
    return (rank < 100) ? ROW_PARALLEL : COLUMN_PARALLEL;
}
```

### I/O System (`json_interface.cpp/h`, `output_formatter.cpp/h`)

**Input Formats:**
- JSON circuit specification
- IBM device noise JSON
- Custom noise model JSON

**Output Formats:**
- CSV: Tabular results (time, rank, fidelity)
- JSON: Structured data (circuit, results, metadata)
- HDF5: Large-scale data (state vectors, density matrices)

**Example JSON Circuit:**
```json
{
  "n_qubits": 4,
  "gates": [
    {"type": "H", "target": 0},
    {"type": "CNOT", "control": 0, "target": 1},
    {"type": "RY", "target": 2, "angle": 0.5}
  ],
  "noise": {
    "type": "depolarizing",
    "level": 0.01
  }
}
```

---

## Data Flow

### 1. CLI Execution

```
User Command
    │
    ▼
cli_parser.cpp
  • Parse arguments
  • Validate parameters
    │
    ▼
simulator.cpp
  • Initialize state
  • Apply gates/noise
  • Truncate rank
    │
    ▼
output_formatter.cpp
  • Format results
  • Write to file/stdout
```

### 2. Python Execution

```
Python Script
    │
    ▼
python_bindings.cpp (pybind11)
  • Expose QuantumSimulator class
  • Convert Python → C++ data types
    │
    ▼
simulator.cpp
  • Run simulation
  • Return results
    │
    ▼
python_bindings.cpp
  • Convert C++ → Python types
  • Return to user
```

### 3. PennyLane Execution

```
PennyLane QNode
    │
    ▼
QLRETDevice (PennyLane plugin)
  • Convert PennyLane ops → LRET gates
  • Set up noise model
    │
    ▼
QuantumSimulator (Python)
    │
    ▼
python_bindings.cpp → simulator.cpp
  • Execute circuit
  • Compute expectation values
    │
    ▼
Return to PennyLane
  • Gradients (parameter-shift)
  • Optimization loop
```

---

## Memory Management

### State Representation

**Primary data structure:**
```cpp
Eigen::MatrixXcd L_;  // Low-rank factor: 2^n × r
```

**Memory usage:**
- Full density matrix: $4^n \times 16$ bytes (complex double)
- LRET factor: $2^n \times r \times 16$ bytes
- Typical rank: $r \ll 2^n$ (e.g., $r \approx 20$ for $n=12$)

**Example (12 qubits, rank 35):**
- Full: $4^{12} \times 16 = 268$ GB
- LRET: $2^{12} \times 35 \times 16 = 2.3$ MB
- **Reduction:** 116,000×

### Rank Truncation Strategy

```cpp
void truncate_rank(MatrixXcd& L, double threshold) {
    // 1. Compute SVD: L = UΣV†
    JacobiSVD<MatrixXcd> svd(L, ComputeThinU | ComputeThinV);
    
    // 2. Find truncation rank
    VectorXd singular_values = svd.singularValues();
    int new_rank = 0;
    double total_weight = singular_values.sum();
    double cumulative = 0.0;
    
    for (int i = 0; i < singular_values.size(); i++) {
        cumulative += singular_values(i);
        if (cumulative / total_weight >= (1.0 - threshold)) {
            new_rank = i + 1;
            break;
        }
    }
    
    // 3. Reconstruct with reduced rank
    MatrixXcd U = svd.matrixU().leftCols(new_rank);
    VectorXd S = singular_values.head(new_rank);
    L = U * S.asDiagonal();
}
```

**Truncation frequency:** After each gate application (configurable)

---

## Performance Optimizations

### 1. SIMD Vectorization (`simd_kernels.cpp/h`)

**Eigen's vectorization:**
- Automatically uses SSE/AVX instructions
- Aligned memory allocations
- Vectorized matrix operations

**Manual optimizations:**
```cpp
// Vectorized gate application
void apply_gate_simd(MatrixXcd& L, const MatrixXcd& gate) {
    // Eigen automatically vectorizes this
    L = gate * L;  // Uses BLAS under the hood
}
```

### 2. Gate Fusion (`gate_fusion.cpp/h`)

**Combine adjacent single-qubit gates:**
```cpp
// Before fusion: RZ(θ1) - RY(θ2) - RZ(θ3)
// After fusion: U(θ1, θ2, θ3) = RZ(θ1) · RY(θ2) · RZ(θ3)

Gate fuse_gates(const std::vector<Gate>& gates) {
    MatrixXcd combined = MatrixXcd::Identity(2, 2);
    for (const auto& gate : gates) {
        combined = gate.unitary * combined;
    }
    return Gate{"FUSED", combined};
}
```

**Speedup:** 2-3× for deep circuits with many single-qubit gates

### 3. Lazy Evaluation

**Batch gate application:**
```cpp
// Apply multiple gates before truncation
for (int i = 0; i < batch_size; i++) {
    L = apply_gate(L, gates[i]);
}
// Truncate once after batch
truncate_rank(L, threshold);
```

**Benefit:** Reduces SVD overhead (expensive operation)

### 4. Memory Pool Allocation

**Pre-allocate scratch space:**
```cpp
class LRETSimulator {
private:
    MatrixXcd scratch_L_;  // Pre-allocated workspace
    
public:
    void apply_gate(const Gate& gate) {
        // Reuse scratch_L_ instead of allocating new memory
        scratch_L_.resize(L_.rows(), L_.cols());
        // ... gate application ...
        L_.swap(scratch_L_);
    }
};
```

---

## Testing Strategy

**Unit Tests:** Individual components (gates, noise, truncation)
**Integration Tests:** End-to-end workflows (CLI, Python, PennyLane)
**Benchmarks:** Performance regression detection
**Validation:** Fidelity vs full-density-matrix simulation

See [Testing Framework](05-testing.md) for details.

---

## Extension Points

### Adding New Gates

1. Define gate matrix in `gates_and_noise.cpp`
2. Compute Choi representation
3. Add to gate registry
4. Expose in Python bindings

See [Extending the Simulator](04-extending-simulator.md).

### Adding New Noise Models

1. Implement Kraus operators
2. Construct Choi matrix
3. Add to noise channel registry
4. Support JSON configuration

### Adding New Parallel Modes

1. Implement in `parallel_modes.cpp`
2. Add selection heuristic
3. Benchmark vs existing modes
4. Document performance characteristics

---

## Performance Characteristics

### Time Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Gate application | $O(2^n \cdot r)$ | Per gate, depends on rank |
| SVD truncation | $O(\min(2^n, r)^3)$ | Bottleneck for large rank |
| Full simulation | $O(d \cdot 2^n \cdot r + T \cdot r^3)$ | $d$ = depth, $T$ = truncations |

### Space Complexity

| Structure | Size | Notes |
|-----------|------|-------|
| State factor $L$ | $O(2^n \cdot r)$ | Main memory usage |
| Gate Choi matrix | $O(4^k)$ | $k$ = gate arity (1, 2, or 3) |
| Scratch space | $O(2^n \cdot r)$ | Temporary allocations |

### Scaling Benchmarks

| Qubits | Rank | Time (LRET) | Time (FDM) | Speedup |
|--------|------|-------------|------------|---------|
| 8      | 12   | 0.05s       | 0.23s      | 4.6×    |
| 10     | 23   | 0.19s       | 1.2s       | 6.3×    |
| 12     | 35   | 0.71s       | 18.4s      | 25.9×   |
| 14     | 51   | 3.2s        | 294s       | 91.9×   |

---

## Further Reading

- **[Building from Source](01-building-from-source.md)** - Compilation instructions
- **[Code Structure](02-code-structure.md)** - Repository organization
- **[LRET Algorithm](03-lret-algorithm.md)** - Mathematical foundations
- **[Extending the Simulator](04-extending-simulator.md)** - Adding features
- **[Performance Guide](06-performance.md)** - Optimization techniques

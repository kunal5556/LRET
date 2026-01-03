# Extending the Simulator

Guide to adding new features, gates, noise models, and backends to LRET.

## Table of Contents

1. [Adding New Gates](#adding-new-gates)
2. [Adding Noise Channels](#adding-noise-channels)
3. [Custom Parallelization Strategies](#custom-parallelization-strategies)
4. [Adding New Backends](#adding-new-backends)
5. [Extending the Python Interface](#extending-the-python-interface)
6. [Adding PennyLane Operations](#adding-pennylane-operations)
7. [Custom Output Formats](#custom-output-formats)
8. [Adding Benchmarks](#adding-benchmarks)

---

## Adding New Gates

### Single-Qubit Gates

**Step 1: Define the gate unitary**

Create unitary matrix in `src/gates_and_noise.cpp`:

```cpp
MatrixXcd my_custom_gate(double theta) {
    MatrixXcd U(2, 2);
    U << cos(theta), -sin(theta),
         sin(theta),  cos(theta);
    return U;
}
```

**Step 2: Compute Choi matrix**

```cpp
MatrixXcd gate_to_choi(const MatrixXcd& U) {
    // For unitary U, Choi matrix is (U* ⊗ U)
    return kronecker_product(U.conjugate(), U);
}

MatrixXcd my_custom_gate_choi(double theta) {
    MatrixXcd U = my_custom_gate(theta);
    return gate_to_choi(U);
}
```

**Step 3: Add to gate registry**

In `include/gates_and_noise.h`:

```cpp
struct Gate {
    std::string name;
    GateType type;
    std::vector<int> targets;
    std::vector<double> params;
    MatrixXcd unitary;
    MatrixXcd choi_matrix;
};

// Add factory function
Gate create_my_custom_gate(int target, double theta);
```

In `src/gates_and_noise.cpp`:

```cpp
Gate create_my_custom_gate(int target, double theta) {
    Gate g;
    g.name = "MyCustomGate";
    g.type = GateType::SINGLE_QUBIT;
    g.targets = {target};
    g.params = {theta};
    g.unitary = my_custom_gate(theta);
    g.choi_matrix = my_custom_gate_choi(theta);
    return g;
}

// Add to registry
std::map<std::string, std::function<Gate(std::vector<int>, std::vector<double>)>> gate_registry = {
    {"H", [](auto t, auto p) { return hadamard_gate(t[0]); }},
    {"X", [](auto t, auto p) { return pauli_x_gate(t[0]); }},
    // ... existing gates ...
    {"MyCustomGate", [](auto t, auto p) { return create_my_custom_gate(t[0], p[0]); }},
};
```

**Step 4: Test the gate**

Create test in `tests/test_gates.cpp`:

```cpp
TEST(GateTest, MyCustomGate) {
    Gate g = create_my_custom_gate(0, M_PI / 4);
    
    EXPECT_EQ(g.name, "MyCustomGate");
    EXPECT_EQ(g.targets.size(), 1);
    EXPECT_EQ(g.targets[0], 0);
    
    // Test unitary properties
    MatrixXcd U = g.unitary;
    MatrixXcd I = MatrixXcd::Identity(2, 2);
    EXPECT_TRUE((U * U.adjoint()).isApprox(I, 1e-10));  // Unitarity
    
    // Test Choi matrix
    MatrixXcd C = g.choi_matrix;
    EXPECT_TRUE(C.isApprox(kronecker_product(U.conjugate(), U), 1e-10));
}
```

---

### Two-Qubit Gates

**Example: Custom controlled gate**

```cpp
MatrixXcd controlled_my_gate(double theta) {
    MatrixXcd U = my_custom_gate(theta);
    MatrixXcd CU = MatrixXcd::Identity(4, 4);
    CU.block(2, 2, 2, 2) = U;  // Apply U to |11⟩ subspace
    return CU;
}

Gate create_controlled_my_gate(int control, int target, double theta) {
    Gate g;
    g.name = "CMyCustomGate";
    g.type = GateType::TWO_QUBIT;
    g.targets = {control, target};
    g.params = {theta};
    g.unitary = controlled_my_gate(theta);
    g.choi_matrix = gate_to_choi(g.unitary);
    return g;
}
```

---

### Three-Qubit Gates

**Example: Toffoli gate (CCNOT)**

```cpp
MatrixXcd toffoli_gate() {
    MatrixXcd U = MatrixXcd::Identity(8, 8);
    // Swap |110⟩ ↔ |111⟩
    U(6, 6) = 0; U(6, 7) = 1;
    U(7, 6) = 1; U(7, 7) = 0;
    return U;
}

Gate create_toffoli_gate(int control1, int control2, int target) {
    Gate g;
    g.name = "Toffoli";
    g.type = GateType::THREE_QUBIT;
    g.targets = {control1, control2, target};
    g.unitary = toffoli_gate();
    g.choi_matrix = gate_to_choi(g.unitary);
    return g;
}
```

---

## Adding Noise Channels

### Single-Qubit Noise

**Example: Generalized amplitude damping**

Combines amplitude damping with thermal excitation.

```cpp
struct NoiseChannel {
    std::string type;
    std::vector<double> params;
    std::vector<int> affected_qubits;
    std::vector<MatrixXcd> kraus_operators;
    MatrixXcd choi_matrix;
};

NoiseChannel generalized_amplitude_damping(int qubit, double gamma, double p_excited) {
    NoiseChannel noise;
    noise.type = "generalized_amplitude_damping";
    noise.params = {gamma, p_excited};
    noise.affected_qubits = {qubit};
    
    // Kraus operators
    double p = p_excited;
    MatrixXcd K0(2, 2), K1(2, 2), K2(2, 2), K3(2, 2);
    
    K0 << sqrt(p), 0,
          0, sqrt(p * (1 - gamma));
    
    K1 << 0, sqrt(p * gamma),
          0, 0;
    
    K2 << sqrt((1-p) * (1-gamma)), 0,
          0, sqrt(1-p);
    
    K3 << 0, 0,
          sqrt((1-p) * gamma), 0;
    
    noise.kraus_operators = {K0, K1, K2, K3};
    
    // Compute Choi matrix
    noise.choi_matrix = MatrixXcd::Zero(4, 4);
    for (const auto& K : noise.kraus_operators) {
        noise.choi_matrix += kronecker_product(K.conjugate(), K);
    }
    
    return noise;
}
```

**Step 2: Add to noise registry**

```cpp
std::map<std::string, std::function<NoiseChannel(int, std::vector<double>)>> noise_registry = {
    {"depolarizing", [](int q, auto p) { return depolarizing_noise(q, p[0]); }},
    {"amplitude_damping", [](int q, auto p) { return amplitude_damping(q, p[0]); }},
    // ... existing noise ...
    {"generalized_amplitude_damping", [](int q, auto p) { 
        return generalized_amplitude_damping(q, p[0], p[1]); 
    }},
};
```

**Step 3: Test the noise channel**

```cpp
TEST(NoiseTest, GeneralizedAmplitudeDamping) {
    NoiseChannel noise = generalized_amplitude_damping(0, 0.1, 0.2);
    
    EXPECT_EQ(noise.kraus_operators.size(), 4);
    
    // Test CPTP conditions
    MatrixXcd sum = MatrixXcd::Zero(2, 2);
    for (const auto& K : noise.kraus_operators) {
        sum += K.adjoint() * K;
    }
    MatrixXcd I = MatrixXcd::Identity(2, 2);
    EXPECT_TRUE(sum.isApprox(I, 1e-10));  // Trace preservation
}
```

---

### Two-Qubit Correlated Noise

**Example: Correlated depolarizing**

```cpp
NoiseChannel correlated_depolarizing(int qubit1, int qubit2, double p, double correlation) {
    NoiseChannel noise;
    noise.type = "correlated_depolarizing";
    noise.params = {p, correlation};
    noise.affected_qubits = {qubit1, qubit2};
    
    // Pauli matrices
    MatrixXcd I = MatrixXcd::Identity(2, 2);
    MatrixXcd X(2, 2); X << 0, 1, 1, 0;
    MatrixXcd Y(2, 2); Y << 0, -1i, 1i, 0;
    MatrixXcd Z(2, 2); Z << 1, 0, 0, -1;
    
    std::vector<MatrixXcd> paulis = {I, X, Y, Z};
    
    // Build Kraus operators with correlation
    std::vector<MatrixXcd> kraus_ops;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            double weight;
            if (i == 0 && j == 0) {
                weight = sqrt(1 - p);
            } else if (i == j && i != 0) {
                weight = sqrt(p * correlation / 3);
            } else if (i != 0 && j != 0) {
                weight = sqrt(p * (1 - correlation) / 15);
            } else {
                weight = 0;
            }
            
            if (weight > 1e-10) {
                MatrixXcd K = weight * kronecker_product(paulis[i], paulis[j]);
                kraus_ops.push_back(K);
            }
        }
    }
    
    noise.kraus_operators = kraus_ops;
    
    // Compute Choi matrix
    noise.choi_matrix = MatrixXcd::Zero(16, 16);
    for (const auto& K : kraus_ops) {
        noise.choi_matrix += kronecker_product(K.conjugate(), K);
    }
    
    return noise;
}
```

---

## Custom Parallelization Strategies

### Adding a New Parallel Mode

**Step 1: Define the mode**

In `include/parallel_modes.h`:

```cpp
enum ParallelMode {
    SEQUENTIAL,
    ROW_PARALLEL,
    COLUMN_PARALLEL,
    HYBRID,
    CUSTOM_MODE  // Add new mode
};
```

**Step 2: Implement the strategy**

In `src/parallel_modes.cpp`:

```cpp
MatrixXcd apply_gate_custom_mode(
    const MatrixXcd& L,
    const MatrixXcd& gate_choi,
    int target_qubit,
    int n_qubits
) {
    int rows = L.rows();
    int cols = L.cols();
    MatrixXcd L_new = MatrixXcd::Zero(rows, cols);
    
    // Custom parallelization strategy
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < rows; ++i) {
        // Custom logic for processing row i
        for (int j = 0; j < cols; ++j) {
            // Compute L_new(i, j)
            // ...
        }
    }
    
    return L_new;
}
```

**Step 3: Add dispatcher**

```cpp
MatrixXcd apply_gate_parallel(
    const MatrixXcd& L,
    const MatrixXcd& gate_choi,
    int target_qubit,
    int n_qubits,
    ParallelMode mode
) {
    switch (mode) {
        case SEQUENTIAL:
            return apply_gate_sequential(L, gate_choi, target_qubit, n_qubits);
        case ROW_PARALLEL:
            return apply_gate_row_parallel(L, gate_choi, target_qubit, n_qubits);
        case COLUMN_PARALLEL:
            return apply_gate_column_parallel(L, gate_choi, target_qubit, n_qubits);
        case HYBRID:
            return apply_gate_hybrid(L, gate_choi, target_qubit, n_qubits);
        case CUSTOM_MODE:
            return apply_gate_custom_mode(L, gate_choi, target_qubit, n_qubits);
        default:
            throw std::invalid_argument("Unknown parallel mode");
    }
}
```

---

## Adding New Backends

### GPU Backend (CUDA)

**Step 1: Define GPU simulator**

In `include/gpu_simulator.h`:

```cpp
#ifdef ENABLE_GPU

class GPUSimulator {
public:
    GPUSimulator(int n_qubits, double noise_level = 0.0);
    ~GPUSimulator();
    
    void apply_gate(const Gate& gate);
    void apply_noise(const NoiseChannel& noise);
    std::map<std::string, int> measure_all(int shots);
    
    MatrixXcd get_density_matrix() const;
    int get_rank() const;
    
private:
    int n_qubits_;
    int current_rank_;
    
    // GPU memory
    cuDoubleComplex* d_L_;  // Device pointer
    cuDoubleComplex* d_temp_;
    
    void allocate_gpu_memory();
    void free_gpu_memory();
    void host_to_device();
    void device_to_host();
};

#endif  // ENABLE_GPU
```

**Step 2: Implement CUDA kernels**

In `src/gpu_simulator.cu`:

```cuda
__global__ void apply_single_qubit_gate_kernel(
    cuDoubleComplex* L_new,
    const cuDoubleComplex* L,
    const cuDoubleComplex* gate_choi,
    int target_qubit,
    int n_qubits,
    int rank
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = (1 << (2 * n_qubits)) * rank;
    
    if (idx < total_size) {
        // Compute L_new[idx] from L and gate_choi
        // ...
    }
}

void GPUSimulator::apply_gate(const Gate& gate) {
    // Copy gate Choi matrix to device
    cuDoubleComplex* d_choi;
    cudaMalloc(&d_choi, gate.choi_matrix.size() * sizeof(cuDoubleComplex));
    cudaMemcpy(d_choi, gate.choi_matrix.data(), /*...*/);
    
    // Launch kernel
    int threads_per_block = 256;
    int blocks = (total_size + threads_per_block - 1) / threads_per_block;
    apply_single_qubit_gate_kernel<<<blocks, threads_per_block>>>(
        d_temp_, d_L_, d_choi, gate.targets[0], n_qubits_, current_rank_
    );
    
    // Swap pointers
    std::swap(d_L_, d_temp_);
    
    cudaFree(d_choi);
}
```

**Step 3: Integrate with main simulator**

```cpp
#include "gpu_simulator.h"

class LRETSimulator {
    // ...
    
    void use_gpu_backend(bool enable) {
#ifdef ENABLE_GPU
        if (enable) {
            gpu_sim_ = std::make_unique<GPUSimulator>(n_qubits_, noise_level_);
            use_gpu_ = true;
        } else {
            gpu_sim_ = nullptr;
            use_gpu_ = false;
        }
#else
        if (enable) {
            throw std::runtime_error("GPU support not compiled");
        }
#endif
    }
    
private:
#ifdef ENABLE_GPU
    std::unique_ptr<GPUSimulator> gpu_sim_;
    bool use_gpu_ = false;
#endif
};
```

---

### MPI Backend (Distributed)

**Step 1: Define MPI communicator**

In `include/mpi_parallel.h`:

```cpp
#ifdef ENABLE_MPI

#include <mpi.h>

class MPISimulator {
public:
    MPISimulator(int n_qubits, double noise_level, MPI_Comm comm);
    ~MPISimulator();
    
    void apply_gate(const Gate& gate);
    void distribute_state();
    void gather_results();
    
private:
    MPI_Comm comm_;
    int rank_;
    int size_;
    
    // Local portion of L
    MatrixXcd L_local_;
    int local_rows_start_;
    int local_rows_end_;
};

#endif  // ENABLE_MPI
```

**Step 2: Implement distributed gate application**

In `src/mpi_parallel.cpp`:

```cpp
void MPISimulator::apply_gate(const Gate& gate) {
    // Distribute Choi matrix to all ranks
    MPI_Bcast(gate.choi_matrix.data(), /*...*/, MPI_DOUBLE_COMPLEX, 0, comm_);
    
    // Apply gate to local rows
    MatrixXcd L_local_new = apply_gate_local(L_local_, gate.choi_matrix, gate.targets[0]);
    
    // Synchronize
    MPI_Barrier(comm_);
    
    L_local_ = L_local_new;
}

void MPISimulator::distribute_state() {
    int total_rows = L_global_.rows();
    int rows_per_rank = total_rows / size_;
    
    local_rows_start_ = rank_ * rows_per_rank;
    local_rows_end_ = (rank_ == size_ - 1) ? total_rows : (rank_ + 1) * rows_per_rank;
    
    // Scatter rows to ranks
    MPI_Scatter(L_global_.data(), rows_per_rank * L_global_.cols(), MPI_DOUBLE_COMPLEX,
                L_local_.data(), rows_per_rank * L_local_.cols(), MPI_DOUBLE_COMPLEX,
                0, comm_);
}
```

---

## Extending the Python Interface

### Adding New Python Methods

**Step 1: Expose in pybind11**

In `src/python_bindings.cpp`:

```cpp
PYBIND11_MODULE(qlret_core, m) {
    py::class_<LRETSimulator>(m, "LRETSimulator")
        .def(py::init<int, double>())
        // ... existing methods ...
        .def("get_expectation_value", &LRETSimulator::get_expectation_value,
             "Compute expectation value of observable")
        .def("get_reduced_density_matrix", &LRETSimulator::get_reduced_density_matrix,
             "Compute reduced density matrix of subsystem")
        ;
}
```

**Step 2: Add C++ implementation**

In `src/simulator.cpp`:

```cpp
double LRETSimulator::get_expectation_value(const MatrixXcd& observable) const {
    // <O> = Tr(O ρ)
    MatrixXcd rho = get_density_matrix();
    return (observable * rho).trace().real();
}

MatrixXcd LRETSimulator::get_reduced_density_matrix(const std::vector<int>& subsystem) const {
    // Partial trace over complement of subsystem
    MatrixXcd rho = get_density_matrix();
    return partial_trace(rho, subsystem, n_qubits_);
}
```

**Step 3: Add Python wrapper**

In `python/qlret/simulator.py`:

```python
class QuantumSimulator:
    # ... existing methods ...
    
    def expectation(self, observable):
        """Compute expectation value.
        
        Args:
            observable: Observable matrix (numpy array or string)
        
        Returns:
            float: Expectation value <ψ|O|ψ>
        
        Example:
            >>> sim.h(0)
            >>> sim.expectation("Z")  # <Z> on qubit 0
            0.0
        """
        if isinstance(observable, str):
            observable = self._parse_observable_string(observable)
        
        return self._sim.get_expectation_value(observable)
    
    def reduced_density_matrix(self, qubits):
        """Get reduced density matrix.
        
        Args:
            qubits: List of qubit indices to keep
        
        Returns:
            numpy.ndarray: Reduced density matrix
        
        Example:
            >>> sim.h(0)
            >>> sim.cnot(0, 1)
            >>> rho_0 = sim.reduced_density_matrix([0])
            >>> print(rho_0)  # Maximally mixed state
        """
        return self._sim.get_reduced_density_matrix(qubits)
```

---

## Adding PennyLane Operations

### Custom PennyLane Operation

**Step 1: Define operation**

In `python/qlret/pennylane_device.py`:

```python
from pennylane.operation import Operation

class MyCustomOperation(Operation):
    """Custom quantum operation."""
    
    num_wires = 1
    num_params = 1
    par_domain = "R"  # Real parameters
    
    @staticmethod
    def compute_matrix(theta):
        """Compute unitary matrix."""
        import numpy as np
        return np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])
```

**Step 2: Register with device**

```python
class QLRETDevice(QubitDevice):
    operations = {
        "PauliX", "PauliY", "PauliZ",
        "Hadamard", "CNOT", "RX", "RY", "RZ",
        "MyCustomOperation",  # Add new operation
    }
    
    def _apply_operation(self, operation):
        if operation.name == "MyCustomOperation":
            theta = operation.parameters[0]
            target = operation.wires[0]
            self.sim.apply_custom_gate(target, theta)
        else:
            # ... existing logic ...
            pass
```

**Step 3: Test with PennyLane**

```python
import pennylane as qml

dev = qml.device("qlret.simulator", wires=2)

@qml.qnode(dev)
def circuit(theta):
    qml.MyCustomOperation(theta, wires=0)
    return qml.expval(qml.PauliZ(0))

result = circuit(0.5)
print(f"Expectation: {result}")
```

---

## Custom Output Formats

### Adding HDF5 Output

**Step 1: Add HDF5 support**

In `CMakeLists.txt`:

```cmake
find_package(HDF5 REQUIRED COMPONENTS CXX)
target_link_libraries(qlret_core PUBLIC HDF5::HDF5)
```

**Step 2: Implement HDF5 writer**

In `include/output_formatter.h`:

```cpp
#ifdef ENABLE_HDF5
#include <H5Cpp.h>

class HDF5OutputFormatter {
public:
    static void write_results(
        const std::string& filename,
        const std::map<std::string, int>& results,
        const MatrixXcd& density_matrix,
        const std::map<std::string, double>& metadata
    );
};
#endif
```

In `src/output_formatter.cpp`:

```cpp
void HDF5OutputFormatter::write_results(
    const std::string& filename,
    const std::map<std::string, int>& results,
    const MatrixXcd& density_matrix,
    const std::map<std::string, double>& metadata
) {
    H5::H5File file(filename, H5F_ACC_TRUNC);
    
    // Write measurement results
    hsize_t dims[1] = {results.size()};
    H5::DataSpace dataspace(1, dims);
    H5::DataSet dataset = file.createDataSet(
        "measurements", H5::PredType::NATIVE_INT, dataspace
    );
    // ... write data ...
    
    // Write density matrix
    // ... write matrix ...
    
    // Write metadata
    // ... write metadata ...
    
    file.close();
}
```

---

## Adding Benchmarks

### Creating a Custom Benchmark

**Step 1: Define benchmark**

In `scripts/benchmark_suite.py`:

```python
def benchmark_custom(n_qubits, depth, noise_level):
    """Custom benchmark scenario."""
    
    results = []
    for trial in range(10):
        # Build custom circuit
        circuit = build_custom_circuit(n_qubits, depth)
        
        # Run simulation
        start_time = time.time()
        sim = QuantumSimulator(n_qubits, noise_level)
        for gate in circuit:
            sim.apply_gate(gate)
        measurements = sim.measure_all(shots=1000)
        elapsed = time.time() - start_time
        
        results.append({
            'trial': trial,
            'time': elapsed,
            'rank': sim.current_rank,
            'memory_mb': sim.memory_usage / 1e6
        })
    
    return pd.DataFrame(results)
```

**Step 2: Add to benchmark registry**

```python
BENCHMARK_REGISTRY = {
    'scaling': benchmark_scaling,
    'parallel': benchmark_parallel,
    'accuracy': benchmark_accuracy,
    'depth': benchmark_depth,
    'memory': benchmark_memory,
    'custom': benchmark_custom,  # Add custom benchmark
}
```

**Step 3: Run benchmark**

```bash
python scripts/benchmark_suite.py --benchmark custom --qubits 10 --depth 50
```

---

## See Also

- **[Code Structure](02-code-structure.md)** - Repository organization
- **[LRET Algorithm](03-lret-algorithm.md)** - Algorithm details
- **[Testing Framework](05-testing.md)** - Testing your extensions
- **[Performance Optimization](06-performance.md)** - Optimizing custom code
- **[Contributing Guidelines](07-contributing.md)** - Submission process

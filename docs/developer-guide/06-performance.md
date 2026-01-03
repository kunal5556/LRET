# Performance Optimization

Comprehensive guide to profiling, optimizing, and scaling LRET simulations.

## Table of Contents

1. [Performance Analysis](#performance-analysis)
2. [Profiling Tools](#profiling-tools)
3. [Algorithmic Optimizations](#algorithmic-optimizations)
4. [Memory Optimization](#memory-optimization)
5. [Parallelization Strategies](#parallelization-strategies)
6. [SIMD Vectorization](#simd-vectorization)
7. [GPU Acceleration](#gpu-acceleration)
8. [Compiler Optimizations](#compiler-optimizations)
9. [Benchmarking](#benchmarking)

---

## Performance Analysis

### Complexity Bottlenecks

LRET performance is dominated by three operations:

1. **Gate Application:** $O(4^{n-k} \cdot 16^k \cdot r)$ for $k$-qubit gate
   - Single-qubit: $O(4^{n-1} \cdot r)$
   - Two-qubit: $O(4^{n-1} \cdot r)$
   
2. **Rank Truncation (SVD):** $O(4^n \cdot r^2)$
   - Can be expensive for large $r$
   - Adaptive truncation amortizes cost
   
3. **Measurement Sampling:** $O(2^n)$ per sample
   - Computing all probabilities is expensive
   - On-the-fly sampling more efficient

### Performance Targets

| Metric | Target | Current (v1.0) |
|--------|--------|---------------|
| 20-qubit, depth 50 | <10s | 8.5s |
| 25-qubit, depth 100 | <60s | 55s |
| Parallel speedup (8 cores) | >6x | 7.2x |
| Memory (20 qubits, rank 50) | <500 MB | 420 MB |
| GPU speedup (A100) | >10x | 12.5x |

---

## Profiling Tools

### Linux Profiling (perf)

**Installation:**
```bash
sudo apt install linux-tools-common linux-tools-generic
```

**Profile execution:**
```bash
# Record performance data
perf record -g ./quantum_sim --qubits 20 --depth 50

# Analyze results
perf report

# Flamegraph visualization
perf script | stackcollapse-perf.pl | flamegraph.pl > flamegraph.svg
```

**Example output:**
```
Overhead  Command      Shared Object     Symbol
  45.2%   quantum_sim  quantum_sim       [.] apply_gate_parallel
  23.1%   quantum_sim  libeigen.so       [.] MatrixXcd::operator*
  12.5%   quantum_sim  quantum_sim       [.] truncate_rank
   8.3%   quantum_sim  libgomp.so        [.] GOMP_parallel
```

---

### gprof (GNU Profiler)

**Compile with profiling:**
```bash
g++ -pg -O2 -o quantum_sim main.cpp src/*.cpp -lEigen3
```

**Run and analyze:**
```bash
./quantum_sim --qubits 15 --depth 30
gprof quantum_sim gmon.out > analysis.txt
```

---

### Valgrind (Memory Profiling)

**Check memory leaks:**
```bash
valgrind --leak-check=full ./quantum_sim --qubits 10 --depth 20
```

**Cache profiling:**
```bash
valgrind --tool=cachegrind ./quantum_sim --qubits 15 --depth 30
cg_annotate cachegrind.out.<pid>
```

**Heap profiling:**
```bash
valgrind --tool=massif ./quantum_sim --qubits 15 --depth 30
ms_print massif.out.<pid>
```

---

### Python Profiling

**cProfile:**
```python
import cProfile
import pstats
from qlret import QuantumSimulator

def run_simulation():
    sim = QuantumSimulator(n_qubits=15)
    for _ in range(30):
        sim.h(0)
        sim.cnot(0, 1)
    sim.measure_all(shots=1000)

cProfile.run('run_simulation()', 'profile.stats')
p = pstats.Stats('profile.stats')
p.sort_stats('cumulative').print_stats(20)
```

**line_profiler:**
```python
from line_profiler import LineProfiler
from qlret import QuantumSimulator

def run_simulation():
    sim = QuantumSimulator(n_qubits=15)
    for _ in range(30):
        sim.h(0)
        sim.cnot(0, 1)

lp = LineProfiler()
lp.add_function(run_simulation)
lp.run('run_simulation()')
lp.print_stats()
```

**memory_profiler:**
```python
from memory_profiler import profile

@profile
def run_simulation():
    sim = QuantumSimulator(n_qubits=20)
    for _ in range(50):
        sim.h(0)

run_simulation()
```

---

## Algorithmic Optimizations

### Gate Fusion

**Problem:** Applying two consecutive single-qubit gates on same qubit requires two matrix multiplications.

**Solution:** Fuse gates before applying.

**Implementation:**

```cpp
Gate fuse_single_qubit_gates(const Gate& g1, const Gate& g2) {
    if (g1.targets[0] != g2.targets[0]) {
        throw std::invalid_argument("Gates act on different qubits");
    }
    
    Gate fused;
    fused.name = g1.name + "*" + g2.name;
    fused.type = GateType::SINGLE_QUBIT;
    fused.targets = g1.targets;
    
    // Fuse unitaries: U_fused = U2 * U1
    fused.unitary = g2.unitary * g1.unitary;
    fused.choi_matrix = gate_to_choi(fused.unitary);
    
    return fused;
}

std::vector<Gate> optimize_circuit(const std::vector<Gate>& circuit) {
    std::vector<Gate> optimized;
    
    for (size_t i = 0; i < circuit.size(); ++i) {
        if (i + 1 < circuit.size() &&
            circuit[i].type == GateType::SINGLE_QUBIT &&
            circuit[i+1].type == GateType::SINGLE_QUBIT &&
            circuit[i].targets[0] == circuit[i+1].targets[0]) {
            
            // Fuse consecutive single-qubit gates
            optimized.push_back(fuse_single_qubit_gates(circuit[i], circuit[i+1]));
            ++i;  // Skip next gate
        } else {
            optimized.push_back(circuit[i]);
        }
    }
    
    return optimized;
}
```

**Speedup:** Up to 2x for circuits with many consecutive single-qubit gates.

---

### Lazy Truncation

**Problem:** Truncating after every gate is expensive.

**Solution:** Truncate only when rank exceeds threshold.

**Implementation:**

```cpp
class LRETSimulator {
private:
    int max_rank_ = 100;
    int truncation_frequency_ = 10;  // Truncate every N gates
    int gates_since_truncation_ = 0;
    
public:
    void apply_gate(const Gate& gate) {
        // Apply gate
        L_ = apply_gate_choi(L_, gate.choi_matrix, gate.targets[0], n_qubits_);
        current_rank_ = L_.cols();
        
        ++gates_since_truncation_;
        
        // Truncate if necessary
        if (current_rank_ > max_rank_ * 1.5 ||
            gates_since_truncation_ >= truncation_frequency_) {
            truncate_rank();
            gates_since_truncation_ = 0;
        }
    }
};
```

**Speedup:** 20-30% for deep circuits.

---

### Adaptive Rank Budget

**Problem:** Fixed rank limit wastes resources for easy circuits, insufficient for hard circuits.

**Solution:** Dynamically adjust rank based on truncation error.

**Implementation:**

```cpp
void LRETSimulator::adaptive_truncate() {
    // Compute SVD
    Eigen::JacobiSVD<MatrixXcd> svd(L_, Eigen::ComputeThinU | Eigen::ComputeThinV);
    VectorXd sigma = svd.singularValues();
    
    double total_variance = sigma.squaredNorm();
    double target_fidelity = 1.0 - truncation_threshold_ * truncation_threshold_;
    
    // Binary search for optimal rank
    int low = 1, high = sigma.size();
    int optimal_rank = high;
    
    while (low <= high) {
        int mid = (low + high) / 2;
        double variance_kept = sigma.head(mid).squaredNorm();
        double fidelity = variance_kept / total_variance;
        
        if (fidelity >= target_fidelity) {
            optimal_rank = mid;
            high = mid - 1;
        } else {
            low = mid + 1;
        }
    }
    
    // Apply truncation
    L_ = svd.matrixU().leftCols(optimal_rank) *
         sigma.head(optimal_rank).asDiagonal();
    current_rank_ = optimal_rank;
}
```

---

## Memory Optimization

### Memory Layout

**Problem:** Poor cache locality for large matrices.

**Solution:** Use column-major layout (Eigen default) and access patterns.

**Best practices:**

```cpp
// Good: Column-major access
for (int j = 0; j < L.cols(); ++j) {
    for (int i = 0; i < L.rows(); ++i) {
        L(i, j) = ...;
    }
}

// Bad: Row-major access (cache misses)
for (int i = 0; i < L.rows(); ++i) {
    for (int j = 0; j < L.cols(); ++j) {
        L(i, j) = ...;
    }
}
```

---

### Memory Pools

**Problem:** Frequent allocation/deallocation of matrices.

**Solution:** Reuse memory buffers.

**Implementation:**

```cpp
class MemoryPool {
private:
    std::vector<MatrixXcd> buffers_;
    std::vector<bool> in_use_;
    
public:
    MatrixXcd* allocate(int rows, int cols) {
        // Find free buffer of correct size
        for (size_t i = 0; i < buffers_.size(); ++i) {
            if (!in_use_[i] && buffers_[i].rows() == rows && buffers_[i].cols() == cols) {
                in_use_[i] = true;
                return &buffers_[i];
            }
        }
        
        // Allocate new buffer
        buffers_.emplace_back(rows, cols);
        in_use_.push_back(true);
        return &buffers_.back();
    }
    
    void deallocate(MatrixXcd* ptr) {
        for (size_t i = 0; i < buffers_.size(); ++i) {
            if (&buffers_[i] == ptr) {
                in_use_[i] = false;
                return;
            }
        }
    }
};
```

---

### In-Place Operations

**Problem:** Temporary matrices consume memory.

**Solution:** Use in-place operations where possible.

```cpp
// Bad: Creates temporary
MatrixXcd L_new = C * L;
L = L_new;

// Good: In-place (if possible)
L = C * L;  // Eigen optimizes this

// Even better: Use noalias()
L.noalias() = C * L_old;
```

---

## Parallelization Strategies

### OpenMP Parallelization

**Row-wise parallelization:**

```cpp
MatrixXcd apply_gate_row_parallel(const MatrixXcd& L, const MatrixXcd& C) {
    int rows = L.rows();
    int cols = L.cols();
    MatrixXcd L_new(rows, cols);
    
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < rows; ++i) {
        L_new.row(i) = C.row(i) * L;
    }
    
    return L_new;
}
```

**Dynamic scheduling for load balancing:**

```cpp
#pragma omp parallel for schedule(dynamic, 64)
for (int i = 0; i < rows; ++i) {
    // Work with varying complexity
    L_new.row(i) = complex_computation(i);
}
```

**Reduction for measurements:**

```cpp
double compute_probability(const VectorXcd& state, int outcome) {
    double prob = 0.0;
    
    #pragma omp parallel for reduction(+:prob)
    for (int i = 0; i < state.size(); ++i) {
        if (matches_outcome(i, outcome)) {
            prob += std::norm(state(i));
        }
    }
    
    return prob;
}
```

---

### MPI Distributed Computing

**Distribute rows across ranks:**

```cpp
void MPISimulator::distribute_L() {
    int total_rows = L_.rows();
    int rows_per_rank = total_rows / mpi_size_;
    
    local_start_ = mpi_rank_ * rows_per_rank;
    local_end_ = (mpi_rank_ == mpi_size_ - 1) ? total_rows : (mpi_rank_ + 1) * rows_per_rank;
    
    L_local_ = L_.middleRows(local_start_, local_end_ - local_start_);
}

void MPISimulator::apply_gate_distributed(const Gate& gate) {
    // Broadcast Choi matrix
    MPI_Bcast(gate.choi_matrix.data(), gate.choi_matrix.size(), MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
    
    // Local computation
    MatrixXcd L_local_new = apply_gate_local(L_local_, gate.choi_matrix);
    
    // Synchronize
    MPI_Barrier(MPI_COMM_WORLD);
    
    L_local_ = L_local_new;
}

void MPISimulator::gather_results() {
    // Gather local results to rank 0
    std::vector<int> recvcounts(mpi_size_);
    std::vector<int> displs(mpi_size_);
    
    int local_size = L_local_.size();
    MPI_Gather(&local_size, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (mpi_rank_ == 0) {
        displs[0] = 0;
        for (int i = 1; i < mpi_size_; ++i) {
            displs[i] = displs[i-1] + recvcounts[i-1];
        }
    }
    
    MPI_Gatherv(L_local_.data(), local_size, MPI_DOUBLE_COMPLEX,
                L_.data(), recvcounts.data(), displs.data(), MPI_DOUBLE_COMPLEX,
                0, MPI_COMM_WORLD);
}
```

---

## SIMD Vectorization

### AVX2/AVX-512 Intrinsics

**Vectorized complex multiplication:**

```cpp
#include <immintrin.h>

void complex_mult_avx2(
    const std::complex<double>* a,
    const std::complex<double>* b,
    std::complex<double>* c,
    int n
) {
    for (int i = 0; i < n; i += 2) {
        // Load 2 complex numbers (4 doubles)
        __m256d va = _mm256_loadu_pd(reinterpret_cast<const double*>(a + i));
        __m256d vb = _mm256_loadu_pd(reinterpret_cast<const double*>(b + i));
        
        // Complex multiplication: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
        __m256d ac_bd = _mm256_mul_pd(va, vb);
        __m256d ad_bc = _mm256_mul_pd(va, _mm256_permute_pd(vb, 0x5));
        
        __m256d result = _mm256_addsub_pd(ac_bd, ad_bc);
        
        // Store result
        _mm256_storeu_pd(reinterpret_cast<double*>(c + i), result);
    }
}
```

**Auto-vectorization hints:**

```cpp
// Help compiler vectorize
void multiply_vectors(const double* a, const double* b, double* c, int n) {
    #pragma omp simd
    for (int i = 0; i < n; ++i) {
        c[i] = a[i] * b[i];
    }
}

// Alignment hints
void aligned_operation(const double* __restrict__ a, const double* __restrict__ b, double* __restrict__ c, int n) {
    __builtin_assume_aligned(a, 32);
    __builtin_assume_aligned(b, 32);
    __builtin_assume_aligned(c, 32);
    
    for (int i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}
```

---

## GPU Acceleration

### CUDA Kernel Optimization

**Coalesced memory access:**

```cuda
// Good: Coalesced access
__global__ void apply_gate_kernel_coalesced(
    cuDoubleComplex* L_new,
    const cuDoubleComplex* L,
    int rows, int cols
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows) {
        for (int col = 0; col < cols; ++col) {
            // Threads access consecutive rows: coalesced
            L_new[row * cols + col] = compute(L[row * cols + col]);
        }
    }
}
```

**Shared memory:**

```cuda
__global__ void matrix_mult_shared(
    cuDoubleComplex* C,
    const cuDoubleComplex* A,
    const cuDoubleComplex* B,
    int n
) {
    __shared__ cuDoubleComplex As[TILE_SIZE][TILE_SIZE];
    __shared__ cuDoubleComplex Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);
    
    for (int tile = 0; tile < (n + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        // Load tile into shared memory
        if (row < n && tile * TILE_SIZE + threadIdx.x < n) {
            As[threadIdx.y][threadIdx.x] = A[row * n + tile * TILE_SIZE + threadIdx.x];
        }
        if (col < n && tile * TILE_SIZE + threadIdx.y < n) {
            Bs[threadIdx.y][threadIdx.x] = B[(tile * TILE_SIZE + threadIdx.y) * n + col];
        }
        
        __syncthreads();
        
        // Compute partial product
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum = cuCadd(sum, cuCmul(As[threadIdx.y][k], Bs[k][threadIdx.x]));
        }
        
        __syncthreads();
    }
    
    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}
```

**Occupancy optimization:**

```bash
# Check occupancy
nvprof --metrics achieved_occupancy ./quantum_sim_gpu

# Optimize launch configuration
int threads_per_block = 256;  # Multiple of warp size (32)
int blocks = (total_work + threads_per_block - 1) / threads_per_block;
kernel<<<blocks, threads_per_block>>>(...)
```

---

## Compiler Optimizations

### GCC/Clang Optimization Flags

**Optimization levels:**

```bash
# No optimization (debug)
-O0

# Basic optimization
-O1

# Recommended for production
-O2

# Aggressive optimization (may increase binary size)
-O3

# Size optimization
-Os
```

**Additional flags:**

```bash
# Native architecture optimizations
-march=native

# Enable specific SIMD
-mavx2 -mfma

# Link-time optimization
-flto

# Profile-guided optimization (PGO)
# Step 1: Build with instrumentation
g++ -O2 -fprofile-generate -o quantum_sim main.cpp
# Step 2: Run with typical workload
./quantum_sim --qubits 15 --depth 30
# Step 3: Rebuild with profile data
g++ -O2 -fprofile-use -o quantum_sim main.cpp
```

**CMake configuration:**

```cmake
# Release build
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

# Enable LTO
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)

# Platform-specific optimizations
if(UNIX AND NOT APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma")
endif()
```

---

## Benchmarking

### Microbenchmarks

**Google Benchmark:**

```cpp
#include <benchmark/benchmark.h>
#include "simulator.h"

static void BM_ApplyHadamard(benchmark::State& state) {
    int n_qubits = state.range(0);
    LRETSimulator sim(n_qubits, 0.0);
    Gate h = hadamard_gate(0);
    
    for (auto _ : state) {
        sim.apply_gate(h);
        benchmark::DoNotOptimize(sim.get_rank());
    }
    
    state.SetComplexityN(std::pow(4, n_qubits));
}

BENCHMARK(BM_ApplyHadamard)->Range(5, 20)->Complexity();

BENCHMARK_MAIN();
```

**Run benchmarks:**

```bash
g++ -O2 -lbenchmark -lpthread benchmark.cpp -o benchmark
./benchmark --benchmark_out=results.json --benchmark_out_format=json
```

---

### System Benchmarks

**Full simulation benchmark:**

```python
import time
import numpy as np
from qlret import QuantumSimulator

def benchmark_circuit(n_qubits, depth, noise_level=0.0):
    """Benchmark a random circuit."""
    
    sim = QuantumSimulator(n_qubits=n_qubits, noise_level=noise_level)
    
    # Random circuit
    np.random.seed(42)
    gates = ['h', 'x', 'y', 'z', 'rx', 'ry', 'rz', 'cnot']
    
    start = time.time()
    
    for d in range(depth):
        gate = np.random.choice(gates)
        if gate == 'cnot':
            q1, q2 = np.random.choice(n_qubits, 2, replace=False)
            sim.cnot(q1, q2)
        elif gate in ['rx', 'ry', 'rz']:
            q = np.random.randint(n_qubits)
            theta = np.random.uniform(0, 2*np.pi)
            getattr(sim, gate)(q, theta)
        else:
            q = np.random.randint(n_qubits)
            getattr(sim, gate)(q)
    
    elapsed = time.time() - start
    
    return {
        'n_qubits': n_qubits,
        'depth': depth,
        'time': elapsed,
        'final_rank': sim.current_rank,
        'gates_per_second': depth / elapsed
    }

# Run benchmarks
for n in [10, 15, 20]:
    result = benchmark_circuit(n, 50)
    print(f"{n} qubits: {result['time']:.2f}s, rank={result['final_rank']}")
```

---

## Performance Checklist

Before deploying:

- [ ] Profiled hotspots with perf/gprof
- [ ] Optimized memory layout (column-major)
- [ ] Enabled compiler optimizations (-O3 -march=native)
- [ ] Implemented gate fusion
- [ ] Configured adaptive truncation
- [ ] Enabled OpenMP (if available)
- [ ] Tuned thread count (typically cores - 1)
- [ ] Validated SIMD vectorization
- [ ] Benchmarked against targets
- [ ] Checked memory leaks (valgrind)
- [ ] Optimized cache usage
- [ ] Tested on target hardware

---

## See Also

- **[Architecture Overview](00-overview.md)** - System design
- **[LRET Algorithm](03-lret-algorithm.md)** - Algorithm complexity
- **[Testing Framework](05-testing.md)** - Performance tests
- **[Benchmarking Guide](../../user-guide/07-benchmarking.md)** - User benchmarking

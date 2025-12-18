# QuantumLRET-Sim

**QuantumLRET-Sim** is a high-performance C++ library and simulator for Low-Rank Exact Tensor (LRET) evolution of noisy quantum circuits. It efficiently simulates density matrix dynamics using a low-rank Cholesky-like factorization (ρ ≈ L L†), avoiding full exponential scaling for up to ~20 qubits. Supports random circuit generation, parallel gate/noise application via OpenMP, eigenvalue-based truncation, and metrics like fidelity and trace distance.

Ideal for benchmarking quantum error models, circuit depths, and parallelism on multi-core systems.

## Features
- **Efficient Simulation**: Parallel batched application of 1/2-qubit gates (e.g., H, CNOT) and Kraus noise (e.g., depolarizing, amplitude damping).
- **Low-Rank Approximation**: Automatic truncation via Gram matrix eigendecomposition to bound rank growth.
- **Visualization**: ASCII art circuit diagrams with gates, wires, and noise indicators.
- **Metrics**: Quantum fidelity, Frobenius norm, and O(rank²) trace distance without full density matrix construction.
- **Tunable Parallelism**: Auto-tuned batch sizes for OpenMP; naive sequential mode for comparison.
- **Scalable**: Handles n=11 in <0.1s on standard hardware; extensible to distributed/GPU.

## Quick Start

### Prerequisites
- C++17 compiler (GCC/Clang/MSVC).
- Eigen3 (header-only; install via `apt install libeigen3-dev`, Homebrew, or vcpkg).
- OpenMP (enabled by default in most compilers).

### Build
Clone the repo and build with CMake:
```bash
git clone https://github.com/yourusername/quantum-lret-sim.git
cd quantum-lret-sim
mkdir build && cd build
cmake ..  # Add -DCMAKE_PREFIX_PATH=/path/to/eigen if needed
make -j$(nproc)  # Or cmake --build .
```

This produces `./quantum_sim` (main benchmark) and `./demo_batch` (batch heuristic tester).

### Run
Execute the benchmark for n=11 qubits, depth=13:
```bash
./quantum_sim
```
**Sample Output**:
```
--------------------------------------------------------------------------------------------------
number of qubits: 11
INFO: n=11 low-workload, batch_size=64
Generated sequence with total noise perc: 0.000523
batch size: 64
current time == 18:02:15
=====================Running LRET simulation for 11 qubits==========================
Simulation Time: 0.123 seconds
Final Rank: 13
...
Speed up with batch size 64 : 4.567
trace distance: 1.23e-05
```

For batch heuristic demo:
```bash
./demo_batch
```
**Sample Output**:
```
INFO: n=11 low-workload, batch_size=64
for 11 qubits number of batches are 64
INFO: n=12 low-workload, batch_size=64
for 12 qubits number of batches are 64
...
INFO: n=16 high-workload, batch_size=128
for 16 qubits number of batches are 128
```

## Usage

### Core API
Include headers and use the `qlret` namespace:
```cpp
#include "gates_and_noise.h"
#include "simulator.h"
#include "utils.h"
using namespace qlret;

// Generate random sequence
auto seq = generate_quantum_sequences(8, 20, true, 0.001);  // n=8, d=20, fixed noise

// Initial state (all-zero)
size_t dim = 1ULL << 8;
MatrixXcd L_init(dim, 1);
L_init(0, 0) = 1.0;

// Run parallel sim
size_t batch = auto_select_batch_size(8);
auto L_final = run_simulation_optimized(L_init, seq, 8, batch, true, true, 1e-4);

// Metrics
double dist = compare_L_matrices_trace(L_init, L_final);
std::cout << "Trace distance: " << dist << std::endl;

// Visualize
print_circuit_diagram(8, seq);
```

### Customization
- **Circuit Params**: Tweak n/d/noise in `generate_quantum_sequences` calls.
- **Sim Options**: Set `verbose=true` for step-by-step logs; `do_truncation=false` to disable rank control.
- **Batch Tuning**: Manual override in runners; use `auto_select_batch_size` for heuristics.
- **Extend**: Add custom gates to `one_qubit_gates` map; implement new noise in `noise_models`.

## Project Structure
- `include/`: Public headers (e.g., `simulator.h` for runners).
- `src/`: Implementations (e.g., `simulator.cpp` for OpenMP logic).
- `main.cpp`: Benchmark tester (n=11 default).
- `tests/demo_batch.cpp`: Standalone batch demo.
- `CMakeLists.txt`: Build config (add targets easily).

## Performance Notes
- **Benchmark**: On 8-core CPU, n=11/d=13: Parallel ~0.1s (4x speedup vs. naive); rank stays ~13 post-trunc.
- **Scaling**: Time ~ O(2^{n-m} * rank * d) per gate; trunc O(rank³). For n>15, increase batch/cap.
- **Profiling**: Use `perf` or add chrono prints.

## Contributing
1. Fork the repo.
2. Create a feature branch (`git checkout -b feat/new-noise`).
3. Commit changes (`git commit -m "feat: add phase damping noise"`).
4. Push (`git push origin feat/new-noise`).
5. Open a Pull Request.

Report issues for bugs/scaling ideas. Pull requests welcome for new gates, GPU support, or distributed sims.

## License
MIT License - see LICENSE file for details.

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

### Using Docker (Alternative)
A pre-built Docker image is available on Docker Hub for easy setup without manual compilation.

#### Pull and Run the Docker Image
```bash
cd lret-
mkdir build
docker run --rm -v $HOME/lret-:/app ajs911/lret777:latest
```

This mounts your local `lret-` directory to `/app` inside the container and runs the simulation. The `--rm` flag automatically removes the container after execution.

### Docker for Massive Workstations (TBs of RAM)

When running simulations on high-end workstations with terabytes of RAM, Docker's default resource limits can prevent you from utilizing all available memory. Here's how to unlock full host resources:

#### Recommended Docker Configuration

```bash
# Run with NO memory limits (uses all host RAM)
docker run --rm -it \
    --memory=0 \
    --memory-swap=-1 \
    --privileged \
    -v $(pwd):/app \
    ajs911/lret777:latest \
    ./quantum_sim -n 25 --fdm --allow-swap --timeout 2d -o results.csv

# With specific resource allocation
docker run --rm -it \
    --memory=512g \           # Limit to 512GB
    --memory-swap=1t \        # Allow 1TB total (RAM + swap)
    --cpus=128 \              # Use 128 CPU cores
    --privileged \
    -v $(pwd):/app \
    ajs911/lret777:latest \
    ./quantum_sim -n 24 --mode hybrid
```

#### Docker Flag Reference

| Flag | Description | Recommended Value |
|------|-------------|-------------------|
| `--memory=0` | No memory limit | Use for unlimited access |
| `--memory-swap=-1` | Unlimited swap | Essential for large simulations |
| `--privileged` | Full host access | Required for some memory operations |
| `--cpus=N` | CPU cores to use | Set to your core count |
| `--shm-size=64g` | Shared memory | Helps with large matrices |
| `--ulimit memlock=-1:-1` | Allow memory locking | Improves performance |

#### Alternative: Singularity/Apptainer for HPC

For HPC clusters and environments where Docker isn't ideal, use Singularity/Apptainer:

```bash
# Convert Docker image to Singularity
singularity pull quantum-lret.sif docker://ajs911/lret777:latest

# Run with full host resources (default behavior in Singularity)
singularity run --bind $(pwd):/app quantum-lret.sif \
    ./quantum_sim -n 26 --fdm --timeout 7d -o massive_run.csv
```

**Why Singularity?**
- No resource isolation by default (uses all host RAM/CPU)
- No root privileges needed
- Native HPC integration (Slurm, PBS, etc.)
- Same container image works everywhere

#### Memory Requirements by Qubit Count

| Qubits | FDM Memory | LRET Peak (est.) | Recommended RAM |
|--------|------------|------------------|-----------------|
| 20 | 17.6 GB | ~1-5 GB | 32 GB |
| 22 | 281.5 GB | ~10-50 GB | 512 GB |
| 24 | 4.5 TB | ~100-500 GB | 8 TB |
| 26 | 72 TB | ~1-5 TB | 128 TB |

*Note: LRET memory depends on rank growth. Noisy circuits maintain lower rank.*

#### Long-Running Simulation Tips

```bash
# Start a named container for monitoring
docker run -d --name lret-run \
    --memory=0 --memory-swap=-1 \
    -v $(pwd):/app \
    ajs911/lret777:latest \
    ./quantum_sim -n 24 --timeout 3d -o run.csv

# Monitor progress (CSV updates in real-time)
tail -f run.csv

# Check container resource usage
docker stats lret-run

# View logs
docker logs -f lret-run

# Graceful stop (triggers Ctrl+C handler)
docker stop --time=30 lret-run
```

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

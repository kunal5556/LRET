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

For detailed agent instructions and capabilities: @agent.md
For beginner-friendly guide: @AGENT_GUIDE.md

# Code Structure

Guide to the LRET repository organization and code layout.

## Repository Structure

```
LRET/
├── CMakeLists.txt              # Root CMake configuration
├── LICENSE                     # MIT License
├── README.md                   # Project overview
├── ROADMAP.md                  # Development roadmap
├── PROJECT_STATUS.md           # Current status
├── TESTING_BACKLOG.md          # Documented untested features
│
├── PHASE_*.md                  # Phase strategy documents
│   ├── PHASE_5_STRATEGY.md
│   ├── PHASE_6_STRATEGY.md
│   ├── PHASE_6B_INTEGRATION_TESTING_STRATEGY.md
│   ├── PHASE_6C_BENCHMARKING_STRATEGY.md
│   └── PHASE_6D_DOCUMENTATION_STRATEGY.md
│
├── include/                    # Public headers (C++)
│   ├── simulator.h             # Main simulator class
│   ├── gates_and_noise.h       # Gate and noise definitions
│   ├── types.h                 # Common types and constants
│   ├── utils.h                 # Utility functions
│   ├── parallel_modes.h        # Parallelization strategies
│   ├── circuit_optimizer.h     # Circuit optimization
│   ├── json_interface.h        # JSON I/O
│   ├── output_formatter.h      # Result formatting
│   ├── cli_parser.h            # Command-line parser
│   ├── resource_monitor.h      # Resource tracking
│   ├── benchmark_types.h       # Benchmarking data structures
│   ├── benchmark_runner.h      # Benchmark executor
│   ├── noise_import.h          # IBM noise import
│   ├── gate_fusion.h           # Gate fusion optimization
│   ├── simd_kernels.h          # SIMD optimizations
│   ├── fdm_simulator.h         # Full-density-matrix fallback
│   ├── advanced_noise.h        # Advanced noise models
│   ├── gpu_simulator.h         # GPU acceleration (CUDA)
│   ├── mpi_parallel.h          # MPI distributed computing
│   ├── progressive_csv.h       # Streaming CSV output
│   └── structured_csv.h        # Structured CSV format
│
├── src/                        # Implementation files (C++)
│   ├── simulator.cpp           # Simulator implementation
│   ├── gates_and_noise.cpp     # Gate operations
│   ├── parallel_modes.cpp      # Parallel execution
│   ├── circuit_optimizer.cpp   # Circuit optimization
│   ├── json_interface.cpp      # JSON parsing/writing
│   ├── output_formatter.cpp    # Output formatting
│   ├── cli_parser.cpp          # CLI argument parsing
│   ├── resource_monitor.cpp    # Resource monitoring
│   ├── benchmark_types.cpp     # Benchmark structures
│   ├── benchmark_runner.cpp    # Benchmark execution
│   ├── noise_import.cpp        # Noise model import
│   ├── gate_fusion.cpp         # Gate fusion logic
│   ├── simd_kernels.cpp        # SIMD implementations
│   ├── fdm_simulator.cpp       # FDM simulation
│   ├── python_bindings.cpp     # pybind11 bindings
│   ├── gpu_simulator.cu        # CUDA kernels
│   └── mpi_parallel.cpp        # MPI implementation
│
├── main.cpp                    # CLI entry point
├── test_*.cpp                  # Standalone test files
│   ├── test_simple.cpp         # Basic functionality
│   ├── test_fidelity.cpp       # Fidelity validation
│   ├── test_minimal.cpp        # Minimal example
│   └── test_noise_import.cpp   # Noise import test
│
├── tests/                      # C++ unit tests (GTest)
│   ├── CMakeLists.txt
│   ├── test_simulator.cpp      # Simulator tests
│   ├── test_gates.cpp          # Gate operation tests
│   ├── test_noise.cpp          # Noise model tests
│   ├── test_parallel.cpp       # Parallelization tests
│   ├── test_truncation.cpp     # Rank truncation tests
│   ├── test_circuit_opt.cpp    # Circuit optimization tests
│   └── test_utils.cpp          # Utility function tests
│
├── python/                     # Python package
│   ├── setup.py                # Python package setup
│   ├── pyproject.toml          # Modern Python config
│   ├── MANIFEST.in             # Package manifest
│   ├── README.md               # Python package docs
│   ├── qlret/                  # Python module
│   │   ├── __init__.py         # Package init
│   │   ├── simulator.py        # Python wrapper
│   │   ├── pennylane_device.py # PennyLane plugin
│   │   ├── noise.py            # Noise utilities
│   │   ├── calibration.py      # Noise calibration
│   │   └── utils.py            # Python utilities
│   └── tests/                  # Python tests (pytest)
│       ├── pytest.ini          # pytest configuration
│       ├── conftest.py         # Test fixtures
│       ├── test_simulator.py   # Simulator tests
│       ├── test_gates.py       # Gate tests
│       ├── test_noise.py       # Noise tests
│       ├── test_pennylane.py   # PennyLane tests
│       └── integration/        # Integration tests
│           ├── conftest.py
│           ├── test_cli_regression.py
│           ├── test_docker_runtime.py
│           ├── test_json_execution.py
│           └── test_pennylane_device.py
│
├── scripts/                    # Python scripts and tools
│   ├── benchmark_suite.py      # Benchmark orchestrator
│   ├── benchmark_analysis.py   # Statistical analysis
│   ├── benchmark_visualize.py  # Plot generation
│   ├── download_ibm_noise.py   # IBM device noise downloader
│   ├── calibrate_noise_model.py# Noise calibration
│   ├── compare_fidelities.py   # Fidelity comparison
│   ├── fit_depolarizing.py     # Fit depolarizing noise
│   ├── fit_t1_t2.py            # Fit relaxation times
│   ├── fit_correlated_errors.py# Fit correlated errors
│   ├── fit_time_scaling.py     # Fit time scaling
│   ├── detect_memory_effects.py# Memory leak detection
│   ├── generate_calibration_data.py # Generate calibration data
│   ├── test_calibration.py     # Test calibration
│   ├── example_commands_*.txt  # Example commands
│   └── sample_noise_with_leakage.json # Example noise
│
├── samples/                    # Example circuits and configs
│   └── json/                   # JSON circuit examples
│       ├── bell_state.json
│       ├── ghz_state.json
│       ├── vqe_ansatz.json
│       └── qaoa_circuit.json
│
├── docs/                       # Documentation
│   ├── user-guide/             # User documentation
│   │   ├── 00-introduction.md
│   │   ├── 01-installation.md
│   │   ├── 02-quick-start.md
│   │   ├── 03-cli-reference.md
│   │   ├── 04-python-interface.md
│   │   ├── 05-pennylane-integration.md
│   │   ├── 06-noise-models.md
│   │   ├── 07-benchmarking.md
│   │   └── 08-troubleshooting.md
│   ├── developer-guide/        # Developer documentation
│   │   ├── 00-overview.md
│   │   ├── 01-building-from-source.md
│   │   ├── 02-code-structure.md
│   │   ├── 03-lret-algorithm.md
│   │   ├── 04-extending-simulator.md
│   │   ├── 05-testing.md
│   │   ├── 06-performance.md
│   │   └── 07-contributing.md
│   ├── api-reference/          # API documentation
│   │   ├── cpp/
│   │   ├── python/
│   │   └── cli/
│   ├── examples/               # Code examples
│   │   ├── python/
│   │   ├── cpp/
│   │   └── jupyter/
│   ├── deployment/             # Deployment guides
│   │   ├── docker-guide.md
│   │   ├── cloud-deployment.md
│   │   └── hpc-deployment.md
│   └── performance/            # Performance docs
│       ├── scaling-analysis.md
│       └── benchmarks/
│
├── build/                      # Build artifacts (gitignored)
│   ├── CMakeCache.txt
│   ├── quantum_sim             # CLI binary
│   ├── test_*                  # Test binaries
│   └── ...
│
├── docker/                     # Docker configurations
│   ├── Dockerfile              # Standard image
│   ├── Dockerfile.dev          # Development image
│   ├── Dockerfile.gpu          # GPU-enabled image
│   └── docker-compose.yml      # Multi-container setup
│
├── .github/                    # GitHub configuration
│   └── workflows/              # CI/CD workflows
│       ├── test.yml
│       ├── build.yml
│       ├── docker.yml
│       ├── docs.yml
│       └── benchmark.yml
│
└── .gitignore                  # Git ignore patterns
```

---

## Core Components

### Simulator Core

**Files:**
- `include/simulator.h` - Main simulator class declaration
- `src/simulator.cpp` - Simulator implementation

**Key classes:**
```cpp
class LRETSimulator {
public:
    // Construction
    LRETSimulator(int n_qubits, double noise_level = 0.0);
    
    // Gate application
    void apply_gate(const Gate& gate);
    void apply_noise(const NoiseChannel& noise);
    
    // State management
    void reset();
    MatrixXcd get_density_matrix() const;
    VectorXcd get_state_vector() const;
    
    // Measurement
    std::map<std::string, int> measure_all(int shots = 1);
    
    // Properties
    int get_rank() const;
    double get_fidelity() const;
    
private:
    MatrixXcd L_;  // Low-rank factor
    int n_qubits_;
    int current_rank_;
    double truncation_threshold_;
    
    // Internal methods
    void truncate_rank();
    MatrixXcd apply_gate_choi(const MatrixXcd& L, const MatrixXcd& choi);
};
```

**Dependencies:**
- Eigen3 for linear algebra
- OpenMP for parallelization
- `gates_and_noise.h` for gate definitions
- `parallel_modes.h` for execution strategies

---

### Gate Library

**Files:**
- `include/gates_and_noise.h` - Gate/noise declarations
- `src/gates_and_noise.cpp` - Gate implementations

**Data structures:**
```cpp
struct Gate {
    std::string name;
    GateType type;  // SINGLE_QUBIT, TWO_QUBIT, THREE_QUBIT
    std::vector<int> targets;
    std::vector<double> params;  // For parametric gates
    MatrixXcd unitary;
    MatrixXcd choi_matrix;
};

struct NoiseChannel {
    std::string type;  // "depolarizing", "amplitude_damping", etc.
    double parameter;
    std::vector<int> affected_qubits;
    std::vector<MatrixXcd> kraus_operators;
    MatrixXcd choi_matrix;
};
```

**Gate registry:**
```cpp
extern std::map<std::string, Gate> single_qubit_gates;
extern std::map<std::string, Gate> two_qubit_gates;
extern std::map<std::string, Gate> three_qubit_gates;

// Access gates by name
Gate hadamard = single_qubit_gates["H"];
Gate cnot = two_qubit_gates["CNOT"];
```

---

### Parallelization

**Files:**
- `include/parallel_modes.h` - Parallel strategy declarations
- `src/parallel_modes.cpp` - Parallel implementations

**Modes:**
```cpp
enum ParallelMode {
    SEQUENTIAL,      // No parallelization
    ROW_PARALLEL,    // Parallelize row updates
    COLUMN_PARALLEL, // Parallelize column updates
    HYBRID           // Auto-select per gate
};

// Mode selection
ParallelMode select_parallel_mode(int rank, int n_qubits);

// Gate application with parallelization
MatrixXcd apply_gate_parallel(
    const MatrixXcd& L,
    const MatrixXcd& gate_choi,
    int target_qubit,
    int n_qubits,
    ParallelMode mode
);
```

---

### I/O System

**JSON Interface:**
- `include/json_interface.h` - JSON parsing/writing
- `src/json_interface.cpp` - Implementation

**Output Formatting:**
- `include/output_formatter.h` - Result formatting
- `src/output_formatter.cpp` - CSV, JSON, HDF5 output

**CSV Streaming:**
- `include/progressive_csv.h` - Real-time CSV updates
- `include/structured_csv.h` - Structured CSV format

---

### CLI Tool

**Files:**
- `main.cpp` - Entry point
- `include/cli_parser.h` - Argument parsing
- `src/cli_parser.cpp` - Implementation

**Flow:**
```
main.cpp
  → parse_arguments()
  → load_circuit()
  → create_simulator()
  → run_simulation()
  → format_output()
  → write_results()
```

---

### Python Bindings

**Files:**
- `src/python_bindings.cpp` - pybind11 bindings
- `python/qlret/` - Python wrapper module

**Exposed classes:**
```cpp
PYBIND11_MODULE(qlret_core, m) {
    py::class_<LRETSimulator>(m, "LRETSimulator")
        .def(py::init<int, double>())
        .def("apply_gate", &LRETSimulator::apply_gate)
        .def("measure_all", &LRETSimulator::measure_all)
        .def("get_rank", &LRETSimulator::get_rank)
        // ... more methods ...
        ;
    
    py::class_<Gate>(m, "Gate")
        .def(py::init<>())
        .def_readwrite("name", &Gate::name)
        .def_readwrite("targets", &Gate::targets)
        // ... more fields ...
        ;
}
```

**Python wrapper:**
```python
# python/qlret/simulator.py
from qlret_core import LRETSimulator as _LRETSimulator

class QuantumSimulator:
    def __init__(self, n_qubits, noise_level=0.0):
        self._sim = _LRETSimulator(n_qubits, noise_level)
    
    def h(self, qubit):
        """Apply Hadamard gate."""
        self._sim.apply_gate(Gate("H", [qubit]))
    
    # ... more methods ...
```

---

### PennyLane Device

**Files:**
- `python/qlret/pennylane_device.py` - Device plugin

**Device implementation:**
```python
from pennylane import QubitDevice
from qlret import QuantumSimulator

class QLRETDevice(QubitDevice):
    name = "LRET quantum simulator"
    short_name = "qlret.simulator"
    pennylane_requires = ">=0.30.0"
    version = "1.0.0"
    author = "LRET Team"
    
    operations = {
        "PauliX", "PauliY", "PauliZ",
        "Hadamard", "CNOT", "RX", "RY", "RZ",
        # ... more ops ...
    }
    
    observables = {"PauliX", "PauliY", "PauliZ", "Identity", "Hermitian"}
    
    def __init__(self, wires, noise_level=0.0, **kwargs):
        super().__init__(wires=wires, **kwargs)
        self.sim = QuantumSimulator(n_qubits=len(wires), noise_level=noise_level)
    
    def apply(self, operations, **kwargs):
        for op in operations:
            # Convert PennyLane op to LRET gate
            self._apply_operation(op)
    
    def expval(self, observable):
        # Compute expectation value
        return self.sim.get_expectation(observable)
```

---

## Testing Infrastructure

### C++ Unit Tests (GTest)

**Location:** `tests/`

**Test structure:**
```cpp
#include <gtest/gtest.h>
#include "simulator.h"

class SimulatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        sim = new LRETSimulator(4, 0.01);
    }
    
    void TearDown() override {
        delete sim;
    }
    
    LRETSimulator* sim;
};

TEST_F(SimulatorTest, HadamardGate) {
    sim->apply_gate(hadamard_gate(0));
    EXPECT_EQ(sim->get_rank(), 2);
}
```

**Run tests:**
```bash
cd build
ctest --output-on-failure
```

---

### Python Tests (pytest)

**Location:** `python/tests/`

**Test structure:**
```python
import pytest
from qlret import QuantumSimulator

@pytest.fixture
def sim():
    return QuantumSimulator(n_qubits=4, noise_level=0.01)

def test_hadamard(sim):
    sim.h(0)
    assert sim.current_rank == 2

def test_bell_state(sim):
    sim.h(0)
    sim.cnot(0, 1)
    results = sim.measure_all(shots=1000)
    assert len(results) == 2  # |00⟩ and |11⟩
```

**Run tests:**
```bash
cd python/tests
pytest -v
```

---

### Integration Tests

**Location:** `python/tests/integration/`

**Test categories:**
- CLI regression tests
- Docker runtime tests
- JSON I/O tests
- PennyLane device tests

---

## Benchmarking Suite

**Files:**
- `scripts/benchmark_suite.py` - Master orchestrator
- `scripts/benchmark_analysis.py` - Statistical analysis
- `scripts/benchmark_visualize.py` - Visualization

**Categories:**
1. **Scaling:** Qubits and depth scaling
2. **Parallel:** Multi-threading performance
3. **Accuracy:** Fidelity validation
4. **Depth:** Deep circuit performance
5. **Memory:** Memory usage tracking

---

## Documentation

### User Documentation
- Installation, quick start, tutorials
- CLI and API references
- Troubleshooting guides

### Developer Documentation
- Architecture, algorithm, code structure
- Build instructions, testing
- Performance optimization

### API Reference
- C++ API documentation
- Python API documentation
- CLI tool reference

---

## Build System (CMake)

**Root CMakeLists.txt:**
```cmake
cmake_minimum_required(VERSION 3.18)
project(LRET VERSION 1.0.0 LANGUAGES CXX)

# Options
option(BUILD_PYTHON "Build Python bindings" ON)
option(BUILD_TESTS "Build C++ tests" ON)
option(ENABLE_OPENMP "Enable OpenMP" ON)
option(ENABLE_GPU "Enable CUDA support" OFF)

# Find dependencies
find_package(Eigen3 3.3 REQUIRED)
find_package(OpenMP)
find_package(Python3 COMPONENTS Interpreter Development)
find_package(pybind11)

# Core library
add_library(qlret_core STATIC
    src/simulator.cpp
    src/gates_and_noise.cpp
    src/parallel_modes.cpp
    # ... more sources ...
)

target_link_libraries(qlret_core PUBLIC Eigen3::Eigen)
if(OpenMP_CXX_FOUND)
    target_link_libraries(qlret_core PUBLIC OpenMP::OpenMP_CXX)
endif()

# CLI tool
add_executable(quantum_sim main.cpp)
target_link_libraries(quantum_sim PRIVATE qlret_core)

# Python bindings
if(BUILD_PYTHON AND pybind11_FOUND)
    pybind11_add_module(qlret_core_py src/python_bindings.cpp)
    target_link_libraries(qlret_core_py PRIVATE qlret_core)
endif()

# Tests
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
```

---

## Coding Conventions

### C++ Style

**Naming:**
- Classes: `PascalCase` (e.g., `LRETSimulator`)
- Functions: `snake_case` (e.g., `apply_gate()`)
- Variables: `snake_case_` for private members (e.g., `L_`, `n_qubits_`)
- Constants: `UPPER_CASE` (e.g., `MAX_QUBITS`)

**Headers:**
```cpp
#pragma once

#include <vector>
#include <complex>
#include <Eigen/Dense>

namespace qlret {

class MyClass {
public:
    MyClass();
    void my_method();
    
private:
    int my_member_;
};

}  // namespace qlret
```

**Implementation:**
```cpp
#include "my_class.h"

namespace qlret {

MyClass::MyClass() : my_member_(0) {
    // Constructor
}

void MyClass::my_method() {
    // Implementation
}

}  // namespace qlret
```

### Python Style

**Follow PEP 8:**
- Classes: `PascalCase`
- Functions: `snake_case`
- Constants: `UPPER_CASE`
- Private: `_leading_underscore`

**Docstrings:**
```python
def my_function(arg1, arg2):
    """Short description.
    
    Longer description with details.
    
    Args:
        arg1 (int): Description of arg1.
        arg2 (str): Description of arg2.
    
    Returns:
        bool: Description of return value.
    
    Example:
        >>> result = my_function(5, "test")
        >>> print(result)
        True
    """
    return True
```

---

## Version Control

### Git Workflow

**Branches:**
- `main` - Stable releases
- `develop` - Development branch
- `feature/*` - Feature branches
- `hotfix/*` - Bug fix branches

**Commit messages:**
```
Type: Short summary (50 chars)

- Detailed bullet points
- Explain what and why, not how
- Reference issues: Fixes #123

Type: feat, fix, docs, test, refactor, perf, chore
```

---

## See Also

- **[Architecture Overview](00-overview.md)** - System design
- **[Building from Source](01-building-from-source.md)** - Compilation
- **[LRET Algorithm](03-lret-algorithm.md)** - Algorithm details
- **[Extending the Simulator](04-extending-simulator.md)** - Adding features
- **[Testing Framework](05-testing.md)** - Testing guide

# Testing Framework

Comprehensive guide to testing LRET components, writing new tests, and CI integration.

## Table of Contents

1. [Testing Philosophy](#testing-philosophy)
2. [C++ Unit Tests](#c-unit-tests)
3. [Python Unit Tests](#python-unit-tests)
4. [Integration Tests](#integration-tests)
5. [Performance Tests](#performance-tests)
6. [Writing New Tests](#writing-new-tests)
7. [Test Coverage](#test-coverage)
8. [Continuous Integration](#continuous-integration)
9. [Debugging Tests](#debugging-tests)

---

## Testing Philosophy

### Test Categories

LRET uses a multi-tiered testing strategy:

1. **Unit Tests:** Test individual components in isolation
   - C++ classes and functions (Google Test)
   - Python modules and functions (pytest)
   
2. **Integration Tests:** Test component interactions
   - CLI tool execution
   - Python-C++ bindings
   - PennyLane device integration
   
3. **System Tests:** End-to-end workflows
   - Full circuit simulations
   - Benchmarking pipelines
   - Docker container execution
   
4. **Performance Tests:** Validate scalability
   - Timing benchmarks
   - Memory profiling
   - Parallel scaling

### Test-Driven Development

When adding new features:

1. **Write test first** - Define expected behavior
2. **Implement feature** - Make test pass
3. **Refactor** - Improve code while keeping test passing
4. **Document** - Add docstrings and comments

### Testing Best Practices

- **Fast:** Unit tests should run in <1s each
- **Isolated:** Tests should not depend on each other
- **Repeatable:** Same input → same output
- **Self-checking:** Use assertions, not manual verification
- **Thorough:** Test edge cases, error paths, and typical usage

---

## C++ Unit Tests

### Google Test Framework

LRET uses **Google Test (gtest)** for C++ testing.

**Installation:**
```bash
# Ubuntu/Debian
sudo apt install libgtest-dev

# macOS
brew install googletest
```

### Test Structure

**Location:** `tests/`

**Example test file:** `tests/test_simulator.cpp`

```cpp
#include <gtest/gtest.h>
#include "simulator.h"
#include "gates_and_noise.h"

// Test fixture for simulator tests
class SimulatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Runs before each test
        sim = new LRETSimulator(4, 0.0);  // 4 qubits, no noise
    }
    
    void TearDown() override {
        // Runs after each test
        delete sim;
    }
    
    LRETSimulator* sim;
};

// Test case: Initial state
TEST_F(SimulatorTest, InitialState) {
    EXPECT_EQ(sim->get_rank(), 1);
    EXPECT_EQ(sim->get_num_qubits(), 4);
    
    VectorXcd state = sim->get_state_vector();
    EXPECT_DOUBLE_EQ(std::abs(state(0)), 1.0);  // |0000⟩
    for (int i = 1; i < state.size(); ++i) {
        EXPECT_DOUBLE_EQ(std::abs(state(i)), 0.0);
    }
}

// Test case: Hadamard gate
TEST_F(SimulatorTest, HadamardGate) {
    Gate h = hadamard_gate(0);
    sim->apply_gate(h);
    
    EXPECT_EQ(sim->get_rank(), 1);  // Pure state
    
    VectorXcd state = sim->get_state_vector();
    double expected = 1.0 / std::sqrt(2.0);
    EXPECT_NEAR(std::abs(state(0)), expected, 1e-10);  // |0000⟩
    EXPECT_NEAR(std::abs(state(8)), expected, 1e-10);  // |1000⟩
}

// Test case: CNOT gate (entanglement)
TEST_F(SimulatorTest, CNOTGate) {
    sim->apply_gate(hadamard_gate(0));
    sim->apply_gate(cnot_gate(0, 1));
    
    EXPECT_EQ(sim->get_rank(), 1);  // Still pure
    
    VectorXcd state = sim->get_state_vector();
    double expected = 1.0 / std::sqrt(2.0);
    EXPECT_NEAR(std::abs(state(0)), expected, 1e-10);   // |0000⟩
    EXPECT_NEAR(std::abs(state(12)), expected, 1e-10);  // |1100⟩
}

// Test case: Depolarizing noise
TEST_F(SimulatorTest, DepolarizingNoise) {
    NoiseChannel noise = depolarizing_noise(0, 0.1);
    sim->apply_noise(noise);
    
    EXPECT_GT(sim->get_rank(), 1);  // Mixed state
    
    double fidelity = sim->get_fidelity();
    EXPECT_LT(fidelity, 1.0);
    EXPECT_GT(fidelity, 0.9);  // Fidelity ≈ 1 - p
}

// Test case: Measurement
TEST_F(SimulatorTest, Measurement) {
    sim->apply_gate(hadamard_gate(0));
    
    std::map<std::string, int> results = sim->measure_all(1000);
    
    // Should get roughly 50-50 split between |0000⟩ and |1000⟩
    EXPECT_GT(results["0000"], 400);
    EXPECT_LT(results["0000"], 600);
    EXPECT_GT(results["1000"], 400);
    EXPECT_LT(results["1000"], 600);
}
```

### Running C++ Tests

**Build tests:**
```bash
mkdir -p build && cd build
cmake .. -DBUILD_TESTS=ON
make -j$(nproc)
```

**Run all tests:**
```bash
ctest --output-on-failure
```

**Run specific test:**
```bash
./test_simulator
```

**Run with filter:**
```bash
./test_simulator --gtest_filter=SimulatorTest.HadamardGate
```

**Verbose output:**
```bash
./test_simulator --gtest_verbose
```

---

### Testing Gates

**Location:** `tests/test_gates.cpp`

```cpp
TEST(GateTest, PauliMatrices) {
    MatrixXcd X = pauli_x();
    MatrixXcd Y = pauli_y();
    MatrixXcd Z = pauli_z();
    MatrixXcd I = MatrixXcd::Identity(2, 2);
    
    // Test Pauli identities
    EXPECT_TRUE((X * X).isApprox(I, 1e-10));
    EXPECT_TRUE((Y * Y).isApprox(I, 1e-10));
    EXPECT_TRUE((Z * Z).isApprox(I, 1e-10));
    
    // Test commutation: [X, Y] = 2iZ
    MatrixXcd commutator = X * Y - Y * X;
    MatrixXcd expected = 2.0 * std::complex<double>(0, 1) * Z;
    EXPECT_TRUE(commutator.isApprox(expected, 1e-10));
}

TEST(GateTest, UnitaryProperty) {
    std::vector<Gate> gates = {
        hadamard_gate(0),
        pauli_x_gate(0),
        rotation_x_gate(0, M_PI / 4),
        cnot_gate(0, 1),
        toffoli_gate(0, 1, 2)
    };
    
    for (const auto& gate : gates) {
        MatrixXcd U = gate.unitary;
        int dim = U.rows();
        MatrixXcd I = MatrixXcd::Identity(dim, dim);
        
        // Test U†U = I (unitarity)
        EXPECT_TRUE((U.adjoint() * U).isApprox(I, 1e-10))
            << "Gate " << gate.name << " is not unitary";
    }
}
```

---

### Testing Noise Channels

**Location:** `tests/test_noise.cpp`

```cpp
TEST(NoiseTest, CPTPConditions) {
    std::vector<NoiseChannel> channels = {
        depolarizing_noise(0, 0.1),
        amplitude_damping(0, 0.2),
        phase_damping(0, 0.15)
    };
    
    for (const auto& noise : channels) {
        // Test trace preservation: Σ_k K_k† K_k = I
        MatrixXcd sum = MatrixXcd::Zero(2, 2);
        for (const auto& K : noise.kraus_operators) {
            sum += K.adjoint() * K;
        }
        MatrixXcd I = MatrixXcd::Identity(2, 2);
        EXPECT_TRUE(sum.isApprox(I, 1e-10))
            << "Noise channel " << noise.type << " violates trace preservation";
        
        // Test complete positivity (Choi matrix is positive semi-definite)
        Eigen::SelfAdjointEigenSolver<MatrixXcd> es(noise.choi_matrix);
        VectorXd eigenvalues = es.eigenvalues();
        for (int i = 0; i < eigenvalues.size(); ++i) {
            EXPECT_GE(eigenvalues(i), -1e-10)
                << "Choi matrix has negative eigenvalue: " << eigenvalues(i);
        }
    }
}
```

---

### Testing Parallel Modes

**Location:** `tests/test_parallel.cpp`

```cpp
TEST(ParallelTest, RowParallelCorrectness) {
    int n_qubits = 4;
    int rank = 10;
    
    LRETSimulator sim_seq(n_qubits, 0.0);
    LRETSimulator sim_par(n_qubits, 0.0);
    
    sim_seq.set_parallel_mode(ParallelMode::SEQUENTIAL);
    sim_par.set_parallel_mode(ParallelMode::ROW_PARALLEL);
    
    // Apply same circuit
    std::vector<Gate> circuit = {
        hadamard_gate(0),
        cnot_gate(0, 1),
        rotation_x_gate(2, M_PI / 4),
        cnot_gate(1, 2)
    };
    
    for (const auto& gate : circuit) {
        sim_seq.apply_gate(gate);
        sim_par.apply_gate(gate);
    }
    
    // Check results match
    VectorXcd state_seq = sim_seq.get_state_vector();
    VectorXcd state_par = sim_par.get_state_vector();
    
    EXPECT_TRUE(state_seq.isApprox(state_par, 1e-10))
        << "Parallel and sequential results differ";
}
```

---

## Python Unit Tests

### pytest Framework

LRET uses **pytest** for Python testing.

**Installation:**
```bash
pip install pytest pytest-cov pytest-xdist
```

### Test Structure

**Location:** `python/tests/`

**Example test file:** `python/tests/test_simulator.py`

```python
import pytest
import numpy as np
from qlret import QuantumSimulator

@pytest.fixture
def sim():
    """Fixture providing a 4-qubit simulator."""
    return QuantumSimulator(n_qubits=4, noise_level=0.0)

def test_initial_state(sim):
    """Test initial state is |0000⟩."""
    assert sim.current_rank == 1
    assert sim.n_qubits == 4
    
    state = sim.get_state_vector()
    assert np.abs(state[0]) == pytest.approx(1.0)
    assert np.allclose(np.abs(state[1:]), 0.0)

def test_hadamard(sim):
    """Test Hadamard gate creates superposition."""
    sim.h(0)
    
    assert sim.current_rank == 1
    
    state = sim.get_state_vector()
    expected = 1 / np.sqrt(2)
    assert np.abs(state[0]) == pytest.approx(expected)
    assert np.abs(state[8]) == pytest.approx(expected)

def test_cnot_entanglement(sim):
    """Test CNOT creates entangled Bell state."""
    sim.h(0)
    sim.cnot(0, 1)
    
    state = sim.get_state_vector()
    expected = 1 / np.sqrt(2)
    
    # Bell state: (|00⟩ + |11⟩) / √2
    # Indices: |00**⟩ = 0, |11**⟩ = 12 (bits 0,1 are 1)
    assert np.abs(state[0]) == pytest.approx(expected)
    assert np.abs(state[12]) == pytest.approx(expected)

def test_measurement_statistics(sim):
    """Test measurement statistics match theory."""
    sim.h(0)
    
    results = sim.measure_all(shots=10000)
    
    # Should get ~50-50 split
    assert 4500 < results.get("0000", 0) < 5500
    assert 4500 < results.get("1000", 0) < 5500
    
    # Total should equal shots
    assert sum(results.values()) == 10000

def test_noise_increases_rank(sim):
    """Test depolarizing noise increases rank."""
    initial_rank = sim.current_rank
    
    sim.apply_depolarizing_noise(0, 0.1)
    
    assert sim.current_rank > initial_rank

def test_parametric_gate():
    """Test parametric rotation gate."""
    sim = QuantumSimulator(n_qubits=1)
    
    # RX(π) should be equivalent to X gate
    sim.rx(0, np.pi)
    state = sim.get_state_vector()
    
    # |0⟩ → -i|1⟩ under RX(π)
    assert np.abs(state[0]) == pytest.approx(0.0)
    assert np.abs(state[1]) == pytest.approx(1.0)
```

### Running Python Tests

**Run all tests:**
```bash
cd python/tests
pytest -v
```

**Run specific test file:**
```bash
pytest test_simulator.py -v
```

**Run specific test function:**
```bash
pytest test_simulator.py::test_hadamard -v
```

**Run with coverage:**
```bash
pytest --cov=qlret --cov-report=html
```

**Parallel execution:**
```bash
pytest -n auto  # Use all CPU cores
```

---

### Testing Python Bindings

**Location:** `python/tests/test_bindings.py`

```python
def test_cpp_python_consistency():
    """Test C++ and Python interfaces give same results."""
    import qlret_core
    from qlret import QuantumSimulator
    
    # C++ interface
    cpp_sim = qlret_core.LRETSimulator(4, 0.0)
    cpp_sim.apply_gate(qlret_core.hadamard_gate(0))
    cpp_state = cpp_sim.get_state_vector()
    
    # Python interface
    py_sim = QuantumSimulator(n_qubits=4)
    py_sim.h(0)
    py_state = py_sim.get_state_vector()
    
    assert np.allclose(cpp_state, py_state)
```

---

### Testing PennyLane Integration

**Location:** `python/tests/test_pennylane.py`

```python
import pennylane as qml
from qlret import QLRETDevice

def test_pennylane_device_creation():
    """Test PennyLane device creation."""
    dev = qml.device("qlret.simulator", wires=2)
    assert dev.num_wires == 2

def test_pennylane_circuit():
    """Test simple PennyLane circuit."""
    dev = qml.device("qlret.simulator", wires=2)
    
    @qml.qnode(dev)
    def circuit():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.probs(wires=[0, 1])
    
    probs = circuit()
    
    # Bell state: 50% |00⟩, 50% |11⟩
    assert probs[0] == pytest.approx(0.5, abs=0.01)
    assert probs[1] == pytest.approx(0.0, abs=0.01)
    assert probs[2] == pytest.approx(0.0, abs=0.01)
    assert probs[3] == pytest.approx(0.5, abs=0.01)

def test_pennylane_gradients():
    """Test PennyLane gradient computation."""
    dev = qml.device("qlret.simulator", wires=1)
    
    @qml.qnode(dev, diff_method="parameter-shift")
    def circuit(theta):
        qml.RX(theta, wires=0)
        return qml.expval(qml.PauliZ(0))
    
    theta = 0.5
    grad = qml.grad(circuit)(theta)
    
    # Analytical gradient: d/dθ <Z> = -sin(θ)
    expected = -np.sin(theta)
    assert grad == pytest.approx(expected, abs=0.01)
```

---

## Integration Tests

### CLI Integration Tests

**Location:** `python/tests/integration/test_cli_regression.py`

```python
import subprocess
import json
import pytest

def test_cli_help():
    """Test CLI help command."""
    result = subprocess.run(
        ["quantum_sim", "--help"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert "usage:" in result.stdout.lower()

def test_cli_json_input(tmp_path):
    """Test CLI with JSON input."""
    circuit_json = {
        "qubits": 4,
        "gates": [
            {"type": "H", "targets": [0]},
            {"type": "CNOT", "targets": [0, 1]}
        ],
        "shots": 1000
    }
    
    # Write JSON file
    json_file = tmp_path / "circuit.json"
    with open(json_file, 'w') as f:
        json.dump(circuit_json, f)
    
    # Run CLI
    result = subprocess.run(
        ["quantum_sim", "--input", str(json_file)],
        capture_output=True,
        text=True
    )
    
    assert result.returncode == 0
    assert "0000" in result.stdout
    assert "1100" in result.stdout

def test_cli_noise_parameter():
    """Test CLI with noise parameter."""
    result = subprocess.run(
        ["quantum_sim", "--qubits", "4", "--depth", "10", "--noise", "0.01"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
```

---

### Docker Integration Tests

**Location:** `python/tests/integration/test_docker_runtime.py`

```python
import docker
import pytest

@pytest.fixture(scope="module")
def docker_client():
    """Docker client fixture."""
    return docker.from_env()

def test_docker_image_exists(docker_client):
    """Test LRET Docker image exists."""
    try:
        image = docker_client.images.get("lret:latest")
        assert image is not None
    except docker.errors.ImageNotFound:
        pytest.skip("Docker image not built")

def test_docker_container_run(docker_client):
    """Test running LRET in Docker."""
    container = docker_client.containers.run(
        "lret:latest",
        "quantum_sim --qubits 4 --depth 10",
        detach=True,
        remove=True
    )
    
    # Wait for completion
    result = container.wait()
    assert result['StatusCode'] == 0
    
    logs = container.logs().decode('utf-8')
    assert "Simulation complete" in logs
```

---

## Performance Tests

### Benchmarking Tests

**Location:** `python/tests/test_performance.py`

```python
import time
import pytest
from qlret import QuantumSimulator

@pytest.mark.slow
def test_scaling_performance():
    """Test performance scales as expected."""
    results = []
    
    for n_qubits in [5, 10, 15]:
        sim = QuantumSimulator(n_qubits=n_qubits)
        
        start = time.time()
        for _ in range(10):
            sim.h(0)
            sim.cnot(0, 1)
        elapsed = time.time() - start
        
        results.append((n_qubits, elapsed))
    
    # Check scaling is reasonable (not exponential)
    assert results[1][1] < results[0][1] * 100
    assert results[2][1] < results[1][1] * 100

@pytest.mark.slow
def test_memory_usage():
    """Test memory usage is within bounds."""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1e6  # MB
    
    sim = QuantumSimulator(n_qubits=20)
    
    # Apply many gates
    for _ in range(100):
        sim.h(0)
    
    mem_after = process.memory_info().rss / 1e6  # MB
    mem_used = mem_after - mem_before
    
    # Should use less than 1 GB
    assert mem_used < 1000
```

---

## Writing New Tests

### C++ Test Template

```cpp
#include <gtest/gtest.h>
#include "your_module.h"

class YourModuleTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize test objects
    }
    
    void TearDown() override {
        // Clean up
    }
    
    // Test fixture members
};

TEST_F(YourModuleTest, TestName) {
    // Arrange
    // ... setup ...
    
    // Act
    // ... perform action ...
    
    // Assert
    EXPECT_EQ(actual, expected);
    EXPECT_TRUE(condition);
    EXPECT_NEAR(value, expected, tolerance);
}
```

### Python Test Template

```python
import pytest
from your_module import YourClass

@pytest.fixture
def your_object():
    """Fixture for test object."""
    return YourClass()

def test_your_feature(your_object):
    """Test description."""
    # Arrange
    # ... setup ...
    
    # Act
    result = your_object.your_method()
    
    # Assert
    assert result == expected
    assert pytest.approx(value, abs=tolerance)
```

---

## Test Coverage

### Measuring Coverage

**C++ coverage (gcov):**
```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug -DENABLE_COVERAGE=ON
make -j$(nproc)
ctest
gcov src/*.cpp
lcov --capture --directory . --output-file coverage.info
genhtml coverage.info --output-directory coverage_html
```

**Python coverage:**
```bash
pytest --cov=qlret --cov-report=html --cov-report=term
```

### Coverage Goals

- **Unit tests:** >90% line coverage
- **Integration tests:** >80% feature coverage
- **Critical paths:** 100% coverage (gate application, measurement)

---

## Continuous Integration

### GitHub Actions Workflow

**Location:** `.github/workflows/test.yml`

```yaml
name: Tests

on: [push, pull_request]

jobs:
  cpp-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Install dependencies
        run: |
          sudo apt-get update
          sudo apt-get install -y cmake g++ libeigen3-dev libgtest-dev
      - name: Build
        run: |
          mkdir build && cd build
          cmake .. -DBUILD_TESTS=ON
          make -j$(nproc)
      - name: Run tests
        run: |
          cd build
          ctest --output-on-failure
  
  python-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install pytest pytest-cov numpy pennylane
          pip install -e python/
      - name: Run tests
        run: |
          cd python/tests
          pytest --cov=qlret --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage.xml
```

---

## Debugging Tests

### Debugging C++ Tests

```bash
# Run with gdb
gdb --args ./test_simulator --gtest_filter=SimulatorTest.HadamardGate

# Set breakpoint
(gdb) break simulator.cpp:150
(gdb) run
(gdb) print L_
(gdb) continue
```

### Debugging Python Tests

```bash
# Run with pdb
pytest --pdb

# Or add breakpoint in code
import pdb; pdb.set_trace()
```

### Common Test Failures

**Floating-point comparison:**
```cpp
// Bad
EXPECT_EQ(value, 0.5);  // May fail due to rounding

// Good
EXPECT_NEAR(value, 0.5, 1e-10);
```

**Non-deterministic tests:**
```python
# Bad
assert results["0000"] == 500  # Exact count

# Good
assert 450 < results["0000"] < 550  # Allow variance
```

---

## See Also

- **[Code Structure](02-code-structure.md)** - Repository organization
- **[Extending the Simulator](04-extending-simulator.md)** - Adding features
- **[Contributing Guidelines](07-contributing.md)** - Submission process
- **[Performance Optimization](06-performance.md)** - Profiling and optimization

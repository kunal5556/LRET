# Phase 6b: Integration Testing - Strategic Implementation Plan

**Date:** January 3, 2026  
**Phase:** 6b (Integration Testing Layer)  
**Duration:** 3-4 hours  
**Complexity:** Medium  
**Risk:** Low (pure testing, no production code changes)  
**Model:** Claude Sonnet 4.5 (strategy), GPT-5.1 Codex Max (implementation)

---

## Executive Summary

Phase 6b creates a comprehensive integration test suite validating all execution modes across the LRET stack: C++ CLI, Python subprocess API, native pybind11 bindings, PennyLane device, and Docker containerization. Unlike unit tests (which test isolated components), integration tests validate end-to-end workflows as users will experience them.

**Core Principle:** Test the interfaces, not the internals. Validate that data flows correctly between components and produces expected results.

**Success Criteria:**
- ✅ 20+ integration tests covering all execution paths
- ✅ Tests pass in Docker container (tester stage already runs them)
- ✅ Fixtures properly isolate test environment
- ✅ Clear failure messages for debugging
- ✅ Fast execution (< 2 minutes total)
- ✅ Zero flaky tests (deterministic results)

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Test Categories](#test-categories)
3. [Implementation Steps](#implementation-steps)
4. [Test Infrastructure](#test-infrastructure)
5. [Test Modules Design](#test-modules-design)
6. [Fixtures and Utilities](#fixtures-and-utilities)
7. [Edge Cases and Error Handling](#edge-cases-and-error-handling)
8. [Docker Integration](#docker-integration)
9. [Execution Plan](#execution-plan)
10. [Success Metrics](#success-metrics)

---

## Architecture Overview

### Testing Stack Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│                   Integration Tests                         │
│                 (tests/integration/)                        │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  JSON API    │    │  PennyLane   │    │  CLI Tests   │
│   (Python)   │    │   Device     │    │ (subprocess) │
└──────────────┘    └──────────────┘    └──────────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Subprocess  │    │   Native     │    │  quantum_sim │
│   Backend    │    │  pybind11    │    │  Executable  │
└──────────────┘    └──────────────┘    └──────────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
                            ▼
                    ┌──────────────┐
                    │  LRET Core   │
                    │   (C++17)    │
                    └──────────────┘
```

### Test Coverage Map

| Component | Test File | Tests | Purpose |
|-----------|-----------|-------|---------|
| JSON API | `test_json_execution.py` | 6 | Validate JSON circuit execution (both backends) |
| PennyLane | `test_pennylane_device.py` | 8 | Device integration, gradients, sampling |
| CLI | `test_cli_regression.py` | 5 | Command-line interface validation |
| Docker | `test_docker_runtime.py` | 3 | Container execution modes |

**Total:** 22 integration tests

---

## Test Categories

### Category 1: JSON Circuit Execution (6 tests)

**File:** `tests/integration/test_json_execution.py`

**Purpose:** Validate that JSON circuit definitions execute correctly via both Python backends.

**Tests:**
1. `test_bell_pair_subprocess` - Bell state via subprocess backend
2. `test_bell_pair_native` - Bell state via native pybind11
3. `test_parametric_circuit` - Circuit with rotation parameters
4. `test_noisy_circuit` - Circuit with depolarizing noise
5. `test_sampling` - Shot-based measurement sampling
6. `test_state_export` - Full quantum state retrieval

**Key Validations:**
- Exit codes (subprocess)
- Result structure (`status`, `expectation_values`, `final_rank`)
- Numerical correctness (Bell state: `<Z₀> ≈ 0`, `<Z₀Z₁> ≈ 1`)
- Rank growth with noise
- Sample distribution matches expected probabilities

---

### Category 2: PennyLane Device (8 tests)

**File:** `tests/integration/test_pennylane_device.py`

**Purpose:** Validate PennyLane plugin integration for variational quantum algorithms.

**Tests:**
1. `test_device_creation` - Device initialization
2. `test_basic_circuit_execution` - Simple QNode evaluation
3. `test_expectation_values` - Multiple observables
4. `test_tensor_observables` - Multi-qubit observables (Z⊗Z)
5. `test_parameter_shift_gradient` - Single parameter gradient
6. `test_multi_param_gradients` - Multiple parameter gradients
7. `test_sampling_mode` - Shot-based sampling
8. `test_hermitian_observable` - Custom matrix observable

**Key Validations:**
- Device capabilities reported correctly
- QNode execution returns correct types
- Gradients match analytical expectations (e.g., d<Z>/dθ = -sin(θ) for RX)
- Sampling produces correct distributions
- Error handling for unsupported operations

---

### Category 3: CLI Regression (5 tests)

**File:** `tests/integration/test_cli_regression.py`

**Purpose:** Ensure command-line interface maintains expected behavior across updates.

**Tests:**
1. `test_basic_simulation` - Default parameters work
2. `test_parallel_modes` - All modes (`sequential`, `row`, `column`, `hybrid`)
3. `test_csv_output` - CSV file generation and format
4. `test_fdm_comparison` - FDM validation flag
5. `test_json_io` - JSON input/output workflow

**Key Validations:**
- Exit codes (0 for success, non-zero for errors)
- Stdout/stderr parsing (extract metrics)
- CSV schema correctness
- FDM fidelity thresholds (>0.99 for small n)
- JSON round-trip integrity

---

### Category 4: Docker Runtime (3 tests)

**File:** `tests/integration/test_docker_runtime.py`

**Purpose:** Validate that Docker container executes both CLI and Python workflows.

**Tests:**
1. `test_docker_cli_execution` - Run quantum_sim in container
2. `test_docker_python_import` - Import qlret in container
3. `test_docker_pennylane` - Execute PennyLane QNode in container

**Key Validations:**
- Container starts without errors
- Executables are on PATH
- Python modules importable
- Results match host execution

---

## Implementation Steps

### Step 1: Create Directory Structure (5 min)

```bash
tests/
├── integration/
│   ├── __init__.py
│   ├── conftest.py              # Shared fixtures
│   ├── test_json_execution.py   # 6 tests
│   ├── test_pennylane_device.py # 8 tests
│   ├── test_cli_regression.py   # 5 tests
│   └── test_docker_runtime.py   # 3 tests
└── pytest.ini                   # pytest configuration
```

**Actions:**
- Create `tests/integration/` directory
- Add `__init__.py` (empty file for package)
- Create placeholder test files with module docstrings

---

### Step 2: Configure pytest (10 min)

**File:** `tests/pytest.ini`

```ini
[pytest]
testpaths = integration
python_files = test_*.py
python_classes = Test*
python_functions = test_*

# Markers for selective testing
markers =
    subprocess: tests using subprocess backend
    native: tests requiring native pybind11 module
    pennylane: tests requiring PennyLane
    docker: tests requiring Docker
    slow: tests taking >10 seconds

# Output formatting
addopts = 
    -v
    --tb=short
    --strict-markers
    --color=yes
    -ra

# Timeout for individual tests (prevent hangs)
timeout = 30

# Coverage reporting (optional)
# addopts = --cov=qlret --cov-report=html
```

**Purpose:**
- Define test discovery patterns
- Create test categories via markers
- Set reasonable timeouts
- Configure output verbosity

---

### Step 3: Build Shared Fixtures (20 min)

**File:** `tests/integration/conftest.py`

```python
"""Shared fixtures for integration tests."""

import pytest
import shutil
import subprocess
from pathlib import Path
import tempfile
import json

# ============================================================================
# Path Configuration
# ============================================================================

@pytest.fixture(scope="session")
def quantum_sim_path():
    """Find quantum_sim executable."""
    # Check common locations
    candidates = [
        Path(__file__).parent.parent.parent / "build" / "quantum_sim",
        Path(__file__).parent.parent.parent / "build" / "quantum_sim.exe",
        shutil.which("quantum_sim"),
    ]
    
    for path in candidates:
        if path and Path(path).exists():
            return Path(path)
    
    pytest.skip("quantum_sim executable not found")


@pytest.fixture(scope="session")
def samples_dir():
    """Path to sample circuits."""
    path = Path(__file__).parent.parent.parent / "samples"
    if not path.exists():
        pytest.skip("samples/ directory not found")
    return path


# ============================================================================
# Backend Detection
# ============================================================================

@pytest.fixture(scope="session")
def has_native_module():
    """Check if native pybind11 module is available."""
    try:
        from qlret import _qlret_native
        return True
    except ImportError:
        return False


@pytest.fixture(scope="session")
def has_pennylane():
    """Check if PennyLane is installed."""
    try:
        import pennylane
        return True
    except ImportError:
        return False


@pytest.fixture(scope="session")
def has_docker():
    """Check if Docker is available."""
    try:
        result = subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            timeout=5
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


# ============================================================================
# Circuit Fixtures
# ============================================================================

@pytest.fixture
def bell_circuit():
    """Standard Bell pair circuit."""
    return {
        "circuit": {
            "num_qubits": 2,
            "operations": [
                {"name": "H", "wires": [0]},
                {"name": "CNOT", "wires": [0, 1]},
            ],
            "observables": [
                {"type": "PAULI", "operator": "Z", "wires": [0]},
                {"type": "TENSOR", "operators": ["Z", "Z"], "wires": [0, 1]},
            ],
        },
        "config": {
            "epsilon": 1e-4,
            "initial_rank": 1,
        },
    }


@pytest.fixture
def noisy_circuit():
    """Circuit with depolarizing noise."""
    return {
        "circuit": {
            "num_qubits": 2,
            "operations": [
                {"name": "H", "wires": [0]},
                {"name": "DEPOLARIZING", "wires": [0], "params": [0.01]},
                {"name": "CNOT", "wires": [0, 1]},
                {"name": "DEPOLARIZING", "wires": [1], "params": [0.01]},
            ],
            "observables": [
                {"type": "PAULI", "operator": "Z", "wires": [0]},
            ],
        },
        "config": {"epsilon": 1e-4},
    }


@pytest.fixture
def temp_output_dir():
    """Temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ============================================================================
# Assertion Helpers
# ============================================================================

def assert_bell_state_expectations(exp_values, tolerance=0.1):
    """Validate Bell state expectation values.
    
    Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2:
    - <Z₀> ≈ 0 (equal superposition on qubit 0)
    - <Z₀⊗Z₁> ≈ 1 (maximally correlated)
    """
    assert len(exp_values) >= 2, "Need at least 2 observables"
    
    z0 = exp_values[0]
    z0z1 = exp_values[1]
    
    assert abs(z0) < tolerance, f"<Z₀> = {z0:.4f}, expected ≈ 0"
    assert abs(z0z1 - 1.0) < tolerance, f"<Z₀Z₁> = {z0z1:.4f}, expected ≈ 1"


def assert_result_valid(result):
    """Validate basic result structure."""
    assert "status" in result, "Missing 'status' field"
    assert result["status"] == "success", f"Status: {result['status']}"
    assert "execution_time_ms" in result
    assert "final_rank" in result
    assert result["final_rank"] > 0, "Rank must be positive"


# Add to conftest.py for pytest to import
pytest.assert_bell_state_expectations = assert_bell_state_expectations
pytest.assert_result_valid = assert_result_valid
```

**Key Features:**
- Session-scoped fixtures (computed once per test run)
- Automatic skipping if dependencies missing
- Reusable circuit definitions
- Helper assertions for common validations

---

### Step 4: JSON Execution Tests (30 min)

**File:** `tests/integration/test_json_execution.py`

```python
"""Integration tests for JSON circuit execution via Python API."""

import pytest
import numpy as np
from qlret import simulate_json, QLRETError


@pytest.mark.subprocess
def test_bell_pair_subprocess(bell_circuit):
    """Test Bell pair via subprocess backend."""
    result = simulate_json(bell_circuit, use_native=False)
    
    pytest.assert_result_valid(result)
    pytest.assert_bell_state_expectations(result["expectation_values"])


@pytest.mark.native
def test_bell_pair_native(bell_circuit, has_native_module):
    """Test Bell pair via native pybind11 backend."""
    if not has_native_module:
        pytest.skip("Native module not available")
    
    result = simulate_json(bell_circuit, use_native=True)
    
    pytest.assert_result_valid(result)
    pytest.assert_bell_state_expectations(result["expectation_values"])


def test_parametric_circuit():
    """Test circuit with rotation parameters."""
    circuit = {
        "circuit": {
            "num_qubits": 1,
            "operations": [
                {"name": "RX", "wires": [0], "params": [1.5708]},  # π/2
            ],
            "observables": [
                {"type": "PAULI", "operator": "Z", "wires": [0]},
            ],
        },
        "config": {"epsilon": 1e-4},
    }
    
    result = simulate_json(circuit, use_native=False)
    
    pytest.assert_result_valid(result)
    # RX(π/2)|0⟩ = (|0⟩ - i|1⟩)/√2, <Z> ≈ 0
    assert abs(result["expectation_values"][0]) < 0.1


def test_noisy_circuit(noisy_circuit):
    """Test circuit with depolarizing noise."""
    result = simulate_json(noisy_circuit, use_native=False)
    
    pytest.assert_result_valid(result)
    
    # Noise should increase rank
    assert result["final_rank"] > 1, "Noise should increase rank"
    
    # Expectation should be slightly reduced
    z_exp = result["expectation_values"][0]
    assert abs(z_exp) < 0.5, "Noise should reduce expectation magnitude"


@pytest.mark.slow
def test_sampling():
    """Test shot-based sampling."""
    circuit = {
        "circuit": {
            "num_qubits": 2,
            "operations": [
                {"name": "H", "wires": [0]},
                {"name": "CNOT", "wires": [0, 1]},
            ],
            "observables": [],
        },
        "config": {
            "epsilon": 1e-4,
            "shots": 100,
        },
    }
    
    result = simulate_json(circuit, use_native=False)
    
    pytest.assert_result_valid(result)
    assert "samples" in result, "Missing samples field"
    assert len(result["samples"]) == 100, "Should have 100 samples"
    
    # Bell state: samples should be 0 (|00⟩) or 3 (|11⟩)
    for sample in result["samples"]:
        assert sample in [0, 3], f"Invalid Bell state sample: {sample}"


def test_state_export():
    """Test quantum state export."""
    circuit = {
        "circuit": {
            "num_qubits": 2,
            "operations": [
                {"name": "H", "wires": [0]},
            ],
            "observables": [],
        },
        "config": {
            "epsilon": 1e-4,
        },
    }
    
    result = simulate_json(circuit, use_native=False, export_state=True)
    
    pytest.assert_result_valid(result)
    assert "state" in result, "Missing state field"
    
    state = result["state"]
    assert "L_real" in state and "L_imag" in state
    assert "rank" in state
    assert state["rank"] == result["final_rank"]
```

**Test Strategy:**
- Minimal circuits (fast execution)
- Clear expected outcomes (Bell state, rotations)
- Separate subprocess vs native paths
- Edge cases (noise, sampling, state export)

---

### Step 5: PennyLane Device Tests (40 min)

**File:** `tests/integration/test_pennylane_device.py`

```python
"""Integration tests for PennyLane device."""

import pytest
import numpy as np

pennylane = pytest.importorskip("pennylane")
qml = pennylane

from qlret import QLRETDevice, QLRETDeviceError


@pytest.mark.pennylane
class TestQLRETDeviceBasics:
    """Basic device functionality tests."""
    
    def test_device_creation(self):
        """Test device initialization."""
        dev = QLRETDevice(wires=4, shots=1000)
        
        assert dev.num_wires == 4
        assert dev.shots == 1000
        assert dev.epsilon == 1e-4  # default
    
    def test_device_capabilities(self):
        """Test device capabilities reporting."""
        caps = QLRETDevice.capabilities()
        
        assert caps["model"] == "qubit"
        assert caps["supports_tensor_observables"] is True
        assert caps["supports_analytic_computation"] is True
    
    def test_basic_circuit_execution(self):
        """Test simple circuit execution."""
        dev = QLRETDevice(wires=2, shots=None)
        
        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=0)
            return qml.expval(qml.PauliZ(0))
        
        result = circuit()
        
        # H|0⟩ = |+⟩, <Z> = 0
        assert abs(result) < 0.1


@pytest.mark.pennylane
class TestObservables:
    """Observable measurement tests."""
    
    def test_single_observable(self):
        """Test single Pauli observable."""
        dev = QLRETDevice(wires=1, shots=None)
        
        @qml.qnode(dev)
        def circuit():
            qml.PauliX(wires=0)  # |0⟩ → |1⟩
            return qml.expval(qml.PauliZ(0))
        
        result = circuit()
        
        # Z|1⟩ = -|1⟩, <Z> = -1
        assert abs(result + 1.0) < 0.1
    
    def test_tensor_observables(self):
        """Test multi-qubit tensor observables."""
        dev = QLRETDevice(wires=2, shots=None)
        
        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
        
        result = circuit()
        
        # Bell state: <Z₀⊗Z₁> = 1
        assert abs(result - 1.0) < 0.1
    
    def test_hermitian_observable(self):
        """Test custom Hermitian observable."""
        dev = QLRETDevice(wires=1, shots=None)
        
        # σ_x observable
        obs_matrix = np.array([[0, 1], [1, 0]], dtype=complex)
        
        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=0)
            return qml.expval(qml.Hermitian(obs_matrix, wires=0))
        
        result = circuit()
        
        # H|0⟩ = |+⟩, <X> = 1
        assert abs(result - 1.0) < 0.1


@pytest.mark.pennylane
class TestGradients:
    """Gradient computation tests."""
    
    def test_parameter_shift_single_param(self):
        """Test single parameter gradient."""
        dev = QLRETDevice(wires=1, shots=None)
        
        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit(theta):
            qml.RX(theta, wires=0)
            return qml.expval(qml.PauliZ(0))
        
        theta = 0.5
        grad = qml.grad(circuit)(theta)
        
        # d<Z>/dθ = -sin(θ) for RX
        expected = -np.sin(theta)
        assert abs(grad - expected) < 0.1
    
    def test_multi_param_gradients(self):
        """Test multiple parameter gradients."""
        dev = QLRETDevice(wires=2, shots=None)
        
        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit(params):
            qml.RX(params[0], wires=0)
            qml.RY(params[1], wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
        
        params = np.array([0.3, 0.7])
        grads = qml.grad(circuit)(params)
        
        # Should return gradient for each parameter
        assert len(grads) == 2
        assert all(isinstance(g, (float, np.floating)) for g in grads)


@pytest.mark.pennylane
@pytest.mark.slow
class TestSampling:
    """Shot-based sampling tests."""
    
    def test_sampling_mode(self):
        """Test shot-based sampling."""
        dev = QLRETDevice(wires=2, shots=100)
        
        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.sample()
        
        samples = circuit()
        
        assert len(samples) == 100
        # Bell state samples (implementation-dependent format)
```

**Test Strategy:**
- Progressive complexity (single qubit → multi-qubit)
- Both analytic and sampling modes
- Gradient computation validation
- Custom observables (Hermitian matrices)

---

### Step 6: CLI Regression Tests (25 min)

**File:** `tests/integration/test_cli_regression.py`

```python
"""Integration tests for CLI executable."""

import pytest
import subprocess
import csv
from pathlib import Path


@pytest.mark.subprocess
def test_basic_simulation(quantum_sim_path):
    """Test basic CLI simulation with default parameters."""
    result = subprocess.run(
        [str(quantum_sim_path), "-n", "6", "-d", "8", "--mode", "sequential"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    
    assert result.returncode == 0, f"CLI failed: {result.stderr}"
    assert "Final Rank" in result.stdout
    assert "Simulation Time" in result.stdout


@pytest.mark.subprocess
def test_parallel_modes(quantum_sim_path):
    """Test all parallel modes execute successfully."""
    modes = ["sequential", "row", "column", "hybrid"]
    
    for mode in modes:
        result = subprocess.run(
            [str(quantum_sim_path), "-n", "6", "-d", "8", "--mode", mode],
            capture_output=True,
            timeout=30,
        )
        
        assert result.returncode == 0, f"Mode {mode} failed"


@pytest.mark.subprocess
def test_csv_output(quantum_sim_path, temp_output_dir):
    """Test CSV output generation and format."""
    output_file = temp_output_dir / "test_results.csv"
    
    result = subprocess.run(
        [
            str(quantum_sim_path),
            "-n", "6", "-d", "8",
            "-o", str(output_file),
        ],
        capture_output=True,
        timeout=30,
    )
    
    assert result.returncode == 0
    assert output_file.exists(), "CSV file not created"
    
    # Validate CSV structure
    with open(output_file, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
        assert len(rows) > 0, "CSV is empty"
        
        # Check required columns
        required_cols = ["num_qubits", "depth", "time_ms", "final_rank"]
        for col in required_cols:
            assert col in rows[0], f"Missing column: {col}"


@pytest.mark.subprocess
@pytest.mark.slow
def test_fdm_comparison(quantum_sim_path):
    """Test FDM validation mode."""
    result = subprocess.run(
        [str(quantum_sim_path), "-n", "8", "-d", "10", "--fdm"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    
    assert result.returncode == 0
    assert "FDM" in result.stdout or "fdm" in result.stdout.lower()
    assert "fidelity" in result.stdout.lower()


@pytest.mark.subprocess
def test_json_io(quantum_sim_path, samples_dir, temp_output_dir):
    """Test JSON input/output workflow."""
    bell_json = samples_dir / "json" / "bell_pair.json"
    
    if not bell_json.exists():
        pytest.skip("Sample JSON not found")
    
    output_json = temp_output_dir / "result.json"
    
    result = subprocess.run(
        [
            str(quantum_sim_path),
            "--input-json", str(bell_json),
            "--output-json", str(output_json),
        ],
        capture_output=True,
        timeout=30,
    )
    
    assert result.returncode == 0
    assert output_json.exists()
    
    # Validate output JSON structure
    import json
    with open(output_json) as f:
        data = json.load(f)
    
    assert "status" in data
    assert data["status"] == "success"
```

**Test Strategy:**
- Focus on exit codes and output format
- Validate file I/O (CSV, JSON)
- Test all execution modes
- Reasonable timeouts (prevent hangs)

---

### Step 7: Docker Runtime Tests (20 min)

**File:** `tests/integration/test_docker_runtime.py`

```python
"""Integration tests for Docker container runtime."""

import pytest
import subprocess


@pytest.mark.docker
def test_docker_cli_execution(has_docker):
    """Test CLI execution inside Docker container."""
    if not has_docker:
        pytest.skip("Docker not available")
    
    result = subprocess.run(
        [
            "docker", "run", "--rm", "qlret:latest",
            "./quantum_sim", "-n", "6", "-d", "8", "--mode", "sequential"
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    
    # Note: If image not built, this will fail
    if "Unable to find image" in result.stderr:
        pytest.skip("Docker image 'qlret:latest' not built")
    
    assert result.returncode == 0, f"Docker CLI failed: {result.stderr}"
    assert "Final Rank" in result.stdout


@pytest.mark.docker
def test_docker_python_import(has_docker):
    """Test Python import inside Docker container."""
    if not has_docker:
        pytest.skip("Docker not available")
    
    result = subprocess.run(
        [
            "docker", "run", "--rm", "qlret:latest",
            "python", "-c", "import qlret; print(qlret.__version__)"
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    
    if "Unable to find image" in result.stderr:
        pytest.skip("Docker image 'qlret:latest' not built")
    
    assert result.returncode == 0
    assert "1.0.0" in result.stdout


@pytest.mark.docker
@pytest.mark.pennylane
def test_docker_pennylane(has_docker):
    """Test PennyLane execution inside Docker container."""
    if not has_docker:
        pytest.skip("Docker not available")
    
    python_code = """
import pennylane as qml
from qlret import QLRETDevice

dev = QLRETDevice(wires=2, shots=None)

@qml.qnode(dev)
def circuit():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

result = circuit()
print(f"Result: {result:.4f}")
"""
    
    result = subprocess.run(
        ["docker", "run", "--rm", "qlret:latest", "python", "-c", python_code],
        capture_output=True,
        text=True,
        timeout=60,
    )
    
    if "Unable to find image" in result.stderr:
        pytest.skip("Docker image 'qlret:latest' not built")
    
    assert result.returncode == 0
    assert "Result:" in result.stdout
```

**Test Strategy:**
- Conditional execution (skip if Docker not available)
- Realistic container workflows
- Validate both CLI and Python paths
- Handle missing image gracefully

---

## Edge Cases and Error Handling

### Error Scenarios to Test

1. **Missing Dependencies**
   - Native module not built → skip native tests
   - PennyLane not installed → skip device tests
   - Docker not running → skip container tests

2. **Invalid Circuits**
   - Malformed JSON → QLRETError
   - Unsupported gates → clear error message
   - Invalid parameters → validation error

3. **Resource Limits**
   - Large circuits (n>15) → timeout or memory error
   - Deep circuits (d>100) → performance degradation
   - High rank explosion → rank capping works

4. **Numerical Issues**
   - Near-zero fidelity → no divide-by-zero
   - NaN propagation → caught and reported
   - Trace preservation violations → warnings

### Error Handling Pattern

```python
def test_invalid_circuit_error():
    """Test proper error handling for invalid circuit."""
    invalid = {
        "circuit": {
            "num_qubits": -1,  # Invalid!
            "operations": [],
        },
        "config": {},
    }
    
    with pytest.raises(QLRETError) as exc_info:
        simulate_json(invalid)
    
    # Should have informative error message
    assert "qubits" in str(exc_info.value).lower()
```

---

## Docker Integration

### Tester Stage Already Runs Tests

Our Dockerfile already has:

```dockerfile
# Stage 3: Tester (run pytest to gate the image)
FROM python-builder AS tester

WORKDIR /app

COPY --from=cpp-builder /app/build/quantum_sim /usr/local/bin/quantum_sim
ENV PATH="/usr/local/bin:${PATH}"

COPY python/tests/ tests/
COPY samples/ samples/

# Run integration tests (fail build on errors)
RUN pytest tests/ -v --tb=short
```

**This means:**
- Building Docker image automatically runs all tests
- Image build fails if any test fails (quality gate)
- No need for separate Docker test step

**Manual Testing:**

```bash
# Build and test in one step
docker build -t qlret:latest .

# If you want to run tests without full build:
docker build --target tester -t qlret:test .

# Run specific test module
docker run --rm qlret:test pytest tests/integration/test_json_execution.py -v
```

---

## Execution Plan

### Development Order (Sequential)

| Step | Task | Time | Output |
|------|------|------|--------|
| 1 | Create directory structure | 5 min | 5 files created |
| 2 | Write pytest.ini config | 10 min | Configuration file |
| 3 | Build conftest.py fixtures | 20 min | Shared infrastructure |
| 4 | Implement test_json_execution.py | 30 min | 6 tests |
| 5 | Implement test_pennylane_device.py | 40 min | 8 tests |
| 6 | Implement test_cli_regression.py | 25 min | 5 tests |
| 7 | Implement test_docker_runtime.py | 20 min | 3 tests |
| 8 | Test locally (pytest) | 15 min | Debug failures |
| 9 | Test in Docker (build image) | 20 min | Validate gate |
| **Total** | | **~3 hours** | 22 tests |

### Parallel Development (if multiple contributors)

Can be split into:
- **Track 1:** JSON + CLI tests (1.5 hours)
- **Track 2:** PennyLane + Docker tests (1.5 hours)
- **Track 3:** Fixtures + config (30 min, prerequisite)

---

## Success Metrics

### Quantitative Metrics

- ✅ **Test Count:** ≥ 20 integration tests
- ✅ **Execution Time:** < 2 minutes total
- ✅ **Coverage:** All execution paths tested
- ✅ **Pass Rate:** 100% in CI (Docker build)
- ✅ **Flakiness:** 0% (deterministic tests)

### Qualitative Metrics

- ✅ Clear test names (describes what is tested)
- ✅ Informative failure messages
- ✅ Tests are independent (can run in any order)
- ✅ Tests are maintainable (easy to update)
- ✅ Documentation in docstrings

### Acceptance Criteria

1. All 22 tests pass on first run
2. Docker build completes successfully
3. Tests skip gracefully if dependencies missing
4. No false positives or false negatives
5. Execution time < 2 minutes

---

## Implementation Checklist

Use this to track progress:

### Infrastructure
- [ ] Create `tests/integration/` directory
- [ ] Add `__init__.py`
- [ ] Write `pytest.ini` configuration
- [ ] Implement `conftest.py` fixtures
- [ ] Add helper assertions

### Test Modules
- [ ] `test_json_execution.py` (6 tests)
  - [ ] test_bell_pair_subprocess
  - [ ] test_bell_pair_native
  - [ ] test_parametric_circuit
  - [ ] test_noisy_circuit
  - [ ] test_sampling
  - [ ] test_state_export

- [ ] `test_pennylane_device.py` (8 tests)
  - [ ] test_device_creation
  - [ ] test_device_capabilities
  - [ ] test_basic_circuit_execution
  - [ ] test_single_observable
  - [ ] test_tensor_observables
  - [ ] test_hermitian_observable
  - [ ] test_parameter_shift_single_param
  - [ ] test_multi_param_gradients

- [ ] `test_cli_regression.py` (5 tests)
  - [ ] test_basic_simulation
  - [ ] test_parallel_modes
  - [ ] test_csv_output
  - [ ] test_fdm_comparison
  - [ ] test_json_io

- [ ] `test_docker_runtime.py` (3 tests)
  - [ ] test_docker_cli_execution
  - [ ] test_docker_python_import
  - [ ] test_docker_pennylane

### Validation
- [ ] Run locally: `pytest tests/integration/ -v`
- [ ] Check coverage (optional)
- [ ] Test in Docker: `docker build -t qlret:latest .`
- [ ] All tests pass in container
- [ ] Document in README (optional)

---

## Conclusion

Phase 6b creates a robust integration test suite ensuring LRET works correctly across all interfaces: CLI, Python API, PennyLane device, and Docker container. By testing end-to-end workflows rather than isolated components, we catch integration bugs that unit tests miss.

**Key Benefits:**
- 🔒 **Safety:** Docker build fails if tests fail (quality gate)
- 🎯 **Coverage:** All execution paths validated
- ⚡ **Speed:** < 2 minutes total execution time
- 🛠️ **Maintainability:** Clear test structure and documentation
- 🐛 **Debugging:** Informative failure messages

**Ready for Implementation with GPT-5.1 Codex Max!** 🚀

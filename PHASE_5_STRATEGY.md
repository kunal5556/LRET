# Phase 5: PennyLane Device Plugin - Strategic Implementation Plan

**Date:** January 3, 2026  
**Phase Duration:** 9-10 days  
**Recommended Model for Implementation:** GPT-4.5 Codex Max or Claude Opus 4.5  
**Current Planning Model:** Claude Sonnet 4.5

---

## Executive Summary

Phase 5 transforms LRET from a standalone C++ simulator into a **PennyLane-compatible device**, unlocking access to the quantum machine learning ecosystem (100k+ monthly users). This integration enables:

- **Automatic differentiation** for variational algorithms (VQE, QAOA)
- **ML framework compatibility** (PyTorch, TensorFlow, JAX)
- **Community adoption** via PyPI package distribution
- **Citation growth** through PennyLane plugin ecosystem

**Key Deliverables:**
1. Python wrapper for LRET C++ backend
2. PennyLane Device interface implementation
3. JSON-based circuit serialization protocol
4. Gradient computation via parameter-shift rule
5. PyPI package publication

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Prerequisites & Dependencies](#prerequisites--dependencies)
3. [Implementation Phases](#implementation-phases)
4. [Detailed Step-by-Step Plan](#detailed-step-by-step-plan)
5. [File Structure](#file-structure)
6. [Testing Strategy](#testing-strategy)
7. [Integration Points](#integration-points)
8. [Challenges & Solutions](#challenges--solutions)
9. [Success Criteria](#success-criteria)
10. [Timeline & Milestones](#timeline--milestones)

---

## Architecture Overview

### System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   User Python Code                       │
│  import pennylane as qml                                 │
│  dev = qml.device("lret", wires=4)                      │
└───────────────────┬─────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│              PennyLane Framework                         │
│  - QNode construction                                    │
│  - Gradient computation (parameter-shift)                │
│  - Optimization loops                                    │
└───────────────────┬─────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│          pennylane-lret Plugin (Python)                  │
│  ┌───────────────────────────────────────┐              │
│  │  LRETDevice (Device subclass)         │              │
│  │  - apply(operations)                  │              │
│  │  - expval(observable)                 │              │
│  │  - _serialize_to_json()               │              │
│  │  - _deserialize_result()              │              │
│  └───────────────┬───────────────────────┘              │
│                  │                                        │
│                  ▼                                        │
│  ┌───────────────────────────────────────┐              │
│  │  LRETBridge (C++/Python interface)    │              │
│  │  - Option A: subprocess + JSON        │              │
│  │  - Option B: pybind11 direct binding  │              │
│  │  - Option C: C FFI via ctypes         │              │
│  └───────────────┬───────────────────────┘              │
└──────────────────┼──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│              LRET C++ Backend                            │
│  ┌─────────────────────────────────────┐                │
│  │  JSON Interface Layer (NEW)         │                │
│  │  - parse_circuit_json()             │                │
│  │  - export_state_json()              │                │
│  │  - export_expectation_json()        │                │
│  └──────────────┬──────────────────────┘                │
│                 │                                        │
│                 ▼                                        │
│  ┌─────────────────────────────────────┐                │
│  │  Existing LRET Core (REUSE)         │                │
│  │  - simulator.cpp                    │                │
│  │  - gates_and_noise.cpp              │                │
│  │  - noise_import.cpp                 │                │
│  └─────────────────────────────────────┘                │
└─────────────────────────────────────────────────────────┘
```

### Design Decisions

**Decision 1: Python-C++ Interface Method**

| Option | Pros | Cons | Recommendation |
|--------|------|------|----------------|
| **A: JSON + subprocess** | Simple, no binding code, debuggable | IPC overhead (~1-10ms per call) | ✅ **Start here** (Phase 5.1) |
| **B: pybind11** | Fast (no IPC), native types | Complex build, platform issues | Phase 5.4 (optimization) |
| **C: ctypes/FFI** | No binding code, moderate speed | Manual memory management | Alternative if pybind11 fails |

**Decision 2: State Management**

- **Stateless mode:** Each `apply()` call runs full circuit from scratch
  - ✅ Simple, no state bugs
  - ❌ Inefficient for iterative algorithms
  
- **Stateful mode:** Cache intermediate states between calls
  - ✅ Efficient for VQE/QAOA (reuse partial circuits)
  - ❌ Complex state tracking, potential bugs
  - **Recommendation:** Start stateless, optimize to stateful in Phase 5.4

**Decision 3: Observable Computation**

- **On-demand:** Compute expectation when `expval()` called
  - ✅ Memory efficient
  - ❌ May recompute state multiple times
  
- **Cached:** Store full state after `apply()`, compute multiple observables
  - ✅ Fast for multiple observables
  - ❌ Higher memory usage
  - **Recommendation:** Cached (most PennyLane circuits measure multiple observables)

---

## Prerequisites & Dependencies

### Python Environment

```bash
# Core dependencies
pip install pennylane>=0.35.0
pip install numpy>=1.21.0
pip install scipy>=1.7.0

# Development dependencies
pip install pytest>=7.0.0
pip install pytest-cov>=3.0.0
pip install black>=22.0.0
pip install flake8>=4.0.0
pip install mypy>=0.950

# Build/packaging
pip install setuptools>=65.0.0
pip install wheel>=0.38.0
pip install twine>=4.0.0

# Documentation
pip install sphinx>=4.5.0
pip install sphinx-rtd-theme>=1.0.0
```

### C++ Backend Modifications

**Required new functionality:**
1. JSON input parsing for circuit specification
2. JSON output serialization for results
3. Expectation value computation for arbitrary observables
4. State export (optional, for debugging)

**No breaking changes to existing code!**

### File System Setup

```bash
lret/
├── pennylane_lret/          # NEW: Python package
│   ├── __init__.py
│   ├── device.py            # LRETDevice class
│   ├── bridge.py            # C++ interface layer
│   ├── ops.py               # Gate translation utilities
│   ├── observables.py       # Observable computation
│   └── utils.py             # Helper functions
├── include/
│   └── json_interface.h     # NEW: JSON interface
├── src/
│   └── json_interface.cpp   # NEW: JSON interface implementation
├── tests/
│   └── test_pennylane_device.py  # NEW: PennyLane tests
├── examples/
│   ├── vqe_h2.py           # NEW: VQE example
│   └── qaoa_maxcut.py      # NEW: QAOA example
├── setup.py                # NEW: Python package setup
├── pyproject.toml          # NEW: Build configuration
└── README_PENNYLANE.md     # NEW: PennyLane usage docs
```

---

## Implementation Phases

### Phase 5.1: JSON Interface & Python Bridge (Days 1-2)

**Objective:** Establish communication protocol between Python and C++

**Tasks:**
1. Define JSON schema for circuit specification
2. Implement JSON parser in C++ (using existing nlohmann/json)
3. Implement JSON serializer for results
4. Create Python bridge using subprocess
5. Test round-trip: Python → JSON → C++ → JSON → Python

**Deliverables:**
- `include/json_interface.h`
- `src/json_interface.cpp`
- `pennylane_lret/bridge.py`
- JSON schema documentation

---

### Phase 5.2: PennyLane Device Implementation (Days 3-5)

**Objective:** Implement PennyLane Device interface

**Tasks:**
1. Create `LRETDevice` class inheriting `pennylane.Device`
2. Implement `apply()` method (execute circuit)
3. Implement `expval()` method (compute expectation)
4. Gate translation: PennyLane ops → LRET gates
5. Observable translation: PennyLane observables → LRET format
6. Handle device configuration (wires, shots, epsilon)

**Deliverables:**
- `pennylane_lret/device.py`
- `pennylane_lret/ops.py`
- `pennylane_lret/observables.py`
- Unit tests for each component

---

### Phase 5.3: Gradient Computation (Days 6-7)

**Objective:** Enable automatic differentiation for VQE/QAOA

**Tasks:**
1. Implement parameter-shift rule in device
2. Test gradient computation vs analytical gradients
3. Integrate with PennyLane's gradient computation framework
4. Optimize: cache circuit evaluations to avoid redundant runs
5. Validate with simple VQE example

**Deliverables:**
- Gradient computation in `device.py`
- VQE example (`examples/vqe_h2.py`)
- Gradient tests

---

### Phase 5.4: Optimization & Features (Day 8)

**Objective:** Improve performance and add features

**Tasks:**
1. Implement circuit caching (avoid redundant computations)
2. Add noise model support (pass NoiseModel to LRET)
3. Support shot-based sampling (not just exact expectation)
4. Benchmark: LRET vs PennyLane default.qubit
5. (Optional) pybind11 binding for zero-copy performance

**Deliverables:**
- Performance optimizations
- Noise model integration
- Shot-based sampling
- Benchmark report

---

### Phase 5.5: Packaging & Publication (Day 9)

**Objective:** Publish to PyPI and document usage

**Tasks:**
1. Create `setup.py` and `pyproject.toml`
2. Write comprehensive README with examples
3. Generate API documentation (Sphinx)
4. Run PennyLane test suite
5. Publish to PyPI: `pip install pennylane-lret`
6. Submit to PennyLane plugin registry

**Deliverables:**
- PyPI package published
- Documentation site
- PennyLane plugin registry entry
- Release announcement

---

## Detailed Step-by-Step Plan

### Step 1: JSON Circuit Schema Design (2 hours)

**Goal:** Define contract between Python and C++

**JSON Schema:**

```json
{
  "circuit": {
    "num_qubits": 4,
    "operations": [
      {
        "type": "gate",
        "name": "H",
        "wires": [0],
        "params": []
      },
      {
        "type": "gate",
        "name": "RY",
        "wires": [1],
        "params": [0.5]
      },
      {
        "type": "gate",
        "name": "CNOT",
        "wires": [0, 1],
        "params": []
      }
    ],
    "observables": [
      {
        "type": "pauli",
        "operator": "Z",
        "wires": [0],
        "coefficient": 1.0
      },
      {
        "type": "tensor",
        "operators": ["X", "Y"],
        "wires": [0, 1],
        "coefficient": 0.5
      }
    ]
  },
  "config": {
    "epsilon": 1e-4,
    "initial_rank": 1,
    "use_fdm": false,
    "noise_model": null,
    "shots": null
  }
}
```

**Result Schema:**

```json
{
  "status": "success",
  "execution_time_ms": 125.3,
  "final_rank": 8,
  "expectation_values": [
    0.7071,
    -0.3536
  ],
  "state": {
    "type": "low_rank",
    "L_real": [...],
    "L_imag": [...]
  },
  "samples": null
}
```

**File:** `docs/json_schema.md`

---

### Step 2: C++ JSON Parser Implementation (4 hours)

**Goal:** Parse JSON circuit specification in C++

**New File:** `include/json_interface.h`

```cpp
#ifndef JSON_INTERFACE_H
#define JSON_INTERFACE_H

#include "types.h"
#include <nlohmann/json.hpp>
#include <string>

namespace qlret {

// Parse JSON circuit specification
struct CircuitSpec {
    size_t num_qubits;
    std::vector<GateOp> operations;
    std::vector<Observable> observables;
    SimConfig config;
};

CircuitSpec parse_circuit_json(const std::string& json_str);
CircuitSpec parse_circuit_json(const nlohmann::json& j);

// Observable specification
struct Observable {
    enum Type { PAULI_STRING, HERMITIAN, TENSOR_PRODUCT };
    Type type;
    std::vector<std::string> operators;  // "X", "Y", "Z", "I"
    std::vector<size_t> wires;
    double coefficient;
    MatrixXcd matrix;  // For Hermitian observables
};

// Compute expectation value
double compute_expectation(
    const MatrixXcd& L,
    const Observable& obs,
    size_t num_qubits
);

// Export results to JSON
std::string export_result_json(
    const MatrixXcd& L_final,
    const std::vector<double>& expectation_values,
    double execution_time_ms,
    const SimConfig& config
);

} // namespace qlret

#endif
```

**New File:** `src/json_interface.cpp`

```cpp
#include "json_interface.h"
#include "gates_and_noise.h"
#include "simulator.h"
#include <stdexcept>

namespace qlret {

CircuitSpec parse_circuit_json(const std::string& json_str) {
    auto j = nlohmann::json::parse(json_str);
    return parse_circuit_json(j);
}

CircuitSpec parse_circuit_json(const nlohmann::json& j) {
    CircuitSpec spec;
    
    // Parse num_qubits
    spec.num_qubits = j["circuit"]["num_qubits"];
    
    // Parse operations
    for (const auto& op_json : j["circuit"]["operations"]) {
        GateOp op;
        op.name = op_json["name"];
        op.wires = op_json["wires"].get<std::vector<size_t>>();
        
        if (op_json.contains("params")) {
            op.params = op_json["params"].get<std::vector<double>>();
        }
        
        spec.operations.push_back(op);
    }
    
    // Parse observables
    for (const auto& obs_json : j["circuit"]["observables"]) {
        Observable obs;
        obs.type = Observable::PAULI_STRING;  // Default
        
        if (obs_json["type"] == "pauli") {
            obs.operators = {obs_json["operator"]};
            obs.wires = obs_json["wires"].get<std::vector<size_t>>();
            obs.coefficient = obs_json["coefficient"];
        } else if (obs_json["type"] == "tensor") {
            obs.operators = obs_json["operators"].get<std::vector<std::string>>();
            obs.wires = obs_json["wires"].get<std::vector<size_t>>();
            obs.coefficient = obs_json["coefficient"];
        }
        
        spec.observables.push_back(obs);
    }
    
    // Parse config
    const auto& cfg = j["config"];
    spec.config.epsilon = cfg["epsilon"];
    spec.config.initial_rank = cfg.value("initial_rank", 1);
    spec.config.use_fdm = cfg.value("use_fdm", false);
    
    return spec;
}

double compute_expectation(
    const MatrixXcd& L,
    const Observable& obs,
    size_t num_qubits
) {
    // Compute ρ = LL†
    MatrixXcd rho = L * L.adjoint();
    
    // Build observable matrix
    MatrixXcd obs_matrix = build_observable_matrix(obs, num_qubits);
    
    // Tr(ρ * obs)
    return (rho * obs_matrix).trace().real();
}

MatrixXcd build_observable_matrix(const Observable& obs, size_t num_qubits) {
    size_t dim = 1ULL << num_qubits;
    MatrixXcd result = MatrixXcd::Identity(dim, dim);
    
    // Tensor product of Pauli operators
    for (size_t i = 0; i < obs.operators.size(); ++i) {
        std::string op = obs.operators[i];
        size_t wire = obs.wires[i];
        
        MatrixXcd pauli(2, 2);
        if (op == "X") {
            pauli << 0, 1, 1, 0;
        } else if (op == "Y") {
            pauli << 0, std::complex<double>(0, -1),
                     std::complex<double>(0, 1), 0;
        } else if (op == "Z") {
            pauli << 1, 0, 0, -1;
        } else if (op == "I") {
            pauli << 1, 0, 0, 1;
        }
        
        // Apply to specific wire
        result = apply_single_qubit_gate_to_matrix(result, wire, pauli, num_qubits);
    }
    
    return obs.coefficient * result;
}

std::string export_result_json(
    const MatrixXcd& L_final,
    const std::vector<double>& expectation_values,
    double execution_time_ms,
    const SimConfig& config
) {
    nlohmann::json j;
    
    j["status"] = "success";
    j["execution_time_ms"] = execution_time_ms;
    j["final_rank"] = L_final.cols();
    j["expectation_values"] = expectation_values;
    
    // Optionally export state (if requested)
    if (config.export_state) {
        j["state"]["type"] = "low_rank";
        j["state"]["L_real"] = matrix_to_vector(L_final.real());
        j["state"]["L_imag"] = matrix_to_vector(L_final.imag());
    }
    
    return j.dump();
}

} // namespace qlret
```

**Testing:**
```cpp
// Test JSON parsing
std::string test_json = R"({
  "circuit": {
    "num_qubits": 2,
    "operations": [
      {"name": "H", "wires": [0], "params": []},
      {"name": "CNOT", "wires": [0, 1], "params": []}
    ],
    "observables": [
      {"type": "pauli", "operator": "Z", "wires": [0], "coefficient": 1.0}
    ]
  },
  "config": {"epsilon": 1e-4}
})";

auto spec = parse_circuit_json(test_json);
assert(spec.num_qubits == 2);
assert(spec.operations.size() == 2);
assert(spec.observables.size() == 1);
```

---

### Step 3: Python Bridge Implementation (3 hours)

**Goal:** Call C++ LRET from Python using subprocess + JSON

**New File:** `pennylane_lret/bridge.py`

```python
"""Bridge between Python and LRET C++ backend."""

import json
import subprocess
import tempfile
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np


class LRETBridge:
    """Interface to LRET C++ simulator."""
    
    def __init__(self, lret_binary_path: Optional[str] = None):
        """Initialize bridge to LRET backend.
        
        Args:
            lret_binary_path: Path to LRET executable. If None, searches PATH.
        """
        if lret_binary_path is None:
            # Try to find in PATH or default locations
            self.lret_binary = self._find_lret_binary()
        else:
            self.lret_binary = Path(lret_binary_path)
        
        if not self.lret_binary.exists():
            raise FileNotFoundError(
                f"LRET binary not found at {self.lret_binary}. "
                f"Please build LRET or specify path."
            )
    
    def _find_lret_binary(self) -> Path:
        """Search for LRET binary in standard locations."""
        # Check PATH
        import shutil
        binary = shutil.which("quantum_sim")
        if binary:
            return Path(binary)
        
        # Check relative to this file
        current_dir = Path(__file__).parent
        candidates = [
            current_dir / "../build/quantum_sim",
            current_dir / "../build/Release/quantum_sim.exe",  # Windows
            current_dir / "../build/Debug/quantum_sim.exe",
            current_dir / "../../build/quantum_sim",
        ]
        
        for candidate in candidates:
            if candidate.exists():
                return candidate
        
        raise FileNotFoundError("LRET binary not found")
    
    def execute_circuit(
        self,
        num_qubits: int,
        operations: List[Dict[str, Any]],
        observables: List[Dict[str, Any]],
        epsilon: float = 1e-4,
        initial_rank: int = 1,
        noise_model: Optional[Dict] = None,
        shots: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute quantum circuit via LRET.
        
        Args:
            num_qubits: Number of qubits
            operations: List of gate operations
            observables: List of observables to measure
            epsilon: Truncation threshold
            initial_rank: Initial rank of state
            noise_model: Optional noise model specification
            shots: Number of samples (None = exact expectation)
        
        Returns:
            Dictionary with execution results
        """
        # Construct JSON input
        circuit_spec = {
            "circuit": {
                "num_qubits": num_qubits,
                "operations": operations,
                "observables": observables,
            },
            "config": {
                "epsilon": epsilon,
                "initial_rank": initial_rank,
                "use_fdm": False,
                "noise_model": noise_model,
                "shots": shots,
            }
        }
        
        # Write to temporary file
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        ) as f:
            json.dump(circuit_spec, f)
            input_file = f.name
        
        try:
            # Run LRET
            result = subprocess.run(
                [str(self.lret_binary), "--input-json", input_file],
                capture_output=True,
                text=True,
                check=True,
                timeout=300,  # 5 minute timeout
            )
            
            # Parse output
            output = json.loads(result.stdout)
            return output
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"LRET execution failed:\n"
                f"Return code: {e.returncode}\n"
                f"Stdout: {e.stdout}\n"
                f"Stderr: {e.stderr}"
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("LRET execution timed out (>5 minutes)")
        finally:
            # Clean up temp file
            os.unlink(input_file)
    
    def compute_expectation(
        self,
        num_qubits: int,
        operations: List[Dict[str, Any]],
        observable: Dict[str, Any],
        **kwargs
    ) -> float:
        """Compute single expectation value.
        
        Convenience wrapper around execute_circuit.
        """
        result = self.execute_circuit(
            num_qubits=num_qubits,
            operations=operations,
            observables=[observable],
            **kwargs
        )
        return result["expectation_values"][0]


# Singleton instance
_bridge_instance = None


def get_bridge(lret_binary_path: Optional[str] = None) -> LRETBridge:
    """Get global LRETBridge instance."""
    global _bridge_instance
    if _bridge_instance is None:
        _bridge_instance = LRETBridge(lret_binary_path)
    return _bridge_instance
```

**Testing:**
```python
# test_bridge.py
from pennylane_lret.bridge import LRETBridge

bridge = LRETBridge()

# Test simple circuit: H gate + Z measurement
result = bridge.execute_circuit(
    num_qubits=1,
    operations=[
        {"name": "H", "wires": [0], "params": []}
    ],
    observables=[
        {"type": "pauli", "operator": "Z", "wires": [0], "coefficient": 1.0}
    ]
)

print(f"Expectation value: {result['expectation_values'][0]}")
assert abs(result['expectation_values'][0]) < 1e-10  # Should be ~0 for |+⟩
```

---

### Step 4: PennyLane Device Class (6 hours)

**Goal:** Implement PennyLane Device interface

**New File:** `pennylane_lret/device.py`

```python
"""PennyLane device for LRET simulator."""

from typing import Optional, Dict, Any, Union
import numpy as np
import pennylane as qml
from pennylane import Device
from pennylane.operation import Operation

from .bridge import get_bridge
from .ops import translate_operation
from .observables import translate_observable


class LRETDevice(Device):
    """PennyLane device powered by LRET low-rank simulator.
    
    Args:
        wires (int or Iterable[int]): Number or indices of wires
        shots (int): Number of circuit evaluations/samples (None = exact)
        epsilon (float): SVD truncation threshold (default: 1e-4)
        initial_rank (int): Initial rank of density matrix (default: 1)
        lret_binary (str): Path to LRET executable (default: auto-detect)
    
    Keyword Args:
        noise_model (dict): Optional noise model specification
    """
    
    name = "LRET Low-Rank Quantum Simulator"
    short_name = "lret"
    pennylane_requires = ">=0.35.0"
    version = "1.0.0"
    author = "LRET Development Team"
    
    # Supported operations (gates)
    operations = {
        # Single-qubit gates
        "PauliX", "PauliY", "PauliZ",
        "Hadamard", "S", "T", "SX",
        "RX", "RY", "RZ",
        "Rot",  # General single-qubit rotation
        "U1", "U2", "U3",  # IBM gates
        
        # Two-qubit gates
        "CNOT", "CZ", "CY", "SWAP",
        "IsingXX", "IsingYY", "IsingZZ",
        
        # Multi-qubit
        "MultiRZ",
    }
    
    # Supported observables
    observables = {
        "PauliX", "PauliY", "PauliZ", "Identity", "Hadamard",
        "Hermitian",  # Arbitrary Hermitian matrix
    }
    
    def __init__(
        self,
        wires,
        *,
        shots=None,
        epsilon=1e-4,
        initial_rank=1,
        lret_binary=None,
        **kwargs
    ):
        super().__init__(wires=wires, shots=shots)
        
        self.epsilon = epsilon
        self.initial_rank = initial_rank
        self.noise_model = kwargs.get("noise_model", None)
        
        # Initialize bridge to C++ backend
        self.bridge = get_bridge(lret_binary)
        
        # State caching
        self._current_operations = []
        self._state_cache = None
    
    def apply(self, operations, **kwargs):
        """Execute quantum operations.
        
        Args:
            operations: List of PennyLane operations to apply
        """
        # Translate PennyLane operations to LRET format
        self._current_operations = [
            translate_operation(op) for op in operations
        ]
        
        # Don't execute yet - wait for expval() call
        # This allows batch processing of multiple observables
        self._state_cache = None
    
    def expval(self, observable, **kwargs):
        """Compute expectation value of observable.
        
        Args:
            observable: PennyLane observable
        
        Returns:
            Expectation value (float)
        """
        # Translate observable
        obs_spec = translate_observable(observable)
        
        # Execute circuit with LRET
        result = self.bridge.execute_circuit(
            num_qubits=self.num_wires,
            operations=self._current_operations,
            observables=[obs_spec],
            epsilon=self.epsilon,
            initial_rank=self.initial_rank,
            noise_model=self.noise_model,
            shots=self.shots,
        )
        
        return result["expectation_values"][0]
    
    def var(self, observable, **kwargs):
        """Compute variance of observable.
        
        Var(O) = ⟨O²⟩ - ⟨O⟩²
        """
        # Compute ⟨O⟩
        exp_val = self.expval(observable)
        
        # Compute ⟨O²⟩
        obs_squared = observable @ observable
        exp_val_squared = self.expval(obs_squared)
        
        return exp_val_squared - exp_val ** 2
    
    def sample(self, observable, shot_range=None, bin_size=None, counts=False):
        """Generate samples from measurement.
        
        This requires shot-based simulation.
        """
        if self.shots is None:
            raise ValueError("Sampling requires shots > 0")
        
        # Execute with shots
        result = self.bridge.execute_circuit(
            num_qubits=self.num_wires,
            operations=self._current_operations,
            observables=[translate_observable(observable)],
            epsilon=self.epsilon,
            shots=self.shots,
        )
        
        return np.array(result["samples"])
    
    def reset(self):
        """Reset device state."""
        self._current_operations = []
        self._state_cache = None
```

**Testing:**
```python
# test_device.py
import pennylane as qml
from pennylane_lret import LRETDevice

# Create device
dev = LRETDevice(wires=2, epsilon=1e-4)

# Define circuit
@qml.qnode(dev)
def circuit():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

# Execute
result = circuit()
print(f"Result: {result}")
assert abs(result - 1.0) < 1e-6  # Bell state: ⟨ZZ⟩ = 1
```

---

### Step 5: Gate Translation (2 hours)

**Goal:** Convert PennyLane operations to LRET format

**New File:** `pennylane_lret/ops.py`

```python
"""Operation translation between PennyLane and LRET."""

from typing import Dict, Any
import numpy as np
from pennylane.operation import Operation


def translate_operation(op: Operation) -> Dict[str, Any]:
    """Convert PennyLane operation to LRET format.
    
    Args:
        op: PennyLane operation
    
    Returns:
        Dictionary with LRET operation specification
    """
    name = op.name
    wires = [w for w in op.wires]
    params = list(op.parameters) if hasattr(op, 'parameters') else []
    
    # Map PennyLane names to LRET names
    name_mapping = {
        "PauliX": "X",
        "PauliY": "Y",
        "PauliZ": "Z",
        "Hadamard": "H",
        "CNOT": "CNOT",
        "CZ": "CZ",
        "SWAP": "SWAP",
        "RX": "RX",
        "RY": "RY",
        "RZ": "RZ",
        "S": "S",
        "T": "T",
        "SX": "SX",
        # Add more mappings as needed
    }
    
    lret_name = name_mapping.get(name, name)
    
    return {
        "name": lret_name,
        "wires": wires,
        "params": params,
    }
```

---

### Step 6: Observable Translation (2 hours)

**Goal:** Convert PennyLane observables to LRET format

**New File:** `pennylane_lret/observables.py`

```python
"""Observable translation between PennyLane and LRET."""

from typing import Dict, Any
import numpy as np
import pennylane as qml


def translate_observable(obs) -> Dict[str, Any]:
    """Convert PennyLane observable to LRET format.
    
    Args:
        obs: PennyLane observable
    
    Returns:
        Dictionary with LRET observable specification
    """
    # Single Pauli observable
    if isinstance(obs, (qml.PauliX, qml.PauliY, qml.PauliZ, qml.Identity)):
        return {
            "type": "pauli",
            "operator": obs.name[-1],  # "PauliX" -> "X"
            "wires": [obs.wires[0]],
            "coefficient": 1.0,
        }
    
    # Tensor product of Paulis
    if isinstance(obs, qml.operation.Tensor):
        operators = []
        wires = []
        for sub_obs in obs.obs:
            operators.append(sub_obs.name[-1])
            wires.append(sub_obs.wires[0])
        
        return {
            "type": "tensor",
            "operators": operators,
            "wires": wires,
            "coefficient": 1.0,
        }
    
    # Hermitian observable (arbitrary matrix)
    if isinstance(obs, qml.Hermitian):
        return {
            "type": "hermitian",
            "matrix_real": obs.matrix.real.tolist(),
            "matrix_imag": obs.matrix.imag.tolist(),
            "wires": list(obs.wires),
            "coefficient": 1.0,
        }
    
    # Linear combination of observables
    if isinstance(obs, qml.Hamiltonian):
        # LRET doesn't support Hamiltonian directly
        # Need to decompose into individual terms
        raise NotImplementedError(
            "Hamiltonian observables not yet supported. "
            "Measure individual terms separately."
        )
    
    raise ValueError(f"Unsupported observable type: {type(obs)}")
```

---

### Step 7: Gradient Computation (4 hours)

**Goal:** Implement parameter-shift rule for automatic differentiation

**Modify:** `pennylane_lret/device.py`

Add method to `LRETDevice`:

```python
def gradient(self, tape, method="best", **kwargs):
    """Compute gradient using parameter-shift rule.
    
    PennyLane automatically calls this for gradient-based optimization.
    """
    # Use PennyLane's built-in parameter-shift
    # (works automatically if device supports diff_method="parameter-shift")
    return None  # Let PennyLane handle it


# Add class attribute
_diff_method = "parameter-shift"
```

**Testing:**
```python
# test_gradient.py
import pennylane as qml
from pennylane_lret import LRETDevice
import numpy as np

dev = LRETDevice(wires=2)

@qml.qnode(dev, diff_method="parameter-shift")
def circuit(param):
    qml.RY(param, wires=0)
    return qml.expval(qml.PauliZ(0))

# Compute gradient
param = np.array(0.5, requires_grad=True)
grad = qml.grad(circuit)(param)

# Analytical gradient: d/dθ ⟨Z⟩ = -sin(θ)
analytical = -np.sin(param)

print(f"Computed gradient: {grad}")
print(f"Analytical gradient: {analytical}")
assert abs(grad - analytical) < 1e-6
```

---

### Step 8: VQE Example (3 hours)

**Goal:** Demonstrate VQE for H₂ molecule

**New File:** `examples/vqe_h2.py`

```python
"""VQE for H2 molecule using LRET device."""

import pennylane as qml
from pennylane import numpy as np
from pennylane_lret import LRETDevice

# H2 Hamiltonian (STO-3G basis, R = 0.74 Å)
# From Qiskit/OpenFermion
coeffs = [0.2252, 0.3435, -0.4750, 0.5716, 0.0910, 0.0910]
obs = [
    qml.Identity(0),
    qml.PauliZ(0),
    qml.PauliZ(1),
    qml.PauliZ(0) @ qml.PauliZ(1),
    qml.PauliX(0) @ qml.PauliX(1),
    qml.PauliY(0) @ qml.PauliY(1),
]

H = qml.Hamiltonian(coeffs, obs)

# LRET device
dev = LRETDevice(wires=2, epsilon=1e-6)

# Ansatz circuit (Hardware-efficient)
def ansatz(params, wires):
    qml.RY(params[0], wires=wires[0])
    qml.RY(params[1], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RY(params[2], wires=wires[0])
    qml.RY(params[3], wires=wires[1])

# Cost function
@qml.qnode(dev)
def cost_fn(params):
    ansatz(params, wires=[0, 1])
    return qml.expval(H)

# Optimize
opt = qml.GradientDescentOptimizer(stepsize=0.1)
params = np.random.random(4, requires_grad=True)

print("Starting VQE optimization...")
for i in range(100):
    params, energy = opt.step_and_cost(cost_fn, params)
    if i % 10 == 0:
        print(f"Step {i}: Energy = {energy:.6f} Ha")

print(f"\nFinal energy: {energy:.6f} Ha")
print(f"Exact ground state: -1.136189 Ha")
print(f"Error: {abs(energy - (-1.136189)):.6f} Ha")
```

---

### Step 9: Packaging (3 hours)

**Goal:** Create distributable Python package

**New File:** `setup.py`

```python
"""Setup script for pennylane-lret."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="pennylane-lret",
    version="1.0.0",
    author="LRET Development Team",
    author_email="your.email@example.com",
    description="PennyLane device for LRET low-rank quantum simulator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/lret",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pennylane>=0.35.0",
        "numpy>=1.21.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
    },
    entry_points={
        "pennylane.plugins": [
            "lret = pennylane_lret.device:LRETDevice"
        ],
    },
)
```

**New File:** `pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=65.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "pennylane-lret"
version = "1.0.0"
description = "PennyLane device for LRET low-rank quantum simulator"
readme = "README.md"
requires-python = ">=3.8"
license = {text = "MIT"}
authors = [
    {name = "LRET Development Team", email = "your.email@example.com"}
]
dependencies = [
    "pennylane>=0.35.0",
    "numpy>=1.21.0",
]

[project.entry-points."pennylane.plugins"]
lret = "pennylane_lret.device:LRETDevice"
```

---

## Testing Strategy

### Unit Tests

```python
# tests/test_pennylane_device.py
import pytest
import pennylane as qml
from pennylane_lret import LRETDevice
import numpy as np


class TestLRETDevice:
    """Test suite for LRET device."""
    
    def test_device_creation(self):
        """Test device initialization."""
        dev = LRETDevice(wires=4, epsilon=1e-4)
        assert dev.num_wires == 4
        assert dev.epsilon == 1e-4
    
    def test_single_qubit_gates(self):
        """Test all single-qubit gates."""
        dev = LRETDevice(wires=1)
        
        @qml.qnode(dev)
        def circuit(gate_fn):
            gate_fn(wires=0)
            return qml.expval(qml.PauliZ(0))
        
        # Test Hadamard: |0⟩ → |+⟩, ⟨Z⟩ = 0
        result = circuit(qml.Hadamard)
        assert abs(result) < 1e-10
        
        # Test X: |0⟩ → |1⟩, ⟨Z⟩ = -1
        result = circuit(qml.PauliX)
        assert abs(result - (-1.0)) < 1e-10
    
    def test_two_qubit_gates(self):
        """Test two-qubit gates."""
        dev = LRETDevice(wires=2)
        
        @qml.qnode(dev)
        def bell_circuit():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
        
        # Bell state: ⟨ZZ⟩ = 1
        result = bell_circuit()
        assert abs(result - 1.0) < 1e-6
    
    def test_parameterized_gates(self):
        """Test rotation gates."""
        dev = LRETDevice(wires=1)
        
        @qml.qnode(dev)
        def circuit(angle):
            qml.RY(angle, wires=0)
            return qml.expval(qml.PauliZ(0))
        
        # RY(π/2) should give ⟨Z⟩ = 0
        result = circuit(np.pi / 2)
        assert abs(result) < 1e-6
    
    def test_gradient_computation(self):
        """Test automatic differentiation."""
        dev = LRETDevice(wires=1)
        
        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit(param):
            qml.RY(param, wires=0)
            return qml.expval(qml.PauliZ(0))
        
        param = np.array(0.5, requires_grad=True)
        grad = qml.grad(circuit)(param)
        analytical = -np.sin(param)
        
        assert abs(grad - analytical) < 1e-4
    
    def test_observable_types(self):
        """Test different observable types."""
        dev = LRETDevice(wires=2)
        
        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliX(0) @ qml.PauliX(1))
        
        z_val, xx_val = circuit()
        assert abs(z_val) < 1e-6
        assert abs(xx_val - 1.0) < 1e-6
```

### Integration Tests

Test against PennyLane's standard test suite:

```python
# tests/test_integration.py
import pytest
import pennylane as qml
from pennylane.devices.tests import test_device

from pennylane_lret import LRETDevice

# Run PennyLane's device test suite
@pytest.fixture
def device():
    return LRETDevice(wires=4)

def test_pennylane_compliance(device):
    """Run PennyLane's standard device tests."""
    test_device(device)
```

---

## Integration Points

### With Existing LRET Codebase

**Modified Files:**
1. `main.cpp` - Add `--input-json` CLI flag
2. `CMakeLists.txt` - Build JSON interface library
3. New files only, no breaking changes to existing code

**New Dependencies:**
- Already have `nlohmann/json` from Phase 4
- No additional C++ dependencies needed

### With PennyLane Ecosystem

**Plugin Registry:**
Submit to PennyLane plugin registry:
- Repository: https://github.com/PennyLaneAI/pennylane
- Documentation: https://pennylane.readthedocs.io/en/stable/development/plugins.html

**Testing:**
Must pass PennyLane's device test suite

---

## Challenges & Solutions

### Challenge 1: Performance Overhead (JSON + subprocess)

**Problem:** IPC overhead ~1-10ms per circuit execution

**Solutions:**
1. **Short-term:** Batch multiple circuits in single JSON call
2. **Medium-term:** Implement pybind11 binding (Phase 5.4)
3. **Long-term:** Shared memory for large state vectors

**Mitigation:**
- For typical VQE (100-1000 iterations), 10ms overhead is acceptable
- For high-frequency calls, pybind11 reduces to ~0.01ms

### Challenge 2: State Management

**Problem:** Should device cache state between calls?

**Solution:**
- Phase 5.2: Stateless (recompute full circuit each time)
- Phase 5.4: Stateful (cache intermediate states)

**Tradeoff:**
- Stateless: Simple, no bugs, but slower
- Stateful: Fast, but complex state tracking

### Challenge 3: Observable Complexity

**Problem:** PennyLane supports complex observables (Hamiltonians, linear combinations)

**Solution:**
- Phase 5.2: Support single observables and tensor products
- Phase 5.4: Decompose Hamiltonians into individual terms
- Future: Native Hamiltonian support in LRET C++

### Challenge 4: Shot-Based Sampling

**Problem:** LRET computes exact expectations, not samples

**Solution:**
- Phase 5.2: Only support `shots=None` (exact mode)
- Phase 5.4: Implement sampling from final state
- Approach: Sample computational basis measurements from |L⟩ columns

### Challenge 5: Noise Model Integration

**Problem:** PennyLane has different noise model format than Qiskit

**Solution:**
- Phase 5.2: Support LRET's native noise model format
- Phase 5.4: Add PennyLane noise channel translation
- Future: Direct support for PennyLane noise channels

---

## Success Criteria

### Functional Requirements

- ✅ Execute all PennyLane operations listed in `Device.operations`
- ✅ Compute expectation values for all listed observables
- ✅ Pass PennyLane device test suite (95%+ tests)
- ✅ Gradient computation works with `diff_method="parameter-shift"`
- ✅ VQE example converges to correct ground state energy
- ✅ QAOA example finds optimal MaxCut solution

### Performance Requirements

- ✅ Circuit execution overhead < 10ms for small circuits (n<10)
- ✅ Gradient computation within 2x of finite difference time
- ✅ VQE convergence in < 5 minutes for H₂ molecule
- ✅ Match FDM fidelity (> 0.999 agreement)

### Quality Requirements

- ✅ Unit test coverage > 90%
- ✅ All examples run without errors
- ✅ Documentation complete (README + API docs)
- ✅ No PennyLane deprecation warnings
- ✅ Compatible with Python 3.8-3.11

### Adoption Requirements

- ✅ Published on PyPI
- ✅ Installable via `pip install pennylane-lret`
- ✅ Listed on PennyLane plugins page
- ✅ At least 2 working examples (VQE + QAOA)

---

## Timeline & Milestones

### Day 1-2: Foundation
- JSON schema design
- C++ JSON parser implementation
- Python bridge (subprocess)
- Round-trip test working

**Milestone:** Can execute simple circuit from Python → C++ → Python

### Day 3-5: PennyLane Integration
- LRETDevice class
- Gate/observable translation
- Unit tests passing
- Basic circuit execution working

**Milestone:** Can run PennyLane circuit on LRET device

### Day 6-7: Gradients & VQE
- Parameter-shift integration
- VQE example
- Gradient tests passing

**Milestone:** VQE converges to H₂ ground state

### Day 8: Optimization
- Circuit caching
- Noise model support
- Shot-based sampling (optional)
- Performance benchmarks

**Milestone:** Competitive performance with default.qubit

### Day 9: Publication
- setup.py and packaging
- Documentation
- PyPI upload
- Plugin registry submission

**Milestone:** Public release, installable via pip

### Day 10 (Buffer): Testing & Fixes
- PennyLane test suite
- Bug fixes
- Documentation improvements
- Community feedback

---

## Next Steps for Implementation

**Immediate Actions (Day 1):**

1. **Create Python package structure**
   ```bash
   mkdir pennylane_lret
   touch pennylane_lret/__init__.py
   touch pennylane_lret/device.py
   touch pennylane_lret/bridge.py
   ```

2. **Design JSON schema**
   - Document in `docs/json_schema.md`
   - Get feedback before implementing

3. **Implement C++ JSON interface**
   - `include/json_interface.h`
   - `src/json_interface.cpp`
   - Test with simple circuits

4. **Implement Python bridge**
   - `pennylane_lret/bridge.py`
   - Test subprocess communication

**Success Checkpoint (End of Day 2):**
- Can execute "H gate + Z measurement" from Python
- JSON round-trip working
- Bridge test passing

---

## Implementation Order Summary

```
Day 1-2:  JSON Interface + Bridge
  ├── JSON schema design
  ├── C++ JSON parser
  ├── C++ result serializer
  └── Python subprocess bridge

Day 3-5:  PennyLane Device
  ├── LRETDevice class
  ├── Gate translation
  ├── Observable translation
  └── Basic execution

Day 6-7:  Gradients & VQE
  ├── Parameter-shift support
  ├── VQE example
  └── Gradient tests

Day 8:    Optimization
  ├── Performance tuning
  ├── Noise model support
  └── Benchmarking

Day 9:    Publication
  ├── Packaging
  ├── Documentation
  └── PyPI release

Day 10:   Polish
  ├── Test suite
  ├── Bug fixes
  └── Community prep
```

---

## Conclusion

Phase 5 implementation is **well-scoped and achievable** in 9-10 days. The architecture is **modular** (can implement incrementally) and **non-invasive** (minimal changes to existing LRET code).

**Key Success Factors:**
1. **Start simple:** JSON + subprocess (not pybind11 initially)
2. **Incremental testing:** Test each component before moving on
3. **Leverage PennyLane:** Use built-in parameter-shift, don't reinvent
4. **Document thoroughly:** Good docs = community adoption

**Ready to begin implementation!**

---

**Strategic Recommendation:** Switch to **GPT-4.5 Codex Max** or **Claude Opus 4.5** for implementation phase. These models excel at:
- Writing production-quality code
- Handling complex API integrations
- Debugging subtle issues
- Maintaining code consistency

Use **Claude Sonnet 4.5** (current) for:
- Reviewing complex architectures
- Debugging performance issues
- Strategic planning for future phases

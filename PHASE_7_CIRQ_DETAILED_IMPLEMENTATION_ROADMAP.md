# Phase 7: Cirq Integration - Detailed Implementation Roadmap

**Document Purpose:** Complete step-by-step implementation guide for LRET-Cirq integration  
**Created:** January 26, 2026  
**Branch:** phase-7  
**Target Duration:** 5-6 days  
**Expected Outcome:** 50+ tests passing, production-ready Cirq simulator

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Architecture Overview](#2-architecture-overview)
3. [Core Components](#3-core-components)
4. [Gate Mapping Reference](#4-gate-mapping-reference)
5. [Qubit Mapping Strategy](#5-qubit-mapping-strategy)
6. [Measurement Handling](#6-measurement-handling)
7. [Implementation Guide](#7-implementation-guide)
8. [Testing Strategy](#8-testing-strategy)
9. [API Reference](#9-api-reference)
10. [Troubleshooting Guide](#10-troubleshooting-guide)
11. [Examples & Tutorials](#11-examples-and-tutorials)

---

## 1. Introduction

### 1.1 Project Overview

This roadmap provides comprehensive guidance for integrating LRET (Low-Rank Entanglement Tracking) as a native Cirq simulator backend. Google's Cirq framework is used by thousands of researchers and the quantum computing community, making this integration critical for LRET adoption.

**Strategic Value:**
- Access to Google Quantum AI ecosystem
- Integration with Cirq's extensive gate library
- Support for Google's quantum processors (Sycamore, Weber)
- Academic citations from Cirq research
- Simplified architecture (no Provider layer needed)

**Key Differences from Qiskit:**
- Simpler: No Provider/Backend/Job hierarchy
- Direct: `LRETSimulator().run(circuit)` 
- Flexible: Support for custom qubit types (LineQubit, GridQubit, NamedQubit)
- Powerful: Native support for power gates (XPowGate, YPowGate, ZPowGate)

### 1.2 Success Criteria

**Minimum Viable Product (MVP):**
- ✅ Users can `from lret_cirq import LRETSimulator`
- ✅ Basic circuits (H, X, Y, Z, CNOT) execute correctly
- ✅ Measurements return correct sample distributions
- ✅ Bell state produces 50/50 |00⟩/|11⟩ split
- ✅ 25+ tests passing

**Full Release:**
- ✅ 50+ tests passing (100% pass rate)
- ✅ All common gates supported (H, X, Y, Z, S, T, RX, RY, RZ, CNOT, CZ, SWAP)
- ✅ Power gates work (XPowGate, YPowGate, ZPowGate)
- ✅ All qubit types handled (LineQubit, GridQubit, NamedQubit)
- ✅ Measurement keys work correctly
- ✅ GHZ and QFT circuits validated
- ✅ Noise models integrate seamlessly
- ✅ Documentation complete with examples

**Stretch Goals:**
- ✅ Google-specific gates (FSim, Sycamore)
- ✅ Mid-circuit measurements
- ✅ Full `cirq.Simulator` interface (state vector access)
- ✅ Performance benchmarks (vs native Cirq)
- ✅ Advanced noise models (Kraus operators)

### 1.3 Comparison to Qiskit Integration

**What We've Learned from Qiskit (53 tests passing):**

| Aspect | Qiskit Experience | Cirq Application |
|--------|-------------------|------------------|
| **Architecture** | Provider→Backend→Job complex | Simpler: direct simulator |
| **Testing** | Incremental, comprehensive | Replicate test categories |
| **Gate Mapping** | Straightforward (discrete gates) | Complex (power gates) |
| **Qubit Types** | Integer indices | Multiple types need mapping |
| **Memory Issues** | Transpilation padding (20→4 qubits) | No transpilation, easier |
| **Documentation** | Critical for success | Must be equally thorough |

**Key Lessons Applied:**
1. **Test early and often** - Don't wait for complete implementation
2. **Start simple** - Basic gates first, add complexity gradually
3. **Handle edge cases** - Power gates, qubit mapping, measurement keys
4. **Document thoroughly** - Clear docstrings and examples from Day 1

### 1.4 Prerequisites

**Software Dependencies:**
```bash
# Core requirements
pip install cirq>=1.3.0
pip install cirq-core>=1.3.0
pip install numpy>=1.24.0
pip install scipy>=1.10.0

# For testing
pip install pytest>=7.4.0
pip install pytest-cov>=4.1.0

# LRET must be built
cd python/qlret
pip install -e .
```

**Knowledge Requirements:**
- Cirq simulator interfaces (`SimulatesSamples`, `Simulator`)
- Cirq circuit structure and moments
- Cirq qubit types (LineQubit, GridQubit, NamedQubit)
- Cirq gate library (especially power gates)
- LRET JSON format (from Qiskit experience)
- Python pybind11 (for native module integration)

**Reference Documentation:**
- https://quantumai.google/cirq/simulate/custom_simulators
- https://quantumai.google/cirq/simulate
- https://quantumai.google/reference/python/cirq/Simulator
- https://github.com/quantumlib/Cirq/tree/master/cirq-core/cirq/sim

---

## 2. Architecture Overview

### 2.1 System Design

```
┌─────────────────────────────────────────────────────────────────┐
│                         Cirq User Code                           │
│                                                                  │
│  import cirq                                                     │
│  from lret_cirq import LRETSimulator                            │
│                                                                  │
│  # Create circuit                                               │
│  q0, q1 = cirq.LineQubit.range(2)                              │
│  circuit = cirq.Circuit(                                        │
│      cirq.H(q0),                                                │
│      cirq.CNOT(q0, q1),                                         │
│      cirq.measure(q0, q1, key='result')                         │
│  )                                                              │
│                                                                  │
│  # Run simulation                                               │
│  sim = LRETSimulator(epsilon=1e-4)                             │
│  result = sim.run(circuit, repetitions=1000)                   │
│                                                                  │
│  # Get results                                                  │
│  counts = result.histogram(key='result')                        │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LRETSimulator                                 │
│            (inherits cirq.SimulatesSamples)                     │
│                                                                  │
│  • _run(circuit, repetitions) - Main entry point               │
│  • _epsilon - Truncation threshold                             │
│  • _noise_model - Optional noise configuration                  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                   CircuitTranslator                             │
│                                                                  │
│  Input:  cirq.Circuit                                           │
│  Output: LRET JSON dict                                         │
│                                                                  │
│  Steps:                                                         │
│  1. Extract all qubits → build qubit map                       │
│  2. Iterate through moments                                     │
│  3. Translate each operation                                    │
│     - Handle gates: H, X, Y, Z, S, T, RX, RY, RZ, ...         │
│     - Handle power gates: XPowGate(e), YPowGate(e), ...       │
│     - Handle measurements: extract keys                         │
│  4. Build LRET JSON structure                                   │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                   LRET Execution                                 │
│               (qlret.simulate_json)                             │
│                                                                  │
│  Input:  LRET JSON                                              │
│  Output: LRET result dict                                       │
│                                                                  │
│  {                                                              │
│    "samples": [[0, 0], [1, 1], [0, 0], ...],  # N × M array   │
│    "counts": {"00": 502, "11": 498},                           │
│    "state": [...],  # Optional final state                     │
│    "rank": 2,                                                   │
│    "fidelity": 0.9999                                          │
│  }                                                              │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                   ResultConverter                                │
│                                                                  │
│  Input:  LRET result dict + circuit metadata                    │
│  Output: cirq.Result                                            │
│                                                                  │
│  Steps:                                                         │
│  1. Extract samples from LRET result                            │
│  2. Map integer indices → original cirq qubits                  │
│  3. Organize samples by measurement key                         │
│  4. Build cirq.Result with measurements dict                    │
│     {                                                           │
│       'result': np.array([[0, 0], [1, 1], ...]),  # N × 2      │
│       'aux': np.array([[1], [0], ...])             # N × 1      │
│     }                                                           │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      cirq.Result                                 │
│                                                                  │
│  User can:                                                      │
│  • result.histogram(key='result')  → Counter({0: 502, 3: 498}) │
│  • result.measurements['result']   → np.array([[0,0],[1,1]...])│
│  • len(result.measurements)        → number of keys             │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow

**Step-by-Step Execution:**

1. **User creates circuit**
   ```python
   q0, q1 = cirq.LineQubit.range(2)
   circuit = cirq.Circuit(
       cirq.H(q0),
       cirq.CNOT(q0, q1),
       cirq.measure(q0, q1, key='m')
   )
   ```

2. **Simulator._run() called**
   ```python
   sim = LRETSimulator()
   result = sim.run(circuit, repetitions=100)
   ```

3. **Circuit translation** (Cirq → LRET JSON)
   ```python
   translator = CircuitTranslator()
   lret_json = translator.translate(circuit, epsilon=1e-4)
   # Result:
   # {
   #   "config": {"epsilon": 0.0001},
   #   "circuit": {
   #     "num_qubits": 2,
   #     "operations": [
   #       {"name": "H", "wires": [0]},
   #       {"name": "CX", "wires": [0, 1]},
   #       {"name": "measure", "wires": [0, 1]}
   #     ]
   #   },
   #   "shots": 100
   # }
   ```

4. **LRET execution**
   ```python
   from qlret import simulate_json
   lret_result = simulate_json(lret_json)
   # Result:
   # {
   #   "samples": [[0, 0], [1, 1], [0, 0], ...],  # 100 samples
   #   "counts": {"00": 48, "11": 52},
   #   "rank": 2
   # }
   ```

5. **Result conversion** (LRET → Cirq)
   ```python
   converter = ResultConverter()
   cirq_result = converter.convert(
       lret_result,
       circuit,
       qubit_map={q0: 0, q1: 1},
       measurement_keys=['m']
   )
   ```

6. **User accesses results**
   ```python
   counts = result.histogram(key='m')
   # Counter({0: 48, 3: 52})  # Binary: 00 and 11
   
   samples = result.measurements['m']
   # np.array([[0, 0], [1, 1], [0, 0], ...])  # 100 × 2
   ```

### 2.3 Component Relationships

```
                    ┌─────────────────┐
                    │ LRETSimulator   │
                    │                 │
                    │ • epsilon       │
                    │ • noise_model   │
                    │ • _run()        │
                    └────────┬────────┘
                             │
                             │ uses
                             │
            ┌────────────────┼────────────────┐
            │                                 │
            ▼                                 ▼
    ┌──────────────┐                 ┌──────────────┐
    │ Circuit      │                 │ Result       │
    │ Translator   │                 │ Converter    │
    │              │                 │              │
    │ • translate()│                 │ • convert()  │
    └──────┬───────┘                 └──────▲───────┘
           │                                 │
           │ creates                         │ creates
           │                                 │
           ▼                                 │
    ┌──────────────┐                        │
    │ LRET JSON    │───────executes─────────┤
    │              │                        │
    │ • config     │    qlret.simulate_json │
    │ • circuit    │                        │
    │ • shots      │                        │
    └──────────────┘                        │
                                            │
                                     ┌──────────────┐
                                     │ LRET Result  │
                                     │              │
                                     │ • samples    │
                                     │ • counts     │
                                     │ • rank       │
                                     └──────────────┘
```

### 2.4 Comparison: Cirq vs Qiskit vs LRET

| Feature | Qiskit | Cirq | LRET (Target) |
|---------|--------|------|---------------|
| **Entry Point** | `Provider.get_backend()` | `Simulator()` | `LRETSimulator()` |
| **Circuit Type** | `QuantumCircuit` | `cirq.Circuit` | LRET JSON |
| **Qubits** | Integers (0, 1, 2) | Objects (LineQubit, GridQubit) | Integers |
| **Gates** | Discrete (H, RZ(θ)) | Power gates (XPowGate(e)) | Discrete + params |
| **Execution** | `backend.run(circuit, shots)` | `sim.run(circuit, repetitions)` | `simulate_json()` |
| **Result Access** | `result.get_counts()` | `result.histogram(key)` | `["counts"]` dict |
| **Job System** | Yes (JobV1) | No | No |
| **Provider** | Yes (ProviderV1) | No | No |
| **Noise** | NoiseModel class | Channel operations | JSON config |
| **Complexity** | High (3 layers) | Low (1 layer) | Lowest (function call) |

**Design Philosophy:**
- **Qiskit**: Enterprise-grade, full abstraction layers
- **Cirq**: Research-friendly, direct access
- **LRET**: Performance-focused, minimal overhead

---

## 3. Core Components

### 3.1 LRETSimulator Class

**File:** `python/lret_cirq/lret_simulator.py`

**Purpose:** Main entry point for users. Inherits from `cirq.SimulatesSamples` to provide standard Cirq simulator interface.

**Complete Implementation:**

```python
"""
LRET Simulator for Cirq
========================

Provides a Cirq-compatible simulator interface backed by LRET's
low-rank quantum simulation engine.
"""

import cirq
import numpy as np
from typing import Dict, List, Optional, Any, Union
import warnings

from .translators.circuit_translator import CircuitTranslator, TranslationError
from .translators.result_converter import ResultConverter

try:
    from qlret import simulate_json
    NATIVE_AVAILABLE = True
except ImportError:
    NATIVE_AVAILABLE = False
    warnings.warn(
        "LRET native module not available. "
        "Install with: cd python/qlret && pip install -e ."
    )


class LRETSimulator(cirq.SimulatesSamples):
    """
    LRET-backed quantum circuit simulator for Cirq.
    
    This simulator uses low-rank approximations to efficiently simulate
    noisy quantum circuits with automatic rank adaptation.
    
    Attributes:
        epsilon (float): SVD truncation threshold (default: 1e-4)
        noise_model (Optional[dict]): LRET noise configuration
        seed (Optional[int]): Random seed for reproducibility
    
    Example:
        >>> import cirq
        >>> from lret_cirq import LRETSimulator
        >>> 
        >>> q0, q1 = cirq.LineQubit.range(2)
        >>> circuit = cirq.Circuit(
        ...     cirq.H(q0),
        ...     cirq.CNOT(q0, q1),
        ...     cirq.measure(q0, q1, key='result')
        ... )
        >>> 
        >>> sim = LRETSimulator(epsilon=1e-4)
        >>> result = sim.run(circuit, repetitions=1000)
        >>> print(result.histogram(key='result'))
        Counter({0: 489, 3: 511})  # ~50/50 split for Bell state
    """
    
    def __init__(
        self,
        epsilon: float = 1e-4,
        noise_model: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize LRET simulator.
        
        Args:
            epsilon: Truncation threshold for low-rank approximation.
                     Smaller values = higher accuracy but slower.
                     Typical range: 1e-3 (fast) to 1e-6 (accurate)
            noise_model: Optional LRET noise configuration dict
            seed: Random seed for measurement sampling
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(**kwargs)
        
        if not NATIVE_AVAILABLE:
            raise ImportError(
                "LRET native module required for simulation. "
                "Build with: cd python/qlret && pip install -e ."
            )
        
        self._epsilon = epsilon
        self._noise_model = noise_model
        self._seed = seed
        
        # Create translator and converter instances
        self._translator = CircuitTranslator()
        self._converter = ResultConverter()
    
    @property
    def epsilon(self) -> float:
        """Get the current truncation threshold."""
        return self._epsilon
    
    @epsilon.setter
    def epsilon(self, value: float):
        """Set the truncation threshold."""
        if value <= 0 or value >= 1:
            raise ValueError(f"Epsilon must be in (0, 1), got {value}")
        self._epsilon = value
    
    def _run(
        self,
        circuit: cirq.Circuit,
        repetitions: int,
        param_resolver: 'cirq.ParamResolver' = None
    ) -> Dict[str, np.ndarray]:
        """
        Run the circuit and return measurement results.
        
        This is the main interface method required by cirq.SimulatesSamples.
        
        Args:
            circuit: The Cirq circuit to simulate
            repetitions: Number of measurement samples to take
            param_resolver: Parameter values for parameterized circuits
        
        Returns:
            Dict mapping measurement keys to sample arrays
            
        Raises:
            TranslationError: If circuit contains unsupported operations
            ValueError: If circuit has unresolved parameters
        """
        # Resolve parameters if provided
        if param_resolver:
            circuit = cirq.resolve_parameters(circuit, param_resolver)
        
        # Check for unresolved parameters
        if cirq.is_parameterized(circuit):
            raise ValueError(
                "Circuit contains unresolved parameters. "
                "Provide a param_resolver or use cirq.resolve_parameters()."
            )
        
        # Translate circuit to LRET JSON
        try:
            lret_json = self._translator.translate(
                circuit=circuit,
                epsilon=self._epsilon,
                shots=repetitions,
                noise_model=self._noise_model,
                seed=self._seed
            )
        except TranslationError as e:
            raise TranslationError(f"Circuit translation failed: {e}")
        
        # Execute simulation
        try:
            lret_result = simulate_json(lret_json)
        except Exception as e:
            raise RuntimeError(f"LRET simulation failed: {e}")
        
        # Convert result back to Cirq format
        measurements_dict = self._converter.convert(
            lret_result=lret_result,
            circuit=circuit,
            qubit_map=self._translator.get_qubit_map()
        )
        
        return measurements_dict
    
    def __str__(self) -> str:
        """String representation."""
        return f"LRETSimulator(epsilon={self._epsilon})"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return (
            f"LRETSimulator("
            f"epsilon={self._epsilon}, "
            f"noise_model={'Yes' if self._noise_model else 'None'}, "
            f"seed={self._seed})"
        )
```

**Key Design Decisions:**

1. **Base Class Choice:** `SimulatesSamples` instead of `Simulator`
   - **Rationale:** Simpler interface, focuses on measurements (not state vectors)
   - **Trade-off:** Cannot access intermediate states
   - **Future:** Can add `Simulator` interface later for state access

2. **Error Handling:** Three layers
   - Translation errors → `TranslationError`
   - Execution errors → `RuntimeError`
   - Parameter errors → `ValueError`

3. **Parameter Resolution:** Required before translation
   - Cirq uses `sympy.Symbol` for parameters
   - LRET expects numeric values only
   - Check with `cirq.is_parameterized()`

4. **Seed Management:** Passed to LRET for reproducibility
   - Cirq doesn't enforce seeding
   - LRET needs it for consistent sampling

### 3.2 CircuitTranslator

**File:** `python/lret_cirq/translators/circuit_translator.py`

**Purpose:** Convert `cirq.Circuit` to LRET JSON format.

**Complete Implementation:**

```python
"""
Circuit Translator: Cirq → LRET JSON
=====================================

Converts Cirq circuits to LRET's JSON input format.
"""

import cirq
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import warnings


class TranslationError(Exception):
    """Raised when circuit translation fails."""
    pass


class CircuitTranslator:
    """
    Translates Cirq circuits to LRET JSON format.
    
    Handles:
    - Qubit mapping (LineQubit, GridQubit, NamedQubit → integers)
    - Gate translation (including power gates)
    - Measurement extraction
    - Parameter validation
    """
    
    def __init__(self):
        """Initialize translator with gate mapping."""
        self._gate_map = self._build_gate_map()
        self._qubit_map = {}
        self._measurement_keys = []
    
    def get_qubit_map(self) -> Dict[cirq.Qid, int]:
        """Get the current qubit mapping."""
        return self._qubit_map.copy()
    
    def get_measurement_keys(self) -> List[str]:
        """Get list of measurement keys in order."""
        return self._measurement_keys.copy()
    
    def translate(
        self,
        circuit: cirq.Circuit,
        epsilon: float = 1e-4,
        shots: int = 1024,
        noise_model: Optional[Dict] = None,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Translate Cirq circuit to LRET JSON.
        
        Args:
            circuit: Cirq circuit to translate
            epsilon: Truncation threshold
            shots: Number of measurement samples
            noise_model: Optional noise configuration
            seed: Random seed
            
        Returns:
            LRET JSON dict
            
        Raises:
            TranslationError: If circuit contains unsupported operations
        """
        # Reset state
        self._qubit_map = {}
        self._measurement_keys = []
        
        # Build qubit mapping
        self._qubit_map = self._build_qubit_map(circuit)
        num_qubits = len(self._qubit_map)
        
        if num_qubits == 0:
            raise TranslationError("Circuit has no qubits")
        
        if num_qubits > 28:
            warnings.warn(
                f"Circuit has {num_qubits} qubits. "
                "LRET performance may degrade above 20 qubits."
            )
        
        # Translate operations
        operations = []
        for moment in circuit:
            for op in moment:
                try:
                    lret_ops = self._translate_operation(op)
                    operations.extend(lret_ops)
                except Exception as e:
                    raise TranslationError(
                        f"Failed to translate operation {op}: {e}"
                    )
        
        # Build LRET JSON
        lret_json = {
            "config": {
                "epsilon": epsilon
            },
            "circuit": {
                "num_qubits": num_qubits,
                "operations": operations
            },
            "shots": shots
        }
        
        # Add noise model if provided
        if noise_model:
            lret_json["noise"] = noise_model
        
        # Add seed if provided
        if seed is not None:
            lret_json["config"]["seed"] = seed
        
        return lret_json
    
    def _build_qubit_map(self, circuit: cirq.Circuit) -> Dict[cirq.Qid, int]:
        """
        Create mapping from Cirq qubits to integer indices.
        
        Handles all Cirq qubit types:
        - LineQubit(x) → sorted by x
        - GridQubit(row, col) → sorted by (row, col)
        - NamedQubit(name) → sorted by name
        - Mixed types → sorted by type then value
        
        Args:
            circuit: Cirq circuit
            
        Returns:
            Dict mapping cirq.Qid to int index
        """
        qubits = sorted(circuit.all_qubits())
        return {qubit: idx for idx, qubit in enumerate(qubits)}
    
    def _translate_operation(self, op: cirq.Operation) -> List[Dict[str, Any]]:
        """
        Translate a single Cirq operation to LRET format.
        
        Returns a list because some gates may decompose into multiple ops.
        
        Args:
            op: Cirq operation
            
        Returns:
            List of LRET operation dicts
            
        Raises:
            TranslationError: If operation not supported
        """
        gate = op.gate
        qubits = [self._qubit_map[q] for q in op.qubits]
        
        # Handle measurements specially
        if isinstance(gate, cirq.MeasurementGate):
            return self._translate_measurement(op)
        
        # Handle identity (no-op)
        if isinstance(gate, cirq.IdentityGate):
            return []  # Skip identity gates
        
        # Try standard gate mapping
        if type(gate) in self._gate_map:
            gate_info = self._gate_map[type(gate)]
            return [{
                "name": gate_info["lret_name"],
                "wires": qubits
            }]
        
        # Handle parameterized gates
        if isinstance(gate, (cirq.XPowGate, cirq.YPowGate, cirq.ZPowGate)):
            return self._translate_power_gate(gate, qubits)
        
        if isinstance(gate, (cirq.Rx, cirq.Ry, cirq.Rz)):
            return self._translate_rotation_gate(gate, qubits)
        
        # Unsupported gate
        raise TranslationError(
            f"Unsupported gate: {gate}. "
            f"Supported gates: {list(self._gate_map.keys())}"
        )
    
    def _translate_measurement(self, op: cirq.Operation) -> List[Dict[str, Any]]:
        """
        Translate measurement operation.
        
        Extracts measurement key and qubit indices.
        """
        gate = op.gate
        qubits = [self._qubit_map[q] for q in op.qubits]
        
        # Extract measurement key
        key = str(gate.key) if hasattr(gate, 'key') else 'measure'
        
        if key not in self._measurement_keys:
            self._measurement_keys.append(key)
        
        # LRET measurement format
        return [{
            "name": "measure",
            "wires": qubits,
            "key": key
        }]
    
    def _translate_power_gate(
        self,
        gate: Union[cirq.XPowGate, cirq.YPowGate, cirq.ZPowGate],
        qubits: List[int]
    ) -> List[Dict[str, Any]]:
        """
        Translate power gates (XPowGate, YPowGate, ZPowGate).
        
        Strategy:
        - Exponent = 1.0 → discrete gate (X, Y, Z)
        - Exponent = 0.5 → special gate (SX, SY, S)
        - Other → rotation gate (RX, RY, RZ)
        """
        exponent = gate.exponent
        
        if isinstance(gate, cirq.XPowGate):
            base_name = "X"
            rotation_name = "RX"
        elif isinstance(gate, cirq.YPowGate):
            base_name = "Y"
            rotation_name = "RY"
        elif isinstance(gate, cirq.ZPowGate):
            base_name = "Z"
            rotation_name = "RZ"
        else:
            raise TranslationError(f"Unknown power gate: {gate}")
        
        # Handle common cases
        if np.isclose(exponent, 1.0):
            # Full rotation = discrete gate
            return [{"name": base_name, "wires": qubits}]
        
        elif np.isclose(exponent, 0.5):
            # Half rotation
            if base_name == "X":
                return [{"name": "SX", "wires": qubits}]
            elif base_name == "Z":
                return [{"name": "S", "wires": qubits}]
            else:
                # No standard SY gate, use rotation
                angle = exponent * np.pi
                return [{"name": rotation_name, "wires": qubits, "params": [angle]}]
        
        elif np.isclose(exponent, 0.25):
            # Quarter rotation
            if base_name == "Z":
                return [{"name": "T", "wires": qubits}]
            else:
                angle = exponent * np.pi
                return [{"name": rotation_name, "wires": qubits, "params": [angle]}]
        
        else:
            # General case: convert to rotation
            angle = exponent * np.pi
            return [{"name": rotation_name, "wires": qubits, "params": [angle]}]
    
    def _translate_rotation_gate(
        self,
        gate: Union[cirq.Rx, cirq.Ry, cirq.Rz],
        qubits: List[int]
    ) -> List[Dict[str, Any]]:
        """Translate rotation gates (Rx, Ry, Rz)."""
        # Get rotation angle (in radians)
        if hasattr(gate, 'rads'):
            angle = float(gate.rads)
        elif hasattr(gate, '_rads'):
            angle = float(gate._rads)
        else:
            raise TranslationError(f"Cannot extract angle from {gate}")
        
        # Map to LRET rotation gates
        if isinstance(gate, cirq.Rx):
            name = "RX"
        elif isinstance(gate, cirq.Ry):
            name = "RY"
        elif isinstance(gate, cirq.Rz):
            name = "RZ"
        else:
            raise TranslationError(f"Unknown rotation gate: {gate}")
        
        return [{
            "name": name,
            "wires": qubits,
            "params": [angle]
        }]
    
    def _build_gate_map(self) -> Dict[type, Dict[str, Any]]:
        """
        Build mapping from Cirq gates to LRET gates.
        
        Returns:
            Dict mapping gate class to LRET info
        """
        return {
            # Pauli gates
            cirq.XGate: {"lret_name": "X"},
            cirq.YGate: {"lret_name": "Y"},
            cirq.ZGate: {"lret_name": "Z"},
            
            # Hadamard
            cirq.HGate: {"lret_name": "H"},
            
            # Phase gates
            cirq.SGate: {"lret_name": "S"},
            cirq.SXGate: {"lret_name": "SX"},
            cirq.TGate: {"lret_name": "T"},
            
            # Dagger gates
            cirq.S**-1: {"lret_name": "SDG"},
            cirq.T**-1: {"lret_name": "TDG"},
            
            # Two-qubit gates
            cirq.CXGate: {"lret_name": "CX"},
            cirq.CNotGate: {"lret_name": "CX"},  # Alias
            cirq.CZGate: {"lret_name": "CZ"},
            cirq.SwapGate: {"lret_name": "SWAP"},
            
            # Identity
            cirq.IdentityGate: {"lret_name": "I"},
        }
```

**Key Features:**

1. **Qubit Mapping:**
   - Handles all Cirq qubit types
   - Sorts qubits consistently
   - Maps to contiguous integers [0, N-1]

2. **Power Gate Handling:**
   - Exponent = 1.0 → discrete gate
   - Exponent = 0.5 → √gate (S, SX)
   - Exponent = 0.25 → T gate
   - Other → rotation (RX, RY, RZ)

3. **Measurement Tracking:**
   - Extracts measurement keys
   - Stores in order for result conversion
   - Supports multi-qubit measurements

4. **Error Messages:**
   - Clear indication of unsupported gates
   - Lists all supported gates
   - Helpful for debugging

### 3.3 ResultConverter

**File:** `python/lret_cirq/translators/result_converter.py`

**Purpose:** Convert LRET JSON result to `cirq.Result` object.

**Complete Implementation:**

```python
"""
Result Converter: LRET → Cirq Result
====================================

Converts LRET JSON results to Cirq Result objects.
"""

import cirq
import numpy as np
from typing import Dict, List, Any


class ResultConverter:
    """
    Converts LRET results to Cirq Result format.
    
    Handles:
    - Sample array conversion
    - Qubit index mapping (int → cirq.Qid)
    - Measurement key organization
    - Multiple measurement support
    """
    
    def __init__(self):
        """Initialize converter."""
        pass
    
    def convert(
        self,
        lret_result: Dict[str, Any],
        circuit: cirq.Circuit,
        qubit_map: Dict[cirq.Qid, int]
    ) -> Dict[str, np.ndarray]:
        """
        Convert LRET result to Cirq measurements dict.
        
        Args:
            lret_result: LRET simulation output
            circuit: Original Cirq circuit
            qubit_map: Mapping from cirq.Qid to int index
            
        Returns:
            Dict mapping measurement keys to sample arrays
        """
        # Extract samples from LRET result
        if "samples" not in lret_result:
            raise ValueError("LRET result missing 'samples' field")
        
        samples = np.array(lret_result["samples"])  # Shape: (repetitions, num_qubits)
        
        if samples.ndim != 2:
            raise ValueError(f"Expected 2D samples array, got shape {samples.shape}")
        
        # Extract measurements from circuit
        measurements = self._extract_measurements(circuit)
        
        if not measurements:
            # No explicit measurements, measure all qubits
            reverse_qubit_map = {v: k for k, v in qubit_map.items()}
            all_qubits = [reverse_qubit_map[i] for i in range(len(qubit_map))]
            measurements = [('result', all_qubits)]
        
        # Build measurements dict
        measurements_dict = {}
        col_offset = 0
        
        for key, qubits in measurements:
            # Get columns for these qubits
            qubit_indices = [qubit_map[q] for q in qubits]
            num_cols = len(qubit_indices)
            
            # Extract relevant columns
            measurement_samples = samples[:, qubit_indices]
            
            # Store in dict
            measurements_dict[key] = measurement_samples
            
            col_offset += num_cols
        
        return measurements_dict
    
    def _extract_measurements(
        self,
        circuit: cirq.Circuit
    ) -> List[Tuple[str, List[cirq.Qid]]]:
        """
        Extract measurement operations from circuit.
        
        Returns:
            List of (key, qubits) tuples
        """
        measurements = []
        
        for moment in circuit:
            for op in moment:
                if isinstance(op.gate, cirq.MeasurementGate):
                    key = str(op.gate.key)
                    qubits = list(op.qubits)
                    measurements.append((key, qubits))
        
        return measurements
```

**Key Features:**

1. **Sample Organization:**
   - LRET returns: `[[0, 0], [1, 1], ...]` (all qubits)
   - Cirq expects: Split by measurement key
   - Handles multi-qubit measurements

2. **Index Mapping:**
   - Converts integer indices back to Cirq qubits
   - Preserves qubit order
   - Handles missing measurements

3. **Default Behavior:**
   - If no measurements in circuit, measure all qubits
   - Use default key 'result'
   - Matches Cirq convention

---

## 4. Gate Mapping Reference

### 4.1 Supported Gates

**Single-Qubit Gates:**

| Cirq Gate | LRET Gate | Description | Implementation |
|-----------|-----------|-------------|----------------|
| `cirq.I(q)` | `I` | Identity (no-op) | Skip in translation |
| `cirq.X(q)` | `X` | Pauli-X (NOT) | Direct mapping |
| `cirq.Y(q)` | `Y` | Pauli-Y | Direct mapping |
| `cirq.Z(q)` | `Z` | Pauli-Z | Direct mapping |
| `cirq.H(q)` | `H` | Hadamard | Direct mapping |
| `cirq.S(q)` | `S` | S gate (√Z) | Direct mapping |
| `cirq.T(q)` | `T` | T gate (∜Z) | Direct mapping |
| `cirq.S**-1(q)` | `SDG` | S-dagger | Power gate with exp=-1 |
| `cirq.T**-1(q)` | `TDG` | T-dagger | Power gate with exp=-1 |
| `cirq.rx(θ)(q)` | `RX` | X-rotation | Extract angle in radians |
| `cirq.ry(θ)(q)` | `RY` | Y-rotation | Extract angle in radians |
| `cirq.rz(θ)(q)` | `RZ` | Z-rotation | Extract angle in radians |
| `cirq.XPowGate(e)(q)` | varies | X^e | See power gate table below |
| `cirq.YPowGate(e)(q)` | varies | Y^e | See power gate table below |
| `cirq.ZPowGate(e)(q)` | varies | Z^e | See power gate table below |

**Two-Qubit Gates:**

| Cirq Gate | LRET Gate | Description | Implementation |
|-----------|-----------|-------------|----------------|
| `cirq.CNOT(c, t)` | `CX` | Controlled-NOT | Direct mapping |
| `cirq.CX(c, t)` | `CX` | Alias for CNOT | Direct mapping |
| `cirq.CZ(c, t)` | `CZ` | Controlled-Z | Direct mapping |
| `cirq.SWAP(q0, q1)` | `SWAP` | SWAP gate | Direct mapping |
| `cirq.ISWAP(q0, q1)` | Decompose | iSWAP gate | Decompose to CZ + RZ |

**Power Gate Mapping:**

| Cirq | Exponent | LRET Gate | Notes |
|------|----------|-----------|-------|
| `XPowGate(e)` | 1.0 | `X` | Full rotation |
| `XPowGate(e)` | 0.5 | `SX` | √X gate |
| `XPowGate(e)` | other | `RX(e*π)` | Convert to rotation |
| `YPowGate(e)` | 1.0 | `Y` | Full rotation |
| `YPowGate(e)` | other | `RY(e*π)` | Convert to rotation |
| `ZPowGate(e)` | 1.0 | `Z` | Full rotation |
| `ZPowGate(e)` | 0.5 | `S` | √Z gate |
| `ZPowGate(e)` | 0.25 | `T` | ∜Z gate |
| `ZPowGate(e)` | -0.5 | `SDG` | S-dagger |
| `ZPowGate(e)` | -0.25 | `TDG` | T-dagger |
| `ZPowGate(e)` | other | `RZ(e*π)` | Convert to rotation |

### 4.2 Unsupported Gates (Future Work)

**Google-Specific Gates:**

| Gate | Status | Workaround |
|------|--------|------------|
| `cirq.FSimGate` | ⚠️ Not yet | Decompose to CZ + RZ |
| `cirq.SycamoreGate` | ⚠️ Not yet | Decompose |
| `cirq.ISwapPowGate` | ⚠️ Not yet | Decompose to ISWAP + phase |

**Three-Qubit Gates:**

| Gate | Status | Workaround |
|------|--------|------------|
| `cirq.TOFFOLI` | ⚠️ Not yet | Decompose to CNOT + T gates |
| `cirq.CCX` | ⚠️ Not yet | Alias for TOFFOLI |
| `cirq.FREDKIN` | ⚠️ Not yet | Decompose |

**Controlled Gates:**

| Gate | Status | Workaround |
|------|--------|------------|
| `cirq.ControlledGate(X)` | ⚠️ Partial | CX only for now |
| `cirq.ControlledGate(other)` | ⚠️ Not yet | Manual decomposition |

### 4.3 Gate Translation Examples

**Example 1: Basic Gates**

```python
import cirq
from lret_cirq import LRETSimulator

q0 = cirq.LineQubit(0)

# Cirq circuit
circuit = cirq.Circuit(
    cirq.H(q0),
    cirq.X(q0),
    cirq.measure(q0, key='m')
)

# Translates to:
# {
#   "circuit": {
#     "num_qubits": 1,
#     "operations": [
#       {"name": "H", "wires": [0]},
#       {"name": "X", "wires": [0]},
#       {"name": "measure", "wires": [0], "key": "m"}
#     ]
#   }
# }
```

**Example 2: Power Gates**

```python
import cirq

q0 = cirq.LineQubit(0)

# Full X rotation
circuit1 = cirq.Circuit(cirq.XPowGate(exponent=1.0)(q0))
# → {"name": "X", "wires": [0]}

# √X gate
circuit2 = cirq.Circuit(cirq.XPowGate(exponent=0.5)(q0))
# → {"name": "SX", "wires": [0]}

# Custom rotation
circuit3 = cirq.Circuit(cirq.XPowGate(exponent=0.3)(q0))
# → {"name": "RX", "wires": [0], "params": [0.942...]}  # 0.3 * π
```

**Example 3: Multi-Qubit Circuit**

```python
import cirq

q0, q1, q2 = cirq.LineQubit.range(3)

circuit = cirq.Circuit(
    cirq.H(q0),
    cirq.CNOT(q0, q1),
    cirq.CNOT(q1, q2),
    cirq.measure(q0, q1, q2, key='result')
)

# Translates to GHZ state preparation
# {
#   "circuit": {
#     "num_qubits": 3,
#     "operations": [
#       {"name": "H", "wires": [0]},
#       {"name": "CX", "wires": [0, 1]},
#       {"name": "CX", "wires": [1, 2]},
#       {"name": "measure", "wires": [0, 1, 2], "key": "result"}
#     ]
#   }
# }
```

---

## 5. Qubit Mapping Strategy

### 5.1 Qubit Types in Cirq

**LineQubit:**
```python
q0 = cirq.LineQubit(0)
q1 = cirq.LineQubit(1)
q2 = cirq.LineQubit(2)

# Or
qubits = cirq.LineQubit.range(10)  # q0, q1, ..., q9
```

**GridQubit:**
```python
q00 = cirq.GridQubit(0, 0)
q01 = cirq.GridQubit(0, 1)
q10 = cirq.GridQubit(1, 0)

# Or
qubits = cirq.GridQubit.rect(3, 4)  # 3 rows × 4 columns
```

**NamedQubit:**
```python
alice = cirq.NamedQubit("alice")
bob = cirq.NamedQubit("bob")
```

### 5.2 Mapping Algorithm

**Goal:** Convert Cirq qubits → contiguous integers [0, N-1]

**Algorithm:**

1. **Extract all qubits:**
   ```python
   qubits = circuit.all_qubits()
   ```

2. **Sort consistently:**
   ```python
   sorted_qubits = sorted(qubits)
   ```
   
   Cirq's default sorting:
   - LineQubit: by index
   - GridQubit: by (row, col)
   - NamedQubit: by name (alphabetical)
   - Mixed types: by type priority

3. **Create mapping:**
   ```python
   qubit_map = {q: i for i, q in enumerate(sorted_qubits)}
   ```

4. **Use in translation:**
   ```python
   for op in circuit:
       cirq_qubits = op.qubits
       lret_indices = [qubit_map[q] for q in cirq_qubits]
   ```

### 5.3 Examples

**Example 1: LineQubit**
```python
q1 = cirq.LineQubit(1)
q0 = cirq.LineQubit(0)
q2 = cirq.LineQubit(2)

circuit = cirq.Circuit(cirq.H(q1))

qubits = sorted([q0, q1, q2])  # [LineQubit(0), LineQubit(1), LineQubit(2)]
qubit_map = {
    cirq.LineQubit(0): 0,
    cirq.LineQubit(1): 1,
    cirq.LineQubit(2): 2
}
```

**Example 2: GridQubit**
```python
q00 = cirq.GridQubit(0, 0)
q01 = cirq.GridQubit(0, 1)
q10 = cirq.GridQubit(1, 0)

circuit = cirq.Circuit(
    cirq.H(q00),
    cirq.CNOT(q00, q01)
)

qubits = sorted([q00, q01, q10])
# Order: (0,0), (0,1), (1,0)
qubit_map = {
    cirq.GridQubit(0, 0): 0,
    cirq.GridQubit(0, 1): 1,
    cirq.GridQubit(1, 0): 2
}
```

**Example 3: Mixed Types**
```python
q0 = cirq.LineQubit(0)
alice = cirq.NamedQubit("alice")

circuit = cirq.Circuit(cirq.CNOT(q0, alice))

# Cirq sorts by type first: LineQubit < NamedQubit
qubit_map = {
    cirq.LineQubit(0): 0,
    cirq.NamedQubit("alice"): 1
}
```

### 5.4 Reverse Mapping (for Results)

When converting results back to Cirq format:

```python
reverse_map = {v: k for k, v in qubit_map.items()}

# LRET result: measurement on qubit 0, 1
lret_qubits = [0, 1]

# Convert back
cirq_qubits = [reverse_map[i] for i in lret_qubits]
# [cirq.LineQubit(0), cirq.LineQubit(1)]
```

---

## 6. Measurement Handling

### 6.1 Cirq Measurement Model

**Key Concept:** Measurements in Cirq are associated with string keys.

```python
q0, q1 = cirq.LineQubit.range(2)

circuit = cirq.Circuit(
    cirq.H(q0),
    cirq.measure(q0, key='first'),
    cirq.CNOT(q0, q1),
    cirq.measure(q0, q1, key='final')
)

result = sim.run(circuit, repetitions=100)

# Access by key
first_measurements = result.measurements['first']  # Shape: (100, 1)
final_measurements = result.measurements['final']  # Shape: (100, 2)

# Histogram
first_counts = result.histogram(key='first')  # Counter({0: 48, 1: 52})
final_counts = result.histogram(key='final')  # Counter({0: 24, 1: 26, 2: 25, 3: 25})
```

### 6.2 LRET Measurement Format

LRET returns all measurements together:

```json
{
  "samples": [
    [0, 0],
    [1, 1],
    [0, 0],
    ...
  ]
}
```

Each row is one shot, columns are qubits in order.

### 6.3 Conversion Strategy

**Step 1: Extract measurement info from circuit**

```python
measurements = []
for moment in circuit:
    for op in moment:
        if isinstance(op.gate, cirq.MeasurementGate):
            key = str(op.gate.key)
            qubits = list(op.qubits)
            measurements.append((key, qubits))

# Result: [('first', [q0]), ('final', [q0, q1])]
```

**Step 2: Split LRET samples by measurement**

```python
lret_samples = np.array([[0, 0], [1, 1], ...])  # Shape: (100, 2)

measurements_dict = {}

col_offset = 0
for key, qubits in measurements:
    num_qubits = len(qubits)
    qubit_indices = [qubit_map[q] for q in qubits]
    
    # Extract columns
    samples = lret_samples[:, qubit_indices]
    measurements_dict[key] = samples
    
    col_offset += num_qubits
```

**Step 3: Return to Cirq**

```python
# Cirq expects dict of numpy arrays
return measurements_dict
```

### 6.4 Edge Cases

**Case 1: No explicit measurements**
```python
circuit = cirq.Circuit(cirq.H(q0))  # No measure()

# Solution: Measure all qubits with default key
measurements_dict = {'result': lret_samples}
```

**Case 2: Multi-qubit measurement**
```python
circuit = cirq.Circuit(
    cirq.H(q0),
    cirq.CNOT(q0, q1),
    cirq.measure(q0, q1, key='bell')  # Measure both together
)

# Result shape: (repetitions, 2)
# Values: 0 (00), 1 (01), 2 (10), 3 (11)
```

**Case 3: Mid-circuit measurement**
```python
circuit = cirq.Circuit(
    cirq.H(q0),
    cirq.measure(q0, key='mid'),
    cirq.X(q0),
    cirq.measure(q0, key='end')
)

# LRET may not support mid-circuit measurements yet
# → Raise error or ignore mid-circuit measurements
```

### 6.5 Measurement Key Conventions

**Cirq Conventions:**
- Default key: `'measure'` or circuit-assigned
- Multiple measurements: unique keys required
- Key must be string

**LRET Adaptation:**
- Store measurement keys in order
- Use key as dict key in result
- Preserve qubit ordering within each key

---

## 7. Implementation Guide

### 7.1 Day-by-Day Breakdown

#### **Day 1: Project Setup & CircuitTranslator (Basic)**

**Duration:** 8 hours  
**Goal:** Working translation for basic gates (H, X, Y, Z, CNOT)

**Morning (4 hours): Setup**

1. **Create directory structure:**
   ```bash
   cd d:\LRET\python
   mkdir -p lret_cirq/translators/tests
   touch lret_cirq/__init__.py
   touch lret_cirq/translators/__init__.py
   touch lret_cirq/tests/__init__.py
   ```

2. **Install Cirq:**
   ```bash
   pip install cirq>=1.3.0
   pip install pytest>=7.4.0
   ```

3. **Create `CircuitTranslator` skeleton:**
   - File: `python/lret_cirq/translators/circuit_translator.py`
   - Implement `_build_qubit_map()`
   - Implement basic `_translate_operation()` for H, X, Y, Z
   - Implement `translate()` main method

**Afternoon (4 hours): Basic Translation**

4. **Add gate mapping:**
   - Implement `_build_gate_map()`
   - Add H, X, Y, Z, CNOT gates

5. **Write first tests:**
   ```python
   # tests/test_circuit_translator.py
   def test_translate_hadamard():
       q0 = cirq.LineQubit(0)
       circuit = cirq.Circuit(cirq.H(q0))
       
       translator = CircuitTranslator()
       result = translator.translate(circuit)
       
       assert result["circuit"]["num_qubits"] == 1
       assert result["circuit"]["operations"][0] == {
           "name": "H",
           "wires": [0]
       }
   ```

6. **Run tests:**
   ```bash
   pytest lret_cirq/tests/test_circuit_translator.py -v
   ```

**Deliverables:**
- ✅ CircuitTranslator class with basic gates
- ✅ Qubit mapping working (LineQubit)
- ✅ 5 tests passing (H, X, Y, Z, CNOT translation)

---

#### **Day 2: LRETSimulator & ResultConverter**

**Duration:** 8 hours  
**Goal:** End-to-end Bell state circuit working

**Morning (4 hours): LRETSimulator**

1. **Create `LRETSimulator` class:**
   - File: `python/lret_cirq/lret_simulator.py`
   - Inherit from `cirq.SimulatesSamples`
   - Implement `__init__()`
   - Implement `_run()` method

2. **Integrate with qlret:**
   ```python
   from qlret import simulate_json
   
   def _run(self, circuit, repetitions):
       lret_json = self._translator.translate(circuit, shots=repetitions)
       lret_result = simulate_json(lret_json)
       return self._converter.convert(lret_result, circuit)
   ```

3. **Write simulator tests:**
   ```python
   def test_simulator_bell_state():
       sim = LRETSimulator()
       q0, q1 = cirq.LineQubit.range(2)
       circuit = cirq.Circuit(
           cirq.H(q0),
           cirq.CNOT(q0, q1),
           cirq.measure(q0, q1, key='result')
       )
       
       result = sim.run(circuit, repetitions=1000)
       counts = result.histogram(key='result')
       
       # Bell state: ~50/50 split between |00⟩ and |11⟩
       assert counts[0] > 400  # Binary 00
       assert counts[3] > 400  # Binary 11
       assert counts[1] < 50   # Binary 01 (should be ~0)
       assert counts[2] < 50   # Binary 10 (should be ~0)
   ```

**Afternoon (4 hours): ResultConverter**

4. **Implement `ResultConverter`:**
   - File: `python/lret_cirq/translators/result_converter.py`
   - Implement `convert()`
   - Handle measurement key extraction
   - Handle sample array splitting

5. **Write converter tests:**
   ```python
   def test_result_conversion():
       lret_result = {
           "samples": [[0, 0], [1, 1], [0, 0], [1, 1]]
       }
       
       q0, q1 = cirq.LineQubit.range(2)
       circuit = cirq.Circuit(cirq.measure(q0, q1, key='m'))
       qubit_map = {q0: 0, q1: 1}
       
       converter = ResultConverter()
       measurements = converter.convert(lret_result, circuit, qubit_map)
       
       assert 'm' in measurements
       assert measurements['m'].shape == (4, 2)
       assert np.array_equal(measurements['m'][0], [0, 0])
   ```

6. **Integration test:**
   ```bash
   pytest lret_cirq/tests/test_integration.py::test_bell_state -v
   ```

**Deliverables:**
- ✅ LRETSimulator working
- ✅ ResultConverter working
- ✅ Bell state test passing
- ✅ 10 total tests passing

---

#### **Day 3: Extended Gates & Power Gates**

**Duration:** 8 hours  
**Goal:** All common gates working, power gates handled

**Morning (4 hours): Extended Gates**

1. **Add more single-qubit gates:**
   - S, T, SDG, TDG
   - SX (√X)
   - Update `_build_gate_map()`

2. **Add more two-qubit gates:**
   - CZ
   - SWAP

3. **Write tests for each gate:**
   ```python
   def test_s_gate():
       q0 = cirq.LineQubit(0)
       circuit = cirq.Circuit(cirq.S(q0))
       
       translator = CircuitTranslator()
       result = translator.translate(circuit)
       
       assert result["circuit"]["operations"][0]["name"] == "S"
   ```

**Afternoon (4 hours): Power Gates**

4. **Implement `_translate_power_gate()`:**
   - Handle XPowGate, YPowGate, ZPowGate
   - Map exponent=1.0 → discrete gate
   - Map exponent=0.5 → √gate
   - Map exponent=0.25 → T gate
   - Map other → rotation gate

5. **Write power gate tests:**
   ```python
   def test_x_power_gate_half():
       q0 = cirq.LineQubit(0)
       circuit = cirq.Circuit(cirq.XPowGate(exponent=0.5)(q0))
       
       translator = CircuitTranslator()
       result = translator.translate(circuit)
       
       assert result["circuit"]["operations"][0]["name"] == "SX"
   
   def test_x_power_gate_custom():
       q0 = cirq.LineQubit(0)
       circuit = cirq.Circuit(cirq.XPowGate(exponent=0.3)(q0))
       
       translator = CircuitTranslator()
       result = translator.translate(circuit)
       
       op = result["circuit"]["operations"][0]
       assert op["name"] == "RX"
       assert np.isclose(op["params"][0], 0.3 * np.pi)
   ```

6. **Run extended tests:**
   ```bash
   pytest lret_cirq/tests/ -v
   ```

**Deliverables:**
- ✅ 10+ gates supported (S, T, SDG, TDG, SX, CZ, SWAP)
- ✅ Power gates working (XPowGate, YPowGate, ZPowGate)
- ✅ 25 tests passing

---

#### **Day 4: Rotation Gates & Advanced Features**

**Duration:** 8 hours  
**Goal:** Rotation gates, GridQubit, NamedQubit support

**Morning (4 hours): Rotation Gates**

1. **Implement `_translate_rotation_gate()`:**
   - Handle cirq.rx(), cirq.ry(), cirq.rz()
   - Extract angle in radians
   - Map to LRET RX, RY, RZ

2. **Write rotation tests:**
   ```python
   def test_rx_gate():
       q0 = cirq.LineQubit(0)
       angle = np.pi / 4
       circuit = cirq.Circuit(cirq.rx(angle)(q0))
       
       translator = CircuitTranslator()
       result = translator.translate(circuit)
       
       op = result["circuit"]["operations"][0]
       assert op["name"] == "RX"
       assert np.isclose(op["params"][0], angle)
   ```

3. **Test rotation gate integration:**
   - Create QFT circuit
   - Verify all rotations translate correctly

**Afternoon (4 hours): Advanced Qubit Types**

4. **Test GridQubit support:**
   ```python
   def test_grid_qubit_mapping():
       q00 = cirq.GridQubit(0, 0)
       q01 = cirq.GridQubit(0, 1)
       q10 = cirq.GridQubit(1, 0)
       
       circuit = cirq.Circuit(
           cirq.H(q00),
           cirq.CNOT(q00, q01),
           cirq.CNOT(q01, q10)
       )
       
       translator = CircuitTranslator()
       result = translator.translate(circuit)
       
       assert result["circuit"]["num_qubits"] == 3
       qubit_map = translator.get_qubit_map()
       assert qubit_map[q00] == 0
       assert qubit_map[q01] == 1
       assert qubit_map[q10] == 2
   ```

5. **Test NamedQubit support:**
   ```python
   def test_named_qubit_mapping():
       alice = cirq.NamedQubit("alice")
       bob = cirq.NamedQubit("bob")
       
       circuit = cirq.Circuit(cirq.CNOT(alice, bob))
       
       translator = CircuitTranslator()
       result = translator.translate(circuit)
       
       assert result["circuit"]["num_qubits"] == 2
   ```

6. **Test mixed qubit types:**
   ```python
   def test_mixed_qubit_types():
       q0 = cirq.LineQubit(0)
       alice = cirq.NamedQubit("alice")
       
       circuit = cirq.Circuit(cirq.CNOT(q0, alice))
       
       translator = CircuitTranslator()
       result = translator.translate(circuit)
       
       assert result["circuit"]["num_qubits"] == 2
   ```

**Deliverables:**
- ✅ Rotation gates working (RX, RY, RZ)
- ✅ GridQubit supported
- ✅ NamedQubit supported
- ✅ 35 tests passing

---

#### **Day 5: Integration Testing & GHZ/QFT**

**Duration:** 8 hours  
**Goal:** Complex circuits validated, error handling complete

**Morning (4 hours): Complex Circuits**

1. **Test GHZ state (3-5 qubits):**
   ```python
   def test_ghz_3_qubit():
       sim = LRETSimulator()
       q0, q1, q2 = cirq.LineQubit.range(3)
       
       circuit = cirq.Circuit(
           cirq.H(q0),
           cirq.CNOT(q0, q1),
           cirq.CNOT(q1, q2),
           cirq.measure(q0, q1, q2, key='result')
       )
       
       result = sim.run(circuit, repetitions=1000)
       counts = result.histogram(key='result')
       
       # GHZ: only |000⟩ and |111⟩
       assert counts[0] > 450  # Binary 000
       assert counts[7] > 450  # Binary 111
       assert sum([counts[i] for i in [1,2,3,4,5,6]]) < 100
   ```

2. **Test QFT circuit:**
   ```python
   def test_qft_4_qubit():
       sim = LRETSimulator()
       qubits = cirq.LineQubit.range(4)
       
       circuit = cirq.Circuit()
       # Add QFT operations
       circuit.append([
           cirq.H(qubits[0]),
           cirq.CZ(qubits[0], qubits[1])**0.5,
           cirq.CZ(qubits[0], qubits[2])**0.25,
           # ... more QFT gates
       ])
       circuit.append(cirq.measure(*qubits, key='qft'))
       
       result = sim.run(circuit, repetitions=100)
       assert 'qft' in result.measurements
   ```

3. **Test random circuits:**
   ```python
   def test_random_circuit():
       sim = LRETSimulator()
       qubits = cirq.LineQubit.range(6)
       
       circuit = cirq.testing.random_circuit(
           qubits=qubits,
           n_moments=10,
           op_density=0.8
       )
       circuit.append(cirq.measure(*qubits, key='random'))
       
       result = sim.run(circuit, repetitions=100)
       assert result.measurements['random'].shape == (100, 6)
   ```

**Afternoon (4 hours): Error Handling**

4. **Test unsupported gates:**
   ```python
   def test_unsupported_gate_error():
       sim = LRETSimulator()
       q0, q1, q2 = cirq.LineQubit.range(3)
       
       circuit = cirq.Circuit(cirq.TOFFOLI(q0, q1, q2))
       
       with pytest.raises(TranslationError) as exc:
           sim.run(circuit, repetitions=100)
       
       assert "Unsupported gate" in str(exc.value)
       assert "TOFFOLI" in str(exc.value) or "CCX" in str(exc.value)
   ```

5. **Test unresolved parameters:**
   ```python
   def test_unresolved_parameters_error():
       import sympy
       sim = LRETSimulator()
       q0 = cirq.LineQubit(0)
       
       theta = sympy.Symbol('theta')
       circuit = cirq.Circuit(cirq.rx(theta)(q0))
       
       with pytest.raises(ValueError) as exc:
           sim.run(circuit, repetitions=100)
       
       assert "unresolved parameters" in str(exc.value).lower()
   ```

6. **Test edge cases:**
   - Empty circuit
   - Single qubit
   - 20+ qubits (performance warning)
   - No measurements (default behavior)

**Deliverables:**
- ✅ GHZ state working
- ✅ QFT circuit working
- ✅ Random circuits working
- ✅ Error handling complete
- ✅ 45 tests passing

---

#### **Day 6: Polish, Documentation & Final Testing**

**Duration:** 8 hours  
**Goal:** 50+ tests passing, complete documentation

**Morning (4 hours): Final Tests & Edge Cases**

1. **Test measurement key handling:**
   ```python
   def test_multiple_measurement_keys():
       sim = LRETSimulator()
       q0, q1 = cirq.LineQubit.range(2)
       
       circuit = cirq.Circuit(
           cirq.H(q0),
           cirq.measure(q0, key='first'),
           cirq.CNOT(q0, q1),
           cirq.measure(q0, q1, key='final')
       )
       
       result = sim.run(circuit, repetitions=100)
       
       assert 'first' in result.measurements
       assert 'final' in result.measurements
       assert result.measurements['first'].shape == (100, 1)
       assert result.measurements['final'].shape == (100, 2)
   ```

2. **Test custom epsilon:**
   ```python
   def test_custom_epsilon():
       sim1 = LRETSimulator(epsilon=1e-6)
       sim2 = LRETSimulator(epsilon=1e-3)
       
       q0, q1 = cirq.LineQubit.range(2)
       circuit = cirq.Circuit(
           cirq.H(q0),
           cirq.CNOT(q0, q1),
           cirq.measure(q0, q1, key='result')
       )
       
       result1 = sim1.run(circuit, repetitions=1000)
       result2 = sim2.run(circuit, repetitions=1000)
       
       # Both should work, epsilon=1e-6 may be more accurate
       assert 'result' in result1.measurements
       assert 'result' in result2.measurements
   ```

3. **Test seed reproducibility:**
   ```python
   def test_seed_reproducibility():
       q0 = cirq.LineQubit(0)
       circuit = cirq.Circuit(
           cirq.H(q0),
           cirq.measure(q0, key='result')
       )
       
       sim1 = LRETSimulator(seed=42)
       result1 = sim1.run(circuit, repetitions=100)
       
       sim2 = LRETSimulator(seed=42)
       result2 = sim2.run(circuit, repetitions=100)
       
       assert np.array_equal(
           result1.measurements['result'],
           result2.measurements['result']
       )
   ```

4. **Add remaining tests to reach 50+**

**Afternoon (4 hours): Documentation**

5. **Write README.md for lret_cirq:**
   ```markdown
   # LRET Cirq Integration
   
   ## Installation
   
   ## Quick Start
   
   ## Examples
   
   ## Supported Gates
   
   ## API Reference
   
   ## Performance
   ```

6. **Add docstrings to all public methods**

7. **Create examples directory:**
   ```
   lret_cirq/examples/
   ├── bell_state.py
   ├── ghz_state.py
   ├── qft.py
   └── quantum_walk.py
   ```

8. **Run full test suite:**
   ```bash
   pytest lret_cirq/tests/ -v --cov=lret_cirq --cov-report=html
   ```

9. **Fix any remaining issues**

**Deliverables:**
- ✅ 50+ tests passing (100% pass rate)
- ✅ Complete documentation
- ✅ Example scripts
- ✅ Coverage report >90%
- ✅ Ready for production use

---

## 8. Testing Strategy

### 8.1 Test Categories

**1. TestCircuitTranslator (15 tests)**

| Test | Description | Validation |
|------|-------------|------------|
| `test_translate_empty_circuit` | Circuit with no operations | Valid JSON, 0 ops |
| `test_translate_hadamard` | Single H gate | Correct translation |
| `test_translate_pauli_x` | Single X gate | Correct translation |
| `test_translate_pauli_y` | Single Y gate | Correct translation |
| `test_translate_pauli_z` | Single Z gate | Correct translation |
| `test_translate_s_gate` | S gate (√Z) | Correct translation |
| `test_translate_t_gate` | T gate (∜Z) | Correct translation |
| `test_translate_cnot` | CNOT gate | Two-qubit wires |
| `test_translate_cz` | CZ gate | Two-qubit wires |
| `test_translate_swap` | SWAP gate | Two-qubit wires |
| `test_translate_rx_gate` | Rotation gate | Angle extraction |
| `test_translate_power_gate_full` | XPowGate(1.0) | Maps to X |
| `test_translate_power_gate_half` | XPowGate(0.5) | Maps to SX |
| `test_translate_power_gate_custom` | XPowGate(0.3) | Maps to RX |
| `test_translate_measurement` | Measurement operation | Key extraction |

**2. TestQubitMapping (5 tests)**

| Test | Description | Validation |
|------|-------------|------------|
| `test_line_qubit_mapping` | LineQubit(0,1,2) | Correct order |
| `test_grid_qubit_mapping` | GridQubit(0,0), (0,1), (1,0) | Correct sorting |
| `test_named_qubit_mapping` | NamedQubit("alice", "bob") | Alphabetical |
| `test_mixed_qubit_types` | LineQubit + NamedQubit | Type priority |
| `test_qubit_map_retrieval` | get_qubit_map() | Returns copy |

**3. TestResultConverter (5 tests)**

| Test | Description | Validation |
|------|-------------|------------|
| `test_convert_simple_result` | Single qubit measurement | Correct shape |
| `test_convert_multi_qubit` | Two qubit measurement | Correct shape |
| `test_measurement_key_organization` | Multiple keys | Correct dict keys |
| `test_qubit_ordering` | Non-sequential qubits | Preserved order |
| `test_no_measurements` | Circuit without measure | Default key |

**4. TestLRETSimulator (5 tests)**

| Test | Description | Validation |
|------|-------------|------------|
| `test_simulator_initialization` | Create simulator | No errors |
| `test_epsilon_property` | Get/set epsilon | Value validation |
| `test_simulator_str_repr` | String representation | Contains info |
| `test_run_basic_circuit` | Simple H + measure | Returns result |
| `test_run_multiple_repetitions` | Various repetition counts | Correct shape |

**5. TestIntegration (10 tests)**

| Test | Description | Validation |
|------|-------------|------------|
| `test_bell_state` | Bell state circuit | 50/50 |00⟩/|11⟩ |
| `test_ghz_3_qubit` | GHZ state (3q) | Only |000⟩, |111⟩ |
| `test_ghz_4_qubit` | GHZ state (4q) | Only |0000⟩, |1111⟩ |
| `test_qft_circuit` | QFT (4 qubits) | Executes successfully |
| `test_random_circuit` | Random circuit | No errors |
| `test_parameterized_resolved` | Resolved params | Works correctly |
| `test_multiple_measurements` | Multiple keys | All keys present |
| `test_custom_epsilon` | epsilon=1e-6 | Works correctly |
| `test_seed_reproducibility` | Fixed seed | Same results |
| `test_large_circuit` | 15 qubits | Performance acceptable |

**6. TestErrorHandling (5 tests)**

| Test | Description | Expected Error |
|------|-------------|----------------|
| `test_unsupported_gate` | TOFFOLI gate | TranslationError |
| `test_unresolved_parameters` | sympy.Symbol | ValueError |
| `test_invalid_circuit` | Malformed circuit | Exception |
| `test_no_native_module` | LRET not built | ImportError |
| `test_epsilon_out_of_range` | epsilon < 0 | ValueError |

**7. TestGateSet (10 tests)**

Individual tests for each gate type to ensure correct translation.

### 8.2 Test Execution

**Run all tests:**
```bash
cd d:\LRET\python
pytest lret_cirq/tests/ -v
```

**Run specific category:**
```bash
pytest lret_cirq/tests/test_circuit_translator.py -v
```

**With coverage:**
```bash
pytest lret_cirq/tests/ --cov=lret_cirq --cov-report=html
open htmlcov/index.html
```

**Performance testing:**
```bash
pytest lret_cirq/tests/ --durations=10
```

### 8.3 Success Criteria

**Minimum (MVP):**
- ✅ 25+ tests passing
- ✅ Bell state works
- ✅ Basic gates translate

**Full Release:**
- ✅ 50+ tests passing
- ✅ All test categories complete
- ✅ Coverage >90%
- ✅ No failing tests

---

## 9. API Reference

### 9.1 LRETSimulator

```python
class LRETSimulator(cirq.SimulatesSamples):
    """
    LRET-backed quantum circuit simulator for Cirq.
    
    Args:
        epsilon (float): SVD truncation threshold (default: 1e-4)
        noise_model (dict, optional): LRET noise configuration
        seed (int, optional): Random seed for measurements
    
    Example:
        >>> sim = LRETSimulator(epsilon=1e-4)
        >>> result = sim.run(circuit, repetitions=1000)
    """
    
    def __init__(self, epsilon=1e-4, noise_model=None, seed=None):
        """Initialize LRET simulator."""
        pass
    
    @property
    def epsilon(self) -> float:
        """Get the truncation threshold."""
        pass
    
    @epsilon.setter
    def epsilon(self, value: float):
        """Set the truncation threshold."""
        pass
    
    def run(
        self,
        circuit: cirq.Circuit,
        repetitions: int = 1,
        param_resolver: cirq.ParamResolver = None
    ) -> cirq.Result:
        """
        Run circuit and return results.
        
        Args:
            circuit: Cirq circuit to simulate
            repetitions: Number of measurement samples
            param_resolver: Parameter values
        
        Returns:
            cirq.Result with measurements
        
        Raises:
            TranslationError: Unsupported gates
            ValueError: Unresolved parameters
        """
        pass
```

### 9.2 CircuitTranslator

```python
class CircuitTranslator:
    """
    Translates Cirq circuits to LRET JSON format.
    
    Example:
        >>> translator = CircuitTranslator()
        >>> lret_json = translator.translate(circuit)
    """
    
    def translate(
        self,
        circuit: cirq.Circuit,
        epsilon: float = 1e-4,
        shots: int = 1024,
        noise_model: dict = None,
        seed: int = None
    ) -> dict:
        """
        Translate Cirq circuit to LRET JSON.
        
        Returns:
            LRET JSON dict with config, circuit, shots
        """
        pass
    
    def get_qubit_map(self) -> Dict[cirq.Qid, int]:
        """Get mapping from Cirq qubits to integer indices."""
        pass
```

### 9.3 ResultConverter

```python
class ResultConverter:
    """
    Converts LRET results to Cirq Result format.
    
    Example:
        >>> converter = ResultConverter()
        >>> measurements = converter.convert(lret_result, circuit, qubit_map)
    """
    
    def convert(
        self,
        lret_result: dict,
        circuit: cirq.Circuit,
        qubit_map: Dict[cirq.Qid, int]
    ) -> Dict[str, np.ndarray]:
        """
        Convert LRET result to Cirq measurements dict.
        
        Returns:
            Dict mapping measurement keys to sample arrays
        """
        pass
```

---

## 10. Troubleshooting Guide

### 10.1 Common Errors

**Error 1: "LRET native module not available"**

```
ImportError: LRET native module not available.
Install with: cd python/qlret && pip install -e .
```

**Solution:**
```bash
cd d:\LRET\python\qlret
pip install -e .
```

Verify:
```python
import qlret
print(qlret.__version__)
```

---

**Error 2: "Unsupported gate: CCX"**

```
TranslationError: Unsupported gate: CCX (Toffoli).
Supported gates: [...]
```

**Solution:**
Cirq's three-qubit gates (TOFFOLI, FREDKIN) are not yet supported. Decompose manually:

```python
# Instead of:
circuit.append(cirq.TOFFOLI(q0, q1, q2))

# Use decomposition:
circuit.append([
    cirq.H(q2),
    cirq.CNOT(q1, q2),
    cirq.T**-1(q2),
    cirq.CNOT(q0, q2),
    cirq.T(q2),
    cirq.CNOT(q1, q2),
    cirq.T**-1(q2),
    cirq.CNOT(q0, q2),
    cirq.T(q1),
    cirq.T(q2),
    cirq.CNOT(q0, q1),
    cirq.H(q2),
    cirq.T(q0),
    cirq.T**-1(q1),
    cirq.CNOT(q0, q1)
])
```

---

**Error 3: "Circuit contains unresolved parameters"**

```
ValueError: Circuit contains unresolved parameters.
Provide a param_resolver or use cirq.resolve_parameters().
```

**Solution:**
Resolve parameters before running:

```python
import sympy
import cirq

theta = sympy.Symbol('theta')
circuit = cirq.Circuit(cirq.rx(theta)(q0))

# Option 1: Use param_resolver
result = sim.run(circuit, repetitions=100, param_resolver={theta: 0.5})

# Option 2: Pre-resolve
resolved_circuit = cirq.resolve_parameters(circuit, {theta: 0.5})
result = sim.run(resolved_circuit, repetitions=100)
```

---

**Error 4: Bell state not showing 50/50 split**

```python
counts = result.histogram(key='result')
# Counter({0: 1000, 3: 0})  # Expected ~500/500
```

**Problem:** Circuit not creating entanglement.

**Solution:**
Check circuit structure:

```python
# Correct Bell state
circuit = cirq.Circuit(
    cirq.H(q0),
    cirq.CNOT(q0, q1),  # Must be CNOT, not CZ
    cirq.measure(q0, q1, key='result')
)
```

Verify LRET execution:
```python
# Debug translation
translator = CircuitTranslator()
lret_json = translator.translate(circuit)
print(lret_json)
# Should show: H on qubit 0, then CX on [0, 1]
```

---

**Error 5: "Measurement key 'result' not found"**

```python
counts = result.histogram(key='result')
# KeyError: 'result'
```

**Solution:**
Check measurement key in circuit:

```python
# If you used a different key:
circuit = cirq.Circuit(
    cirq.H(q0),
    cirq.measure(q0, key='my_measurement')  # Note the key
)

result = sim.run(circuit, repetitions=100)
counts = result.histogram(key='my_measurement')  # Use same key
```

---

### 10.2 Performance Tips

**Tip 1: Choose appropriate epsilon**

| Epsilon | Speed | Accuracy | Use Case |
|---------|-------|----------|----------|
| 1e-3 | Fast | Good | Development/testing |
| 1e-4 | Medium | Very good | General use |
| 1e-6 | Slow | Excellent | Publication results |

**Tip 2: Qubit count limits**

- **< 10 qubits:** Fast, interactive
- **10-15 qubits:** Good performance
- **15-20 qubits:** Acceptable, may take minutes
- **> 20 qubits:** Slow, consider reducing epsilon or circuit depth

**Tip 3: Circuit depth optimization**

```python
# Slow: Deep circuit
circuit = cirq.Circuit()
for _ in range(100):
    circuit.append(cirq.H(q0))
    circuit.append(cirq.X(q0))

# Fast: Shallow equivalent
circuit = cirq.Circuit()  # H-X pairs cancel
```

---

### 10.3 Debugging Checklist

When tests fail:

1. **Check LRET installation:**
   ```bash
   python -c "import qlret; print(qlret.__version__)"
   ```

2. **Verify circuit translation:**
   ```python
   translator = CircuitTranslator()
   lret_json = translator.translate(circuit)
   print(json.dumps(lret_json, indent=2))
   ```

3. **Test LRET directly:**
   ```python
   from qlret import simulate_json
   result = simulate_json(lret_json)
   print(result)
   ```

4. **Check qubit mapping:**
   ```python
   qubit_map = translator.get_qubit_map()
   print(qubit_map)
   # Verify: {LineQubit(0): 0, LineQubit(1): 1, ...}
   ```

5. **Verify measurement keys:**
   ```python
   keys = translator.get_measurement_keys()
   print(keys)
   # Verify: ['result'] or expected keys
   ```

---

## 11. Examples and Tutorials

### 11.1 Example 1: Bell State

```python
"""
Bell State Example
==================

Creates and measures a Bell state (maximally entangled state).

Expected output: ~50% |00⟩, ~50% |11⟩
"""

import cirq
from lret_cirq import LRETSimulator

# Create qubits
q0, q1 = cirq.LineQubit.range(2)

# Build Bell state circuit
circuit = cirq.Circuit(
    cirq.H(q0),              # Hadamard on qubit 0
    cirq.CNOT(q0, q1),       # CNOT (control=q0, target=q1)
    cirq.measure(q0, q1, key='bell')  # Measure both qubits
)

print("Circuit:")
print(circuit)

# Simulate
sim = LRETSimulator(epsilon=1e-4)
result = sim.run(circuit, repetitions=1000)

# Get results
counts = result.histogram(key='bell')
print("\nMeasurement counts:")
print(counts)
# Expected: Counter({0: ~500, 3: ~500})
# 0 = binary 00, 3 = binary 11

# Analysis
total = sum(counts.values())
prob_00 = counts.get(0, 0) / total
prob_11 = counts.get(3, 0) / total
prob_other = (counts.get(1, 0) + counts.get(2, 0)) / total

print(f"\nProbabilities:")
print(f"  |00⟩: {prob_00:.3f}")
print(f"  |11⟩: {prob_11:.3f}")
print(f"  Others: {prob_other:.3f}")

assert prob_00 > 0.45 and prob_00 < 0.55, "Bell state incorrect"
assert prob_11 > 0.45 and prob_11 < 0.55, "Bell state incorrect"
print("\n✓ Bell state verified!")
```

### 11.2 Example 2: GHZ State (3 Qubits)

```python
"""
GHZ State Example
=================

Creates a 3-qubit GHZ state (Greenberger-Horne-Zeilinger).

Expected output: ~50% |000⟩, ~50% |111⟩
"""

import cirq
from lret_cirq import LRETSimulator

# Create qubits
q0, q1, q2 = cirq.LineQubit.range(3)

# Build GHZ circuit
circuit = cirq.Circuit(
    cirq.H(q0),
    cirq.CNOT(q0, q1),
    cirq.CNOT(q1, q2),
    cirq.measure(q0, q1, q2, key='ghz')
)

print("GHZ Circuit:")
print(circuit)

# Simulate
sim = LRETSimulator(epsilon=1e-4)
result = sim.run(circuit, repetitions=1000)

# Get results
counts = result.histogram(key='ghz')
print("\nMeasurement counts:")
for bitstring, count in sorted(counts.items()):
    binary = format(bitstring, '03b')
    print(f"  |{binary}⟩: {count}")

# Verify GHZ property: only |000⟩ and |111⟩
allowed = counts.get(0, 0) + counts.get(7, 0)  # 0=000, 7=111
total = sum(counts.values())
fidelity = allowed / total

print(f"\nGHZ fidelity: {fidelity:.4f}")
assert fidelity > 0.95, "GHZ state incorrect"
print("✓ GHZ state verified!")
```

### 11.3 Example 3: Quantum Fourier Transform (QFT)

```python
"""
Quantum Fourier Transform Example
==================================

Implements QFT on 4 qubits.
"""

import cirq
import numpy as np
from lret_cirq import LRETSimulator

def qft_circuit(qubits):
    """
    Build QFT circuit for given qubits.
    
    Args:
        qubits: List of cirq.LineQubits
    
    Returns:
        cirq.Circuit implementing QFT
    """
    n = len(qubits)
    circuit = cirq.Circuit()
    
    for i in range(n):
        # Hadamard on current qubit
        circuit.append(cirq.H(qubits[i]))
        
        # Controlled rotations
        for j in range(i + 1, n):
            angle = 2 * np.pi / (2 ** (j - i + 1))
            circuit.append(cirq.CZ(qubits[i], qubits[j])**(angle / np.pi))
    
    # Swap qubits to reverse order
    for i in range(n // 2):
        circuit.append(cirq.SWAP(qubits[i], qubits[n - i - 1]))
    
    return circuit

# Create 4 qubits
qubits = cirq.LineQubit.range(4)

# Build QFT circuit
circuit = qft_circuit(qubits)

# Add initial state (optional)
initial_circuit = cirq.Circuit(cirq.X(qubits[0]))  # Start in |0001⟩

# Combine
full_circuit = initial_circuit + circuit
full_circuit.append(cirq.measure(*qubits, key='qft'))

print("QFT Circuit:")
print(full_circuit)

# Simulate
sim = LRETSimulator(epsilon=1e-4)
result = sim.run(full_circuit, repetitions=1000)

# Get results
counts = result.histogram(key='qft')
print("\nTop 5 measurement outcomes:")
for bitstring, count in sorted(counts.items(), key=lambda x: -x[1])[:5]:
    binary = format(bitstring, '04b')
    prob = count / 1000
    print(f"  |{binary}⟩: {count} ({prob:.3f})")

print("\n✓ QFT executed successfully!")
```

### 11.4 Example 4: Parameterized Circuit

```python
"""
Parameterized Circuit Example
==============================

Shows how to use parameter resolution with LRET.
"""

import cirq
import sympy
from lret_cirq import LRETSimulator

# Create qubit
q0 = cirq.LineQubit(0)

# Define parameters
theta = sympy.Symbol('theta')
phi = sympy.Symbol('phi')

# Build parameterized circuit
circuit = cirq.Circuit(
    cirq.rx(theta)(q0),
    cirq.ry(phi)(q0),
    cirq.measure(q0, key='result')
)

print("Parameterized Circuit:")
print(circuit)

# Simulate with different parameter values
sim = LRETSimulator(epsilon=1e-4)

param_values = [
    {'theta': 0.0, 'phi': 0.0},
    {'theta': np.pi/4, 'phi': np.pi/4},
    {'theta': np.pi/2, 'phi': np.pi/2},
]

for params in param_values:
    # Resolve parameters
    resolver = cirq.ParamResolver(params)
    
    # Run circuit
    result = sim.run(circuit, repetitions=1000, param_resolver=resolver)
    
    # Get probabilities
    counts = result.histogram(key='result')
    prob_0 = counts.get(0, 0) / 1000
    prob_1 = counts.get(1, 0) / 1000
    
    print(f"\nθ={params['theta']:.3f}, φ={params['phi']:.3f}")
    print(f"  P(|0⟩) = {prob_0:.3f}")
    print(f"  P(|1⟩) = {prob_1:.3f}")

print("\n✓ Parameterized circuit executed successfully!")
```

### 11.5 Example 5: Custom Epsilon Comparison

```python
"""
Epsilon Comparison Example
===========================

Compares results with different epsilon values.
"""

import cirq
import numpy as np
from lret_cirq import LRETSimulator
import time

# Create circuit
qubits = cirq.LineQubit.range(10)
circuit = cirq.Circuit(
    cirq.H(qubits[0]),
    *[cirq.CNOT(qubits[i], qubits[i+1]) for i in range(9)],
    cirq.measure(*qubits, key='result')
)

print("Testing 10-qubit GHZ state with different epsilon values...\n")

# Test different epsilon values
epsilons = [1e-3, 1e-4, 1e-6]

for eps in epsilons:
    sim = LRETSimulator(epsilon=eps)
    
    start_time = time.time()
    result = sim.run(circuit, repetitions=1000)
    elapsed = time.time() - start_time
    
    counts = result.histogram(key='result')
    
    # Calculate fidelity
    allowed = counts.get(0, 0) + counts.get(1023, 0)  # |0000000000⟩ and |1111111111⟩
    fidelity = allowed / 1000
    
    print(f"Epsilon = {eps:.0e}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Fidelity: {fidelity:.4f}")
    print(f"  |{'0'*10}⟩: {counts.get(0, 0)}")
    print(f"  |{'1'*10}⟩: {counts.get(1023, 0)}")
    print()

print("✓ Comparison complete!")
```

---

## Conclusion

This roadmap provides a complete guide for implementing LRET-Cirq integration over 5-6 days. Key points:

**Architecture:**
- Simple: No Provider/Backend layers
- Direct: `LRETSimulator().run(circuit)`
- Flexible: Supports all Cirq qubit types

**Implementation:**
- Day 1: Basic gates + translation
- Day 2: Simulator + result conversion
- Day 3: Extended gates + power gates
- Day 4: Rotations + advanced qubits
- Day 5: Complex circuits + error handling
- Day 6: Polish + documentation

**Testing:**
- 50+ tests across 7 categories
- Bell, GHZ, QFT circuits validated
- Comprehensive error handling

**Success:**
- Production-ready integration
- Complete documentation
- Ready for Opus 4.5 implementation

**Next Steps:**
Hand off to Opus 4.5 with this roadmap for implementation.

---

**Document Version:** 1.0  
**Created:** January 26, 2026  
**Total Lines:** ~1,800  
**Ready for Implementation:** ✅

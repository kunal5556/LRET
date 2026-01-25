# Phase 7 - Cirq Integration Roadmap Context

**Date**: January 25, 2026  
**Purpose**: Context document for writing comprehensive Cirq integration roadmap  
**Target Model**: Sonnet 4.5 (better for large documentation)  
**Expected Output**: 1,500+ line detailed implementation guide  
**Follow-up**: Opus 4.5 will implement over 5-6 days

---

## Current Status Summary

### ✅ Completed: Qiskit Integration
- **Location**: `d:\LRET\python\lret_qiskit\`
- **Test Status**: 53/53 tests passing (100%)
- **Documentation**: [PHASE_7_QISKIT_TESTING_SUMMARY.md](PHASE_7_QISKIT_TESTING_SUMMARY.md)
- **Components**:
  - `LRETProvider` - Manages 3 backend variants (epsilon: 1e-4, 1e-6, 1e-8)
  - `LRETBackend` - BackendV2 implementation with full target system
  - `CircuitTranslator` - Qiskit → LRET JSON (50+ gates)
  - `ResultConverter` - LRET → Qiskit Result
  - `LRETJob` - JobV1 lifecycle management

### 📋 Next: Cirq Integration
- **Target Location**: `d:\LRET\python\lret_cirq\` (to be created)
- **Timeline**: 5-6 days implementation (after roadmap)
- **Pattern**: Follow Qiskit integration architecture
- **Goal**: LRET as native Cirq simulator backend

---

## Existing Cirq Infrastructure

### Current Cirq Work (Benchmarking Focus)
Located in `d:\LRET\cirq_comparison\`:
- **Purpose**: Performance comparison (LRET vs Cirq FDM)
- **Status**: Infrastructure complete, benchmarks run
- **Key Files**:
  - `cirq_fdm_wrapper.py` - Cirq simulator wrapper for comparisons
  - `benchmark_*.py` - Various benchmark scripts
  - `results/` - Benchmark outputs
  - `plots/` - Visualization outputs

**IMPORTANT**: This is NOT the integration we're building. We need a **Cirq device/simulator that users can import**, not just benchmarking infrastructure.

### Existing Documentation
1. **CIRQ_COMPARISON_GUIDE.md** (398 lines)
   - Focus: Benchmarking LRET vs Cirq
   - OpenCode automation instructions
   - Phase-by-phase comparison workflow
   - NOT an integration guide

2. **ROADMAP.md** - Section 7.1 (brief Cirq mention)
   - High-level concept: "LRET as Cirq Simulator backend"
   - Basic code sketch (50 lines)
   - No implementation details

**What's Missing**: Comprehensive guide like `PHASE_7_DETAILED_IMPLEMENTATION_ROADMAP.md` but for Cirq.

---

## Cirq vs Qiskit Architecture Differences

| Aspect | Qiskit | Cirq | Implications for LRET |
|--------|--------|------|----------------------|
| **Base Class** | `BackendV2` | `cirq.SimulatesSamples` or `cirq.Simulator` | Different interface requirements |
| **Provider System** | `Provider → Backend → Job` | Direct simulator import | No provider layer needed |
| **Circuit Type** | `QuantumCircuit` | `cirq.Circuit` | Different circuit structure |
| **Gate Library** | Unified (H, X, CX, RZ, etc.) | Google gates (XPowGate, YPowGate, etc.) | Gate mapping more complex |
| **Qubits** | Integer indices (0, 1, 2) | `cirq.LineQubit`, `cirq.GridQubit` | Need qubit mapping |
| **Operations** | `qc.h(0)` | `circuit.append(cirq.H(q0))` | Different operation model |
| **Measurements** | Explicit `measure()` | Measurement moments | Different measurement handling |
| **Result Type** | `Result` object with counts | `Result` with measurements array | Different result conversion |
| **Noise** | `NoiseModel` class | `cirq.Channel` operations | Different noise integration |
| **Parameters** | `Parameter` class | `sympy.Symbol` | Different parameterization |

---

## Cirq Integration Requirements

### Must-Have Components

#### 1. Cirq Simulator Class
```python
# python/lret_cirq/cirq_simulator.py
import cirq
from qlret import simulate_json

class LRETSimulator(cirq.SimulatesSamples):
    """LRET-backed Cirq simulator."""
    
    def __init__(self, epsilon=1e-4, **kwargs):
        self._epsilon = epsilon
        super().__init__(**kwargs)
    
    def _run(self, circuit, repetitions):
        """Run circuit and return samples."""
        # Translate Cirq circuit → LRET JSON
        # Execute via simulate_json()
        # Convert LRET result → Cirq result
        pass
```

**Base Class Options**:
- `cirq.SimulatesSamples` - For sampling-based simulation (measurement results)
- `cirq.Simulator` - For full state vector/density matrix access
- **Recommendation**: Start with `SimulatesSamples` (simpler), add `Simulator` later

#### 2. Circuit Translator
```python
# python/lret_cirq/translators/circuit_translator.py
import cirq

class CircuitTranslator:
    """Translate Cirq circuits to LRET JSON."""
    
    def translate(self, circuit: cirq.Circuit, epsilon: float) -> dict:
        """Convert cirq.Circuit to LRET JSON format."""
        # Map cirq.LineQubit → integers
        # Translate gates (handle XPowGate, YPowGate, etc.)
        # Build JSON structure
        pass
```

**Gate Mapping Challenge**: Cirq uses `XPowGate(exponent=0.5)` instead of discrete gates.
- `XPowGate(exponent=1.0)` = X gate
- `XPowGate(exponent=0.5)` = √X gate
- `YPowGate(exponent=0.5)` = √Y gate
- Need to map to LRET's gate set or add new gates

#### 3. Result Converter
```python
# python/lret_cirq/translators/result_converter.py
import cirq
import numpy as np

class ResultConverter:
    """Convert LRET results to Cirq format."""
    
    def convert(self, lret_result: dict, circuit: cirq.Circuit, repetitions: int):
        """Convert LRET JSON result to cirq.Result."""
        # Extract measurements from LRET
        # Map qubit indices back to cirq.LineQubit
        # Create cirq.Result with measurements
        pass
```

#### 4. Tests (50+ tests)
```python
# python/lret_cirq/tests/test_integration.py
import cirq
import pytest
from lret_cirq import LRETSimulator

class TestLRETCirqSimulator:
    def test_bell_state(self):
        sim = LRETSimulator()
        q0, q1 = cirq.LineQubit.range(2)
        circuit = cirq.Circuit(
            cirq.H(q0),
            cirq.CNOT(q0, q1),
            cirq.measure(q0, q1, key='m')
        )
        result = sim.run(circuit, repetitions=100)
        # Check Bell state distribution
```

---

## Cirq Gate Set Reference

### Common Cirq Gates

| Cirq Gate | Description | LRET Equivalent |
|-----------|-------------|-----------------|
| `cirq.H(q)` | Hadamard | `H` |
| `cirq.X(q)` | Pauli-X (NOT) | `X` |
| `cirq.Y(q)` | Pauli-Y | `Y` |
| `cirq.Z(q)` | Pauli-Z | `Z` |
| `cirq.S(q)` | S gate (√Z) | `S` |
| `cirq.T(q)` | T gate | `T` |
| `cirq.CNOT(q1, q2)` | Controlled-NOT | `CX` |
| `cirq.CZ(q1, q2)` | Controlled-Z | `CZ` |
| `cirq.SWAP(q1, q2)` | SWAP | `SWAP` |
| `cirq.rx(θ)(q)` | X-rotation | `RX` with angle |
| `cirq.ry(θ)(q)` | Y-rotation | `RY` with angle |
| `cirq.rz(θ)(q)` | Z-rotation | `RZ` with angle |
| `cirq.XPowGate(exponent=e)` | X^e gate | Complex - may need extension |
| `cirq.YPowGate(exponent=e)` | Y^e gate | Complex - may need extension |
| `cirq.ZPowGate(exponent=e)` | Z^e gate | Map to RZ if exponent linear |

### Google-Specific Gates (Lower Priority)
- `cirq.FSimGate` - Fermionic simulation gate
- `cirq.SycamoreGate` - Google Sycamore gate
- `cirq.ISwapPowGate` - iSWAP variations

**Strategy**: Support common gates first, add Google gates later if needed.

---

## Qubit Mapping Strategy

### Cirq Qubit Types
```python
# Line qubits (1D chain)
q0, q1, q2 = cirq.LineQubit.range(3)

# Grid qubits (2D lattice)
q00 = cirq.GridQubit(0, 0)
q01 = cirq.GridQubit(0, 1)

# Named qubits
qa = cirq.NamedQubit("alice")
qb = cirq.NamedQubit("bob")
```

### LRET Format (Integer Indices)
```json
{
  "circuit": {
    "num_qubits": 3,
    "operations": [
      {"name": "H", "wires": [0]},
      {"name": "CX", "wires": [0, 1]}
    ]
  }
}
```

### Mapping Solution
```python
def build_qubit_map(circuit: cirq.Circuit) -> dict:
    """Map cirq qubits to integer indices."""
    qubits = sorted(circuit.all_qubits())
    return {qubit: idx for idx, qubit in enumerate(qubits)}

# Example:
# {LineQubit(0): 0, LineQubit(1): 1, LineQubit(2): 2}
```

---

## Measurement Handling

### Cirq Measurements
```python
circuit = cirq.Circuit(
    cirq.H(q0),
    cirq.measure(q0, key='m0'),  # Measure with key
    cirq.measure(q1, key='m1')
)

result = sim.run(circuit, repetitions=100)
counts = result.histogram(key='m0')  # Get counts for specific measurement
measurements = result.measurements['m0']  # Get all samples (100×1 array)
```

### LRET Format
```json
{
  "circuit": {
    "operations": [
      {"name": "H", "wires": [0]},
      {"name": "measure", "wires": [0]}
    ]
  }
}

// Result:
{
  "samples": [[0], [1], [0], [1], ...],  // 100 samples
  "counts": {"0": 52, "1": 48}
}
```

### Conversion Strategy
1. Extract measurement operations from Cirq circuit
2. Track measurement keys → qubit mapping
3. Translate to LRET measure operations
4. Convert LRET samples back to Cirq format with correct keys

---

## Noise Model Integration

### Cirq Noise Channels
```python
import cirq

# Depolarizing noise
noise_model = cirq.ConstantQubitNoiseModel(
    cirq.DepolarizingChannel(0.01)
)

# Apply to circuit
noisy_circuit = noise_model.noisy_moment(circuit)
```

### LRET Noise Format
```json
{
  "noise": [
    {
      "type": "depolarizing",
      "qubits": [0, 1, 2],
      "probability": 0.01
    }
  ]
}
```

### Integration Strategy
1. Parse Cirq noise operations (if present)
2. Convert to LRET noise JSON
3. Include in simulation config

---

## Parameterized Circuits

### Cirq Parameters
```python
import sympy

theta = sympy.Symbol('theta')
circuit = cirq.Circuit(
    cirq.rx(theta)(q0),
    cirq.ry(2 * theta)(q1)
)

# Bind parameters
resolved = cirq.resolve_parameters(circuit, {'theta': 0.5})
```

### LRET Handling
- LRET expects numeric values only
- **Strategy**: Require parameters to be resolved before translation
- **Error**: Raise `ValueError` if unresolved symbols detected

---

## Testing Strategy (50+ Tests)

### Test Categories (Based on Qiskit Success)

#### 1. TestLRETCirqSimulator (5 tests)
- Instantiation
- Basic run with repetitions
- Multiple circuit batch execution
- Custom epsilon values
- String representation

#### 2. TestCircuitTranslator (15 tests)
- Empty circuit
- Single-qubit gates (H, X, Y, Z, S, T)
- Two-qubit gates (CNOT, CZ, SWAP)
- Rotation gates (RX, RY, RZ)
- Power gates (XPowGate, YPowGate, ZPowGate)
- Unsupported gates (raise error)
- Qubit mapping (LineQubit, GridQubit)
- Measurement translation
- Batch translation

#### 3. TestResultConverter (5 tests)
- Sample to counts conversion
- Multiple measurement keys
- Qubit ordering preservation
- Repetitions validation
- Metadata inclusion

#### 4. TestIntegration (10 tests)
- Bell state circuit
- GHZ state (3+ qubits)
- QFT circuit
- Random circuit
- Parameterized circuit (resolved)
- Circuit with noise
- Multi-qubit measurement
- Custom repetitions
- Transpiled circuit (if applicable)
- Edge cases (single qubit, many qubits)

#### 5. TestGateSet (10 tests)
- All single-qubit gates individually
- All two-qubit gates individually
- Rotation gates with various angles
- Power gates with fractional exponents
- Gate composition (multiple gates)

#### 6. TestErrorHandling (5 tests)
- Unresolved parameters
- Unsupported gates
- Invalid circuit structure
- Measurement key conflicts
- Qubit index out of range

---

## Directory Structure Proposal

```
d:\LRET\python\lret_cirq\
├── __init__.py                      # Package init, export LRETSimulator
├── cirq_simulator.py                # Main simulator class
├── translators/
│   ├── __init__.py
│   ├── circuit_translator.py        # Cirq → LRET JSON
│   └── result_converter.py          # LRET → Cirq Result
├── gate_mapping.py                  # Cirq gate → LRET gate lookup
├── qubit_utils.py                   # Qubit mapping helpers
└── tests/
    ├── __init__.py
    ├── test_simulator.py            # Simulator tests
    ├── test_translator.py           # Translation tests
    ├── test_result_converter.py     # Result conversion tests
    └── test_integration.py          # End-to-end tests
```

---

## Implementation Phases (5-6 Days)

### Day 1: Core Infrastructure
- Create package structure
- Implement `CircuitTranslator` skeleton
- Implement basic gate mapping (H, X, Y, Z, CNOT)
- Write 10 translator unit tests

### Day 2: Simulator Class
- Implement `LRETSimulator` class
- Integrate with `qlret.simulate_json`
- Implement qubit mapping
- Write 5 simulator tests

### Day 3: Result Conversion
- Implement `ResultConverter`
- Handle measurement keys
- Handle multi-qubit measurements
- Write 5 result converter tests

### Day 4: Extended Gates
- Add rotation gates (RX, RY, RZ)
- Add S, T, SWAP gates
- Handle power gates (XPowGate, etc.)
- Write 15 gate tests

### Day 5: Integration & Testing
- End-to-end integration tests
- Bell state, GHZ state tests
- Noise model integration
- Error handling tests

### Day 6: Polish & Documentation
- Complete test coverage (50+ tests)
- Fix any failing tests
- Write user documentation
- Create usage examples

---

## Expected Challenges

### Challenge 1: Power Gates
**Problem**: Cirq uses `XPowGate(exponent=0.5)` for √X, but LRET may not support fractional powers.

**Solutions**:
1. **Option A**: Map common exponents to discrete gates (0.5 → SX, 1.0 → X)
2. **Option B**: Extend LRET to support parameterized power gates
3. **Option C**: Decompose to rotation gates (RX, RY, RZ)

**Recommendation**: Option A for initial implementation, Option B for future.

### Challenge 2: Qubit Types
**Problem**: Cirq supports LineQubit, GridQubit, NamedQubit; LRET uses integers.

**Solution**: Build mapping dict during translation, preserve qubit objects for result conversion.

### Challenge 3: Measurement Moments
**Problem**: Cirq allows measurements at any moment; LRET may expect final measurements only.

**Solution**: Support mid-circuit measurements if LRET supports it, else raise error.

### Challenge 4: Noise Models
**Problem**: Cirq noise is operation-based; LRET noise is circuit-wide.

**Solution**: Aggregate Cirq noise operations into LRET noise config (may be lossy).

---

## Success Criteria

### Minimum Viable Product (MVP)
- ✅ Users can run `from lret_cirq import LRETSimulator`
- ✅ Basic circuits (H, X, CNOT) work
- ✅ Measurements produce correct counts
- ✅ 25+ tests passing

### Full Release
- ✅ All common gates supported (H, X, Y, Z, S, T, RX, RY, RZ, CNOT, CZ, SWAP)
- ✅ Power gates supported (XPowGate, YPowGate, ZPowGate)
- ✅ Noise models work
- ✅ Qubit types handled (LineQubit, GridQubit)
- ✅ 50+ tests passing
- ✅ Documentation complete

### Stretch Goals
- ✅ Google-specific gates (FSim, Sycamore)
- ✅ Mid-circuit measurements
- ✅ Full `cirq.Simulator` interface (not just `SimulatesSamples`)
- ✅ Performance benchmarks vs native Cirq

---

## Resources for Roadmap Writer

### Existing LRET Documentation
1. **PHASE_7_DETAILED_IMPLEMENTATION_ROADMAP.md** - Qiskit integration guide (1,835 lines)
   - Use as template for structure
   - Copy section organization
   - Adapt examples for Cirq

2. **PHASE_7_QISKIT_TESTING_SUMMARY.md** - Test coverage guide (481 lines)
   - Test categories to replicate
   - Success metrics
   - Common patterns

3. **python/lret_qiskit/** - Working Qiskit integration
   - Reference implementation
   - Proven patterns
   - Test structure

### Cirq Official Documentation
- **Cirq Docs**: https://quantumai.google/cirq
- **Simulator Guide**: https://quantumai.google/cirq/simulate
- **Custom Simulators**: https://quantumai.google/cirq/simulate/custom_simulators
- **Gate Reference**: https://quantumai.google/reference/python/cirq/ops

### Key Cirq Examples
```python
# Basic Cirq workflow
import cirq

# Create qubits
q0, q1 = cirq.LineQubit.range(2)

# Build circuit
circuit = cirq.Circuit(
    cirq.H(q0),
    cirq.CNOT(q0, q1),
    cirq.measure(q0, q1, key='result')
)

# Simulate
simulator = cirq.Simulator()  # Native Cirq simulator
result = simulator.run(circuit, repetitions=100)

# Get results
print(result.histogram(key='result'))
# Output: Counter({0: 52, 3: 48})  # Binary: 00 and 11
```

---

## Deliverables for Roadmap

The comprehensive Cirq roadmap should include:

1. **Architecture Overview** (~200 lines)
   - System design
   - Component relationships
   - Data flow diagrams

2. **API Specification** (~300 lines)
   - `LRETSimulator` class full API
   - `CircuitTranslator` methods
   - `ResultConverter` methods
   - Usage examples

3. **Gate Mapping Reference** (~200 lines)
   - Complete gate table
   - Translation examples
   - Unsupported gates list

4. **Implementation Guide** (~400 lines)
   - Step-by-step code templates
   - Day-by-day breakdown
   - Code snippets for each component

5. **Testing Strategy** (~200 lines)
   - Test categories
   - Example tests
   - Coverage targets

6. **Troubleshooting Guide** (~100 lines)
   - Common errors
   - Solutions
   - Debugging tips

7. **Examples & Tutorials** (~100 lines)
   - Bell state example
   - QFT example
   - Noise simulation example

**Total**: ~1,500 lines

---

## Next Steps

### For Sonnet 4.5 (Roadmap Writing)
1. Read this context document
2. Review `PHASE_7_DETAILED_IMPLEMENTATION_ROADMAP.md` for structure
3. Review Cirq documentation
4. Create `PHASE_7_CIRQ_DETAILED_IMPLEMENTATION_ROADMAP.md` (1,500+ lines)

### For Opus 4.5 (Implementation)
1. Read the completed roadmap
2. Implement over 5-6 days following roadmap phases
3. Write 50+ tests
4. Validate with real Cirq circuits

---

**Document Version**: 1.0  
**Last Updated**: January 25, 2026  
**Branch**: phase-7  
**Status**: Ready for roadmap writing

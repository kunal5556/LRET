# Handoff to Sonnet 4.5 - Cirq Roadmap Writing

**Date**: January 25, 2026  
**From**: Opus 4.5  
**To**: Sonnet 4.5  
**Task**: Write comprehensive Cirq integration roadmap (1,500+ lines)  
**Branch**: phase-7  
**Status**: All preparation complete, ready to write

---

## What Has Been Done ✅

### 1. Qiskit Integration Complete
- **53/53 tests passing** (100%)
- Full BackendV2 implementation
- Complete documentation
- Committed and pushed to `phase-7` branch
- **Summary**: [PHASE_7_QISKIT_TESTING_SUMMARY.md](PHASE_7_QISKIT_TESTING_SUMMARY.md)

### 2. Context Document Created
- **File**: [PHASE_7_CIRQ_ROADMAP_CONTEXT.md](PHASE_7_CIRQ_ROADMAP_CONTEXT.md)
- **Size**: 627 lines
- **Content**:
  - Cirq vs Qiskit architecture comparison
  - Gate mapping reference
  - Qubit mapping strategy
  - Measurement handling
  - Noise integration
  - Testing strategy (50+ tests)
  - 5-6 day implementation phases
  - Expected challenges and solutions

### 3. Repository Prepared
- All commits pushed to remote
- Clean working directory
- Documentation up to date

---

## Your Task 📝

Create a comprehensive Cirq integration roadmap following the pattern of the successful Qiskit integration.

### Target Document
**File**: `d:\LRET\PHASE_7_CIRQ_DETAILED_IMPLEMENTATION_ROADMAP.md`

### Requirements
- **Length**: 1,500+ lines (similar to Qiskit roadmap)
- **Style**: Technical, detailed, implementation-focused
- **Audience**: Opus 4.5 who will implement over 5-6 days

### Template to Follow
Use `PHASE_7_DETAILED_IMPLEMENTATION_ROADMAP.md` as your structural template:
- Read it completely to understand format
- Copy section organization
- Adapt content for Cirq instead of Qiskit
- Maintain similar depth and detail

---

## Required Sections

### 1. Introduction (~100 lines)
- Project overview
- Why Cirq integration matters
- Comparison to Qiskit integration
- Success criteria

### 2. Architecture Overview (~200 lines)
- System design diagram (ASCII art)
- Component relationships
- Data flow: Cirq Circuit → LRET → Cirq Result
- Comparison table: Cirq vs Qiskit vs LRET

### 3. Core Components (~400 lines)

#### 3.1 LRETSimulator Class
- Inherit from `cirq.SimulatesSamples`
- Complete class structure
- All methods with signatures
- Usage examples

#### 3.2 CircuitTranslator
- Cirq Circuit → LRET JSON
- Gate mapping logic
- Qubit mapping logic
- Code templates

#### 3.3 ResultConverter
- LRET JSON → Cirq Result
- Measurement key handling
- Sample array construction
- Code templates

### 4. Gate Mapping Reference (~200 lines)
- Complete gate table (Cirq → LRET)
- Common gates (H, X, Y, Z, CNOT, etc.)
- Rotation gates (RX, RY, RZ)
- Power gates (XPowGate, YPowGate, ZPowGate)
- Google gates (FSim, Sycamore) - optional
- Translation examples for each gate

### 5. Qubit Mapping Strategy (~100 lines)
- LineQubit handling
- GridQubit handling
- NamedQubit handling
- Mapping algorithm
- Reverse mapping for results

### 6. Measurement Handling (~100 lines)
- Measurement keys
- Multi-qubit measurements
- Measurement moments
- Mid-circuit measurements (if supported)

### 7. Implementation Guide (~400 lines)

#### Day 1: Core Infrastructure
- Package structure creation
- CircuitTranslator skeleton
- Basic gate mapping (H, X, Y, Z, CNOT)
- 10 unit tests

#### Day 2: Simulator Class
- LRETSimulator implementation
- Integration with qlret.simulate_json
- Qubit mapping
- 5 simulator tests

#### Day 3: Result Conversion
- ResultConverter implementation
- Measurement key handling
- Multi-qubit measurements
- 5 result tests

#### Day 4: Extended Gates
- Rotation gates (RX, RY, RZ)
- Additional gates (S, T, SWAP)
- Power gate handling
- 15 gate tests

#### Day 5: Integration Testing
- End-to-end tests
- Bell state, GHZ state
- Complex circuits
- Error handling tests

#### Day 6: Polish & Documentation
- Complete test coverage (50+ tests)
- Fix any issues
- User documentation
- Usage examples

### 8. Testing Strategy (~200 lines)
- Test categories (based on Qiskit success)
- TestLRETCirqSimulator (5 tests)
- TestCircuitTranslator (15 tests)
- TestResultConverter (5 tests)
- TestIntegration (10 tests)
- TestGateSet (10 tests)
- TestErrorHandling (5 tests)
- Example test code for each category

### 9. API Reference (~200 lines)
- Complete class signatures
- Method documentation
- Parameter descriptions
- Return types
- Usage examples

### 10. Troubleshooting Guide (~100 lines)
- Common errors and solutions
- Debugging strategies
- Performance tips
- FAQ

### 11. Examples & Tutorials (~100 lines)
- Bell state example
- GHZ state example
- QFT example
- Noise simulation example
- Parameterized circuit example

---

## Key Resources

### Must Read Before Writing
1. **[PHASE_7_CIRQ_ROADMAP_CONTEXT.md](PHASE_7_CIRQ_ROADMAP_CONTEXT.md)** - All technical context
2. **[PHASE_7_DETAILED_IMPLEMENTATION_ROADMAP.md](PHASE_7_DETAILED_IMPLEMENTATION_ROADMAP.md)** - Template structure
3. **[PHASE_7_QISKIT_TESTING_SUMMARY.md](PHASE_7_QISKIT_TESTING_SUMMARY.md)** - Testing patterns

### Reference During Writing
4. **python/lret_qiskit/** - Working implementation to reference
5. **Cirq Documentation** - https://quantumai.google/cirq/simulate/custom_simulators

---

## Key Differences to Address

### Cirq-Specific Challenges

#### 1. Power Gates
Cirq uses `XPowGate(exponent=e)` instead of discrete gates:
```python
# Cirq
cirq.XPowGate(exponent=1.0)  # X gate
cirq.XPowGate(exponent=0.5)  # √X gate

# LRET - Need to handle this!
```

**Solution to Document**:
- Map common exponents (0.5, 1.0) to gates
- Or decompose to rotation gates
- Or extend LRET gate set

#### 2. Qubit Objects
Cirq has multiple qubit types; LRET uses integers:
```python
# Cirq
q0 = cirq.LineQubit(0)
q1 = cirq.GridQubit(0, 1)
qa = cirq.NamedQubit("alice")

# LRET expects: 0, 1, 2
```

**Solution to Document**:
- Build qubit → integer mapping during translation
- Preserve mapping for result conversion

#### 3. No Provider System
Unlike Qiskit, Cirq doesn't use Provider/Backend pattern:
```python
# Qiskit
provider = LRETProvider()
backend = provider.get_backend()
job = backend.run(circuit)

# Cirq (simpler)
sim = LRETSimulator()
result = sim.run(circuit, repetitions=100)
```

**Advantage**: Simpler architecture!

---

## Writing Guidelines

### Style
- **Technical but accessible** - Opus 4.5 needs to implement this
- **Complete code examples** - Not just pseudocode
- **Detailed explanations** - Why, not just what
- **Practical focus** - Real implementation guidance

### Code Snippets
- Use actual Python syntax
- Include imports
- Add comments explaining key points
- Show both Cirq and LRET equivalents

### Tables
- Use markdown tables for comparisons
- Keep them readable (not too wide)
- Include all relevant information

### Diagrams
- Use ASCII art for architecture diagrams
- Keep them simple but informative
- Show data flow clearly

---

## Success Criteria for Your Roadmap

### Must Have
- ✅ 1,500+ lines
- ✅ All 11 sections complete
- ✅ Complete code templates for all components
- ✅ 50+ test examples
- ✅ Day-by-day implementation guide
- ✅ Troubleshooting section

### Quality Indicators
- ✅ Opus 4.5 can implement without asking clarifications
- ✅ All Cirq-specific challenges addressed
- ✅ Clear migration path from Qiskit patterns
- ✅ Examples are copy-paste ready

### Bonus
- ✅ Performance optimization tips
- ✅ Comparison to other Cirq simulators
- ✅ Future enhancement roadmap
- ✅ Publication-ready benchmarking guide

---

## After You Complete the Roadmap

### Handoff to Opus 4.5
You will create a similar handoff document:
```markdown
# Handoff to Opus 4.5 - Cirq Implementation

**Task**: Implement Cirq integration following roadmap over 5-6 days
**Roadmap**: PHASE_7_CIRQ_DETAILED_IMPLEMENTATION_ROADMAP.md
**Goal**: 50+ tests passing, production-ready integration
...
```

### Commit & Push
```bash
cd d:\LRET
git add PHASE_7_CIRQ_DETAILED_IMPLEMENTATION_ROADMAP.md
git commit -m "Add comprehensive Cirq integration roadmap (1,500+ lines)

Complete implementation guide for LRET-Cirq integration:
- Architecture and component design
- Gate mapping reference (50+ gates)
- Day-by-day implementation plan (6 days)
- 50+ test strategy with examples
- API reference and troubleshooting guide"

git push origin phase-7
```

---

## Questions to Answer in Roadmap

1. How do we handle `XPowGate(exponent=0.5)` in LRET?
2. What's the best base class: `SimulatesSamples` or `Simulator`?
3. How do we map `GridQubit(x, y)` to integer indices?
4. Can LRET support mid-circuit measurements?
5. How do we aggregate Cirq noise operations to LRET noise config?
6. What's the performance target vs native Cirq?
7. Which Google-specific gates should we support?
8. How do we handle parameterized circuits with `sympy.Symbol`?

**Don't just ask these - answer them in the roadmap!**

---

## Example of Good Section

Here's what a good section should look like:

```markdown
### 4.2 CircuitTranslator Implementation

The `CircuitTranslator` converts Cirq circuits to LRET JSON format.

#### Architecture

                Cirq Circuit
                     ↓
         ┌───────────────────────┐
         │  Extract qubits       │
         │  Build qubit mapping  │
         └───────────────────────┘
                     ↓
         ┌───────────────────────┐
         │  Iterate moments       │
         │  Translate gates       │
         │  Collect measurements  │
         └───────────────────────┘
                     ↓
         ┌───────────────────────┐
         │  Build LRET JSON      │
         │  Add config (epsilon) │
         └───────────────────────┘
                     ↓
                 LRET JSON

#### Complete Implementation

```python
# python/lret_cirq/translators/circuit_translator.py
import cirq
from typing import Dict, List, Any

class CircuitTranslator:
    """Translate Cirq circuits to LRET JSON format."""
    
    def __init__(self):
        self._gate_map = self._build_gate_map()
    
    def translate(self, circuit: cirq.Circuit, epsilon: float = 1e-4) -> Dict[str, Any]:
        """
        Translate a Cirq circuit to LRET JSON.
        
        Args:
            circuit: cirq.Circuit to translate
            epsilon: LRET truncation threshold
            
        Returns:
            LRET JSON dict
            
        Raises:
            TranslationError: If circuit contains unsupported gates
        """
        # Step 1: Build qubit mapping
        qubit_map = self._build_qubit_map(circuit)
        num_qubits = len(qubit_map)
        
        # Step 2: Translate operations
        operations = []
        for moment in circuit:
            for op in moment:
                lret_op = self._translate_operation(op, qubit_map)
                operations.append(lret_op)
        
        # Step 3: Build JSON
        return {
            "config": {"epsilon": epsilon},
            "circuit": {
                "num_qubits": num_qubits,
                "operations": operations
            }
        }
    
    # ... more methods ...
```

**Key Design Decisions:**
- Moments are flattened (LRET doesn't have moment concept)
- Qubits mapped to contiguous integers starting at 0
- Unsupported gates raise `TranslationError` immediately

**Usage Example:**
```python
import cirq
from lret_cirq.translators import CircuitTranslator

# Create circuit
q0, q1 = cirq.LineQubit.range(2)
circuit = cirq.Circuit(
    cirq.H(q0),
    cirq.CNOT(q0, q1)
)

# Translate
translator = CircuitTranslator()
lret_json = translator.translate(circuit, epsilon=1e-4)

print(lret_json)
# Output:
# {
#   "config": {"epsilon": 0.0001},
#   "circuit": {
#     "num_qubits": 2,
#     "operations": [
#       {"name": "H", "wires": [0]},
#       {"name": "CX", "wires": [0, 1]}
#     ]
#   }
# }
```
```

---

## Final Checklist Before You Start

- [ ] Read PHASE_7_CIRQ_ROADMAP_CONTEXT.md completely
- [ ] Skim PHASE_7_DETAILED_IMPLEMENTATION_ROADMAP.md (Qiskit template)
- [ ] Understand Cirq's simulator interface from official docs
- [ ] Have a clear mental model of Cirq → LRET → Cirq data flow
- [ ] Know the target audience (Opus 4.5, who needs complete guidance)

---

## Ready to Start?

Your roadmap should be so detailed that Opus 4.5 can implement the entire Cirq integration in 5-6 days with minimal clarification questions. Think of it as a comprehensive blueprint, not just an outline.

**Good luck! Looking forward to reviewing your 1,500+ line masterpiece.**

---

**Document Version**: 1.0  
**Created**: January 25, 2026  
**Status**: Ready for Sonnet 4.5  
**Branch**: phase-7  
**Next Step**: Write PHASE_7_CIRQ_DETAILED_IMPLEMENTATION_ROADMAP.md

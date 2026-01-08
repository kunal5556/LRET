# Manual Circuit Execution Guide for LRET with OpenCode

This guide shows how to manually provide quantum circuits to OpenCode and execute them on LRET.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Circuit Format Options](#circuit-format-options)
3. [Execution Methods](#execution-methods)
4. [Using OpenCode for Circuit Execution](#using-opencode-for-circuit-execution)
5. [Advanced: Custom Circuit Templates](#advanced-custom-circuit-templates)
6. [Batch Execution](#batch-execution)
7. [Comparing Multiple Circuits](#comparing-multiple-circuits)
8. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Method 1: Direct JSON File

Create a circuit file `my_circuit.json`:

```json
{
  "circuit": {
    "num_qubits": 3,
    "operations": [
      { "name": "H", "wires": [0] },
      { "name": "H", "wires": [1] },
      { "name": "H", "wires": [2] },
      { "name": "CNOT", "wires": [0, 1] },
      { "name": "CNOT", "wires": [1, 2] }
    ],
    "observables": [
      { "type": "PAULI", "operator": "Z", "wires": [0], "coefficient": 1.0 }
    ]
  },
  "config": {
    "epsilon": 1e-4,
    "initial_rank": 1,
    "shots": 1000
  }
}
```

Execute:
```bash
./build/quantum_sim my_circuit.json
```

### Method 2: Via OpenCode Natural Language

```bash
opencode run "Execute my circuit file my_circuit.json using quantum_sim and show me the results"
```

---

## Circuit Format Options

### Option 1: LRET JSON Format

**Structure:**
```json
{
  "circuit": {
    "num_qubits": <int>,
    "operations": [
      { "name": "<gate>", "wires": [<qubit_indices>], "params": [<float>] }
    ],
    "observables": [
      { "type": "<type>", "operator": "<pauli>", "wires": [<indices>] }
    ]
  },
  "config": {
    "epsilon": <float>,          // Rank truncation threshold
    "initial_rank": <int>,       // Starting rank
    "shots": <int>,              // Measurement shots
    "export_state": <bool>,      // Save final state
    "noise_model": {...}         // Optional noise
  }
}
```

**Supported Gates:**
- Single-qubit: `H`, `X`, `Y`, `Z`, `RX`, `RY`, `RZ`, `S`, `T`
- Two-qubit: `CNOT`, `CZ`, `SWAP`, `CRX`, `CRY`, `CRZ`
- Three-qubit: `TOFFOLI`, `FREDKIN`

**Example - GHZ State:**
```json
{
  "circuit": {
    "num_qubits": 5,
    "operations": [
      { "name": "H", "wires": [0] },
      { "name": "CNOT", "wires": [0, 1] },
      { "name": "CNOT", "wires": [1, 2] },
      { "name": "CNOT", "wires": [2, 3] },
      { "name": "CNOT", "wires": [3, 4] }
    ]
  },
  "config": {
    "epsilon": 1e-6,
    "shots": 10000
  }
}
```

### Option 2: Python API

**Using qlret package:**
```python
from qlret import QLRETDevice, simulate_json
import json

# Method A: From JSON file
circuit_data = json.load(open('my_circuit.json'))
result = simulate_json(circuit_data)
print(f"Expectation values: {result['observables']}")

# Method B: Using PennyLane
import pennylane as qml

dev = QLRETDevice(wires=3, epsilon=1e-4)

@qml.qnode(dev)
def my_circuit():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    return qml.expval(qml.PauliZ(0))

result = my_circuit()
print(f"Result: {result}")
```

### Option 3: Describe to OpenCode

**Natural language:**
```bash
opencode run "
Create and execute a quantum circuit:
- 4 qubits
- Apply Hadamard to all qubits
- Apply CNOT between adjacent qubits (0-1, 1-2, 2-3)
- Measure all qubits
- Run with 1000 shots
- Save results to my_results.json
"
```

---

## Execution Methods

### Method 1: Direct Binary Execution

```bash
# Basic execution
./build/quantum_sim my_circuit.json

# With output to file
./build/quantum_sim my_circuit.json > results.txt

# Multiple circuits
for circuit in circuits/*.json; do
    echo "Running $circuit..."
    ./build/quantum_sim "$circuit"
done
```

### Method 2: Python Script Execution

Create `run_circuit.py`:
```python
#!/usr/bin/env python3
import sys
import json
from qlret import simulate_json

if __name__ == "__main__":
    circuit_file = sys.argv[1]
    
    with open(circuit_file) as f:
        circuit_data = json.load(f)
    
    result = simulate_json(circuit_data)
    
    print(f"Circuit: {circuit_file}")
    print(f"Observables: {result['observables']}")
    print(f"Execution time: {result.get('time_ms', 'N/A')} ms")
```

Execute:
```bash
python3 run_circuit.py my_circuit.json
```

### Method 3: OpenCode-Assisted Execution

```bash
# Execute and analyze
opencode run "
Run my_circuit.json and:
1. Show execution time
2. Display measurement results
3. Plot the probability distribution
4. Compare with theoretical expectations
"

# Batch execution with comparison
opencode run "
Execute all circuits in circuits/ directory:
- Run each circuit on LRET
- Record time and results
- Create a comparison table
- Save to circuits_comparison.csv
"
```

---

## Using OpenCode for Circuit Execution

### Scenario 1: Create Circuit from Description

```bash
opencode run "
I want to create a quantum circuit that:
1. Creates a W-state on 4 qubits
2. Applies some noise (depolarizing, p=0.01)
3. Measures in the computational basis

Steps:
1. Create the circuit JSON file at circuits/w_state_4q.json
2. Execute it using quantum_sim
3. Show me the measurement probabilities
4. Plot the results
"
```

### Scenario 2: Modify Existing Circuit

```bash
opencode run "
Take the Bell pair circuit at samples/json/bell_pair.json and:
1. Increase to 4 qubits (make 2 Bell pairs)
2. Add entangling gate between the pairs
3. Save as circuits/double_bell.json
4. Execute and compare fidelity with single Bell pair
"
```

### Scenario 3: Circuit Optimization

```bash
opencode run "
I have a circuit at circuits/my_deep_circuit.json.
Analyze it and:
1. Check for gate cancellations
2. Suggest more efficient gate sequence
3. Create optimized version
4. Run both and compare execution times
"
```

### Scenario 4: Parameter Sweep

```bash
opencode run "
For the circuit circuits/parametric_circuit.json:
1. Sweep the RX gate angle from 0 to 2π in 20 steps
2. Run each variant
3. Plot expectation value vs angle
4. Find the angle that maximizes <Z>
"
```

### Scenario 5: Noise Analysis

```bash
opencode run "
Take circuits/ghz_5q.json and:
1. Run without noise (baseline)
2. Run with depolarizing noise: p = [0.001, 0.01, 0.05, 0.1]
3. For each noise level, run 10 times
4. Plot fidelity vs noise level
5. Save data to noise_analysis.csv
"
```

---

## Advanced: Custom Circuit Templates

### Template 1: Variational Circuit Generator

Create `generate_variational_circuit.py`:
```python
#!/usr/bin/env python3
import json
import numpy as np

def variational_circuit(num_qubits, num_layers, params=None):
    """Generate a variational quantum circuit"""
    if params is None:
        params = np.random.random((num_layers, num_qubits, 3))
    
    operations = []
    
    for layer in range(num_layers):
        # Rotation layer
        for q in range(num_qubits):
            operations.append({
                "name": "RX",
                "wires": [q],
                "params": [params[layer, q, 0]]
            })
            operations.append({
                "name": "RY",
                "wires": [q],
                "params": [params[layer, q, 1]]
            })
            operations.append({
                "name": "RZ",
                "wires": [q],
                "params": [params[layer, q, 2]]
            })
        
        # Entangling layer
        for q in range(num_qubits - 1):
            operations.append({
                "name": "CNOT",
                "wires": [q, q + 1]
            })
    
    circuit = {
        "circuit": {
            "num_qubits": num_qubits,
            "operations": operations,
            "observables": [
                {"type": "PAULI", "operator": "Z", "wires": [0], "coefficient": 1.0}
            ]
        },
        "config": {
            "epsilon": 1e-4,
            "shots": 1000
        }
    }
    
    return circuit

if __name__ == "__main__":
    import sys
    num_qubits = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    num_layers = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    output_file = sys.argv[3] if len(sys.argv) > 3 else "variational_circuit.json"
    
    circuit = variational_circuit(num_qubits, num_layers)
    
    with open(output_file, 'w') as f:
        json.dump(circuit, f, indent=2)
    
    print(f"Generated {num_qubits}-qubit, {num_layers}-layer circuit")
    print(f"Saved to: {output_file}")
```

**Usage:**
```bash
# Generate circuit
python3 generate_variational_circuit.py 6 4 circuits/vqe_circuit.json

# Execute with OpenCode
opencode run "Execute circuits/vqe_circuit.json and optimize the parameters to minimize <Z>"
```

### Template 2: Quantum Fourier Transform

Create `generate_qft.py`:
```python
#!/usr/bin/env python3
import json
import math

def qft_circuit(num_qubits):
    """Generate Quantum Fourier Transform circuit"""
    operations = []
    
    for target in range(num_qubits):
        # Hadamard on target
        operations.append({"name": "H", "wires": [target]})
        
        # Controlled rotations
        for control in range(target + 1, num_qubits):
            k = control - target + 1
            angle = 2 * math.pi / (2 ** k)
            operations.append({
                "name": "CRZ",
                "wires": [control, target],
                "params": [angle]
            })
    
    # Swap qubits to reverse order
    for i in range(num_qubits // 2):
        operations.append({
            "name": "SWAP",
            "wires": [i, num_qubits - 1 - i]
        })
    
    circuit = {
        "circuit": {
            "num_qubits": num_qubits,
            "operations": operations
        },
        "config": {
            "epsilon": 1e-6,
            "shots": 1000
        }
    }
    
    return circuit

if __name__ == "__main__":
    import sys
    num_qubits = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    output_file = sys.argv[2] if len(sys.argv) > 2 else f"qft_{num_qubits}q.json"
    
    circuit = qft_circuit(num_qubits)
    
    with open(output_file, 'w') as f:
        json.dump(circuit, f, indent=2)
    
    print(f"Generated {num_qubits}-qubit QFT circuit")
    print(f"Saved to: {output_file}")
```

**Usage:**
```bash
# Generate QFT for 8 qubits
python3 generate_qft.py 8 circuits/qft_8q.json

# Execute with OpenCode
opencode run "Execute circuits/qft_8q.json and measure execution time"
```

---

## Batch Execution

### Script: Batch Circuit Runner

Create `batch_execute.sh`:
```bash
#!/bin/bash
# Batch execute multiple circuits

CIRCUIT_DIR="${1:-circuits}"
OUTPUT_DIR="${2:-results}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$OUTPUT_DIR"

echo "Batch Circuit Execution - $TIMESTAMP"
echo "Circuit Directory: $CIRCUIT_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "========================================"

TOTAL=0
SUCCESS=0
FAILED=0

for circuit_file in "$CIRCUIT_DIR"/*.json; do
    if [ ! -f "$circuit_file" ]; then
        continue
    fi
    
    TOTAL=$((TOTAL + 1))
    circuit_name=$(basename "$circuit_file" .json)
    output_file="$OUTPUT_DIR/${circuit_name}_$TIMESTAMP.txt"
    
    echo ""
    echo "[$TOTAL] Executing: $circuit_name"
    
    if ./build/quantum_sim "$circuit_file" > "$output_file" 2>&1; then
        echo "    ✅ Success"
        SUCCESS=$((SUCCESS + 1))
    else
        echo "    ❌ Failed"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "========================================"
echo "Summary:"
echo "  Total:   $TOTAL"
echo "  Success: $SUCCESS"
echo "  Failed:  $FAILED"
echo "========================================"
echo "Results saved to: $OUTPUT_DIR"
```

**Usage:**
```bash
chmod +x batch_execute.sh

# Execute all circuits in circuits/ directory
./batch_execute.sh circuits results

# With OpenCode
opencode run "Execute batch_execute.sh and summarize the results in a table"
```

---

## Comparing Multiple Circuits

### Script: Circuit Comparison

Create `compare_circuits.py`:
```python
#!/usr/bin/env python3
import json
import subprocess
import time
from pathlib import Path
import pandas as pd

def execute_circuit(circuit_file):
    """Execute a circuit and return timing/results"""
    start = time.time()
    
    try:
        result = subprocess.run(
            ['./build/quantum_sim', circuit_file],
            capture_output=True,
            text=True,
            timeout=60
        )
        elapsed = time.time() - start
        
        return {
            'success': result.returncode == 0,
            'time': elapsed,
            'output': result.stdout
        }
    except Exception as e:
        return {
            'success': False,
            'time': -1,
            'error': str(e)
        }

def compare_circuits(circuit_files, output_csv='circuit_comparison.csv'):
    """Compare multiple circuits"""
    results = []
    
    for circuit_file in circuit_files:
        circuit_name = Path(circuit_file).stem
        print(f"Running {circuit_name}...")
        
        # Load circuit info
        with open(circuit_file) as f:
            circuit_data = json.load(f)
        
        num_qubits = circuit_data['circuit']['num_qubits']
        num_gates = len(circuit_data['circuit']['operations'])
        
        # Execute
        exec_result = execute_circuit(circuit_file)
        
        results.append({
            'Circuit': circuit_name,
            'Qubits': num_qubits,
            'Gates': num_gates,
            'Time (s)': exec_result['time'],
            'Success': exec_result['success']
        })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    
    print(f"\nComparison saved to: {output_csv}")
    print("\nSummary:")
    print(df.to_string(index=False))
    
    return df

if __name__ == "__main__":
    import sys
    import glob
    
    if len(sys.argv) > 1:
        # Specific files
        circuit_files = sys.argv[1:]
    else:
        # All circuits in circuits/
        circuit_files = glob.glob('circuits/*.json')
    
    compare_circuits(circuit_files)
```

**Usage:**
```bash
# Compare specific circuits
python3 compare_circuits.py circuit1.json circuit2.json circuit3.json

# Compare all circuits
python3 compare_circuits.py circuits/*.json

# With OpenCode
opencode run "Run compare_circuits.py on all circuits in circuits/ and create a bar plot of execution times"
```

---

## Troubleshooting

### Common Issues

#### Issue 1: Invalid JSON Format
```bash
opencode run "Validate my circuit file circuits/my_circuit.json and fix any JSON syntax errors"
```

#### Issue 2: Gate Not Recognized
```bash
opencode run "Check if all gates in circuits/my_circuit.json are supported by LRET. List any unsupported gates."
```

#### Issue 3: Circuit Too Large
```bash
opencode run "Analyze circuits/large_circuit.json and suggest optimizations to reduce gate count or depth"
```

#### Issue 4: Numerical Precision Issues
```bash
opencode run "
Run circuits/precision_test.json with different epsilon values:
- 1e-3, 1e-4, 1e-5, 1e-6, 1e-8
Compare results and recommend optimal epsilon
"
```

---

## Integration with OpenCode

### Full Workflow Example

```bash
opencode run "
Complete workflow:

1. Create a 6-qubit GHZ state circuit
   - Save as circuits/ghz_6q.json
   
2. Create a noisy version (depolarizing, p=0.01)
   - Save as circuits/ghz_6q_noisy.json
   
3. Execute both circuits:
   - Run each 5 times
   - Record times and fidelities
   
4. Analyze results:
   - Compare clean vs noisy
   - Calculate fidelity loss due to noise
   - Plot probability distributions
   
5. Generate report:
   - Summary statistics
   - Visualization plots
   - Save to ghz_analysis_report.pdf
   
Proceed step-by-step, showing results after each phase.
"
```

---

## Quick Reference Commands

```bash
# Create circuit from description
opencode run "Create a 5-qubit circuit that implements <description>"

# Execute single circuit
./build/quantum_sim my_circuit.json

# Execute with Python
python3 -c "from qlret import simulate_json; import json; print(simulate_json(json.load(open('my_circuit.json'))))"

# Batch execution
./batch_execute.sh circuits/ results/

# Comparison
python3 compare_circuits.py circuits/*.json

# With OpenCode analysis
opencode run "Execute my_circuit.json, analyze the results, and explain what the circuit does"
```

---

## Summary

**Three Ways to Execute Circuits:**

1. **Direct:** `./build/quantum_sim circuit.json`
2. **Python:** `from qlret import simulate_json`
3. **OpenCode:** `opencode run "Execute circuit.json"`

**Circuit Sources:**
- Hand-written JSON files
- Python generators (QFT, variational, etc.)
- OpenCode natural language creation
- Converted from other frameworks (Cirq, Qiskit)

**Best Practices:**
- Validate JSON before execution
- Start with small circuits, scale up
- Use appropriate epsilon for precision
- Save results with timestamps
- Compare multiple runs for statistical significance

**Next Steps:**
- Try the examples in this guide
- Create your own circuit templates
- Use OpenCode for automated workflows
- Integrate with Cirq comparison (CIRQ_COMPARISON_GUIDE.md)

# Python Examples

This directory contains comprehensive Python examples demonstrating various quantum algorithms and features of the LRET simulator.

## Prerequisites

```bash
pip install qlret numpy matplotlib pennylane
```

## Examples Overview

### 1. Bell State Creation ([01_bell_state.py](01_bell_state.py))
**Difficulty:** Beginner  
**Concepts:** Superposition, entanglement, Bell states

Creates and visualizes Bell states (maximally entangled two-qubit states). Demonstrates basic gate operations and measurement.

```bash
python 01_bell_state.py
```

**Output:** Probability distributions and visualization plots for all four Bell states.

---

### 2. GHZ State Creation ([02_ghz_state.py](02_ghz_state.py))
**Difficulty:** Beginner  
**Concepts:** Multi-qubit entanglement, GHZ states

Creates Greenberger-Horne-Zeilinger states for various numbers of qubits, demonstrating multi-qubit entanglement.

```bash
python 02_ghz_state.py
```

**Output:** Measurement statistics and correlation analysis for GHZ states.

---

### 3. Quantum Fourier Transform ([03_qft.py](03_qft.py))
**Difficulty:** Intermediate  
**Concepts:** QFT, basis transformations, quantum algorithms

Implements the Quantum Fourier Transform and its inverse, demonstrating basis transformations and period finding.

```bash
python 03_qft.py
```

**Output:** QFT visualizations, phase encoding demonstrations, period finding examples.

---

### 4. Noisy Simulation ([04_noisy_simulation.py](04_noisy_simulation.py))
**Difficulty:** Intermediate  
**Concepts:** Noise models, decoherence, error analysis

Compares different noise models (depolarizing, amplitude damping, phase damping) and their effects on quantum circuits.

```bash
python 04_noisy_simulation.py
```

**Output:** Fidelity plots, noise comparison charts, decoherence analysis.

---

### 5. Variational Quantum Eigensolver ([05_vqe.py](05_vqe.py))
**Difficulty:** Advanced  
**Concepts:** VQE, optimization, quantum chemistry

Implements VQE to find ground state energies of molecular Hamiltonians, including H₂ molecule simulation.

```bash
python 05_vqe.py
```

**Output:** Energy convergence plots, ansatz comparisons, optimization trajectories.

---

### 6. PennyLane Integration ([06_pennylane_integration.py](06_pennylane_integration.py))
**Difficulty:** Intermediate  
**Concepts:** PennyLane, automatic differentiation, variational algorithms

Demonstrates how to use LRET as a PennyLane device for quantum machine learning and variational algorithms.

```bash
python 06_pennylane_integration.py
```

**Output:** Gradient computations, VQE with PennyLane, QAOA examples, noisy device comparisons.

---

### 7. Quantum Teleportation ([07_quantum_teleportation.py](07_quantum_teleportation.py))
**Difficulty:** Intermediate  
**Concepts:** Quantum teleportation, entanglement, Bell measurements

Implements the quantum teleportation protocol and superdense coding.

```bash
python 07_quantum_teleportation.py
```

**Output:** Teleportation success rates, fidelity under noise, superdense coding demonstrations.

---

### 8. Grover's Search Algorithm ([08_grover_search.py](08_grover_search.py))
**Difficulty:** Advanced  
**Concepts:** Grover's algorithm, amplitude amplification, quantum search

Implements Grover's search algorithm with analysis of iteration count, scaling, and multiple solutions.

```bash
python 08_grover_search.py
```

**Output:** Success probability plots, iteration analysis, scaling demonstrations, multiple solution examples.

---

### 9. Quantum Phase Estimation ([09_phase_estimation.py](09_phase_estimation.py))
**Difficulty:** Advanced  
**Concepts:** Phase estimation, QFT, eigenvalue problems

Implements quantum phase estimation (QPE) algorithm with precision analysis.

```bash
python 09_phase_estimation.py
```

**Output:** Precision analysis, phase kickback demonstrations, accuracy plots.

---

## Running All Examples

To run all examples sequentially:

```bash
for file in *.py; do
    echo "Running $file..."
    python "$file"
    echo "---"
done
```

Or on Windows:

```powershell
Get-ChildItem -Filter *.py | ForEach-Object {
    Write-Host "Running $($_.Name)..."
    python $_.Name
    Write-Host "---"
}
```

## Example Categories

### Beginner Examples
- Bell state creation
- GHZ state creation
- Basic measurements

### Intermediate Examples
- Quantum Fourier Transform
- Noisy simulation
- PennyLane integration
- Quantum teleportation

### Advanced Examples
- Variational Quantum Eigensolver
- Grover's search algorithm
- Quantum phase estimation

## Common Patterns

### Basic Circuit Pattern
```python
from qlret import QuantumSimulator

# Create simulator
sim = QuantumSimulator(n_qubits=2)

# Apply gates
sim.h(0)
sim.cx(0, 1)

# Measure
result = sim.measure()
```

### Noisy Simulation Pattern
```python
# Create simulator with noise
sim = QuantumSimulator(n_qubits=2, noise_level=0.01)

# Circuit remains the same
sim.h(0)
sim.cx(0, 1)
```

### PennyLane Integration Pattern
```python
import pennylane as qml

# Create LRET device
dev = qml.device("qlret.simulator", wires=2)

# Define QNode
@qml.qnode(dev)
def circuit(params):
    qml.RY(params[0], wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(0))
```

### Expectation Value Pattern
```python
# Measure expectation value of Pauli operator
expectation = sim.expectation_value('Z', [0])

# Multiple qubits
expectation_zz = sim.expectation_value('ZZ', [0, 1])
```

## Visualization

All examples that generate plots save them to the current directory with descriptive filenames:

- `bell_states.png` - Bell state distributions
- `ghz_correlations.png` - GHZ state correlations
- `qft_transformation.png` - QFT visualizations
- `noise_comparison.png` - Noise model comparisons
- `vqe_convergence.png` - VQE optimization
- `pennylane_vqe.png` - PennyLane VQE results
- `teleportation_noise.png` - Teleportation fidelity
- `grover_iterations.png` - Grover iteration analysis
- `qpe_precision.png` - QPE precision analysis

## Performance Tips

### For Large Simulations
```python
# Use lower precision for faster computation
sim = QuantumSimulator(n_qubits, precision='single')

# Disable intermediate state storage
sim = QuantumSimulator(n_qubits, store_history=False)
```

### For Noisy Simulations
```python
# Use density matrix for better noise modeling
sim = QuantumSimulator(n_qubits, use_density_matrix=True, noise_level=0.01)
```

### For Parallel Execution
```python
# Enable multithreading
sim = QuantumSimulator(n_qubits, num_threads=4)
```

## Troubleshooting

### Memory Issues
If you encounter memory errors with large qubit counts:
```python
# Reduce qubit count or use sparse matrices
sim = QuantumSimulator(n_qubits, sparse=True)
```

### Slow Execution
For slow simulations:
```python
# Check simulation backend
print(sim.backend)  # Should show 'AVX2' or 'SIMD' for best performance

# Reduce shot count
n_shots = 100  # Instead of 1000
```

### Import Errors
```bash
# Ensure LRET is installed
pip install qlret

# For PennyLane examples
pip install pennylane

# For visualization
pip install matplotlib
```

## Further Reading

- [LRET User Guide](../../user-guide/)
- [Python API Reference](../../api-reference/python/)
- [Developer Guide](../../developer-guide/)
- [Jupyter Notebooks](../jupyter/) - Interactive tutorials

## Contributing

Found a bug or want to add an example? See our [Contributing Guide](../../developer-guide/07-contributing.md).

## License

These examples are part of the LRET project and are released under the same license. See [LICENSE](../../../LICENSE) for details.

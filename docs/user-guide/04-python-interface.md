# Python Interface

Complete Python API reference for LRET quantum simulator.

## Installation

```bash
pip install qlret  # When available on PyPI
# OR
cd LRET/python && pip install -e .
```

---

## Quick Start

```python
from qlret import QuantumSimulator

# Create simulator
sim = QuantumSimulator(n_qubits=4, noise_level=0.01)

# Apply gates
sim.h(0)
sim.cnot(0, 1)

# Measure
results = sim.measure_all(shots=1000)
print(results)
```

---

## QuantumSimulator Class

### Constructor

```python
QuantumSimulator(
    n_qubits: int,
    noise_level: float = 0.0,
    noise_model: Optional[Union[str, dict]] = None,
    parallel_mode: str = "hybrid",
    truncation_threshold: float = 1e-4,
    max_rank: Optional[int] = None,
    seed: Optional[int] = None,
    verbose: bool = False
)
```

**Parameters:**
- `n_qubits` (int): Number of qubits to simulate
- `noise_level` (float): Global depolarizing noise level (0.0-1.0)
- `noise_model` (str | dict | None): Custom noise model (file path or dict)
- `parallel_mode` (str): Parallelization mode (`"sequential"`, `"row"`, `"column"`, `"hybrid"`)
- `truncation_threshold` (float): SVD truncation threshold for rank reduction
- `max_rank` (int | None): Maximum allowed rank (hard limit)
- `seed` (int | None): Random seed for reproducibility
- `verbose` (bool): Enable verbose output

**Example:**
```python
# Basic simulator
sim = QuantumSimulator(n_qubits=8, noise_level=0.01)

# Custom configuration
sim = QuantumSimulator(
    n_qubits=10,
    noise_level=0.02,
    parallel_mode="hybrid",
    truncation_threshold=1e-5,
    max_rank=100,
    seed=42,
    verbose=True
)

# IBM device noise
sim = QuantumSimulator(
    n_qubits=5,
    noise_model="ibmq_manila.json"
)
```

---

### Properties

#### `n_qubits`
```python
@property
def n_qubits(self) -> int
```
Number of qubits in the simulator.

**Example:**
```python
sim = QuantumSimulator(n_qubits=8)
print(sim.n_qubits)  # 8
```

#### `current_rank`
```python
@property
def current_rank(self) -> int
```
Current rank of the density matrix factorization.

**Example:**
```python
sim = QuantumSimulator(n_qubits=8, noise_level=0.01)
sim.h(0)
sim.cnot(0, 1)
print(sim.current_rank)  # e.g., 4
```

#### `fidelity`
```python
@property
def fidelity(self) -> float
```
Fidelity of current state vs initial state (if tracked).

**Example:**
```python
sim = QuantumSimulator(n_qubits=4, noise_level=0.01)
sim.h(0)
print(sim.fidelity)  # e.g., 0.9998
```

#### `last_runtime`
```python
@property
def last_runtime(self) -> float
```
Runtime of last gate application (seconds).

**Example:**
```python
sim = QuantumSimulator(n_qubits=10)
sim.h(0)
print(f"Gate took {sim.last_runtime:.6f} seconds")
```

---

### Single-Qubit Gates

#### `h(qubit: int)`
Apply Hadamard gate.

$$H = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$$

```python
sim.h(0)  # Hadamard on qubit 0
```

#### `x(qubit: int)`
Apply Pauli-X (NOT) gate.

$$X = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$$

```python
sim.x(1)  # X on qubit 1
```

#### `y(qubit: int)`
Apply Pauli-Y gate.

$$Y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}$$

```python
sim.y(2)  # Y on qubit 2
```

#### `z(qubit: int)`
Apply Pauli-Z gate.

$$Z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$$

```python
sim.z(3)  # Z on qubit 3
```

#### `rx(angle: float, qubit: int)`
Apply X-rotation gate.

$$R_X(\theta) = \begin{pmatrix} \cos(\theta/2) & -i\sin(\theta/2) \\ -i\sin(\theta/2) & \cos(\theta/2) \end{pmatrix}$$

```python
import numpy as np
sim.rx(np.pi/4, 0)  # RX(π/4) on qubit 0
```

#### `ry(angle: float, qubit: int)`
Apply Y-rotation gate.

$$R_Y(\theta) = \begin{pmatrix} \cos(\theta/2) & -\sin(\theta/2) \\ \sin(\theta/2) & \cos(\theta/2) \end{pmatrix}$$

```python
sim.ry(0.5, 1)  # RY(0.5) on qubit 1
```

#### `rz(angle: float, qubit: int)`
Apply Z-rotation gate.

$$R_Z(\theta) = \begin{pmatrix} e^{-i\theta/2} & 0 \\ 0 & e^{i\theta/2} \end{pmatrix}$$

```python
sim.rz(1.2, 2)  # RZ(1.2) on qubit 2
```

#### `s(qubit: int)`
Apply S gate (phase gate).

$$S = \begin{pmatrix} 1 & 0 \\ 0 & i \end{pmatrix}$$

```python
sim.s(0)
```

#### `sdg(qubit: int)`
Apply S† gate (conjugate of S).

$$S^\dagger = \begin{pmatrix} 1 & 0 \\ 0 & -i \end{pmatrix}$$

```python
sim.sdg(0)
```

#### `t(qubit: int)`
Apply T gate (π/8 gate).

$$T = \begin{pmatrix} 1 & 0 \\ 0 & e^{i\pi/4} \end{pmatrix}$$

```python
sim.t(0)
```

#### `tdg(qubit: int)`
Apply T† gate (conjugate of T).

$$T^\dagger = \begin{pmatrix} 1 & 0 \\ 0 & e^{-i\pi/4} \end{pmatrix}$$

```python
sim.tdg(0)
```

---

### Two-Qubit Gates

#### `cnot(control: int, target: int)`
Apply CNOT (controlled-X) gate.

$$\text{CNOT} = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{pmatrix}$$

```python
sim.cnot(0, 1)  # Control: qubit 0, Target: qubit 1
```

#### `cz(control: int, target: int)`
Apply CZ (controlled-Z) gate.

$$\text{CZ} = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & -1 \end{pmatrix}$$

```python
sim.cz(0, 1)
```

#### `swap(qubit1: int, qubit2: int)`
Apply SWAP gate.

$$\text{SWAP} = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}$$

```python
sim.swap(0, 1)
```

#### `crx(angle: float, control: int, target: int)`
Apply controlled X-rotation.

```python
sim.crx(np.pi/2, 0, 1)
```

#### `cry(angle: float, control: int, target: int)`
Apply controlled Y-rotation.

```python
sim.cry(0.5, 0, 1)
```

#### `crz(angle: float, control: int, target: int)`
Apply controlled Z-rotation.

```python
sim.crz(1.2, 0, 1)
```

---

### Three-Qubit Gates

#### `toffoli(control1: int, control2: int, target: int)`
Apply Toffoli (CCNOT) gate.

```python
sim.toffoli(0, 1, 2)  # Controls: 0, 1; Target: 2
```

#### `fredkin(control: int, target1: int, target2: int)`
Apply Fredkin (CSWAP) gate.

```python
sim.fredkin(0, 1, 2)  # Control: 0; Swap: 1, 2
```

---

### Measurement

#### `measure(qubit: int, shots: int = 1) -> dict`
Measure a single qubit.

**Parameters:**
- `qubit` (int): Qubit index to measure
- `shots` (int): Number of measurement shots

**Returns:**
- `dict`: Measurement outcomes with counts

**Example:**
```python
sim = QuantumSimulator(n_qubits=4)
sim.h(0)
results = sim.measure(0, shots=1000)
print(results)  # {'0': 502, '1': 498}
```

#### `measure_all(shots: int = 1) -> dict`
Measure all qubits.

**Parameters:**
- `shots` (int): Number of measurement shots

**Returns:**
- `dict`: Measurement outcomes (bitstrings) with counts

**Example:**
```python
sim = QuantumSimulator(n_qubits=4)
sim.h(0)
sim.cnot(0, 1)
results = sim.measure_all(shots=1000)
print(results)  # {'0000': 256, '0011': 244, '1100': 251, '1111': 249}
```

---

### State Inspection

#### `get_state() -> np.ndarray`
Get current state vector (converts from density matrix).

**Returns:**
- `np.ndarray`: State vector of shape `(2^n,)`

**Example:**
```python
sim = QuantumSimulator(n_qubits=2)
sim.h(0)
sim.cnot(0, 1)
state = sim.get_state()
print(state)  # Array of length 4
```

#### `get_density_matrix() -> np.ndarray`
Get full density matrix (constructs from low-rank form).

**Returns:**
- `np.ndarray`: Density matrix of shape `(2^n, 2^n)`

**Warning:** Memory intensive for large `n`!

**Example:**
```python
sim = QuantumSimulator(n_qubits=3, noise_level=0.01)
sim.h(0)
rho = sim.get_density_matrix()
print(rho.shape)  # (8, 8)
print(np.trace(rho))  # Should be ≈ 1.0
```

#### `get_reduced_density_matrix(qubits: list) -> np.ndarray`
Get reduced density matrix for subset of qubits.

**Parameters:**
- `qubits` (list): List of qubit indices

**Returns:**
- `np.ndarray`: Reduced density matrix

**Example:**
```python
sim = QuantumSimulator(n_qubits=4, noise_level=0.01)
sim.h(0)
sim.cnot(0, 1)
# Get reduced density matrix for qubits 0 and 1
rho_01 = sim.get_reduced_density_matrix([0, 1])
print(rho_01.shape)  # (4, 4)
```

#### `get_expectation(observable: Union[str, np.ndarray], qubits: Optional[list] = None) -> float`
Compute expectation value of an observable.

**Parameters:**
- `observable` (str | ndarray): Observable name (`"X"`, `"Y"`, `"Z"`) or matrix
- `qubits` (list | None): Qubits to apply observable (default: all)

**Returns:**
- `float`: Expectation value

**Example:**
```python
sim = QuantumSimulator(n_qubits=2)
sim.h(0)
sim.cnot(0, 1)

# Pauli-Z expectation on qubit 0
exp_z = sim.get_expectation("Z", qubits=[0])
print(exp_z)

# Custom observable
import numpy as np
obs = np.array([[1, 0], [0, -1]])  # Z operator
exp = sim.get_expectation(obs, qubits=[0])
```

---

### Circuit Management

#### `reset()`
Reset simulator to initial state (|0⟩^n).

```python
sim = QuantumSimulator(n_qubits=4)
sim.h(0)
sim.cnot(0, 1)
print(sim.current_rank)  # e.g., 4

sim.reset()
print(sim.current_rank)  # 1
```

#### `get_circuit_depth() -> int`
Get current circuit depth (number of gates applied).

```python
sim = QuantumSimulator(n_qubits=4)
sim.h(0)
sim.cnot(0, 1)
sim.h(2)
print(sim.get_circuit_depth())  # 3
```

#### `print_circuit()`
Print ASCII circuit diagram.

```python
sim = QuantumSimulator(n_qubits=3)
sim.h(0)
sim.cnot(0, 1)
sim.cnot(1, 2)
sim.print_circuit()
```

**Output:**
```
q0: ──H────●─────
           │
q1: ───────X──●──
              │
q2: ──────────X──
```

#### `to_qiskit() -> qiskit.QuantumCircuit`
Convert to Qiskit circuit (requires `qiskit`).

```python
from qlret import QuantumSimulator

sim = QuantumSimulator(n_qubits=3)
sim.h(0)
sim.cnot(0, 1)

qiskit_circuit = sim.to_qiskit()
print(qiskit_circuit)
```

---

### Utilities

#### `save_state(filename: str)`
Save current state to file.

**Parameters:**
- `filename` (str): Output file path (`.npy` or `.h5`)

```python
sim = QuantumSimulator(n_qubits=8, noise_level=0.01)
sim.h(0)
sim.cnot(0, 1)
sim.save_state("my_state.npy")
```

#### `load_state(filename: str)`
Load state from file.

**Parameters:**
- `filename` (str): Input file path

```python
sim = QuantumSimulator(n_qubits=8)
sim.load_state("my_state.npy")
```

#### `compute_fidelity(other: Union[QuantumSimulator, np.ndarray]) -> float`
Compute fidelity with another state.

**Parameters:**
- `other` (QuantumSimulator | ndarray): Reference state

**Returns:**
- `float`: Fidelity (0-1)

```python
sim1 = QuantumSimulator(n_qubits=4, noise_level=0.0)
sim1.h(0)
sim1.cnot(0, 1)

sim2 = QuantumSimulator(n_qubits=4, noise_level=0.01)
sim2.h(0)
sim2.cnot(0, 1)

fidelity = sim1.compute_fidelity(sim2)
print(f"Fidelity: {fidelity:.4f}")  # e.g., 0.9998
```

---

## Noise Models

### `load_noise_model(source: Union[str, dict]) -> dict`
Load noise model from file or dict.

**Parameters:**
- `source` (str | dict): File path or noise model dict

**Returns:**
- `dict`: Noise model specification

**Example:**
```python
from qlret import load_noise_model, QuantumSimulator

# Load from JSON file
noise = load_noise_model("ibmq_manila.json")
sim = QuantumSimulator(n_qubits=5, noise_model=noise)

# Load from dict
noise_dict = {
    "global_depolarizing": 0.01,
    "gate_errors": {"H": 0.0005, "CNOT": 0.01}
}
sim = QuantumSimulator(n_qubits=4, noise_model=noise_dict)
```

### Creating Custom Noise Models

```python
import json

# Define custom noise
custom_noise = {
    "model_type": "mixed",
    "global_depolarizing": 0.001,
    "gate_specific": {
        "H": {"depolarizing": 0.0005},
        "CNOT": {"depolarizing": 0.01},
        "RX": {"amplitude_damping": 0.002, "phase_damping": 0.001}
    },
    "qubit_specific": {
        "0": {"T1": 50e-6, "T2": 70e-6},
        "1": {"T1": 45e-6, "T2": 65e-6},
        "2": {"T1": 52e-6, "T2": 68e-6}
    },
    "readout_errors": [
        [0.02, 0.01],  # Qubit 0: P(0|1), P(1|0)
        [0.015, 0.025],  # Qubit 1
        [0.018, 0.022]   # Qubit 2
    ]
}

# Save to file
with open("custom_noise.json", "w") as f:
    json.dump(custom_noise, f, indent=2)

# Use in simulator
sim = QuantumSimulator(n_qubits=3, noise_model="custom_noise.json")
```

---

## Examples

### Bell State Preparation

```python
from qlret import QuantumSimulator

sim = QuantumSimulator(n_qubits=2, noise_level=0.01)

# Prepare |Φ+⟩ = (|00⟩ + |11⟩)/√2
sim.h(0)
sim.cnot(0, 1)

# Measure
results = sim.measure_all(shots=1000)
print(results)  # {'00': ~500, '11': ~500}
```

### Quantum Teleportation

```python
from qlret import QuantumSimulator
import numpy as np

sim = QuantumSimulator(n_qubits=3, noise_level=0.01)

# Prepare state to teleport on qubit 0
sim.rx(np.pi/3, 0)

# Create entangled pair between qubits 1 and 2
sim.h(1)
sim.cnot(1, 2)

# Bell measurement on qubits 0 and 1
sim.cnot(0, 1)
sim.h(0)

# Measure qubits 0 and 1
m0 = sim.measure(0, shots=1)
m1 = sim.measure(1, shots=1)

# Classical corrections on qubit 2
if m1['1'] > 0:
    sim.x(2)
if m0['1'] > 0:
    sim.z(2)

# Qubit 2 now has the teleported state
print(f"Teleportation fidelity: {sim.fidelity:.4f}")
```

### Grover's Algorithm

```python
from qlret import QuantumSimulator
import numpy as np

def grover_search(n_qubits, target, noise_level=0.01):
    sim = QuantumSimulator(n_qubits=n_qubits, noise_level=noise_level)
    
    # Initialize superposition
    for i in range(n_qubits):
        sim.h(i)
    
    # Grover iterations
    iterations = int(np.pi/4 * np.sqrt(2**n_qubits))
    for _ in range(iterations):
        # Oracle (marks target state)
        # Simplified oracle for demonstration
        if target & (1 << (n_qubits-1)):
            sim.z(n_qubits-1)
        
        # Diffusion operator
        for i in range(n_qubits):
            sim.h(i)
            sim.x(i)
        
        # Multi-controlled Z
        sim.h(n_qubits-1)
        sim.toffoli(0, 1, n_qubits-1)  # Simplified
        sim.h(n_qubits-1)
        
        for i in range(n_qubits):
            sim.x(i)
            sim.h(i)
    
    # Measure
    results = sim.measure_all(shots=1000)
    return results

# Search for state |101⟩ in 3 qubits
results = grover_search(n_qubits=3, target=0b101)
print(results)  # '101' should have highest count
```

### VQE (Variational Quantum Eigensolver)

```python
from qlret import QuantumSimulator
import numpy as np
from scipy.optimize import minimize

def ansatz(sim, params):
    """Parameterized ansatz circuit"""
    n = sim.n_qubits
    for i in range(n):
        sim.ry(params[i], i)
    for i in range(n-1):
        sim.cnot(i, i+1)
    for i in range(n):
        sim.rz(params[n + i], i)

def vqe_cost(params, hamiltonian_coeffs, hamiltonian_ops, n_qubits, noise_level):
    """VQE cost function"""
    sim = QuantumSimulator(n_qubits=n_qubits, noise_level=noise_level)
    ansatz(sim, params)
    
    energy = 0.0
    for coeff, op in zip(hamiltonian_coeffs, hamiltonian_ops):
        exp_val = sim.get_expectation(op, qubits=list(range(n_qubits)))
        energy += coeff * exp_val
    
    return energy

# H2 molecule Hamiltonian (simplified)
h_coeffs = [0.2252, -0.4347, 0.5716]
h_ops = ["I", "Z", "ZZ"]

# Optimize
n_qubits = 2
n_params = 2 * n_qubits
initial_params = np.random.random(n_params) * 2 * np.pi

result = minimize(
    vqe_cost,
    initial_params,
    args=(h_coeffs, h_ops, n_qubits, 0.005),
    method='COBYLA'
)

print(f"Ground state energy: {result.fun:.4f}")
print(f"Optimal parameters: {result.x}")
```

---

## See Also

- **[Quick Start Tutorial](02-quick-start.md)** - Getting started guide
- **[CLI Reference](03-cli-reference.md)** - Command-line interface
- **[PennyLane Integration](05-pennylane-integration.md)** - Hybrid algorithms
- **[Noise Models](06-noise-models.md)** - Configuring noise
- **[API Examples](../examples/python/)** - More code examples

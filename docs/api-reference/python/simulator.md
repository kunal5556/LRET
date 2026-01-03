# QuantumSimulator Class

Main simulator class for quantum state evolution using the LRET algorithm.

## Import

```python
from qlret import QuantumSimulator
```

## Class Definition

```python
class QuantumSimulator:
    """Quantum simulator using Low-Rank Evolution in Time (LRET)."""
    
    def __init__(
        self,
        n_qubits: int,
        noise_level: float = 0.0,
        truncation_threshold: float = 1e-6,
        max_rank: int = 100
    ):
        """Initialize simulator.
        
        Args:
            n_qubits: Number of qubits
            noise_level: Global depolarizing noise level (0 to 1)
            truncation_threshold: Fidelity threshold for rank truncation
            max_rank: Maximum allowed rank
        """
```

---

## Constructor

### `__init__(n_qubits, noise_level=0.0, truncation_threshold=1e-6, max_rank=100)`

Create a new quantum simulator.

**Parameters:**
- `n_qubits` (int): Number of qubits (≥ 1)
- `noise_level` (float, optional): Global noise level, default 0.0
- `truncation_threshold` (float, optional): Truncation threshold, default 1e-6
- `max_rank` (int, optional): Maximum rank, default 100

**Raises:**
- `ValueError`: If parameters are invalid

**Example:**
```python
from qlret import QuantumSimulator

# Noiseless simulator
sim = QuantumSimulator(n_qubits=4)

# With noise
sim = QuantumSimulator(n_qubits=4, noise_level=0.01)

# Custom configuration
sim = QuantumSimulator(
    n_qubits=10,
    noise_level=0.005,
    truncation_threshold=1e-8,
    max_rank=200
)
```

---

## Single-Qubit Gates

### `h(qubit: int)`

Apply Hadamard gate.

**Parameters:**
- `qubit` (int): Target qubit index

**Example:**
```python
sim.h(0)  # Hadamard on qubit 0
```

---

### `x(qubit: int)`, `y(qubit: int)`, `z(qubit: int)`

Apply Pauli gates.

**Example:**
```python
sim.x(0)  # Pauli-X (NOT gate)
sim.y(1)  # Pauli-Y
sim.z(2)  # Pauli-Z
```

---

### `s(qubit: int)`, `t(qubit: int)`

Apply S and T gates.

**Example:**
```python
sim.s(0)  # S gate (√Z)
sim.t(1)  # T gate (√S)
```

---

### `rx(qubit: int, theta: float)`
### `ry(qubit: int, theta: float)`
### `rz(qubit: int, theta: float)`

Apply rotation gates.

**Parameters:**
- `qubit` (int): Target qubit
- `theta` (float): Rotation angle in radians

**Example:**
```python
import numpy as np

sim.rx(0, np.pi / 4)  # X-rotation by π/4
sim.ry(1, np.pi / 2)  # Y-rotation by π/2
sim.rz(2, np.pi)      # Z-rotation by π
```

---

### `u3(qubit: int, theta: float, phi: float, lambda_: float)`

Apply general single-qubit unitary.

**Parameters:**
- `qubit` (int): Target qubit
- `theta`, `phi`, `lambda_` (float): Euler angles

**Example:**
```python
sim.u3(0, np.pi/4, np.pi/2, np.pi)
```

---

## Two-Qubit Gates

### `cnot(control: int, target: int)`

Apply CNOT (controlled-X) gate.

**Parameters:**
- `control` (int): Control qubit
- `target` (int): Target qubit

**Example:**
```python
sim.cnot(0, 1)  # Control: 0, Target: 1
```

---

### `cz(control: int, target: int)`

Apply CZ (controlled-Z) gate.

**Example:**
```python
sim.cz(0, 1)
```

---

### `swap(qubit1: int, qubit2: int)`

Apply SWAP gate.

**Example:**
```python
sim.swap(0, 1)  # Swap qubits 0 and 1
```

---

### `crx(control: int, target: int, theta: float)`
### `cry(control: int, target: int, theta: float)`
### `crz(control: int, target: int, theta: float)`

Apply controlled rotation gates.

**Example:**
```python
sim.crx(0, 1, np.pi/4)
```

---

## Three-Qubit Gates

### `toffoli(control1: int, control2: int, target: int)`

Apply Toffoli (CCNOT) gate.

**Example:**
```python
sim.toffoli(0, 1, 2)  # Controls: 0, 1; Target: 2
```

---

### `fredkin(control: int, target1: int, target2: int)`

Apply Fredkin (controlled-SWAP) gate.

**Example:**
```python
sim.fredkin(0, 1, 2)  # Control: 0; Targets: 1, 2
```

---

## Noise Operations

### `apply_depolarizing_noise(qubit: int, p: float)`

Apply depolarizing noise to a qubit.

**Parameters:**
- `qubit` (int): Target qubit
- `p` (float): Depolarizing parameter (0 to 1)

**Example:**
```python
sim.apply_depolarizing_noise(0, 0.01)  # 1% depolarizing noise
```

---

### `apply_amplitude_damping(qubit: int, gamma: float)`

Apply amplitude damping (energy loss).

**Parameters:**
- `qubit` (int): Target qubit
- `gamma` (float): Damping parameter (0 to 1)

**Example:**
```python
sim.apply_amplitude_damping(0, 0.05)
```

---

### `apply_phase_damping(qubit: int, lambda_: float)`

Apply phase damping (dephasing).

**Parameters:**
- `qubit` (int): Target qubit
- `lambda_` (float): Dephasing parameter (0 to 1)

**Example:**
```python
sim.apply_phase_damping(0, 0.02)
```

---

### `apply_custom_noise(qubit: int, kraus_operators: list)`

Apply custom noise channel.

**Parameters:**
- `qubit` (int): Target qubit
- `kraus_operators` (list): List of Kraus operators (numpy arrays)

**Example:**
```python
import numpy as np

# Custom bit-flip channel
K0 = np.sqrt(0.9) * np.eye(2)
K1 = np.sqrt(0.1) * np.array([[0, 1], [1, 0]])

sim.apply_custom_noise(0, [K0, K1])
```

---

## Measurement

### `measure_all(shots: int = 1) -> Dict[str, int]`

Measure all qubits in computational basis.

**Parameters:**
- `shots` (int, optional): Number of measurements, default 1

**Returns:**
- `dict`: Mapping from outcome strings to counts

**Example:**
```python
results = sim.measure_all(shots=1000)
print(results)
# Output: {'00': 487, '11': 513}
```

---

### `measure_qubits(qubits: List[int], shots: int = 1) -> Dict[str, int]`

Measure specific qubits.

**Parameters:**
- `qubits` (list): List of qubit indices to measure
- `shots` (int, optional): Number of measurements

**Example:**
```python
results = sim.measure_qubits([0, 2], shots=1000)
```

---

### `get_probability(outcome: str) -> float`

Get probability of a specific outcome.

**Parameters:**
- `outcome` (str): Outcome string (e.g., "0101")

**Returns:**
- `float`: Probability (0 to 1)

**Example:**
```python
prob = sim.get_probability("00")
print(f"P(00) = {prob:.4f}")
```

---

### `get_probabilities() -> np.ndarray`

Get probabilities of all outcomes.

**Returns:**
- `numpy.ndarray`: Array of probabilities (length $2^n$)

**Example:**
```python
probs = sim.get_probabilities()
print(f"Sum of probabilities: {probs.sum():.6f}")  # Should be 1.0
```

---

## State Access

### `get_state_vector() -> np.ndarray`

Get the state vector (pure states only).

**Returns:**
- `numpy.ndarray`: Complex state vector of length $2^n$

**Raises:**
- `RuntimeError`: If state is mixed (rank > 1)

**Example:**
```python
try:
    psi = sim.get_state_vector()
    print(f"State vector norm: {np.linalg.norm(psi):.6f}")
except RuntimeError:
    print("State is mixed")
```

---

### `get_density_matrix() -> np.ndarray`

Get the full density matrix.

**Returns:**
- `numpy.ndarray`: Complex density matrix of shape $(2^n, 2^n)$

**Warning:** Exponential memory for large $n$

**Example:**
```python
rho = sim.get_density_matrix()
print(f"Trace: {np.trace(rho):.6f}")  # Should be 1.0
print(f"Purity: {np.trace(rho @ rho):.6f}")  # ≤ 1.0
```

---

### `get_reduced_density_matrix(qubits: List[int]) -> np.ndarray`

Get reduced density matrix of subsystem.

**Parameters:**
- `qubits` (list): Qubits to keep

**Returns:**
- `numpy.ndarray`: Reduced density matrix

**Example:**
```python
# Entangled state
sim.h(0)
sim.cnot(0, 1)

# Reduced density matrix of qubit 0
rho_0 = sim.get_reduced_density_matrix([0])
print(rho_0)
# Output: [[0.5, 0], [0, 0.5]]  (maximally mixed)
```

---

## Expectation Values

### `expectation(observable: Union[str, np.ndarray]) -> float`

Compute expectation value of an observable.

**Parameters:**
- `observable` (str or ndarray): Observable ("X", "Y", "Z", or matrix)

**Returns:**
- `float`: Expectation value

**Example:**
```python
# Prepare |+⟩ state
sim.h(0)

# Measure <X>
exp_x = sim.expectation("X")
print(f"<X> = {exp_x:.4f}")  # Should be ~1.0

# Measure <Z>
exp_z = sim.expectation("Z")
print(f"<Z> = {exp_z:.4f}")  # Should be ~0.0

# Custom observable
obs = np.array([[1, 0], [0, -1]])  # Z matrix
exp = sim.expectation(obs)
```

---

## Properties

### `n_qubits: int`

Number of qubits (read-only).

```python
print(f"Number of qubits: {sim.n_qubits}")
```

---

### `current_rank: int`

Current rank of the state representation (read-only).

```python
print(f"Current rank: {sim.current_rank}")
```

---

### `noise_level: float`

Global noise level.

```python
print(f"Noise level: {sim.noise_level}")
sim.noise_level = 0.02  # Change noise level
```

---

## Configuration

### `set_truncation_threshold(threshold: float)`

Set truncation threshold.

**Example:**
```python
sim.set_truncation_threshold(1e-8)  # Higher accuracy
```

---

### `set_max_rank(max_rank: int)`

Set maximum rank.

**Example:**
```python
sim.set_max_rank(200)
```

---

### `reset()`

Reset to initial state $|0\rangle^{\otimes n}$.

**Example:**
```python
sim.reset()
```

---

## Examples

### Bell State Creation

```python
from qlret import QuantumSimulator

sim = QuantumSimulator(n_qubits=2)

# Create Bell state
sim.h(0)
sim.cnot(0, 1)

# Measure
results = sim.measure_all(shots=1000)
print(results)
# Output: {'00': 502, '11': 498}
```

---

### GHZ State

```python
sim = QuantumSimulator(n_qubits=3)

# Create GHZ state: (|000⟩ + |111⟩) / √2
sim.h(0)
sim.cnot(0, 1)
sim.cnot(0, 2)

results = sim.measure_all(shots=1000)
print(results)
# Output: {'000': 496, '111': 504}
```

---

### Quantum Fourier Transform

```python
import numpy as np

def qft(sim, n):
    """Apply QFT to first n qubits."""
    for j in range(n):
        sim.h(j)
        for k in range(j + 1, n):
            sim.crz(k, j, np.pi / (2 ** (k - j)))
    
    # Swap qubits
    for j in range(n // 2):
        sim.swap(j, n - j - 1)

sim = QuantumSimulator(n_qubits=4)
sim.x(0)  # Start from |0001⟩
qft(sim, 4)

probs = sim.get_probabilities()
print(f"Max probability: {probs.max():.4f}")
```

---

### Variational Quantum Eigensolver (VQE)

```python
import numpy as np
from scipy.optimize import minimize

def ansatz(sim, params):
    """Variational ansatz."""
    sim.reset()
    for i in range(sim.n_qubits):
        sim.ry(i, params[i])
    for i in range(sim.n_qubits - 1):
        sim.cnot(i, i + 1)
    for i in range(sim.n_qubits):
        sim.ry(i, params[sim.n_qubits + i])

def energy(params):
    """Compute energy expectation."""
    sim = QuantumSimulator(n_qubits=4)
    ansatz(sim, params)
    return sim.expectation("Z")  # Hamiltonian: Z₀

# Optimize
init_params = np.random.rand(8)
result = minimize(energy, init_params, method='COBYLA')
print(f"Ground state energy: {result.fun:.6f}")
```

---

## See Also

- [Gate Reference](gates.md) - Available gates
- [Noise Models](noise.md) - Noise channels
- [PennyLane Device](pennylane.md) - PennyLane integration
- [Examples](../../examples/python/) - More examples
- [User Guide](../../user-guide/04-python-interface.md) - Detailed guide

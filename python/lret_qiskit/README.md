# LRET Qiskit Backend

Qiskit integration for the LRET (Low-Rank Exact Tensor) quantum simulator.

## Installation

```bash
# Install from the python directory
cd python/lret_qiskit
pip install .

# Or install with development dependencies
pip install .[dev]
```

**Note:** This package requires the `qlret` package to be installed for actual
simulation. Install it from the parent directory:

```bash
cd python
pip install .
```

## Usage

```python
from qiskit import QuantumCircuit
from lret_qiskit import LRETProvider

# Get the provider and backend
provider = LRETProvider()
backend = provider.get_backend("lret_simulator")

# Create a circuit
qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure([0, 1], [0, 1])

# Run on LRET backend
job = backend.run(qc, shots=1024)
result = job.result()
print(result.get_counts())
```

## Available Backends

| Backend | Epsilon | Use Case |
|---------|---------|----------|
| `lret_simulator` | 1e-4 | Balanced accuracy/speed (default) |
| `lret_simulator_accurate` | 1e-6 | High-precision simulations |
| `lret_simulator_fast` | 1e-3 | Quick prototyping |

## Supported Gates

### Single-Qubit Gates
- Clifford: H, X, Y, Z, S, Sdg, T, Tdg, SX
- Rotation: RX, RY, RZ
- Phase: P (U1), U2, U3

### Two-Qubit Gates
- CX (CNOT), CY, CZ
- SWAP, iSWAP

### Measurement
- measure

## Running Tests

```bash
pytest python/lret_qiskit/tests/ -v
```

## License

Apache 2.0

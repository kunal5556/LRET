# Python API Reference

Complete reference for the LRET Python package.

## Installation

```bash
pip install qlret
```

## Modules

### Core Simulator

- **[QuantumSimulator](simulator.md)** - Main simulator class
- **[Gate](gates.md)** - Gate operations and builders
- **[NoiseModel](noise.md)** - Noise channel definitions

### PennyLane Integration

- **[QLRETDevice](pennylane.md)** - PennyLane device plugin

### Utilities

- **[calibration](calibration.md)** - Noise calibration tools
- **[utils](utils.md)** - Helper functions
- **[visualization](visualization.md)** - Plotting and visualization

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

## PennyLane Integration

```python
import pennylane as qml
from qlret import QLRETDevice

dev = qml.device("qlret.simulator", wires=2, noise_level=0.01)

@qml.qnode(dev)
def circuit():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.probs(wires=[0, 1])

print(circuit())
```

## Quick Links

- [Examples](../examples/python/) - Python code examples
- [Jupyter Notebooks](../examples/jupyter/) - Interactive tutorials
- [User Guide](../../user-guide/04-python-interface.md) - Detailed guide
- [PennyLane Guide](../../user-guide/05-pennylane-integration.md) - PennyLane usage

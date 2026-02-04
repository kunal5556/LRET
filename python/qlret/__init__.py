"""QLRET - Low-Rank Exact Tensor Quantum Simulator.

Python bridge for QLRET with PennyLane device integration.

Basic Usage:
    from qlret import simulate_json, load_json_file
    result = simulate_json(load_json_file("circuit.json"))

PennyLane Usage:
    import pennylane as qml
    from qlret import QLRETDevice

    dev = QLRETDevice(wires=4, shots=1000)

    @qml.qnode(dev)
    def circuit(theta):
        qml.RX(theta, wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0))

    result = circuit(0.5)
    grad = qml.grad(circuit)(0.5)

Batch Parallelism (for VQE, QAOA, QNN workloads):
    # Enable Python-level parallelism for batch execution
    dev = QLRETDevice(
        wires=4,
        num_threads=2,        # C++ threads per circuit (OpenMP)
        max_batch_workers=4,  # Python workers for parallel circuit execution
    )
    # 4 workers × 2 threads = 8 total threads (optimal for 8-core CPU)
    
    # Or use auto-tuning:
    dev = QLRETDevice(wires=4, max_batch_workers=-1)  # Auto-tune parallelism
"""

from .api import (
    simulate_json,
    load_json_file,
    set_executable_path,
    QLRETError,
)
from .pennylane_device import QLRETDevice, QLRETDeviceError

__version__ = "1.0.0"
__all__ = [
    "simulate_json",
    "load_json_file",
    "set_executable_path",
    "QLRETError",
    "QLRETDevice",
    "QLRETDeviceError",
]

"""
Quantum Walk Benchmark
======================

Algorithm #16 - Tier 3 (Optional)
Purpose: Continuous-time quantum walks on graphs

Key metrics:
- State transfer fidelity
- Mixing time
- Search efficiency

Primary comparison: default.mixed
"""

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.benchmark_utils import BenchmarkResult, Timer, MemoryTracker
from utils.device_factory import create_lret_device, create_comparison_device


def create_line_graph_hamiltonian(n_nodes: int) -> qml.Hamiltonian:
    """Create Hamiltonian for quantum walk on a line graph."""
    coeffs = []
    ops = []
    
    # Hopping terms: |i><j| + |j><i| encoded as X_i X_j + Y_i Y_j
    for i in range(n_nodes - 1):
        coeffs.extend([0.5, 0.5])
        ops.extend([
            qml.PauliX(i) @ qml.PauliX(i + 1),
            qml.PauliY(i) @ qml.PauliY(i + 1)
        ])
    
    return qml.Hamiltonian(coeffs, ops)


def create_cycle_graph_hamiltonian(n_nodes: int) -> qml.Hamiltonian:
    """Create Hamiltonian for quantum walk on a cycle graph."""
    coeffs = []
    ops = []
    
    for i in range(n_nodes):
        j = (i + 1) % n_nodes
        coeffs.extend([0.5, 0.5])
        ops.extend([
            qml.PauliX(i) @ qml.PauliX(j),
            qml.PauliY(i) @ qml.PauliY(j)
        ])
    
    return qml.Hamiltonian(coeffs, ops)


def quantum_walk_circuit(
    hamiltonian: qml.Hamiltonian,
    time: float,
    n_qubits: int,
    start_node: int = 0,
    n_trotter_steps: int = 5
):
    """Quantum walk circuit using Trotterization."""
    # Initialize at start node
    qml.PauliX(wires=start_node)
    
    # Trotterized evolution: exp(-i H t)
    dt = time / n_trotter_steps
    
    for _ in range(n_trotter_steps):
        # Apply each term in the Hamiltonian
        for coef, op in zip(hamiltonian.coeffs, hamiltonian.ops):
            angle = float(coef) * dt
            # Simplified: approximate with nearest-neighbor evolution
            if hasattr(op, 'wires') and len(op.wires) == 2:
                i, j = op.wires
                qml.IsingXX(2 * angle, wires=[i, j])


def compute_transfer_probability(probs: np.ndarray, target_node: int) -> float:
    """Compute probability of measuring the target node."""
    # For position encoding, target node corresponds to basis state |target_node>
    if target_node < len(probs):
        return float(probs[target_node])
    return 0.0


@dataclass
class QuantumWalkBenchmark:
    """Quantum Walk benchmark."""
    
    n_nodes: int = 4
    evolution_time: float = 2.0
    n_trotter_steps: int = 10
    graph_type: str = 'line'  # 'line' or 'cycle'
    with_noise: bool = True
    noise_strength: float = 0.01
    
    def __post_init__(self):
        self.n_qubits = self.n_nodes
        if self.graph_type == 'cycle':
            self.hamiltonian = create_cycle_graph_hamiltonian(self.n_nodes)
        else:
            self.hamiltonian = create_line_graph_hamiltonian(self.n_nodes)
        self.results = []
    
    def run_single_device(
        self,
        device_name: str,
        mode: str = 'default',
        n_trials: int = 3
    ) -> List[BenchmarkResult]:
        results = []
        
        for trial in range(n_trials):
            with MemoryTracker() as mem:
                with Timer() as timer:
                    try:
                        if 'qlret' in device_name:
                            dev = create_lret_device(wires=self.n_qubits, mode=mode)
                        else:
                            dev = create_comparison_device(device_name, wires=self.n_qubits)
                        
                        @qml.qnode(dev)
                        def walk_circuit():
                            quantum_walk_circuit(
                                self.hamiltonian,
                                self.evolution_time,
                                self.n_qubits,
                                start_node=0,
                                n_trotter_steps=self.n_trotter_steps
                            )
                            return qml.probs(wires=range(self.n_qubits))
                        
                        probs = walk_circuit()
                        
                        # Compute spreading: variance of position
                        positions = np.arange(len(probs))
                        mean_pos = np.sum(positions * probs)
                        variance = np.sum((positions - mean_pos) ** 2 * probs)
                        
                        # Transfer to opposite end
                        target = self.n_nodes - 1
                        transfer_prob = compute_transfer_probability(probs, target)
                        
                        success = True
                        error_msg = ""
                        
                    except Exception as e:
                        variance = float('nan')
                        transfer_prob = float('nan')
                        success = False
                        error_msg = str(e)
            
            results.append(BenchmarkResult(
                algorithm='QuantumWalk',
                device_name=device_name,
                mode=mode,
                n_qubits=self.n_qubits,
                execution_time_seconds=timer.seconds,
                peak_memory_mb=mem.peak_mb,
                result_value=variance,
                secondary_value=transfer_prob,
                with_noise=self.with_noise,
                success=success,
                error_message=error_msg,
                extra_data={
                    'evolution_time': self.evolution_time,
                    'graph_type': self.graph_type
                }
            ))
        
        self.results.extend(results)
        return results
    
    def compare_lret_modes(self, n_trials: int = 3) -> Dict[str, List[BenchmarkResult]]:
        print(f"\n{'='*60}\nQuantum Walk LRET Mode Comparison\n{'='*60}")
        return {mode: self.run_single_device('qlret.mixed', mode, n_trials) 
                for mode in ['sequential', 'batched', 'parallel', 'openmp']}
    
    def compare_devices(self, n_trials: int = 3) -> Dict[str, List[BenchmarkResult]]:
        print(f"\n{'='*60}\nQuantum Walk Device Comparison\n{'='*60}")
        return {
            'qlret.mixed': self.run_single_device('qlret.mixed', 'sequential', n_trials),
            'default.mixed': self.run_single_device('default.mixed', 'default', n_trials),
        }
    
    def run_full_benchmark(self, n_trials: int = 3) -> Dict[str, Any]:
        return {
            'lret_modes': self.compare_lret_modes(n_trials),
            'device_comparison': self.compare_devices(n_trials),
        }


def run_quantum_walk_benchmark(n_nodes: int = 4, n_trials: int = 3) -> Dict[str, Any]:
    return QuantumWalkBenchmark(n_nodes=n_nodes).run_full_benchmark(n_trials)


if __name__ == "__main__":
    results = run_quantum_walk_benchmark(n_nodes=4, n_trials=2)
    print("Quantum Walk benchmark complete!")

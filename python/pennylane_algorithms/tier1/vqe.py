"""
VQE (Variational Quantum Eigensolver) Benchmark
================================================

Algorithm #1 - Tier 1 (Must Test)
Purpose: Find ground state energy of molecular Hamiltonians

Key metrics:
- Energy accuracy vs exact diagonalization
- Convergence rate
- Noise resilience

Primary comparison: default.mixed
"""

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.benchmark_utils import BenchmarkResult, Timer, MemoryTracker
from utils.device_factory import create_lret_device, create_comparison_device


def get_h2_hamiltonian() -> Tuple[qml.Hamiltonian, float]:
    """Create H2 molecule Hamiltonian at equilibrium bond length."""
    coeffs = [-0.81261, 0.17120, 0.17120, -0.22343, 0.16862, 0.04532]
    obs = [
        qml.Identity(0),
        qml.PauliZ(0),
        qml.PauliZ(1),
        qml.PauliZ(0) @ qml.PauliZ(1),
        qml.PauliX(0) @ qml.PauliX(1),
        qml.PauliY(0) @ qml.PauliY(1),
    ]
    exact_energy = -1.1373
    return qml.Hamiltonian(coeffs, obs), exact_energy


def hardware_efficient_ansatz(params: np.ndarray, n_qubits: int, n_layers: int = 2):
    """Hardware-efficient ansatz with RY-RZ rotations and CNOT entanglement."""
    param_idx = 0
    for layer in range(n_layers):
        for i in range(n_qubits):
            qml.RY(params[param_idx], wires=i)
            param_idx += 1
            qml.RZ(params[param_idx], wires=i)
            param_idx += 1
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])


@dataclass
class VQEBenchmark:
    """VQE benchmark class."""
    
    n_qubits: int = 4
    n_layers: int = 2
    max_iterations: int = 50
    with_noise: bool = True
    noise_strength: float = 0.01
    
    def __post_init__(self):
        self.hamiltonian, self.exact_energy = get_h2_hamiltonian()
        self.n_params = 2 * self.n_qubits * self.n_layers
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
                        
                        @qml.qnode(dev, interface='autograd')
                        def circuit(params):
                            hardware_efficient_ansatz(params, self.n_qubits, self.n_layers)
                            return qml.expval(self.hamiltonian)
                        
                        params = pnp.random.uniform(-np.pi, np.pi, self.n_params, requires_grad=True)
                        opt = qml.GradientDescentOptimizer(stepsize=0.1)
                        
                        for _ in range(self.max_iterations):
                            params, energy = opt.step_and_cost(circuit, params)
                        
                        energy = float(energy)
                        error = abs(energy - self.exact_energy)
                        success = True
                        error_msg = ""
                        
                    except Exception as e:
                        energy = float('nan')
                        error = float('nan')
                        success = False
                        error_msg = str(e)
            
            results.append(BenchmarkResult(
                algorithm='VQE',
                device_name=device_name,
                mode=mode,
                n_qubits=self.n_qubits,
                execution_time_seconds=timer.seconds,
                peak_memory_mb=mem.peak_mb,
                result_value=energy,
                secondary_value=error,
                with_noise=self.with_noise,
                success=success,
                error_message=error_msg,
                extra_data={'n_layers': self.n_layers, 'max_iterations': self.max_iterations}
            ))
        
        self.results.extend(results)
        return results
    
    def compare_lret_modes(self, n_trials: int = 3) -> Dict[str, List[BenchmarkResult]]:
        print(f"\n{'='*60}\nVQE LRET Mode Comparison\n{'='*60}")
        return {mode: self.run_single_device('qlret.mixed', mode, n_trials) 
                for mode in ['sequential', 'batched', 'parallel', 'openmp']}
    
    def compare_devices(self, n_trials: int = 3) -> Dict[str, List[BenchmarkResult]]:
        print(f"\n{'='*60}\nVQE Device Comparison\n{'='*60}")
        return {
            'qlret.mixed': self.run_single_device('qlret.mixed', 'sequential', n_trials),
            'default.mixed': self.run_single_device('default.mixed', 'default', n_trials),
        }
    
    def run_full_benchmark(self, n_trials: int = 3) -> Dict[str, Any]:
        return {
            'lret_modes': self.compare_lret_modes(n_trials),
            'device_comparison': self.compare_devices(n_trials),
        }


def run_vqe_benchmark(n_qubits: int = 4, n_trials: int = 3) -> Dict[str, Any]:
    return VQEBenchmark(n_qubits=n_qubits).run_full_benchmark(n_trials)


if __name__ == "__main__":
    results = run_vqe_benchmark(n_qubits=4, n_trials=2)
    print("VQE benchmark complete!")

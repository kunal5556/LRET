"""
UCCSD-VQE Benchmark
===================

Algorithm #8 - Tier 2 (Should Test)
Purpose: Gold-standard chemistry ansatz with many parameters

Key metrics:
- Chemical accuracy
- Parameter count handling
- Gradient efficiency

Primary comparison: default.mixed
Secondary comparison: lightning.qubit
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
from utils.parallel_modes import get_parallel_modes, ParallelExecutor


def get_h2_uccsd_params(n_qubits: int = 4) -> Tuple[int, int]:
    """Get parameter counts for H2 UCCSD."""
    n_electrons = 2
    n_orbitals = n_qubits
    n_singles = n_electrons * (n_orbitals - n_electrons)
    n_doubles = n_singles * (n_singles - 1) // 2
    return n_singles, n_doubles


def uccsd_circuit(params: np.ndarray, n_qubits: int):
    """UCCSD ansatz circuit."""
    n_electrons = n_qubits // 2
    
    # Hartree-Fock initial state
    for i in range(n_electrons):
        qml.PauliX(wires=i)
    
    # Single and double excitations
    param_idx = 0
    
    # Singles
    for i in range(n_electrons):
        for a in range(n_electrons, n_qubits):
            if param_idx < len(params):
                qml.SingleExcitation(params[param_idx], wires=[i, a])
                param_idx += 1
    
    # Doubles
    for i in range(n_electrons):
        for j in range(i + 1, n_electrons):
            for a in range(n_electrons, n_qubits):
                for b in range(a + 1, n_qubits):
                    if param_idx < len(params):
                        qml.DoubleExcitation(params[param_idx], wires=[i, j, a, b])
                        param_idx += 1


def get_h2_hamiltonian_simple() -> Tuple[qml.Hamiltonian, float]:
    """Simplified H2 Hamiltonian."""
    coeffs = [-0.81261, 0.17120, 0.17120, -0.22343, 0.16862, 0.12054]
    ops = [
        qml.Identity(0),
        qml.PauliZ(0), qml.PauliZ(1),
        qml.PauliZ(0) @ qml.PauliZ(1),
        qml.PauliZ(0) @ qml.PauliZ(2),
        qml.PauliX(0) @ qml.PauliX(1) @ qml.PauliY(2) @ qml.PauliY(3),
    ]
    return qml.Hamiltonian(coeffs, ops), -1.1373


@dataclass
class UCCSDBenchmark:
    """UCCSD-VQE benchmark."""
    
    n_qubits: int = 4
    max_iterations: int = 30
    with_noise: bool = True
    noise_strength: float = 0.01
    
    def __post_init__(self):
        self.hamiltonian, self.exact_energy = get_h2_hamiltonian_simple()
        n_singles, n_doubles = get_h2_uccsd_params(self.n_qubits)
        self.n_params = min(n_singles + n_doubles, self.n_qubits * 2)
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
                        
                        @qml.qnode(dev, interface='autograd', diff_method='parameter-shift')
                        def circuit(params):
                            uccsd_circuit(params, self.n_qubits)
                            return qml.expval(self.hamiltonian)
                        
                        params = pnp.random.uniform(-0.1, 0.1, self.n_params, requires_grad=True)
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
                algorithm='UCCSD-VQE',
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
                extra_data={'n_params': self.n_params}
            ))
        
        self.results.extend(results)
        return results
    
    def compare_lret_modes(self, n_trials: int = 3) -> Dict[str, List[BenchmarkResult]]:
        print(f"\n{'='*60}\nUCCSD-VQE LRET Mode Comparison\n{'='*60}")
        return {mode: self.run_single_device('qlret.mixed', mode, n_trials) 
                for mode in ['sequential', 'batched', 'parallel', 'openmp']}
    
    def compare_devices(self, n_trials: int = 3) -> Dict[str, List[BenchmarkResult]]:
        print(f"\n{'='*60}\nUCCSD-VQE Device Comparison\n{'='*60}")
        return {
            'qlret.mixed': self.run_single_device('qlret.mixed', 'sequential', n_trials),
            'default.mixed': self.run_single_device('default.mixed', 'default', n_trials),
        }
    
    def run_full_benchmark(self, n_trials: int = 3) -> Dict[str, Any]:
        return {
            'lret_modes': self.compare_lret_modes(n_trials),
            'device_comparison': self.compare_devices(n_trials),
        }


def run_uccsd_benchmark(n_qubits: int = 4, n_trials: int = 3) -> Dict[str, Any]:
    return UCCSDBenchmark(n_qubits=n_qubits).run_full_benchmark(n_trials)


if __name__ == "__main__":
    results = run_uccsd_benchmark(n_qubits=4, n_trials=2)
    print("UCCSD-VQE benchmark complete!")

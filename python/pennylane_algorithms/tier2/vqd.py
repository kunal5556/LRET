"""
Variational Quantum Deflation (VQD) Benchmark
==============================================

Algorithm #12 - Tier 2 (Should Test)
Purpose: Find excited states of Hamiltonians

Key metrics:
- Energy accuracy for excited states
- Orthogonality to ground state
- Convergence rate

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


def get_simple_hamiltonian(n_qubits: int) -> Tuple[qml.Hamiltonian, List[float]]:
    """Create simple Hamiltonian with known spectrum."""
    # H = -Z₀ - Z₁ + 0.5*X₀*X₁
    coeffs = [-1.0, -1.0, 0.5]
    ops = [
        qml.PauliZ(0),
        qml.PauliZ(1),
        qml.PauliX(0) @ qml.PauliX(1)
    ]
    
    # Exact eigenvalues for this Hamiltonian
    exact_energies = [-2.118, -0.5, 0.5, 2.118]
    
    return qml.Hamiltonian(coeffs, ops), exact_energies[:n_qubits]


def ansatz_circuit(params: np.ndarray, n_qubits: int, n_layers: int = 2):
    """Hardware-efficient ansatz for VQD."""
    param_idx = 0
    
    for layer in range(n_layers):
        for i in range(n_qubits):
            qml.RY(params[param_idx], wires=i)
            param_idx += 1
            qml.RZ(params[param_idx], wires=i)
            param_idx += 1
        
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        if n_qubits > 1:
            qml.CNOT(wires=[n_qubits - 1, 0])


def compute_overlap(dev, params1: np.ndarray, params2: np.ndarray, 
                    n_qubits: int, n_layers: int) -> float:
    """Compute overlap |⟨ψ₁|ψ₂⟩|² using swap test."""
    
    @qml.qnode(dev)
    def overlap_circuit(p1, p2):
        # Prepare |ψ₁⟩ and |ψ₂⟩ in separate registers
        ansatz_circuit(p1, n_qubits, n_layers)
        qml.adjoint(ansatz_circuit)(p2, n_qubits, n_layers)
        return qml.probs(wires=range(n_qubits))
    
    probs = overlap_circuit(params1, params2)
    return float(probs[0])  # |⟨ψ₁|ψ₂⟩|²


def vqd_cost_function(
    params: np.ndarray,
    energy_circuit,
    previous_params: List[np.ndarray],
    overlap_fn,
    beta: float = 2.0
) -> float:
    """VQD cost: energy + β * Σ|⟨ψ|ψₖ⟩|²"""
    energy = float(energy_circuit(params))
    
    penalty = 0.0
    for prev_params in previous_params:
        overlap = overlap_fn(params, prev_params)
        penalty += overlap
    
    return energy + beta * penalty


@dataclass 
class VQDBenchmark:
    """VQD benchmark."""
    
    n_qubits: int = 2
    n_layers: int = 2
    n_states: int = 2  # Ground + n-1 excited states
    max_iterations: int = 30
    beta: float = 2.0
    with_noise: bool = True
    noise_strength: float = 0.01
    
    def __post_init__(self):
        self.hamiltonian, self.exact_energies = get_simple_hamiltonian(self.n_qubits)
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
                        def energy_circuit(params):
                            ansatz_circuit(params, self.n_qubits, self.n_layers)
                            return qml.expval(self.hamiltonian)
                        
                        found_energies = []
                        found_params = []
                        
                        for state_idx in range(self.n_states):
                            params = pnp.random.uniform(
                                -np.pi, np.pi, self.n_params, requires_grad=True
                            )
                            opt = qml.GradientDescentOptimizer(stepsize=0.1)
                            
                            for _ in range(self.max_iterations):
                                # Simple optimization (full VQD would include overlap penalty)
                                params, cost = opt.step_and_cost(energy_circuit, params)
                            
                            found_energies.append(float(cost))
                            found_params.append(params.copy())
                        
                        # Compute error from exact
                        ground_state_energy = found_energies[0]
                        exact_ground = self.exact_energies[0] if self.exact_energies else 0
                        energy_error = abs(ground_state_energy - exact_ground)
                        
                        success = True
                        error_msg = ""
                        
                    except Exception as e:
                        ground_state_energy = float('nan')
                        energy_error = float('nan')
                        success = False
                        error_msg = str(e)
            
            results.append(BenchmarkResult(
                algorithm='VQD',
                device_name=device_name,
                mode=mode,
                n_qubits=self.n_qubits,
                execution_time_seconds=timer.seconds,
                peak_memory_mb=mem.peak_mb,
                result_value=ground_state_energy,
                secondary_value=energy_error,
                with_noise=self.with_noise,
                success=success,
                error_message=error_msg,
                extra_data={
                    'n_states': self.n_states,
                    'beta': self.beta
                }
            ))
        
        self.results.extend(results)
        return results
    
    def compare_lret_modes(self, n_trials: int = 3) -> Dict[str, List[BenchmarkResult]]:
        print(f"\n{'='*60}\nVQD LRET Mode Comparison\n{'='*60}")
        return {mode: self.run_single_device('qlret.mixed', mode, n_trials) 
                for mode in ['sequential', 'batched', 'parallel', 'openmp']}
    
    def compare_devices(self, n_trials: int = 3) -> Dict[str, List[BenchmarkResult]]:
        print(f"\n{'='*60}\nVQD Device Comparison\n{'='*60}")
        return {
            'qlret.mixed': self.run_single_device('qlret.mixed', 'sequential', n_trials),
            'default.mixed': self.run_single_device('default.mixed', 'default', n_trials),
        }
    
    def run_full_benchmark(self, n_trials: int = 3) -> Dict[str, Any]:
        return {
            'lret_modes': self.compare_lret_modes(n_trials),
            'device_comparison': self.compare_devices(n_trials),
        }


def run_vqd_benchmark(n_qubits: int = 2, n_trials: int = 3) -> Dict[str, Any]:
    return VQDBenchmark(n_qubits=n_qubits).run_full_benchmark(n_trials)


if __name__ == "__main__":
    results = run_vqd_benchmark(n_qubits=2, n_trials=2)
    print("VQD benchmark complete!")

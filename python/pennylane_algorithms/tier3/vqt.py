"""
Variational Quantum Thermalizer (VQT) Benchmark
================================================

Algorithm #15 - Tier 3 (Optional)
Purpose: Prepare thermal/Gibbs states of Hamiltonians

Key metrics:
- Free energy minimization
- Temperature accuracy
- Mixed state fidelity

Primary comparison: default.mixed (required for thermal states)
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


def get_ising_hamiltonian(n_qubits: int, J: float = 1.0, h: float = 0.5) -> qml.Hamiltonian:
    """Create 1D transverse field Ising model Hamiltonian."""
    coeffs = []
    ops = []
    
    # ZZ interactions
    for i in range(n_qubits - 1):
        coeffs.append(-J)
        ops.append(qml.PauliZ(i) @ qml.PauliZ(i + 1))
    
    # Transverse field
    for i in range(n_qubits):
        coeffs.append(-h)
        ops.append(qml.PauliX(i))
    
    return qml.Hamiltonian(coeffs, ops)


def vqt_ansatz(params: np.ndarray, n_qubits: int, n_layers: int = 2):
    """VQT ansatz with noise channels for thermal state preparation."""
    param_idx = 0
    
    for layer in range(n_layers):
        # Parameterized rotations
        for i in range(n_qubits):
            qml.RY(params[param_idx], wires=i)
            param_idx += 1
            qml.RZ(params[param_idx], wires=i)
            param_idx += 1
        
        # Entangling layer
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        
        # Add depolarizing noise to create mixed states
        for i in range(n_qubits):
            if param_idx < len(params):
                # Use amplitude damping to create thermal mixing
                gamma = np.clip(np.abs(params[param_idx]), 0, 1)
                qml.AmplitudeDamping(gamma, wires=i)
                param_idx += 1


def compute_free_energy(
    energy: float,
    entropy: float,
    temperature: float
) -> float:
    """Compute free energy F = E - T*S."""
    return energy - temperature * entropy


def estimate_von_neumann_entropy(probs: np.ndarray) -> float:
    """Estimate von Neumann entropy from probability distribution."""
    probs = np.clip(probs, 1e-10, 1.0)
    return float(-np.sum(probs * np.log2(probs)))


@dataclass
class VQTBenchmark:
    """VQT benchmark."""
    
    n_qubits: int = 3
    n_layers: int = 2
    temperature: float = 1.0
    max_iterations: int = 30
    with_noise: bool = True
    noise_strength: float = 0.01
    
    def __post_init__(self):
        self.hamiltonian = get_ising_hamiltonian(self.n_qubits)
        self.n_params = (3 * self.n_qubits) * self.n_layers
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
                        def energy_circuit(params):
                            vqt_ansatz(params, self.n_qubits, self.n_layers)
                            return qml.expval(self.hamiltonian)
                        
                        @qml.qnode(dev)
                        def prob_circuit(params):
                            vqt_ansatz(params, self.n_qubits, self.n_layers)
                            return qml.probs(wires=range(self.n_qubits))
                        
                        params = pnp.random.uniform(0, 0.5, self.n_params)
                        
                        # VQT optimization: minimize free energy
                        def cost(params):
                            energy = energy_circuit(params)
                            probs = prob_circuit(params)
                            entropy = estimate_von_neumann_entropy(probs)
                            return energy - self.temperature * entropy
                        
                        opt = qml.GradientDescentOptimizer(stepsize=0.05)
                        
                        for _ in range(self.max_iterations):
                            params = opt.step(cost, params)
                        
                        final_energy = float(energy_circuit(params))
                        final_probs = prob_circuit(params)
                        final_entropy = estimate_von_neumann_entropy(final_probs)
                        final_free_energy = final_energy - self.temperature * final_entropy
                        
                        success = True
                        error_msg = ""
                        
                    except Exception as e:
                        final_free_energy = float('nan')
                        final_entropy = float('nan')
                        success = False
                        error_msg = str(e)
            
            results.append(BenchmarkResult(
                algorithm='VQT',
                device_name=device_name,
                mode=mode,
                n_qubits=self.n_qubits,
                execution_time_seconds=timer.seconds,
                peak_memory_mb=mem.peak_mb,
                result_value=final_free_energy,
                secondary_value=final_entropy,
                with_noise=self.with_noise,
                success=success,
                error_message=error_msg,
                extra_data={
                    'temperature': self.temperature,
                    'n_layers': self.n_layers
                }
            ))
        
        self.results.extend(results)
        return results
    
    def compare_lret_modes(self, n_trials: int = 3) -> Dict[str, List[BenchmarkResult]]:
        print(f"\n{'='*60}\nVQT LRET Mode Comparison\n{'='*60}")
        return {mode: self.run_single_device('qlret.mixed', mode, n_trials) 
                for mode in ['sequential', 'batched', 'parallel', 'openmp']}
    
    def compare_devices(self, n_trials: int = 3) -> Dict[str, List[BenchmarkResult]]:
        print(f"\n{'='*60}\nVQT Device Comparison\n{'='*60}")
        return {
            'qlret.mixed': self.run_single_device('qlret.mixed', 'sequential', n_trials),
            'default.mixed': self.run_single_device('default.mixed', 'default', n_trials),
        }
    
    def run_full_benchmark(self, n_trials: int = 3) -> Dict[str, Any]:
        return {
            'lret_modes': self.compare_lret_modes(n_trials),
            'device_comparison': self.compare_devices(n_trials),
        }


def run_vqt_benchmark(n_qubits: int = 3, n_trials: int = 3) -> Dict[str, Any]:
    return VQTBenchmark(n_qubits=n_qubits).run_full_benchmark(n_trials)


if __name__ == "__main__":
    results = run_vqt_benchmark(n_qubits=3, n_trials=2)
    print("VQT benchmark complete!")

"""
Hardware-Efficient Ansatz (HEA) Benchmark
=========================================

Algorithm #19 - Tier 3 (Optional)
Purpose: Study general-purpose variational ansatz performance

Key metrics:
- Expressibility
- Entanglement capability
- Optimization landscape

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


def hea_circuit_v1(params: np.ndarray, n_qubits: int, n_layers: int = 3):
    """Hardware-efficient ansatz variant 1: RY-RZ + linear CNOT."""
    param_idx = 0
    
    for layer in range(n_layers):
        # Single-qubit rotations
        for i in range(n_qubits):
            qml.RY(params[param_idx], wires=i)
            param_idx += 1
            qml.RZ(params[param_idx], wires=i)
            param_idx += 1
        
        # Linear entangling
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])


def hea_circuit_v2(params: np.ndarray, n_qubits: int, n_layers: int = 3):
    """Hardware-efficient ansatz variant 2: RY-RZ + circular CNOT."""
    param_idx = 0
    
    for layer in range(n_layers):
        for i in range(n_qubits):
            qml.RY(params[param_idx], wires=i)
            param_idx += 1
            qml.RZ(params[param_idx], wires=i)
            param_idx += 1
        
        # Circular entangling
        for i in range(n_qubits):
            qml.CNOT(wires=[i, (i + 1) % n_qubits])


def hea_circuit_v3(params: np.ndarray, n_qubits: int, n_layers: int = 3):
    """Hardware-efficient ansatz variant 3: Full rotation + all-to-all."""
    param_idx = 0
    
    for layer in range(n_layers):
        for i in range(n_qubits):
            qml.Rot(
                params[param_idx],
                params[param_idx + 1],
                params[param_idx + 2],
                wires=i
            )
            param_idx += 3
        
        # All-to-all entangling (for small systems)
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                qml.CZ(wires=[i, j])


def get_random_hamiltonian(n_qubits: int, seed: int = 42) -> qml.Hamiltonian:
    """Generate random Hamiltonian for testing."""
    np.random.seed(seed)
    coeffs = []
    ops = []
    
    # Random ZZ interactions
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            coeffs.append(np.random.uniform(-1, 1))
            ops.append(qml.PauliZ(i) @ qml.PauliZ(j))
    
    # Random single-qubit terms
    for i in range(n_qubits):
        coeffs.append(np.random.uniform(-0.5, 0.5))
        ops.append(qml.PauliX(i))
        coeffs.append(np.random.uniform(-0.5, 0.5))
        ops.append(qml.PauliZ(i))
    
    return qml.Hamiltonian(coeffs, ops)


def compute_expressibility(dev, params_list: List[np.ndarray], 
                           n_qubits: int, n_layers: int,
                           circuit_fn) -> float:
    """Estimate ansatz expressibility via state overlap distribution."""
    n_samples = len(params_list)
    overlaps = []
    
    @qml.qnode(dev)
    def circuit(params):
        circuit_fn(params, n_qubits, n_layers)
        return qml.state()
    
    for i in range(min(n_samples, 10)):
        for j in range(i + 1, min(n_samples, 10)):
            state1 = circuit(params_list[i])
            state2 = circuit(params_list[j])
            overlap = np.abs(np.vdot(state1, state2)) ** 2
            overlaps.append(overlap)
    
    if overlaps:
        # Expressibility metric: KL divergence from Haar
        return float(np.var(overlaps))
    return 0.0


@dataclass
class HEABenchmark:
    """Hardware-Efficient Ansatz benchmark."""
    
    n_qubits: int = 4
    n_layers: int = 3
    ansatz_variant: str = 'v1'  # 'v1', 'v2', 'v3'
    max_iterations: int = 30
    with_noise: bool = True
    noise_strength: float = 0.01
    
    def __post_init__(self):
        self.hamiltonian = get_random_hamiltonian(self.n_qubits)
        
        if self.ansatz_variant == 'v1':
            self.circuit_fn = hea_circuit_v1
            self.n_params = 2 * self.n_qubits * self.n_layers
        elif self.ansatz_variant == 'v2':
            self.circuit_fn = hea_circuit_v2
            self.n_params = 2 * self.n_qubits * self.n_layers
        else:
            self.circuit_fn = hea_circuit_v3
            self.n_params = 3 * self.n_qubits * self.n_layers
        
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
                            self.circuit_fn(params, self.n_qubits, self.n_layers)
                            return qml.expval(self.hamiltonian)
                        
                        params = pnp.random.uniform(
                            -np.pi, np.pi, self.n_params, requires_grad=True
                        )
                        opt = qml.GradientDescentOptimizer(stepsize=0.1)
                        
                        energies = []
                        for _ in range(self.max_iterations):
                            params, energy = opt.step_and_cost(circuit, params)
                            energies.append(float(energy))
                        
                        final_energy = energies[-1]
                        
                        # Convergence metric: how much did energy drop?
                        energy_drop = energies[0] - energies[-1] if len(energies) > 1 else 0
                        
                        success = True
                        error_msg = ""
                        
                    except Exception as e:
                        final_energy = float('nan')
                        energy_drop = float('nan')
                        success = False
                        error_msg = str(e)
            
            results.append(BenchmarkResult(
                algorithm='HEA',
                device_name=device_name,
                mode=mode,
                n_qubits=self.n_qubits,
                execution_time_seconds=timer.seconds,
                peak_memory_mb=mem.peak_mb,
                result_value=final_energy,
                secondary_value=energy_drop,
                with_noise=self.with_noise,
                success=success,
                error_message=error_msg,
                extra_data={
                    'ansatz_variant': self.ansatz_variant,
                    'n_layers': self.n_layers
                }
            ))
        
        self.results.extend(results)
        return results
    
    def compare_lret_modes(self, n_trials: int = 3) -> Dict[str, List[BenchmarkResult]]:
        print(f"\n{'='*60}\nHEA LRET Mode Comparison\n{'='*60}")
        return {mode: self.run_single_device('qlret.mixed', mode, n_trials) 
                for mode in ['sequential', 'batched', 'parallel', 'openmp']}
    
    def compare_devices(self, n_trials: int = 3) -> Dict[str, List[BenchmarkResult]]:
        print(f"\n{'='*60}\nHEA Device Comparison\n{'='*60}")
        return {
            'qlret.mixed': self.run_single_device('qlret.mixed', 'sequential', n_trials),
            'default.mixed': self.run_single_device('default.mixed', 'default', n_trials),
        }
    
    def run_full_benchmark(self, n_trials: int = 3) -> Dict[str, Any]:
        return {
            'lret_modes': self.compare_lret_modes(n_trials),
            'device_comparison': self.compare_devices(n_trials),
        }


def run_hea_benchmark(n_qubits: int = 4, n_trials: int = 3) -> Dict[str, Any]:
    return HEABenchmark(n_qubits=n_qubits).run_full_benchmark(n_trials)


if __name__ == "__main__":
    results = run_hea_benchmark(n_qubits=4, n_trials=2)
    print("HEA benchmark complete!")

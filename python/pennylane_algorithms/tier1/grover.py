"""
Grover's Search Algorithm Benchmark
====================================

Algorithm #6 - Tier 1 (Must Test)
Purpose: Unstructured database search

Key metrics:
- Success probability
- Optimal iterations
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.benchmark_utils import BenchmarkResult, Timer, MemoryTracker
from utils.device_factory import create_lret_device, create_comparison_device


def oracle_single_target(n_qubits: int, target: int):
    """Oracle that marks a single target state."""
    binary = format(target, f'0{n_qubits}b')
    for i, bit in enumerate(binary):
        if bit == '0':
            qml.PauliX(wires=i)
    
    if n_qubits == 2:
        qml.CZ(wires=[0, 1])
    else:
        qml.ctrl(qml.PauliZ, control=list(range(n_qubits - 1)))(wires=n_qubits - 1)
    
    for i, bit in enumerate(binary):
        if bit == '0':
            qml.PauliX(wires=i)


def diffusion_operator(n_qubits: int):
    """Grover diffusion operator."""
    for i in range(n_qubits):
        qml.Hadamard(wires=i)
        qml.PauliX(wires=i)
    
    if n_qubits == 2:
        qml.CZ(wires=[0, 1])
    else:
        qml.ctrl(qml.PauliZ, control=list(range(n_qubits - 1)))(wires=n_qubits - 1)
    
    for i in range(n_qubits):
        qml.PauliX(wires=i)
        qml.Hadamard(wires=i)


def grover_circuit(n_qubits: int, target: int, n_iterations: int):
    """Complete Grover's search circuit."""
    for i in range(n_qubits):
        qml.Hadamard(wires=i)
    
    for _ in range(n_iterations):
        oracle_single_target(n_qubits, target)
        diffusion_operator(n_qubits)


@dataclass
class GroverBenchmark:
    """Grover's Search benchmark."""
    
    n_qubits: int = 4
    target: int = 5
    with_noise: bool = True
    
    def __post_init__(self):
        N = 2 ** self.n_qubits
        self.optimal_iterations = int(np.round(np.pi / 4 * np.sqrt(N)))
        self.target = min(self.target, N - 1)
        self.results = []
    
    def run_single_device(self, device_name: str, mode: str = 'default', n_trials: int = 3) -> List[BenchmarkResult]:
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
                        def circuit():
                            grover_circuit(self.n_qubits, self.target, self.optimal_iterations)
                            return qml.probs(wires=range(self.n_qubits))
                        
                        probs = circuit()
                        success_prob = float(probs[self.target])
                        success = True
                        error_msg = ""
                    except Exception as e:
                        success_prob = 0.0
                        success = False
                        error_msg = str(e)
            
            results.append(BenchmarkResult(
                algorithm='Grover',
                device_name=device_name,
                mode=mode,
                n_qubits=self.n_qubits,
                execution_time_seconds=timer.seconds,
                peak_memory_mb=mem.peak_mb,
                result_value=success_prob,
                secondary_value=1.0 - success_prob,
                with_noise=self.with_noise,
                success=success,
                error_message=error_msg,
                extra_data={'target': self.target, 'iterations': self.optimal_iterations}
            ))
        
        self.results.extend(results)
        return results
    
    def compare_lret_modes(self, n_trials: int = 3) -> Dict[str, List[BenchmarkResult]]:
        print(f"\n{'='*60}\nGrover LRET Mode Comparison\n{'='*60}")
        return {mode: self.run_single_device('qlret.mixed', mode, n_trials) 
                for mode in ['sequential', 'batched', 'parallel', 'openmp']}
    
    def compare_devices(self, n_trials: int = 3) -> Dict[str, List[BenchmarkResult]]:
        print(f"\n{'='*60}\nGrover Device Comparison\n{'='*60}")
        return {
            'qlret.mixed': self.run_single_device('qlret.mixed', 'sequential', n_trials),
            'default.mixed': self.run_single_device('default.mixed', 'default', n_trials),
        }
    
    def run_full_benchmark(self, n_trials: int = 3) -> Dict[str, Any]:
        return {
            'lret_modes': self.compare_lret_modes(n_trials),
            'device_comparison': self.compare_devices(n_trials),
        }


def run_grover_benchmark(n_qubits: int = 4, n_trials: int = 3) -> Dict[str, Any]:
    return GroverBenchmark(n_qubits=n_qubits).run_full_benchmark(n_trials)


if __name__ == "__main__":
    results = run_grover_benchmark(n_qubits=4, n_trials=2)
    print("Grover benchmark complete!")

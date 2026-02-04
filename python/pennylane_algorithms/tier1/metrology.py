"""
Quantum Metrology Benchmark
============================

Algorithm #7 - Tier 1 (Must Test)
Purpose: Quantum-enhanced parameter estimation using GHZ states

Key metrics:
- Quantum Fisher Information (QFI)
- Phase sensitivity
- Heisenberg scaling

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


def prepare_ghz_state(n_qubits: int):
    """Prepare GHZ state: (|00...0⟩ + |11...1⟩)/√2"""
    qml.Hadamard(wires=0)
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])


def phase_encoding(theta: float, n_qubits: int):
    """Encode phase on all qubits."""
    for i in range(n_qubits):
        qml.RZ(theta, wires=i)


def ghz_interferometry(theta: float, n_qubits: int):
    """GHZ-based interferometry circuit."""
    prepare_ghz_state(n_qubits)
    phase_encoding(theta, n_qubits)
    
    for i in range(n_qubits - 1, 0, -1):
        qml.CNOT(wires=[i - 1, i])
    qml.Hadamard(wires=0)


def compute_qfi_numerical(dev, n_qubits: int, theta: float = 0.1, delta: float = 0.01) -> float:
    """Compute QFI numerically via finite differences."""
    @qml.qnode(dev)
    def circuit(t):
        ghz_interferometry(t, n_qubits)
        return qml.probs(wires=range(n_qubits))
    
    probs_plus = circuit(theta + delta)
    probs_minus = circuit(theta - delta)
    
    grad = (probs_plus - probs_minus) / (2 * delta)
    probs = circuit(theta)
    
    qfi = 0.0
    for i in range(len(probs)):
        if probs[i] > 1e-10:
            qfi += grad[i] ** 2 / probs[i]
    
    return float(4 * qfi)


@dataclass
class MetrologyBenchmark:
    """Quantum Metrology benchmark."""
    
    n_qubits: int = 4
    theta: float = 0.1
    with_noise: bool = True
    
    def __post_init__(self):
        self.classical_limit = self.n_qubits
        self.heisenberg_limit = self.n_qubits ** 2
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
                        
                        qfi = compute_qfi_numerical(dev, self.n_qubits, self.theta)
                        heisenberg_ratio = qfi / self.heisenberg_limit
                        success = True
                        error_msg = ""
                    except Exception as e:
                        qfi = 0.0
                        heisenberg_ratio = 0.0
                        success = False
                        error_msg = str(e)
            
            results.append(BenchmarkResult(
                algorithm='Metrology',
                device_name=device_name,
                mode=mode,
                n_qubits=self.n_qubits,
                execution_time_seconds=timer.seconds,
                peak_memory_mb=mem.peak_mb,
                result_value=qfi,
                secondary_value=heisenberg_ratio,
                with_noise=self.with_noise,
                success=success,
                error_message=error_msg,
                extra_data={'theta': self.theta, 'heisenberg_limit': self.heisenberg_limit}
            ))
        
        self.results.extend(results)
        return results
    
    def compare_lret_modes(self, n_trials: int = 3) -> Dict[str, List[BenchmarkResult]]:
        print(f"\n{'='*60}\nMetrology LRET Mode Comparison\n{'='*60}")
        return {mode: self.run_single_device('qlret.mixed', mode, n_trials) 
                for mode in ['sequential', 'batched', 'parallel', 'openmp']}
    
    def compare_devices(self, n_trials: int = 3) -> Dict[str, List[BenchmarkResult]]:
        print(f"\n{'='*60}\nMetrology Device Comparison\n{'='*60}")
        return {
            'qlret.mixed': self.run_single_device('qlret.mixed', 'sequential', n_trials),
            'default.mixed': self.run_single_device('default.mixed', 'default', n_trials),
        }
    
    def run_full_benchmark(self, n_trials: int = 3) -> Dict[str, Any]:
        return {
            'lret_modes': self.compare_lret_modes(n_trials),
            'device_comparison': self.compare_devices(n_trials),
        }


def run_metrology_benchmark(n_qubits: int = 4, n_trials: int = 3) -> Dict[str, Any]:
    return MetrologyBenchmark(n_qubits=n_qubits).run_full_benchmark(n_trials)


if __name__ == "__main__":
    results = run_metrology_benchmark(n_qubits=4, n_trials=2)
    print("Metrology benchmark complete!")

"""
QPE (Quantum Phase Estimation) Benchmark
=========================================

Algorithm #5 - Tier 1 (Must Test)
Purpose: Estimate eigenvalue phases

Key metrics:
- Phase estimation accuracy
- Bit precision
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


def qpe_circuit(n_counting: int, target_phase: float):
    """QPE circuit for phase estimation."""
    n_total = n_counting + 1
    
    qml.PauliX(wires=n_counting)
    
    for i in range(n_counting):
        qml.Hadamard(wires=i)
    
    for i in range(n_counting):
        power = 2 ** i
        for _ in range(power):
            qml.ctrl(qml.PhaseShift, control=i)(2 * np.pi * target_phase, wires=n_counting)
    
    for i in range(n_counting // 2):
        qml.SWAP(wires=[i, n_counting - 1 - i])
    
    for i in range(n_counting):
        qml.Hadamard(wires=i)
        for j in range(i + 1, n_counting):
            qml.ctrl(qml.PhaseShift, control=j)(-np.pi / (2 ** (j - i)), wires=i)


def estimate_phase_from_probs(probs: np.ndarray, n_counting: int) -> float:
    """Extract phase estimate from measurement probabilities."""
    most_likely = np.argmax(probs[:2**n_counting])
    return most_likely / (2 ** n_counting)


@dataclass
class QPEBenchmark:
    """QPE benchmark."""
    
    n_counting_qubits: int = 4
    target_phase: float = 0.25
    with_noise: bool = True
    
    def __post_init__(self):
        self.n_qubits = self.n_counting_qubits + 1
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
                            qpe_circuit(self.n_counting_qubits, self.target_phase)
                            return qml.probs(wires=range(self.n_counting_qubits))
                        
                        probs = circuit()
                        estimated_phase = estimate_phase_from_probs(probs, self.n_counting_qubits)
                        phase_error = abs(estimated_phase - self.target_phase)
                        success = True
                        error_msg = ""
                    except Exception as e:
                        estimated_phase = 0.0
                        phase_error = 1.0
                        success = False
                        error_msg = str(e)
            
            results.append(BenchmarkResult(
                algorithm='QPE',
                device_name=device_name,
                mode=mode,
                n_qubits=self.n_qubits,
                execution_time_seconds=timer.seconds,
                peak_memory_mb=mem.peak_mb,
                result_value=estimated_phase,
                secondary_value=phase_error,
                with_noise=self.with_noise,
                success=success,
                error_message=error_msg,
                extra_data={'target_phase': self.target_phase}
            ))
        
        self.results.extend(results)
        return results
    
    def compare_lret_modes(self, n_trials: int = 3) -> Dict[str, List[BenchmarkResult]]:
        print(f"\n{'='*60}\nQPE LRET Mode Comparison\n{'='*60}")
        return {mode: self.run_single_device('qlret.mixed', mode, n_trials) 
                for mode in ['sequential', 'batched', 'parallel', 'openmp']}
    
    def compare_devices(self, n_trials: int = 3) -> Dict[str, List[BenchmarkResult]]:
        print(f"\n{'='*60}\nQPE Device Comparison\n{'='*60}")
        return {
            'qlret.mixed': self.run_single_device('qlret.mixed', 'sequential', n_trials),
            'default.mixed': self.run_single_device('default.mixed', 'default', n_trials),
        }
    
    def run_full_benchmark(self, n_trials: int = 3) -> Dict[str, Any]:
        return {
            'lret_modes': self.compare_lret_modes(n_trials),
            'device_comparison': self.compare_devices(n_trials),
        }


def run_qpe_benchmark(n_qubits: int = 5, n_trials: int = 3) -> Dict[str, Any]:
    return QPEBenchmark(n_counting_qubits=n_qubits-1).run_full_benchmark(n_trials)


if __name__ == "__main__":
    results = run_qpe_benchmark(n_qubits=5, n_trials=2)
    print("QPE benchmark complete!")

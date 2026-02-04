"""
QFT (Quantum Fourier Transform) Benchmark
==========================================

Algorithm #4 - Tier 1 (Must Test)
Purpose: Test QFT fidelity and roundtrip accuracy

Key metrics:
- State fidelity
- Roundtrip accuracy
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


def qft_circuit(n_qubits: int):
    """Standard QFT circuit."""
    for i in range(n_qubits):
        qml.Hadamard(wires=i)
        for j in range(i + 1, n_qubits):
            qml.ctrl(qml.PhaseShift, control=j)(np.pi / (2 ** (j - i)), wires=i)
    
    for i in range(n_qubits // 2):
        qml.SWAP(wires=[i, n_qubits - 1 - i])


def inverse_qft_circuit(n_qubits: int):
    """Inverse QFT circuit."""
    for i in range(n_qubits // 2):
        qml.SWAP(wires=[i, n_qubits - 1 - i])
    
    for i in range(n_qubits - 1, -1, -1):
        for j in range(n_qubits - 1, i, -1):
            qml.ctrl(qml.PhaseShift, control=j)(-np.pi / (2 ** (j - i)), wires=i)
        qml.Hadamard(wires=i)


@dataclass
class QFTBenchmark:
    """QFT Fidelity benchmark."""
    
    n_qubits: int = 4
    with_noise: bool = True
    
    def __post_init__(self):
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
                        def qft_roundtrip():
                            qml.PauliX(wires=0)
                            qft_circuit(self.n_qubits)
                            inverse_qft_circuit(self.n_qubits)
                            return qml.probs(wires=range(self.n_qubits))
                        
                        probs = qft_roundtrip()
                        fidelity = float(probs[1])
                        success = True
                        error_msg = ""
                    except Exception as e:
                        fidelity = 0.0
                        success = False
                        error_msg = str(e)
            
            results.append(BenchmarkResult(
                algorithm='QFT',
                device_name=device_name,
                mode=mode,
                n_qubits=self.n_qubits,
                execution_time_seconds=timer.seconds,
                peak_memory_mb=mem.peak_mb,
                result_value=fidelity,
                secondary_value=1.0 - fidelity,
                with_noise=self.with_noise,
                success=success,
                error_message=error_msg,
                extra_data={}
            ))
        
        self.results.extend(results)
        return results
    
    def compare_lret_modes(self, n_trials: int = 3) -> Dict[str, List[BenchmarkResult]]:
        print(f"\n{'='*60}\nQFT LRET Mode Comparison\n{'='*60}")
        return {mode: self.run_single_device('qlret.mixed', mode, n_trials) 
                for mode in ['sequential', 'batched', 'parallel', 'openmp']}
    
    def compare_devices(self, n_trials: int = 3) -> Dict[str, List[BenchmarkResult]]:
        print(f"\n{'='*60}\nQFT Device Comparison\n{'='*60}")
        return {
            'qlret.mixed': self.run_single_device('qlret.mixed', 'sequential', n_trials),
            'default.mixed': self.run_single_device('default.mixed', 'default', n_trials),
        }
    
    def run_full_benchmark(self, n_trials: int = 3) -> Dict[str, Any]:
        return {
            'lret_modes': self.compare_lret_modes(n_trials),
            'device_comparison': self.compare_devices(n_trials),
        }


def run_qft_benchmark(n_qubits: int = 4, n_trials: int = 3) -> Dict[str, Any]:
    return QFTBenchmark(n_qubits=n_qubits).run_full_benchmark(n_trials)


if __name__ == "__main__":
    results = run_qft_benchmark(n_qubits=4, n_trials=2)
    print("QFT benchmark complete!")

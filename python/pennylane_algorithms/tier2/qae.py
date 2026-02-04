"""
Quantum Amplitude Estimation (QAE) Benchmark
=============================================

Algorithm #11 - Tier 2 (Should Test)
Purpose: Estimate probability amplitudes with quantum speedup

Key metrics:
- Estimation accuracy
- Query complexity
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


def state_preparation_A(target_amplitude: float, n_qubits: int):
    """Prepare state with known amplitude a: |ψ⟩ = √a|good⟩ + √(1-a)|bad⟩"""
    # Use RY rotation to create superposition
    theta = 2 * np.arcsin(np.sqrt(target_amplitude))
    qml.RY(theta, wires=n_qubits - 1)
    
    # Mark the good state (last qubit = 1)
    # For simplicity, good = |1⟩ on last qubit


def grover_operator_Q(target_amplitude: float, n_qubits: int):
    """Grover operator Q = -AS_0A^{-1}S_χ for amplitude estimation."""
    # S_χ: flip sign of good states (|1⟩ on last qubit)
    qml.PauliZ(wires=n_qubits - 1)
    
    # A^{-1}: inverse state preparation
    theta = 2 * np.arcsin(np.sqrt(target_amplitude))
    qml.RY(-theta, wires=n_qubits - 1)
    
    # S_0: flip sign of |0⟩ (reflection about zero)
    qml.PauliX(wires=n_qubits - 1)
    qml.PauliZ(wires=n_qubits - 1)
    qml.PauliX(wires=n_qubits - 1)
    
    # A: state preparation again
    qml.RY(theta, wires=n_qubits - 1)


def qae_circuit_iterative(target_amplitude: float, n_iterations: int, n_qubits: int):
    """Iterative QAE circuit."""
    # Initial state preparation
    state_preparation_A(target_amplitude, n_qubits)
    
    # Apply Q operator multiple times
    for _ in range(n_iterations):
        grover_operator_Q(target_amplitude, n_qubits)


def qpe_qae_circuit(
    target_amplitude: float,
    n_counting_qubits: int,
    n_state_qubits: int
):
    """QPE-based QAE circuit."""
    total_qubits = n_counting_qubits + n_state_qubits
    
    # Prepare state A|0⟩ on state register
    state_preparation_A(target_amplitude, n_state_qubits)
    
    # Hadamard on counting qubits
    for i in range(n_counting_qubits):
        qml.Hadamard(wires=i)
    
    # Controlled Q^{2^k} operations
    for k in range(n_counting_qubits):
        power = 2 ** k
        # Controlled Grover operator
        for _ in range(power):
            # Simplified: controlled-RY for demonstration
            theta = 2 * np.arcsin(np.sqrt(target_amplitude))
            qml.ctrl(qml.RY, control=k)(2 * theta, wires=n_counting_qubits)
    
    # Inverse QFT on counting register
    for i in range(n_counting_qubits // 2):
        qml.SWAP(wires=[i, n_counting_qubits - 1 - i])
    
    for i in range(n_counting_qubits):
        qml.Hadamard(wires=i)
        for j in range(i + 1, n_counting_qubits):
            qml.ctrl(qml.PhaseShift, control=j)(
                -np.pi / (2 ** (j - i)), wires=i
            )


def estimate_amplitude_from_measurement(
    probs: np.ndarray,
    n_counting_qubits: int
) -> float:
    """Estimate amplitude from QPE measurement outcome."""
    # Find most likely measurement outcome
    most_likely = np.argmax(probs)
    
    # Convert to phase estimate
    theta_estimate = 2 * np.pi * most_likely / (2 ** n_counting_qubits)
    
    # Amplitude is sin^2(theta/2)
    amplitude_estimate = np.sin(theta_estimate / 2) ** 2
    
    return amplitude_estimate


@dataclass
class QAEBenchmark:
    """Quantum Amplitude Estimation benchmark."""
    
    n_counting_qubits: int = 4
    n_state_qubits: int = 2
    target_amplitude: float = 0.3
    with_noise: bool = True
    noise_strength: float = 0.01
    
    def __post_init__(self):
        self.n_qubits = self.n_counting_qubits + self.n_state_qubits
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
                        def qae_circuit():
                            qpe_qae_circuit(
                                self.target_amplitude,
                                self.n_counting_qubits,
                                self.n_state_qubits
                            )
                            return qml.probs(wires=range(self.n_counting_qubits))
                        
                        probs = qae_circuit()
                        
                        amplitude_estimate = estimate_amplitude_from_measurement(
                            probs, self.n_counting_qubits
                        )
                        
                        estimation_error = abs(amplitude_estimate - self.target_amplitude)
                        success = True
                        error_msg = ""
                        
                    except Exception as e:
                        amplitude_estimate = float('nan')
                        estimation_error = float('nan')
                        success = False
                        error_msg = str(e)
            
            results.append(BenchmarkResult(
                algorithm='QAE',
                device_name=device_name,
                mode=mode,
                n_qubits=self.n_qubits,
                execution_time_seconds=timer.seconds,
                peak_memory_mb=mem.peak_mb,
                result_value=amplitude_estimate,
                secondary_value=estimation_error,
                with_noise=self.with_noise,
                success=success,
                error_message=error_msg,
                extra_data={
                    'target_amplitude': self.target_amplitude,
                    'n_counting_qubits': self.n_counting_qubits
                }
            ))
        
        self.results.extend(results)
        return results
    
    def compare_lret_modes(self, n_trials: int = 3) -> Dict[str, List[BenchmarkResult]]:
        print(f"\n{'='*60}\nQAE LRET Mode Comparison\n{'='*60}")
        return {mode: self.run_single_device('qlret.mixed', mode, n_trials) 
                for mode in ['sequential', 'batched', 'parallel', 'openmp']}
    
    def compare_devices(self, n_trials: int = 3) -> Dict[str, List[BenchmarkResult]]:
        print(f"\n{'='*60}\nQAE Device Comparison\n{'='*60}")
        return {
            'qlret.mixed': self.run_single_device('qlret.mixed', 'sequential', n_trials),
            'default.mixed': self.run_single_device('default.mixed', 'default', n_trials),
        }
    
    def run_full_benchmark(self, n_trials: int = 3) -> Dict[str, Any]:
        return {
            'lret_modes': self.compare_lret_modes(n_trials),
            'device_comparison': self.compare_devices(n_trials),
        }


def run_qae_benchmark(n_counting_qubits: int = 4, n_trials: int = 3) -> Dict[str, Any]:
    return QAEBenchmark(n_counting_qubits=n_counting_qubits).run_full_benchmark(n_trials)


if __name__ == "__main__":
    results = run_qae_benchmark(n_counting_qubits=4, n_trials=2)
    print("QAE benchmark complete!")

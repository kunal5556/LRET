"""
QNN (Quantum Neural Network) Classifier Benchmark
==================================================

Algorithm #3 - Tier 1 (Must Test)
Purpose: Binary classification with quantum circuits

Key metrics:
- Classification accuracy
- Training convergence
- Generalization

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


def generate_classification_data(n_samples: int = 50, n_features: int = 4, seed: int = 42):
    """Generate synthetic classification data."""
    np.random.seed(seed)
    X = np.random.randn(n_samples, n_features)
    y = (np.sum(X[:, :2], axis=1) > 0).astype(float)
    return X, y


def qnn_circuit(x: np.ndarray, params: np.ndarray, n_qubits: int, n_layers: int = 2):
    """QNN circuit with data encoding and variational layers."""
    for i in range(n_qubits):
        qml.RY(x[i % len(x)] * np.pi, wires=i)
    
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
class QNNBenchmark:
    """QNN Classifier benchmark."""
    
    n_qubits: int = 4
    n_layers: int = 2
    n_samples: int = 30
    n_epochs: int = 10
    with_noise: bool = True
    
    def __post_init__(self):
        X, y = generate_classification_data(self.n_samples, self.n_qubits)
        split = int(0.8 * self.n_samples)
        self.X_train, self.X_test = X[:split], X[split:]
        self.y_train, self.y_test = y[:split], y[split:]
        self.n_params = 2 * self.n_qubits * self.n_layers
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
                        
                        @qml.qnode(dev, interface='autograd')
                        def qnn(x, params):
                            qnn_circuit(x, params, self.n_qubits, self.n_layers)
                            return qml.expval(qml.PauliZ(0))
                        
                        params = pnp.random.uniform(-np.pi, np.pi, self.n_params, requires_grad=True)
                        opt = qml.AdamOptimizer(stepsize=0.1)
                        
                        for epoch in range(self.n_epochs):
                            for x, y in zip(self.X_train, self.y_train):
                                def cost(params):
                                    pred = qnn(x, params)
                                    return (pred - (2 * y - 1)) ** 2
                                params = opt.step(cost, params)
                        
                        correct = sum(1 for x, y in zip(self.X_test, self.y_test) 
                                     if (qnn(x, params) > 0) == y)
                        accuracy = correct / len(self.y_test)
                        success = True
                        error_msg = ""
                    except Exception as e:
                        accuracy = 0.0
                        success = False
                        error_msg = str(e)
            
            results.append(BenchmarkResult(
                algorithm='QNN-Classifier',
                device_name=device_name,
                mode=mode,
                n_qubits=self.n_qubits,
                execution_time_seconds=timer.seconds,
                peak_memory_mb=mem.peak_mb,
                result_value=accuracy,
                secondary_value=0.0,
                with_noise=self.with_noise,
                success=success,
                error_message=error_msg,
                extra_data={'n_epochs': self.n_epochs, 'n_samples': self.n_samples}
            ))
        
        self.results.extend(results)
        return results
    
    def compare_lret_modes(self, n_trials: int = 3) -> Dict[str, List[BenchmarkResult]]:
        print(f"\n{'='*60}\nQNN LRET Mode Comparison\n{'='*60}")
        return {mode: self.run_single_device('qlret.mixed', mode, n_trials) 
                for mode in ['sequential', 'batched', 'parallel', 'openmp']}
    
    def compare_devices(self, n_trials: int = 3) -> Dict[str, List[BenchmarkResult]]:
        print(f"\n{'='*60}\nQNN Device Comparison\n{'='*60}")
        return {
            'qlret.mixed': self.run_single_device('qlret.mixed', 'sequential', n_trials),
            'default.mixed': self.run_single_device('default.mixed', 'default', n_trials),
        }
    
    def run_full_benchmark(self, n_trials: int = 3) -> Dict[str, Any]:
        return {
            'lret_modes': self.compare_lret_modes(n_trials),
            'device_comparison': self.compare_devices(n_trials),
        }


def run_qnn_benchmark(n_qubits: int = 4, n_trials: int = 3) -> Dict[str, Any]:
    return QNNBenchmark(n_qubits=n_qubits).run_full_benchmark(n_trials)


if __name__ == "__main__":
    results = run_qnn_benchmark(n_qubits=4, n_trials=2)
    print("QNN benchmark complete!")

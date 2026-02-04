"""
Sub-sampling QNN Benchmark
==========================

Algorithm #18 - Tier 3 (Optional)
Purpose: Large-scale QNN training with data sub-sampling

Uses stochastic gradient descent with mini-batches,
critical for scaling to larger datasets.

Key metrics:
- Training time with batching
- Accuracy vs batch size
- Memory efficiency

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


def generate_dataset(
    n_samples: int = 100,
    n_features: int = 4,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic classification dataset."""
    np.random.seed(seed)
    
    X = np.random.randn(n_samples, n_features)
    
    # Labels based on non-linear function
    weights = np.random.randn(n_features)
    scores = X @ weights + 0.1 * np.sum(X ** 2, axis=1)
    y = (scores > np.median(scores)).astype(float)
    
    return X, y


def qnn_circuit(x: np.ndarray, params: np.ndarray, n_qubits: int, n_layers: int = 2):
    """QNN circuit with data encoding and variational layers."""
    param_idx = 0
    
    # Data encoding
    for i in range(n_qubits):
        qml.RY(x[i % len(x)] * np.pi, wires=i)
    
    # Variational layers
    for layer in range(n_layers):
        for i in range(n_qubits):
            qml.RY(params[param_idx], wires=i)
            param_idx += 1
            qml.RZ(params[param_idx], wires=i)
            param_idx += 1
        
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])


def create_data_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool = True
):
    """Create data loader for mini-batch training."""
    n_samples = len(X)
    indices = np.arange(n_samples)
    
    if shuffle:
        np.random.shuffle(indices)
    
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_idx = indices[start:end]
        yield X[batch_idx], y[batch_idx]


@dataclass
class SubsamplingQNNBenchmark:
    """Sub-sampling QNN benchmark."""
    
    n_qubits: int = 4
    n_layers: int = 2
    n_samples: int = 100
    batch_size: int = 10
    n_epochs: int = 5
    with_noise: bool = True
    noise_strength: float = 0.01
    
    def __post_init__(self):
        X, y = generate_dataset(self.n_samples, self.n_qubits)
        split = int(0.8 * self.n_samples)
        self.X_train, self.X_test = X[:split], X[split:]
        self.y_train, self.y_test = y[:split], y[split:]
        
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
                        def qnn(x, params):
                            qnn_circuit(x, params, self.n_qubits, self.n_layers)
                            return qml.expval(qml.PauliZ(0))
                        
                        params = pnp.random.uniform(
                            -np.pi, np.pi, self.n_params, requires_grad=True
                        )
                        opt = qml.AdamOptimizer(stepsize=0.1)
                        
                        # Training with mini-batches
                        for epoch in range(self.n_epochs):
                            for X_batch, y_batch in create_data_loader(
                                self.X_train, self.y_train, self.batch_size
                            ):
                                def batch_cost(params):
                                    predictions = []
                                    for x in X_batch:
                                        pred = qnn(x, params)
                                        predictions.append(pred)
                                    predictions = pnp.array(predictions)
                                    # MSE loss
                                    return pnp.mean((predictions - (2 * y_batch - 1)) ** 2)
                                
                                params = opt.step(batch_cost, params)
                        
                        # Evaluate accuracy
                        correct = 0
                        for x, y in zip(self.X_test, self.y_test):
                            pred = qnn(x, params)
                            predicted_class = 1 if pred > 0 else 0
                            if predicted_class == y:
                                correct += 1
                        
                        accuracy = correct / len(self.y_test)
                        success = True
                        error_msg = ""
                        
                    except Exception as e:
                        accuracy = 0.0
                        success = False
                        error_msg = str(e)
            
            results.append(BenchmarkResult(
                algorithm='SubsamplingQNN',
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
                extra_data={
                    'batch_size': self.batch_size,
                    'n_epochs': self.n_epochs,
                    'n_samples': self.n_samples
                }
            ))
        
        self.results.extend(results)
        return results
    
    def compare_lret_modes(self, n_trials: int = 3) -> Dict[str, List[BenchmarkResult]]:
        print(f"\n{'='*60}\nSubsampling QNN LRET Mode Comparison\n{'='*60}")
        return {mode: self.run_single_device('qlret.mixed', mode, n_trials) 
                for mode in ['sequential', 'batched', 'parallel', 'openmp']}
    
    def compare_devices(self, n_trials: int = 3) -> Dict[str, List[BenchmarkResult]]:
        print(f"\n{'='*60}\nSubsampling QNN Device Comparison\n{'='*60}")
        return {
            'qlret.mixed': self.run_single_device('qlret.mixed', 'sequential', n_trials),
            'default.mixed': self.run_single_device('default.mixed', 'default', n_trials),
        }
    
    def run_full_benchmark(self, n_trials: int = 3) -> Dict[str, Any]:
        return {
            'lret_modes': self.compare_lret_modes(n_trials),
            'device_comparison': self.compare_devices(n_trials),
        }


def run_subsampling_qnn_benchmark(n_qubits: int = 4, n_trials: int = 3) -> Dict[str, Any]:
    return SubsamplingQNNBenchmark(n_qubits=n_qubits).run_full_benchmark(n_trials)


if __name__ == "__main__":
    results = run_subsampling_qnn_benchmark(n_qubits=4, n_trials=2)
    print("Subsampling QNN benchmark complete!")

"""
Quantum SVM (QSVM) Benchmark
============================

Algorithm #10 - Tier 2 (Should Test)
Purpose: Quantum kernel-based classification

Key metrics:
- Classification accuracy
- Kernel computation time
- Training efficiency

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


def generate_classification_data(
    n_samples: int = 50,
    n_features: int = 4,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic classification data."""
    np.random.seed(seed)
    
    # Class 0: centered at origin
    X0 = np.random.randn(n_samples // 2, n_features) * 0.5
    y0 = np.zeros(n_samples // 2)
    
    # Class 1: centered at [1, 1, ...]
    X1 = np.random.randn(n_samples // 2, n_features) * 0.5 + 1.0
    y1 = np.ones(n_samples // 2)
    
    X = np.vstack([X0, X1])
    y = np.hstack([y0, y1])
    
    # Shuffle
    perm = np.random.permutation(n_samples)
    X, y = X[perm], y[perm]
    
    # Train/test split
    split = int(0.8 * n_samples)
    return X[:split], X[split:], y[:split], y[split:]


def feature_map(x: np.ndarray, n_qubits: int):
    """Quantum feature map for QSVM."""
    n_features = min(len(x), n_qubits)
    
    # First layer
    for i in range(n_qubits):
        qml.Hadamard(wires=i)
        qml.RZ(x[i % n_features], wires=i)
    
    # Entangling layer
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])
        qml.RZ((x[i % n_features] * x[(i + 1) % n_features]), wires=i + 1)
        qml.CNOT(wires=[i, i + 1])
    
    # Second layer
    for i in range(n_qubits):
        qml.RY(x[i % n_features] * np.pi, wires=i)


def quantum_kernel(x1: np.ndarray, x2: np.ndarray, dev, n_qubits: int) -> float:
    """Compute quantum kernel between two data points."""
    
    @qml.qnode(dev)
    def kernel_circuit(x1, x2):
        feature_map(x1, n_qubits)
        qml.adjoint(feature_map)(x2, n_qubits)
        return qml.probs(wires=range(n_qubits))
    
    probs = kernel_circuit(x1, x2)
    return float(probs[0])  # Probability of |00...0⟩


def compute_kernel_matrix(
    X: np.ndarray,
    dev,
    n_qubits: int,
    Y: Optional[np.ndarray] = None
) -> np.ndarray:
    """Compute full kernel matrix."""
    if Y is None:
        Y = X
    
    n_x, n_y = len(X), len(Y)
    K = np.zeros((n_x, n_y))
    
    for i in range(n_x):
        for j in range(n_y):
            K[i, j] = quantum_kernel(X[i], Y[j], dev, n_qubits)
    
    return K


def svm_train_simple(K: np.ndarray, y: np.ndarray, C: float = 1.0) -> np.ndarray:
    """Simple SVM training using kernel matrix (ridge regression approximation)."""
    n = len(y)
    # Convert labels to {-1, +1}
    y_signed = 2 * y - 1
    
    # Ridge regression on kernel: alpha = (K + lambda*I)^{-1} y
    alpha = np.linalg.solve(K + (1/C) * np.eye(n), y_signed)
    return alpha


def svm_predict(K_test: np.ndarray, alpha: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    """Predict using SVM."""
    y_signed = 2 * y_train - 1
    scores = K_test @ alpha
    return (np.sign(scores) + 1) / 2


@dataclass
class QSVMBenchmark:
    """QSVM benchmark."""
    
    n_qubits: int = 4
    n_samples: int = 30
    with_noise: bool = True
    noise_strength: float = 0.01
    
    def __post_init__(self):
        self.X_train, self.X_test, self.y_train, self.y_test = \
            generate_classification_data(self.n_samples, self.n_qubits)
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
                        
                        # Compute kernel matrices
                        K_train = compute_kernel_matrix(self.X_train, dev, self.n_qubits)
                        K_test = compute_kernel_matrix(
                            self.X_test, dev, self.n_qubits, self.X_train
                        )
                        
                        # Train and predict
                        alpha = svm_train_simple(K_train, self.y_train)
                        y_pred = svm_predict(K_test, alpha, self.y_train)
                        
                        accuracy = np.mean(y_pred == self.y_test)
                        success = True
                        error_msg = ""
                        
                    except Exception as e:
                        accuracy = 0.0
                        success = False
                        error_msg = str(e)
            
            results.append(BenchmarkResult(
                algorithm='QSVM',
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
                extra_data={'n_samples': self.n_samples}
            ))
        
        self.results.extend(results)
        return results
    
    def compare_lret_modes(self, n_trials: int = 3) -> Dict[str, List[BenchmarkResult]]:
        print(f"\n{'='*60}\nQSVM LRET Mode Comparison\n{'='*60}")
        return {mode: self.run_single_device('qlret.mixed', mode, n_trials) 
                for mode in ['sequential', 'batched', 'parallel', 'openmp']}
    
    def compare_devices(self, n_trials: int = 3) -> Dict[str, List[BenchmarkResult]]:
        print(f"\n{'='*60}\nQSVM Device Comparison\n{'='*60}")
        return {
            'qlret.mixed': self.run_single_device('qlret.mixed', 'sequential', n_trials),
            'default.mixed': self.run_single_device('default.mixed', 'default', n_trials),
        }
    
    def run_full_benchmark(self, n_trials: int = 3) -> Dict[str, Any]:
        return {
            'lret_modes': self.compare_lret_modes(n_trials),
            'device_comparison': self.compare_devices(n_trials),
        }


def run_qsvm_benchmark(n_qubits: int = 4, n_trials: int = 3) -> Dict[str, Any]:
    return QSVMBenchmark(n_qubits=n_qubits).run_full_benchmark(n_trials)


if __name__ == "__main__":
    results = run_qsvm_benchmark(n_qubits=4, n_trials=2)
    print("QSVM benchmark complete!")

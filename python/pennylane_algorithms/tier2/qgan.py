"""
Quantum GAN (qGAN) Benchmark
============================

Algorithm #13 - Tier 2 (Should Test)
Purpose: Quantum generative adversarial network for distribution learning

Key metrics:
- Distribution matching (KL divergence, Wasserstein distance)
- Training stability
- Sample quality

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


def generate_target_distribution(n_qubits: int, seed: int = 42) -> np.ndarray:
    """Generate a target probability distribution."""
    np.random.seed(seed)
    dim = 2 ** n_qubits
    
    # Create a structured distribution (e.g., Gaussian-like)
    x = np.arange(dim)
    probs = np.exp(-((x - dim/2) ** 2) / (2 * (dim/4) ** 2))
    probs = probs / np.sum(probs)
    
    return probs


def generator_circuit(params: np.ndarray, n_qubits: int, n_layers: int = 2):
    """Quantum generator circuit."""
    param_idx = 0
    
    # Initial Hadamards for superposition
    for i in range(n_qubits):
        qml.Hadamard(wires=i)
    
    # Variational layers
    for layer in range(n_layers):
        for i in range(n_qubits):
            qml.RY(params[param_idx], wires=i)
            param_idx += 1
            qml.RZ(params[param_idx], wires=i)
            param_idx += 1
        
        # Entangling
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])


def discriminator_circuit(params: np.ndarray, data: np.ndarray, n_qubits: int):
    """Quantum discriminator circuit."""
    # Encode data
    for i in range(min(len(data), n_qubits)):
        qml.RY(data[i] * np.pi, wires=i)
    
    # Variational layers
    param_idx = 0
    for i in range(n_qubits):
        qml.RY(params[param_idx], wires=i)
        param_idx += 1
        qml.RZ(params[param_idx], wires=i)
        param_idx += 1
    
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])


def kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
    """Compute KL divergence D_KL(p||q)."""
    p = np.clip(p, epsilon, 1.0)
    q = np.clip(q, epsilon, 1.0)
    return float(np.sum(p * np.log(p / q)))


def total_variation_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Compute total variation distance."""
    return float(0.5 * np.sum(np.abs(p - q)))


@dataclass
class QGANBenchmark:
    """Quantum GAN benchmark."""
    
    n_qubits: int = 3
    n_layers: int = 2
    max_iterations: int = 30
    with_noise: bool = True
    noise_strength: float = 0.01
    
    def __post_init__(self):
        self.target_dist = generate_target_distribution(self.n_qubits)
        self.n_gen_params = 2 * self.n_qubits * self.n_layers
        self.n_disc_params = 2 * self.n_qubits
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
                        def generator(params):
                            generator_circuit(params, self.n_qubits, self.n_layers)
                            return qml.probs(wires=range(self.n_qubits))
                        
                        # Simple generator training (minimize distance to target)
                        gen_params = pnp.random.uniform(
                            -np.pi, np.pi, self.n_gen_params, requires_grad=True
                        )
                        
                        def generator_loss(params):
                            gen_probs = generator(params)
                            return kl_divergence(self.target_dist, gen_probs)
                        
                        opt = qml.GradientDescentOptimizer(stepsize=0.1)
                        
                        for _ in range(self.max_iterations):
                            gen_params, loss = opt.step_and_cost(generator_loss, gen_params)
                        
                        # Evaluate final distribution
                        final_probs = generator(gen_params)
                        final_kl = kl_divergence(self.target_dist, final_probs)
                        final_tvd = total_variation_distance(self.target_dist, final_probs)
                        
                        success = True
                        error_msg = ""
                        
                    except Exception as e:
                        final_kl = float('nan')
                        final_tvd = float('nan')
                        success = False
                        error_msg = str(e)
            
            results.append(BenchmarkResult(
                algorithm='qGAN',
                device_name=device_name,
                mode=mode,
                n_qubits=self.n_qubits,
                execution_time_seconds=timer.seconds,
                peak_memory_mb=mem.peak_mb,
                result_value=final_kl,
                secondary_value=final_tvd,
                with_noise=self.with_noise,
                success=success,
                error_message=error_msg,
                extra_data={
                    'n_layers': self.n_layers,
                    'n_gen_params': self.n_gen_params
                }
            ))
        
        self.results.extend(results)
        return results
    
    def compare_lret_modes(self, n_trials: int = 3) -> Dict[str, List[BenchmarkResult]]:
        print(f"\n{'='*60}\nqGAN LRET Mode Comparison\n{'='*60}")
        return {mode: self.run_single_device('qlret.mixed', mode, n_trials) 
                for mode in ['sequential', 'batched', 'parallel', 'openmp']}
    
    def compare_devices(self, n_trials: int = 3) -> Dict[str, List[BenchmarkResult]]:
        print(f"\n{'='*60}\nqGAN Device Comparison\n{'='*60}")
        return {
            'qlret.mixed': self.run_single_device('qlret.mixed', 'sequential', n_trials),
            'default.mixed': self.run_single_device('default.mixed', 'default', n_trials),
        }
    
    def run_full_benchmark(self, n_trials: int = 3) -> Dict[str, Any]:
        return {
            'lret_modes': self.compare_lret_modes(n_trials),
            'device_comparison': self.compare_devices(n_trials),
        }


def run_qgan_benchmark(n_qubits: int = 3, n_trials: int = 3) -> Dict[str, Any]:
    return QGANBenchmark(n_qubits=n_qubits).run_full_benchmark(n_trials)


if __name__ == "__main__":
    results = run_qgan_benchmark(n_qubits=3, n_trials=2)
    print("qGAN benchmark complete!")

"""
Number Partitioning Benchmark
=============================

Algorithm #14 - Tier 2 (Should Test)
Purpose: Combinatorial optimization using QAOA

Given a set of numbers, partition into two subsets with equal sum.
This is NP-hard and maps naturally to Ising optimization.

Key metrics:
- Solution quality (partition difference)
- Approximation ratio
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


def generate_number_set(n_numbers: int, seed: int = 42) -> np.ndarray:
    """Generate a random set of positive integers."""
    np.random.seed(seed)
    return np.random.randint(1, 20, size=n_numbers).astype(float)


def partition_hamiltonian(numbers: np.ndarray) -> qml.Hamiltonian:
    """Create Hamiltonian for number partitioning.
    
    Objective: minimize (Σᵢ nᵢ * sᵢ)² where sᵢ ∈ {-1, +1}
    Maps to: minimize Σᵢⱼ nᵢ nⱼ Zᵢ Zⱼ
    """
    n = len(numbers)
    coeffs = []
    ops = []
    
    for i in range(n):
        for j in range(i, n):
            coef = numbers[i] * numbers[j]
            if i == j:
                coeffs.append(coef)
                ops.append(qml.Identity(0))
            else:
                coeffs.append(2 * coef)
                ops.append(qml.PauliZ(i) @ qml.PauliZ(j))
    
    return qml.Hamiltonian(coeffs, ops)


def qaoa_partition_circuit(
    params: np.ndarray,
    numbers: np.ndarray,
    depth: int = 2
):
    """QAOA circuit for number partitioning."""
    n = len(numbers)
    
    # Initial superposition
    for i in range(n):
        qml.Hadamard(wires=i)
    
    for layer in range(depth):
        gamma = params[layer * 2]
        beta = params[layer * 2 + 1]
        
        # Cost layer: exp(-i γ H_C)
        for i in range(n):
            for j in range(i + 1, n):
                angle = gamma * numbers[i] * numbers[j]
                qml.CNOT(wires=[i, j])
                qml.RZ(2 * angle, wires=j)
                qml.CNOT(wires=[i, j])
        
        # Mixer layer: exp(-i β H_M)
        for i in range(n):
            qml.RX(2 * beta, wires=i)


def evaluate_partition(bitstring: np.ndarray, numbers: np.ndarray) -> float:
    """Evaluate partition quality (difference between sums)."""
    # Convert 0/1 to -1/+1
    spins = 2 * bitstring - 1
    partition_diff = np.abs(np.sum(spins * numbers))
    return partition_diff


def find_optimal_partition(numbers: np.ndarray) -> float:
    """Find optimal partition by brute force (for small instances)."""
    n = len(numbers)
    best_diff = float('inf')
    
    for i in range(2 ** n):
        bitstring = np.array([(i >> j) & 1 for j in range(n)])
        diff = evaluate_partition(bitstring, numbers)
        best_diff = min(best_diff, diff)
    
    return best_diff


@dataclass
class NumberPartitioningBenchmark:
    """Number Partitioning benchmark."""
    
    n_numbers: int = 4
    depth: int = 2
    max_iterations: int = 30
    with_noise: bool = True
    noise_strength: float = 0.01
    
    def __post_init__(self):
        self.numbers = generate_number_set(self.n_numbers)
        self.hamiltonian = partition_hamiltonian(self.numbers)
        self.n_params = self.depth * 2
        self.n_qubits = self.n_numbers
        
        # Compute optimal for small instances
        if self.n_numbers <= 10:
            self.optimal_diff = find_optimal_partition(self.numbers)
        else:
            self.optimal_diff = None
        
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
                            qaoa_partition_circuit(params, self.numbers, self.depth)
                            return qml.expval(self.hamiltonian)
                        
                        params = pnp.random.uniform(0, np.pi, self.n_params, requires_grad=True)
                        opt = qml.GradientDescentOptimizer(stepsize=0.1)
                        
                        for _ in range(self.max_iterations):
                            params, cost = opt.step_and_cost(circuit, params)
                        
                        final_cost = float(cost)
                        
                        # Approximation ratio
                        if self.optimal_diff is not None and self.optimal_diff > 0:
                            approx_ratio = self.optimal_diff / max(1.0, np.sqrt(final_cost))
                        else:
                            approx_ratio = 1.0
                        
                        success = True
                        error_msg = ""
                        
                    except Exception as e:
                        final_cost = float('nan')
                        approx_ratio = 0.0
                        success = False
                        error_msg = str(e)
            
            results.append(BenchmarkResult(
                algorithm='NumberPartitioning',
                device_name=device_name,
                mode=mode,
                n_qubits=self.n_qubits,
                execution_time_seconds=timer.seconds,
                peak_memory_mb=mem.peak_mb,
                result_value=final_cost,
                secondary_value=approx_ratio,
                with_noise=self.with_noise,
                success=success,
                error_message=error_msg,
                extra_data={
                    'numbers': self.numbers.tolist(),
                    'optimal_diff': self.optimal_diff
                }
            ))
        
        self.results.extend(results)
        return results
    
    def compare_lret_modes(self, n_trials: int = 3) -> Dict[str, List[BenchmarkResult]]:
        print(f"\n{'='*60}\nNumber Partitioning LRET Mode Comparison\n{'='*60}")
        return {mode: self.run_single_device('qlret.mixed', mode, n_trials) 
                for mode in ['sequential', 'batched', 'parallel', 'openmp']}
    
    def compare_devices(self, n_trials: int = 3) -> Dict[str, List[BenchmarkResult]]:
        print(f"\n{'='*60}\nNumber Partitioning Device Comparison\n{'='*60}")
        return {
            'qlret.mixed': self.run_single_device('qlret.mixed', 'sequential', n_trials),
            'default.mixed': self.run_single_device('default.mixed', 'default', n_trials),
        }
    
    def run_full_benchmark(self, n_trials: int = 3) -> Dict[str, Any]:
        return {
            'lret_modes': self.compare_lret_modes(n_trials),
            'device_comparison': self.compare_devices(n_trials),
        }


def run_number_partitioning_benchmark(n_numbers: int = 4, n_trials: int = 3) -> Dict[str, Any]:
    return NumberPartitioningBenchmark(n_numbers=n_numbers).run_full_benchmark(n_trials)


if __name__ == "__main__":
    results = run_number_partitioning_benchmark(n_numbers=4, n_trials=2)
    print("Number Partitioning benchmark complete!")

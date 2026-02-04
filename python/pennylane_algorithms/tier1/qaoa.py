"""
QAOA (Quantum Approximate Optimization Algorithm) Benchmark
============================================================

Algorithm #2 - Tier 1 (Must Test)
Purpose: Solve combinatorial optimization (MaxCut)

Key metrics:
- Approximation ratio
- Solution quality
- Convergence

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


def generate_random_graph(n_nodes: int, edge_prob: float = 0.5, seed: int = 42) -> List[Tuple[int, int]]:
    """Generate random graph edges."""
    np.random.seed(seed)
    edges = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if np.random.random() < edge_prob:
                edges.append((i, j))
    return edges if edges else [(0, 1)]


def maxcut_hamiltonian(edges: List[Tuple[int, int]]) -> qml.Hamiltonian:
    """Create MaxCut cost Hamiltonian."""
    coeffs = []
    obs = []
    for i, j in edges:
        coeffs.append(0.5)
        obs.append(qml.Identity(i))
        coeffs.append(-0.5)
        obs.append(qml.PauliZ(i) @ qml.PauliZ(j))
    return qml.Hamiltonian(coeffs, obs)


def qaoa_circuit(params: np.ndarray, edges: List[Tuple[int, int]], n_qubits: int, depth: int):
    """QAOA circuit with cost and mixer layers."""
    for i in range(n_qubits):
        qml.Hadamard(wires=i)
    
    for layer in range(depth):
        gamma = params[layer * 2]
        beta = params[layer * 2 + 1]
        
        for i, j in edges:
            qml.CNOT(wires=[i, j])
            qml.RZ(gamma, wires=j)
            qml.CNOT(wires=[i, j])
        
        for i in range(n_qubits):
            qml.RX(2 * beta, wires=i)


@dataclass
class QAOABenchmark:
    """QAOA MaxCut benchmark."""
    
    n_qubits: int = 6
    depth: int = 2
    max_iterations: int = 50
    with_noise: bool = True
    
    def __post_init__(self):
        self.edges = generate_random_graph(self.n_qubits)
        self.hamiltonian = maxcut_hamiltonian(self.edges)
        self.n_params = 2 * self.depth
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
                        def circuit(params):
                            qaoa_circuit(params, self.edges, self.n_qubits, self.depth)
                            return qml.expval(self.hamiltonian)
                        
                        params = pnp.random.uniform(0, np.pi, self.n_params, requires_grad=True)
                        opt = qml.GradientDescentOptimizer(stepsize=0.1)
                        
                        for _ in range(self.max_iterations):
                            params, cost = opt.step_and_cost(circuit, params)
                        
                        cost = float(cost)
                        approx_ratio = cost / len(self.edges)
                        success = True
                        error_msg = ""
                    except Exception as e:
                        cost = float('nan')
                        approx_ratio = float('nan')
                        success = False
                        error_msg = str(e)
            
            results.append(BenchmarkResult(
                algorithm='QAOA-MaxCut',
                device_name=device_name,
                mode=mode,
                n_qubits=self.n_qubits,
                execution_time_seconds=timer.seconds,
                peak_memory_mb=mem.peak_mb,
                result_value=cost,
                secondary_value=approx_ratio,
                with_noise=self.with_noise,
                success=success,
                error_message=error_msg,
                extra_data={'depth': self.depth, 'n_edges': len(self.edges)}
            ))
        
        self.results.extend(results)
        return results
    
    def compare_lret_modes(self, n_trials: int = 3) -> Dict[str, List[BenchmarkResult]]:
        print(f"\n{'='*60}\nQAOA LRET Mode Comparison\n{'='*60}")
        return {mode: self.run_single_device('qlret.mixed', mode, n_trials) 
                for mode in ['sequential', 'batched', 'parallel', 'openmp']}
    
    def compare_devices(self, n_trials: int = 3) -> Dict[str, List[BenchmarkResult]]:
        print(f"\n{'='*60}\nQAOA Device Comparison\n{'='*60}")
        return {
            'qlret.mixed': self.run_single_device('qlret.mixed', 'sequential', n_trials),
            'default.mixed': self.run_single_device('default.mixed', 'default', n_trials),
        }
    
    def run_full_benchmark(self, n_trials: int = 3) -> Dict[str, Any]:
        return {
            'lret_modes': self.compare_lret_modes(n_trials),
            'device_comparison': self.compare_devices(n_trials),
        }


def run_qaoa_benchmark(n_qubits: int = 6, n_trials: int = 3) -> Dict[str, Any]:
    return QAOABenchmark(n_qubits=n_qubits).run_full_benchmark(n_trials)


if __name__ == "__main__":
    results = run_qaoa_benchmark(n_qubits=6, n_trials=2)
    print("QAOA benchmark complete!")

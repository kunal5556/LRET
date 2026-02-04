"""
Portfolio Optimization Benchmark
================================

Algorithm #9 - Tier 2 (Should Test)
Purpose: Finance application - optimize asset allocation

Uses QAOA-like approach for quadratic optimization.

Key metrics:
- Portfolio return
- Risk minimization
- Constraint satisfaction

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
from utils.parallel_modes import get_parallel_modes, ParallelExecutor


def generate_portfolio_problem(n_assets: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generate random portfolio optimization problem.
    
    Returns:
        expected_returns: Expected return for each asset
        covariance_matrix: Covariance matrix of returns
    """
    np.random.seed(seed)
    expected_returns = np.random.uniform(0.05, 0.15, n_assets)
    
    # Generate positive semi-definite covariance matrix
    A = np.random.randn(n_assets, n_assets) * 0.1
    covariance_matrix = A @ A.T + np.eye(n_assets) * 0.01
    
    return expected_returns, covariance_matrix


def portfolio_hamiltonian(
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
    risk_aversion: float = 0.5,
    budget_penalty: float = 2.0
) -> qml.Hamiltonian:
    """Create portfolio optimization Hamiltonian.
    
    Objective: max return - risk_aversion * risk - budget_penalty * (sum - 1)^2
    """
    n_assets = len(expected_returns)
    coeffs = []
    ops = []
    
    # Return terms (linear in Z)
    for i, ret in enumerate(expected_returns):
        coeffs.append(-ret / 2)  # Negative for maximization
        ops.append(qml.PauliZ(i))
    
    # Risk terms (quadratic in Z)
    for i in range(n_assets):
        for j in range(i, n_assets):
            coef = risk_aversion * covariance_matrix[i, j] / 4
            if i == j:
                coeffs.append(coef)
                ops.append(qml.Identity(0))
            else:
                coeffs.append(coef)
                ops.append(qml.PauliZ(i) @ qml.PauliZ(j))
    
    # Budget constraint penalty
    for i in range(n_assets):
        coeffs.append(budget_penalty / 4)
        ops.append(qml.Identity(0))
        for j in range(i + 1, n_assets):
            coeffs.append(budget_penalty / 2)
            ops.append(qml.PauliZ(i) @ qml.PauliZ(j))
    
    return qml.Hamiltonian(coeffs, ops)


def portfolio_qaoa_circuit(params: np.ndarray, n_assets: int, depth: int = 2):
    """Portfolio optimization QAOA circuit."""
    # Initial superposition
    for i in range(n_assets):
        qml.Hadamard(wires=i)
    
    for layer in range(depth):
        # Cost layer (simplified)
        for i in range(n_assets):
            qml.RZ(params[layer * 2], wires=i)
        for i in range(n_assets - 1):
            qml.CNOT(wires=[i, i + 1])
            qml.RZ(params[layer * 2], wires=i + 1)
            qml.CNOT(wires=[i, i + 1])
        
        # Mixer layer
        for i in range(n_assets):
            qml.RX(params[layer * 2 + 1], wires=i)


def evaluate_portfolio(bitstring: np.ndarray, returns: np.ndarray, 
                       covariance: np.ndarray, risk_aversion: float = 0.5) -> float:
    """Evaluate portfolio value from bitstring."""
    portfolio_return = np.sum(bitstring * returns)
    portfolio_risk = bitstring @ covariance @ bitstring
    return portfolio_return - risk_aversion * portfolio_risk


@dataclass
class PortfolioBenchmark:
    """Portfolio Optimization benchmark."""
    
    n_assets: int = 6
    depth: int = 2
    max_iterations: int = 30
    risk_aversion: float = 0.5
    with_noise: bool = True
    noise_strength: float = 0.01
    
    def __post_init__(self):
        self.returns, self.covariance = generate_portfolio_problem(self.n_assets)
        self.hamiltonian = portfolio_hamiltonian(
            self.returns, self.covariance, self.risk_aversion
        )
        self.n_params = self.depth * 2
        self.n_qubits = self.n_assets
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
                            portfolio_qaoa_circuit(params, self.n_assets, self.depth)
                            return qml.expval(self.hamiltonian)
                        
                        params = pnp.random.uniform(0, np.pi, self.n_params, requires_grad=True)
                        opt = qml.GradientDescentOptimizer(stepsize=0.1)
                        
                        for _ in range(self.max_iterations):
                            params, cost = opt.step_and_cost(circuit, params)
                        
                        cost = float(cost)
                        success = True
                        error_msg = ""
                        
                    except Exception as e:
                        cost = float('nan')
                        success = False
                        error_msg = str(e)
            
            results.append(BenchmarkResult(
                algorithm='Portfolio-Optimization',
                device_name=device_name,
                mode=mode,
                n_qubits=self.n_qubits,
                execution_time_seconds=timer.seconds,
                peak_memory_mb=mem.peak_mb,
                result_value=cost,
                secondary_value=0.0,
                with_noise=self.with_noise,
                success=success,
                error_message=error_msg,
                extra_data={
                    'n_assets': self.n_assets,
                    'risk_aversion': self.risk_aversion
                }
            ))
        
        self.results.extend(results)
        return results
    
    def compare_lret_modes(self, n_trials: int = 3) -> Dict[str, List[BenchmarkResult]]:
        print(f"\n{'='*60}\nPortfolio Optimization LRET Mode Comparison\n{'='*60}")
        return {mode: self.run_single_device('qlret.mixed', mode, n_trials) 
                for mode in ['sequential', 'batched', 'parallel', 'openmp']}
    
    def compare_devices(self, n_trials: int = 3) -> Dict[str, List[BenchmarkResult]]:
        print(f"\n{'='*60}\nPortfolio Optimization Device Comparison\n{'='*60}")
        return {
            'qlret.mixed': self.run_single_device('qlret.mixed', 'sequential', n_trials),
            'default.mixed': self.run_single_device('default.mixed', 'default', n_trials),
        }
    
    def run_full_benchmark(self, n_trials: int = 3) -> Dict[str, Any]:
        return {
            'lret_modes': self.compare_lret_modes(n_trials),
            'device_comparison': self.compare_devices(n_trials),
        }


def run_portfolio_benchmark(n_assets: int = 6, n_trials: int = 3) -> Dict[str, Any]:
    return PortfolioBenchmark(n_assets=n_assets).run_full_benchmark(n_trials)


if __name__ == "__main__":
    results = run_portfolio_benchmark(n_assets=6, n_trials=2)
    print("Portfolio Optimization benchmark complete!")

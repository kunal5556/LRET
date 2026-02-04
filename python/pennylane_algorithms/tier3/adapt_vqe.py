"""
ADAPT-VQE Benchmark
===================

Algorithm #20 - Tier 3 (Optional)
Purpose: Adaptive ansatz construction for chemistry

ADAPT-VQE iteratively grows the ansatz by selecting
operators from a pool that have the largest gradient.

Key metrics:
- Chemical accuracy
- Operator count
- Circuit depth efficiency

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


def get_h2_hamiltonian() -> Tuple[qml.Hamiltonian, float]:
    """H2 molecule Hamiltonian."""
    coeffs = [-0.81261, 0.17120, 0.17120, -0.22343, 0.16862]
    ops = [
        qml.Identity(0),
        qml.PauliZ(0),
        qml.PauliZ(1),
        qml.PauliZ(0) @ qml.PauliZ(1),
        qml.PauliX(0) @ qml.PauliX(1)
    ]
    return qml.Hamiltonian(coeffs, ops), -1.1373


def get_operator_pool(n_qubits: int) -> List[Tuple[str, callable]]:
    """Get operator pool for ADAPT-VQE."""
    pool = []
    
    # Single excitations: a†_p a_q
    for p in range(n_qubits // 2, n_qubits):
        for q in range(n_qubits // 2):
            def single_exc(angle, p=p, q=q):
                qml.SingleExcitation(angle, wires=[q, p])
            pool.append((f"S_{q}->{p}", single_exc))
    
    # Double excitations
    for p in range(n_qubits // 2, n_qubits):
        for q in range(p + 1, n_qubits):
            for r in range(n_qubits // 2):
                for s in range(r + 1, n_qubits // 2):
                    def double_exc(angle, p=p, q=q, r=r, s=s):
                        qml.DoubleExcitation(angle, wires=[r, s, p, q])
                    pool.append((f"D_{r}{s}->{p}{q}", double_exc))
    
    # If pool is empty (small system), add simple rotations
    if not pool:
        for i in range(n_qubits):
            def ry_op(angle, i=i):
                qml.RY(angle, wires=i)
            pool.append((f"RY_{i}", ry_op))
            
            def rz_op(angle, i=i):
                qml.RZ(angle, wires=i)
            pool.append((f"RZ_{i}", rz_op))
    
    return pool


def hartree_fock_state(n_qubits: int, n_electrons: int = 2):
    """Prepare Hartree-Fock initial state."""
    for i in range(min(n_electrons, n_qubits)):
        qml.PauliX(wires=i)


def compute_gradient_for_operator(
    dev,
    hamiltonian: qml.Hamiltonian,
    current_ops: List[Tuple[str, callable, float]],
    candidate_op: Tuple[str, callable],
    n_qubits: int,
    n_electrons: int
) -> float:
    """Compute gradient of energy w.r.t. new operator parameter."""
    
    @qml.qnode(dev, diff_method='parameter-shift')
    def circuit(new_angle):
        hartree_fock_state(n_qubits, n_electrons)
        
        # Apply current operators
        for name, op_fn, angle in current_ops:
            op_fn(angle)
        
        # Apply candidate operator
        candidate_op[1](new_angle)
        
        return qml.expval(hamiltonian)
    
    # Gradient at angle = 0
    grad = qml.grad(circuit)(pnp.array(0.0))
    return abs(float(grad))


@dataclass
class ADAPTVQEBenchmark:
    """ADAPT-VQE benchmark."""
    
    n_qubits: int = 4
    n_electrons: int = 2
    max_operators: int = 5
    gradient_threshold: float = 1e-3
    max_iterations: int = 20
    with_noise: bool = True
    noise_strength: float = 0.01
    
    def __post_init__(self):
        self.hamiltonian, self.exact_energy = get_h2_hamiltonian()
        self.operator_pool = get_operator_pool(self.n_qubits)
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
                        
                        # ADAPT-VQE loop
                        selected_ops: List[Tuple[str, callable, float]] = []
                        
                        for adapt_iter in range(self.max_operators):
                            # Find operator with largest gradient
                            best_grad = 0.0
                            best_op = None
                            
                            for candidate in self.operator_pool:
                                grad = compute_gradient_for_operator(
                                    dev, self.hamiltonian,
                                    selected_ops, candidate,
                                    self.n_qubits, self.n_electrons
                                )
                                if grad > best_grad:
                                    best_grad = grad
                                    best_op = candidate
                            
                            if best_grad < self.gradient_threshold or best_op is None:
                                break
                            
                            # Add operator and optimize
                            selected_ops.append((best_op[0], best_op[1], 0.0))
                            
                            # Optimize all parameters
                            def cost_fn(angles):
                                @qml.qnode(dev)
                                def circuit():
                                    hartree_fock_state(self.n_qubits, self.n_electrons)
                                    for idx, (name, op_fn, _) in enumerate(selected_ops):
                                        op_fn(angles[idx])
                                    return qml.expval(self.hamiltonian)
                                return circuit()
                            
                            angles = pnp.array([op[2] for op in selected_ops], requires_grad=True)
                            opt = qml.GradientDescentOptimizer(stepsize=0.1)
                            
                            for _ in range(self.max_iterations):
                                angles = opt.step(cost_fn, angles)
                            
                            # Update stored angles
                            selected_ops = [
                                (name, op_fn, float(angles[idx]))
                                for idx, (name, op_fn, _) in enumerate(selected_ops)
                            ]
                        
                        # Final energy
                        final_energy = cost_fn(pnp.array([op[2] for op in selected_ops]))
                        energy_error = abs(float(final_energy) - self.exact_energy)
                        n_ops = len(selected_ops)
                        
                        success = True
                        error_msg = ""
                        
                    except Exception as e:
                        final_energy = float('nan')
                        energy_error = float('nan')
                        n_ops = 0
                        success = False
                        error_msg = str(e)
            
            results.append(BenchmarkResult(
                algorithm='ADAPT-VQE',
                device_name=device_name,
                mode=mode,
                n_qubits=self.n_qubits,
                execution_time_seconds=timer.seconds,
                peak_memory_mb=mem.peak_mb,
                result_value=float(final_energy) if success else float('nan'),
                secondary_value=energy_error,
                with_noise=self.with_noise,
                success=success,
                error_message=error_msg,
                extra_data={
                    'n_operators': n_ops,
                    'exact_energy': self.exact_energy
                }
            ))
        
        self.results.extend(results)
        return results
    
    def compare_lret_modes(self, n_trials: int = 3) -> Dict[str, List[BenchmarkResult]]:
        print(f"\n{'='*60}\nADAPT-VQE LRET Mode Comparison\n{'='*60}")
        return {mode: self.run_single_device('qlret.mixed', mode, n_trials) 
                for mode in ['sequential', 'batched', 'parallel', 'openmp']}
    
    def compare_devices(self, n_trials: int = 3) -> Dict[str, List[BenchmarkResult]]:
        print(f"\n{'='*60}\nADAPT-VQE Device Comparison\n{'='*60}")
        return {
            'qlret.mixed': self.run_single_device('qlret.mixed', 'sequential', n_trials),
            'default.mixed': self.run_single_device('default.mixed', 'default', n_trials),
        }
    
    def run_full_benchmark(self, n_trials: int = 3) -> Dict[str, Any]:
        return {
            'lret_modes': self.compare_lret_modes(n_trials),
            'device_comparison': self.compare_devices(n_trials),
        }


def run_adapt_vqe_benchmark(n_qubits: int = 4, n_trials: int = 3) -> Dict[str, Any]:
    return ADAPTVQEBenchmark(n_qubits=n_qubits).run_full_benchmark(n_trials)


if __name__ == "__main__":
    results = run_adapt_vqe_benchmark(n_qubits=4, n_trials=2)
    print("ADAPT-VQE benchmark complete!")

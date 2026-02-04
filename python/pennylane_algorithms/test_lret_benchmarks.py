"""
LRET Algorithm Benchmark Runner
================================

This script runs all algorithms and documents:
1. Which algorithms work with LRET device
2. Which need modifications (Hamiltonian decomposition)
3. Speedup comparisons between modes
4. Error documentation

LRET Device Limitations:
- Supports: PauliX, PauliY, PauliZ, Identity, Hermitian observables
- Does NOT support: Hamiltonian (qml.Hamiltonian) directly
- Workaround: Decompose Hamiltonian into individual Pauli terms
"""

import sys
import os
import time
import json
import numpy as np
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Tuple

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

import pennylane as qml
from pennylane import numpy as pnp


@dataclass
class TestResult:
    """Result from a single test."""
    algorithm: str
    device: str
    mode: str
    n_qubits: int
    time_seconds: float
    success: bool
    error: str
    result_value: float


def test_lret_device_basic(n_qubits: int = 4) -> Tuple[bool, str, float]:
    """Test basic LRET device functionality."""
    try:
        dev = qml.device('qlret.mixed', wires=n_qubits)
        
        @qml.qnode(dev)
        def circuit(params):
            for i in range(n_qubits):
                qml.RY(params[i], wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            return qml.expval(qml.PauliZ(0))
        
        params = np.random.randn(n_qubits)
        start = time.time()
        result = circuit(params)
        elapsed = time.time() - start
        
        return True, "", elapsed
    except Exception as e:
        return False, str(e), 0.0


def test_lret_modes(n_qubits: int = 4, n_iterations: int = 10) -> Dict[str, Dict]:
    """Test all LRET modes and measure performance."""
    results = {}
    
    modes = {
        'sequential': {},
        'batched': {'batch_size': 5},
        'openmp': {'use_openmp': True},
    }
    
    for mode_name, mode_kwargs in modes.items():
        try:
            dev = qml.device('qlret.mixed', wires=n_qubits, **mode_kwargs)
            
            @qml.qnode(dev)
            def circuit(params):
                for i in range(n_qubits):
                    qml.RY(params[i], wires=i)
                    qml.RZ(params[i + n_qubits], wires=i)
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                return qml.expval(qml.PauliZ(0))
            
            params = np.random.randn(2 * n_qubits)
            
            # Warmup
            _ = circuit(params)
            
            # Time it
            start = time.time()
            for _ in range(n_iterations):
                _ = circuit(params)
            elapsed = time.time() - start
            
            results[mode_name] = {
                'success': True,
                'total_time': elapsed,
                'avg_time': elapsed / n_iterations,
                'error': ''
            }
        except Exception as e:
            results[mode_name] = {
                'success': False,
                'total_time': 0,
                'avg_time': 0,
                'error': str(e)
            }
    
    return results


def test_algorithm_compatibility() -> Dict[str, Dict]:
    """Test which algorithms work with LRET device."""
    
    algorithms = {
        'VQE': test_vqe_compatibility,
        'QFT': test_qft_compatibility,
        'QPE': test_qpe_compatibility,
        'Grover': test_grover_compatibility,
        'QNN': test_qnn_compatibility,
        'QAOA': test_qaoa_compatibility,
    }
    
    results = {}
    for name, test_fn in algorithms.items():
        print(f"Testing {name}...", end=" ")
        try:
            success, error, time_s, notes = test_fn()
            results[name] = {
                'compatible': success,
                'error': error,
                'time_seconds': time_s,
                'notes': notes
            }
            status = "✓" if success else "✗"
            print(f"{status}")
        except Exception as e:
            results[name] = {
                'compatible': False,
                'error': str(e),
                'time_seconds': 0,
                'notes': 'Exception during test'
            }
            print(f"✗ (exception)")
    
    return results


def test_vqe_compatibility() -> Tuple[bool, str, float, str]:
    """Test VQE - needs Hamiltonian decomposition."""
    n_qubits = 2
    
    # VQE requires Hamiltonian which LRET doesn't support directly
    # We need to decompose into individual Pauli expectations
    
    try:
        dev = qml.device('qlret.mixed', wires=n_qubits)
        
        # H2 Hamiltonian coefficients
        coeffs = [-0.81261, 0.17120, 0.17120, -0.22343, 0.16862, 0.04532]
        
        @qml.qnode(dev)
        def measure_identity():
            return qml.expval(qml.Identity(0))
        
        @qml.qnode(dev)
        def measure_z0():
            return qml.expval(qml.PauliZ(0))
        
        @qml.qnode(dev)
        def measure_z1():
            return qml.expval(qml.PauliZ(1))
        
        @qml.qnode(dev)
        def ansatz_z0z1(params):
            qml.RY(params[0], wires=0)
            qml.RY(params[1], wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
        
        params = np.array([0.5, 0.5])
        start = time.time()
        
        # This will fail because tensor products aren't supported
        try:
            result = ansatz_z0z1(params)
            elapsed = time.time() - start
            return True, "", elapsed, "VQE works with decomposed Hamiltonians"
        except:
            # Individual measurements work
            _ = measure_identity()
            _ = measure_z0()
            _ = measure_z1()
            elapsed = time.time() - start
            return False, "Tensor product observables not supported", elapsed, "Need to decompose to single-qubit measurements"
            
    except Exception as e:
        return False, str(e), 0, ""


def test_qft_compatibility() -> Tuple[bool, str, float, str]:
    """Test QFT - should work with probs measurement."""
    n_qubits = 4
    
    try:
        dev = qml.device('qlret.mixed', wires=n_qubits)
        
        @qml.qnode(dev)
        def qft_circuit():
            qml.PauliX(wires=0)
            # QFT
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
                for j in range(i + 1, n_qubits):
                    qml.ctrl(qml.PhaseShift, control=j)(np.pi / (2 ** (j - i)), wires=i)
            return qml.probs(wires=range(n_qubits))
        
        start = time.time()
        probs = qft_circuit()
        elapsed = time.time() - start
        
        return True, "", elapsed, "QFT works with probability measurements"
    except Exception as e:
        return False, str(e), 0, ""


def test_qpe_compatibility() -> Tuple[bool, str, float, str]:
    """Test QPE - should work with probs measurement."""
    n_counting = 3
    n_total = n_counting + 1
    
    try:
        dev = qml.device('qlret.mixed', wires=n_total)
        
        @qml.qnode(dev)
        def qpe_circuit():
            # Eigenstate
            qml.PauliX(wires=n_counting)
            # Hadamards on counting qubits
            for i in range(n_counting):
                qml.Hadamard(wires=i)
            # Controlled phase gates
            for i in range(n_counting):
                qml.ctrl(qml.PhaseShift, control=i)(np.pi / 2, wires=n_counting)
            return qml.probs(wires=range(n_counting))
        
        start = time.time()
        probs = qpe_circuit()
        elapsed = time.time() - start
        
        return True, "", elapsed, "QPE works with probability measurements"
    except Exception as e:
        return False, str(e), 0, ""


def test_grover_compatibility() -> Tuple[bool, str, float, str]:
    """Test Grover's algorithm."""
    n_qubits = 3
    
    try:
        dev = qml.device('qlret.mixed', wires=n_qubits)
        
        @qml.qnode(dev)
        def grover_circuit():
            # Initial superposition
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
            
            # Oracle for target |101⟩
            qml.PauliX(wires=1)
            qml.ctrl(qml.PauliZ, control=[0, 1])(wires=2)
            qml.PauliX(wires=1)
            
            # Diffusion
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
                qml.PauliX(wires=i)
            qml.ctrl(qml.PauliZ, control=[0, 1])(wires=2)
            for i in range(n_qubits):
                qml.PauliX(wires=i)
                qml.Hadamard(wires=i)
            
            return qml.probs(wires=range(n_qubits))
        
        start = time.time()
        probs = grover_circuit()
        elapsed = time.time() - start
        
        return True, "", elapsed, "Grover works with probability measurements"
    except Exception as e:
        return False, str(e), 0, ""


def test_qnn_compatibility() -> Tuple[bool, str, float, str]:
    """Test QNN - should work with single Pauli measurement."""
    n_qubits = 4
    
    try:
        dev = qml.device('qlret.mixed', wires=n_qubits)
        
        @qml.qnode(dev, interface='autograd')
        def qnn(x, params):
            # Data encoding
            for i in range(n_qubits):
                qml.RY(x[i % len(x)] * np.pi, wires=i)
            # Variational layer
            for i in range(n_qubits):
                qml.RY(params[i], wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            return qml.expval(qml.PauliZ(0))
        
        x = np.random.randn(n_qubits)
        params = pnp.random.randn(n_qubits, requires_grad=True)
        
        start = time.time()
        result = qnn(x, params)
        elapsed = time.time() - start
        
        return True, "", elapsed, "QNN works with single PauliZ measurement"
    except Exception as e:
        return False, str(e), 0, ""


def test_qaoa_compatibility() -> Tuple[bool, str, float, str]:
    """Test QAOA - needs Hamiltonian decomposition."""
    n_qubits = 4
    
    try:
        dev = qml.device('qlret.mixed', wires=n_qubits)
        
        # QAOA MaxCut - test if we can measure individual ZZ terms
        @qml.qnode(dev)
        def qaoa_circuit_decomposed(gamma, beta):
            # Initial superposition
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
            
            # Cost layer (ZZ interactions)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
                qml.RZ(gamma, wires=i + 1)
                qml.CNOT(wires=[i, i + 1])
            
            # Mixer layer
            for i in range(n_qubits):
                qml.RX(2 * beta, wires=i)
            
            # Measure ZZ on edge (0,1)
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
        
        start = time.time()
        try:
            result = qaoa_circuit_decomposed(0.5, 0.5)
            elapsed = time.time() - start
            return True, "", elapsed, "QAOA works with decomposed ZZ measurements"
        except:
            # Try with single qubit
            @qml.qnode(dev)
            def qaoa_single():
                for i in range(n_qubits):
                    qml.Hadamard(wires=i)
                return qml.expval(qml.PauliZ(0))
            
            result = qaoa_single()
            elapsed = time.time() - start
            return False, "Tensor products not supported", elapsed, "Need decomposition to single qubit"
            
    except Exception as e:
        return False, str(e), 0, ""


def compare_lret_vs_default(n_qubits_list: List[int] = [4, 6, 8]) -> Dict[str, Any]:
    """Compare LRET vs default.mixed performance."""
    results = {}
    
    for n_qubits in n_qubits_list:
        print(f"\nTesting {n_qubits} qubits...")
        
        try:
            dev_lret = qml.device('qlret.mixed', wires=n_qubits)
            dev_default = qml.device('default.mixed', wires=n_qubits)
            
            @qml.qnode(dev_lret)
            def circuit_lret(params):
                for i in range(n_qubits):
                    qml.RY(params[i], wires=i)
                    qml.RZ(params[i + n_qubits], wires=i)
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                for i in range(n_qubits):
                    qml.RY(params[i + 2*n_qubits], wires=i)
                return qml.probs(wires=range(n_qubits))
            
            @qml.qnode(dev_default)
            def circuit_default(params):
                for i in range(n_qubits):
                    qml.RY(params[i], wires=i)
                    qml.RZ(params[i + n_qubits], wires=i)
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                for i in range(n_qubits):
                    qml.RY(params[i + 2*n_qubits], wires=i)
                return qml.probs(wires=range(n_qubits))
            
            params = np.random.randn(3 * n_qubits)
            
            # Warmup
            _ = circuit_lret(params)
            _ = circuit_default(params)
            
            n_iterations = 20
            
            # Time LRET
            start = time.time()
            for _ in range(n_iterations):
                _ = circuit_lret(params)
            lret_time = time.time() - start
            
            # Time default
            start = time.time()
            for _ in range(n_iterations):
                _ = circuit_default(params)
            default_time = time.time() - start
            
            speedup = default_time / lret_time if lret_time > 0 else 0
            
            results[n_qubits] = {
                'lret_time': lret_time / n_iterations,
                'default_time': default_time / n_iterations,
                'speedup': speedup,
                'success': True
            }
            
            print(f"  LRET: {lret_time/n_iterations:.4f}s, default: {default_time/n_iterations:.4f}s, speedup: {speedup:.2f}x")
            
        except Exception as e:
            results[n_qubits] = {
                'error': str(e),
                'success': False
            }
            print(f"  Error: {e}")
    
    return results


def main():
    """Run all benchmark tests."""
    print("=" * 70)
    print("LRET PennyLane Device Benchmark")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    
    all_results = {}
    
    # Test 1: Basic device functionality
    print("\n1. Testing basic LRET device functionality...")
    success, error, time_s = test_lret_device_basic(4)
    print(f"   Status: {'✓ Working' if success else '✗ Failed'}")
    if error:
        print(f"   Error: {error}")
    else:
        print(f"   Time: {time_s:.4f}s")
    all_results['basic_test'] = {'success': success, 'error': error, 'time': time_s}
    
    # Test 2: LRET modes
    print("\n2. Testing LRET execution modes...")
    mode_results = test_lret_modes(n_qubits=4, n_iterations=20)
    for mode, data in mode_results.items():
        status = "✓" if data['success'] else "✗"
        print(f"   {mode}: {status} (avg: {data['avg_time']*1000:.2f}ms)")
        if data['error']:
            print(f"      Error: {data['error']}")
    all_results['mode_tests'] = mode_results
    
    # Test 3: Algorithm compatibility
    print("\n3. Testing algorithm compatibility with LRET...")
    compat_results = test_algorithm_compatibility()
    all_results['algorithm_compatibility'] = compat_results
    
    # Test 4: Performance comparison
    print("\n4. Performance comparison: LRET vs default.mixed...")
    perf_results = compare_lret_vs_default([4, 6, 8])
    all_results['performance_comparison'] = perf_results
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print("\nAlgorithm Compatibility:")
    for algo, data in compat_results.items():
        status = "✓" if data['compatible'] else "✗"
        print(f"  {algo}: {status}")
        if data['notes']:
            print(f"    Notes: {data['notes']}")
        if data['error'] and not data['compatible']:
            print(f"    Error: {data['error'][:60]}...")
    
    print("\nLRET Mode Performance (4 qubits, 20 iterations):")
    if 'mode_tests' in all_results:
        baseline = mode_results.get('sequential', {}).get('avg_time', 1)
        for mode, data in mode_results.items():
            if data['success']:
                speedup = baseline / data['avg_time'] if data['avg_time'] > 0 else 0
                print(f"  {mode}: {data['avg_time']*1000:.2f}ms (speedup: {speedup:.2f}x vs sequential)")
    
    print("\nLRET vs default.mixed Speedup:")
    if 'performance_comparison' in all_results:
        for n_qubits, data in perf_results.items():
            if data.get('success'):
                print(f"  {n_qubits} qubits: {data['speedup']:.2f}x speedup")
    
    # Save results
    output_file = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_file}")
    
    return all_results


if __name__ == "__main__":
    results = main()

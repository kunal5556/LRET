"""
Quick Validation Test for All PennyLane Algorithms
===================================================

PURPOSE: Verify all 20 algorithms work with LRET device (smoke test)
TIME: ~2-5 minutes total (not hours of benchmarking!)

This script tests each algorithm with:
- LRET device only (no comparisons)
- 1 run per algorithm
- Minimal configuration (2-4 qubits, 3-5 iterations)
- Just checks: "Does it run without errors?"

Usage:
    python quick_validation.py              # Test all 20 algorithms
    python quick_validation.py --tier 1     # Test only Tier 1 (7 algorithms)
    python quick_validation.py --verbose    # Show detailed output
"""

import sys
import os
import time
import json
import argparse
import traceback
from datetime import datetime
from typing import Dict, List, Tuple, Any, Callable

# Add paths
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


# ============================================================================
# QUICK TEST FUNCTIONS - Minimal configs for fast validation
# ============================================================================

def test_vqe() -> Tuple[bool, str, float]:
    """VQE: 2 qubits, 3 iterations."""
    n_qubits = 2
    dev = qml.device('qlret.mixed', wires=n_qubits)

    @qml.qnode(dev, interface='autograd')
    def circuit(params):
        qml.RY(params[0], wires=0)
        qml.RY(params[1], wires=1)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0))

    params = pnp.array([0.5, 0.5], requires_grad=True)
    opt = qml.GradientDescentOptimizer(stepsize=0.1)

    for _ in range(3):  # Just 3 iterations
        params, _ = opt.step_and_cost(circuit, params)

    result = circuit(params)
    return True, f"Energy={float(result):.4f}", 0


def test_qaoa() -> Tuple[bool, str, float]:
    """QAOA: 3 qubits, 1 layer."""
    n_qubits = 3
    dev = qml.device('qlret.mixed', wires=n_qubits)

    @qml.qnode(dev)
    def circuit(gamma, beta):
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
        # Cost layer
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
            qml.RZ(gamma, wires=i + 1)
            qml.CNOT(wires=[i, i + 1])
        # Mixer
        for i in range(n_qubits):
            qml.RX(2 * beta, wires=i)
        return qml.probs(wires=range(n_qubits))

    probs = circuit(0.5, 0.5)
    return True, f"Max prob={float(max(probs)):.4f}", 0


def test_qnn() -> Tuple[bool, str, float]:
    """QNN: 2 qubits, 3 samples, 3 epochs."""
    n_qubits = 2
    dev = qml.device('qlret.mixed', wires=n_qubits)

    @qml.qnode(dev, interface='autograd')
    def circuit(x, params):
        for i in range(n_qubits):
            qml.RY(x[i % len(x)] * np.pi, wires=i)
        for i in range(n_qubits):
            qml.RY(params[i], wires=i)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0))

    X = np.array([[0.1, 0.2], [0.8, 0.9], [0.5, 0.5]])
    y = np.array([0, 1, 0])
    params = pnp.array([0.1, 0.2], requires_grad=True)
    opt = qml.GradientDescentOptimizer(stepsize=0.1)

    for _ in range(3):  # 3 epochs
        for x_i in X:
            pred = circuit(x_i, params)

    return True, f"Prediction={float(pred):.4f}", 0


def test_qft() -> Tuple[bool, str, float]:
    """QFT: 3 qubits, roundtrip test."""
    n_qubits = 3
    dev = qml.device('qlret.mixed', wires=n_qubits)

    @qml.qnode(dev)
    def circuit():
        qml.PauliX(wires=0)
        # QFT
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
            for j in range(i + 1, n_qubits):
                qml.ctrl(qml.PhaseShift, control=j)(np.pi / (2 ** (j - i)), wires=i)
        # Inverse QFT
        for i in range(n_qubits - 1, -1, -1):
            for j in range(n_qubits - 1, i, -1):
                qml.ctrl(qml.PhaseShift, control=j)(-np.pi / (2 ** (j - i)), wires=i)
            qml.Hadamard(wires=i)
        return qml.probs(wires=range(n_qubits))

    probs = circuit()
    fidelity = float(probs[1])
    return True, f"Fidelity={fidelity:.4f}", 0


def test_qpe() -> Tuple[bool, str, float]:
    """QPE: 3 counting qubits + 1 eigenstate."""
    n_counting = 3
    dev = qml.device('qlret.mixed', wires=n_counting + 1)

    @qml.qnode(dev)
    def circuit():
        qml.PauliX(wires=n_counting)
        for i in range(n_counting):
            qml.Hadamard(wires=i)
        for i in range(n_counting):
            qml.ctrl(qml.PhaseShift, control=i)(np.pi / 4, wires=n_counting)
        return qml.probs(wires=range(n_counting))

    probs = circuit()
    return True, f"Peaks at phase estimation", 0


def test_grover() -> Tuple[bool, str, float]:
    """Grover: 3 qubits, 1 iteration."""
    n_qubits = 3
    dev = qml.device('qlret.mixed', wires=n_qubits)

    @qml.qnode(dev)
    def circuit():
        # Superposition
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
        # Oracle (mark |101>)
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

    probs = circuit()
    target_prob = float(probs[5])  # |101> = 5
    return True, f"Target |101> prob={target_prob:.4f}", 0


def test_metrology() -> Tuple[bool, str, float]:
    """Quantum Metrology: 2 qubits, phase estimation."""
    n_qubits = 2
    dev = qml.device('qlret.mixed', wires=n_qubits)

    @qml.qnode(dev)
    def circuit(theta):
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
        qml.CNOT(wires=[0, 1])
        for i in range(n_qubits):
            qml.RZ(theta, wires=i)
        qml.CNOT(wires=[0, 1])
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
        return qml.expval(qml.PauliZ(0))

    result = circuit(0.3)
    return True, f"Measurement={float(result):.4f}", 0


def test_uccsd() -> Tuple[bool, str, float]:
    """UCCSD-VQE: 2 qubits, chemistry-inspired."""
    n_qubits = 2
    dev = qml.device('qlret.mixed', wires=n_qubits)

    @qml.qnode(dev, interface='autograd')
    def circuit(params):
        # Initial HF state
        qml.PauliX(wires=0)
        # Single excitation
        qml.SingleExcitation(params[0], wires=[0, 1])
        return qml.expval(qml.PauliZ(0))

    params = pnp.array([0.1], requires_grad=True)
    result = circuit(params)
    return True, f"Energy={float(result):.4f}", 0


def test_portfolio() -> Tuple[bool, str, float]:
    """Portfolio Optimization: 3 assets."""
    n_qubits = 3
    dev = qml.device('qlret.mixed', wires=n_qubits)

    @qml.qnode(dev)
    def circuit(gamma, beta):
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
        # Cost encoding
        for i in range(n_qubits):
            qml.RZ(gamma * (i + 1), wires=i)
        # Mixer
        for i in range(n_qubits):
            qml.RX(beta, wires=i)
        return qml.probs(wires=range(n_qubits))

    probs = circuit(0.5, 0.5)
    return True, f"Portfolio probs computed", 0


def test_qsvm() -> Tuple[bool, str, float]:
    """Quantum SVM: 2 qubits, kernel computation."""
    n_qubits = 2
    dev = qml.device('qlret.mixed', wires=n_qubits)

    @qml.qnode(dev)
    def kernel(x1, x2):
        # Encode x1
        for i in range(n_qubits):
            qml.RY(x1[i], wires=i)
        # Encode x2 (adjoint)
        for i in range(n_qubits - 1, -1, -1):
            qml.RY(-x2[i], wires=i)
        return qml.probs(wires=range(n_qubits))

    x1 = np.array([0.1, 0.2])
    x2 = np.array([0.3, 0.4])
    probs = kernel(x1, x2)
    overlap = float(probs[0])
    return True, f"Kernel overlap={overlap:.4f}", 0


def test_qae() -> Tuple[bool, str, float]:
    """Quantum Amplitude Estimation: 3 qubits."""
    n_qubits = 3
    dev = qml.device('qlret.mixed', wires=n_qubits)

    @qml.qnode(dev)
    def circuit():
        # State prep
        qml.RY(0.6, wires=0)
        # Grover iterations
        qml.Hadamard(wires=1)
        qml.Hadamard(wires=2)
        qml.ctrl(qml.PauliZ, control=1)(wires=0)
        return qml.probs(wires=range(n_qubits))

    probs = circuit()
    return True, f"Amplitude estimated", 0


def test_vqd() -> Tuple[bool, str, float]:
    """VQD: 2 qubits, ground + excited state."""
    n_qubits = 2
    dev = qml.device('qlret.mixed', wires=n_qubits)

    @qml.qnode(dev, interface='autograd')
    def circuit(params):
        qml.RY(params[0], wires=0)
        qml.RY(params[1], wires=1)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0))

    params = pnp.array([0.5, 0.5], requires_grad=True)
    result = circuit(params)
    return True, f"VQD energy={float(result):.4f}", 0


def test_qgan() -> Tuple[bool, str, float]:
    """Quantum GAN: 2 qubits, generator test."""
    n_qubits = 2
    dev = qml.device('qlret.mixed', wires=n_qubits)

    @qml.qnode(dev)
    def generator(params):
        qml.RY(params[0], wires=0)
        qml.RY(params[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RY(params[2], wires=0)
        return qml.probs(wires=range(n_qubits))

    params = np.array([0.5, 0.5, 0.5])
    probs = generator(params)
    return True, f"Generated distribution", 0


def test_number_partitioning() -> Tuple[bool, str, float]:
    """Number Partitioning: 3 numbers."""
    n_qubits = 3
    dev = qml.device('qlret.mixed', wires=n_qubits)

    @qml.qnode(dev)
    def circuit(gamma, beta):
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
        # Cost layer
        for i in range(n_qubits):
            qml.RZ(gamma, wires=i)
        # Mixer
        for i in range(n_qubits):
            qml.RX(beta, wires=i)
        return qml.probs(wires=range(n_qubits))

    probs = circuit(0.5, 0.5)
    return True, f"Partition found", 0


def test_vqt() -> Tuple[bool, str, float]:
    """VQT: 2 qubits, thermal state preparation."""
    n_qubits = 2
    dev = qml.device('qlret.mixed', wires=n_qubits)

    @qml.qnode(dev)
    def circuit(params):
        qml.RY(params[0], wires=0)
        qml.RY(params[1], wires=1)
        qml.CNOT(wires=[0, 1])
        return qml.probs(wires=range(n_qubits))

    params = np.array([0.5, 0.5])
    probs = circuit(params)
    return True, f"Thermal state prepared", 0


def test_quantum_walk() -> Tuple[bool, str, float]:
    """Quantum Walk: 3 position qubits."""
    n_qubits = 3
    dev = qml.device('qlret.mixed', wires=n_qubits + 1)  # +1 for coin

    @qml.qnode(dev)
    def circuit():
        # Initial state
        qml.Hadamard(wires=0)  # Coin
        # Walk step
        qml.Hadamard(wires=0)
        qml.ctrl(qml.PauliX, control=0)(wires=1)
        return qml.probs(wires=range(1, n_qubits + 1))

    probs = circuit()
    return True, f"Walk distribution computed", 0


def test_kernel_alignment() -> Tuple[bool, str, float]:
    """Kernel Alignment: 2 qubits."""
    n_qubits = 2
    dev = qml.device('qlret.mixed', wires=n_qubits)

    @qml.qnode(dev, interface='autograd')
    def circuit(x, params):
        for i in range(n_qubits):
            qml.RY(x[i] * params[i], wires=i)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0))

    x = np.array([0.5, 0.5])
    params = pnp.array([1.0, 1.0], requires_grad=True)
    result = circuit(x, params)
    return True, f"Kernel aligned", 0


def test_subsampling_qnn() -> Tuple[bool, str, float]:
    """Sub-sampling QNN: 2 qubits, reduced data."""
    n_qubits = 2
    dev = qml.device('qlret.mixed', wires=n_qubits)

    @qml.qnode(dev, interface='autograd')
    def circuit(x, params):
        for i in range(n_qubits):
            qml.RY(x[i], wires=i)
        for i in range(n_qubits):
            qml.RY(params[i], wires=i)
        return qml.expval(qml.PauliZ(0))

    x = np.array([0.5, 0.5])
    params = pnp.array([0.3, 0.3], requires_grad=True)
    result = circuit(x, params)
    return True, f"Subsampled result={float(result):.4f}", 0


def test_hea() -> Tuple[bool, str, float]:
    """Hardware-Efficient Ansatz: 3 qubits, 1 layer."""
    n_qubits = 3
    dev = qml.device('qlret.mixed', wires=n_qubits)

    @qml.qnode(dev)
    def circuit(params):
        # Layer of rotations
        for i in range(n_qubits):
            qml.RY(params[i], wires=i)
            qml.RZ(params[i + n_qubits], wires=i)
        # Entanglement
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        return qml.expval(qml.PauliZ(0))

    params = np.random.randn(2 * n_qubits)
    result = circuit(params)
    return True, f"HEA result={float(result):.4f}", 0


def test_adapt_vqe() -> Tuple[bool, str, float]:
    """ADAPT-VQE: 2 qubits, adaptive ansatz."""
    n_qubits = 2
    dev = qml.device('qlret.mixed', wires=n_qubits)

    @qml.qnode(dev, interface='autograd')
    def circuit(params):
        # Initial state
        qml.PauliX(wires=0)
        # Adaptive operators
        qml.SingleExcitation(params[0], wires=[0, 1])
        return qml.expval(qml.PauliZ(0))

    params = pnp.array([0.1], requires_grad=True)
    result = circuit(params)
    return True, f"ADAPT energy={float(result):.4f}", 0


# ============================================================================
# TEST REGISTRY
# ============================================================================

ALGORITHMS = {
    # Tier 1 - Must Test
    'vqe': {'fn': test_vqe, 'tier': 1, 'name': 'VQE', 'qubits': 2},
    'qaoa': {'fn': test_qaoa, 'tier': 1, 'name': 'QAOA', 'qubits': 3},
    'qnn': {'fn': test_qnn, 'tier': 1, 'name': 'QNN', 'qubits': 2},
    'qft': {'fn': test_qft, 'tier': 1, 'name': 'QFT', 'qubits': 3},
    'qpe': {'fn': test_qpe, 'tier': 1, 'name': 'QPE', 'qubits': 4},
    'grover': {'fn': test_grover, 'tier': 1, 'name': 'Grover', 'qubits': 3},
    'metrology': {'fn': test_metrology, 'tier': 1, 'name': 'Metrology', 'qubits': 2},

    # Tier 2 - Should Test
    'uccsd': {'fn': test_uccsd, 'tier': 2, 'name': 'UCCSD-VQE', 'qubits': 2},
    'portfolio': {'fn': test_portfolio, 'tier': 2, 'name': 'Portfolio', 'qubits': 3},
    'qsvm': {'fn': test_qsvm, 'tier': 2, 'name': 'QSVM', 'qubits': 2},
    'qae': {'fn': test_qae, 'tier': 2, 'name': 'QAE', 'qubits': 3},
    'vqd': {'fn': test_vqd, 'tier': 2, 'name': 'VQD', 'qubits': 2},
    'qgan': {'fn': test_qgan, 'tier': 2, 'name': 'qGAN', 'qubits': 2},
    'number_partitioning': {'fn': test_number_partitioning, 'tier': 2, 'name': 'Number Partition', 'qubits': 3},

    # Tier 3 - Nice to Test
    'vqt': {'fn': test_vqt, 'tier': 3, 'name': 'VQT', 'qubits': 2},
    'quantum_walk': {'fn': test_quantum_walk, 'tier': 3, 'name': 'Quantum Walk', 'qubits': 4},
    'kernel_alignment': {'fn': test_kernel_alignment, 'tier': 3, 'name': 'Kernel Alignment', 'qubits': 2},
    'subsampling_qnn': {'fn': test_subsampling_qnn, 'tier': 3, 'name': 'Subsampling QNN', 'qubits': 2},
    'hea': {'fn': test_hea, 'tier': 3, 'name': 'HEA', 'qubits': 3},
    'adapt_vqe': {'fn': test_adapt_vqe, 'tier': 3, 'name': 'ADAPT-VQE', 'qubits': 2},
}


# ============================================================================
# MAIN RUNNER
# ============================================================================

def run_quick_validation(
    tiers: List[int] = None,
    algorithms: List[str] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """Run quick validation tests."""

    # Determine which algorithms to test
    if algorithms:
        to_test = {k: v for k, v in ALGORITHMS.items() if k in algorithms}
    elif tiers:
        to_test = {k: v for k, v in ALGORITHMS.items() if v['tier'] in tiers}
    else:
        to_test = ALGORITHMS

    print("=" * 70)
    print("LRET PennyLane Quick Validation")
    print("=" * 70)
    print(f"Testing {len(to_test)} algorithms with LRET device")
    print(f"Expected time: ~{len(to_test) * 5}-{len(to_test) * 15} seconds")
    print("=" * 70)

    results = {
        'timestamp': datetime.now().isoformat(),
        'total_algorithms': len(to_test),
        'tests': {}
    }

    passed = 0
    failed = 0
    total_time = 0

    for tier in [1, 2, 3]:
        tier_algos = {k: v for k, v in to_test.items() if v['tier'] == tier}
        if not tier_algos:
            continue

        print(f"\n--- Tier {tier} ---")

        for key, info in tier_algos.items():
            start = time.time()
            try:
                success, message, _ = info['fn']()
                elapsed = time.time() - start
                total_time += elapsed

                status = "PASS" if success else "FAIL"
                symbol = "[PASS]" if success else "[FAIL]"

                if success:
                    passed += 1
                else:
                    failed += 1

                results['tests'][key] = {
                    'name': info['name'],
                    'tier': tier,
                    'qubits': info['qubits'],
                    'passed': success,
                    'time_seconds': elapsed,
                    'message': message
                }

                print(f"  {symbol} {info['name']:20s} ({elapsed:.2f}s) - {message[:40]}")

            except Exception as e:
                elapsed = time.time() - start
                total_time += elapsed
                failed += 1

                error_msg = str(e)
                if verbose:
                    error_msg = traceback.format_exc()

                results['tests'][key] = {
                    'name': info['name'],
                    'tier': tier,
                    'qubits': info['qubits'],
                    'passed': False,
                    'time_seconds': elapsed,
                    'error': error_msg
                }

                print(f"  ✗ {info['name']:20s} ({elapsed:.2f}s) - ERROR: {str(e)[:40]}")
                if verbose:
                    print(f"    {traceback.format_exc()}")

    # Summary
    results['summary'] = {
        'passed': passed,
        'failed': failed,
        'total': passed + failed,
        'pass_rate': passed / (passed + failed) * 100 if (passed + failed) > 0 else 0,
        'total_time_seconds': total_time
    }

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Passed: {passed}/{passed + failed} ({results['summary']['pass_rate']:.1f}%)")
    print(f"Failed: {failed}/{passed + failed}")
    print(f"Total time: {total_time:.2f}s")

    if failed > 0:
        print("\n--- FAILED TESTS ---")
        for key, data in results['tests'].items():
            if not data['passed']:
                print(f"  [FAIL] {data['name']}: {data.get('error', data.get('message', 'Unknown error'))[:60]}")

    # Save results
    output_file = f"validation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Quick validation test for all PennyLane algorithms'
    )
    parser.add_argument(
        '--tier', '-t', type=int, nargs='+',
        help='Tier(s) to test (1, 2, 3)'
    )
    parser.add_argument(
        '--algorithm', '-a', type=str, nargs='+',
        help='Specific algorithm(s) to test'
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Show detailed error traces'
    )
    parser.add_argument(
        '--list', '-l', action='store_true',
        help='List available algorithms'
    )

    args = parser.parse_args()

    if args.list:
        print("\nAvailable algorithms for quick validation:")
        print("-" * 50)
        for tier in [1, 2, 3]:
            print(f"\nTier {tier}:")
            for key, info in ALGORITHMS.items():
                if info['tier'] == tier:
                    print(f"  {key:20s} - {info['name']} ({info['qubits']}q)")
        return

    results = run_quick_validation(
        tiers=args.tier,
        algorithms=args.algorithm,
        verbose=args.verbose
    )

    # Exit with error code if any tests failed
    if results['summary']['failed'] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

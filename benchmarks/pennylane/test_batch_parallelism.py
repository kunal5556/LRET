#!/usr/bin/env python3
"""
Test script for LRET batch parallelism feature.

This script demonstrates and benchmarks the Python-level batch parallelism
capability added to the LRET PennyLane device.

Usage:
    python test_batch_parallelism.py
    python test_batch_parallelism.py --batch-size 20 --num-params 10
"""

import time
import argparse
import sys
import os

# Add parent paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))

import numpy as np

try:
    import pennylane as qml
    from qlret import QLRETDevice
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure LRET is installed: cd python && pip install -e .")
    sys.exit(1)


def create_test_circuit(dev, n_params):
    """Create a parametrized test circuit."""
    @qml.qnode(dev)
    def circuit(params):
        # Simple variational ansatz
        for i in range(dev.num_wires):
            qml.RY(params[i], wires=i)
        
        # Entangling layer
        for i in range(dev.num_wires - 1):
            qml.CNOT(wires=[i, i + 1])
        
        # Another rotation layer
        for i in range(dev.num_wires):
            qml.RZ(params[i + dev.num_wires] if i + dev.num_wires < len(params) else 0, wires=i)
        
        return qml.expval(qml.PauliZ(0))
    
    return circuit


def benchmark_batch_execution(n_qubits, n_params, batch_size, max_batch_workers, num_threads):
    """Benchmark batch circuit execution with different parallelism settings."""
    
    print(f"\n{'='*70}")
    print(f"Configuration:")
    print(f"  Qubits: {n_qubits}")
    print(f"  Parameters: {n_params}")
    print(f"  Batch size: {batch_size}")
    print(f"  max_batch_workers: {max_batch_workers}")
    print(f"  num_threads: {num_threads}")
    print(f"{'='*70}")
    
    # Create device
    dev = QLRETDevice(
        wires=n_qubits,
        epsilon=1e-4,
        num_threads=num_threads,
        parallel_mode="hybrid",
        max_batch_workers=max_batch_workers,
    )
    
    # Show computed strategy
    workers, threads_per_circuit = dev._compute_execution_strategy(batch_size)
    print(f"\nExecution strategy:")
    print(f"  Workers: {workers}")
    print(f"  Threads per circuit: {threads_per_circuit}")
    print(f"  Total threads: {workers * threads_per_circuit}")
    
    # Create circuit
    circuit = create_test_circuit(dev, n_params)
    
    # Generate random parameter sets
    np.random.seed(42)
    param_sets = [np.random.uniform(-np.pi, np.pi, n_params) for _ in range(batch_size)]
    
    # Warm-up run
    print(f"\nWarm-up run...")
    _ = circuit(param_sets[0])
    
    # Timed batch execution
    print(f"Executing {batch_size} circuits...")
    start_time = time.perf_counter()
    
    results = []
    for params in param_sets:
        result = circuit(params)
        results.append(result)
    
    end_time = time.perf_counter()
    total_time = end_time - start_time
    avg_time = total_time / batch_size
    
    print(f"\n{'='*70}")
    print(f"Results:")
    print(f"  Total time: {total_time:.3f} seconds")
    print(f"  Average per circuit: {avg_time*1000:.2f} ms")
    print(f"  Throughput: {batch_size/total_time:.2f} circuits/second")
    print(f"{'='*70}")
    
    return {
        "total_time": total_time,
        "avg_time": avg_time,
        "throughput": batch_size / total_time,
        "results": results,
    }


def run_comparison(n_qubits=4, n_params=8, batch_size=16, num_threads=0):
    """Run comparison between sequential and parallel execution."""
    
    print("\n" + "="*70)
    print("LRET Batch Parallelism Benchmark")
    print("="*70)
    print(f"System CPU cores: {os.cpu_count()}")
    
    # Auto-detect threads if 0
    if num_threads == 0:
        num_threads = os.cpu_count() or 4
    
    # Sequential execution
    print("\n" + "-"*70)
    print("TEST 1: Sequential execution (max_batch_workers=0)")
    print("-"*70)
    seq_result = benchmark_batch_execution(
        n_qubits, n_params, batch_size,
        max_batch_workers=0,
        num_threads=num_threads,
    )
    
    # Parallel execution with 4 workers
    n_workers = min(4, os.cpu_count() // 2) if os.cpu_count() else 2
    threads_per_worker = max(1, num_threads // n_workers)
    
    print("\n" + "-"*70)
    print(f"TEST 2: Parallel execution (max_batch_workers={n_workers})")
    print("-"*70)
    par_result = benchmark_batch_execution(
        n_qubits, n_params, batch_size,
        max_batch_workers=n_workers,
        num_threads=num_threads,
    )
    
    # Auto-tune execution
    print("\n" + "-"*70)
    print("TEST 3: Auto-tune execution (max_batch_workers=-1)")
    print("-"*70)
    auto_result = benchmark_batch_execution(
        n_qubits, n_params, batch_size,
        max_batch_workers=-1,
        num_threads=num_threads,
    )
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Mode':<25} {'Total Time':>12} {'Speedup':>10}")
    print("-"*70)
    
    seq_time = seq_result["total_time"]
    par_time = par_result["total_time"]
    auto_time = auto_result["total_time"]
    
    print(f"{'Sequential (baseline)':<25} {seq_time:>10.3f}s {1.0:>10.2f}x")
    print(f"{'Parallel (N workers)':<25} {par_time:>10.3f}s {seq_time/par_time:>10.2f}x")
    print(f"{'Auto-tune':<25} {auto_time:>10.3f}s {seq_time/auto_time:>10.2f}x")
    print("="*70)
    
    # Verify results match
    print("\nVerifying result consistency...")
    seq_vals = np.array(seq_result["results"])
    par_vals = np.array(par_result["results"])
    auto_vals = np.array(auto_result["results"])
    
    if np.allclose(seq_vals, par_vals, atol=1e-6) and np.allclose(seq_vals, auto_vals, atol=1e-6):
        print("✓ All results match (within tolerance)")
    else:
        print("✗ Results differ!")
        print(f"  Max diff (seq vs par): {np.max(np.abs(seq_vals - par_vals))}")
        print(f"  Max diff (seq vs auto): {np.max(np.abs(seq_vals - auto_vals))}")
    
    return seq_result, par_result, auto_result


def main():
    parser = argparse.ArgumentParser(description="Test LRET batch parallelism")
    parser.add_argument("--qubits", type=int, default=4, help="Number of qubits (default: 4)")
    parser.add_argument("--params", type=int, default=8, help="Number of parameters (default: 8)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--num-threads", type=int, default=0, help="C++ threads (default: 0 = auto)")
    args = parser.parse_args()
    
    run_comparison(
        n_qubits=args.qubits,
        n_params=args.params,
        batch_size=args.batch_size,
        num_threads=args.num_threads,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Quick Parallel Modes Benchmark

This script runs a quick version of the parallel modes benchmark for rapid testing
and validation during development. Estimated runtime: 5-10 minutes.

Configuration:
- Qubit range: [4, 6, 8]
- Circuit depth: 10
- Noise: 0.01 (1%)
- Epsilon: 1e-4
- Modes: [sequential, row, hybrid]
- Circuit types: [random, qft, ghz]
- Trials: 2

Usage:
    python benchmarks/parallel_modes_benchmark_quick.py

Output:
    results/parallel_modes_quick/
    ├── results.json
    ├── summary.json
    ├── config.json
    └── benchmark.log
"""

from parallel_modes_benchmark import (
    BenchmarkConfig,
    CircuitConfig,
    ParallelModesBenchmark
)

def main():
    """Run quick benchmark."""
    print("=" * 80)
    print("QUICK PARALLEL MODES BENCHMARK")
    print("=" * 80)
    print("Estimated runtime: 5-10 minutes")
    print("")
    print("Configuration:")
    print("  Qubits: [4, 6, 8]")
    print("  Depth: 10")
    print("  Noise: 0.01 (1%)")
    print("  Modes: [sequential, row, hybrid]")
    print("  Trials: 2 per configuration")
    print("=" * 80)
    print("")

    # Define quick configuration
    circuit_configs = [
        CircuitConfig("random", n_qubits=4, depth=10, noise_prob=0.01, epsilon=1e-4),
        CircuitConfig("random", n_qubits=6, depth=10, noise_prob=0.01, epsilon=1e-4),
        CircuitConfig("random", n_qubits=8, depth=10, noise_prob=0.01, epsilon=1e-4),
        CircuitConfig("qft", n_qubits=6, depth=6, noise_prob=0.01, epsilon=1e-4),
        CircuitConfig("ghz", n_qubits=8, depth=8, noise_prob=0.01, epsilon=1e-4),
    ]

    config = BenchmarkConfig(
        circuit_configs=circuit_configs,
        modes=["sequential", "row", "hybrid"],
        trials=2,
        output_dir="results/parallel_modes_quick",
        save_intermediate=True,
        export_state=False  # Set to True for validation
    )

    # Run benchmark
    benchmark = ParallelModesBenchmark(config)
    results = benchmark.run_full_benchmark()

    # Summary
    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETED")
    print("=" * 80)
    print(f"Results directory: {benchmark.run_dir}")
    print(f"Total runs: {len(results)}")

    success_count = sum(1 for r in results if r["status"] == "success")
    failed_count = sum(1 for r in results if r["status"] == "error")
    timeout_count = sum(1 for r in results if r["status"] == "timeout")

    print(f"  Successful: {success_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Timeout: {timeout_count}")
    print("")
    print("Next steps:")
    print("  1. Visualize results:")
    print(f"     python scripts/benchmark_visualize_modes.py {benchmark.run_dir}/results.json")
    print("  2. Validate correctness:")
    print(f"     python benchmarks/validation_utils.py {benchmark.run_dir}/results.json")
    print("=" * 80)


if __name__ == "__main__":
    main()

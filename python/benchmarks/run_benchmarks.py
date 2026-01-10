"""Main benchmark runner script.

This script orchestrates the complete benchmarking process, running circuits
across multiple devices and collecting performance metrics.

Usage:
    python run_benchmarks.py --mode quick   # Quick test (2-10 minutes)
    python run_benchmarks.py --mode standard  # Standard benchmarks (1-2 hours)
    python run_benchmarks.py --mode full     # Full suite (many hours)
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import pennylane as qml

from .config import (
    BenchmarkConfig,
    DEFAULT_CONFIG,
    QUICK_TEST_CONFIG,
    get_available_devices,
    create_device,
)
from .circuits import (
    CircuitSpec,
    CircuitType,
    create_benchmark_qnode,
    QUICK_SUITE,
    STANDARD_SUITE,
    SCALABILITY_SUITE,
)
from .metrics import (
    ExecutionMetrics,
    BenchmarkResults,
    measure_execution,
)


def print_header(text: str, char: str = "="):
    """Print a formatted header."""
    line = char * 60
    print(f"\n{line}")
    print(f" {text}")
    print(f"{line}\n")


def print_progress(current: int, total: int, prefix: str = ""):
    """Print progress indicator."""
    pct = current / total * 100
    bar_len = 30
    filled = int(bar_len * current / total)
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"\r{prefix}[{bar}] {current}/{total} ({pct:.1f}%)", end="", flush=True)


def run_single_benchmark(
    device_name: str,
    circuit_spec: CircuitSpec,
    trial: int,
    timeout_seconds: float = 300.0,
) -> ExecutionMetrics:
    """Run a single benchmark trial.
    
    Parameters
    ----------
    device_name : str
        Name of the device to benchmark.
    circuit_spec : CircuitSpec
        Circuit specification.
    trial : int
        Trial number.
    timeout_seconds : float
        Maximum execution time.
    
    Returns
    -------
    ExecutionMetrics
        Collected metrics from the run.
    """
    metrics = ExecutionMetrics(
        device_name=device_name,
        circuit_type=circuit_spec.circuit_type.value,
        num_qubits=circuit_spec.num_qubits,
        circuit_depth=circuit_spec.depth,
        trial=trial,
        execution_time_ms=0.0,
    )
    
    try:
        # Create device
        device = create_device(device_name, circuit_spec.num_qubits)
        if device is None:
            metrics.success = False
            metrics.error_message = f"Failed to create device: {device_name}"
            return metrics
        
        # Create QNode
        qnode, num_params, init_params = create_benchmark_qnode(
            device, circuit_spec, seed=42 + trial
        )
        
        # Run with measurement
        if num_params == 0:
            result, exec_time, peak_mem, mem_delta = measure_execution(
                qnode, warmup=(trial == 0)
            )
        else:
            # Split params for QNN (weights, features)
            if circuit_spec.circuit_type == CircuitType.QNN:
                weights = init_params[:-circuit_spec.num_qubits]
                features = init_params[-circuit_spec.num_qubits:]
                result, exec_time, peak_mem, mem_delta = measure_execution(
                    qnode, weights, features, warmup=(trial == 0)
                )
            else:
                result, exec_time, peak_mem, mem_delta = measure_execution(
                    qnode, init_params, warmup=(trial == 0)
                )
        
        # Update metrics
        metrics.execution_time_ms = exec_time
        metrics.peak_memory_mb = peak_mem
        metrics.memory_delta_mb = mem_delta
        
        # Store result value
        if isinstance(result, np.ndarray):
            metrics.result_value = float(result[0]) if len(result) > 0 else None
        else:
            metrics.result_value = float(result) if result is not None else None
        
        metrics.success = True
        
    except MemoryError as e:
        metrics.success = False
        metrics.error_message = f"MemoryError: {e}"
    
    except Exception as e:
        metrics.success = False
        metrics.error_message = f"{type(e).__name__}: {e}"
    
    return metrics


def run_benchmark_suite(
    device_names: List[str],
    circuit_suite: List[CircuitSpec],
    config: BenchmarkConfig,
    results: BenchmarkResults,
    verbose: bool = True,
) -> None:
    """Run a suite of benchmarks across multiple devices.
    
    Parameters
    ----------
    device_names : list of str
        Device names to benchmark.
    circuit_suite : list of CircuitSpec
        Circuit specifications to run.
    config : BenchmarkConfig
        Benchmark configuration.
    results : BenchmarkResults
        Results collector.
    verbose : bool
        Whether to print progress.
    """
    total_runs = len(device_names) * len(circuit_suite) * config.num_trials
    current_run = 0
    
    for device_name in device_names:
        if verbose:
            print_header(f"Benchmarking: {device_name}", "-")
        
        for circuit_spec in circuit_suite:
            if verbose:
                print(f"\n  Circuit: {circuit_spec.name}")
            
            for trial in range(config.num_trials):
                current_run += 1
                if verbose:
                    print_progress(current_run, total_runs, "  Progress: ")
                
                metrics = run_single_benchmark(
                    device_name,
                    circuit_spec,
                    trial,
                    config.timeout_seconds,
                )
                results.add(metrics)
                
                # Early termination on repeated failures
                if not metrics.success and trial == 0:
                    if verbose:
                        print(f"\n    ⚠ Failed: {metrics.error_message[:50]}...")
                    # Skip remaining trials for this circuit
                    for remaining_trial in range(trial + 1, config.num_trials):
                        skip_metrics = ExecutionMetrics(
                            device_name=device_name,
                            circuit_type=circuit_spec.circuit_type.value,
                            num_qubits=circuit_spec.num_qubits,
                            circuit_depth=circuit_spec.depth,
                            trial=remaining_trial,
                            execution_time_ms=0.0,
                            success=False,
                            error_message="Skipped due to prior failure",
                        )
                        results.add(skip_metrics)
                        current_run += 1
                    break
        
        if verbose:
            print()  # New line after progress bar


def run_quick_benchmark(output_dir: Path, verbose: bool = True) -> BenchmarkResults:
    """Run quick benchmark suite.
    
    Parameters
    ----------
    output_dir : Path
        Directory to save results.
    verbose : bool
        Whether to print progress.
    
    Returns
    -------
    BenchmarkResults
        Collected results.
    """
    if verbose:
        print_header("QLRET Quick Benchmark Suite")
    
    # Get available devices
    available_devices = get_available_devices()
    if verbose:
        print(f"Available devices: {', '.join(available_devices)}")
    
    # Initialize results
    results = BenchmarkResults()
    config = QUICK_TEST_CONFIG
    
    # Run benchmarks
    run_benchmark_suite(
        available_devices,
        QUICK_SUITE,
        config,
        results,
        verbose,
    )
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results.save_json(output_dir / f"quick_benchmark_{timestamp}.json")
    
    if verbose:
        print_header("Benchmark Complete")
        print(f"Results saved to: {output_dir}")
    
    return results


def run_standard_benchmark(output_dir: Path, verbose: bool = True) -> BenchmarkResults:
    """Run standard benchmark suite.
    
    Parameters
    ----------
    output_dir : Path
        Directory to save results.
    verbose : bool
        Whether to print progress.
    
    Returns
    -------
    BenchmarkResults
        Collected results.
    """
    if verbose:
        print_header("QLRET Standard Benchmark Suite")
    
    available_devices = get_available_devices()
    if verbose:
        print(f"Available devices: {', '.join(available_devices)}")
    
    results = BenchmarkResults()
    config = DEFAULT_CONFIG
    
    run_benchmark_suite(
        available_devices,
        STANDARD_SUITE,
        config,
        results,
        verbose,
    )
    
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results.save_json(output_dir / f"standard_benchmark_{timestamp}.json")
    
    try:
        results.save_csv(output_dir / f"standard_benchmark_{timestamp}.csv")
    except ImportError:
        pass  # pandas not available
    
    if verbose:
        print_header("Benchmark Complete")
        print(f"Results saved to: {output_dir}")
    
    return results


def run_scalability_benchmark(
    output_dir: Path,
    max_qubits: int = 16,
    verbose: bool = True
) -> BenchmarkResults:
    """Run scalability benchmark.
    
    Parameters
    ----------
    output_dir : Path
        Directory to save results.
    max_qubits : int
        Maximum qubits to test.
    verbose : bool
        Whether to print progress.
    
    Returns
    -------
    BenchmarkResults
        Collected results.
    """
    if verbose:
        print_header("QLRET Scalability Benchmark")
        print(f"Testing up to {max_qubits} qubits")
    
    available_devices = get_available_devices()
    if verbose:
        print(f"Available devices: {', '.join(available_devices)}")
    
    # Filter scalability suite by max_qubits
    suite = [spec for spec in SCALABILITY_SUITE if spec.num_qubits <= max_qubits]
    
    results = BenchmarkResults()
    config = DEFAULT_CONFIG
    
    run_benchmark_suite(
        available_devices,
        suite,
        config,
        results,
        verbose,
    )
    
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results.save_json(output_dir / f"scalability_benchmark_{timestamp}.json")
    
    if verbose:
        print_header("Benchmark Complete")
        print(f"Results saved to: {output_dir}")
    
    return results


def print_summary(results: BenchmarkResults):
    """Print a summary of benchmark results."""
    print_header("Results Summary")
    
    summary = results.summary_by_device()
    
    for device, stats in summary.items():
        print(f"\n{device}:")
        print(f"  Successful runs: {stats['num_successful']}")
        print(f"  Avg execution time: {stats['avg_time_ms']:.2f} ms")
        print(f"  Max memory: {stats['max_memory_mb']:.1f} MB")
    
    # Compare to QLRET baseline
    if "qlret.mixed" in summary and len(summary) > 1:
        print("\n--- Comparison to QLRET ---")
        time_ratios = results.compare_devices("qlret.mixed", "execution_time_ms")
        for device, ratio in time_ratios.items():
            if device != "qlret.mixed":
                if ratio > 1:
                    print(f"  {device}: QLRET is {ratio:.2f}x faster")
                else:
                    print(f"  {device}: QLRET is {1/ratio:.2f}x slower")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="QLRET PennyLane Plugin Benchmarks"
    )
    parser.add_argument(
        "--mode",
        choices=["quick", "standard", "scalability"],
        default="quick",
        help="Benchmark mode (default: quick)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Output directory for results (default: results)",
    )
    parser.add_argument(
        "--max-qubits",
        type=int,
        default=16,
        help="Maximum qubits for scalability test (default: 16)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )
    
    args = parser.parse_args()
    output_dir = Path(args.output)
    verbose = not args.quiet
    
    start_time = time.time()
    
    if args.mode == "quick":
        results = run_quick_benchmark(output_dir, verbose)
    elif args.mode == "standard":
        results = run_standard_benchmark(output_dir, verbose)
    elif args.mode == "scalability":
        results = run_scalability_benchmark(output_dir, args.max_qubits, verbose)
    
    elapsed = time.time() - start_time
    
    if verbose:
        print_summary(results)
        print(f"\nTotal time: {elapsed:.1f} seconds")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

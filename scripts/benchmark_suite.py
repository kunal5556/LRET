#!/usr/bin/env python3
"""Master benchmark suite orchestrator for LRET quantum simulator.

This module runs comprehensive benchmarks across multiple categories:
- Scaling: Time vs qubit count (exponential analysis)
- Parallel: Speedup comparison across modes
- Accuracy: LRET vs FDM fidelity validation
- Depth: Circuit depth scaling analysis
- Memory: Memory usage profiling

Usage:
    python scripts/benchmark_suite.py --output benchmark_results.csv
    python scripts/benchmark_suite.py --categories scaling,parallel
    python scripts/benchmark_suite.py --quick  # Fast subset for CI

Examples:
    # Run all benchmarks with default settings
    python scripts/benchmark_suite.py

    # Quick CI mode (fewer trials, smaller qubit range)
    python scripts/benchmark_suite.py --quick

    # Only scaling and parallel benchmarks
    python scripts/benchmark_suite.py --categories scaling,parallel

    # Custom output directory
    python scripts/benchmark_suite.py --output-dir results/

Author: LRET Development Team
Date: January 2026
"""

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
import re


@dataclass
class BenchmarkResult:
    """Single benchmark measurement result.
    
    Attributes:
        category: Benchmark category (scaling, parallel, accuracy, etc.)
        n_qubits: Number of qubits in simulation
        depth: Circuit depth
        mode: Parallelization mode (sequential, row, column, hybrid)
        trial: Trial number (0-indexed)
        time_ms: Wall-clock execution time in milliseconds
        final_rank: Final density matrix rank after simulation
        memory_mb: Peak memory usage in megabytes (optional)
        fidelity: LRET vs FDM fidelity (accuracy category only)
        trace_distance: Trace distance metric (optional)
        noise_level: Noise level used (optional)
        reported_time_ms: Simulator-reported execution time (optional)
        error_message: Error message if benchmark failed (optional)
    """
    category: str
    n_qubits: int
    depth: int
    mode: str
    trial: int
    time_ms: float
    final_rank: int
    memory_mb: Optional[float] = None
    fidelity: Optional[float] = None
    trace_distance: Optional[float] = None
    noise_level: Optional[float] = None
    reported_time_ms: Optional[float] = None
    error_message: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for CSV/JSON serialization."""
        return asdict(self)
    
    def is_success(self) -> bool:
        """Check if benchmark completed successfully."""
        return self.error_message is None and self.time_ms > 0


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark suite execution.
    
    Attributes:
        scaling_qubits: List of qubit counts for scaling benchmarks
        scaling_depth: Circuit depth for scaling benchmarks
        parallel_qubits: Qubit count for parallel benchmarks
        parallel_depth: Circuit depth for parallel benchmarks
        accuracy_qubits: List of qubit counts for accuracy benchmarks
        accuracy_depth: Circuit depth for accuracy benchmarks
        depth_qubits: Qubit count for depth scaling benchmarks
        depth_values: List of depths for depth scaling benchmarks
        memory_qubits: List of qubit counts for memory benchmarks
        trials_scaling: Number of trials for scaling benchmarks
        trials_parallel: Number of trials for parallel benchmarks
        trials_accuracy: Number of trials for accuracy benchmarks
        trials_depth: Number of trials for depth benchmarks
        timeout: Maximum execution time per benchmark in seconds
    """
    # Scaling configuration
    scaling_qubits: List[int] = field(default_factory=lambda: [8, 9, 10, 11, 12, 13, 14])
    scaling_depth: int = 15
    
    # Parallel configuration
    parallel_qubits: int = 12
    parallel_depth: int = 20
    parallel_modes: List[str] = field(default_factory=lambda: ["sequential", "row", "column", "hybrid"])
    
    # Accuracy configuration
    accuracy_qubits: List[int] = field(default_factory=lambda: [6, 7, 8, 9, 10])
    accuracy_depth: int = 10
    accuracy_noise_levels: List[float] = field(default_factory=lambda: [0.0, 0.001, 0.01])
    
    # Depth scaling configuration
    depth_qubits: int = 10
    depth_values: List[int] = field(default_factory=lambda: [5, 10, 15, 20, 25, 30])
    
    # Memory profiling configuration
    memory_qubits: List[int] = field(default_factory=lambda: [10, 11, 12, 13, 14])
    memory_depth: int = 15
    
    # Trial counts
    trials_scaling: int = 3
    trials_parallel: int = 5
    trials_accuracy: int = 3
    trials_depth: int = 3
    trials_memory: int = 2
    
    # Execution settings
    timeout: int = 300  # 5 minutes max per benchmark
    
    @classmethod
    def quick_mode(cls) -> "BenchmarkConfig":
        """Create configuration for quick CI mode."""
        return cls(
            scaling_qubits=[8, 10, 12],
            scaling_depth=10,
            parallel_qubits=10,
            parallel_depth=15,
            accuracy_qubits=[6, 8],
            accuracy_depth=8,
            accuracy_noise_levels=[0.001],
            depth_qubits=8,
            depth_values=[5, 10, 15],
            memory_qubits=[10, 12],
            memory_depth=10,
            trials_scaling=1,
            trials_parallel=2,
            trials_accuracy=1,
            trials_depth=1,
            trials_memory=1,
            timeout=120,
        )


class BenchmarkRunner:
    """Orchestrates all benchmark categories and data collection.
    
    This class manages the execution of benchmarks, parsing of results,
    and aggregation of metrics into structured output formats.
    """
    
    def __init__(
        self, 
        quantum_sim_path: Path, 
        output_dir: Path,
        config: Optional[BenchmarkConfig] = None,
        verbose: bool = True
    ):
        """Initialize benchmark runner.
        
        Args:
            quantum_sim_path: Path to quantum_sim executable
            output_dir: Directory for output files
            config: Benchmark configuration (uses defaults if None)
            verbose: Whether to print progress messages
        """
        self.quantum_sim = quantum_sim_path.resolve()
        self.output_dir = output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or BenchmarkConfig()
        self.verbose = verbose
        self.results: List[BenchmarkResult] = []
        self.start_time = datetime.now()
        
        # Validate executable
        if not self.quantum_sim.exists():
            raise FileNotFoundError(f"quantum_sim not found at: {self.quantum_sim}")
    
    def _log(self, message: str, indent: int = 0):
        """Print log message if verbose mode is enabled."""
        if self.verbose:
            prefix = "  " * indent
            print(f"{prefix}{message}")
    
    def run_single_benchmark(
        self, 
        n: int, 
        d: int, 
        mode: str = "hybrid",
        fdm: bool = False,
        noise: Optional[float] = None,
        extra_args: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Run a single simulation and extract metrics.
        
        Args:
            n: Number of qubits
            d: Circuit depth
            mode: Parallelization mode
            fdm: Whether to use FDM (Full Density Matrix) mode
            noise: Noise level (overrides default)
            extra_args: Additional CLI arguments
            
        Returns:
            Dictionary with extracted metrics
            
        Raises:
            RuntimeError: If benchmark execution fails
            subprocess.TimeoutExpired: If benchmark exceeds timeout
        """
        cmd = [
            str(self.quantum_sim),
            "-n", str(n),
            "-d", str(d),
            "--mode", mode,
        ]
        
        if fdm:
            cmd.append("--fdm")
        
        if noise is not None:
            cmd.extend(["--noise", str(noise)])
        
        if extra_args:
            cmd.extend(extra_args)
        
        # Execute benchmark
        start = time.perf_counter()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout,
                cwd=self.quantum_sim.parent,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Benchmark timed out after {self.config.timeout}s")
        
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        if result.returncode != 0:
            error_msg = result.stderr.strip() or f"Exit code {result.returncode}"
            raise RuntimeError(f"Benchmark failed: {error_msg}")
        
        # Parse output
        metrics = self._parse_output(result.stdout)
        metrics["time_ms"] = elapsed_ms
        metrics["command"] = " ".join(cmd)
        
        return metrics
    
    def _parse_output(self, stdout: str) -> dict:
        """Extract metrics from CLI output.
        
        Parses the quantum_sim output to extract:
        - Final rank
        - Reported simulation time
        - Fidelity values
        - Memory usage
        
        Args:
            stdout: Standard output from quantum_sim
            
        Returns:
            Dictionary of extracted metrics
        """
        metrics = {}
        
        for line in stdout.split('\n'):
            line = line.strip()
            
            # Final rank extraction
            if "Final Rank:" in line or "final rank:" in line.lower():
                match = re.search(r'(\d+)', line.split(':')[-1])
                if match:
                    metrics["final_rank"] = int(match.group(1))
            
            # Reported simulation time
            elif "Simulation Time:" in line or "simulation time:" in line.lower():
                match = re.search(r'([\d.]+)', line.split(':')[-1])
                if match:
                    # Convert to ms (assuming seconds)
                    metrics["reported_time_ms"] = float(match.group(1)) * 1000
            
            # Fidelity extraction
            elif "fidelity" in line.lower():
                match = re.search(r'([\d.]+)', line.split(':')[-1])
                if match:
                    fid = float(match.group(1))
                    if 0 <= fid <= 1:
                        metrics["fidelity"] = fid
            
            # Trace distance
            elif "trace distance" in line.lower():
                match = re.search(r'([\d.e+-]+)', line.split(':')[-1])
                if match:
                    metrics["trace_distance"] = float(match.group(1))
            
            # Memory usage
            elif "memory" in line.lower() and ("mb" in line.lower() or "peak" in line.lower()):
                match = re.search(r'([\d.]+)', line)
                if match:
                    metrics["memory_mb"] = float(match.group(1))
        
        return metrics
    
    # -------------------------------------------------------------------------
    # Benchmark Categories
    # -------------------------------------------------------------------------
    
    def run_scaling_benchmarks(self) -> List[BenchmarkResult]:
        """Category 1: Qubit scaling benchmarks.
        
        Measures how execution time and rank scale with qubit count.
        Expected behavior: ~O(2^n) time scaling.
        
        Returns:
            List of benchmark results for this category
        """
        self._log("\n" + "=" * 60)
        self._log("RUNNING SCALING BENCHMARKS")
        self._log("=" * 60)
        
        results = []
        
        for n in self.config.scaling_qubits:
            self._log(f"n={n} qubits...", indent=1)
            
            for trial in range(self.config.trials_scaling):
                try:
                    metrics = self.run_single_benchmark(
                        n=n, 
                        d=self.config.scaling_depth, 
                        mode="hybrid"
                    )
                    
                    result = BenchmarkResult(
                        category="scaling",
                        n_qubits=n,
                        depth=self.config.scaling_depth,
                        mode="hybrid",
                        trial=trial,
                        time_ms=metrics["time_ms"],
                        final_rank=metrics.get("final_rank", -1),
                        memory_mb=metrics.get("memory_mb"),
                        reported_time_ms=metrics.get("reported_time_ms"),
                    )
                    results.append(result)
                    self.results.append(result)
                    
                    self._log(f"Trial {trial+1}: {metrics['time_ms']:.1f} ms, rank={metrics.get('final_rank', 'N/A')}", indent=2)
                    
                except Exception as e:
                    result = BenchmarkResult(
                        category="scaling",
                        n_qubits=n,
                        depth=self.config.scaling_depth,
                        mode="hybrid",
                        trial=trial,
                        time_ms=-1,
                        final_rank=-1,
                        error_message=str(e),
                    )
                    results.append(result)
                    self.results.append(result)
                    self._log(f"Trial {trial+1} FAILED: {e}", indent=2)
        
        return results
    
    def run_parallel_benchmarks(self) -> List[BenchmarkResult]:
        """Category 2: Parallel mode comparison benchmarks.
        
        Compares execution time across different parallelization strategies.
        Computes speedup relative to sequential baseline.
        
        Returns:
            List of benchmark results for this category
        """
        self._log("\n" + "=" * 60)
        self._log("RUNNING PARALLEL MODE BENCHMARKS")
        self._log("=" * 60)
        
        results = []
        n = self.config.parallel_qubits
        d = self.config.parallel_depth
        
        for mode in self.config.parallel_modes:
            self._log(f"Mode: {mode}...", indent=1)
            
            for trial in range(self.config.trials_parallel):
                try:
                    metrics = self.run_single_benchmark(n=n, d=d, mode=mode)
                    
                    result = BenchmarkResult(
                        category="parallel",
                        n_qubits=n,
                        depth=d,
                        mode=mode,
                        trial=trial,
                        time_ms=metrics["time_ms"],
                        final_rank=metrics.get("final_rank", -1),
                    )
                    results.append(result)
                    self.results.append(result)
                    
                    self._log(f"Trial {trial+1}: {metrics['time_ms']:.1f} ms", indent=2)
                    
                except Exception as e:
                    result = BenchmarkResult(
                        category="parallel",
                        n_qubits=n,
                        depth=d,
                        mode=mode,
                        trial=trial,
                        time_ms=-1,
                        final_rank=-1,
                        error_message=str(e),
                    )
                    results.append(result)
                    self.results.append(result)
                    self._log(f"Trial {trial+1} FAILED: {e}", indent=2)
        
        return results
    
    def run_accuracy_benchmarks(self) -> List[BenchmarkResult]:
        """Category 3: LRET vs FDM accuracy validation.
        
        Compares LRET approximation against full density matrix simulation.
        Validates that fidelity exceeds threshold (>0.999).
        
        Returns:
            List of benchmark results for this category
        """
        self._log("\n" + "=" * 60)
        self._log("RUNNING ACCURACY VALIDATION BENCHMARKS")
        self._log("=" * 60)
        
        results = []
        
        for n in self.config.accuracy_qubits:
            self._log(f"n={n} qubits (LRET vs FDM)...", indent=1)
            
            for noise in self.config.accuracy_noise_levels:
                self._log(f"noise={noise}...", indent=2)
                
                for trial in range(self.config.trials_accuracy):
                    try:
                        # Run with FDM comparison enabled
                        metrics = self.run_single_benchmark(
                            n=n, 
                            d=self.config.accuracy_depth, 
                            mode="hybrid",
                            fdm=True,
                            noise=noise,
                        )
                        
                        result = BenchmarkResult(
                            category="accuracy",
                            n_qubits=n,
                            depth=self.config.accuracy_depth,
                            mode="hybrid",
                            trial=trial,
                            time_ms=metrics["time_ms"],
                            final_rank=metrics.get("final_rank", -1),
                            fidelity=metrics.get("fidelity"),
                            trace_distance=metrics.get("trace_distance"),
                            noise_level=noise,
                        )
                        results.append(result)
                        self.results.append(result)
                        
                        fid_str = f"{metrics.get('fidelity', 'N/A'):.6f}" if metrics.get('fidelity') else "N/A"
                        self._log(f"Trial {trial+1}: fidelity={fid_str}", indent=3)
                        
                    except Exception as e:
                        result = BenchmarkResult(
                            category="accuracy",
                            n_qubits=n,
                            depth=self.config.accuracy_depth,
                            mode="hybrid",
                            trial=trial,
                            time_ms=-1,
                            final_rank=-1,
                            noise_level=noise,
                            error_message=str(e),
                        )
                        results.append(result)
                        self.results.append(result)
                        self._log(f"Trial {trial+1} FAILED: {e}", indent=3)
        
        return results
    
    def run_depth_benchmarks(self) -> List[BenchmarkResult]:
        """Category 4: Depth scaling benchmarks.
        
        Measures how execution time and rank scale with circuit depth.
        Expected behavior: ~O(d) time scaling (linear).
        
        Returns:
            List of benchmark results for this category
        """
        self._log("\n" + "=" * 60)
        self._log("RUNNING DEPTH SCALING BENCHMARKS")
        self._log("=" * 60)
        
        results = []
        n = self.config.depth_qubits
        
        for d in self.config.depth_values:
            self._log(f"depth={d}...", indent=1)
            
            for trial in range(self.config.trials_depth):
                try:
                    metrics = self.run_single_benchmark(n=n, d=d, mode="hybrid")
                    
                    result = BenchmarkResult(
                        category="depth_scaling",
                        n_qubits=n,
                        depth=d,
                        mode="hybrid",
                        trial=trial,
                        time_ms=metrics["time_ms"],
                        final_rank=metrics.get("final_rank", -1),
                    )
                    results.append(result)
                    self.results.append(result)
                    
                    self._log(f"Trial {trial+1}: {metrics['time_ms']:.1f} ms, rank={metrics.get('final_rank', 'N/A')}", indent=2)
                    
                except Exception as e:
                    result = BenchmarkResult(
                        category="depth_scaling",
                        n_qubits=n,
                        depth=d,
                        mode="hybrid",
                        trial=trial,
                        time_ms=-1,
                        final_rank=-1,
                        error_message=str(e),
                    )
                    results.append(result)
                    self.results.append(result)
                    self._log(f"Trial {trial+1} FAILED: {e}", indent=2)
        
        return results
    
    def run_memory_benchmarks(self) -> List[BenchmarkResult]:
        """Category 5: Memory profiling benchmarks.
        
        Tracks peak memory usage across different qubit counts.
        Validates that LRET keeps memory bounded via truncation.
        
        Returns:
            List of benchmark results for this category
        """
        self._log("\n" + "=" * 60)
        self._log("RUNNING MEMORY PROFILING BENCHMARKS")
        self._log("=" * 60)
        
        results = []
        
        for n in self.config.memory_qubits:
            self._log(f"n={n} qubits...", indent=1)
            
            for trial in range(self.config.trials_memory):
                try:
                    # Request memory profiling output
                    metrics = self.run_single_benchmark(
                        n=n, 
                        d=self.config.memory_depth, 
                        mode="hybrid",
                        extra_args=["--profile-memory"] if self._supports_memory_profiling() else None,
                    )
                    
                    result = BenchmarkResult(
                        category="memory",
                        n_qubits=n,
                        depth=self.config.memory_depth,
                        mode="hybrid",
                        trial=trial,
                        time_ms=metrics["time_ms"],
                        final_rank=metrics.get("final_rank", -1),
                        memory_mb=metrics.get("memory_mb"),
                    )
                    results.append(result)
                    self.results.append(result)
                    
                    mem_str = f"{metrics.get('memory_mb', 'N/A'):.1f} MB" if metrics.get('memory_mb') else "N/A"
                    self._log(f"Trial {trial+1}: {mem_str}", indent=2)
                    
                except Exception as e:
                    result = BenchmarkResult(
                        category="memory",
                        n_qubits=n,
                        depth=self.config.memory_depth,
                        mode="hybrid",
                        trial=trial,
                        time_ms=-1,
                        final_rank=-1,
                        error_message=str(e),
                    )
                    results.append(result)
                    self.results.append(result)
                    self._log(f"Trial {trial+1} FAILED: {e}", indent=2)
        
        return results
    
    def _supports_memory_profiling(self) -> bool:
        """Check if quantum_sim supports memory profiling flag."""
        try:
            result = subprocess.run(
                [str(self.quantum_sim), "--help"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return "--profile-memory" in result.stdout
        except:
            return False
    
    # -------------------------------------------------------------------------
    # Run All Categories
    # -------------------------------------------------------------------------
    
    def run_all(self, categories: Optional[List[str]] = None) -> List[BenchmarkResult]:
        """Run all requested benchmark categories.
        
        Args:
            categories: List of categories to run (default: all)
                       Options: scaling, parallel, accuracy, depth, memory
        
        Returns:
            List of all benchmark results
        """
        if categories is None:
            categories = ["scaling", "parallel", "accuracy", "depth", "memory"]
        
        self._log(f"\n{'=' * 60}")
        self._log("LRET BENCHMARK SUITE")
        self._log(f"{'=' * 60}")
        self._log(f"Executable: {self.quantum_sim}")
        self._log(f"Output dir: {self.output_dir}")
        self._log(f"Categories: {', '.join(categories)}")
        self._log(f"Start time: {self.start_time.isoformat()}")
        
        for category in categories:
            if category == "scaling":
                self.run_scaling_benchmarks()
            elif category == "parallel":
                self.run_parallel_benchmarks()
            elif category == "accuracy":
                self.run_accuracy_benchmarks()
            elif category == "depth":
                self.run_depth_benchmarks()
            elif category == "memory":
                self.run_memory_benchmarks()
            else:
                self._log(f"Unknown category: {category}")
        
        return self.results
    
    # -------------------------------------------------------------------------
    # Output Generation
    # -------------------------------------------------------------------------
    
    def save_results(self, csv_filename: str = "benchmark_results.csv"):
        """Save all results to CSV and JSON files.
        
        Args:
            csv_filename: Name for the CSV output file
        """
        if not self.results:
            self._log("No results to save.")
            return
        
        csv_path = self.output_dir / csv_filename
        json_path = self.output_dir / "benchmark_summary.json"
        
        # CSV output
        fieldnames = list(self.results[0].to_dict().keys())
        
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in self.results:
                writer.writerow(result.to_dict())
        
        self._log(f"\nResults saved to: {csv_path}")
        
        # JSON summary
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        successful = [r for r in self.results if r.is_success()]
        failed = [r for r in self.results if not r.is_success()]
        
        summary = {
            "metadata": {
                "timestamp": end_time.isoformat(),
                "duration_seconds": duration,
                "executable": str(self.quantum_sim),
                "platform": platform.platform(),
                "python_version": platform.python_version(),
            },
            "statistics": {
                "total_benchmarks": len(self.results),
                "successful": len(successful),
                "failed": len(failed),
                "categories": list(set(r.category for r in self.results)),
            },
            "results": [r.to_dict() for r in self.results],
        }
        
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        
        self._log(f"Summary saved to: {json_path}")
    
    def print_summary(self):
        """Print quick summary statistics to console."""
        if not self.results:
            self._log("No results to summarize.")
            return
        
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)
        
        # Group by category
        by_category: Dict[str, List[BenchmarkResult]] = {}
        for r in self.results:
            if r.category not in by_category:
                by_category[r.category] = []
            by_category[r.category].append(r)
        
        for category, results in sorted(by_category.items()):
            print(f"\n{category.upper()}:")
            
            successful = [r for r in results if r.is_success()]
            failed = [r for r in results if not r.is_success()]
            
            print(f"  Total runs: {len(results)} ({len(successful)} successful, {len(failed)} failed)")
            
            if successful:
                times = [r.time_ms for r in successful]
                print(f"  Time range: {min(times):.1f} - {max(times):.1f} ms")
                print(f"  Mean time: {sum(times)/len(times):.1f} ms")
                
                if category == "parallel" and len(set(r.mode for r in successful)) > 1:
                    # Compute speedups
                    by_mode = {}
                    for r in successful:
                        if r.mode not in by_mode:
                            by_mode[r.mode] = []
                        by_mode[r.mode].append(r.time_ms)
                    
                    if "sequential" in by_mode:
                        seq_mean = sum(by_mode["sequential"]) / len(by_mode["sequential"])
                        print(f"  Speedups vs sequential:")
                        for mode, times in sorted(by_mode.items()):
                            mean = sum(times) / len(times)
                            speedup = seq_mean / mean
                            print(f"    {mode}: {speedup:.2f}x")
                
                if category == "accuracy":
                    fidelities = [r.fidelity for r in successful if r.fidelity is not None]
                    if fidelities:
                        print(f"  Fidelity range: {min(fidelities):.6f} - {max(fidelities):.6f}")
                        print(f"  Mean fidelity: {sum(fidelities)/len(fidelities):.6f}")
        
        # Overall stats
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        print(f"\n{'=' * 60}")
        print(f"Total benchmarks: {len(self.results)}")
        print(f"Total duration: {duration:.1f} seconds")
        print(f"Output directory: {self.output_dir}")
        print("=" * 60)


def main():
    """Main entry point for benchmark suite CLI."""
    parser = argparse.ArgumentParser(
        description="LRET Quantum Simulator Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark_suite.py                           # Run all benchmarks
  python benchmark_suite.py --quick                   # Quick CI mode
  python benchmark_suite.py --categories scaling      # Only scaling benchmarks
  python benchmark_suite.py --categories scaling,parallel
  python benchmark_suite.py --output-dir results/
        """
    )
    
    parser.add_argument(
        "--quantum-sim",
        type=Path,
        default=Path("build/quantum_sim"),
        help="Path to quantum_sim executable (default: build/quantum_sim)",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.csv",
        help="Output CSV filename (default: benchmark_results.csv)",
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_output"),
        help="Output directory for all artifacts (default: benchmark_output/)",
    )
    
    parser.add_argument(
        "--categories",
        type=str,
        default="all",
        help="Comma-separated list of categories: scaling,parallel,accuracy,depth,memory (default: all)",
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick subset for CI (fewer trials, smaller ranges)",
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    
    args = parser.parse_args()
    
    # Validate executable exists
    quantum_sim = args.quantum_sim
    if not quantum_sim.exists():
        # Try alternate locations
        alt_paths = [
            Path("./quantum_sim"),
            Path("./build/Release/quantum_sim"),
            Path("./build/Debug/quantum_sim"),
            Path("../build/quantum_sim"),
        ]
        for alt in alt_paths:
            if alt.exists():
                quantum_sim = alt
                break
        else:
            print(f"Error: quantum_sim not found at {args.quantum_sim}")
            print("Please build the project first or specify --quantum-sim path")
            sys.exit(1)
    
    # Create configuration
    config = BenchmarkConfig.quick_mode() if args.quick else BenchmarkConfig()
    
    # Create runner
    runner = BenchmarkRunner(
        quantum_sim_path=quantum_sim,
        output_dir=args.output_dir,
        config=config,
        verbose=not args.quiet,
    )
    
    # Determine categories to run
    if args.categories == "all":
        categories = ["scaling", "parallel", "accuracy", "depth", "memory"]
    else:
        categories = [c.strip() for c in args.categories.split(",")]
    
    # Run benchmarks
    print(f"\nStarting LRET benchmark suite...")
    if args.quick:
        print("(Quick mode enabled - reduced trials and ranges)")
    
    try:
        runner.run_all(categories=categories)
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user.")
    
    # Save results
    runner.save_results(csv_filename=args.output)
    
    # Print summary
    runner.print_summary()
    
    print("\n✅ Benchmark suite complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

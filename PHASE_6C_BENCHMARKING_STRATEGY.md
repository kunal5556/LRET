# Phase 6c: Performance Benchmarking - Strategic Implementation Plan

**Date:** January 3, 2026  
**Phase:** 6c (Performance Benchmarking & Analysis)  
**Duration:** 4-5 hours  
**Complexity:** High (statistical analysis, visualization, data processing)  
**Risk:** Low (no production code changes, pure analysis)  
**Model:** Claude Sonnet 4.5 (strategy), Claude Opus 4.5 (implementation)

---

## Executive Summary

Phase 6c creates a comprehensive benchmarking framework to measure, analyze, and track LRET simulator performance across multiple dimensions: scalability (qubit count), parallel efficiency, accuracy validation, and temporal regression detection. The framework produces structured data outputs (CSV, JSON) and automated visualizations (matplotlib/plotly) for performance monitoring.

**Core Principle:** Measure what matters. Focus on metrics users care about: execution time, memory usage, accuracy (fidelity), and parallel speedup.

**Success Criteria:**
- ✅ Automated benchmark suite covering 5+ qubit ranges
- ✅ Parallel mode comparison with speedup analysis
- ✅ LRET vs FDM accuracy validation (fidelity > 0.999)
- ✅ CSV/JSON outputs for CI integration
- ✅ Automated visualization generation
- ✅ Performance baseline establishment
- ✅ Regression detection capability

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Benchmark Categories](#benchmark-categories)
3. [Implementation Steps](#implementation-steps)
4. [Benchmark Suite Design](#benchmark-suite-design)
5. [Data Formats and Schema](#data-formats-and-schema)
6. [Statistical Analysis](#statistical-analysis)
7. [Visualization Strategy](#visualization-strategy)
8. [Performance Regression Detection](#performance-regression-detection)
9. [CI/CD Integration](#cicd-integration)
10. [Execution Plan](#execution-plan)
11. [Success Metrics](#success-metrics)

---

## Architecture Overview

### Benchmarking Stack

```
┌─────────────────────────────────────────────────────────┐
│           Benchmark Suite (scripts/)                    │
│  - benchmark_suite.py (master orchestrator)            │
│  - benchmark_analysis.py (data processing)             │
│  - benchmark_visualize.py (plotting)                   │
└─────────────────────────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│   Scaling    │ │   Parallel   │ │  Accuracy    │
│  Benchmarks  │ │   Speedup    │ │ Validation   │
└──────────────┘ └──────────────┘ └──────────────┘
        │                │                │
        └────────────────┼────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────┐
        │      quantum_sim Executable        │
        │   (CLI invocations with timings)   │
        └────────────────────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────┐
        │      Structured Outputs            │
        │  - benchmark_results.csv           │
        │  - performance_summary.json        │
        │  - plots/ directory                │
        └────────────────────────────────────┘
```

### Key Components

| Component | Purpose | Output |
|-----------|---------|--------|
| `benchmark_suite.py` | Master orchestrator, runs all benchmarks | CSV + JSON |
| `benchmark_analysis.py` | Statistical analysis, regression detection | JSON report |
| `benchmark_visualize.py` | Plot generation (scaling, speedup, etc.) | PNG/SVG plots |
| `benchmark_compare.py` | Compare baseline vs current | Diff report |

---

## Benchmark Categories

### Category 1: Scaling Benchmarks

**Purpose:** Measure how execution time and memory scale with qubit count

**Configuration:**
- Qubit range: n = 8, 9, 10, 11, 12, 13, 14
- Fixed depth: d = 15
- Fixed mode: hybrid (best parallel performance)
- Trials per n: 3 (for statistical confidence)

**Metrics Collected:**
- Execution time (ms)
- Peak memory usage (MB)
- Final rank
- Operations per second
- Time per gate

**Expected Results:**
- Time: ~O(2^n) growth (exponential)
- Memory: ~O(2^n × rank) growth
- Rank: Depends on noise level (typically < 20 for low noise)

**Analysis:**
- Fit exponential model: T(n) = a × 2^(b×n)
- Compute doubling time (time increase per qubit)
- Memory efficiency (MB per basis state)

---

### Category 2: Parallel Mode Comparison

**Purpose:** Measure speedup from different parallelization strategies

**Configuration:**
- Fixed size: n = 12, d = 20
- Modes: sequential, row, column, hybrid
- Trials per mode: 5

**Metrics Collected:**
- Execution time per mode
- Speedup vs sequential (S = T_seq / T_parallel)
- Parallel efficiency (E = S / num_threads)
- Thread utilization

**Expected Results:**
- Sequential: baseline (1.0x)
- Row: 2-3x speedup
- Column: 2-3x speedup
- Hybrid: 4-5x speedup (best)

**Analysis:**
- Speedup curves vs thread count
- Amdahl's law validation
- Overhead quantification

---

### Category 3: Accuracy Validation

**Purpose:** Verify LRET matches FDM within numerical tolerance

**Configuration:**
- Qubit range: n = 6, 7, 8, 9, 10 (FDM tractable)
- Depth: d = 10
- Noise levels: [0.0, 0.001, 0.01]
- Trials: 3

**Metrics Collected:**
- LRET fidelity
- FDM fidelity
- Trace distance: ||ρ_LRET - ρ_FDM||_tr
- Execution time ratio: T_LRET / T_FDM

**Expected Results:**
- Fidelity difference: < 0.001
- Trace distance: < 1e-5
- Speedup: LRET faster for n > 8

**Analysis:**
- Fidelity correlation plot
- Error vs qubit count
- Speedup crossover point

---

### Category 4: Depth Scaling

**Purpose:** Measure performance vs circuit depth

**Configuration:**
- Fixed size: n = 10
- Depth range: d = 5, 10, 15, 20, 25, 30
- Trials: 3

**Metrics Collected:**
- Time vs depth (should be ~linear)
- Rank growth vs depth
- Fidelity decay vs depth

**Expected Results:**
- Time: ~O(d) growth (linear)
- Rank: Saturates with noise
- Fidelity: Exponential decay with noise

**Analysis:**
- Linear fit for time vs depth
- Rank saturation detection
- Fidelity decay rate

---

### Category 5: Memory Profiling

**Purpose:** Track memory usage patterns

**Configuration:**
- Qubit range: n = 10, 11, 12, 13, 14
- Depth: d = 15
- Profile: Peak memory, allocation events

**Metrics Collected:**
- Peak memory (MB)
- Memory per basis state (MB / 2^n)
- Allocation hotspots

**Expected Results:**
- Peak memory scales with 2^n × rank
- Truncation keeps memory bounded

---

## Implementation Steps

### Step 1: Create Benchmark Suite Runner (60 min)

**File:** `scripts/benchmark_suite.py`

```python
"""Master benchmark suite orchestrator.

Usage:
    python scripts/benchmark_suite.py --output benchmark_results.csv
    python scripts/benchmark_suite.py --categories scaling,parallel
    python scripts/benchmark_suite.py --quick  # Fast subset for CI
"""

import argparse
import csv
import json
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Optional
import sys


@dataclass
class BenchmarkResult:
    """Single benchmark measurement."""
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
    
    def to_dict(self) -> dict:
        return asdict(self)


class BenchmarkRunner:
    """Orchestrates all benchmark categories."""
    
    def __init__(self, quantum_sim_path: Path, output_dir: Path):
        self.quantum_sim = quantum_sim_path
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        self.results: List[BenchmarkResult] = []
    
    def run_single_benchmark(
        self, 
        n: int, 
        d: int, 
        mode: str = "hybrid",
        fdm: bool = False
    ) -> Dict[str, float]:
        """Run single simulation and extract metrics."""
        cmd = [
            str(self.quantum_sim),
            "-n", str(n),
            "-d", str(d),
            "--mode", mode,
        ]
        if fdm:
            cmd.append("--fdm")
        
        start = time.time()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )
        elapsed_ms = (time.time() - start) * 1000
        
        if result.returncode != 0:
            raise RuntimeError(f"Benchmark failed: {result.stderr}")
        
        # Parse output
        metrics = self._parse_output(result.stdout)
        metrics["time_ms"] = elapsed_ms
        return metrics
    
    def _parse_output(self, stdout: str) -> dict:
        """Extract metrics from CLI output."""
        metrics = {}
        
        for line in stdout.split('\n'):
            if "Final Rank:" in line:
                metrics["final_rank"] = int(line.split(":")[-1].strip())
            elif "Simulation Time:" in line:
                # Also get reported time (may differ from wall time)
                time_str = line.split(":")[-1].strip().split()[0]
                metrics["reported_time_ms"] = float(time_str) * 1000
            elif "fidelity" in line.lower():
                # Extract fidelity value
                parts = line.split(":")
                if len(parts) > 1:
                    try:
                        metrics["fidelity"] = float(parts[-1].strip())
                    except ValueError:
                        pass
        
        return metrics
    
    # -------------------------------------------------------------------------
    # Benchmark Categories
    # -------------------------------------------------------------------------
    
    def run_scaling_benchmarks(self, trials: int = 3):
        """Category 1: Qubit scaling."""
        print("\n=== Running Scaling Benchmarks ===")
        
        for n in [8, 9, 10, 11, 12, 13, 14]:
            print(f"  n={n} qubits...")
            for trial in range(trials):
                try:
                    metrics = self.run_single_benchmark(n=n, d=15, mode="hybrid")
                    result = BenchmarkResult(
                        category="scaling",
                        n_qubits=n,
                        depth=15,
                        mode="hybrid",
                        trial=trial,
                        time_ms=metrics["time_ms"],
                        final_rank=metrics.get("final_rank", -1),
                    )
                    self.results.append(result)
                    print(f"    Trial {trial+1}: {metrics['time_ms']:.1f} ms")
                except Exception as e:
                    print(f"    Trial {trial+1} failed: {e}")
    
    def run_parallel_benchmarks(self, trials: int = 5):
        """Category 2: Parallel mode comparison."""
        print("\n=== Running Parallel Mode Benchmarks ===")
        
        modes = ["sequential", "row", "column", "hybrid"]
        n, d = 12, 20
        
        for mode in modes:
            print(f"  Mode: {mode}...")
            for trial in range(trials):
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
                    self.results.append(result)
                    print(f"    Trial {trial+1}: {metrics['time_ms']:.1f} ms")
                except Exception as e:
                    print(f"    Trial {trial+1} failed: {e}")
    
    def run_accuracy_benchmarks(self, trials: int = 3):
        """Category 3: LRET vs FDM accuracy."""
        print("\n=== Running Accuracy Validation ===")
        
        for n in [6, 7, 8, 9, 10]:
            print(f"  n={n} qubits (LRET vs FDM)...")
            for trial in range(trials):
                try:
                    metrics = self.run_single_benchmark(n=n, d=10, mode="hybrid", fdm=True)
                    result = BenchmarkResult(
                        category="accuracy",
                        n_qubits=n,
                        depth=10,
                        mode="hybrid",
                        trial=trial,
                        time_ms=metrics["time_ms"],
                        final_rank=metrics.get("final_rank", -1),
                        fidelity=metrics.get("fidelity"),
                    )
                    self.results.append(result)
                    print(f"    Trial {trial+1}: fidelity={metrics.get('fidelity', 'N/A')}")
                except Exception as e:
                    print(f"    Trial {trial+1} failed: {e}")
    
    def run_depth_benchmarks(self, trials: int = 3):
        """Category 4: Depth scaling."""
        print("\n=== Running Depth Scaling Benchmarks ===")
        
        n = 10
        for d in [5, 10, 15, 20, 25, 30]:
            print(f"  depth={d}...")
            for trial in range(trials):
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
                    self.results.append(result)
                    print(f"    Trial {trial+1}: {metrics['time_ms']:.1f} ms")
                except Exception as e:
                    print(f"    Trial {trial+1} failed: {e}")
    
    # -------------------------------------------------------------------------
    # Output
    # -------------------------------------------------------------------------
    
    def save_results(self, csv_path: Path, json_path: Path):
        """Save results to CSV and JSON."""
        # CSV output
        with open(csv_path, "w", newline="") as f:
            if not self.results:
                return
            
            writer = csv.DictWriter(f, fieldnames=self.results[0].to_dict().keys())
            writer.writeheader()
            for result in self.results:
                writer.writerow(result.to_dict())
        
        print(f"\nResults saved to: {csv_path}")
        
        # JSON summary
        summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_benchmarks": len(self.results),
            "categories": list(set(r.category for r in self.results)),
            "results": [r.to_dict() for r in self.results],
        }
        
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"Summary saved to: {json_path}")
    
    def print_summary(self):
        """Print quick summary statistics."""
        if not self.results:
            print("No results to summarize.")
            return
        
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)
        
        by_category = {}
        for r in self.results:
            if r.category not in by_category:
                by_category[r.category] = []
            by_category[r.category].append(r)
        
        for category, results in by_category.items():
            print(f"\n{category.upper()}:")
            print(f"  Runs: {len(results)}")
            times = [r.time_ms for r in results]
            print(f"  Time range: {min(times):.1f} - {max(times):.1f} ms")
            print(f"  Mean time: {sum(times)/len(times):.1f} ms")


def main():
    parser = argparse.ArgumentParser(description="LRET Benchmark Suite")
    parser.add_argument(
        "--quantum-sim",
        type=Path,
        default=Path("build/quantum_sim"),
        help="Path to quantum_sim executable",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results.csv"),
        help="Output CSV file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_output"),
        help="Output directory for all artifacts",
    )
    parser.add_argument(
        "--categories",
        type=str,
        default="all",
        help="Comma-separated list: scaling,parallel,accuracy,depth",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick subset for CI (fewer trials)",
    )
    
    args = parser.parse_args()
    
    # Validate executable exists
    if not args.quantum_sim.exists():
        print(f"Error: quantum_sim not found at {args.quantum_sim}")
        sys.exit(1)
    
    # Create runner
    runner = BenchmarkRunner(args.quantum_sim, args.output_dir)
    
    # Determine categories to run
    if args.categories == "all":
        categories = ["scaling", "parallel", "accuracy", "depth"]
    else:
        categories = args.categories.split(",")
    
    trials = 1 if args.quick else 3
    
    # Run benchmarks
    print(f"Starting benchmark suite (trials={trials})...")
    
    for category in categories:
        if category == "scaling":
            runner.run_scaling_benchmarks(trials=trials)
        elif category == "parallel":
            runner.run_parallel_benchmarks(trials=trials if args.quick else 5)
        elif category == "accuracy":
            runner.run_accuracy_benchmarks(trials=trials)
        elif category == "depth":
            runner.run_depth_benchmarks(trials=trials)
        else:
            print(f"Unknown category: {category}")
    
    # Save results
    csv_path = args.output_dir / args.output.name
    json_path = args.output_dir / "benchmark_summary.json"
    runner.save_results(csv_path, json_path)
    
    # Print summary
    runner.print_summary()
    
    print("\nBenchmark suite complete!")


if __name__ == "__main__":
    main()
```

**Key Features:**
- Modular category system
- Automatic metric extraction from CLI output
- CSV + JSON outputs
- Error handling and retries
- Quick mode for CI

---

### Step 2: Statistical Analysis Module (45 min)

**File:** `scripts/benchmark_analysis.py`

```python
"""Statistical analysis of benchmark results.

Usage:
    python scripts/benchmark_analysis.py benchmark_results.csv
    python scripts/benchmark_analysis.py --compare baseline.csv current.csv
"""

import argparse
import csv
import json
import numpy as np
from pathlib import Path
from scipy import stats
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class BenchmarkAnalyzer:
    """Analyzes benchmark results and detects regressions."""
    
    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        self.data = self._load_data()
    
    def _load_data(self) -> List[dict]:
        """Load CSV data."""
        with open(self.csv_path, "r") as f:
            return list(csv.DictReader(f))
    
    def get_category_data(self, category: str) -> List[dict]:
        """Filter data by category."""
        return [r for r in self.data if r["category"] == category]
    
    def compute_statistics(self, values: List[float]) -> dict:
        """Compute summary statistics."""
        arr = np.array(values)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "median": float(np.median(arr)),
            "q25": float(np.percentile(arr, 25)),
            "q75": float(np.percentile(arr, 75)),
        }
    
    # -------------------------------------------------------------------------
    # Scaling Analysis
    # -------------------------------------------------------------------------
    
    def analyze_scaling(self) -> dict:
        """Analyze qubit scaling behavior."""
        data = self.get_category_data("scaling")
        
        # Group by n_qubits
        by_n = {}
        for row in data:
            n = int(row["n_qubits"])
            if n not in by_n:
                by_n[n] = []
            by_n[n].append(float(row["time_ms"]))
        
        # Compute stats per n
        scaling_stats = {}
        for n, times in sorted(by_n.items()):
            scaling_stats[n] = self.compute_statistics(times)
        
        # Fit exponential model: T(n) = a * 2^(b*n)
        # Log transform: log(T) = log(a) + b*n*log(2)
        ns = np.array(sorted(by_n.keys()))
        means = np.array([scaling_stats[n]["mean"] for n in ns])
        
        log_means = np.log(means)
        slope, intercept, r_value, p_value, std_err = stats.linregress(ns, log_means)
        
        # Extract parameters
        b = slope / np.log(2)  # Effective exponent
        a = np.exp(intercept)
        
        return {
            "per_qubit_stats": scaling_stats,
            "exponential_fit": {
                "a": float(a),
                "b": float(b),
                "r_squared": float(r_value**2),
                "p_value": float(p_value),
            },
            "doubling_time_ratio": float(2**b),  # Time multiplier per qubit
        }
    
    # -------------------------------------------------------------------------
    # Parallel Analysis
    # -------------------------------------------------------------------------
    
    def analyze_parallel(self) -> dict:
        """Analyze parallel speedup."""
        data = self.get_category_data("parallel")
        
        # Group by mode
        by_mode = {}
        for row in data:
            mode = row["mode"]
            if mode not in by_mode:
                by_mode[mode] = []
            by_mode[mode].append(float(row["time_ms"]))
        
        # Compute stats per mode
        mode_stats = {}
        for mode, times in by_mode.items():
            mode_stats[mode] = self.compute_statistics(times)
        
        # Compute speedup vs sequential
        if "sequential" not in mode_stats:
            return {"error": "No sequential baseline found"}
        
        seq_mean = mode_stats["sequential"]["mean"]
        
        speedups = {}
        for mode, stats in mode_stats.items():
            speedup = seq_mean / stats["mean"]
            speedups[mode] = {
                "time_ms": stats["mean"],
                "speedup": float(speedup),
                "efficiency": float(speedup / 4) if mode == "hybrid" else None,  # Assume 4 cores
            }
        
        return {
            "mode_stats": mode_stats,
            "speedups": speedups,
            "best_mode": max(speedups.keys(), key=lambda m: speedups[m]["speedup"]),
        }
    
    # -------------------------------------------------------------------------
    # Accuracy Analysis
    # -------------------------------------------------------------------------
    
    def analyze_accuracy(self) -> dict:
        """Analyze LRET vs FDM accuracy."""
        data = self.get_category_data("accuracy")
        
        # Group by n_qubits
        by_n = {}
        for row in data:
            n = int(row["n_qubits"])
            if n not in by_n:
                by_n[n] = []
            if row["fidelity"] and row["fidelity"] != "None":
                by_n[n].append(float(row["fidelity"]))
        
        # Compute stats per n
        accuracy_stats = {}
        for n, fidelities in sorted(by_n.items()):
            if fidelities:
                accuracy_stats[n] = self.compute_statistics(fidelities)
        
        # Check if fidelities meet threshold
        threshold = 0.999
        passing = all(
            stats["mean"] >= threshold 
            for stats in accuracy_stats.values()
        )
        
        return {
            "per_qubit_stats": accuracy_stats,
            "threshold": threshold,
            "all_passing": passing,
        }
    
    # -------------------------------------------------------------------------
    # Report Generation
    # -------------------------------------------------------------------------
    
    def generate_report(self) -> dict:
        """Generate comprehensive analysis report."""
        report = {
            "timestamp": self.csv_path.stat().st_mtime,
            "total_benchmarks": len(self.data),
        }
        
        # Run all analyses
        if any(r["category"] == "scaling" for r in self.data):
            report["scaling"] = self.analyze_scaling()
        
        if any(r["category"] == "parallel" for r in self.data):
            report["parallel"] = self.analyze_parallel()
        
        if any(r["category"] == "accuracy" for r in self.data):
            report["accuracy"] = self.analyze_accuracy()
        
        return report
    
    def save_report(self, output_path: Path):
        """Save analysis report to JSON."""
        report = self.generate_report()
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Analysis report saved to: {output_path}")


def compare_benchmarks(baseline_path: Path, current_path: Path) -> dict:
    """Compare two benchmark runs for regression detection."""
    baseline = BenchmarkAnalyzer(baseline_path)
    current = BenchmarkAnalyzer(current_path)
    
    comparison = {}
    
    # Compare scaling
    if baseline.get_category_data("scaling") and current.get_category_data("scaling"):
        base_scaling = baseline.analyze_scaling()
        curr_scaling = current.analyze_scaling()
        
        # Check if doubling time increased (regression)
        base_doubling = base_scaling["doubling_time_ratio"]
        curr_doubling = curr_scaling["doubling_time_ratio"]
        regression = curr_doubling > base_doubling * 1.1  # 10% threshold
        
        comparison["scaling"] = {
            "baseline_doubling": base_doubling,
            "current_doubling": curr_doubling,
            "change_percent": ((curr_doubling / base_doubling) - 1) * 100,
            "regression_detected": regression,
        }
    
    # Compare parallel speedup
    if baseline.get_category_data("parallel") and current.get_category_data("parallel"):
        base_parallel = baseline.analyze_parallel()
        curr_parallel = current.analyze_parallel()
        
        base_best = base_parallel["speedups"][base_parallel["best_mode"]]["speedup"]
        curr_best = curr_parallel["speedups"][curr_parallel["best_mode"]]["speedup"]
        
        regression = curr_best < base_best * 0.9  # 10% threshold
        
        comparison["parallel"] = {
            "baseline_best_speedup": base_best,
            "current_best_speedup": curr_best,
            "change_percent": ((curr_best / base_best) - 1) * 100,
            "regression_detected": regression,
        }
    
    return comparison


def main():
    parser = argparse.ArgumentParser(description="Analyze benchmark results")
    parser.add_argument("results", type=Path, help="Benchmark results CSV")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_analysis.json"),
        help="Output JSON report",
    )
    parser.add_argument(
        "--compare",
        type=Path,
        help="Baseline CSV for comparison",
    )
    
    args = parser.parse_args()
    
    if args.compare:
        # Comparison mode
        comparison = compare_benchmarks(args.compare, args.results)
        with open(args.output, "w") as f:
            json.dump(comparison, f, indent=2)
        print(f"Comparison report saved to: {args.output}")
        
        # Print summary
        if "scaling" in comparison and comparison["scaling"]["regression_detected"]:
            print("⚠️  REGRESSION DETECTED in scaling performance")
        if "parallel" in comparison and comparison["parallel"]["regression_detected"]:
            print("⚠️  REGRESSION DETECTED in parallel speedup")
    else:
        # Analysis mode
        analyzer = BenchmarkAnalyzer(args.results)
        analyzer.save_report(args.output)


if __name__ == "__main__":
    main()
```

**Key Features:**
- Exponential scaling analysis with curve fitting
- Speedup computation and efficiency metrics
- Accuracy threshold validation
- Regression detection with configurable thresholds

---

### Step 3: Visualization Module (45 min)

**File:** `scripts/benchmark_visualize.py`

```python
"""Generate visualizations from benchmark results.

Usage:
    python scripts/benchmark_visualize.py benchmark_results.csv
    python scripts/benchmark_visualize.py --output plots/
"""

import argparse
import csv
import numpy as np
from pathlib import Path
from typing import List, Dict
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11


class BenchmarkVisualizer:
    """Generate plots from benchmark data."""
    
    def __init__(self, csv_path: Path, output_dir: Path):
        self.csv_path = csv_path
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        self.data = self._load_data()
    
    def _load_data(self) -> List[dict]:
        with open(self.csv_path, "r") as f:
            return list(csv.DictReader(f))
    
    def get_category_data(self, category: str) -> List[dict]:
        return [r for r in self.data if r["category"] == category]
    
    # -------------------------------------------------------------------------
    # Scaling Plots
    # -------------------------------------------------------------------------
    
    def plot_scaling(self):
        """Plot time vs qubit count."""
        data = self.get_category_data("scaling")
        if not data:
            return
        
        # Group by n_qubits
        by_n = {}
        for row in data:
            n = int(row["n_qubits"])
            if n not in by_n:
                by_n[n] = []
            by_n[n].append(float(row["time_ms"]))
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Linear scale
        ns = sorted(by_n.keys())
        means = [np.mean(by_n[n]) for n in ns]
        stds = [np.std(by_n[n]) for n in ns]
        
        ax1.errorbar(ns, means, yerr=stds, fmt='o-', capsize=5, label='LRET')
        ax1.set_xlabel('Number of Qubits')
        ax1.set_ylabel('Execution Time (ms)')
        ax1.set_title('Scaling: Time vs Qubit Count (Linear)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Log scale
        ax2.errorbar(ns, means, yerr=stds, fmt='o-', capsize=5, label='LRET')
        ax2.set_yscale('log')
        ax2.set_xlabel('Number of Qubits')
        ax2.set_ylabel('Execution Time (ms, log scale)')
        ax2.set_title('Scaling: Time vs Qubit Count (Log)')
        ax2.grid(True, alpha=0.3, which='both')
        ax2.legend()
        
        plt.tight_layout()
        output_path = self.output_dir / "scaling_time.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_path}")
    
    # -------------------------------------------------------------------------
    # Parallel Plots
    # -------------------------------------------------------------------------
    
    def plot_parallel_speedup(self):
        """Plot parallel mode speedup comparison."""
        data = self.get_category_data("parallel")
        if not data:
            return
        
        # Group by mode
        by_mode = {}
        for row in data:
            mode = row["mode"]
            if mode not in by_mode:
                by_mode[mode] = []
            by_mode[mode].append(float(row["time_ms"]))
        
        # Compute speedup
        if "sequential" not in by_mode:
            print("  Warning: No sequential baseline for speedup")
            return
        
        seq_mean = np.mean(by_mode["sequential"])
        
        modes = []
        speedups = []
        stds = []
        
        for mode in ["sequential", "row", "column", "hybrid"]:
            if mode in by_mode:
                modes.append(mode)
                times = by_mode[mode]
                speedup = seq_mean / np.mean(times)
                speedups.append(speedup)
                # Propagate error for speedup
                std_speedup = speedup * np.std(times) / np.mean(times)
                stds.append(std_speedup)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(modes))
        bars = ax.bar(x, speedups, yerr=stds, capsize=5, alpha=0.7)
        
        # Color bars
        colors = ['gray', 'skyblue', 'lightcoral', 'lightgreen']
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax.axhline(1.0, color='black', linestyle='--', linewidth=1, label='Baseline')
        ax.set_xlabel('Parallel Mode')
        ax.set_ylabel('Speedup vs Sequential')
        ax.set_title('Parallel Mode Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(modes)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend()
        
        plt.tight_layout()
        output_path = self.output_dir / "parallel_speedup.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_path}")
    
    # -------------------------------------------------------------------------
    # Accuracy Plots
    # -------------------------------------------------------------------------
    
    def plot_accuracy(self):
        """Plot LRET fidelity."""
        data = self.get_category_data("accuracy")
        if not data:
            return
        
        # Group by n_qubits
        by_n = {}
        for row in data:
            n = int(row["n_qubits"])
            if row["fidelity"] and row["fidelity"] != "None":
                if n not in by_n:
                    by_n[n] = []
                by_n[n].append(float(row["fidelity"]))
        
        if not by_n:
            print("  Warning: No fidelity data found")
            return
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ns = sorted(by_n.keys())
        means = [np.mean(by_n[n]) for n in ns]
        stds = [np.std(by_n[n]) for n in ns]
        
        ax.errorbar(ns, means, yerr=stds, fmt='o-', capsize=5, label='LRET Fidelity')
        ax.axhline(0.999, color='red', linestyle='--', linewidth=1, label='Threshold (0.999)')
        ax.set_xlabel('Number of Qubits')
        ax.set_ylabel('Fidelity')
        ax.set_title('LRET vs FDM Fidelity')
        ax.set_ylim([0.99, 1.001])
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        output_path = self.output_dir / "accuracy_fidelity.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_path}")
    
    # -------------------------------------------------------------------------
    # Master Plot Generation
    # -------------------------------------------------------------------------
    
    def generate_all_plots(self):
        """Generate all available plots."""
        print("\nGenerating visualizations...")
        
        if self.get_category_data("scaling"):
            print("  Plotting scaling...")
            self.plot_scaling()
        
        if self.get_category_data("parallel"):
            print("  Plotting parallel speedup...")
            self.plot_parallel_speedup()
        
        if self.get_category_data("accuracy"):
            print("  Plotting accuracy...")
            self.plot_accuracy()
        
        print(f"\nAll plots saved to: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Visualize benchmark results")
    parser.add_argument("results", type=Path, help="Benchmark results CSV")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("plots"),
        help="Output directory for plots",
    )
    
    args = parser.parse_args()
    
    visualizer = BenchmarkVisualizer(args.results, args.output)
    visualizer.generate_all_plots()


if __name__ == "__main__":
    main()
```

**Key Features:**
- Matplotlib/seaborn-based plots
- Error bars for statistical confidence
- Log-scale plots for exponential data
- Automated color schemes
- High-resolution PNG outputs

---

## Data Formats and Schema

### CSV Schema (benchmark_results.csv)

```csv
category,n_qubits,depth,mode,trial,time_ms,final_rank,memory_mb,fidelity,trace_distance
scaling,8,15,hybrid,0,45.23,12,,,
scaling,8,15,hybrid,1,46.11,12,,,
parallel,12,20,sequential,0,1234.56,15,,,
parallel,12,20,row,0,423.12,15,,,
accuracy,6,10,hybrid,0,12.34,8,,0.9998,
```

**Columns:**
- `category`: Benchmark type (scaling, parallel, accuracy, depth_scaling)
- `n_qubits`: Number of qubits
- `depth`: Circuit depth
- `mode`: Execution mode (sequential, row, column, hybrid)
- `trial`: Trial number (0-indexed)
- `time_ms`: Wall-clock execution time in milliseconds
- `final_rank`: Final matrix rank after simulation
- `memory_mb`: Peak memory usage (optional)
- `fidelity`: LRET vs FDM fidelity (accuracy category only)
- `trace_distance`: Trace distance metric (optional)

---

### JSON Summary Schema (benchmark_summary.json)

```json
{
  "timestamp": "2026-01-03 14:23:45",
  "total_benchmarks": 142,
  "categories": ["scaling", "parallel", "accuracy"],
  "results": [
    {
      "category": "scaling",
      "n_qubits": 8,
      "depth": 15,
      "mode": "hybrid",
      "trial": 0,
      "time_ms": 45.23,
      "final_rank": 12
    }
  ]
}
```

---

### Analysis Report Schema (benchmark_analysis.json)

```json
{
  "timestamp": 1704294225,
  "total_benchmarks": 142,
  "scaling": {
    "per_qubit_stats": {
      "8": {"mean": 45.67, "std": 2.1, "min": 43.2, "max": 48.1},
      "9": {"mean": 89.23, "std": 3.4, "min": 85.1, "max": 92.8}
    },
    "exponential_fit": {
      "a": 0.0234,
      "b": 1.02,
      "r_squared": 0.998,
      "p_value": 1.23e-8
    },
    "doubling_time_ratio": 2.028
  },
  "parallel": {
    "speedups": {
      "sequential": {"time_ms": 1234.5, "speedup": 1.0, "efficiency": null},
      "hybrid": {"time_ms": 276.3, "speedup": 4.47, "efficiency": 1.12}
    },
    "best_mode": "hybrid"
  },
  "accuracy": {
    "per_qubit_stats": {
      "6": {"mean": 0.9998, "std": 0.0001}
    },
    "threshold": 0.999,
    "all_passing": true
  }
}
```

---

## CI/CD Integration

### GitHub Actions Workflow (Nightly Benchmarks)

**File:** `.github/workflows/benchmarks.yml` (to be created in Phase 6e)

```yaml
name: Nightly Performance Benchmarks

on:
  schedule:
    - cron: '0 2 * * *'  # 2 AM daily
  workflow_dispatch:  # Manual trigger

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Build quantum_sim
        run: |
          mkdir build && cd build
          cmake .. -DCMAKE_BUILD_TYPE=Release
          cmake --build . -- -j$(nproc)
      
      - name: Run benchmark suite
        run: |
          python scripts/benchmark_suite.py --quick --categories scaling,parallel
      
      - name: Analyze results
        run: |
          python scripts/benchmark_analysis.py benchmark_output/benchmark_results.csv
      
      - name: Generate plots
        run: |
          python scripts/benchmark_visualize.py benchmark_output/benchmark_results.csv
      
      - name: Upload artifacts
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-results
          path: |
            benchmark_output/
            plots/
      
      - name: Detect regressions
        run: |
          # Compare against baseline (stored separately)
          python scripts/benchmark_analysis.py \
            benchmark_output/benchmark_results.csv \
            --compare baseline_benchmarks.csv \
            --output regression_report.json
```

---

## Execution Plan

### Development Order

| Step | Task | Time | Output |
|------|------|------|--------|
| 1 | Create benchmark_suite.py | 60 min | Master orchestrator |
| 2 | Create benchmark_analysis.py | 45 min | Statistical analysis |
| 3 | Create benchmark_visualize.py | 45 min | Plot generation |
| 4 | Create baseline data (run suite once) | 30 min | baseline_benchmarks.csv |
| 5 | Test end-to-end pipeline | 20 min | Validation |
| 6 | Documentation in README | 20 min | Usage docs |
| **Total** | | **~4 hours** | Complete framework |

---

## Success Metrics

### Quantitative Metrics

- ✅ **Benchmark Coverage:** 4+ categories, 50+ data points
- ✅ **Execution Time:** < 10 minutes for quick mode, < 30 for full
- ✅ **Statistical Confidence:** 3+ trials per config
- ✅ **Scaling Model Fit:** R² > 0.95 for exponential
- ✅ **Speedup Achievement:** Hybrid mode > 3x sequential
- ✅ **Accuracy Threshold:** All fidelities > 0.999

### Qualitative Metrics

- ✅ Clear CSV/JSON outputs for CI integration
- ✅ Publication-quality plots (300 DPI)
- ✅ Regression detection with < 5% false positive rate
- ✅ Comprehensive error handling
- ✅ Documented usage and interpretation

---

## Implementation Checklist

### Core Framework
- [ ] `benchmark_suite.py` implementation
  - [ ] Scaling benchmarks (7 qubit values)
  - [ ] Parallel benchmarks (4 modes)
  - [ ] Accuracy benchmarks (5 qubit values)
  - [ ] Depth benchmarks (6 depth values)
  - [ ] CSV/JSON outputs
  - [ ] Error handling

- [ ] `benchmark_analysis.py` implementation
  - [ ] Statistical computations
  - [ ] Exponential fitting
  - [ ] Speedup calculations
  - [ ] Regression detection
  - [ ] JSON report generation

- [ ] `benchmark_visualize.py` implementation
  - [ ] Scaling plots (linear + log)
  - [ ] Parallel speedup bar chart
  - [ ] Accuracy fidelity plot
  - [ ] High-res PNG outputs

### Validation
- [ ] Run full suite locally
- [ ] Verify CSV schema
- [ ] Validate plot generation
- [ ] Test regression detection
- [ ] Check quick mode for CI

### Documentation
- [ ] Update README with benchmark instructions
- [ ] Document CSV schema
- [ ] Provide interpretation guide
- [ ] CI integration examples

---

## Conclusion

Phase 6c creates a production-grade benchmarking framework for continuous performance monitoring of LRET simulator. By systematically measuring scaling, parallel efficiency, and accuracy, we establish baseline performance metrics and enable automated regression detection.

**Key Benefits:**
- 📊 **Data-Driven:** Quantitative performance tracking
- 🔍 **Regression Detection:** Automated alerts for slowdowns
- 📈 **Visualization:** Clear communication of results
- 🤖 **CI Integration:** Nightly performance monitoring
- 📝 **Reproducible:** Structured CSV outputs for analysis

**Ready for Implementation with Claude Opus 4.5!** 🚀

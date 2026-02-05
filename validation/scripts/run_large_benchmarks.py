#!/usr/bin/env python3
"""
LRET Large-Scale Benchmark Runner - Phase C
Runs comprehensive benchmarks with statistical analysis

Features:
- Multiple trials with statistical analysis (mean, std, CI)
- Speedup scaling analysis by qubit count
- Memory-safe timeout handling
- Progress tracking and resumption
- Detailed CSV and JSON output

Usage:
    python run_large_benchmarks.py --max-qubits 14 --trials 5
    python run_large_benchmarks.py --resume results/20260205_123456
"""

import json
import os
import subprocess
import time
import argparse
import csv
import statistics
import math
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

# =============================================================================
# Configuration
# =============================================================================

BASELINE_EXE = "baseline\\quantum_sim.exe"
OPTIMIZED_EXE = "optimized\\quantum_sim.exe"
GENERATED_DIR = "test_circuits\\large"
RESULTS_DIR = "results"

# Timeout scaling: more qubits = more time allowed
TIMEOUT_PER_QUBIT = 30  # seconds per qubit
BASE_TIMEOUT = 60       # minimum timeout

# =============================================================================
# Statistics Helpers
# =============================================================================

def calculate_ci(data: List[float], confidence: float = 0.95) -> Tuple[float, float]:
    """Calculate confidence interval for mean."""
    if len(data) < 2:
        return (data[0], data[0]) if data else (0, 0)
    
    n = len(data)
    mean = statistics.mean(data)
    std = statistics.stdev(data)
    
    # t-value approximation for 95% CI
    t_values = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57, 7: 2.45, 8: 2.36, 9: 2.31, 10: 2.26}
    t = t_values.get(n, 1.96)  # Use normal distribution for n > 10
    
    margin = t * std / math.sqrt(n)
    return (mean - margin, mean + margin)

def format_time(seconds: float) -> str:
    """Format time nicely."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"

# =============================================================================
# Result Data Structures
# =============================================================================

@dataclass
class TrialResult:
    """Result from a single trial."""
    success: bool
    wall_time: float
    sim_time: Optional[float]
    final_rank: Optional[int]
    error: Optional[str]

@dataclass
class BenchmarkResult:
    """Aggregated benchmark result for one circuit."""
    file: str
    category: str
    subtype: str
    n_qubits: int
    n_operations: int
    cnot_count: int
    trials: int
    
    # Baseline stats
    baseline_mean: float
    baseline_std: float
    baseline_ci_low: float
    baseline_ci_high: float
    baseline_rank: int
    baseline_success_rate: float
    
    # Optimized stats
    optimized_mean: float
    optimized_std: float
    optimized_ci_low: float
    optimized_ci_high: float
    optimized_rank: int
    optimized_success_rate: float
    
    # Comparison
    speedup: float
    speedup_ci_low: float
    speedup_ci_high: float
    ranks_match: bool

# =============================================================================
# Benchmark Runner
# =============================================================================

class LargeBenchmarkRunner:
    """Run comprehensive benchmarks on large circuits."""
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.baseline_exe = self.base_dir / BASELINE_EXE
        self.optimized_exe = self.base_dir / OPTIMIZED_EXE
        self.generated_dir = self.base_dir / GENERATED_DIR
        self.results_dir = self.base_dir / RESULTS_DIR
        
        # Verify executables exist
        if not self.baseline_exe.exists():
            raise FileNotFoundError(f"Baseline exe not found: {self.baseline_exe}")
        if not self.optimized_exe.exists():
            raise FileNotFoundError(f"Optimized exe not found: {self.optimized_exe}")
        
        # Load manifest
        manifest_path = self.generated_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                self.manifest = json.load(f)
        else:
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    
    def get_timeout(self, n_qubits: int) -> int:
        """Calculate timeout based on qubit count."""
        return BASE_TIMEOUT + TIMEOUT_PER_QUBIT * n_qubits
    
    def run_single(self, circuit_path: Path, exe_path: Path, 
                   timeout: int) -> TrialResult:
        """Run a single circuit execution."""
        cmd = [
            str(exe_path),
            "--input-json", str(circuit_path),
            "--allow-swap",
            "--non-interactive"
        ]
        
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.base_dir)
            )
            elapsed = time.time() - start_time
            
            # Parse output
            output = result.stdout + result.stderr
            
            sim_time = None
            final_rank = None
            
            for line in output.split('\n'):
                if 'Time:' in line:
                    try:
                        sim_time = float(line.split(':')[1].strip().replace('s', '').strip())
                    except:
                        pass
                if 'Final Rank:' in line:
                    try:
                        final_rank = int(line.split(':')[1].strip())
                    except:
                        pass
            
            return TrialResult(
                success=result.returncode == 0,
                wall_time=elapsed,
                sim_time=sim_time,
                final_rank=final_rank,
                error=None if result.returncode == 0 else f"returncode={result.returncode}"
            )
            
        except subprocess.TimeoutExpired:
            return TrialResult(
                success=False,
                wall_time=timeout,
                sim_time=None,
                final_rank=None,
                error="timeout"
            )
        except Exception as e:
            return TrialResult(
                success=False,
                wall_time=time.time() - start_time,
                sim_time=None,
                final_rank=None,
                error=str(e)
            )
    
    def run_benchmark(self, max_qubits: int = 14, min_qubits: int = 8,
                      trials: int = 3, subtypes: List[str] = None,
                      resume_from: str = None) -> List[BenchmarkResult]:
        """Run full benchmark suite."""
        
        # Filter circuits
        circuits = [c for c in self.manifest 
                    if min_qubits <= c["n_qubits"] <= max_qubits]
        if subtypes:
            circuits = [c for c in circuits if c["subtype"] in subtypes]
        
        # Sort by qubit count for better progress tracking
        circuits.sort(key=lambda c: (c["n_qubits"], c["subtype"]))
        
        # Resume support
        completed = set()
        results = []
        if resume_from:
            resume_path = Path(resume_from) / "circuit_benchmark.csv"
            if resume_path.exists():
                with open(resume_path) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        completed.add(row["file"])
                        results.append(self._row_to_result(row))
                print(f"Resuming from {len(completed)} completed circuits")
        
        # Calculate ETA
        total = len(circuits)
        remaining = sum(1 for c in circuits if c["file"] not in completed)
        
        print()
        print("=" * 70)
        print("LRET Large-Scale Benchmark Runner - Phase C")
        print("=" * 70)
        print(f"Total circuits: {total}")
        print(f"Remaining: {remaining}")
        print(f"Qubit range: {min_qubits}-{max_qubits}")
        print(f"Trials per circuit: {trials}")
        print(f"Estimated time: {format_time(remaining * trials * 2 * 30)}")
        print("=" * 70)
        print()
        
        start_time = time.time()
        processed = 0
        
        for i, circuit_info in enumerate(circuits):
            if circuit_info["file"] in completed:
                continue
            
            circuit_path = self.generated_dir / circuit_info["file"]
            n_qubits = circuit_info["n_qubits"]
            n_ops = circuit_info["n_operations"]
            subtype = circuit_info["subtype"]
            cnot_count = circuit_info.get("cnot_count", 0)
            timeout = self.get_timeout(n_qubits)
            
            processed += 1
            elapsed = time.time() - start_time
            eta = (elapsed / processed) * (remaining - processed) if processed > 0 else 0
            
            print(f"\n[{processed}/{remaining}] {circuit_info['file']}")
            print(f"  {n_qubits}q, {n_ops} ops, {cnot_count} CNOTs | timeout={timeout}s | ETA: {format_time(eta)}")
            
            # Run trials
            baseline_trials = []
            optimized_trials = []
            
            for t in range(trials):
                # Baseline
                print(f"  Trial {t+1}/{trials}: baseline...", end="", flush=True)
                baseline_result = self.run_single(circuit_path, self.baseline_exe, timeout)
                if baseline_result.success:
                    baseline_trials.append(baseline_result)
                    print(f" {baseline_result.wall_time:.2f}s", end="")
                else:
                    print(f" FAIL ({baseline_result.error})", end="")
                
                # Optimized
                print(" | optimized...", end="", flush=True)
                optimized_result = self.run_single(circuit_path, self.optimized_exe, timeout)
                if optimized_result.success:
                    optimized_trials.append(optimized_result)
                    print(f" {optimized_result.wall_time:.2f}s")
                else:
                    print(f" FAIL ({optimized_result.error})")
            
            # Aggregate results
            if baseline_trials and optimized_trials:
                result = self._aggregate_results(
                    circuit_info, baseline_trials, optimized_trials, trials
                )
                results.append(result)
                
                # Print summary
                color = "\033[92m" if result.speedup > 1.1 else ("\033[91m" if result.speedup < 0.9 else "")
                reset = "\033[0m" if color else ""
                print(f"  {color}Speedup: {result.speedup:.2f}x [{result.speedup_ci_low:.2f}-{result.speedup_ci_high:.2f}]{reset}")
                
                if not result.ranks_match:
                    print(f"  \033[93mWARNING: Rank mismatch! {result.baseline_rank} vs {result.optimized_rank}\033[0m")
            else:
                print("  SKIPPED (not enough successful trials)")
        
        return results
    
    def _aggregate_results(self, circuit_info: Dict, 
                           baseline_trials: List[TrialResult],
                           optimized_trials: List[TrialResult],
                           total_trials: int) -> BenchmarkResult:
        """Aggregate trial results with statistics."""
        
        baseline_times = [t.wall_time for t in baseline_trials]
        optimized_times = [t.wall_time for t in optimized_trials]
        
        baseline_mean = statistics.mean(baseline_times)
        optimized_mean = statistics.mean(optimized_times)
        
        baseline_std = statistics.stdev(baseline_times) if len(baseline_times) > 1 else 0
        optimized_std = statistics.stdev(optimized_times) if len(optimized_times) > 1 else 0
        
        baseline_ci = calculate_ci(baseline_times)
        optimized_ci = calculate_ci(optimized_times)
        
        speedup = baseline_mean / optimized_mean if optimized_mean > 0 else 0
        
        # Speedup confidence interval (pessimistic)
        speedup_ci_low = baseline_ci[0] / optimized_ci[1] if optimized_ci[1] > 0 else 0
        speedup_ci_high = baseline_ci[1] / optimized_ci[0] if optimized_ci[0] > 0 else speedup * 2
        
        baseline_rank = baseline_trials[0].final_rank if baseline_trials and baseline_trials[0].final_rank else 0
        optimized_rank = optimized_trials[0].final_rank if optimized_trials and optimized_trials[0].final_rank else 0
        
        return BenchmarkResult(
            file=circuit_info["file"],
            category=circuit_info["category"],
            subtype=circuit_info["subtype"],
            n_qubits=circuit_info["n_qubits"],
            n_operations=circuit_info["n_operations"],
            cnot_count=circuit_info.get("cnot_count", 0),
            trials=total_trials,
            
            baseline_mean=round(baseline_mean, 4),
            baseline_std=round(baseline_std, 4),
            baseline_ci_low=round(baseline_ci[0], 4),
            baseline_ci_high=round(baseline_ci[1], 4),
            baseline_rank=baseline_rank,
            baseline_success_rate=round(len(baseline_trials) / total_trials, 2),
            
            optimized_mean=round(optimized_mean, 4),
            optimized_std=round(optimized_std, 4),
            optimized_ci_low=round(optimized_ci[0], 4),
            optimized_ci_high=round(optimized_ci[1], 4),
            optimized_rank=optimized_rank,
            optimized_success_rate=round(len(optimized_trials) / total_trials, 2),
            
            speedup=round(speedup, 3),
            speedup_ci_low=round(speedup_ci_low, 3),
            speedup_ci_high=round(speedup_ci_high, 3),
            ranks_match=baseline_rank == optimized_rank
        )
    
    def _row_to_result(self, row: Dict) -> BenchmarkResult:
        """Convert CSV row back to BenchmarkResult."""
        return BenchmarkResult(
            file=row["file"],
            category=row["category"],
            subtype=row["subtype"],
            n_qubits=int(row["n_qubits"]),
            n_operations=int(row["n_operations"]),
            cnot_count=int(row.get("cnot_count", 0)),
            trials=int(row["trials"]),
            baseline_mean=float(row["baseline_mean"]),
            baseline_std=float(row["baseline_std"]),
            baseline_ci_low=float(row["baseline_ci_low"]),
            baseline_ci_high=float(row["baseline_ci_high"]),
            baseline_rank=int(row["baseline_rank"]),
            baseline_success_rate=float(row["baseline_success_rate"]),
            optimized_mean=float(row["optimized_mean"]),
            optimized_std=float(row["optimized_std"]),
            optimized_ci_low=float(row["optimized_ci_low"]),
            optimized_ci_high=float(row["optimized_ci_high"]),
            optimized_rank=int(row["optimized_rank"]),
            optimized_success_rate=float(row["optimized_success_rate"]),
            speedup=float(row["speedup"]),
            speedup_ci_low=float(row["speedup_ci_low"]),
            speedup_ci_high=float(row["speedup_ci_high"]),
            ranks_match=row["ranks_match"] == "True"
        )
    
    def save_results(self, results: List[BenchmarkResult], 
                     output_dir: str = None) -> Path:
        """Save results to CSV and JSON."""
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = self.results_dir / f"large_{timestamp}"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save CSV
        csv_path = output_dir / "circuit_benchmark.csv"
        if results:
            fieldnames = list(asdict(results[0]).keys())
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in results:
                    writer.writerow(asdict(r))
        
        # Save JSON
        json_path = output_dir / "circuit_benchmark.json"
        with open(json_path, 'w') as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        
        print(f"\nResults saved to: {output_dir}")
        return output_dir
    
    def analyze_results(self, results: List[BenchmarkResult]):
        """Comprehensive analysis of results."""
        print("\n")
        print("=" * 70)
        print("                    COMPREHENSIVE BENCHMARK ANALYSIS")
        print("=" * 70)
        
        if not results:
            print("No results to analyze.")
            return
        
        # Overall statistics
        speedups = [r.speedup for r in results]
        avg_speedup = statistics.mean(speedups)
        median_speedup = statistics.median(speedups)
        std_speedup = statistics.stdev(speedups) if len(speedups) > 1 else 0
        
        print(f"\n📊 OVERALL STATISTICS")
        print(f"   Total circuits benchmarked: {len(results)}")
        print(f"   Average speedup: {avg_speedup:.3f}x")
        print(f"   Median speedup: {median_speedup:.3f}x")
        print(f"   Std deviation: {std_speedup:.3f}")
        print(f"   Min speedup: {min(speedups):.3f}x")
        print(f"   Max speedup: {max(speedups):.3f}x")
        
        # Speedup > 1 analysis
        above_1 = sum(1 for s in speedups if s > 1.0)
        above_1_5 = sum(1 for s in speedups if s > 1.5)
        above_2 = sum(1 for s in speedups if s > 2.0)
        print(f"\n   Circuits with speedup > 1.0x: {above_1}/{len(results)} ({100*above_1/len(results):.1f}%)")
        print(f"   Circuits with speedup > 1.5x: {above_1_5}/{len(results)} ({100*above_1_5/len(results):.1f}%)")
        print(f"   Circuits with speedup > 2.0x: {above_2}/{len(results)} ({100*above_2/len(results):.1f}%)")
        
        # By qubit count
        print(f"\n📈 SPEEDUP BY QUBIT COUNT")
        qubit_counts = sorted(set(r.n_qubits for r in results))
        for n in qubit_counts:
            q_results = [r for r in results if r.n_qubits == n]
            q_speedups = [r.speedup for r in q_results]
            q_avg = statistics.mean(q_speedups)
            q_std = statistics.stdev(q_speedups) if len(q_speedups) > 1 else 0
            q_ci = calculate_ci(q_speedups)
            
            bar_len = int(q_avg * 10)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            
            print(f"   {n:2d} qubits: {q_avg:.2f}x ± {q_std:.2f} [{q_ci[0]:.2f}-{q_ci[1]:.2f}] | {bar} | n={len(q_results)}")
        
        # By circuit type
        print(f"\n📋 SPEEDUP BY CIRCUIT TYPE")
        subtypes = sorted(set(r.subtype for r in results))
        type_results = []
        for subtype in subtypes:
            t_results = [r for r in results if r.subtype == subtype]
            t_speedups = [r.speedup for r in t_results]
            t_avg = statistics.mean(t_speedups)
            type_results.append((subtype, t_avg, len(t_results)))
        
        type_results.sort(key=lambda x: -x[1])
        for subtype, avg, count in type_results:
            bar_len = int(avg * 10)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            print(f"   {subtype:20s}: {avg:.2f}x | {bar} | n={count}")
        
        # Rank analysis (proxy for complexity)
        print(f"\n🔢 SPEEDUP VS FINAL RANK")
        ranks = sorted(set(r.baseline_rank for r in results if r.baseline_rank > 0))
        rank_buckets = [(0, 10), (10, 50), (50, 100), (100, 500), (500, 1000), (1000, float('inf'))]
        for low, high in rank_buckets:
            bucket_results = [r for r in results if low <= r.baseline_rank < high]
            if bucket_results:
                bucket_speedups = [r.speedup for r in bucket_results]
                bucket_avg = statistics.mean(bucket_speedups)
                label = f"{low}-{high}" if high != float('inf') else f"{low}+"
                print(f"   Rank {label:>10s}: {bucket_avg:.2f}x (n={len(bucket_results)})")
        
        # Top performers
        print(f"\n🏆 TOP 10 FASTEST SPEEDUPS")
        sorted_results = sorted(results, key=lambda r: -r.speedup)
        for i, r in enumerate(sorted_results[:10]):
            print(f"   {i+1:2d}. {r.speedup:.2f}x - {r.subtype} {r.n_qubits}q ({r.n_operations} ops, rank={r.baseline_rank})")
        
        # Slowest (regressions)
        regressions = [r for r in results if r.speedup < 1.0]
        if regressions:
            print(f"\n⚠️  REGRESSIONS (speedup < 1.0x)")
            for r in sorted(regressions, key=lambda x: x.speedup)[:5]:
                print(f"   {r.speedup:.2f}x - {r.subtype} {r.n_qubits}q ({r.n_operations} ops, rank={r.baseline_rank})")
        
        # Correctness
        mismatches = [r for r in results if not r.ranks_match]
        if mismatches:
            print(f"\n⛔ RANK MISMATCHES: {len(mismatches)}")
            for r in mismatches[:5]:
                print(f"   {r.file}: baseline={r.baseline_rank}, optimized={r.optimized_rank}")
        else:
            print(f"\n✅ CORRECTNESS: All {len(results)} circuits produce matching ranks")
        
        # Time savings
        total_baseline = sum(r.baseline_mean * r.trials for r in results)
        total_optimized = sum(r.optimized_mean * r.trials for r in results)
        saved = total_baseline - total_optimized
        
        print(f"\n⏱️  TIME SAVINGS")
        print(f"   Total baseline time: {format_time(total_baseline)}")
        print(f"   Total optimized time: {format_time(total_optimized)}")
        print(f"   Time saved: {format_time(saved)} ({100*saved/total_baseline:.1f}%)")
        
        print("\n" + "=" * 70)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="LRET Large-Scale Benchmark Runner")
    parser.add_argument("--min-qubits", type=int, default=8,
                        help="Minimum qubit count to test")
    parser.add_argument("--max-qubits", "-q", type=int, default=14,
                        help="Maximum qubit count to test")
    parser.add_argument("--trials", "-t", type=int, default=3,
                        help="Trials per circuit")
    parser.add_argument("--subtypes", nargs="+",
                        help="Only run specific subtypes")
    parser.add_argument("--resume", type=str,
                        help="Resume from existing results directory")
    parser.add_argument("--output", "-o", type=str,
                        help="Output directory for results")
    
    args = parser.parse_args()
    
    try:
        runner = LargeBenchmarkRunner()
        
        results = runner.run_benchmark(
            max_qubits=args.max_qubits,
            min_qubits=args.min_qubits,
            trials=args.trials,
            subtypes=args.subtypes,
            resume_from=args.resume
        )
        
        if results:
            output_dir = runner.save_results(results, args.output)
            runner.analyze_results(results)
        else:
            print("No results collected.")
            
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Make sure you have generated large circuits first:")
        print("  python scripts/generate_large_circuits.py")
        sys.exit(1)


if __name__ == "__main__":
    main()

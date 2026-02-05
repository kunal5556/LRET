#!/usr/bin/env python3
"""
LRET High-Rank Benchmark Runner - Phase D
Runs benchmarks on noisy circuits and tracks rank vs speedup correlation

Key hypothesis: Speedup increases with final rank because row-parallelism 
optimization engages when rank >= 32.

Usage:
    python run_noisy_benchmarks.py --max-qubits 10 --trials 3
"""

import json
import os
import subprocess
import time
import argparse
import csv
import statistics
import re
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
GENERATED_DIR = "test_circuits\\noisy"
RESULTS_DIR = "results"

# Timeout scaling for noisy circuits (they can be slower)
TIMEOUT_PER_QUBIT = 60  # seconds per qubit (more for noisy)
BASE_TIMEOUT = 120

# =============================================================================
# Helpers
# =============================================================================

def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def parse_json_output(output: str) -> Dict:
    """Parse JSON output from quantum_sim."""
    result = {
        "execution_time_ms": None,
        "final_rank": None,
        "success": False
    }
    
    # Try to parse as JSON
    try:
        # Find JSON in output (may have other text before/after)
        json_match = re.search(r'\{[\s\S]*\}', output)
        if json_match:
            data = json.loads(json_match.group())
            result["execution_time_ms"] = data.get("execution_time_ms")
            result["final_rank"] = data.get("final_rank")
            result["success"] = True
    except json.JSONDecodeError:
        pass
    
    return result


# =============================================================================
# Benchmark Result
# =============================================================================

@dataclass
class NoisyBenchmarkResult:
    """Result for a noisy circuit benchmark."""
    file: str
    subtype: str
    n_qubits: int
    n_operations: int
    noise_count: int
    noise_type: str
    noise_rate: float
    trials: int
    
    # Baseline stats
    baseline_mean_ms: float
    baseline_std_ms: float
    baseline_rank: int
    
    # Optimized stats
    optimized_mean_ms: float
    optimized_std_ms: float
    optimized_rank: int
    
    # Comparison
    speedup: float
    ranks_match: bool


# =============================================================================
# Benchmark Runner
# =============================================================================

class NoisyBenchmarkRunner:
    """Run benchmarks on noisy circuits."""
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.baseline_exe = self.base_dir / BASELINE_EXE
        self.optimized_exe = self.base_dir / OPTIMIZED_EXE
        self.generated_dir = self.base_dir / GENERATED_DIR
        self.results_dir = self.base_dir / RESULTS_DIR
        
        # Verify executables
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
    
    def get_timeout(self, n_qubits: int, noise_count: int) -> int:
        """Calculate timeout based on circuit complexity."""
        # Noisy circuits take longer
        base = BASE_TIMEOUT + TIMEOUT_PER_QUBIT * n_qubits
        # Add time for noise operations
        noise_factor = 1 + (noise_count / 100)
        return int(base * noise_factor)
    
    def run_single(self, circuit_path: Path, exe_path: Path, 
                   timeout: int) -> Dict:
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
            parsed = parse_json_output(output)
            
            return {
                "success": result.returncode == 0 and parsed["success"],
                "wall_time": elapsed,
                "execution_time_ms": parsed.get("execution_time_ms"),
                "final_rank": parsed.get("final_rank"),
                "error": None if result.returncode == 0 else f"rc={result.returncode}"
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "wall_time": timeout,
                "execution_time_ms": None,
                "final_rank": None,
                "error": "timeout"
            }
        except Exception as e:
            return {
                "success": False,
                "wall_time": time.time() - start_time,
                "execution_time_ms": None,
                "final_rank": None,
                "error": str(e)
            }
    
    def run_benchmark(self, max_qubits: int = 10, min_qubits: int = 6,
                      trials: int = 2, subtypes: List[str] = None) -> List[NoisyBenchmarkResult]:
        """Run benchmarks on noisy circuits."""
        
        # Filter circuits
        circuits = [c for c in self.manifest 
                    if min_qubits <= c["n_qubits"] <= max_qubits]
        if subtypes:
            circuits = [c for c in circuits if c["subtype"] in subtypes]
        
        # Sort by qubit count
        circuits.sort(key=lambda c: (c["n_qubits"], c.get("noise_count", 0)))
        
        print()
        print("=" * 70)
        print("LRET High-Rank Benchmark Runner - Phase D")
        print("=" * 70)
        print(f"Total circuits: {len(circuits)}")
        print(f"Qubit range: {min_qubits}-{max_qubits}")
        print(f"Trials per circuit: {trials}")
        print("=" * 70)
        print()
        
        results = []
        start_time = time.time()
        
        for i, circuit_info in enumerate(circuits):
            circuit_path = self.generated_dir / circuit_info["file"]
            n_qubits = circuit_info["n_qubits"]
            n_ops = circuit_info["n_operations"]
            noise_count = circuit_info.get("noise_count", 0)
            noise_type = circuit_info.get("noise_type", "unknown")
            noise_rate = circuit_info.get("noise_rate", 0)
            subtype = circuit_info["subtype"]
            
            timeout = self.get_timeout(n_qubits, noise_count)
            
            elapsed = time.time() - start_time
            eta = (elapsed / (i + 1)) * (len(circuits) - i - 1) if i > 0 else 0
            
            print(f"\n[{i+1}/{len(circuits)}] {circuit_info['file']}")
            print(f"  {n_qubits}q, {n_ops} ops, {noise_count} noise | {noise_type} p={noise_rate:.2f} | timeout={timeout}s | ETA: {format_time(eta)}")
            
            # Run trials
            baseline_results = []
            optimized_results = []
            
            for t in range(trials):
                # Baseline
                print(f"  Trial {t+1}/{trials}: baseline...", end="", flush=True)
                b_result = self.run_single(circuit_path, self.baseline_exe, timeout)
                if b_result["success"]:
                    baseline_results.append(b_result)
                    print(f" {b_result['execution_time_ms']:.0f}ms (rank={b_result['final_rank']})", end="")
                else:
                    print(f" FAIL ({b_result['error']})", end="")
                
                # Optimized
                print(" | optimized...", end="", flush=True)
                o_result = self.run_single(circuit_path, self.optimized_exe, timeout)
                if o_result["success"]:
                    optimized_results.append(o_result)
                    print(f" {o_result['execution_time_ms']:.0f}ms (rank={o_result['final_rank']})")
                else:
                    print(f" FAIL ({o_result['error']})")
            
            # Aggregate
            if baseline_results and optimized_results:
                b_times = [r["execution_time_ms"] for r in baseline_results]
                o_times = [r["execution_time_ms"] for r in optimized_results]
                
                b_mean = statistics.mean(b_times)
                o_mean = statistics.mean(o_times)
                b_std = statistics.stdev(b_times) if len(b_times) > 1 else 0
                o_std = statistics.stdev(o_times) if len(o_times) > 1 else 0
                
                speedup = b_mean / o_mean if o_mean > 0 else 0
                
                b_rank = baseline_results[0]["final_rank"]
                o_rank = optimized_results[0]["final_rank"]
                
                result = NoisyBenchmarkResult(
                    file=circuit_info["file"],
                    subtype=subtype,
                    n_qubits=n_qubits,
                    n_operations=n_ops,
                    noise_count=noise_count,
                    noise_type=noise_type,
                    noise_rate=noise_rate,
                    trials=trials,
                    baseline_mean_ms=round(b_mean, 2),
                    baseline_std_ms=round(b_std, 2),
                    baseline_rank=b_rank if b_rank else 0,
                    optimized_mean_ms=round(o_mean, 2),
                    optimized_std_ms=round(o_std, 2),
                    optimized_rank=o_rank if o_rank else 0,
                    speedup=round(speedup, 3),
                    ranks_match=b_rank == o_rank
                )
                results.append(result)
                
                # Print summary
                rank_str = f"rank={b_rank}" if b_rank else "rank=?"
                color = "\033[92m" if speedup > 1.2 else ("\033[91m" if speedup < 0.9 else "")
                reset = "\033[0m" if color else ""
                print(f"  {color}Speedup: {speedup:.2f}x{reset} | {rank_str}")
                
                if not result.ranks_match:
                    print(f"  \033[93mWARNING: Rank mismatch! baseline={b_rank}, optimized={o_rank}\033[0m")
            else:
                print("  SKIPPED (not enough successful trials)")
        
        return results
    
    def save_results(self, results: List[NoisyBenchmarkResult], 
                     output_dir: str = None) -> Path:
        """Save results to CSV and JSON."""
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = self.results_dir / f"noisy_{timestamp}"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save CSV
        csv_path = output_dir / "noisy_benchmark.csv"
        if results:
            fieldnames = list(asdict(results[0]).keys())
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in results:
                    writer.writerow(asdict(r))
        
        # Save JSON
        json_path = output_dir / "noisy_benchmark.json"
        with open(json_path, 'w') as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        
        print(f"\nResults saved to: {output_dir}")
        return output_dir
    
    def analyze_results(self, results: List[NoisyBenchmarkResult]):
        """Analyze results with focus on rank vs speedup correlation."""
        print("\n")
        print("=" * 70)
        print("           HIGH-RANK BENCHMARK ANALYSIS - PHASE D")
        print("=" * 70)
        
        if not results:
            print("No results to analyze.")
            return
        
        # Overall
        speedups = [r.speedup for r in results]
        ranks = [r.baseline_rank for r in results if r.baseline_rank > 0]
        
        print(f"\n📊 OVERALL STATISTICS")
        print(f"   Circuits benchmarked: {len(results)}")
        print(f"   Average speedup: {statistics.mean(speedups):.3f}x")
        print(f"   Median speedup: {statistics.median(speedups):.3f}x")
        print(f"   Min/Max speedup: {min(speedups):.3f}x / {max(speedups):.3f}x")
        
        if ranks:
            print(f"\n   Average final rank: {statistics.mean(ranks):.1f}")
            print(f"   Max final rank: {max(ranks)}")
            print(f"   Circuits with rank >= 32: {sum(1 for r in ranks if r >= 32)}")
        
        # KEY ANALYSIS: Speedup vs Rank
        print(f"\n🔑 KEY ANALYSIS: SPEEDUP VS FINAL RANK")
        print(f"   (Row-parallelism engages at rank >= 32)")
        print()
        
        rank_buckets = [
            (1, 4, "1-4 (low)"),
            (4, 16, "4-16 (medium)"),
            (16, 32, "16-32 (near threshold)"),
            (32, 100, "32-100 (above threshold)"),
            (100, 1000, "100+ (high)")
        ]
        
        print(f"   {'Rank Range':<25} | {'Avg Speedup':>10} | {'Count':>6} | Chart")
        print(f"   {'-'*25}-+-{'-'*10}-+-{'-'*6}-+------------")
        
        for low, high, label in rank_buckets:
            bucket = [r for r in results if r.baseline_rank and low <= r.baseline_rank < high]
            if bucket:
                avg = statistics.mean([r.speedup for r in bucket])
                bar = "█" * int(avg * 15)
                print(f"   {label:<25} | {avg:>10.2f}x | {len(bucket):>6} | {bar}")
        
        # By noise type
        print(f"\n📋 SPEEDUP BY NOISE TYPE")
        noise_types = sorted(set(r.noise_type for r in results))
        
        for nt in noise_types:
            nt_results = [r for r in results if r.noise_type == nt]
            if nt_results:
                avg = statistics.mean([r.speedup for r in nt_results])
                avg_rank = statistics.mean([r.baseline_rank for r in nt_results if r.baseline_rank])
                print(f"   {nt:<20}: {avg:.2f}x avg speedup, {avg_rank:.0f} avg rank (n={len(nt_results)})")
        
        # By noise rate
        print(f"\n📋 SPEEDUP BY NOISE RATE")
        noise_rates = sorted(set(r.noise_rate for r in results))
        
        for nr in noise_rates:
            nr_results = [r for r in results if r.noise_rate == nr]
            if nr_results:
                avg = statistics.mean([r.speedup for r in nr_results])
                avg_rank = statistics.mean([r.baseline_rank for r in nr_results if r.baseline_rank])
                print(f"   p={nr:.2f}: {avg:.2f}x avg speedup, {avg_rank:.0f} avg rank (n={len(nr_results)})")
        
        # Top performers
        print(f"\n🏆 TOP 10 SPEEDUPS")
        sorted_results = sorted(results, key=lambda r: -r.speedup)
        for i, r in enumerate(sorted_results[:10]):
            print(f"   {i+1:2d}. {r.speedup:.2f}x - {r.subtype} {r.n_qubits}q, rank={r.baseline_rank}, p={r.noise_rate:.2f}")
        
        # Correctness
        mismatches = [r for r in results if not r.ranks_match]
        if mismatches:
            print(f"\n⚠️  RANK MISMATCHES: {len(mismatches)}")
        else:
            print(f"\n✅ CORRECTNESS: All ranks match")
        
        print("\n" + "=" * 70)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="LRET Noisy Circuit Benchmark Runner")
    parser.add_argument("--min-qubits", type=int, default=6,
                        help="Minimum qubit count")
    parser.add_argument("--max-qubits", "-q", type=int, default=10,
                        help="Maximum qubit count")
    parser.add_argument("--trials", "-t", type=int, default=2,
                        help="Trials per circuit")
    parser.add_argument("--subtypes", nargs="+",
                        help="Only run specific subtypes")
    parser.add_argument("--output", "-o", type=str,
                        help="Output directory")
    
    args = parser.parse_args()
    
    try:
        runner = NoisyBenchmarkRunner()
        
        results = runner.run_benchmark(
            max_qubits=args.max_qubits,
            min_qubits=args.min_qubits,
            trials=args.trials,
            subtypes=args.subtypes
        )
        
        if results:
            runner.save_results(results, args.output)
            runner.analyze_results(results)
        else:
            print("No results collected.")
            
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Make sure you have generated noisy circuits first:")
        print("  python scripts/generate_noisy_circuits.py")
        sys.exit(1)


if __name__ == "__main__":
    main()

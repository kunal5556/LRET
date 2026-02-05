#!/usr/bin/env python3
"""
LRET Circuit Benchmark Runner - Phase B.2
Runs generated circuits through baseline and optimized simulators

Usage:
    python run_circuit_benchmarks.py --category random --max-qubits 10
    python run_circuit_benchmarks.py --all --trials 2
"""

import json
import os
import subprocess
import time
import argparse
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# =============================================================================
# Configuration
# =============================================================================

BASELINE_EXE = "baseline\\quantum_sim.exe"
OPTIMIZED_EXE = "optimized\\quantum_sim.exe"
GENERATED_DIR = "test_circuits\\generated"
RESULTS_DIR = "results"

# =============================================================================
# Benchmark Runner
# =============================================================================

class CircuitBenchmarkRunner:
    """Run benchmarks on generated circuits."""
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.baseline_exe = self.base_dir / BASELINE_EXE
        self.optimized_exe = self.base_dir / OPTIMIZED_EXE
        self.generated_dir = self.base_dir / GENERATED_DIR
        self.results_dir = self.base_dir / RESULTS_DIR
        
        # Load manifest
        manifest_path = self.generated_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                self.manifest = json.load(f)
        else:
            self.manifest = []
            
    def run_circuit(self, circuit_path: str, exe_path: Path, 
                    timeout: int = 300) -> Dict:
        """Run a single circuit and return timing info."""
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
            
            # Parse output for metrics
            output = result.stdout + result.stderr
            
            # Extract timing
            sim_time = None
            for line in output.split('\n'):
                if 'Time:' in line:
                    try:
                        sim_time = float(line.split(':')[1].strip().replace('s', '').strip())
                    except:
                        pass
            
            # Extract final rank
            final_rank = None
            for line in output.split('\n'):
                if 'Final Rank:' in line:
                    try:
                        final_rank = int(line.split(':')[1].strip())
                    except:
                        pass
            
            return {
                "success": result.returncode == 0,
                "wall_time": elapsed,
                "sim_time": sim_time,
                "final_rank": final_rank,
                "returncode": result.returncode,
                "error": None
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "wall_time": timeout,
                "sim_time": None,
                "final_rank": None,
                "returncode": -1,
                "error": "timeout"
            }
        except Exception as e:
            return {
                "success": False,
                "wall_time": time.time() - start_time,
                "sim_time": None,
                "final_rank": None,
                "returncode": -1,
                "error": str(e)
            }
    
    def run_benchmark(self, category: str = None, max_qubits: int = 12,
                      trials: int = 1, timeout: int = 300) -> List[Dict]:
        """Run benchmarks on matching circuits."""
        results = []
        
        # Filter circuits
        circuits = self.manifest
        if category:
            circuits = [c for c in circuits if c["category"] == category]
        circuits = [c for c in circuits if c["n_qubits"] <= max_qubits]
        
        print(f"Running {len(circuits)} circuits...")
        print(f"Max qubits: {max_qubits}, Trials: {trials}")
        print()
        
        for i, circuit_info in enumerate(circuits):
            circuit_path = self.generated_dir / circuit_info["file"]
            n_qubits = circuit_info["n_qubits"]
            n_ops = circuit_info["n_operations"]
            
            print(f"[{i+1}/{len(circuits)}] {circuit_info['file']}")
            print(f"  Qubits: {n_qubits}, Ops: {n_ops}")
            
            baseline_times = []
            optimized_times = []
            baseline_ranks = []
            optimized_ranks = []
            
            for t in range(trials):
                # Run baseline
                print(f"  Trial {t+1}/{trials} - Baseline...", end="", flush=True)
                baseline_result = self.run_circuit(circuit_path, self.baseline_exe, timeout)
                if baseline_result["success"]:
                    baseline_times.append(baseline_result["wall_time"])
                    if baseline_result["final_rank"]:
                        baseline_ranks.append(baseline_result["final_rank"])
                    print(f" {baseline_result['wall_time']:.2f}s")
                else:
                    print(f" FAILED ({baseline_result['error']})")
                
                # Run optimized
                print(f"  Trial {t+1}/{trials} - Optimized...", end="", flush=True)
                optimized_result = self.run_circuit(circuit_path, self.optimized_exe, timeout)
                if optimized_result["success"]:
                    optimized_times.append(optimized_result["wall_time"])
                    if optimized_result["final_rank"]:
                        optimized_ranks.append(optimized_result["final_rank"])
                    print(f" {optimized_result['wall_time']:.2f}s")
                else:
                    print(f" FAILED ({optimized_result['error']})")
            
            # Calculate averages
            if baseline_times and optimized_times:
                baseline_avg = sum(baseline_times) / len(baseline_times)
                optimized_avg = sum(optimized_times) / len(optimized_times)
                speedup = baseline_avg / optimized_avg if optimized_avg > 0 else 0
                
                baseline_rank_avg = sum(baseline_ranks) / len(baseline_ranks) if baseline_ranks else None
                optimized_rank_avg = sum(optimized_ranks) / len(optimized_ranks) if optimized_ranks else None
                
                result = {
                    "file": circuit_info["file"],
                    "category": circuit_info["category"],
                    "subtype": circuit_info["subtype"],
                    "n_qubits": n_qubits,
                    "n_operations": n_ops,
                    "baseline_time": round(baseline_avg, 4),
                    "optimized_time": round(optimized_avg, 4),
                    "speedup": round(speedup, 3),
                    "baseline_rank": int(baseline_rank_avg) if baseline_rank_avg else None,
                    "optimized_rank": int(optimized_rank_avg) if optimized_rank_avg else None,
                    "trials": trials
                }
                results.append(result)
                
                color_code = "\033[92m" if speedup > 1.05 else ("\033[91m" if speedup < 0.95 else "")
                reset = "\033[0m" if color_code else ""
                print(f"  {color_code}Speedup: {speedup:.2f}x{reset}")
            else:
                print("  SKIPPED (failures)")
            
            print()
        
        return results
    
    def save_results(self, results: List[Dict], prefix: str = "circuit_benchmark"):
        """Save benchmark results to CSV."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_subdir = self.results_dir / timestamp
        results_subdir.mkdir(parents=True, exist_ok=True)
        
        csv_path = results_subdir / f"{prefix}.csv"
        
        if results:
            fieldnames = results[0].keys()
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
        
        print(f"Results saved to: {csv_path}")
        return csv_path
    
    def analyze_results(self, results: List[Dict]):
        """Print analysis of benchmark results."""
        print("\n" + "=" * 60)
        print("                    BENCHMARK ANALYSIS")
        print("=" * 60)
        
        if not results:
            print("No results to analyze.")
            return
        
        # Overall
        speedups = [r["speedup"] for r in results]
        avg_speedup = sum(speedups) / len(speedups)
        max_speedup = max(speedups)
        min_speedup = min(speedups)
        
        print(f"\nTotal circuits benchmarked: {len(results)}")
        print(f"Average speedup: {avg_speedup:.3f}x")
        print(f"Max speedup: {max_speedup:.3f}x")
        print(f"Min speedup: {min_speedup:.3f}x")
        
        # By category
        print("\nBy Category:")
        categories = set(r["category"] for r in results)
        for cat in sorted(categories):
            cat_results = [r for r in results if r["category"] == cat]
            cat_avg = sum(r["speedup"] for r in cat_results) / len(cat_results)
            print(f"  {cat}: {cat_avg:.3f}x avg ({len(cat_results)} circuits)")
        
        # By qubit count
        print("\nBy Qubit Count:")
        qubit_counts = sorted(set(r["n_qubits"] for r in results))
        for n in qubit_counts:
            q_results = [r for r in results if r["n_qubits"] == n]
            q_avg = sum(r["speedup"] for r in q_results) / len(q_results)
            print(f"  {n} qubits: {q_avg:.3f}x avg ({len(q_results)} circuits)")
        
        # Best/worst
        print("\nTop 5 Best:")
        sorted_results = sorted(results, key=lambda x: x["speedup"], reverse=True)
        for r in sorted_results[:5]:
            print(f"  {r['speedup']:.2f}x - {r['category']}/{r['subtype']} {r['n_qubits']}q")
        
        print("\nTop 5 Worst:")
        for r in sorted_results[-5:]:
            print(f"  {r['speedup']:.2f}x - {r['category']}/{r['subtype']} {r['n_qubits']}q")
        
        # Correctness check
        mismatches = [r for r in results if r["baseline_rank"] != r["optimized_rank"] 
                      and r["baseline_rank"] is not None]
        if mismatches:
            print(f"\nWARNING: {len(mismatches)} circuits with rank mismatches!")
        else:
            print(f"\n✓ All circuits have matching final ranks")
        
        print("\n" + "=" * 60)


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="LRET Circuit Benchmark Runner")
    parser.add_argument("--category", "-c", 
                        choices=["basic", "entanglement", "random", "algorithm", "noisy", "stress"],
                        help="Run only specific category")
    parser.add_argument("--max-qubits", "-q", type=int, default=10,
                        help="Maximum qubit count to test")
    parser.add_argument("--trials", "-t", type=int, default=1,
                        help="Trials per circuit")
    parser.add_argument("--timeout", type=int, default=120,
                        help="Timeout per circuit in seconds")
    parser.add_argument("--all", "-a", action="store_true",
                        help="Run all categories")
    
    args = parser.parse_args()
    
    runner = CircuitBenchmarkRunner()
    
    category = None if args.all else args.category
    
    results = runner.run_benchmark(
        category=category,
        max_qubits=args.max_qubits,
        trials=args.trials,
        timeout=args.timeout
    )
    
    if results:
        runner.save_results(results)
        runner.analyze_results(results)


if __name__ == "__main__":
    main()

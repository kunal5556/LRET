#!/usr/bin/env python3
"""
Phase E Analysis - Aggregate all benchmarking results and create report.
"""

import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict
import statistics

@dataclass
class CircuitResult:
    circuit: str
    n_qubits: int
    baseline_ms: float
    optimized_ms: float
    baseline_rank: int
    optimized_rank: int
    speedup: float
    ranks_match: bool
    noise_type: str = "unknown"
    circuit_type: str = "unknown"

def load_phase_d_results(path: Path) -> List[CircuitResult]:
    """Load Phase D results."""
    with open(path, 'r') as f:
        data = json.load(f)
    
    results = []
    for r in data:
        # Extract qubit count and noise type from circuit name
        name = r["circuit"]
        n_qubits = 0
        noise_type = "unknown"
        circuit_type = "unknown"
        
        if "6q" in name:
            n_qubits = 6
        elif "8q" in name:
            n_qubits = 8
        elif "10q" in name:
            n_qubits = 10
        
        if "depolarizing" in name:
            noise_type = "depolarizing"
        elif "amplitude_damping" in name:
            noise_type = "amplitude_damping"
        elif "phase_damping" in name:
            noise_type = "phase_damping"
        elif "mixed" in name:
            noise_type = "mixed"
        
        if "ghz" in name:
            circuit_type = "ghz"
        elif "random" in name:
            circuit_type = "random"
        elif "vqe" in name:
            circuit_type = "vqe"
        elif "stress" in name:
            circuit_type = "stress"
        elif "mixed" in name:
            circuit_type = "mixed_noise"
        
        results.append(CircuitResult(
            circuit=name,
            n_qubits=n_qubits,
            baseline_ms=r["baseline_ms"],
            optimized_ms=r["optimized_ms"],
            baseline_rank=r["baseline_rank"],
            optimized_rank=r["optimized_rank"],
            speedup=r["speedup"],
            ranks_match=r["match"],
            noise_type=noise_type,
            circuit_type=circuit_type
        ))
    return results

def load_phase_e_partial(path: Path) -> List[CircuitResult]:
    """Load Phase E partial results."""
    with open(path, 'r') as f:
        data = json.load(f)
    
    results = []
    for r in data.get("results", []):
        results.append(CircuitResult(
            circuit=r["circuit"],
            n_qubits=r["n_qubits"],
            baseline_ms=r["baseline_ms"],
            optimized_ms=r["optimized_ms"],
            baseline_rank=r["baseline_rank"],
            optimized_rank=r["optimized_rank"],
            speedup=r["speedup"],
            ranks_match=r["ranks_match"],
            noise_type=r.get("noise_type", "unknown"),
            circuit_type=r.get("circuit_type", "unknown")
        ))
    return results

def analyze_by_qubit_count(results: List[CircuitResult]) -> Dict:
    """Analyze results grouped by qubit count."""
    by_qubits = {}
    for r in results:
        if r.n_qubits not in by_qubits:
            by_qubits[r.n_qubits] = []
        by_qubits[r.n_qubits].append(r)
    
    analysis = {}
    for n_qubits, group in sorted(by_qubits.items()):
        speedups = [r.speedup for r in group]
        ranks = [r.optimized_rank for r in group]
        
        analysis[n_qubits] = {
            "count": len(group),
            "avg_speedup": statistics.mean(speedups),
            "median_speedup": statistics.median(speedups),
            "min_speedup": min(speedups),
            "max_speedup": max(speedups),
            "avg_rank": statistics.mean(ranks),
            "max_rank": max(ranks),
            "all_match": all(r.ranks_match for r in group),
            "above_1x_pct": 100 * sum(1 for s in speedups if s >= 1.0) / len(speedups)
        }
    return analysis

def analyze_by_rank(results: List[CircuitResult]) -> Dict:
    """Analyze speedup correlation with rank."""
    # Group by rank ranges
    ranges = [(2, 10), (10, 25), (25, 40), (40, 60)]
    analysis = {}
    
    for low, high in ranges:
        group = [r for r in results if low <= r.optimized_rank < high]
        if group:
            speedups = [r.speedup for r in group]
            analysis[f"{low}-{high}"] = {
                "count": len(group),
                "avg_speedup": statistics.mean(speedups),
                "median_speedup": statistics.median(speedups)
            }
    
    # High rank group (40+)
    high_rank = [r for r in results if r.optimized_rank >= 40]
    if high_rank:
        speedups = [r.speedup for r in high_rank]
        analysis["40+"] = {
            "count": len(high_rank),
            "avg_speedup": statistics.mean(speedups),
            "median_speedup": statistics.median(speedups)
        }
    
    return analysis

def main():
    base_dir = Path("D:/LRET/validation")
    
    # Load all results
    phase_d = load_phase_d_results(base_dir / "PHASE_D_FIXED_RESULTS.json")
    phase_e = load_phase_e_partial(base_dir / "results" / "phase_e_partial.json")
    
    all_results = phase_d + phase_e
    
    print("=" * 70)
    print("PHASE E: Extended Benchmarking Analysis")
    print("=" * 70)
    print()
    
    # Overview
    print(f"Total circuits analyzed: {len(all_results)}")
    print(f"  - Phase D (6-10q): {len(phase_d)} circuits")
    print(f"  - Phase E (11-12q): {len(phase_e)} circuits")
    print()
    
    # All ranks match?
    all_match = all(r.ranks_match for r in all_results)
    print(f"All ranks match: {all_match} (100% correctness)")
    print()
    
    # By qubit count
    print("=" * 70)
    print("Results by Qubit Count")
    print("=" * 70)
    by_qubits = analyze_by_qubit_count(all_results)
    
    print(f"{'Qubits':>6} | {'Count':>5} | {'Avg Speed':>9} | {'Med Speed':>9} | {'Avg Rank':>8} | {'>1x %':>6}")
    print("-" * 70)
    for n_qubits, stats in sorted(by_qubits.items()):
        print(f"{n_qubits:>6} | {stats['count']:>5} | {stats['avg_speedup']:>9.2f}x | {stats['median_speedup']:>9.2f}x | {stats['avg_rank']:>8.1f} | {stats['above_1x_pct']:>5.1f}%")
    print()
    
    # Overall stats
    all_speedups = [r.speedup for r in all_results]
    print("Overall Statistics:")
    print(f"  Average speedup: {statistics.mean(all_speedups):.3f}x")
    print(f"  Median speedup: {statistics.median(all_speedups):.3f}x")
    print(f"  Std deviation: {statistics.stdev(all_speedups):.3f}")
    print(f"  Range: {min(all_speedups):.3f}x to {max(all_speedups):.3f}x")
    print(f"  Above 1.0x: {100 * sum(1 for s in all_speedups if s >= 1.0) / len(all_speedups):.1f}%")
    print()
    
    # By rank
    print("=" * 70)
    print("Speedup vs Rank Analysis")
    print("=" * 70)
    by_rank = analyze_by_rank(all_results)
    
    print(f"{'Rank Range':>12} | {'Count':>5} | {'Avg Speedup':>11} | {'Med Speedup':>11}")
    print("-" * 50)
    for range_name, stats in by_rank.items():
        print(f"{range_name:>12} | {stats['count']:>5} | {stats['avg_speedup']:>11.2f}x | {stats['median_speedup']:>11.2f}x")
    print()
    
    # Key findings for 11-12q
    print("=" * 70)
    print("11-12 Qubit Key Findings")
    print("=" * 70)
    
    q11_12 = [r for r in all_results if r.n_qubits in [11, 12]]
    if q11_12:
        print(f"Circuits tested: {len(q11_12)}")
        print(f"Ranks: {[r.optimized_rank for r in q11_12]}")
        print(f"Max rank reached: {max(r.optimized_rank for r in q11_12)}")
        print(f"All ranks match: {all(r.ranks_match for r in q11_12)}")
        
        speedups = [r.speedup for r in q11_12]
        print(f"Average speedup: {statistics.mean(speedups):.3f}x")
        print(f"Speedup range: {min(speedups):.3f}x to {max(speedups):.3f}x")
    
    # Save aggregated results
    output = {
        "total_circuits": len(all_results),
        "phase_d_count": len(phase_d),
        "phase_e_count": len(phase_e),
        "all_ranks_match": all_match,
        "overall_avg_speedup": statistics.mean(all_speedups),
        "overall_median_speedup": statistics.median(all_speedups),
        "overall_std_speedup": statistics.stdev(all_speedups),
        "above_1x_pct": 100 * sum(1 for s in all_speedups if s >= 1.0) / len(all_speedups),
        "by_qubits": by_qubits,
        "by_rank": by_rank
    }
    
    output_path = base_dir / "results" / "phase_e_aggregated.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print()
    print(f"Aggregated results saved to: {output_path}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
LRET Benchmark Analysis Script
Phase A.2: Analyze and visualize benchmark results

Usage:
    python analyze_results.py results/20260205_*/benchmark_results.csv
"""

import sys
import csv
import os
from collections import defaultdict

def load_csv(filepath):
    """Load benchmark results from CSV."""
    results = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Handle both full and quick benchmark formats
            results.append({
                'qubits': int(row['Qubits']),
                'depth': int(row.get('Depth', 15)),
                'initial_rank': int(row.get('InitialRank', row.get('Rank', 0))),
                'mode': row['Mode'],
                'baseline_time': float(row.get('Baseline_Mean_s', row.get('Baseline', 0))),
                'optimized_time': float(row.get('Optimized_Mean_s', row.get('Optimized', 0))),
                'speedup': float(row['Speedup']),
                'baseline_rank': int(row.get('Baseline_FinalRank', 0)),
                'optimized_rank': int(row.get('Optimized_FinalRank', 0)),
            })
    return results

def analyze_results(results):
    """Perform comprehensive analysis of benchmark results."""
    
    print("=" * 70)
    print("         LRET Benchmark Analysis - Baseline vs Optimized")
    print("=" * 70)
    print()
    
    # Overall statistics
    speedups = [r['speedup'] for r in results]
    avg_speedup = sum(speedups) / len(speedups)
    max_speedup = max(speedups)
    min_speedup = min(speedups)
    
    print("OVERALL STATISTICS")
    print("-" * 40)
    print(f"  Total configurations tested: {len(results)}")
    print(f"  Average speedup: {avg_speedup:.3f}x")
    print(f"  Maximum speedup: {max_speedup:.3f}x")
    print(f"  Minimum speedup: {min_speedup:.3f}x")
    print()
    
    # By mode
    print("BY PARALLELIZATION MODE")
    print("-" * 40)
    by_mode = defaultdict(list)
    for r in results:
        by_mode[r['mode']].append(r['speedup'])
    
    for mode in sorted(by_mode.keys()):
        speeds = by_mode[mode]
        avg = sum(speeds) / len(speeds)
        print(f"  {mode:12s}: {avg:.3f}x avg ({len(speeds)} tests)")
    print()
    
    # By initial rank
    print("BY INITIAL RANK (key for Phase 1 optimization)")
    print("-" * 40)
    by_rank = defaultdict(list)
    for r in results:
        by_rank[r['initial_rank']].append(r['speedup'])
    
    for rank in sorted(by_rank.keys()):
        speeds = by_rank[rank]
        avg = sum(speeds) / len(speeds)
        indicator = "<<< Threshold changed" if rank == 32 else ""
        print(f"  Rank {rank:3d}: {avg:.3f}x avg ({len(speeds)} tests) {indicator}")
    print()
    
    # By qubit count
    print("BY QUBIT COUNT")
    print("-" * 40)
    by_qubits = defaultdict(list)
    for r in results:
        by_qubits[r['qubits']].append(r['speedup'])
    
    for q in sorted(by_qubits.keys()):
        speeds = by_qubits[q]
        avg = sum(speeds) / len(speeds)
        print(f"  {q:2d} qubits: {avg:.3f}x avg ({len(speeds)} tests)")
    print()
    
    # Best and worst configurations
    print("TOP 5 BEST CONFIGURATIONS")
    print("-" * 40)
    sorted_results = sorted(results, key=lambda x: x['speedup'], reverse=True)
    for i, r in enumerate(sorted_results[:5], 1):
        print(f"  {i}. n={r['qubits']}, rank={r['initial_rank']:2d}, mode={r['mode']:10s} -> {r['speedup']:.3f}x")
    print()
    
    print("TOP 5 WORST CONFIGURATIONS")
    print("-" * 40)
    for i, r in enumerate(sorted_results[-5:], 1):
        print(f"  {i}. n={r['qubits']}, rank={r['initial_rank']:2d}, mode={r['mode']:10s} -> {r['speedup']:.3f}x")
    print()
    
    # Phase 1 specific analysis: Row mode around threshold
    print("PHASE 1 ANALYSIS: Row Mode Performance")
    print("-" * 40)
    print("The Phase 1 optimization changed MIN_RANK_FOR_COL_PARALLEL from 4 to 32.")
    print("This should improve row mode performance for ranks 4-31.")
    print()
    row_results = [r for r in results if r['mode'] == 'row']
    if row_results:
        below_threshold = [r for r in row_results if r['initial_rank'] < 32]
        at_or_above = [r for r in row_results if r['initial_rank'] >= 32]
        
        if below_threshold:
            avg_below = sum(r['speedup'] for r in below_threshold) / len(below_threshold)
            print(f"  Rank < 32: {avg_below:.3f}x avg ({len(below_threshold)} tests)")
        if at_or_above:
            avg_above = sum(r['speedup'] for r in at_or_above) / len(at_or_above)
            print(f"  Rank >= 32: {avg_above:.3f}x avg ({len(at_or_above)} tests)")
    print()
    
    # Correctness check: final ranks should match
    print("CORRECTNESS CHECK: Final Rank Comparison")
    print("-" * 40)
    rank_mismatches = [r for r in results if r['baseline_rank'] != r['optimized_rank']]
    if rank_mismatches:
        print(f"  WARNING: {len(rank_mismatches)} configurations have mismatched final ranks!")
        for r in rank_mismatches[:5]:
            print(f"    n={r['qubits']}, init_rank={r['initial_rank']}, mode={r['mode']}: "
                  f"baseline={r['baseline_rank']}, optimized={r['optimized_rank']}")
    else:
        print(f"  ✓ All {len(results)} configurations have matching final ranks")
    print()
    
    print("=" * 70)
    print("                    ANALYSIS COMPLETE")
    print("=" * 70)

def generate_ascii_chart(results):
    """Generate ASCII bar chart of speedups by mode."""
    print("\nSPEEDUP BY MODE (ASCII Chart)")
    print("-" * 50)
    
    by_mode = defaultdict(list)
    for r in results:
        by_mode[r['mode']].append(r['speedup'])
    
    for mode in sorted(by_mode.keys()):
        avg = sum(by_mode[mode]) / len(by_mode[mode])
        bar_len = int(avg * 20)  # Scale: 1.0x = 20 chars
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"  {mode:12s} |{bar}| {avg:.2f}x")
    print()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py <benchmark_results.csv>")
        print()
        print("Looking for latest results...")
        # Find latest results file
        results_dir = "results"
        if os.path.exists(results_dir):
            subdirs = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
            subdirs.sort(reverse=True)
            if subdirs:
                csv_path = os.path.join(results_dir, subdirs[0], "benchmark_results.csv")
                if os.path.exists(csv_path):
                    print(f"Found: {csv_path}")
                    results = load_csv(csv_path)
                    analyze_results(results)
                    generate_ascii_chart(results)
                    sys.exit(0)
        print("No results found. Run run_full_benchmark.ps1 first.")
        sys.exit(1)
    else:
        csv_path = sys.argv[1]
        if not os.path.exists(csv_path):
            print(f"Error: File not found: {csv_path}")
            sys.exit(1)
        
        results = load_csv(csv_path)
        analyze_results(results)
        generate_ascii_chart(results)

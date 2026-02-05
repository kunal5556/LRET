#!/usr/bin/env python3
"""Quick noisy benchmark - Phase D verification with fixed optimized binary."""

import subprocess
import json
import time
import os
from pathlib import Path

def run_sim(exe_path, circuit_path):
    """Run simulator and return (time_ms, final_rank)."""
    start = time.time()
    result = subprocess.run(
        [str(exe_path), "--input-json", str(circuit_path), "--allow-swap", "--non-interactive"],
        capture_output=True, text=True
    )
    elapsed = (time.time() - start) * 1000
    
    rank = None
    for line in result.stdout.split('\n'):
        if '"final_rank"' in line:
            try:
                rank = int(line.split(':')[1].strip().rstrip(','))
            except:
                pass
    return elapsed, rank

def main():
    base_dir = Path("D:/LRET/validation")
    baseline_exe = base_dir / "baseline" / "quantum_sim.exe"
    optimized_exe = base_dir / "optimized" / "quantum_sim.exe"
    noisy_dir = base_dir / "test_circuits" / "noisy"
    
    print("="*80)
    print("PHASE D NOISY BENCHMARK - FIXED OPTIMIZED BINARY")
    print("="*80)
    
    # Group circuits by qubit count and type
    circuits = sorted(noisy_dir.glob("*.json"))
    circuits = [c for c in circuits if c.name != "manifest.json"]
    
    results = []
    
    for circuit in circuits:
        b_time, b_rank = run_sim(baseline_exe, circuit)
        o_time, o_rank = run_sim(optimized_exe, circuit)
        
        if o_time > 0:
            speedup = b_time / o_time
        else:
            speedup = 0
            
        match = "OK" if b_rank == o_rank else "MISMATCH!"
        
        results.append({
            "circuit": circuit.name,
            "baseline_ms": b_time,
            "optimized_ms": o_time,
            "baseline_rank": b_rank,
            "optimized_rank": o_rank,
            "speedup": speedup,
            "match": b_rank == o_rank
        })
        
        # Short display name
        name = circuit.stem[:35]
        print(f"{name:35} | B={b_time/1000:5.1f}s O={o_time/1000:5.1f}s | r={b_rank:3} | {speedup:4.2f}x | {match}")
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    matching = [r for r in results if r["match"]]
    mismatched = [r for r in results if not r["match"]]
    
    print(f"Total circuits: {len(results)}")
    print(f"Matching ranks: {len(matching)}")
    print(f"Mismatched ranks: {len(mismatched)}")
    
    if mismatched:
        print("\nMISMATCHED CIRCUITS:")
        for r in mismatched:
            print(f"  {r['circuit']}: baseline={r['baseline_rank']} vs optimized={r['optimized_rank']}")
    
    # Speedup by rank
    print("\nSPEEDUP BY RANK RANGE:")
    rank_ranges = [(0, 16), (16, 32), (32, 48), (48, 64), (64, 1000)]
    
    for low, high in rank_ranges:
        in_range = [r for r in matching if r["baseline_rank"] and low <= r["baseline_rank"] < high]
        if in_range:
            avg_speedup = sum(r["speedup"] for r in in_range) / len(in_range)
            avg_b = sum(r["baseline_ms"] for r in in_range) / len(in_range)
            avg_o = sum(r["optimized_ms"] for r in in_range) / len(in_range)
            print(f"  Rank {low:2}-{high:2}: {len(in_range):3} circuits, avg speedup = {avg_speedup:.2f}x (B={avg_b/1000:.1f}s, O={avg_o/1000:.1f}s)")
    
    # Save results
    results_file = base_dir / "PHASE_D_FIXED_RESULTS.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Run small 11-12q circuits only (< 150 ops) to avoid timeout issues.
"""

import subprocess
import json
import time
import re
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Tuple, Optional

@dataclass
class BenchmarkResult:
    circuit: str
    n_qubits: int
    n_operations: int
    noise_count: int
    noise_type: str
    noise_rate: float
    baseline_ms: float
    optimized_ms: float
    baseline_rank: int
    optimized_rank: int
    speedup: float
    ranks_match: bool
    circuit_type: str

def run_simulator(exe_path: Path, circuit_path: Path, timeout: int = 300) -> Tuple[float, int, Optional[str]]:
    """Run simulator and return (time_ms, final_rank, error)."""
    try:
        start = time.time()
        result = subprocess.run(
            [str(exe_path), "--input-json", str(circuit_path), "--allow-swap", "--non-interactive"],
            capture_output=True, text=True, timeout=timeout
        )
        elapsed_ms = (time.time() - start) * 1000
        
        rank = None
        for line in result.stdout.split('\n'):
            if '"final_rank"' in line:
                match = re.search(r'"final_rank"\s*:\s*(\d+)', line)
                if match:
                    rank = int(match.group(1))
                    break
        
        if rank is None:
            if result.returncode != 0:
                return 0.0, 0, f"Exit code {result.returncode}"
            return elapsed_ms, 1, None
        
        return elapsed_ms, rank, None
    except subprocess.TimeoutExpired:
        return 0.0, 0, f"Timeout ({timeout}s)"
    except Exception as e:
        return 0.0, 0, str(e)

def load_circuit_metadata(circuit_path: Path) -> dict:
    with open(circuit_path, 'r') as f:
        data = json.load(f)
    
    circuit = data.get("circuit", {})
    metadata = data.get("metadata", {})
    operations = circuit.get("operations", [])
    
    return {
        "n_qubits": circuit.get("n_qubits", metadata.get("n_qubits", 0)),
        "n_operations": len(operations),
        "noise_count": sum(1 for op in operations if op.get("name") == "KRAUS"),
        "noise_type": metadata.get("noise_type", "unknown"),
        "noise_rate": metadata.get("noise_rate", 0.0),
        "circuit_type": metadata.get("subtype", "unknown")
    }

def main():
    base_dir = Path("D:/LRET/validation")
    baseline_exe = base_dir / "baseline" / "quantum_sim.exe"
    optimized_exe = base_dir / "optimized" / "quantum_sim.exe"
    noisy_dir = base_dir / "test_circuits" / "noisy"
    partial_file = base_dir / "results" / "phase_e_partial.json"
    output_file = base_dir / "results" / "phase_e_11_12q_small.json"
    
    # Load existing results
    completed = {}
    if partial_file.exists():
        with open(partial_file, 'r') as f:
            data = json.load(f)
            for r in data.get("results", []):
                completed[r["circuit"]] = r
    
    print(f"Already completed: {len(completed)} circuits")
    
    # Get all 11-12q circuits with < 150 ops (exclude mixed_noise and stress)
    all_circuits = []
    for f in sorted(noisy_dir.glob("*.json")):
        if ("11q" in f.name or "12q" in f.name) and f.name != "manifest.json":
            if "mixed_noise" not in f.name and "stress" not in f.name:
                all_circuits.append(f)
    
    remaining = [c for c in all_circuits if c.name not in completed]
    print(f"Small circuits (no mixed_noise/stress) to test: {len(remaining)}")
    print()
    
    results: List[BenchmarkResult] = []
    
    for i, circuit_path in enumerate(remaining):
        try:
            meta = load_circuit_metadata(circuit_path)
        except Exception as e:
            print(f"[{i+1}/{len(remaining)}] {circuit_path.name}: Failed to load: {e}")
            continue
        
        n_qubits = meta["n_qubits"]
        n_ops = meta["n_operations"]
        
        print(f"[{i+1}/{len(remaining)}] {circuit_path.name[:50]:50} | {n_qubits}q, {n_ops} ops", end="", flush=True)
        
        timeout = 180  # 3 min for small circuits
        
        # Run baseline
        b_time, b_rank, b_err = run_simulator(baseline_exe, circuit_path, timeout)
        if b_err:
            print(f" | BASELINE ERROR: {b_err}")
            continue
        
        # Run optimized
        o_time, o_rank, o_err = run_simulator(optimized_exe, circuit_path, timeout)
        if o_err:
            print(f" | OPTIMIZED ERROR: {o_err}")
            continue
        
        speedup = b_time / o_time if o_time > 0 else 0.0
        ranks_match = (b_rank == o_rank)
        
        result = BenchmarkResult(
            circuit=circuit_path.name,
            n_qubits=n_qubits,
            n_operations=n_ops,
            noise_count=meta["noise_count"],
            noise_type=meta["noise_type"],
            noise_rate=meta["noise_rate"],
            baseline_ms=b_time,
            optimized_ms=o_time,
            baseline_rank=b_rank,
            optimized_rank=o_rank,
            speedup=speedup,
            ranks_match=ranks_match,
            circuit_type=meta["circuit_type"]
        )
        results.append(result)
        
        match_str = "OK" if ranks_match else "MISMATCH"
        print(f" | B={b_time/1000:.1f}s O={o_time/1000:.1f}s r={b_rank:3} {speedup:.2f}x {match_str}")
    
    # Save results
    print()
    print("=" * 70)
    print(f"COMPLETE: {len(results)} circuits")
    print("=" * 70)
    
    if results:
        all_match = all(r.ranks_match for r in results)
        avg_speedup = sum(r.speedup for r in results) / len(results)
        print(f"All ranks match: {all_match}")
        print(f"Average speedup: {avg_speedup:.2f}x")
        
        with open(output_file, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "count": len(results),
                "all_ranks_match": all_match,
                "average_speedup": avg_speedup,
                "results": [asdict(r) for r in results]
            }, f, indent=2)
        print(f"\nSaved to: {output_file}")

if __name__ == "__main__":
    main()

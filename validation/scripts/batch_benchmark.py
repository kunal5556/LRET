#!/usr/bin/env python3
"""
Run 11-12q benchmarks in small batches with recovery.
"""

import subprocess
import json
import time
import re
import sys
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

def run_simulator(exe_path: Path, circuit_path: Path, timeout: int = 600) -> Tuple[float, int, Optional[str]]:
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
    output_file = base_dir / "results" / "phase_e_complete.json"
    
    # Load all existing results from various files
    completed = {}  # circuit name -> result
    
    # Load partial results
    for partial_file in [
        base_dir / "results" / "phase_e_partial.json",
        base_dir / "results" / "phase_e_11_12q_results.json",
        base_dir / "results" / "phase_e_complete.json"
    ]:
        if partial_file.exists():
            with open(partial_file, 'r') as f:
                data = json.load(f)
                for r in data.get("results", []):
                    completed[r["circuit"]] = r
    
    print(f"Already completed: {len(completed)} circuits")
    
    # Get batch number from command line
    batch_size = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    batch_num = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    
    # Get all 11-12q circuits
    all_circuits = sorted([
        f for f in noisy_dir.glob("*.json")
        if ("11q" in f.name or "12q" in f.name) and f.name != "manifest.json"
    ])
    
    remaining = [c for c in all_circuits if c.name not in completed]
    print(f"Remaining to test: {len(remaining)} circuits")
    
    # Select batch
    start_idx = batch_num * batch_size
    batch = remaining[start_idx:start_idx + batch_size]
    print(f"Batch {batch_num}: circuits {start_idx+1}-{start_idx+len(batch)} of {len(remaining)}")
    print()
    
    results: List[BenchmarkResult] = list(completed.values())  # Start with existing
    
    for i, circuit_path in enumerate(batch):
        try:
            meta = load_circuit_metadata(circuit_path)
        except Exception as e:
            print(f"[{i+1}/{len(batch)}] {circuit_path.name}: Failed to load metadata: {e}")
            continue
        
        n_qubits = meta["n_qubits"]
        n_ops = meta["n_operations"]
        
        # Set timeout based on circuit size
        if n_ops > 400:
            timeout = 600  # 10 min max
        elif n_ops > 200:
            timeout = 300  # 5 min
        else:
            timeout = 180  # 3 min
        
        print(f"[{i+1}/{len(batch)}] {circuit_path.name[:50]:50} | {n_qubits}q, {n_ops} ops", end="", flush=True)
        
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
        completed[circuit_path.name] = asdict(result)
        
        match_str = "✓" if ranks_match else "✗"
        print(f" | B={b_time/1000:.1f}s O={o_time/1000:.1f}s r={b_rank:3} {speedup:.2f}x {match_str}")
    
    # Save all results
    print()
    print(f"Batch complete. Total results: {len(results)}")
    
    # Filter to only 11-12q results
    results_11_12q = [r for r in results if isinstance(r, BenchmarkResult) and r.n_qubits in [11, 12]]
    if not results_11_12q:
        results_11_12q = [BenchmarkResult(**r) for r in results if isinstance(r, dict) and r.get("n_qubits") in [11, 12]]
    
    all_match = all((r.ranks_match if isinstance(r, BenchmarkResult) else r.get("ranks_match", False)) for r in results)
    
    with open(output_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "status": "partial" if len(completed) < len(all_circuits) else "complete",
            "count": len(completed),
            "total_11_12q": len(all_circuits),
            "all_ranks_match": all_match,
            "results": list(completed.values())
        }, f, indent=2)
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    main()

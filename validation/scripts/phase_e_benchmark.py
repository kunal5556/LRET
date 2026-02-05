#!/usr/bin/env python3
"""
Phase E Extended Benchmarking - LRET Optimization Validation
Comprehensive benchmark comparing baseline vs optimized for 6-12 qubit noisy circuits.

Collects:
- Execution time (ms)
- Final rank
- Memory estimate
- Speedup ratio
- Correlation data for analysis
"""

import subprocess
import json
import time
import os
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import statistics

@dataclass
class BenchmarkResult:
    """Single benchmark result."""
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
    """
    Run simulator and return (time_ms, final_rank, error).
    Returns (0.0, 0, error_string) on failure.
    """
    try:
        start = time.time()
        result = subprocess.run(
            [str(exe_path), "--input-json", str(circuit_path), "--allow-swap", "--non-interactive"],
            capture_output=True, text=True, timeout=timeout
        )
        elapsed_ms = (time.time() - start) * 1000
        
        # Parse final_rank from output
        rank = None
        for line in result.stdout.split('\n'):
            if '"final_rank"' in line:
                match = re.search(r'"final_rank"\s*:\s*(\d+)', line)
                if match:
                    rank = int(match.group(1))
                    break
        
        if rank is None:
            # Check for errors
            if result.returncode != 0:
                return 0.0, 0, f"Exit code {result.returncode}"
            return elapsed_ms, 1, None  # Default rank=1 if not found
        
        return elapsed_ms, rank, None
        
    except subprocess.TimeoutExpired:
        return 0.0, 0, f"Timeout ({timeout}s)"
    except Exception as e:
        return 0.0, 0, str(e)

def load_circuit_metadata(circuit_path: Path) -> Dict:
    """Load circuit and extract metadata."""
    with open(circuit_path, 'r') as f:
        data = json.load(f)
    
    circuit = data.get("circuit", {})
    metadata = data.get("metadata", {})
    
    # Count operations
    operations = circuit.get("operations", [])
    n_ops = len(operations)
    
    # Count noise operations
    noise_count = sum(1 for op in operations if op.get("name") == "KRAUS")
    
    # Extract info
    return {
        "n_qubits": circuit.get("n_qubits", metadata.get("n_qubits", 0)),
        "n_operations": n_ops,
        "noise_count": noise_count,
        "noise_type": metadata.get("noise_type", "unknown"),
        "noise_rate": metadata.get("noise_rate", 0.0),
        "circuit_type": metadata.get("subtype", "unknown")
    }

class PhaseEBenchmark:
    """Phase E Extended Benchmarking."""
    
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.baseline_exe = base_dir / "baseline" / "quantum_sim.exe"
        self.optimized_exe = base_dir / "optimized" / "quantum_sim.exe"
        self.noisy_dir = base_dir / "test_circuits" / "noisy"
        self.results: List[BenchmarkResult] = []
        self.results_file = base_dir / "results" / "phase_e_results.json"
        self.partial_file = base_dir / "results" / "phase_e_partial.json"
    
    def save_partial_results(self):
        """Save partial results for recovery."""
        if self.results:
            self.partial_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.partial_file, 'w') as f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "status": "partial",
                    "count": len(self.results),
                    "results": [asdict(r) for r in self.results]
                }, f, indent=2)
    
    def verify_setup(self) -> bool:
        """Verify executables and circuits exist."""
        if not self.baseline_exe.exists():
            print(f"ERROR: Baseline not found: {self.baseline_exe}")
            return False
        if not self.optimized_exe.exists():
            print(f"ERROR: Optimized not found: {self.optimized_exe}")
            return False
        if not self.noisy_dir.exists():
            print(f"ERROR: Noisy circuits not found: {self.noisy_dir}")
            return False
        return True
    
    def run_benchmark(self, 
                      min_qubits: int = 6, 
                      max_qubits: int = 12,
                      timeout_per_qubit: int = 60) -> List[BenchmarkResult]:
        """Run comprehensive benchmark."""
        
        if not self.verify_setup():
            return []
        
        # Collect all circuit files
        circuits = sorted([f for f in self.noisy_dir.glob("*.json") if f.name != "manifest.json"])
        
        print("=" * 80)
        print(f"PHASE E: Extended Benchmarking ({len(circuits)} circuits)")
        print("=" * 80)
        print(f"Baseline: {self.baseline_exe}")
        print(f"Optimized: {self.optimized_exe}")
        print(f"Qubit filter: {min_qubits}-{max_qubits}")
        print()
        
        self.results = []
        errors = []
        
        for i, circuit_path in enumerate(circuits):
            # Load metadata
            try:
                meta = load_circuit_metadata(circuit_path)
            except Exception as e:
                errors.append(f"{circuit_path.name}: Failed to load metadata: {e}")
                continue
            
            n_qubits = meta["n_qubits"]
            
            # Filter by qubit count
            if n_qubits < min_qubits or n_qubits > max_qubits:
                continue
            
            # Calculate timeout based on qubit count
            timeout = timeout_per_qubit * (2 ** (n_qubits - 6))
            timeout = max(60, min(timeout, 1800))  # 1min to 30min
            
            # Progress
            progress = f"[{i+1}/{len(circuits)}]"
            print(f"{progress} {circuit_path.name[:50]:50} | {n_qubits}q, {meta['n_operations']} ops", end="", flush=True)
            
            # Run baseline
            b_time, b_rank, b_err = run_simulator(self.baseline_exe, circuit_path, timeout)
            if b_err:
                print(f" | BASELINE ERROR: {b_err}")
                errors.append(f"{circuit_path.name}: Baseline: {b_err}")
                continue
            
            # Run optimized
            o_time, o_rank, o_err = run_simulator(self.optimized_exe, circuit_path, timeout)
            if o_err:
                print(f" | OPTIMIZED ERROR: {o_err}")
                errors.append(f"{circuit_path.name}: Optimized: {o_err}")
                continue
            
            # Calculate speedup
            speedup = b_time / o_time if o_time > 0 else 0.0
            ranks_match = (b_rank == o_rank)
            
            # Store result
            result = BenchmarkResult(
                circuit=circuit_path.name,
                n_qubits=n_qubits,
                n_operations=meta["n_operations"],
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
            self.results.append(result)
            
            # Save incrementally every 5 results
            if len(self.results) % 5 == 0:
                self.save_partial_results()
            
            # Output
            match_str = "✓" if ranks_match else "✗"
            print(f" | B={b_time/1000:.1f}s O={o_time/1000:.1f}s r={b_rank:3} {speedup:.2f}x {match_str}")
        
        # Summary
        print()
        print("=" * 80)
        print("SUMMARY")
        print("=" * 80)
        
        if self.results:
            self._print_summary()
        
        if errors:
            print(f"\nErrors: {len(errors)}")
            for e in errors[:10]:
                print(f"  {e}")
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more")
        
        return self.results
    
    def _print_summary(self):
        """Print benchmark summary statistics."""
        
        # Overall
        total = len(self.results)
        matching = sum(1 for r in self.results if r.ranks_match)
        print(f"Total circuits: {total}")
        print(f"Matching ranks: {matching} ({100*matching/total:.1f}%)")
        
        # By qubit count
        print("\nBy Qubit Count:")
        qubit_groups = {}
        for r in self.results:
            q = r.n_qubits
            if q not in qubit_groups:
                qubit_groups[q] = []
            qubit_groups[q].append(r)
        
        for q in sorted(qubit_groups.keys()):
            group = qubit_groups[q]
            speedups = [r.speedup for r in group if r.speedup > 0]
            ranks = [r.baseline_rank for r in group]
            avg_speedup = statistics.mean(speedups) if speedups else 0
            avg_rank = statistics.mean(ranks) if ranks else 0
            matching_pct = 100 * sum(1 for r in group if r.ranks_match) / len(group)
            print(f"  {q:2}q: {len(group):3} circuits, avg speedup={avg_speedup:.2f}x, avg rank={avg_rank:.0f}, match={matching_pct:.0f}%")
        
        # By rank range
        print("\nBy Rank Range:")
        rank_ranges = [(0, 16), (16, 32), (32, 64), (64, 128), (128, 256), (256, 1024)]
        for low, high in rank_ranges:
            in_range = [r for r in self.results if low <= r.baseline_rank < high]
            if in_range:
                speedups = [r.speedup for r in in_range if r.speedup > 0]
                avg_speedup = statistics.mean(speedups) if speedups else 0
                min_speedup = min(speedups) if speedups else 0
                max_speedup = max(speedups) if speedups else 0
                print(f"  Rank {low:3}-{high:3}: {len(in_range):3} circuits, speedup={avg_speedup:.2f}x (min={min_speedup:.2f}, max={max_speedup:.2f})")
        
        # Top speedups
        print("\nTop 10 Speedups:")
        sorted_results = sorted(self.results, key=lambda r: r.speedup, reverse=True)
        for r in sorted_results[:10]:
            print(f"  {r.speedup:.2f}x: {r.circuit} (rank={r.baseline_rank}, {r.n_qubits}q)")
    
    def save_results(self, output_path: Path):
        """Save results to JSON."""
        data = {
            "timestamp": datetime.now().isoformat(),
            "total_circuits": len(self.results),
            "results": [asdict(r) for r in self.results]
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"\nResults saved to: {output_path}")

def main():
    import argparse
    import signal
    
    parser = argparse.ArgumentParser(description="Phase E Extended Benchmarking")
    parser.add_argument("--min-qubits", type=int, default=6, help="Minimum qubits")
    parser.add_argument("--max-qubits", type=int, default=12, help="Maximum qubits")
    parser.add_argument("--timeout", type=int, default=60, help="Timeout per qubit (seconds, scales with 2^n)")
    parser.add_argument("--output", "-o", default="results/phase_e_results.json", help="Output file")
    
    args = parser.parse_args()
    
    base_dir = Path("D:/LRET/validation")
    benchmark = PhaseEBenchmark(base_dir)
    
    # Handle interrupt gracefully
    def handle_interrupt(sig, frame):
        print("\n\n*** INTERRUPTED - Saving partial results ***")
        benchmark.save_partial_results()
        print(f"Partial results saved ({len(benchmark.results)} circuits)")
        sys.exit(1)
    
    signal.signal(signal.SIGINT, handle_interrupt)
    
    try:
        results = benchmark.run_benchmark(
            min_qubits=args.min_qubits,
            max_qubits=args.max_qubits,
            timeout_per_qubit=args.timeout
        )
        
        if results:
            output_path = base_dir / args.output
            output_path.parent.mkdir(parents=True, exist_ok=True)
            benchmark.save_results(output_path)
    except KeyboardInterrupt:
        handle_interrupt(None, None)
    except Exception as e:
        print(f"\n*** ERROR: {e} - Saving partial results ***")
        benchmark.save_partial_results()
        raise

if __name__ == "__main__":
    main()

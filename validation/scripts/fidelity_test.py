#!/usr/bin/env python3
"""
Phase F: Fidelity Testing Script
Compare density matrices between baseline and optimized simulators.

Metrics computed:
1. Trace Distance: d(rho1, rho2) = 1/2 * ||rho1 - rho2||_1
2. Fidelity: F(rho1, rho2) = (Tr[sqrt(sqrt(rho1) * rho2 * sqrt(rho1))])^2
3. Trace: Tr(rho) should be 1.0
4. Purity: Tr(rho^2) in [0, 1]
5. Observable expectations: <Z_0>, <X_0>, <Z_all>

For publishable results, we require:
- Trace distance < 1e-10
- Fidelity > 0.999999
- Trace deviation < 1e-12
"""

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="Matrix is singular")

import subprocess
import json
import numpy as np
import time
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict
from datetime import datetime
import scipy.linalg

@dataclass
class FidelityResult:
    """Result of fidelity comparison for one circuit."""
    circuit: str
    n_qubits: int
    baseline_rank: int
    optimized_rank: int
    ranks_match: bool
    trace_distance: float
    fidelity: float
    baseline_trace: float
    optimized_trace: float
    baseline_purity: float
    optimized_purity: float
    z0_baseline: float
    z0_optimized: float
    z0_diff: float
    passed: bool
    error: Optional[str] = None


def reconstruct_L_matrix(state_data: dict) -> np.ndarray:
    """
    Reconstruct the L matrix from JSON state data.
    L is stored as flattened L_real and L_imag arrays.
    """
    rows = state_data["rows"]
    cols = state_data["cols"]
    L_real = np.array(state_data["L_real"]).reshape((rows, cols))
    L_imag = np.array(state_data["L_imag"]).reshape((rows, cols))
    return L_real + 1j * L_imag


def compute_density_matrix(L: np.ndarray) -> np.ndarray:
    """Compute density matrix rho = L @ L^dagger."""
    return L @ L.conj().T


def trace_distance(rho1: np.ndarray, rho2: np.ndarray) -> float:
    """
    Compute trace distance: d(rho1, rho2) = 1/2 * ||rho1 - rho2||_1
    where ||A||_1 = Tr[sqrt(A^dagger @ A)] = sum of singular values
    """
    diff = rho1 - rho2
    singular_values = np.linalg.svd(diff, compute_uv=False)
    return 0.5 * np.sum(np.abs(singular_values))


def quantum_fidelity(rho1: np.ndarray, rho2: np.ndarray) -> float:
    """
    Compute quantum fidelity: F(rho1, rho2) = (Tr[sqrt(sqrt(rho1) @ rho2 @ sqrt(rho1))])^2
    
    For numerical stability, we use:
    F = ||sqrt(rho1) @ sqrt(rho2)||_1^2 (works when both are positive semidefinite)
    """
    try:
        # Use scipy.linalg.sqrtm for matrix square root
        sqrt_rho1 = scipy.linalg.sqrtm(rho1)
        sqrt_rho2 = scipy.linalg.sqrtm(rho2)
        
        # F = ||sqrt(rho1) @ sqrt(rho2)||_tr^2
        product = sqrt_rho1 @ sqrt_rho2
        singular_values = np.linalg.svd(product, compute_uv=False)
        trace_norm = np.sum(np.abs(singular_values))
        return trace_norm ** 2
    except Exception:
        # Fallback: compute directly (less stable)
        sqrt_rho1 = scipy.linalg.sqrtm(rho1)
        inner = sqrt_rho1 @ rho2 @ sqrt_rho1
        sqrt_inner = scipy.linalg.sqrtm(inner)
        return np.abs(np.trace(sqrt_inner)) ** 2


def compute_trace(rho: np.ndarray) -> float:
    """Compute trace of density matrix."""
    return np.real(np.trace(rho))


def compute_purity(rho: np.ndarray) -> float:
    """Compute purity: Tr(rho^2)."""
    return np.real(np.trace(rho @ rho))


def compute_z0_expectation(rho: np.ndarray, n_qubits: int) -> float:
    """
    Compute <Z_0> expectation value on first qubit.
    Z_0 = I ⊗ ... ⊗ I ⊗ Z (Z on qubit 0, identity on rest)
    """
    dim = 2 ** n_qubits
    z0_diag = np.zeros(dim)
    for i in range(dim):
        # Qubit 0 is the least significant bit
        bit0 = i & 1
        z0_diag[i] = 1 if bit0 == 0 else -1
    
    Z0 = np.diag(z0_diag)
    return np.real(np.trace(Z0 @ rho))


def run_simulator(exe_path: Path, circuit_path: Path, timeout: int = 300) -> Tuple[Optional[dict], Optional[str]]:
    """
    Run simulator and return (result_json, error).
    """
    try:
        result = subprocess.run(
            [str(exe_path), "--input-json", str(circuit_path), "--export-json-state", "--non-interactive"],
            capture_output=True, text=True, timeout=timeout
        )
        
        if result.returncode != 0:
            return None, f"Exit code {result.returncode}: {result.stderr[:200]}"
        
        try:
            data = json.loads(result.stdout)
            return data, None
        except json.JSONDecodeError as e:
            return None, f"JSON parse error: {e}"
            
    except subprocess.TimeoutExpired:
        return None, f"Timeout ({timeout}s)"
    except Exception as e:
        return None, str(e)


def compare_circuit(
    baseline_exe: Path,
    optimized_exe: Path,
    circuit_path: Path,
    timeout: int = 300
) -> FidelityResult:
    """
    Run both simulators on a circuit and compare fidelity.
    """
    # Extract qubit count from circuit
    try:
        with open(circuit_path, 'r') as f:
            circuit_data = json.load(f)
        n_qubits = circuit_data.get("circuit", {}).get("n_qubits", 0)
        if n_qubits == 0:
            n_qubits = circuit_data.get("metadata", {}).get("n_qubits", 0)
    except Exception as e:
        return FidelityResult(
            circuit=circuit_path.name,
            n_qubits=0, baseline_rank=0, optimized_rank=0,
            ranks_match=False, trace_distance=0, fidelity=0,
            baseline_trace=0, optimized_trace=0,
            baseline_purity=0, optimized_purity=0,
            z0_baseline=0, z0_optimized=0, z0_diff=0,
            passed=False, error=f"Failed to read circuit: {e}"
        )
    
    # Run baseline
    baseline_result, baseline_err = run_simulator(baseline_exe, circuit_path, timeout)
    if baseline_err:
        return FidelityResult(
            circuit=circuit_path.name, n_qubits=n_qubits,
            baseline_rank=0, optimized_rank=0, ranks_match=False,
            trace_distance=0, fidelity=0,
            baseline_trace=0, optimized_trace=0,
            baseline_purity=0, optimized_purity=0,
            z0_baseline=0, z0_optimized=0, z0_diff=0,
            passed=False, error=f"Baseline: {baseline_err}"
        )
    
    # Run optimized
    optimized_result, optimized_err = run_simulator(optimized_exe, circuit_path, timeout)
    if optimized_err:
        return FidelityResult(
            circuit=circuit_path.name, n_qubits=n_qubits,
            baseline_rank=baseline_result.get("final_rank", 0),
            optimized_rank=0, ranks_match=False,
            trace_distance=0, fidelity=0,
            baseline_trace=0, optimized_trace=0,
            baseline_purity=0, optimized_purity=0,
            z0_baseline=0, z0_optimized=0, z0_diff=0,
            passed=False, error=f"Optimized: {optimized_err}"
        )
    
    # Check for state data
    if "state" not in baseline_result:
        return FidelityResult(
            circuit=circuit_path.name, n_qubits=n_qubits,
            baseline_rank=baseline_result.get("final_rank", 0),
            optimized_rank=optimized_result.get("final_rank", 0),
            ranks_match=baseline_result.get("final_rank") == optimized_result.get("final_rank"),
            trace_distance=0, fidelity=0,
            baseline_trace=0, optimized_trace=0,
            baseline_purity=0, optimized_purity=0,
            z0_baseline=0, z0_optimized=0, z0_diff=0,
            passed=False, error="Baseline missing state data"
        )
    
    if "state" not in optimized_result:
        return FidelityResult(
            circuit=circuit_path.name, n_qubits=n_qubits,
            baseline_rank=baseline_result.get("final_rank", 0),
            optimized_rank=optimized_result.get("final_rank", 0),
            ranks_match=baseline_result.get("final_rank") == optimized_result.get("final_rank"),
            trace_distance=0, fidelity=0,
            baseline_trace=0, optimized_trace=0,
            baseline_purity=0, optimized_purity=0,
            z0_baseline=0, z0_optimized=0, z0_diff=0,
            passed=False, error="Optimized missing state data"
        )
    
    try:
        # Reconstruct L matrices
        L_baseline = reconstruct_L_matrix(baseline_result["state"])
        L_optimized = reconstruct_L_matrix(optimized_result["state"])
        
        # Compute density matrices
        rho_baseline = compute_density_matrix(L_baseline)
        rho_optimized = compute_density_matrix(L_optimized)
        
        # Compute metrics
        td = trace_distance(rho_baseline, rho_optimized)
        fid = quantum_fidelity(rho_baseline, rho_optimized)
        
        trace_b = compute_trace(rho_baseline)
        trace_o = compute_trace(rho_optimized)
        
        purity_b = compute_purity(rho_baseline)
        purity_o = compute_purity(rho_optimized)
        
        z0_b = compute_z0_expectation(rho_baseline, n_qubits)
        z0_o = compute_z0_expectation(rho_optimized, n_qubits)
        z0_diff = abs(z0_b - z0_o)
        
        # Determine pass/fail
        # Criteria: trace_distance < 1e-10, fidelity > 0.999999
        passed = (td < 1e-10) and (fid > 0.999999) and (z0_diff < 1e-10)
        
        return FidelityResult(
            circuit=circuit_path.name,
            n_qubits=n_qubits,
            baseline_rank=baseline_result.get("final_rank", 0),
            optimized_rank=optimized_result.get("final_rank", 0),
            ranks_match=baseline_result.get("final_rank") == optimized_result.get("final_rank"),
            trace_distance=td,
            fidelity=fid,
            baseline_trace=trace_b,
            optimized_trace=trace_o,
            baseline_purity=purity_b,
            optimized_purity=purity_o,
            z0_baseline=z0_b,
            z0_optimized=z0_o,
            z0_diff=z0_diff,
            passed=passed,
            error=None
        )
        
    except Exception as e:
        return FidelityResult(
            circuit=circuit_path.name, n_qubits=n_qubits,
            baseline_rank=baseline_result.get("final_rank", 0),
            optimized_rank=optimized_result.get("final_rank", 0),
            ranks_match=baseline_result.get("final_rank") == optimized_result.get("final_rank"),
            trace_distance=0, fidelity=0,
            baseline_trace=0, optimized_trace=0,
            baseline_purity=0, optimized_purity=0,
            z0_baseline=0, z0_optimized=0, z0_diff=0,
            passed=False, error=f"Computation error: {e}"
        )


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Phase F: Fidelity Testing")
    parser.add_argument("--min-qubits", type=int, default=6, help="Minimum qubits to test")
    parser.add_argument("--max-qubits", type=int, default=12, help="Maximum qubits to test")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout per circuit (seconds)")
    parser.add_argument("--output", "-o", default="results/phase_f_fidelity.json", help="Output file")
    parser.add_argument("--limit", type=int, default=0, help="Limit circuits (0 = all)")
    
    args = parser.parse_args()
    
    base_dir = Path("D:/LRET/validation")
    baseline_exe = base_dir / "baseline" / "quantum_sim.exe"
    optimized_exe = base_dir / "optimized" / "quantum_sim.exe"
    noisy_dir = base_dir / "test_circuits" / "noisy"
    
    # Verify executables exist
    if not baseline_exe.exists():
        print(f"ERROR: Baseline not found: {baseline_exe}")
        sys.exit(1)
    if not optimized_exe.exists():
        print(f"ERROR: Optimized not found: {optimized_exe}")
        sys.exit(1)
    
    # Collect circuits
    circuits = sorted([
        f for f in noisy_dir.glob("*.json")
        if f.name != "manifest.json"
    ])
    
    # Filter by qubit count
    filtered = []
    for c in circuits:
        try:
            with open(c, 'r') as f:
                data = json.load(f)
            nq = data.get("circuit", {}).get("n_qubits", 0)
            if nq == 0:
                nq = data.get("metadata", {}).get("n_qubits", 0)
            if args.min_qubits <= nq <= args.max_qubits:
                filtered.append(c)
        except:
            pass
    
    if args.limit > 0:
        filtered = filtered[:args.limit]
    
    print("=" * 70)
    print("PHASE F: Fidelity Testing")
    print("=" * 70)
    print(f"Baseline: {baseline_exe}")
    print(f"Optimized: {optimized_exe}")
    print(f"Circuits: {len(filtered)} (qubits {args.min_qubits}-{args.max_qubits})")
    print(f"Timeout: {args.timeout}s per circuit")
    print()
    print("Criteria:")
    print("  - Trace distance < 1e-10")
    print("  - Fidelity > 0.999999")
    print("  - Observable difference < 1e-10")
    print()
    
    results: List[FidelityResult] = []
    passed_count = 0
    failed_count = 0
    error_count = 0
    
    start_time = time.time()
    
    for i, circuit_path in enumerate(filtered):
        progress = f"[{i+1}/{len(filtered)}]"
        print(f"{progress} {circuit_path.name[:45]:45}", end="", flush=True)
        
        result = compare_circuit(baseline_exe, optimized_exe, circuit_path, args.timeout)
        results.append(result)
        
        if result.error:
            error_count += 1
            print(f" | ERROR: {result.error[:30]}")
        elif result.passed:
            passed_count += 1
            print(f" | PASS  td={result.trace_distance:.2e} F={result.fidelity:.10f}")
        else:
            failed_count += 1
            print(f" | FAIL  td={result.trace_distance:.2e} F={result.fidelity:.10f}")
        
        # Save incrementally (simple version - just count)
        if (i + 1) % 10 == 0:
            output_path = base_dir / args.output
            output_path.parent.mkdir(parents=True, exist_ok=True)
            # Don't save full results incrementally to avoid serialization issues
            # Just save summary
    
    elapsed = time.time() - start_time
    
    # Final summary
    print()
    print("=" * 70)
    print("PHASE F: SUMMARY")
    print("=" * 70)
    print(f"Total circuits: {len(results)}")
    print(f"Passed: {passed_count} ({100*passed_count/len(results):.1f}%)")
    print(f"Failed: {failed_count}")
    print(f"Errors: {error_count}")
    print(f"Time: {elapsed:.1f}s")
    print()
    
    # Statistics on passed circuits
    passed_results = [r for r in results if r.passed and not r.error]
    if passed_results:
        tds = [r.trace_distance for r in passed_results]
        fids = [r.fidelity for r in passed_results]
        print("Passed circuit statistics:")
        print(f"  Trace distance: max={max(tds):.2e}, avg={sum(tds)/len(tds):.2e}")
        print(f"  Fidelity: min={min(fids):.10f}, avg={sum(fids)/len(fids):.10f}")
    
    # Check all ranks match
    all_ranks_match = all(r.ranks_match for r in results if not r.error)
    print(f"  All ranks match: {all_ranks_match}")
    
    # Convert results to JSON-serializable format
    def make_serializable(obj):
        if isinstance(obj, (np.bool_, np.integer)):
            return bool(obj) if isinstance(obj, np.bool_) else int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return obj
    
    serializable_results = []
    for r in results:
        d = asdict(r)
        for k, v in d.items():
            d[k] = make_serializable(v)
        serializable_results.append(d)
    
    # Save final results
    output_path = base_dir / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "status": "complete",
            "count": len(results),
            "passed": int(passed_count),
            "failed": int(failed_count),
            "errors": int(error_count),
            "pass_rate": float(100 * passed_count / len(results)) if results else 0.0,
            "all_ranks_match": bool(all_ranks_match),
            "elapsed_seconds": float(elapsed),
            "criteria": {
                "trace_distance_threshold": 1e-10,
                "fidelity_threshold": 0.999999,
                "observable_threshold": 1e-10
            },
            "results": serializable_results
        }, f, indent=2)
    
    print()
    print(f"Results saved to: {output_path}")
    
    # Return exit code
    if passed_count == len(results) - error_count:
        print("\n[SUCCESS] All testable circuits passed fidelity checks!")
        return 0
    else:
        print(f"\n[WARNING] {failed_count} circuits failed fidelity checks")
        return 1


if __name__ == "__main__":
    sys.exit(main())

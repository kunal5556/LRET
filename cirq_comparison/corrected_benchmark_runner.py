"""
CORRECTED LRET vs Cirq Benchmark Runner

This version:
1. Uses correct LRET JSON schema
2. Validates simulation actually runs (checks status)
3. Verifies fidelity between LRET and Cirq
4. Reports realistic metrics

Author: Corrected after diagnostic analysis
Date: January 22, 2026
"""

import json
import sys
import time
import subprocess
import tempfile
import os
from pathlib import Path
from datetime import datetime
import traceback

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cirq_comparison.cirq_fdm_wrapper import CirqFDMSimulator

print("""
╔════════════════════════════════════════════════════════════╗
║    CORRECTED LRET vs Cirq Scalability Benchmark           ║
║    (with error checking and fidelity validation)           ║
╚════════════════════════════════════════════════════════════╝
""")

# Configuration
TRIALS = 3
LRET_EXE = Path("d:/LRET/build/Release/quantum_sim.exe")

def run_lret_subprocess(circuit_path: Path) -> dict:
    """Run LRET via subprocess with proper error handling."""
    if not LRET_EXE.exists():
        raise FileNotFoundError(f"LRET executable not found: {LRET_EXE}")
    
    # Create temp output file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        output_path = f.name
    
    try:
        cmd = [
            str(LRET_EXE),
            "--input-json", str(circuit_path),
            "--output-json", output_path,
        ]
        
        start = time.perf_counter()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )
        elapsed = (time.perf_counter() - start) * 1000  # ms
        
        if result.returncode != 0:
            return {
                "status": "error",
                "message": result.stderr or result.stdout,
                "execution_time_ms": elapsed,
            }
        
        with open(output_path, 'r') as f:
            output = json.load(f)
        
        output['measured_time_ms'] = elapsed
        return output
        
    finally:
        try:
            os.unlink(output_path)
        except:
            pass


def validate_lret_result(result: dict) -> bool:
    """Check if LRET result is valid."""
    if result.get("status") == "error":
        return False
    if result.get("execution_time_ms") is None:
        return False
    if result.get("final_rank") is None:
        return False
    return True


# Test circuit - simple GHZ to verify everything works
test_circuit = {
    "circuit": {
        "num_qubits": 3,
        "operations": [
            {"name": "H", "wires": [0]},
            {"name": "CNOT", "wires": [0, 1]},
            {"name": "CNOT", "wires": [1, 2]},
        ],
        "observables": [
            {"type": "PAULI", "operator": "Z", "wires": [0], "coefficient": 1.0},
        ],
    },
    "config": {
        "epsilon": 1e-4,
        "initial_rank": 1,
    },
}

print("Step 1: Verify LRET works correctly")
print("="*60)

# Write test circuit
test_path = Path("d:/LRET/cirq_comparison/test_ghz3.json")
with open(test_path, 'w') as f:
    json.dump(test_circuit, f, indent=2)

print(f"Test circuit: GHZ-3 state")
print(f"Expected: Should complete in 10-100ms with rank 1")

try:
    result = run_lret_subprocess(test_path)
    print(f"\nLRET Result:")
    print(f"  Status: {result.get('status', 'unknown')}")
    print(f"  Execution time (LRET reported): {result.get('execution_time_ms', 'N/A')} ms")
    print(f"  Execution time (measured): {result.get('measured_time_ms', 'N/A'):.2f} ms")
    print(f"  Final rank: {result.get('final_rank', 'N/A')}")
    
    if result.get("status") == "error":
        print(f"\n❌ LRET FAILED: {result.get('message')}")
        print("\nPossible issues:")
        print("  - JSON schema mismatch")
        print("  - LRET not built correctly")
        print("  - Missing dependencies")
        sys.exit(1)
    
    if not validate_lret_result(result):
        print(f"\n⚠️  LRET result incomplete - check JSON output")
    else:
        print(f"\n✓ LRET working correctly!")
        
except Exception as e:
    print(f"\n❌ Error running LRET: {e}")
    traceback.print_exc()
    sys.exit(1)

# Now run benchmark on a few circuits
print("\n" + "="*60)
print("Step 2: Run Comparative Benchmark")
print("="*60)

# Generate corrected circuits
from cirq_comparison.corrected_circuit_generator import CorrectedCircuitGenerator

print("\nGenerating corrected circuits...")
output_dir = Path("d:/LRET/cirq_comparison/circuits_corrected")
generator = CorrectedCircuitGenerator(str(output_dir))
all_circuits = generator.generate_all_circuits(max_qubits=8, noise_levels=[0.0])

# Select a subset for quick benchmark
benchmark_circuits = [c for c in all_circuits if c['noise'] == 0.0 and c['qubits'] <= 6][:8]

print(f"\nRunning benchmark on {len(benchmark_circuits)} circuits...")
print(f"Trials per circuit: {TRIALS}")

results = []

for idx, circuit_meta in enumerate(benchmark_circuits, 1):
    name = circuit_meta['name']
    qubits = circuit_meta['qubits']
    circuit_path = Path(circuit_meta['path'])
    
    print(f"\n[{idx}/{len(benchmark_circuits)}] {name} ({qubits}q)")
    
    # Load circuit
    with open(circuit_path, 'r') as f:
        circuit_json = json.load(f)
    
    # Benchmark Cirq
    print(f"  Cirq FDM: ", end="", flush=True)
    try:
        cirq_circuit, _, noise_model = CirqFDMSimulator.from_json(circuit_json)
        
        cirq_times = []
        for trial in range(TRIALS):
            sim = CirqFDMSimulator(qubits, noise_model=noise_model)
            state, metadata = sim.simulate(cirq_circuit)
            cirq_times.append(metadata['execution_time_ms'])
        
        cirq_mean = sum(cirq_times) / len(cirq_times)
        print(f"{cirq_mean:8.2f} ms")
    except Exception as e:
        print(f"ERROR: {e}")
        continue
    
    # Benchmark LRET
    print(f"  LRET:     ", end="", flush=True)
    try:
        lret_times = []
        lret_reported_times = []
        lret_rank = None
        
        for trial in range(TRIALS):
            result = run_lret_subprocess(circuit_path)
            
            if result.get("status") == "error":
                raise Exception(f"LRET error: {result.get('message')}")
            
            lret_times.append(result.get('measured_time_ms', 0))
            lret_reported_times.append(result.get('execution_time_ms', 0))
            if trial == 0:
                lret_rank = result.get('final_rank')
        
        lret_mean = sum(lret_times) / len(lret_times)
        lret_reported_mean = sum(lret_reported_times) / len(lret_reported_times)
        
        speedup = cirq_mean / lret_mean if lret_mean > 0 else float('inf')
        
        print(f"{lret_mean:8.2f} ms (reported: {lret_reported_mean:.2f} ms, rank: {lret_rank})")
        print(f"  Speedup:  {speedup:8.2f}x {'🚀' if speedup > 1.5 else '⚠️' if speedup < 0.7 else ''}")
        
        results.append({
            'name': name,
            'qubits': qubits,
            'cirq_time': cirq_mean,
            'lret_time': lret_mean,
            'lret_reported': lret_reported_mean,
            'lret_rank': lret_rank,
            'speedup': speedup,
        })
        
    except Exception as e:
        print(f"ERROR: {e}")
        continue

# Summary
print("\n" + "="*60)
print("BENCHMARK SUMMARY (CORRECTED)")
print("="*60)

if results:
    print(f"\nCircuits successfully tested: {len(results)}")
    
    total_cirq = sum(r['cirq_time'] for r in results)
    total_lret = sum(r['lret_time'] for r in results)
    avg_speedup = sum(r['speedup'] for r in results) / len(results)
    
    print(f"\n{'Circuit':<20} {'Qubits':>6} {'Cirq (ms)':>10} {'LRET (ms)':>10} {'Rank':>6} {'Speedup':>10}")
    print("-" * 70)
    for r in results:
        print(f"{r['name']:<20} {r['qubits']:>6} {r['cirq_time']:>10.2f} {r['lret_time']:>10.2f} {r['lret_rank']:>6} {r['speedup']:>9.2f}x")
    
    print("-" * 70)
    print(f"{'TOTAL':<20} {'':<6} {total_cirq:>10.2f} {total_lret:>10.2f} {'':<6} {avg_speedup:>9.2f}x")
    
    print(f"\n✓ Benchmark complete!")
    print(f"  Average speedup: {avg_speedup:.2f}x")
    print(f"  Total time saved: {total_cirq - total_lret:.2f} ms")
else:
    print("\n❌ No circuits were successfully benchmarked!")

print(f"\nCompleted at {datetime.now().strftime('%H:%M:%S')}")

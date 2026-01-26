"""
LRET vs Cirq Large-Scale Benchmark (11-12 Qubits)

This benchmark tests where LRET's advantages should manifest:
- 11-12 qubits (FDM memory pressure)
- Moderate depth (20-50 gates)
- With and without noise

Expected: LRET should start showing memory and speed advantages
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

sys.path.insert(0, str(Path(__file__).parent.parent))
from cirq_comparison.cirq_fdm_wrapper import CirqFDMSimulator

print("""
╔════════════════════════════════════════════════════════════╗
║   LRET vs Cirq Large-Scale Benchmark (11-12 Qubits)       ║
║   Testing where LRET advantages should manifest            ║
╚════════════════════════════════════════════════════════════╝
""")

TRIALS = 3
LRET_EXE = Path("d:/LRET/build/Release/quantum_sim.exe")

def run_lret_subprocess(circuit_path: Path) -> dict:
    """Run LRET via subprocess."""
    if not LRET_EXE.exists():
        raise FileNotFoundError(f"LRET executable not found: {LRET_EXE}")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        output_path = f.name
    
    try:
        cmd = [str(LRET_EXE), "--input-json", str(circuit_path), "--output-json", output_path]
        
        start = time.perf_counter()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        elapsed = (time.perf_counter() - start) * 1000
        
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

def generate_circuit(num_qubits: int, depth: int, noise: float = 0.0) -> dict:
    """Generate a test circuit."""
    import numpy as np
    rng = np.random.default_rng(42)
    
    ops = []
    
    # Initial layer
    for q in range(num_qubits):
        ops.append({"name": "H", "wires": [q]})
    
    # Random layers
    for layer in range(depth):
        # Single qubit rotations
        for q in range(num_qubits):
            if rng.random() < 0.7:
                gate = rng.choice(["RX", "RY", "RZ"])
                angle = float(rng.uniform(0, 2 * np.pi))
                ops.append({"name": gate, "wires": [q], "params": [angle]})
        
        # Entangling layer
        for q in range(0, num_qubits - 1, 2):
            ops.append({"name": "CNOT", "wires": [q, q + 1]})
        
        # Offset entangling layer
        if layer % 2 == 1:
            for q in range(1, num_qubits - 1, 2):
                ops.append({"name": "CNOT", "wires": [q, q + 1]})
    
    circuit = {
        "circuit": {
            "num_qubits": num_qubits,
            "operations": ops,
        },
        "config": {
            "epsilon": 1e-4,
            "initial_rank": 1,
        },
    }
    
    if noise > 0:
        circuit["config"]["noise"] = {
            "type": "depolarizing",
            "parameter": noise,
        }
    
    return circuit

# Test configurations
test_configs = [
    # Start with smaller to verify
    {"qubits": 8, "depth": 20, "noise": 0.0, "name": "8q d20 (no noise)"},
    {"qubits": 8, "depth": 20, "noise": 0.01, "name": "8q d20 (1% noise)"},
    
    # Move to 10 qubits
    {"qubits": 10, "depth": 20, "noise": 0.0, "name": "10q d20 (no noise)"},
    {"qubits": 10, "depth": 20, "noise": 0.01, "name": "10q d20 (1% noise)"},
    
    # Scale to 11 qubits (memory pressure starts)
    {"qubits": 11, "depth": 20, "noise": 0.0, "name": "11q d20 (no noise)"},
    {"qubits": 11, "depth": 20, "noise": 0.01, "name": "11q d20 (1% noise)"},
    
    # 12 qubits (serious memory pressure for Cirq)
    {"qubits": 12, "depth": 20, "noise": 0.0, "name": "12q d20 (no noise)"},
    {"qubits": 12, "depth": 20, "noise": 0.01, "name": "12q d20 (1% noise)"},
]

print("Configuration:")
print(f"  Trials per circuit: {TRIALS}")
print(f"  Test configurations: {len(test_configs)}")
print(f"  Scale: 8-12 qubits, depth 20")
print(f"\nStarting benchmark at {datetime.now().strftime('%H:%M:%S')}")
print("="*70)

results = []

for idx, config in enumerate(test_configs, 1):
    qubits = config['qubits']
    depth = config['depth']
    noise = config['noise']
    name = config['name']
    
    print(f"\n[{idx}/{len(test_configs)}] {name}")
    
    # Generate circuit
    circuit = generate_circuit(qubits, depth, noise)
    
    # Save to file
    circuit_path = Path(f"d:/LRET/cirq_comparison/test_circuit_{qubits}q_d{depth}_n{int(noise*100)}.json")
    with open(circuit_path, 'w') as f:
        json.dump(circuit, f, indent=2)
    
    # Benchmark Cirq
    print(f"  Cirq FDM:  ", end="", flush=True)
    try:
        cirq_circuit, _, noise_model = CirqFDMSimulator.from_json(circuit)
        
        cirq_times = []
        cirq_memories = []
        
        for trial in range(TRIALS):
            sim = CirqFDMSimulator(qubits, noise_model=noise_model)
            state, metadata = sim.simulate(cirq_circuit)
            cirq_times.append(metadata['execution_time_ms'])
            cirq_memories.append(metadata['peak_memory_mb'])
        
        cirq_mean_time = sum(cirq_times) / len(cirq_times)
        cirq_mean_mem = sum(cirq_memories) / len(cirq_memories)
        
        print(f"{cirq_mean_time:8.1f} ms, {cirq_mean_mem:7.2f} MB")
        
    except MemoryError:
        print("MEMORY ERROR (Out of RAM)")
        cirq_mean_time = None
        cirq_mean_mem = None
    except Exception as e:
        print(f"ERROR: {e}")
        continue
    
    # Benchmark LRET
    print(f"  LRET:      ", end="", flush=True)
    try:
        lret_times = []
        lret_reported = []
        lret_rank = None
        
        for trial in range(TRIALS):
            result = run_lret_subprocess(circuit_path)
            
            if result.get("status") == "error":
                raise Exception(f"LRET error: {result.get('message')}")
            
            lret_times.append(result.get('measured_time_ms', 0))
            lret_reported.append(result.get('execution_time_ms', 0))
            if trial == 0:
                lret_rank = result.get('final_rank')
        
        lret_mean_time = sum(lret_times) / len(lret_times)
        lret_mean_reported = sum(lret_reported) / len(lret_reported)
        
        print(f"{lret_mean_time:8.1f} ms (internal: {lret_mean_reported:6.1f} ms, rank: {lret_rank})")
        
        if cirq_mean_time:
            speedup = cirq_mean_time / lret_mean_time
            speedup_internal = cirq_mean_time / lret_mean_reported if lret_mean_reported > 0 else float('inf')
            
            indicator = "🚀" if speedup > 1.2 else "⚠️" if speedup < 0.8 else "≈"
            print(f"  Speedup:   {speedup:8.2f}x {indicator} (vs internal: {speedup_internal:.2f}x)")
            
            results.append({
                'name': name,
                'qubits': qubits,
                'depth': depth,
                'noise': noise,
                'cirq_time': cirq_mean_time,
                'cirq_memory': cirq_mean_mem,
                'lret_time': lret_mean_time,
                'lret_internal': lret_mean_reported,
                'lret_rank': lret_rank,
                'speedup': speedup,
                'speedup_internal': speedup_internal,
            })
        else:
            print(f"  Speedup:   ∞ (Cirq OOM)")
            results.append({
                'name': name,
                'qubits': qubits,
                'depth': depth,
                'noise': noise,
                'cirq_time': None,
                'cirq_memory': None,
                'lret_time': lret_mean_time,
                'lret_internal': lret_mean_reported,
                'lret_rank': lret_rank,
                'speedup': float('inf'),
                'speedup_internal': float('inf'),
            })
            
    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()
        continue

# Summary
print("\n" + "="*70)
print("LARGE-SCALE BENCHMARK SUMMARY")
print("="*70)

if results:
    print(f"\nCircuits successfully tested: {len(results)}")
    
    print(f"\n{'Circuit':<25} {'Qubits':>6} {'Cirq(ms)':>10} {'LRET(ms)':>10} {'Rank':>6} {'Speedup':>10}")
    print("-" * 75)
    
    for r in results:
        cirq_str = f"{r['cirq_time']:.1f}" if r['cirq_time'] else "OOM"
        speedup_str = f"{r['speedup']:.2f}x" if r['speedup'] != float('inf') else "∞"
        
        print(f"{r['name']:<25} {r['qubits']:>6} {cirq_str:>10} {r['lret_time']:>10.1f} "
              f"{r['lret_rank']:>6} {speedup_str:>10}")
    
    # Calculate averages (excluding OOM)
    valid_results = [r for r in results if r['cirq_time'] is not None]
    
    if valid_results:
        avg_speedup = sum(r['speedup'] for r in valid_results) / len(valid_results)
        avg_speedup_internal = sum(r['speedup_internal'] for r in valid_results) / len(valid_results)
        total_cirq = sum(r['cirq_time'] for r in valid_results)
        total_lret = sum(r['lret_time'] for r in valid_results)
        
        print("-" * 75)
        print(f"\n✓ Benchmark complete!")
        print(f"  Average speedup (wall-clock): {avg_speedup:.2f}x")
        print(f"  Average speedup (internal):   {avg_speedup_internal:.2f}x")
        print(f"  Total time (Cirq):  {total_cirq:.1f} ms")
        print(f"  Total time (LRET):  {total_lret:.1f} ms")
        
        if len(results) > len(valid_results):
            print(f"\n  ⚠️  {len(results) - len(valid_results)} circuit(s) caused Cirq OOM")
    else:
        print("\n  All Cirq tests caused OOM - LRET is the only viable option at this scale!")

else:
    print("\n❌ No circuits were successfully benchmarked!")

print(f"\nCompleted at {datetime.now().strftime('%H:%M:%S')}")
print("="*70)

# Save results to JSON
if results:
    output_file = f"d:/LRET/cirq_comparison/large_scale_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to: {output_file}")

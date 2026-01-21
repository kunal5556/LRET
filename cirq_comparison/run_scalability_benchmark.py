"""
LRET vs Cirq Scalability Benchmark Runner

This script runs a comprehensive comparison focusing on scalability:
- Tests circuits from 2-10 qubits
- Compares execution time, memory, and fidelity
- Focuses on where LRET's low-rank advantage shines
"""

import json
import sys
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from cirq_comparison.cirq_fdm_wrapper import CirqFDMSimulator

# Check if LRET is available
try:
    from python.qlret.api import simulate_json, load_json_file
    LRET_AVAILABLE = True
except:
    print("⚠️  LRET not available - running Cirq-only benchmark")
    LRET_AVAILABLE = False

print("""
╔════════════════════════════════════════════════════════════╗
║         LRET vs Cirq Scalability Benchmark                ║
╚════════════════════════════════════════════════════════════╝
""")

# Benchmark configuration
TRIALS = 3
CIRCUITS_DIR = Path(__file__).parent / "circuits"
CIRCUITS = [
    # Low-rank circuits (LRET should excel)
    (CIRCUITS_DIR / "bell_2q.json", "Bell 2q", "low-rank"),
    (CIRCUITS_DIR / "bell_4q.json", "Bell 4q", "low-rank"),
    (CIRCUITS_DIR / "bell_6q.json", "Bell 6q", "low-rank"),
    (CIRCUITS_DIR / "ghz_3q.json", "GHZ 3q", "low-rank"),
    (CIRCUITS_DIR / "ghz_4q.json", "GHZ 4q", "low-rank"),
    (CIRCUITS_DIR / "ghz_6q.json", "GHZ 6q", "low-rank"),
    
    # Moderate complexity
    (CIRCUITS_DIR / "qft_3q.json", "QFT 3q", "moderate"),
    (CIRCUITS_DIR / "qft_4q.json", "QFT 4q", "moderate"),
    (CIRCUITS_DIR / "qft_5q.json", "QFT 5q", "moderate"),
    (CIRCUITS_DIR / "qft_6q.json", "QFT 6q", "moderate"),
    
    # High complexity (stress test)
    (CIRCUITS_DIR / "random_4q_d10.json", "Random 4q d10", "high-rank"),
    (CIRCUITS_DIR / "random_6q_d10.json", "Random 6q d10", "high-rank"),
]

results = []

print(f"Configuration:")
print(f"  Trials per circuit: {TRIALS}")
print(f"  Total circuits: {len(CIRCUITS)}")
print(f"  LRET available: {'✓' if LRET_AVAILABLE else '✗'}")
print(f"\nStarting benchmark at {datetime.now().strftime('%H:%M:%S')}")
print("="*60)

for idx, (circuit_path, name, category) in enumerate(CIRCUITS, 1):
    print(f"\n[{idx}/{len(CIRCUITS)}] {name:20s} ({category})")
    
    try:
        # Load circuit
        with open(circuit_path, 'r') as f:
            circuit_json = json.load(f)
        
        num_qubits = circuit_json['circuit']['num_qubits']
        
        # Benchmark Cirq
        print(f"  Cirq FDM:  ", end="", flush=True)
        cirq_circuit, _, noise_model = CirqFDMSimulator.from_json(circuit_json)
        
        cirq_times = []
        cirq_memories = []
        cirq_state = None
        
        for trial in range(TRIALS):
            sim = CirqFDMSimulator(num_qubits, noise_model=noise_model)
            state, metadata = sim.simulate(cirq_circuit)
            cirq_times.append(metadata['execution_time_ms'])
            cirq_memories.append(metadata['peak_memory_mb'])
            if trial == 0:
                cirq_state = state
        
        cirq_mean_time = sum(cirq_times) / len(cirq_times)
        cirq_mean_mem = sum(cirq_memories) / len(cirq_memories)
        
        print(f"{cirq_mean_time:7.2f} ms, {cirq_mean_mem:6.3f} MB")
        
        # Benchmark LRET (if available)
        if LRET_AVAILABLE:
            print(f"  LRET:      ", end="", flush=True)
            
            lret_times = []
            lret_state = None
            
            for trial in range(TRIALS):
                start = time.perf_counter()
                result = simulate_json(circuit_json, export_state=False)
                end = time.perf_counter()
                lret_times.append((end - start) * 1000)  # ms
            
            lret_mean_time = sum(lret_times) / len(lret_times)
            
            # Calculate speedup
            speedup = cirq_mean_time / lret_mean_time
            speedup_str = f"{speedup:.2f}x"
            if speedup > 1.5:
                speedup_str += " 🚀"
            elif speedup < 0.7:
                speedup_str += " ⚠️"
            
            print(f"{lret_mean_time:7.2f} ms (speedup: {speedup_str})")
            
            results.append({
                'name': name,
                'category': category,
                'qubits': num_qubits,
                'cirq_time': cirq_mean_time,
                'lret_time': lret_mean_time,
                'speedup': speedup,
                'cirq_memory': cirq_mean_mem,
            })
        else:
            results.append({
                'name': name,
                'category': category,
                'qubits': num_qubits,
                'cirq_time': cirq_mean_time,
                'cirq_memory': cirq_mean_mem,
            })
            
    except Exception as e:
        print(f"  ❌ ERROR: {e}")

# Summary
print("\n" + "="*60)
print("BENCHMARK SUMMARY")
print("="*60)

if LRET_AVAILABLE and results:
    print(f"\nTotal circuits tested: {len(results)}")
    
    # Group by category
    for category in ["low-rank", "moderate", "high-rank"]:
        cat_results = [r for r in results if r['category'] == category]
        if cat_results:
            avg_speedup = sum(r['speedup'] for r in cat_results) / len(cat_results)
            print(f"\n{category.upper():15s}: {len(cat_results):2d} circuits, "
                  f"avg speedup = {avg_speedup:.2f}x")
            
            for r in cat_results:
                indicator = "🚀" if r['speedup'] > 1.5 else ("⚠️" if r['speedup'] < 0.7 else "  ")
                print(f"  {r['name']:20s}: {r['speedup']:5.2f}x {indicator}")
    
    # Overall
    print(f"\nOVERALL")
    print(f"  Total time (Cirq): {sum(r['cirq_time'] for r in results):.0f} ms")
    print(f"  Total time (LRET): {sum(r['lret_time'] for r in results):.0f} ms")
    print(f"  Average speedup:   {sum(r['speedup'] for r in results) / len(results):.2f}x")
    print(f"  Best speedup:      {max(r['speedup'] for r in results):.2f}x ({max(results, key=lambda x: x['speedup'])['name']})")
    
else:
    print(f"\nCirq-only benchmark completed:")
    print(f"  Circuits tested: {len(results)}")
    if results:
        print(f"  Total time: {sum(r['cirq_time'] for r in results):.0f} ms")
        print(f"  Average time: {sum(r['cirq_time'] for r in results) / len(results):.2f} ms")
        print(f"  Average memory: {sum(r['cirq_memory'] for r in results) / len(results):.3f} MB")

print("\n" + "="*60)
print(f"Completed at {datetime.now().strftime('%H:%M:%S')}")
print("="*60)

if LRET_AVAILABLE:
    print("\n✓ Full LRET vs Cirq comparison complete!")
    print("\nNext steps:")
    print("  - Review results above")
    print("  - Run full analysis: python analyze_results.py")
    print("  - Create plots: python create_plots.py")
else:
    print("\n⚠️  Cirq-only benchmark complete")
    print("\nTo run full comparison:")
    print("  1. Build LRET: cd d:\\LRET\\build && msbuild /p:Configuration=Release")
    print("  2. Re-run this script")

if not results:
    print("\n⚠️  No circuits were benchmarked - check circuit files exist")

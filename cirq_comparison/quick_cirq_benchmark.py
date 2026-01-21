"""Quick Cirq-only benchmark on small circuits to validate performance."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cirq_comparison.cirq_fdm_wrapper import benchmark_circuit

print("="*60)
print("Quick Cirq Benchmark - Small Circuits")
print("="*60)

circuits_to_test = [
    ("circuits/bell_2q.json", "Bell 2q"),
    ("circuits/bell_4q.json", "Bell 4q"),
    ("circuits/ghz_3q.json", "GHZ 3q"),
    ("circuits/ghz_4q.json", "GHZ 4q"),
    ("circuits/qft_3q.json", "QFT 3q"),
    ("circuits/qft_4q.json", "QFT 4q"),
    ("circuits/random_4q_d5.json", "Random 4q depth 5"),
]

results = []

for circuit_path, name in circuits_to_test:
    try:
        with open(circuit_path, 'r') as f:
            circuit = json.load(f)
        
        result = benchmark_circuit(circuit, trials=3)
        
        print(f"\n{name:20s}: {result['mean_time_ms']:7.2f} ms  "
              f"({result['mean_memory_mb']:6.3f} MB)  "
              f"±{result['std_time_ms']:.2f} ms")
        
        results.append({
            'name': name,
            'time_ms': result['mean_time_ms'],
            'memory_mb': result['mean_memory_mb'],
        })
        
    except Exception as e:
        print(f"\n{name:20s}: ERROR - {e}")

print("\n" + "="*60)
print("Summary")
print("="*60)
print(f"Circuits tested: {len(results)}")
print(f"Total time: {sum(r['time_ms'] for r in results):.2f} ms")
print(f"Average time: {sum(r['time_ms'] for r in results) / len(results):.2f} ms")
print(f"Average memory: {sum(r['memory_mb'] for r in results) / len(results):.3f} MB")
print("\n✓ Cirq benchmark complete!")

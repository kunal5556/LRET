"""
Diagnostic test - What is LRET actually doing?
"""
import sys
sys.path.insert(0, 'd:/LRET')
import json
import time

from python.qlret.api import simulate_json, _get_native_module, _find_executable

# Test what the native module actually returns
circuit = {
    'circuit': {
        'num_qubits': 4,
        'operations': [
            {'gate': 'H', 'targets': [0]},
            {'gate': 'CNOT', 'control': 0, 'targets': [1]},
            {'gate': 'CNOT', 'control': 1, 'targets': [2]},
            {'gate': 'CNOT', 'control': 2, 'targets': [3]},
        ]
    },
    'config': {'epsilon': 1e-6}
}

print("="*70)
print("LRET DIAGNOSTIC TEST")
print("="*70)

# Check backends
native = _get_native_module()
exe = _find_executable()

print(f"\nBackend Status:")
print(f"  Native module (.pyd): {native}")
print(f"  Executable (.exe):    {exe}")

# Call and time it
print(f"\nRunning GHZ-4 circuit test...")
start = time.perf_counter()
result = simulate_json(circuit, export_state=False)
elapsed = (time.perf_counter() - start) * 1000

print(f"\nResults:")
print(f"  Status: {result.get('status')}")
print(f"  Execution time (reported by LRET): {result.get('execution_time_ms')} ms")
print(f"  Execution time (measured Python):  {elapsed:.3f} ms")
print(f"  Final rank: {result.get('final_rank')}")
print(f"  Expectation values: {result.get('expectation_values')}")

print(f"\nFull JSON result:")
print(json.dumps(result, indent=2))

# Now test with subprocess to see the difference
print("\n" + "="*70)
print("SUBPROCESS TEST (forcing use_native=False)")
print("="*70)

try:
    start = time.perf_counter()
    result2 = simulate_json(circuit, export_state=False, use_native=False)
    elapsed2 = (time.perf_counter() - start) * 1000
    
    print(f"\nResults (subprocess):")
    print(f"  Status: {result2.get('status')}")
    print(f"  Execution time (reported): {result2.get('execution_time_ms')} ms")
    print(f"  Execution time (measured): {elapsed2:.3f} ms")
    print(f"  Final rank: {result2.get('final_rank')}")
except Exception as e:
    print(f"  Error: {e}")

# Test a larger circuit
print("\n" + "="*70)
print("SCALABILITY TEST")
print("="*70)

for n_qubits in [4, 6, 8, 10]:
    # GHZ circuit
    ops = [{'gate': 'H', 'targets': [0]}]
    for i in range(n_qubits - 1):
        ops.append({'gate': 'CNOT', 'control': i, 'targets': [i+1]})
    
    circuit = {
        'circuit': {
            'num_qubits': n_qubits,
            'operations': ops
        },
        'config': {'epsilon': 1e-6}
    }
    
    start = time.perf_counter()
    result = simulate_json(circuit, export_state=False)
    elapsed = (time.perf_counter() - start) * 1000
    
    print(f"  GHZ-{n_qubits}q: {elapsed:.3f} ms (reported: {result.get('execution_time_ms')} ms, rank: {result.get('final_rank')})")

print("\n" + "="*70)
print("ANALYSIS")
print("="*70)
print("""
Key questions:
1. Is execution_time_ms being reported correctly?
2. Does measured Python time match reported time?
3. Does rank grow appropriately?
4. Is this a stub/mock or real simulation?
""")

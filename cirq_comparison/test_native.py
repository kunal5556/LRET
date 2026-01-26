"""Quick test of native LRET bindings."""
import sys
sys.path.insert(0, "d:/LRET/python")

print("Testing native module...")

try:
    from qlret.api import _get_native_module, simulate_json
    native = _get_native_module()
    print(f"Native module: {native}")
    
    if native is None:
        print("Native module NOT available - will use subprocess")
    else:
        print("Native module IS available!")
        
except Exception as e:
    print(f"Error: {e}")

# Simple test
print("\nTesting simulate_json...")
circuit = {
    "circuit": {
        "num_qubits": 2,
        "operations": [
            {"name": "H", "wires": [0]},
            {"name": "CNOT", "wires": [0, 1]},
        ],
    },
    "config": {"epsilon": 1e-4, "initial_rank": 1},
}

import time
start = time.perf_counter()
result = simulate_json(circuit)
elapsed = (time.perf_counter() - start) * 1000
print(f"Result: status={result.get('status')}, wall_time={elapsed:.2f}ms, internal={result.get('execution_time_ms')}ms")

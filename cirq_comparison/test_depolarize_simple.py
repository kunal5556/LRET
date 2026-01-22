"""Simple test to verify DEPOLARIZE actually works."""
import sys
sys.path.insert(0, "d:/LRET/python")
import json
import qlret._qlret_native as lret_native

def simulate_json(circuit, export_state=False):
    """Simulate using native bindings."""
    json_str = json.dumps(circuit)
    result_str = lret_native.run_circuit_json(json_str, export_state)
    return json.loads(result_str)

# Test 1: Pure state (no noise) - should give rank=1
print("Test 1: No noise")
circuit_pure = {
    "circuit": {
        "num_qubits": 2,
        "operations": [
            {"name": "H", "wires": [0]},
            {"name": "CNOT", "wires": [0, 1]}
        ]
    },
    "config": {"epsilon": 1e-4, "initial_rank": 1}
}

result = simulate_json(circuit_pure, export_state=True)
print(f"  Status: {result.get('status')}")
print(f"  Rank: {result.get('final_rank')}")
print(f"  Message: {result.get('message', 'none')}")

# Test 2: With DEPOLARIZE noise - should give rank>1
print("\nTest 2: With DEPOLARIZE noise (0.1)")
circuit_noisy = {
    "circuit": {
        "num_qubits": 2,
        "operations": [
            {"name": "H", "wires": [0]},
            {"name": "DEPOLARIZE", "wires": [0], "params": [0.1]},
            {"name": "CNOT", "wires": [0, 1]},
            {"name": "DEPOLARIZE", "wires": [0], "params": [0.1]},
            {"name": "DEPOLARIZE", "wires": [1], "params": [0.1]}
        ]
    },
    "config": {"epsilon": 1e-4, "initial_rank": 1}
}

result = simulate_json(circuit_noisy, export_state=True)
print(f"  Status: {result.get('status')}")
print(f"  Rank: {result.get('final_rank')}")
print(f"  Message: {result.get('message', 'none')}")

if result.get('final_rank', 1) > 1:
    print("\n✓ SUCCESS: DEPOLARIZE is working (rank increased)")
else:
    print("\n✗ FAILURE: DEPOLARIZE not working (rank still 1)")

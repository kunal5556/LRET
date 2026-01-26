"""Test if DEPOLARIZE gates work with LRET."""
import sys
sys.path.insert(0, "d:/LRET/python")

from qlret import simulate_json

print("Testing DEPOLARIZE gate support...")

# Simple circuit with depolarizing noise
circuit = {
    "circuit": {
        "num_qubits": 2,
        "operations": [
            {"name": "H", "wires": [0]},
            {"name": "DEPOLARIZE", "wires": [0], "params": [0.001]},
            {"name": "CNOT", "wires": [0, 1]},
            {"name": "DEPOLARIZE", "wires": [0], "params": [0.001]},
            {"name": "DEPOLARIZE", "wires": [1], "params": [0.001]},
        ],
    },
    "config": {"epsilon": 1e-4, "initial_rank": 1},
}

print("Running circuit...")
try:
    result = simulate_json(circuit, export_state=True, use_native=True)
    print(f"Result keys: {list(result.keys())}")
    print(f"Status: {result.get('status')}")
    if 'message' in result:
        print(f"Message: {result.get('message')}")
    if 'error' in result:
        print(f"Error: {result.get('error')}")
    print(f"Rank: {result.get('final_rank')}")
    print(f"Has state: {'state' in result}")
    if 'state' in result:
        print(f"State keys: {list(result['state'].keys())}")
    if result.get('status') == 'success':
        print("\nSUCCESS!")
    else:
        print("\nFAILED!")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

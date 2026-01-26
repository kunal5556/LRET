"""
DEEP DIAGNOSTIC: Is LRET Actually Simulating?

This script thoroughly investigates:
1. Is LRET running real simulations or returning dummy data?
2. Are output states physically valid (trace=1, positive semidefinite)?
3. Do LRET and Cirq produce the SAME output states?
4. Why are speedups so high (14x-2700x)?

Author: Deep analysis after suspicious benchmark results
Date: January 22, 2026
"""

import json
import sys
import time
import subprocess
import tempfile
import os
import traceback
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

print("""
╔════════════════════════════════════════════════════════════════════╗
║  DEEP DIAGNOSTIC: Verifying LRET Actually Simulates               ║
║  Investigating suspicious speedup values (14x-2700x)              ║
╚════════════════════════════════════════════════════════════════════╝
""")

LRET_EXE = Path("d:/LRET/build/Release/quantum_sim.exe")

# ============================================================================
# STEP 1: Check if LRET exports actual state data
# ============================================================================
print("="*70)
print("STEP 1: Verify LRET exports actual quantum state")
print("="*70)

# Simple Bell state circuit
bell_circuit = {
    "circuit": {
        "num_qubits": 2,
        "operations": [
            {"name": "H", "wires": [0]},
            {"name": "CNOT", "wires": [0, 1]},
        ],
        "observables": [
            {"type": "PAULI", "operator": "Z", "wires": [0], "coefficient": 1.0},
            {"type": "PAULI", "operator": "Z", "wires": [1], "coefficient": 1.0},
        ],
    },
    "config": {
        "epsilon": 1e-6,
        "initial_rank": 1,
        "export_state": True,  # Request state export
    },
}

# Save circuit
circuit_path = Path("d:/LRET/cirq_comparison/diagnostic_bell.json")
with open(circuit_path, 'w') as f:
    json.dump(bell_circuit, f, indent=2)

# Run LRET with state export
output_path = Path("d:/LRET/cirq_comparison/diagnostic_bell_output.json")
cmd = [
    str(LRET_EXE),
    "--input-json", str(circuit_path),
    "--output-json", str(output_path),
    "--export-json-state",  # Export the state
]

print(f"\nRunning LRET with state export...")
print(f"Command: {' '.join(cmd)}")

start = time.perf_counter()
result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
elapsed = (time.perf_counter() - start) * 1000

print(f"\nLRET execution time: {elapsed:.2f} ms")
print(f"Return code: {result.returncode}")

if result.returncode != 0:
    print(f"STDERR: {result.stderr}")
    print(f"STDOUT: {result.stdout}")
else:
    # Load and analyze output
    with open(output_path, 'r') as f:
        lret_output = json.load(f)
    
    print(f"\nLRET Output Keys: {list(lret_output.keys())}")
    print(f"Status: {lret_output.get('status')}")
    print(f"Execution time (reported): {lret_output.get('execution_time_ms')} ms")
    print(f"Final rank: {lret_output.get('final_rank')}")
    print(f"Expectation values: {lret_output.get('expectation_values')}")
    
    # Check if state was exported
    if 'state' in lret_output:
        state_data = lret_output['state']
        print(f"\n✓ State data exported!")
        print(f"  State keys: {list(state_data.keys()) if isinstance(state_data, dict) else type(state_data)}")
        
        if isinstance(state_data, dict):
            if 'L_matrix' in state_data:
                L = state_data['L_matrix']
                print(f"  L_matrix shape: {len(L)} x {len(L[0]) if L else 0}")
            if 'density_matrix' in state_data:
                print(f"  Density matrix provided directly")
    else:
        print(f"\n⚠️  No state data in output!")
        print(f"  Full output: {json.dumps(lret_output, indent=2)[:500]}...")

# ============================================================================
# STEP 2: Compare LRET vs Cirq on Bell state
# ============================================================================
print("\n" + "="*70)
print("STEP 2: Compare LRET vs Cirq output on Bell state")
print("="*70)

import cirq

# Create Cirq Bell state
qubits = cirq.LineQubit.range(2)
cirq_circuit = cirq.Circuit([
    cirq.H(qubits[0]),
    cirq.CNOT(qubits[0], qubits[1]),
])

# Simulate with Cirq
cirq_sim = cirq.DensityMatrixSimulator()
cirq_result = cirq_sim.simulate(cirq_circuit)
cirq_state = cirq_result.final_density_matrix

print(f"\nCirq Bell state density matrix:")
print(f"  Shape: {cirq_state.shape}")
print(f"  Trace: {np.trace(cirq_state).real:.6f}")
print(f"  Is Hermitian: {np.allclose(cirq_state, cirq_state.conj().T)}")

# Expected Bell state: |00⟩ + |11⟩ / sqrt(2)
# Density matrix should be:
# [[0.5, 0, 0, 0.5],
#  [0,   0, 0, 0  ],
#  [0,   0, 0, 0  ],
#  [0.5, 0, 0, 0.5]]
expected_bell = np.array([
    [0.5, 0, 0, 0.5],
    [0,   0, 0, 0  ],
    [0,   0, 0, 0  ],
    [0.5, 0, 0, 0.5]
], dtype=complex)

print(f"\nExpected Bell state |Φ+⟩:")
print(expected_bell.real)

print(f"\nCirq output (real part):")
print(np.round(cirq_state.real, 4))

fidelity_cirq_expected = np.abs(np.trace(
    np.sqrt(np.sqrt(expected_bell) @ cirq_state @ np.sqrt(expected_bell))
)) ** 2
print(f"\nFidelity(Cirq, Expected): {fidelity_cirq_expected:.6f}")

# ============================================================================
# STEP 3: Deep dive into LRET execution
# ============================================================================
print("\n" + "="*70)
print("STEP 3: Analyze LRET execution in detail")
print("="*70)

# Run LRET with verbose output
cmd_verbose = [
    str(LRET_EXE),
    "-n", "4",
    "-d", "10",
    "--noise", "0.0",
    "-v",  # Verbose
    "--show-timing",
]

print(f"\nRunning LRET with verbose output (4 qubits, depth 10)...")
print(f"Command: {' '.join(cmd_verbose)}")

start = time.perf_counter()
result = subprocess.run(cmd_verbose, capture_output=True, text=True, timeout=60)
elapsed = (time.perf_counter() - start) * 1000

print(f"\nExecution time: {elapsed:.2f} ms")
print(f"\nSTDOUT (first 2000 chars):")
print(result.stdout[:2000] if result.stdout else "(empty)")

if result.stderr:
    print(f"\nSTDERR:")
    print(result.stderr[:500])

# ============================================================================
# STEP 4: Time breakdown analysis
# ============================================================================
print("\n" + "="*70)
print("STEP 4: Time breakdown - Why is LRET so fast?")
print("="*70)

# Test multiple configurations
configs = [
    {"qubits": 4, "depth": 20},
    {"qubits": 6, "depth": 20},
    {"qubits": 8, "depth": 20},
]

for config in configs:
    qubits = config['qubits']
    depth = config['depth']
    
    # Generate circuit
    ops = [{"name": "H", "wires": [i]} for i in range(qubits)]
    for _ in range(depth):
        for i in range(0, qubits - 1, 2):
            ops.append({"name": "CNOT", "wires": [i, i+1]})
    
    circuit = {
        "circuit": {
            "num_qubits": qubits,
            "operations": ops,
        },
        "config": {
            "epsilon": 1e-4,
            "initial_rank": 1,
        },
    }
    
    # Save
    test_path = Path(f"d:/LRET/cirq_comparison/timing_test_{qubits}q.json")
    with open(test_path, 'w') as f:
        json.dump(circuit, f)
    
    out_path = Path(f"d:/LRET/cirq_comparison/timing_test_{qubits}q_out.json")
    
    # Measure subprocess overhead (run empty command)
    start = time.perf_counter()
    subprocess.run([str(LRET_EXE), "--help"], capture_output=True)
    help_time = (time.perf_counter() - start) * 1000
    
    # Run actual simulation
    start = time.perf_counter()
    cmd = [str(LRET_EXE), "--input-json", str(test_path), "--output-json", str(out_path)]
    subprocess.run(cmd, capture_output=True)
    total_time = (time.perf_counter() - start) * 1000
    
    # Get reported internal time
    try:
        with open(out_path, 'r') as f:
            output = json.load(f)
        internal_time = output.get('execution_time_ms', 0)
        final_rank = output.get('final_rank', 'N/A')
        status = output.get('status', 'unknown')
    except:
        internal_time = 0
        final_rank = 'N/A'
        status = 'error'
    
    overhead = total_time - internal_time
    
    print(f"\n{qubits}q depth-{depth}:")
    print(f"  Status: {status}")
    print(f"  Total wall-clock: {total_time:.2f} ms")
    print(f"  LRET internal:    {internal_time:.2f} ms")
    print(f"  Subprocess overhead: {overhead:.2f} ms")
    print(f"  Final rank: {final_rank}")
    print(f"  Help command time: {help_time:.2f} ms (baseline)")

# ============================================================================
# STEP 5: Verify rank evolution
# ============================================================================
print("\n" + "="*70)
print("STEP 5: Verify rank evolution (should grow with entanglement)")
print("="*70)

# Test with noise (rank should grow)
noise_circuit = {
    "circuit": {
        "num_qubits": 6,
        "operations": [
            {"name": "H", "wires": [i]} for i in range(6)
        ] + [
            {"name": "CNOT", "wires": [i, i+1]} for i in range(5)
        ] * 5,  # Repeat CNOT pattern
    },
    "config": {
        "epsilon": 1e-4,
        "initial_rank": 1,
    },
}

# Without noise
noise_path = Path("d:/LRET/cirq_comparison/rank_test_no_noise.json")
with open(noise_path, 'w') as f:
    json.dump(noise_circuit, f)

out_no_noise = Path("d:/LRET/cirq_comparison/rank_test_no_noise_out.json")
subprocess.run([str(LRET_EXE), "--input-json", str(noise_path), "--output-json", str(out_no_noise)], capture_output=True)

with open(out_no_noise, 'r') as f:
    result_no_noise = json.load(f)

print(f"\n6q circuit WITHOUT noise:")
print(f"  Final rank: {result_no_noise.get('final_rank')}")
print(f"  Execution time: {result_no_noise.get('execution_time_ms')} ms")

# With noise
noise_circuit["config"]["noise"] = {"type": "depolarizing", "parameter": 0.01}
noise_path_with = Path("d:/LRET/cirq_comparison/rank_test_with_noise.json")
with open(noise_path_with, 'w') as f:
    json.dump(noise_circuit, f)

out_with_noise = Path("d:/LRET/cirq_comparison/rank_test_with_noise_out.json")
subprocess.run([str(LRET_EXE), "--input-json", str(noise_path_with), "--output-json", str(out_with_noise)], capture_output=True)

with open(out_with_noise, 'r') as f:
    result_with_noise = json.load(f)

print(f"\n6q circuit WITH 1% noise:")
print(f"  Final rank: {result_with_noise.get('final_rank')}")
print(f"  Execution time: {result_with_noise.get('execution_time_ms')} ms")

# ============================================================================
# STEP 6: Conclusion
# ============================================================================
print("\n" + "="*70)
print("DIAGNOSTIC CONCLUSIONS")
print("="*70)

print("""
Key Questions to Answer:

1. Is LRET running real simulations?
   → Check if rank evolves correctly
   → Check if expectation values are physically reasonable
   → Check if timing scales with circuit complexity

2. Why are speedups so high (14x-2700x)?
   → Subprocess overhead (~15ms) is significant at small scale
   → LRET internal time is extremely fast (<1ms for small circuits)
   → Cirq DensityMatrixSimulator is NOT optimized (pure Python)
   → LRET may be optimized for specific circuit patterns

3. Is the comparison fair?
   → Both should produce the SAME output states (verify fidelity)
   → We need to export LRET state and compare directly
   → If states differ significantly, one simulator is wrong

RECOMMENDATIONS:
1. Export LRET state and compute fidelity vs Cirq
2. Test with observables and compare expectation values
3. Verify rank grows with noise (confirms real simulation)
4. Use CLI mode for accurate timing (avoid subprocess overhead)
""")

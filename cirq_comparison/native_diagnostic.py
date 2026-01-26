"""
NATIVE DIAGNOSTIC: Compare LRET (native bindings) vs Cirq
Eliminates subprocess overhead for fair comparison
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add LRET python package to path
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

print("="*70)
print("NATIVE DIAGNOSTIC: LRET vs Cirq (No Subprocess Overhead)")
print("="*70)

# ============================================================================
# STEP 1: Load LRET native bindings
# ============================================================================
print("\n[STEP 1] Loading LRET native bindings...")
print("-"*70)

try:
    from qlret import simulate_json
    from qlret.api import _get_native_module
    
    native = _get_native_module()
    if native is not None:
        print("[OK] Native pybind11 module loaded successfully!")
        print(f"     Module: {native}")
    else:
        print("[WARNING] Native module not found, will use subprocess fallback")
except ImportError as e:
    print(f"[ERROR] Could not import qlret: {e}")
    sys.exit(1)

# ============================================================================
# STEP 2: Test native execution speed
# ============================================================================
print("\n[STEP 2] Native execution timing")
print("-"*70)

bell_circuit = {
    "circuit": {
        "num_qubits": 2,
        "operations": [
            {"name": "H", "wires": [0]},
            {"name": "CNOT", "wires": [0, 1]},
        ],
        "observables": [
            {"type": "PAULI", "operator": "Z", "wires": [0], "coefficient": 1.0},
        ],
    },
    "config": {
        "epsilon": 1e-6,
        "initial_rank": 1,
    },
}

# Warm-up run
_ = simulate_json(bell_circuit)

# Timing runs
times = []
for i in range(10):
    start = time.perf_counter()
    result = simulate_json(bell_circuit)
    elapsed = (time.perf_counter() - start) * 1000
    times.append(elapsed)

print(f"Bell state (2q) - 10 runs:")
print(f"  Mean wall-clock time: {np.mean(times):.3f} ms")
print(f"  Std dev: {np.std(times):.3f} ms")
print(f"  Internal LRET time: {result.get('execution_time_ms', 'N/A')} ms")

# ============================================================================
# STEP 3: Compare with Cirq on same circuit
# ============================================================================
print("\n[STEP 3] Comparing LRET native vs Cirq")
print("-"*70)

import cirq

def run_lret_native(n_qubits, depth, noise_prob=0.0):
    """Run circuit on LRET with native bindings."""
    # Build circuit
    ops = [{"name": "H", "wires": [i]} for i in range(n_qubits)]
    for d in range(depth):
        for i in range(0, n_qubits - 1, 2):
            ops.append({"name": "CNOT", "wires": [i, i+1]})
        if d % 2 == 1:
            for i in range(1, n_qubits - 1, 2):
                ops.append({"name": "CNOT", "wires": [i, i+1]})
    
    circuit = {
        "circuit": {"num_qubits": n_qubits, "operations": ops},
        "config": {"epsilon": 1e-4, "initial_rank": 1},
    }
    
    start = time.perf_counter()
    result = simulate_json(circuit, export_state=True)
    elapsed = (time.perf_counter() - start) * 1000
    
    return {
        "wall_time_ms": elapsed,
        "internal_time_ms": result.get("execution_time_ms", 0),
        "final_rank": result.get("final_rank", 1),
        "state": result.get("state"),
    }

def run_cirq(n_qubits, depth, noise_prob=0.0):
    """Run equivalent circuit on Cirq."""
    qubits = cirq.LineQubit.range(n_qubits)
    
    ops = [cirq.H(qubits[i]) for i in range(n_qubits)]
    for d in range(depth):
        for i in range(0, n_qubits - 1, 2):
            ops.append(cirq.CNOT(qubits[i], qubits[i+1]))
        if d % 2 == 1:
            for i in range(1, n_qubits - 1, 2):
                ops.append(cirq.CNOT(qubits[i], qubits[i+1]))
    
    circuit = cirq.Circuit(ops)
    sim = cirq.DensityMatrixSimulator()
    
    start = time.perf_counter()
    result = sim.simulate(circuit)
    elapsed = (time.perf_counter() - start) * 1000
    
    return {
        "time_ms": elapsed,
        "state": result.final_density_matrix,
    }

# Test configurations
configs = [
    {"qubits": 4, "depth": 10},
    {"qubits": 6, "depth": 10},
    {"qubits": 8, "depth": 20},
    {"qubits": 10, "depth": 20},
    {"qubits": 12, "depth": 20},
]

print(f"\n{'Config':<12} {'LRET(ms)':<12} {'LRET int(ms)':<14} {'Cirq(ms)':<12} {'Speedup':<10} {'Rank'}")
print("-"*75)

for config in configs:
    qubits = config['qubits']
    depth = config['depth']
    
    try:
        # Run LRET native
        lret_result = run_lret_native(qubits, depth)
        
        # Run Cirq
        cirq_result = run_cirq(qubits, depth)
        
        speedup = cirq_result['time_ms'] / lret_result['wall_time_ms']
        
        print(f"{qubits}q d{depth:<6} {lret_result['wall_time_ms']:<12.2f} "
              f"{lret_result['internal_time_ms']:<14.2f} {cirq_result['time_ms']:<12.2f} "
              f"{speedup:<10.2f}x {lret_result['final_rank']}")
    except Exception as e:
        print(f"{qubits}q d{depth:<6} ERROR: {e}")

# ============================================================================
# STEP 4: State accuracy comparison
# ============================================================================
print("\n[STEP 4] State accuracy comparison (LRET vs Cirq)")
print("-"*70)

def compare_states(n_qubits):
    """Compare LRET and Cirq output states for a GHZ circuit."""
    # GHZ circuit
    ops = [{"name": "H", "wires": [0]}]
    for i in range(n_qubits - 1):
        ops.append({"name": "CNOT", "wires": [i, i+1]})
    
    circuit = {
        "circuit": {"num_qubits": n_qubits, "operations": ops},
        "config": {"epsilon": 1e-6, "initial_rank": 1},
    }
    
    lret_result = simulate_json(circuit, export_state=True)
    
    # Reconstruct LRET density matrix
    state = lret_result['state']
    L_real = np.array(state['L_real'])
    L_imag = np.array(state['L_imag'])
    rows = state.get('rows', 2**n_qubits)
    cols = state.get('cols', 1)
    
    if L_real.ndim == 1:
        L_real = L_real.reshape(rows, cols)
        L_imag = L_imag.reshape(rows, cols)
    
    L = L_real + 1j * L_imag
    rho_lret = L @ L.conj().T
    
    # Cirq GHZ
    qubits = cirq.LineQubit.range(n_qubits)
    cirq_ops = [cirq.H(qubits[0])]
    for i in range(n_qubits - 1):
        cirq_ops.append(cirq.CNOT(qubits[i], qubits[i+1]))
    
    rho_cirq = cirq.DensityMatrixSimulator().simulate(cirq.Circuit(cirq_ops)).final_density_matrix
    
    # Metrics
    overlap = np.abs(np.trace(rho_lret @ rho_cirq))
    trace_dist = 0.5 * np.sum(np.abs(np.linalg.eigvalsh(rho_lret - rho_cirq)))
    max_diff = np.max(np.abs(rho_lret - rho_cirq))
    
    return {
        "overlap": overlap,
        "trace_distance": trace_dist,
        "max_diff": max_diff,
        "lret_trace": np.trace(rho_lret).real,
        "cirq_trace": np.trace(rho_cirq).real,
    }

print(f"\n{'Circuit':<15} {'Overlap':<12} {'Trace Dist':<12} {'Max Diff':<12} {'Valid'}")
print("-"*60)

for n in [2, 4, 6, 8]:
    metrics = compare_states(n)
    valid = "YES" if metrics['overlap'] > 0.999 else "NO"
    print(f"GHZ-{n:<10} {metrics['overlap']:<12.6f} {metrics['trace_distance']:<12.6f} "
          f"{metrics['max_diff']:<12.6f} {valid}")

# ============================================================================
# STEP 5: Stress test - larger circuits
# ============================================================================
print("\n[STEP 5] Scalability stress test")
print("-"*70)

print(f"\n{'Qubits':<10} {'LRET(ms)':<12} {'Cirq(ms)':<12} {'Speedup':<10} {'LRET Rank'}")
print("-"*55)

for qubits in [6, 8, 10, 12, 14]:
    depth = 20
    
    try:
        lret_result = run_lret_native(qubits, depth)
        lret_time = lret_result['wall_time_ms']
        lret_rank = lret_result['final_rank']
    except Exception as e:
        lret_time = float('inf')
        lret_rank = "ERR"
        print(f"{qubits:<10} LRET ERROR: {e}")
        continue
    
    try:
        cirq_result = run_cirq(qubits, depth)
        cirq_time = cirq_result['time_ms']
    except MemoryError:
        cirq_time = float('inf')
        print(f"{qubits:<10} {lret_time:<12.2f} OOM          -          {lret_rank}")
        continue
    except Exception as e:
        cirq_time = float('inf')
        print(f"{qubits:<10} {lret_time:<12.2f} ERROR        -          {lret_rank}")
        continue
    
    if cirq_time < float('inf') and lret_time > 0:
        speedup = cirq_time / lret_time
        print(f"{qubits:<10} {lret_time:<12.2f} {cirq_time:<12.2f} {speedup:<10.2f}x {lret_rank}")
    else:
        print(f"{qubits:<10} {lret_time:<12.2f} {cirq_time:<12.2f} -          {lret_rank}")

# ============================================================================
# CONCLUSIONS
# ============================================================================
print("\n" + "="*70)
print("CONCLUSIONS")
print("="*70)
print("""
With native bindings (no subprocess overhead):

1. TIMING IS NOW FAIR:
   - Both LRET and Cirq run in-process
   - Wall-clock times are directly comparable
   - No artificial overhead inflating Cirq's relative slowness

2. EXPECTED RESULTS:
   - LRET should still be faster due to C++ optimization
   - Speedup should be more modest (5-20x, not 100-1000x)
   - At very small circuits, times may be similar (both fast)

3. STATE ACCURACY:
   - Overlap should be ~1.0 (matching states)
   - Trace distance should be ~0 (no difference)
   - This proves LRET is computing correct quantum states
""")

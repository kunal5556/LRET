"""
LRET vs Cirq Benchmark - CRITICAL ISSUES ANALYSIS

Date: January 22, 2026
Status: BENCHMARK RESULTS INVALID

## Executive Summary

The previous benchmark results showing "97× speedup" are COMPLETELY INVALID because:
1. LRET was returning errors, not actually simulating
2. We were measuring JSON parsing error time (~0.1ms), not simulation time
3. Circuit JSON schema was incompatible with LRET's expected format

## Root Cause Analysis

### Issue #1: Wrong JSON Schema

**What we generated:**
```json
{
  "circuit": {
    "operations": [
      {"gate": "H", "targets": [0]},
      {"gate": "CNOT", "control": 0, "targets": [1]}
    ]
  }
}
```

**What LRET expects:**
```json
{
  "circuit": {
    "operations": [
      {"name": "H", "wires": [0]},
      {"name": "CNOT", "wires": [0, 1]}
    ]
  }
}
```

**Key differences:**
- `"gate"` should be `"name"`
- `"targets"` should be `"wires"`
- CNOT uses `"wires": [control, target]` not separate fields

### Issue #2: Silent Failures

The Python API returned error results but we didn't check:
```python
{
  "status": "error",
  "message": "[json.exception.out_of_range.403] key 'name' not found"
}
```

We assumed success and measured the JSON error return time.

### Issue #3: No Fidelity Validation

We never compared LRET output states with Cirq states to verify correctness.

## What the "Results" Actually Showed

| Metric | Claimed | Reality |
|--------|---------|---------|
| LRET time | 0.08-1.59 ms | JSON error return time |
| Actual simulation | N/A | Never happened |
| Speedup | 97× | Meaningless |
| Fidelity | Not checked | N/A |

## Required Fixes

### Fix 1: Update Circuit Generator

Change field names:
- `"gate"` → `"name"`
- `"targets"` → `"wires"`
- `"control": c, "targets": [t]` → `"wires": [c, t]`

### Fix 2: Add Error Checking

```python
result = simulate_json(circuit)
if result.get("status") == "error":
    raise Exception(f"LRET error: {result.get('message')}")
```

### Fix 3: Add Fidelity Validation

Compare LRET and Cirq output density matrices:
```python
fidelity = compute_fidelity(lret_state, cirq_state)
assert fidelity > 0.99, f"Fidelity too low: {fidelity}"
```

### Fix 4: Verify Execution Time is Realistic

For 6-qubit circuits with depth 10, expect:
- Cirq FDM: 50-200 ms (stores 64×64 = 4096 element matrix)
- LRET: 10-100 ms (with proper rank evolution)

Times < 1ms indicate no actual computation occurred.

## Parameters That Should Have Been Used

| Parameter | Value | Description |
|-----------|-------|-------------|
| Qubits | 2-10 | Range to test scaling |
| Depth | 10-50 | Circuit depth |
| Epsilon | 1e-4 to 1e-6 | Truncation threshold |
| Noise | 0.01-0.05 | Depolarizing probability |
| Trials | 5 | For statistical significance |
| Parallel Mode | row/hybrid | OpenMP modes |
| Fidelity Check | YES | Validate correctness |

## Corrective Actions

1. ❌ Delete or archive invalid results (BENCHMARK_RESULTS.md)
2. 🔧 Fix circuit generator to use correct JSON schema
3. 🔧 Add error checking to benchmark runner
4. 🔧 Add fidelity validation
5. 🔄 Re-run benchmarks with corrected code
6. ✅ Verify LRET actually runs and produces correct results

## Lessons Learned

1. Always validate that simulations actually executed (check status)
2. Always verify output fidelity against a reference implementation
3. Unrealistic results (97× speedup) should trigger skepticism
4. Test the benchmark infrastructure before trusting results
"""

print(__doc__)

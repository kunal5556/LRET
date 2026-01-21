# Deep Analysis: Why the Original Benchmark Was Wrong

## Your Concerns - ADDRESSED

### a) Parameters Used in Original Benchmarking

| Parameter | Original Value | Issue |
|-----------|---------------|-------|
| **Qubits** | 2-6 | Too small - FDM is naturally fast |
| **Depth** | 2-10 gates | Too shallow - not stressing rank |
| **Noise** | 0% | No noise = rank stays at 1 forever |
| **Epsilon** | 1e-6 | Fine, but irrelevant when rank=1 |
| **Initial Rank** | 1 | Correct for pure state |
| **Parallel Mode** | Unknown | LRET returned errors, so N/A |
| **Fidelity Check** | NONE | Never validated correctness |

**Key Missing Parameters**:
- No noise applied → rank never grows → LRET's advantage never manifests
- Only tested 2-6 qubits → FDM has no memory pressure
- Never verified simulation actually ran

### b) Was LRET Run Correct? Was Fidelity Appropriate?

**NO - LRET DID NOT RUN AT ALL!**

**Evidence from Diagnostic Test**:
```
Results:
  Status: error
  Execution time (reported by LRET): None ms
  Execution time (measured Python):  0.319 ms
  Final rank: None
  Expectation values: None

Full JSON result:
{
  "message": "[json.exception.out_of_range.403] key 'name' not found",
  "status": "error"
}
```

The benchmark script used wrong JSON format:
- Used `"gate": "H"` instead of `"name": "H"`
- Used `"targets": [0], "control": 0` instead of `"wires": [0, 1]`

LRET's JSON parser threw an exception and returned an error dictionary in ~0.1ms.
We measured that 0.1ms error-return time and called it "simulation time"!

### c) Addressing Your Concerns

| Concern | Status | Explanation |
|---------|--------|-------------|
| **Ran too fast** | ✅ CONFIRMED | Was measuring error return, not simulation |
| **LRET not built** | ❌ LRET was built | But JSON schema mismatch caused failures |
| **Fallback simulator** | ❌ No fallback | Just error returns |
| **Unfair comparison** | ✅ CONFIRMED | Cirq ran real simulation, LRET returned errors |

## Root Cause Summary

```
ORIGINAL BENCHMARK FLOW:
1. Generate circuits with "gate", "targets", "control" format
2. Send to LRET (expects "name", "wires" format)
3. LRET throws JSON parse error → returns {"status": "error"} in 0.1ms
4. Benchmark script ignores status, measures 0.1ms
5. Cirq runs real simulation in 5-50ms
6. Speedup = 50ms / 0.1ms = 500× !!! ← COMPLETELY WRONG

CORRECTED BENCHMARK FLOW:
1. Generate circuits with "name", "wires" format  
2. Send to LRET via subprocess
3. LRET runs simulation → returns {"status": "success", "final_rank": 1}
4. Subprocess overhead adds ~12ms
5. Measured time = 12-17ms (mostly overhead)
6. Cirq runs real simulation in 5-10ms
7. Speedup = 7ms / 15ms = 0.47× (LRET slower due to subprocess)
```

## Corrected Results

| Circuit | Cirq (ms) | LRET (ms) | LRET Internal | Rank | Real Speedup |
|---------|-----------|-----------|---------------|------|--------------|
| Bell 2q | 3.84 | 13.17 | 0.03 | 1 | **0.29×** |
| Bell 4q | 5.41 | 13.64 | 0.04 | 1 | **0.40×** |
| Bell 6q | 7.31 | 14.18 | 0.03 | 1 | **0.52×** |
| GHZ 3q | 5.07 | 16.55 | 0.03 | 1 | **0.31×** |
| GHZ 4q | 7.53 | 16.00 | 0.05 | 1 | **0.47×** |
| GHZ 5q | 8.75 | 15.57 | 0.03 | 1 | **0.56×** |
| GHZ 6q | 9.34 | 12.39 | 0.03 | 1 | **0.75×** |

**Average: 0.47× (LRET is 2× SLOWER, not 97× faster!)**

## Why LRET is Slower in This Test

1. **Subprocess overhead dominates**: ~12-15ms per call
   - Process spawn, file I/O, JSON serialization
   - LRET's actual simulation time is 0.03ms!

2. **Small scale (2-6 qubits)**: 
   - FDM stores 64×64 = 4096 numbers → trivial for modern CPUs
   - No memory pressure → no LRET advantage

3. **No noise**:
   - Rank stays at 1 forever
   - LRET's truncation never activates
   - No compression benefit

## What's Needed for Meaningful Comparison

| Parameter | For Small Scale Test | For LRET Advantage Test |
|-----------|---------------------|------------------------|
| Qubits | 2-6 | **10-20** |
| Depth | 2-10 | **50-200** |
| Noise | 0% | **1-5%** |
| Backend | Subprocess | **Native pybind11** |
| Expected Rank | 1 | 10-1000 |
| Expected Speedup | 0.5-1× | **10-100×** |

## Action Items

1. ✅ **Fixed circuit generator** - now uses correct LRET schema
2. ✅ **Added error checking** - validates LRET returns success
3. ✅ **Documented the issue** - this analysis
4. ⬜ **Rebuild native module** - eliminate subprocess overhead
5. ⬜ **Test at larger scale** - 10-16 qubits
6. ⬜ **Add noise** - to stress LRET's rank truncation
7. ⬜ **Validate fidelity** - compare output states

## Lessons Learned

1. **Always check return status** - errors can look like fast results
2. **Validate with known outputs** - Bell state should give specific expectation values
3. **Suspicious results need investigation** - 97× speedup was too good to be true
4. **Test infrastructure before trusting results** - run simple known tests first
5. **Understand what you're measuring** - subprocess overhead vs simulation time

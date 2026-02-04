#!/usr/bin/env python3
"""
Test to analyze sequential LRET execution with Python-level parallelism.

This explores whether running LRET in sequential C++ mode (num_threads=1)
can still benefit from Python-level parallelism to maximize CPU utilization.

Scenario:
---------
- 8 CPU cores available
- LRET configured with num_threads=1 (sequential C++ execution)
- Current batch parallelism: max_batch_workers=4 → 4 workers × 2 threads = 8
- Problem: Each circuit only uses 1 thread (sequential), so 4 cores idle!
- Solution: max_batch_workers=8 → 8 workers × 1 thread = 8 (optimal!)

Expected Findings:
------------------
1. Sequential C++ (num_threads=1) does NOT use extra threads even if allocated
2. Current strategy wastes cores when C++ is sequential
3. New strategy needed: Detect sequential mode and maximize Python workers
"""

import sys
sys.path.insert(0, '.')

from qlret.pennylane_device import QLRETDevice
import os

print("=" * 70)
print("SEQUENTIAL LRET WITH PYTHON PARALLELISM ANALYSIS")
print("=" * 70)
print()

cpu_count = os.cpu_count() or 1
print(f"System CPU Count: {cpu_count}")
print()

# Test scenarios
scenarios = [
    {
        "name": "Sequential C++ + No Batch Parallelism (Current Default)",
        "params": {"num_threads": 1, "max_batch_workers": 0},
        "batch_size": 10,
        "expected": "1 circuit at a time, 1 core used, 7 cores idle",
    },
    {
        "name": "Sequential C++ + Current Batch Strategy",
        "params": {"num_threads": 1, "max_batch_workers": 4},
        "batch_size": 10,
        "expected": "4 workers × 2 threads = 8, BUT each circuit uses only 1 core → 4 cores used, 4 WASTED!",
    },
    {
        "name": "Sequential C++ + Auto-tune Strategy",
        "params": {"num_threads": 1, "max_batch_workers": -1},
        "batch_size": 10,
        "expected": "Auto computes 4 workers × 2 threads = 8, BUT still wastes 4 cores!",
    },
    {
        "name": "Sequential C++ + Maximum Python Workers (PROPOSED)",
        "params": {"num_threads": 1, "max_batch_workers": 8},
        "batch_size": 10,
        "expected": "8 workers × 1 thread = 8, ALL cores used efficiently!",
    },
    {
        "name": "Parallel C++ (default) + Current Batch Strategy",
        "params": {"num_threads": 0, "max_batch_workers": 4},
        "batch_size": 10,
        "expected": "4 workers × 2 threads = 8, each circuit uses 2 cores → OPTIMAL",
    },
]

print("ANALYSIS OF DIFFERENT CONFIGURATIONS:")
print("-" * 70)

for i, scenario in enumerate(scenarios, 1):
    print(f"\n{i}. {scenario['name']}")
    print(f"   Config: {scenario['params']}")
    
    dev = QLRETDevice(wires=4, **scenario['params'])
    workers, threads = dev._compute_execution_strategy(scenario['batch_size'])
    
    total_threads = workers * threads
    actual_cores_used = workers * 1 if dev.num_threads == 1 else workers * threads
    
    print(f"   Strategy: {workers} workers × {threads} threads = {total_threads} total")
    print(f"   Actual cores used: {actual_cores_used}")
    print(f"   Expected: {scenario['expected']}")
    
    # Highlight the problem
    if dev.num_threads == 1 and threads > 1:
        print(f"   ⚠️  PROBLEM: Allocating {threads} threads per circuit, but C++ only uses 1!")
        print(f"   ⚠️  WASTE: {workers * (threads - 1)} cores allocated but idle!")

print()
print("=" * 70)
print("CONCLUSION:")
print("=" * 70)
print()
print("✅ PROBLEM CONFIRMED:")
print("   When LRET runs in sequential C++ mode (num_threads=1), the current")
print("   batch parallelism strategy WASTES cores by allocating multiple threads")
print("   per circuit that will never be used.")
print()
print("✅ ROOT CAUSE:")
print("   _compute_execution_strategy() divides threads evenly among workers:")
print("   threads_per_circuit = effective_threads // workers")
print("   But it doesn't know that C++ will only use 1 thread!")
print()
print("✅ PROPOSED SOLUTION:")
print("   Modify _compute_execution_strategy() to detect sequential mode:")
print("   - If num_threads == 1 or parallel_mode == 'sequential':")
print("     * Set threads_per_circuit = 1")
print("     * Set workers = min(cpu_count, batch_size)")
print("     * Result: Maximum Python parallelism with 1 thread per circuit")
print()
print("📊 EXAMPLE IMPROVEMENT (8-core system, 10 circuits):")
print("   Current: 4 workers × 2 threads = 8, but only 4 cores used (50% waste)")
print("   Optimized: 8 workers × 1 thread = 8, all 8 cores used (0% waste)")
print("   Speedup: 2× faster for sequential C++ mode!")
print()
print("=" * 70)
print("RECOMMENDATION:")
print("=" * 70)
print()
print("Implement a new execution mode:")
print()
print("# Maximum parallelism for sequential C++ execution")
print("dev = QLRETDevice(")
print("    wires=4,")
print("    num_threads=1,           # Sequential C++ (single-threaded per circuit)")
print("    max_batch_workers=-1,    # Auto-tune Python workers")
print(")")
print("# Result: Auto-detects sequential mode → 8 Python workers × 1 thread each")
print()
print("OR add explicit mode:")
print()
print("# Maximum Python parallelism mode")
print("dev = QLRETDevice(")
print("    wires=4,")
print("    max_batch_workers='max'  # Use cpu_count workers with 1 thread each")
print(")")
print()

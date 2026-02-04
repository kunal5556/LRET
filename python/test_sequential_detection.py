#!/usr/bin/env python3
"""
Comprehensive test of sequential mode detection and optimal thread allocation.

Tests all combinations of num_threads, parallel_mode, and max_batch_workers
to verify that CPU resources are utilized optimally.
"""

import sys
sys.path.insert(0, '.')

from qlret.pennylane_device import QLRETDevice
import os

print("=" * 80)
print("SEQUENTIAL MODE DETECTION & OPTIMAL THREAD ALLOCATION TEST")
print("=" * 80)
print()

cpu_count = os.cpu_count() or 1
print(f"System CPU Count: {cpu_count}")
print()

# Define test cases
test_cases = [
    # Group 1: Sequential C++ Mode Detection
    {
        "group": "Sequential C++ Detection (num_threads=1)",
        "tests": [
            {
                "name": "Sequential + No parallelism (default)",
                "params": {"num_threads": 1, "max_batch_workers": 0},
                "batch": 10,
                "expect_workers": 1,
                "expect_threads": 1,
                "optimal": False,
                "note": "Only 1 core used, 7 idle"
            },
            {
                "name": "Sequential + Explicit workers=4",
                "params": {"num_threads": 1, "max_batch_workers": 4},
                "batch": 10,
                "expect_workers": 4,
                "expect_threads": 1,  # FIXED: Now detects sequential!
                "optimal": True,
                "note": "4 cores used efficiently"
            },
            {
                "name": "Sequential + Auto-tune",
                "params": {"num_threads": 1, "max_batch_workers": -1},
                "batch": 10,
                "expect_workers": cpu_count,  # FIXED: Now maximizes workers!
                "expect_threads": 1,
                "optimal": True,
                "note": "All cores used efficiently"
            },
            {
                "name": "Sequential + 'max' mode (NEW!)",
                "params": {"num_threads": 1, "max_batch_workers": 'max'},
                "batch": 10,
                "expect_workers": cpu_count,
                "expect_threads": 1,
                "optimal": True,
                "note": "All cores used with explicit 'max' mode"
            },
        ]
    },
    {
        "group": "Sequential C++ Detection (parallel_mode='sequential')",
        "tests": [
            {
                "name": "parallel_mode=sequential + Auto-tune",
                "params": {"parallel_mode": "sequential", "max_batch_workers": -1},
                "batch": 10,
                "expect_workers": cpu_count,
                "expect_threads": 1,
                "optimal": True,
                "note": "Detected via parallel_mode setting"
            },
            {
                "name": "parallel_mode=sequential + 'max'",
                "params": {"parallel_mode": "sequential", "max_batch_workers": 'max'},
                "batch": 10,
                "expect_workers": cpu_count,
                "expect_threads": 1,
                "optimal": True,
                "note": "Explicit max mode"
            },
        ]
    },
    {
        "group": "Parallel C++ Mode (normal operation)",
        "tests": [
            {
                "name": "Parallel + Auto-tune",
                "params": {"num_threads": 0, "max_batch_workers": -1},
                "batch": 10,
                "expect_workers": cpu_count // 2,
                "expect_threads": 2,
                "optimal": True,
                "note": "Balanced workers and threads"
            },
            {
                "name": "Parallel + Explicit workers=4",
                "params": {"num_threads": 8, "max_batch_workers": 4},
                "batch": 10,
                "expect_workers": 4,
                "expect_threads": 2,
                "optimal": True,
                "note": "4 workers × 2 threads = 8"
            },
            {
                "name": "Parallel + 'max' mode",
                "params": {"num_threads": 8, "max_batch_workers": 'max'},
                "batch": 10,
                "expect_workers": cpu_count,
                "expect_threads": 1,
                "optimal": True,
                "note": "'max' forces 1 thread per worker"
            },
        ]
    },
    {
        "group": "Edge Cases",
        "tests": [
            {
                "name": "Small batch (3 circuits) + Auto-tune",
                "params": {"num_threads": 1, "max_batch_workers": -1},
                "batch": 3,
                "expect_workers": 1,
                "expect_threads": 1,
                "optimal": True,
                "note": "Too small for parallelism"
            },
            {
                "name": "Single circuit + 'max'",
                "params": {"num_threads": 1, "max_batch_workers": 'max'},
                "batch": 1,
                "expect_workers": 1,
                "expect_threads": 1,
                "optimal": True,
                "note": "Single circuit always sequential"
            },
            {
                "name": "Batch smaller than workers",
                "params": {"num_threads": 1, "max_batch_workers": 10},
                "batch": 5,
                "expect_workers": 5,  # min(10, 5)
                "expect_threads": 1,
                "optimal": True,
                "note": "Workers limited by batch size"
            },
        ]
    },
]

# Run tests
total_tests = 0
passed_tests = 0
failed_tests = []

for group_data in test_cases:
    print(f"\n{'=' * 80}")
    print(f"GROUP: {group_data['group']}")
    print(f"{'=' * 80}\n")
    
    for test in group_data['tests']:
        total_tests += 1
        name = test['name']
        params = test['params']
        batch = test['batch']
        expect_workers = test['expect_workers']
        expect_threads = test['expect_threads']
        optimal = test['optimal']
        note = test['note']
        
        # Create device and compute strategy
        dev = QLRETDevice(wires=4, **params)
        workers, threads = dev._compute_execution_strategy(batch)
        
        # Check if results match expectations
        workers_match = workers == expect_workers
        threads_match = threads == expect_threads
        passed = workers_match and threads_match
        
        # Display results
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} | {name}")
        print(f"       Config: {params}")
        print(f"       Batch size: {batch}")
        print(f"       Expected: {expect_workers} workers × {expect_threads} threads")
        print(f"       Got:      {workers} workers × {threads} threads")
        
        if not workers_match or not threads_match:
            print(f"       ⚠️  MISMATCH!")
            failed_tests.append({
                'name': name,
                'expected': (expect_workers, expect_threads),
                'got': (workers, threads),
            })
        
        # Show utilization
        total_threads = workers * threads
        utilization = (total_threads / cpu_count) * 100 if cpu_count > 0 else 0
        print(f"       Total threads: {total_threads} ({utilization:.0f}% CPU utilization)")
        print(f"       Optimal: {'YES' if optimal else 'NO'} - {note}")
        print()
        
        if passed:
            passed_tests += 1

# Summary
print("\n" + "=" * 80)
print("TEST SUMMARY")
print("=" * 80)
print(f"Total tests: {total_tests}")
print(f"Passed: {passed_tests} ✅")
print(f"Failed: {len(failed_tests)} ❌")
print()

if failed_tests:
    print("FAILED TESTS:")
    for fail in failed_tests:
        print(f"  - {fail['name']}")
        print(f"    Expected: {fail['expected']}, Got: {fail['got']}")
    print()
    print("❌ SOME TESTS FAILED!")
else:
    print("✅ ALL TESTS PASSED!")
    print()
    print("VERIFICATION COMPLETE:")
    print("  ✅ Sequential C++ mode detection works (num_threads=1)")
    print("  ✅ Sequential C++ mode detection works (parallel_mode='sequential')")
    print("  ✅ Auto-tune intelligently adapts to C++ parallelism mode")
    print("  ✅ 'max' mode forces maximum Python parallelism")
    print("  ✅ Parallel C++ mode still works optimally")
    print("  ✅ Edge cases handled correctly")
    print()
    print("IMPLEMENTATION SUCCESS! 🎉")

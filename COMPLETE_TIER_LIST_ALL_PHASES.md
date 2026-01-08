# Complete Testing Tiers: ALL Phases (1-9 with subdivisions)

## Full Phase Structure from TESTING_BACKLOG.md

The TESTING_BACKLOG.md actually documents these phases:

- **Phase 1:** Core LRET Tests
- **Phase 2:** GPU Tests (Optional)
- **Phase 3:** MPI Tests (Optional)
- **Phase 4.1-4.5:** Noise Model & Calibration (5 sub-phases)
- **Phase 5:** Python Integration Tests
- **Phase 6a-6c:** Docker Integration & Benchmarking (3 sub-phases)
- **Phase 8:** GPU Memory & Autodiff (multiple sub-phases)
- **Phase 9.1-9.3:** Quantum Error Correction (3 sub-phases)

---

## COMPLETE TIER BREAKDOWN: 17 Tiers Total

```
TIER 1   ──→ Phase 1: Core LRET Tests
             Status: ✅ Ready
             Duration: 1-2h
             Tests: test_simple, test_fidelity, test_minimal, main, batch
             Dependencies: None

TIER 2a  ──→ Phase 2: GPU Tests [OPTIONAL]
             Status: ⏳ Ready (needs CUDA)
             Duration: 1-2h
             Tests: test_distributed_gpu, single-GPU smoke test
             Prerequisites: CUDA 11.8+

TIER 2b  ──→ Phase 2 (Multi-GPU): GPU + MPI Tests [OPTIONAL]
             Status: ⏳ Ready (needs CUDA + MPI)
             Duration: 1-2h
             Tests: test_distributed_gpu_mpi, multi-GPU collectives, load balance
             Prerequisites: CUDA 11.8+, MPI, NCCL

TIER 3   ──→ Phase 3: MPI Distribution Tests [OPTIONAL]
             Status: ⏳ Ready (needs MPI)
             Duration: 1-2h
             Tests: Distributed simulator, row-wise partitioning, collectives
             Prerequisites: MPI library (Open-MPI, MPICH)

TIER 4a  ──→ Phase 4.1: Noise Import Tests
             Status: ⏳ Ready
             Duration: 30min
             Tests: test_noise_import, Qiskit noise model loading
             Prerequisites: None

TIER 4b  ──→ Phase 4.2: Calibration Script Tests
             Status: ⏳ Ready
             Duration: 1h
             Tests: fit_depolarizing, fit_t1_t2, calibrate_noise_model.py
             Prerequisites: None

TIER 4c  ──→ Phase 4.3: Advanced Noise Tests
             Status: ⏳ Ready
             Duration: 30min
             Tests: test_advanced_noise, time-varying noise, memory effects
             Prerequisites: None

TIER 4d  ──→ Phase 4.4-4.5: Leakage & Measurement Tests
             Status: ⏳ Ready
             Duration: 30min
             Tests: test_leakage_measurement, leakage channels, readout errors
             Prerequisites: None

TIER 5   ──→ Phase 5: Python Integration Tests
             Status: ⏳ Ready
             Duration: 2-3h
             Tests: test_qlret_device (15 tests), PennyLane, JSON API
                    test_jax_interface, test_pytorch_interface (optional)
             Prerequisites: Python 3.10+, pytest

TIER 6a  ──→ Phase 6a: Docker Multi-Stage Build Tests
             Status: ⏳ Ready
             Duration: 1h
             Tests: Docker build validation, multi-stage pipeline
             Prerequisites: Docker

TIER 6b  ──→ Phase 6b: Docker Runtime & Integration Tests
             Status: ⏳ Ready
             Duration: 1h
             Tests: Container CLI tests, Python runtime, volume mounting
             Prerequisites: Docker

TIER 6c  ──→ Phase 6c: Performance Benchmarking in Docker
             Status: ⏳ Ready
             Duration: 1-2h
             Tests: Benchmark suite, regression detection, result analysis
             Prerequisites: Docker

TIER 8a  ──→ Phase 8.1-8.2: GPU Performance Optimization [OPTIONAL]
             Status: ⏳ Ready (needs CUDA)
             Duration: 1-2h
             Tests: GPU bandwidth measurement, kernel optimization, shared memory
             Prerequisites: CUDA 11.8+, Nsight Systems

TIER 8b  ──→ Phase 8.3: Autodiff & ML Frameworks [OPTIONAL]
             Status: ⏳ Ready
             Duration: 2-3h
             Tests: test_autodiff, test_autodiff_multi, JAX/PyTorch integration
             Prerequisites: JAX/Flax or PyTorch (optional)

TIER 8c  ──→ Phase 8.4-8.5: Distributed Autodiff & Fault Tolerance [OPTIONAL]
             Status: ⏳ Ready
             Duration: 2h
             Tests: Distributed autodiff, checkpoint/recovery, ML training
             Prerequisites: MPI (optional), JAX (optional)

TIER 9a  ──→ Phase 9.1: Core QEC Tests
             Status: ✅ PASSING (60+ tests)
             Duration: 1-2h
             Tests: test_qec_stabilizer, test_qec_syndrome, test_qec_decoder, test_qec_logical
             Dependencies: Tier 1

TIER 9b  ──→ Phase 9.2: Distributed QEC Tests
             Status: ⏳ Ready (disabled)
             Duration: 1.5-2h
             Tests: test_qec_distributed (52 tests), multi-rank QEC simulation
             Dependencies: Tier 9a, Tier 3 (MPI optional)

TIER 9c  ──→ Phase 9.3: Adaptive QEC Tests
             Status: ⏳ Ready (disabled)
             Duration: 1-1.5h
             Tests: test_qec_adaptive (45 tests), ML decoder, code selection
             Dependencies: Tier 9a, Tier 9b

TIER 10  ──→ Phase 0: Documentation & Release
             Status: ⏳ Ready
             Duration: 1-2h
             Tests: README, API docs, release notes, contribution guide
             Dependencies: All other tiers
```

---

## Tier Matrix with Phase Mapping

| Tier | Phase | Component | Type | Status | Hardware | Duration |
|------|-------|-----------|------|--------|----------|----------|
| 1 | 1 | Core LRET | Mandatory | ✅ Ready | None | 1-2h |
| 2a | 2 | GPU Single | Optional | ⏳ Ready | CUDA | 1-2h |
| 2b | 2 | GPU Multi | Optional | ⏳ Ready | CUDA+MPI | 1-2h |
| 3 | 3 | MPI Dist | Optional | ⏳ Ready | MPI | 1-2h |
| 4a | 4.1 | Noise Import | Mandatory | ⏳ Ready | None | 30m |
| 4b | 4.2 | Calibration | Mandatory | ⏳ Ready | None | 1h |
| 4c | 4.3 | Adv. Noise | Mandatory | ⏳ Ready | None | 30m |
| 4d | 4.4-4.5 | Leakage/Meas | Mandatory | ⏳ Ready | None | 30m |
| 5 | 5 | Python | Mandatory | ⏳ Ready | None | 2-3h |
| 6a | 6a | Docker Build | Mandatory | ⏳ Ready | Docker | 1h |
| 6b | 6b | Docker Runtime | Mandatory | ⏳ Ready | Docker | 1h |
| 6c | 6c | Docker Bench | Mandatory | ⏳ Ready | Docker | 1-2h |
| 8a | 8.1-8.2 | GPU Perf | Optional | ⏳ Ready | CUDA | 1-2h |
| 8b | 8.3 | Autodiff | Optional | ⏳ Ready | JAX/PT | 2-3h |
| 8c | 8.4-8.5 | Dist Autodiff | Optional | ⏳ Ready | MPI | 2h |
| 9a | 9.1 | Core QEC | Mandatory | ✅ Passing | None | 1-2h |
| 9b | 9.2 | Dist QEC | Mandatory | ⏳ Ready | MPI* | 1.5-2h |
| 9c | 9.3 | Adaptive QEC | Mandatory | ⏳ Ready | None | 1-1.5h |
| 10 | 0 | Documentation | Mandatory | ⏳ Ready | None | 1-2h |

---

## Three Execution Paths

### **Path A: Complete (All 17 Tiers) - 25-35 hours**

```
Mandatory Tiers (14): 1, 4a, 4b, 4c, 4d, 5, 6a, 6b, 6c, 9a, 9b, 9c, 10
                     Total: ~18-20 hours

Optional Tiers (3):  2a, 2b, 3, 8a, 8b, 8c
                     Total: ~7-9 hours (if hardware available)
                     
Dependencies:
  - Tier 2a, 2b: Need CUDA
  - Tier 3: Need MPI
  - Tier 8a: Need CUDA + profiling tools
  - Tier 8b: Need JAX or PyTorch
  - Tier 8c: Need MPI + JAX (optional)
```

### **Path B: Core Only (12 Tiers) - 18-22 hours** ← **RECOMMENDED**

```
Mandatory Tiers: 1, 4a, 4b, 4c, 4d, 5, 6a, 6b, 6c, 9a, 9b, 9c, 10

Skip Optional:   2a, 2b, 3, 8a, 8b, 8c (GPU, MPI, Autodiff)

Time: ~18-22 hours
Hardware: None beyond normal dev environment
```

### **Path C: Quick Validation (3 Tiers) - 2 hours**

```
Essential Tiers: 1, 9a, 5 (or skip 5 for ultra-fast)

Time: ~2-3 hours
Purpose: Rapid verification of core functionality
```

---

## Simplified View: 10 Main Test Suites

If you group related tiers:

```
SUITE 1: Core LRET (Tier 1)
         1-2h | ✅ Ready | No deps

SUITE 2: Noise & Calibration (Tiers 4a, 4b, 4c, 4d)
         2.5h | ⏳ Ready | No deps

SUITE 3: Python Integration (Tier 5)
         2-3h | ⏳ Ready | Python

SUITE 4: Core QEC (Tier 9a)
         1-2h | ✅ Passing | No deps

SUITE 5: Distributed QEC (Tier 9b)
         1.5-2h | ⏳ Ready | MPI optional

SUITE 6: Adaptive QEC (Tier 9c)
         1-1.5h | ⏳ Ready | No deps

SUITE 7: GPU Acceleration (Tiers 2a, 2b) [OPTIONAL]
         2-4h | ⏳ Ready | CUDA + MPI

SUITE 8: MPI Distribution (Tier 3) [OPTIONAL]
         1-2h | ⏳ Ready | MPI

SUITE 9: Autodiff & ML (Tiers 8b, 8c) [OPTIONAL]
         4-5h | ⏳ Ready | JAX/PyTorch

SUITE 10: Docker & Documentation (Tiers 6a, 6b, 6c, 10)
          4-5h | ⏳ Ready | Docker
```

---

## Quick Reference: All Tests

### Core Tests (No Dependencies)
- Tier 1: test_simple, test_fidelity, test_minimal
- Tier 4a: test_noise_import
- Tier 4b: calibration scripts
- Tier 4c: test_advanced_noise
- Tier 4d: test_leakage_measurement
- Tier 5: test_qlret_device (15 tests), JSON API
- Tier 9a: test_qec_* (58 tests) ✅
- Tier 9b: test_qec_distributed (52 tests)
- Tier 9c: test_qec_adaptive (45 tests)

### Optional Tests (Hardware-Dependent)
- Tier 2a: test_distributed_gpu (single GPU)
- Tier 2b: test_distributed_gpu_mpi, test_multi_gpu_* (multi-GPU)
- Tier 3: Distributed QEC with MPI
- Tier 8a: GPU bandwidth & optimization
- Tier 8b: test_autodiff, test_autodiff_multi
- Tier 8c: test_autodiff_multi_gpu, distributed ML

### Infrastructure Tests
- Tier 6a: Docker multi-stage build
- Tier 6b: Container runtime validation
- Tier 6c: benchmark_suite.py (170+ benchmarks)
- Tier 10: Documentation & release

---

## Total Test Count

```
Mandatory Tiers (Paths A & B):
  - Tier 1: ~5 tests
  - Tier 4a-4d: ~25 tests (scripts + C++)
  - Tier 5: ~15 tests
  - Tier 6a-6c: ~25 tests + 170+ benchmarks
  - Tier 9a-9c: 145 tests (58+52+45)
  - Tier 10: Documentation (N/A)
  ─────────────────────────────────
  Total: 385+ C++ tests + 170+ benchmarks

Optional Tiers (Path A only):
  - Tier 2a-2b: ~20 tests
  - Tier 3: Integrated into Tier 9b
  - Tier 8a-8c: ~30 tests
  ─────────────────────────────────
  Total: 50+ additional tests
```

---

## Execution Timeline (Path B - Recommended)

```
Week 1: Foundation
├─ Day 1: Tier 1 (Core LRET)         [1-2h]  ✅
├─ Day 2: Tier 4a-4d (Noise)         [2.5h]  ⏳
├─ Day 3: Tier 5 (Python)            [2-3h]  ⏳
├─ Day 4-5: Tier 9a (Core QEC)       [1-2h]  ✅ + setup
└─ Total: 8-10h

Week 2: Distributed & Finalization
├─ Day 1: Tier 9b (Dist QEC)         [1.5-2h]⏳
├─ Day 2: Tier 9c (Adaptive QEC)     [1-1.5h]⏳
├─ Day 3-4: Tiers 6a-6c (Docker)     [3-4h]  ⏳
├─ Day 5: Tier 10 (Documentation)    [1-2h]  ⏳
└─ Total: 7.5-10h

Combined: ~18-22 hours over 2 weeks
Average: 2-2.5 hours per day
```

---

## Summary

**Original "9 phases"** in TESTING_BACKLOG.md actually breaks down to:

- Phase 1 → 1 Tier
- Phase 2 → 2 Tiers (single + multi-GPU)
- Phase 3 → 1 Tier
- Phase 4 → 4 Tiers (4.1, 4.2, 4.3, 4.4-4.5)
- Phase 5 → 1 Tier
- Phase 6 → 3 Tiers (6a, 6b, 6c)
- Phase 8 → 3 Tiers (8.1-8.2, 8.3, 8.4-8.5)
- Phase 9 → 3 Tiers (9.1, 9.2, 9.3)
- Phase 0 (Documentation) → 1 Tier

**Total: 17 comprehensive testing tiers**

✅ **14 Tiers mandatory** (18-22 hours)  
⏳ **3 Tiers optional** (7-9 hours if hardware)  
🎯 **Complete coverage** of all 9 phases + documentation

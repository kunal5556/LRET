# Row Parallelism Optimization - Quick Reference

**Full Strategy**: See [ROW_PARALLELISM_OPTIMIZATION_STRATEGY.md](ROW_PARALLELISM_OPTIMIZATION_STRATEGY.md) (18,500 lines)

**Branch**: `row-parallelism-optimization`  
**Created**: February 5, 2026

---

## 🎯 Quick Summary

This optimization strategy combines insights from:
1. **MPS (Matrix Product States)** - Tensor network techniques for low-rank quantum states
2. **Grok AI Analysis** - 4 scenarios + 5 techniques for row parallelism in LRET

**Expected Gains**: 1.5× to 19× speedup (depending on implementation phase)

---

## 📊 Performance at a Glance

| Phase | Time to Implement | Speedup | Key Optimizations |
|-------|------------------|---------|------------------|
| **Phase 1** | 1 week | **1.5×** | Rank threshold, SIMD, row-local ops |
| **Phase 2** | 1 week | **1.8×** | Parallelism oracle |
| **Phase 3** | 2 weeks | **2.2×** | Cholesky QR, qubit reordering |
| **Phase 4 (GPU)** | 3 days | **3.0×** | GPU Kraus summation |
| **Phase 4 (MPI)** | 3 days | **16×** | Distributed row-parallel (8 nodes) |
| **Phase 4 (GPU+MPI)** | 1 week | **19×** | Combined (8 GPUs) |

---

## 🔥 Phase 1: Quick Wins (CRITICAL - Start Here!)

**Effort**: 1 week  
**Gain**: 1.5× speedup  
**Files to modify**: `src/parallel_modes.cpp`, `src/simd_kernels.cpp`, `src/utils.cpp`

### Task Checklist
- [ ] Raise `MIN_RANK_FOR_COL_PARALLEL` from 4 to 32 (30 min)
- [ ] Add `#pragma omp simd` to row-parallel loops (1 hour)
- [ ] Implement stride-aware scheduling (2 hours)
- [ ] Optimize `compute_trace()` and `sample_measurement()` (3 hours)

### Code Changes

**1. Raise Rank Threshold** (`src/parallel_modes.cpp:56`):
```cpp
// OLD:
constexpr size_t MIN_RANK_FOR_COL_PARALLEL = 4;

// NEW:
constexpr size_t MIN_RANK_FOR_COL_PARALLEL = 32;
```

**2. Add SIMD to Row-Parallel Gate** (`src/parallel_modes.cpp:~250`):
```cpp
#pragma omp parallel for schedule(static, 64) if(dim > 4096)
for (int64_t block = 0; block < (int64_t)dim; block += 2 * step) {
    for (size_t i = block; i < block + step && i < dim; ++i) {
        size_t i0 = i;
        size_t i1 = i + step;
        
        // ADD THIS:
        #pragma omp simd aligned(result:64)
        for (size_t r = 0; r < rank; ++r) {
            Complex v0 = L(i0, r);
            Complex v1 = L(i1, r);
            result(i0, r) = U(0,0)*v0 + U(0,1)*v1;
            result(i1, r) = U(1,0)*v0 + U(1,1)*v1;
        }
    }
}
```

**3. Row-Parallel Trace** (`src/utils.cpp` or `src/simulator.cpp`):
```cpp
double compute_trace_row_parallel(const MatrixXcd& L) {
    size_t dim = L.rows();
    double trace = 0.0;
    
    #pragma omp parallel for reduction(+:trace) schedule(static, 256)
    for (int64_t i = 0; i < (int64_t)dim; ++i) {
        trace += L.row(i).squaredNorm();
    }
    
    return trace;
}
```

---

## ⚡ Phase 2: Parallelism Oracle (HIGH Priority)

**Effort**: 1 week  
**Gain**: +20% (1.8× total)  
**New files**: `src/parallelism_oracle.cpp`, `include/parallelism_oracle.h`

### Key Concept
Runtime decision: row vs column parallelism based on:
- Current rank (< 32 → row)
- Target qubit (low index → row)
- Cache size (stride fits in L2 → row)

### Minimal Oracle Implementation
```cpp
ParallelMode select_mode(const MatrixXcd& L, const GateOp& gate) {
    size_t rank = L.cols();
    
    // Heuristic 1: Low rank → row
    if (rank < 32) return ParallelMode::ROW;
    
    // Heuristic 2: Low qubit index → row
    if (gate.qubits.size() > 0) {
        size_t max_qubit = *std::max_element(gate.qubits.begin(), gate.qubits.end());
        if (max_qubit < 5) return ParallelMode::ROW;
    }
    
    // Heuristic 3: High rank → column
    if (rank > 64) return ParallelMode::COLUMN;
    
    return ParallelMode::ROW;  // Default
}
```

---

## 🚀 Phase 3: Advanced Optimizations (MEDIUM Priority)

### 3.1 Cholesky QR (2.5× faster truncation)

**File**: `src/simulator.cpp`  
**Function**: `truncate_L_with_cholesky_qr()`

```cpp
MatrixXcd orthonormalize_L_cholesky_qr(const MatrixXcd& L) {
    MatrixXcd G = L.adjoint() * L;
    Eigen::LLT<MatrixXcd> llt(G);
    MatrixXcd R_inv = llt.matrixU().inverse();
    
    MatrixXcd Q(L.rows(), L.cols());
    #pragma omp parallel for
    for (int64_t i = 0; i < (int64_t)L.rows(); ++i) {
        Q.row(i) = L.row(i) * R_inv;
    }
    return Q;
}
```

### 3.2 Qubit Reordering (1.8× for QNN)

**New file**: `src/qubit_reordering.cpp`

**Concept**: Reorder qubits so most-used qubits → lowest indices (better cache locality)

### 3.3 Community Detection Batching (2× for random circuits)

**New file**: `src/community_batching.cpp`

**Concept**: Group rows by gate connectivity, process communities in parallel

---

## 🎮 Phase 4: GPU & Distributed (HIGH Impact)

### 4.1 GPU Kraus Summation (5× speedup)

**File**: `src/gpu_simulator.cu`  
**Gain**: 5× faster noise application

**Key Idea**: Batch all Kraus operators, apply in parallel on GPU

### 4.2 MPI Row Distribution (16× on 8 nodes)

**File**: `src/mpi_parallel.cpp`  
**Gain**: Near-linear scaling to 8-16 nodes

**Key Technique**: HALO exchange with pipelining (overlap communication + computation)

---

## 📋 Testing Checklist

### Correctness
- [ ] Fidelity > 0.999 vs baseline for n=8-12 qubits
- [ ] All unit tests pass (run `ctest`)
- [ ] Edge cases: rank=1, rank=128, empty circuits

### Performance
- [ ] Benchmark VQE circuit (n=15, d=50): Should be 1.5× faster after Phase 1
- [ ] Benchmark QNN (n=12, feature map heavy): Should be 2× faster after qubit reordering
- [ ] Benchmark noisy circuit: Should be 3× faster with GPU Kraus

### Regression
- [ ] No slowdown for column-parallel cases (rank > 64)
- [ ] No slowdown for sequential cases (n < 8)

---

## 🗂️ File Modification Map

| File | Phase 1 | Phase 2 | Phase 3 | Phase 4 |
|------|---------|---------|---------|---------|
| `src/parallel_modes.cpp` | ✅ Heavy | ⚠️ Integrate oracle | ⚠️ Add reordering | ⚠️ MPI |
| `src/simd_kernels.cpp` | ✅ Add SIMD | | | |
| `src/simulator.cpp` | ⚠️ Row-parallel trace | | ✅ Cholesky QR | |
| `src/utils.cpp` | ⚠️ Sampling | | | |
| `src/parallelism_oracle.cpp` | | ✅ Create | ✅ Enhance | |
| `src/qubit_reordering.cpp` | | | ✅ Create | |
| `src/community_batching.cpp` | | | ✅ Create | |
| `src/gpu_simulator.cu` | | | | ✅ Batched Kraus |
| `src/mpi_parallel.cpp` | | | | ✅ HALO exchange |

---

## 📈 Benchmarking Command

```bash
# Baseline (before optimization)
./build/quantum_sim samples/vqe_h2_n15_d50.json --verbose

# After Phase 1
./build/quantum_sim samples/vqe_h2_n15_d50.json --verbose --row-parallel-threshold 32

# Compare performance
python scripts/compare_benchmarks.py baseline.csv phase1.csv
```

---

## 🔬 Key Insights from Research

### From MPS (Matrix Product States)
1. **Sequential tensor updates** → Row-parallel gate application for low qubits
2. **Bond dimension = rank** → Adaptive truncation thresholds
3. **Chunked storage** → Cache-aware L matrix layout

### From Grok Analysis - 4 Scenarios
1. **Scenario 1**: Low rank (r < 32) → Row stride fits in L2 cache → 2× speedup
2. **Scenario 2**: Low qubits (t < 5) → Cache-line friendly access → 1.8× speedup
3. **Scenario 3**: Row-local ops (trace, sampling) → Perfect parallelism → 5-8× speedup
4. **Scenario 4**: MPI distribution → 10× less communication vs column-wise

### Grok - 5 Advanced Techniques
1. **Cholesky QR**: Row-parallel orthonormalization → 2.5× faster
2. **GPU Kraus**: Batched matrix ops → 5× faster
3. **Hybrid TTN**: Tree topology for deep circuits → 2.5× (niche)
4. **Community Detection**: Graph-based scheduling → 2× (random circuits)
5. **Parallelism Oracle**: Runtime heuristics → 1.2× adaptive gain

---

## 🎯 Recommended Implementation Order

### Week 1 (CRITICAL)
**Phase 1: Quick Wins** → 1.5× speedup
- Day 1: Rank threshold + SIMD (file: `parallel_modes.cpp`)
- Day 2-3: Stride-aware scheduling + row-local operations
- Day 4-5: Testing and benchmarking

### Week 2 (HIGH)
**Phase 2: Oracle** → 1.8× total
- Day 1: Cache detection + basic oracle (new files)
- Day 2-3: Integration with simulation loop
- Day 4-5: Performance logging and tuning

### Week 3-4 (MEDIUM)
**Phase 3: Advanced** → 2.2× total
- Week 3: Cholesky QR + qubit reordering
- Week 4: Community batching (optional)

### Week 5-6 (MEDIUM, if GPU/cluster available)
**Phase 4: GPU & MPI** → 3-19× depending on hardware
- Week 5: GPU Kraus implementation
- Week 6: MPI HALO exchange + testing

---

## 🚨 Common Pitfalls

1. **Don't forget SIMD alignment**: Use `aligned(result:64)` pragma
2. **Test with rank=1**: Edge case that breaks many optimizations
3. **Verify trace preservation**: After Cholesky QR, check `Tr[ρ] = 1`
4. **Profile before optimizing**: Oracle should learn, not guess
5. **MPI requires balanced circuits**: Very imbalanced gate distributions → poor scaling

---

## 📞 Need Help?

- **Full details**: See [ROW_PARALLELISM_OPTIMIZATION_STRATEGY.md](ROW_PARALLELISM_OPTIMIZATION_STRATEGY.md)
- **Code examples**: Check `Part V: Code Examples` in full strategy
- **MPS background**: See `Part I: MPS Research` (4500 lines)
- **Grok analysis**: See `Part II: Grok Analysis` (8000 lines)
- **Performance math**: See `Part IV: Performance Projections`

---

## 🎉 Success Metrics

After Phase 1 implementation, you should see:
- ✅ VQE (n=15, d=50): 3550ms → **2370ms** (1.5× speedup)
- ✅ QNN (n=12, feature map): Significant improvement with qubit reordering
- ✅ Sampling-heavy circuits: 5× faster measurements
- ✅ All unit tests passing with fidelity > 0.999

**Good luck! Start with Phase 1 - the quick wins will validate the approach.**

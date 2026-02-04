# LRET Optimized - Row Parallelism Implementation

**Purpose**: Experimental folder for Phase 1 row parallelism optimizations

## Structure

```
LRET_optimized/
├── src/           # Optimized source files (copied from ../src/)
├── include/       # Optimized headers (copied from ../include/)
├── build/         # Build directory for optimized version
└── CMakeLists.txt # Build configuration
```

## Key Changes (Phase 1)

Will be made to these files:
1. `src/parallel_modes.cpp` - Rank threshold, SIMD, stride-aware
2. `src/simulator.cpp` - Row-parallel trace
3. `src/utils.cpp` - Row-parallel sampling
4. `src/simd_kernels.cpp` - SIMD optimizations

## Build Instructions

```bash
cd LRET_optimized/build
cmake ..
make -j8
```

## Baseline Comparison

- **Baseline**: `../build/quantum_sim` (untouched)
- **Optimized**: `build/quantum_sim` (this folder)

Run benchmarks and compare using `../scripts/compare_performance.py`

## Status

- ✅ Folder created
- ✅ Files copied (31 .cpp, 35 .h)
- ⏳ Awaiting Phase 1 implementation

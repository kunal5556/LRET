# PennyLane QNN Classifier Benchmark Report

**Generated**: 2026-01-10 01:38:52

## Executive Summary

This benchmark compares LRET PennyLane plugin against PennyLane's default.mixed device for Quantum Neural Network (QNN) classifier training.

## Test Configurations

| Qubits | Epochs | Batch Size | Layers | Parameters |
|--------|--------|------------|--------|------------|
| 8 | 100 | 10 | 2 | 32 |
| 8 | 100 | 10 | 2 | 32 |
| 10 | 100 | 10 | 2 | 40 |
| 10 | 100 | 10 | 2 | 40 |

## Performance Results

| Qubits | Device | Time (s) | Memory (MB) | Accuracy | Status |
|--------|--------|----------|-------------|----------|--------|
| 8 | lret | 25.35 | 13.91 | 70.0% | success |
| 8 | default.mixed | 86.94 | 79.34 | 70.0% | success |
| 10 | lret | 129.85 | 29.95 | 70.0% | success |
| 10 | default.mixed | 1405.33 | 908.91 | 70.0% | success |

## Comparison Analysis

### 8 Qubits

**Performance:**
- LRET: 25.35s
- default.mixed: 86.94s
- **Speedup: 3.43x** ✓ (LRET faster)

**Memory:**
- LRET: 13.91 MB
- default.mixed: 79.34 MB
- **Memory Ratio: 5.71x** ✓ (LRET uses less)

**Accuracy:**
- LRET: 70.0%
- default.mixed: 70.0%
- Difference: 0.0%

### 10 Qubits

**Performance:**
- LRET: 129.85s
- default.mixed: 1405.33s
- **Speedup: 10.82x** ✓ (LRET faster)

**Memory:**
- LRET: 29.95 MB
- default.mixed: 908.91 MB
- **Memory Ratio: 30.34x** ✓ (LRET uses less)

**Accuracy:**
- LRET: 70.0%
- default.mixed: 70.0%
- Difference: 0.0%

## Conclusion

**Overall Performance:**
- Average Speedup: 9.61x
- Average Memory Reduction: 22.53x

✓✓ **LRET demonstrates clear advantages in both speed and memory efficiency.**

**Recommendation:**
- For 8-10 qubits: Both backends perform adequately
- For 12+ qubits: LRET expected to show significant advantages
- For production: Run larger-scale tests (14-16 qubits) to confirm scalability

---

**Note**: This benchmark tests QNN classifier training at small scale (8-10 qubits). LRET's advantages are more pronounced at larger scales (12-24 qubits) where tensor network compression and rank truncation provide significant memory and speed benefits.

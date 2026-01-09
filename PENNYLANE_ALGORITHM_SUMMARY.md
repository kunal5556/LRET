# PennyLane Algorithm Test Suite - Quick Reference

**Complete list of algorithms for LRET benchmarking**

---

## Overview

**Total Algorithms**: 20 (14 core + 6 variants)

**Testing Priority Tiers**:
- **Tier 1**: Must test (7 algorithms) - Core functionality validation
- **Tier 2**: Should test (7 algorithms) - Comprehensive coverage
- **Tier 3**: Nice to test (6 algorithms) - Advanced features

---

## Quick Reference Table

| # | Algorithm | Category | Qubits | Priority | Noise Sens. | Why Test |
|---|-----------|----------|--------|----------|-------------|----------|
| 1 | **VQE (H2, LiH)** | Chemistry | 2-6 | Tier 1 | Medium | Most important NISQ app |
| 2 | **QAOA (MaxCut)** | Optimization | 4-10 | Tier 1 | High | Popular optimization |
| 3 | **QNN Classifier** | ML | 4-8 | Tier 1 | Medium | Core QML application |
| 4 | **QFT** | Simulation | 4-12 | Tier 1 | High | Fundamental algorithm |
| 5 | **QPE** | Simulation | 6-10 | Tier 1 | Very High | Chemistry applications |
| 6 | **Grover's Algorithm** | Search | 4-12 | Tier 1 | High | Famous quantum algorithm |
| 7 | **Quantum Metrology** | Sensing | 4-8 | Tier 1 | Very High | Quantum advantage demo |
| 8 | **UCCSD-VQE** | Chemistry | 4-8 | Tier 2 | High | Gold standard chemistry |
| 9 | **Portfolio Optimization** | Optimization | 6-10 | Tier 2 | Medium | Real-world finance |
| 10 | **QSVM** | ML | 2-6 | Tier 2 | Low-Medium | Quantum kernel methods |
| 11 | **QAE** | Estimation | 6-10 | Tier 2 | High | Monte Carlo speedup |
| 12 | **VQD** | Chemistry | 2-6 | Tier 2 | Medium-High | Excited states |
| 13 | **qGAN** | ML | 3-6 | Tier 2 | High | Advanced QML |
| 14 | **Number Partitioning** | Optimization | 4-8 | Tier 2 | Medium | Optimization variant |
| 15 | **VQT** | Thermalization | 4-8 | Tier 3 | High | Thermal states |
| 16 | **Quantum Walk** | Simulation | 4-10 | Tier 3 | Medium | Algorithmic primitive |
| 17 | **Quantum Kernel Alignment** | ML | 2-6 | Tier 3 | Low | Advanced kernels |
| 18 | **Sub-sampling QNN** | ML | 4-8 | Tier 3 | Medium | Efficiency techniques |
| 19 | **Hardware-Efficient Ansatz** | Variational | 4-12 | Tier 3 | Medium | Ansatz benchmarking |
| 20 | **ADAPT-VQE** | Chemistry | 4-8 | Tier 3 | Medium | Adaptive methods |

---

## Algorithms by Category

### 🧪 Quantum Chemistry (5 algorithms)

1. **VQE** - Variational Quantum Eigensolver
   - Purpose: Ground state energy
   - Variants: H2 (2 qubits), LiH (4 qubits), BeH2 (6 qubits)
   - Test: Convergence, accuracy, gradient efficiency

2. **UCCSD-VQE** - Unitary Coupled Cluster
   - Purpose: High-accuracy molecular energies
   - Challenge: Many parameters, deep circuits
   - Test: Chemical accuracy (<1 kcal/mol)

3. **VQD** - Variational Quantum Deflation
   - Purpose: Excited states
   - Method: Penalty terms for orthogonality
   - Test: Multiple state preparation

4. **ADAPT-VQE** - Adaptive VQE
   - Purpose: Parameter-efficient chemistry
   - Method: Dynamically build ansatz
   - Test: Operator pool selection

5. **VQT** - Variational Quantum Thermalization
   - Purpose: Thermal state preparation
   - Test: Temperature-dependent properties

### 🎯 Optimization (4 algorithms)

6. **QAOA (MaxCut)** - Quantum Approximate Optimization
   - Purpose: Graph partitioning
   - Variants: Complete graphs, random graphs
   - Test: Approximation ratio, layer scaling

7. **Portfolio Optimization**
   - Purpose: Financial asset selection
   - Constraints: Risk-return tradeoff
   - Test: Solution quality vs classical

8. **Number Partitioning**
   - Purpose: Divide set into equal sums
   - Method: QAOA or VQE encoding
   - Test: Partition balance quality

9. **Traveling Salesman (small)**
   - Purpose: Route optimization
   - Instances: 3-5 cities
   - Test: Tour optimality

### 🤖 Quantum Machine Learning (6 algorithms)

10. **QNN Classifier** - Quantum Neural Network
    - Purpose: Binary/multi-class classification
    - Architecture: Data encoding + variational layers
    - Test: Training speed, accuracy, gradients

11. **QSVM** - Quantum Support Vector Machine
    - Purpose: Classification with quantum kernels
    - Method: Kernel matrix computation
    - Test: Kernel speed, classification accuracy

12. **qGAN** - Quantum Generative Adversarial Network
    - Purpose: Distribution generation
    - Components: Quantum generator, classical discriminator
    - Test: Training stability, distribution matching

13. **Quantum Kernel Alignment**
    - Purpose: Optimize quantum feature maps
    - Method: Kernel-target alignment maximization
    - Test: Alignment improvement

14. **Sub-sampling QNN**
    - Purpose: Efficient gradient estimation
    - Method: Subset of parameters per iteration
    - Test: Convergence vs full gradient

15. **Hardware-Efficient Ansatz**
    - Purpose: Device-specific optimization
    - Variants: Different entangling patterns
    - Test: Expressibility vs trainability

### 🔄 Quantum Simulation (3 algorithms)

16. **QFT** - Quantum Fourier Transform
    - Purpose: Basis transformation
    - Circuit: O(n²) gates
    - Test: Output fidelity, noise resilience

17. **QPE** - Quantum Phase Estimation
    - Purpose: Eigenvalue estimation
    - Applications: Chemistry, cryptography
    - Test: Phase accuracy, precision scaling

18. **Quantum Walk**
    - Purpose: Quantum algorithm primitive
    - Types: Discrete-time, continuous-time
    - Test: Propagation dynamics

### 🔍 Search & Estimation (2 algorithms)

19. **Grover's Algorithm**
    - Purpose: Unstructured search
    - Speedup: Quadratic over classical
    - Test: Success probability, optimal iterations

20. **QAE** - Quantum Amplitude Estimation
    - Purpose: Probability estimation
    - Application: Monte Carlo acceleration
    - Test: Estimation accuracy, precision

### 📏 Quantum Metrology (1 algorithm)

21. **Quantum Parameter Estimation**
    - Purpose: Precise parameter measurement
    - Advantage: Heisenberg limit scaling
    - Test: Quantum Fisher Information, precision

---

## Benchmarking Metrics per Algorithm

### For All Algorithms

1. **Performance**:
   - Execution time (total, per iteration)
   - Memory usage (peak, average)
   - Rank evolution for LRET

2. **Accuracy**:
   - Solution quality (problem-specific)
   - Fidelity vs exact solution
   - Error vs classical methods

3. **Noise Impact**:
   - Performance degradation curve
   - Critical noise threshold
   - Error mitigation effectiveness

4. **Scalability**:
   - Time vs qubit count
   - Time vs circuit depth
   - Memory vs problem size

5. **Comparison**:
   - LRET vs default.mixed
   - LRET vs competitors (Qiskit, Cirq)
   - Speedup ratios (10-500×)

### Algorithm-Specific Metrics

| Algorithm | Key Metric | Target Value | Comparison |
|-----------|------------|--------------|------------|
| VQE | Ground energy | <1 kcal/mol error | vs FCI |
| QAOA | Approximation ratio | >0.85 for p=3 | vs optimal |
| QNN | Classification accuracy | >90% | vs classical NN |
| QFT | Output fidelity | >99% | vs exact QFT |
| QPE | Phase error | <0.01 | vs true eigenvalue |
| Grover | Success probability | >0.9 at optimal | vs classical |
| QSVM | Kernel computation | <1s per 100 pairs | vs classical kernel |
| qGAN | KL divergence | <0.1 | vs target distribution |

---

## Implementation Details

### File Structure

```
PENNYLANE_ALGORITHM_CATALOG.md       # Detailed implementations (80+ pages)
PENNYLANE_BENCHMARKING_STRATEGY.md   # Testing methodology (80+ pages)
PENNYLANE_ALGORITHM_SUMMARY.md       # This quick reference (5 pages)

benchmarks/
├── 06_applications/
│   ├── vqe_benchmark.py              # Tests 1, 2, 8
│   ├── qaoa_benchmark.py             # Tests 6, 7
│   ├── qml_benchmark.py              # Tests 3, 10, 11, 12
│   ├── quantum_simulation_benchmark.py  # Tests 4, 5, 16, 18
│   ├── advanced_chemistry_benchmark.py  # Tests 8, 20
│   ├── extended_optimization_benchmark.py  # Tests 7, 9
│   └── quantum_metrology_benchmark.py    # Tests 19, 21
```

### Testing Timeline (5 weeks)

**Week 1**: Categories 1-2 (Memory, Speed)
- Infrastructure setup
- Basic benchmarks

**Week 2**: Categories 3-4 (Accuracy, Gradients)
- Fidelity testing
- Gradient verification

**Week 3**: Category 5 (Scalability)
- Qubit scaling
- Depth scaling

**Week 4**: Category 6 (Applications)
- **Tier 1 algorithms** (days 1-3): VQE, QAOA, QNN, QFT, QPE, Grover, Metrology
- **Tier 2 algorithms** (days 4-5): UCCSD-VQE, Portfolio, QSVM, QAE, VQD, qGAN, Number Partitioning

**Week 5**: Categories 7-8 (Framework Integration, Cross-Simulator)
- PyTorch/JAX integration
- Qiskit/Cirq comparison
- **Tier 3 algorithms** (if time permits)

---

## Expected Results Summary

### Memory Efficiency
- **Baseline**: 10-500× reduction vs default.mixed
- **Range**: 10× (low noise) to 500× (high noise)
- **Mechanism**: Low-rank decomposition (rank 10-50 vs 2ⁿ)

### Execution Speed
- **Baseline**: 50-200× faster vs default.mixed
- **Range**: 50× (small circuits) to 200× (large circuits)
- **Mechanism**: Rank compression + efficient operations

### Accuracy
- **Fidelity**: >99.9% vs exact simulation
- **Chemical**: <1 kcal/mol for VQE
- **Optimization**: Approximation ratio >0.85
- **Classification**: >90% accuracy for QML

### Scalability
- **Qubits**: Can handle 16-20 qubits (vs 12-14 for default.mixed)
- **Depth**: Linear scaling after rank saturation
- **Noise**: Better resilience at 1-5% error rates

### Gradients
- **Parameter-shift**: Overhead 2-3× (expected)
- **Accuracy**: Machine precision
- **Speed**: 5-20× faster than default.mixed

---

## Key Validation Points

### Must Verify

1. ✅ **VQE converges** to chemical accuracy for H2, LiH
2. ✅ **QAOA finds** near-optimal cuts (>85% approximation)
3. ✅ **QNN trains** successfully (>90% accuracy)
4. ✅ **QFT output** matches exact QFT (>99% fidelity)
5. ✅ **Gradients are correct** (vs finite differences)
6. ✅ **Memory savings** confirmed (10-500×)
7. ✅ **Speed improvements** confirmed (50-200×)

### Should Verify

8. ✅ **UCCSD-VQE** achieves chemical accuracy
9. ✅ **QSVM** computes kernels efficiently
10. ✅ **QPE** estimates phases accurately
11. ✅ **Grover** finds targets probabilistically
12. ✅ **Scalability** follows predicted trends

### Nice to Verify

13. ⏸ **qGAN** training stability
14. ⏸ **VQD** excited state orthogonality
15. ⏸ **ADAPT-VQE** operator selection efficiency
16. ⏸ **Quantum Walk** dynamics correctness

---

## Publication Readiness Checklist

- [ ] All Tier 1 algorithms tested (7/7)
- [ ] All Tier 2 algorithms tested (7/7)
- [ ] Memory benchmarks complete
- [ ] Speed benchmarks complete
- [ ] Accuracy benchmarks complete
- [ ] Gradient benchmarks complete
- [ ] Scalability analysis done
- [ ] Cross-simulator comparison done
- [ ] Framework integration verified
- [ ] Plots and figures generated
- [ ] Results tables compiled
- [ ] Paper draft written
- [ ] Code repository cleaned
- [ ] Documentation complete
- [ ] Reproducibility verified

---

## Next Actions

### Immediate (This Week)
1. ✅ Review algorithm list with stakeholders
2. ✅ Approve testing priorities (Tier 1 first)
3. ⏳ Set up benchmark directory structure
4. ⏳ Implement Tier 1 algorithm benchmarks

### Short-term (Weeks 1-2)
5. ⏳ Run Categories 1-4 benchmarks
6. ⏳ Collect initial performance data
7. ⏳ Verify correctness for all Tier 1 algorithms
8. ⏳ Generate preliminary plots

### Medium-term (Weeks 3-5)
9. ⏳ Complete all 8 benchmark categories
10. ⏳ Analyze scalability trends
11. ⏳ Cross-simulator comparisons
12. ⏳ Framework integration tests

### Final Steps
13. ⏳ Compile results into publication tables/figures
14. ⏳ Write benchmarking results section for paper
15. ⏳ Prepare supplementary materials
16. ⏳ Submit to arXiv and journal

---

**Document Version**: 1.0.0  
**Last Updated**: January 9, 2026  
**Status**: Ready for Implementation

**Related Documents**:
- [PENNYLANE_ALGORITHM_CATALOG.md](PENNYLANE_ALGORITHM_CATALOG.md) - Complete implementations
- [PENNYLANE_BENCHMARKING_STRATEGY.md](PENNYLANE_BENCHMARKING_STRATEGY.md) - Detailed testing plan
- [PENNYLANE_COMPREHENSIVE_DOCUMENTATION.md](PENNYLANE_COMPREHENSIVE_DOCUMENTATION.md) - Full integration guide

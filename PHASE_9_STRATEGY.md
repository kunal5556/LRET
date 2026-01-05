# Phase 9: Quantum Error Correction at Scale - Strategic Implementation Plan

**Objective:** Implement fault-tolerant quantum computing via quantum error correction (QEC) codes, integrated with LRET's distributed multi-GPU infrastructure.

**Timeline:** 4-6 weeks (implementation + validation)  
**Model Recommendation:** Claude Opus 4.5 (QEC theory + distributed systems expertise)  
**Prerequisites:** Phase 8 complete (distributed GPU, autodiff, fault tolerance)

---

## 1. Overview & Architecture

### 1.1 QEC Fundamentals for LRET

**Key Concepts:**
- **Logical Qubits:** Encoded in multiple physical qubits to protect against errors
- **Stabilizer Codes:** Surface codes, repetition codes (detect errors without measuring state)
- **Syndrome Measurement:** Ancilla qubits measure error patterns without collapsing data
- **Error Correction Cycle:** Syndrome extraction → Classical decoding → Correction gates

**LRET Integration Points:**
- Physical qubits simulated via LRET's density matrix formalism (ρ = L @ L†)
- Noise models from Phase 4 calibration (T1, T2, gate errors, leakage)
- Distributed QEC: partition code blocks across GPUs
- Syndrome circuits as parameterized gate sequences (autodiff-compatible)

### 1.2 Scope & Milestones

**Phase 9.1:** Surface Code Implementation (CPU)
- Distance-3 and distance-5 surface codes
- Stabilizer measurement circuits
- Syndrome extraction and error tracking

**Phase 9.2:** Distributed QEC (Multi-GPU)
- Partition code blocks across GPU ranks
- Collective syndrome aggregation
- Parallel decoder execution

**Phase 9.3:** Adaptive QEC (ML-Driven)
- Dynamic code selection based on noise profile
- ML-based decoder training (via JAX/PyTorch integration)
- Feedback loop with Phase 4 noise calibration

---

## 2. Phase 9.1: Surface Code Implementation (CPU Foundation)

### 2.1 Stabilizer Formalism

**Implementation:**
```cpp
// include/qec_stabilizer.h
class StabilizerCode {
    size_t num_data_qubits;
    size_t num_ancilla_qubits;
    size_t code_distance;
    
    // Pauli operators for X/Z stabilizers
    std::vector<PauliString> x_stabilizers;
    std::vector<PauliString> z_stabilizers;
    
    // Logical operators
    PauliString logical_x;
    PauliString logical_z;
};

struct PauliString {
    std::vector<char> paulis;  // 'I', 'X', 'Y', 'Z'
    std::vector<size_t> qubits;
    Complex phase;
};
```

**Key Methods:**
- `generate_stabilizers()` - Build X/Z checks for surface code lattice
- `measure_stabilizer()` - Apply measurement circuit for one stabilizer
- `detect_errors()` - Compare syndrome with previous round
- `compute_logical_operator()` - Chain of X/Z forming logical qubit

**Test:** `test_stabilizer_generation.cpp`
- Distance-3 surface code: 9 data + 8 ancilla qubits
- Verify stabilizer commutation relations
- Check logical operator anticommutation

### 2.2 Syndrome Extraction Circuits

**Circuit Construction:**
```cpp
// For X-stabilizer: |0⟩_anc → H → CNOT(anc, data1) → CNOT(anc, data2) → ... → H → Measure
// For Z-stabilizer: |0⟩_anc → CNOT(data1, anc) → CNOT(data2, anc) → ... → Measure

QuantumSequence build_syndrome_circuit(const StabilizerCode& code, bool measure_x_stabs);
```

**Noise-Aware Syndrome:**
- Include realistic gate errors during CNOT cascades
- Ancilla preparation/measurement errors
- Idle noise on data qubits during syndrome extraction

**Test:** `test_syndrome_circuit.cpp`
- Apply single-qubit X error, measure Z-stabilizers → syndrome = 1
- No error → syndrome = 0
- Verify fault-tolerance: single error during syndrome doesn't corrupt data

### 2.3 Classical Decoder

**Minimum-Weight Perfect Matching (MWPM) Decoder:**
```cpp
// include/qec_decoder.h
class MWPMDecoder {
public:
    std::vector<size_t> decode(const std::vector<int>& syndrome,
                                const StabilizerCode& code,
                                ErrorModel error_model);
private:
    Graph syndrome_graph;  // Nodes = syndrome locations, edges = error chains
    double compute_edge_weight(size_t q1, size_t q2, ErrorModel model);
};
```

**Algorithm:**
1. Build syndrome graph from defects (flipped stabilizers)
2. Assign edge weights based on error probability
3. Solve minimum-weight perfect matching (Blossom algorithm)
4. Trace paths to determine correction gates

**External Dependency:** PyMatching (Python library) or implement Blossom V in C++

**Test:** `test_mwpm_decoder.cpp`
- Inject known error pattern
- Verify decoder recovers correct error location
- Measure logical error rate vs physical error rate

### 2.4 Logical Qubit Interface

**High-Level API:**
```cpp
class LogicalQubit {
public:
    LogicalQubit(size_t distance, ErrorModel noise);
    
    // Logical gate implementation
    void apply_logical_x();
    void apply_logical_z();
    void apply_logical_h();  // Hadamard via lattice surgery
    void apply_logical_cnot(LogicalQubit& target);
    
    // Error correction cycle
    SyndromeResult measure_syndrome();
    void apply_correction(const std::vector<size_t>& corrections);
    
    // Measurement
    int measure_logical_z();
    int measure_logical_x();
};
```

**Test:** `test_logical_qubit.cpp`
- Initialize |0⟩_L, apply X_L, measure Z_L → expect -1
- Inject errors below threshold → correction recovers state
- Verify fidelity > 0.999 after correction cycle

---

## 3. Phase 9.2: Distributed QEC (Multi-GPU)

### 3.1 Code Block Partitioning

**Strategy:** Partition surface code lattice across GPU ranks
- **Row-wise partitioning:** Each rank owns horizontal strip of lattice
- **Column-wise partitioning:** Vertical strips (alternative)
- **Hybrid:** 2D block decomposition for large codes

**Implementation:**
```cpp
struct DistributedQECConfig {
    size_t code_distance;
    int world_size;
    int rank;
    PartitionStrategy partition;  // ROW, COLUMN, BLOCK_2D
};

class DistributedLogicalQubit {
public:
    DistributedLogicalQubit(const DistributedQECConfig& config,
                            const DistributedGPUConfig& gpu_config);
    
    // Distributed syndrome measurement
    LocalSyndrome measure_local_syndrome();
    GlobalSyndrome all_gather_syndrome();
    
    // Decoder runs on rank 0, broadcasts corrections
    std::vector<size_t> decode_global(const GlobalSyndrome& syn);
    void apply_distributed_correction(const std::vector<size_t>& corrections);
};
```

**Communication Patterns:**
- **Syndrome extraction:** Each rank measures local stabilizers
- **AllGather:** Collect all syndromes to rank 0 for decoding
- **Broadcast:** Decoder sends correction gates back to all ranks
- **Boundary stabilizers:** Require inter-rank qubit state (use NCCL P2P)

**Test:** `test_distributed_qec.cpp`
- Distance-5 code split across 2 GPUs
- Inject errors on both ranks
- Verify global decoder finds all errors
- Compare fidelity with single-GPU reference

### 3.2 Parallel Decoder

**Strategy:** Distribute decoding workload
- **Graph partitioning:** Split syndrome graph across ranks
- **Local matching:** Each rank solves local subgraph
- **Merge:** Combine solutions at boundaries

**Implementation:**
```cpp
class ParallelMWPMDecoder {
public:
    ParallelMWPMDecoder(const DistributedQECConfig& config);
    
    // Each rank decodes its partition
    LocalCorrection decode_local(const LocalSyndrome& syn);
    
    // Boundary resolution via MPI
    GlobalCorrection merge_corrections(const std::vector<LocalCorrection>& local);
};
```

**Challenges:**
- Matching across partition boundaries (requires coordination)
- Load balancing (uneven syndrome distribution)
- Communication overhead (syndrome size ∝ O(d²) for distance d)

**Optimization:** Pipeline syndrome measurement + decoding + correction

**Test:** `test_parallel_decoder.cpp`
- Verify parallel decoder matches serial MWPM
- Measure speedup: 2 ranks should decode in ~0.6× time
- Check correctness for boundary-spanning errors

### 3.3 Fault-Tolerant Circuit Execution

**Integration with Phase 8 Fault Tolerance:**
```cpp
class FaultTolerantQECRunner {
public:
    FaultTolerantQECRunner(const DistributedQECConfig& qec_config,
                          const FaultToleranceConfig& ft_config);
    
    // Run logical circuit with continuous error correction
    void run_logical_circuit(const LogicalCircuit& circuit);
    
    // Checkpoint includes syndrome history + correction log
    void checkpoint_with_qec_state();
    bool recover_qec_state(const std::string& ckpt_path);
};
```

**QEC-Specific Checkpointing:**
- Save syndrome history (for streak detection)
- Log applied corrections (for debugging)
- Store decoder state (graph, weights)

**Test:** `test_ft_qec_execution.cpp`
- Run 1000-cycle logical circuit with checkpointing every 100 cycles
- Simulate failure at cycle 350
- Recover and continue → verify logical state matches no-failure case

---

## 4. Phase 9.3: Adaptive QEC (ML-Driven)

### 4.1 Noise-Adaptive Code Selection

**Dynamic Code Choice:**
```cpp
enum QECCodeType {
    REPETITION_CODE,     // For highly biased noise (X >> Z or Z >> X)
    SURFACE_CODE,        // General-purpose
    COLOR_CODE,          // Faster logical gates (tradeoff: higher overhead)
    CONCATENATED_CODE    // For very high error rates
};

class AdaptiveQECSelector {
public:
    QECCodeType select_code(const NoiseProfile& noise);
    
    // Use Phase 4 calibration data
    double predict_logical_error_rate(QECCodeType code, const NoiseProfile& noise);
};
```

**Strategy:**
- If T1 ≪ T2: Prefer Z-biased codes (Bacon-Shor)
- If gate errors ≫ idle errors: Minimize gate count (color codes)
- If error rate > threshold: Increase code distance or switch to concatenated codes

**Test:** `test_adaptive_code_selection.cpp`
- Load various noise profiles from Phase 4 calibration
- Verify selector chooses expected code for each regime
- Measure logical error rate improvement vs fixed code

### 4.2 ML-Based Decoder

**Neural Network Decoder:**
```cpp
// Python-side (JAX/PyTorch)
class NeuralDecoder:
    def __init__(self, code_distance):
        self.model = build_transformer_model(code_distance)
    
    def train(self, syndrome_dataset, error_labels):
        # Supervised learning: syndrome → error location
        pass
    
    def decode(self, syndrome):
        return self.model.predict(syndrome)

// C++ interface
class MLDecoder {
public:
    MLDecoder(const std::string& model_path);
    std::vector<size_t> decode(const std::vector<int>& syndrome);
private:
    std::unique_ptr<PythonMLModel> model_;  // Via pybind11
};
```

**Training Data Generation:**
- Simulate 10M error patterns + syndrome measurements
- Label with ground-truth error locations
- Train transformer/CNN to map syndrome → corrections

**Advantages over MWPM:**
- Can learn correlations in real hardware noise
- Handles time-dependent noise (via syndrome history)
- Potentially faster inference (parallel GPU forward pass)

**Test:** `test_ml_decoder.cpp`
- Compare ML decoder vs MWPM on 1000 test cases
- Measure accuracy (% correct error identification)
- Verify ML decoder improves fidelity by ≥5% on correlated noise

### 4.3 Closed-Loop Calibration

**Feedback Loop:**
```
Phase 4 Noise Calibration → Phase 9 Adaptive QEC → Runtime Error Tracking → Update Noise Model
```

**Implementation:**
```cpp
class ClosedLoopQEC {
public:
    void run_calibration_cycle();  // Call Phase 4 tools
    void update_decoder_weights(const NoiseProfile& updated_noise);
    void monitor_logical_error_rate();  // Alert if threshold exceeded
};
```

**Strategy:**
- Every N correction cycles, measure actual error rates
- If logical error rate drifts, re-calibrate noise model
- Re-train ML decoder if noise profile changes significantly

**Test:** `test_closed_loop_qec.cpp`
- Simulate drifting noise (T1 decreases over time)
- Verify system detects drift and recalibrates
- Confirm logical error rate stabilizes after update

---

## 5. Testing & Validation Strategy

### 5.1 Unit Tests (CPU, No GPU Required)

| Test | Purpose | Success Criteria |
|------|---------|------------------|
| `test_stabilizer_generation` | Verify stabilizer construction | All commutation relations hold |
| `test_syndrome_circuit` | Syndrome measurement correctness | Known error → expected syndrome |
| `test_mwpm_decoder` | Decoder accuracy | 100% correct on < threshold errors |
| `test_logical_qubit` | Logical gate fidelity | > 0.999 after correction |
| `test_adaptive_code_selection` | Code choice logic | Optimal code selected per noise |

### 5.2 Integration Tests (CPU, Phase 8 Components)

| Test | Purpose | Success Criteria |
|------|---------|------------------|
| `test_qec_with_autodiff` | Gradient of logical circuit | Gradients match numerical diff |
| `test_qec_checkpointing` | Checkpoint QEC state | Recovery restores syndrome history |
| `test_qec_scheduler` | Prioritize correction cycles | Scheduler reduces logical error rate |

### 5.3 Distributed Tests (Multi-GPU, MPI, NCCL)

| Test | Purpose | Success Criteria |
|------|---------|------------------|
| `test_distributed_qec` | Multi-rank syndrome + decode | Matches single-GPU fidelity |
| `test_parallel_decoder` | Parallel MWPM speedup | 2× ranks → 1.5× faster decode |
| `test_ft_qec_execution` | Fault tolerance + QEC | Recovery completes logical circuit |

All marked ⏭️ SKIPPED on Windows; validated on HPC.

---

## 6. Implementation Roadmap

### Week 1-2: Phase 9.1 (CPU Foundation)
- Day 1-3: Stabilizer formalism + code generation
- Day 4-7: Syndrome extraction circuits
- Day 8-10: MWPM decoder (integrate Blossom library or implement)
- Day 11-14: Logical qubit interface + unit tests

### Week 3-4: Phase 9.2 (Distributed QEC)
- Day 15-17: Code block partitioning strategy
- Day 18-21: Distributed syndrome measurement + AllGather
- Day 22-24: Parallel decoder implementation
- Day 25-28: Integration with Phase 8 fault tolerance

### Week 5-6: Phase 9.3 (Adaptive QEC)
- Day 29-31: Noise-adaptive code selection
- Day 32-35: ML decoder (training pipeline + inference)
- Day 36-38: Closed-loop calibration integration
- Day 39-42: End-to-end testing + benchmarking

---

## 7. Dependencies & Prerequisites

### External Libraries
- **Blossom V:** Minimum-weight perfect matching (C++ library, free for research)
- **PyMatching:** Alternative MWPM solver (Python, GPU-compatible)
- **NetworkX:** Graph algorithms for syndrome graph construction (Python)
- **SciPy:** Sparse matrix operations for large codes

### LRET Components Required
- ✅ Phase 4 noise models (T1, T2, gate errors, leakage)
- ✅ Phase 8 distributed GPU infrastructure
- ✅ Phase 8 autodiff (for gradient-based decoder tuning)
- ✅ Phase 8 checkpointing (for QEC state persistence)

### Knowledge Requirements
- Stabilizer formalism (Nielsen & Chuang Ch. 10)
- Surface code topology (Fowler et al. review papers)
- Classical decoding algorithms (MWPM, belief propagation)
- Lattice surgery (for logical gate implementation)

---

## 8. Performance Targets

### Logical Error Rate
- **Target:** p_L < p²_phys for distance-3 code (below threshold)
- **Stretch Goal:** p_L < 10⁻⁶ with distance-7 + optimized decoder

### Decoding Speed
- **Single-GPU:** < 10 ms/cycle for distance-5 code
- **Multi-GPU (4 ranks):** < 15 ms/cycle (communication overhead)
- **ML Decoder:** < 5 ms/cycle (GPU inference)

### Scalability
- **Code Distance:** Support d=3, 5, 7, 9 (9→81 data qubits)
- **Distributed:** Scale to 16 GPUs for d=11+ codes
- **Throughput:** 100 correction cycles/sec sustained

---

## 9. Risk Mitigation

### Technical Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| MWPM decoder too slow | High | Pre-compute edge weights; use approximate matching |
| Boundary errors in distributed QEC | Medium | Add ghost stabilizers at partition edges |
| ML decoder overfits training noise | Medium | Use noise augmentation; validate on diverse profiles |
| Syndrome measurement errors cascade | High | Implement flag qubits (fault-tolerant syndrome) |

### Integration Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| QEC conflicts with Phase 8 checkpointing | Low | Extend checkpoint format for syndrome history |
| Autodiff breaks on syndrome circuits | Medium | Treat measurement as non-differentiable boundary |
| NCCL overhead dominates small codes | Low | Use MPI for small syndromes, NCCL for large |

---

## 10. Success Criteria

### Phase 9.1 (CPU)
- ✅ Distance-3 and distance-5 surface codes implemented
- ✅ Syndrome extraction circuits generate correct syndromes
- ✅ MWPM decoder achieves > 99% accuracy below threshold
- ✅ Logical qubit interface supports X, Z, H gates
- ✅ All unit tests pass

### Phase 9.2 (Distributed)
- ✅ Code partitioning across 2-4 GPU ranks functional
- ✅ Distributed decoder matches single-GPU fidelity (within 1%)
- ✅ Parallel decoder achieves 1.5× speedup on 2 ranks
- ✅ Fault-tolerant execution completes 1000-cycle logical circuit
- ✅ Integration tests pass

### Phase 9.3 (Adaptive)
- ✅ Adaptive code selector chooses optimal code per noise profile
- ✅ ML decoder improves fidelity by ≥5% on correlated noise
- ✅ Closed-loop calibration stabilizes logical error rate under drift
- ✅ End-to-end test: 10,000-cycle logical circuit with continuous QEC

---

## 11. Documentation Deliverables

1. **API Reference:** `docs/api-reference/qec.md`
   - Stabilizer code classes
   - Decoder interfaces
   - Logical qubit operations

2. **User Guide:** `docs/user-guide/qec-setup.md`
   - Choosing code distance
   - Configuring noise models
   - Running distributed QEC

3. **Developer Guide:** `docs/developer-guide/qec-internals.md`
   - Stabilizer formalism primer
   - Decoder algorithm details
   - Extending to new codes

4. **Benchmarks:** `docs/performance/qec-benchmarks.md`
   - Logical error rates vs physical error rates
   - Decoder speed comparisons
   - Scaling plots (GPUs vs code distance)

---

## 12. Next Steps After Phase 9

Upon completion, proceed to:

**Phase 10:** ML Framework Integration at Scale
- VQE/QAOA on logical qubits
- Distributed gradient accumulation with QEC
- Hybrid classical-quantum optimization loops

**Phase 11:** Production Hardening
- Large-scale benchmarks (>4 GPUs, distance-9+ codes)
- Reliability testing (inject hardware failures)
- Deployment (Docker, Singularity, cloud platforms)

---

**Ready to begin implementation with Claude Opus 4.5?**

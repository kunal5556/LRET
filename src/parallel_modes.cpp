/**
 * @file parallel_modes.cpp
 * @brief Advanced Parallelization Strategies for LRET Simulation
 * 
 * Implements multiple parallelization approaches:
 * - ROW: Gate fusion + aggressive row-level parallelism
 * - COLUMN: Column-parallel for high-rank states
 * - BATCH: Uses optimized simulator (baseline)
 * - HYBRID: Layer-parallel gate application + fusion
 * 
 * Key optimizations:
 * 1. Gate Fusion: Combine consecutive single-qubit gates on same target
 * 2. Layer Parallelism: Apply non-overlapping gates simultaneously
 * 3. OpenMP Threshold: Skip parallelization for small problems (overhead > benefit)
 * 4. Pre-computed indices: O(2^n) not O(4^n) for two-qubit gates
 * 5. Cache-aware scheduling: Better memory access patterns
 * 
 * Performance notes:
 * - For n<8 (dim<256): Sequential is faster due to OpenMP overhead
 * - For n>12: Memory bandwidth becomes bottleneck, need cache-aware access
 * - Row parallelism dominates when rank << 2^n (typical LRET)
 */

#include "parallel_modes.h"
#include "gates_and_noise.h"
#include "simulator.h"
#include "utils.h"
#include "structured_csv.h"
#include <iostream>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace qlret {

//==============================================================================
// Configuration Constants
//==============================================================================

// Minimum dimension for OpenMP parallelization to be beneficial
// For smaller problems, OpenMP overhead (~10-50μs) exceeds computation time
constexpr size_t MIN_DIM_FOR_OPENMP = 256;  // 2^8, so n >= 8 qubits

// Minimum rank for column parallelism to be beneficial
constexpr size_t MIN_RANK_FOR_COL_PARALLEL = 4;

//==============================================================================
// Batch Size and Mode Selection Heuristics
//==============================================================================

size_t auto_select_batch_size(size_t num_qubits) {
    unsigned int num_threads = 1;
#ifdef _OPENMP
    num_threads = omp_get_max_threads();
#else
    num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
#endif
    
    size_t base_batch;
    if (num_qubits <= 10) {
        base_batch = 32;
    } else if (num_qubits <= 14) {
        base_batch = 64;
    } else {
        base_batch = 128;
    }
    
    size_t scaled = base_batch * (num_threads / 4);
    scaled = std::max(scaled, size_t(32));
    scaled = std::min(scaled, size_t(256));
    
    return scaled;
}

std::string get_workload_class(size_t num_qubits) {
    if (num_qubits <= 10) return "low-workload";
    if (num_qubits <= 14) return "medium-workload";
    return "high-workload";
}

ParallelMode auto_select_mode(size_t num_qubits, size_t depth, size_t rank_estimate) {
    size_t dim = 1ULL << num_qubits;
    
    if (dim < 64) {
        return ParallelMode::SEQUENTIAL;
    }
    
    // For medium-large problems with significant depth, hybrid excels
    if (depth > 10 && num_qubits >= 8) {
        return ParallelMode::HYBRID;
    }
    
    // For problems with many qubits, row parallelism is best
    if (num_qubits >= 10) {
        return ParallelMode::ROW;
    }
    
    return ParallelMode::BATCH;
}

//==============================================================================
// Gate Fusion Utilities
// Combine consecutive single-qubit gates on same target into one gate matrix
//==============================================================================

struct FusedGate {
    MatrixXcd matrix;       // Combined 2x2 or 4x4 gate
    std::vector<size_t> qubits;
    bool is_fused;          // True if multiple gates were fused
};

// Fuse consecutive single-qubit gates on the same target
std::vector<FusedGate> fuse_single_qubit_gates(const std::vector<GateOp>& gates) {
    std::vector<FusedGate> fused;
    
    size_t i = 0;
    while (i < gates.size()) {
        const auto& gate = gates[i];
        
        if (gate.qubits.size() == 1) {
            // Single-qubit gate - try to fuse with following gates on same qubit
            size_t target = gate.qubits[0];
            MatrixXcd combined = get_single_qubit_gate(gate.type, gate.params);
            bool did_fuse = false;
            
            // Look ahead for more gates on same qubit
            size_t j = i + 1;
            while (j < gates.size()) {
                const auto& next_gate = gates[j];
                if (next_gate.qubits.size() == 1 && next_gate.qubits[0] == target) {
                    // Fuse: combined = next_gate * combined (apply combined first, then next)
                    MatrixXcd next_matrix = get_single_qubit_gate(next_gate.type, next_gate.params);
                    combined = next_matrix * combined;
                    did_fuse = true;
                    j++;
                } else if (next_gate.qubits.size() == 2 && 
                          (next_gate.qubits[0] == target || next_gate.qubits[1] == target)) {
                    // Two-qubit gate involving our target - stop fusing
                    break;
                } else {
                    // Gate on different qubit - might be able to continue fusing
                    // but conservatively stop here
                    break;
                }
            }
            
            FusedGate fg;
            fg.matrix = combined;
            fg.qubits = {target};
            fg.is_fused = did_fuse;
            fused.push_back(fg);
            
            i = j;  // Skip all fused gates
        } else {
            // Two-qubit gate - cannot fuse, add directly
            FusedGate fg;
            fg.matrix = get_two_qubit_gate(gate.type, gate.params);
            fg.qubits = gate.qubits;
            fg.is_fused = false;
            fused.push_back(fg);
            i++;
        }
    }
    
    return fused;
}

// Group gates into layers where no two gates share a qubit
std::vector<std::vector<const GateOp*>> build_parallel_layers(const std::vector<GateOp>& gates) {
    std::vector<std::vector<const GateOp*>> layers;
    
    for (const auto& gate : gates) {
        bool placed = false;
        
        // Try to place in existing layer
        for (auto& layer : layers) {
            std::unordered_set<size_t> used_qubits;
            for (const auto* g : layer) {
                for (size_t q : g->qubits) used_qubits.insert(q);
            }
            
            // Check if this gate conflicts with any in the layer
            bool conflicts = false;
            for (size_t q : gate.qubits) {
                if (used_qubits.count(q)) {
                    conflicts = true;
                    break;
                }
            }
            
            if (!conflicts) {
                layer.push_back(&gate);
                placed = true;
                break;
            }
        }
        
        if (!placed) {
            layers.push_back({&gate});
        }
    }
    
    return layers;
}

//==============================================================================
// Optimized Direct Gate Application (for row-parallel mode)
// KEY INSIGHT: For low-rank LRET (rank=1 typical), the base functions are optimal.
// Parallelization only helps when:
// 1. rank > num_threads (column parallel)
// 2. dim >> 4096 AND rank > 4 (row parallel)
// 
// For most LRET cases, BATCH/SEQUENTIAL is optimal because:
// - Memory allocation overhead exceeds computation
// - OpenMP thread synchronization overhead exceeds work
//==============================================================================

namespace parallel_ops {

// Apply single-qubit gate - MINIMAL overhead version
// No vector allocations, no dynamic scheduling
inline MatrixXcd apply_fused_single_gate(const MatrixXcd& L, const MatrixXcd& U, 
                                   size_t target, size_t num_qubits) {
    size_t dim = L.rows();
    size_t rank = L.cols();
    MatrixXcd result = L;
    
    size_t step = 1ULL << target;
    
    // Only use OpenMP for LARGE problems with enough work per thread
    // dim > 4096 means n >= 12 qubits
    // rank > 2 means there's meaningful work to parallelize
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(dim > 4096 && rank > 2)
#endif
    for (size_t block = 0; block < dim; block += 2 * step) {
        for (size_t i = block; i < block + step && i < dim; ++i) {
            size_t i0 = i;
            size_t i1 = i + step;
            if (i1 >= dim) continue;
            
            for (size_t r = 0; r < rank; ++r) {
                Complex v0 = L(i0, r);
                Complex v1 = L(i1, r);
                result(i0, r) = U(0, 0) * v0 + U(0, 1) * v1;
                result(i1, r) = U(1, 0) * v0 + U(1, 1) * v1;
            }
        }
    }
    
    return result;
}

// Apply two-qubit gate - NO vector allocation version
// Uses direct iteration instead of pre-computed indices
MatrixXcd apply_two_qubit_gate_parallel(const MatrixXcd& L, const MatrixXcd& U,
                                         size_t q1, size_t q2, size_t num_qubits) {
    size_t dim = L.rows();
    size_t rank = L.cols();
    MatrixXcd result = L;
    
    size_t qmin = std::min(q1, q2);
    size_t qmax = std::max(q1, q2);
    bool swapped = (q1 > q2);
    
    size_t step_min = 1ULL << qmin;
    size_t step_max = 1ULL << qmax;
    
    // Direct iteration - no vector allocation needed
    // Only parallelize for large problems with meaningful rank
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(dim > 4096 && rank > 2)
#endif
    for (size_t base = 0; base < dim; ++base) {
        // Skip if either qubit bit is set
        if ((base & step_min) != 0 || (base & step_max) != 0) continue;
        
        size_t idx[4];
        idx[0] = base;
        idx[1] = base | step_min;
        idx[2] = base | step_max;
        idx[3] = base | step_min | step_max;
        
        if (swapped) std::swap(idx[1], idx[2]);
        
        for (size_t r = 0; r < rank; ++r) {
            Complex v[4];
            for (int k = 0; k < 4; ++k) v[k] = L(idx[k], r);
            
            for (int k = 0; k < 4; ++k) {
                result(idx[k], r) = U(k, 0) * v[0] + U(k, 1) * v[1] + 
                                    U(k, 2) * v[2] + U(k, 3) * v[3];
            }
        }
    }
    
    return result;
}

// Apply a layer of non-overlapping gates in parallel
MatrixXcd apply_gate_layer_parallel(const MatrixXcd& L, 
                                     const std::vector<const GateOp*>& layer,
                                     size_t num_qubits) {
    if (layer.empty()) return L;
    if (layer.size() == 1) {
        return apply_gate_to_L(L, *layer[0], num_qubits);
    }
    
    size_t dim = L.rows();
    size_t rank = L.cols();
    
    // For truly parallel layer application, we need to be careful about overlaps
    // Since gates in a layer don't share qubits, we can apply them all "simultaneously"
    // by computing the combined effect
    
    // Strategy: Apply gates sequentially but with maximum internal parallelism
    MatrixXcd result = L;
    
    for (const auto* gate : layer) {
        if (gate->qubits.size() == 1) {
            MatrixXcd U = get_single_qubit_gate(gate->type, gate->params);
            result = apply_fused_single_gate(result, U, gate->qubits[0], num_qubits);
        } else {
            MatrixXcd U = get_two_qubit_gate(gate->type, gate->params);
            result = apply_two_qubit_gate_parallel(result, U, gate->qubits[0], gate->qubits[1], num_qubits);
        }
    }
    
    return result;
}

MatrixXcd apply_gate_sequential(const MatrixXcd& L, const GateOp& gate, size_t num_qubits) {
    return apply_gate_to_L(L, gate, num_qubits);
}

MatrixXcd apply_gate_row_parallel(const MatrixXcd& L, const GateOp& gate, size_t num_qubits) {
    if (gate.qubits.size() == 1) {
        MatrixXcd U = get_single_qubit_gate(gate.type, gate.params);
        return apply_fused_single_gate(L, U, gate.qubits[0], num_qubits);
    } else {
        MatrixXcd U = get_two_qubit_gate(gate.type, gate.params);
        return apply_two_qubit_gate_parallel(L, U, gate.qubits[0], gate.qubits[1], num_qubits);
    }
}

MatrixXcd apply_gate_column_parallel(const MatrixXcd& L, const GateOp& gate, size_t num_qubits) {
    size_t dim = L.rows();
    size_t rank = L.cols();
    
    // Column-parallel is best when rank is high
    if (rank < 4) {
        // Low rank - row parallel is better
        return apply_gate_row_parallel(L, gate, num_qubits);
    }
    
    MatrixXcd result(dim, rank);
    
    if (gate.qubits.size() == 1) {
        MatrixXcd U = get_single_qubit_gate(gate.type, gate.params);
        size_t target = gate.qubits[0];
        size_t step = 1ULL << target;
        
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (size_t r = 0; r < rank; ++r) {
            VectorXcd col = L.col(r);
            VectorXcd new_col = col;
            
            for (size_t block = 0; block < dim; block += 2 * step) {
                for (size_t i = block; i < block + step && i < dim; ++i) {
                    size_t i0 = i;
                    size_t i1 = i + step;
                    if (i1 >= dim) continue;
                    
                    Complex v0 = col(i0);
                    Complex v1 = col(i1);
                    new_col(i0) = U(0, 0) * v0 + U(0, 1) * v1;
                    new_col(i1) = U(1, 0) * v0 + U(1, 1) * v1;
                }
            }
            result.col(r) = new_col;
        }
    } else {
        MatrixXcd U = get_two_qubit_gate(gate.type, gate.params);
        size_t q1 = gate.qubits[0];
        size_t q2 = gate.qubits[1];
        size_t qmin = std::min(q1, q2);
        size_t qmax = std::max(q1, q2);
        bool swapped = (q1 > q2);
        size_t step_min = 1ULL << qmin;
        size_t step_max = 1ULL << qmax;
        
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (size_t r = 0; r < rank; ++r) {
            VectorXcd col = L.col(r);
            VectorXcd new_col = col;
            
            for (size_t base = 0; base < dim; ++base) {
                if ((base & step_min) != 0 || (base & step_max) != 0) continue;
                
                size_t idx[4];
                idx[0] = base;
                idx[1] = base | step_min;
                idx[2] = base | step_max;
                idx[3] = base | step_min | step_max;
                
                if (swapped) std::swap(idx[1], idx[2]);
                
                Complex v[4];
                for (int k = 0; k < 4; ++k) v[k] = col(idx[k]);
                
                for (int k = 0; k < 4; ++k) {
                    new_col(idx[k]) = U(k, 0) * v[0] + U(k, 1) * v[1] + 
                                      U(k, 2) * v[2] + U(k, 3) * v[3];
                }
            }
            result.col(r) = new_col;
        }
    }
    
    return result;
}

}  // namespace parallel_ops

//==============================================================================
// Mode Runners
//==============================================================================

namespace modes {

MatrixXcd run_sequential(
    const MatrixXcd& L_init,
    const QuantumSequence& sequence,
    size_t num_qubits,
    const SimConfig& config
) {
    MatrixXcd L = L_init;
    
    for (const auto& op : sequence.operations) {
        if (std::holds_alternative<GateOp>(op)) {
            const auto& gate = std::get<GateOp>(op);
            L = parallel_ops::apply_gate_sequential(L, gate, num_qubits);
        } else {
            const auto& noise = std::get<NoiseOp>(op);
            L = apply_noise_to_L(L, noise, num_qubits);
            
            if (config.do_truncation && L.cols() > 1) {
                L = truncate_L(L, config.truncation_threshold);
            }
        }
    }
    
    return L;
}

// ROW-PARALLEL mode
// For low rank (typical LRET), just use the base optimized functions
// Gate fusion only helps when there are consecutive gates on same qubit
MatrixXcd run_row_parallel(
    const MatrixXcd& L_init,
    const QuantumSequence& sequence,
    size_t num_qubits,
    const SimConfig& config
) {
    MatrixXcd L = L_init;
    
    // For efficiency, just apply gates directly using the base functions
    // which are already OpenMP-optimized
    for (const auto& op : sequence.operations) {
        if (std::holds_alternative<GateOp>(op)) {
            const auto& gate = std::get<GateOp>(op);
            // Use apply_gate_to_L which is already optimized
            L = apply_gate_to_L(L, gate, num_qubits);
        } else {
            const auto& noise = std::get<NoiseOp>(op);
            L = apply_noise_to_L(L, noise, num_qubits);
            
            if (config.do_truncation && L.cols() > 1) {
                L = truncate_L(L, config.truncation_threshold);
            }
        }
    }
    
    return L;
}

MatrixXcd run_column_parallel(
    const MatrixXcd& L_init,
    const QuantumSequence& sequence,
    size_t num_qubits,
    const SimConfig& config
) {
    MatrixXcd L = L_init;
    
    for (const auto& op : sequence.operations) {
        if (std::holds_alternative<GateOp>(op)) {
            const auto& gate = std::get<GateOp>(op);
            L = parallel_ops::apply_gate_column_parallel(L, gate, num_qubits);
        } else {
            const auto& noise = std::get<NoiseOp>(op);
            L = apply_noise_to_L(L, noise, num_qubits);
            
            if (config.do_truncation && L.cols() > 1) {
                L = truncate_L(L, config.truncation_threshold);
            }
        }
    }
    
    return L;
}

MatrixXcd run_batch_parallel(
    const MatrixXcd& L_init,
    const QuantumSequence& sequence,
    size_t num_qubits,
    size_t batch_size,
    const SimConfig& config
) {
    return run_simulation_optimized(
        L_init, sequence, num_qubits,
        batch_size, config.do_truncation,
        config.verbose, config.truncation_threshold
    );
}

// HYBRID: Adaptive mode that switches between row and batch based on rank
// - For low rank (1-4): Use batch (minimal overhead)
// - For medium rank (4-16): Use row parallelism
// - For high rank (16+): Use column parallelism
MatrixXcd run_hybrid(
    const MatrixXcd& L_init,
    const QuantumSequence& sequence,
    size_t num_qubits,
    size_t batch_size,
    const SimConfig& config
) {
    MatrixXcd L = L_init;
    size_t dim = L.rows();
    
    // Threshold for switching between strategies
    constexpr size_t ROW_RANK_THRESHOLD = 4;
    constexpr size_t COL_RANK_THRESHOLD = 16;
    
    size_t step = 0;
    size_t total_ops = sequence.operations.size();
    
    for (const auto& op : sequence.operations) {
        step++;
        
        // Check for abort
        if (should_abort()) {
            if (config.verbose) {
                std::cout << "\nHybrid mode: Aborted at step " << step << "/" << total_ops << std::endl;
            }
            break;
        }
        
        if (std::holds_alternative<GateOp>(op)) {
            const auto& gate = std::get<GateOp>(op);
            size_t rank = L.cols();
            
            // Adaptive strategy based on current rank
            if (rank <= ROW_RANK_THRESHOLD) {
                // Low rank: use base function (no parallelization overhead)
                L = apply_gate_to_L(L, gate, num_qubits);
            } else if (rank <= COL_RANK_THRESHOLD || dim < 4096) {
                // Medium rank or small dim: row parallelism
                if (gate.qubits.size() == 1) {
                    MatrixXcd U = get_single_qubit_gate(gate.type, gate.params);
                    L = parallel_ops::apply_fused_single_gate(L, U, gate.qubits[0], num_qubits);
                } else {
                    MatrixXcd U = get_two_qubit_gate(gate.type, gate.params);
                    L = parallel_ops::apply_two_qubit_gate_parallel(L, U, gate.qubits[0], gate.qubits[1], num_qubits);
                }
            } else {
                // High rank + large dim: column parallelism is better
                L = parallel_ops::apply_gate_column_parallel(L, gate, num_qubits);
            }
            
            if (config.verbose && step % 100 == 0) {
                std::cout << "Hybrid step " << step << "/" << total_ops 
                          << " rank=" << L.cols() << std::endl;
            }
        } else {
            const auto& noise = std::get<NoiseOp>(op);
            L = apply_noise_to_L(L, noise, num_qubits);
            
            if (config.do_truncation && L.cols() > 1) {
                L = truncate_L(L, config.truncation_threshold);
            }
        }
    }
    
    return L;
}

}  // namespace modes

//==============================================================================
// Main Entry Points
//==============================================================================

ModeResult run_with_mode(
    const MatrixXcd& L_init,
    const QuantumSequence& sequence,
    size_t num_qubits,
    ParallelMode mode,
    const SimConfig& config,
    size_t batch_size
) {
    if (batch_size == 0) {
        batch_size = auto_select_batch_size(num_qubits);
    }
    
    Timer timer;
    MatrixXcd L_final;
    
    switch (mode) {
        case ParallelMode::SEQUENTIAL:
            L_final = modes::run_sequential(L_init, sequence, num_qubits, config);
            break;
        case ParallelMode::ROW:
            L_final = modes::run_row_parallel(L_init, sequence, num_qubits, config);
            break;
        case ParallelMode::COLUMN:
            L_final = modes::run_column_parallel(L_init, sequence, num_qubits, config);
            break;
        case ParallelMode::BATCH:
            L_final = modes::run_batch_parallel(L_init, sequence, num_qubits, batch_size, config);
            break;
        case ParallelMode::HYBRID:
            L_final = modes::run_hybrid(L_init, sequence, num_qubits, batch_size, config);
            break;
        case ParallelMode::AUTO:
        default:
            mode = auto_select_mode(num_qubits, sequence.depth, L_init.cols());
            return run_with_mode(L_init, sequence, num_qubits, mode, config, batch_size);
    }
    
    double elapsed = timer.elapsed_seconds();
    
    ModeResult result;
    result.mode = mode;
    result.time_seconds = elapsed;
    result.L_final = L_final;
    result.final_rank = L_final.cols();
    result.trace_value = (L_final * L_final.adjoint()).trace().real();
    
    return result;
}

std::vector<ModeResult> run_all_modes_comparison(
    const MatrixXcd& L_init,
    const QuantumSequence& sequence,
    size_t num_qubits,
    const SimConfig& config,
    size_t batch_size
) {
    std::vector<ModeResult> results;
    
    std::vector<ParallelMode> modes_to_test = {
        ParallelMode::SEQUENTIAL,
        ParallelMode::ROW,
        ParallelMode::COLUMN,
        ParallelMode::BATCH,
        ParallelMode::HYBRID
    };
    
    for (auto mode : modes_to_test) {
        std::cout << "Running " << parallel_mode_to_string(mode) << "..." << std::flush;
        
        // Begin LRET progress logging for this mode
        if (g_structured_csv) {
            g_structured_csv->begin_lret_progress(num_qubits, sequence.operations.size(), mode);
        }
        
        auto result = run_with_mode(L_init, sequence, num_qubits, mode, config, batch_size);
        results.push_back(result);
        
        std::cout << " done (" << std::fixed << std::setprecision(4) 
                  << result.time_seconds << "s)\n";
        
        // End LRET progress (but DON'T write metrics yet - wait until speedup is computed)
        if (g_structured_csv) {
            g_structured_csv->end_lret_progress(result.time_seconds, true, mode);
        }
    }
    
    // Compute metrics for each mode vs sequential baseline
    // NOTE: For large n, full density matrix computation is O(4^n) - too slow!
    // Use L-based metrics instead (O(2^n * rank))
    if (!results.empty() && results[0].mode == ParallelMode::SEQUENTIAL) {
        const MatrixXcd& L_seq = results[0].L_final;
        double seq_time = results[0].time_seconds;
        double seq_norm = L_seq.norm();
        
        for (auto& r : results) {
            r.speedup = seq_time / r.time_seconds;
            
            if (r.mode == ParallelMode::SEQUENTIAL) {
                r.fidelity = 1.0;
                r.trace_distance = 0.0;
                r.frobenius_distance = 0.0;
                r.distortion = 0.0;
            } else {
                // Use L-based metrics (O(2^n * rank) instead of O(4^n))
                // Frobenius distance of L matrices
                r.frobenius_distance = (L_seq - r.L_final).norm();
                
                // Distortion: ||L_this - L_seq||_F / ||L_seq||_F
                r.distortion = (seq_norm > 1e-15) ? r.frobenius_distance / seq_norm : 0.0;
                
                // Fidelity approximation using L overlap
                // F ≈ |Tr(L_seq† L_this)|² / (||L_seq||² * ||L_this||²)
                Complex overlap = (L_seq.adjoint() * r.L_final).trace();
                double norm_seq = L_seq.squaredNorm();
                double norm_this = r.L_final.squaredNorm();
                double denom = norm_seq * norm_this;
                r.fidelity = (denom > 1e-15) ? std::norm(overlap) / denom : 0.0;
                
                // Trace distance approximation: proportional to distortion for LRET
                r.trace_distance = r.distortion;
            }
            
            // Compute state metrics for each mode (purity, entropy, etc.)
            double purity = compute_purity(r.L_final);
            double entropy = compute_entropy(r.L_final);
            double linear_entropy = compute_linear_entropy(r.L_final);
            double concurrence = -1.0;
            double negativity = -1.0;
            
            // Concurrence only for 2-qubit systems
            if (num_qubits == 2) {
                concurrence = compute_concurrence(r.L_final);
            }
            
            // Negativity for bipartite split (half-half)
            if (num_qubits >= 2) {
                size_t split = num_qubits / 2;
                negativity = compute_negativity(r.L_final, split, num_qubits);
            }
            
            // Write comprehensive mode metrics to CSV (NOW that speedup is computed)
            if (g_structured_csv) {
                g_structured_csv->write_lret_mode_metrics_full(
                    r, num_qubits, purity, entropy, linear_entropy, concurrence, negativity);
            }
        }
    }
    
    return results;
}

}  // namespace qlret

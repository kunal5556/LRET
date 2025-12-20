#include "parallel_modes.h"
#include "gates_and_noise.h"
#include "simulator.h"
#include "utils.h"
#include <iostream>
#include <thread>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace qlret {

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
    // Based on benchmarking, BATCH mode with optimized direct gate application
    // is consistently the fastest. For very small problems, sequential is fine.
    size_t dim = 1ULL << num_qubits;
    
    if (dim < 64) {
        return ParallelMode::SEQUENTIAL;
    }
    
    // For all larger problems, batch mode is empirically fastest
    return ParallelMode::BATCH;
}

//==============================================================================
// Optimized Parallel Gate Application
// 
// KEY INSIGHT: We parallelize WITHIN the efficient direct gate application,
// NOT by expanding to full 2^n × 2^n space (which is O(4^n) and terrible).
//
// All modes use O(2^n × rank) complexity via direct application.
// The difference is HOW we parallelize the inner loops.
//==============================================================================

namespace parallel_ops {

//------------------------------------------------------------------------------
// Sequential: No parallelization, uses optimized direct application
//------------------------------------------------------------------------------
MatrixXcd apply_gate_sequential(const MatrixXcd& L, const GateOp& gate, size_t num_qubits) {
    // This already uses apply_single_gate_direct / apply_two_qubit_gate_direct
    // which are O(2^n × rank) - optimal complexity
    return apply_gate_to_L(L, gate, num_qubits);
}

//------------------------------------------------------------------------------
// Row-Parallel: Parallelize over row-pairs in direct application
// For single-qubit gate on qubit q, we process pairs (i, i + 2^q)
// The parallelization is over the "block" index in the direct method
//------------------------------------------------------------------------------
MatrixXcd apply_gate_row_parallel(const MatrixXcd& L, const GateOp& gate, size_t num_qubits) {
    if (gate.qubits.size() == 1) {
        // Single-qubit gate: parallelize over row blocks
        MatrixXcd U = get_single_qubit_gate(gate.type, gate.params);
        size_t dim = L.rows();
        size_t rank = L.cols();
        MatrixXcd result = L;
        
        size_t target = gate.qubits[0];
        size_t step = 1ULL << target;
        
        // Number of independent pairs to process
        size_t num_pairs = dim / 2;
        
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (size_t pair_idx = 0; pair_idx < num_pairs; ++pair_idx) {
            // Convert pair_idx to actual row indices
            size_t block = pair_idx / step;
            size_t offset = pair_idx % step;
            size_t i0 = block * (2 * step) + offset;
            size_t i1 = i0 + step;
            
            if (i1 >= dim) continue;
            
            for (size_t r = 0; r < rank; ++r) {
                Complex v0 = L(i0, r);
                Complex v1 = L(i1, r);
                result(i0, r) = U(0, 0) * v0 + U(0, 1) * v1;
                result(i1, r) = U(1, 0) * v0 + U(1, 1) * v1;
            }
        }
        return result;
        
    } else {
        // Two-qubit gate: more complex but still O(2^n × rank)
        MatrixXcd U = get_two_qubit_gate(gate.type, gate.params);
        size_t dim = L.rows();
        size_t rank = L.cols();
        MatrixXcd result = L;
        
        size_t q1 = gate.qubits[0];
        size_t q2 = gate.qubits[1];
        size_t qmin = std::min(q1, q2);
        size_t qmax = std::max(q1, q2);
        bool swapped = (q1 > q2);
        
        size_t step_min = 1ULL << qmin;
        size_t step_max = 1ULL << qmax;
        
        // Count valid base indices (where both qubit bits are 0)
        std::vector<size_t> base_indices;
        for (size_t base = 0; base < dim; ++base) {
            if ((base & step_min) == 0 && (base & step_max) == 0) {
                base_indices.push_back(base);
            }
        }
        
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (size_t bi = 0; bi < base_indices.size(); ++bi) {
            size_t base = base_indices[bi];
            
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
}

//------------------------------------------------------------------------------
// Column-Parallel: Parallelize over rank columns
// Each column of L is processed independently
//------------------------------------------------------------------------------
MatrixXcd apply_gate_column_parallel(const MatrixXcd& L, const GateOp& gate, size_t num_qubits) {
    size_t dim = L.rows();
    size_t rank = L.cols();
    MatrixXcd result(dim, rank);
    
    if (gate.qubits.size() == 1) {
        MatrixXcd U = get_single_qubit_gate(gate.type, gate.params);
        size_t target = gate.qubits[0];
        size_t step = 1ULL << target;
        
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (size_t r = 0; r < rank; ++r) {
            // Process this column
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

MatrixXcd run_row_parallel(
    const MatrixXcd& L_init,
    const QuantumSequence& sequence,
    size_t num_qubits,
    const SimConfig& config
) {
    MatrixXcd L = L_init;
    
    for (const auto& op : sequence.operations) {
        if (std::holds_alternative<GateOp>(op)) {
            const auto& gate = std::get<GateOp>(op);
            L = parallel_ops::apply_gate_row_parallel(L, gate, num_qubits);
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
    // Uses the highly optimized simulator with batched gate application
    return run_simulation_optimized(
        L_init, sequence, num_qubits,
        batch_size, config.do_truncation,
        config.verbose, config.truncation_threshold
    );
}

MatrixXcd run_hybrid(
    const MatrixXcd& L_init,
    const QuantumSequence& sequence,
    size_t num_qubits,
    size_t batch_size,
    const SimConfig& config
) {
    // Hybrid: batch gates together, apply each with row parallelization
    MatrixXcd L = L_init;
    std::vector<GateOp> gate_batch;
    
    for (const auto& op : sequence.operations) {
        if (std::holds_alternative<GateOp>(op)) {
            gate_batch.push_back(std::get<GateOp>(op));
            
            if (gate_batch.size() >= batch_size) {
                for (const auto& gate : gate_batch) {
                    L = parallel_ops::apply_gate_row_parallel(L, gate, num_qubits);
                }
                gate_batch.clear();
            }
        } else {
            // Flush gates before noise
            for (const auto& gate : gate_batch) {
                L = parallel_ops::apply_gate_row_parallel(L, gate, num_qubits);
            }
            gate_batch.clear();
            
            const auto& noise = std::get<NoiseOp>(op);
            L = apply_noise_to_L(L, noise, num_qubits);
            
            if (config.do_truncation && L.cols() > 1) {
                L = truncate_L(L, config.truncation_threshold);
            }
        }
    }
    
    // Flush remaining
    for (const auto& gate : gate_batch) {
        L = parallel_ops::apply_gate_row_parallel(L, gate, num_qubits);
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
        
        auto result = run_with_mode(L_init, sequence, num_qubits, mode, config, batch_size);
        results.push_back(result);
        
        std::cout << " done (" << std::fixed << std::setprecision(3) 
                  << result.time_seconds << "s)\n";
    }
    
    return results;
}

}  // namespace qlret

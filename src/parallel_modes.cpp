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

// Auto-select batch size based on qubit count
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

// Auto-select best parallelization mode
ParallelMode auto_select_mode(size_t num_qubits, size_t depth, size_t rank_estimate) {
    size_t dim = 1ULL << num_qubits;
    
    if (dim < 64) {
        return ParallelMode::SEQUENTIAL;
    }
    
    if (dim < 256) {
        return depth > 50 ? ParallelMode::BATCH : ParallelMode::SEQUENTIAL;
    }
    
    // For larger problems, hybrid is usually best
    if (depth > 20) {
        return ParallelMode::HYBRID;
    }
    
    return ParallelMode::ROW;
}

namespace parallel_ops {

// Sequential gate application
MatrixXcd apply_gate_sequential(const MatrixXcd& L, const GateOp& gate, size_t num_qubits) {
    return apply_gate_to_L(L, gate, num_qubits);
}

// Row-parallel gate application
MatrixXcd apply_gate_row_parallel(const MatrixXcd& L, const GateOp& gate, size_t num_qubits) {
    MatrixXcd U = get_single_qubit_gate(gate.type, gate.params);
    if (gate.qubits.size() == 2) {
        U = get_two_qubit_gate(gate.type, gate.params);
    }
    
    // Expand to full space
    MatrixXcd U_full;
    if (gate.qubits.size() == 1) {
        U_full = expand_single_gate(U, gate.qubits[0], num_qubits);
    } else {
        U_full = expand_two_qubit_gate(U, gate.qubits[0], gate.qubits[1], num_qubits);
    }
    
    size_t dim = L.rows();
    size_t rank = L.cols();
    MatrixXcd L_new(dim, rank);
    
    // Parallelize over rows
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (size_t row = 0; row < dim; ++row) {
        for (size_t col = 0; col < rank; ++col) {
            Complex sum = 0;
            for (size_t k = 0; k < dim; ++k) {
                sum += U_full(row, k) * L(k, col);
            }
            L_new(row, col) = sum;
        }
    }
    
    return L_new;
}

// Column-parallel gate application
MatrixXcd apply_gate_column_parallel(const MatrixXcd& L, const GateOp& gate, size_t num_qubits) {
    MatrixXcd U = get_single_qubit_gate(gate.type, gate.params);
    if (gate.qubits.size() == 2) {
        U = get_two_qubit_gate(gate.type, gate.params);
    }
    
    MatrixXcd U_full;
    if (gate.qubits.size() == 1) {
        U_full = expand_single_gate(U, gate.qubits[0], num_qubits);
    } else {
        U_full = expand_two_qubit_gate(U, gate.qubits[0], gate.qubits[1], num_qubits);
    }
    
    size_t dim = L.rows();
    size_t rank = L.cols();
    MatrixXcd L_new(dim, rank);
    
    // Parallelize over columns (rank components)
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (size_t col = 0; col < rank; ++col) {
        L_new.col(col) = U_full * L.col(col);
    }
    
    return L_new;
}

}  // namespace parallel_ops

namespace modes {

// Sequential simulation
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

// Row-parallel simulation
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

// Column-parallel simulation
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

// Batch parallel simulation (existing optimized implementation)
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

// Hybrid: row parallel for gates + batch organization
MatrixXcd run_hybrid(
    const MatrixXcd& L_init,
    const QuantumSequence& sequence,
    size_t num_qubits,
    size_t batch_size,
    const SimConfig& config
) {
    MatrixXcd L = L_init;
    std::vector<GateOp> gate_batch;
    
    for (const auto& op : sequence.operations) {
        if (std::holds_alternative<GateOp>(op)) {
            gate_batch.push_back(std::get<GateOp>(op));
            
            // Apply batch when full
            if (gate_batch.size() >= batch_size) {
                for (const auto& gate : gate_batch) {
                    L = parallel_ops::apply_gate_row_parallel(L, gate, num_qubits);
                }
                gate_batch.clear();
            }
        } else {
            // Flush remaining gates before noise
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
    
    // Apply remaining gates
    for (const auto& gate : gate_batch) {
        L = parallel_ops::apply_gate_row_parallel(L, gate, num_qubits);
    }
    
    return L;
}

}  // namespace modes

// Run with specific mode
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

// Run all modes for comparison
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

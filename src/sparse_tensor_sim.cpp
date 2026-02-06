/**
 * @file sparse_tensor_sim.cpp
 * @brief Sparse Tensor Approximation for LRET Simulation
 * 
 * Phase 2B of Advanced Row-Parallel Optimization.
 * 
 * IMPLEMENTATION NOTES:
 * 
 * The hybrid sparse-dense approach works as follows:
 * 
 * 1. SPARSIFICATION:
 *    After noise application, many L elements decay to near-zero.
 *    We threshold: if |L_{ij}| < ε, set L_{ij} = 0.
 *    Then remove columns that are entirely zero (rank reduction).
 *    
 * 2. GATE APPLICATION:
 *    Gates operate on dense L (row-pair operations need random access).
 *    After gate application, we re-sparsify.
 *    
 * 3. ADAPTIVE SWITCHING:
 *    We track the sparsity ratio after each noise operation.
 *    If sparsity ratio < 0.5 (>50% zeros), sparsification is beneficial.
 *    If sparsity ratio > 0.8 (too dense), skip sparsification overhead.
 * 
 * 4. MEMORY SAVINGS:
 *    Dense: 16 × dim × rank bytes (complex<double> = 16 bytes each)
 *    Sparse CSC: (16 + 4) × nnz + 4 × (rank + 1) bytes
 *    Beneficial when nnz/total < 0.5
 * 
 * @see sparse_tensor_sim.h for API documentation
 * @see ROW_PARALLEL_ADVANCED_OPTIMIZATION_PLAN.md Phase 2B
 */

#include "sparse_tensor_sim.h"
#include "gates_and_noise.h"
#include "simulator.h"
#include "iterative_compression.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <chrono>

namespace qlret {

//==============================================================================
// Sparsity Analysis
//==============================================================================

SparsityStats analyze_sparsity(const MatrixXcd& L, double threshold) {
    SparsityStats stats;
    const size_t rows = static_cast<size_t>(L.rows());
    const size_t cols = static_cast<size_t>(L.cols());

    stats.total_elements = rows * cols;

    size_t nnz = 0;
    for (size_t j = 0; j < cols; ++j) {
        for (size_t i = 0; i < rows; ++i) {
            if (std::abs(L(i, j)) >= threshold) {
                nnz++;
            }
        }
    }

    stats.nonzero_elements = nnz;
    stats.sparsity_ratio = static_cast<double>(nnz) / static_cast<double>(stats.total_elements);

    // Memory estimates
    // Dense: 16 bytes per element (complex<double>)
    stats.dense_memory_bytes = stats.total_elements * 16;

    // Sparse CSC: (16 + 4) per nonzero + 4 per column pointer + overhead
    // value array: 16 * nnz, inner index array: 4 * nnz, outer index: 4 * (cols+1)
    stats.sparse_memory_bytes = nnz * 20 + (cols + 1) * 4 + 64;  // 64 for overhead

    stats.memory_ratio = static_cast<double>(stats.sparse_memory_bytes)
                       / static_cast<double>(stats.dense_memory_bytes + 1);

    return stats;
}

//==============================================================================
// Dense ↔ Sparse Conversion
//==============================================================================

SparseMatrixXcd to_sparse(const MatrixXcd& L, double threshold) {
    const int rows = static_cast<int>(L.rows());
    const int cols = static_cast<int>(L.cols());

    // Count nonzeros for preallocation
    std::vector<int> nnz_per_col(cols, 0);
    for (int j = 0; j < cols; ++j) {
        for (int i = 0; i < rows; ++i) {
            if (std::abs(L(i, j)) >= threshold) {
                nnz_per_col[j]++;
            }
        }
    }

    SparseMatrixXcd S(rows, cols);
    S.reserve(Eigen::VectorXi::Map(nnz_per_col.data(), cols));

    for (int j = 0; j < cols; ++j) {
        for (int i = 0; i < rows; ++i) {
            if (std::abs(L(i, j)) >= threshold) {
                S.insert(i, j) = L(i, j);
            }
        }
    }

    S.makeCompressed();
    return S;
}

MatrixXcd to_dense(const SparseMatrixXcd& L_sparse) {
    return MatrixXcd(L_sparse);
}

//==============================================================================
// In-place Sparsification
//==============================================================================

size_t sparsify_inplace(MatrixXcd& L, double threshold) {
    const size_t rows = static_cast<size_t>(L.rows());
    const size_t cols = static_cast<size_t>(L.cols());
    size_t zeroed = 0;

    for (size_t j = 0; j < cols; ++j) {
        for (size_t i = 0; i < rows; ++i) {
            if (std::abs(L(i, j)) < threshold) {
                L(i, j) = Complex(0.0, 0.0);
                zeroed++;
            }
        }
    }

    return zeroed;
}

/**
 * Remove zero columns from L and return the compacted matrix.
 * This reduces the rank of L.
 */
static MatrixXcd remove_zero_columns(const MatrixXcd& L, double threshold = 1e-15) {
    const size_t cols = static_cast<size_t>(L.cols());

    // Find non-zero columns
    std::vector<size_t> kept;
    for (size_t j = 0; j < cols; ++j) {
        double col_norm = L.col(j).norm();
        if (col_norm > threshold) {
            kept.push_back(j);
        }
    }

    if (kept.empty()) {
        // Keep at least one column (the first, even if zero)
        return L.col(0);
    }

    if (kept.size() == cols) {
        return L;  // No columns removed
    }

    MatrixXcd L_compact(L.rows(), kept.size());
    for (size_t i = 0; i < kept.size(); ++i) {
        L_compact.col(i) = L.col(kept[i]);
    }

    return L_compact;
}

//==============================================================================
// Sparse Mode Detection
//==============================================================================

bool should_use_sparse(const QuantumSequence& sequence, const SparseConfig& config) {
    if (sequence.num_qubits < config.min_qubits) {
        return false;
    }

    size_t noise_count = 0;
    size_t gate_count = 0;

    for (const auto& op : sequence.operations) {
        if (std::holds_alternative<NoiseOp>(op)) {
            noise_count++;
        } else if (std::holds_alternative<GateOp>(op)) {
            gate_count++;
        }
    }

    size_t total = noise_count + gate_count;
    if (total == 0) return false;

    double noise_ratio = static_cast<double>(noise_count) / static_cast<double>(total);

    // Use sparse mode when noise is dominant (>50% of operations)
    return noise_ratio > 0.5;
}

//==============================================================================
// Sparse-Aware Noise Application
//==============================================================================

MatrixXcd apply_noise_sparse(
    const MatrixXcd& L,
    const NoiseOp& noise_op,
    size_t num_qubits,
    const SparseConfig& config
) {
    // Step 1: Apply noise using standard Kraus concatenation
    MatrixXcd L_noisy = apply_noise_to_L(L, noise_op, num_qubits);

    // Step 2: Sparsify — zero out small elements
    size_t zeroed = sparsify_inplace(L_noisy, config.sparsity_threshold);

    // Step 3: Remove zero columns (reduce rank)
    MatrixXcd L_compact = remove_zero_columns(L_noisy, config.sparsity_threshold);

    // Step 4: Renormalize to preserve trace(ρ) = 1
    double trace = L_compact.squaredNorm();
    if (trace > 1e-15) {
        L_compact /= std::sqrt(trace);
    }

    if (config.verbose) {
        SparsityStats stats = analyze_sparsity(L_compact, config.sparsity_threshold);
        std::cout << "  [Sparse] Zeroed " << zeroed << " elements, "
                  << "rank " << L.cols() << " → " << L_compact.cols()
                  << ", sparsity ratio = " << stats.sparsity_ratio
                  << std::endl;
    }

    return L_compact;
}

//==============================================================================
// Full Sparse-Aware Simulation
//==============================================================================

MatrixXcd run_simulation_sparse(
    const MatrixXcd& L_init,
    const QuantumSequence& sequence,
    size_t num_qubits,
    const SparseConfig& config,
    double truncation_threshold,
    bool verbose
) {
    MatrixXcd L = L_init;
    size_t step = 0;
    bool sparse_active = false;
    size_t ops_since_compress = 0;

    // Collect gates for batched application
    std::vector<GateOp> gate_batch;
    const size_t batch_size = 64;

    for (const auto& op : sequence.operations) {
        step++;

        if (std::holds_alternative<GateOp>(op)) {
            const auto& gate = std::get<GateOp>(op);
            gate_batch.push_back(gate);

            if (gate_batch.size() >= batch_size) {
                L = apply_gates_batched(L, gate_batch, num_qubits, batch_size);
                gate_batch.clear();
            }
        } else if (std::holds_alternative<NoiseOp>(op)) {
            // Flush pending gates
            if (!gate_batch.empty()) {
                L = apply_gates_batched(L, gate_batch, num_qubits, batch_size);
                gate_batch.clear();
            }

            const auto& noise = std::get<NoiseOp>(op);

            // Apply noise with sparse-aware truncation
            if (sparse_active) {
                L = apply_noise_sparse(L, noise, num_qubits, config);
            } else {
                // Use iterative compression (Phase 1 method)
                L = apply_noise_iterative_simple(L, noise, num_qubits, truncation_threshold);
            }

            ops_since_compress++;

            // Check if we should switch sparse mode on/off
            if (ops_since_compress >= config.compress_interval) {
                SparsityStats stats = analyze_sparsity(L, config.sparsity_threshold);

                if (!sparse_active && stats.sparsity_ratio < config.min_sparsity_benefit) {
                    sparse_active = true;
                    if (verbose) {
                        std::cout << "Step " << step << ": Activating sparse mode "
                                  << "(sparsity ratio = " << stats.sparsity_ratio << ")"
                                  << std::endl;
                    }
                } else if (sparse_active && stats.sparsity_ratio > config.redensify_threshold) {
                    sparse_active = false;
                    if (verbose) {
                        std::cout << "Step " << step << ": Deactivating sparse mode "
                                  << "(sparsity ratio = " << stats.sparsity_ratio << ")"
                                  << std::endl;
                    }
                }

                // Apply sparsification as additional truncation when in sparse mode
                if (sparse_active) {
                    sparsify_inplace(L, config.sparsity_threshold);
                    L = remove_zero_columns(L, config.sparsity_threshold);

                    double trace = L.squaredNorm();
                    if (trace > 1e-15) {
                        L /= std::sqrt(trace);
                    }
                }

                ops_since_compress = 0;
            }

            // Fallback truncation if rank is still too high
            if (L.cols() > 1) {
                L = truncate_L(L, truncation_threshold);
            }

            if (verbose) {
                std::cout << "Step " << step << ": noise applied, rank = " << L.cols()
                          << (sparse_active ? " [SPARSE]" : " [DENSE]")
                          << std::endl;
            }
        }
    }

    // Flush remaining gates
    if (!gate_batch.empty()) {
        L = apply_gates_batched(L, gate_batch, num_qubits, batch_size);
    }

    // Final truncation
    if (L.cols() > 1) {
        L = truncate_L(L, truncation_threshold);
    }

    return L;
}

}  // namespace qlret

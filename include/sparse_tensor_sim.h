#pragma once

/**
 * @file sparse_tensor_sim.h
 * @brief Sparse Tensor Approximation for LRET Simulation
 * 
 * Phase 2B of Advanced Row-Parallel Optimization.
 * 
 * BACKGROUND:
 * In noisy quantum circuits, many elements of the L matrix (dim × rank)
 * decay to near-zero due to decoherence. After noise + truncation:
 * - Amplitude damping drives off-diagonal elements toward 0
 * - Depolarizing noise pushes toward the maximally mixed state
 * - Phase damping kills coherences
 * 
 * When L becomes >50% zero (by element count), storing it as
 * Eigen::SparseMatrix saves memory and speeds up gate application.
 * 
 * HYBRID REPRESENTATION:
 * We use a hybrid approach:
 * 1. Sparse L for storage-heavy bulk operations
 * 2. Dense L_core for active subspace (gate application uses dense ops)
 * 
 * Gate application: convert sparse → dense → apply gate → sparsify
 * Noise application: apply Kraus (dense) → sparsify result
 * 
 * The benefit comes from:
 * - Memory: 3-10× savings for highly sparse L (>80% zeros)
 * - Speed: Sparse matrix operations skip zero elements
 * - Truncation: Sparsification is an additional truncation pathway
 * 
 * WHEN TO USE:
 * Auto-detected when noise ratio is high (>50% noise operations in circuit)
 * and qubit count >= 8 (smaller circuits don't benefit enough).
 * 
 * Reference:
 *   ROW_PARALLEL_ADVANCED_OPTIMIZATION_PLAN.md Phase 2B
 */

#include "types.h"
#include <Eigen/Sparse>
#include <vector>
#include <string>

namespace qlret {

// Type alias for sparse complex matrix
using SparseMatrixXcd = Eigen::SparseMatrix<std::complex<double>>;

//==============================================================================
// Configuration
//==============================================================================

/**
 * @brief Configuration for sparse tensor simulation
 */
struct SparseConfig {
    /// Elements with |value| < threshold are zeroed out
    double sparsity_threshold = 1e-8;

    /// Maximum rank for dense core
    size_t max_dense_rank = 64;

    /// Minimum sparsity ratio (nonzero/total) to use sparse mode.
    /// Below this ratio, sparse representation is worthwhile.
    /// 0.5 means "use sparse when >50% of L is zero"
    double min_sparsity_benefit = 0.5;

    /// Minimum qubit count to activate sparse mode
    size_t min_qubits = 6;

    /// Re-densify if sparsity drops below this (matrix is too dense)
    double redensify_threshold = 0.8;

    /// Compress sparse matrix every N operations
    size_t compress_interval = 10;

    /// Enable verbose logging
    bool verbose = false;
};

//==============================================================================
// Sparsity Statistics
//==============================================================================

/**
 * @brief Statistics about L matrix sparsity
 */
struct SparsityStats {
    size_t total_elements = 0;    ///< dim × rank
    size_t nonzero_elements = 0;  ///< Count of |L_{ij}| >= threshold
    double sparsity_ratio = 0.0;  ///< nonzero / total (lower = more sparse)
    size_t dense_memory_bytes = 0;  ///< Memory if stored dense
    size_t sparse_memory_bytes = 0; ///< Approximate memory as sparse
    double memory_ratio = 0.0;     ///< sparse / dense (lower = more savings)

    bool is_sparse_beneficial() const {
        return sparsity_ratio < 0.5 && memory_ratio < 0.8;
    }
};

//==============================================================================
// Core Functions
//==============================================================================

/**
 * @brief Analyze sparsity of an L matrix
 * 
 * Counts nonzero elements and estimates memory savings from
 * sparse representation.
 * 
 * @param L Dense L matrix to analyze
 * @param threshold Elements with |value| < threshold count as zero
 * @return Sparsity statistics
 */
SparsityStats analyze_sparsity(const MatrixXcd& L, double threshold = 1e-8);

/**
 * @brief Convert dense L to sparse representation
 * 
 * Elements with |value| < threshold are dropped.
 * 
 * @param L Dense L matrix
 * @param threshold Sparsification threshold
 * @return Sparse L matrix
 */
SparseMatrixXcd to_sparse(const MatrixXcd& L, double threshold = 1e-8);

/**
 * @brief Convert sparse L back to dense representation
 * 
 * @param L_sparse Sparse L matrix
 * @return Dense L matrix
 */
MatrixXcd to_dense(const SparseMatrixXcd& L_sparse);

/**
 * @brief Sparsify a dense L matrix in-place
 * 
 * Sets elements with |value| < threshold to zero, then re-normalizes.
 * This is a lightweight truncation that reduces rank growth from noise.
 * 
 * @param L Dense L matrix (modified in place)
 * @param threshold Sparsification threshold
 * @return Number of elements zeroed out
 */
size_t sparsify_inplace(MatrixXcd& L, double threshold = 1e-8);

/**
 * @brief Determine if sparse mode would benefit this circuit
 * 
 * Heuristic: count noise operations as fraction of total operations.
 * High noise ratio → state becomes sparse → sparse mode benefits.
 * 
 * @param sequence The quantum circuit
 * @param config Sparse configuration
 * @return true if sparse mode is recommended
 */
bool should_use_sparse(const QuantumSequence& sequence, const SparseConfig& config);

/**
 * @brief Apply noise with sparsity-aware truncation
 * 
 * Algorithm:
 * 1. Apply Kraus operators (standard path, result is dense)
 * 2. Sparsify: zero out small elements
 * 3. Remove zero columns (reduce rank)
 * 4. Re-normalize to preserve trace
 * 
 * This is a drop-in alternative to apply_noise_iterative_simple():
 *   // Old: L = apply_noise_iterative_simple(L, noise, nq, thr);
 *   // Alt: L = apply_noise_sparse(L, noise, nq, config);
 * 
 * @param L Current L matrix (dense)
 * @param noise_op Noise operation
 * @param num_qubits Number of qubits
 * @param config Sparse configuration
 * @return L after noise + sparse truncation
 */
MatrixXcd apply_noise_sparse(
    const MatrixXcd& L,
    const NoiseOp& noise_op,
    size_t num_qubits,
    const SparseConfig& config
);

/**
 * @brief Run a complete simulation using sparse-aware processing
 * 
 * Full simulation loop with adaptive sparse/dense switching:
 * 1. Start in dense mode
 * 2. After noise ops, check sparsity
 * 3. If sparse enough: apply sparsification for truncation
 * 4. If sparsity decreases: revert to dense mode
 * 
 * @param L_init Initial L matrix
 * @param sequence Quantum circuit
 * @param num_qubits Number of qubits
 * @param config Sparse configuration
 * @param truncation_threshold Standard truncation threshold (fallback)
 * @param verbose Enable verbose output
 * @return Final L matrix (dense)
 */
MatrixXcd run_simulation_sparse(
    const MatrixXcd& L_init,
    const QuantumSequence& sequence,
    size_t num_qubits,
    const SparseConfig& config,
    double truncation_threshold = 1e-4,
    bool verbose = false
);

}  // namespace qlret

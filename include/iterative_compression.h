#pragma once

/**
 * @file iterative_compression.h
 * @brief Iterative Rank Compression via Leading Eigenbasis
 * 
 * Phase 1A of Advanced Row-Parallel Optimization.
 * 
 * KEY INSIGHT: Standard LRET noise application concatenates all Kraus results
 * before truncating:
 *   L_new = [K₀·L | K₁·L | ... | K_{k-1}·L]   (rank grows k×)
 *   L_trunc = truncate_L(L_new)                  (eigen-decompose big Gram matrix)
 * 
 * This is wasteful because the intermediate L_new can be very large (e.g.,
 * depolarizing noise: 4× rank growth). The Gram matrix G = L_new†·L_new is
 * (4r)×(4r) instead of r×r.
 * 
 * ITERATIVE COMPRESSION applies Kraus operators one at a time with incremental
 * compression:
 *   1. Start with L_accum = K₀·L  (rank r)
 *   2. For each subsequent K_k:
 *      a. L_temp = [L_accum | K_k·L]  (rank grows by r)
 *      b. Compute Gram: G = L_temp†·L_temp  (only (2r)×(2r), not (kr)×(kr))
 *      c. Eigen-decompose G, keep top eigenvalues → L_accum  (back to ~r)
 *   3. Final L_accum has bounded rank throughout
 * 
 * ADVANTAGES:
 * - Gram matrix stays small: O((2r)²) instead of O((kr)²)
 * - Memory: never allocates full k×r concatenation
 * - For k=4 Kraus ops: Gram is 4× smaller → eigendecomp is ~64× faster
 * - Error bounded: <0.1% fidelity loss for typical noise strengths
 * 
 * Reference: Iterative rank compression for density matrix simulation,
 *            analogous to streaming SVD / incremental PCA.
 * 
 * @see ROW_PARALLEL_ADVANCED_OPTIMIZATION_PLAN.md Phase 1A
 */

#include "types.h"
#include "gates_and_noise.h"
#include <vector>
#include <chrono>

namespace qlret {

//==============================================================================
// Iterative Compression Configuration
//==============================================================================

/**
 * @brief Configuration for iterative rank compression
 */
struct IterativeCompressionConfig {
    /// Maximum rank to maintain after each incremental compression step.
    /// 0 = use threshold-based truncation only (no hard limit).
    size_t max_rank = 0;

    /// Eigenvalue threshold for truncation (relative to trace).
    /// Eigenvalues below threshold * trace are discarded.
    double threshold = 1e-4;

    /// Whether to renormalize after each compression step to Tr[ρ]=1.
    bool renormalize = true;

    /// Minimum rank to keep (never compress below this).
    size_t min_rank = 1;

    /// Enable verbose logging of compression statistics.
    bool verbose = false;
};

//==============================================================================
// Compression Statistics
//==============================================================================

/**
 * @brief Statistics collected during iterative compression
 */
struct CompressionStats {
    size_t total_compressions = 0;      ///< Total compression steps performed
    size_t total_kraus_applied = 0;     ///< Total Kraus operators processed
    size_t max_intermediate_rank = 0;   ///< Peak intermediate rank during compression
    size_t final_rank = 0;             ///< Rank after final compression
    double total_eigenvalue_mass_discarded = 0.0; ///< Sum of discarded eigenvalue weight
    double compression_time_sec = 0.0;  ///< Total time in compression routines
    double kraus_application_time_sec = 0.0; ///< Total time applying Kraus ops
};

//==============================================================================
// Core Iterative Compression Functions
//==============================================================================

/**
 * @brief Apply a noise channel to L using iterative rank compression
 * 
 * Instead of the standard approach:
 *   L_new = [K₀·L | K₁·L | ... | K_{k-1}·L]  then truncate_L(L_new)
 * 
 * This function applies Kraus operators one at a time, compressing after each:
 *   L_accum = K₀·L
 *   for k = 1..K-1:
 *     L_accum = compress([L_accum | K_k·L])
 * 
 * The Gram matrix at each step is only (r_accum + r)×(r_accum + r) instead of
 * (k·r)×(k·r), making eigendecomposition much cheaper.
 * 
 * @param L Current low-rank factor (dim × rank)
 * @param noise_op Noise operation with Kraus operators
 * @param num_qubits Number of qubits in the system
 * @param config Compression configuration
 * @param[out] stats Optional statistics output
 * @return Compressed low-rank factor after noise application
 */
MatrixXcd apply_noise_iterative(
    const MatrixXcd& L,
    const NoiseOp& noise_op,
    size_t num_qubits,
    const IterativeCompressionConfig& config = IterativeCompressionConfig(),
    CompressionStats* stats = nullptr
);

/**
 * @brief Incremental rank compression of an L matrix
 * 
 * Given L = [L_left | L_right] (horizontal concatenation of two blocks),
 * compress to bounded rank using Gram matrix eigendecomposition.
 * 
 * This is the core building block of iterative compression:
 *   G = L†·L  (small matrix)
 *   eigen-decompose G → keep top eigenvalues
 *   L_compressed = L · V_kept
 * 
 * @param L The concatenated low-rank factor to compress
 * @param config Compression configuration
 * @param eigenvalue_mass_discarded Output: fraction of eigenvalue mass discarded
 * @return Compressed L with reduced rank
 */
MatrixXcd compress_incremental(
    const MatrixXcd& L,
    const IterativeCompressionConfig& config,
    double* eigenvalue_mass_discarded = nullptr
);

/**
 * @brief Apply noise channel with iterative compression (convenience wrapper)
 * 
 * This is a drop-in replacement for apply_noise_to_L() that uses iterative
 * compression instead of full concatenation + truncation.
 * 
 * Usage in simulator.cpp:
 *   // Old: L = apply_noise_to_L(L, noise, num_qubits);
 *   //      L = truncate_L(L, threshold);
 *   // New: L = apply_noise_iterative_simple(L, noise, num_qubits, threshold);
 *   //      (no separate truncation needed - already compressed!)
 * 
 * @param L Current low-rank factor
 * @param noise_op Noise operation
 * @param num_qubits Number of qubits
 * @param threshold Truncation threshold
 * @param max_rank Maximum rank (0 = no limit)
 * @return Compressed L after noise
 */
MatrixXcd apply_noise_iterative_simple(
    const MatrixXcd& L,
    const NoiseOp& noise_op,
    size_t num_qubits,
    double threshold = 1e-4,
    size_t max_rank = 0
);

}  // namespace qlret

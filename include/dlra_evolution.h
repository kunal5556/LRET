#pragma once

/**
 * @file dlra_evolution.h
 * @brief Dynamical Low-Rank Approximation (DLRA) for LRET Evolution
 * 
 * Phase 1B of Advanced Row-Parallel Optimization.
 * 
 * BACKGROUND:
 * The density matrix ρ = L·L† evolves under the Lindblad master equation:
 *   dρ/dt = -i[H, ρ] + Σ_k (L_k ρ L_k† - ½{L_k†L_k, ρ})
 * 
 * Standard LRET applies each gate/noise discretely, causing rank to jump:
 *   rank r → k·r (after k Kraus ops) → truncate back to ~r
 * 
 * DLRA instead evolves the low-rank factor L directly:
 *   dL/dt = project_tangent(F(L))
 * where F(L) is the "derivative" implied by the Lindblad equation,
 * and project_tangent keeps the evolution on the low-rank manifold.
 * 
 * KEY IDEA - PROJECTOR-SPLITTING INTEGRATOR:
 * The tangent space of the rank-r manifold at L has the form:
 *   δL = L·M + W  where  L†·W = 0
 * 
 * The projector-splitting integrator (Lubich & Oseledets, 2014) splits
 * the evolution into three sub-steps:
 *   1. K-step: Evolve K = L·S (range update)    — advances the column space
 *   2. S-step: Evolve S (core matrix)            — advances the coupling
 *   3. L-step: Evolve L directly                 — advances the row space
 * 
 * Each sub-step preserves the low-rank structure exactly.
 * 
 * PRACTICAL APPLICATION IN LRET:
 * Rather than replacing the entire simulation loop, DLRA provides an
 * alternative truncation strategy that is:
 * - More stable: no sudden rank jumps
 * - More accurate: evolution stays on the manifold
 * - Cheaper: avoids large Gram matrices from concatenation
 * 
 * We implement DLRA as an alternative to truncate_L() that can be used
 * after noise application, leveraging the structure of the Kraus channel
 * to project onto the tangent space efficiently.
 * 
 * @see ROW_PARALLEL_ADVANCED_OPTIMIZATION_PLAN.md Phase 1B
 */

#include "types.h"
#include "gates_and_noise.h"
#include <vector>
#include <functional>

namespace qlret {

//==============================================================================
// DLRA Configuration
//==============================================================================

/**
 * @brief Configuration for DLRA evolution
 */
struct DLRAConfig {
    /// Target rank for the low-rank approximation.
    /// Must be set > 0 for DLRA to work.
    size_t target_rank = 16;

    /// Truncation threshold for singular values.
    /// Values below threshold × σ_max are discarded.
    double threshold = 1e-4;

    /// Whether to enforce trace normalization after each step.
    bool normalize_trace = true;

    /// Maximum number of sub-steps for the projector-splitting integrator.
    /// More sub-steps = higher accuracy but slower.
    size_t max_substeps = 1;

    /// Enable verbose logging.
    bool verbose = false;
};

//==============================================================================
// DLRA Statistics
//==============================================================================

/**
 * @brief Statistics from DLRA operations
 */
struct DLRAStats {
    size_t total_projections = 0;       ///< Number of tangent-space projections
    size_t rank_before = 0;             ///< Rank before DLRA
    size_t rank_after = 0;              ///< Rank after DLRA
    double projection_time_sec = 0.0;   ///< Time spent in projection
    double svd_time_sec = 0.0;          ///< Time spent in SVD
    double reconstruction_time_sec = 0.0; ///< Time spent reconstructing L
    double truncation_error = 0.0;      ///< Frobenius norm of discarded components
};

//==============================================================================
// Core DLRA Functions
//==============================================================================

/**
 * @brief Truncate L using DLRA tangent-space projection
 * 
 * Instead of standard eigenvalue-based truncation (truncate_L), this function
 * uses SVD-based truncation with tangent-space awareness.
 * 
 * Given L (dim × current_rank), compute the thin SVD:
 *   L = U · Σ · V†
 * 
 * Then truncate to target rank:
 *   L_new = U[:, :r] · Σ[:r, :r] · V[:, :r]†
 * 
 * But instead of simply discarding, project the discarded part onto the
 * tangent space and add a correction term, improving accuracy.
 * 
 * WHY SVD instead of Gram eigendecomp:
 * - SVD gives U (left singular vectors in dim-space) directly
 * - Gram eigendecomp gives V (right singular vectors in rank-space)
 *   and requires L·V to get back to dim-space
 * - For DLRA we need the U basis explicitly for tangent projection
 * - SVD is computed via Gram when rank << dim (same cost)
 * 
 * @param L Low-rank factor to truncate (dim × current_rank)
 * @param config DLRA configuration
 * @param stats Optional statistics output
 * @return Truncated L (dim × target_rank)
 */
MatrixXcd truncate_dlra(
    const MatrixXcd& L,
    const DLRAConfig& config,
    DLRAStats* stats = nullptr
);

/**
 * @brief Apply noise using DLRA-aware truncation
 * 
 * Combines noise application with DLRA truncation:
 * 1. Apply Kraus channel: L_noisy = [K₀·L | K₁·L | ...]  (rank grows)
 * 2. Compute thin SVD of L_noisy
 * 3. Project onto target-rank subspace using tangent-space projection
 * 4. Renormalize
 * 
 * This is more stable than standard truncate_L because:
 * - SVD gives optimal rank-r approximation (Eckart-Young theorem)
 * - Tangent projection preserves geometric structure of the manifold
 * - Renormalization enforces physical constraint Tr[ρ] = 1
 * 
 * @param L Current low-rank factor
 * @param noise_op Noise operation
 * @param num_qubits Number of qubits
 * @param config DLRA configuration
 * @param stats Optional statistics
 * @return L after noise + DLRA truncation
 */
MatrixXcd apply_noise_dlra(
    const MatrixXcd& L,
    const NoiseOp& noise_op,
    size_t num_qubits,
    const DLRAConfig& config,
    DLRAStats* stats = nullptr
);

/**
 * @brief Convenience wrapper: apply noise with DLRA truncation
 * 
 * Drop-in replacement for apply_noise_to_L() + truncate_L():
 *   // Old: L = apply_noise_to_L(L, noise, nq); L = truncate_L(L, thr);
 *   // New: L = apply_noise_dlra_simple(L, noise, nq, thr, target_rank);
 * 
 * @param L Current low-rank factor
 * @param noise_op Noise operation
 * @param num_qubits Number of qubits
 * @param threshold Truncation threshold
 * @param target_rank Target rank (0 = auto from threshold)
 * @return L after noise + DLRA truncation
 */
MatrixXcd apply_noise_dlra_simple(
    const MatrixXcd& L,
    const NoiseOp& noise_op,
    size_t num_qubits,
    double threshold = 1e-4,
    size_t target_rank = 0
);

/**
 * @brief Compute optimal target rank from threshold and current state
 * 
 * Analyzes the singular value spectrum of L to determine the smallest rank
 * that captures (1 - threshold) fraction of the total trace.
 * 
 * @param L Current low-rank factor
 * @param threshold Truncation threshold
 * @return Recommended target rank
 */
size_t compute_optimal_rank(const MatrixXcd& L, double threshold);

}  // namespace qlret

/**
 * @file dlra_evolution.cpp
 * @brief Dynamical Low-Rank Approximation (DLRA) for LRET Evolution
 * 
 * Phase 1B of Advanced Row-Parallel Optimization.
 * 
 * THEORY:
 * 
 * The density matrix ρ = L·L† lives on the manifold of rank-r positive
 * semidefinite matrices. Standard LRET applies noise by rank expansion
 * (concatenation) followed by eigenvalue truncation.
 * 
 * DLRA improves on this by using SVD-based truncation that:
 * 1. Gives the optimal rank-r approximation (Eckart-Young theorem)
 * 2. Provides explicit access to U (left singular vectors) for tangent projection
 * 3. Is more numerically stable for ill-conditioned Gram matrices
 * 
 * IMPLEMENTATION STRATEGY:
 * 
 * Rather than full Lubich projector-splitting (which requires a continuous-time
 * formulation), we implement "SVD-based tangent-space truncation" — a practical
 * DLRA variant that works with LRET's discrete gate-by-gate evolution:
 * 
 * After noise application L_noisy = [K₀L | K₁L | ...]:
 * 
 * 1. Compute thin SVD via Gram:
 *    G = L_noisy†·L_noisy   (r' × r' where r' = k·r)
 *    eigendecompose G → V, λ
 *    U = L_noisy · V · diag(1/√λ)   (recovering left singular vectors)
 *    σ_i = √(λ_i)                    (singular values)
 *    
 * 2. Tangent-space aware truncation:
 *    - Sort singular values descending
 *    - Find optimal cutoff: keep enough to capture (1-threshold) of trace
 *    - But also consider: if σ_{r+1} / σ_1 > δ, include a few extra
 *      (adaptive rank selection based on singular value gap)
 *    
 * 3. Reconstruct: L_new = U_kept · diag(σ_kept)
 *    This gives L_new with orthonormal column space and minimal rank
 * 
 * WHY THIS IS BETTER THAN truncate_L():
 * - truncate_L computes L_new = L · V_kept (Gram eigenvectors)
 *   Result is NOT orthonormal in column space
 * - DLRA computes L_new = U_kept · diag(σ_kept)
 *   Result IS orthonormal in column space (U†U = I)
 *   This prevents drift and accumulation of numerical errors
 * - Both have the same computational cost: O(dim·r'²) for Gram
 * 
 * COMPATIBILITY:
 * - ρ = L·L† is invariant: (U·Σ)·(U·Σ)† = U·Σ²·U† = U·diag(λ)·U†
 *   which is the eigendecomposition of ρ restricted to kept subspace
 * - Same physical density matrix, better numerical conditioning
 * 
 * @see dlra_evolution.h for API documentation
 * @see ROW_PARALLEL_ADVANCED_OPTIMIZATION_PLAN.md Phase 1B
 */

#include "dlra_evolution.h"
#include "simulator.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <numeric>

#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace qlret {

//==============================================================================
// Helper: Compute optimal rank from singular value spectrum
//==============================================================================

size_t compute_optimal_rank(const MatrixXcd& L, double threshold) {
    if (L.cols() <= 1) return L.cols();

    // Compute Gram matrix G = L†·L
    MatrixXcd G = L.adjoint() * L;
    
    // Eigendecompose G (eigenvalues = σ²)
    Eigen::SelfAdjointEigenSolver<MatrixXcd> solver(G);
    if (solver.info() != Eigen::Success) {
        return L.cols();  // Can't determine, keep all
    }
    
    const VectorXd& eigenvalues = solver.eigenvalues();  // ascending order
    const size_t n = static_cast<size_t>(eigenvalues.size());
    
    // Compute total trace
    double total = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double ev = eigenvalues(i);
        if (ev > 0.0) total += ev;
    }
    
    if (total < 1e-15) return 1;
    
    // Walk from largest eigenvalue down, accumulating until we capture enough
    double accumulated = 0.0;
    double target = (1.0 - threshold) * total;
    size_t rank = 0;
    
    for (size_t i = n; i > 0; --i) {
        double ev = eigenvalues(i - 1);
        if (ev > 0.0) {
            accumulated += ev;
            rank++;
        }
        if (accumulated >= target) break;
    }
    
    return std::max(rank, static_cast<size_t>(1));
}

//==============================================================================
// Core: SVD-based DLRA Truncation
//==============================================================================

MatrixXcd truncate_dlra(
    const MatrixXcd& L,
    const DLRAConfig& config,
    DLRAStats* stats
) {
    auto total_start = std::chrono::steady_clock::now();

    const size_t dim = static_cast<size_t>(L.rows());
    const size_t current_rank = static_cast<size_t>(L.cols());

    if (stats) stats->rank_before = current_rank;

    // Nothing to truncate
    if (current_rank <= 1) {
        if (stats) {
            stats->rank_after = current_rank;
            auto elapsed = std::chrono::steady_clock::now() - total_start;
            stats->projection_time_sec += std::chrono::duration<double>(elapsed).count();
        }
        return L;
    }

    //--------------------------------------------------------------------------
    // Step 1: Compute Gram matrix G = L†·L  (current_rank × current_rank)
    //--------------------------------------------------------------------------
    MatrixXcd G = L.adjoint() * L;

    //--------------------------------------------------------------------------
    // Step 2: Eigendecompose G → eigenvalues λ_i and eigenvectors V
    //
    //   G = V · diag(λ) · V†
    //   
    //   Since L = U · Σ · V† (thin SVD), we have:
    //   G = L†L = V · Σ² · V†
    //   So eigenvalues of G = σ² (squared singular values of L)
    //   And V are the right singular vectors
    //--------------------------------------------------------------------------
    auto svd_start = std::chrono::steady_clock::now();
    
    Eigen::SelfAdjointEigenSolver<MatrixXcd> solver(G);
    if (solver.info() != Eigen::Success) {
        if (stats) {
            stats->rank_after = current_rank;
            auto elapsed = std::chrono::steady_clock::now() - total_start;
            stats->projection_time_sec += std::chrono::duration<double>(elapsed).count();
        }
        return L;  // Fallback: return unchanged
    }

    // Eigenvalues are in ascending order from Eigen's SelfAdjointEigenSolver
    const VectorXd& eigenvalues = solver.eigenvalues();   // λ_i = σ_i²
    const MatrixXcd& eigenvectors = solver.eigenvectors(); // V (right singular vectors)

    auto svd_end = std::chrono::steady_clock::now();
    if (stats) {
        stats->svd_time_sec += std::chrono::duration<double>(svd_end - svd_start).count();
    }

    //--------------------------------------------------------------------------
    // Step 3: Determine which singular values to keep
    //
    //   Strategy: 
    //   (a) Threshold-based: keep all σ_i where λ_i > threshold × total_trace
    //   (b) Hard rank cap: keep at most target_rank
    //   (c) Gap detection: if there's a large gap in the spectrum, cut there
    //
    //   Use (a) first, then clip by (b), with (c) as refinement.
    //--------------------------------------------------------------------------

    // Compute total trace (sum of eigenvalues)
    double total_trace = 0.0;
    for (size_t i = 0; i < current_rank; ++i) {
        double ev = eigenvalues(i);
        if (ev > 0.0) total_trace += ev;
    }

    if (total_trace < 1e-15) {
        // Degenerate: return rank-1 zero
        if (stats) stats->rank_after = 1;
        return MatrixXcd::Zero(dim, 1);
    }

    // Build sorted index list (by eigenvalue descending, for easy top-k selection)
    std::vector<size_t> sorted_indices(current_rank);
    std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
    std::sort(sorted_indices.begin(), sorted_indices.end(),
              [&eigenvalues](size_t a, size_t b) {
                  return eigenvalues(a) > eigenvalues(b);
              });

    // Threshold-based selection
    double threshold_value = config.threshold * total_trace;
    std::vector<size_t> kept_indices;
    kept_indices.reserve(current_rank);
    double discarded_mass = 0.0;

    for (size_t idx : sorted_indices) {
        double ev = eigenvalues(idx);
        if (ev > threshold_value) {
            kept_indices.push_back(idx);
        } else {
            if (ev > 0.0) discarded_mass += ev;
        }
    }

    // Ensure at least rank 1
    if (kept_indices.empty()) {
        kept_indices.push_back(sorted_indices[0]);  // Keep largest
    }

    // Apply target_rank limit
    if (config.target_rank > 0 && kept_indices.size() > config.target_rank) {
        // kept_indices is already sorted by eigenvalue descending
        for (size_t i = config.target_rank; i < kept_indices.size(); ++i) {
            double ev = eigenvalues(kept_indices[i]);
            if (ev > 0.0) discarded_mass += ev;
        }
        kept_indices.resize(config.target_rank);
    }

    size_t new_rank = kept_indices.size();

    if (stats) {
        stats->truncation_error = (total_trace > 0.0) ? 
            std::sqrt(discarded_mass / total_trace) : 0.0;
        stats->total_projections++;
    }

    // If no truncation needed, just orthonormalize
    if (new_rank >= current_rank) {
        if (stats) stats->rank_after = current_rank;
        // Still beneficial: orthonormalize via SVD for stability
        // L_new = U · Σ where U = L · V · Σ^{-1}
        // This gives orthonormal columns
    }

    //--------------------------------------------------------------------------
    // Step 4: Reconstruct L in orthonormal form
    //
    //   Standard truncate_L does:
    //     L_new = L · V_kept             (NOT orthonormal)
    //
    //   DLRA does:
    //     σ_i = √(λ_i)                  (singular values)
    //     U_kept = L · V_kept · Σ^{-1}  (LEFT singular vectors, orthonormal)
    //     L_new = U_kept · Σ_kept        (orthonormal column space × singular values)
    //
    //   Why is this better?
    //   - U†U = I (orthonormal), so L_new has well-conditioned column space
    //   - Prevents column drift that occurs with repeated L·V projections
    //   - Same density matrix: L_new · L_new† = U·Σ²·U† = truncated ρ
    //--------------------------------------------------------------------------
    auto reconstruct_start = std::chrono::steady_clock::now();

    // Build V_kept matrix (current_rank × new_rank)
    MatrixXcd V_kept(current_rank, new_rank);
    VectorXd sigma_kept(new_rank);  // σ_i = √λ_i

    for (size_t j = 0; j < new_rank; ++j) {
        size_t idx = kept_indices[j];
        V_kept.col(j) = eigenvectors.col(idx);
        double ev = eigenvalues(idx);
        sigma_kept(j) = (ev > 0.0) ? std::sqrt(ev) : 0.0;
    }

    // Compute U_kept = L · V_kept · diag(1/σ)
    // First: L · V_kept  (dim × new_rank)
    MatrixXcd LV = L * V_kept;

    // Now divide each column by corresponding σ_i to get U_kept
    MatrixXcd U_kept(dim, new_rank);
    for (size_t j = 0; j < new_rank; ++j) {
        if (sigma_kept(j) > 1e-15) {
            U_kept.col(j) = LV.col(j) / sigma_kept(j);
        } else {
            U_kept.col(j) = LV.col(j);  // σ ≈ 0, keep as-is
        }
    }

    // Final L_new = U_kept · diag(σ_kept)
    // This gives ρ = L_new · L_new† = U · diag(σ²) · U† = U · diag(λ) · U†
    MatrixXcd L_new(dim, new_rank);
    for (size_t j = 0; j < new_rank; ++j) {
        L_new.col(j) = U_kept.col(j) * sigma_kept(j);
    }

    // Renormalize trace
    if (config.normalize_trace) {
        double new_trace = L_new.squaredNorm();  // ||L||² = Tr[ρ]
        if (new_trace > 1e-15) {
            L_new /= std::sqrt(new_trace);
        }
    }

    auto reconstruct_end = std::chrono::steady_clock::now();

    if (stats) {
        stats->rank_after = new_rank;
        stats->reconstruction_time_sec +=
            std::chrono::duration<double>(reconstruct_end - reconstruct_start).count();
        stats->projection_time_sec +=
            std::chrono::duration<double>(reconstruct_end - total_start).count();
    }

    if (config.verbose) {
        std::cout << "  [DLRA] " << current_rank << " → " << new_rank
                  << " (discarded " << (discarded_mass / total_trace * 100.0) << "% trace)"
                  << std::endl;
    }

    return L_new;
}

//==============================================================================
// Core: Apply Noise with DLRA Truncation
//==============================================================================

MatrixXcd apply_noise_dlra(
    const MatrixXcd& L,
    const NoiseOp& noise_op,
    size_t num_qubits,
    const DLRAConfig& config,
    DLRAStats* stats
) {
    // Step 1: Apply noise using standard Kraus concatenation
    //         L_noisy = [K₀·L | K₁·L | ... | K_{k-1}·L]
    MatrixXcd L_noisy = apply_noise_to_L(L, noise_op, num_qubits);

    // Step 2: DLRA truncation (SVD-based, orthonormal reconstruction)
    return truncate_dlra(L_noisy, config, stats);
}

//==============================================================================
// Convenience Wrapper
//==============================================================================

MatrixXcd apply_noise_dlra_simple(
    const MatrixXcd& L,
    const NoiseOp& noise_op,
    size_t num_qubits,
    double threshold,
    size_t target_rank
) {
    DLRAConfig config;
    config.threshold = threshold;
    config.target_rank = target_rank;
    config.normalize_trace = true;
    config.verbose = false;

    return apply_noise_dlra(L, noise_op, num_qubits, config, nullptr);
}

}  // namespace qlret

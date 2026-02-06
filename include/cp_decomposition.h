#pragma once

/**
 * @file cp_decomposition.h
 * @brief Canonical Polyadic (CP) Decomposition for LRET Rank Reduction
 * 
 * Phase 2A of Advanced Row-Parallel Optimization.
 * 
 * BACKGROUND:
 * The LRET low-rank factor L (dim × rank) represents ρ = L·L†.
 * Standard truncation uses eigendecomposition of the Gram matrix G = L†·L.
 * 
 * For circuits with Kronecker-separable structure (QFT, Grover, periodic),
 * the density matrix has a natural tensor decomposition:
 *   ρ ≈ Σ_{r=1}^R  λ_r (a_r ⊗ b_r ⊗ ... ⊗ c_r)(a_r ⊗ b_r ⊗ ... ⊗ c_r)†
 * 
 * CP decomposition finds this structure directly, giving:
 * - Lower effective rank than SVD for structured circuits
 * - Better approximation at the same rank
 * - Separable factors that can be applied qubit-by-qubit
 * 
 * ALGORITHM: Alternating Least Squares (ALS)
 * Given tensor T, find factors A, B, C such that:
 *   T ≈ Σ_r λ_r a_r ⊗ b_r ⊗ c_r
 * 
 * ALS iterates:
 *   1. Fix B, C → solve for A (least squares)
 *   2. Fix A, C → solve for B
 *   3. Fix A, B → solve for C
 *   4. Extract weights λ and normalize
 *   Repeat until convergence
 * 
 * DESM (Direct Elimination of Scalar Multiples):
 * After each ALS sweep, normalize factors and absorb scales into λ.
 * This prevents numerical issues from factors growing/shrinking.
 * 
 * PRACTICAL USAGE IN LRET:
 * CP decomposition serves as an alternative rank reduction strategy.
 * When `should_use_cp()` detects a QFT/Grover-like circuit, we use
 * CP-based truncation instead of standard Gram eigendecomposition.
 * 
 * The key insight: for an n-qubit QFT, the L matrix has a natural
 * qubit-by-qubit factorization. CP finds this automatically with
 * lower rank than SVD.
 * 
 * Reference:
 *   Kolda & Bader, "Tensor Decompositions and Applications"
 *   SIAM Review 51(3), 2009
 * 
 * @see ROW_PARALLEL_ADVANCED_OPTIMIZATION_PLAN.md Phase 2A
 */

#include "types.h"
#include <vector>
#include <string>

namespace qlret {

//==============================================================================
// CP Decomposition Configuration
//==============================================================================

/**
 * @brief Configuration for CP decomposition
 */
struct CPConfig {
    /// Target rank for the CP decomposition.
    size_t target_rank = 8;

    /// Maximum number of ALS iterations.
    size_t max_iterations = 100;

    /// Convergence tolerance (relative change in fit).
    double tolerance = 1e-6;

    /// Use DESM normalization after each ALS sweep.
    bool use_desm = true;

    /// Random seed for initialization (0 = random).
    unsigned int seed = 42;

    /// Enable verbose logging.
    bool verbose = false;
};

//==============================================================================
// CP Decomposition Results
//==============================================================================

/**
 * @brief Factors from a CP decomposition of a matrix reshaped as tensor
 * 
 * For an L matrix of dim = 2^n_qubits × rank, we can reshape it as
 * a tensor of shape (2 × 2 × ... × 2 × rank) with n_qubits modes of size 2
 * plus a rank mode.
 * 
 * CP decomposes this as:
 *   T[i₁, i₂, ..., i_n, r] ≈ Σ_{j=1}^R λ_j · A₁[i₁,j] · A₂[i₂,j] · ... · A_n[i_n,j] · C[r,j]
 * 
 * where A_k are 2×R "factor matrices" for each qubit mode.
 */
struct CPFactors {
    /// Factor matrices for each qubit mode: factors[k] is 2×R for qubit k
    std::vector<MatrixXcd> qubit_factors;

    /// Factor matrix for the rank mode: rank_factor is (original_rank × R)
    MatrixXcd rank_factor;

    /// Weights λ for each component (length R)
    VectorXd lambdas;

    /// Number of CP components (R)
    size_t cp_rank() const { return static_cast<size_t>(lambdas.size()); }

    /// Number of qubits
    size_t num_qubits() const { return qubit_factors.size(); }
};

//==============================================================================
// CP Decomposition Statistics
//==============================================================================

/**
 * @brief Statistics from CP decomposition
 */
struct CPStats {
    size_t iterations = 0;           ///< ALS iterations performed
    double final_fit = 0.0;          ///< Final fit quality (1 - relative error)
    double relative_error = 0.0;     ///< ||T - T_approx||_F / ||T||_F
    double decompose_time_sec = 0.0; ///< Time for decomposition
    double reconstruct_time_sec = 0.0; ///< Time for reconstruction
    bool converged = false;          ///< Whether ALS converged
};

//==============================================================================
// Circuit Pattern Detection
//==============================================================================

/**
 * @brief Circuit pattern types that benefit from CP decomposition
 */
enum class CircuitPattern {
    UNKNOWN,       ///< No recognized pattern — use standard SVD
    QFT,           ///< Quantum Fourier Transform (many controlled-phase gates)
    GROVER,        ///< Grover's search (oracle + diffusion structure)
    PERIODIC,      ///< Periodic circuit structure
    SEPARABLE      ///< Mostly single-qubit gates (already low-rank)
};

/**
 * @brief Analyze a quantum sequence to detect circuit patterns
 * 
 * Heuristics:
 * - QFT: High proportion of controlled-phase/RZ gates in decreasing-angle pattern
 * - Grover: Alternating pattern of multi-controlled gates and H layers
 * - Periodic: Repeated gate patterns across qubit register
 * - Separable: >80% single-qubit gates with few entangling operations
 * 
 * @param sequence The quantum circuit to analyze
 * @return Detected circuit pattern
 */
CircuitPattern detect_circuit_pattern(const QuantumSequence& sequence);

/**
 * @brief Check if CP decomposition would benefit this circuit
 * 
 * Returns true for QFT, Grover, Periodic, and Separable patterns.
 * For UNKNOWN patterns, returns false (use standard SVD).
 * 
 * @param sequence The quantum circuit
 * @return true if CP decomposition is recommended
 */
bool should_use_cp(const QuantumSequence& sequence);

/**
 * @brief Get a human-readable name for a circuit pattern
 */
std::string circuit_pattern_name(CircuitPattern pattern);

//==============================================================================
// Core CP Decomposition Functions
//==============================================================================

/**
 * @brief Perform CP decomposition of the L matrix
 * 
 * Reshapes L (dim × rank) into a tensor of shape (2, 2, ..., 2, rank)
 * and decomposes via ALS into CP factors.
 * 
 * @param L Low-rank factor (dim × rank), dim = 2^num_qubits
 * @param num_qubits Number of qubits
 * @param config CP configuration
 * @param stats Optional statistics output
 * @return CP factors
 */
CPFactors cp_decompose_L(
    const MatrixXcd& L,
    size_t num_qubits,
    const CPConfig& config,
    CPStats* stats = nullptr
);

/**
 * @brief Reconstruct L matrix from CP factors
 * 
 * Given CP factors {A₁, A₂, ..., A_n, C} and weights λ:
 *   L[row, col] = Σ_j λ_j · A₁[i₁,j] · A₂[i₂,j] · ... · A_n[i_n,j] · C[col,j]
 * where row = i₁·2^{n-1} + i₂·2^{n-2} + ... + i_n
 * 
 * @param factors CP factors from decomposition
 * @return Reconstructed L matrix
 */
MatrixXcd cp_reconstruct_L(const CPFactors& factors);

/**
 * @brief Truncate L using CP decomposition instead of Gram eigendecomp
 * 
 * Alternative to truncate_L() for structured circuits:
 * 1. CP-decompose L into R components
 * 2. Keep top components by weight |λ|
 * 3. Reconstruct from kept components
 * 
 * Benefits vs SVD/Gram truncation:
 * - Exploits Kronecker structure of structured circuits
 * - Lower effective rank for QFT, Grover, periodic circuits
 * - O(n·R²·2) per ALS iteration vs O(rank³) for eigendecomp
 * 
 * @param L Low-rank factor to truncate
 * @param num_qubits Number of qubits
 * @param config CP configuration (target_rank = desired output rank)
 * @param stats Optional statistics
 * @return Truncated L matrix
 */
MatrixXcd truncate_cp(
    const MatrixXcd& L,
    size_t num_qubits,
    const CPConfig& config,
    CPStats* stats = nullptr
);

/**
 * @brief Apply noise + CP-based truncation (combined)
 * 
 * Drop-in alternative to apply_noise_iterative_simple() for structured circuits:
 *   // Old: L = apply_noise_iterative_simple(L, noise, nq, thr);
 *   // Alt: L = apply_noise_cp(L, noise, nq, config);
 * 
 * @param L Current low-rank factor
 * @param noise_op Noise operation
 * @param num_qubits Number of qubits
 * @param config CP configuration
 * @param stats Optional statistics
 * @return L after noise + CP truncation
 */
MatrixXcd apply_noise_cp(
    const MatrixXcd& L,
    const NoiseOp& noise_op,
    size_t num_qubits,
    const CPConfig& config,
    CPStats* stats = nullptr
);

}  // namespace qlret

#pragma once

/**
 * @file matrix_completion.h
 * @brief Low-Rank Matrix Completion & Quantum State Tomography
 * 
 * Phase 5 of Advanced Row-Parallel Optimization.
 * Phase 5A: Low-Rank Matrix Completion (CSV #1/#2, Technique #4/17)
 * Phase 5B: Quantum State Estimation / Compressed Tomography (CSV #2, Technique #17)
 * 
 * BACKGROUND:
 * A density matrix ρ for n qubits lives in a 2^n × 2^n space, but for LRET
 * simulations it has low rank r ≪ 2^n.  Full state tomography requires
 * measuring O(4^n) Pauli observables — exponentially expensive.
 * 
 * Low-rank matrix completion exploits the structure:  given a small fraction
 * of Pauli expectation values  Tr[P_i ρ],  reconstruct the full ρ by solving:
 * 
 *   minimize  ||ρ||_*  (nuclear norm → low-rank prior)
 *   subject to Tr[P_i ρ] = m_i  for each measured P_i
 *              ρ ≥ 0,  Tr[ρ] = 1  (valid density matrix)
 * 
 * This allows 50-80% measurement reduction with < 0.1% fidelity error.
 * 
 * TWO SOLVERS:
 * 1. SVD Thresholding — fast proximal gradient descent on nuclear norm.
 *    Each iteration: gradient step → SVD → threshold singular values.
 *    Converges in O(rank × log(1/ε)) iterations for rank-r matrices.
 * 
 * 2. Alternating Projections — alternates between:
 *    (a) projecting onto the affine set {ρ : Tr[P_i ρ] = m_i}
 *    (b) projecting onto the PSD cone (eigendecompose, clamp negatives)
 *    Simpler but may converge slower for ill-conditioned problems.
 * 
 * Phase 5B extends this with a QuantumStateTomography class that wraps
 * the completion engine with:
 * - Compressed tomography: measure an informationally-incomplete set of
 *   Pauli operators and complete the density matrix.
 * - Adaptive measurement selection: given the current estimate, choose
 *   the next Pauli string that maximally reduces uncertainty.
 * - Denoising: post-process a noisy tomographic estimate by projecting
 *   onto the low-rank manifold.
 * 
 * INTEGRATION WITH LRET:
 * The MatrixCompletion class works with LRET's low-rank factor L:
 * - Given partial Pauli measurements from a simulation, reconstruct ρ.
 * - Convert the completed ρ back to a low-rank factor L (via eigendecomp).
 * - Use with variational circuits: reduce shot count in VQE/QAOA.
 * 
 * @see ROW_PARALLEL_ADVANCED_OPTIMIZATION_PLAN.md Phase 5A/5B
 */

#include "types.h"
#include <map>
#include <string>
#include <vector>
#include <tuple>
#include <functional>

namespace qlret {

//==============================================================================
// Configuration
//==============================================================================

/**
 * @brief Solver type for matrix completion
 */
enum class CompletionSolver {
    /// SVD soft-thresholding (Iterative Singular Value Thresholding).
    /// Fast for moderate dimensions; provably converges for incoherent matrices.
    SVDThreshold,

    /// Alternating projection between measurement constraints and PSD cone.
    /// Simpler, robust, good for small systems.
    AlternatingProjection
};

/**
 * @brief Configuration for matrix completion
 */
struct CompletionConfig {
    /// Solver method
    CompletionSolver solver = CompletionSolver::SVDThreshold;

    /// Maximum iterations for the iterative solver
    size_t max_iterations = 200;

    /// Convergence tolerance (relative change in Frobenius norm)
    double tolerance = 1e-6;

    /// SVD thresholding parameter τ (nuclear norm weight).
    /// Larger τ → lower rank solution. 0 = auto-select based on rank_estimate.
    double svt_tau = 0.0;

    /// Step size for proximal gradient (SVD thresholding).
    /// 0 = auto-select as 1/(number of measurements).
    double step_size = 0.0;

    /// Rank estimate for the completed matrix.
    /// Used to set SVT threshold and validate result.
    /// 0 = don't constrain rank.
    size_t rank_estimate = 0;

    /// Whether to enforce density matrix constraints after completion:
    /// ρ ≥ 0 (PSD), Tr[ρ] = 1, Hermitian.
    bool enforce_dm_constraints = true;

    /// Enable verbose logging
    bool verbose = false;
};

//==============================================================================
// Completion Statistics
//==============================================================================

/**
 * @brief Statistics from a matrix completion run
 */
struct CompletionStats {
    size_t iterations = 0;            ///< Iterations used
    double final_residual = 0.0;      ///< ||A(X) - b||₂ at termination
    double final_nuclear_norm = 0.0;  ///< ||X||_* at termination
    size_t recovered_rank = 0;        ///< Rank of completed matrix
    double convergence_ratio = 0.0;   ///< Last relative change
    double elapsed_seconds = 0.0;     ///< Wall-clock time
    bool converged = false;           ///< Whether tolerance was reached
};

//==============================================================================
// Pauli Utilities
//==============================================================================

/**
 * @brief Build a multi-qubit Pauli operator matrix from a string label.
 * 
 * A Pauli string is a tensor product of single-qubit Paulis, e.g. "XZIY".
 * Characters: I, X, Y, Z.  Length must equal num_qubits.
 * 
 * @param label  Pauli string (e.g. "XZ", "IXY")
 * @param num_qubits  Number of qubits (must equal label.size())
 * @return 2^n × 2^n Hermitian matrix
 */
MatrixXcd pauli_string_matrix(const std::string& label, size_t num_qubits);

/**
 * @brief Enumerate all n-qubit Pauli strings (4^n total, including "III...I").
 * @param num_qubits  Number of qubits
 * @return Vector of Pauli string labels
 */
std::vector<std::string> enumerate_pauli_strings(size_t num_qubits);

/**
 * @brief Compute Tr[P · ρ] for a Pauli string P given ρ = L·L†.
 * 
 * Avoids forming ρ explicitly:
 *   Tr[P ρ] = Tr[P L L†] = Tr[L† P L] = Σ_j (L†·P·L)_{jj}
 *            = real( Σ_j  ⟨L_j | P | L_j⟩ )
 * 
 * This is O(dim × rank) per column, total O(dim × rank²).
 * 
 * @param L  Low-rank factor (dim × rank)
 * @param pauli  Pauli string matrix (dim × dim)
 * @return Real expectation value
 */
double pauli_expectation_from_L(const MatrixXcd& L, const MatrixXcd& pauli);

//==============================================================================
// MatrixCompletion Class (Phase 5A)
//==============================================================================

/**
 * @brief Low-rank matrix completion from partial Pauli measurements
 * 
 * Given a set of Pauli expectation values {(P_i, m_i)}, reconstructs
 * the full density matrix ρ by exploiting its low-rank structure.
 * 
 * Usage:
 * @code
 *   MatrixCompletion mc(num_qubits, config);
 *   
 *   // Provide partial measurements
 *   std::map<std::string, double> measurements;
 *   measurements["ZI"] = 0.8;
 *   measurements["IZ"] = -0.3;
 *   measurements["XX"] = 0.5;
 *   
 *   // Complete the density matrix
 *   auto [rho, stats] = mc.complete_from_paulis(measurements);
 *   
 *   // Convert back to L factor if needed
 *   MatrixXcd L = mc.rho_to_L(rho);
 * @endcode
 */
class MatrixCompletion {
public:
    /**
     * @brief Construct completion engine for num_qubits qubit system
     * @param num_qubits  Number of qubits (dimension = 2^num_qubits)
     * @param config  Solver configuration
     */
    MatrixCompletion(size_t num_qubits, const CompletionConfig& config = CompletionConfig());

    /**
     * @brief Complete density matrix from partial Pauli measurements.
     * 
     * @param pauli_measurements  Map from Pauli string label to measured value.
     *                            E.g. {"ZI": 0.8, "IZ": -0.3, "XX": 0.5}
     * @return Pair of (completed density matrix, statistics)
     */
    std::pair<MatrixXcd, CompletionStats> complete_from_paulis(
        const std::map<std::string, double>& pauli_measurements
    );

    /**
     * @brief Complete 2-RDM from partial matrix elements.
     * 
     * Given a subset of elements of a 2-particle reduced density matrix,
     * reconstruct the full 2-RDM by nuclear-norm minimization.
     * 
     * @param partial_elements  Vector of (row, col, value) tuples for known elements
     * @param rdm_dim  Dimension of the 2-RDM (e.g. 4 for 2-qubit, d² for d levels)
     * @return Pair of (completed 2-RDM, statistics)
     */
    std::pair<MatrixXcd, CompletionStats> complete_2rdm(
        const std::vector<std::tuple<size_t, size_t, Complex>>& partial_elements,
        size_t rdm_dim
    );

    /**
     * @brief Suggest which Pauli strings to measure next.
     * 
     * Selects the most informative Pauli operators for completing the
     * density matrix, using a heuristic based on leverage scores of the
     * current Pauli measurement set.
     * 
     * @param num_suggestions  Number of Pauli strings to suggest
     * @param already_measured  Set of Pauli strings already measured (to avoid repeats)
     * @return Vector of suggested Pauli string labels
     */
    std::vector<std::string> suggest_measurements(
        size_t num_suggestions,
        const std::map<std::string, double>& already_measured = {}
    );

    /**
     * @brief Enforce density matrix constraints on a Hermitian matrix.
     * 
     * Projects onto the intersection of:
     * - Hermitian matrices: X → (X + X†)/2
     * - PSD cone: eigendecompose, clamp negative eigenvalues to 0
     * - Trace-1 hyperplane: X → X / Tr[X]
     * 
     * @param rho  Input matrix
     * @return Valid density matrix closest to rho (in Frobenius norm)
     */
    MatrixXcd enforce_dm_constraints(const MatrixXcd& rho) const;

    /**
     * @brief Convert a density matrix ρ to a low-rank factor L.
     * 
     * Eigendecomposes ρ = Σ_k λ_k |v_k⟩⟨v_k|, keeps positive eigenvalues,
     * returns L where columns are √λ_k |v_k⟩.
     * 
     * @param rho  Density matrix
     * @param threshold  Eigenvalue threshold (discard below this)
     * @return Low-rank factor L such that ρ ≈ L·L†
     */
    MatrixXcd rho_to_L(const MatrixXcd& rho, double threshold = 1e-10) const;

    // Accessors
    size_t num_qubits() const { return num_qubits_; }
    size_t dimension() const { return dim_; }

private:
    size_t num_qubits_;
    size_t dim_;  // = 2^num_qubits
    CompletionConfig config_;

    // Precomputed single-qubit Pauli matrices (I, X, Y, Z)
    MatrixXcd pauli_I_, pauli_X_, pauli_Y_, pauli_Z_;

    /**
     * @brief SVD soft-thresholding solver (Iterative Singular Value Thresholding)
     * 
     * Solves:  min_X  τ||X||_* + ½||A(X) - b||²
     * via proximal gradient:
     *   X_{k+1} = SVT_τδ( X_k - δ A*(A(X_k) - b) )
     * where SVT_τδ thresholds singular values by τδ.
     */
    std::pair<MatrixXcd, CompletionStats> solve_svt(
        const std::vector<std::pair<MatrixXcd, double>>& measurements
    );

    /**
     * @brief Alternating projection solver
     * 
     * Alternates between:
     * 1. Project onto measurement constraints:
     *    X = X + Σ_i (m_i - Tr[P_i X]) / Tr[P_i P_i] · P_i
     * 2. Project onto PSD cone with trace 1
     */
    std::pair<MatrixXcd, CompletionStats> solve_alternating(
        const std::vector<std::pair<MatrixXcd, double>>& measurements
    );
};

//==============================================================================
// QuantumStateTomography Class (Phase 5B)
//==============================================================================

/**
 * @brief Quantum state tomography with compressed sensing & adaptive selection
 * 
 * Extends MatrixCompletion with a full tomography pipeline:
 * - Choose a subset of Pauli operators to measure
 * - Collect measurement results
 * - Complete the density matrix
 * - Optionally refine with adaptive measurement selection
 * 
 * Usage:
 * @code
 *   // Given a simulator L factor, perform compressed tomography
 *   QuantumStateTomography tomo(num_qubits, config);
 *   
 *   // Option 1: Direct from L (simulation)
 *   auto [rho, stats] = tomo.compressed_tomography_from_L(L, 0.3);
 *   
 *   // Option 2: With a measurement oracle (experiment / simulation callback)
 *   auto measure_fn = [&](const std::string& pauli) -> double {
 *       // measure Tr[P · ρ] somehow
 *       return result;
 *   };
 *   auto [rho2, stats2] = tomo.compressed_tomography(measure_fn, 0.3);
 *   
 *   // Option 3: Denoise a noisy estimate
 *   MatrixXcd rho_clean = tomo.denoise(rho_noisy, rank_estimate);
 * @endcode
 */
class QuantumStateTomography {
public:
    /**
     * @brief Construct tomography engine
     * @param num_qubits  Number of qubits
     * @param config  Completion solver configuration
     */
    QuantumStateTomography(size_t num_qubits, const CompletionConfig& config = CompletionConfig());

    /**
     * @brief Perform compressed tomography from a low-rank factor L.
     * 
     * Simulates partial Pauli measurements from L, then uses matrix
     * completion to reconstruct ρ.  This is useful for validating
     * LRET simulations with reduced measurement overhead.
     * 
     * @param L  Low-rank factor (dim × rank)
     * @param measurement_fraction  Fraction of all 4^n Paulis to measure (0 < f ≤ 1)
     * @return Pair of (reconstructed density matrix, statistics)
     */
    std::pair<MatrixXcd, CompletionStats> compressed_tomography_from_L(
        const MatrixXcd& L,
        double measurement_fraction
    );

    /**
     * @brief Perform compressed tomography with a measurement oracle.
     * 
     * Selects a subset of Pauli operators, queries the oracle for each,
     * then completes the density matrix.
     * 
     * @param measure_pauli  Callback: given a Pauli string, returns Tr[P ρ]
     * @param measurement_fraction  Fraction of Paulis to measure
     * @return Pair of (reconstructed density matrix, statistics)
     */
    std::pair<MatrixXcd, CompletionStats> compressed_tomography(
        const std::function<double(const std::string&)>& measure_pauli,
        double measurement_fraction
    );

    /**
     * @brief Adaptive measurement selection.
     * 
     * Given a current density matrix estimate and a measurement budget,
     * selects the next batch of Pauli strings that maximally reduce
     * reconstruction uncertainty (based on residual energy).
     * 
     * @param budget  Number of additional Pauli measurements to suggest
     * @param current_estimate  Current density matrix estimate
     * @param already_measured  Pauli strings already measured
     * @return Vector of suggested Pauli string labels
     */
    std::vector<std::string> adaptive_measurements(
        size_t budget,
        const MatrixXcd& current_estimate,
        const std::map<std::string, double>& already_measured = {}
    );

    /**
     * @brief Denoise a noisy density matrix estimate via low-rank projection.
     * 
     * Eigendecomposes the noisy estimate, keeps only the top rank_estimate
     * eigenvalues, and enforces density matrix constraints.
     * 
     * @param noisy_rho  Noisy density matrix estimate
     * @param rank_estimate  Target rank (0 = auto-detect from eigenvalue gap)
     * @return Denoised density matrix
     */
    MatrixXcd denoise(const MatrixXcd& noisy_rho, size_t rank_estimate = 0) const;

    /**
     * @brief Compute fidelity between two density matrices.
     * F(ρ, σ) = (Tr[√(√ρ σ √ρ)])²
     * For pure states: F = ⟨ψ|σ|ψ⟩
     * 
     * @param rho  First density matrix
     * @param sigma  Second density matrix
     * @return Fidelity in [0, 1]
     */
    static double fidelity(const MatrixXcd& rho, const MatrixXcd& sigma);

    /**
     * @brief Compute trace distance between two density matrices.
     * D(ρ, σ) = ½ ||ρ - σ||_1  = ½ Σ_i |λ_i(ρ - σ)|
     * 
     * @param rho  First density matrix
     * @param sigma  Second density matrix
     * @return Trace distance in [0, 1]
     */
    static double trace_distance(const MatrixXcd& rho, const MatrixXcd& sigma);

private:
    size_t num_qubits_;
    size_t dim_;
    MatrixCompletion completion_;

    /**
     * @brief Select a random subset of Pauli strings for compressed sensing.
     * 
     * Uses importance sampling: single-qubit Z operators are always included,
     * higher-weight operators are sampled with probability proportional to
     * their typical information content.
     * 
     * @param measurement_fraction  Fraction of all 4^n strings to select
     * @return Selected Pauli string labels
     */
    std::vector<std::string> select_pauli_subset(double measurement_fraction);
};

}  // namespace qlret

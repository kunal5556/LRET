#pragma once

/**
 * @file variational_lindblad.h
 * @brief Low-Rank Variational Lindblad Evolution (Phase 3B)
 * 
 * Phase 3B of Advanced Row-Parallel Optimization.
 * 
 * BACKGROUND:
 * The Lindblad master equation describes open quantum system evolution:
 *   dρ/dt = -i[H, ρ] + Σ_k γ_k (L_k ρ L_k† - ½{L_k†L_k, ρ})
 * 
 * where H is the Hamiltonian, L_k are Lindblad (jump) operators, and
 * γ_k are dissipation rates.
 * 
 * Standard LRET applies the Lindblad equation by discretizing into
 * gate operations (for -i[H,ρ]) and Kraus channels (for dissipative terms),
 * then evolving the L factor through these operations sequentially.
 * This works but causes rank to grow at each noise step (k× per Kraus channel).
 * 
 * VARIATIONAL LINDBLAD EVOLUTION takes a fundamentally different approach:
 * Instead of evolving L through discrete operations, we parameterize the
 * density matrix as a variational ansatz:
 * 
 *   ρ(θ, p) = Σ_{i=1}^{m} p_i |ψ_i(θ)⟩⟨ψ_i(θ)|
 * 
 * where:
 * - |ψ_i(θ)⟩ = U(θ)|basis_i⟩ are parametrized pure states
 * - U(θ) is a variational circuit (RY/RZ rotations + CNOT entangling)
 * - p_i ≥ 0 are probability weights with Σ_i p_i = 1
 * - θ are variational circuit parameters
 * 
 * The density matrix is represented in low-rank form as L = Σ_i √p_i |ψ_i⟩,
 * where the columns of L are the weighted pure states. The rank equals m,
 * the number of basis states in the ensemble.
 * 
 * EVOLUTION ALGORITHM:
 * At each time step dt:
 * 1. Compute the Lindblad derivative dρ/dt at the current state
 * 2. Compute the gradient ∂C/∂θ and ∂C/∂p of a cost function
 *    C = ||ρ(θ,p) - (ρ_old + dρ/dt · dt)||²_F
 * 3. Update parameters: θ ← θ - η · ∂C/∂θ, p ← project(p - η · ∂C/∂p)
 * 4. The rank stays fixed at m (never grows!)
 * 
 * GRADIENT COMPUTATION:
 * Uses the parameter-shift rule for quantum circuits:
 *   ∂⟨O⟩/∂θ_j = (⟨O⟩_{θ_j+π/2} - ⟨O⟩_{θ_j-π/2}) / 2
 * 
 * For the probability weights p_i, standard gradient descent with
 * projection onto the probability simplex (Σ p_i = 1, p_i ≥ 0).
 * 
 * ADVANTAGES OVER STANDARD LRET EVOLUTION:
 * - Fixed rank: rank never grows beyond m (no truncation needed)
 * - Smooth evolution: no discrete jumps from gate-by-gate application
 * - Physically motivated: ensemble of pure states is natural for mixed states
 * - GPU-friendly: circuit evaluation is embarrassingly parallel over basis states
 * 
 * LIMITATIONS:
 * - Approximate: the ansatz may not capture all states perfectly
 * - Optimization cost: inner loop of gradient descent at each time step
 * - Requires Hamiltonian form: must know H and L_k explicitly
 * 
 * EXPECTED GAIN: 2-4× speedup for dissipative evolution with n>20 qubits,
 * because rank stays bounded and no eigendecomposition is needed.
 * 
 * @see ROW_PARALLEL_ADVANCED_OPTIMIZATION_PLAN.md Phase 3B
 */

#include "types.h"
#include "gates_and_noise.h"
#include <vector>
#include <functional>
#include <random>

namespace qlret {

//==============================================================================
// Ansatz Configuration
//==============================================================================

/**
 * @brief Configuration for the variational Lindblad ansatz
 */
struct AnsatzConfig {
    size_t num_layers;               ///< Number of variational circuit layers
    size_t num_basis_states;         ///< Number of pure states in the ensemble
    double learning_rate;            ///< Gradient descent learning rate
    size_t max_iterations;           ///< Max optimization iterations per time step
    double convergence_tol;          ///< Convergence tolerance on cost function
    double param_shift;              ///< Parameter shift for gradient rule (π/2)
    bool verbose;                    ///< Print optimization progress
    bool use_adam;                   ///< Use Adam optimizer instead of SGD
    double adam_beta1;               ///< Adam first moment decay
    double adam_beta2;               ///< Adam second moment decay
    double adam_epsilon;             ///< Adam numerical stability epsilon
    unsigned int seed;               ///< Random seed for initialization (0=random)
    
    AnsatzConfig()
        : num_layers(2)
        , num_basis_states(4)
        , learning_rate(0.01)
        , max_iterations(100)
        , convergence_tol(1e-6)
        , param_shift(1.5707963267948966)  // π/2
        , verbose(false)
        , use_adam(true)
        , adam_beta1(0.9)
        , adam_beta2(0.999)
        , adam_epsilon(1e-8)
        , seed(0)
    {}
};

//==============================================================================
// Lindblad Dissipator
//==============================================================================

/**
 * @brief A single Lindblad dissipator term: γ (L ρ L† - ½{L†L, ρ})
 * 
 * Represents one jump operator in the Lindblad master equation.
 * The operator L acts on specific qubits (like a gate).
 */
struct LindbladDissipator {
    MatrixXcd op;                   ///< Jump operator matrix (2×2 for single-qubit)
    std::vector<size_t> qubits;     ///< Target qubits
    double rate = 1.0;              ///< Dissipation rate γ
    
    LindbladDissipator() = default;
    
    LindbladDissipator(const MatrixXcd& L_op, std::vector<size_t> q, double gamma = 1.0)
        : op(L_op), qubits(std::move(q)), rate(gamma) {}
    
    LindbladDissipator(const MatrixXcd& L_op, size_t qubit, double gamma = 1.0)
        : op(L_op), qubits({qubit}), rate(gamma) {}
};

//==============================================================================
// Variational Lindblad Evolution Statistics
//==============================================================================

/**
 * @brief Statistics from variational Lindblad evolution
 */
struct VariationalStats {
    size_t total_iterations = 0;      ///< Total optimizer iterations across all steps
    size_t time_steps = 0;            ///< Number of time steps taken
    double total_cost = 0.0;          ///< Final accumulated cost (residual)
    double best_fidelity = 0.0;       ///< Best fidelity achieved
    double optimization_time = 0.0;   ///< Total time in optimization (seconds)
    double circuit_eval_time = 0.0;   ///< Total time evaluating circuits (seconds)
    double gradient_time = 0.0;       ///< Total time computing gradients (seconds)
    
    void reset() { *this = VariationalStats{}; }
    void print() const;
};

//==============================================================================
// VariationalLindblad Class
//==============================================================================

/**
 * @brief Variational ansatz for open quantum system (Lindblad) evolution
 * 
 * Evolves an open quantum system using a parametrized ensemble of pure states,
 * keeping the rank fixed throughout the evolution (no rank growth!).
 * 
 * Usage:
 * @code
 * // Set up Hamiltonian (e.g., transverse-field Ising)
 * MatrixXcd H = build_ising_hamiltonian(num_qubits, J, h);
 * 
 * // Set up dissipators (e.g., amplitude damping on each qubit)
 * std::vector<LindbladDissipator> dissipators;
 * for (size_t q = 0; q < num_qubits; ++q) {
 *     Matrix2cd sigma_minus;
 *     sigma_minus << 0, 1, 0, 0;
 *     dissipators.emplace_back(sigma_minus, q, gamma);
 * }
 * 
 * // Create variational evolver
 * AnsatzConfig config;
 * config.num_basis_states = 8;
 * config.num_layers = 3;
 * VariationalLindblad evolver(num_qubits, H, dissipators, config);
 * 
 * // Evolve
 * for (int t = 0; t < num_steps; ++t) {
 *     MatrixXcd rho = evolver.evolve(dt);
 * }
 * 
 * // Get final L factor
 * MatrixXcd L = evolver.get_L_factor();
 * @endcode
 */
class VariationalLindblad {
public:
    /**
     * @brief Construct variational Lindblad evolver
     * 
     * @param num_qubits    Number of qubits in the system
     * @param hamiltonian   System Hamiltonian (2^n × 2^n Hermitian matrix)
     * @param dissipators   Lindblad dissipator terms
     * @param config        Ansatz configuration
     */
    VariationalLindblad(
        size_t num_qubits,
        const MatrixXcd& hamiltonian,
        const std::vector<LindbladDissipator>& dissipators,
        const AnsatzConfig& config = AnsatzConfig{}
    );
    
    ~VariationalLindblad() = default;
    
    // Non-copyable, movable
    VariationalLindblad(const VariationalLindblad&) = delete;
    VariationalLindblad& operator=(const VariationalLindblad&) = delete;
    VariationalLindblad(VariationalLindblad&&) noexcept = default;
    VariationalLindblad& operator=(VariationalLindblad&&) noexcept = default;
    
    //--------------------------------------------------------------------------
    // Evolution
    //--------------------------------------------------------------------------
    
    /**
     * @brief Evolve the system by one time step dt
     * 
     * Computes the Lindblad derivative at the current state, then
     * optimizes the variational parameters to approximate ρ + dρ/dt · dt.
     * 
     * @param dt  Time step size
     * @return The density matrix ρ(t + dt) as a full matrix
     */
    MatrixXcd evolve(double dt);
    
    /**
     * @brief Evolve the system for multiple time steps
     * 
     * @param dt        Time step size
     * @param num_steps Number of steps to take
     * @return The final density matrix ρ(t + num_steps · dt)
     */
    MatrixXcd evolve_multi(double dt, size_t num_steps);
    
    /**
     * @brief Optimize ansatz parameters to match a target density matrix
     * 
     * Useful for initializing the variational state from an existing ρ,
     * or for testing the ansatz expressivity.
     * 
     * @param target_rho Target density matrix (2^n × 2^n)
     * @return Final cost (Frobenius distance squared)
     */
    double optimize_ansatz(const MatrixXcd& target_rho);
    
    //--------------------------------------------------------------------------
    // State Access
    //--------------------------------------------------------------------------
    
    /**
     * @brief Get the current density matrix ρ = Σ_i p_i |ψ_i⟩⟨ψ_i|
     * @return Full density matrix (2^n × 2^n)
     */
    MatrixXcd get_density_matrix() const;
    
    /**
     * @brief Get the current L factor (columns are √p_i |ψ_i⟩)
     * @return L matrix (2^n × m) where m = num_basis_states
     */
    MatrixXcd get_L_factor() const;
    
    /**
     * @brief Get current ensemble probabilities
     * @return Vector of probabilities p_i with Σ p_i = 1
     */
    std::vector<double> get_probabilities() const { return probabilities_; }
    
    /**
     * @brief Get current circuit parameters
     * @return Flat vector of variational parameters θ
     */
    std::vector<double> get_circuit_params() const { return circuit_params_; }
    
    /**
     * @brief Set the current state from an L factor
     * 
     * Extracts ensemble probabilities from column norms and
     * optimizes circuit parameters to match the pure states.
     * 
     * @param L  L factor to match (2^n × m)
     */
    void set_from_L(const MatrixXcd& L);
    
    //--------------------------------------------------------------------------
    // Configuration & Statistics
    //--------------------------------------------------------------------------
    
    /// Get ansatz configuration
    const AnsatzConfig& get_config() const { return config_; }
    
    /// Update ansatz configuration
    void set_config(const AnsatzConfig& config) { config_ = config; }
    
    /// Get accumulated statistics
    const VariationalStats& get_stats() const { return stats_; }
    
    /// Reset statistics
    void reset_stats() { stats_.reset(); }
    
    /// Get number of variational parameters
    size_t num_params() const { return circuit_params_.size(); }
    
    /// Get the number of qubits
    size_t num_qubits() const { return num_qubits_; }

private:
    size_t num_qubits_;
    size_t dim_;                    ///< 2^num_qubits
    MatrixXcd hamiltonian_;         ///< System Hamiltonian
    std::vector<LindbladDissipator> dissipators_;
    AnsatzConfig config_;
    VariationalStats stats_;
    
    // Variational parameters
    std::vector<double> circuit_params_;    ///< θ: circuit rotation angles
    std::vector<double> probabilities_;     ///< p: ensemble weights
    
    // Adam optimizer state (if config_.use_adam)
    std::vector<double> adam_m_params_;     ///< First moment for circuit params
    std::vector<double> adam_v_params_;     ///< Second moment for circuit params
    std::vector<double> adam_m_probs_;      ///< First moment for probabilities
    std::vector<double> adam_v_probs_;      ///< Second moment for probabilities
    size_t adam_step_ = 0;                  ///< Adam step counter
    
    std::mt19937 rng_;                      ///< Random number generator
    
    //--------------------------------------------------------------------------
    // Internal: Circuit Construction
    //--------------------------------------------------------------------------
    
    /**
     * @brief Construct a parametrized quantum circuit from parameters θ
     * 
     * Circuit structure (per layer):
     *   For each qubit i: RY(θ[...]) → RZ(θ[...])
     *   For each pair (i, i+1): CNOT(i, i+1)
     * 
     * @param params Variational parameters
     * @return QuantumSequence representing the circuit
     */
    QuantumSequence construct_ansatz_circuit(const std::vector<double>& params) const;
    
    /**
     * @brief Evaluate the ansatz circuit on a basis state
     * 
     * Applies U(θ)|basis_i⟩ to get |ψ_i(θ)⟩ using LRET's gate application.
     * 
     * @param basis_index Index of computational basis state
     * @param params      Circuit parameters
     * @return State vector |ψ_i(θ)⟩ (as column of L, dim × 1)
     */
    VectorXcd evaluate_basis_state(size_t basis_index, 
                                    const std::vector<double>& params) const;
    
    //--------------------------------------------------------------------------
    // Internal: Lindblad Derivative
    //--------------------------------------------------------------------------
    
    /**
     * @brief Compute the Lindblad derivative dρ/dt at the current state
     * 
     * dρ/dt = -i[H, ρ] + Σ_k γ_k (L_k ρ L_k† - ½{L_k†L_k, ρ})
     * 
     * @param rho Current density matrix
     * @return dρ/dt as a full matrix
     */
    MatrixXcd compute_lindblad_derivative(const MatrixXcd& rho) const;
    
    //--------------------------------------------------------------------------
    // Internal: Gradient Computation
    //--------------------------------------------------------------------------
    
    /**
     * @brief Compute gradient of cost function w.r.t. circuit parameters
     * 
     * Cost function: C = ||ρ(θ,p) - ρ_target||²_F
     * 
     * Uses parameter-shift rule:
     *   ∂C/∂θ_j = [C(θ_j + s) - C(θ_j - s)] / (2 sin(s))
     * where s = π/2 (standard shift).
     * 
     * @param target_rho Target density matrix
     * @return Gradient vector ∂C/∂θ
     */
    std::vector<double> compute_param_gradient(const MatrixXcd& target_rho);
    
    /**
     * @brief Compute gradient of cost function w.r.t. probabilities
     * 
     * ∂C/∂p_i = 2 · Tr[(ρ - ρ_target) · |ψ_i⟩⟨ψ_i|]
     *         = 2 · ⟨ψ_i|(ρ - ρ_target)|ψ_i⟩
     * 
     * @param target_rho Target density matrix
     * @return Gradient vector ∂C/∂p
     */
    std::vector<double> compute_prob_gradient(const MatrixXcd& target_rho);
    
    //--------------------------------------------------------------------------
    // Internal: Optimization
    //--------------------------------------------------------------------------
    
    /**
     * @brief Project probabilities onto the probability simplex
     * 
     * Ensures Σ_i p_i = 1 and p_i ≥ 0 using Duchi et al. (2008) algorithm.
     * 
     * @param probs  Probabilities to project (modified in-place)
     */
    static void project_simplex(std::vector<double>& probs);
    
    /**
     * @brief Compute Frobenius cost ||ρ(θ,p) - target||²_F
     * 
     * @param target_rho Target density matrix
     * @return Cost value
     */
    double compute_cost(const MatrixXcd& target_rho) const;
    
    /**
     * @brief Compute fidelity F(ρ, σ) = [Tr(√(√ρ σ √ρ))]²
     * 
     * @param rho First density matrix
     * @param sigma Second density matrix
     * @return Fidelity ∈ [0, 1]
     */
    static double compute_fidelity(const MatrixXcd& rho, const MatrixXcd& sigma);
    
    /**
     * @brief Adam optimizer update step
     * 
     * @param params      Parameters to update
     * @param gradient    Gradient of cost w.r.t. params
     * @param m           First moment estimate (updated in-place)
     * @param v           Second moment estimate (updated in-place)
     */
    void adam_update(std::vector<double>& params,
                     const std::vector<double>& gradient,
                     std::vector<double>& m,
                     std::vector<double>& v);
    
    /**
     * @brief Number of parameters per variational circuit layer
     * @return 2 * num_qubits (one RY and one RZ per qubit per layer)
     */
    size_t params_per_layer() const { return 2 * num_qubits_; }
    
    /**
     * @brief Total number of circuit parameters
     * @return num_layers * 2 * num_qubits
     */
    size_t total_circuit_params() const { 
        return config_.num_layers * params_per_layer(); 
    }
};

//==============================================================================
// Free Functions
//==============================================================================

/**
 * @brief Apply variational Lindblad evolution to a noise operation
 * 
 * High-level entry point for integrating variational Lindblad into
 * the LRET simulation pipeline. Converts a NoiseOp into a Lindblad
 * dissipator and evolves one time step.
 * 
 * @param L            Current L factor
 * @param noise        Noise operation
 * @param num_qubits   Number of qubits
 * @param hamiltonian  System Hamiltonian (if empty, uses zero Hamiltonian)
 * @param dt           Time step (default 1.0, representing one noise application)
 * @param config       Ansatz configuration
 * @return Updated L factor
 */
MatrixXcd apply_noise_variational(
    const MatrixXcd& L,
    const NoiseOp& noise,
    size_t num_qubits,
    const MatrixXcd& hamiltonian = MatrixXcd(),
    double dt = 1.0,
    const AnsatzConfig& config = AnsatzConfig{}
);

/**
 * @brief Convert a NoiseOp to a LindbladDissipator
 * 
 * Extracts the jump operator from the Kraus representation:
 * For a noise channel with Kraus ops {K_0, K_1, ...}, the
 * non-identity Kraus operators correspond to jump operators.
 * 
 * @param noise      Noise operation
 * @return Vector of LindbladDissipators
 */
std::vector<LindbladDissipator> noise_to_dissipators(const NoiseOp& noise);

}  // namespace qlret

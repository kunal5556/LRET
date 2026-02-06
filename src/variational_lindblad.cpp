/**
 * @file variational_lindblad.cpp
 * @brief Implementation of Low-Rank Variational Lindblad Evolution (Phase 3B)
 * 
 * Implements the variational ansatz ρ = Σ_i p_i |ψ_i(θ)⟩⟨ψ_i(θ)| for
 * Lindblad master equation evolution. The ansatz keeps rank fixed throughout
 * evolution, avoiding the rank explosion problem in standard LRET noise
 * application.
 * 
 * @see variational_lindblad.h for API documentation
 * @see ROW_PARALLEL_ADVANCED_OPTIMIZATION_PLAN.md Phase 3B
 */

#include "variational_lindblad.h"
#include "simulator.h"
#include <algorithm>
#include <numeric>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cassert>
#include <cmath>

#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace qlret {

//==============================================================================
// VariationalStats
//==============================================================================

void VariationalStats::print() const {
    std::cout << "=== Variational Lindblad Statistics ===" << std::endl;
    std::cout << "  Time steps:          " << time_steps << std::endl;
    std::cout << "  Total iterations:    " << total_iterations << std::endl;
    std::cout << "  Final cost:          " << std::scientific << total_cost << std::endl;
    std::cout << "  Best fidelity:       " << std::fixed << std::setprecision(6) 
              << best_fidelity << std::endl;
    std::cout << "  Optimization time:   " << std::setprecision(4) 
              << optimization_time << " s" << std::endl;
    std::cout << "  Circuit eval time:   " << circuit_eval_time << " s" << std::endl;
    std::cout << "  Gradient time:       " << gradient_time << " s" << std::endl;
    std::cout << "========================================" << std::endl;
}

//==============================================================================
// Constructor
//==============================================================================

VariationalLindblad::VariationalLindblad(
    size_t num_qubits,
    const MatrixXcd& hamiltonian,
    const std::vector<LindbladDissipator>& dissipators,
    const AnsatzConfig& config
)
    : num_qubits_(num_qubits)
    , dim_(1ULL << num_qubits)
    , hamiltonian_(hamiltonian)
    , dissipators_(dissipators)
    , config_(config)
{
    // Validate Hamiltonian dimensions
    if (hamiltonian_.rows() == 0 && hamiltonian_.cols() == 0) {
        // Zero Hamiltonian: pure dissipative evolution
        hamiltonian_ = MatrixXcd::Zero(dim_, dim_);
    }
    assert(static_cast<size_t>(hamiltonian_.rows()) == dim_ && 
           static_cast<size_t>(hamiltonian_.cols()) == dim_);
    
    // Initialize RNG
    if (config_.seed != 0) {
        rng_.seed(config_.seed);
    } else {
        rng_.seed(std::random_device{}());
    }
    
    // Initialize circuit parameters randomly in [0, 2π)
    size_t n_params = total_circuit_params();
    circuit_params_.resize(n_params);
    std::uniform_real_distribution<double> param_dist(0.0, 2.0 * qlret::PI);
    for (size_t i = 0; i < n_params; ++i) {
        circuit_params_[i] = param_dist(rng_);
    }
    
    // Initialize ensemble probabilities uniformly
    size_t m = config_.num_basis_states;
    probabilities_.resize(m, 1.0 / static_cast<double>(m));
    
    // Initialize Adam state
    if (config_.use_adam) {
        adam_m_params_.resize(n_params, 0.0);
        adam_v_params_.resize(n_params, 0.0);
        adam_m_probs_.resize(m, 0.0);
        adam_v_probs_.resize(m, 0.0);
        adam_step_ = 0;
    }
}

//==============================================================================
// Circuit Construction
//==============================================================================

QuantumSequence VariationalLindblad::construct_ansatz_circuit(
    const std::vector<double>& params
) const {
    QuantumSequence seq(num_qubits_);
    size_t param_idx = 0;
    
    for (size_t layer = 0; layer < config_.num_layers; ++layer) {
        // Rotation sub-layer: RY and RZ on each qubit
        for (size_t q = 0; q < num_qubits_; ++q) {
            double ry_angle = params[param_idx++];
            double rz_angle = params[param_idx++];
            
            seq.add_gate(GateOp(GateType::RY, q, std::vector<double>{ry_angle}));
            seq.add_gate(GateOp(GateType::RZ, q, std::vector<double>{rz_angle}));
        }
        
        // Entangling sub-layer: CNOT ladder
        for (size_t q = 0; q + 1 < num_qubits_; ++q) {
            seq.add_gate(GateOp(GateType::CNOT, q, q + 1));
        }
        
        // Wrap-around CNOT for periodic boundary (if more than 2 qubits)
        if (num_qubits_ > 2) {
            seq.add_gate(GateOp(GateType::CNOT, num_qubits_ - 1, static_cast<size_t>(0)));
        }
    }
    
    return seq;
}

//==============================================================================
// Basis State Evaluation
//==============================================================================

VectorXcd VariationalLindblad::evaluate_basis_state(
    size_t basis_index,
    const std::vector<double>& params
) const {
    // Start with computational basis state |basis_index⟩
    // Represented as L = e_{basis_index} (standard basis vector, dim_ × 1)
    MatrixXcd L = MatrixXcd::Zero(dim_, 1);
    if (basis_index < dim_) {
        L(basis_index, 0) = Complex(1.0, 0.0);
    } else {
        // Wrap around if basis_index >= dim_ (shouldn't happen normally)
        L(basis_index % dim_, 0) = Complex(1.0, 0.0);
    }
    
    // Construct and apply the variational circuit
    QuantumSequence circuit = construct_ansatz_circuit(params);
    
    // Apply each gate operation to the L factor
    for (const auto& op : circuit.operations) {
        if (std::holds_alternative<GateOp>(op)) {
            L = apply_gate_to_L(L, std::get<GateOp>(op), num_qubits_);
        }
    }
    
    return L.col(0);
}

//==============================================================================
// Density Matrix Construction
//==============================================================================

MatrixXcd VariationalLindblad::get_density_matrix() const {
    auto t_start = std::chrono::steady_clock::now();
    
    MatrixXcd rho = MatrixXcd::Zero(dim_, dim_);
    
    size_t m = config_.num_basis_states;
    for (size_t i = 0; i < m; ++i) {
        VectorXcd psi_i = evaluate_basis_state(i, circuit_params_);
        rho += probabilities_[i] * psi_i * psi_i.adjoint();
    }
    
    return rho;
}

MatrixXcd VariationalLindblad::get_L_factor() const {
    size_t m = config_.num_basis_states;
    MatrixXcd L(dim_, m);
    
    for (size_t i = 0; i < m; ++i) {
        VectorXcd psi_i = evaluate_basis_state(i, circuit_params_);
        L.col(i) = std::sqrt(probabilities_[i]) * psi_i;
    }
    
    return L;
}

//==============================================================================
// Lindblad Derivative
//==============================================================================

MatrixXcd VariationalLindblad::compute_lindblad_derivative(
    const MatrixXcd& rho
) const {
    // dρ/dt = -i[H, ρ] + Σ_k γ_k (L_k ρ L_k† - ½{L_k†L_k, ρ})
    
    MatrixXcd drho = MatrixXcd::Zero(dim_, dim_);
    
    // Hamiltonian part: -i[H, ρ] = -i(Hρ - ρH)
    drho -= Complex(0.0, 1.0) * (hamiltonian_ * rho - rho * hamiltonian_);
    
    // Dissipative part: Σ_k γ_k (L_k ρ L_k† - ½{L_k†L_k, ρ})
    for (const auto& diss : dissipators_) {
        // Expand jump operator to full Hilbert space
        // For single-qubit operators, need to embed in 2^n space
        MatrixXcd L_full;
        
        if (diss.op.rows() == 2 && diss.qubits.size() == 1) {
            // Single-qubit: embed as I ⊗ ... ⊗ L_k ⊗ ... ⊗ I
            L_full = MatrixXcd::Identity(dim_, dim_);
            size_t qubit = diss.qubits[0];
            size_t bit = 1ULL << qubit;
            
            // Apply the 2×2 operator to the qubit's subspace
            MatrixXcd L_embed = MatrixXcd::Zero(dim_, dim_);
            for (size_t row = 0; row < dim_; ++row) {
                for (size_t col = 0; col < dim_; ++col) {
                    // Check if row and col differ only in qubit 'qubit'
                    size_t row_rest = row & ~bit;
                    size_t col_rest = col & ~bit;
                    if (row_rest != col_rest) continue;
                    
                    size_t row_bit = (row >> qubit) & 1;
                    size_t col_bit = (col >> qubit) & 1;
                    L_embed(row, col) = diss.op(row_bit, col_bit);
                }
            }
            L_full = L_embed;
        } else if (diss.op.rows() == 4 && diss.qubits.size() == 2) {
            // Two-qubit: embed similarly
            size_t q1 = diss.qubits[0];
            size_t q2 = diss.qubits[1];
            size_t bit1 = 1ULL << q1;
            size_t bit2 = 1ULL << q2;
            
            L_full = MatrixXcd::Zero(dim_, dim_);
            for (size_t row = 0; row < dim_; ++row) {
                for (size_t col = 0; col < dim_; ++col) {
                    // Check non-target bits match
                    size_t row_rest = row & ~bit1 & ~bit2;
                    size_t col_rest = col & ~bit1 & ~bit2;
                    if (row_rest != col_rest) continue;
                    
                    size_t r1 = (row >> q1) & 1;
                    size_t r2 = (row >> q2) & 1;
                    size_t c1 = (col >> q1) & 1;
                    size_t c2 = (col >> q2) & 1;
                    size_t row_idx = (r1 << 1) | r2;
                    size_t col_idx = (c1 << 1) | c2;
                    L_full(row, col) = diss.op(row_idx, col_idx);
                }
            }
        } else {
            // Full-space operator: use directly
            L_full = diss.op;
        }
        
        MatrixXcd L_dag = L_full.adjoint();
        MatrixXcd L_dag_L = L_dag * L_full;
        
        drho += diss.rate * (
            L_full * rho * L_dag 
            - 0.5 * L_dag_L * rho 
            - 0.5 * rho * L_dag_L
        );
    }
    
    return drho;
}

//==============================================================================
// Gradient Computation
//==============================================================================

std::vector<double> VariationalLindblad::compute_param_gradient(
    const MatrixXcd& target_rho
) {
    auto t_start = std::chrono::steady_clock::now();
    
    size_t n_params = circuit_params_.size();
    std::vector<double> gradient(n_params, 0.0);
    double shift = config_.param_shift;
    
    // Parameter-shift rule: for each parameter θ_j
    // ∂C/∂θ_j ≈ [C(θ_j + shift) - C(θ_j - shift)] / (2 sin(shift))
    double denom = 2.0 * std::sin(shift);
    
    for (size_t j = 0; j < n_params; ++j) {
        // Forward shift
        std::vector<double> params_plus = circuit_params_;
        params_plus[j] += shift;
        
        // Backward shift
        std::vector<double> params_minus = circuit_params_;
        params_minus[j] -= shift;
        
        // Compute cost at shifted parameters
        // Temporarily swap circuit_params_ for cost evaluation
        auto saved = circuit_params_;
        
        circuit_params_ = params_plus;
        double cost_plus = compute_cost(target_rho);
        
        circuit_params_ = params_minus;
        double cost_minus = compute_cost(target_rho);
        
        circuit_params_ = saved;
        
        gradient[j] = (cost_plus - cost_minus) / denom;
    }
    
    stats_.gradient_time += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_start).count();
    
    return gradient;
}

std::vector<double> VariationalLindblad::compute_prob_gradient(
    const MatrixXcd& target_rho
) {
    auto t_start = std::chrono::steady_clock::now();
    
    size_t m = config_.num_basis_states;
    std::vector<double> gradient(m, 0.0);
    
    // ∂C/∂p_i = 2 · ⟨ψ_i| (ρ - ρ_target) |ψ_i⟩
    // where ρ = current density matrix
    MatrixXcd rho = get_density_matrix();
    MatrixXcd diff = rho - target_rho;
    
    for (size_t i = 0; i < m; ++i) {
        VectorXcd psi_i = evaluate_basis_state(i, circuit_params_);
        // ⟨ψ_i| diff |ψ_i⟩ is a real number (since diff is Hermitian for ρ-ρ_target)
        Complex inner = psi_i.adjoint() * diff * psi_i;
        gradient[i] = 2.0 * inner.real();
    }
    
    stats_.gradient_time += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_start).count();
    
    return gradient;
}

//==============================================================================
// Cost Function
//==============================================================================

double VariationalLindblad::compute_cost(const MatrixXcd& target_rho) const {
    auto t_start = std::chrono::steady_clock::now();
    
    MatrixXcd rho = get_density_matrix();
    double cost = (rho - target_rho).squaredNorm();
    
    // Note: squaredNorm() gives Σ|a_ij|² = ||A||²_F
    
    return cost;
}

//==============================================================================
// Fidelity
//==============================================================================

double VariationalLindblad::compute_fidelity(
    const MatrixXcd& rho, 
    const MatrixXcd& sigma
) {
    // F(ρ, σ) = [Tr(√(√ρ σ √ρ))]²
    // Simplified for density matrices: use eigendecomposition
    
    // Compute √ρ via eigendecomposition
    Eigen::SelfAdjointEigenSolver<MatrixXcd> solver_rho(rho);
    if (solver_rho.info() != Eigen::Success) return 0.0;
    
    VectorXd evals = solver_rho.eigenvalues().real();
    MatrixXcd evecs = solver_rho.eigenvectors();
    
    // √ρ = V diag(√λ) V†
    VectorXd sqrt_evals(evals.size());
    for (int i = 0; i < evals.size(); ++i) {
        sqrt_evals(i) = (evals(i) > 0) ? std::sqrt(evals(i)) : 0.0;
    }
    MatrixXcd sqrt_rho = evecs * sqrt_evals.asDiagonal() * evecs.adjoint();
    
    // M = √ρ σ √ρ
    MatrixXcd M = sqrt_rho * sigma * sqrt_rho;
    
    // Tr(√M) = Σ √(eigenvalues of M)
    Eigen::SelfAdjointEigenSolver<MatrixXcd> solver_M(M);
    if (solver_M.info() != Eigen::Success) return 0.0;
    
    VectorXd M_evals = solver_M.eigenvalues().real();
    double trace_sqrt_M = 0.0;
    for (int i = 0; i < M_evals.size(); ++i) {
        if (M_evals(i) > 0) {
            trace_sqrt_M += std::sqrt(M_evals(i));
        }
    }
    
    return trace_sqrt_M * trace_sqrt_M;
}

//==============================================================================
// Simplex Projection (Duchi et al. 2008)
//==============================================================================

void VariationalLindblad::project_simplex(std::vector<double>& probs) {
    size_t n = probs.size();
    if (n == 0) return;
    
    // Sort in descending order
    std::vector<double> sorted(probs);
    std::sort(sorted.begin(), sorted.end(), std::greater<double>());
    
    // Find the threshold τ such that:
    // τ = (Σ_{j ∈ S} u_j - 1) / |S|
    // where S = { j : u_j - τ > 0 }
    double cumsum = 0.0;
    double tau = 0.0;
    size_t rho_idx = 0;
    
    for (size_t j = 0; j < n; ++j) {
        cumsum += sorted[j];
        double t = (cumsum - 1.0) / static_cast<double>(j + 1);
        if (sorted[j] - t > 0) {
            tau = t;
            rho_idx = j + 1;
        }
    }
    
    // Project: p_i = max(p_i - τ, 0)
    for (size_t i = 0; i < n; ++i) {
        probs[i] = std::max(probs[i] - tau, 0.0);
    }
    
    // Ensure exact summation to 1.0 (numerical safety)
    double sum = std::accumulate(probs.begin(), probs.end(), 0.0);
    if (sum > 0.0) {
        for (double& p : probs) {
            p /= sum;
        }
    } else {
        // Fallback: uniform distribution
        for (double& p : probs) {
            p = 1.0 / static_cast<double>(n);
        }
    }
}

//==============================================================================
// Adam Optimizer
//==============================================================================

void VariationalLindblad::adam_update(
    std::vector<double>& params,
    const std::vector<double>& gradient,
    std::vector<double>& m,
    std::vector<double>& v
) {
    adam_step_++;
    double beta1 = config_.adam_beta1;
    double beta2 = config_.adam_beta2;
    double eps = config_.adam_epsilon;
    double lr = config_.learning_rate;
    
    // Bias-corrected learning rate
    double lr_t = lr * std::sqrt(1.0 - std::pow(beta2, adam_step_)) 
                     / (1.0 - std::pow(beta1, adam_step_));
    
    for (size_t i = 0; i < params.size(); ++i) {
        // Update biased first and second moment estimates
        m[i] = beta1 * m[i] + (1.0 - beta1) * gradient[i];
        v[i] = beta2 * v[i] + (1.0 - beta2) * gradient[i] * gradient[i];
        
        // Update parameters
        params[i] -= lr_t * m[i] / (std::sqrt(v[i]) + eps);
    }
}

//==============================================================================
// Optimize Ansatz
//==============================================================================

double VariationalLindblad::optimize_ansatz(const MatrixXcd& target_rho) {
    auto t_start = std::chrono::steady_clock::now();
    
    double best_cost = compute_cost(target_rho);
    std::vector<double> best_params = circuit_params_;
    std::vector<double> best_probs = probabilities_;
    
    for (size_t iter = 0; iter < config_.max_iterations; ++iter) {
        // Compute gradients
        auto param_grad = compute_param_gradient(target_rho);
        auto prob_grad = compute_prob_gradient(target_rho);
        
        // Update parameters
        if (config_.use_adam) {
            adam_update(circuit_params_, param_grad, adam_m_params_, adam_v_params_);
            adam_update(probabilities_, prob_grad, adam_m_probs_, adam_v_probs_);
        } else {
            // Simple gradient descent
            for (size_t j = 0; j < circuit_params_.size(); ++j) {
                circuit_params_[j] -= config_.learning_rate * param_grad[j];
            }
            for (size_t i = 0; i < probabilities_.size(); ++i) {
                probabilities_[i] -= config_.learning_rate * prob_grad[i];
            }
        }
        
        // Project probabilities onto simplex
        project_simplex(probabilities_);
        
        // Compute cost
        double cost = compute_cost(target_rho);
        
        if (cost < best_cost) {
            best_cost = cost;
            best_params = circuit_params_;
            best_probs = probabilities_;
        }
        
        stats_.total_iterations++;
        
        // Convergence check
        if (cost < config_.convergence_tol) {
            if (config_.verbose) {
                std::cout << "[VariationalLindblad] Converged at iteration " 
                          << iter << ", cost = " << std::scientific << cost 
                          << std::endl;
            }
            break;
        }
        
        if (config_.verbose && (iter % 10 == 0 || iter == config_.max_iterations - 1)) {
            std::cout << "[VariationalLindblad] Iter " << iter 
                      << ": cost = " << std::scientific << cost << std::endl;
        }
    }
    
    // Restore best parameters
    circuit_params_ = best_params;
    probabilities_ = best_probs;
    stats_.total_cost = best_cost;
    
    // Compute fidelity
    MatrixXcd rho = get_density_matrix();
    stats_.best_fidelity = compute_fidelity(rho, target_rho);
    
    stats_.optimization_time += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_start).count();
    
    return best_cost;
}

//==============================================================================
// Evolve
//==============================================================================

MatrixXcd VariationalLindblad::evolve(double dt) {
    auto t_start = std::chrono::steady_clock::now();
    
    // Step 1: Get current density matrix
    MatrixXcd rho = get_density_matrix();
    
    // Step 2: Compute Lindblad derivative
    MatrixXcd drho = compute_lindblad_derivative(rho);
    
    // Step 3: Target = ρ + dρ/dt · dt
    MatrixXcd target_rho = rho + drho * dt;
    
    // Ensure target is a valid density matrix:
    // - Make Hermitian: target = (target + target†) / 2
    target_rho = 0.5 * (target_rho + target_rho.adjoint());
    
    // - Make positive semidefinite: project eigenvalues to [0, ∞)
    Eigen::SelfAdjointEigenSolver<MatrixXcd> solver(target_rho);
    if (solver.info() == Eigen::Success) {
        VectorXd evals = solver.eigenvalues().real();
        MatrixXcd evecs = solver.eigenvectors();
        
        // Clamp negative eigenvalues to zero
        for (int i = 0; i < evals.size(); ++i) {
            if (evals(i) < 0.0) evals(i) = 0.0;
        }
        
        // Renormalize to trace 1
        double trace = evals.sum();
        if (trace > 1e-10) {
            evals /= trace;
        }
        
        target_rho = evecs * evals.asDiagonal() * evecs.adjoint();
    }
    
    // Step 4: Optimize ansatz to match target
    optimize_ansatz(target_rho);
    
    stats_.time_steps++;
    stats_.optimization_time += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_start).count();
    
    return get_density_matrix();
}

MatrixXcd VariationalLindblad::evolve_multi(double dt, size_t num_steps) {
    MatrixXcd rho;
    for (size_t step = 0; step < num_steps; ++step) {
        rho = evolve(dt);
        
        if (config_.verbose && (step % 10 == 0 || step == num_steps - 1)) {
            double trace = rho.trace().real();
            std::cout << "[VariationalLindblad] Step " << step + 1 << "/" << num_steps
                      << ", Tr(ρ) = " << std::fixed << std::setprecision(6) << trace
                      << std::endl;
        }
    }
    return rho;
}

//==============================================================================
// Set from L Factor
//==============================================================================

void VariationalLindblad::set_from_L(const MatrixXcd& L) {
    size_t m = std::min(static_cast<size_t>(L.cols()), config_.num_basis_states);
    
    // Extract probabilities from column norms
    probabilities_.resize(config_.num_basis_states, 0.0);
    double total_norm_sq = 0.0;
    
    for (size_t i = 0; i < m; ++i) {
        double norm_sq = L.col(i).squaredNorm();
        probabilities_[i] = norm_sq;
        total_norm_sq += norm_sq;
    }
    
    // Normalize
    if (total_norm_sq > 1e-10) {
        for (size_t i = 0; i < config_.num_basis_states; ++i) {
            probabilities_[i] /= total_norm_sq;
        }
    } else {
        // Fallback to uniform
        for (size_t i = 0; i < config_.num_basis_states; ++i) {
            probabilities_[i] = 1.0 / static_cast<double>(config_.num_basis_states);
        }
    }
    
    // Optimize circuit parameters to match the pure states
    // Construct target density matrix from L
    MatrixXcd target_rho = L * L.adjoint();
    double trace = target_rho.trace().real();
    if (trace > 1e-10) {
        target_rho /= trace;
    }
    
    optimize_ansatz(target_rho);
}

//==============================================================================
// Free Functions
//==============================================================================

std::vector<LindbladDissipator> noise_to_dissipators(const NoiseOp& noise) {
    std::vector<LindbladDissipator> dissipators;
    
    // Get Kraus operators for this noise type
    std::vector<MatrixXcd> kraus_ops;
    if (noise.type == NoiseType::CUSTOM && !noise.custom_kraus.empty()) {
        kraus_ops = noise.custom_kraus;
    } else {
        kraus_ops = get_noise_kraus_operators(noise.type, noise.probability, noise.params);
    }
    
    if (kraus_ops.empty()) return dissipators;
    
    // The first Kraus operator K_0 ≈ I - ε·(Σ L_k†L_k)/2 represents
    // the "no-jump" evolution. The remaining K_k correspond to jump operators.
    // 
    // For a noise channel ρ → Σ_k K_k ρ K_k†, the Lindblad form is:
    //   dρ/dt ≈ Σ_{k≥1} (K_k ρ K_k† - ½{K_k†K_k, ρ})
    //
    // We extract jump operators from non-identity Kraus operators.
    
    for (size_t k = 0; k < kraus_ops.size(); ++k) {
        const MatrixXcd& K = kraus_ops[k];
        
        // Check if this Kraus operator is close to identity (the "no-jump" operator)
        MatrixXcd identity = MatrixXcd::Identity(K.rows(), K.cols());
        double dist_to_identity = (K - identity).norm();
        
        if (dist_to_identity > 0.01) {
            // This is a jump operator
            dissipators.emplace_back(K, noise.qubits, 1.0);
        }
    }
    
    return dissipators;
}

MatrixXcd apply_noise_variational(
    const MatrixXcd& L,
    const NoiseOp& noise,
    size_t num_qubits,
    const MatrixXcd& hamiltonian,
    double dt,
    const AnsatzConfig& config
) {
    // Convert noise to dissipators
    auto dissipators = noise_to_dissipators(noise);
    
    if (dissipators.empty()) {
        // No dissipation: just return L unchanged
        return L;
    }
    
    // Set up Hamiltonian (zero if not provided)
    MatrixXcd H = hamiltonian;
    if (H.rows() == 0) {
        size_t dim = 1ULL << num_qubits;
        H = MatrixXcd::Zero(dim, dim);
    }
    
    // Create variational Lindblad evolver
    VariationalLindblad evolver(num_qubits, H, dissipators, config);
    
    // Initialize from current L factor
    evolver.set_from_L(L);
    
    // Evolve one time step
    evolver.evolve(dt);
    
    // Return updated L factor
    return evolver.get_L_factor();
}

}  // namespace qlret

/**
 * @file matrix_completion.cpp
 * @brief Implementation of Low-Rank Matrix Completion & Quantum State Tomography
 * 
 * Phase 5 of Advanced Row-Parallel Optimization.
 * 
 * This file implements two main classes:
 * 1. MatrixCompletion (Phase 5A) — Reconstructs density matrices from partial
 *    Pauli measurements using nuclear-norm minimization.
 * 2. QuantumStateTomography (Phase 5B) — Compressed tomography pipeline with
 *    adaptive measurement selection and denoising.
 * 
 * ALGORITHMS IMPLEMENTED:
 * 
 * SVD Thresholding (Iterative Singular Value Thresholding / ISVT):
 *   The nuclear norm ||X||_* = Σ σ_i is a convex surrogate for rank.
 *   We solve:  min_X  τ||X||_*  +  ½ ||A(X) - b||²
 *   via proximal gradient descent:
 *     X_{k+1} = SVT_{τδ}( X_k  -  δ · A*(A(X_k) - b) )
 *   where SVT_{λ}(M) = U diag(max(σ_i - λ, 0)) V† is the soft-thresholding
 *   operator applied to singular values.
 * 
 *   For density matrices (Hermitian PSD), we use eigendecomposition instead
 *   of SVD, which is slightly cheaper and preserves Hermiticity.
 * 
 * Alternating Projection:
 *   Alternates between two convex sets:
 *   1. Measurement constraints:  {X : Tr[P_i X] = m_i  ∀i}
 *   2. PSD + trace-1 cone:       {X : X ≥ 0, Tr[X] = 1}
 *   Converges by Dykstra/von Neumann theorem for convex sets.
 * 
 * Reference:
 *   Candes & Recht, "Exact Matrix Completion via Convex Optimization"
 *   Gross et al., "Quantum State Tomography via Compressed Sensing"
 * 
 * @see matrix_completion.h for API documentation
 * @see ROW_PARALLEL_ADVANCED_OPTIMIZATION_PLAN.md Phase 5A/5B
 */

#include "matrix_completion.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <random>
#include <numeric>
#include <cassert>
#include <set>

namespace qlret {

//==============================================================================
// Pauli Utilities
//==============================================================================

/**
 * Build the 2x2 Pauli matrix for a single character.
 */
static MatrixXcd single_pauli(char c) {
    MatrixXcd m(2, 2);
    switch (c) {
        case 'I':
            m << Complex(1,0), Complex(0,0),
                 Complex(0,0), Complex(1,0);
            break;
        case 'X':
            m << Complex(0,0), Complex(1,0),
                 Complex(1,0), Complex(0,0);
            break;
        case 'Y':
            m << Complex(0,0), Complex(0,-1),
                 Complex(0,1), Complex(0,0);
            break;
        case 'Z':
            m << Complex(1,0), Complex(0,0),
                 Complex(0,0), Complex(-1,0);
            break;
        default:
            // Treat unknown as identity
            m << Complex(1,0), Complex(0,0),
                 Complex(0,0), Complex(1,0);
            break;
    }
    return m;
}

MatrixXcd pauli_string_matrix(const std::string& label, size_t num_qubits) {
    assert(label.size() == num_qubits);
    if (num_qubits == 0) {
        MatrixXcd m(1, 1);
        m(0, 0) = Complex(1, 0);
        return m;
    }

    // Build tensor product from left to right:  P[0] ⊗ P[1] ⊗ ... ⊗ P[n-1]
    MatrixXcd result = single_pauli(label[0]);
    for (size_t i = 1; i < num_qubits; ++i) {
        MatrixXcd pi = single_pauli(label[i]);
        // Kronecker product: result ⊗ pi
        const Eigen::Index r1 = result.rows(), c1 = result.cols();
        const Eigen::Index r2 = pi.rows(), c2 = pi.cols();
        MatrixXcd kron(r1 * r2, c1 * c2);
        for (Eigen::Index a = 0; a < r1; ++a)
            for (Eigen::Index b = 0; b < c1; ++b)
                kron.block(a * r2, b * c2, r2, c2) = result(a, b) * pi;
        result = std::move(kron);
    }
    return result;
}

std::vector<std::string> enumerate_pauli_strings(size_t num_qubits) {
    // 4^n strings over {I, X, Y, Z}
    const char chars[] = {'I', 'X', 'Y', 'Z'};
    size_t total = 1;
    for (size_t i = 0; i < num_qubits; ++i) total *= 4;

    std::vector<std::string> result;
    result.reserve(total);

    for (size_t idx = 0; idx < total; ++idx) {
        std::string s(num_qubits, 'I');
        size_t tmp = idx;
        for (size_t q = num_qubits; q > 0; --q) {
            s[q - 1] = chars[tmp % 4];
            tmp /= 4;
        }
        result.push_back(std::move(s));
    }
    return result;
}

double pauli_expectation_from_L(const MatrixXcd& L, const MatrixXcd& pauli) {
    // Tr[P ρ] = Tr[P L L†] = Tr[L† P L]
    // = Σ_j (L† P L)_{jj}  = Σ_j L_j† P L_j  where L_j is col j of L
    //
    // We compute P·L first (dim × rank), then dot with L column-by-column.
    const MatrixXcd PL = pauli * L;  // (dim × rank)
    // Trace of L† PL = sum of element-wise conjugate-multiply
    Complex trace = (L.conjugate().array() * PL.array()).sum();
    return trace.real();
}

//==============================================================================
// MatrixCompletion - Constructor
//==============================================================================

MatrixCompletion::MatrixCompletion(size_t num_qubits, const CompletionConfig& config)
    : num_qubits_(num_qubits)
    , dim_(static_cast<size_t>(1) << num_qubits)
    , config_(config)
{
    // Precompute single-qubit Paulis
    pauli_I_ = single_pauli('I');
    pauli_X_ = single_pauli('X');
    pauli_Y_ = single_pauli('Y');
    pauli_Z_ = single_pauli('Z');
}

//==============================================================================
// MatrixCompletion - complete_from_paulis
//==============================================================================

std::pair<MatrixXcd, CompletionStats> MatrixCompletion::complete_from_paulis(
    const std::map<std::string, double>& pauli_measurements
) {
    // Build the list of (Pauli_matrix, measured_value) pairs
    std::vector<std::pair<MatrixXcd, double>> measurements;
    measurements.reserve(pauli_measurements.size());

    for (const auto& [label, value] : pauli_measurements) {
        MatrixXcd P = pauli_string_matrix(label, num_qubits_);
        measurements.emplace_back(std::move(P), value);
    }

    // Dispatch to solver
    std::pair<MatrixXcd, CompletionStats> result;
    switch (config_.solver) {
        case CompletionSolver::SVDThreshold:
            result = solve_svt(measurements);
            break;
        case CompletionSolver::AlternatingProjection:
            result = solve_alternating(measurements);
            break;
    }

    // Enforce density matrix constraints if requested
    if (config_.enforce_dm_constraints) {
        result.first = enforce_dm_constraints(result.first);
    }

    return result;
}

//==============================================================================
// MatrixCompletion - complete_2rdm
//==============================================================================

std::pair<MatrixXcd, CompletionStats> MatrixCompletion::complete_2rdm(
    const std::vector<std::tuple<size_t, size_t, Complex>>& partial_elements,
    size_t rdm_dim
) {
    // For 2-RDM completion, we convert the known elements into a set of
    // "measurement" operators.  Each known element (i,j,v) means:
    //   Tr[ |i⟩⟨j| · ρ ] = v    (where |i⟩⟨j| is the elementary matrix E_ij)
    //
    // Since ρ is Hermitian, we also have Tr[ |j⟩⟨i| · ρ ] = conj(v).
    // We use these as our "Pauli" measurements (they're not Pauli, but the
    // solver handles arbitrary Hermitian measurement operators).

    std::vector<std::pair<MatrixXcd, double>> measurements;
    measurements.reserve(partial_elements.size() * 2);

    for (const auto& [row, col, val] : partial_elements) {
        if (row >= rdm_dim || col >= rdm_dim) continue;

        if (row == col) {
            // Diagonal element: Tr[ |i⟩⟨i| · ρ ] = ρ_{ii} (real for Hermitian)
            MatrixXcd E = MatrixXcd::Zero(rdm_dim, rdm_dim);
            E(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(col)) = Complex(1, 0);
            measurements.emplace_back(std::move(E), val.real());
        } else {
            // Off-diagonal: use Hermitian operators to extract real and imaginary parts
            // Real part: (|i⟩⟨j| + |j⟩⟨i|) has trace = 2·Re(ρ_{ij})
            MatrixXcd E_real = MatrixXcd::Zero(rdm_dim, rdm_dim);
            E_real(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(col)) = Complex(1, 0);
            E_real(static_cast<Eigen::Index>(col), static_cast<Eigen::Index>(row)) = Complex(1, 0);
            measurements.emplace_back(std::move(E_real), 2.0 * val.real());

            // Imaginary part: i(|j⟩⟨i| - |i⟩⟨j|) has trace = 2·Im(ρ_{ij})
            MatrixXcd E_imag = MatrixXcd::Zero(rdm_dim, rdm_dim);
            E_imag(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(col)) = Complex(0, -1);
            E_imag(static_cast<Eigen::Index>(col), static_cast<Eigen::Index>(row)) = Complex(0, 1);
            measurements.emplace_back(std::move(E_imag), 2.0 * val.imag());
        }
    }

    // Temporarily override dim_ for the solver (2-RDM may be smaller than 2^n)
    const size_t saved_dim = dim_;
    dim_ = rdm_dim;

    std::pair<MatrixXcd, CompletionStats> result;
    switch (config_.solver) {
        case CompletionSolver::SVDThreshold:
            result = solve_svt(measurements);
            break;
        case CompletionSolver::AlternatingProjection:
            result = solve_alternating(measurements);
            break;
    }

    dim_ = saved_dim;

    if (config_.enforce_dm_constraints) {
        result.first = enforce_dm_constraints(result.first);
    }

    return result;
}

//==============================================================================
// MatrixCompletion - suggest_measurements
//==============================================================================

std::vector<std::string> MatrixCompletion::suggest_measurements(
    size_t num_suggestions,
    const std::map<std::string, double>& already_measured
) {
    // Strategy: use leverage score heuristic.
    // 
    // For density matrices, low-weight Pauli strings (few non-I positions)
    // typically carry the most information.  We prioritize:
    // 1. Weight-1 operators: Z_i, X_i, Y_i (always include)
    // 2. Weight-2 operators: Z_i Z_j, X_i X_j, etc. (most of the correlations)
    // 3. Random higher-weight operators (fill remaining budget)
    //
    // Skip any that are already measured.

    std::set<std::string> measured_set;
    for (const auto& [label, _] : already_measured) {
        measured_set.insert(label);
    }

    std::vector<std::string> suggestions;
    suggestions.reserve(num_suggestions);

    const char paulis[] = {'X', 'Y', 'Z'};

    // Priority 1: Weight-1 operators (3n total)
    for (size_t q = 0; q < num_qubits_; ++q) {
        for (char p : paulis) {
            std::string label(num_qubits_, 'I');
            label[q] = p;
            if (measured_set.count(label) == 0) {
                suggestions.push_back(label);
                measured_set.insert(label);
                if (suggestions.size() >= num_suggestions) return suggestions;
            }
        }
    }

    // Priority 2: Weight-2 operators (most informative for correlations)
    for (size_t q1 = 0; q1 < num_qubits_ && suggestions.size() < num_suggestions; ++q1) {
        for (size_t q2 = q1 + 1; q2 < num_qubits_ && suggestions.size() < num_suggestions; ++q2) {
            for (char p1 : paulis) {
                for (char p2 : paulis) {
                    if (suggestions.size() >= num_suggestions) return suggestions;
                    std::string label(num_qubits_, 'I');
                    label[q1] = p1;
                    label[q2] = p2;
                    if (measured_set.count(label) == 0) {
                        suggestions.push_back(label);
                        measured_set.insert(label);
                    }
                }
            }
        }
    }

    // Priority 3: Random higher-weight operators
    if (suggestions.size() < num_suggestions) {
        std::mt19937 rng(42);
        auto all_paulis = enumerate_pauli_strings(num_qubits_);

        // Shuffle and pick unmeasured ones
        std::shuffle(all_paulis.begin(), all_paulis.end(), rng);
        for (const auto& label : all_paulis) {
            if (suggestions.size() >= num_suggestions) break;
            if (measured_set.count(label) == 0) {
                suggestions.push_back(label);
                measured_set.insert(label);
            }
        }
    }

    return suggestions;
}

//==============================================================================
// MatrixCompletion - enforce_dm_constraints
//==============================================================================

MatrixXcd MatrixCompletion::enforce_dm_constraints(const MatrixXcd& rho) const {
    const Eigen::Index d = rho.rows();

    // Step 1: Hermitianize
    MatrixXcd X = 0.5 * (rho + rho.adjoint());

    // Step 2: Project onto PSD cone
    Eigen::SelfAdjointEigenSolver<MatrixXcd> solver(X);
    if (solver.info() != Eigen::Success) {
        // Fallback: return Hermitianized matrix
        return X / X.trace();
    }

    VectorXd eigenvalues = solver.eigenvalues();
    MatrixXcd eigenvectors = solver.eigenvectors();

    // Clamp negative eigenvalues to zero
    for (Eigen::Index i = 0; i < d; ++i) {
        if (eigenvalues(i) < 0.0) {
            eigenvalues(i) = 0.0;
        }
    }

    // Step 3: Normalize to trace 1
    double trace = eigenvalues.sum();
    if (trace > 1e-15) {
        eigenvalues /= trace;
    } else {
        // Degenerate case: uniform mixed state
        eigenvalues.setConstant(1.0 / static_cast<double>(d));
    }

    // Reconstruct
    MatrixXcd result = eigenvectors * eigenvalues.asDiagonal().toDenseMatrix().cast<Complex>() * eigenvectors.adjoint();
    return result;
}

//==============================================================================
// MatrixCompletion - rho_to_L
//==============================================================================

MatrixXcd MatrixCompletion::rho_to_L(const MatrixXcd& rho, double threshold) const {
    const Eigen::Index d = rho.rows();

    Eigen::SelfAdjointEigenSolver<MatrixXcd> solver(rho);
    if (solver.info() != Eigen::Success) {
        // Fallback: return single column (|0⟩)
        MatrixXcd L = MatrixXcd::Zero(d, 1);
        L(0, 0) = Complex(1.0, 0.0);
        return L;
    }

    VectorXd eigenvalues = solver.eigenvalues();
    MatrixXcd eigenvectors = solver.eigenvectors();

    // Count eigenvalues above threshold
    size_t rank = 0;
    for (Eigen::Index i = d - 1; i >= 0; --i) {
        if (eigenvalues(i) > threshold) {
            ++rank;
        } else {
            break;
        }
    }
    if (rank == 0) rank = 1;  // At least rank 1

    // Build L from top eigenvalues: L_j = √λ_j · v_j
    MatrixXcd L(d, static_cast<Eigen::Index>(rank));
    for (size_t k = 0; k < rank; ++k) {
        Eigen::Index idx = d - 1 - static_cast<Eigen::Index>(k);
        double sqrt_lam = std::sqrt(std::max(eigenvalues(idx), 0.0));
        L.col(static_cast<Eigen::Index>(k)) = sqrt_lam * eigenvectors.col(idx);
    }

    return L;
}

//==============================================================================
// MatrixCompletion - SVD Thresholding Solver
//==============================================================================

std::pair<MatrixXcd, CompletionStats> MatrixCompletion::solve_svt(
    const std::vector<std::pair<MatrixXcd, double>>& measurements
) {
    auto t_start = std::chrono::high_resolution_clock::now();

    const Eigen::Index d = static_cast<Eigen::Index>(dim_);
    const size_t m = measurements.size();

    CompletionStats stats;

    // Auto-select parameters
    double tau = config_.svt_tau;
    if (tau <= 0.0) {
        // Heuristic: τ ∝ dimension / √(num_measurements)
        // For density matrices, we want low nuclear norm (low rank).
        tau = static_cast<double>(d) / std::sqrt(static_cast<double>(std::max(m, size_t(1))));
        // Clamp to reasonable range
        tau = std::max(tau, 0.01);
        tau = std::min(tau, static_cast<double>(d));
    }

    double delta = config_.step_size;
    if (delta <= 0.0) {
        // Step size: inversely proportional to the spectral norm of the
        // measurement operator.  For Pauli operators (unitary, trace = d or 0),
        // the operator norm of A*A is bounded by m (each Pauli has ||P||₂ = 1
        // but Tr[P P] = d, so we use 1/d as a rough step size).
        delta = 1.0 / static_cast<double>(std::max(m, size_t(1)));
    }

    // Initialize X to the zero matrix (could also warm-start from identity/d)
    MatrixXcd X = MatrixXcd::Zero(d, d);

    // Precompute b vector and measurement matrices are already provided
    double prev_residual = 1e20;

    for (size_t iter = 0; iter < config_.max_iterations; ++iter) {
        // Compute residual: r_i = Tr[P_i X] - m_i
        // and gradient: A*(r) = Σ_i r_i P_i
        double residual_sq = 0.0;
        MatrixXcd gradient = MatrixXcd::Zero(d, d);

        for (size_t i = 0; i < m; ++i) {
            const MatrixXcd& P = measurements[i].first;
            double target = measurements[i].second;

            // Compute Tr[P X] efficiently
            Complex tr = (P.array() * X.transpose().array()).sum();
            double residual_i = tr.real() - target;

            residual_sq += residual_i * residual_i;
            gradient += residual_i * P;
        }

        double residual = std::sqrt(residual_sq);

        // Check convergence
        double rel_change = std::abs(prev_residual - residual) / (prev_residual + 1e-15);
        if (config_.verbose && (iter % 20 == 0 || iter < 5)) {
            std::cout << "  SVT iter " << iter 
                      << ": residual=" << residual 
                      << " rel_change=" << rel_change << std::endl;
        }

        if (rel_change < config_.tolerance && iter > 0) {
            stats.converged = true;
            stats.iterations = iter + 1;
            stats.final_residual = residual;
            stats.convergence_ratio = rel_change;
            break;
        }

        prev_residual = residual;
        stats.iterations = iter + 1;
        stats.final_residual = residual;
        stats.convergence_ratio = rel_change;

        // Proximal gradient step: Y = X - δ · gradient
        MatrixXcd Y = X - delta * gradient;

        // Hermitianize (density matrices are Hermitian)
        Y = 0.5 * (Y + Y.adjoint());

        // SVT (Singular Value Thresholding) via eigendecomposition
        // Since Y is Hermitian, singular values = |eigenvalues|
        Eigen::SelfAdjointEigenSolver<MatrixXcd> solver(Y);
        if (solver.info() != Eigen::Success) {
            // If eigendecomp fails, keep Y as-is
            X = Y;
            continue;
        }

        VectorXd eigvals = solver.eigenvalues();
        MatrixXcd eigvecs = solver.eigenvectors();

        // Soft-threshold eigenvalues by τ·δ
        double threshold = tau * delta;
        for (Eigen::Index i = 0; i < d; ++i) {
            double val = eigvals(i);
            if (val > threshold) {
                eigvals(i) = val - threshold;
            } else if (val < -threshold) {
                eigvals(i) = val + threshold;
            } else {
                eigvals(i) = 0.0;
            }
        }

        // Reconstruct X from thresholded eigenvalues
        X = eigvecs * eigvals.asDiagonal().toDenseMatrix().cast<Complex>() * eigvecs.adjoint();
    }

    // Compute nuclear norm of result
    Eigen::SelfAdjointEigenSolver<MatrixXcd> final_solver(X);
    if (final_solver.info() == Eigen::Success) {
        VectorXd ev = final_solver.eigenvalues();
        stats.final_nuclear_norm = ev.array().abs().sum();
        stats.recovered_rank = static_cast<size_t>((ev.array().abs() > 1e-10).count());
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    stats.elapsed_seconds = std::chrono::duration<double>(t_end - t_start).count();

    return {X, stats};
}

//==============================================================================
// MatrixCompletion - Alternating Projection Solver
//==============================================================================

std::pair<MatrixXcd, CompletionStats> MatrixCompletion::solve_alternating(
    const std::vector<std::pair<MatrixXcd, double>>& measurements
) {
    auto t_start = std::chrono::high_resolution_clock::now();

    const Eigen::Index d = static_cast<Eigen::Index>(dim_);
    const size_t m = measurements.size();

    CompletionStats stats;

    // Initialize X to the maximally mixed state (ρ = I/d)
    MatrixXcd X = MatrixXcd::Identity(d, d) / static_cast<double>(d);

    // Precompute Tr[P_i P_i] for each measurement (for projection formula)
    // For Pauli strings: Tr[P_i²] = d (since P_i is unitary and Hermitian)
    std::vector<double> P_norm_sq(m);
    for (size_t i = 0; i < m; ++i) {
        const MatrixXcd& P = measurements[i].first;
        Complex tr = (P.array() * P.transpose().array()).sum();
        P_norm_sq[i] = tr.real();
        if (P_norm_sq[i] < 1e-15) P_norm_sq[i] = 1.0;  // safety
    }

    double prev_residual = 1e20;

    for (size_t iter = 0; iter < config_.max_iterations; ++iter) {
        // Step 1: Project onto measurement constraints
        // For each measurement (P_i, m_i):
        //   X ← X + (m_i - Tr[P_i X]) / Tr[P_i²] · P_i
        //
        // This is an additive Kaczmarz-style projection.
        double residual_sq = 0.0;

        for (size_t i = 0; i < m; ++i) {
            const MatrixXcd& P = measurements[i].first;
            double target = measurements[i].second;

            Complex tr = (P.array() * X.transpose().array()).sum();
            double current = tr.real();
            double gap = target - current;
            residual_sq += gap * gap;

            X += (gap / P_norm_sq[i]) * P;
        }

        double residual = std::sqrt(residual_sq);

        // Step 2: Project onto PSD + trace-1 cone
        // Hermitianize
        X = 0.5 * (X + X.adjoint());

        // Eigendecompose, clamp negatives, normalize
        Eigen::SelfAdjointEigenSolver<MatrixXcd> solver(X);
        if (solver.info() == Eigen::Success) {
            VectorXd eigvals = solver.eigenvalues();
            MatrixXcd eigvecs = solver.eigenvectors();

            // Clamp negative eigenvalues
            for (Eigen::Index i = 0; i < d; ++i) {
                if (eigvals(i) < 0.0) eigvals(i) = 0.0;
            }

            // Normalize trace
            double trace = eigvals.sum();
            if (trace > 1e-15) {
                eigvals /= trace;
            } else {
                eigvals.setConstant(1.0 / static_cast<double>(d));
            }

            X = eigvecs * eigvals.asDiagonal().toDenseMatrix().cast<Complex>() * eigvecs.adjoint();
        }

        // Check convergence
        double rel_change = std::abs(prev_residual - residual) / (prev_residual + 1e-15);
        if (config_.verbose && (iter % 20 == 0 || iter < 5)) {
            std::cout << "  AltProj iter " << iter 
                      << ": residual=" << residual 
                      << " rel_change=" << rel_change << std::endl;
        }

        if (rel_change < config_.tolerance && iter > 0) {
            stats.converged = true;
            stats.iterations = iter + 1;
            stats.final_residual = residual;
            stats.convergence_ratio = rel_change;
            break;
        }

        prev_residual = residual;
        stats.iterations = iter + 1;
        stats.final_residual = residual;
        stats.convergence_ratio = rel_change;
    }

    // Compute nuclear norm and rank
    Eigen::SelfAdjointEigenSolver<MatrixXcd> final_solver(X);
    if (final_solver.info() == Eigen::Success) {
        VectorXd ev = final_solver.eigenvalues();
        stats.final_nuclear_norm = ev.array().abs().sum();
        stats.recovered_rank = static_cast<size_t>((ev.array().abs() > 1e-10).count());
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    stats.elapsed_seconds = std::chrono::duration<double>(t_end - t_start).count();

    return {X, stats};
}

//==============================================================================
// QuantumStateTomography - Constructor
//==============================================================================

QuantumStateTomography::QuantumStateTomography(size_t num_qubits, const CompletionConfig& config)
    : num_qubits_(num_qubits)
    , dim_(static_cast<size_t>(1) << num_qubits)
    , completion_(num_qubits, config)
{
}

//==============================================================================
// QuantumStateTomography - select_pauli_subset
//==============================================================================

std::vector<std::string> QuantumStateTomography::select_pauli_subset(double measurement_fraction) {
    auto all_paulis = enumerate_pauli_strings(num_qubits_);
    const size_t total = all_paulis.size();
    size_t budget = static_cast<size_t>(std::ceil(measurement_fraction * static_cast<double>(total)));
    budget = std::min(budget, total);
    budget = std::max(budget, size_t(1));

    if (budget >= total) {
        return all_paulis;
    }

    // Always include the identity string (provides trace constraint)
    std::set<std::string> selected;
    std::string identity(num_qubits_, 'I');
    selected.insert(identity);

    // Importance sampling strategy:
    // 1. Always include all weight-1 operators (3n of them)
    // 2. Include weight-2 operators proportionally
    // 3. Fill remaining budget with random higher-weight operators

    const char paulis[] = {'X', 'Y', 'Z'};

    // Weight-1: Z_i, X_i, Y_i for each qubit
    for (size_t q = 0; q < num_qubits_; ++q) {
        for (char p : paulis) {
            if (selected.size() >= budget) break;
            std::string label(num_qubits_, 'I');
            label[q] = p;
            selected.insert(label);
        }
    }

    // Weight-2 operators (if budget allows)
    if (selected.size() < budget) {
        for (size_t q1 = 0; q1 < num_qubits_; ++q1) {
            for (size_t q2 = q1 + 1; q2 < num_qubits_; ++q2) {
                for (char p1 : paulis) {
                    for (char p2 : paulis) {
                        if (selected.size() >= budget) goto done_weight2;
                        std::string label(num_qubits_, 'I');
                        label[q1] = p1;
                        label[q2] = p2;
                        selected.insert(label);
                    }
                }
            }
        }
    }
    done_weight2:

    // Fill remaining with random selection from all Pauli strings
    if (selected.size() < budget) {
        std::mt19937 rng(42);
        std::shuffle(all_paulis.begin(), all_paulis.end(), rng);
        for (const auto& label : all_paulis) {
            if (selected.size() >= budget) break;
            selected.insert(label);
        }
    }

    return std::vector<std::string>(selected.begin(), selected.end());
}

//==============================================================================
// QuantumStateTomography - compressed_tomography_from_L
//==============================================================================

std::pair<MatrixXcd, CompletionStats> QuantumStateTomography::compressed_tomography_from_L(
    const MatrixXcd& L,
    double measurement_fraction
) {
    // Select which Paulis to measure
    auto pauli_subset = select_pauli_subset(measurement_fraction);

    // "Measure" each Pauli: Tr[P ρ] = Tr[P L L†]
    std::map<std::string, double> measurements;
    for (const auto& label : pauli_subset) {
        MatrixXcd P = pauli_string_matrix(label, num_qubits_);
        double value = pauli_expectation_from_L(L, P);
        measurements[label] = value;
    }

    // Complete the density matrix
    return completion_.complete_from_paulis(measurements);
}

//==============================================================================
// QuantumStateTomography - compressed_tomography (with oracle)
//==============================================================================

std::pair<MatrixXcd, CompletionStats> QuantumStateTomography::compressed_tomography(
    const std::function<double(const std::string&)>& measure_pauli,
    double measurement_fraction
) {
    auto pauli_subset = select_pauli_subset(measurement_fraction);

    std::map<std::string, double> measurements;
    for (const auto& label : pauli_subset) {
        measurements[label] = measure_pauli(label);
    }

    return completion_.complete_from_paulis(measurements);
}

//==============================================================================
// QuantumStateTomography - adaptive_measurements
//==============================================================================

std::vector<std::string> QuantumStateTomography::adaptive_measurements(
    size_t budget,
    const MatrixXcd& current_estimate,
    const std::map<std::string, double>& already_measured
) {
    // Adaptive strategy: choose Pauli operators that have the largest
    // discrepancy between the current estimate and what we've measured,
    // or that probe the most uncertain subspace.
    //
    // Heuristic: for each unmeasured Pauli P, compute the "information gain":
    //   info(P) = |Tr[P ρ_est]| × (1 - weight_penalty)
    //
    // High |Tr[P ρ]| means the operator has a strong signal (not near zero),
    // which means it carries important information about ρ.
    // Weight penalty discourages very high-weight operators (they're noisy).

    std::set<std::string> measured_set;
    for (const auto& [label, _] : already_measured) {
        measured_set.insert(label);
    }

    // Score all unmeasured Pauli strings
    struct Candidate {
        std::string label;
        double score;
    };
    std::vector<Candidate> candidates;

    auto all_paulis = enumerate_pauli_strings(num_qubits_);

    for (const auto& label : all_paulis) {
        if (measured_set.count(label) > 0) continue;

        // Compute weight (number of non-I characters)
        size_t weight = 0;
        for (char c : label) {
            if (c != 'I') ++weight;
        }

        // Compute expected value under current estimate
        MatrixXcd P = pauli_string_matrix(label, num_qubits_);
        Complex tr = (P.array() * current_estimate.transpose().array()).sum();
        double abs_exp = std::abs(tr.real());

        // Weight penalty: prefer lower-weight operators (less noise in practice)
        double weight_penalty = 0.1 * static_cast<double>(weight);
        double score = abs_exp * (1.0 + 0.5 / (1.0 + weight_penalty));

        // Boost score for operators in uncertain subspace
        // Use the variance proxy: |⟨P²⟩ - ⟨P⟩²| = 1 - ⟨P⟩² (for Pauli with eigenvalues ±1)
        double variance_proxy = 1.0 - tr.real() * tr.real();
        score += 0.5 * variance_proxy;

        candidates.push_back({label, score});
    }

    // Sort by score (descending)
    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b) { return a.score > b.score; });

    // Return top-budget candidates
    std::vector<std::string> result;
    result.reserve(budget);
    for (size_t i = 0; i < std::min(budget, candidates.size()); ++i) {
        result.push_back(candidates[i].label);
    }

    return result;
}

//==============================================================================
// QuantumStateTomography - denoise
//==============================================================================

MatrixXcd QuantumStateTomography::denoise(const MatrixXcd& noisy_rho, size_t rank_estimate) const {
    const Eigen::Index d = noisy_rho.rows();

    // Step 1: Hermitianize
    MatrixXcd X = 0.5 * (noisy_rho + noisy_rho.adjoint());

    // Step 2: Eigendecompose
    Eigen::SelfAdjointEigenSolver<MatrixXcd> solver(X);
    if (solver.info() != Eigen::Success) {
        return X;
    }

    VectorXd eigvals = solver.eigenvalues();
    MatrixXcd eigvecs = solver.eigenvectors();

    // Step 3: Determine rank
    size_t rank = rank_estimate;
    if (rank == 0) {
        // Auto-detect rank by finding the gap between signal and noise eigenvalues.
        // Strategy: Use the largest *relative* gap between consecutive eigenvalues
        // (relative to the largest eigenvalue), but only consider gaps where the
        // eigenvalue below is near zero. This distinguishes signal-noise boundary
        // from gaps within signal eigenvalues.
        std::vector<double> sorted_ev(eigvals.data(), eigvals.data() + d);
        std::sort(sorted_ev.begin(), sorted_ev.end(), std::greater<double>());

        double max_ev = sorted_ev[0];
        double threshold = max_ev * 0.05;  // eigenvalues below 5% of max are "noise"

        // Find the first eigenvalue that falls below the threshold
        size_t gap_idx = static_cast<size_t>(d);  // default: full rank
        for (size_t i = 0; i < sorted_ev.size(); ++i) {
            if (sorted_ev[i] < threshold) {
                gap_idx = i;
                break;
            }
        }

        // Fallback: if no eigenvalue is below threshold, use largest gap method
        if (gap_idx == static_cast<size_t>(d) || gap_idx == 0) {
            double max_gap = 0.0;
            gap_idx = 1;
            for (size_t i = 0; i < sorted_ev.size() - 1; ++i) {
                double gap = sorted_ev[i] - sorted_ev[i + 1];
                if (gap > max_gap && sorted_ev[i] > 1e-10) {
                    max_gap = gap;
                    gap_idx = i + 1;
                }
            }
        }
        rank = gap_idx;
    }
    rank = std::max(rank, size_t(1));
    rank = std::min(rank, static_cast<size_t>(d));

    // Step 4: Keep only top-rank eigenvalues
    // Eigenvalues from SelfAdjointEigenSolver are in ascending order
    for (Eigen::Index i = 0; i < d - static_cast<Eigen::Index>(rank); ++i) {
        eigvals(i) = 0.0;
    }

    // Step 5: Clamp remaining negatives, normalize
    for (Eigen::Index i = 0; i < d; ++i) {
        if (eigvals(i) < 0.0) eigvals(i) = 0.0;
    }
    double trace = eigvals.sum();
    if (trace > 1e-15) {
        eigvals /= trace;
    }

    // Reconstruct
    return eigvecs * eigvals.asDiagonal().toDenseMatrix().cast<Complex>() * eigvecs.adjoint();
}

//==============================================================================
// QuantumStateTomography - fidelity
//==============================================================================

double QuantumStateTomography::fidelity(const MatrixXcd& rho, const MatrixXcd& sigma) {
    // F(ρ, σ) = (Tr[√(√ρ σ √ρ)])²
    //
    // Implementation: compute eigendecomposition of ρ to get √ρ,
    // then eigendecompose √ρ σ √ρ to get the trace of its sqrt.

    const Eigen::Index d = rho.rows();

    // Compute √ρ via eigendecomposition
    Eigen::SelfAdjointEigenSolver<MatrixXcd> solver_rho(rho);
    if (solver_rho.info() != Eigen::Success) return 0.0;

    VectorXd eigvals_rho = solver_rho.eigenvalues();
    MatrixXcd eigvecs_rho = solver_rho.eigenvectors();

    // √ρ
    VectorXd sqrt_eigvals(d);
    for (Eigen::Index i = 0; i < d; ++i) {
        sqrt_eigvals(i) = std::sqrt(std::max(eigvals_rho(i), 0.0));
    }
    MatrixXcd sqrt_rho = eigvecs_rho * sqrt_eigvals.asDiagonal().toDenseMatrix().cast<Complex>() * eigvecs_rho.adjoint();

    // M = √ρ σ √ρ
    MatrixXcd M = sqrt_rho * sigma * sqrt_rho;

    // Hermitianize M (it should be Hermitian PSD)
    M = 0.5 * (M + M.adjoint());

    // Eigendecompose M
    Eigen::SelfAdjointEigenSolver<MatrixXcd> solver_M(M);
    if (solver_M.info() != Eigen::Success) return 0.0;

    VectorXd eigvals_M = solver_M.eigenvalues();

    // Tr[√M] = Σ √λ_i(M)
    double trace_sqrt_M = 0.0;
    for (Eigen::Index i = 0; i < d; ++i) {
        trace_sqrt_M += std::sqrt(std::max(eigvals_M(i), 0.0));
    }

    // F = (Tr[√M])²
    double F = trace_sqrt_M * trace_sqrt_M;

    // Clamp to [0, 1] (numerical precision)
    return std::max(0.0, std::min(1.0, F));
}

//==============================================================================
// QuantumStateTomography - trace_distance
//==============================================================================

double QuantumStateTomography::trace_distance(const MatrixXcd& rho, const MatrixXcd& sigma) {
    // D(ρ, σ) = ½ ||ρ - σ||_1 = ½ Σ |λ_i(ρ - σ)|
    MatrixXcd diff = rho - sigma;

    // Hermitianize (ρ - σ should be Hermitian)
    diff = 0.5 * (diff + diff.adjoint());

    Eigen::SelfAdjointEigenSolver<MatrixXcd> solver(diff);
    if (solver.info() != Eigen::Success) {
        return (rho - sigma).norm();  // Frobenius fallback
    }

    VectorXd eigvals = solver.eigenvalues();
    double trace_norm = eigvals.array().abs().sum();

    return 0.5 * trace_norm;
}

}  // namespace qlret

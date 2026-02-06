/**
 * @file cp_decomposition.cpp
 * @brief Canonical Polyadic (CP) Decomposition for LRET Rank Reduction
 * 
 * Phase 2A of Advanced Row-Parallel Optimization.
 * 
 * ALGORITHM DETAILS:
 * 
 * The L matrix (dim × rank) is treated as an order-(n+1) tensor:
 *   T[i₁, i₂, ..., i_n, r] = L[row(i₁,...,i_n), r]
 * where row(i₁,...,i_n) = Σ_k i_k · 2^(n-1-k)
 * 
 * CP decomposition finds factors A_1, ..., A_n, C such that:
 *   T ≈ Σ_{j=1}^R λ_j · a₁_j ⊗ a₂_j ⊗ ... ⊗ a_n_j ⊗ c_j
 * 
 * ALS ITERATION (for mode k):
 *   Given current factors for all modes except k,
 *   the tensor unfolding along mode k gives:
 *     T_(k) ≈ A_k · diag(λ) · (A_{-k})†
 *   where A_{-k} is the Khatri-Rao product of all other factors.
 *   
 *   Optimal A_k = T_(k) · A_{-k} · (V)⁻¹
 *   where V = (A₁†A₁ ⊙ ... ⊙ A_n†A_n ⊙ C†C) with mode k excluded
 *   (⊙ = Hadamard/elementwise product)
 * 
 * COST ANALYSIS:
 * Each ALS iteration: O(n · R² · dim · rank / 2^{n-1})
 * For typical R=8, n=10: much cheaper than Gram eigendecomp O(rank³)
 * when rank is large but R is small (exploiting structure).
 * 
 * @see cp_decomposition.h for API documentation
 * @see ROW_PARALLEL_ADVANCED_OPTIMIZATION_PLAN.md Phase 2A
 */

#include "cp_decomposition.h"
#include "gates_and_noise.h"
#include "simulator.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <random>
#include <numeric>

namespace qlret {

//==============================================================================
// Helper: Index conversion between flat row and qubit indices
//==============================================================================

/**
 * Convert a flat row index to qubit indices.
 * row = i₁·2^(n-1) + i₂·2^(n-2) + ... + i_n
 * Returns vector {i₁, i₂, ..., i_n} where each i_k ∈ {0,1}
 */
static std::vector<size_t> row_to_qubit_indices(size_t row, size_t num_qubits) {
    std::vector<size_t> indices(num_qubits);
    for (size_t k = 0; k < num_qubits; ++k) {
        // Qubit k corresponds to bit position (num_qubits - 1 - k)
        indices[k] = (row >> (num_qubits - 1 - k)) & 1;
    }
    return indices;
}

/**
 * Convert qubit indices back to flat row index.
 */
static size_t qubit_indices_to_row(const std::vector<size_t>& indices) {
    size_t row = 0;
    size_t n = indices.size();
    for (size_t k = 0; k < n; ++k) {
        row |= (indices[k] << (n - 1 - k));
    }
    return row;
}

//==============================================================================
// Helper: Khatri-Rao product and Hadamard product
//==============================================================================

/**
 * Compute the Hadamard (elementwise) product of two matrices.
 * Both must have the same dimensions.
 */
static MatrixXcd hadamard_product(const MatrixXcd& A, const MatrixXcd& B) {
    return A.array() * B.array();
}

/**
 * Compute the "V matrix" for ALS update of mode k:
 *   V = ⊙_{m ≠ k} (A_m† · A_m)
 * where ⊙ denotes Hadamard (elementwise) product.
 * 
 * Each A_m† · A_m is R × R, and the Hadamard product is R × R.
 * This avoids forming the full Khatri-Rao product.
 */
static MatrixXcd compute_V_matrix(
    const CPFactors& factors,
    size_t skip_mode  // mode to skip (qubit index, or num_qubits for rank mode)
) {
    const size_t R = factors.cp_rank();
    const size_t n = factors.num_qubits();

    // Initialize V = ones(R, R) (Hadamard identity)
    MatrixXcd V = MatrixXcd::Ones(R, R);

    // Multiply in the qubit factors (except the skipped mode)
    for (size_t m = 0; m < n; ++m) {
        if (m == skip_mode) continue;
        // A_m is 2 × R
        MatrixXcd gram = factors.qubit_factors[m].adjoint() * factors.qubit_factors[m];
        V = hadamard_product(V, gram);
    }

    // Multiply in the rank factor (unless that's the skipped mode)
    if (skip_mode != n) {
        // rank_factor is original_rank × R
        MatrixXcd gram = factors.rank_factor.adjoint() * factors.rank_factor;
        V = hadamard_product(V, gram);
    }

    return V;
}

/**
 * Compute the "matricized tensor times Khatri-Rao product" (MTTKRP)
 * for mode k.
 * 
 * This is the key operation in ALS: compute T_(k) · Z_(k)
 * where Z_(k) is the Khatri-Rao product of all factors except mode k.
 * 
 * Instead of forming Z explicitly (which can be huge), we compute
 * the MTTKRP elementwise using the tensor structure.
 * 
 * For mode k (a qubit mode with dimension 2):
 *   result[i_k, j] = Σ_{i₁,...,i_{k-1},i_{k+1},...,i_n,r}
 *                     T[i₁,...,i_n,r] · Π_{m≠k} A_m[i_m, j] · C[r, j]
 * 
 * This can be computed by iterating over all tensor entries.
 */
static MatrixXcd compute_MTTKRP_qubit(
    const MatrixXcd& L,
    const CPFactors& factors,
    size_t qubit_mode,
    size_t num_qubits
) {
    const size_t dim = static_cast<size_t>(L.rows());
    const size_t rank = static_cast<size_t>(L.cols());
    const size_t R = factors.cp_rank();

    // Result is 2 × R (dimension of qubit mode × CP rank)
    MatrixXcd result = MatrixXcd::Zero(2, R);

    // Iterate over all entries of L
    for (size_t row = 0; row < dim; ++row) {
        auto qubits = row_to_qubit_indices(row, num_qubits);
        size_t i_k = qubits[qubit_mode];  // This qubit's index (0 or 1)

        for (size_t r = 0; r < rank; ++r) {
            Complex L_val = L(row, r);
            if (std::abs(L_val) < 1e-15) continue;  // Skip zeros

            // For each CP component j, compute the contribution
            for (size_t j = 0; j < R; ++j) {
                // Product of all factors except mode k at index j
                Complex product = L_val;
                for (size_t m = 0; m < num_qubits; ++m) {
                    if (m == qubit_mode) continue;
                    product *= std::conj(factors.qubit_factors[m](qubits[m], j));
                }
                product *= std::conj(factors.rank_factor(r, j));

                result(i_k, j) += product;
            }
        }
    }

    return result;
}

/**
 * MTTKRP for the rank mode.
 * 
 *   result[r, j] = Σ_{i₁,...,i_n}
 *                  T[i₁,...,i_n,r] · Π_m A_m[i_m, j]
 */
static MatrixXcd compute_MTTKRP_rank(
    const MatrixXcd& L,
    const CPFactors& factors,
    size_t num_qubits
) {
    const size_t dim = static_cast<size_t>(L.rows());
    const size_t rank = static_cast<size_t>(L.cols());
    const size_t R = factors.cp_rank();

    // Result is rank × R
    MatrixXcd result = MatrixXcd::Zero(rank, R);

    for (size_t row = 0; row < dim; ++row) {
        auto qubits = row_to_qubit_indices(row, num_qubits);

        for (size_t r = 0; r < rank; ++r) {
            Complex L_val = L(row, r);
            if (std::abs(L_val) < 1e-15) continue;

            for (size_t j = 0; j < R; ++j) {
                Complex product = L_val;
                for (size_t m = 0; m < num_qubits; ++m) {
                    product *= std::conj(factors.qubit_factors[m](qubits[m], j));
                }
                result(r, j) += product;
            }
        }
    }

    return result;
}

//==============================================================================
// Circuit Pattern Detection
//==============================================================================

std::string circuit_pattern_name(CircuitPattern pattern) {
    switch (pattern) {
        case CircuitPattern::QFT:       return "QFT";
        case CircuitPattern::GROVER:    return "Grover";
        case CircuitPattern::PERIODIC:  return "Periodic";
        case CircuitPattern::SEPARABLE: return "Separable";
        default:                        return "Unknown";
    }
}

CircuitPattern detect_circuit_pattern(const QuantumSequence& sequence) {
    if (sequence.operations.empty()) return CircuitPattern::UNKNOWN;

    const size_t nq = sequence.num_qubits;
    size_t total_gates = 0;
    size_t single_qubit_gates = 0;
    size_t two_qubit_gates = 0;
    size_t h_gates = 0;
    size_t controlled_phase_count = 0;  // CZ, RZ-like controlled gates
    size_t x_gates = 0;
    size_t cnot_gates = 0;

    // Count gate types
    for (const auto& op : sequence.operations) {
        if (!std::holds_alternative<GateOp>(op)) continue;
        const auto& gate = std::get<GateOp>(op);
        total_gates++;

        if (gate.qubits.size() == 1) {
            single_qubit_gates++;
            switch (gate.type) {
                case GateType::H:  h_gates++; break;
                case GateType::X:  x_gates++; break;
                case GateType::RZ:
                case GateType::U1:
                    // Single-qubit phase: part of QFT structure
                    controlled_phase_count++;
                    break;
                default: break;
            }
        } else {
            two_qubit_gates++;
            switch (gate.type) {
                case GateType::CZ:
                    controlled_phase_count++;
                    break;
                case GateType::CNOT:
                    cnot_gates++;
                    break;
                default: break;
            }
        }
    }

    if (total_gates == 0) return CircuitPattern::UNKNOWN;

    //--------------------------------------------------------------------------
    // QFT Detection:
    // QFT(n) has n H gates and n(n-1)/2 controlled-phase gates.
    // Ratio of controlled-phase to total ≈ 0.5 for large n.
    // Also: H gates ≈ n, which is significant fraction for small circuits.
    //--------------------------------------------------------------------------
    double phase_ratio = static_cast<double>(controlled_phase_count) / total_gates;
    double h_ratio = static_cast<double>(h_gates) / total_gates;

    // QFT: many H gates + many controlled phases
    if (nq >= 3 && h_gates >= nq / 2 && phase_ratio > 0.3) {
        return CircuitPattern::QFT;
    }

    //--------------------------------------------------------------------------
    // Grover Detection:
    // Grover has: oracle (multi-controlled Z) + diffusion (H⊗n, X⊗n, MCZ, X⊗n, H⊗n)
    // Signature: many X and H gates in blocks, with CNOT/CZ gates
    //--------------------------------------------------------------------------
    double x_ratio = static_cast<double>(x_gates) / total_gates;
    double entangling_ratio = static_cast<double>(two_qubit_gates) / total_gates;

    if (nq >= 3 && h_ratio > 0.15 && x_ratio > 0.15 && entangling_ratio > 0.1) {
        return CircuitPattern::GROVER;
    }

    //--------------------------------------------------------------------------
    // Periodic Detection:
    // Look for repeating gate patterns.
    // If the gate sequence repeats with period P, it's periodic.
    //--------------------------------------------------------------------------
    if (total_gates >= 8) {
        // Check for period P = total_gates / 2 (at least 2 repetitions)
        // Simple heuristic: check if first half matches second half
        std::vector<const GateOp*> gate_list;
        for (const auto& op : sequence.operations) {
            if (std::holds_alternative<GateOp>(op)) {
                gate_list.push_back(&std::get<GateOp>(op));
            }
        }

        size_t half = gate_list.size() / 2;
        if (half >= 4) {
            bool is_periodic = true;
            for (size_t i = 0; i < half && is_periodic; ++i) {
                if (gate_list[i]->type != gate_list[i + half]->type) {
                    is_periodic = false;
                }
                if (gate_list[i]->qubits != gate_list[i + half]->qubits) {
                    is_periodic = false;
                }
            }
            if (is_periodic) return CircuitPattern::PERIODIC;
        }
    }

    //--------------------------------------------------------------------------
    // Separable Detection:
    // >80% single-qubit gates → the state is approximately separable
    //--------------------------------------------------------------------------
    double single_ratio = static_cast<double>(single_qubit_gates) / total_gates;
    if (single_ratio > 0.8) {
        return CircuitPattern::SEPARABLE;
    }

    return CircuitPattern::UNKNOWN;
}

bool should_use_cp(const QuantumSequence& sequence) {
    CircuitPattern pattern = detect_circuit_pattern(sequence);
    return pattern != CircuitPattern::UNKNOWN;
}

//==============================================================================
// Core: CP Decomposition via ALS
//==============================================================================

CPFactors cp_decompose_L(
    const MatrixXcd& L,
    size_t num_qubits,
    const CPConfig& config,
    CPStats* stats
) {
    auto start = std::chrono::steady_clock::now();

    const size_t dim = static_cast<size_t>(L.rows());
    const size_t rank = static_cast<size_t>(L.cols());
    const size_t R = config.target_rank;

    // Validate dimensions
    if (dim != (1ULL << num_qubits)) {
        throw std::invalid_argument(
            "L dimension " + std::to_string(dim) +
            " does not match 2^num_qubits = " + std::to_string(1ULL << num_qubits)
        );
    }

    CPFactors factors;

    //--------------------------------------------------------------------------
    // Step 1: Initialize factors randomly
    //--------------------------------------------------------------------------
    std::mt19937 gen(config.seed);
    std::normal_distribution<double> dist(0.0, 1.0);

    // Initialize qubit factors: each is 2 × R
    factors.qubit_factors.resize(num_qubits);
    for (size_t k = 0; k < num_qubits; ++k) {
        factors.qubit_factors[k] = MatrixXcd::Zero(2, R);
        for (size_t i = 0; i < 2; ++i) {
            for (size_t j = 0; j < R; ++j) {
                factors.qubit_factors[k](i, j) = Complex(dist(gen), dist(gen));
            }
        }
        // Normalize columns
        for (size_t j = 0; j < R; ++j) {
            double norm = factors.qubit_factors[k].col(j).norm();
            if (norm > 1e-15) {
                factors.qubit_factors[k].col(j) /= norm;
            }
        }
    }

    // Initialize rank factor: original_rank × R
    factors.rank_factor = MatrixXcd::Zero(rank, R);
    for (size_t i = 0; i < rank; ++i) {
        for (size_t j = 0; j < R; ++j) {
            factors.rank_factor(i, j) = Complex(dist(gen), dist(gen));
        }
    }
    for (size_t j = 0; j < R; ++j) {
        double norm = factors.rank_factor.col(j).norm();
        if (norm > 1e-15) {
            factors.rank_factor.col(j) /= norm;
        }
    }

    // Initialize weights
    factors.lambdas = VectorXd::Ones(R);

    //--------------------------------------------------------------------------
    // Step 2: Compute initial ||T||² for fit calculation
    //--------------------------------------------------------------------------
    double tensor_norm_sq = L.squaredNorm();

    //--------------------------------------------------------------------------
    // Step 3: ALS Iterations
    //--------------------------------------------------------------------------
    double prev_fit = 0.0;

    for (size_t iter = 0; iter < config.max_iterations; ++iter) {

        //----------------------------------------------------------------------
        // Update each qubit factor A_k
        //----------------------------------------------------------------------
        for (size_t k = 0; k < num_qubits; ++k) {
            // Compute V matrix: Hadamard product of all grams except mode k
            MatrixXcd V = compute_V_matrix(factors, k);

            // Include lambda scaling: V = V ⊙ (λ·λ†)
            for (size_t i = 0; i < R; ++i) {
                for (size_t j = 0; j < R; ++j) {
                    V(i, j) *= factors.lambdas(i) * factors.lambdas(j);
                }
            }

            // Compute MTTKRP for mode k
            MatrixXcd M = compute_MTTKRP_qubit(L, factors, k, num_qubits);

            // Include lambda scaling in MTTKRP
            for (size_t j = 0; j < R; ++j) {
                M.col(j) *= factors.lambdas(j);
            }

            // Solve: A_k_new = M · V^{-1}
            // V is R × R, typically small → use direct solve
            // Add regularization for stability
            MatrixXcd V_reg = V + 1e-12 * MatrixXcd::Identity(R, R);
            
            // Use LDLT for Hermitian positive definite solve
            Eigen::LDLT<MatrixXcd> solver(V_reg);
            if (solver.info() == Eigen::Success) {
                // Solve V · A_k† = M† → A_k = (solve(V, M†))†
                MatrixXcd A_new = solver.solve(M.adjoint()).adjoint();
                factors.qubit_factors[k] = A_new;
            }
            // else: keep current factor (solver failed)

            // DESM: Normalize columns and absorb norms into λ
            if (config.use_desm) {
                for (size_t j = 0; j < R; ++j) {
                    double col_norm = factors.qubit_factors[k].col(j).norm();
                    if (col_norm > 1e-15) {
                        factors.qubit_factors[k].col(j) /= col_norm;
                        factors.lambdas(j) *= col_norm;
                    }
                }
            }
        }

        //----------------------------------------------------------------------
        // Update rank factor C
        //----------------------------------------------------------------------
        {
            MatrixXcd V = compute_V_matrix(factors, num_qubits);  // skip rank mode

            for (size_t i = 0; i < R; ++i) {
                for (size_t j = 0; j < R; ++j) {
                    V(i, j) *= factors.lambdas(i) * factors.lambdas(j);
                }
            }

            MatrixXcd M = compute_MTTKRP_rank(L, factors, num_qubits);

            for (size_t j = 0; j < R; ++j) {
                M.col(j) *= factors.lambdas(j);
            }

            MatrixXcd V_reg = V + 1e-12 * MatrixXcd::Identity(R, R);
            Eigen::LDLT<MatrixXcd> solver(V_reg);
            if (solver.info() == Eigen::Success) {
                MatrixXcd C_new = solver.solve(M.adjoint()).adjoint();
                factors.rank_factor = C_new;
            }

            if (config.use_desm) {
                for (size_t j = 0; j < R; ++j) {
                    double col_norm = factors.rank_factor.col(j).norm();
                    if (col_norm > 1e-15) {
                        factors.rank_factor.col(j) /= col_norm;
                        factors.lambdas(j) *= col_norm;
                    }
                }
            }
        }

        //----------------------------------------------------------------------
        // Compute fit = 1 - ||T - T_approx||² / ||T||²
        //
        // Using the identity:
        //   ||T - T_approx||² = ||T||² - 2·Re(⟨T, T_approx⟩) + ||T_approx||²
        //
        //   ⟨T, T_approx⟩ = Σ_j λ_j · trace(A_j† · MTTKRP_j) (for any mode j)
        //   ||T_approx||² = Σ_{i,j} λ_i·λ_j · Π_m (A_m†·A_m)_{i,j} · (C†·C)_{i,j}
        //----------------------------------------------------------------------

        // Compute ||T_approx||²
        MatrixXcd V_all = MatrixXcd::Ones(R, R);
        for (size_t m = 0; m < num_qubits; ++m) {
            MatrixXcd gram = factors.qubit_factors[m].adjoint() * factors.qubit_factors[m];
            V_all = hadamard_product(V_all, gram);
        }
        {
            MatrixXcd gram = factors.rank_factor.adjoint() * factors.rank_factor;
            V_all = hadamard_product(V_all, gram);
        }

        double approx_norm_sq = 0.0;
        for (size_t i = 0; i < R; ++i) {
            for (size_t j = 0; j < R; ++j) {
                approx_norm_sq += (factors.lambdas(i) * factors.lambdas(j) * V_all(i, j)).real();
            }
        }

        // Compute ⟨T, T_approx⟩ using the last mode's MTTKRP
        // Use rank mode MTTKRP for convenience
        MatrixXcd M_rank = compute_MTTKRP_rank(L, factors, num_qubits);
        double inner_product = 0.0;
        for (size_t j = 0; j < R; ++j) {
            Complex ip = factors.rank_factor.col(j).dot(M_rank.col(j));
            inner_product += (factors.lambdas(j) * ip).real();
        }

        double residual_sq = tensor_norm_sq - 2.0 * inner_product + approx_norm_sq;
        if (residual_sq < 0.0) residual_sq = 0.0;  // Numerical guard

        double fit = 1.0 - std::sqrt(residual_sq) / std::sqrt(tensor_norm_sq + 1e-15);

        if (config.verbose) {
            std::cout << "  [CP-ALS] Iter " << iter + 1
                      << ": fit = " << fit
                      << ", residual = " << std::sqrt(residual_sq)
                      << std::endl;
        }

        // Check convergence
        if (iter > 0 && std::abs(fit - prev_fit) < config.tolerance) {
            if (stats) {
                stats->iterations = iter + 1;
                stats->final_fit = fit;
                stats->relative_error = std::sqrt(residual_sq / (tensor_norm_sq + 1e-15));
                stats->converged = true;
            }
            break;
        }

        prev_fit = fit;

        if (stats) {
            stats->iterations = iter + 1;
            stats->final_fit = fit;
            stats->relative_error = std::sqrt(residual_sq / (tensor_norm_sq + 1e-15));
            stats->converged = false;
        }
    }

    auto end = std::chrono::steady_clock::now();
    if (stats) {
        stats->decompose_time_sec = std::chrono::duration<double>(end - start).count();
    }

    return factors;
}

//==============================================================================
// Core: Reconstruct L from CP Factors
//==============================================================================

MatrixXcd cp_reconstruct_L(const CPFactors& factors) {
    auto start = std::chrono::steady_clock::now();

    const size_t n = factors.num_qubits();
    const size_t R = factors.cp_rank();
    const size_t dim = 1ULL << n;
    const size_t rank = static_cast<size_t>(factors.rank_factor.rows());

    MatrixXcd L = MatrixXcd::Zero(dim, rank);

    // L[row, col] = Σ_j λ_j · Π_k A_k[i_k, j] · C[col, j]
    for (size_t row = 0; row < dim; ++row) {
        auto qubits = row_to_qubit_indices(row, n);

        for (size_t col = 0; col < rank; ++col) {
            Complex val(0.0, 0.0);
            for (size_t j = 0; j < R; ++j) {
                Complex product(factors.lambdas(j), 0.0);
                for (size_t k = 0; k < n; ++k) {
                    product *= factors.qubit_factors[k](qubits[k], j);
                }
                product *= factors.rank_factor(col, j);
                val += product;
            }
            L(row, col) = val;
        }
    }

    return L;
}

//==============================================================================
// Core: CP-based Truncation
//==============================================================================

MatrixXcd truncate_cp(
    const MatrixXcd& L,
    size_t num_qubits,
    const CPConfig& config,
    CPStats* stats
) {
    const size_t current_rank = static_cast<size_t>(L.cols());

    // If rank is already small enough, no truncation needed
    if (current_rank <= 1) return L;

    // Perform CP decomposition
    CPStats local_stats;
    CPFactors factors = cp_decompose_L(L, num_qubits, config, &local_stats);

    if (stats) *stats = local_stats;

    // Sort components by weight |λ| descending
    std::vector<size_t> order(factors.cp_rank());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&factors](size_t a, size_t b) {
                  return std::abs(factors.lambdas(a)) > std::abs(factors.lambdas(b));
              });

    // Reorder factors by weight
    CPFactors sorted_factors;
    sorted_factors.qubit_factors.resize(num_qubits);
    sorted_factors.rank_factor.resize(factors.rank_factor.rows(), factors.cp_rank());
    sorted_factors.lambdas.resize(factors.cp_rank());

    for (size_t j = 0; j < factors.cp_rank(); ++j) {
        size_t src = order[j];
        sorted_factors.lambdas(j) = factors.lambdas(src);
        for (size_t k = 0; k < num_qubits; ++k) {
            if (j == 0) {
                sorted_factors.qubit_factors[k].resize(2, factors.cp_rank());
            }
            sorted_factors.qubit_factors[k].col(j) = factors.qubit_factors[k].col(src);
        }
        sorted_factors.rank_factor.col(j) = factors.rank_factor.col(src);
    }

    // Reconstruct from all components (reconstruction is already at target rank R)
    auto reconstruct_start = std::chrono::steady_clock::now();
    MatrixXcd L_new = cp_reconstruct_L(sorted_factors);
    auto reconstruct_end = std::chrono::steady_clock::now();

    if (stats) {
        stats->reconstruct_time_sec =
            std::chrono::duration<double>(reconstruct_end - reconstruct_start).count();
    }

    // Renormalize to preserve trace
    double new_trace = L_new.squaredNorm();
    if (new_trace > 1e-15) {
        L_new /= std::sqrt(new_trace);
    }

    return L_new;
}

//==============================================================================
// Core: Apply Noise + CP Truncation
//==============================================================================

MatrixXcd apply_noise_cp(
    const MatrixXcd& L,
    const NoiseOp& noise_op,
    size_t num_qubits,
    const CPConfig& config,
    CPStats* stats
) {
    // Step 1: Apply noise using standard Kraus concatenation
    MatrixXcd L_noisy = apply_noise_to_L(L, noise_op, num_qubits);

    // Step 2: CP-based truncation
    return truncate_cp(L_noisy, num_qubits, config, stats);
}

}  // namespace qlret

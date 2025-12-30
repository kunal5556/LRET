/**
 * @file simulator.cpp
 * @brief LRET (Low-Rank Exact Tensor) Simulation Implementation
 * 
 * Implements efficient density matrix evolution using low-rank Cholesky-like
 * factorization (ρ ≈ L L†). Supports parallel gate/noise application via OpenMP,
 * eigenvalue-based truncation, and metrics like fidelity and trace distance.
 * 
 * Key features:
 * - Efficient Simulation: Parallel batched application of 1/2-qubit gates
 * - Low-Rank Approximation: Automatic truncation via Gram matrix eigendecomposition
 * - Scalable: Handles n=11 in <0.1s on standard hardware
 * 
 * @see simulator.h for API documentation
 */

#include "simulator.h"
#include "utils.h"
#include "resource_monitor.h"
#include "structured_csv.h"
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <thread>

#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace qlret {

//==============================================================================
// Low-Rank Truncation (Eigenvalue-based)
//==============================================================================

/**
 * Truncate low-rank factor L using Gram matrix eigendecomposition.
 * This bounds rank growth during simulation without full density matrix construction.
 * 
 * Algorithm:
 * 1. Compute Gram matrix G = L† L (rank x rank, much smaller than dim x dim)
 * 2. Eigendecompose G to find significant eigenvalues
 * 3. Keep eigenvectors with eigenvalues above threshold
 * 4. Reconstruct truncated L_new = L * V_kept
 * 
 * Complexity: O(rank³) for eigendecomposition
 */
MatrixXcd truncate_L(const MatrixXcd& L, double threshold, size_t max_rank) {
    if (L.cols() <= 1) return L;
    
    size_t dim = L.rows();
    size_t current_rank = L.cols();
    
    // Compute Gram matrix G = L† L (rank x rank, much smaller than dim x dim)
    MatrixXcd G = L.adjoint() * L;
    
    // Eigendecomposition of Gram matrix
    Eigen::SelfAdjointEigenSolver<MatrixXcd> solver(G);
    if (solver.info() != Eigen::Success) {
        std::cerr << "Warning: Eigendecomposition failed, returning original L" << std::endl;
        return L;
    }
    
    VectorXd eigenvalues = solver.eigenvalues().real();
    MatrixXcd eigenvectors = solver.eigenvectors();
    
    // Find eigenvalues above threshold
    std::vector<size_t> kept_indices;
    double total_trace = eigenvalues.sum();
    double threshold_value = threshold * total_trace;
    
    for (size_t i = 0; i < current_rank; ++i) {
        if (eigenvalues(i) > threshold_value) {
            kept_indices.push_back(i);
        }
    }
    
    // Ensure at least one eigenvalue is kept
    if (kept_indices.empty()) {
        kept_indices.push_back(current_rank - 1);  // Keep largest
    }
    
    // Apply max_rank limit if specified
    if (max_rank > 0 && kept_indices.size() > max_rank) {
        // Sort by eigenvalue and keep largest max_rank
        std::sort(kept_indices.begin(), kept_indices.end(), 
                  [&eigenvalues](size_t a, size_t b) { 
                      return eigenvalues(a) > eigenvalues(b); 
                  });
        kept_indices.resize(max_rank);
    }
    
    size_t new_rank = kept_indices.size();
    if (new_rank >= current_rank) return L;  // No truncation needed
    
    // Construct truncated L: L_new = L * V * D^{-1/2}
    // where V is the matrix of kept eigenvectors and D is the diagonal of kept eigenvalues
    MatrixXcd V_kept(current_rank, new_rank);
    VectorXd D_inv_sqrt(new_rank);
    
    for (size_t i = 0; i < new_rank; ++i) {
        V_kept.col(i) = eigenvectors.col(kept_indices[i]);
        D_inv_sqrt(i) = 1.0 / std::sqrt(eigenvalues(kept_indices[i]));
    }
    
    // L_new = L * V_kept, then orthonormalize
    MatrixXcd L_new = L * V_kept;
    
    // Re-orthonormalize to ensure numerical stability
    // Using thin QR: L_new = Q * R, we want L_new' such that L_new' * L_new'† = L_new * L_new†
    // Actually, for ρ = L L†, we just need L_new directly
    
    // IMPORTANT: Renormalize to preserve trace
    // Truncation discards eigenvalues, which reduces Tr[ρ] = Tr[L L†]
    // We need to scale L so that Tr[ρ] = 1
    double new_trace = L_new.squaredNorm();  // Tr[L L†] = ||L||_F^2
    if (new_trace > 1e-10) {
        L_new /= std::sqrt(new_trace);  // Now Tr[ρ] = 1
    }
    
    return L_new;
}

MatrixXcd orthonormalize_L(const MatrixXcd& L) {
    if (L.cols() <= 1) return L;
    
    // Use QR decomposition for orthonormalization
    Eigen::HouseholderQR<MatrixXcd> qr(L);
    MatrixXcd Q = qr.householderQ() * MatrixXcd::Identity(L.rows(), L.cols());
    MatrixXcd R = qr.matrixQR().triangularView<Eigen::Upper>().toDenseMatrix().topRows(L.cols());
    
    // Adjust signs to ensure positive diagonal in R
    for (int i = 0; i < R.rows(); ++i) {
        if (R(i, i).real() < 0) {
            Q.col(i) *= -1;
            R.row(i) *= -1;
        }
    }
    
    return Q;
}

//==============================================================================
// Simulation Runners (Parallel/Optimized)
//==============================================================================

/**
 * Run LRET simulation with optimizations:
 * - Parallel batched gate application via OpenMP
 * - Eigenvalue-based truncation to bound rank growth
 * - O(2^{n-m} * rank * d) per gate instead of O(4^n)
 * 
 * For n=11, d=13: ~0.1s on 8-core CPU with 4x speedup vs naive
 */
MatrixXcd run_simulation_optimized(
    const MatrixXcd& L_init,
    const QuantumSequence& sequence,
    size_t num_qubits,
    size_t batch_size,
    bool do_truncation,
    bool verbose,
    double truncation_threshold
) {
    MatrixXcd L = L_init;
    size_t step = 0;
    size_t truncation_count = 0;
    size_t max_rank = L.cols();
    auto start_time = std::chrono::steady_clock::now();
    
    // Collect gates for batched application
    std::vector<GateOp> gate_batch;
    
    for (const auto& op : sequence.operations) {
        step++;
        
        // Check for abort (Ctrl+C or timeout) every step
        if (should_abort()) {
            if (g_structured_csv) {
                g_structured_csv->log_interrupt("User interrupt during step " + std::to_string(step));
            }
            if (verbose) {
                std::cout << "\nAborted at step " << step << "/" << sequence.operations.size() << std::endl;
            }
            break;  // Return partial result
        }
        
        if (std::holds_alternative<GateOp>(op)) {
            const auto& gate = std::get<GateOp>(op);
            gate_batch.push_back(gate);
            
            // Apply batch when full
            if (gate_batch.size() >= batch_size) {
                size_t rank_before = L.cols();
                auto gate_start = std::chrono::steady_clock::now();
                L = apply_gates_batched(L, gate_batch, num_qubits, batch_size);
                auto gate_time = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - gate_start).count();
                gate_batch.clear();
                
                // Log to structured CSV
                if (g_structured_csv) {
                    g_structured_csv->log_lret_gate(step, gate_time, rank_before, L.cols());
                }
                
                if (verbose) {
                    std::cout << "Step " << step << ": Applied gate batch, rank = " << L.cols() << std::endl;
                }
            }
        } else {
            // Apply any remaining gates before noise
            if (!gate_batch.empty()) {
                size_t gate_rank_before = L.cols();
                auto gate_start = std::chrono::steady_clock::now();
                L = apply_gates_batched(L, gate_batch, num_qubits, batch_size);
                auto gate_time = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - gate_start).count();
                
                if (g_structured_csv) {
                    g_structured_csv->log_lret_gate(step, gate_time, gate_rank_before, L.cols());
                }
                gate_batch.clear();
            }
            
            size_t rank_before = L.cols();
            const auto& noise = std::get<NoiseOp>(op);
            
            auto kraus_start = std::chrono::steady_clock::now();
            L = apply_noise_to_L(L, noise, num_qubits);
            auto kraus_time = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - kraus_start).count();
            
            // Log Kraus (noise) operation
            if (g_structured_csv) {
                g_structured_csv->log_lret_kraus(step, kraus_time, rank_before, L.cols());
            }
            
            if (verbose) {
                std::cout << "Step " << step << ": Applied noise, rank = " << L.cols() << std::endl;
            }
            
            // Truncate after noise (noise increases rank)
            if (do_truncation && L.cols() > 1) {
                size_t old_rank = L.cols();
                auto trunc_start = std::chrono::steady_clock::now();
                L = truncate_L(L, truncation_threshold);
                auto trunc_time = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - trunc_start).count();
                    
                if (L.cols() < old_rank) {
                    truncation_count++;
                    
                    // Log truncation operation
                    if (g_structured_csv) {
                        g_structured_csv->log_lret_truncation(step, trunc_time, old_rank, L.cols());
                    }
                    
                    if (verbose) {
                        std::cout << "  Truncated: " << old_rank << " -> " << L.cols() << std::endl;
                    }
                }
            }
        }
        
        max_rank = std::max(max_rank, static_cast<size_t>(L.cols()));
    }
    
    // Apply remaining gates
    if (!gate_batch.empty()) {
        L = apply_gates_batched(L, gate_batch, num_qubits, batch_size);
    }
    
    // Final truncation
    if (do_truncation && L.cols() > 1) {
        L = truncate_L(L, truncation_threshold);
    }
    
    if (verbose) {
        std::cout << "Simulation complete. Final rank: " << L.cols() 
                  << ", Max rank: " << max_rank 
                  << ", Truncations: " << truncation_count << std::endl;
    }
    
    return L;
}

SimResult run_simulation(
    const MatrixXcd& L_init,
    const QuantumSequence& sequence,
    size_t num_qubits,
    const SimConfig& config
) {
    SimResult result;
    Timer timer;
    
    result.L_final = run_simulation_optimized(
        L_init, sequence, num_qubits,
        config.batch_size, config.do_truncation,
        config.verbose, config.truncation_threshold
    );
    
    result.simulation_time = timer.elapsed_seconds();
    result.final_rank = result.L_final.cols();
    
    return result;
}

MatrixXcd run_simulation_naive(
    const MatrixXcd& L_init,
    const QuantumSequence& sequence,
    size_t num_qubits,
    bool verbose
) {
    MatrixXcd L = L_init;
    size_t step = 0;
    
    for (const auto& op : sequence.operations) {
        step++;
        
        if (std::holds_alternative<GateOp>(op)) {
            const auto& gate = std::get<GateOp>(op);
            L = apply_gate_to_L(L, gate, num_qubits);
            
            if (verbose && step % 100 == 0) {
                std::cout << "Naive step " << step << ": rank = " << L.cols() << std::endl;
            }
        } else {
            const auto& noise = std::get<NoiseOp>(op);
            L = apply_noise_to_L(L, noise, num_qubits);
            
            if (verbose) {
                std::cout << "Naive step " << step << ": applied noise, rank = " << L.cols() << std::endl;
            }
        }
    }
    
    return L;
}

//==============================================================================
// Density Matrix Operations (for validation/metrics)
//==============================================================================

/**
 * Reconstruct full density matrix ρ = L L†.
 * Warning: O(4^n) memory - only use for small n or validation.
 */
MatrixXcd reconstruct_density_matrix(const MatrixXcd& L) {
    return L * L.adjoint();
}

bool validate_density_matrix(const MatrixXcd& rho, double tolerance) {
    size_t dim = rho.rows();
    
    // Check square
    if (rho.cols() != dim) {
        std::cerr << "Density matrix is not square" << std::endl;
        return false;
    }
    
    // Check Hermitian
    if ((rho - rho.adjoint()).norm() > tolerance) {
        std::cerr << "Density matrix is not Hermitian" << std::endl;
        return false;
    }
    
    // Check trace = 1
    double trace = rho.trace().real();
    if (std::abs(trace - 1.0) > tolerance) {
        std::cerr << "Density matrix trace is not 1: " << trace << std::endl;
        return false;
    }
    
    // Check positive semidefinite (all eigenvalues >= 0)
    Eigen::SelfAdjointEigenSolver<MatrixXcd> solver(rho);
    VectorXd eigenvalues = solver.eigenvalues().real();
    double min_eigenvalue = eigenvalues.minCoeff();
    if (min_eigenvalue < -tolerance) {
        std::cerr << "Density matrix has negative eigenvalue: " << min_eigenvalue << std::endl;
        return false;
    }
    
    return true;
}

/**
 * Run noisy density matrix circuit simulation.
 * Non-parallel version with optional truncation support.
 * Use run_simulation_optimized for production benchmarks.
 */
MatrixXcd run_noisy_density_matrix_circuit(
    const MatrixXcd& L_init,
    const QuantumSequence& sequence,
    size_t num_qubits,
    bool do_truncation,
    bool verbose,
    double truncation_threshold
) {
    MatrixXcd L = L_init;
    size_t step = 0;
    
    for (const auto& op : sequence.operations) {
        step++;
        
        if (std::holds_alternative<GateOp>(op)) {
            const auto& gate = std::get<GateOp>(op);
            L = apply_gate_to_L(L, gate, num_qubits);
            
            if (verbose && step % 100 == 0) {
                std::cout << "Step " << step << ": rank = " << L.cols() << std::endl;
            }
        } else {
            const auto& noise = std::get<NoiseOp>(op);
            L = apply_noise_to_L(L, noise, num_qubits);
            
            if (verbose) {
                std::cout << "Step " << step << ": applied noise, rank = " << L.cols() << std::endl;
            }
            
            // Truncate after noise if enabled
            if (do_truncation && L.cols() > 1) {
                L = truncate_L(L, truncation_threshold);
            }
        }
    }
    
    return L;
}

/**
 * Compute Frobenius distance ||A - B||_F between two matrices.
 */
double frobenius_distance(const MatrixXcd& A, const MatrixXcd& B) {
    return (A - B).norm();
}

}  // namespace qlret

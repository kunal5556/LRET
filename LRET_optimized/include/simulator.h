#pragma once

#include "types.h"
#include "gates_and_noise.h"

namespace qlret {

//==============================================================================
// Low-Rank Truncation
//==============================================================================

/**
 * @brief Truncate low-rank factor L using Gram matrix eigendecomposition
 * 
 * Given L where ρ ≈ L L†, compute Gram matrix G = L† L and find eigenvalues.
 * Keep only eigenvectors corresponding to eigenvalues above threshold.
 * 
 * @param L Current low-rank factor (dim x rank)
 * @param threshold Eigenvalue threshold for truncation
 * @param max_rank Maximum allowed rank (0 = no limit)
 * @return Truncated L with reduced rank
 */
MatrixXcd truncate_L(const MatrixXcd& L, double threshold, size_t max_rank = 0);

/**
 * @brief Orthonormalize columns of L while preserving ρ = L L†
 * Uses QR decomposition
 * @param L Low-rank factor
 * @return Orthonormalized L
 */
MatrixXcd orthonormalize_L(const MatrixXcd& L);

//==============================================================================
// Simulation Runners
//==============================================================================

/**
 * @brief Run LRET simulation with optimizations (parallelism, truncation)
 * 
 * @param L_init Initial low-rank factor (usually |0><0| state as column vector)
 * @param sequence Quantum sequence to simulate
 * @param num_qubits Number of qubits
 * @param batch_size Batch size for parallel application
 * @param do_truncation Whether to perform eigenvalue-based truncation
 * @param verbose Print progress information
 * @param truncation_threshold Eigenvalue threshold for truncation
 * @return Final low-rank factor L
 */
MatrixXcd run_simulation_optimized(
    const MatrixXcd& L_init,
    const QuantumSequence& sequence,
    size_t num_qubits,
    size_t batch_size = 64,
    bool do_truncation = true,
    bool verbose = false,
    double truncation_threshold = 1e-4
);

/**
 * @brief Run LRET simulation with timing instrumentation
 * 
 * Same as run_simulation_optimized but also returns timing breakdown.
 * 
 * @param L_init Initial low-rank factor
 * @param sequence Quantum sequence
 * @param num_qubits Number of qubits
 * @param config Simulation configuration
 * @param[out] gate_time Total time spent on gate application
 * @param[out] noise_time Total time spent on noise/Kraus operations
 * @param[out] truncation_time Total time spent on SVD truncation
 * @param[out] truncation_count Number of truncation operations performed
 * @return Final low-rank factor L
 */
MatrixXcd run_simulation_with_timing(
    const MatrixXcd& L_init,
    const QuantumSequence& sequence,
    size_t num_qubits,
    const SimConfig& config,
    double& gate_time,
    double& noise_time,
    double& truncation_time,
    size_t& truncation_count
);

/**
 * @brief Run LRET simulation with full configuration
 * @param L_init Initial low-rank factor
 * @param sequence Quantum sequence
 * @param num_qubits Number of qubits
 * @param config Simulation configuration
 * @return SimResult with timing and statistics
 */
SimResult run_simulation(
    const MatrixXcd& L_init,
    const QuantumSequence& sequence,
    size_t num_qubits,
    const SimConfig& config = SimConfig()
);

/**
 * @brief Run naive (sequential) simulation for comparison
 * No parallelism, no truncation optimizations
 * @param L_init Initial low-rank factor
 * @param sequence Quantum sequence
 * @param num_qubits Number of qubits
 * @param verbose Print progress
 * @return Final low-rank factor
 */
MatrixXcd run_simulation_naive(
    const MatrixXcd& L_init,
    const QuantumSequence& sequence,
    size_t num_qubits,
    bool verbose = false
);

//==============================================================================
// Batch Size Heuristics
//==============================================================================

/**
 * @brief Automatically select optimal batch size based on problem size
 * Uses heuristics based on number of qubits and available cores
 * @param num_qubits Number of qubits in simulation
 * @return Recommended batch size
 */
size_t auto_select_batch_size(size_t num_qubits);

/**
 * @brief Get workload classification string
 * @param num_qubits Number of qubits
 * @return "low-workload", "medium-workload", or "high-workload"
 */
std::string get_workload_class(size_t num_qubits);

//==============================================================================
// Density Matrix Operations (for validation)
//==============================================================================

/**
 * @brief Reconstruct full density matrix from low-rank factor
 * Warning: This is O(4^n) and should only be used for small n
 * @param L Low-rank factor
 * @return Full density matrix ρ = L L†
 */
MatrixXcd reconstruct_density_matrix(const MatrixXcd& L);

/**
 * @brief Check if density matrix properties are satisfied
 * Verifies: Hermitian, positive semidefinite, trace = 1
 * @param rho Density matrix
 * @param tolerance Numerical tolerance
 * @return true if valid density matrix
 */
bool validate_density_matrix(const MatrixXcd& rho, double tolerance = 1e-10);

/**
 * @brief Run noisy density matrix circuit simulation (non-parallel)
 * Similar to run_simulation_naive but with truncation support
 * @param L_init Initial low-rank factor
 * @param sequence Quantum sequence
 * @param num_qubits Number of qubits
 * @param do_truncation Whether to truncate
 * @param verbose Print progress
 * @param truncation_threshold Threshold for truncation
 * @return Final low-rank factor
 */
MatrixXcd run_noisy_density_matrix_circuit(
    const MatrixXcd& L_init,
    const QuantumSequence& sequence,
    size_t num_qubits,
    bool do_truncation = true,
    bool verbose = false,
    double truncation_threshold = 1e-4
);

/**
 * @brief Compute Frobenius distance between two matrices
 * @param A First matrix
 * @param B Second matrix
 * @return ||A - B||_F
 */
double frobenius_distance(const MatrixXcd& A, const MatrixXcd& B);

}  // namespace qlret

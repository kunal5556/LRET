#pragma once

#include "types.h"
#include "output_formatter.h"

namespace qlret {

// Check if FDM should run and estimate resources
struct FDMCheckResult {
    bool should_run;
    std::string skip_reason;
    size_t estimated_memory_mb;
    double estimated_time_s;
};

// Get available system memory in MB
size_t get_available_memory_mb();

// Check FDM feasibility based on actual system resources
// If force_run is true, bypasses memory check (for testing limits)
FDMCheckResult check_fdm_feasibility(size_t num_qubits, bool user_enabled, bool force_run = false);

// Run FDM simulation with memory safety (wrapped in try-catch)
FDMResult run_fdm_simulation(
    const QuantumSequence& sequence,
    size_t num_qubits,
    bool verbose = false
);

// Run FDM simulation with custom initial state
FDMResult run_fdm_simulation(
    const QuantumSequence& sequence,
    size_t num_qubits,
    const MatrixXcd& rho_init,
    bool verbose = false
);

// Create initial density matrix |0><0|
MatrixXcd create_zero_density_matrix(size_t num_qubits);

// Convert L factor to density matrix: rho = L L†
MatrixXcd L_to_density_matrix(const MatrixXcd& L);

// Apply gate to density matrix: rho' = U rho U†
MatrixXcd apply_gate_to_rho(
    const MatrixXcd& rho,
    const GateOp& gate,
    size_t num_qubits
);

// Apply noise to density matrix: rho' = sum_i K_i rho K_i†
MatrixXcd apply_noise_to_rho(
    const MatrixXcd& rho,
    const NoiseOp& noise,
    size_t num_qubits
);

// Memory estimate in bytes
size_t estimate_fdm_memory(size_t num_qubits);

// Memory estimate in MB (for display)
size_t estimate_fdm_memory_mb(size_t num_qubits);

}  // namespace qlret

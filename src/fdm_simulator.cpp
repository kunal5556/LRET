#include "fdm_simulator.h"
#include "gates_and_noise.h"
#include "utils.h"
#include <iostream>

namespace qlret {

size_t estimate_fdm_memory(size_t num_qubits) {
    // Full density matrix: 2^n x 2^n complex numbers
    // Each complex = 16 bytes
    size_t dim = 1ULL << num_qubits;
    return dim * dim * 16;
}

size_t estimate_fdm_memory_mb(size_t num_qubits) {
    return estimate_fdm_memory(num_qubits) / (1024 * 1024);
}

FDMCheckResult check_fdm_feasibility(size_t num_qubits, size_t fdm_threshold, bool user_enabled) {
    FDMCheckResult result;
    result.estimated_memory_mb = estimate_fdm_memory_mb(num_qubits);
    result.estimated_time_s = 0.1 * std::pow(4.0, num_qubits - 4);  // Rough estimate
    
    if (!user_enabled) {
        result.should_run = false;
        result.skip_reason = "FDM not enabled (use --fdm flag)";
        return result;
    }
    
    if (num_qubits > fdm_threshold) {
        result.should_run = false;
        result.skip_reason = "Qubits (" + std::to_string(num_qubits) + 
                             ") exceed threshold (" + std::to_string(fdm_threshold) + 
                             ") - would require ~" + std::to_string(result.estimated_memory_mb) + " MB";
        return result;
    }
    
    result.should_run = true;
    result.skip_reason = "";
    return result;
}

MatrixXcd create_zero_density_matrix(size_t num_qubits) {
    size_t dim = 1ULL << num_qubits;
    MatrixXcd rho = MatrixXcd::Zero(dim, dim);
    rho(0, 0) = 1.0;  // |0><0|
    return rho;
}

MatrixXcd L_to_density_matrix(const MatrixXcd& L) {
    return L * L.adjoint();
}

MatrixXcd apply_gate_to_rho(const MatrixXcd& rho, const GateOp& gate, size_t num_qubits) {
    // Get gate matrix
    MatrixXcd U;
    if (gate.qubits.size() == 1) {
        MatrixXcd U_small = get_single_qubit_gate(gate.type, gate.params);
        U = expand_single_gate(U_small, gate.qubits[0], num_qubits);
    } else {
        MatrixXcd U_small = get_two_qubit_gate(gate.type, gate.params);
        U = expand_two_qubit_gate(U_small, gate.qubits[0], gate.qubits[1], num_qubits);
    }
    
    // rho' = U rho U†
    return U * rho * U.adjoint();
}

MatrixXcd apply_noise_to_rho(const MatrixXcd& rho, const NoiseOp& noise, size_t num_qubits) {
    // Get Kraus operators
    auto kraus_ops = get_noise_kraus_operators(noise.type, noise.probability, noise.params);
    
    size_t dim = rho.rows();
    MatrixXcd rho_new = MatrixXcd::Zero(dim, dim);
    
    // rho' = sum_i K_i rho K_i†
    for (const auto& K_small : kraus_ops) {
        MatrixXcd K = expand_single_gate(K_small, noise.qubits[0], num_qubits);
        rho_new += K * rho * K.adjoint();
    }
    
    return rho_new;
}

FDMResult run_fdm_simulation(
    const QuantumSequence& sequence,
    size_t num_qubits,
    bool verbose
) {
    FDMResult result;
    
    Timer timer;
    
    // Initialize |0><0|
    MatrixXcd rho = create_zero_density_matrix(num_qubits);
    
    size_t step = 0;
    for (const auto& op : sequence.operations) {
        step++;
        
        if (std::holds_alternative<GateOp>(op)) {
            const auto& gate = std::get<GateOp>(op);
            rho = apply_gate_to_rho(rho, gate, num_qubits);
            
            if (verbose && step % 50 == 0) {
                std::cout << "FDM step " << step << std::endl;
            }
        } else {
            const auto& noise = std::get<NoiseOp>(op);
            rho = apply_noise_to_rho(rho, noise, num_qubits);
        }
    }
    
    result.was_run = true;
    result.time_seconds = timer.elapsed_seconds();
    result.rho_final = rho;
    result.trace_value = rho.trace().real();
    
    return result;
}

}  // namespace qlret

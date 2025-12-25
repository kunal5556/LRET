#include "fdm_simulator.h"
#include "gates_and_noise.h"
#include "utils.h"
#include "structured_csv.h"
#include <iostream>
#include <new>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/sysinfo.h>
#include <unistd.h>
#endif

namespace qlret {

size_t get_available_memory_mb() {
#ifdef _WIN32
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status)) {
        return static_cast<size_t>(status.ullAvailPhys / (1024 * 1024));
    }
    return 0;  // Unknown
#else
    struct sysinfo info;
    if (sysinfo(&info) == 0) {
        return static_cast<size_t>((info.freeram * info.mem_unit) / (1024 * 1024));
    }
    return 0;  // Unknown
#endif
}

size_t estimate_fdm_memory(size_t num_qubits) {
    // Full density matrix: 2^n x 2^n complex numbers
    // Each complex = 16 bytes
    size_t dim = 1ULL << num_qubits;
    return dim * dim * 16;
}

size_t estimate_fdm_memory_mb(size_t num_qubits) {
    return estimate_fdm_memory(num_qubits) / (1024 * 1024);
}

FDMCheckResult check_fdm_feasibility(size_t num_qubits, bool user_enabled, bool force_run) {
    FDMCheckResult result;
    result.estimated_memory_mb = estimate_fdm_memory_mb(num_qubits);
    result.estimated_time_s = 0.1 * std::pow(4.0, static_cast<double>(num_qubits) - 4);
    
    if (!user_enabled) {
        result.should_run = false;
        result.skip_reason = "FDM not enabled (use --fdm flag)";
        return result;
    }
    
    // Check available memory
    size_t available_mb = get_available_memory_mb();
    
    if (available_mb == 0) {
        // Couldn't determine memory - warn but allow (will catch allocation failure)
        std::cout << "Warning: Could not determine available memory. "
                  << "FDM requires ~" << result.estimated_memory_mb << " MB.\n";
        result.should_run = true;
        result.skip_reason = "";
        return result;
    }
    
    // For "testing to the limit": only check if BASE memory is available
    // No buffer - we'll catch allocation failures at runtime
    // This allows FDM to run even when memory is tight
    if (result.estimated_memory_mb > available_mb) {
        if (force_run) {
            std::cout << "Warning: --fdm-force enabled. Attempting FDM despite insufficient memory.\n"
                      << "  Required: ~" << result.estimated_memory_mb << " MB, Available: " 
                      << available_mb << " MB\n"
                      << "  This may cause allocation failure or system instability.\n";
            result.should_run = true;
            result.skip_reason = "";
            return result;
        }
        
        result.should_run = false;
        result.skip_reason = "Insufficient memory: need ~" + 
                             std::to_string(result.estimated_memory_mb) + " MB, available: " + 
                             std::to_string(available_mb) + " MB. "
                             "Use --fdm-force to attempt anyway.";
        return result;
    }
    
    // Warn if memory is tight (less than 20% buffer)
    double memory_ratio = static_cast<double>(available_mb) / result.estimated_memory_mb;
    if (memory_ratio < 1.2) {
        std::cout << "Warning: Available memory (" << available_mb << " MB) is close to "
                  << "requirement (" << result.estimated_memory_mb << " MB). "
                  << "FDM may fail during computation.\n";
    }
    
    // Warn for very large simulations (time estimate)
    if (result.estimated_memory_mb > 10000) {  // > 10 GB
        std::cout << "Warning: FDM will use ~" << result.estimated_memory_mb / 1024 
                  << " GB of memory. Estimated time: " << result.estimated_time_s << "s\n";
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
    size_t total_gates = 0;
    size_t total_noise = 0;
    
    for (const auto& op : sequence.operations) {
        step++;
        
        if (std::holds_alternative<GateOp>(op)) {
            const auto& gate = std::get<GateOp>(op);
            auto gate_start = std::chrono::steady_clock::now();
            rho = apply_gate_to_rho(rho, gate, num_qubits);
            auto gate_time = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - gate_start).count();
            total_gates++;
            
            // Log gate operation
            if (g_structured_csv) {
                g_structured_csv->log_fdm_gate(step, gate_time);
            }
            
            if (verbose && step % 50 == 0) {
                std::cout << "FDM step " << step << std::endl;
            }
        } else {
            const auto& noise = std::get<NoiseOp>(op);
            auto noise_start = std::chrono::steady_clock::now();
            rho = apply_noise_to_rho(rho, noise, num_qubits);
            auto noise_time = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - noise_start).count();
            total_noise++;
            
            // Log noise operation
            if (g_structured_csv) {
                g_structured_csv->log_fdm_noise(step, noise_time);
            }
        }
    }
    
    result.was_run = true;
    result.time_seconds = timer.elapsed_seconds();
    result.rho_final = rho;
    result.trace_value = rho.trace().real();
    
    return result;
}

}  // namespace qlret

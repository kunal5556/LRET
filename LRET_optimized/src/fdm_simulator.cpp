#include "fdm_simulator.h"
#include "gates_and_noise.h"
#include "utils.h"
#include "structured_csv.h"
#include <iostream>
#include <new>
#include <stdexcept>

#ifdef _WIN32
#include <windows.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#include <mach/mach.h>
#include <unistd.h>
#else
#include <sys/sysinfo.h>
#include <unistd.h>
#endif

namespace qlret {

namespace {

Matrix2cd fdm_pauli_from_index(int idx) {
    Matrix2cd m;
    switch (idx) {
        case 1: // X
            m << 0, 1,
                 1, 0;
            break;
        case 2: { // Y
            Complex i(0.0, 1.0);
            m << 0, -i,
                 i, 0;
            break;
        }
        case 3: // Z
            m << 1, 0,
                 0, -1;
            break;
        default:
            m.setIdentity();
            break;
    }
    return m;
}

Matrix4cd fdm_kron_pauli(int idx0, int idx1) {
    Matrix2cd p0 = fdm_pauli_from_index(idx0);
    Matrix2cd p1 = fdm_pauli_from_index(idx1);
    Matrix4cd result;
    result.setZero();
    for (int r0 = 0; r0 < 2; ++r0) {
        for (int c0 = 0; c0 < 2; ++c0) {
            for (int r1 = 0; r1 < 2; ++r1) {
                for (int c1 = 0; c1 < 2; ++c1) {
                    result(2 * r0 + r1, 2 * c0 + c1) = p0(r0, c0) * p1(r1, c1);
                }
            }
        }
    }
    return result;
}

} // namespace

size_t get_available_memory_mb() {
#ifdef _WIN32
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status)) {
        return static_cast<size_t>(status.ullAvailPhys / (1024 * 1024));
    }
    return 0;  // Unknown
#elif defined(__APPLE__)
    int64_t mem_size = 0;
    size_t len = sizeof(mem_size);
    if (sysctlbyname("hw.memsize", &mem_size, &len, NULL, 0) == 0) {
        // Get available memory using vm_statistics
        mach_msg_type_number_t count = HOST_VM_INFO_COUNT;
        vm_statistics_data_t vmstat;
        if (host_statistics(mach_host_self(), HOST_VM_INFO, (host_info_t)&vmstat, &count) == KERN_SUCCESS) {
            size_t free_mem = vmstat.free_count * vm_page_size;
            return free_mem / (1024 * 1024);
        }
        // Fallback: just return total memory / 2 as rough estimate
        return static_cast<size_t>(mem_size / (2 * 1024 * 1024));
    }
    return 0;
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
    size_t dim = rho.rows();
    MatrixXcd rho_new = MatrixXcd::Zero(dim, dim);

    // Correlated Pauli channel (two-qubit)
    if (noise.type == NoiseType::CORRELATED_PAULI) {
        if (noise.qubits.size() != 2) {
            throw std::invalid_argument("Correlated Pauli noise requires two qubits");
        }
        if (noise.params.size() != 16) {
            throw std::invalid_argument("Correlated Pauli noise expects 16 probability entries");
        }

        std::vector<Matrix4cd> kraus_ops;
        kraus_ops.reserve(16);
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                double p = noise.params[static_cast<size_t>(i * 4 + j)];
                if (p <= 1e-12) continue;
                kraus_ops.push_back(fdm_kron_pauli(i, j) * std::sqrt(p));
            }
        }

        for (const auto& K_small : kraus_ops) {
            MatrixXcd K = expand_two_qubit_gate(K_small, noise.qubits[0], noise.qubits[1], num_qubits);
            rho_new += K * rho * K.adjoint();
        }
        return rho_new;
    }

    // Leakage channels (Phase 4.4) - effective 2-level handling
    if (noise.type == NoiseType::LEAKAGE || noise.type == NoiseType::LEAKAGE_RELAXATION) {
        // Use standard Kraus operators from get_noise_kraus_operators
        auto kraus_ops = get_noise_kraus_operators(noise.type, noise.probability, noise.params);
        for (const auto& K_small : kraus_ops) {
            MatrixXcd K = expand_single_gate(K_small, noise.qubits[0], num_qubits);
            rho_new += K * rho * K.adjoint();
        }
        return rho_new;
    }

    // Standard single-qubit channels
    auto kraus_ops = get_noise_kraus_operators(noise.type, noise.probability, noise.params);
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
    // Default to |0><0| initial state
    MatrixXcd rho_init = create_zero_density_matrix(num_qubits);
    return run_fdm_simulation(sequence, num_qubits, rho_init, verbose);
}

FDMResult run_fdm_simulation(
    const QuantumSequence& sequence,
    size_t num_qubits,
    const MatrixXcd& rho_init,
    bool verbose
) {
    FDMResult result;
    
    Timer timer;
    
    // Use provided initial state
    MatrixXcd rho = rho_init;
    
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
        } else if (std::holds_alternative<NoiseOp>(op)) {
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
        } else if (std::holds_alternative<MeasurementOp>(op)) {
            // Measurement operation (Phase 4.5) - for FDM we trace out or store
            // For now, we do projective measurement sampling (non-deterministic)
            const auto& meas = std::get<MeasurementOp>(op);
            if (meas.collapse_state) {
                std::array<double, 2> probs;
                auto [rho0, rho1] = apply_measurement_to_rho(rho, meas.qubit, num_qubits, probs);
                // For simulation we average both branches (ensemble interpretation)
                rho = probs[0] * rho0 + probs[1] * rho1;
            }
            if (verbose) {
                std::cout << "FDM step " << step << ": Measurement on qubit " << meas.qubit << std::endl;
            }
        } else if (std::holds_alternative<ConditionalOp>(op)) {
            // Conditional operation - in FDM we apply gate weighted by classical bit probability
            // This is an approximation; exact handling requires tracking classical register
            const auto& cond = std::get<ConditionalOp>(op);
            // For now, apply gate unconditionally (conservative approximation)
            rho = apply_gate_to_rho(rho, cond.gate, num_qubits);
            if (verbose) {
                std::cout << "FDM step " << step << ": Conditional gate (applied unconditionally)" << std::endl;
            }
        }
    }
    
    result.was_run = true;
    result.time_seconds = timer.elapsed_seconds();
    result.rho_final = rho;
    result.trace_value = rho.trace().real();
    
    return result;
}

//==============================================================================
// Measurement for FDM (Phase 4.5)
//==============================================================================

std::pair<MatrixXcd, MatrixXcd> apply_measurement_to_rho(
    const MatrixXcd& rho,
    size_t qubit,
    size_t num_qubits,
    std::array<double, 2>& outcome_probs
) {
    size_t dim = rho.rows();
    size_t step = 1ULL << qubit;
    
    // Build projectors P0 and P1 for target qubit
    MatrixXcd P0 = MatrixXcd::Zero(dim, dim);
    MatrixXcd P1 = MatrixXcd::Zero(dim, dim);
    
    for (size_t i = 0; i < dim; ++i) {
        size_t qubit_val = (i >> qubit) & 1;
        if (qubit_val == 0) {
            P0(i, i) = 1.0;
        } else {
            P1(i, i) = 1.0;
        }
    }
    
    // Compute post-measurement states
    MatrixXcd rho0 = P0 * rho * P0;
    MatrixXcd rho1 = P1 * rho * P1;
    
    // Probabilities
    double p0 = rho0.trace().real();
    double p1 = rho1.trace().real();
    double total = p0 + p1;
    
    if (total > 1e-15) {
        outcome_probs[0] = p0 / total;
        outcome_probs[1] = p1 / total;
    } else {
        outcome_probs[0] = 0.5;
        outcome_probs[1] = 0.5;
    }
    
    // Normalize post-measurement states
    if (p0 > 1e-15) {
        rho0 /= p0;
    }
    if (p1 > 1e-15) {
        rho1 /= p1;
    }
    
    return {rho0, rho1};
}

}  // namespace qlret

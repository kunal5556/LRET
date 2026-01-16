/**
 * @file test_lret_fdm_large_scale.cpp
 * @brief Large-scale LRET vs FDM comparison
 * 
 * Parameters: n=13 qubits, d=50, noise=0.1 per layer, epsilon=1e-6
 * Expected: Several minutes runtime, ~1GB memory
 */

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <chrono>
#include "gates_and_noise.h"
#include "simulator.h"
#include "fdm_simulator.h"
#include "utils.h"

using namespace qlret;

void log_to_file(const std::string& message, const std::string& filename = "large_scale_test.log") {
    std::ofstream logfile(filename, std::ios::app);
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    logfile << "[" << std::ctime(&time) << "] " << message << std::endl;
    logfile.close();
    std::cout << message << std::endl;
}

void print_density_matrix_summary(const std::string& name, const MatrixXcd& rho) {
    std::stringstream ss;
    ss << "\n" << name << " Summary:\n";
    ss << std::string(70, '-') << "\n";
    ss << "Dimensions: " << rho.rows() << "×" << rho.cols() << "\n";
    ss << "Trace: " << std::fixed << std::setprecision(10) << rho.trace().real() << "\n";
    
    // Compute purity Tr[ρ²]
    MatrixXcd rho2 = rho * rho;
    double purity = rho2.trace().real();
    ss << "Purity Tr[ρ²]: " << std::fixed << std::setprecision(10) << purity << "\n";
    
    // Compute entropy -Tr[ρ log ρ] (approximate via eigenvalues)
    Eigen::SelfAdjointEigenSolver<MatrixXcd> solver(rho);
    VectorXd eigenvalues = solver.eigenvalues().real();
    double entropy = 0.0;
    for (int i = 0; i < eigenvalues.size(); ++i) {
        if (eigenvalues(i) > 1e-15) {
            entropy -= eigenvalues(i) * std::log(eigenvalues(i));
        }
    }
    ss << "Von Neumann Entropy: " << std::fixed << std::setprecision(10) << entropy << "\n";
    
    // Show some diagonal elements
    ss << "\nDiagonal elements (first 10):\n";
    for (int i = 0; i < std::min(10, (int)rho.rows()); ++i) {
        ss << "  ρ[" << i << "," << i << "] = " << std::fixed << std::setprecision(8) 
           << rho(i,i).real() << "\n";
    }
    
    // Find max off-diagonal element
    double max_offdiag = 0.0;
    int max_i = 0, max_j = 0;
    for (int i = 0; i < rho.rows(); ++i) {
        for (int j = i+1; j < rho.cols(); ++j) {
            double val = std::abs(rho(i,j));
            if (val > max_offdiag) {
                max_offdiag = val;
                max_i = i;
                max_j = j;
            }
        }
    }
    ss << "\nMax off-diagonal: |ρ[" << max_i << "," << max_j << "]| = " 
       << std::scientific << std::setprecision(6) << max_offdiag << "\n";
    
    ss << std::string(70, '-') << "\n";
    
    std::string output = ss.str();
    log_to_file(output);
}

void compare_metrics(const MatrixXcd& rho_lret, const MatrixXcd& rho_fdm) {
    std::stringstream ss;
    ss << "\n" << std::string(70, '=') << "\n";
    ss << "DETAILED COMPARISON METRICS\n";
    ss << std::string(70, '=') << "\n\n";
    
    // Fidelity
    auto start = std::chrono::steady_clock::now();
    double fidelity = compute_fidelity_rho(rho_lret, rho_fdm);
    auto end = std::chrono::steady_clock::now();
    double fid_time = std::chrono::duration<double>(end - start).count();
    
    ss << "1. Fidelity F(ρ_LRET, ρ_FDM):\n";
    ss << "   Value: " << std::fixed << std::setprecision(15) << fidelity << "\n";
    ss << "   Computation time: " << fid_time << " seconds\n";
    ss << "   Status: " << (fidelity > 0.999 ? "✅ PASS" : "❌ FAIL") << "\n\n";
    
    // Trace distance
    start = std::chrono::steady_clock::now();
    double trace_dist = compare_L_matrices_trace(rho_lret, rho_fdm);
    end = std::chrono::steady_clock::now();
    double td_time = std::chrono::duration<double>(end - start).count();
    
    ss << "2. Trace Distance D(ρ_LRET, ρ_FDM):\n";
    ss << "   Value: " << std::scientific << std::setprecision(10) << trace_dist << "\n";
    ss << "   Computation time: " << td_time << " seconds\n";
    ss << "   Status: " << (trace_dist < 0.01 ? "✅ PASS" : "❌ FAIL") << "\n\n";
    
    // Frobenius norm
    start = std::chrono::steady_clock::now();
    MatrixXcd diff = rho_lret - rho_fdm;
    double frobenius = diff.norm();
    end = std::chrono::steady_clock::now();
    double frob_time = std::chrono::duration<double>(end - start).count();
    
    ss << "3. Frobenius Norm ||ρ_LRET - ρ_FDM||:\n";
    ss << "   Value: " << std::scientific << std::setprecision(10) << frobenius << "\n";
    ss << "   Computation time: " << frob_time << " seconds\n";
    ss << "   Status: " << (frobenius < 0.01 ? "✅ PASS" : "❌ FAIL") << "\n\n";
    
    // Trace difference
    double trace_lret = rho_lret.trace().real();
    double trace_fdm = rho_fdm.trace().real();
    double trace_diff = std::abs(trace_lret - trace_fdm);
    
    ss << "4. Trace Preservation:\n";
    ss << "   Tr[ρ_LRET]: " << std::fixed << std::setprecision(10) << trace_lret << "\n";
    ss << "   Tr[ρ_FDM]:  " << std::fixed << std::setprecision(10) << trace_fdm << "\n";
    ss << "   Difference: " << std::scientific << std::setprecision(6) << trace_diff << "\n";
    ss << "   Status: " << (trace_diff < 0.01 ? "✅ PASS" : "❌ FAIL") << "\n\n";
    
    // Purity comparison
    MatrixXcd rho2_lret = rho_lret * rho_lret;
    MatrixXcd rho2_fdm = rho_fdm * rho_fdm;
    double purity_lret = rho2_lret.trace().real();
    double purity_fdm = rho2_fdm.trace().real();
    double purity_diff = std::abs(purity_lret - purity_fdm);
    
    ss << "5. Purity Tr[ρ²]:\n";
    ss << "   LRET: " << std::fixed << std::setprecision(10) << purity_lret << "\n";
    ss << "   FDM:  " << std::fixed << std::setprecision(10) << purity_fdm << "\n";
    ss << "   Difference: " << std::scientific << std::setprecision(6) << purity_diff << "\n";
    ss << "   Status: " << (purity_diff < 0.01 ? "✅ PASS" : "❌ FAIL") << "\n\n";
    
    ss << std::string(70, '=') << "\n";
    
    // Overall verdict
    bool all_pass = (fidelity > 0.999) && (trace_dist < 0.01) && 
                    (frobenius < 0.01) && (trace_diff < 0.01) && (purity_diff < 0.01);
    
    if (all_pass) {
        ss << "✅ OVERALL VERDICT: LRET PERFECTLY MATCHES FDM!\n";
        ss << "   All metrics pass with high precision.\n";
        ss << "   LRET implementation is CORRECT at scale!\n";
    } else if (fidelity > 0.99) {
        ss << "⚠️  OVERALL VERDICT: LRET closely matches FDM\n";
        ss << "   Minor numerical differences present.\n";
    } else {
        ss << "❌ OVERALL VERDICT: Significant differences detected\n";
        ss << "   Further investigation needed.\n";
    }
    
    ss << std::string(70, '=') << "\n\n";
    
    std::string output = ss.str();
    log_to_file(output);
}

int main() {
    // Clear previous log
    std::ofstream logfile("large_scale_test.log", std::ios::trunc);
    logfile.close();
    
    log_to_file(std::string(70, '='));
    log_to_file("LARGE SCALE TEST: LRET vs FDM");
    log_to_file(std::string(70, '='));
    log_to_file("");
    
    // =========================================================================
    // Test Configuration
    // =========================================================================
    size_t num_qubits = 11;
    size_t circuit_depth = 25;
    double noise_prob = 0.1;  // 10% noise per layer
    double truncation_threshold = 1e-6;  // Strict truncation
    uint64_t seed = 42;
    
    log_to_file("Test Configuration:");
    log_to_file("  Qubits: " + std::to_string(num_qubits));
    log_to_file("  Circuit Depth: " + std::to_string(circuit_depth));
    log_to_file("  Noise Probability: " + std::to_string(noise_prob * 100) + "%");
    log_to_file("  LRET Truncation ε: " + std::to_string(truncation_threshold));
    log_to_file("  Mode: Sequential");
    log_to_file("  Random Seed: " + std::to_string(seed));
    log_to_file("");
    
    size_t dim = 1ULL << num_qubits;
    size_t fdm_memory_mb = (dim * dim * sizeof(Complex)) / (1024 * 1024);
    log_to_file("Expected FDM memory: ~" + std::to_string(fdm_memory_mb) + " MB");
    log_to_file("This test may take several minutes...");
    log_to_file("");
    
    // =========================================================================
    // Generate Circuit
    // =========================================================================
    log_to_file("Generating quantum circuit...");
    auto circuit_start = std::chrono::steady_clock::now();
    
    QuantumSequence circuit = generate_quantum_sequences(
        num_qubits,
        circuit_depth,
        true,  // include_noise
        noise_prob,
        seed
    );
    
    auto circuit_end = std::chrono::steady_clock::now();
    double circuit_gen_time = std::chrono::duration<double>(circuit_end - circuit_start).count();
    
    log_to_file("✓ Circuit generated in " + std::to_string(circuit_gen_time) + " seconds");
    log_to_file("  Total operations: " + std::to_string(circuit.operations.size()));
    
    size_t gate_count = 0, noise_count = 0;
    for (const auto& op : circuit.operations) {
        if (std::holds_alternative<GateOp>(op)) gate_count++;
        else if (std::holds_alternative<NoiseOp>(op)) noise_count++;
    }
    log_to_file("  Gates: " + std::to_string(gate_count));
    log_to_file("  Noise ops: " + std::to_string(noise_count));
    log_to_file("");
    
    // =========================================================================
    // LRET Simulation
    // =========================================================================
    log_to_file(std::string(70, '='));
    log_to_file("RUNNING LRET SIMULATION");
    log_to_file(std::string(70, '='));
    log_to_file("");
    
    MatrixXcd L_init = create_zero_state(num_qubits);
    MatrixXcd rho_lret_init = L_init * L_init.adjoint();
    
    log_to_file("Initial state: |0⟩⊗" + std::to_string(num_qubits));
    print_density_matrix_summary("ρ_LRET (initial)", rho_lret_init);
    
    log_to_file("\nStarting LRET simulation...");
    auto lret_start = std::chrono::steady_clock::now();
    
    MatrixXcd L_final = run_simulation_optimized(
        L_init,
        circuit,
        num_qubits,
        1,  // batch_size = 1 (sequential mode)
        true,  // do_truncation
        true,  // verbose
        truncation_threshold
    );
    
    auto lret_end = std::chrono::steady_clock::now();
    double lret_time = std::chrono::duration<double>(lret_end - lret_start).count();
    
    MatrixXcd rho_lret_final = L_final * L_final.adjoint();
    
    log_to_file("\n✓ LRET SIMULATION COMPLETE!");
    log_to_file("  Total time: " + std::to_string(lret_time) + " seconds (" + 
                std::to_string(lret_time / 60.0) + " minutes)");
    log_to_file("  Final rank: " + std::to_string(L_final.cols()));
    size_t lret_memory_mb = (L_final.rows() * L_final.cols() * sizeof(Complex)) / (1024 * 1024);
    log_to_file("  Memory usage: ~" + std::to_string(lret_memory_mb) + " MB");
    log_to_file("");
    
    print_density_matrix_summary("ρ_LRET (final)", rho_lret_final);
    
    // =========================================================================
    // FDM Simulation
    // =========================================================================
    log_to_file("\n" + std::string(70, '='));
    log_to_file("RUNNING FDM SIMULATION");
    log_to_file(std::string(70, '='));
    log_to_file("");
    
    MatrixXcd rho_fdm_init = create_zero_density_matrix(num_qubits);
    
    log_to_file("Initial state: |0⟩⊗" + std::to_string(num_qubits));
    print_density_matrix_summary("ρ_FDM (initial)", rho_fdm_init);
    
    log_to_file("\nStarting FDM simulation...");
    log_to_file("WARNING: This may take several minutes and use ~" + 
                std::to_string(fdm_memory_mb) + " MB memory");
    
    auto fdm_start = std::chrono::steady_clock::now();
    
    FDMResult fdm_result = run_fdm_simulation(circuit, num_qubits, true);
    
    auto fdm_end = std::chrono::steady_clock::now();
    double fdm_time = std::chrono::duration<double>(fdm_end - fdm_start).count();
    
    MatrixXcd rho_fdm_final = fdm_result.rho_final;
    
    log_to_file("\n✓ FDM SIMULATION COMPLETE!");
    log_to_file("  Total time: " + std::to_string(fdm_time) + " seconds (" + 
                std::to_string(fdm_time / 60.0) + " minutes)");
    log_to_file("  Memory usage: ~" + std::to_string(fdm_memory_mb) + " MB");
    log_to_file("");
    
    print_density_matrix_summary("ρ_FDM (final)", rho_fdm_final);
    
    // =========================================================================
    // Comparison
    // =========================================================================
    log_to_file("\n" + std::string(70, '='));
    log_to_file("COMPARING RESULTS");
    log_to_file(std::string(70, '='));
    log_to_file("");
    
    compare_metrics(rho_lret_final, rho_fdm_final);
    
    // =========================================================================
    // Performance Summary
    // =========================================================================
    log_to_file("\n" + std::string(70, '='));
    log_to_file("PERFORMANCE SUMMARY");
    log_to_file(std::string(70, '='));
    log_to_file("");
    
    log_to_file("LRET Time:  " + std::to_string(lret_time) + " seconds");
    log_to_file("FDM Time:   " + std::to_string(fdm_time) + " seconds");
    
    if (lret_time < fdm_time) {
        double speedup = fdm_time / lret_time;
        log_to_file("LRET Speedup: " + std::to_string(speedup) + "×");
    } else {
        double slowdown = lret_time / fdm_time;
        log_to_file("LRET Slowdown: " + std::to_string(slowdown) + "×");
    }
    
    double memory_savings = (double)fdm_memory_mb / lret_memory_mb;
    log_to_file("\nLRET Memory: " + std::to_string(lret_memory_mb) + " MB");
    log_to_file("FDM Memory:  " + std::to_string(fdm_memory_mb) + " MB");
    log_to_file("Memory Savings: " + std::to_string(memory_savings) + "×");
    
    log_to_file("\n" + std::string(70, '='));
    log_to_file("TEST COMPLETE - Check large_scale_test.log for full details");
    log_to_file(std::string(70, '='));
    
    return 0;
}

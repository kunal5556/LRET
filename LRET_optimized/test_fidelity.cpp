// test_fidelity.cpp - Debug fidelity calculation between LRET and FDM
#include <iostream>
#include <iomanip>
#include "gates_and_noise.h"
#include "utils.h"
#include "fdm_simulator.h"
#include "simulator.h"

using namespace qlret;

void print_matrix(const std::string& name, const MatrixXcd& M, size_t max_dim = 8) {
    std::cout << name << " (" << M.rows() << "x" << M.cols() << "):\n";
    size_t rows = std::min((size_t)M.rows(), max_dim);
    size_t cols = std::min((size_t)M.cols(), max_dim);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            std::cout << std::setw(12) << std::setprecision(4) << M(i, j) << " ";
        }
        if (cols < (size_t)M.cols()) std::cout << "...";
        std::cout << "\n";
    }
    if (rows < (size_t)M.rows()) std::cout << "...\n";
    std::cout << "\n";
}

double check_trace(const std::string& name, const MatrixXcd& L) {
    double trace = L.squaredNorm();  // ||L||_F^2 = Tr[L L†]
    std::cout << "   " << name << " trace (||L||_F^2): " << trace << "\n";
    return trace;
}

int main() {
    std::cout << "=== Fidelity Debug Test ===\n\n";
    
    try {
        size_t num_qubits = 2;
        
        // Create identical initial states
        std::cout << "1. Creating initial states (n=" << num_qubits << " qubits)...\n";
        MatrixXcd L = create_zero_state(num_qubits);
        MatrixXcd rho_fdm = create_zero_density_matrix(num_qubits);
        
        check_trace("L initial", L);
        
        MatrixXcd rho_lret = L * L.adjoint();
        
        std::cout << "   rho_lret trace: " << rho_lret.trace().real() << "\n";
        std::cout << "   rho_fdm trace: " << rho_fdm.trace().real() << "\n";
        
        double fid0 = compute_fidelity_rho(rho_lret, rho_fdm);
        std::cout << "   Initial fidelity: " << fid0 << " (should be 1.0)\n\n";
        
        // Test 1: Apply single H gate
        std::cout << "2. Applying H gate to qubit 0...\n";
        GateOp h_gate(GateType::H, 0);
        
        L = apply_gate_to_L(L, h_gate, num_qubits);
        rho_fdm = apply_gate_to_rho(rho_fdm, h_gate, num_qubits);
        
        check_trace("L after H", L);
        
        rho_lret = L * L.adjoint();
        
        std::cout << "   rho_lret trace: " << rho_lret.trace().real() << "\n";
        std::cout << "   rho_fdm trace: " << rho_fdm.trace().real() << "\n";
        
        double fid1 = compute_fidelity_rho(rho_lret, rho_fdm);
        std::cout << "   Fidelity after H: " << fid1 << " (should be 1.0)\n";
        
        // Print matrices to compare
        print_matrix("rho_lret after H", rho_lret);
        print_matrix("rho_fdm after H", rho_fdm);
        
        // Test 2: Apply CNOT gate
        std::cout << "3. Applying CNOT(0,1) gate...\n";
        GateOp cnot_gate(GateType::CNOT, 0, 1);
        
        L = apply_gate_to_L(L, cnot_gate, num_qubits);
        rho_fdm = apply_gate_to_rho(rho_fdm, cnot_gate, num_qubits);
        
        check_trace("L after CNOT", L);
        
        rho_lret = L * L.adjoint();
        
        double fid2 = compute_fidelity_rho(rho_lret, rho_fdm);
        std::cout << "   Fidelity after CNOT: " << fid2 << " (should be 1.0)\n";
        
        print_matrix("rho_lret after CNOT", rho_lret);
        print_matrix("rho_fdm after CNOT", rho_fdm);
        
        // Test 3: Apply depolarizing noise
        std::cout << "4. Applying depolarizing noise (p=0.01) to qubit 0...\n";
        NoiseOp noise(NoiseType::DEPOLARIZING, 0, 0.01);
        
        L = apply_noise_to_L(L, noise, num_qubits);
        rho_fdm = apply_noise_to_rho(rho_fdm, noise, num_qubits);
        
        check_trace("L after noise", L);
        
        std::cout << "   L dims after noise: " << L.rows() << "x" << L.cols() << "\n";
        
        rho_lret = L * L.adjoint();
        
        std::cout << "   rho_lret trace: " << rho_lret.trace().real() << "\n";
        std::cout << "   rho_fdm trace: " << rho_fdm.trace().real() << "\n";
        
        double fid3 = compute_fidelity_rho(rho_lret, rho_fdm);
        std::cout << "   Fidelity after noise: " << fid3 << " (should be ~1.0)\n";
        
        print_matrix("rho_lret after noise", rho_lret);
        print_matrix("rho_fdm after noise", rho_fdm);
        
        // Compute difference
        MatrixXcd diff = rho_lret - rho_fdm;
        double frobenius_diff = diff.norm();
        std::cout << "   Frobenius norm of difference: " << frobenius_diff << "\n\n";
        
        // Test 4: Test truncation trace preservation
        std::cout << "5. Testing truncation trace preservation...\n";
        
        // Create a state with higher rank
        MatrixXcd L_mixed = create_random_mixed_state(3, 4, 42);  // 3 qubits, rank 4
        double trace_before = L_mixed.squaredNorm();
        std::cout << "   Before truncation: rank=" << L_mixed.cols() 
                  << ", trace=" << trace_before << "\n";
        
        MatrixXcd L_truncated = truncate_L(L_mixed, 0.1);  // Aggressive truncation
        double trace_after = L_truncated.squaredNorm();
        std::cout << "   After truncation: rank=" << L_truncated.cols() 
                  << ", trace=" << trace_after << "\n";
        
        std::cout << "   Trace preserved: " << (std::abs(trace_after - 1.0) < 0.01 ? "YES" : "NO") << "\n\n";
        
        // Test 5: Full simulation comparison
        std::cout << "6. Full simulation test (n=3, d=5)...\n";
        num_qubits = 3;
        size_t depth = 5;
        
        QuantumSequence seq = generate_quantum_sequences(num_qubits, depth, true, 0.01, 42);
        std::cout << "   Generated " << seq.operations.size() << " operations\n";
        
        // Run LRET
        L = create_zero_state(num_qubits);
        
        MatrixXcd L_final = run_simulation_optimized(L, seq, num_qubits, 1, true, false, 1e-6);
        
        double lret_trace = L_final.squaredNorm();
        std::cout << "   LRET final rank: " << L_final.cols() << "\n";
        std::cout << "   LRET ||L||_F^2 (trace): " << lret_trace << "\n";
        
        rho_lret = L_final * L_final.adjoint();
        std::cout << "   rho_lret trace: " << rho_lret.trace().real() << "\n";
        
        // Run FDM
        FDMResult fdm_result = run_fdm_simulation(seq, num_qubits, false);
        
        std::cout << "   rho_fdm trace: " << fdm_result.rho_final.trace().real() << "\n";
        
        double fid_final = compute_fidelity_rho(rho_lret, fdm_result.rho_final);
        std::cout << "   Final fidelity: " << fid_final << "\n";
        
        if (fid_final < 0) {
            std::cout << "   WARNING: Fidelity returned -1 (computation failed)\n";
        } else if (fid_final < 0.9) {
            std::cout << "   WARNING: Fidelity is low. Expected close to 1.0\n";
        } else {
            std::cout << "   SUCCESS: Fidelity is good!\n";
        }
        
        frobenius_diff = (rho_lret - fdm_result.rho_final).norm();
        std::cout << "   Frobenius norm of difference: " << frobenius_diff << "\n";
        
        double trace_dist = compute_trace_distance_rho(rho_lret, fdm_result.rho_final);
        std::cout << "   Trace distance: " << trace_dist << "\n";
        
        // Print diagonal elements
        std::cout << "\n   Diagonal elements comparison:\n";
        for (size_t i = 0; i < std::min((size_t)8, (size_t)rho_lret.rows()); ++i) {
            std::cout << "   [" << i << "] LRET: " << std::setprecision(6) << rho_lret(i,i).real() 
                      << " vs FDM: " << fdm_result.rho_final(i,i).real()
                      << " diff: " << std::abs(rho_lret(i,i) - fdm_result.rho_final(i,i)) << "\n";
        }
        
        std::cout << "\n=== Test Complete ===\n";
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}

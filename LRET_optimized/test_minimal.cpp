// test_minimal.cpp - Minimal test to debug LRET vs FDM mismatch
#include <iostream>
#include <iomanip>
#include "gates_and_noise.h"
#include "utils.h"
#include "fdm_simulator.h"
#include "simulator.h"

using namespace qlret;

void print_state(const std::string& name, const MatrixXcd& rho) {
    std::cout << name << ":\n";
    std::cout << std::fixed << std::setprecision(4);
    for (int i = 0; i < std::min((int)rho.rows(), 8); ++i) {
        for (int j = 0; j < std::min((int)rho.cols(), 8); ++j) {
            double re = rho(i, j).real();
            double im = rho(i, j).imag();
            if (std::abs(im) < 1e-10) {
                std::cout << std::setw(8) << re << " ";
            } else {
                std::cout << "(" << re << "," << im << ") ";
            }
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

void print_L(const std::string& name, const MatrixXcd& L) {
    std::cout << name << " (" << L.rows() << "x" << L.cols() << "):\n";
    for (int i = 0; i < (int)L.rows(); ++i) {
        std::cout << "  [" << i << "] = ";
        for (int j = 0; j < (int)L.cols(); ++j) {
            double re = L(i, j).real();
            double im = L(i, j).imag();
            if (std::abs(im) < 1e-10) {
                std::cout << re << " ";
            } else {
                std::cout << "(" << re << "," << im << ") ";
            }
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

int main() {
    std::cout << "=== Minimal LRET vs FDM Test ===\n\n";
    
    size_t n = 2;  // 2 qubits
    
    // Create initial states
    MatrixXcd L = create_zero_state(n);
    MatrixXcd rho_fdm = create_zero_density_matrix(n);
    
    print_L("Initial L", L);
    
    MatrixXcd rho_lret = L * L.adjoint();
    print_state("Initial rho_lret", rho_lret);
    print_state("Initial rho_fdm", rho_fdm);
    
    double fid = compute_fidelity_rho(rho_lret, rho_fdm);
    double frob = (rho_lret - rho_fdm).norm();
    std::cout << "Initial: Fidelity=" << fid << ", Frobenius=" << frob << " (should be 1.0, 0.0)\n\n";
    
    // Test 1: Apply H to qubit 0
    std::cout << "=== Test 1: H(0) ===\n";
    GateOp h0(GateType::H, 0);
    
    L = apply_gate_to_L(L, h0, n);
    rho_fdm = apply_gate_to_rho(rho_fdm, h0, n);
    
    print_L("L after H(0)", L);
    
    rho_lret = L * L.adjoint();
    print_state("rho_lret after H(0)", rho_lret);
    print_state("rho_fdm after H(0)", rho_fdm);
    
    fid = compute_fidelity_rho(rho_lret, rho_fdm);
    frob = (rho_lret - rho_fdm).norm();
    std::cout << "After H(0): Fidelity=" << fid << ", Frobenius=" << frob << " (should be 1.0, 0.0)\n\n";
    
    // Test 2: Reset and apply H to qubit 1
    std::cout << "=== Test 2: H(1) from |00> ===\n";
    L = create_zero_state(n);
    rho_fdm = create_zero_density_matrix(n);
    
    GateOp h1(GateType::H, 1);
    L = apply_gate_to_L(L, h1, n);
    rho_fdm = apply_gate_to_rho(rho_fdm, h1, n);
    
    print_L("L after H(1)", L);
    
    rho_lret = L * L.adjoint();
    print_state("rho_lret after H(1)", rho_lret);
    print_state("rho_fdm after H(1)", rho_fdm);
    
    fid = compute_fidelity_rho(rho_lret, rho_fdm);
    frob = (rho_lret - rho_fdm).norm();
    std::cout << "After H(1): Fidelity=" << fid << ", Frobenius=" << frob << " (should be 1.0, 0.0)\n\n";
    
    // Test 3: Apply CNOT(0,1) from |+0> state
    std::cout << "=== Test 3: CNOT(0,1) from H(0)|00> ===\n";
    L = create_zero_state(n);
    rho_fdm = create_zero_density_matrix(n);
    
    // Apply H(0) first
    L = apply_gate_to_L(L, h0, n);
    rho_fdm = apply_gate_to_rho(rho_fdm, h0, n);
    
    std::cout << "After H(0):\n";
    print_L("L", L);
    
    // Apply CNOT(0,1)
    GateOp cnot01(GateType::CNOT, 0, 1);
    L = apply_gate_to_L(L, cnot01, n);
    rho_fdm = apply_gate_to_rho(rho_fdm, cnot01, n);
    
    print_L("L after CNOT(0,1)", L);
    
    rho_lret = L * L.adjoint();
    print_state("rho_lret after CNOT(0,1)", rho_lret);
    print_state("rho_fdm after CNOT(0,1)", rho_fdm);
    
    fid = compute_fidelity_rho(rho_lret, rho_fdm);
    frob = (rho_lret - rho_fdm).norm();
    std::cout << "After CNOT(0,1): Fidelity=" << fid << ", Frobenius=" << frob << " (should be 1.0, 0.0)\n\n";
    
    // Test 4: Full sequence test
    std::cout << "=== Test 4: Full Sequence with seed ===\n";
    size_t num_qubits = 3;
    size_t depth = 3;
    
    // Generate a fixed sequence with no noise
    QuantumSequence seq = generate_quantum_sequences(num_qubits, depth, true, 0.0, 12345);
    
    std::cout << "Circuit has " << seq.operations.size() << " operations:\n";
    int op_idx = 0;
    for (const auto& op : seq.operations) {
        if (std::holds_alternative<GateOp>(op)) {
            const auto& gate = std::get<GateOp>(op);
            std::cout << "  [" << op_idx << "] Gate ";
            if (gate.qubits.size() == 1) {
                std::cout << gate_type_to_string(gate.type) << "(" << gate.qubits[0] << ")";
            } else {
                std::cout << gate_type_to_string(gate.type) << "(" << gate.qubits[0] << "," << gate.qubits[1] << ")";
            }
            std::cout << "\n";
        } else {
            const auto& noise = std::get<NoiseOp>(op);
            std::cout << "  [" << op_idx << "] Noise on qubit " << noise.qubits[0] << " p=" << noise.probability << "\n";
        }
        op_idx++;
    }
    std::cout << "\n";
    
    // Apply to LRET
    L = create_zero_state(num_qubits);
    for (const auto& op : seq.operations) {
        if (std::holds_alternative<GateOp>(op)) {
            const auto& gate = std::get<GateOp>(op);
            L = apply_gate_to_L(L, gate, num_qubits);
        } else {
            const auto& noise = std::get<NoiseOp>(op);
            L = apply_noise_to_L(L, noise, num_qubits);
        }
    }
    
    // Apply to FDM
    rho_fdm = create_zero_density_matrix(num_qubits);
    for (const auto& op : seq.operations) {
        if (std::holds_alternative<GateOp>(op)) {
            const auto& gate = std::get<GateOp>(op);
            rho_fdm = apply_gate_to_rho(rho_fdm, gate, num_qubits);
        } else {
            const auto& noise = std::get<NoiseOp>(op);
            rho_fdm = apply_noise_to_rho(rho_fdm, noise, num_qubits);
        }
    }
    
    rho_lret = L * L.adjoint();
    
    std::cout << "LRET trace: " << rho_lret.trace().real() << "\n";
    std::cout << "FDM trace: " << rho_fdm.trace().real() << "\n";
    
    fid = compute_fidelity_rho(rho_lret, rho_fdm);
    frob = (rho_lret - rho_fdm).norm();
    double trace_dist = compute_trace_distance_rho(rho_lret, rho_fdm);
    
    std::cout << "Fidelity: " << fid << " (should be 1.0)\n";
    std::cout << "Frobenius: " << frob << " (should be 0.0)\n";
    std::cout << "Trace Distance: " << trace_dist << " (should be 0.0)\n";
    
    // Print diagonal elements
    std::cout << "\nDiagonal comparison:\n";
    for (size_t i = 0; i < std::min((size_t)8, (size_t)rho_lret.rows()); ++i) {
        std::cout << "  [" << i << "] LRET=" << std::setprecision(6) << rho_lret(i,i).real() 
                  << " FDM=" << rho_fdm(i,i).real() << "\n";
    }
    
    return 0;
}

/**
 * QuantumLRET-Sim - Batch Size Heuristic Demo
 * 
 * This demo tests the auto_select_batch_size heuristic across
 * different qubit counts to show workload classification and
 * batch size selection.
 */

#include "simulator.h"
#include "utils.h"

#include <iostream>
#include <iomanip>

using namespace qlret;

int main() {
    std::cout << "QuantumLRET-Sim Batch Heuristic Demo" << std::endl;
    std::cout << std::string(50, '=') << std::endl;
    std::cout << std::endl;
    
    // Test batch size selection for various qubit counts
    std::vector<size_t> qubit_counts = {6, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20};
    
    for (size_t n : qubit_counts) {
        size_t batch_size = auto_select_batch_size(n);
        std::string workload = get_workload_class(n);
        
        std::cout << "INFO: n=" << std::setw(2) << n << " " << workload 
                  << ", batch_size=" << batch_size << std::endl;
        std::cout << "for " << n << " qubits number of batches are " 
                  << batch_size << std::endl;
    }
    
    std::cout << std::endl;
    std::cout << std::string(50, '=') << std::endl;
    std::cout << "Batch heuristic demo complete." << std::endl;
    
    // Additional demo: Show Hilbert space dimensions
    std::cout << std::endl;
    std::cout << "Hilbert Space Dimensions:" << std::endl;
    std::cout << std::string(30, '-') << std::endl;
    
    for (size_t n : {4, 6, 8, 10, 12, 14, 16, 18, 20}) {
        size_t dim = 1ULL << n;
        double memory_bytes = static_cast<double>(dim * dim) * 16.0;  // complex<double> = 16 bytes
        
        std::cout << "n=" << std::setw(2) << n << ": dim=2^" << n << "=" << std::setw(10) << dim;
        
        if (memory_bytes < 1024) {
            std::cout << " (full ρ: " << memory_bytes << " B)" << std::endl;
        } else if (memory_bytes < 1024 * 1024) {
            std::cout << " (full ρ: " << std::fixed << std::setprecision(1) 
                      << memory_bytes / 1024 << " KB)" << std::endl;
        } else if (memory_bytes < 1024 * 1024 * 1024) {
            std::cout << " (full ρ: " << std::fixed << std::setprecision(1) 
                      << memory_bytes / (1024 * 1024) << " MB)" << std::endl;
        } else if (memory_bytes < 1024.0 * 1024 * 1024 * 1024) {
            std::cout << " (full ρ: " << std::fixed << std::setprecision(1) 
                      << memory_bytes / (1024 * 1024 * 1024) << " GB)" << std::endl;
        } else {
            std::cout << " (full ρ: " << std::fixed << std::setprecision(1) 
                      << memory_bytes / (1024.0 * 1024 * 1024 * 1024) << " TB)" << std::endl;
        }
    }
    
    std::cout << std::endl;
    std::cout << "Note: LRET avoids full density matrix storage!" << std::endl;
    std::cout << "Low-rank L matrix: dim × rank (typically rank << dim)" << std::endl;
    
    // Demo: Generate and analyze a small circuit
    std::cout << std::endl;
    std::cout << std::string(50, '=') << std::endl;
    std::cout << "Small Circuit Analysis (n=4, d=5)" << std::endl;
    std::cout << std::string(50, '-') << std::endl;
    
    size_t n = 4;
    size_t d = 5;
    auto seq = generate_quantum_sequences(n, d, true, 0.01);
    
    std::cout << "Generated " << seq.operations.size() << " operations" << std::endl;
    std::cout << "Total noise probability: " << seq.total_noise_probability << std::endl;
    
    // Count gates and noise
    size_t gate_count = 0;
    size_t noise_count = 0;
    for (const auto& op : seq.operations) {
        if (std::holds_alternative<GateOp>(op)) {
            gate_count++;
        } else {
            noise_count++;
        }
    }
    std::cout << "Gates: " << gate_count << ", Noise ops: " << noise_count << std::endl;
    
    // Print circuit
    std::cout << std::endl << "Circuit diagram:" << std::endl;
    print_circuit_diagram(n, seq, 80);
    
    // Run simulation
    std::cout << std::endl << "Running simulation..." << std::endl;
    MatrixXcd L_init = create_zero_state(n);
    
    Timer timer;
    MatrixXcd L_final = run_simulation_optimized(
        L_init, seq, n, 
        auto_select_batch_size(n), 
        true, false, 1e-4
    );
    double elapsed = timer.elapsed_seconds();
    
    std::cout << "Time: " << std::fixed << std::setprecision(4) << elapsed << " s" << std::endl;
    std::cout << "Final rank: " << L_final.cols() << std::endl;
    std::cout << "Purity: " << compute_purity(L_final) << std::endl;
    
    return 0;
}

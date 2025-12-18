/**
 * QuantumLRET-Sim - Main Benchmark
 * 
 * Benchmark for n=11 qubits, depth=13 as per README specification.
 * Compares parallel optimized vs naive sequential execution.
 */
#include "gates_and_noise.h"
#include "simulator.h"
#include "utils.h"
#include <iostream>
#include <iomanip>

using namespace qlret;

int main() {
    std::cout << "Starting QuantumLRET-Sim..." << std::endl;
    std::cout.flush();
    
    try {
    // Smaller circuit for clear visualization
    const size_t num_qubits = 3;
    const size_t depth = 5;
    const double noise_prob = 0.05;  // Lower noise
    const double truncation_threshold = 1e-4;
    const bool verbose = false;
    
    std::cout << std::string(98, '-') << std::endl;
    std::cout << "number of qubits: " << num_qubits << std::endl;
    
    // Auto-select batch size based on workload
    size_t batch_size = auto_select_batch_size(num_qubits);
    std::string workload = get_workload_class(num_qubits);
    std::cout << "INFO: n=" << num_qubits << " " << workload 
              << ", batch_size=" << batch_size << std::endl;
    
    // Generate random quantum sequence with noise
    auto sequence = generate_quantum_sequences(num_qubits, depth, true, noise_prob);
    std::cout << "Generated sequence with total noise perc: " 
              << std::fixed << std::setprecision(6) 
              << sequence.total_noise_probability << std::endl;
    std::cout << "batch size: " << batch_size << std::endl;
    std::cout << "current time == " << get_current_time_string() << std::endl;
    
    // Print circuit diagram (compact - first 60 chars per line)
    std::cout << "\nCircuit diagram:" << std::endl;
    print_circuit_diagram(num_qubits, sequence, 0);  // 0 = no wrapping
    
    // Create initial |0...0> state
    MatrixXcd L_init = create_zero_state(num_qubits);
    
    // Print header
    std::cout << "=====================Running LRET simulation for " 
              << num_qubits << " qubits==========================" << std::endl;
    
    // Run optimized parallel simulation
    Timer parallel_timer;
    MatrixXcd L_parallel = run_simulation_optimized(
        L_init, sequence, num_qubits,
        batch_size, true, verbose, truncation_threshold
    );
    double parallel_time = parallel_timer.elapsed_seconds();
    
    std::cout << "Simulation Time: " << std::fixed << std::setprecision(3) 
              << parallel_time << " seconds" << std::endl;
    std::cout << "Final Rank: " << L_parallel.cols() << std::endl;
    
    // Run naive simulation for comparison
    std::cout << "\nRunning naive (sequential) simulation for comparison..." << std::endl;
    Timer naive_timer;
    MatrixXcd L_naive = run_simulation_naive(L_init, sequence, num_qubits, false);
    double naive_time = naive_timer.elapsed_seconds();
    
    std::cout << "Naive Time: " << std::fixed << std::setprecision(3) 
              << naive_time << " seconds" << std::endl;
    
    // Compute speedup
    if (parallel_time > 0) {
        double speedup = naive_time / parallel_time;
        std::cout << "\nSpeed up with batch size " << batch_size << " : " 
                  << std::fixed << std::setprecision(3) << speedup << std::endl;
    }
    
    // Compute trace distance
    double trace_dist = compare_L_matrices_trace(L_parallel, L_naive);
    std::cout << "trace distance: " << std::scientific << std::setprecision(2) 
              << trace_dist << std::endl;
    
    std::cout << std::string(98, '-') << std::endl;
    
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown exception" << std::endl;
        return 1;
    }
    
    return 0;
}

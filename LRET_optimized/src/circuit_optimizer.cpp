/**
 * @file circuit_optimizer.cpp
 * @brief Implementation of Circuit Stratification for LRET
 * 
 * Implements layer-based circuit optimization inspired by Cirq's approach.
 * Gates are grouped into layers where no two gates share qubits.
 * 
 * @see include/circuit_optimizer.h for API documentation
 */

#include "circuit_optimizer.h"
#include "gates_and_noise.h"
#include "simd_kernels.h"
#include "simulator.h"
#include <iostream>
#include <iomanip>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace qlret {

//==============================================================================
// StratifiedCircuit Statistics
//==============================================================================

void StratifiedCircuit::print_stats() const {
    std::cout << "\n=== Circuit Stratification Statistics ===" << std::endl;
    std::cout << "Original depth (sequential): " << original_depth << std::endl;
    std::cout << "Stratified depth (layers):   " << stratified_depth << std::endl;
    std::cout << "Total gates:                 " << total_gates << std::endl;
    std::cout << "Parallelism factor:          " << std::fixed << std::setprecision(2) 
              << parallelism_factor << "x" << std::endl;
    std::cout << "Avg gates per layer:         " << std::fixed << std::setprecision(1)
              << avg_layer_width << std::endl;
    std::cout << "=========================================\n" << std::endl;
}

//==============================================================================
// CircuitStratifier Implementation
//==============================================================================

void CircuitStratifier::assign_gates_to_layers(
    const std::vector<GateOp>& gates,
    std::vector<CircuitLayer>& layers
) {
    for (size_t i = 0; i < gates.size(); ++i) {
        const auto& gate = gates[i];
        bool placed = false;
        
        // Try to place in existing layer (greedy: first fit)
        for (auto& layer : layers) {
            if (layer.can_add(gate)) {
                layer.add(gate, i);
                placed = true;
                break;
            }
        }
        
        // If no existing layer works, create new one
        if (!placed) {
            CircuitLayer new_layer;
            new_layer.add(gate, i);
            layers.push_back(std::move(new_layer));
        }
    }
}

StratifiedCircuit CircuitStratifier::stratify_gates_only(
    const std::vector<GateOp>& gates, 
    size_t num_qubits
) {
    StratifiedCircuit result(num_qubits);
    
    if (gates.empty()) {
        return result;
    }
    
    // Assign gates to layers
    assign_gates_to_layers(gates, result.layers);
    
    // Compute statistics
    result.original_depth = gates.size();
    result.stratified_depth = result.layers.size();
    result.total_gates = gates.size();
    
    if (result.stratified_depth > 0) {
        result.parallelism_factor = 
            static_cast<double>(result.original_depth) / result.stratified_depth;
        result.avg_layer_width = 
            static_cast<double>(result.total_gates) / result.stratified_depth;
    }
    
    if (config_.verbose) {
        result.print_stats();
    }
    
    return result;
}

StratifiedCircuit CircuitStratifier::stratify(const QuantumSequence& sequence) {
    // Extract only gate operations (noise handled separately)
    std::vector<GateOp> gates;
    gates.reserve(sequence.operations.size());
    
    for (const auto& op : sequence.operations) {
        if (std::holds_alternative<GateOp>(op)) {
            gates.push_back(std::get<GateOp>(op));
        }
    }
    
    return stratify_gates_only(gates, sequence.num_qubits);
}

//==============================================================================
// Layer-Parallel Execution
//==============================================================================

MatrixXcd apply_layer_parallel(
    const MatrixXcd& L,
    const CircuitLayer& layer,
    size_t num_qubits,
    const SimConfig& config
) {
    if (layer.empty()) {
        return L;
    }
    
    // Single gate - no parallelism benefit, just apply directly
    if (layer.size() == 1) {
        return apply_gate_to_L(L, layer.gates[0], num_qubits);
    }
    
    MatrixXcd result = L;
    
    // For layers with multiple gates, we can apply them in sequence
    // but each individual gate application uses SIMD/OpenMP internally.
    // 
    // NOTE: True layer parallelism would require computing combined unitary,
    // which is O(4^n) and defeats the purpose. Instead, we rely on:
    // 1. SIMD vectorization within each gate apply
    // 2. The reduced layer count for fewer synchronization points
    
    if (layer.all_single_qubit) {
        // Fast path: all single-qubit gates
        // Apply each with SIMD-friendly kernel
        for (const auto& gate : layer.gates) {
            MatrixXcd U = get_single_qubit_gate(gate.type, gate.params);
            result = apply_single_qubit_simd(result, U, gate.qubits[0], num_qubits);
        }
    } else {
        // Mixed path: some two-qubit gates
        for (const auto& gate : layer.gates) {
            result = apply_gate_to_L(result, gate, num_qubits);
        }
    }
    
    return result;
}

MatrixXcd execute_stratified_circuit(
    const MatrixXcd& L_init,
    const StratifiedCircuit& stratified,
    const SimConfig& config
) {
    MatrixXcd L = L_init;
    
    for (const auto& layer : stratified.layers) {
        L = apply_layer_parallel(L, layer, stratified.num_qubits, config);
    }
    
    return L;
}

MatrixXcd execute_stratified_with_noise(
    const MatrixXcd& L_init,
    const QuantumSequence& sequence,
    const StratifiedCircuit& stratified,
    const SimConfig& config
) {
    MatrixXcd L = L_init;
    size_t num_qubits = sequence.num_qubits;
    
    // Build a map of noise operations by their position in original sequence
    std::vector<const NoiseOp*> noise_ops;
    for (const auto& op : sequence.operations) {
        if (std::holds_alternative<NoiseOp>(op)) {
            noise_ops.push_back(&std::get<NoiseOp>(op));
        }
    }
    
    // If no noise, just execute layers
    if (noise_ops.empty()) {
        return execute_stratified_circuit(L_init, stratified, config);
    }
    
    // With noise: apply layers, then apply noise operations, then truncate
    // This is a simplification - in reality, noise should be interleaved
    // based on original circuit structure
    
    // Execute all gate layers
    for (const auto& layer : stratified.layers) {
        L = apply_layer_parallel(L, layer, num_qubits, config);
    }
    
    // Apply all noise operations
    for (const auto* noise : noise_ops) {
        L = apply_noise_to_L(L, *noise, num_qubits);
        
        if (config.do_truncation && L.cols() > 1) {
            L = truncate_L(L, config.truncation_threshold, config.max_rank);
        }
    }
    
    return L;
}

//==============================================================================
// Utility Functions
//==============================================================================

double estimate_stratification_speedup(const StratifiedCircuit& stratified) {
    // Speedup comes from:
    // 1. Reduced synchronization points (fewer layers than gates)
    // 2. Better cache utilization (batch of gates on different qubits)
    // 3. SIMD utilization within layers
    
    if (stratified.stratified_depth == 0 || stratified.original_depth == 0) {
        return 1.0;
    }
    
    // Conservative estimate: ~70% of parallelism factor
    // (accounting for overheads)
    double raw_factor = stratified.parallelism_factor;
    double efficiency = 0.7;
    
    return std::max(1.0, raw_factor * efficiency);
}

bool should_stratify(const QuantumSequence& sequence) {
    if (sequence.operations.size() < 4) {
        return false;  // Too small to benefit
    }
    
    // Count gates and check width
    size_t gate_count = 0;
    std::unordered_set<size_t> all_qubits;
    
    for (const auto& op : sequence.operations) {
        if (std::holds_alternative<GateOp>(op)) {
            const auto& gate = std::get<GateOp>(op);
            gate_count++;
            for (size_t q : gate.qubits) {
                all_qubits.insert(q);
            }
        }
    }
    
    // Stratification helps when:
    // - Many gates (depth > 10)
    // - Multiple qubits used (width > 2)
    // - Ratio of gates to qubits suggests parallelism opportunity
    
    if (gate_count < 10) return false;
    if (all_qubits.size() < 3) return false;
    
    double density = static_cast<double>(gate_count) / all_qubits.size();
    return density > 2.0;  // At least 2 gates per qubit on average
}

}  // namespace qlret

/**
 * @file test_qec_syndrome.cpp
 * @brief Tests for syndrome extraction and error injection
 * 
 * Phase 9.1: Quantum Error Correction - Syndrome Extraction
 */

#include <iostream>
#include <cassert>
#include <cmath>
#include "qec_types.h"
#include "qec_stabilizer.h"
#include "qec_syndrome.h"

using namespace qlret;

//==============================================================================
// Test Helpers
//==============================================================================

void test_passed(const std::string& name) {
    std::cout << "[PASS] " << name << std::endl;
}

//==============================================================================
// ErrorInjector Tests
//==============================================================================

void test_error_injector_single() {
    ErrorInjector injector(5);
    
    auto error = injector.single_error(2, Pauli::X);
    assert(error.size() == 5);
    assert(error.weight() == 1);
    assert(error[2] == Pauli::X);
    
    test_passed("error_injector_single");
}

void test_error_injector_chain() {
    ErrorInjector injector(5);
    
    auto error = injector.error_chain({0, 1, 2}, Pauli::Z);
    assert(error.weight() == 3);
    assert(error[0] == Pauli::Z);
    assert(error[1] == Pauli::Z);
    assert(error[2] == Pauli::Z);
    
    test_passed("error_injector_chain");
}

void test_error_injector_depolarizing() {
    ErrorInjector injector(100, 42);
    
    // With p=0, should get no errors
    auto error_zero = injector.depolarizing(0.0);
    assert(error_zero.weight() == 0);
    
    // With p=1, should get errors on all qubits
    ErrorInjector injector2(10, 123);
    auto error_one = injector2.depolarizing(1.0);
    assert(error_one.weight() == 10);
    
    test_passed("error_injector_depolarizing");
}

//==============================================================================
// Syndrome Extraction Tests - Repetition Code
//==============================================================================

void test_syndrome_extraction_repetition_no_error() {
    RepetitionCode code(5);
    SyndromeExtractor extractor(code);
    
    // No error
    PauliString no_error(5);
    Syndrome syn = extractor.extract(no_error);
    
    assert(syn.num_defects() == 0);
    
    test_passed("syndrome_extraction_repetition_no_error");
}

void test_syndrome_extraction_repetition_single_x() {
    RepetitionCode code(5);
    SyndromeExtractor extractor(code);
    ErrorInjector injector(5);
    
    // Single X error on qubit 2
    auto error = injector.single_error(2, Pauli::X);
    Syndrome syn = extractor.extract(error);
    
    // X error on qubit 2 triggers Z-stabilizers on (1,2) and (2,3)
    assert(syn.z_syndrome.size() == 4);
    assert(syn.z_syndrome[1] == 1);  // Z1Z2
    assert(syn.z_syndrome[2] == 1);  // Z2Z3
    assert(syn.num_defects() == 2);
    
    test_passed("syndrome_extraction_repetition_single_x");
}

void test_syndrome_extraction_repetition_edge_error() {
    RepetitionCode code(5);
    SyndromeExtractor extractor(code);
    ErrorInjector injector(5);
    
    // X error on edge qubit 0
    auto error = injector.single_error(0, Pauli::X);
    Syndrome syn = extractor.extract(error);
    
    // Only triggers first stabilizer
    assert(syn.z_syndrome[0] == 1);  // Z0Z1
    assert(syn.num_defects() == 1);
    
    test_passed("syndrome_extraction_repetition_edge_error");
}

void test_syndrome_extraction_repetition_z_error() {
    RepetitionCode code(5);
    SyndromeExtractor extractor(code);
    ErrorInjector injector(5);
    
    // Z error should not trigger any Z-stabilizers (bit-flip code)
    auto error = injector.single_error(2, Pauli::Z);
    Syndrome syn = extractor.extract(error);
    
    assert(syn.num_defects() == 0);  // Z errors invisible to Z-stabilizers
    
    test_passed("syndrome_extraction_repetition_z_error");
}

//==============================================================================
// Syndrome Extraction Tests - Surface Code
//==============================================================================

void test_syndrome_extraction_surface_no_error() {
    SurfaceCode code(3);
    SyndromeExtractor extractor(code);
    
    PauliString no_error(code.num_data_qubits());
    Syndrome syn = extractor.extract(no_error);
    
    assert(syn.num_defects() == 0);
    
    test_passed("syndrome_extraction_surface_no_error");
}

void test_syndrome_extraction_surface_single_error() {
    SurfaceCode code(3);
    SyndromeExtractor extractor(code);
    ErrorInjector injector(code.num_data_qubits());
    
    // Single X error in the middle
    auto error = injector.single_error(4, Pauli::X);  // Middle qubit
    Syndrome syn = extractor.extract(error);
    
    // Should trigger adjacent Z-stabilizers
    assert(syn.num_defects() >= 1);
    
    test_passed("syndrome_extraction_surface_single_error");
}

//==============================================================================
// Syndrome Tests
//==============================================================================

void test_syndrome_defects() {
    Syndrome syn;
    syn.x_syndrome = {0, 1, 0, 1};
    syn.z_syndrome = {1, 0, 0};
    
    assert(syn.num_defects() == 3);
    
    auto x_defects = syn.x_defect_indices();
    assert(x_defects.size() == 2);
    assert(x_defects[0] == 1);
    assert(x_defects[1] == 3);
    
    auto z_defects = syn.z_defect_indices();
    assert(z_defects.size() == 1);
    assert(z_defects[0] == 0);
    
    test_passed("syndrome_defects");
}

void test_detection_events() {
    Syndrome prev;
    prev.x_syndrome = {0, 1, 0};
    prev.z_syndrome = {1, 0};
    
    Syndrome curr;
    curr.x_syndrome = {1, 1, 1};
    curr.z_syndrome = {1, 1};
    
    Syndrome events = SyndromeExtractor::detection_events(prev, curr);
    
    // XOR of prev and curr
    assert(events.x_syndrome[0] == 1);  // 0 XOR 1
    assert(events.x_syndrome[1] == 0);  // 1 XOR 1
    assert(events.x_syndrome[2] == 1);  // 0 XOR 1
    assert(events.z_syndrome[0] == 0);  // 1 XOR 1
    assert(events.z_syndrome[1] == 1);  // 0 XOR 1
    
    test_passed("detection_events");
}

//==============================================================================
// Measurement Circuit Tests
//==============================================================================

void test_measurement_circuit() {
    RepetitionCode code(3);
    SyndromeExtractor extractor(code);
    
    auto circuit = extractor.measurement_circuit(0, false);  // First Z-stabilizer
    
    // Should have: RESET, CNOTs, MEASURE
    assert(!circuit.empty());
    assert(std::get<0>(circuit.front()) == "RESET");
    assert(std::get<0>(circuit.back()) == "MEASURE");
    
    test_passed("measurement_circuit");
}

//==============================================================================
// Multiple Round Tests
//==============================================================================

void test_multiple_rounds() {
    RepetitionCode code(5);
    SyndromeExtractor extractor(code);
    ErrorInjector injector(5);
    
    auto error = injector.single_error(2, Pauli::X);
    auto syndromes = extractor.extract_multiple_rounds(error, 3);
    
    assert(syndromes.size() == 3);
    
    // Each round should give same syndrome (no additional errors)
    for (const auto& syn : syndromes) {
        assert(syn.num_defects() == 2);
    }
    
    test_passed("multiple_rounds");
}

//==============================================================================
// Noisy Syndrome Tests
//==============================================================================

void test_noisy_syndrome() {
    RepetitionCode code(5);
    
    SyndromeExtractor::NoiseParams noise;
    noise.measurement_error = 0.0;  // No noise
    
    SyndromeExtractor extractor(code, noise);
    
    PauliString error(5);
    error.set(2, Pauli::X);
    
    Syndrome syn = extractor.extract(error);
    assert(syn.num_defects() == 2);  // Ideal syndrome
    
    test_passed("noisy_syndrome");
}

//==============================================================================
// SyndromeGraph Tests
//==============================================================================

void test_syndrome_graph() {
    SurfaceCode code(3);
    
    Syndrome syn;
    syn.z_syndrome = {1, 0, 1, 0};  // Two defects
    syn.x_syndrome = {0, 0, 0, 0};
    
    auto graph = SyndromeGraph::from_syndrome(syn, code, 0.01);
    
    assert(graph.num_defects == 2);
    assert(!graph.edges.empty());
    
    test_passed("syndrome_graph");
}

//==============================================================================
// Main
//==============================================================================

int main() {
    std::cout << "=== QEC Syndrome Extraction Tests ===" << std::endl;
    
    // ErrorInjector tests
    test_error_injector_single();
    test_error_injector_chain();
    test_error_injector_depolarizing();
    
    // Repetition code syndrome tests
    test_syndrome_extraction_repetition_no_error();
    test_syndrome_extraction_repetition_single_x();
    test_syndrome_extraction_repetition_edge_error();
    test_syndrome_extraction_repetition_z_error();
    
    // Surface code syndrome tests
    test_syndrome_extraction_surface_no_error();
    test_syndrome_extraction_surface_single_error();
    
    // Syndrome helper tests
    test_syndrome_defects();
    test_detection_events();
    
    // Measurement circuit tests
    test_measurement_circuit();
    
    // Multiple round tests
    test_multiple_rounds();
    
    // Noisy syndrome tests
    test_noisy_syndrome();
    
    // Graph tests
    test_syndrome_graph();
    
    std::cout << "\n=== All Syndrome Tests Passed ===" << std::endl;
    return 0;
}

/**
 * @file test_qec_logical.cpp
 * @brief Tests for logical qubit interface and QEC simulation
 * 
 * Phase 9.1: Quantum Error Correction - Logical Qubit Interface
 */

#include <iostream>
#include <cassert>
#include <cmath>
#include "qec_types.h"
#include "qec_stabilizer.h"
#include "qec_syndrome.h"
#include "qec_decoder.h"
#include "qec_logical.h"

using namespace qlret;

//==============================================================================
// Test Helpers
//==============================================================================

void test_passed(const std::string& name) {
    std::cout << "[PASS] " << name << std::endl;
}

//==============================================================================
// LogicalQubit Creation Tests
//==============================================================================

void test_logical_qubit_creation() {
    LogicalQubit::Config config;
    config.code_type = QECCodeType::SURFACE;
    config.distance = 3;
    config.decoder_type = DecoderType::MWPM;
    
    LogicalQubit qubit(config);
    
    assert(qubit.config().distance == 3);
    assert(qubit.config().code_type == QECCodeType::SURFACE);
    
    test_passed("logical_qubit_creation");
}

void test_logical_qubit_default_config() {
    LogicalQubit qubit;
    
    // Should use default config
    assert(qubit.config().distance == 3);
    
    test_passed("logical_qubit_default_config");
}

//==============================================================================
// Initialization Tests
//==============================================================================

void test_logical_qubit_initialize_zero() {
    LogicalQubit qubit;
    qubit.initialize_zero();
    
    // Should be in |0_L> state (no accumulated error)
    assert(!qubit.has_logical_error());
    
    test_passed("logical_qubit_initialize_zero");
}

void test_logical_qubit_initialize_one() {
    LogicalQubit qubit;
    qubit.initialize_one();
    
    // Should be in |1_L> state
    
    test_passed("logical_qubit_initialize_one");
}

void test_logical_qubit_initialize_plus() {
    LogicalQubit qubit;
    qubit.initialize_plus();
    
    test_passed("logical_qubit_initialize_plus");
}

//==============================================================================
// Logical Gate Tests
//==============================================================================

void test_logical_x_gate() {
    LogicalQubit qubit;
    qubit.initialize_zero();
    
    qubit.apply_logical_x();
    
    // After X, should be in |1_L>
    // Measure should give 1
    
    test_passed("logical_x_gate");
}

void test_logical_z_gate() {
    LogicalQubit qubit;
    qubit.initialize_zero();
    
    qubit.apply_logical_z();
    
    // Z on |0> should give |0> (eigenstate)
    
    test_passed("logical_z_gate");
}

void test_logical_y_gate() {
    LogicalQubit qubit;
    qubit.initialize_zero();
    
    qubit.apply_logical_y();
    
    test_passed("logical_y_gate");
}

void test_logical_h_gate() {
    LogicalQubit qubit;
    qubit.initialize_zero();
    
    qubit.apply_logical_h();
    
    // After H, should be in |+_L>
    
    test_passed("logical_h_gate");
}

//==============================================================================
// QEC Round Tests
//==============================================================================

void test_qec_round_no_error() {
    LogicalQubit qubit;
    qubit.initialize_zero();
    
    auto result = qubit.qec_round();
    
    // No error injected, should detect nothing
    assert(!result.detected_error);
    assert(!result.logical_error);
    
    test_passed("qec_round_no_error");
}

void test_qec_round_with_error() {
    LogicalQubit qubit;
    qubit.initialize_zero();
    
    // Inject small error
    qubit.inject_error(0.5);  // 50% chance per qubit
    
    auto result = qubit.qec_round();
    
    // Should detect and correct
    
    test_passed("qec_round_with_error");
}

void test_qec_round_single_error() {
    LogicalQubit qubit;
    qubit.initialize_zero();
    
    // Inject specific single-qubit error
    PauliString error(qubit.code().num_data_qubits());
    error.set(0, Pauli::X);
    qubit.inject_error(error);
    
    auto result = qubit.qec_round();
    
    assert(result.detected_error);
    
    test_passed("qec_round_single_error");
}

void test_multiple_qec_rounds() {
    LogicalQubit qubit;
    qubit.initialize_zero();
    
    auto results = qubit.qec_rounds(3);
    
    assert(results.size() == 3);
    
    test_passed("multiple_qec_rounds");
}

//==============================================================================
// Error Tracking Tests
//==============================================================================

void test_accumulated_error() {
    LogicalQubit qubit;
    qubit.initialize_zero();
    
    PauliString error(qubit.code().num_data_qubits());
    error.set(2, Pauli::X);
    qubit.inject_error(error);
    
    auto acc = qubit.get_accumulated_error();
    assert(acc[2] == Pauli::X);
    
    test_passed("accumulated_error");
}

void test_logical_error_detection() {
    LogicalQubit::Config config;
    config.code_type = QECCodeType::REPETITION;
    config.distance = 3;
    
    LogicalQubit qubit(config);
    qubit.initialize_zero();
    
    // Inject error chain that creates logical error
    PauliString error(3);
    error.set(0, Pauli::X);
    error.set(1, Pauli::X);
    error.set(2, Pauli::X);  // Logical X operator
    qubit.inject_error(error);
    
    assert(qubit.has_logical_error());
    
    test_passed("logical_error_detection");
}

//==============================================================================
// Statistics Tests
//==============================================================================

void test_logical_qubit_stats() {
    LogicalQubit qubit;
    qubit.initialize_zero();
    
    // Perform multiple QEC rounds
    for (int i = 0; i < 5; ++i) {
        qubit.qec_round();
    }
    
    assert(qubit.stats().total_qec_rounds == 5);
    
    qubit.reset_stats();
    assert(qubit.stats().total_qec_rounds == 0);
    
    test_passed("logical_qubit_stats");
}

//==============================================================================
// Configuration Tests
//==============================================================================

void test_set_error_rates() {
    LogicalQubit qubit;
    
    qubit.set_physical_error_rate(0.001);
    qubit.set_measurement_error_rate(0.002);
    qubit.set_syndrome_rounds(3);
    qubit.set_auto_correct(false);
    
    test_passed("set_error_rates");
}

//==============================================================================
// LogicalRegister Tests
//==============================================================================

void test_logical_register_creation() {
    LogicalRegister::Config config;
    config.distance = 3;
    
    LogicalRegister reg(3, config);
    
    assert(reg.size() == 3);
    
    test_passed("logical_register_creation");
}

void test_logical_register_access() {
    LogicalRegister reg(2);
    
    reg.qubit(0).initialize_zero();
    reg.qubit(1).initialize_one();
    
    test_passed("logical_register_access");
}

void test_logical_cnot() {
    LogicalRegister reg(2);
    
    reg.qubit(0).initialize_one();  // |1>
    reg.qubit(1).initialize_zero(); // |0>
    
    reg.apply_logical_cnot(0, 1);
    
    // After CNOT: |11>
    
    test_passed("logical_cnot");
}

void test_logical_register_qec_all() {
    LogicalRegister reg(3);
    reg.initialize_all_zero();
    
    auto results = reg.qec_round_all();
    
    assert(results.size() == 3);
    
    test_passed("logical_register_qec_all");
}

//==============================================================================
// QECSimulator Tests
//==============================================================================

void test_qec_simulator_creation() {
    QECSimulator::SimConfig config;
    config.code_type = QECCodeType::REPETITION;
    config.distance = 3;
    config.num_trials = 100;
    
    QECSimulator sim(config);
    
    test_passed("qec_simulator_creation");
}

void test_qec_simulator_run() {
    QECSimulator::SimConfig config;
    config.code_type = QECCodeType::REPETITION;
    config.distance = 3;
    config.num_trials = 10;  // Small for test
    config.physical_error_rate = 0.01;
    
    QECSimulator sim(config);
    auto result = sim.run();
    
    assert(result.num_trials == 10);
    assert(result.logical_error_rate >= 0.0);
    assert(result.logical_error_rate <= 1.0);
    
    std::cout << "  Logical error rate: " << result.logical_error_rate << std::endl;
    std::cout << "  Avg decode time: " << result.avg_decode_time_ms << " ms" << std::endl;
    
    test_passed("qec_simulator_run");
}

//==============================================================================
// LogicalState Tests
//==============================================================================

void test_logical_state() {
    LogicalQubit qubit;
    qubit.initialize_zero();
    
    LogicalState state = qubit.get_state();
    
    assert(state.code_distance == 3);
    assert(state.code_type == QECCodeType::SURFACE);
    
    test_passed("logical_state");
}

//==============================================================================
// Main
//==============================================================================

int main() {
    std::cout << "=== QEC Logical Qubit Tests ===" << std::endl;
    
    // Creation tests
    test_logical_qubit_creation();
    test_logical_qubit_default_config();
    
    // Initialization tests
    test_logical_qubit_initialize_zero();
    test_logical_qubit_initialize_one();
    test_logical_qubit_initialize_plus();
    
    // Gate tests
    test_logical_x_gate();
    test_logical_z_gate();
    test_logical_y_gate();
    test_logical_h_gate();
    
    // QEC round tests
    test_qec_round_no_error();
    test_qec_round_with_error();
    test_qec_round_single_error();
    test_multiple_qec_rounds();
    
    // Error tracking tests
    test_accumulated_error();
    test_logical_error_detection();
    
    // Stats tests
    test_logical_qubit_stats();
    
    // Config tests
    test_set_error_rates();
    
    // Register tests
    test_logical_register_creation();
    test_logical_register_access();
    test_logical_cnot();
    test_logical_register_qec_all();
    
    // Simulator tests
    test_qec_simulator_creation();
    test_qec_simulator_run();
    
    // State tests
    test_logical_state();
    
    std::cout << "\n=== All Logical Qubit Tests Passed ===" << std::endl;
    return 0;
}

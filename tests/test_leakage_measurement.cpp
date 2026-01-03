/**
 * @file test_leakage_measurement.cpp
 * @brief Tests for Phase 4.4 (Leakage) and Phase 4.5 (Measurement) features
 */

#include "types.h"
#include "gates_and_noise.h"
#include "fdm_simulator.h"
#include "noise_import.h"
#include <cassert>
#include <iostream>
#include <cmath>

using namespace qlret;

//==============================================================================
// Leakage Channel Tests (Phase 4.4)
//==============================================================================

void test_leakage_kraus_operators() {
    std::cout << "Testing leakage Kraus operators..." << std::endl;
    
    // Get Kraus operators for leakage channel
    double p_leak = 0.1;
    auto kraus = get_noise_kraus_operators(NoiseType::LEAKAGE, p_leak, {});
    
    // Should have 2 Kraus operators
    assert(kraus.size() == 2);
    
    // Verify completeness: sum(K†K) = I
    MatrixXcd sum = MatrixXcd::Zero(2, 2);
    for (const auto& K : kraus) {
        sum += K.adjoint() * K;
    }
    
    double trace_err = std::abs(sum.trace() - 2.0);
    assert(trace_err < 1e-10);
    
    std::cout << "  Leakage Kraus operators: PASS" << std::endl;
}

void test_leakage_relaxation_kraus() {
    std::cout << "Testing leakage relaxation Kraus operators..." << std::endl;
    
    double p_relax = 0.2;
    auto kraus = get_noise_kraus_operators(NoiseType::LEAKAGE_RELAXATION, p_relax, {});
    
    // Should have 2 Kraus operators (amplitude damping style)
    assert(kraus.size() == 2);
    
    // Verify trace preservation
    MatrixXcd sum = MatrixXcd::Zero(2, 2);
    for (const auto& K : kraus) {
        sum += K.adjoint() * K;
    }
    double trace_err = std::abs(sum.trace() - 2.0);
    assert(trace_err < 1e-10);
    
    std::cout << "  Leakage relaxation Kraus: PASS" << std::endl;
}

void test_apply_leakage_channel() {
    std::cout << "Testing apply_leakage_channel..." << std::endl;
    
    // Create a simple 2-qubit state |00>
    size_t num_qubits = 2;
    size_t dim = 1ULL << num_qubits;
    MatrixXcd L = MatrixXcd::Zero(dim, 1);
    L(0, 0) = 1.0;  // |00>
    
    // Apply leakage to qubit 0
    double p_leak = 0.1;
    MatrixXcd L_new = apply_leakage_channel(L, 0, p_leak, num_qubits);
    
    // Rank should increase (2 Kraus operators)
    assert(L_new.cols() == 2);
    
    // Trace should be preserved
    double trace_before = L.squaredNorm();
    double trace_after = L_new.squaredNorm();
    assert(std::abs(trace_before - trace_after) < 1e-10);
    
    std::cout << "  apply_leakage_channel: PASS" << std::endl;
}

void test_leakage_full() {
    std::cout << "Testing apply_leakage_full..." << std::endl;
    
    size_t num_qubits = 2;
    size_t dim = 1ULL << num_qubits;
    MatrixXcd L = MatrixXcd::Zero(dim, 1);
    L(0, 0) = 1.0;
    
    LeakageChannel ch;
    ch.p_leak = 0.05;
    ch.p_relax = 0.02;
    ch.p_phase = 0.01;
    
    MatrixXcd L_new = apply_leakage_full(L, 0, ch, num_qubits);
    
    // Should have expanded rank due to multiple channels
    assert(L_new.cols() > L.cols());
    
    std::cout << "  apply_leakage_full: PASS" << std::endl;
}

//==============================================================================
// Measurement Tests (Phase 4.5)
//==============================================================================

void test_measurement_projectors() {
    std::cout << "Testing measurement projectors..." << std::endl;
    
    size_t num_qubits = 2;
    size_t dim = 1ULL << num_qubits;
    
    // State |+> on qubit 0, |0> on qubit 1: (|00> + |10>)/sqrt(2)
    MatrixXcd L = MatrixXcd::Zero(dim, 1);
    L(0, 0) = 1.0 / std::sqrt(2.0);  // |00>
    L(2, 0) = 1.0 / std::sqrt(2.0);  // |10>
    
    std::array<double, 2> probs;
    auto [L0, L1] = apply_measurement_projectors(L, 0, num_qubits, probs);
    
    // Should get 50/50 probabilities for qubit 0
    assert(std::abs(probs[0] - 0.5) < 1e-10);
    assert(std::abs(probs[1] - 0.5) < 1e-10);
    
    // L0 should be |00>, L1 should be |10> (normalized)
    assert(std::abs(L0(0, 0)) > 0.9);  // Main amplitude at |00>
    assert(std::abs(L1(2, 0)) > 0.9);  // Main amplitude at |10>
    
    std::cout << "  Measurement projectors: PASS" << std::endl;
}

void test_confusion_matrix() {
    std::cout << "Testing confusion matrix application..." << std::endl;
    
    // Perfect confusion matrix (identity)
    MatrixXd confusion_perfect(2, 2);
    confusion_perfect << 1.0, 0.0,
                         0.0, 1.0;
    
    std::array<double, 2> ideal = {0.3, 0.7};
    auto observed = apply_confusion_matrix(ideal, confusion_perfect);
    
    assert(std::abs(observed[0] - ideal[0]) < 1e-10);
    assert(std::abs(observed[1] - ideal[1]) < 1e-10);
    
    // Realistic confusion matrix (some bit-flip error)
    MatrixXd confusion_noisy(2, 2);
    confusion_noisy << 0.95, 0.05,   // P(measure 0 | actual 0), P(measure 0 | actual 1)
                       0.05, 0.95;   // P(measure 1 | actual 0), P(measure 1 | actual 1)
    
    observed = apply_confusion_matrix(ideal, confusion_noisy);
    
    // Should mix probabilities
    double expected_0 = 0.95 * 0.3 + 0.05 * 0.7;  // 0.285 + 0.035 = 0.32
    double expected_1 = 0.05 * 0.3 + 0.95 * 0.7;  // 0.015 + 0.665 = 0.68
    
    assert(std::abs(observed[0] - expected_0) < 0.01);
    assert(std::abs(observed[1] - expected_1) < 0.01);
    
    std::cout << "  Confusion matrix: PASS" << std::endl;
}

void test_sample_measurement() {
    std::cout << "Testing measurement sampling..." << std::endl;
    
    std::array<double, 2> probs = {0.3, 0.7};
    
    // Sample below threshold -> outcome 0
    int outcome = sample_measurement_outcome(probs, 0.2);
    assert(outcome == 0);
    
    // Sample above threshold -> outcome 1
    outcome = sample_measurement_outcome(probs, 0.5);
    assert(outcome == 1);
    
    // Edge case at threshold
    outcome = sample_measurement_outcome(probs, 0.3);
    assert(outcome == 1);
    
    std::cout << "  Measurement sampling: PASS" << std::endl;
}

void test_fdm_measurement() {
    std::cout << "Testing FDM measurement..." << std::endl;
    
    size_t num_qubits = 2;
    size_t dim = 1ULL << num_qubits;
    
    // Create |+> state on qubit 0
    MatrixXcd rho = MatrixXcd::Zero(dim, dim);
    rho(0, 0) = 0.5;
    rho(0, 2) = 0.5;
    rho(2, 0) = 0.5;
    rho(2, 2) = 0.5;
    
    std::array<double, 2> probs;
    auto [rho0, rho1] = apply_measurement_to_rho(rho, 0, num_qubits, probs);
    
    // 50/50 measurement
    assert(std::abs(probs[0] - 0.5) < 1e-10);
    assert(std::abs(probs[1] - 0.5) < 1e-10);
    
    // Post-measurement states should be normalized
    assert(std::abs(rho0.trace().real() - 1.0) < 1e-10);
    assert(std::abs(rho1.trace().real() - 1.0) < 1e-10);
    
    std::cout << "  FDM measurement: PASS" << std::endl;
}

//==============================================================================
// JSON Parsing Tests
//==============================================================================

void test_leakage_json_parsing() {
    std::cout << "Testing leakage JSON parsing..." << std::endl;
    
    std::string json_str = R"({
        "errors": [],
        "leakage": {
            "enabled": true,
            "default": {
                "p_leak": 0.001,
                "p_relax": 0.0005,
                "p_phase": 0.0001,
                "use_qutrit": false
            },
            "qubits": [
                {"id": 0, "p_leak": 0.002},
                {"id": 1, "p_leak": 0.0015, "p_relax": 0.001}
            ]
        }
    })";
    
    NoiseModelImporter importer;
    NoiseModel model = importer.load_from_json_string(json_str);
    
    assert(model.use_leakage == true);
    assert(std::abs(model.default_leakage.p_leak - 0.001) < 1e-10);
    assert(std::abs(model.default_leakage.p_relax - 0.0005) < 1e-10);
    assert(model.leakage_channels.size() == 2);
    assert(std::abs(model.leakage_channels[0].p_leak - 0.002) < 1e-10);
    assert(std::abs(model.leakage_channels[1].p_relax - 0.001) < 1e-10);
    
    std::cout << "  Leakage JSON parsing: PASS" << std::endl;
}

void test_measurement_confusion_json_parsing() {
    std::cout << "Testing measurement confusion JSON parsing..." << std::endl;
    
    std::string json_str = R"({
        "errors": [],
        "measurement_confusion": {
            "enabled": true,
            "entries": [
                {
                    "qubit": 0,
                    "conditional": false,
                    "matrix": [[0.98, 0.02], [0.03, 0.97]]
                },
                {
                    "qubit": 1,
                    "conditional": true,
                    "matrix": [[0.95, 0.05], [0.05, 0.95]]
                }
            ]
        }
    })";
    
    NoiseModelImporter importer;
    NoiseModel model = importer.load_from_json_string(json_str);
    
    assert(model.use_measurement_confusion == true);
    assert(model.measurement_specs.size() == 2);
    
    auto& spec0 = model.measurement_specs[0];
    assert(spec0.qubit == 0);
    assert(spec0.conditional == false);
    assert(std::abs(spec0.confusion(0, 0) - 0.98) < 1e-10);
    assert(std::abs(spec0.confusion(0, 1) - 0.02) < 1e-10);
    
    auto& spec1 = model.measurement_specs[1];
    assert(spec1.conditional == true);
    
    std::cout << "  Measurement confusion JSON parsing: PASS" << std::endl;
}

//==============================================================================
// Main
//==============================================================================

int main() {
    std::cout << "=== Phase 4.4/4.5 Leakage & Measurement Tests ===" << std::endl;
    std::cout << std::endl;
    
    // Leakage tests
    std::cout << "--- Leakage Channel Tests (Phase 4.4) ---" << std::endl;
    test_leakage_kraus_operators();
    test_leakage_relaxation_kraus();
    test_apply_leakage_channel();
    test_leakage_full();
    std::cout << std::endl;
    
    // Measurement tests
    std::cout << "--- Measurement Tests (Phase 4.5) ---" << std::endl;
    test_measurement_projectors();
    test_confusion_matrix();
    test_sample_measurement();
    test_fdm_measurement();
    std::cout << std::endl;
    
    // JSON parsing tests
    std::cout << "--- JSON Parsing Tests ---" << std::endl;
    test_leakage_json_parsing();
    test_measurement_confusion_json_parsing();
    std::cout << std::endl;
    
    std::cout << "=== All Phase 4.4/4.5 tests passed! ===" << std::endl;
    return 0;
}

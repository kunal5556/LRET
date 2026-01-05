/**
 * @file test_qec_stabilizer.cpp
 * @brief Tests for stabilizer codes and Pauli string operations
 * 
 * Phase 9.1: Quantum Error Correction - Stabilizer Code Foundation
 */

#include <iostream>
#include <cassert>
#include <cmath>
#include "qec_types.h"
#include "qec_stabilizer.h"

using namespace qlret;

//==============================================================================
// Test Helpers
//==============================================================================

void test_passed(const std::string& name) {
    std::cout << "[PASS] " << name << std::endl;
}

void test_failed(const std::string& name, const std::string& reason) {
    std::cout << "[FAIL] " << name << ": " << reason << std::endl;
}

//==============================================================================
// Pauli Multiplication Tests
//==============================================================================

void test_pauli_mult() {
    // I * X = X
    assert(pauli_mult(Pauli::I, Pauli::X) == Pauli::X);
    // X * X = I
    assert(pauli_mult(Pauli::X, Pauli::X) == Pauli::I);
    // X * Y = iZ
    assert(pauli_mult(Pauli::X, Pauli::Y) == Pauli::Z);
    // Y * Z = iX
    assert(pauli_mult(Pauli::Y, Pauli::Z) == Pauli::X);
    // Z * X = iY
    assert(pauli_mult(Pauli::Z, Pauli::X) == Pauli::Y);
    
    test_passed("pauli_mult");
}

void test_pauli_mult_phase() {
    // X * Y = iZ (phase = 1)
    assert(pauli_mult_phase(Pauli::X, Pauli::Y) == 1);
    // Y * X = -iZ (phase = 3)
    assert(pauli_mult_phase(Pauli::Y, Pauli::X) == 3);
    // X * X = I (phase = 0)
    assert(pauli_mult_phase(Pauli::X, Pauli::X) == 0);
    
    test_passed("pauli_mult_phase");
}

//==============================================================================
// PauliString Tests
//==============================================================================

void test_pauli_string_creation() {
    // From string
    PauliString ps1("XZIY");
    assert(ps1.size() == 4);
    assert(ps1[0] == Pauli::X);
    assert(ps1[1] == Pauli::Z);
    assert(ps1[2] == Pauli::I);
    assert(ps1[3] == Pauli::Y);
    
    // From size
    PauliString ps2(5);
    assert(ps2.size() == 5);
    assert(ps2.is_identity());
    
    test_passed("pauli_string_creation");
}

void test_pauli_string_multiplication() {
    PauliString ps1("XX");
    PauliString ps2("XZ");
    
    PauliString product = ps1 * ps2;
    assert(product.size() == 2);
    assert(product[0] == Pauli::I);  // X * X = I
    assert(product[1] == Pauli::Y);  // X * Z = -iY
    
    test_passed("pauli_string_multiplication");
}

void test_pauli_string_commutation() {
    // X and Z anti-commute
    PauliString px("X");
    PauliString pz("Z");
    assert(!px.commutes_with(pz));
    
    // XX and ZZ commute (even number of anti-commuting pairs)
    PauliString pxx("XX");
    PauliString pzz("ZZ");
    assert(pxx.commutes_with(pzz));
    
    // XZ and ZX anti-commute
    PauliString pxz("XZ");
    PauliString pzx("ZX");
    assert(!pxz.commutes_with(pzx));
    
    // Same Paulis commute
    assert(px.commutes_with(px));
    
    test_passed("pauli_string_commutation");
}

void test_pauli_string_weight() {
    PauliString ps1("IXIZI");
    assert(ps1.weight() == 2);
    
    PauliString ps2("IIIII");
    assert(ps2.weight() == 0);
    
    PauliString ps3("XYZXYZ");
    assert(ps3.weight() == 6);
    
    test_passed("pauli_string_weight");
}

void test_pauli_string_support() {
    PauliString ps("IXZIY");
    auto sup = ps.support();
    assert(sup.size() == 3);
    assert(sup[0] == 1);
    assert(sup[1] == 2);
    assert(sup[2] == 4);
    
    test_passed("pauli_string_support");
}

//==============================================================================
// RepetitionCode Tests
//==============================================================================

void test_repetition_code_creation() {
    RepetitionCode code(3);
    
    assert(code.distance() == 3);
    assert(code.num_data_qubits() == 3);
    assert(code.num_logical_qubits() == 1);
    assert(code.code_type() == QECCodeType::REPETITION);
    
    test_passed("repetition_code_creation");
}

void test_repetition_code_stabilizers() {
    RepetitionCode code(5);  // 5-qubit repetition code
    
    // Should have 4 stabilizers: ZZ on pairs (0,1), (1,2), (2,3), (3,4)
    assert(code.z_stabilizers().size() == 4);
    assert(code.x_stabilizers().empty());  // Bit-flip code
    
    // Each stabilizer should have weight 2
    for (const auto& stab : code.z_stabilizers()) {
        assert(stab.weight() == 2);
    }
    
    test_passed("repetition_code_stabilizers");
}

void test_repetition_code_logical_operators() {
    RepetitionCode code(3);
    
    const auto& log_x = code.logical_x(0);
    const auto& log_z = code.logical_z(0);
    
    // Logical X: X on first qubit
    assert(log_x.weight() == 1);
    
    // Logical Z: Z on all qubits
    assert(log_z.weight() == 3);
    
    // Logical operators should anti-commute
    assert(!log_x.commutes_with(log_z));
    
    // Logical operators should commute with stabilizers
    for (const auto& stab : code.z_stabilizers()) {
        assert(log_x.commutes_with(stab));
        assert(log_z.commutes_with(stab));
    }
    
    test_passed("repetition_code_logical_operators");
}

void test_repetition_code_validation() {
    RepetitionCode code(5);
    
    assert(code.validate_stabilizers());
    assert(code.validate_logical_operators());
    
    test_passed("repetition_code_validation");
}

//==============================================================================
// SurfaceCode Tests
//==============================================================================

void test_surface_code_creation() {
    SurfaceCode code(3);
    
    assert(code.distance() == 3);
    assert(code.num_logical_qubits() == 1);
    assert(code.code_type() == QECCodeType::SURFACE);
    
    // Distance-3 surface code: 9 data qubits
    assert(code.num_data_qubits() == 9);
    
    test_passed("surface_code_creation");
}

void test_surface_code_stabilizers() {
    SurfaceCode code(3);
    
    const auto& x_stabs = code.x_stabilizers();
    const auto& z_stabs = code.z_stabilizers();
    
    // Should have stabilizers
    assert(!x_stabs.empty());
    assert(!z_stabs.empty());
    
    // Stabilizers should have weight 2, 3, or 4
    for (const auto& stab : x_stabs) {
        size_t w = stab.weight();
        assert(w >= 2 && w <= 4);
    }
    
    for (const auto& stab : z_stabs) {
        size_t w = stab.weight();
        assert(w >= 2 && w <= 4);
    }
    
    test_passed("surface_code_stabilizers");
}

void test_surface_code_validation() {
    SurfaceCode code(3);
    
    // All stabilizers should commute
    assert(code.validate_stabilizers());
    
    // Logical operators should satisfy requirements
    assert(code.validate_logical_operators());
    
    test_passed("surface_code_validation");
}

void test_surface_code_distance_5() {
    SurfaceCode code(5);
    
    assert(code.distance() == 5);
    assert(code.num_data_qubits() == 25);  // 5x5 = 25 data qubits
    
    assert(code.validate_stabilizers());
    assert(code.validate_logical_operators());
    
    // Logical operators should have minimum weight = distance
    assert(code.logical_x(0).weight() >= 5);
    assert(code.logical_z(0).weight() >= 5);
    
    test_passed("surface_code_distance_5");
}

//==============================================================================
// Factory Tests
//==============================================================================

void test_stabilizer_code_factory() {
    auto rep_code = create_stabilizer_code(QECCodeType::REPETITION, 5);
    assert(rep_code != nullptr);
    assert(rep_code->distance() == 5);
    assert(rep_code->code_type() == QECCodeType::REPETITION);
    
    auto surf_code = create_stabilizer_code(QECCodeType::SURFACE, 3);
    assert(surf_code != nullptr);
    assert(surf_code->distance() == 3);
    assert(surf_code->code_type() == QECCodeType::SURFACE);
    
    test_passed("stabilizer_code_factory");
}

//==============================================================================
// Main
//==============================================================================

int main() {
    std::cout << "=== QEC Stabilizer Code Tests ===" << std::endl;
    
    // Pauli tests
    test_pauli_mult();
    test_pauli_mult_phase();
    
    // PauliString tests
    test_pauli_string_creation();
    test_pauli_string_multiplication();
    test_pauli_string_commutation();
    test_pauli_string_weight();
    test_pauli_string_support();
    
    // Repetition code tests
    test_repetition_code_creation();
    test_repetition_code_stabilizers();
    test_repetition_code_logical_operators();
    test_repetition_code_validation();
    
    // Surface code tests
    test_surface_code_creation();
    test_surface_code_stabilizers();
    test_surface_code_validation();
    test_surface_code_distance_5();
    
    // Factory tests
    test_stabilizer_code_factory();
    
    std::cout << "\n=== All Stabilizer Tests Passed ===" << std::endl;
    return 0;
}

#pragma once

/**
 * @file qec_types.h
 * @brief Phase 9.1: Core types for Quantum Error Correction
 *
 * Provides Pauli string representation and basic QEC utilities.
 */

#include "types.h"
#include <vector>
#include <string>
#include <cstdint>

namespace qlret {

//==============================================================================
// Pauli Operators
//==============================================================================

enum class Pauli : uint8_t {
    I = 0,  // Identity
    X = 1,  // Pauli-X
    Y = 2,  // Pauli-Y
    Z = 3   // Pauli-Z
};

inline char pauli_to_char(Pauli p) {
    switch (p) {
        case Pauli::I: return 'I';
        case Pauli::X: return 'X';
        case Pauli::Y: return 'Y';
        case Pauli::Z: return 'Z';
        default: return '?';
    }
}

inline Pauli char_to_pauli(char c) {
    switch (c) {
        case 'I': case 'i': return Pauli::I;
        case 'X': case 'x': return Pauli::X;
        case 'Y': case 'y': return Pauli::Y;
        case 'Z': case 'z': return Pauli::Z;
        default: return Pauli::I;
    }
}

// Pauli multiplication table (mod phases)
inline Pauli pauli_mult(Pauli a, Pauli b) {
    // XY=iZ, YZ=iX, ZX=iY, XX=I, etc.
    static const Pauli table[4][4] = {
        {Pauli::I, Pauli::X, Pauli::Y, Pauli::Z},  // I * ...
        {Pauli::X, Pauli::I, Pauli::Z, Pauli::Y},  // X * ...
        {Pauli::Y, Pauli::Z, Pauli::I, Pauli::X},  // Y * ...
        {Pauli::Z, Pauli::Y, Pauli::X, Pauli::I}   // Z * ...
    };
    return table[static_cast<int>(a)][static_cast<int>(b)];
}

// Phase from Pauli multiplication: returns 0,1,2,3 for 1,i,-1,-i
inline int pauli_mult_phase(Pauli a, Pauli b) {
    // Returns phase exponent k where a*b = i^k * pauli_mult(a,b)
    static const int phase_table[4][4] = {
        {0, 0, 0, 0},  // I * ...
        {0, 0, 1, 3},  // X * Y = iZ (1), X * Z = -iY (3)
        {0, 3, 0, 1},  // Y * X = -iZ (3), Y * Z = iX (1)
        {0, 1, 3, 0}   // Z * X = iY (1), Z * Y = -iX (3)
    };
    return phase_table[static_cast<int>(a)][static_cast<int>(b)];
}

//==============================================================================
// Pauli String (tensor product of Paulis)
//==============================================================================

/**
 * @brief Represents a Pauli string P = i^phase * P_0 ⊗ P_1 ⊗ ... ⊗ P_{n-1}
 */
struct PauliString {
    std::vector<Pauli> paulis;  ///< Pauli on each qubit
    int phase = 0;              ///< Phase exponent (0,1,2,3 for 1,i,-1,-i)

    PauliString() = default;
    explicit PauliString(size_t n) : paulis(n, Pauli::I), phase(0) {}
    PauliString(const std::string& s);

    size_t size() const { return paulis.size(); }
    bool is_identity() const;
    
    // Set Pauli at qubit q
    void set(size_t q, Pauli p) { paulis[q] = p; }
    Pauli get(size_t q) const { return paulis[q]; }

    // String representation
    std::string to_string() const;

    // Multiplication
    PauliString operator*(const PauliString& other) const;

    // Commutation check: returns true if [this, other] = 0
    bool commutes_with(const PauliString& other) const;

    // Weight (number of non-identity Paulis)
    size_t weight() const;

    // Support (qubits with non-identity Paulis)
    std::vector<size_t> support() const;
};

//==============================================================================
// Syndrome Types
//==============================================================================

struct Syndrome {
    std::vector<int> x_syndrome;  ///< X-stabilizer measurements (0 or 1)
    std::vector<int> z_syndrome;  ///< Z-stabilizer measurements (0 or 1)
    size_t round = 0;             ///< Measurement round number

    size_t num_defects() const;
    std::vector<size_t> x_defect_indices() const;
    std::vector<size_t> z_defect_indices() const;
};

//==============================================================================
// Error Types
//==============================================================================

struct ErrorLocation {
    size_t qubit;
    Pauli error_type;  // X, Y, or Z error
    double time = 0.0; // When error occurred (for time-correlated decoding)
};

struct Correction {
    std::vector<ErrorLocation> corrections;
    double confidence = 1.0;
};

}  // namespace qlret

#include "qec_types.h"
#include <sstream>

namespace qlret {

//==============================================================================
// PauliString Implementation
//==============================================================================

PauliString::PauliString(const std::string& s) : phase(0) {
    paulis.reserve(s.size());
    for (char c : s) {
        paulis.push_back(char_to_pauli(c));
    }
}

bool PauliString::is_identity() const {
    for (auto p : paulis) {
        if (p != Pauli::I) return false;
    }
    return (phase % 4) == 0;
}

std::string PauliString::to_string() const {
    std::ostringstream oss;
    if (phase == 1) oss << "i";
    else if (phase == 2) oss << "-";
    else if (phase == 3) oss << "-i";
    for (auto p : paulis) {
        oss << pauli_to_char(p);
    }
    return oss.str();
}

PauliString PauliString::operator*(const PauliString& other) const {
    size_t n = std::max(paulis.size(), other.paulis.size());
    PauliString result(n);
    result.phase = (phase + other.phase) % 4;

    for (size_t i = 0; i < n; ++i) {
        Pauli a = (i < paulis.size()) ? paulis[i] : Pauli::I;
        Pauli b = (i < other.paulis.size()) ? other.paulis[i] : Pauli::I;
        result.paulis[i] = pauli_mult(a, b);
        result.phase = (result.phase + pauli_mult_phase(a, b)) % 4;
    }
    return result;
}

bool PauliString::commutes_with(const PauliString& other) const {
    // Count anti-commuting pairs
    int anticommute_count = 0;
    size_t n = std::min(paulis.size(), other.paulis.size());

    for (size_t i = 0; i < n; ++i) {
        Pauli a = paulis[i];
        Pauli b = other.paulis[i];
        // Non-identity Paulis anti-commute if different (and neither is I)
        if (a != Pauli::I && b != Pauli::I && a != b) {
            ++anticommute_count;
        }
    }
    // Commutes if even number of anti-commuting pairs
    return (anticommute_count % 2) == 0;
}

size_t PauliString::weight() const {
    size_t w = 0;
    for (auto p : paulis) {
        if (p != Pauli::I) ++w;
    }
    return w;
}

std::vector<size_t> PauliString::support() const {
    std::vector<size_t> sup;
    for (size_t i = 0; i < paulis.size(); ++i) {
        if (paulis[i] != Pauli::I) {
            sup.push_back(i);
        }
    }
    return sup;
}

//==============================================================================
// Syndrome Implementation
//==============================================================================

size_t Syndrome::num_defects() const {
    size_t count = 0;
    for (int s : x_syndrome) count += (s != 0);
    for (int s : z_syndrome) count += (s != 0);
    return count;
}

std::vector<size_t> Syndrome::x_defect_indices() const {
    std::vector<size_t> defects;
    for (size_t i = 0; i < x_syndrome.size(); ++i) {
        if (x_syndrome[i] != 0) defects.push_back(i);
    }
    return defects;
}

std::vector<size_t> Syndrome::z_defect_indices() const {
    std::vector<size_t> defects;
    for (size_t i = 0; i < z_syndrome.size(); ++i) {
        if (z_syndrome[i] != 0) defects.push_back(i);
    }
    return defects;
}

}  // namespace qlret

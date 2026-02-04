#include "qec_stabilizer.h"
#include <stdexcept>
#include <algorithm>
#include <cmath>

namespace qlret {

//==============================================================================
// StabilizerCode Validation
//==============================================================================

bool StabilizerCode::validate_stabilizers() const {
    // All stabilizers must commute with each other
    const auto& x_stabs = x_stabilizers();
    const auto& z_stabs = z_stabilizers();

    // X-stabilizers commute with each other
    for (size_t i = 0; i < x_stabs.size(); ++i) {
        for (size_t j = i + 1; j < x_stabs.size(); ++j) {
            if (!x_stabs[i].commutes_with(x_stabs[j])) return false;
        }
    }

    // Z-stabilizers commute with each other
    for (size_t i = 0; i < z_stabs.size(); ++i) {
        for (size_t j = i + 1; j < z_stabs.size(); ++j) {
            if (!z_stabs[i].commutes_with(z_stabs[j])) return false;
        }
    }

    // X and Z stabilizers commute
    for (const auto& xs : x_stabs) {
        for (const auto& zs : z_stabs) {
            if (!xs.commutes_with(zs)) return false;
        }
    }

    return true;
}

bool StabilizerCode::validate_logical_operators() const {
    const auto& x_stabs = x_stabilizers();
    const auto& z_stabs = z_stabilizers();

    for (size_t k = 0; k < num_logical_qubits(); ++k) {
        const auto& lx = logical_x(k);
        const auto& lz = logical_z(k);

        // Logical operators must commute with all stabilizers
        for (const auto& xs : x_stabs) {
            if (!lx.commutes_with(xs)) return false;
            if (!lz.commutes_with(xs)) return false;
        }
        for (const auto& zs : z_stabs) {
            if (!lx.commutes_with(zs)) return false;
            if (!lz.commutes_with(zs)) return false;
        }

        // Logical X and Z must anti-commute
        if (lx.commutes_with(lz)) return false;
    }

    return true;
}

//==============================================================================
// RepetitionCode Implementation
//==============================================================================

RepetitionCode::RepetitionCode(size_t distance, bool phase_flip)
    : distance_(distance), phase_flip_(phase_flip) {
    if (distance < 3 || distance % 2 == 0) {
        throw std::invalid_argument("Repetition code distance must be odd >= 3");
    }
    generate_stabilizers();
}

void RepetitionCode::generate_stabilizers() {
    size_t n = distance_;

    // For bit-flip code: Z stabilizers (Z_i Z_{i+1})
    // For phase-flip code: X stabilizers (X_i X_{i+1})
    Pauli stab_pauli = phase_flip_ ? Pauli::X : Pauli::Z;

    std::vector<PauliString>& stabs = phase_flip_ ? x_stabs_ : z_stabs_;
    stabs.clear();

    for (size_t i = 0; i + 1 < n; ++i) {
        PauliString s(n);
        s.set(i, stab_pauli);
        s.set(i + 1, stab_pauli);
        stabs.push_back(s);
    }

    // Logical operators
    logical_x_ = PauliString(n);
    logical_z_ = PauliString(n);

    if (phase_flip_) {
        // Phase-flip code: logical Z = Z_0, logical X = X_all
        logical_z_.set(0, Pauli::Z);
        for (size_t i = 0; i < n; ++i) logical_x_.set(i, Pauli::X);
    } else {
        // Bit-flip code: logical X = X_0, logical Z = Z_all
        logical_x_.set(0, Pauli::X);
        for (size_t i = 0; i < n; ++i) logical_z_.set(i, Pauli::Z);
    }
}

const PauliString& RepetitionCode::logical_x(size_t) const {
    return logical_x_;
}

const PauliString& RepetitionCode::logical_z(size_t) const {
    return logical_z_;
}

std::pair<int, int> RepetitionCode::qubit_coords(size_t qubit) const {
    return {0, static_cast<int>(qubit)};
}

size_t RepetitionCode::qubit_at_coords(int, int col) const {
    return static_cast<size_t>(col);
}

//==============================================================================
// SurfaceCode Implementation
//==============================================================================

SurfaceCode::SurfaceCode(size_t distance) : distance_(distance) {
    if (distance < 3 || distance % 2 == 0) {
        throw std::invalid_argument("Surface code distance must be odd >= 3");
    }
    generate_lattice();
    generate_stabilizers();
    generate_logical_operators();
}

size_t SurfaceCode::num_ancilla_qubits() const {
    // For rotated surface code: (d²-1)/2 X-ancillas + (d²-1)/2 Z-ancillas
    return (distance_ * distance_ - 1);
}

void SurfaceCode::generate_lattice() {
    // Rotated surface code layout
    // Data qubits at (row, col) where (row + col) is even
    // X-ancilla at (row, col) where row is odd, col is odd
    // Z-ancilla at (row, col) where row is even, col is even (except corners)

    data_coords_.clear();
    x_ancilla_coords_.clear();
    z_ancilla_coords_.clear();

    size_t grid = grid_size();

    for (size_t r = 0; r < grid; ++r) {
        for (size_t c = 0; c < grid; ++c) {
            if (is_data_qubit(static_cast<int>(r), static_cast<int>(c))) {
                data_coords_.push_back({static_cast<int>(r), static_cast<int>(c)});
            } else if (is_x_ancilla(static_cast<int>(r), static_cast<int>(c))) {
                x_ancilla_coords_.push_back({static_cast<int>(r), static_cast<int>(c)});
            } else if (is_z_ancilla(static_cast<int>(r), static_cast<int>(c))) {
                z_ancilla_coords_.push_back({static_cast<int>(r), static_cast<int>(c)});
            }
        }
    }
}

bool SurfaceCode::is_data_qubit(int row, int col) const {
    int grid = static_cast<int>(grid_size());
    if (row < 0 || col < 0 || row >= grid || col >= grid) return false;
    return ((row + col) % 2) == 0;
}

bool SurfaceCode::is_x_ancilla(int row, int col) const {
    int grid = static_cast<int>(grid_size());
    if (row < 0 || col < 0 || row >= grid || col >= grid) return false;
    if ((row + col) % 2 == 0) return false;  // Data qubit position
    // X-ancillas on odd rows (for rotated code)
    return (row % 2) == 1;
}

bool SurfaceCode::is_z_ancilla(int row, int col) const {
    int grid = static_cast<int>(grid_size());
    if (row < 0 || col < 0 || row >= grid || col >= grid) return false;
    if ((row + col) % 2 == 0) return false;  // Data qubit position
    // Z-ancillas on even rows
    return (row % 2) == 0;
}

void SurfaceCode::generate_stabilizers() {
    size_t n_data = num_data_qubits();
    x_stabs_.clear();
    z_stabs_.clear();

    // Build map from coords to data qubit index
    auto coord_to_data_idx = [this](int r, int c) -> int {
        for (size_t i = 0; i < data_coords_.size(); ++i) {
            if (data_coords_[i].first == r && data_coords_[i].second == c) {
                return static_cast<int>(i);
            }
        }
        return -1;
    };

    // X-stabilizers: for each X-ancilla, X on adjacent data qubits
    for (const auto& [ar, ac] : x_ancilla_coords_) {
        PauliString stab(n_data);
        // Adjacent data qubits: (ar±1, ac), (ar, ac±1)
        std::vector<std::pair<int, int>> neighbors = {
            {ar - 1, ac}, {ar + 1, ac}, {ar, ac - 1}, {ar, ac + 1}
        };
        for (const auto& [nr, nc] : neighbors) {
            int idx = coord_to_data_idx(nr, nc);
            if (idx >= 0) {
                stab.set(static_cast<size_t>(idx), Pauli::X);
            }
        }
        if (stab.weight() > 0) {
            x_stabs_.push_back(stab);
        }
    }

    // Z-stabilizers: for each Z-ancilla, Z on adjacent data qubits
    for (const auto& [ar, ac] : z_ancilla_coords_) {
        PauliString stab(n_data);
        std::vector<std::pair<int, int>> neighbors = {
            {ar - 1, ac}, {ar + 1, ac}, {ar, ac - 1}, {ar, ac + 1}
        };
        for (const auto& [nr, nc] : neighbors) {
            int idx = coord_to_data_idx(nr, nc);
            if (idx >= 0) {
                stab.set(static_cast<size_t>(idx), Pauli::Z);
            }
        }
        if (stab.weight() > 0) {
            z_stabs_.push_back(stab);
        }
    }
}

void SurfaceCode::generate_logical_operators() {
    size_t n_data = num_data_qubits();

    // Logical X: horizontal chain of X operators
    logical_x_ = PauliString(n_data);
    for (size_t i = 0; i < data_coords_.size(); ++i) {
        if (data_coords_[i].first == 0) {  // Top row
            logical_x_.set(i, Pauli::X);
        }
    }

    // Logical Z: vertical chain of Z operators
    logical_z_ = PauliString(n_data);
    for (size_t i = 0; i < data_coords_.size(); ++i) {
        if (data_coords_[i].second == 0) {  // Left column
            logical_z_.set(i, Pauli::Z);
        }
    }
}

const PauliString& SurfaceCode::logical_x(size_t) const {
    return logical_x_;
}

const PauliString& SurfaceCode::logical_z(size_t) const {
    return logical_z_;
}

std::pair<int, int> SurfaceCode::qubit_coords(size_t qubit) const {
    if (qubit < data_coords_.size()) {
        return data_coords_[qubit];
    }
    return {-1, -1};
}

size_t SurfaceCode::qubit_at_coords(int row, int col) const {
    for (size_t i = 0; i < data_coords_.size(); ++i) {
        if (data_coords_[i].first == row && data_coords_[i].second == col) {
            return i;
        }
    }
    return SIZE_MAX;
}

std::vector<size_t> SurfaceCode::stabilizer_data_qubits(size_t stab_idx, bool is_x) const {
    const auto& stabs = is_x ? x_stabs_ : z_stabs_;
    if (stab_idx >= stabs.size()) return {};
    return stabs[stab_idx].support();
}

//==============================================================================
// Factory
//==============================================================================

std::unique_ptr<StabilizerCode> create_stabilizer_code(QECCodeType type, size_t distance) {
    switch (type) {
        case QECCodeType::REPETITION:
            return std::make_unique<RepetitionCode>(distance);
        case QECCodeType::SURFACE:
            return std::make_unique<SurfaceCode>(distance);
        default:
            throw std::invalid_argument("Unsupported code type");
    }
}

}  // namespace qlret

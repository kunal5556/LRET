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
    return x_ancilla_coords_.size() + z_ancilla_coords_.size();
}

void SurfaceCode::generate_lattice() {
    // Simple surface code on d x d data qubit grid
    // Data qubit index = row * d + col, coords = (row, col)

    data_coords_.clear();
    x_ancilla_coords_.clear();
    z_ancilla_coords_.clear();

    int d = static_cast<int>(distance_);

    // Data qubits on d x d grid
    for (int r = 0; r < d; ++r) {
        for (int c = 0; c < d; ++c) {
            data_coords_.push_back({r, c});
        }
    }

    // X-stabilizers (plaquettes): placed at centers of faces in a checkerboard
    // For a d x d grid, faces are at (r+0.5, c+0.5) for 0 <= r < d-1, 0 <= c < d-1
    // We use (r, c) to denote the face between data qubits (r,c), (r,c+1), (r+1,c), (r+1,c+1)
    // X-plaquettes: faces where (r + c) is even
    // Z-plaquettes: faces where (r + c) is odd
    // Plus boundary weight-2 stabilizers

    for (int r = 0; r < d - 1; ++r) {
        for (int c = 0; c < d - 1; ++c) {
            if ((r + c) % 2 == 0) {
                x_ancilla_coords_.push_back({r, c});  // face index
            } else {
                z_ancilla_coords_.push_back({r, c});  // face index
            }
        }
    }

    // Boundary stabilizers: weight-2 on edges without interior plaquettes
    // Top boundary: Z-type, touching (0, c) and (0, c+1)
    for (int c = 0; c < d - 1; ++c) {
        if (c % 2 == 1) {  // Where there's no X-plaquette at row=-1
            z_ancilla_coords_.push_back({-1, c});  // boundary marker
        }
    }
    // Bottom boundary
    for (int c = 0; c < d - 1; ++c) {
        if ((d - 1 + c) % 2 == 1) {
            z_ancilla_coords_.push_back({d - 1, c});
        }
    }
    // Left boundary: X-type
    for (int r = 0; r < d - 1; ++r) {
        if (r % 2 == 1) {
            x_ancilla_coords_.push_back({r, -1});
        }
    }
    // Right boundary
    for (int r = 0; r < d - 1; ++r) {
        if ((r + d - 1) % 2 == 1) {
            x_ancilla_coords_.push_back({r, d - 1});
        }
    }
}

bool SurfaceCode::is_data_qubit(int row, int col) const {
    int d = static_cast<int>(distance_);
    return (row >= 0 && col >= 0 && row < d && col < d);
}

bool SurfaceCode::is_x_ancilla(int row, int col) const {
    for (const auto& [r, c] : x_ancilla_coords_) {
        if (r == row && c == col) return true;
    }
    return false;
}

bool SurfaceCode::is_z_ancilla(int row, int col) const {
    for (const auto& [r, c] : z_ancilla_coords_) {
        if (r == row && c == col) return true;
    }
    return false;
}

void SurfaceCode::generate_stabilizers() {
    size_t n_data = num_data_qubits();
    int d = static_cast<int>(distance_);
    x_stabs_.clear();
    z_stabs_.clear();

    // Helper: convert (row, col) in d x d grid to data qubit index
    auto coord_to_idx = [d](int r, int c) -> int {
        if (r < 0 || c < 0 || r >= d || c >= d) return -1;
        return r * d + c;
    };

    // X-stabilizers from X-ancilla positions
    for (const auto& [fr, fc] : x_ancilla_coords_) {
        PauliString stab(n_data);
        // Each face/plaquette (fr, fc) touches 4 data qubits:
        // (fr, fc), (fr, fc+1), (fr+1, fc), (fr+1, fc+1)
        // But boundary stabilizers (fc == -1 or fc == d-1) only touch 2
        std::vector<std::pair<int, int>> corners;
        if (fc >= 0 && fc < d - 1) {
            // Interior plaquette
            corners = {{fr, fc}, {fr, fc + 1}, {fr + 1, fc}, {fr + 1, fc + 1}};
        } else if (fc == -1) {
            // Left boundary: touches (fr, 0) and (fr+1, 0)
            corners = {{fr, 0}, {fr + 1, 0}};
        } else if (fc == d - 1) {
            // Right boundary: touches (fr, d-1) and (fr+1, d-1)
            corners = {{fr, d - 1}, {fr + 1, d - 1}};
        }
        for (const auto& [cr, cc] : corners) {
            int idx = coord_to_idx(cr, cc);
            if (idx >= 0) {
                stab.set(static_cast<size_t>(idx), Pauli::X);
            }
        }
        if (stab.weight() > 0) {
            x_stabs_.push_back(stab);
        }
    }

    // Z-stabilizers from Z-ancilla positions
    for (const auto& [fr, fc] : z_ancilla_coords_) {
        PauliString stab(n_data);
        std::vector<std::pair<int, int>> corners;
        if (fr >= 0 && fr < d - 1) {
            // Interior plaquette
            corners = {{fr, fc}, {fr, fc + 1}, {fr + 1, fc}, {fr + 1, fc + 1}};
        } else if (fr == -1) {
            // Top boundary: touches (0, fc) and (0, fc+1)
            corners = {{0, fc}, {0, fc + 1}};
        } else if (fr == d - 1) {
            // Bottom boundary: touches (d-1, fc) and (d-1, fc+1)
            corners = {{d - 1, fc}, {d - 1, fc + 1}};
        }
        for (const auto& [cr, cc] : corners) {
            int idx = coord_to_idx(cr, cc);
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
    int d = static_cast<int>(distance_);
    size_t n_data = num_data_qubits();

    // Logical X: horizontal chain of X operators across the top row
    logical_x_ = PauliString(n_data);
    for (int c = 0; c < d; ++c) {
        logical_x_.set(static_cast<size_t>(0 * d + c), Pauli::X);  // row 0, all cols
    }

    // Logical Z: vertical chain of Z operators down the left column
    logical_z_ = PauliString(n_data);
    for (int r = 0; r < d; ++r) {
        logical_z_.set(static_cast<size_t>(r * d + 0), Pauli::Z);  // all rows, col 0
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

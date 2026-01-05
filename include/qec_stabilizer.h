#pragma once

/**
 * @file qec_stabilizer.h
 * @brief Phase 9.1: Stabilizer Code Implementation
 *
 * Implements surface codes and other stabilizer codes for QEC.
 */

#include "qec_types.h"
#include "types.h"
#include <vector>
#include <memory>

namespace qlret {

//==============================================================================
// Stabilizer Code Base Class
//==============================================================================

enum class QECCodeType {
    REPETITION,      ///< Simple repetition code (1D)
    SURFACE,         ///< 2D surface code (rotated)
    STEANE,          ///< [[7,1,3]] Steane code
    SHOR             ///< [[9,1,3]] Shor code
};

/**
 * @brief Abstract base class for stabilizer codes.
 */
class StabilizerCode {
public:
    virtual ~StabilizerCode() = default;

    // Code parameters
    virtual size_t num_data_qubits() const = 0;
    virtual size_t num_ancilla_qubits() const = 0;
    virtual size_t num_physical_qubits() const {
        return num_data_qubits() + num_ancilla_qubits();
    }
    virtual size_t code_distance() const = 0;
    virtual size_t num_logical_qubits() const = 0;
    virtual QECCodeType code_type() const = 0;

    // Stabilizers
    virtual const std::vector<PauliString>& x_stabilizers() const = 0;
    virtual const std::vector<PauliString>& z_stabilizers() const = 0;
    virtual size_t num_x_stabilizers() const { return x_stabilizers().size(); }
    virtual size_t num_z_stabilizers() const { return z_stabilizers().size(); }

    // Logical operators
    virtual const PauliString& logical_x(size_t idx = 0) const = 0;
    virtual const PauliString& logical_z(size_t idx = 0) const = 0;

    // Qubit layout (for visualization/partitioning)
    virtual std::pair<int, int> qubit_coords(size_t qubit) const = 0;
    virtual size_t qubit_at_coords(int row, int col) const = 0;

    // Validation
    bool validate_stabilizers() const;
    bool validate_logical_operators() const;
};

//==============================================================================
// Repetition Code
//==============================================================================

/**
 * @brief [[n,1,n]] repetition code for bit-flip or phase-flip errors.
 */
class RepetitionCode : public StabilizerCode {
public:
    explicit RepetitionCode(size_t distance, bool phase_flip = false);

    size_t num_data_qubits() const override { return distance_; }
    size_t num_ancilla_qubits() const override { return distance_ - 1; }
    size_t code_distance() const override { return distance_; }
    size_t num_logical_qubits() const override { return 1; }
    QECCodeType code_type() const override { return QECCodeType::REPETITION; }

    const std::vector<PauliString>& x_stabilizers() const override { return x_stabs_; }
    const std::vector<PauliString>& z_stabilizers() const override { return z_stabs_; }

    const PauliString& logical_x(size_t idx = 0) const override;
    const PauliString& logical_z(size_t idx = 0) const override;

    std::pair<int, int> qubit_coords(size_t qubit) const override;
    size_t qubit_at_coords(int row, int col) const override;

private:
    void generate_stabilizers();

    size_t distance_;
    bool phase_flip_;  // If true, Z stabilizers; otherwise X stabilizers
    std::vector<PauliString> x_stabs_;
    std::vector<PauliString> z_stabs_;
    PauliString logical_x_;
    PauliString logical_z_;
};

//==============================================================================
// Surface Code (Rotated)
//==============================================================================

/**
 * @brief Rotated surface code with distance d.
 *
 * Layout for d=3 (9 data qubits, 8 ancilla):
 *
 *     X - D - Z
 *     |   |   |
 *     D - X - D
 *     |   |   |
 *     Z - D - X
 *
 * D = data qubit, X = X-stabilizer ancilla, Z = Z-stabilizer ancilla
 */
class SurfaceCode : public StabilizerCode {
public:
    explicit SurfaceCode(size_t distance);

    size_t num_data_qubits() const override { return distance_ * distance_; }
    size_t num_ancilla_qubits() const override;
    size_t code_distance() const override { return distance_; }
    size_t num_logical_qubits() const override { return 1; }
    QECCodeType code_type() const override { return QECCodeType::SURFACE; }

    const std::vector<PauliString>& x_stabilizers() const override { return x_stabs_; }
    const std::vector<PauliString>& z_stabilizers() const override { return z_stabs_; }

    const PauliString& logical_x(size_t idx = 0) const override;
    const PauliString& logical_z(size_t idx = 0) const override;

    std::pair<int, int> qubit_coords(size_t qubit) const override;
    size_t qubit_at_coords(int row, int col) const override;

    // Surface code specific
    size_t grid_size() const { return 2 * distance_ - 1; }
    bool is_data_qubit(int row, int col) const;
    bool is_x_ancilla(int row, int col) const;
    bool is_z_ancilla(int row, int col) const;

    // Get data qubits in a stabilizer's support
    std::vector<size_t> stabilizer_data_qubits(size_t stab_idx, bool is_x) const;

private:
    void generate_lattice();
    void generate_stabilizers();
    void generate_logical_operators();

    size_t distance_;
    std::vector<PauliString> x_stabs_;
    std::vector<PauliString> z_stabs_;
    PauliString logical_x_;
    PauliString logical_z_;

    // Qubit index maps
    std::vector<std::pair<int, int>> data_coords_;
    std::vector<std::pair<int, int>> x_ancilla_coords_;
    std::vector<std::pair<int, int>> z_ancilla_coords_;
};

//==============================================================================
// Factory
//==============================================================================

std::unique_ptr<StabilizerCode> create_stabilizer_code(QECCodeType type, size_t distance);

}  // namespace qlret

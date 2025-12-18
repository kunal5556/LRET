#pragma once

#include "types.h"
#include <map>
#include <functional>

namespace qlret {

//==============================================================================
// Gate Matrix Generation
//==============================================================================

/**
 * @brief Get the matrix representation of a single-qubit gate
 * @param type The type of gate
 * @param params Optional parameters for parameterized gates
 * @return 2x2 complex matrix
 */
MatrixXcd get_single_qubit_gate(GateType type, const std::vector<double>& params = {});

/**
 * @brief Get the matrix representation of a two-qubit gate
 * @param type The type of gate
 * @param params Optional parameters
 * @return 4x4 complex matrix
 */
MatrixXcd get_two_qubit_gate(GateType type, const std::vector<double>& params = {});

/**
 * @brief Expand a single-qubit gate to full Hilbert space
 * @param gate 2x2 gate matrix
 * @param target Target qubit index
 * @param num_qubits Total number of qubits
 * @return (2^n x 2^n) expanded matrix
 */
MatrixXcd expand_single_gate(const MatrixXcd& gate, size_t target, size_t num_qubits);

/**
 * @brief Expand a two-qubit gate to full Hilbert space
 * @param gate 4x4 gate matrix
 * @param control Control qubit index
 * @param target Target qubit index
 * @param num_qubits Total number of qubits
 * @return (2^n x 2^n) expanded matrix
 */
MatrixXcd expand_two_qubit_gate(const MatrixXcd& gate, size_t control, size_t target, size_t num_qubits);

/**
 * @brief Apply a gate operation to a low-rank factor L
 * @param L Current low-rank factor (dim x rank)
 * @param gate_op Gate operation to apply
 * @param num_qubits Total number of qubits
 * @return Updated low-rank factor
 */
MatrixXcd apply_gate_to_L(const MatrixXcd& L, const GateOp& gate_op, size_t num_qubits);

//==============================================================================
// Noise Models (Kraus Operators)
//==============================================================================

/**
 * @brief Get Kraus operators for a noise channel
 * @param type Type of noise
 * @param probability Noise probability/strength
 * @param params Additional parameters
 * @return Vector of Kraus operators (2x2 matrices for single-qubit noise)
 */
std::vector<MatrixXcd> get_noise_kraus_operators(NoiseType type, double probability, 
                                                  const std::vector<double>& params = {});

/**
 * @brief Apply noise to low-rank factor L using Kraus operators
 * This expands the rank: if L has rank r and there are k Kraus operators,
 * the result has rank at most k*r
 * @param L Current low-rank factor
 * @param noise_op Noise operation
 * @param num_qubits Total number of qubits
 * @return Updated low-rank factor (potentially with increased rank)
 */
MatrixXcd apply_noise_to_L(const MatrixXcd& L, const NoiseOp& noise_op, size_t num_qubits);

/**
 * @brief Apply depolarizing noise channel
 * @param L Low-rank factor
 * @param qubit Target qubit
 * @param probability Depolarizing probability
 * @param num_qubits Total number of qubits
 * @return Updated L with expanded rank
 */
MatrixXcd apply_depolarizing_noise(const MatrixXcd& L, size_t qubit, double probability, size_t num_qubits);

/**
 * @brief Apply amplitude damping noise channel
 * @param L Low-rank factor
 * @param qubit Target qubit
 * @param gamma Damping parameter
 * @param num_qubits Total number of qubits
 * @return Updated L
 */
MatrixXcd apply_amplitude_damping(const MatrixXcd& L, size_t qubit, double gamma, size_t num_qubits);

/**
 * @brief Apply phase damping noise channel
 * @param L Low-rank factor
 * @param qubit Target qubit
 * @param lambda Damping parameter
 * @param num_qubits Total number of qubits
 * @return Updated L
 */
MatrixXcd apply_phase_damping(const MatrixXcd& L, size_t qubit, double lambda, size_t num_qubits);

//==============================================================================
// Batched Gate Application (for parallelism)
//==============================================================================

/**
 * @brief Apply a batch of gate operations in parallel
 * @param L Current low-rank factor
 * @param gates Vector of gate operations to apply
 * @param num_qubits Total number of qubits
 * @param batch_size Size of parallel batches
 * @return Updated low-rank factor
 */
MatrixXcd apply_gates_batched(const MatrixXcd& L, const std::vector<GateOp>& gates, 
                               size_t num_qubits, size_t batch_size);

/**
 * @brief Apply a batch of noise operations
 * @param L Current low-rank factor
 * @param noises Vector of noise operations
 * @param num_qubits Total number of qubits
 * @param batch_size Size of parallel batches
 * @return Updated low-rank factor
 */
MatrixXcd apply_noise_batched(const MatrixXcd& L, const std::vector<NoiseOp>& noises,
                               size_t num_qubits, size_t batch_size);

//==============================================================================
// Predefined Gate Maps
//==============================================================================

// Map from gate name string to GateType
extern const std::map<std::string, GateType> gate_name_to_type;

// Map from noise name string to NoiseType
extern const std::map<std::string, NoiseType> noise_name_to_type;

}  // namespace qlret

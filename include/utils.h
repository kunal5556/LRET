#pragma once

#include "types.h"
#include <random>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <sstream>

namespace qlret {

//==============================================================================
// Random Circuit Generation
//==============================================================================

/**
 * @brief Generate a random quantum sequence (circuit)
 * 
 * Creates a circuit with random gates from the gate set, optionally
 * interspersed with noise operations.
 * 
 * @param num_qubits Number of qubits
 * @param depth Circuit depth (layers of gates)
 * @param fixed_noise If true, use fixed noise probability
 * @param noise_prob Base probability for noise (used if fixed_noise=true)
 * @param seed Random seed (0 for time-based)
 * @return Generated quantum sequence
 */
QuantumSequence generate_quantum_sequences(
    size_t num_qubits,
    size_t depth,
    bool fixed_noise = false,
    double noise_prob = 0.001,
    unsigned int seed = 0
);

/**
 * @brief Generate a random Clifford circuit
 * Only uses H, S, CNOT gates (Clifford group)
 * @param num_qubits Number of qubits
 * @param depth Circuit depth
 * @param seed Random seed
 * @return Clifford circuit sequence
 */
QuantumSequence generate_clifford_circuit(
    size_t num_qubits,
    size_t depth,
    unsigned int seed = 0
);

/**
 * @brief Generate a random Clifford+T circuit
 * Uses H, S, T, CNOT gates
 * @param num_qubits Number of qubits
 * @param depth Circuit depth
 * @param t_fraction Fraction of T gates (vs other single-qubit gates)
 * @param seed Random seed
 * @return Clifford+T circuit sequence
 */
QuantumSequence generate_clifford_t_circuit(
    size_t num_qubits,
    size_t depth,
    double t_fraction = 0.3,
    unsigned int seed = 0
);

//==============================================================================
// Quantum Metrics
//==============================================================================

/**
 * @brief Compute quantum fidelity between two states (via L factors)
 * F(ρ, σ) = (Tr[sqrt(sqrt(ρ)σsqrt(ρ))])^2
 * For pure states: F = |<ψ|φ>|^2
 * @param L1 First low-rank factor
 * @param L2 Second low-rank factor
 * @return Fidelity in [0, 1]
 */
double compute_fidelity(const MatrixXcd& L1, const MatrixXcd& L2);

/**
 * @brief Compute trace distance between two states
 * D(ρ, σ) = (1/2) ||ρ - σ||_1 = (1/2) Tr[|ρ - σ|]
 * 
 * Uses O(rank²) algorithm without constructing full density matrices
 * @param L1 First low-rank factor
 * @param L2 Second low-rank factor
 * @return Trace distance in [0, 1]
 */
double compare_L_matrices_trace(const MatrixXcd& L1, const MatrixXcd& L2);

/**
 * @brief Compute Frobenius norm difference
 * ||L1 L1† - L2 L2†||_F
 * @param L1 First low-rank factor
 * @param L2 Second low-rank factor
 * @return Frobenius norm
 */
double compute_frobenius_distance(const MatrixXcd& L1, const MatrixXcd& L2);

/**
 * @brief Compute purity of a state
 * γ = Tr[ρ²]
 * @param L Low-rank factor
 * @return Purity in [1/d, 1] where d is dimension
 */
double compute_purity(const MatrixXcd& L);

/**
 * @brief Compute von Neumann entropy
 * S(ρ) = -Tr[ρ log ρ]
 * @param L Low-rank factor
 * @return Entropy (in bits if using log2)
 */
double compute_entropy(const MatrixXcd& L);

/**
 * @brief Compute variational distance (distortion)
 * Sum of absolute differences of diagonal elements
 */
double compute_variational_distance(const MatrixXcd& rho1, const MatrixXcd& rho2);

/**
 * @brief Compute variational distance from L factors
 */
double compute_variational_distance_L(const MatrixXcd& L1, const MatrixXcd& L2);

/**
 * @brief Compute Frobenius distance between density matrices
 */
double compute_frobenius_distance_rho(const MatrixXcd& rho1, const MatrixXcd& rho2);

/**
 * @brief Compute trace distance between density matrices
 */
double compute_trace_distance_rho(const MatrixXcd& rho1, const MatrixXcd& rho2);

/**
 * @brief Compute fidelity from density matrices
 */
double compute_fidelity_rho(const MatrixXcd& rho1, const MatrixXcd& rho2);

//==============================================================================
// Visualization
//==============================================================================

/**
 * @brief Print ASCII circuit diagram
 * @param num_qubits Number of qubits
 * @param sequence Quantum sequence to visualize
 * @param max_width Maximum width before wrapping (0 = no limit)
 */
void print_circuit_diagram(size_t num_qubits, const QuantumSequence& sequence, 
                           size_t max_width = 120);

/**
 * @brief Get string representation of a gate type
 * @param type Gate type
 * @return Short string representation (e.g., "H", "CX")
 */
std::string gate_type_to_string(GateType type);

/**
 * @brief Get string representation of a noise type
 * @param type Noise type
 * @return String representation (e.g., "Depol", "AmpDamp")
 */
std::string noise_type_to_string(NoiseType type);

//==============================================================================
// Timing Utilities
//==============================================================================

/**
 * @brief Get current time as formatted string
 * @return Time string in HH:MM:SS format
 */
std::string get_current_time_string();

/**
 * @brief High-resolution timer class
 */
class Timer {
public:
    Timer() : start_time_(std::chrono::high_resolution_clock::now()) {}
    
    void reset() {
        start_time_ = std::chrono::high_resolution_clock::now();
    }
    
    double elapsed_seconds() const {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double>(now - start_time_).count();
    }
    
    double elapsed_ms() const {
        return elapsed_seconds() * 1000.0;
    }
    
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time_;
};

//==============================================================================
// Initial State Preparation
//==============================================================================

/**
 * @brief Create initial |0...0> state as low-rank factor
 * @param num_qubits Number of qubits
 * @return L such that L L† = |0...0><0...0|
 */
MatrixXcd create_zero_state(size_t num_qubits);

/**
 * @brief Create maximally mixed state as low-rank factor
 * @param num_qubits Number of qubits
 * @return L such that L L† = I/2^n
 */
MatrixXcd create_maximally_mixed_state(size_t num_qubits);

/**
 * @brief Create arbitrary pure state from coefficient vector
 * @param coefficients State vector (will be normalized)
 * @return L representing the pure state
 */
MatrixXcd create_pure_state(const VectorXcd& coefficients);

//==============================================================================
// Printing Utilities
//==============================================================================

/**
 * @brief Print simulation header
 * @param num_qubits Number of qubits
 */
void print_simulation_header(size_t num_qubits);

/**
 * @brief Print simulation results summary
 * @param result Simulation result
 * @param parallel_time Time for parallel run
 * @param naive_time Time for naive run (0 to skip comparison)
 */
void print_simulation_summary(const SimResult& result, double parallel_time, 
                               double naive_time = 0.0);

}  // namespace qlret

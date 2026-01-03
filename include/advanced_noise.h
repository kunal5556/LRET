#pragma once

#include "types.h"
#include <array>
#include <deque>
#include <map>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace qlret {

// Scaling model for depth-dependent noise rates
enum class TimeScalingModel {
    LINEAR,
    EXPONENTIAL,
    POLYNOMIAL
};

// Two-qubit correlated error specification
struct CorrelatedError {
    size_t qubit_i = 0;
    size_t qubit_j = 0;
    double coupling_strength_hz = 0.0;                    // Optional ZZ rate in Hz
    std::array<std::array<double, 4>, 4> joint_probs{};   // Dense joint Pauli probabilities
    std::vector<std::tuple<int, int, double>> sparse_probs; // Non-zero entries only
    std::vector<std::string> applicable_gates;            // Gates that trigger this correlation
};

// Parameters for time-dependent noise scaling
struct TimeVaryingNoiseParams {
    double base_t1_ns = 0.0;
    double base_t2_ns = 0.0;
    double base_depol_prob = 0.0;
    TimeScalingModel model = TimeScalingModel::LINEAR;
    double alpha = 0.0;   // Linear or exponential slope
    double beta = 0.0;    // Quadratic term for polynomial
    size_t max_depth = 1; // Normalization factor (avoid div-by-zero)
};

// Memory rule describing non-Markovian modification to subsequent gates
struct MemoryEffect {
    std::string prev_gate_type;               // Trigger gate type
    std::vector<size_t> prev_qubits;          // Trigger qubits (empty = any)
    std::string affected_gate_type;           // Gate type modified by this rule ("any" allowed)
    std::vector<size_t> affected_qubits;      // Qubits affected (empty = any/overlap)
    double error_scale_factor = 1.0;          // Multiplicative factor applied to noise rates
    size_t memory_depth = 1;                  // How many past gates to consider
};

// Rolling state for memory-aware simulation
struct CircuitMemoryState {
    std::deque<std::tuple<std::string, std::vector<size_t>, size_t>> gate_history; // name, qubits, depth
    size_t max_memory_depth = 2;
    size_t current_depth = 0;
    std::map<size_t, std::string> last_gate_per_qubit; // Fast path when memory_depth == 1
};

// Compute depth-dependent scaling factor for a base probability or rate
double compute_time_scaled_rate(double base_rate, size_t depth, const TimeVaryingNoiseParams& params);

// Evaluate multiplicative scaling due to memory effects
double evaluate_memory_scale(const CircuitMemoryState& memory_state,
                             const std::string& gate_name,
                             const std::vector<size_t>& qubits,
                             const std::vector<MemoryEffect>& effects);

// Append gate info into memory history (bounded by max_memory_depth)
void append_memory_history(CircuitMemoryState& memory_state,
                           const std::string& gate_name,
                           const std::vector<size_t>& qubits);

// Sample a Pauli pair index (0=I,1=X,2=Y,3=Z) from correlated probabilities
std::pair<int, int> sample_correlated_pauli(const CorrelatedError& error, double random01);

// Apply correlated Pauli channel to low-rank factor (two-qubit)
MatrixXcd apply_correlated_pauli_channel(const MatrixXcd& L,
                                         size_t qubit_i,
                                         size_t qubit_j,
                                         const std::vector<double>& joint_probs,
                                         size_t num_qubits);

} // namespace qlret

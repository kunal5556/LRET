#pragma once

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <complex>
#include <vector>
#include <string>
#include <variant>
#include <optional>

namespace qlret {

// Type aliases for convenience
using Complex = std::complex<double>;
using MatrixXcd = Eigen::MatrixXcd;
using VectorXcd = Eigen::VectorXcd;
using MatrixXd = Eigen::MatrixXd;
using VectorXd = Eigen::VectorXd;

// Constants
constexpr double PI = 3.14159265358979323846;
constexpr double SQRT2 = 1.41421356237309504880;
constexpr double INV_SQRT2 = 0.70710678118654752440;

// Gate types enumeration
enum class GateType {
    // Single-qubit gates
    H,      // Hadamard
    X,      // Pauli-X
    Y,      // Pauli-Y
    Z,      // Pauli-Z
    S,      // Phase gate (sqrt(Z))
    T,      // T gate (sqrt(S))
    Sdg,    // S dagger
    Tdg,    // T dagger
    SX,     // sqrt(X)
    RX,     // Rotation around X
    RY,     // Rotation around Y
    RZ,     // Rotation around Z
    U1,     // U1 gate
    U2,     // U2 gate
    U3,     // U3 gate
    
    // Two-qubit gates
    CNOT,   // Controlled-NOT (CX)
    CZ,     // Controlled-Z
    CY,     // Controlled-Y
    SWAP,   // SWAP gate
    ISWAP,  // iSWAP gate
    
    // Custom
    CUSTOM
};

// Noise types enumeration
enum class NoiseType {
    DEPOLARIZING,
    AMPLITUDE_DAMPING,
    PHASE_DAMPING,
    BIT_FLIP,
    PHASE_FLIP,
    BIT_PHASE_FLIP,
    THERMAL,
    CUSTOM
};

// Structure representing a gate operation
struct GateOp {
    GateType type;
    std::vector<size_t> qubits;  // Target qubits (1 for single, 2 for two-qubit)
    std::vector<double> params;  // Optional parameters (e.g., rotation angles)
    std::optional<MatrixXcd> custom_matrix;  // For custom gates
    
    GateOp(GateType t, std::vector<size_t> q, std::vector<double> p = {})
        : type(t), qubits(std::move(q)), params(std::move(p)) {}
    
    GateOp(GateType t, size_t q, std::vector<double> p = {})
        : type(t), qubits({q}), params(std::move(p)) {}
    
    GateOp(GateType t, size_t q1, size_t q2)
        : type(t), qubits({q1, q2}) {}
};

// Structure representing a noise operation
struct NoiseOp {
    NoiseType type;
    std::vector<size_t> qubits;
    double probability;  // Noise probability/strength
    std::vector<double> params;  // Additional parameters
    
    NoiseOp(NoiseType t, std::vector<size_t> q, double prob, std::vector<double> p = {})
        : type(t), qubits(std::move(q)), probability(prob), params(std::move(p)) {}
    
    NoiseOp(NoiseType t, size_t q, double prob, std::vector<double> p = {})
        : type(t), qubits({q}), probability(prob), params(std::move(p)) {}
};

// Union type for sequence elements
using SequenceElement = std::variant<GateOp, NoiseOp>;

// Quantum sequence (circuit)
struct QuantumSequence {
    size_t num_qubits;
    size_t depth;
    std::vector<SequenceElement> operations;
    double total_noise_probability;
    
    QuantumSequence(size_t n = 0) 
        : num_qubits(n), depth(0), total_noise_probability(0.0) {}
    
    void add_gate(const GateOp& gate) {
        operations.push_back(gate);
    }
    
    void add_noise(const NoiseOp& noise) {
        operations.push_back(noise);
        total_noise_probability += noise.probability;
    }
    
    size_t size() const { return operations.size(); }
};

// Simulation configuration
struct SimConfig {
    bool verbose = false;
    bool do_truncation = true;
    double truncation_threshold = 1e-4;
    size_t max_rank = 0;  // 0 means no limit
    size_t batch_size = 64;
    bool use_parallel = true;
    
    SimConfig() = default;
    
    SimConfig& set_verbose(bool v) { verbose = v; return *this; }
    SimConfig& set_truncation(bool t) { do_truncation = t; return *this; }
    SimConfig& set_threshold(double th) { truncation_threshold = th; return *this; }
    SimConfig& set_max_rank(size_t r) { max_rank = r; return *this; }
    SimConfig& set_batch_size(size_t b) { batch_size = b; return *this; }
    SimConfig& set_parallel(bool p) { use_parallel = p; return *this; }
};

// Simulation result
struct SimResult {
    MatrixXcd L_final;          // Final low-rank factor
    double simulation_time;      // Time in seconds
    size_t final_rank;          // Final rank of L
    size_t max_rank_reached;    // Maximum rank during simulation
    size_t truncation_count;    // Number of truncations performed
    
    SimResult() : simulation_time(0.0), final_rank(0), max_rank_reached(0), truncation_count(0) {}
};

}  // namespace qlret

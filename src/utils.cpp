#include "utils.h"
#include "gates_and_noise.h"
#include <algorithm>
#include <numeric>
#include <ctime>

namespace qlret {

//==============================================================================
// Random Circuit Generation
//==============================================================================

QuantumSequence generate_quantum_sequences(
    size_t num_qubits,
    size_t depth,
    bool fixed_noise,
    double noise_prob,
    unsigned int seed
) {
    if (seed == 0) {
        seed = static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count());
    }
    std::mt19937 rng(seed);
    std::uniform_real_distribution<> uniform(0.0, 1.0);
    std::uniform_int_distribution<size_t> qubit_dist(0, num_qubits - 1);
    
    // Available single-qubit gates
    std::vector<GateType> single_gates = {
        GateType::H, GateType::X, GateType::Y, GateType::Z,
        GateType::S, GateType::T, GateType::SX
    };
    
    // Available two-qubit gates
    std::vector<GateType> two_gates = {
        GateType::CNOT, GateType::CZ
    };
    
    // Available noise types
    std::vector<NoiseType> noise_types = {
        NoiseType::DEPOLARIZING, NoiseType::AMPLITUDE_DAMPING, NoiseType::PHASE_DAMPING
    };
    
    QuantumSequence sequence(num_qubits);
    sequence.depth = depth;
    
    for (size_t d = 0; d < depth; ++d) {
        // Layer of single-qubit gates (on random qubits)
        size_t num_single_gates = num_qubits / 2 + 1;
        std::vector<size_t> qubits_used;
        
        for (size_t i = 0; i < num_single_gates; ++i) {
            size_t qubit = qubit_dist(rng);
            
            // Avoid using same qubit twice in one layer
            if (std::find(qubits_used.begin(), qubits_used.end(), qubit) != qubits_used.end()) {
                continue;
            }
            qubits_used.push_back(qubit);
            
            size_t gate_idx = rng() % single_gates.size();
            sequence.add_gate(GateOp(single_gates[gate_idx], qubit));
        }
        
        // Layer of two-qubit gates
        if (num_qubits >= 2) {
            size_t num_two_gates = (num_qubits - 1) / 2;
            for (size_t i = 0; i < num_two_gates; ++i) {
                size_t q1 = qubit_dist(rng);
                size_t q2 = qubit_dist(rng);
                while (q2 == q1) {
                    q2 = qubit_dist(rng);
                }
                
                size_t gate_idx = rng() % two_gates.size();
                sequence.add_gate(GateOp(two_gates[gate_idx], q1, q2));
            }
        }
        
        // Add noise with some probability
        if (fixed_noise) {
            // Add noise to random qubits
            for (size_t q = 0; q < num_qubits; ++q) {
                if (uniform(rng) < noise_prob) {
                    size_t noise_idx = rng() % noise_types.size();
                    double prob = noise_prob * uniform(rng);
                    sequence.add_noise(NoiseOp(noise_types[noise_idx], q, prob));
                }
            }
        }
    }
    
    return sequence;
}

QuantumSequence generate_clifford_circuit(
    size_t num_qubits,
    size_t depth,
    unsigned int seed
) {
    if (seed == 0) {
        seed = static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count());
    }
    std::mt19937 rng(seed);
    std::uniform_int_distribution<size_t> qubit_dist(0, num_qubits - 1);
    
    std::vector<GateType> clifford_gates = {GateType::H, GateType::S};
    
    QuantumSequence sequence(num_qubits);
    sequence.depth = depth;
    
    for (size_t d = 0; d < depth; ++d) {
        // Single-qubit Clifford gates
        for (size_t q = 0; q < num_qubits; ++q) {
            if (rng() % 2 == 0) {
                size_t gate_idx = rng() % clifford_gates.size();
                sequence.add_gate(GateOp(clifford_gates[gate_idx], q));
            }
        }
        
        // CNOT layer
        if (num_qubits >= 2) {
            for (size_t q = 0; q < num_qubits - 1; q += 2) {
                if (rng() % 2 == 0) {
                    sequence.add_gate(GateOp(GateType::CNOT, q, q + 1));
                }
            }
        }
    }
    
    return sequence;
}

QuantumSequence generate_clifford_t_circuit(
    size_t num_qubits,
    size_t depth,
    double t_fraction,
    unsigned int seed
) {
    if (seed == 0) {
        seed = static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count());
    }
    std::mt19937 rng(seed);
    std::uniform_real_distribution<> uniform(0.0, 1.0);
    std::uniform_int_distribution<size_t> qubit_dist(0, num_qubits - 1);
    
    QuantumSequence sequence(num_qubits);
    sequence.depth = depth;
    
    for (size_t d = 0; d < depth; ++d) {
        // Single-qubit gates
        for (size_t q = 0; q < num_qubits; ++q) {
            double r = uniform(rng);
            if (r < t_fraction) {
                sequence.add_gate(GateOp(GateType::T, q));
            } else if (r < t_fraction + 0.3) {
                sequence.add_gate(GateOp(GateType::H, q));
            } else if (r < t_fraction + 0.5) {
                sequence.add_gate(GateOp(GateType::S, q));
            }
        }
        
        // CNOT layer
        if (num_qubits >= 2) {
            for (size_t q = 0; q < num_qubits - 1; q += 2) {
                if (uniform(rng) < 0.5) {
                    sequence.add_gate(GateOp(GateType::CNOT, q, q + 1));
                }
            }
        }
    }
    
    return sequence;
}

//==============================================================================
// Quantum Metrics
//==============================================================================

double compute_fidelity(const MatrixXcd& L1, const MatrixXcd& L2) {
    // For low-rank factors, compute overlap
    // If both are pure states (rank 1), F = |<L1|L2>|^2
    // For mixed states, use general formula via reconstructed density matrices
    
    if (L1.cols() == 1 && L2.cols() == 1) {
        // Pure state fidelity
        Complex overlap = L1.col(0).adjoint() * L2.col(0);
        return std::norm(overlap);
    }
    
    // General case: reconstruct and compute
    // This is expensive, but accurate
    MatrixXcd rho1 = L1 * L1.adjoint();
    MatrixXcd rho2 = L2 * L2.adjoint();
    
    // F = (Tr[sqrt(sqrt(rho1) * rho2 * sqrt(rho1))])^2
    // Compute sqrt(rho1)
    Eigen::SelfAdjointEigenSolver<MatrixXcd> solver1(rho1);
    MatrixXcd sqrt_rho1 = solver1.operatorSqrt();
    
    MatrixXcd inner = sqrt_rho1 * rho2 * sqrt_rho1;
    Eigen::SelfAdjointEigenSolver<MatrixXcd> solver2(inner);
    MatrixXcd sqrt_inner = solver2.operatorSqrt();
    
    double trace = sqrt_inner.trace().real();
    return trace * trace;
}

double compare_L_matrices_trace(const MatrixXcd& L1, const MatrixXcd& L2) {
    // Trace distance D(ρ, σ) = (1/2) ||ρ - σ||_1
    // For low-rank: use eigenvalues of ρ - σ
    
    // Construct difference efficiently
    // ρ - σ = L1 L1† - L2 L2†
    
    size_t dim = L1.rows();
    size_t r1 = L1.cols();
    size_t r2 = L2.cols();
    
    // If ranks are small, use efficient computation
    if (r1 + r2 < dim / 2) {
        // Form combined low-rank representation
        // [L1, L2] and compute eigenvalues of Gram-like matrix
        MatrixXcd combined(dim, r1 + r2);
        combined.leftCols(r1) = L1;
        combined.rightCols(r2) = L2;
        
        // Gram matrix
        MatrixXcd G = combined.adjoint() * combined;
        
        // Build the "signed" Gram matrix for ρ - σ
        MatrixXcd S = MatrixXcd::Zero(r1 + r2, r1 + r2);
        S.topLeftCorner(r1, r1) = MatrixXcd::Identity(r1, r1);
        S.bottomRightCorner(r2, r2) = -MatrixXcd::Identity(r2, r2);
        
        MatrixXcd M = G * S;
        
        Eigen::ComplexEigenSolver<MatrixXcd> solver(M);
        VectorXcd eigenvalues = solver.eigenvalues();
        
        double trace_dist = 0.0;
        for (int i = 0; i < eigenvalues.size(); ++i) {
            trace_dist += std::abs(eigenvalues(i).real());
        }
        
        return 0.5 * trace_dist;
    }
    
    // Fall back to full reconstruction
    MatrixXcd diff = L1 * L1.adjoint() - L2 * L2.adjoint();
    Eigen::SelfAdjointEigenSolver<MatrixXcd> solver(diff);
    VectorXd eigenvalues = solver.eigenvalues().real();
    
    double trace_dist = 0.0;
    for (int i = 0; i < eigenvalues.size(); ++i) {
        trace_dist += std::abs(eigenvalues(i));
    }
    
    return 0.5 * trace_dist;
}

double compute_frobenius_distance(const MatrixXcd& L1, const MatrixXcd& L2) {
    // ||ρ1 - ρ2||_F^2 = Tr[(ρ1 - ρ2)²]
    // = Tr[ρ1²] + Tr[ρ2²] - 2 Tr[ρ1 ρ2]
    
    // Tr[ρ1²] = Tr[(L1 L1†)²] = ||L1† L1||_F²
    MatrixXcd G1 = L1.adjoint() * L1;
    MatrixXcd G2 = L2.adjoint() * L2;
    MatrixXcd G12 = L1.adjoint() * L2;
    
    double tr_rho1_sq = (G1 * G1).trace().real();
    double tr_rho2_sq = (G2 * G2).trace().real();
    double tr_rho1_rho2 = (G12 * G12.adjoint()).trace().real();
    
    return std::sqrt(tr_rho1_sq + tr_rho2_sq - 2.0 * tr_rho1_rho2);
}

double compute_purity(const MatrixXcd& L) {
    // Purity γ = Tr[ρ²] = ||L† L||_F²
    MatrixXcd G = L.adjoint() * L;
    return (G * G).trace().real();
}

double compute_entropy(const MatrixXcd& L) {
    // S(ρ) = -Tr[ρ log ρ] = -Σ λ_i log λ_i
    // Use Gram matrix eigenvalues (same as ρ eigenvalues for nonzero ones)
    
    MatrixXcd G = L.adjoint() * L;
    Eigen::SelfAdjointEigenSolver<MatrixXcd> solver(G);
    VectorXd eigenvalues = solver.eigenvalues().real();
    
    double entropy = 0.0;
    for (int i = 0; i < eigenvalues.size(); ++i) {
        double lambda = eigenvalues(i);
        if (lambda > 1e-15) {
            entropy -= lambda * std::log2(lambda);
        }
    }
    
    return entropy;
}

double compute_variational_distance(const MatrixXcd& rho1, const MatrixXcd& rho2) {
    // Sum of absolute differences of diagonal elements
    double dist = 0.0;
    for (size_t i = 0; i < static_cast<size_t>(rho1.rows()); ++i) {
        dist += std::abs(rho1(i, i).real() - rho2(i, i).real());
    }
    return dist;
}

double compute_variational_distance_L(const MatrixXcd& L1, const MatrixXcd& L2) {
    MatrixXcd rho1 = L1 * L1.adjoint();
    MatrixXcd rho2 = L2 * L2.adjoint();
    return compute_variational_distance(rho1, rho2);
}

double compute_frobenius_distance_rho(const MatrixXcd& rho1, const MatrixXcd& rho2) {
    return (rho1 - rho2).norm();
}

double compute_trace_distance_rho(const MatrixXcd& rho1, const MatrixXcd& rho2) {
    MatrixXcd diff = rho1 - rho2;
    Eigen::SelfAdjointEigenSolver<MatrixXcd> solver(diff);
    VectorXd eigenvalues = solver.eigenvalues().real();
    
    double trace_dist = 0.0;
    for (int i = 0; i < eigenvalues.size(); ++i) {
        trace_dist += std::abs(eigenvalues(i));
    }
    return 0.5 * trace_dist;
}

double compute_fidelity_rho(const MatrixXcd& rho1, const MatrixXcd& rho2) {
    // F = (Tr[sqrt(sqrt(rho1) * rho2 * sqrt(rho1))])^2
    Eigen::SelfAdjointEigenSolver<MatrixXcd> solver1(rho1);
    MatrixXcd sqrt_rho1 = solver1.operatorSqrt();
    
    MatrixXcd inner = sqrt_rho1 * rho2 * sqrt_rho1;
    Eigen::SelfAdjointEigenSolver<MatrixXcd> solver2(inner);
    MatrixXcd sqrt_inner = solver2.operatorSqrt();
    
    double trace = sqrt_inner.trace().real();
    return trace * trace;
}

//==============================================================================
// Visualization
//==============================================================================

std::string gate_type_to_string(GateType type) {
    switch (type) {
        case GateType::H: return "H";
        case GateType::X: return "X";
        case GateType::Y: return "Y";
        case GateType::Z: return "Z";
        case GateType::S: return "S";
        case GateType::T: return "T";
        case GateType::Sdg: return "S†";
        case GateType::Tdg: return "T†";
        case GateType::SX: return "√X";
        case GateType::RX: return "Rx";
        case GateType::RY: return "Ry";
        case GateType::RZ: return "Rz";
        case GateType::U1: return "U1";
        case GateType::U2: return "U2";
        case GateType::U3: return "U3";
        case GateType::CNOT: return "CX";
        case GateType::CZ: return "CZ";
        case GateType::CY: return "CY";
        case GateType::SWAP: return "SW";
        case GateType::ISWAP: return "iS";
        default: return "??";
    }
}

std::string noise_type_to_string(NoiseType type) {
    switch (type) {
        case NoiseType::DEPOLARIZING: return "Dep";
        case NoiseType::AMPLITUDE_DAMPING: return "AD";
        case NoiseType::PHASE_DAMPING: return "PD";
        case NoiseType::BIT_FLIP: return "BF";
        case NoiseType::PHASE_FLIP: return "PF";
        case NoiseType::BIT_PHASE_FLIP: return "BPF";
        case NoiseType::THERMAL: return "Th";
        default: return "??";
    }
}

void print_circuit_diagram(size_t num_qubits, const QuantumSequence& sequence, size_t max_width) {
    // Create wire lines
    std::vector<std::string> wires(num_qubits);
    for (size_t q = 0; q < num_qubits; ++q) {
        wires[q] = "q" + std::to_string(q) + ": ";
    }
    
    // Track column position
    size_t col = 4;
    
    for (const auto& op : sequence.operations) {
        if (std::holds_alternative<GateOp>(op)) {
            const auto& gate = std::get<GateOp>(op);
            std::string symbol = gate_type_to_string(gate.type);
            
            if (gate.qubits.size() == 1) {
                size_t q = gate.qubits[0];
                // Pad other wires
                for (size_t i = 0; i < num_qubits; ++i) {
                    if (i == q) {
                        wires[i] += "-[" + symbol + "]-";
                    } else {
                        wires[i] += std::string(symbol.length() + 4, '-');
                    }
                }
            } else if (gate.qubits.size() == 2) {
                size_t q1 = gate.qubits[0];  // Control
                size_t q2 = gate.qubits[1];  // Target
                size_t min_q = std::min(q1, q2);
                size_t max_q = std::max(q1, q2);
                
                for (size_t i = 0; i < num_qubits; ++i) {
                    if (i == q1) {
                        wires[i] += "--*--";  // ASCII control dot
                    } else if (i == q2) {
                        wires[i] += "-[" + symbol.substr(1) + "]-";
                    } else if (i > min_q && i < max_q) {
                        wires[i] += "--+--";  // ASCII vertical connector
                    } else {
                        wires[i] += "-----";
                    }
                }
            }
            col += 5;
        } else {
            const auto& noise = std::get<NoiseOp>(op);
            std::string symbol = "~" + noise_type_to_string(noise.type);
            size_t q = noise.qubits[0];
            
            for (size_t i = 0; i < num_qubits; ++i) {
                if (i == q) {
                    wires[i] += "-" + symbol + "-";
                } else {
                    wires[i] += std::string(symbol.length() + 2, '-');
                }
            }
            col += symbol.length() + 2;
        }
        
        // Wrap if too wide
        if (max_width > 0 && col > max_width) {
            for (const auto& wire : wires) {
                std::cout << wire << std::endl;
            }
            std::cout << std::endl;
            
            for (size_t q = 0; q < num_qubits; ++q) {
                wires[q] = "   : ";
            }
            col = 5;
        }
    }
    
    // Print remaining
    for (const auto& wire : wires) {
        std::cout << wire << std::endl;
    }
}

//==============================================================================
// Timing Utilities
//==============================================================================

std::string get_current_time_string() {
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm* local_time = std::localtime(&now_time);
    
    std::ostringstream oss;
    oss << std::setfill('0') << std::setw(2) << local_time->tm_hour << ":"
        << std::setfill('0') << std::setw(2) << local_time->tm_min << ":"
        << std::setfill('0') << std::setw(2) << local_time->tm_sec;
    
    return oss.str();
}

//==============================================================================
// Initial State Preparation
//==============================================================================

MatrixXcd create_zero_state(size_t num_qubits) {
    size_t dim = 1ULL << num_qubits;
    MatrixXcd L(dim, 1);
    L.setZero();
    L(0, 0) = 1.0;
    return L;
}

MatrixXcd create_maximally_mixed_state(size_t num_qubits) {
    size_t dim = 1ULL << num_qubits;
    double norm = 1.0 / std::sqrt(static_cast<double>(dim));
    
    MatrixXcd L(dim, dim);
    L = MatrixXcd::Identity(dim, dim) * norm;
    return L;
}

MatrixXcd create_pure_state(const VectorXcd& coefficients) {
    VectorXcd normalized = coefficients / coefficients.norm();
    MatrixXcd L(normalized.size(), 1);
    L.col(0) = normalized;
    return L;
}

//==============================================================================
// Printing Utilities
//==============================================================================

void print_simulation_header(size_t num_qubits) {
    std::cout << std::string(90, '=') << std::endl;
    std::cout << "Running LRET simulation for " << num_qubits << " qubits" << std::endl;
    std::cout << std::string(90, '=') << std::endl;
}

void print_simulation_summary(const SimResult& result, double parallel_time, double naive_time) {
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Simulation Time: " << parallel_time << " seconds" << std::endl;
    std::cout << "Final Rank: " << result.final_rank << std::endl;
    
    if (naive_time > 0.0) {
        double speedup = naive_time / parallel_time;
        std::cout << "Naive Time: " << naive_time << " seconds" << std::endl;
        std::cout << "Speedup: " << speedup << "x" << std::endl;
    }
}

}  // namespace qlret

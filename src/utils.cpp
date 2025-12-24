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
    // Uhlmann Fidelity: F(ρ, σ) = (Tr[sqrt(sqrt(ρ)σsqrt(ρ))])^2
    // For pure states (rank 1): F = |<ψ|φ>|^2
    // For mixed states: use general formula via reconstructed density matrices
    
    if (L1.cols() == 1 && L2.cols() == 1) {
        // Pure state fidelity - simple overlap
        Complex overlap = L1.col(0).adjoint() * L2.col(0);
        return std::norm(overlap);
    }
    
    // For very large matrices, use approximate fidelity
    size_t dim = L1.rows();
    if (dim > 16384) {  // 2^14 = 16384, beyond this full reconstruction is expensive
        // Use lower bound approximation: F ≥ |Tr(L1† L2)|² / (||L1||² ||L2||²)
        Complex overlap = (L1.adjoint() * L2).trace();
        double norm1_sq = L1.squaredNorm();
        double norm2_sq = L2.squaredNorm();
        double denom = norm1_sq * norm2_sq;
        if (denom < 1e-30) return 0.0;
        return std::norm(overlap) / denom;
    }
    
    // General case: reconstruct density matrices and compute
    MatrixXcd rho1 = L1 * L1.adjoint();
    MatrixXcd rho2 = L2 * L2.adjoint();
    
    return compute_fidelity_rho(rho1, rho2);
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

double compute_linear_entropy(const MatrixXcd& L) {
    // Linear entropy: S_L = 1 - Tr[ρ²] = 1 - purity
    // For pure states: S_L = 0
    // For maximally mixed (dim d): S_L = 1 - 1/d
    return 1.0 - compute_purity(L);
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

double compute_relative_entropy(const MatrixXcd& L1, const MatrixXcd& L2) {
    // Relative entropy: S(ρ||σ) = Tr[ρ(log ρ - log σ)]
    // = Tr[ρ log ρ] - Tr[ρ log σ]
    // = -S(ρ) - Tr[ρ log σ]
    //
    // Only defined when supp(ρ) ⊆ supp(σ), otherwise +∞
    
    // For efficiency, work with eigendecompositions
    MatrixXcd G1 = L1.adjoint() * L1;  // Same eigenvalues as ρ
    MatrixXcd G2 = L2.adjoint() * L2;  // Same eigenvalues as σ
    
    Eigen::SelfAdjointEigenSolver<MatrixXcd> solver1(G1);
    Eigen::SelfAdjointEigenSolver<MatrixXcd> solver2(G2);
    
    VectorXd eigvals1 = solver1.eigenvalues().real();
    VectorXd eigvals2 = solver2.eigenvalues().real();
    
    // Check support condition: if ρ has non-zero eigenvalue where σ is zero
    // For small dimensions, compute full density matrices
    size_t dim = L1.rows();
    if (dim > 1024) {
        // Too large for full reconstruction - return approximation
        // using von Neumann entropies
        double S1 = compute_entropy(L1);
        double S2 = compute_entropy(L2);
        return std::max(0.0, S1 - S2);  // Very rough approximation
    }
    
    MatrixXcd rho = L1 * L1.adjoint();
    MatrixXcd sigma = L2 * L2.adjoint();
    
    // Compute log(σ) using eigendecomposition
    Eigen::SelfAdjointEigenSolver<MatrixXcd> sigma_solver(sigma);
    VectorXd sigma_eigvals = sigma_solver.eigenvalues().real();
    MatrixXcd sigma_eigvecs = sigma_solver.eigenvectors();
    
    // Check if support of ρ is within support of σ
    for (int i = 0; i < sigma_eigvals.size(); ++i) {
        if (sigma_eigvals(i) < 1e-15) {
            // Check if ρ has weight on this eigenvector
            Complex weight = (sigma_eigvecs.col(i).adjoint() * rho * sigma_eigvecs.col(i))(0, 0);
            if (std::abs(weight) > 1e-10) {
                return std::numeric_limits<double>::infinity();
            }
        }
    }
    
    // Compute log(σ)
    MatrixXcd log_sigma = MatrixXcd::Zero(dim, dim);
    for (int i = 0; i < sigma_eigvals.size(); ++i) {
        if (sigma_eigvals(i) > 1e-15) {
            log_sigma += std::log2(sigma_eigvals(i)) * 
                         sigma_eigvecs.col(i) * sigma_eigvecs.col(i).adjoint();
        }
    }
    
    // S(ρ||σ) = -S(ρ) - Tr[ρ log σ]
    double S_rho = compute_entropy(L1);
    double tr_rho_log_sigma = (rho * log_sigma).trace().real();
    
    return -S_rho - tr_rho_log_sigma;
}

double compute_concurrence(const MatrixXcd& L) {
    // Concurrence for 2-qubit states only
    // C(ρ) = max(0, λ₁ - λ₂ - λ₃ - λ₄)
    
    size_t dim = L.rows();
    if (dim != 4) {
        return -1.0;  // Not a 2-qubit state
    }
    
    // Construct full density matrix
    MatrixXcd rho = L * L.adjoint();
    
    // Pauli Y matrix
    MatrixXcd sigma_y(2, 2);
    sigma_y << Complex(0, 0), Complex(0, -1),
               Complex(0, 1), Complex(0, 0);
    
    // σy ⊗ σy (4x4 tensor product)
    MatrixXcd sigma_yy = MatrixXcd::Zero(4, 4);
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            for (int k = 0; k < 2; ++k) {
                for (int l = 0; l < 2; ++l) {
                    sigma_yy(2*i + k, 2*j + l) = sigma_y(i, j) * sigma_y(k, l);
                }
            }
        }
    }
    
    // ρ̃ = (σy ⊗ σy) ρ* (σy ⊗ σy)
    MatrixXcd rho_tilde = sigma_yy * rho.conjugate() * sigma_yy;
    
    // R = sqrt(sqrt(ρ) * ρ̃ * sqrt(ρ))
    // Simpler: eigenvalues of ρ * ρ̃ give λᵢ²
    MatrixXcd R = rho * rho_tilde;
    
    Eigen::ComplexEigenSolver<MatrixXcd> solver(R);
    VectorXcd eigenvalues = solver.eigenvalues();
    
    // Get real parts and sort in decreasing order
    std::vector<double> lambdas(4);
    for (int i = 0; i < 4; ++i) {
        double val = eigenvalues(i).real();
        lambdas[i] = (val > 0) ? std::sqrt(val) : 0.0;
    }
    std::sort(lambdas.begin(), lambdas.end(), std::greater<double>());
    
    // C = max(0, λ₁ - λ₂ - λ₃ - λ₄)
    double concurrence = lambdas[0] - lambdas[1] - lambdas[2] - lambdas[3];
    return std::max(0.0, concurrence);
}

double compute_negativity(const MatrixXcd& L, size_t num_qubits_A, size_t num_qubits) {
    // Negativity: N(ρ) = (||ρ^Tₐ||₁ - 1) / 2
    // where ρ^Tₐ is partial transpose over subsystem A
    
    size_t dim = L.rows();
    size_t dim_A = 1ULL << num_qubits_A;
    size_t dim_B = 1ULL << (num_qubits - num_qubits_A);
    
    if (dim != dim_A * dim_B) {
        return -1.0;  // Dimension mismatch
    }
    
    // Construct full density matrix
    MatrixXcd rho = L * L.adjoint();
    
    // Partial transpose over A
    // ρ^Tₐ[i,j,k,l] = ρ[j,i,k,l] where (i,k) are row indices and (j,l) are column indices
    // in the A⊗B basis
    MatrixXcd rho_pt = MatrixXcd::Zero(dim, dim);
    
    for (size_t i = 0; i < dim_A; ++i) {
        for (size_t j = 0; j < dim_A; ++j) {
            for (size_t k = 0; k < dim_B; ++k) {
                for (size_t l = 0; l < dim_B; ++l) {
                    // Original index: row = i*dim_B + k, col = j*dim_B + l
                    // After partial transpose on A: row = j*dim_B + k, col = i*dim_B + l
                    rho_pt(j * dim_B + k, i * dim_B + l) = rho(i * dim_B + k, j * dim_B + l);
                }
            }
        }
    }
    
    // Compute trace norm ||ρ^Tₐ||₁ = Σ|λᵢ|
    Eigen::SelfAdjointEigenSolver<MatrixXcd> solver(rho_pt);
    VectorXd eigenvalues = solver.eigenvalues().real();
    
    double trace_norm = 0.0;
    for (int i = 0; i < eigenvalues.size(); ++i) {
        trace_norm += std::abs(eigenvalues(i));
    }
    
    // Negativity = (||ρ^Tₐ||₁ - 1) / 2
    return (trace_norm - 1.0) / 2.0;
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
    // Uhlmann Fidelity: F = (Tr[sqrt(sqrt(rho1) * rho2 * sqrt(rho1))])^2
    // 
    // Numerically stable implementation that handles:
    // 1. Near-zero eigenvalues (clamp negative to zero)
    // 2. Failed eigendecomposition (return -1)
    // 3. Invalid density matrices (check trace)
    
    // Validate density matrices
    double tr1 = rho1.trace().real();
    double tr2 = rho2.trace().real();
    if (std::abs(tr1 - 1.0) > 0.5 || std::abs(tr2 - 1.0) > 0.5) {
        // Traces too far from 1 - invalid density matrices
        return -1.0;
    }
    
    // Compute sqrt(rho1) using eigendecomposition with numerical fixes
    Eigen::SelfAdjointEigenSolver<MatrixXcd> solver1(rho1);
    if (solver1.info() != Eigen::Success) {
        return -1.0;  // Eigendecomposition failed
    }
    
    VectorXd eigvals1 = solver1.eigenvalues();
    MatrixXcd eigvecs1 = solver1.eigenvectors();
    
    // Clamp negative eigenvalues to zero (numerical fix for near-zero negatives)
    for (int i = 0; i < eigvals1.size(); ++i) {
        if (eigvals1(i) < 0) eigvals1(i) = 0;
    }
    
    // Compute sqrt(rho1) = V * sqrt(D) * V†
    VectorXd sqrt_eigvals1 = eigvals1.array().sqrt();
    MatrixXcd sqrt_rho1 = eigvecs1 * sqrt_eigvals1.asDiagonal() * eigvecs1.adjoint();
    
    // Compute inner = sqrt(rho1) * rho2 * sqrt(rho1)
    MatrixXcd inner = sqrt_rho1 * rho2 * sqrt_rho1;
    
    // Make inner Hermitian (fix numerical asymmetry)
    inner = 0.5 * (inner + inner.adjoint());
    
    // Compute sqrt(inner) using eigendecomposition
    Eigen::SelfAdjointEigenSolver<MatrixXcd> solver2(inner);
    if (solver2.info() != Eigen::Success) {
        return -1.0;  // Eigendecomposition failed
    }
    
    VectorXd eigvals2 = solver2.eigenvalues();
    
    // Sum of sqrt of positive eigenvalues (clamp negatives)
    double trace_sum = 0.0;
    for (int i = 0; i < eigvals2.size(); ++i) {
        if (eigvals2(i) > 0) {
            trace_sum += std::sqrt(eigvals2(i));
        }
    }
    
    double fidelity = trace_sum * trace_sum;
    
    // Clamp to [0, 1] range (numerical safety)
    if (fidelity < 0.0) fidelity = 0.0;
    if (fidelity > 1.0) fidelity = 1.0;
    
    return fidelity;
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

MatrixXcd create_random_mixed_state(size_t num_qubits, size_t target_rank, unsigned int seed) {
    if (seed == 0) {
        seed = static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count());
    }
    std::mt19937 rng(seed);
    std::normal_distribution<> normal(0.0, 1.0);
    
    size_t dim = 1ULL << num_qubits;
    target_rank = std::min(target_rank, dim);  // Can't exceed dimension
    
    // Create random L matrix with specified rank
    // L is dim x rank, L*L† is the density matrix
    MatrixXcd L(dim, target_rank);
    
    for (size_t i = 0; i < dim; ++i) {
        for (size_t j = 0; j < target_rank; ++j) {
            L(i, j) = Complex(normal(rng), normal(rng));
        }
    }
    
    // Normalize so trace(L*L†) = 1
    double trace = (L * L.adjoint()).trace().real();
    L /= std::sqrt(trace);
    
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

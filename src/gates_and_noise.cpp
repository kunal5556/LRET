#include "gates_and_noise.h"
#include <cmath>
#include <stdexcept>

#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace qlret {

//==============================================================================
// Gate Name Maps
//==============================================================================

const std::map<std::string, GateType> gate_name_to_type = {
    {"H", GateType::H},
    {"X", GateType::X},
    {"Y", GateType::Y},
    {"Z", GateType::Z},
    {"S", GateType::S},
    {"T", GateType::T},
    {"Sdg", GateType::Sdg},
    {"Tdg", GateType::Tdg},
    {"SX", GateType::SX},
    {"RX", GateType::RX},
    {"RY", GateType::RY},
    {"RZ", GateType::RZ},
    {"U1", GateType::U1},
    {"U2", GateType::U2},
    {"U3", GateType::U3},
    {"CNOT", GateType::CNOT},
    {"CX", GateType::CNOT},
    {"CZ", GateType::CZ},
    {"CY", GateType::CY},
    {"SWAP", GateType::SWAP},
    {"ISWAP", GateType::ISWAP}
};

const std::map<std::string, NoiseType> noise_name_to_type = {
    {"depolarizing", NoiseType::DEPOLARIZING},
    {"amplitude_damping", NoiseType::AMPLITUDE_DAMPING},
    {"phase_damping", NoiseType::PHASE_DAMPING},
    {"bit_flip", NoiseType::BIT_FLIP},
    {"phase_flip", NoiseType::PHASE_FLIP},
    {"bit_phase_flip", NoiseType::BIT_PHASE_FLIP},
    {"thermal", NoiseType::THERMAL}
};

//==============================================================================
// Single-Qubit Gate Matrices
//==============================================================================

MatrixXcd get_single_qubit_gate(GateType type, const std::vector<double>& params) {
    MatrixXcd gate(2, 2);
    Complex i(0.0, 1.0);
    
    switch (type) {
        case GateType::H:
            gate << INV_SQRT2, INV_SQRT2,
                    INV_SQRT2, -INV_SQRT2;
            break;
            
        case GateType::X:
            gate << 0, 1,
                    1, 0;
            break;
            
        case GateType::Y:
            gate << 0, -i,
                    i, 0;
            break;
            
        case GateType::Z:
            gate << 1, 0,
                    0, -1;
            break;
            
        case GateType::S:
            gate << 1, 0,
                    0, i;
            break;
            
        case GateType::T:
            gate << 1, 0,
                    0, std::exp(i * PI / 4.0);
            break;
            
        case GateType::Sdg:
            gate << 1, 0,
                    0, -i;
            break;
            
        case GateType::Tdg:
            gate << 1, 0,
                    0, std::exp(-i * PI / 4.0);
            break;
            
        case GateType::SX:
            gate << Complex(0.5, 0.5), Complex(0.5, -0.5),
                    Complex(0.5, -0.5), Complex(0.5, 0.5);
            break;
            
        case GateType::RX: {
            if (params.empty()) throw std::invalid_argument("RX requires theta parameter");
            double theta = params[0];
            double c = std::cos(theta / 2.0);
            double s = std::sin(theta / 2.0);
            gate << c, -i * s,
                    -i * s, c;
            break;
        }
        
        case GateType::RY: {
            if (params.empty()) throw std::invalid_argument("RY requires theta parameter");
            double theta = params[0];
            double c = std::cos(theta / 2.0);
            double s = std::sin(theta / 2.0);
            gate << c, -s,
                    s, c;
            break;
        }
        
        case GateType::RZ: {
            if (params.empty()) throw std::invalid_argument("RZ requires theta parameter");
            double theta = params[0];
            gate << std::exp(-i * theta / 2.0), 0,
                    0, std::exp(i * theta / 2.0);
            break;
        }
        
        case GateType::U1: {
            if (params.empty()) throw std::invalid_argument("U1 requires lambda parameter");
            double lambda = params[0];
            gate << 1, 0,
                    0, std::exp(i * lambda);
            break;
        }
        
        case GateType::U2: {
            if (params.size() < 2) throw std::invalid_argument("U2 requires phi, lambda parameters");
            double phi = params[0];
            double lambda = params[1];
            gate << INV_SQRT2, -INV_SQRT2 * std::exp(i * lambda),
                    INV_SQRT2 * std::exp(i * phi), INV_SQRT2 * std::exp(i * (phi + lambda));
            break;
        }
        
        case GateType::U3: {
            if (params.size() < 3) throw std::invalid_argument("U3 requires theta, phi, lambda parameters");
            double theta = params[0];
            double phi = params[1];
            double lambda = params[2];
            double c = std::cos(theta / 2.0);
            double s = std::sin(theta / 2.0);
            gate << c, -std::exp(i * lambda) * s,
                    std::exp(i * phi) * s, std::exp(i * (phi + lambda)) * c;
            break;
        }
        
        default:
            throw std::invalid_argument("Unknown single-qubit gate type");
    }
    
    return gate;
}

//==============================================================================
// Two-Qubit Gate Matrices
//==============================================================================

MatrixXcd get_two_qubit_gate(GateType type, const std::vector<double>& params) {
    MatrixXcd gate(4, 4);
    gate.setZero();
    Complex i(0.0, 1.0);
    
    switch (type) {
        case GateType::CNOT:
            // |00> -> |00>, |01> -> |01>, |10> -> |11>, |11> -> |10>
            gate << 1, 0, 0, 0,
                    0, 1, 0, 0,
                    0, 0, 0, 1,
                    0, 0, 1, 0;
            break;
            
        case GateType::CZ:
            gate << 1, 0, 0, 0,
                    0, 1, 0, 0,
                    0, 0, 1, 0,
                    0, 0, 0, -1;
            break;
            
        case GateType::CY:
            gate << 1, 0, 0, 0,
                    0, 1, 0, 0,
                    0, 0, 0, -i,
                    0, 0, i, 0;
            break;
            
        case GateType::SWAP:
            gate << 1, 0, 0, 0,
                    0, 0, 1, 0,
                    0, 1, 0, 0,
                    0, 0, 0, 1;
            break;
            
        case GateType::ISWAP:
            gate << 1, 0, 0, 0,
                    0, 0, i, 0,
                    0, i, 0, 0,
                    0, 0, 0, 1;
            break;
            
        default:
            throw std::invalid_argument("Unknown two-qubit gate type");
    }
    
    return gate;
}

//==============================================================================
// Optimized Gate Application (without full expansion)
//==============================================================================

// Apply single-qubit gate directly to L matrix (O(2^n * rank) instead of O(4^n))
MatrixXcd apply_single_gate_direct(const MatrixXcd& L, const MatrixXcd& gate, size_t target, size_t num_qubits) {
    size_t dim = L.rows();
    size_t rank = L.cols();
    MatrixXcd result = L;  // Copy
    
    size_t step = 1ULL << target;
    
    // Process pairs of rows that differ only in the target qubit
    #pragma omp parallel for schedule(dynamic) if(dim > 256)
    for (size_t block = 0; block < dim; block += 2 * step) {
        for (size_t i = block; i < block + step && i < dim; ++i) {
            size_t i0 = i;           // target qubit = 0
            size_t i1 = i + step;    // target qubit = 1
            
            if (i1 >= dim) continue;
            
            // For each column (rank), apply 2x2 gate
            for (size_t r = 0; r < rank; ++r) {
                Complex v0 = L(i0, r);
                Complex v1 = L(i1, r);
                
                result(i0, r) = gate(0, 0) * v0 + gate(0, 1) * v1;
                result(i1, r) = gate(1, 0) * v0 + gate(1, 1) * v1;
            }
        }
    }
    
    return result;
}

// Apply two-qubit gate directly to L matrix
// Gate matrix convention: row/col index = (q1_bit << 1) | q2_bit
// where q1 is the first qubit argument (control for CNOT) and q2 is the second
MatrixXcd apply_two_qubit_gate_direct(const MatrixXcd& L, const MatrixXcd& gate, 
                                       size_t q1, size_t q2, size_t num_qubits) {
    size_t dim = L.rows();
    size_t rank = L.cols();
    MatrixXcd result = L;
    
    // Bit steps for each qubit
    size_t step_q1 = 1ULL << q1;
    size_t step_q2 = 1ULL << q2;
    
    // For iteration, we need the min and max to skip properly
    size_t qmin = std::min(q1, q2);
    size_t qmax = std::max(q1, q2);
    size_t step_min = 1ULL << qmin;
    size_t step_max = 1ULL << qmax;
    
    for (size_t base = 0; base < dim; ++base) {
        // Only process if both qubit positions are 0
        if ((base & step_min) != 0 || (base & step_max) != 0) continue;
        
        // Four basis states indexed by gate matrix convention:
        // idx[k] where k = (q1_bit << 1) | q2_bit
        // idx[0] = q1=0, q2=0
        // idx[1] = q1=0, q2=1
        // idx[2] = q1=1, q2=0
        // idx[3] = q1=1, q2=1
        size_t idx[4];
        idx[0] = base;                          // q1=0, q2=0
        idx[1] = base | step_q2;                // q1=0, q2=1
        idx[2] = base | step_q1;                // q1=1, q2=0
        idx[3] = base | step_q1 | step_q2;      // q1=1, q2=1
        
        for (size_t r = 0; r < rank; ++r) {
            Complex v[4];
            for (int k = 0; k < 4; ++k) {
                v[k] = L(idx[k], r);
            }
            
            for (int k = 0; k < 4; ++k) {
                result(idx[k], r) = gate(k, 0) * v[0] + gate(k, 1) * v[1] + 
                                    gate(k, 2) * v[2] + gate(k, 3) * v[3];
            }
        }
    }
    
    return result;
}

//==============================================================================
// Gate Expansion to Full Hilbert Space (kept for compatibility)
//==============================================================================

MatrixXcd expand_single_gate(const MatrixXcd& gate, size_t target, size_t num_qubits) {
    size_t dim = 1ULL << num_qubits;
    MatrixXcd full_gate(dim, dim);
    full_gate.setZero();
    
    for (size_t i = 0; i < dim; ++i) {
        for (size_t j = 0; j < dim; ++j) {
            // Check if all qubits except target are the same
            size_t mask = ~(1ULL << target);
            if ((i & mask) != (j & mask)) continue;
            
            // Get the target qubit values
            size_t i_target = (i >> target) & 1;
            size_t j_target = (j >> target) & 1;
            
            full_gate(i, j) = gate(i_target, j_target);
        }
    }
    
    return full_gate;
}

MatrixXcd expand_two_qubit_gate(const MatrixXcd& gate, size_t control, size_t target, size_t num_qubits) {
    size_t dim = 1ULL << num_qubits;
    MatrixXcd full_gate(dim, dim);
    full_gate.setZero();
    
    for (size_t i = 0; i < dim; ++i) {
        for (size_t j = 0; j < dim; ++j) {
            // Check if all qubits except control and target are the same
            size_t mask = ~((1ULL << control) | (1ULL << target));
            if ((i & mask) != (j & mask)) continue;
            
            // Get the control and target qubit values
            size_t i_ctrl = (i >> control) & 1;
            size_t j_ctrl = (j >> control) & 1;
            size_t i_targ = (i >> target) & 1;
            size_t j_targ = (j >> target) & 1;
            
            // Map to 4x4 matrix indices (control is MSB)
            size_t row = (i_ctrl << 1) | i_targ;
            size_t col = (j_ctrl << 1) | j_targ;
            
            full_gate(i, j) = gate(row, col);
        }
    }
    
    return full_gate;
}

//==============================================================================
// Gate Application to Low-Rank Factor
//==============================================================================

MatrixXcd apply_gate_to_L(const MatrixXcd& L, const GateOp& gate_op, size_t num_qubits) {
    // Use optimized direct application (O(2^n * rank) instead of O(4^n))
    if (gate_op.qubits.size() == 1) {
        MatrixXcd gate_matrix = get_single_qubit_gate(gate_op.type, gate_op.params);
        return apply_single_gate_direct(L, gate_matrix, gate_op.qubits[0], num_qubits);
    } else if (gate_op.qubits.size() == 2) {
        MatrixXcd gate_matrix = get_two_qubit_gate(gate_op.type, gate_op.params);
        return apply_two_qubit_gate_direct(L, gate_matrix, gate_op.qubits[0], gate_op.qubits[1], num_qubits);
    } else {
        throw std::invalid_argument("Only 1 and 2 qubit gates supported");
    }
}

//==============================================================================
// Noise Kraus Operators
//==============================================================================

std::vector<MatrixXcd> get_noise_kraus_operators(NoiseType type, double p, 
                                                  const std::vector<double>& params) {
    std::vector<MatrixXcd> kraus;
    Complex i(0.0, 1.0);
    
    switch (type) {
        case NoiseType::DEPOLARIZING: {
            // Depolarizing channel: ρ -> (1-p)ρ + (p/3)(XρX + YρY + ZρZ)
            double c0 = std::sqrt(1.0 - p);
            double c1 = std::sqrt(p / 3.0);
            
            MatrixXcd K0(2, 2), K1(2, 2), K2(2, 2), K3(2, 2);
            K0 << c0, 0, 0, c0;
            K1 << 0, c1, c1, 0;  // sqrt(p/3) * X
            K2 << 0, -i*c1, i*c1, 0;  // sqrt(p/3) * Y
            K3 << c1, 0, 0, -c1;  // sqrt(p/3) * Z
            
            kraus = {K0, K1, K2, K3};
            break;
        }
        
        case NoiseType::AMPLITUDE_DAMPING: {
            // Amplitude damping: |1> -> |0> with probability gamma
            double gamma = p;
            MatrixXcd K0(2, 2), K1(2, 2);
            K0 << 1, 0, 0, std::sqrt(1.0 - gamma);
            K1 << 0, std::sqrt(gamma), 0, 0;
            
            kraus = {K0, K1};
            break;
        }
        
        case NoiseType::PHASE_DAMPING: {
            // Phase damping (dephasing)
            double lambda = p;
            MatrixXcd K0(2, 2), K1(2, 2);
            K0 << 1, 0, 0, std::sqrt(1.0 - lambda);
            K1 << 0, 0, 0, std::sqrt(lambda);
            
            kraus = {K0, K1};
            break;
        }
        
        case NoiseType::BIT_FLIP: {
            // Bit flip: X with probability p
            double c0 = std::sqrt(1.0 - p);
            double c1 = std::sqrt(p);
            
            MatrixXcd K0(2, 2), K1(2, 2);
            K0 << c0, 0, 0, c0;
            K1 << 0, c1, c1, 0;
            
            kraus = {K0, K1};
            break;
        }
        
        case NoiseType::PHASE_FLIP: {
            // Phase flip: Z with probability p
            double c0 = std::sqrt(1.0 - p);
            double c1 = std::sqrt(p);
            
            MatrixXcd K0(2, 2), K1(2, 2);
            K0 << c0, 0, 0, c0;
            K1 << c1, 0, 0, -c1;
            
            kraus = {K0, K1};
            break;
        }
        
        case NoiseType::BIT_PHASE_FLIP: {
            // Bit-phase flip: Y with probability p
            double c0 = std::sqrt(1.0 - p);
            double c1 = std::sqrt(p);
            
            MatrixXcd K0(2, 2), K1(2, 2);
            K0 << c0, 0, 0, c0;
            K1 << 0, -i*c1, i*c1, 0;
            
            kraus = {K0, K1};
            break;
        }
        
        case NoiseType::THERMAL: {
            // Generalized amplitude damping (thermal relaxation)
            if (params.empty()) throw std::invalid_argument("Thermal noise requires N parameter");
            double gamma = p;
            double N = params[0];  // Thermal population
            
            MatrixXcd K0(2, 2), K1(2, 2), K2(2, 2), K3(2, 2);
            double sqrt_1mN = std::sqrt(1.0 - N);
            double sqrt_N = std::sqrt(N);
            double sqrt_gamma = std::sqrt(gamma);
            double sqrt_1mg = std::sqrt(1.0 - gamma);
            
            K0 << sqrt_1mN, 0, 0, sqrt_1mN * sqrt_1mg;
            K1 << 0, sqrt_1mN * sqrt_gamma, 0, 0;
            K2 << sqrt_N * sqrt_1mg, 0, 0, sqrt_N;
            K3 << 0, 0, sqrt_N * sqrt_gamma, 0;
            
            kraus = {K0, K1, K2, K3};
            break;
        }
        
        default:
            throw std::invalid_argument("Unknown noise type");
    }
    
    return kraus;
}

//==============================================================================
// Noise Application to Low-Rank Factor
//==============================================================================

MatrixXcd apply_noise_to_L(const MatrixXcd& L, const NoiseOp& noise_op, size_t num_qubits) {
    auto kraus_ops = get_noise_kraus_operators(noise_op.type, noise_op.probability, noise_op.params);
    size_t dim = L.rows();
    size_t rank = L.cols();
    size_t num_kraus = kraus_ops.size();
    
    // New L will have rank = old_rank * num_kraus
    MatrixXcd L_new(dim, rank * num_kraus);
    
    for (size_t k = 0; k < num_kraus; ++k) {
        // Use optimized direct application instead of full expansion
        MatrixXcd L_k = apply_single_gate_direct(L, kraus_ops[k], noise_op.qubits[0], num_qubits);
        L_new.block(0, k * rank, dim, rank) = L_k;
    }
    
    return L_new;
}

MatrixXcd apply_depolarizing_noise(const MatrixXcd& L, size_t qubit, double probability, size_t num_qubits) {
    NoiseOp noise(NoiseType::DEPOLARIZING, qubit, probability);
    return apply_noise_to_L(L, noise, num_qubits);
}

MatrixXcd apply_amplitude_damping(const MatrixXcd& L, size_t qubit, double gamma, size_t num_qubits) {
    NoiseOp noise(NoiseType::AMPLITUDE_DAMPING, qubit, gamma);
    return apply_noise_to_L(L, noise, num_qubits);
}

MatrixXcd apply_phase_damping(const MatrixXcd& L, size_t qubit, double lambda, size_t num_qubits) {
    NoiseOp noise(NoiseType::PHASE_DAMPING, qubit, lambda);
    return apply_noise_to_L(L, noise, num_qubits);
}

//==============================================================================
// Batched Gate Application
//==============================================================================

MatrixXcd apply_gates_batched(const MatrixXcd& L, const std::vector<GateOp>& gates, 
                               size_t num_qubits, size_t batch_size) {
    if (gates.empty()) return L;
    
    MatrixXcd result = L;
    
    // Apply gates sequentially using optimized direct method
    // (batching with full matrix combination is now slower than direct application)
    for (const auto& gate_op : gates) {
        result = apply_gate_to_L(result, gate_op, num_qubits);
    }
    
    return result;
}

MatrixXcd apply_noise_batched(const MatrixXcd& L, const std::vector<NoiseOp>& noises,
                               size_t num_qubits, size_t batch_size) {
    if (noises.empty()) return L;
    
    MatrixXcd result = L;
    
    // Noise operations increase rank, so apply sequentially
    for (const auto& noise_op : noises) {
        result = apply_noise_to_L(result, noise_op, num_qubits);
    }
    
    return result;
}

}  // namespace qlret

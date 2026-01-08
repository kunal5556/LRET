#include "autodiff.h"
#include "gates_and_noise.h"
#include <cmath>
#include <stdexcept>

namespace qlret {

AutoDiffCircuit::AutoDiffCircuit(size_t num_qubits,
                                 QuantumSequence circuit_template,
                                 std::vector<int> param_indices)
    : num_qubits_(num_qubits),
      circuit_template_(std::move(circuit_template)),
      param_indices_(std::move(param_indices)) {
    if (circuit_template_.operations.size() != param_indices_.size()) {
        throw std::invalid_argument("param_indices size must match circuit operations");
    }
    int max_idx = -1;
    for (int idx : param_indices_) {
        if (idx >= 0) {
            max_idx = std::max(max_idx, idx);
        }
    }
    num_params_ = (max_idx >= 0) ? static_cast<size_t>(max_idx + 1) : 0;
}

// Forward pass with tape recording
double AutoDiffCircuit::forward(const std::vector<double>& params, const Observable& obs) {
    if (params.size() < num_params_) {
        throw std::invalid_argument("Parameter vector too small for circuit");
    }
    tape_.clear();

    // |0...0> state
    size_t dim = 1ULL << num_qubits_;
    MatrixXcd L = MatrixXcd::Zero(dim, 1);
    L(0, 0) = Complex(1.0, 0.0);

    for (size_t i = 0; i < circuit_template_.operations.size(); ++i) {
        const auto& op = circuit_template_.operations[i];

        if (std::holds_alternative<GateOp>(op)) {
            GateOp gate = std::get<GateOp>(op);
            int pidx = param_indices_[i];
            if (pidx >= 0) {
                gate.params = {params[static_cast<size_t>(pidx)]};
            }
            L = apply_gate_to_L(L, gate, num_qubits_);

            if (pidx >= 0) {
                tape_.push_back({gate.type, gate.qubits, gate.params,
                                 static_cast<size_t>(pidx), true});
            }
        } else if (std::holds_alternative<NoiseOp>(op)) {
            const auto& noise = std::get<NoiseOp>(op);
            L = apply_noise_to_L(L, noise, num_qubits_);
        } else {
            // Measurements/conditionals not yet supported in autodiff
            continue;
        }
    }

    return compute_expectation(L, obs);
}

// Forward pass without tape recording (used for parameter shift)
double AutoDiffCircuit::forward_no_record(const std::vector<double>& params,
                                          const Observable& obs) const {
    if (params.size() < num_params_) {
        throw std::invalid_argument("Parameter vector too small for circuit");
    }

    size_t dim = 1ULL << num_qubits_;
    MatrixXcd L = MatrixXcd::Zero(dim, 1);
    L(0, 0) = Complex(1.0, 0.0);

    for (size_t i = 0; i < circuit_template_.operations.size(); ++i) {
        const auto& op = circuit_template_.operations[i];

        if (std::holds_alternative<GateOp>(op)) {
            GateOp gate = std::get<GateOp>(op);
            int pidx = param_indices_[i];
            if (pidx >= 0) {
                gate.params = {params[static_cast<size_t>(pidx)]};
            }
            L = apply_gate_to_L(L, gate, num_qubits_);
        } else if (std::holds_alternative<NoiseOp>(op)) {
            const auto& noise = std::get<NoiseOp>(op);
            L = apply_noise_to_L(L, noise, num_qubits_);
        } else {
            continue;
        }
    }

    return compute_expectation(L, obs);
}

// Forward pass with shift applied to a single gate occurrence
// gate_index: which gate in the circuit to shift
// shift: the shift to apply to that gate's parameter
double AutoDiffCircuit::forward_with_single_shift(const std::vector<double>& params,
                                                   const Observable& obs,
                                                   size_t gate_index,
                                                   double shift) const {
    if (params.size() < num_params_) {
        throw std::invalid_argument("Parameter vector too small for circuit");
    }

    size_t dim = 1ULL << num_qubits_;
    MatrixXcd L = MatrixXcd::Zero(dim, 1);
    L(0, 0) = Complex(1.0, 0.0);

    for (size_t i = 0; i < circuit_template_.operations.size(); ++i) {
        const auto& op = circuit_template_.operations[i];

        if (std::holds_alternative<GateOp>(op)) {
            GateOp gate = std::get<GateOp>(op);
            int pidx = param_indices_[i];
            if (pidx >= 0) {
                double param_val = params[static_cast<size_t>(pidx)];
                // Apply shift only to the specified gate
                if (i == gate_index) {
                    param_val += shift;
                }
                gate.params = {param_val};
            }
            L = apply_gate_to_L(L, gate, num_qubits_);
        } else if (std::holds_alternative<NoiseOp>(op)) {
            const auto& noise = std::get<NoiseOp>(op);
            L = apply_noise_to_L(L, noise, num_qubits_);
        } else {
            continue;
        }
    }

    return compute_expectation(L, obs);
}

std::vector<double> AutoDiffCircuit::backward(const std::vector<double>& params,
                                              const Observable& obs) {
    // Run forward pass to populate tape
    forward(params, obs);

    std::vector<double> grads(num_params_, 0.0);
    const double shift = PI / 2.0;  // parameter-shift rule for Pauli rotations

    // For each gate in the circuit, compute the gradient contribution
    // and accumulate to the corresponding parameter
    for (size_t i = 0; i < circuit_template_.operations.size(); ++i) {
        int pidx = param_indices_[i];
        if (pidx < 0 || static_cast<size_t>(pidx) >= grads.size()) continue;
        
        // Only process parameterized gates
        const auto& op = circuit_template_.operations[i];
        if (!std::holds_alternative<GateOp>(op)) continue;

        // Parameter shift applied to this specific gate occurrence
        double exp_plus = forward_with_single_shift(params, obs, i, shift);
        double exp_minus = forward_with_single_shift(params, obs, i, -shift);

        grads[static_cast<size_t>(pidx)] += (exp_plus - exp_minus) * 0.5;
    }

    return grads;
}

// Expectation value for simple single-qubit observables (Pauli X/Y/Z on one qubit)
double AutoDiffCircuit::compute_expectation(const MatrixXcd& L, const Observable& obs) const {
    if (obs.terms.empty() && obs.qubit >= num_qubits_) {
        throw std::invalid_argument("Observable qubit index out of range");
    }

    MatrixXcd rho = reconstruct_density_matrix(L);
    size_t dim = static_cast<size_t>(rho.rows());

    // Single-qubit Pauli (legacy path)
    if (obs.terms.empty()) {
        size_t mask = 1ULL << obs.qubit;
        double exp_val = 0.0;

        switch (obs.type) {
            case ObservableType::PauliZ: {
                for (size_t i = 0; i < dim; ++i) {
                    double sign = ((i & mask) == 0) ? 1.0 : -1.0;
                    exp_val += sign * rho(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(i)).real();
                }
                return obs.coefficient * exp_val;
            }
            case ObservableType::PauliX: {
                for (size_t i = 0; i < dim; ++i) {
                    size_t j = i ^ mask;
                    exp_val += rho(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)).real();
                }
                return obs.coefficient * exp_val;
            }
            case ObservableType::PauliY: {
                for (size_t i = 0; i < dim; ++i) {
                    size_t j = i ^ mask;
                    double sign = ((i & mask) == 0) ? -1.0 : 1.0;  // Y = [[0, -i], [i, 0]]
                    exp_val += sign * rho(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)).imag();
                }
                return obs.coefficient * exp_val;
            }
            default:
                throw std::invalid_argument("Unsupported observable type");
        }
    }

    // Multi-qubit Pauli string path
    size_t mask_x = 0;
    for (const auto& term : obs.terms) {
        if (term.second >= num_qubits_) {
            throw std::invalid_argument("Observable qubit index out of range");
        }
        if (term.first == ObservableType::PauliX || term.first == ObservableType::PauliY) {
            mask_x |= (1ULL << term.second);
        }
    }

    Complex accum(0.0, 0.0);
    for (size_t i = 0; i < dim; ++i) {
        size_t j = i ^ mask_x;  // X/Y flip bits
        Complex phase(1.0, 0.0);

        for (const auto& term : obs.terms) {
            bool bit = ((i >> term.second) & 1ULL) != 0ULL;
            switch (term.first) {
                case ObservableType::PauliZ:
                    phase *= bit ? Complex(-1.0, 0.0) : Complex(1.0, 0.0);
                    break;
                case ObservableType::PauliX:
                    // Already handled by mask_x flip; no phase contribution
                    break;
                case ObservableType::PauliY:
                    // Y contributes ±i and flips the bit (handled above)
                    phase *= bit ? Complex(0.0, -1.0) : Complex(0.0, 1.0);
                    break;
                default:
                    throw std::invalid_argument("Unsupported observable type in Pauli string");
            }
        }

        accum += phase * rho(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j));
    }

    // Expectation of Hermitian Pauli string should be real; take real part for robustness
    return obs.coefficient * accum.real();
}

}  // namespace qlret

#pragma once

#include "types.h"
#include "simulator.h"
#include <vector>
#include <optional>
#include <utility>

namespace qlret {

// Simple observable types supported for autodiff
enum class ObservableType {
    PauliZ,
    PauliX,
    PauliY
};

struct Observable {
    ObservableType type = ObservableType::PauliZ;
    size_t qubit = 0;  // target qubit
        // Optional multi-qubit Pauli string terms; when non-empty, use these instead of (type, qubit)
        std::vector<std::pair<ObservableType, size_t>> terms;
        double coefficient = 1.0;
};

// Tape entry for parameterized gate
struct TapeEntry {
    GateType gate_type;
    std::vector<size_t> qubits;
    std::vector<double> params;
    size_t param_idx = 0;
    bool is_parameterized = false;
};

// AutoDiffCircuit implements tape-based parameter-shift gradients for LRET
class AutoDiffCircuit {
public:
    AutoDiffCircuit(size_t num_qubits,
                    QuantumSequence circuit_template,
                    std::vector<int> param_indices);

    // Forward pass: executes circuit with given params and returns expectation
    double forward(const std::vector<double>& params, const Observable& obs);

    // Backward pass: parameter-shift gradients for all parameters
    std::vector<double> backward(const std::vector<double>& params,
                                 const Observable& obs);

    size_t num_parameters() const { return num_params_; }

private:
    double forward_no_record(const std::vector<double>& params,
                             const Observable& obs) const;
    
    // Forward with shift applied to a single gate occurrence
    double forward_with_single_shift(const std::vector<double>& params,
                                     const Observable& obs,
                                     size_t gate_index,
                                     double shift) const;

    double compute_expectation(const MatrixXcd& L, const Observable& obs) const;

    size_t num_qubits_ = 0;
    QuantumSequence circuit_template_;
    std::vector<int> param_indices_;  // -1 for non-parameterized ops
    size_t num_params_ = 0;

    // Tape captured during forward()
    std::vector<TapeEntry> tape_;
};

}  // namespace qlret

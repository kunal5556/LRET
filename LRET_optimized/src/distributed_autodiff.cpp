#include "distributed_autodiff.h"
#include "gates_and_noise.h"
#include <cmath>
#include <stdexcept>
#include <numeric>

namespace qlret {

DistributedAutoDiffCircuit::DistributedAutoDiffCircuit(
    size_t num_qubits,
    QuantumSequence circuit_template,
    std::vector<int> param_indices,
    const DistributedGPUConfig& gpu_config)
    : num_qubits_(num_qubits),
      circuit_template_(std::move(circuit_template)),
      param_indices_(std::move(param_indices)),
      gpu_config_(gpu_config) {
    if (circuit_template_.operations.size() != param_indices_.size()) {
        throw std::invalid_argument("param_indices size must match circuit operations");
    }
    int max_idx = -1;
    for (int idx : param_indices_) {
        if (idx >= 0) max_idx = std::max(max_idx, idx);
    }
    num_params_ = (max_idx >= 0) ? static_cast<size_t>(max_idx + 1) : 0;

#if defined(USE_GPU) && defined(USE_MPI) && defined(USE_NCCL)
    dist_sim_ = std::make_unique<DistributedGPUSimulator>(gpu_config_);
#endif
}

DistributedAutoDiffCircuit::~DistributedAutoDiffCircuit() = default;

int DistributedAutoDiffCircuit::rank() const {
#if defined(USE_MPI)
    return get_mpi_rank();
#else
    return 0;
#endif
}

int DistributedAutoDiffCircuit::world_size() const {
#if defined(USE_MPI)
    return get_mpi_size();
#else
    return 1;
#endif
}

double DistributedAutoDiffCircuit::forward_local(const std::vector<double>& params,
                                                  const Observable& obs) {
    if (params.size() < num_params_) {
        throw std::invalid_argument("Parameter vector too small");
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
        }
    }

    // Compute local expectation contribution
    MatrixXcd rho = reconstruct_density_matrix(L);
    size_t rdim = static_cast<size_t>(rho.rows());

    if (obs.terms.empty()) {
        size_t mask = 1ULL << obs.qubit;
        double exp_val = 0.0;
        for (size_t i = 0; i < rdim; ++i) {
            double sign = ((i & mask) == 0) ? 1.0 : -1.0;
            exp_val += sign * rho(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(i)).real();
        }
        return obs.coefficient * exp_val;
    }

    // Multi-qubit Pauli string
    size_t mask_x = 0;
    for (const auto& term : obs.terms) {
        if (term.first == ObservableType::PauliX || term.first == ObservableType::PauliY) {
            mask_x |= (1ULL << term.second);
        }
    }

    Complex accum(0.0, 0.0);
    for (size_t i = 0; i < rdim; ++i) {
        size_t j = i ^ mask_x;
        Complex phase(1.0, 0.0);
        for (const auto& term : obs.terms) {
            bool bit = ((i >> term.second) & 1ULL) != 0ULL;
            switch (term.first) {
                case ObservableType::PauliZ:
                    phase *= bit ? Complex(-1.0, 0.0) : Complex(1.0, 0.0);
                    break;
                case ObservableType::PauliX:
                    break;
                case ObservableType::PauliY:
                    phase *= bit ? Complex(0.0, -1.0) : Complex(0.0, 1.0);
                    break;
                default:
                    break;
            }
        }
        accum += phase * rho(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j));
    }
    return obs.coefficient * accum.real();
}

double DistributedAutoDiffCircuit::forward(const std::vector<double>& params,
                                            const Observable& obs) {
    double local_exp = forward_local(params, obs);

#if defined(USE_GPU) && defined(USE_MPI) && defined(USE_NCCL)
    if (dist_sim_) {
        return dist_sim_->all_reduce_expectation(local_exp) / static_cast<double>(world_size());
    }
#endif
    return local_exp;
}

std::vector<double> DistributedAutoDiffCircuit::backward(const std::vector<double>& params,
                                                          const Observable& obs) {
    std::vector<double> grads(num_params_, 0.0);
    const double shift = PI / 2.0;

    for (size_t pidx = 0; pidx < num_params_; ++pidx) {
        auto params_plus = params;
        params_plus[pidx] += shift;
        double exp_plus = forward(params_plus, obs);

        auto params_minus = params;
        params_minus[pidx] -= shift;
        double exp_minus = forward(params_minus, obs);

        grads[pidx] = (exp_plus - exp_minus) * 0.5;
    }

    return grads;
}

bool validate_distributed_gradients(size_t num_qubits, size_t depth, double tolerance) {
#if defined(USE_GPU) && defined(USE_MPI) && defined(USE_NCCL)
    int rank = get_mpi_rank();
    int world = get_mpi_size();

    if (world < 2) {
        if (rank == 0) {
            std::cout << "Skipping validation: requires >=2 ranks\n";
        }
        return true;
    }

    // Build simple parameterized circuit: alternating RY layers
    QuantumSequence seq(num_qubits);
    std::vector<int> param_indices;
    int param_count = 0;

    for (size_t d = 0; d < depth; ++d) {
        for (size_t q = 0; q < num_qubits; ++q) {
            seq.add_gate(GateOp(GateType::RY, q));
            param_indices.push_back(param_count++);
        }
        // Add CNOT layer (entangling)
        for (size_t q = 0; q + 1 < num_qubits; ++q) {
            seq.add_gate(GateOp(GateType::CNOT, q, q + 1));
            param_indices.push_back(-1);
        }
    }

    std::vector<double> params(static_cast<size_t>(param_count));
    for (size_t i = 0; i < params.size(); ++i) {
        params[i] = 0.1 * static_cast<double>(i + 1);
    }

    Observable obs{ObservableType::PauliZ, 0};

    // Distributed gradients
    DistributedGPUConfig cfg;
    cfg.world_size = world;
    cfg.rank = rank;
    cfg.device_id = rank;
    cfg.enable_collectives = true;

    DistributedAutoDiffCircuit dist_circuit(num_qubits, seq, param_indices, cfg);
    auto dist_grads = dist_circuit.backward(params, obs);

    // Single-GPU reference (rank 0 only)
    std::vector<double> ref_grads;
    if (rank == 0) {
        AutoDiffCircuit ref_circuit(num_qubits, seq, param_indices);
        ref_grads = ref_circuit.backward(params, obs);
    }

    // Broadcast reference gradients from rank 0
    if (rank != 0) {
        ref_grads.resize(dist_grads.size());
    }
    // Simple MPI_Bcast substitute (would use MPI_Bcast in real impl)

    // Compute L2 error (on rank 0)
    if (rank == 0) {
        double l2_err = 0.0;
        for (size_t i = 0; i < ref_grads.size(); ++i) {
            double diff = dist_grads[i] - ref_grads[i];
            l2_err += diff * diff;
        }
        l2_err = std::sqrt(l2_err);

        if (l2_err < tolerance) {
            std::cout << "[PASS] Distributed gradients match reference (L2 error: " << l2_err << ")\n";
            return true;
        } else {
            std::cerr << "[FAIL] Distributed gradients mismatch (L2 error: " << l2_err << ")\n";
            return false;
        }
    }
    return true;
#else
    (void)num_qubits;
    (void)depth;
    (void)tolerance;
    return true;  // Skip on non-GPU builds
#endif
}

}  // namespace qlret

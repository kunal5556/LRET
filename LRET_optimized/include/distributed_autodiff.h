#pragma once

/**
 * @file distributed_autodiff.h
 * @brief Phase 8.4: Distributed Autodiff for Multi-GPU Gradient Computation
 *
 * Extends AutoDiffCircuit to run on distributed GPU backend with:
 * - Row-wise L matrix partitioning across ranks
 * - All-reduce for expectation values
 * - Gradient aggregation via collective ops
 */

#include "autodiff.h"
#include "distributed_gpu.h"
#include "mpi_parallel.h"
#include <vector>
#include <memory>

namespace qlret {

/**
 * @brief Distributed autodiff circuit for multi-GPU parameter-shift gradients.
 *
 * Each rank holds a local partition of L; expectation values are all-reduced,
 * and parameter-shift gradients are computed in parallel then aggregated.
 */
class DistributedAutoDiffCircuit {
public:
    /**
     * @brief Construct distributed autodiff circuit.
     * @param num_qubits Number of qubits
     * @param circuit_template Quantum sequence template
     * @param param_indices Mapping from op index to param index (-1 = non-parameterized)
     * @param gpu_config Distributed GPU configuration
     */
    DistributedAutoDiffCircuit(size_t num_qubits,
                               QuantumSequence circuit_template,
                               std::vector<int> param_indices,
                               const DistributedGPUConfig& gpu_config);

    ~DistributedAutoDiffCircuit();

    DistributedAutoDiffCircuit(const DistributedAutoDiffCircuit&) = delete;
    DistributedAutoDiffCircuit& operator=(const DistributedAutoDiffCircuit&) = delete;

    /**
     * @brief Forward pass: execute circuit and return global expectation.
     * @param params Parameter vector
     * @param obs Observable
     * @return Expectation value (all-reduced across ranks)
     */
    double forward(const std::vector<double>& params, const Observable& obs);

    /**
     * @brief Backward pass: parameter-shift gradients aggregated across ranks.
     * @param params Parameter vector
     * @param obs Observable
     * @return Gradient vector
     */
    std::vector<double> backward(const std::vector<double>& params, const Observable& obs);

    size_t num_parameters() const { return num_params_; }
    int rank() const;
    int world_size() const;

private:
    double forward_local(const std::vector<double>& params, const Observable& obs);

    size_t num_qubits_ = 0;
    QuantumSequence circuit_template_;
    std::vector<int> param_indices_;
    size_t num_params_ = 0;

    std::unique_ptr<DistributedGPUSimulator> dist_sim_;
    DistributedGPUConfig gpu_config_;
};

/**
 * @brief Compare distributed gradients vs single-GPU reference.
 * @param num_qubits Number of qubits
 * @param depth Circuit depth
 * @param tolerance L2 error tolerance
 * @return true if gradients match within tolerance
 */
bool validate_distributed_gradients(size_t num_qubits, size_t depth, double tolerance = 1e-6);

}  // namespace qlret

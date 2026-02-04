#pragma once

/**
 * @file fault_tolerance.h
 * @brief Phase 8.4: Fault Tolerance Integration for Distributed Simulations
 *
 * Combines checkpointing + scheduler + distributed simulator for:
 * - Periodic checkpointing during simulation
 * - Recovery from checkpoint after failure
 * - Scheduler-driven execution with fault awareness
 */

#include "checkpoint.h"
#include "scheduler.h"
#include "distributed_gpu.h"
#include "types.h"
#include <string>
#include <functional>
#include <atomic>

namespace qlret {

/**
 * @brief Configuration for fault-tolerant simulation.
 */
struct FaultToleranceConfig {
    size_t checkpoint_interval = 100;      ///< Checkpoint every N operations
    std::string checkpoint_dir = "./ckpt"; ///< Directory for checkpoint files
    bool async_checkpoint = true;          ///< Use async writer
    SchedulePolicy schedule_policy = SchedulePolicy::FIFO;
    bool verbose = false;
};

/**
 * @brief Fault-tolerant simulation runner.
 *
 * Wraps distributed simulation with periodic checkpointing and recovery.
 */
class FaultTolerantRunner {
public:
    FaultTolerantRunner(size_t num_qubits,
                        const DistributedGPUConfig& gpu_config,
                        const FaultToleranceConfig& ft_config);
    ~FaultTolerantRunner();

    FaultTolerantRunner(const FaultTolerantRunner&) = delete;
    FaultTolerantRunner& operator=(const FaultTolerantRunner&) = delete;

    /**
     * @brief Initialize state (|0...0>).
     */
    void initialize();

    /**
     * @brief Load state from checkpoint.
     * @param checkpoint_path Path to checkpoint file
     * @return true if loaded successfully
     */
    bool recover_from_checkpoint(const std::string& checkpoint_path);

    /**
     * @brief Submit operation for execution.
     * @param op Operation to execute
     * @param priority Priority (for priority scheduler)
     * @param cost Estimated cost (for adaptive scheduler)
     */
    void submit_operation(const QuantumOp& op, int priority = 0, double cost = 1.0);

    /**
     * @brief Execute all pending operations with fault tolerance.
     * @return Number of operations executed
     */
    size_t execute_all();

    /**
     * @brief Force immediate checkpoint.
     */
    void force_checkpoint();

    /**
     * @brief Get current step index.
     */
    size_t current_step() const { return step_; }

    /**
     * @brief Get current L matrix (gathered on rank 0).
     */
    MatrixXcd gather_result() const;

    /**
     * @brief Check if a failure was simulated (for testing).
     */
    bool had_failure() const { return had_failure_; }

    /**
     * @brief Simulate a failure at a given step (for testing).
     */
    void simulate_failure_at(size_t step) { failure_step_ = step; }

private:
    void maybe_checkpoint();
    std::string checkpoint_path() const;

    size_t num_qubits_ = 0;
    DistributedGPUConfig gpu_config_;
    FaultToleranceConfig ft_config_;

    std::unique_ptr<DistributedGPUSimulator> dist_sim_;
    std::unique_ptr<Scheduler> scheduler_;
    AsyncCheckpointWriter async_writer_;

    MatrixXcd L_;  // Current state (on host for checkpointing)
    size_t step_ = 0;
    size_t op_index_ = 0;

    std::atomic<bool> had_failure_{false};
    size_t failure_step_ = SIZE_MAX;
};

/**
 * @brief Test fault tolerance: run, fail, recover, continue.
 * @param num_qubits Number of qubits
 * @param num_ops Total operations to run
 * @param fail_at_step Step at which to simulate failure
 * @return true if recovered state matches expected
 */
bool test_fault_recovery(size_t num_qubits, size_t num_ops, size_t fail_at_step);

}  // namespace qlret

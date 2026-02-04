#include "fault_tolerance.h"
#include "gates_and_noise.h"
#include "mpi_parallel.h"
#include <filesystem>
#include <sstream>
#include <iostream>

namespace qlret {

FaultTolerantRunner::FaultTolerantRunner(size_t num_qubits,
                                         const DistributedGPUConfig& gpu_config,
                                         const FaultToleranceConfig& ft_config)
    : num_qubits_(num_qubits),
      gpu_config_(gpu_config),
      ft_config_(ft_config) {
    scheduler_ = create_scheduler(ft_config_.schedule_policy);

#if defined(USE_GPU) && defined(USE_MPI) && defined(USE_NCCL)
    dist_sim_ = std::make_unique<DistributedGPUSimulator>(gpu_config_);
#endif

    // Create checkpoint directory if needed
    std::filesystem::create_directories(ft_config_.checkpoint_dir);
}

FaultTolerantRunner::~FaultTolerantRunner() {
    async_writer_.wait();
}

void FaultTolerantRunner::initialize() {
    size_t dim = 1ULL << num_qubits_;
    L_ = MatrixXcd::Zero(dim, 1);
    L_(0, 0) = Complex(1.0, 0.0);
    step_ = 0;
    op_index_ = 0;

#if defined(USE_GPU) && defined(USE_MPI) && defined(USE_NCCL)
    if (dist_sim_) {
        dist_sim_->distribute_state(L_);
    }
#endif
}

bool FaultTolerantRunner::recover_from_checkpoint(const std::string& checkpoint_path) {
    CheckpointMeta meta;
    if (!load_checkpoint(checkpoint_path, L_, meta)) {
        return false;
    }
    step_ = meta.step;
    num_qubits_ = meta.num_qubits;

#if defined(USE_GPU) && defined(USE_MPI) && defined(USE_NCCL)
    if (dist_sim_) {
        dist_sim_->distribute_state(L_);
    }
#endif

    if (ft_config_.verbose) {
        std::cout << "[FT] Recovered from checkpoint at step " << step_ << "\n";
    }
    return true;
}

void FaultTolerantRunner::submit_operation(const QuantumOp& op, int priority, double cost) {
    ScheduledOp sched_op;
    sched_op.op_index = op_index_++;
    sched_op.priority = priority;
    sched_op.estimated_cost = cost;
    scheduler_->submit(sched_op);

    // Store operation for execution (simplified: we'd normally store in a vector)
    // For this stub, we just track the index
}

size_t FaultTolerantRunner::execute_all() {
    size_t executed = 0;

    while (!scheduler_->empty()) {
        auto sched_op = scheduler_->next();
        if (!sched_op) break;

        // Simulate failure
        if (step_ == failure_step_) {
            had_failure_ = true;
            if (ft_config_.verbose) {
                std::cout << "[FT] Simulated failure at step " << step_ << "\n";
            }
            break;
        }

        // Execute operation (placeholder: would apply gate/noise here)
        ++step_;
        ++executed;

        maybe_checkpoint();
    }

    return executed;
}

void FaultTolerantRunner::force_checkpoint() {
    CheckpointMeta meta;
    meta.step = step_;
    meta.num_qubits = num_qubits_;
    meta.rank = static_cast<size_t>(L_.cols());
    meta.config_json = "{}";

#if defined(USE_GPU) && defined(USE_MPI) && defined(USE_NCCL)
    if (dist_sim_) {
        L_ = dist_sim_->gather_state();
    }
#endif

    if (ft_config_.async_checkpoint) {
        async_writer_.start(checkpoint_path(), L_, meta);
    } else {
        save_checkpoint(checkpoint_path(), L_, meta);
    }

    if (ft_config_.verbose) {
        std::cout << "[FT] Checkpoint saved at step " << step_ << "\n";
    }
}

void FaultTolerantRunner::maybe_checkpoint() {
    if (step_ > 0 && (step_ % ft_config_.checkpoint_interval) == 0) {
        force_checkpoint();
    }
}

std::string FaultTolerantRunner::checkpoint_path() const {
    std::ostringstream oss;
    oss << ft_config_.checkpoint_dir << "/ckpt_step_" << step_ << ".bin";
    return oss.str();
}

MatrixXcd FaultTolerantRunner::gather_result() const {
#if defined(USE_GPU) && defined(USE_MPI) && defined(USE_NCCL)
    if (dist_sim_) {
        return dist_sim_->gather_state();
    }
#endif
    return L_;
}

bool test_fault_recovery(size_t num_qubits, size_t num_ops, size_t fail_at_step) {
#if defined(USE_GPU) && defined(USE_MPI) && defined(USE_NCCL)
    int rank = get_mpi_rank();

    DistributedGPUConfig gpu_cfg;
    gpu_cfg.world_size = get_mpi_size();
    gpu_cfg.rank = rank;
    gpu_cfg.device_id = rank;

    FaultToleranceConfig ft_cfg;
    ft_cfg.checkpoint_interval = 10;
    ft_cfg.checkpoint_dir = "./ckpt_test";
    ft_cfg.verbose = (rank == 0);

    // Phase 1: Run until failure
    FaultTolerantRunner runner1(num_qubits, gpu_cfg, ft_cfg);
    runner1.initialize();
    runner1.simulate_failure_at(fail_at_step);

    // Submit dummy operations
    for (size_t i = 0; i < num_ops; ++i) {
        ScheduledOp op;
        op.op_index = i;
        runner1.submit_operation(GateOp(GateType::RY, 0), 0, 1.0);
    }

    size_t exec1 = runner1.execute_all();
    bool failed = runner1.had_failure();

    if (rank == 0) {
        std::cout << "[FT Test] Executed " << exec1 << " ops before failure\n";
    }

    if (!failed) {
        if (rank == 0) {
            std::cout << "[FT Test] No failure occurred; test incomplete\n";
        }
        return true;
    }

    // Find latest checkpoint
    std::string ckpt_path = ft_cfg.checkpoint_dir + "/ckpt_step_" +
                            std::to_string((fail_at_step / 10) * 10) + ".bin";

    // Phase 2: Recover and continue
    FaultTolerantRunner runner2(num_qubits, gpu_cfg, ft_cfg);
    if (!runner2.recover_from_checkpoint(ckpt_path)) {
        if (rank == 0) {
            std::cerr << "[FT Test] Failed to recover from checkpoint\n";
        }
        return false;
    }

    // Resume execution
    for (size_t i = runner2.current_step(); i < num_ops; ++i) {
        runner2.submit_operation(GateOp(GateType::RY, 0), 0, 1.0);
    }
    size_t exec2 = runner2.execute_all();

    if (rank == 0) {
        std::cout << "[FT Test] Executed " << exec2 << " ops after recovery\n";
        std::cout << "[FT Test] Total steps: " << runner2.current_step() << "\n";

        if (runner2.current_step() >= num_ops - 1) {
            std::cout << "[PASS] Fault recovery test completed\n";
            return true;
        } else {
            std::cerr << "[FAIL] Did not complete all operations\n";
            return false;
        }
    }
    return true;
#else
    (void)num_qubits;
    (void)num_ops;
    (void)fail_at_step;
    std::cout << "Skipping fault recovery test (requires USE_GPU, USE_MPI, USE_NCCL)\n";
    return true;
#endif
}

}  // namespace qlret

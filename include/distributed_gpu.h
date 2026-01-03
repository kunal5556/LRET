#pragma once

/**
 * @file distributed_gpu.h
 * @brief Multi-GPU distributed simulation support (Phase 8.1)
 *
 * Provides NCCL/MPI-based collectives to scale LRET across multiple GPUs.
 * Designed to interoperate with the existing single-GPU simulator and keep
 * CUDA/NCCL symbols hidden from CPU-only builds via a PIMPL wrapper.
 */

#include "types.h"
#include <memory>
#include <string>
#include <stdexcept>

namespace qlret {

struct DistributedGPUConfig {
    int world_size = 1;             ///< Total number of GPU ranks participating
    int rank = 0;                   ///< Rank of this process (MPI rank)
    int device_id = -1;             ///< GPU device to bind (-1 = rank-based)
    bool overlap_comm_compute = true; ///< Try to overlap communication and compute
    bool enable_collectives = true; ///< Enable NCCL/MPI collectives when available
    bool verbose = false;           ///< Print setup diagnostics
};

/**
 * @brief Multi-GPU simulator front-end.
 *
 * Responsibilities:
 *  - GPU device binding per rank
 *  - NCCL/MPI communicator setup
 *  - State distribution (row-partitioned L matrix)
 *  - Collectives: all-reduce (expectations) and gather (final state)
 */
class DistributedGPUSimulator {
public:
    explicit DistributedGPUSimulator(const DistributedGPUConfig& config);
    ~DistributedGPUSimulator();

    DistributedGPUSimulator(const DistributedGPUSimulator&) = delete;
    DistributedGPUSimulator& operator=(const DistributedGPUSimulator&) = delete;
    DistributedGPUSimulator(DistributedGPUSimulator&&) noexcept;
    DistributedGPUSimulator& operator=(DistributedGPUSimulator&&) noexcept;

    /**
     * @brief Distribute a global L matrix across GPUs (row-wise block).
     * @param L_full Full L matrix on host (2^n x r, column-major Eigen)
     */
    void distribute_state(const MatrixXcd& L_full);

    /**
     * @brief Gather the full L matrix back to rank 0 (host).
     * @return Full L on host (rank 0) or empty matrix on other ranks.
     */
    MatrixXcd gather_state() const;

    /**
     * @brief All-reduce a scalar expectation value across GPUs.
     * @param local_exp Local contribution
     * @return Global reduced value
     */
    double all_reduce_expectation(double local_exp) const;

    /**
     * @brief Hook to overlap communication for upcoming two-qubit gates.
     * @param needs_remote Whether the next gate depends on remote data.
     */
    void overlap_for_two_qubit(bool needs_remote);

    /**
     * @brief Copy the local partition from device to host.
     * @return Local block of L (row-contiguous segment) on host.
     */
    MatrixXcd copy_local_to_host() const;

    /**
     * @brief Upload a host-local partition back to device memory.
     * @param local Updated local block (must match local_rows x columns).
     */
    void upload_local_from_host(const MatrixXcd& local);

    // Introspection helpers
    size_t local_rows() const;
    size_t global_rows() const;
    size_t columns() const;
    int device_id() const;
    bool is_multi_gpu() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace qlret

#ifndef USE_GPU
// CPU-only placeholder to keep builds working without CUDA/NCCL
namespace qlret {
inline DistributedGPUSimulator::DistributedGPUSimulator(const DistributedGPUConfig&) {
    throw std::runtime_error("DistributedGPUSimulator requires USE_GPU=ON");
}
inline DistributedGPUSimulator::~DistributedGPUSimulator() = default;
inline DistributedGPUSimulator::DistributedGPUSimulator(DistributedGPUSimulator&&) noexcept = default;
inline DistributedGPUSimulator& DistributedGPUSimulator::operator=(DistributedGPUSimulator&&) noexcept = default;
inline void DistributedGPUSimulator::distribute_state(const MatrixXcd&) {
    throw std::runtime_error("DistributedGPUSimulator requires USE_GPU=ON");
}
inline MatrixXcd DistributedGPUSimulator::gather_state() const { return MatrixXcd(); }
inline double DistributedGPUSimulator::all_reduce_expectation(double local_exp) const { return local_exp; }
inline void DistributedGPUSimulator::overlap_for_two_qubit(bool) {}
inline MatrixXcd DistributedGPUSimulator::copy_local_to_host() const { return MatrixXcd(); }
inline void DistributedGPUSimulator::upload_local_from_host(const MatrixXcd&) {}
inline size_t DistributedGPUSimulator::local_rows() const { return 0; }
inline size_t DistributedGPUSimulator::global_rows() const { return 0; }
inline size_t DistributedGPUSimulator::columns() const { return 0; }
inline int DistributedGPUSimulator::device_id() const { return -1; }
inline bool DistributedGPUSimulator::is_multi_gpu() const { return false; }
}  // namespace qlret
#endif  // USE_GPU

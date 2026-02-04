#pragma once

/**
 * @file checkpoint.h
 * @brief Phase 8.3+ Fault Tolerance: Checkpointing for Distributed Simulations
 *
 * Provides:
 * - Checkpoint save/restore for L matrix state
 * - Metadata (step index, parameters, config)
 * - Async checkpoint writer (non-blocking)
 */

#include "types.h"
#include <string>
#include <memory>
#include <optional>
#include <functional>

namespace qlret {

struct CheckpointMeta {
    size_t step = 0;               ///< Simulation step index
    size_t num_qubits = 0;         ///< Number of qubits
    size_t rank = 0;               ///< Matrix rank (columns of L)
    std::string config_json;       ///< Serialized simulation config
};

/**
 * @brief Save checkpoint to file.
 * @param path File path (will overwrite if exists)
 * @param L Current L matrix (host, column-major)
 * @param meta Checkpoint metadata
 * @return true on success
 */
bool save_checkpoint(const std::string& path, const MatrixXcd& L, const CheckpointMeta& meta);

/**
 * @brief Load checkpoint from file.
 * @param path File path
 * @param L Output L matrix
 * @param meta Output metadata
 * @return true on success
 */
bool load_checkpoint(const std::string& path, MatrixXcd& L, CheckpointMeta& meta);

/**
 * @brief Async checkpoint writer (non-blocking).
 *
 * Spawns a background thread to write checkpoint while simulation continues.
 * Call wait_checkpoint() before next checkpoint to ensure completion.
 */
class AsyncCheckpointWriter {
public:
    AsyncCheckpointWriter();
    ~AsyncCheckpointWriter();

    AsyncCheckpointWriter(const AsyncCheckpointWriter&) = delete;
    AsyncCheckpointWriter& operator=(const AsyncCheckpointWriter&) = delete;

    /**
     * @brief Start async write.
     * @param path File path
     * @param L L matrix (copied internally)
     * @param meta Metadata
     */
    void start(const std::string& path, const MatrixXcd& L, const CheckpointMeta& meta);

    /**
     * @brief Wait for pending write to complete.
     * @return true if last write succeeded
     */
    bool wait();

    /**
     * @brief Check if a write is in progress.
     */
    bool is_busy() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace qlret

#pragma once

/**
 * @file distributed_perf.h
 * @brief Phase 8.2 Performance Optimization Utilities for Distributed GPU
 *
 * Provides helpers for:
 * - Pinned memory allocation/deallocation
 * - Comm/compute overlap scheduling
 * - Bandwidth measurement utilities
 */

#include "types.h"
#include <cstddef>
#include <memory>

namespace qlret {

//==============================================================================
// Pinned Memory Allocator (for GPU-Direct / faster H2D/D2H transfers)
//==============================================================================

/**
 * @brief Allocate pinned (page-locked) host memory.
 * @param bytes Number of bytes to allocate
 * @return Pointer to pinned memory, or nullptr on failure
 */
void* alloc_pinned(size_t bytes);

/**
 * @brief Free pinned host memory.
 * @param ptr Pointer previously returned by alloc_pinned
 */
void free_pinned(void* ptr);

/**
 * @brief RAII wrapper for pinned memory
 */
template <typename T>
class PinnedBuffer {
public:
    explicit PinnedBuffer(size_t count)
        : ptr_(static_cast<T*>(alloc_pinned(count * sizeof(T)))), count_(count) {}
    ~PinnedBuffer() { if (ptr_) free_pinned(ptr_); }

    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
    PinnedBuffer(PinnedBuffer&& o) noexcept : ptr_(o.ptr_), count_(o.count_) { o.ptr_ = nullptr; }
    PinnedBuffer& operator=(PinnedBuffer&& o) noexcept {
        if (this != &o) { if (ptr_) free_pinned(ptr_); ptr_ = o.ptr_; count_ = o.count_; o.ptr_ = nullptr; }
        return *this;
    }

    T* data() { return ptr_; }
    const T* data() const { return ptr_; }
    size_t size() const { return count_; }
    T& operator[](size_t i) { return ptr_[i]; }
    const T& operator[](size_t i) const { return ptr_[i]; }

private:
    T* ptr_ = nullptr;
    size_t count_ = 0;
};

//==============================================================================
// Overlap Scheduling Utilities
//==============================================================================

/**
 * @brief Hint to begin async communication for upcoming gate.
 * @param needs_remote Whether the gate requires remote data
 */
void schedule_overlap_begin(bool needs_remote);

/**
 * @brief Wait for async communication to complete before gate execution.
 */
void schedule_overlap_sync();

//==============================================================================
// Bandwidth Measurement
//==============================================================================

struct BandwidthResult {
    double h2d_gbps = 0.0;   ///< Host-to-device bandwidth (GB/s)
    double d2h_gbps = 0.0;   ///< Device-to-host bandwidth (GB/s)
    double d2d_gbps = 0.0;   ///< Device-to-device (P2P) bandwidth (GB/s)
};

/**
 * @brief Measure memory bandwidth for pinned and pageable transfers.
 * @param bytes Transfer size in bytes
 * @param iterations Number of iterations to average
 * @return BandwidthResult with measured throughput
 */
BandwidthResult measure_bandwidth(size_t bytes, int iterations = 10);

}  // namespace qlret

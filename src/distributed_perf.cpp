#include "distributed_perf.h"
#include <cstdlib>
#include <cstring>

#ifdef USE_GPU
#include <cuda_runtime.h>
#endif

namespace qlret {

void* alloc_pinned(size_t bytes) {
#ifdef USE_GPU
    void* ptr = nullptr;
    cudaError_t err = cudaMallocHost(&ptr, bytes);
    if (err != cudaSuccess) return nullptr;
    return ptr;
#else
    // Fallback: aligned malloc (no pinning without CUDA)
    return std::aligned_alloc(64, bytes);
#endif
}

void free_pinned(void* ptr) {
    if (!ptr) return;
#ifdef USE_GPU
    cudaFreeHost(ptr);
#else
    std::free(ptr);
#endif
}

void schedule_overlap_begin(bool /*needs_remote*/) {
    // Stub: actual implementation would launch async NCCL send/recv
}

void schedule_overlap_sync() {
    // Stub: actual implementation would synchronize streams
#ifdef USE_GPU
    cudaDeviceSynchronize();
#endif
}

BandwidthResult measure_bandwidth(size_t bytes, int iterations) {
    BandwidthResult result;
#ifdef USE_GPU
    void* h_pinned = alloc_pinned(bytes);
    if (!h_pinned) return result;
    std::memset(h_pinned, 0, bytes);

    void* d_buf = nullptr;
    cudaMalloc(&d_buf, bytes);
    if (!d_buf) { free_pinned(h_pinned); return result; }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    cudaMemcpy(d_buf, h_pinned, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(h_pinned, d_buf, bytes, cudaMemcpyDeviceToHost);

    // H2D
    cudaEventRecord(start);
    for (int i = 0; i < iterations; ++i) {
        cudaMemcpy(d_buf, h_pinned, bytes, cudaMemcpyHostToDevice);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    result.h2d_gbps = (static_cast<double>(bytes) * iterations / 1e9) / (ms / 1e3);

    // D2H
    cudaEventRecord(start);
    for (int i = 0; i < iterations; ++i) {
        cudaMemcpy(h_pinned, d_buf, bytes, cudaMemcpyDeviceToHost);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    result.d2h_gbps = (static_cast<double>(bytes) * iterations / 1e9) / (ms / 1e3);

    // D2D (self-copy as proxy; true P2P requires second device)
    cudaEventRecord(start);
    for (int i = 0; i < iterations; ++i) {
        cudaMemcpy(d_buf, d_buf, bytes, cudaMemcpyDeviceToDevice);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    result.d2d_gbps = (static_cast<double>(bytes) * iterations / 1e9) / (ms / 1e3);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_buf);
    free_pinned(h_pinned);
#else
    (void)bytes;
    (void)iterations;
#endif
    return result;
}

}  // namespace qlret

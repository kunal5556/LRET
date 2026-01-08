/**
 * @file distributed_gpu.cu
 * @brief NCCL/MPI-based multi-GPU support (Phase 8.1)
 */

#include "distributed_gpu.h"

#ifdef USE_GPU

#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>
#include <numeric>
#include <iostream>
#include <cmath>
#include <cstring>
#include <complex>

#ifdef USE_NCCL
#include <nccl.h>
#endif

#ifdef USE_MPI
#include <mpi.h>
#endif

namespace qlret {

namespace {
#ifdef USE_NCCL
#define NCCL_CHECK(call) \
    do { \
        ncclResult_t _status = call; \
        if (_status != ncclSuccess) { \
            throw std::runtime_error(std::string("NCCL error: ") + ncclGetErrorString(_status)); \
        } \
    } while (0)
#endif

#define CUDA_CHECK(call) \
    do { \
        cudaError_t _err = call; \
        if (_err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(_err)); \
        } \
    } while (0)
}

class DistributedGPUSimulator::Impl {
public:
    explicit Impl(const DistributedGPUConfig& cfg) : config_(cfg) {
        if (config_.world_size < 1) config_.world_size = 1;
        if (config_.rank < 0) config_.rank = 0;

#ifdef USE_MPI
        int mpi_world = 0;
        int mpi_rank = 0;
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_world);
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
        mpi_enabled_ = true;
        mpi_world_size_ = mpi_world;
        if (config_.world_size != mpi_world) {
            config_.world_size = mpi_world;
        }
        if (config_.rank != mpi_rank) {
            config_.rank = mpi_rank;
        }
#endif

        // Device selection: default to rank-based mapping
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        if (device_count == 0) {
            throw std::runtime_error("No CUDA devices available for DistributedGPUSimulator");
        }

        device_id_ = (config_.device_id >= 0) ? config_.device_id : (config_.rank % device_count);
        CUDA_CHECK(cudaSetDevice(device_id_));

        // Only initialize NCCL when multi-GPU is requested
        if (config_.world_size > 1) {
#ifndef USE_NCCL
            if (config_.enable_collectives) {
                throw std::runtime_error("NCCL not enabled. Rebuild with USE_NCCL=ON for multi-GPU.");
            }
#else
            nccl_enabled_ = true;
#ifdef USE_MPI
            if (!mpi_enabled_) {
                throw std::runtime_error("MPI not available for NCCL unique ID broadcast");
            }
#endif
            ncclUniqueId id{};
            if (config_.rank == 0) {
                NCCL_CHECK(ncclGetUniqueId(&id));
            }
#ifdef USE_MPI
            MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
#endif
            NCCL_CHECK(ncclCommInitRank(&nccl_comm_, config_.world_size, id, config_.rank));
#endif
        }

        CUDA_CHECK(cudaStreamCreate(&compute_stream_));
        CUDA_CHECK(cudaStreamCreate(&comm_stream_));
        CUDA_CHECK(cudaEventCreateWithFlags(&prefetch_event_, cudaEventDisableTiming));

        if (config_.verbose) {
            std::cout << "[DistributedGPUSimulator] rank=" << config_.rank
                      << "/" << config_.world_size
                      << " device=" << device_id_ << std::endl;
        }
    }

    ~Impl() {
#ifdef USE_NCCL
        if (nccl_enabled_) {
            ncclCommDestroy(nccl_comm_);
        }
#endif
        if (compute_stream_) cudaStreamDestroy(compute_stream_);
        if (comm_stream_) cudaStreamDestroy(comm_stream_);
        if (prefetch_event_) cudaEventDestroy(prefetch_event_);
        if (d_L_) cudaFree(d_L_);
    }

    Impl(Impl&& other) noexcept { move_from(std::move(other)); }
    Impl& operator=(Impl&& other) noexcept {
        if (this != &other) {
#ifdef USE_NCCL
            if (nccl_enabled_) {
                ncclCommDestroy(nccl_comm_);
            }
#endif
            if (compute_stream_) cudaStreamDestroy(compute_stream_);
            if (comm_stream_) cudaStreamDestroy(comm_stream_);
            if (prefetch_event_) cudaEventDestroy(prefetch_event_);
            if (d_L_) cudaFree(d_L_);
            move_from(std::move(other));
        }
        return *this;
    }

    void distribute_state(const MatrixXcd& L_full) {
        global_rows_ = static_cast<size_t>(L_full.rows());
        columns_ = static_cast<size_t>(L_full.cols());

        if (global_rows_ == 0 || columns_ == 0) {
            throw std::invalid_argument("Empty L matrix provided to distribute_state");
        }

        // Validate power-of-two rows
        double nq = std::log2(static_cast<double>(global_rows_));
        if (std::fabs(nq - std::round(nq)) > 1e-9) {
            throw std::invalid_argument("L rows must be 2^n for distribution");
        }

        size_t base_rows = global_rows_ / config_.world_size;
        size_t remainder = global_rows_ % config_.world_size;
        local_rows_ = base_rows + (static_cast<size_t>(config_.rank) < remainder ? 1 : 0);
        start_row_ = base_rows * static_cast<size_t>(config_.rank) + std::min(remainder, static_cast<size_t>(config_.rank));

        if (local_rows_ == 0) {
            throw std::runtime_error("Computed local_rows_ is zero. Check world size vs problem size.");
        }

        size_t local_elems = local_rows_ * columns_;
        host_local_.assign(local_elems, std::complex<double>(0.0, 0.0));

        // Extract row block in column-major order then copy to contiguous buffer
        for (size_t col = 0; col < columns_; ++col) {
            const std::complex<double>* src_col = L_full.data() + col * global_rows_ + start_row_;
            std::memcpy(host_local_.data() + col * local_rows_, src_col, sizeof(std::complex<double>) * local_rows_);
        }

        // Allocate and upload to GPU
        if (d_L_) cudaFree(d_L_);
        CUDA_CHECK(cudaMalloc(&d_L_, local_elems * sizeof(cuDoubleComplex)));
        CUDA_CHECK(cudaMemcpyAsync(d_L_, host_local_.data(), local_elems * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice, compute_stream_));
        CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
    }

    MatrixXcd gather_state() const {
        if (global_rows_ == 0 || columns_ == 0) {
            return MatrixXcd();
        }

        size_t local_elems = local_rows_ * columns_;
        std::vector<std::complex<double>> host_col_major(local_elems);
        CUDA_CHECK(cudaMemcpyAsync(host_col_major.data(), d_L_, local_elems * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost, compute_stream_));
        CUDA_CHECK(cudaStreamSynchronize(compute_stream_));

        // Convert to row-major for MPI gather
        std::vector<std::complex<double>> host_row_major(local_elems);
        for (size_t col = 0; col < columns_; ++col) {
            for (size_t r = 0; r < local_rows_; ++r) {
                host_row_major[r * columns_ + col] = host_col_major[col * local_rows_ + r];
            }
        }

        if (config_.world_size == 1) {
            MatrixXcd out(global_rows_, columns_);
            for (size_t r = 0; r < global_rows_; ++r) {
                for (size_t c = 0; c < columns_; ++c) {
                    out(r, c) = host_row_major[r * columns_ + c];
                }
            }
            return out;
        }

#ifndef USE_MPI
        throw std::runtime_error("gather_state requires MPI for multi-GPU builds");
#else
        int send_count = static_cast<int>(host_row_major.size());
        std::vector<int> recv_counts(config_.world_size, 0);
        std::vector<int> displs(config_.world_size, 0);

        MPI_Allgather(&send_count, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
        int total = std::accumulate(recv_counts.begin(), recv_counts.end(), 0);

        for (int i = 1; i < config_.world_size; ++i) {
            displs[i] = displs[i - 1] + recv_counts[i - 1];
        }

        std::vector<std::complex<double>> gathered;
        if (config_.rank == 0) {
            gathered.resize(static_cast<size_t>(total));
        }

        MPI_Gatherv(host_row_major.data(), send_count, MPI_DOUBLE_COMPLEX,
                gathered.data(), recv_counts.data(), displs.data(), MPI_DOUBLE_COMPLEX,
                    0, MPI_COMM_WORLD);

        if (config_.rank != 0) {
            return MatrixXcd();
        }

        if (static_cast<size_t>(total) != global_rows_ * columns_) {
            throw std::runtime_error("Gathered element count mismatch");
        }

        MatrixXcd out(global_rows_, columns_);
        for (size_t r = 0; r < global_rows_; ++r) {
            for (size_t c = 0; c < columns_; ++c) {
                out(r, c) = gathered[r * columns_ + c];
            }
        }
        return out;
#endif
    }

    double all_reduce_expectation(double local_exp) const {
        if (!config_.enable_collectives || config_.world_size == 1) {
            return local_exp;
        }

#ifdef USE_NCCL
        // NCCL operates on device memory, so we need to copy to/from device
        double* d_local = nullptr;
        double* d_result = nullptr;
        CUDA_CHECK(cudaMalloc(&d_local, sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_result, sizeof(double)));
        CUDA_CHECK(cudaMemcpyAsync(d_local, &local_exp, sizeof(double), cudaMemcpyHostToDevice, compute_stream_));
        
        NCCL_CHECK(ncclAllReduce(d_local, d_result, 1, ncclDouble, ncclSum, nccl_comm_, compute_stream_));
        
        double result = 0.0;
        CUDA_CHECK(cudaMemcpyAsync(&result, d_result, sizeof(double), cudaMemcpyDeviceToHost, compute_stream_));
        CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
        
        cudaFree(d_local);
        cudaFree(d_result);
        return result;
#elif defined(USE_MPI)
        double result = 0.0;
        MPI_Allreduce(&local_exp, &result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        return result;
#else
        throw std::runtime_error("all_reduce_expectation requires NCCL or MPI for multi-GPU");
#endif
    }

    void overlap_for_two_qubit(bool needs_remote) {
        if (!needs_remote || !config_.overlap_comm_compute) {
            return;
        }
        // Record an event on the comm stream and make compute stream wait,
        // establishing a hook for future async prefetch/comm work.
        CUDA_CHECK(cudaEventRecord(prefetch_event_, comm_stream_));
        CUDA_CHECK(cudaStreamWaitEvent(compute_stream_, prefetch_event_, 0));
    }

    MatrixXcd copy_local_to_host() const {
        if (local_rows_ == 0 || columns_ == 0) {
            return MatrixXcd();
        }

        size_t local_elems = local_rows_ * columns_;
        std::vector<std::complex<double>> host_col_major(local_elems);
        CUDA_CHECK(cudaMemcpyAsync(host_col_major.data(), d_L_, local_elems * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost, compute_stream_));
        CUDA_CHECK(cudaStreamSynchronize(compute_stream_));

        MatrixXcd out(local_rows_, columns_);
        for (size_t c = 0; c < columns_; ++c) {
            for (size_t r = 0; r < local_rows_; ++r) {
                out(static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(c)) = host_col_major[c * local_rows_ + r];
            }
        }
        return out;
    }

    void upload_local_from_host(const MatrixXcd& local) {
        if (local_rows_ == 0 || columns_ == 0) {
            return;
        }
        if (local.rows() != static_cast<Eigen::Index>(local_rows_) || local.cols() != static_cast<Eigen::Index>(columns_)) {
            throw std::invalid_argument("upload_local_from_host shape mismatch");
        }

        size_t local_elems = local_rows_ * columns_;
        host_local_.assign(local_elems, std::complex<double>(0.0, 0.0));
        for (size_t c = 0; c < columns_; ++c) {
            for (size_t r = 0; r < local_rows_; ++r) {
                host_local_[c * local_rows_ + r] = local(static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(c));
            }
        }

        if (!d_L_) {
            CUDA_CHECK(cudaMalloc(&d_L_, local_elems * sizeof(cuDoubleComplex)));
        }

        CUDA_CHECK(cudaMemcpyAsync(d_L_, host_local_.data(), local_elems * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice, compute_stream_));
        CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
    }

    size_t local_rows() const { return local_rows_; }
    size_t global_rows() const { return global_rows_; }
    size_t columns() const { return columns_; }
    int device_id() const { return device_id_; }
    bool is_multi_gpu() const { return config_.world_size > 1; }

private:
    void move_from(Impl&& other) {
        config_ = other.config_;
        nccl_enabled_ = other.nccl_enabled_;
        compute_stream_ = other.compute_stream_;
        comm_stream_ = other.comm_stream_;
        d_L_ = other.d_L_;
        global_rows_ = other.global_rows_;
        columns_ = other.columns_;
        local_rows_ = other.local_rows_;
        start_row_ = other.start_row_;
        device_id_ = other.device_id_;
        host_local_ = std::move(other.host_local_);
#ifdef USE_NCCL
        nccl_comm_ = other.nccl_comm_;
#endif
        other.compute_stream_ = nullptr;
        other.comm_stream_ = nullptr;
        other.d_L_ = nullptr;
#ifdef USE_NCCL
        other.nccl_enabled_ = false;
#endif
    }

    DistributedGPUConfig config_;

#ifdef USE_NCCL
    bool nccl_enabled_ = false;
    ncclComm_t nccl_comm_{};
#endif

#ifdef USE_MPI
    bool mpi_enabled_ = false;
    int mpi_world_size_ = 1;
#endif

    cudaStream_t compute_stream_ = nullptr;
    cudaStream_t comm_stream_ = nullptr;
    cudaEvent_t prefetch_event_ = nullptr;
    cuDoubleComplex* d_L_ = nullptr;

    size_t global_rows_ = 0;
    size_t columns_ = 0;
    size_t local_rows_ = 0;
    size_t start_row_ = 0;
    int device_id_ = 0;

    std::vector<std::complex<double>> host_local_;
};

//------------------------------------------------------------------------------
// Public wrapper
//------------------------------------------------------------------------------

DistributedGPUSimulator::DistributedGPUSimulator(const DistributedGPUConfig& config)
    : impl_(std::make_unique<Impl>(config)) {}

DistributedGPUSimulator::~DistributedGPUSimulator() = default;
DistributedGPUSimulator::DistributedGPUSimulator(DistributedGPUSimulator&&) noexcept = default;
DistributedGPUSimulator& DistributedGPUSimulator::operator=(DistributedGPUSimulator&&) noexcept = default;

void DistributedGPUSimulator::distribute_state(const MatrixXcd& L_full) {
    impl_->distribute_state(L_full);
}

MatrixXcd DistributedGPUSimulator::gather_state() const {
    return impl_->gather_state();
}

double DistributedGPUSimulator::all_reduce_expectation(double local_exp) const {
    return impl_->all_reduce_expectation(local_exp);
}

void DistributedGPUSimulator::overlap_for_two_qubit(bool needs_remote) {
    impl_->overlap_for_two_qubit(needs_remote);
}

MatrixXcd DistributedGPUSimulator::copy_local_to_host() const {
    return impl_->copy_local_to_host();
}

void DistributedGPUSimulator::upload_local_from_host(const MatrixXcd& local) {
    impl_->upload_local_from_host(local);
}

size_t DistributedGPUSimulator::local_rows() const { return impl_->local_rows(); }
size_t DistributedGPUSimulator::global_rows() const { return impl_->global_rows(); }
size_t DistributedGPUSimulator::columns() const { return impl_->columns(); }
int DistributedGPUSimulator::device_id() const { return impl_->device_id(); }
bool DistributedGPUSimulator::is_multi_gpu() const { return impl_->is_multi_gpu(); }

}  // namespace qlret

#endif  // USE_GPU

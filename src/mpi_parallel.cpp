#include "mpi_parallel.h"
#include "simulator.h"
#include "gates_and_noise.h"
#include "utils.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <iostream>
#include <random>

#ifdef USE_MPI

namespace qlret {

//==============================================================================
// Helper utilities
//==============================================================================

static size_t compute_rows_for_rank(int rank, size_t global_dim, int world_size, size_t& row_start_out) {
    size_t base = global_dim / world_size;
    size_t rem  = global_dim % world_size;
    size_t start = rank * base + std::min<size_t>(rank, rem);
    size_t rows  = base + (static_cast<size_t>(rank) < rem ? 1 : 0);
    row_start_out = start;
    return rows;
}

static size_t floor_log2_size_t(size_t v) {
    size_t r = 0;
    while ((1ULL << (r + 1)) <= v) ++r;
    return r;
}

//==============================================================================
// MPIProcessInfo
//==============================================================================

int MPIProcessInfo::get_partner_rank(size_t qubit) const {
    size_t local_bits = floor_log2_size_t(local_rows);
    if (qubit < local_bits) return world_rank;
    size_t offset_bit = qubit - local_bits;
    int partner = world_rank ^ (1 << static_cast<int>(offset_bit));
    return partner;
}

void MPIProcessInfo::print() const {
    std::cout << "MPI rank " << world_rank << "/" << world_size
              << " rows [" << row_start << "," << row_end << ")"
              << " local_rows=" << local_rows
              << " local_qubit_bits=" << local_qubit
              << std::endl;
}

//==============================================================================
// MPICommStats
//==============================================================================

void MPICommStats::print() const {
    std::cout << "MPI Comm Stats: messages=" << total_messages_sent
              << " bytes=" << total_bytes_sent
              << " local_ops=" << local_gate_ops
              << " remote_ops=" << remote_gate_ops
              << " efficiency=" << efficiency()
              << std::endl;
}

void MPICommStats::reset() {
    total_messages_sent = 0;
    total_bytes_sent = 0;
    local_gate_ops = 0;
    remote_gate_ops = 0;
    total_comm_time = 0.0;
    total_compute_time = 0.0;
}

//==============================================================================
// MPIBufferManager
//==============================================================================

MPIBufferManager::MPIBufferManager(size_t buffer_size) {
    if (buffer_size > 0) {
        send_buffer_.resize(buffer_size);
        recv_buffer_.resize(buffer_size);
        buffer_size_ = buffer_size;
    }
}

MPIBufferManager::~MPIBufferManager() = default;

Complex* MPIBufferManager::get_send_buffer(size_t size) {
    ensure_capacity(size);
    return send_buffer_.data();
}

Complex* MPIBufferManager::get_recv_buffer(size_t size) {
    ensure_capacity(size);
    return recv_buffer_.data();
}

void MPIBufferManager::ensure_capacity(size_t size) {
    if (size > buffer_size_) {
        send_buffer_.resize(size);
        recv_buffer_.resize(size);
        buffer_size_ = size;
    }
}

//==============================================================================
// MPISimulatorImpl
//==============================================================================

class MPISimulatorImpl {
public:
    MPIConfig config_;
    MPIProcessInfo info_;
    MPICommStats stats_;
    MPIBufferManager buffers_;
    MatrixXcd local_L_;

    MPISimulatorImpl(size_t num_qubits, const MPIConfig& config)
        : config_(config) {
        MPI_Comm_rank(MPI_COMM_WORLD, &info_.world_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &info_.world_size);

        info_.num_qubits = num_qubits;
        info_.global_dim = (1ULL << num_qubits);

        size_t row_start = 0;
        info_.local_rows = compute_rows_for_rank(info_.world_rank, info_.global_dim, info_.world_size, row_start);
        info_.row_start = row_start;
        info_.row_end = row_start + info_.local_rows;
        info_.local_qubit = floor_log2_size_t(info_.local_rows);

        buffers_ = MPIBufferManager(config.comm_buffer_size);

        if (config.verbose && info_.is_root()) {
            std::cout << "[MPI] world_size=" << info_.world_size
                      << " global_dim=" << info_.global_dim
                      << " local_rows(rank0)=" << info_.local_rows << std::endl;
        }
    }

    void initialize_state(const MatrixXcd& L, bool is_local_chunk) {
        if (is_local_chunk) {
            local_L_ = L;
            return;
        }

        int rank = info_.world_rank;
        int size = info_.world_size;

        std::vector<int> send_counts(size, 0);
        std::vector<int> displs(size, 0);

        if (rank == 0) {
            for (int r = 0; r < size; ++r) {
                size_t start;
                size_t rows = compute_rows_for_rank(r, info_.global_dim, size, start);
                send_counts[r] = static_cast<int>(rows * L.cols());
                displs[r] = static_cast<int>(start * L.cols());
            }
        }

        int local_count = static_cast<int>(info_.local_rows * L.cols());
        local_L_.resize(info_.local_rows, L.cols());

        MPI_Scatterv(
            rank == 0 ? L.data() : nullptr,
            send_counts.data(),
            displs.data(),
            MPI_CXX_DOUBLE_COMPLEX,
            local_L_.data(),
            local_count,
            MPI_CXX_DOUBLE_COMPLEX,
            0,
            MPI_COMM_WORLD
        );
    }

    void initialize_zero_state() {
        local_L_ = MatrixXcd::Zero(info_.local_rows, 1);
        if (info_.row_start == 0) {
            local_L_(0, 0) = Complex(1.0, 0.0);
        }
    }

    void initialize_random_state(size_t rank_cols, unsigned int seed) {
        std::mt19937 rng(seed ? seed + info_.world_rank : static_cast<unsigned int>(std::random_device{}()));
        std::normal_distribution<double> dist(0.0, 1.0);
        local_L_.resize(info_.local_rows, rank_cols);
        for (int r = 0; r < local_L_.rows(); ++r) {
            for (int c = 0; c < local_L_.cols(); ++c) {
                local_L_(r, c) = Complex(dist(rng), dist(rng));
            }
        }
    }

    void apply_single_local(size_t qubit, const Matrix2cd& gate) {
        size_t dim_local = info_.local_rows;
        size_t rank_cols = static_cast<size_t>(local_L_.cols());
        size_t bit = 1ULL << qubit;

        for (size_t local_r = 0; local_r < dim_local; ++local_r) {
            size_t global_r = info_.row_start + local_r;
            if ((global_r & bit) != 0) continue;
            size_t partner_global = global_r ^ bit;
            if (partner_global < info_.row_start || partner_global >= info_.row_end) continue;
            size_t partner_local = partner_global - info_.row_start;

            for (size_t c = 0; c < rank_cols; ++c) {
                Complex v0 = local_L_(static_cast<int>(local_r), static_cast<int>(c));
                Complex v1 = local_L_(static_cast<int>(partner_local), static_cast<int>(c));
                local_L_(static_cast<int>(local_r), static_cast<int>(c)) = gate(0,0)*v0 + gate(0,1)*v1;
                local_L_(static_cast<int>(partner_local), static_cast<int>(c)) = gate(1,0)*v0 + gate(1,1)*v1;
            }
        }
    }

    void apply_single_remote(size_t qubit, const Matrix2cd& gate) {
        int partner = info_.get_partner_rank(qubit);
        if (partner == info_.world_rank) {
            apply_single_local(qubit, gate);
            return;
        }

        size_t partner_start = 0;
        size_t partner_rows = compute_rows_for_rank(partner, info_.global_dim, info_.world_size, partner_start);
        size_t rank_cols = static_cast<size_t>(local_L_.cols());

        size_t send_elems = info_.local_rows * rank_cols;
        size_t recv_elems = partner_rows * rank_cols;

        Complex* send_buf = buffers_.get_send_buffer(send_elems);
        Complex* recv_buf = buffers_.get_recv_buffer(recv_elems);
        std::copy(local_L_.data(), local_L_.data() + send_elems, send_buf);

        MPI_Sendrecv(
            send_buf, static_cast<int>(send_elems), MPI_CXX_DOUBLE_COMPLEX, partner, 0,
            recv_buf, static_cast<int>(recv_elems), MPI_CXX_DOUBLE_COMPLEX, partner, 0,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE
        );

        size_t bit = 1ULL << qubit;
        for (size_t local_r = 0; local_r < info_.local_rows; ++local_r) {
            size_t global_r = info_.row_start + local_r;
            size_t partner_global = global_r ^ bit;
            bool partner_is_remote = (partner_global < info_.row_start || partner_global >= info_.row_end);
            if (!partner_is_remote) continue;

            size_t partner_local_index = partner_global - partner_start;
            for (size_t c = 0; c < rank_cols; ++c) {
                Complex v0 = local_L_(static_cast<int>(local_r), static_cast<int>(c));
                Complex v1 = recv_buf[partner_local_index * rank_cols + c];
                local_L_(static_cast<int>(local_r), static_cast<int>(c)) = gate(0,0)*v0 + gate(0,1)*v1;
            }
        }

        stats_.total_messages_sent += 2;
        stats_.total_bytes_sent += (send_elems + recv_elems) * sizeof(Complex);
        stats_.remote_gate_ops += 1;
    }

    void apply_two_local(size_t q1, size_t q2, const Matrix4cd& gate) {
        size_t dim_local = info_.local_rows;
        size_t rank_cols = static_cast<size_t>(local_L_.cols());
        size_t b1 = 1ULL << q1;
        size_t b2 = 1ULL << q2;

        for (size_t local_r = 0; local_r < dim_local; ++local_r) {
            size_t g = info_.row_start + local_r;
            if (((g & b1) != 0) || ((g & b2) != 0)) continue;

            size_t r01 = g ^ b1;
            size_t r10 = g ^ b2;
            size_t r11 = g ^ b1 ^ b2;

            if (r01 < info_.row_start || r01 >= info_.row_end) continue;
            if (r10 < info_.row_start || r10 >= info_.row_end) continue;
            if (r11 < info_.row_start || r11 >= info_.row_end) continue;

            size_t l00 = local_r;
            size_t l01 = r01 - info_.row_start;
            size_t l10 = r10 - info_.row_start;
            size_t l11 = r11 - info_.row_start;

            for (size_t c = 0; c < rank_cols; ++c) {
                Complex v00 = local_L_(static_cast<int>(l00), static_cast<int>(c));
                Complex v01 = local_L_(static_cast<int>(l01), static_cast<int>(c));
                Complex v10 = local_L_(static_cast<int>(l10), static_cast<int>(c));
                Complex v11 = local_L_(static_cast<int>(l11), static_cast<int>(c));

                Complex out00 = gate(0,0)*v00 + gate(0,1)*v01 + gate(0,2)*v10 + gate(0,3)*v11;
                Complex out01 = gate(1,0)*v00 + gate(1,1)*v01 + gate(1,2)*v10 + gate(1,3)*v11;
                Complex out10 = gate(2,0)*v00 + gate(2,1)*v01 + gate(2,2)*v10 + gate(2,3)*v11;
                Complex out11 = gate(3,0)*v00 + gate(3,1)*v01 + gate(3,2)*v10 + gate(3,3)*v11;

                local_L_(static_cast<int>(l00), static_cast<int>(c)) = out00;
                local_L_(static_cast<int>(l01), static_cast<int>(c)) = out01;
                local_L_(static_cast<int>(l10), static_cast<int>(c)) = out10;
                local_L_(static_cast<int>(l11), static_cast<int>(c)) = out11;
            }
        }
    }

    void apply_two_remote(size_t q1, size_t q2, const Matrix4cd& gate) {
        size_t q_high = std::max(q1, q2);
        int partner = info_.get_partner_rank(q_high);
        if (partner == info_.world_rank) {
            apply_two_local(q1, q2, gate);
            return;
        }

        size_t partner_start = 0;
        size_t partner_rows = compute_rows_for_rank(partner, info_.global_dim, info_.world_size, partner_start);
        size_t rank_cols = static_cast<size_t>(local_L_.cols());

        size_t send_elems = info_.local_rows * rank_cols;
        size_t recv_elems = partner_rows * rank_cols;

        Complex* send_buf = buffers_.get_send_buffer(send_elems);
        Complex* recv_buf = buffers_.get_recv_buffer(recv_elems);
        std::copy(local_L_.data(), local_L_.data() + send_elems, send_buf);

        MPI_Sendrecv(
            send_buf, static_cast<int>(send_elems), MPI_CXX_DOUBLE_COMPLEX, partner, 0,
            recv_buf, static_cast<int>(recv_elems), MPI_CXX_DOUBLE_COMPLEX, partner, 0,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE
        );

        size_t b1 = 1ULL << q1;
        size_t b2 = 1ULL << q2;

        for (size_t local_r = 0; local_r < info_.local_rows; ++local_r) {
            size_t g = info_.row_start + local_r;
            if (((g & b1) != 0) || ((g & b2) != 0)) continue;

            size_t r01 = g ^ b1;
            size_t r10 = g ^ b2;
            size_t r11 = g ^ b1 ^ b2;

            for (size_t c = 0; c < rank_cols; ++c) {
                Complex v00 = local_L_(static_cast<int>(local_r), static_cast<int>(c));
                Complex v01, v10, v11;

                if (r01 >= info_.row_start && r01 < info_.row_end) {
                    v01 = local_L_(static_cast<int>(r01 - info_.row_start), static_cast<int>(c));
                } else {
                    v01 = recv_buf[(r01 - partner_start) * rank_cols + c];
                }
                if (r10 >= info_.row_start && r10 < info_.row_end) {
                    v10 = local_L_(static_cast<int>(r10 - info_.row_start), static_cast<int>(c));
                } else {
                    v10 = recv_buf[(r10 - partner_start) * rank_cols + c];
                }
                if (r11 >= info_.row_start && r11 < info_.row_end) {
                    v11 = local_L_(static_cast<int>(r11 - info_.row_start), static_cast<int>(c));
                } else {
                    v11 = recv_buf[(r11 - partner_start) * rank_cols + c];
                }

                Complex out00 = gate(0,0)*v00 + gate(0,1)*v01 + gate(0,2)*v10 + gate(0,3)*v11;
                local_L_(static_cast<int>(local_r), static_cast<int>(c)) = out00;
            }
        }

        stats_.total_messages_sent += 2;
        stats_.total_bytes_sent += (send_elems + recv_elems) * sizeof(Complex);
        stats_.remote_gate_ops += 1;
    }

    void apply_gate(const GateOp& gate) {
        if (gate.qubits.size() == 1) {
            Matrix2cd U = get_single_qubit_gate(gate.type, gate.params);
            if (info_.is_qubit_local(gate.qubits[0])) {
                apply_single_local(gate.qubits[0], U);
                stats_.local_gate_ops += 1;
            } else {
                apply_single_remote(gate.qubits[0], U);
            }
        } else if (gate.qubits.size() == 2) {
            Matrix4cd U = get_two_qubit_gate(gate.type, gate.params);
            bool local1 = info_.is_qubit_local(gate.qubits[0]);
            bool local2 = info_.is_qubit_local(gate.qubits[1]);
            if (local1 && local2) {
                apply_two_local(gate.qubits[0], gate.qubits[1], U);
                stats_.local_gate_ops += 1;
            } else {
                apply_two_remote(gate.qubits[0], gate.qubits[1], U);
            }
        }
    }

    void apply_noise(const NoiseOp& noise, const SimConfig& config) {
        auto kraus_ops = get_kraus_operators(noise.type, noise.probability);
        if (!kraus_ops.empty()) {
            apply_single_local(noise.qubit, kraus_ops[0]);
        }
        if (config.do_truncation) {
            truncate(config.truncation_threshold);
        }
    }

    size_t truncate(double threshold) {
        double local_norm_sq = 0.0;
        for (int c = 0; c < local_L_.cols(); ++c) {
            local_norm_sq += local_L_.col(c).squaredNorm();
        }

        double global_norm_sq = 0.0;
        MPI_Allreduce(&local_norm_sq, &global_norm_sq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        std::vector<int> keep;
        for (int c = 0; c < local_L_.cols(); ++c) {
            double col_norm = local_L_.col(c).squaredNorm();
            if (col_norm > threshold * threshold * global_norm_sq) {
                keep.push_back(c);
            }
        }
        if (keep.empty()) keep.push_back(0);

        MatrixXcd truncated(local_L_.rows(), static_cast<int>(keep.size()));
        for (size_t i = 0; i < keep.size(); ++i) {
            truncated.col(static_cast<int>(i)) = local_L_.col(keep[i]);
        }
        local_L_.swap(truncated);
        return keep.size();
    }

    MatrixXcd gather_result() const {
        int rank = info_.world_rank;
        int size = info_.world_size;
        int cols = local_L_.cols();

        std::vector<int> recv_counts(size, 0);
        std::vector<int> displs(size, 0);
        int local_count = static_cast<int>(info_.local_rows * cols);

        if (rank == 0) {
            for (int r = 0; r < size; ++r) {
                size_t start;
                size_t rows = compute_rows_for_rank(r, info_.global_dim, size, start);
                recv_counts[r] = static_cast<int>(rows * cols);
                displs[r] = static_cast<int>(start * cols);
            }
        }

        MatrixXcd out;
        if (rank == 0) {
            out.resize(static_cast<int>(info_.global_dim), cols);
        }

        MPI_Gatherv(
            local_L_.data(), local_count, MPI_CXX_DOUBLE_COMPLEX,
            rank == 0 ? out.data() : nullptr,
            recv_counts.data(), displs.data(), MPI_CXX_DOUBLE_COMPLEX,
            0, MPI_COMM_WORLD
        );
        return out;
    }

    MatrixXcd get_local_chunk() const { return local_L_; }
    size_t get_global_rank_estimate() const { return static_cast<size_t>(local_L_.cols()); }
    double get_global_trace() const {
        double local_sq = local_L_.squaredNorm();
        double global_sq = 0.0;
        MPI_Allreduce(&local_sq, &global_sq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        return global_sq;
    }
};

//==============================================================================
// MPISimulator facade
//==============================================================================

MPISimulator::MPISimulator(size_t num_qubits, const MPIConfig& config)
    : impl_(std::make_unique<MPISimulatorImpl>(num_qubits, config)) {}

MPISimulator::~MPISimulator() = default;
MPISimulator::MPISimulator(MPISimulator&&) noexcept = default;
MPISimulator& MPISimulator::operator=(MPISimulator&&) noexcept = default;

void MPISimulator::initialize_state(const MatrixXcd& L, bool is_local_chunk) {
    impl_->initialize_state(L, is_local_chunk);
}

void MPISimulator::initialize_zero_state() { impl_->initialize_zero_state(); }
void MPISimulator::initialize_random_state(size_t rank, unsigned int seed) { impl_->initialize_random_state(rank, seed); }
void MPISimulator::apply_gate(const GateOp& gate) { impl_->apply_gate(gate); }
void MPISimulator::apply_single_qubit_gate(size_t qubit, const Matrix2cd& gate) { (void)qubit; (void)gate; }
void MPISimulator::apply_two_qubit_gate(size_t qubit1, size_t qubit2, const Matrix4cd& gate) { (void)qubit1; (void)qubit2; (void)gate; }
void MPISimulator::apply_noise(const NoiseOp& noise) { SimConfig dummy; impl_->apply_noise(noise, dummy); }
size_t MPISimulator::truncate(double threshold) { return impl_->truncate(threshold); }
MatrixXcd MPISimulator::gather_result() const { return impl_->gather_result(); }
MatrixXcd MPISimulator::get_local_chunk() const { return impl_->get_local_chunk(); }
size_t MPISimulator::get_global_rank() const { return impl_->get_global_rank_estimate(); }
double MPISimulator::get_global_trace() const { return impl_->get_global_trace(); }
const MPIProcessInfo& MPISimulator::get_process_info() const { return impl_->info_; }
const MPICommStats& MPISimulator::get_comm_stats() const { return impl_->stats_; }
void MPISimulator::reset_comm_stats() { impl_->stats_.reset(); }
void MPISimulator::barrier() const { MPI_Barrier(MPI_COMM_WORLD); }
bool MPISimulator::is_root() const { return impl_->info_.is_root(); }

//==============================================================================
// High-level simulation helpers
//==============================================================================

MatrixXcd simulate_mpi(
    const MatrixXcd& L_init,
    const QuantumSequence& sequence,
    size_t num_qubits,
    const SimConfig& config,
    const MPIConfig& mpi_config
) {
    MPISimulator sim(num_qubits, mpi_config);
    sim.initialize_state(L_init, false);

    for (const auto& op : sequence.operations) {
        if (std::holds_alternative<GateOp>(op)) {
            sim.apply_gate(std::get<GateOp>(op));
        } else {
            sim.apply_noise(std::get<NoiseOp>(op));
        }
    }

    return sim.gather_result();
}

MatrixXcd simulate_distributed(
    const MatrixXcd& L_init,
    const QuantumSequence& sequence,
    size_t num_qubits,
    const SimConfig& config,
    bool prefer_mpi
) {
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized || !prefer_mpi) {
        extern MatrixXcd run_lret_simulation(
            const MatrixXcd&, const QuantumSequence&, size_t, const SimConfig&);
        return run_lret_simulation(L_init, sequence, num_qubits, config);
    }

    MPIConfig mpi_cfg;
    return simulate_mpi(L_init, sequence, num_qubits, config, mpi_cfg);
}

//==============================================================================
// MPI query helpers
//==============================================================================

bool is_mpi_available() {
    int flag = 0;
    MPI_Initialized(&flag);
    return flag != 0;
}

bool is_mpi_root() {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank == 0;
}

int get_mpi_size() {
    int size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    return size;
}

int get_mpi_rank() {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
}

void mpi_init(int* argc, char*** argv) {
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized) {
        MPI_Init(argc, argv);
    }
}

void mpi_finalize() {
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (!finalized) {
        MPI_Finalize();
    }
}

void print_mpi_info() {
    int rank = get_mpi_rank();
    int size = get_mpi_size();
    if (rank == 0) {
        std::cout << "MPI world size: " << size << std::endl;
    }
}

//==============================================================================
// mpi_ops utilities
//==============================================================================

namespace mpi_ops {

double allreduce_sum(double local_value) {
    double global = 0.0;
    MPI_Allreduce(&local_value, &global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return global;
}

double allreduce_max(double local_value) {
    double global = 0.0;
    MPI_Allreduce(&local_value, &global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return global;
}

double allreduce_min(double local_value) {
    double global = 0.0;
    MPI_Allreduce(&local_value, &global, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    return global;
}

void broadcast(double& value, int root) {
    MPI_Bcast(&value, 1, MPI_DOUBLE, root, MPI_COMM_WORLD);
}

void broadcast(size_t& value, int root) {
    MPI_Bcast(&value, 1, MPI_UNSIGNED_LONG_LONG, root, MPI_COMM_WORLD);
}

void broadcast_matrix(MatrixXcd& matrix, int root) {
    int rank = get_mpi_rank();
    int rows = static_cast<int>(matrix.rows());
    int cols = static_cast<int>(matrix.cols());
    MPI_Bcast(&rows, 1, MPI_INT, root, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, root, MPI_COMM_WORLD);
    if (rank != root) {
        matrix.resize(rows, cols);
    }
    MPI_Bcast(matrix.data(), rows * cols, MPI_CXX_DOUBLE_COMPLEX, root, MPI_COMM_WORLD);
}

MatrixXcd scatter_rows(const MatrixXcd& full_matrix, int root) {
    int rank = get_mpi_rank();
    int size = get_mpi_size();
    size_t global_rows = static_cast<size_t>(full_matrix.rows());
    size_t cols = static_cast<size_t>(full_matrix.cols());

    std::vector<int> send_counts(size, 0);
    std::vector<int> displs(size, 0);
    if (rank == root) {
        for (int r = 0; r < size; ++r) {
            size_t start;
            size_t rows = compute_rows_for_rank(r, global_rows, size, start);
            send_counts[r] = static_cast<int>(rows * cols);
            displs[r] = static_cast<int>(start * cols);
        }
    }

    size_t start_local = 0;
    size_t local_rows = compute_rows_for_rank(rank, global_rows, size, start_local);
    MatrixXcd local(local_rows, cols);

    MPI_Scatterv(
        rank == root ? full_matrix.data() : nullptr,
        send_counts.data(), displs.data(), MPI_CXX_DOUBLE_COMPLEX,
        local.data(), static_cast<int>(local_rows * cols), MPI_CXX_DOUBLE_COMPLEX,
        root, MPI_COMM_WORLD);
    return local;
}

MatrixXcd gather_rows(const MatrixXcd& local_chunk, int root) {
    int rank = get_mpi_rank();
    int size = get_mpi_size();

    size_t cols = static_cast<size_t>(local_chunk.cols());
    int local_rows_int = static_cast<int>(local_chunk.rows());
    int global_rows_int = 0;
    MPI_Allreduce(&local_rows_int, &global_rows_int, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    size_t global_rows = static_cast<size_t>(global_rows_int);

    std::vector<int> recv_counts(size, 0);
    std::vector<int> displs(size, 0);
    if (rank == root) {
        for (int r = 0; r < size; ++r) {
            size_t start;
            size_t rows = compute_rows_for_rank(r, global_rows, size, start);
            recv_counts[r] = static_cast<int>(rows * cols);
            displs[r] = static_cast<int>(start * cols);
        }
    }

    MatrixXcd gathered;
    if (rank == root) {
        gathered.resize(static_cast<int>(global_rows), static_cast<int>(cols));
    }

    MPI_Gatherv(
        local_chunk.data(), static_cast<int>(local_chunk.size()), MPI_CXX_DOUBLE_COMPLEX,
        rank == root ? gathered.data() : nullptr,
        recv_counts.data(), displs.data(), MPI_CXX_DOUBLE_COMPLEX,
        root, MPI_COMM_WORLD);
    return gathered;
}

} // namespace mpi_ops

} // namespace qlret

#else // USE_MPI not defined

//==============================================================================
// Stubs when MPI is not available
//==============================================================================

namespace qlret {

MPISimulator::MPISimulator(size_t, const MPIConfig&) {
    throw std::runtime_error("MPI not enabled. Build with -DUSE_MPI=ON");
}
MPISimulator::~MPISimulator() = default;
MPISimulator::MPISimulator(MPISimulator&&) noexcept = default;
MPISimulator& MPISimulator::operator=(MPISimulator&&) noexcept = default;
void MPISimulator::initialize_state(const MatrixXcd&, bool) {}
void MPISimulator::initialize_zero_state() {}
void MPISimulator::initialize_random_state(size_t, unsigned int) {}
void MPISimulator::apply_gate(const GateOp&) {}
void MPISimulator::apply_single_qubit_gate(size_t, const Matrix2cd&) {}
void MPISimulator::apply_two_qubit_gate(size_t, size_t, const Matrix4cd&) {}
void MPISimulator::apply_noise(const NoiseOp&) {}
size_t MPISimulator::truncate(double) { return 0; }
MatrixXcd MPISimulator::gather_result() const { return MatrixXcd(); }
MatrixXcd MPISimulator::get_local_chunk() const { return MatrixXcd(); }
size_t MPISimulator::get_global_rank() const { return 0; }
double MPISimulator::get_global_trace() const { return 0.0; }
const MPIProcessInfo& MPISimulator::get_process_info() const { static MPIProcessInfo info; return info; }
const MPICommStats& MPISimulator::get_comm_stats() const { static MPICommStats s; return s; }
void MPISimulator::reset_comm_stats() {}
void MPISimulator::barrier() const {}
bool MPISimulator::is_root() const { return true; }

MatrixXcd simulate_mpi(const MatrixXcd&, const QuantumSequence&, size_t, const SimConfig&, const MPIConfig&) {
    throw std::runtime_error("MPI not enabled. Build with -DUSE_MPI=ON");
}

MatrixXcd simulate_distributed(const MatrixXcd& L_init, const QuantumSequence& sequence,
                               size_t num_qubits, const SimConfig& config, bool) {
    extern MatrixXcd run_lret_simulation(
        const MatrixXcd&, const QuantumSequence&, size_t, const SimConfig&);
    return run_lret_simulation(L_init, sequence, num_qubits, config);
}

bool is_mpi_available() { return false; }
bool is_mpi_root() { return true; }
int get_mpi_size() { return 1; }
int get_mpi_rank() { return 0; }
void mpi_init(int*, char***) {}
void mpi_finalize() {}
void print_mpi_info() { std::cout << "MPI not enabled. Build with -DUSE_MPI=ON" << std::endl; }

namespace mpi_ops {
double allreduce_sum(double v) { return v; }
double allreduce_max(double v) { return v; }
double allreduce_min(double v) { return v; }
void broadcast(double&, int) {}
void broadcast(size_t&, int) {}
void broadcast_matrix(MatrixXcd&, int) {}
MatrixXcd scatter_rows(const MatrixXcd& m, int) { return m; }
MatrixXcd gather_rows(const MatrixXcd& m, int) { return m; }
} // namespace mpi_ops

} // namespace qlret

#endif // USE_MPI

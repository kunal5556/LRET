#include "mpi_parallel.h"
#include "simulator.h"
#include "gates_and_noise.h"
#include "utils.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <iostream>

#ifdef USE_MPI

namespace qlret {

//==============================================================================
// Helper utilities
//==============================================================================

static size_t compute_rows_for_rank(int rank, size_t global_dim, int world_size, size_t& row_start_out) {
    size_t base = global_dim / world_size;
    size_t rem  = global_dim % world_size;
    size_t start = rank * base + std::min<size_t>(rank, rem);
    size_t rows  = base + (rank < rem ? 1 : 0);
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
    // When processes are treated as contiguous blocks, partner rank toggles the
    // bit corresponding to (qubit - local_qubit)
    size_t local_bits = floor_log2_size_t(local_rows);
    if (qubit < local_bits) return world_rank; // local, no partner needed
    size_t offset_bit = qubit - local_bits;
    int partner = world_rank ^ (1 << static_cast<int>(offset_bit));
    return partner;
}

void MPIProcessInfo::print() const {
    std::cout << "MPI rank " << world_rank << "/" << world_size
              << " rows [" << row_start << "," << row_end << ")"
              << " cols [" << col_start << "," << col_end << ")"
              << " local_rows=" << local_rows
              << " local_cols=" << local_cols
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
              << " comm_time=" << total_comm_time
              << " compute_time=" << total_compute_time
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
    MatrixXcd local_L_; // local rows only

    MPISimulatorImpl(size_t num_qubits, const MPIConfig& config)
        : config_(config) {
        // Basic MPI info
        MPI_Comm_rank(MPI_COMM_WORLD, &info_.world_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &info_.world_size);

        info_.num_qubits = num_qubits;
        info_.global_dim = (1ULL << num_qubits);

        size_t row_start = 0;
        info_.local_rows = compute_rows_for_rank(info_.world_rank, info_.global_dim, info_.world_size, row_start);
        info_.row_start = row_start;
        info_.row_end = row_start + info_.local_rows;
        info_.local_qubit = floor_log2_size_t(info_.local_rows);

        // Column distribution (not used yet, placeholder for future)
        info_.col_start = 0;
        info_.col_end = 0;
        info_.local_cols = 0;

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

        // Root scatters rows to all processes
        int rank = info_.world_rank;
        int size = info_.world_size;

        // Prepare send counts and displacements
        std::vector<int> send_counts(size, 0);
        std::vector<int> displs(size, 0);

        if (rank == 0) {
            size_t offset = 0;
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

    void initialize_random_state(size_t rank, unsigned int seed) {
        std::mt19937 rng(seed ? seed + info_.world_rank : static_cast<unsigned int>(std::random_device{}()));
        std::normal_distribution<double> dist(0.0, 1.0);
        local_L_.resize(info_.local_rows, rank);
        for (int r = 0; r < local_L_.rows(); ++r) {
            for (int c = 0; c < local_L_.cols(); ++c) {
                local_L_(r, c) = Complex(dist(rng), dist(rng));
            }
        }
    }

    //---------------------------------------------------------------------------
    // Local gate application helpers
    //---------------------------------------------------------------------------

    void apply_single_local(size_t qubit, const Matrix2cd& gate) {
        size_t dim_local = info_.local_rows;
        size_t rank_cols = static_cast<size_t>(local_L_.cols());
        size_t bit = 1ULL << qubit;

        for (size_t local_r = 0; local_r < dim_local; ++local_r) {
            size_t global_r = info_.row_start + local_r;
            if ((global_r & bit) != 0) continue; // handle each pair once
            size_t partner_global = global_r ^ bit;
            if (partner_global < info_.row_start || partner_global >= info_.row_end) continue; // partner not local
            size_t partner_local = partner_global - info_.row_start;

            for (size_t c = 0; c < rank_cols; ++c) {
                Complex v0 = local_L_(static_cast<int>(local_r), static_cast<int>(c));
                Complex v1 = local_L_(static_cast<int>(partner_local), static_cast<int>(c));
                local_L_(static_cast<int>(local_r), static_cast<int>(c)) = gate(0,0)*v0 + gate(0,1)*v1;
                local_L_(static_cast<int>(partner_local), static_cast<int>(c)) = gate(1,0)*v0 + gate(1,1)*v1;
            }
        }
    }

    // Apply gate when partner rows are remote. Full exchange with partner rank.
    void apply_single_remote(size_t qubit, const Matrix2cd& gate) {
        int partner = info_.get_partner_rank(qubit);
        if (partner == info_.world_rank) {
            apply_single_local(qubit, gate);
            return;
        }

        size_t partner_start = 0;
        size_t partner_rows = compute_rows_for_rank(partner, info_.global_dim, info_.world_size, partner_start);
        size_t rank_cols = static_cast<size_t>(local_L_.cols());

        // Exchange full chunks (simpler, though not minimal)
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

        // Update local rows that pair with partner rows
        size_t bit = 1ULL << qubit;
        for (size_t local_r = 0; local_r < info_.local_rows; ++local_r) {
            size_t global_r = info_.row_start + local_r;
            size_t partner_global = global_r ^ bit;
            bool partner_is_remote = (partner_global < info_.row_start || partner_global >= info_.row_end);
            if (!partner_is_remote) continue;

            size_t partner_local_index = partner_global - partner_start; // index in partner chunk
            for (size_t c = 0; c < rank_cols; ++c) {
                Complex v0 = local_L_(static_cast<int>(local_r), static_cast<int>(c));
                Complex v1 = recv_buf[partner_local_index * rank_cols + c];
                local_L_(static_cast<int>(local_r), static_cast<int>(c)) = gate(0,0)*v0 + gate(0,1)*v1;
            }
        }

        stats_.total_messages_sent += 2; // send + recv
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
            if (((g & b1) != 0) || ((g & b2) != 0)) continue; // process base |00






















































































































































































































































































































































































































































































































































































































#endif // USE_MPI} // namespace qlret} // namespace mpi_opsMatrixXcd gather_rows(const MatrixXcd& m, int) { return m; }MatrixXcd scatter_rows(const MatrixXcd& m, int) { return m; }void broadcast_matrix(MatrixXcd&, int) {}void broadcast(size_t&, int) {}void broadcast(double&, int) {}double allreduce_min(double v) { return v; }double allreduce_max(double v) { return v; }double allreduce_sum(double v) { return v; }namespace mpi_ops {#endif}    throw std::runtime_error("MPI/GPU hybrid not enabled. Build with -DUSE_MPI=ON");MatrixXcd simulate_mpi_gpu(const MatrixXcd&, const QuantumSequence&, size_t, const SimConfig&, const MPIConfig&, const GPUConfig&) {#ifdef USE_GPUvoid print_mpi_info() { std::cout << "MPI not enabled. Build with -DUSE_MPI=ON" << std::endl; }void mpi_finalize() {}void mpi_init(int*, char***) {}int get_mpi_rank() { return 0; }int get_mpi_size() { return 1; }bool is_mpi_root() { return true; }bool is_mpi_available() { return false; }}    return run_lret_simulation(L_init, sequence, num_qubits, config);        const MatrixXcd&, const QuantumSequence&, size_t, const SimConfig&);    extern MatrixXcd run_lret_simulation(                               size_t num_qubits, const SimConfig& config, bool) {MatrixXcd simulate_distributed(const MatrixXcd& L_init, const QuantumSequence& sequence,}    throw std::runtime_error("MPI not enabled. Build with -DUSE_MPI=ON");MatrixXcd simulate_mpi(const MatrixXcd&, const QuantumSequence&, size_t, const SimConfig&, const MPIConfig&) {bool MPISimulator::is_root() const { return true; }void MPISimulator::barrier() const {}void MPISimulator::reset_comm_stats() {}const MPICommStats& MPISimulator::get_comm_stats() const { static MPICommStats s; return s; }const MPIProcessInfo& MPISimulator::get_process_info() const { static MPIProcessInfo info; return info; }double MPISimulator::get_global_trace() const { return 0.0; }size_t MPISimulator::get_global_rank() const { return 0; }MatrixXcd MPISimulator::get_local_chunk() const { return MatrixXcd(); }MatrixXcd MPISimulator::gather_result() const { return MatrixXcd(); }size_t MPISimulator::truncate(double) { return 0; }void MPISimulator::apply_noise(const NoiseOp&) {}void MPISimulator::apply_two_qubit_gate(size_t, size_t, const Matrix4cd&) {}void MPISimulator::apply_single_qubit_gate(size_t, const Matrix2cd&) {}void MPISimulator::apply_gate(const GateOp&) {}void MPISimulator::initialize_random_state(size_t, unsigned int) {}void MPISimulator::initialize_zero_state() {}void MPISimulator::initialize_state(const MatrixXcd&, bool) {}MPISimulator& MPISimulator::operator=(MPISimulator&&) noexcept = default;MPISimulator::MPISimulator(MPISimulator&&) noexcept = default;MPISimulator::~MPISimulator() = default;}    throw std::runtime_error("MPI not enabled. Build with -DUSE_MPI=ON");MPISimulator::MPISimulator(size_t, const MPIConfig&) {// Stubs when MPI is not availablenamespace qlret {#else // USE_MPI not defined} // namespace qlret} // namespace mpi_ops}    return gathered;        root, MPI_COMM_WORLD);        recv_counts.data(), displs.data(), MPI_CXX_DOUBLE_COMPLEX,        rank == root ? gathered.data() : nullptr,        local_chunk.data(), static_cast<int>(local_chunk.size()), MPI_CXX_DOUBLE_COMPLEX,    MPI_Gatherv(    }        gathered.resize(static_cast<int>(global_rows), static_cast<int>(cols));    if (rank == root) {    MatrixXcd gathered;    }        }            displs[r] = static_cast<int>(start * cols);            recv_counts[r] = static_cast<int>(rows * cols);            size_t rows = compute_rows_for_rank(r, global_rows, size, start);            size_t start;        for (int r = 0; r < size; ++r) {        int disp = 0;    if (rank == root) {    std::vector<int> displs(size, 0);    std::vector<int> recv_counts(size, 0);    MPI_Allreduce(&local_chunk.rows(), &global_rows, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);    size_t global_rows = 0;    size_t cols = static_cast<size_t>(local_chunk.cols());    int size = get_mpi_size();    int rank = get_mpi_rank();MatrixXcd gather_rows(const MatrixXcd& local_chunk, int root) {}    return local;        root, MPI_COMM_WORLD);        local.data(), static_cast<int>(local_rows * cols), MPI_CXX_DOUBLE_COMPLEX,        send_counts.data(), displs.data(), MPI_CXX_DOUBLE_COMPLEX,        rank == root ? full_matrix.data() : nullptr,    MPI_Scatterv(    MatrixXcd local(local_rows, cols);    size_t local_rows = compute_rows_for_rank(rank, global_rows, size, start_local);    size_t start_local = 0;    }        }            displs[r] = static_cast<int>(start * cols);            send_counts[r] = static_cast<int>(rows * cols);            size_t rows = compute_rows_for_rank(r, global_rows, size, start);            size_t start;        for (int r = 0; r < size; ++r) {    if (rank == root) {    std::vector<int> displs(size, 0);    std::vector<int> send_counts(size, 0);    size_t cols = static_cast<size_t>(full_matrix.cols());    size_t global_rows = static_cast<size_t>(full_matrix.rows());    int size = get_mpi_size();    int rank = get_mpi_rank();MatrixXcd scatter_rows(const MatrixXcd& full_matrix, int root) {}    MPI_Bcast(matrix.data(), rows * cols, MPI_CXX_DOUBLE_COMPLEX, root, MPI_COMM_WORLD);    }        matrix.resize(rows, cols);    if (rank != root) {    MPI_Bcast(&cols, 1, MPI_INT, root, MPI_COMM_WORLD);    MPI_Bcast(&rows, 1, MPI_INT, root, MPI_COMM_WORLD);    int cols = static_cast<int>(matrix.cols());    int rows = static_cast<int>(matrix.rows());    int rank = get_mpi_rank();void broadcast_matrix(MatrixXcd& matrix, int root) {}    MPI_Bcast(&value, 1, MPI_UNSIGNED_LONG_LONG, root, MPI_COMM_WORLD);void broadcast(size_t& value, int root) {}    MPI_Bcast(&value, 1, MPI_DOUBLE, root, MPI_COMM_WORLD);void broadcast(double& value, int root) {}    return global;    MPI_Allreduce(&local_value, &global, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);    double global = 0.0;double allreduce_min(double local_value) {}    return global;    MPI_Allreduce(&local_value, &global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);    double global = 0.0;double allreduce_max(double local_value) {}    return global;    MPI_Allreduce(&local_value, &global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);    double global = 0.0;double allreduce_sum(double local_value) {namespace mpi_ops {//==============================================================================// mpi_ops utilities//==============================================================================#endif}    return simulate_mpi(L_init, sequence, num_qubits, config, mpi_config);    (void)gpu_config;    // For now, fall back to CPU MPI simulation; GPU-aware MPI TBD) {    const GPUConfig& gpu_config    const MPIConfig& mpi_config,    const SimConfig& config,    size_t num_qubits,    const QuantumSequence& sequence,    const MatrixXcd& L_init,MatrixXcd simulate_mpi_gpu(#ifdef USE_GPU//==============================================================================// MPI + GPU hybrid stub (for future GPU-aware MPI)//==============================================================================}    }        std::cout << "MPI world size: " << size << std::endl;    if (rank == 0) {    int size = get_mpi_size();    int rank = get_mpi_rank();void print_mpi_info() {}    }        MPI_Finalize();    if (!finalized) {    MPI_Finalized(&finalized);    int finalized = 0;void mpi_finalize() {}    }        MPI_Init(argc, argv);    if (!initialized) {    MPI_Initialized(&initialized);    int initialized = 0;void mpi_init(int* argc, char*** argv) {}    return rank;    MPI_Comm_rank(MPI_COMM_WORLD, &rank);    int rank = 0;int get_mpi_rank() {}    return size;    MPI_Comm_size(MPI_COMM_WORLD, &size);    int size = 1;int get_mpi_size() {}    return rank == 0;    MPI_Comm_rank(MPI_COMM_WORLD, &rank);    int rank = 0;bool is_mpi_root() {}    return flag != 0;    MPI_Initialized(&flag);    int flag = 0;bool is_mpi_available() {//==============================================================================// MPI query helpers//==============================================================================}    return simulate_mpi(L_init, sequence, num_qubits, config, mpi_cfg);    MPIConfig mpi_cfg;    }        return run_lret_simulation(L_init, sequence, num_qubits, config);            const MatrixXcd&, const QuantumSequence&, size_t, const SimConfig&);        extern MatrixXcd run_lret_simulation(    if (!prefer_mpi) {    }        return run_lret_simulation(L_init, sequence, num_qubits, config);            const MatrixXcd&, const QuantumSequence&, size_t, const SimConfig&);        extern MatrixXcd run_lret_simulation(        // MPI not initialized; fall back    if (!initialized) {    MPI_Initialized(&initialized);    int initialized = 0;) {    bool prefer_mpi    const SimConfig& config,    size_t num_qubits,    const QuantumSequence& sequence,    const MatrixXcd& L_init,MatrixXcd simulate_distributed(}    return sim.gather_result();    }        }            sim.apply_noise(std::get<NoiseOp>(op));        } else {            sim.apply_gate(std::get<GateOp>(op));        if (std::holds_alternative<GateOp>(op)) {        ++step;    for (const auto& op : sequence.operations) {    size_t step = 0;    sim.initialize_state(L_init, false);    MPISimulator sim(num_qubits, mpi_config);) {    const MPIConfig& mpi_config    const SimConfig& config,    size_t num_qubits,    const QuantumSequence& sequence,    const MatrixXcd& L_init,MatrixXcd simulate_mpi(//==============================================================================// High-level simulation helpers//==============================================================================bool MPISimulator::is_root() const { return impl_->info_.is_root(); }void MPISimulator::barrier() const { MPI_Barrier(MPI_COMM_WORLD); }void MPISimulator::reset_comm_stats() { impl_->stats_.reset(); }const MPICommStats& MPISimulator::get_comm_stats() const { return impl_->stats_; }const MPIProcessInfo& MPISimulator::get_process_info() const { return impl_->info_; }}    return impl_->get_global_trace();double MPISimulator::get_global_trace() const {}    return impl_->get_global_rank_estimate();size_t MPISimulator::get_global_rank() const {}    return impl_->get_local_chunk();MatrixXcd MPISimulator::get_local_chunk() const {}    return impl_->gather_result();MatrixXcd MPISimulator::gather_result() const {}    return impl_->truncate(threshold);size_t MPISimulator::truncate(double threshold) {}    impl_->apply_noise(noise, dummy);    SimConfig dummy;void MPISimulator::apply_noise(const NoiseOp& noise) {}    (void)qubit1; (void)qubit2; (void)gate; // placeholdervoid MPISimulator::apply_two_qubit_gate(size_t qubit1, size_t qubit2, const Matrix4cd& gate) {}    (void)gate; // avoid unused warning    impl_->apply_gate(GateOp{GateType::CUSTOM_SINGLE, {qubit}, {}}); // placeholder unusedvoid MPISimulator::apply_single_qubit_gate(size_t qubit, const Matrix2cd& gate) {}    impl_->apply_gate(gate);void MPISimulator::apply_gate(const GateOp& gate) {}    impl_->initialize_random_state(rank, seed);void MPISimulator::initialize_random_state(size_t rank, unsigned int seed) {}    impl_->initialize_zero_state();void MPISimulator::initialize_zero_state() {}    impl_->initialize_state(L, is_local_chunk);void MPISimulator::initialize_state(const MatrixXcd& L, bool is_local_chunk) {MPISimulator& MPISimulator::operator=(MPISimulator&&) noexcept = default;MPISimulator::MPISimulator(MPISimulator&&) noexcept = default;MPISimulator::~MPISimulator() = default;    : impl_(std::make_unique<MPISimulatorImpl>(num_qubits, config)) {}MPISimulator::MPISimulator(size_t num_qubits, const MPIConfig& config)//==============================================================================// MPISimulator facade//==============================================================================};    }        return global_sq;        MPI_Allreduce(&local_sq, &global_sq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);        double global_sq = 0.0;        double local_sq = local_L_.squaredNorm();        // Trace = ||L||_F^2    double get_global_trace() const {    }        return static_cast<size_t>(local_L_.cols());        // approximate: assume all ranks same columns    size_t get_global_rank_estimate() const {    }        return local_L_;    MatrixXcd get_local_chunk() const {    }        return out;        );            0, MPI_COMM_WORLD            recv_counts.data(), displs.data(), MPI_CXX_DOUBLE_COMPLEX,            rank == 0 ? out.data() : nullptr,            local_L_.data(), local_count, MPI_CXX_DOUBLE_COMPLEX,        MPI_Gatherv(        }            out.resize(static_cast<int>(info_.global_dim), cols);        if (rank == 0) {        MatrixXcd out;        }            }                displs[r] = static_cast<int>(start * cols);                recv_counts[r] = static_cast<int>(rows * cols);                size_t rows = compute_rows_for_rank(r, info_.global_dim, size, start);                size_t start;            for (int r = 0; r < size; ++r) {        if (rank == 0) {        size_t dummy_start = 0;        int local_count = static_cast<int>(info_.local_rows * cols);        std::vector<int> displs(size, 0);        std::vector<int> recv_counts(size, 0);        // Gather row counts        int cols = local_L_.cols();        int size = info_.world_size;        int rank = info_.world_rank;    MatrixXcd gather_result() const {    }        return keep.size();        local_L_.swap(truncated);        }            truncated.col(static_cast<int>(i)) = local_L_.col(keep[i]);        for (size_t i = 0; i < keep.size(); ++i) {        MatrixXcd truncated(local_L_.rows(), static_cast<int>(keep.size()));        if (keep.empty()) keep.push_back(0);        }            }                keep.push_back(c);            if (col_norm > threshold * threshold * global_norm_sq) {            double col_norm = local_L_.col(c).squaredNorm();        for (int c = 0; c < local_L_.cols(); ++c) {        std::vector<int> keep;        // Simple column-wise thresholding based on global norm fraction        MPI_Allreduce(&local_norm_sq, &global_norm_sq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);        double global_norm_sq = 0.0;        }            local_norm_sq += local_L_.col(c).squaredNorm();        for (int c = 0; c < local_L_.cols(); ++c) {        double local_norm_sq = 0.0;        // Compute local squared norms    size_t truncate(double threshold) {    }        }            truncate(config.truncation_threshold);        if (config.do_truncation) {        }            apply_single_local(noise.qubit, kraus_ops[0]);        if (!kraus_ops.empty()) {        auto kraus_ops = get_kraus_operators(noise.type, noise.probability);        // Simplified: apply first Kraus operator; full expansion TODO    void apply_noise(const NoiseOp& noise, const SimConfig& config) {    }        }            }                apply_two_remote(gate.qubits[0], gate.qubits[1], U);            } else {                stats_.local_gate_ops += 1;                apply_two_local(gate.qubits[0], gate.qubits[1], U);            if (local1 && local2) {            bool local2 = info_.is_qubit_local(gate.qubits[1]);            bool local1 = info_.is_qubit_local(gate.qubits[0]);            Matrix4cd U = get_two_qubit_gate(gate.type, gate.params);        } else if (gate.qubits.size() == 2) {            }                apply_single_remote(gate.qubits[0], U);            } else {                stats_.local_gate_ops += 1;                apply_single_local(gate.qubits[0], U);            if (info_.is_qubit_local(gate.qubits[0])) {            Matrix2cd U = get_single_qubit_gate(gate.type, gate.params);        if (gate.qubits.size() == 1) {    void apply_gate(const GateOp& gate) {    //---------------------------------------------------------------------------    // Public operations    //---------------------------------------------------------------------------    }        stats_.remote_gate_ops += 1;        stats_.total_bytes_sent += (send_elems + recv_elems) * sizeof(Complex);        stats_.total_messages_sent += 2;        }            }                local_L_(static_cast<int>(local_r), static_cast<int>(c)) = out00;                Complex out00 = gate(0,0)*v00 + gate(0,1)*v01 + gate(0,2)*v10 + gate(0,3)*v11;                }                    v11 = recv_buf[(r11 - partner_start) * rank_cols + c];                } else {                    v11 = local_L_(static_cast<int>(r11 - info_.row_start), static_cast<int>(c));                if (r11 >= info_.row_start && r11 < info_.row_end) {                }                    v10 = recv_buf[(r10 - partner_start) * rank_cols + c];                } else {                    v10 = local_L_(static_cast<int>(r10 - info_.row_start), static_cast<int>(c));                if (r10 >= info_.row_start && r10 < info_.row_end) {                }                    v01 = recv_buf[(r01 - partner_start) * rank_cols + c];                } else {                    v01 = local_L_(static_cast<int>(r01 - info_.row_start), static_cast<int>(c));                if (r01 >= info_.row_start && r01 < info_.row_end) {                // fetch partner/local values                Complex v01, v10, v11;                Complex v00 = local_L_(static_cast<int>(local_r), static_cast<int>(c));            for (size_t c = 0; c < rank_cols; ++c) {            };                }                    out = recv_buf[l * rank_cols + 0]; // col handled outside loop                    size_t l = global_row - partner_start;                } else {                    out = local_L_(static_cast<int>(l), 0); // placeholder overwritten below                    size_t l = global_row - info_.row_start;                if (global_row >= info_.row_start && global_row < info_.row_end) {            auto get_val = [&](size_t global_row, Complex& out) {            // Resolve where each partner row lives            size_t r11 = g ^ b1 ^ b2;            size_t r10 = g ^ b2;            size_t r01 = g ^ b1;            if (((g & b1) != 0) || ((g & b2) != 0)) continue;            size_t g = info_.row_start + local_r;        for (size_t local_r = 0; local_r < info_.local_rows; ++local_r) {        size_t b2 = 1ULL << q2;        size_t b1 = 1ULL << q1;        );            MPI_COMM_WORLD, MPI_STATUS_IGNORE            recv_buf, static_cast<int>(recv_elems), MPI_CXX_DOUBLE_COMPLEX, partner, 0,            send_buf, static_cast<int>(send_elems), MPI_CXX_DOUBLE_COMPLEX, partner, 0,        MPI_Sendrecv(        std::copy(local_L_.data(), local_L_.data() + send_elems, send_buf);        Complex* recv_buf = buffers_.get_recv_buffer(recv_elems);        Complex* send_buf = buffers_.get_send_buffer(send_elems);        size_t recv_elems = partner_rows * rank_cols;        size_t send_elems = info_.local_rows * rank_cols;        size_t rank_cols = static_cast<size_t>(local_L_.cols());        size_t partner_rows = compute_rows_for_rank(partner, info_.global_dim, info_.world_size, partner_start);        size_t partner_start = 0;        }            return;            apply_two_local(q1, q2, gate);        if (partner == info_.world_rank) {        int partner = info_.get_partner_rank(q_high);        size_t q_high = std::max(q1, q2);        // Simplified: exchange full chunk with partner determined by highest qubit    void apply_two_remote(size_t q1, size_t q2, const Matrix4cd& gate) {    }        }            }                local_L_(static_cast<int>(l11), static_cast<int>(c)) = out11;                local_L_(static_cast<int>(l10), static_cast<int>(c)) = out10;                local_L_(static_cast<int>(l01), static_cast<int>(c)) = out01;                local_L_(static_cast<int>(l00), static_cast<int>(c)) = out00;                Complex out11 = gate(3,0)*v00 + gate(3,1)*v01 + gate(3,2)*v10 + gate(3,3)*v11;                Complex out10 = gate(2,0)*v00 + gate(2,1)*v01 + gate(2,2)*v10 + gate(2,3)*v11;                Complex out01 = gate(1,0)*v00 + gate(1,1)*v01 + gate(1,2)*v10 + gate(1,3)*v11;                Complex out00 = gate(0,0)*v00 + gate(0,1)*v01 + gate(0,2)*v10 + gate(0,3)*v11;                Complex v11 = local_L_(static_cast<int>(l11), static_cast<int>(c));                Complex v10 = local_L_(static_cast<int>(l10), static_cast<int>(c));                Complex v01 = local_L_(static_cast<int>(l01), static_cast<int>(c));                Complex v00 = local_L_(static_cast<int>(l00), static_cast<int>(c));            for (size_t c = 0; c < rank_cols; ++c) {            size_t l11 = r11 - info_.row_start;            size_t l10 = r10 - info_.row_start;            size_t l01 = r01 - info_.row_start;            size_t l00 = local_r;            if (r11 < info_.row_start || r11 >= info_.row_end) continue;            if (r10 < info_.row_start || r10 >= info_.row_end) continue;            if (r01 < info_.row_start || r01 >= info_.row_end) continue;            size_t r11 = g ^ b1 ^ b2;            size_t r10 = g ^ b2;            size_t r01 = g ^ b1;angle only
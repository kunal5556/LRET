#pragma once

#include "qec_types.h"
#include "qec_stabilizer.h"
#include "qec_syndrome.h"
#include "qec_decoder.h"
#include "qec_logical.h"
#include "types.h"
#include <vector>
#include <memory>
#include <functional>
#include <map>

namespace qlret {

//==============================================================================
// Partition Strategy
//==============================================================================

enum class PartitionStrategy {
    ROW_WISE,      // Each rank owns horizontal strip
    COLUMN_WISE,   // Each rank owns vertical strip
    BLOCK_2D,      // 2D block decomposition
    ROUND_ROBIN    // Stabilizers distributed round-robin
};

//==============================================================================
// Distributed QEC Configuration
//==============================================================================

struct DistributedQECConfig {
    size_t code_distance = 3;
    int world_size = 1;
    int rank = 0;
    PartitionStrategy partition = PartitionStrategy::ROW_WISE;
    QECCodeType code_type = QECCodeType::SURFACE;
    DecoderType decoder_type = DecoderType::MWPM;
    double physical_error_rate = 0.001;
    double measurement_error_rate = 0.001;
    
    // Communication settings
    bool use_async_comm = true;
    size_t comm_buffer_size = 4096;
    
    // Decoder settings
    bool parallel_decode = true;
    bool use_local_decode_first = true;  // Local decoding before merge
};

//==============================================================================
// Local Syndrome (per-rank)
//==============================================================================

struct LocalSyndrome {
    int rank;
    std::vector<int> x_syndrome;
    std::vector<int> z_syndrome;
    std::vector<size_t> local_stabilizer_indices;
    std::vector<size_t> boundary_stabilizer_indices;
    double extraction_time_ms = 0.0;
    
    size_t num_defects() const;
    bool has_boundary_defects() const;
};

//==============================================================================
// Global Syndrome (aggregated)
//==============================================================================

struct GlobalSyndrome {
    Syndrome full_syndrome;
    std::vector<LocalSyndrome> local_syndromes;
    std::vector<size_t> boundary_defects;  // Defects requiring multi-rank resolution
    double aggregation_time_ms = 0.0;
    
    size_t total_defects() const;
};

//==============================================================================
// Local Correction (per-rank)
//==============================================================================

struct LocalCorrection {
    int rank;
    PauliString x_correction;
    PauliString z_correction;
    std::vector<size_t> corrected_qubits;
    bool requires_boundary_merge = false;
    double decode_time_ms = 0.0;
};

//==============================================================================
// Global Correction (merged)
//==============================================================================

struct GlobalCorrection {
    Correction full_correction;
    std::vector<LocalCorrection> local_corrections;
    std::vector<std::pair<int, int>> boundary_merges;  // (rank1, rank2) pairs
    double merge_time_ms = 0.0;
};

//==============================================================================
// Partition Map
//==============================================================================

/**
 * @brief Maps qubits and stabilizers to ranks
 */
class PartitionMap {
public:
    PartitionMap(const StabilizerCode& code, 
                 const DistributedQECConfig& config);

    // Qubit ownership
    int qubit_owner(size_t qubit) const;
    std::vector<size_t> local_qubits(int rank) const;
    std::vector<size_t> boundary_qubits(int rank) const;
    
    // Stabilizer ownership
    int stabilizer_owner(size_t stab_idx, bool is_x) const;
    std::vector<size_t> local_x_stabilizers(int rank) const;
    std::vector<size_t> local_z_stabilizers(int rank) const;
    std::vector<size_t> boundary_stabilizers(int rank) const;
    
    // Rank communication partners
    std::vector<int> neighbor_ranks(int rank) const;
    
    const StabilizerCode& code() const { return code_; }
    const DistributedQECConfig& config() const { return config_; }

private:
    const StabilizerCode& code_;
    DistributedQECConfig config_;
    
    std::vector<int> qubit_to_rank_;
    std::vector<int> x_stab_to_rank_;
    std::vector<int> z_stab_to_rank_;
    std::map<int, std::vector<int>> rank_neighbors_;
    
    void build_row_partition();
    void build_column_partition();
    void build_block_partition();
    void build_round_robin_partition();
};

//==============================================================================
// Distributed Syndrome Extractor
//==============================================================================

/**
 * @brief Extracts syndrome in distributed setting
 */
class DistributedSyndromeExtractor {
public:
    DistributedSyndromeExtractor(const StabilizerCode& code,
                                  const PartitionMap& partition,
                                  const DistributedQECConfig& config);

    /**
     * @brief Extract local syndrome for this rank
     * @param error Current error pattern (local portion)
     * @return LocalSyndrome for this rank
     */
    LocalSyndrome extract_local(const PauliString& local_error);

    /**
     * @brief Gather syndromes from all ranks (simulated)
     * @param local Local syndrome from this rank
     * @param all_local Syndromes from other ranks (for simulation)
     * @return GlobalSyndrome with all data
     */
    GlobalSyndrome gather_syndromes(const LocalSyndrome& local,
                                    const std::vector<LocalSyndrome>& all_local);

    /**
     * @brief AllGather implementation using MPI (stub)
     */
    GlobalSyndrome allgather_syndromes_mpi(const LocalSyndrome& local);

    /**
     * @brief Handle boundary stabilizers requiring cross-rank data
     */
    void resolve_boundary_stabilizers(LocalSyndrome& local,
                                      const std::vector<LocalSyndrome>& neighbors);

    const PartitionMap& partition() const { return partition_; }

private:
    const StabilizerCode& code_;
    const PartitionMap& partition_;
    DistributedQECConfig config_;
    SyndromeExtractor local_extractor_;
};

//==============================================================================
// Parallel Decoder
//==============================================================================

/**
 * @brief Decodes syndrome in parallel across ranks
 */
class ParallelMWPMDecoder {
public:
    struct Config {
        bool use_local_decode = true;
        bool merge_at_boundaries = true;
        double boundary_weight_factor = 1.5;  // Extra weight for boundary edges
        size_t max_merge_iterations = 10;
    };

    ParallelMWPMDecoder(const StabilizerCode& code,
                        const PartitionMap& partition,
                        const DistributedQECConfig& qec_config);
    ParallelMWPMDecoder(const StabilizerCode& code,
                        const PartitionMap& partition,
                        const DistributedQECConfig& qec_config,
                        Config decoder_config);

    /**
     * @brief Decode local subgraph
     * @param local_syn Local syndrome for this rank
     * @return LocalCorrection for this rank's qubits
     */
    LocalCorrection decode_local(const LocalSyndrome& local_syn);

    /**
     * @brief Merge corrections across rank boundaries
     * @param all_local All local corrections
     * @return GlobalCorrection with merged result
     */
    GlobalCorrection merge_corrections(const std::vector<LocalCorrection>& all_local);

    /**
     * @brief Full parallel decode: local + merge
     * @param global_syn Global syndrome
     * @return GlobalCorrection
     */
    GlobalCorrection decode_parallel(const GlobalSyndrome& global_syn);

    /**
     * @brief Fallback: decode globally on rank 0
     * @param global_syn Global syndrome
     * @return GlobalCorrection
     */
    GlobalCorrection decode_global(const GlobalSyndrome& global_syn);

    const QECDecoder::DecoderStats& stats() const { return stats_; }

private:
    const StabilizerCode& code_;
    const PartitionMap& partition_;
    DistributedQECConfig qec_config_;
    Config decoder_config_;
    std::unique_ptr<QECDecoder> local_decoder_;
    QECDecoder::DecoderStats stats_;

    // Boundary matching
    std::vector<std::pair<size_t, size_t>> 
    find_boundary_defect_pairs(const GlobalSyndrome& global_syn);
    
    void apply_boundary_correction(GlobalCorrection& correction,
                                   const std::vector<std::pair<size_t, size_t>>& pairs);
};

//==============================================================================
// Distributed Logical Qubit
//==============================================================================

/**
 * @brief Logical qubit distributed across multiple ranks
 */
class DistributedLogicalQubit {
public:
    struct Stats {
        size_t total_qec_rounds = 0;
        size_t local_decode_count = 0;
        size_t global_decode_count = 0;
        size_t boundary_merges = 0;
        double total_extract_time_ms = 0.0;
        double total_decode_time_ms = 0.0;
        double total_comm_time_ms = 0.0;
    };

    explicit DistributedLogicalQubit(const DistributedQECConfig& config);

    //--------------------------------------------------------------------------
    // Initialization
    //--------------------------------------------------------------------------
    
    void initialize_zero();
    void initialize_plus();

    //--------------------------------------------------------------------------
    // Logical Gates (transversal, distributed)
    //--------------------------------------------------------------------------
    
    void apply_logical_x();
    void apply_logical_z();
    void apply_logical_h();

    //--------------------------------------------------------------------------
    // Distributed QEC
    //--------------------------------------------------------------------------

    /**
     * @brief Perform distributed QEC round
     * @return QECRoundResult with timing and correction info
     */
    QECRoundResult qec_round();

    /**
     * @brief Multi-round with time-domain decoding
     * @param rounds Number of syndrome rounds
     */
    std::vector<QECRoundResult> qec_rounds(size_t rounds);

    /**
     * @brief Inject error on local qubits
     * @param p Error probability
     */
    void inject_local_error(double p);

    /**
     * @brief Inject specific error pattern
     */
    void inject_error(const PauliString& error);

    //--------------------------------------------------------------------------
    // State Access
    //--------------------------------------------------------------------------

    int my_rank() const { return config_.rank; }
    int world_size() const { return config_.world_size; }
    const StabilizerCode& code() const { return *code_; }
    const PartitionMap& partition() const { return *partition_; }
    const Stats& stats() const { return stats_; }

    void reset_stats() { stats_ = Stats{}; }

private:
    DistributedQECConfig config_;
    std::unique_ptr<StabilizerCode> code_;
    std::unique_ptr<PartitionMap> partition_;
    std::unique_ptr<DistributedSyndromeExtractor> extractor_;
    std::unique_ptr<ParallelMWPMDecoder> decoder_;
    std::unique_ptr<ErrorInjector> error_injector_;

    PauliString local_error_;  // Error on local qubits
    Stats stats_;

    // Simulate MPI communication (for single-process testing)
    std::vector<LocalSyndrome> simulate_allgather(const LocalSyndrome& local);
    std::vector<LocalCorrection> simulate_gather_corrections(const LocalCorrection& local);
};

//==============================================================================
// Fault-Tolerant QEC Runner (integrates with Phase 8)
//==============================================================================

/**
 * @brief Runs QEC with fault tolerance (checkpoint + recovery)
 */
class FaultTolerantQECRunner {
public:
    struct Config {
        DistributedQECConfig qec_config;
        size_t checkpoint_interval = 100;  // Cycles between checkpoints
        std::string checkpoint_dir = "./qec_checkpoints";
        bool enable_recovery = true;
        bool log_syndrome_history = true;
    };

    struct QECState {
        size_t cycle_number = 0;
        std::vector<Syndrome> syndrome_history;
        std::vector<Correction> correction_history;
        PauliString accumulated_error;
        double logical_fidelity = 1.0;
    };

    explicit FaultTolerantQECRunner(Config config);

    /**
     * @brief Run logical circuit with continuous QEC
     * @param num_cycles Number of QEC cycles
     * @param apply_gates Optional gates to apply between cycles
     * @return Final QEC state
     */
    QECState run(size_t num_cycles,
                 std::function<void(DistributedLogicalQubit&, size_t)> apply_gates = nullptr);

    /**
     * @brief Save QEC state checkpoint
     */
    bool checkpoint(const std::string& path);

    /**
     * @brief Recover from checkpoint
     */
    bool recover(const std::string& path);

    /**
     * @brief Get current state
     */
    const QECState& state() const { return state_; }

private:
    Config config_;
    QECState state_;
    std::unique_ptr<DistributedLogicalQubit> logical_qubit_;

    bool save_state(const std::string& path);
    bool load_state(const std::string& path);
};

//==============================================================================
// Distributed QEC Simulator
//==============================================================================

/**
 * @brief Monte Carlo simulation for distributed QEC
 */
class DistributedQECSimulator {
public:
    struct SimConfig {
        DistributedQECConfig qec_config;
        size_t num_trials = 1000;
        size_t qec_rounds_per_trial = 10;
        bool measure_comm_overhead = true;
        unsigned seed = 42;
    };

    struct SimResult {
        size_t num_trials;
        size_t num_logical_errors;
        double logical_error_rate;
        double avg_local_decode_time_ms;
        double avg_merge_time_ms;
        double avg_comm_time_ms;
        double speedup_vs_serial;  // Compared to single-rank
        std::map<int, size_t> errors_per_rank;  // Distribution
    };

    explicit DistributedQECSimulator(SimConfig config);

    /**
     * @brief Run distributed QEC simulation
     */
    SimResult run();

    /**
     * @brief Compare distributed vs serial performance
     */
    std::pair<SimResult, SimResult> compare_distributed_vs_serial();

private:
    SimConfig config_;
};

}  // namespace qlret

#include "qec_distributed.h"
#include <algorithm>
#include <numeric>
#include <chrono>
#include <cmath>
#include <fstream>
#include <sstream>
#include <random>
#include <stdexcept>

namespace qlret {

//==============================================================================
// LocalSyndrome Implementation
//==============================================================================

size_t LocalSyndrome::num_defects() const {
    size_t count = 0;
    for (int s : x_syndrome) count += (s != 0);
    for (int s : z_syndrome) count += (s != 0);
    return count;
}

bool LocalSyndrome::has_boundary_defects() const {
    for (size_t idx : boundary_stabilizer_indices) {
        // Check if this stabilizer index is in x_syndrome or z_syndrome
        for (size_t i = 0; i < x_syndrome.size(); ++i) {
            if (local_stabilizer_indices.size() > i && 
                local_stabilizer_indices[i] == idx && x_syndrome[i]) {
                return true;
            }
        }
        for (size_t i = 0; i < z_syndrome.size(); ++i) {
            if (local_stabilizer_indices.size() > i && 
                local_stabilizer_indices[i] == idx && z_syndrome[i]) {
                return true;
            }
        }
    }
    return false;
}

//==============================================================================
// GlobalSyndrome Implementation
//==============================================================================

size_t GlobalSyndrome::total_defects() const {
    return full_syndrome.num_defects();
}

//==============================================================================
// PartitionMap Implementation
//==============================================================================

PartitionMap::PartitionMap(const StabilizerCode& code, 
                           const DistributedQECConfig& config)
    : code_(code), config_(config) {
    
    size_t n_qubits = code.num_data_qubits();
    size_t n_x_stabs = code.x_stabilizers().size();
    size_t n_z_stabs = code.z_stabilizers().size();
    
    qubit_to_rank_.resize(n_qubits, 0);
    x_stab_to_rank_.resize(n_x_stabs, 0);
    z_stab_to_rank_.resize(n_z_stabs, 0);
    
    switch (config.partition) {
        case PartitionStrategy::ROW_WISE:
            build_row_partition();
            break;
        case PartitionStrategy::COLUMN_WISE:
            build_column_partition();
            break;
        case PartitionStrategy::BLOCK_2D:
            build_block_partition();
            break;
        case PartitionStrategy::ROUND_ROBIN:
            build_round_robin_partition();
            break;
    }
}

void PartitionMap::build_row_partition() {
    size_t n_qubits = code_.num_data_qubits();
    size_t d = code_.distance();
    int ws = config_.world_size;
    
    // For surface code: row = qubit / d
    for (size_t q = 0; q < n_qubits; ++q) {
        auto [row, col] = code_.qubit_coords(q);
        int rank = (row * ws) / static_cast<int>(d);
        rank = std::min(rank, ws - 1);
        qubit_to_rank_[q] = rank;
    }
    
    // Stabilizers owned by the rank that owns majority of their qubits
    for (size_t i = 0; i < code_.x_stabilizers().size(); ++i) {
        auto support = code_.x_stabilizers()[i].support();
        if (!support.empty()) {
            x_stab_to_rank_[i] = qubit_to_rank_[support[0]];
        }
    }
    for (size_t i = 0; i < code_.z_stabilizers().size(); ++i) {
        auto support = code_.z_stabilizers()[i].support();
        if (!support.empty()) {
            z_stab_to_rank_[i] = qubit_to_rank_[support[0]];
        }
    }
    
    // Build neighbor map
    for (int r = 0; r < ws; ++r) {
        if (r > 0) rank_neighbors_[r].push_back(r - 1);
        if (r < ws - 1) rank_neighbors_[r].push_back(r + 1);
    }
}

void PartitionMap::build_column_partition() {
    size_t n_qubits = code_.num_data_qubits();
    size_t d = code_.distance();
    int ws = config_.world_size;
    
    for (size_t q = 0; q < n_qubits; ++q) {
        auto [row, col] = code_.qubit_coords(q);
        int rank = (col * ws) / static_cast<int>(d);
        rank = std::min(rank, ws - 1);
        qubit_to_rank_[q] = rank;
    }
    
    for (size_t i = 0; i < code_.x_stabilizers().size(); ++i) {
        auto support = code_.x_stabilizers()[i].support();
        if (!support.empty()) {
            x_stab_to_rank_[i] = qubit_to_rank_[support[0]];
        }
    }
    for (size_t i = 0; i < code_.z_stabilizers().size(); ++i) {
        auto support = code_.z_stabilizers()[i].support();
        if (!support.empty()) {
            z_stab_to_rank_[i] = qubit_to_rank_[support[0]];
        }
    }
    
    for (int r = 0; r < ws; ++r) {
        if (r > 0) rank_neighbors_[r].push_back(r - 1);
        if (r < ws - 1) rank_neighbors_[r].push_back(r + 1);
    }
}

void PartitionMap::build_block_partition() {
    // 2D block decomposition
    int ws = config_.world_size;
    int sqrt_ws = static_cast<int>(std::sqrt(ws));
    if (sqrt_ws * sqrt_ws != ws) {
        // Fall back to row partition if not perfect square
        build_row_partition();
        return;
    }
    
    size_t n_qubits = code_.num_data_qubits();
    size_t d = code_.distance();
    
    for (size_t q = 0; q < n_qubits; ++q) {
        auto [row, col] = code_.qubit_coords(q);
        int block_row = (row * sqrt_ws) / static_cast<int>(d);
        int block_col = (col * sqrt_ws) / static_cast<int>(d);
        block_row = std::min(block_row, sqrt_ws - 1);
        block_col = std::min(block_col, sqrt_ws - 1);
        qubit_to_rank_[q] = block_row * sqrt_ws + block_col;
    }
    
    for (size_t i = 0; i < code_.x_stabilizers().size(); ++i) {
        auto support = code_.x_stabilizers()[i].support();
        if (!support.empty()) {
            x_stab_to_rank_[i] = qubit_to_rank_[support[0]];
        }
    }
    for (size_t i = 0; i < code_.z_stabilizers().size(); ++i) {
        auto support = code_.z_stabilizers()[i].support();
        if (!support.empty()) {
            z_stab_to_rank_[i] = qubit_to_rank_[support[0]];
        }
    }
    
    // 2D neighbor connectivity
    for (int r = 0; r < ws; ++r) {
        int block_row = r / sqrt_ws;
        int block_col = r % sqrt_ws;
        if (block_row > 0) rank_neighbors_[r].push_back(r - sqrt_ws);
        if (block_row < sqrt_ws - 1) rank_neighbors_[r].push_back(r + sqrt_ws);
        if (block_col > 0) rank_neighbors_[r].push_back(r - 1);
        if (block_col < sqrt_ws - 1) rank_neighbors_[r].push_back(r + 1);
    }
}

void PartitionMap::build_round_robin_partition() {
    int ws = config_.world_size;
    
    for (size_t q = 0; q < qubit_to_rank_.size(); ++q) {
        qubit_to_rank_[q] = static_cast<int>(q % ws);
    }
    
    for (size_t i = 0; i < x_stab_to_rank_.size(); ++i) {
        x_stab_to_rank_[i] = static_cast<int>(i % ws);
    }
    for (size_t i = 0; i < z_stab_to_rank_.size(); ++i) {
        z_stab_to_rank_[i] = static_cast<int>(i % ws);
    }
    
    // All ranks are neighbors in round-robin
    for (int r = 0; r < ws; ++r) {
        for (int other = 0; other < ws; ++other) {
            if (other != r) rank_neighbors_[r].push_back(other);
        }
    }
}

int PartitionMap::qubit_owner(size_t qubit) const {
    return (qubit < qubit_to_rank_.size()) ? qubit_to_rank_[qubit] : 0;
}

std::vector<size_t> PartitionMap::local_qubits(int rank) const {
    std::vector<size_t> qubits;
    for (size_t i = 0; i < qubit_to_rank_.size(); ++i) {
        if (qubit_to_rank_[i] == rank) {
            qubits.push_back(i);
        }
    }
    return qubits;
}

std::vector<size_t> PartitionMap::boundary_qubits(int rank) const {
    std::vector<size_t> boundary;
    auto local = local_qubits(rank);
    auto neighbors = neighbor_ranks(rank);
    
    for (size_t q : local) {
        // Check if any stabilizer containing q is owned by a neighbor
        for (size_t i = 0; i < code_.x_stabilizers().size(); ++i) {
            if (x_stab_to_rank_[i] != rank) {
                auto support = code_.x_stabilizers()[i].support();
                if (std::find(support.begin(), support.end(), q) != support.end()) {
                    boundary.push_back(q);
                    break;
                }
            }
        }
    }
    return boundary;
}

int PartitionMap::stabilizer_owner(size_t stab_idx, bool is_x) const {
    if (is_x) {
        return (stab_idx < x_stab_to_rank_.size()) ? x_stab_to_rank_[stab_idx] : 0;
    } else {
        return (stab_idx < z_stab_to_rank_.size()) ? z_stab_to_rank_[stab_idx] : 0;
    }
}

std::vector<size_t> PartitionMap::local_x_stabilizers(int rank) const {
    std::vector<size_t> stabs;
    for (size_t i = 0; i < x_stab_to_rank_.size(); ++i) {
        if (x_stab_to_rank_[i] == rank) {
            stabs.push_back(i);
        }
    }
    return stabs;
}

std::vector<size_t> PartitionMap::local_z_stabilizers(int rank) const {
    std::vector<size_t> stabs;
    for (size_t i = 0; i < z_stab_to_rank_.size(); ++i) {
        if (z_stab_to_rank_[i] == rank) {
            stabs.push_back(i);
        }
    }
    return stabs;
}

std::vector<size_t> PartitionMap::boundary_stabilizers(int rank) const {
    std::vector<size_t> boundary;
    auto local_x = local_x_stabilizers(rank);
    auto local_z = local_z_stabilizers(rank);
    
    // Stabilizers that touch qubits owned by other ranks
    for (size_t i : local_x) {
        auto support = code_.x_stabilizers()[i].support();
        for (size_t q : support) {
            if (qubit_to_rank_[q] != rank) {
                boundary.push_back(i);
                break;
            }
        }
    }
    for (size_t i : local_z) {
        auto support = code_.z_stabilizers()[i].support();
        for (size_t q : support) {
            if (qubit_to_rank_[q] != rank) {
                boundary.push_back(i);
                break;
            }
        }
    }
    return boundary;
}

std::vector<int> PartitionMap::neighbor_ranks(int rank) const {
    auto it = rank_neighbors_.find(rank);
    if (it != rank_neighbors_.end()) {
        return it->second;
    }
    return {};
}

//==============================================================================
// DistributedSyndromeExtractor Implementation
//==============================================================================

DistributedSyndromeExtractor::DistributedSyndromeExtractor(
    const StabilizerCode& code,
    const PartitionMap& partition,
    const DistributedQECConfig& config)
    : code_(code), partition_(partition), config_(config),
      local_extractor_(code) {}

LocalSyndrome DistributedSyndromeExtractor::extract_local(const PauliString& local_error) {
    auto start = std::chrono::high_resolution_clock::now();
    
    LocalSyndrome local;
    local.rank = config_.rank;
    
    // Get local stabilizers
    auto local_x_stabs = partition_.local_x_stabilizers(config_.rank);
    auto local_z_stabs = partition_.local_z_stabilizers(config_.rank);
    auto boundary_stabs = partition_.boundary_stabilizers(config_.rank);
    
    local.local_stabilizer_indices.insert(
        local.local_stabilizer_indices.end(),
        local_x_stabs.begin(), local_x_stabs.end());
    local.local_stabilizer_indices.insert(
        local.local_stabilizer_indices.end(),
        local_z_stabs.begin(), local_z_stabs.end());
    local.boundary_stabilizer_indices = boundary_stabs;
    
    // Extract syndrome for local stabilizers only
    const auto& x_stabs = code_.x_stabilizers();
    const auto& z_stabs = code_.z_stabilizers();
    
    for (size_t i : local_x_stabs) {
        const auto& stab = x_stabs[i];
        int parity = 0;
        for (size_t q : stab.support()) {
            if (q < local_error.size()) {
                Pauli e = local_error[q];
                if (e == Pauli::Z || e == Pauli::Y) {
                    parity ^= 1;
                }
            }
        }
        local.x_syndrome.push_back(parity);
    }
    
    for (size_t i : local_z_stabs) {
        const auto& stab = z_stabs[i];
        int parity = 0;
        for (size_t q : stab.support()) {
            if (q < local_error.size()) {
                Pauli e = local_error[q];
                if (e == Pauli::X || e == Pauli::Y) {
                    parity ^= 1;
                }
            }
        }
        local.z_syndrome.push_back(parity);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    local.extraction_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    return local;
}

GlobalSyndrome DistributedSyndromeExtractor::gather_syndromes(
    const LocalSyndrome& local,
    const std::vector<LocalSyndrome>& all_local) {
    
    auto start = std::chrono::high_resolution_clock::now();
    
    GlobalSyndrome global;
    global.local_syndromes = all_local;
    global.local_syndromes[config_.rank] = local;
    
    // Reconstruct full syndrome
    global.full_syndrome.x_syndrome.resize(code_.x_stabilizers().size(), 0);
    global.full_syndrome.z_syndrome.resize(code_.z_stabilizers().size(), 0);
    
    for (const auto& ls : global.local_syndromes) {
        auto local_x_stabs = partition_.local_x_stabilizers(ls.rank);
        auto local_z_stabs = partition_.local_z_stabilizers(ls.rank);
        
        for (size_t i = 0; i < ls.x_syndrome.size() && i < local_x_stabs.size(); ++i) {
            global.full_syndrome.x_syndrome[local_x_stabs[i]] = ls.x_syndrome[i];
        }
        for (size_t i = 0; i < ls.z_syndrome.size() && i < local_z_stabs.size(); ++i) {
            global.full_syndrome.z_syndrome[local_z_stabs[i]] = ls.z_syndrome[i];
        }
    }
    
    // Identify boundary defects
    for (const auto& ls : global.local_syndromes) {
        for (size_t idx : ls.boundary_stabilizer_indices) {
            if (idx < global.full_syndrome.x_syndrome.size() &&
                global.full_syndrome.x_syndrome[idx]) {
                global.boundary_defects.push_back(idx);
            }
            if (idx < global.full_syndrome.z_syndrome.size() &&
                global.full_syndrome.z_syndrome[idx]) {
                global.boundary_defects.push_back(idx);
            }
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    global.aggregation_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    return global;
}

GlobalSyndrome DistributedSyndromeExtractor::allgather_syndromes_mpi(
    const LocalSyndrome& local) {
    // MPI stub - in real implementation would use MPI_Allgather
    // For now, return single-rank result
    GlobalSyndrome global;
    global.local_syndromes.resize(config_.world_size);
    global.local_syndromes[config_.rank] = local;
    
    // Reconstruct from single rank
    global.full_syndrome.x_syndrome.resize(code_.x_stabilizers().size(), 0);
    global.full_syndrome.z_syndrome.resize(code_.z_stabilizers().size(), 0);
    
    auto local_x_stabs = partition_.local_x_stabilizers(config_.rank);
    auto local_z_stabs = partition_.local_z_stabilizers(config_.rank);
    
    for (size_t i = 0; i < local.x_syndrome.size() && i < local_x_stabs.size(); ++i) {
        global.full_syndrome.x_syndrome[local_x_stabs[i]] = local.x_syndrome[i];
    }
    for (size_t i = 0; i < local.z_syndrome.size() && i < local_z_stabs.size(); ++i) {
        global.full_syndrome.z_syndrome[local_z_stabs[i]] = local.z_syndrome[i];
    }
    
    return global;
}

void DistributedSyndromeExtractor::resolve_boundary_stabilizers(
    LocalSyndrome& local,
    const std::vector<LocalSyndrome>& neighbors) {
    // Placeholder for boundary resolution
    // In real MPI implementation, would exchange boundary qubit states
    (void)neighbors;
}

//==============================================================================
// ParallelMWPMDecoder Implementation
//==============================================================================

ParallelMWPMDecoder::ParallelMWPMDecoder(
    const StabilizerCode& code,
    const PartitionMap& partition,
    const DistributedQECConfig& qec_config)
    : ParallelMWPMDecoder(code, partition, qec_config, Config{}) {}

ParallelMWPMDecoder::ParallelMWPMDecoder(
    const StabilizerCode& code,
    const PartitionMap& partition,
    const DistributedQECConfig& qec_config,
    Config decoder_config)
    : code_(code), partition_(partition), 
      qec_config_(qec_config), decoder_config_(decoder_config) {
    
    MWPMDecoder::Config mwpm_config;
    mwpm_config.physical_error_rate = qec_config.physical_error_rate;
    mwpm_config.measurement_error_rate = qec_config.measurement_error_rate;
    
    local_decoder_ = std::make_unique<MWPMDecoder>(code, mwpm_config);
}

LocalCorrection ParallelMWPMDecoder::decode_local(const LocalSyndrome& local_syn) {
    auto start = std::chrono::high_resolution_clock::now();
    
    LocalCorrection local_corr;
    local_corr.rank = local_syn.rank;
    local_corr.x_correction = PauliString(code_.num_data_qubits());
    local_corr.z_correction = PauliString(code_.num_data_qubits());
    
    // Build local syndrome
    Syndrome local_full;
    local_full.x_syndrome = local_syn.x_syndrome;
    local_full.z_syndrome = local_syn.z_syndrome;
    
    // Only decode if there are defects
    if (local_syn.num_defects() > 0) {
        Correction corr = local_decoder_->decode(local_full);
        
        // Only keep corrections for local qubits
        auto local_qubits = partition_.local_qubits(local_syn.rank);
        for (size_t q : local_qubits) {
            if (q < corr.x_correction.size()) {
                local_corr.x_correction.set(q, corr.x_correction[q]);
            }
            if (q < corr.z_correction.size()) {
                local_corr.z_correction.set(q, corr.z_correction[q]);
            }
        }
        
        local_corr.corrected_qubits = local_qubits;
        local_corr.requires_boundary_merge = local_syn.has_boundary_defects();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    local_corr.decode_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    stats_.num_decodes++;
    stats_.total_time_ms += local_corr.decode_time_ms;
    
    return local_corr;
}

GlobalCorrection ParallelMWPMDecoder::merge_corrections(
    const std::vector<LocalCorrection>& all_local) {
    
    auto start = std::chrono::high_resolution_clock::now();
    
    GlobalCorrection global;
    global.local_corrections = all_local;
    global.full_correction.x_correction = PauliString(code_.num_data_qubits());
    global.full_correction.z_correction = PauliString(code_.num_data_qubits());
    
    // Merge all local corrections
    for (const auto& local : all_local) {
        for (size_t q : local.corrected_qubits) {
            if (q < local.x_correction.size()) {
                Pauli p = local.x_correction[q];
                if (p != Pauli::I) {
                    global.full_correction.x_correction.set(q, p);
                }
            }
            if (q < local.z_correction.size()) {
                Pauli p = local.z_correction[q];
                if (p != Pauli::I) {
                    global.full_correction.z_correction.set(q, p);
                }
            }
        }
    }
    
    // Record boundary merges
    for (size_t i = 0; i < all_local.size(); ++i) {
        if (all_local[i].requires_boundary_merge) {
            auto neighbors = partition_.neighbor_ranks(all_local[i].rank);
            for (int n : neighbors) {
                if (n > all_local[i].rank) {  // Avoid duplicates
                    global.boundary_merges.push_back({all_local[i].rank, n});
                }
            }
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    global.merge_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    return global;
}

GlobalCorrection ParallelMWPMDecoder::decode_parallel(const GlobalSyndrome& global_syn) {
    // Decode locally on each rank (simulated)
    std::vector<LocalCorrection> all_local;
    
    for (const auto& local_syn : global_syn.local_syndromes) {
        all_local.push_back(decode_local(local_syn));
    }
    
    return merge_corrections(all_local);
}

GlobalCorrection ParallelMWPMDecoder::decode_global(const GlobalSyndrome& global_syn) {
    auto start = std::chrono::high_resolution_clock::now();
    
    GlobalCorrection global;
    
    // Decode full syndrome on rank 0
    Correction full_corr = local_decoder_->decode(global_syn.full_syndrome);
    global.full_correction = full_corr;
    
    auto end = std::chrono::high_resolution_clock::now();
    global.merge_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    return global;
}

std::vector<std::pair<size_t, size_t>> 
ParallelMWPMDecoder::find_boundary_defect_pairs(const GlobalSyndrome& global_syn) {
    std::vector<std::pair<size_t, size_t>> pairs;
    
    // Simple greedy pairing of boundary defects
    auto defects = global_syn.boundary_defects;
    std::vector<bool> matched(defects.size(), false);
    
    for (size_t i = 0; i < defects.size(); ++i) {
        if (matched[i]) continue;
        
        double best_dist = 1e9;
        size_t best_j = i;
        
        for (size_t j = i + 1; j < defects.size(); ++j) {
            if (matched[j]) continue;
            
            // Simple distance based on stabilizer indices
            double dist = std::abs(static_cast<double>(defects[i]) - 
                                   static_cast<double>(defects[j]));
            if (dist < best_dist) {
                best_dist = dist;
                best_j = j;
            }
        }
        
        if (best_j != i) {
            pairs.push_back({defects[i], defects[best_j]});
            matched[i] = true;
            matched[best_j] = true;
        }
    }
    
    return pairs;
}

void ParallelMWPMDecoder::apply_boundary_correction(
    GlobalCorrection& correction,
    const std::vector<std::pair<size_t, size_t>>& /* pairs */) {
    // Placeholder - would apply corrections for boundary-crossing error chains
}

//==============================================================================
// DistributedLogicalQubit Implementation
//==============================================================================

DistributedLogicalQubit::DistributedLogicalQubit(const DistributedQECConfig& config)
    : config_(config) {
    
    code_ = create_stabilizer_code(config_.code_type, config_.code_distance);
    partition_ = std::make_unique<PartitionMap>(*code_, config_);
    extractor_ = std::make_unique<DistributedSyndromeExtractor>(*code_, *partition_, config_);
    
    ParallelMWPMDecoder::Config decoder_cfg;
    decoder_cfg.use_local_decode = config_.use_local_decode_first;
    decoder_ = std::make_unique<ParallelMWPMDecoder>(*code_, *partition_, config_, decoder_cfg);
    
    auto local_qubits = partition_->local_qubits(config_.rank);
    error_injector_ = std::make_unique<ErrorInjector>(code_->num_data_qubits());
    local_error_ = PauliString(code_->num_data_qubits());
}

void DistributedLogicalQubit::initialize_zero() {
    local_error_ = PauliString(code_->num_data_qubits());
}

void DistributedLogicalQubit::initialize_plus() {
    local_error_ = PauliString(code_->num_data_qubits());
    apply_logical_h();
}

void DistributedLogicalQubit::apply_logical_x() {
    const auto& log_x = code_->logical_x(0);
    auto local_qubits = partition_->local_qubits(config_.rank);
    
    for (size_t q : local_qubits) {
        if (q < log_x.size() && log_x[q] != Pauli::I) {
            local_error_.set(q, pauli_mult(local_error_[q], log_x[q]));
        }
    }
}

void DistributedLogicalQubit::apply_logical_z() {
    const auto& log_z = code_->logical_z(0);
    auto local_qubits = partition_->local_qubits(config_.rank);
    
    for (size_t q : local_qubits) {
        if (q < log_z.size() && log_z[q] != Pauli::I) {
            local_error_.set(q, pauli_mult(local_error_[q], log_z[q]));
        }
    }
}

void DistributedLogicalQubit::apply_logical_h() {
    // H swaps X and Z errors
    auto local_qubits = partition_->local_qubits(config_.rank);
    for (size_t q : local_qubits) {
        Pauli p = local_error_[q];
        if (p == Pauli::X) {
            local_error_.set(q, Pauli::Z);
        } else if (p == Pauli::Z) {
            local_error_.set(q, Pauli::X);
        }
        // Y stays as Y under H (up to phase)
    }
}

QECRoundResult DistributedLogicalQubit::qec_round() {
    QECRoundResult result;
    
    // Extract local syndrome
    auto local_syn = extractor_->extract_local(local_error_);
    stats_.total_extract_time_ms += local_syn.extraction_time_ms;
    
    // Simulate gathering from all ranks
    auto all_local = simulate_allgather(local_syn);
    GlobalSyndrome global_syn = extractor_->gather_syndromes(local_syn, all_local);
    
    result.syndrome = global_syn.full_syndrome;
    result.detected_error = global_syn.total_defects() > 0;
    
    // Decode
    GlobalCorrection global_corr;
    if (config_.parallel_decode && config_.world_size > 1) {
        global_corr = decoder_->decode_parallel(global_syn);
        stats_.local_decode_count++;
        stats_.boundary_merges += global_corr.boundary_merges.size();
    } else {
        global_corr = decoder_->decode_global(global_syn);
        stats_.global_decode_count++;
    }
    
    result.correction = global_corr.full_correction;
    stats_.total_decode_time_ms += global_corr.merge_time_ms;
    
    // Apply correction to local error
    auto local_qubits = partition_->local_qubits(config_.rank);
    for (size_t q : local_qubits) {
        if (q < result.correction.x_correction.size()) {
            Pauli cx = result.correction.x_correction[q];
            if (cx != Pauli::I) {
                local_error_.set(q, pauli_mult(local_error_[q], cx));
            }
        }
        if (q < result.correction.z_correction.size()) {
            Pauli cz = result.correction.z_correction[q];
            if (cz != Pauli::I) {
                local_error_.set(q, pauli_mult(local_error_[q], cz));
            }
        }
    }
    
    // Check for logical error
    const auto& log_x = code_->logical_x(0);
    const auto& log_z = code_->logical_z(0);
    result.logical_error = !local_error_.commutes_with(log_x) || 
                           !local_error_.commutes_with(log_z);
    
    stats_.total_qec_rounds++;
    
    return result;
}

std::vector<QECRoundResult> DistributedLogicalQubit::qec_rounds(size_t rounds) {
    std::vector<QECRoundResult> results;
    results.reserve(rounds);
    
    for (size_t r = 0; r < rounds; ++r) {
        results.push_back(qec_round());
    }
    
    return results;
}

void DistributedLogicalQubit::inject_local_error(double p) {
    auto local_qubits = partition_->local_qubits(config_.rank);
    PauliString error = error_injector_->depolarizing(p);
    
    for (size_t q : local_qubits) {
        if (q < error.size() && error[q] != Pauli::I) {
            local_error_.set(q, pauli_mult(local_error_[q], error[q]));
        }
    }
}

void DistributedLogicalQubit::inject_error(const PauliString& error) {
    auto local_qubits = partition_->local_qubits(config_.rank);
    
    for (size_t q : local_qubits) {
        if (q < error.size() && error[q] != Pauli::I) {
            local_error_.set(q, pauli_mult(local_error_[q], error[q]));
        }
    }
}

std::vector<LocalSyndrome> DistributedLogicalQubit::simulate_allgather(
    const LocalSyndrome& local) {
    // Simulate having all ranks' syndromes (for single-process testing)
    std::vector<LocalSyndrome> all_local(config_.world_size);
    all_local[config_.rank] = local;
    
    // For other ranks, create empty syndromes
    for (int r = 0; r < config_.world_size; ++r) {
        if (r != config_.rank) {
            all_local[r].rank = r;
            auto x_stabs = partition_->local_x_stabilizers(r);
            auto z_stabs = partition_->local_z_stabilizers(r);
            all_local[r].x_syndrome.resize(x_stabs.size(), 0);
            all_local[r].z_syndrome.resize(z_stabs.size(), 0);
            all_local[r].local_stabilizer_indices.insert(
                all_local[r].local_stabilizer_indices.end(),
                x_stabs.begin(), x_stabs.end());
            all_local[r].local_stabilizer_indices.insert(
                all_local[r].local_stabilizer_indices.end(),
                z_stabs.begin(), z_stabs.end());
        }
    }
    
    return all_local;
}

std::vector<LocalCorrection> DistributedLogicalQubit::simulate_gather_corrections(
    const LocalCorrection& local) {
    std::vector<LocalCorrection> all_local(config_.world_size);
    all_local[config_.rank] = local;
    
    for (int r = 0; r < config_.world_size; ++r) {
        if (r != config_.rank) {
            all_local[r].rank = r;
            all_local[r].x_correction = PauliString(code_->num_data_qubits());
            all_local[r].z_correction = PauliString(code_->num_data_qubits());
        }
    }
    
    return all_local;
}

//==============================================================================
// FaultTolerantQECRunner Implementation
//==============================================================================

FaultTolerantQECRunner::FaultTolerantQECRunner(Config config)
    : config_(config) {
    logical_qubit_ = std::make_unique<DistributedLogicalQubit>(config.qec_config);
}

FaultTolerantQECRunner::QECState FaultTolerantQECRunner::run(
    size_t num_cycles,
    std::function<void(DistributedLogicalQubit&, size_t)> apply_gates) {
    
    state_.cycle_number = 0;
    state_.syndrome_history.clear();
    state_.correction_history.clear();
    state_.logical_fidelity = 1.0;
    
    logical_qubit_->initialize_zero();
    
    for (size_t cycle = 0; cycle < num_cycles; ++cycle) {
        state_.cycle_number = cycle;
        
        // Apply gates if specified
        if (apply_gates) {
            apply_gates(*logical_qubit_, cycle);
        }
        
        // Inject error
        logical_qubit_->inject_local_error(config_.qec_config.physical_error_rate);
        
        // QEC round
        auto result = logical_qubit_->qec_round();
        
        if (config_.log_syndrome_history) {
            state_.syndrome_history.push_back(result.syndrome);
            state_.correction_history.push_back(result.correction);
        }
        
        if (result.logical_error) {
            state_.logical_fidelity *= 0.5;  // Simplified fidelity tracking
        }
        
        // Checkpoint
        if (config_.enable_recovery && 
            (cycle + 1) % config_.checkpoint_interval == 0) {
            std::string ckpt_path = config_.checkpoint_dir + "/qec_cycle_" + 
                                    std::to_string(cycle) + ".ckpt";
            checkpoint(ckpt_path);
        }
    }
    
    state_.accumulated_error = PauliString(logical_qubit_->code().num_data_qubits());
    
    return state_;
}

bool FaultTolerantQECRunner::checkpoint(const std::string& path) {
    return save_state(path);
}

bool FaultTolerantQECRunner::recover(const std::string& path) {
    return load_state(path);
}

bool FaultTolerantQECRunner::save_state(const std::string& path) {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) return false;
    
    // Save cycle number
    file.write(reinterpret_cast<const char*>(&state_.cycle_number), sizeof(state_.cycle_number));
    
    // Save syndrome history size
    size_t hist_size = state_.syndrome_history.size();
    file.write(reinterpret_cast<const char*>(&hist_size), sizeof(hist_size));
    
    // Save logical fidelity
    file.write(reinterpret_cast<const char*>(&state_.logical_fidelity), sizeof(state_.logical_fidelity));
    
    return file.good();
}

bool FaultTolerantQECRunner::load_state(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) return false;
    
    file.read(reinterpret_cast<char*>(&state_.cycle_number), sizeof(state_.cycle_number));
    
    size_t hist_size;
    file.read(reinterpret_cast<char*>(&hist_size), sizeof(hist_size));
    
    file.read(reinterpret_cast<char*>(&state_.logical_fidelity), sizeof(state_.logical_fidelity));
    
    return file.good();
}

//==============================================================================
// DistributedQECSimulator Implementation
//==============================================================================

DistributedQECSimulator::DistributedQECSimulator(SimConfig config)
    : config_(config) {}

DistributedQECSimulator::SimResult DistributedQECSimulator::run() {
    SimResult result;
    result.num_trials = config_.num_trials;
    result.num_logical_errors = 0;
    
    double total_local_decode = 0.0;
    double total_merge = 0.0;
    double total_comm = 0.0;
    
    std::mt19937 rng(config_.seed);
    
    for (size_t trial = 0; trial < config_.num_trials; ++trial) {
        DistributedLogicalQubit qubit(config_.qec_config);
        qubit.initialize_zero();
        
        bool had_logical_error = false;
        
        for (size_t round = 0; round < config_.qec_rounds_per_trial; ++round) {
            qubit.inject_local_error(config_.qec_config.physical_error_rate);
            auto qec_result = qubit.qec_round();
            
            if (qec_result.logical_error) {
                had_logical_error = true;
            }
        }
        
        if (had_logical_error) {
            result.num_logical_errors++;
            result.errors_per_rank[config_.qec_config.rank]++;
        }
        
        total_local_decode += qubit.stats().total_decode_time_ms;
        total_comm += qubit.stats().total_comm_time_ms;
    }
    
    result.logical_error_rate = static_cast<double>(result.num_logical_errors) / 
                                 config_.num_trials;
    result.avg_local_decode_time_ms = total_local_decode / 
                                       (config_.num_trials * config_.qec_rounds_per_trial);
    result.avg_merge_time_ms = total_merge / 
                                (config_.num_trials * config_.qec_rounds_per_trial);
    result.avg_comm_time_ms = total_comm / 
                               (config_.num_trials * config_.qec_rounds_per_trial);
    result.speedup_vs_serial = 1.0;  // Placeholder
    
    return result;
}

std::pair<DistributedQECSimulator::SimResult, DistributedQECSimulator::SimResult>
DistributedQECSimulator::compare_distributed_vs_serial() {
    // Run distributed
    auto dist_result = run();
    
    // Run serial (single rank)
    SimConfig serial_config = config_;
    serial_config.qec_config.world_size = 1;
    serial_config.qec_config.rank = 0;
    serial_config.qec_config.parallel_decode = false;
    
    DistributedQECSimulator serial_sim(serial_config);
    auto serial_result = serial_sim.run();
    
    // Compute speedup
    if (serial_result.avg_local_decode_time_ms > 0) {
        dist_result.speedup_vs_serial = serial_result.avg_local_decode_time_ms / 
                                         dist_result.avg_local_decode_time_ms;
    }
    
    return {dist_result, serial_result};
}

}  // namespace qlret

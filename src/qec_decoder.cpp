#include "qec_decoder.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <queue>
#include <stdexcept>

namespace qlret {

//==============================================================================
// QECDecoder Base
//==============================================================================

Correction QECDecoder::decode_multiple_rounds(const std::vector<Syndrome>& syndromes) {
    // Default: just decode the last syndrome
    if (syndromes.empty()) {
        return Correction{};
    }
    return decode(syndromes.back());
}

void QECDecoder::record_decode(double time_ms, bool failed) {
    stats_.num_decodes++;
    stats_.total_time_ms += time_ms;
    if (failed) stats_.num_failures++;
}

//==============================================================================
// MWPMDecoder Implementation
//==============================================================================

MWPMDecoder::MWPMDecoder(const StabilizerCode& code, Config config)
    : code_(code), config_(config) {}

Correction MWPMDecoder::decode(const Syndrome& syndrome) {
    auto start = std::chrono::high_resolution_clock::now();
    
    Correction corr;
    corr.x_correction = PauliString(code_.num_data_qubits());
    corr.z_correction = PauliString(code_.num_data_qubits());

    // Decode X errors (from Z-syndrome)
    {
        std::vector<Syndrome> syn_vec = {syndrome};
        auto graph = build_matching_graph(syn_vec, false);  // Z-syndrome
        
        if (graph.defects.size() > 0) {
            auto matching = greedy_matching(graph.distances);
            auto x_corr = matching_to_correction(matching, graph, true);
            corr.x_correction = x_corr.x_correction;
        }
    }

    // Decode Z errors (from X-syndrome)
    {
        std::vector<Syndrome> syn_vec = {syndrome};
        auto graph = build_matching_graph(syn_vec, true);  // X-syndrome
        
        if (graph.defects.size() > 0) {
            auto matching = greedy_matching(graph.distances);
            auto z_corr = matching_to_correction(matching, graph, false);
            corr.z_correction = z_corr.z_correction;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    record_decode(time_ms);

    return corr;
}

Correction MWPMDecoder::decode_multiple_rounds(const std::vector<Syndrome>& syndromes) {
    if (!config_.use_3d_matching) {
        return decode(syndromes.back());
    }

    auto start = std::chrono::high_resolution_clock::now();
    
    Correction corr;
    corr.x_correction = PauliString(code_.num_data_qubits());
    corr.z_correction = PauliString(code_.num_data_qubits());

    // 3D matching for X errors
    {
        auto graph = build_matching_graph(syndromes, false);
        if (graph.defects.size() > 0) {
            auto matching = greedy_matching(graph.distances);
            auto x_corr = matching_to_correction(matching, graph, true);
            corr.x_correction = x_corr.x_correction;
        }
    }

    // 3D matching for Z errors
    {
        auto graph = build_matching_graph(syndromes, true);
        if (graph.defects.size() > 0) {
            auto matching = greedy_matching(graph.distances);
            auto z_corr = matching_to_correction(matching, graph, false);
            corr.z_correction = z_corr.z_correction;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    record_decode(time_ms);

    return corr;
}

bool MWPMDecoder::has_logical_error(const Correction& correction, 
                                     const PauliString& actual_error) const {
    // Combine correction and actual error
    PauliString combined_x = correction.x_correction * actual_error;
    PauliString combined_z = correction.z_correction * actual_error;

    // Check if result anti-commutes with logical operators
    const auto& log_x = code_.logical_x(0);
    const auto& log_z = code_.logical_z(0);

    // X-type logical error if combined_x anti-commutes with logical_z
    bool x_logical = !combined_x.commutes_with(log_z);
    
    // Z-type logical error if combined_z anti-commutes with logical_x
    bool z_logical = !combined_z.commutes_with(log_x);

    return x_logical || z_logical;
}

void MWPMDecoder::update_weights(double physical_error, double measurement_error) {
    config_.physical_error_rate = physical_error;
    config_.measurement_error_rate = measurement_error;
}

MWPMDecoder::MatchingGraph MWPMDecoder::build_matching_graph(
    const std::vector<Syndrome>& syndromes, bool is_x_syndrome) const {
    
    MatchingGraph graph;

    // Collect defects with space-time coordinates
    Syndrome prev;
    if (is_x_syndrome) {
        prev.x_syndrome.resize(code_.x_stabilizers().size(), 0);
    } else {
        prev.z_syndrome.resize(code_.z_stabilizers().size(), 0);
    }

    for (size_t t = 0; t < syndromes.size(); ++t) {
        const auto& syn = is_x_syndrome ? syndromes[t].x_syndrome : syndromes[t].z_syndrome;
        const auto& prev_syn = is_x_syndrome ? prev.x_syndrome : prev.z_syndrome;

        for (size_t i = 0; i < syn.size(); ++i) {
            int prev_val = (i < prev_syn.size()) ? prev_syn[i] : 0;
            if (syn[i] != prev_val) {
                graph.defects.push_back({i, t});
            }
        }
        if (is_x_syndrome) {
            prev.x_syndrome = syndromes[t].x_syndrome;
        } else {
            prev.z_syndrome = syndromes[t].z_syndrome;
        }
    }

    size_t n = graph.defects.size();
    if (n == 0) return graph;

    // Add virtual boundary vertices (one per defect for potential boundary matching)
    size_t n_with_boundary = 2 * n;  // Each defect can match to boundary

    // Build distance matrix
    graph.distances.resize(n_with_boundary, std::vector<double>(n_with_boundary, 1e9));

    double space_weight = -std::log(std::max(config_.physical_error_rate, 1e-10));
    double time_weight = -std::log(std::max(config_.measurement_error_rate, 1e-10));

    // Real defect pairs
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            int space_dist = std::abs(static_cast<int>(graph.defects[i].first) - 
                                      static_cast<int>(graph.defects[j].first));
            int time_dist = std::abs(static_cast<int>(graph.defects[i].second) - 
                                     static_cast<int>(graph.defects[j].second));
            
            double weight = space_weight * space_dist + time_weight * time_dist;
            graph.distances[i][j] = weight;
            graph.distances[j][i] = weight;
        }
    }

    // Boundary matching (virtual vertices n + i matches with defect i to boundary)
    for (size_t i = 0; i < n; ++i) {
        // Approximate boundary distance
        int space_idx = static_cast<int>(graph.defects[i].first);
        int dist_to_boundary = std::min(space_idx, 
            static_cast<int>(code_.distance()) - 1 - space_idx);
        
        double boundary_weight = space_weight * dist_to_boundary;
        graph.distances[i][n + i] = boundary_weight;
        graph.distances[n + i][i] = boundary_weight;
    }

    // Virtual vertices can match each other at zero cost (both to boundary)
    for (size_t i = n; i < n_with_boundary; ++i) {
        for (size_t j = i + 1; j < n_with_boundary; ++j) {
            graph.distances[i][j] = 0;
            graph.distances[j][i] = 0;
        }
    }

    return graph;
}

std::vector<std::pair<size_t, size_t>> MWPMDecoder::greedy_matching(
    const std::vector<std::vector<double>>& distances) const {
    
    size_t n = distances.size();
    if (n == 0) return {};

    std::vector<bool> matched(n, false);
    std::vector<std::pair<size_t, size_t>> result;

    // Build sorted edge list
    struct Edge {
        size_t u, v;
        double weight;
        bool operator<(const Edge& other) const { return weight < other.weight; }
    };

    std::vector<Edge> edges;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            if (distances[i][j] < 1e8) {
                edges.push_back({i, j, distances[i][j]});
            }
        }
    }
    std::sort(edges.begin(), edges.end());

    // Greedy matching
    for (const auto& e : edges) {
        if (!matched[e.u] && !matched[e.v]) {
            matched[e.u] = true;
            matched[e.v] = true;
            result.push_back({e.u, e.v});
        }
    }

    return result;
}

std::vector<std::pair<size_t, size_t>> MWPMDecoder::blossom_matching(
    const std::vector<std::vector<double>>& distances) const {
    // Placeholder: Blossom V requires external library
    // Fall back to greedy for now
    return greedy_matching(distances);
}

Correction MWPMDecoder::matching_to_correction(
    const std::vector<std::pair<size_t, size_t>>& matching,
    const MatchingGraph& graph,
    bool is_x_correction) const {
    
    Correction corr;
    size_t n_data = code_.num_data_qubits();
    corr.x_correction = PauliString(n_data);
    corr.z_correction = PauliString(n_data);

    PauliString& target = is_x_correction ? corr.x_correction : corr.z_correction;
    Pauli pauli_type = is_x_correction ? Pauli::X : Pauli::Z;

    size_t n_real_defects = graph.defects.size();

    for (const auto& [u, v] : matching) {
        // Skip virtual-virtual matches
        if (u >= n_real_defects && v >= n_real_defects) continue;

        // Get real defect indices
        size_t real_u = (u < n_real_defects) ? u : v;
        size_t real_v = (v < n_real_defects) ? v : u;

        if (real_v >= n_real_defects) {
            // Boundary match: apply correction chain to boundary
            size_t stab_idx = graph.defects[real_u].first;
            // Simple: apply error at first qubit in stabilizer support
            const auto& stabs = is_x_correction ? code_.z_stabilizers() : code_.x_stabilizers();
            if (stab_idx < stabs.size()) {
                auto support = stabs[stab_idx].support();
                if (!support.empty()) {
                    target.set(support[0], pauli_type);
                }
            }
        } else {
            // Match between two defects: find correction path
            size_t stab_u = graph.defects[real_u].first;
            size_t stab_v = graph.defects[real_v].first;
            
            // Apply corrections along path (simplified: just endpoints)
            const auto& stabs = is_x_correction ? code_.z_stabilizers() : code_.x_stabilizers();
            if (stab_u < stabs.size() && stab_v < stabs.size()) {
                auto support_u = stabs[stab_u].support();
                auto support_v = stabs[stab_v].support();
                
                // Find shared qubits or path between stabilizers
                for (size_t q : support_u) {
                    for (size_t q2 : support_v) {
                        if (q == q2) {
                            target.set(q, pauli_type);
                            goto found_path;
                        }
                    }
                }
                // No shared qubit found, use first qubits
                if (!support_u.empty()) target.set(support_u[0], pauli_type);
                found_path:;
            }
        }
    }

    return corr;
}

double MWPMDecoder::compute_distance(int r1, int c1, int r2, int c2) const {
    return std::abs(r1 - r2) + std::abs(c1 - c2);
}

//==============================================================================
// UnionFindDecoder Implementation
//==============================================================================

UnionFindDecoder::UnionFindDecoder(const StabilizerCode& code, Config config)
    : code_(code), config_(config) {}

Correction UnionFindDecoder::decode(const Syndrome& syndrome) {
    auto start = std::chrono::high_resolution_clock::now();

    Correction corr;
    corr.x_correction = grow_clusters(syndrome, false);  // Z-syndrome -> X-correction
    corr.z_correction = grow_clusters(syndrome, true);   // X-syndrome -> Z-correction

    auto end = std::chrono::high_resolution_clock::now();
    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    record_decode(time_ms);

    return corr;
}

bool UnionFindDecoder::has_logical_error(const Correction& correction, 
                                          const PauliString& actual_error) const {
    PauliString combined = correction.x_correction * correction.z_correction * actual_error;
    return !combined.commutes_with(code_.logical_x(0)) || 
           !combined.commutes_with(code_.logical_z(0));
}

size_t UnionFindDecoder::find(size_t x) {
    if (parent_[x] != x) {
        parent_[x] = find(parent_[x]);  // Path compression
    }
    return parent_[x];
}

void UnionFindDecoder::unite(size_t x, size_t y) {
    size_t px = find(x);
    size_t py = find(y);
    
    if (px == py) return;

    // Union by rank
    if (rank_[px] < rank_[py]) {
        parent_[px] = py;
    } else if (rank_[px] > rank_[py]) {
        parent_[py] = px;
    } else {
        parent_[py] = px;
        rank_[px]++;
    }
}

void UnionFindDecoder::reset_uf(size_t n) {
    parent_.resize(n);
    rank_.resize(n);
    std::iota(parent_.begin(), parent_.end(), 0);
    std::fill(rank_.begin(), rank_.end(), 0);
}

Correction UnionFindDecoder::grow_clusters(const Syndrome& syndrome, bool is_x_syndrome) {
    const auto& syn = is_x_syndrome ? syndrome.x_syndrome : syndrome.z_syndrome;
    const auto& stabs = is_x_syndrome ? code_.x_stabilizers() : code_.z_stabilizers();
    
    size_t n_stabs = stabs.size();
    size_t n_data = code_.num_data_qubits();

    // Initialize Union-Find with stabilizer + boundary nodes
    reset_uf(n_stabs + 1);  // +1 for virtual boundary

    // Collect defects
    std::vector<size_t> defects;
    for (size_t i = 0; i < syn.size(); ++i) {
        if (syn[i]) defects.push_back(i);
    }

    if (defects.empty()) {
        return Correction{PauliString(n_data), PauliString(n_data)};
    }

    // Grow clusters by uniting neighboring stabilizers
    // This is a simplified version of the full algorithm
    for (size_t d : defects) {
        // Unite with adjacent stabilizers that are also defects
        for (size_t other : defects) {
            if (d != other) {
                // Check if adjacent (share a data qubit)
                auto sup_d = stabs[d].support();
                auto sup_other = stabs[other].support();
                
                for (size_t q1 : sup_d) {
                    for (size_t q2 : sup_other) {
                        if (q1 == q2) {
                            unite(d, other);
                            goto next_pair;
                        }
                    }
                }
                next_pair:;
            }
        }
    }

    // Build correction from clusters
    PauliString corr_pauli(n_data);
    Pauli pauli_type = is_x_syndrome ? Pauli::Z : Pauli::X;

    // Simple correction: for each cluster, flip one qubit to neutralize
    std::unordered_map<size_t, std::vector<size_t>> clusters;
    for (size_t d : defects) {
        clusters[find(d)].push_back(d);
    }

    for (const auto& [root, members] : clusters) {
        if (members.size() % 2 == 1) {
            // Odd parity cluster: connect to boundary
            auto sup = stabs[members[0]].support();
            if (!sup.empty()) {
                corr_pauli.set(sup[0], pauli_type);
            }
        } else {
            // Even parity: pair up defects internally
            for (size_t i = 0; i + 1 < members.size(); i += 2) {
                auto sup1 = stabs[members[i]].support();
                auto sup2 = stabs[members[i + 1]].support();
                
                // Find connecting qubit
                for (size_t q1 : sup1) {
                    for (size_t q2 : sup2) {
                        if (q1 == q2) {
                            corr_pauli.set(q1, pauli_type);
                            goto next_cluster;
                        }
                    }
                }
                // Fallback
                if (!sup1.empty()) corr_pauli.set(sup1[0], pauli_type);
                next_cluster:;
            }
        }
    }

    Correction result;
    result.x_correction = is_x_syndrome ? PauliString(n_data) : corr_pauli;
    result.z_correction = is_x_syndrome ? corr_pauli : PauliString(n_data);
    return result;
}

//==============================================================================
// LookupTableDecoder Implementation
//==============================================================================

LookupTableDecoder::LookupTableDecoder(const StabilizerCode& code) 
    : code_(code) {
    build_table();
}

Correction LookupTableDecoder::decode(const Syndrome& syndrome) {
    auto start = std::chrono::high_resolution_clock::now();

    Correction corr;
    size_t n_data = code_.num_data_qubits();
    corr.x_correction = PauliString(n_data);
    corr.z_correction = PauliString(n_data);

    size_t hash = syndrome_hash(syndrome);
    auto it = lookup_table_.find(hash);
    
    if (it != lookup_table_.end()) {
        // Split into X and Z corrections
        const PauliString& full = it->second;
        for (size_t i = 0; i < n_data && i < full.size(); ++i) {
            Pauli p = full[i];
            if (p == Pauli::X) {
                corr.x_correction.set(i, Pauli::X);
            } else if (p == Pauli::Z) {
                corr.z_correction.set(i, Pauli::Z);
            } else if (p == Pauli::Y) {
                corr.x_correction.set(i, Pauli::X);
                corr.z_correction.set(i, Pauli::Z);
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    record_decode(time_ms, it == lookup_table_.end());

    return corr;
}

bool LookupTableDecoder::has_logical_error(const Correction& correction, 
                                            const PauliString& actual_error) const {
    PauliString combined = correction.x_correction * correction.z_correction * actual_error;
    return !combined.commutes_with(code_.logical_x(0)) || 
           !combined.commutes_with(code_.logical_z(0));
}

void LookupTableDecoder::build_table() {
    size_t n_data = code_.num_data_qubits();
    size_t max_weight = (code_.distance() - 1) / 2;

    SyndromeExtractor extractor(code_);

    // Enumerate all errors up to weight max_weight
    std::function<void(size_t, size_t, PauliString&)> enumerate;
    enumerate = [&](size_t pos, size_t weight, PauliString& error) {
        if (weight > max_weight) return;
        
        Syndrome syn = extractor.extract(error);
        size_t hash = syndrome_hash(syn);
        
        // Keep lowest weight error for each syndrome
        auto it = lookup_table_.find(hash);
        if (it == lookup_table_.end() || error.weight() < it->second.weight()) {
            lookup_table_[hash] = error;
        }

        if (pos >= n_data) return;

        // Try all Paulis at current position
        for (int p = 1; p <= 3; ++p) {
            error.set(pos, static_cast<Pauli>(p));
            enumerate(pos + 1, weight + 1, error);
            error.set(pos, Pauli::I);
        }
        
        // Skip this position
        enumerate(pos + 1, weight, error);
    };

    PauliString error(n_data);
    enumerate(0, 0, error);
}

size_t LookupTableDecoder::syndrome_hash(const Syndrome& syn) const {
    size_t hash = 0;
    size_t bit = 1;
    
    for (int s : syn.x_syndrome) {
        if (s) hash |= bit;
        bit <<= 1;
    }
    for (int s : syn.z_syndrome) {
        if (s) hash |= bit;
        bit <<= 1;
    }
    
    return hash;
}

//==============================================================================
// Factory
//==============================================================================

std::unique_ptr<QECDecoder> create_decoder(
    DecoderType type,
    const StabilizerCode& code,
    double physical_error_rate) {
    
    switch (type) {
        case DecoderType::MWPM: {
            MWPMDecoder::Config cfg;
            cfg.physical_error_rate = physical_error_rate;
            return std::make_unique<MWPMDecoder>(code, cfg);
        }
        case DecoderType::UNION_FIND:
            return std::make_unique<UnionFindDecoder>(code);
        case DecoderType::LOOKUP_TABLE:
            return std::make_unique<LookupTableDecoder>(code);
        default:
            throw std::invalid_argument("Unknown decoder type");
    }
}

}  // namespace qlret

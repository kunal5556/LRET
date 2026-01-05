#pragma once

#include "qec_types.h"
#include "qec_stabilizer.h"
#include "qec_syndrome.h"
#include <vector>
#include <unordered_map>
#include <memory>
#include <chrono>

namespace qlret {

//==============================================================================
// Decoder Base Class
//==============================================================================

/**
 * @brief Abstract base class for QEC decoders
 */
class QECDecoder {
public:
    struct DecoderStats {
        size_t num_decodes = 0;
        size_t num_failures = 0;  // Detected as uncorrectable
        double total_time_ms = 0.0;
        double avg_time_ms() const { 
            return num_decodes > 0 ? total_time_ms / num_decodes : 0.0; 
        }
        double failure_rate() const {
            return num_decodes > 0 ? static_cast<double>(num_failures) / num_decodes : 0.0;
        }
    };

    virtual ~QECDecoder() = default;

    /**
     * @brief Decode syndrome to find correction
     * @param syndrome Measured syndrome
     * @return Correction to apply (Pauli string)
     */
    virtual Correction decode(const Syndrome& syndrome) = 0;

    /**
     * @brief Decode multiple syndrome rounds (time-domain)
     * @param syndromes Vector of syndromes across time
     * @return Correction to apply
     */
    virtual Correction decode_multiple_rounds(const std::vector<Syndrome>& syndromes);

    /**
     * @brief Check if correction results in logical error
     * @param correction Proposed correction
     * @param actual_error True error (for simulation)
     * @return True if logical error after correction
     */
    virtual bool has_logical_error(const Correction& correction, 
                                   const PauliString& actual_error) const = 0;

    const DecoderStats& stats() const { return stats_; }
    void reset_stats() { stats_ = DecoderStats{}; }

protected:
    DecoderStats stats_;
    void record_decode(double time_ms, bool failed = false);
};

//==============================================================================
// Minimum Weight Perfect Matching Decoder
//==============================================================================

/**
 * @brief MWPM decoder for surface codes
 * 
 * Uses Blossom algorithm for minimum weight perfect matching.
 * Supports both 2D (single round) and 3D (multiple rounds) matching.
 */
class MWPMDecoder : public QECDecoder {
public:
    struct Config {
        double physical_error_rate = 0.001;
        double measurement_error_rate = 0.001;
        bool use_3d_matching = false;  // Time-domain matching
        size_t max_matching_iterations = 1000;
    };

    MWPMDecoder(const StabilizerCode& code, Config config = {});

    Correction decode(const Syndrome& syndrome) override;
    Correction decode_multiple_rounds(const std::vector<Syndrome>& syndromes) override;

    bool has_logical_error(const Correction& correction, 
                           const PauliString& actual_error) const override;

    /**
     * @brief Set edge weights based on error rates
     * @param physical_error Physical error rate
     * @param measurement_error Measurement error rate (for 3D)
     */
    void update_weights(double physical_error, double measurement_error = 0.0);

private:
    const StabilizerCode& code_;
    Config config_;
    
    // Matching structures
    struct MatchingGraph {
        std::vector<std::pair<size_t, size_t>> defects;  // (space_idx, time_idx)
        std::vector<std::vector<double>> distances;       // Distance matrix
        std::vector<std::vector<size_t>> paths;           // Path for each edge
    };

    // Build matching graph from syndrome
    MatchingGraph build_matching_graph(const std::vector<Syndrome>& syndromes,
                                       bool is_x_syndrome) const;

    // Simple greedy matching (fallback when no Blossom)
    std::vector<std::pair<size_t, size_t>> greedy_matching(
        const std::vector<std::vector<double>>& distances) const;

    // Blossom V matching (optional external)
    std::vector<std::pair<size_t, size_t>> blossom_matching(
        const std::vector<std::vector<double>>& distances) const;

    // Convert matching to correction
    Correction matching_to_correction(
        const std::vector<std::pair<size_t, size_t>>& matching,
        const MatchingGraph& graph,
        bool is_x_correction) const;

    // Compute Manhattan distance on surface code lattice
    double compute_distance(int r1, int c1, int r2, int c2) const;
};

//==============================================================================
// Union-Find Decoder (faster but suboptimal)
//==============================================================================

/**
 * @brief Union-Find (Almost-Linear Time) decoder for surface codes
 * 
 * Faster than MWPM but may produce suboptimal corrections.
 * Good for real-time decoding applications.
 */
class UnionFindDecoder : public QECDecoder {
public:
    struct Config {
        bool use_weighted = true;  // Use weighted union-find
        bool use_peeling = true;   // Use peeling decoder optimization
    };

    UnionFindDecoder(const StabilizerCode& code, Config config = {});

    Correction decode(const Syndrome& syndrome) override;

    bool has_logical_error(const Correction& correction, 
                           const PauliString& actual_error) const override;

private:
    const StabilizerCode& code_;
    Config config_;

    // Union-Find data structures
    std::vector<size_t> parent_;
    std::vector<size_t> rank_;

    size_t find(size_t x);
    void unite(size_t x, size_t y);
    void reset_uf(size_t n);

    // Cluster growing algorithm
    Correction grow_clusters(const Syndrome& syndrome, bool is_x_syndrome);
};

//==============================================================================
// Lookup Table Decoder (for small codes)
//==============================================================================

/**
 * @brief Precomputed lookup table decoder for small codes
 * 
 * Fast constant-time decoding using precomputed syndrome -> correction map.
 * Memory-intensive, practical only for distance <= 5.
 */
class LookupTableDecoder : public QECDecoder {
public:
    explicit LookupTableDecoder(const StabilizerCode& code);

    Correction decode(const Syndrome& syndrome) override;

    bool has_logical_error(const Correction& correction, 
                           const PauliString& actual_error) const override;

    /**
     * @brief Get lookup table size
     */
    size_t table_size() const { return lookup_table_.size(); }

private:
    const StabilizerCode& code_;

    // Syndrome hash -> optimal correction
    std::unordered_map<size_t, PauliString> lookup_table_;

    // Build lookup table by enumerating all low-weight errors
    void build_table();

    // Syndrome to hash key
    size_t syndrome_hash(const Syndrome& syn) const;
};

//==============================================================================
// Decoder Factory
//==============================================================================

enum class DecoderType {
    MWPM,
    UNION_FIND,
    LOOKUP_TABLE
};

/**
 * @brief Create decoder by type
 */
std::unique_ptr<QECDecoder> create_decoder(
    DecoderType type,
    const StabilizerCode& code,
    double physical_error_rate = 0.001);

}  // namespace qlret

/**
 * @file tuning_params.cpp
 * @brief Performance Tuning Infrastructure implementation
 *
 * Phase 10, Phase 4B: Performance Tuning Infrastructure
 * From CSV #3, Technique #22: "1.5-3× throughput improvement through empirical tuning"
 *
 * Implements:
 *   - JSON load/save using nlohmann::json
 *   - Heuristic get_optimal() with empirical rules
 *   - Cache-aware batch size and tile size computation
 *   - Noise-adaptive truncation threshold
 *
 * @see include/tuning_params.h
 */

#include "tuning_params.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <thread>
#include <iostream>

namespace qlret {

//==============================================================================
// JSON Persistence
//==============================================================================

TunedParameters TunedParameters::load_from_file(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        throw std::runtime_error(
            "TunedParameters::load_from_file: cannot open '" + path + "'");
    }

    nlohmann::json j;
    try {
        ifs >> j;
    } catch (const nlohmann::json::parse_error& e) {
        throw std::runtime_error(
            "TunedParameters::load_from_file: JSON parse error in '" + path + "': " +
            std::string(e.what()));
    }

    TunedParameters p;

    // Load each field if present; missing fields keep defaults.
    if (j.contains("batch_size"))             p.batch_size             = j["batch_size"].get<size_t>();
    if (j.contains("truncation_threshold"))   p.truncation_threshold   = j["truncation_threshold"].get<double>();
    if (j.contains("openmp_threads"))         p.openmp_threads         = j["openmp_threads"].get<size_t>();
    if (j.contains("row_rank_threshold"))     p.row_rank_threshold     = j["row_rank_threshold"].get<size_t>();
    if (j.contains("column_rank_threshold"))  p.column_rank_threshold  = j["column_rank_threshold"].get<size_t>();
    if (j.contains("morton_qubit_threshold")) p.morton_qubit_threshold = j["morton_qubit_threshold"].get<size_t>();
    if (j.contains("morton_min_qubits"))      p.morton_min_qubits      = j["morton_min_qubits"].get<size_t>();
    if (j.contains("morton_min_batch_gates")) p.morton_min_batch_gates = j["morton_min_batch_gates"].get<size_t>();
    if (j.contains("prefetch_distance"))      p.prefetch_distance      = j["prefetch_distance"].get<size_t>();
    if (j.contains("tile_size"))              p.tile_size              = j["tile_size"].get<size_t>();
    if (j.contains("l1_cache_bytes"))         p.l1_cache_bytes         = j["l1_cache_bytes"].get<size_t>();
    if (j.contains("l2_cache_bytes"))         p.l2_cache_bytes         = j["l2_cache_bytes"].get<size_t>();
    if (j.contains("cache_line_bytes"))       p.cache_line_bytes       = j["cache_line_bytes"].get<size_t>();
    if (j.contains("noise_threshold_scale"))  p.noise_threshold_scale  = j["noise_threshold_scale"].get<double>();
    if (j.contains("max_rank_limit"))         p.max_rank_limit         = j["max_rank_limit"].get<size_t>();
    if (j.contains("source"))                 p.source                 = j["source"].get<std::string>();
    if (j.contains("version"))                p.version                = j["version"].get<std::string>();

    return p;
}

void TunedParameters::save_to_file(const std::string& path) const {
    nlohmann::json j;

    j["batch_size"]             = batch_size;
    j["truncation_threshold"]   = truncation_threshold;
    j["openmp_threads"]         = openmp_threads;
    j["row_rank_threshold"]     = row_rank_threshold;
    j["column_rank_threshold"]  = column_rank_threshold;
    j["morton_qubit_threshold"] = morton_qubit_threshold;
    j["morton_min_qubits"]      = morton_min_qubits;
    j["morton_min_batch_gates"] = morton_min_batch_gates;
    j["prefetch_distance"]      = prefetch_distance;
    j["tile_size"]              = tile_size;
    j["l1_cache_bytes"]         = l1_cache_bytes;
    j["l2_cache_bytes"]         = l2_cache_bytes;
    j["cache_line_bytes"]       = cache_line_bytes;
    j["noise_threshold_scale"]  = noise_threshold_scale;
    j["max_rank_limit"]         = max_rank_limit;
    j["source"]                 = source;
    j["version"]                = version;

    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        throw std::runtime_error(
            "TunedParameters::save_to_file: cannot create '" + path + "'");
    }

    ofs << j.dump(2) << std::endl;
}

//==============================================================================
// Heuristic Optimal Parameter Selection
//==============================================================================

/**
 * Select parameters based on circuit characteristics.
 *
 * Empirical rules derived from profiling on 8-core x86-64, 32 GB RAM:
 *
 * Batch size:
 *   - Small circuits (n < 10): batch_size = min(depth, 128)
 *   - Medium circuits (10 <= n < 16): batch_size = min(depth/2, 64)
 *   - Large circuits (n >= 16): batch_size = min(depth/4, 32)
 *   Rationale: larger circuits have larger L matrices, so smaller batches
 *   keep working set in cache.
 *
 * Truncation threshold:
 *   - Low noise (p < 0.01):  1e-6  (tight, preserve accuracy)
 *   - Medium noise (0.01 <= p < 0.05):  1e-4  (balanced)
 *   - High noise (p >= 0.05):  1e-3  (aggressive truncation to control rank)
 *
 * OpenMP threads:
 *   - n < 8:   1 (overhead exceeds benefit)
 *   - n < 14:  min(hardware_threads, 4)
 *   - n >= 14: min(hardware_threads, 8)
 *
 * Morton parameters:
 *   - n < 14: disabled (MIN_QUBITS_FOR_MORTON)
 *   - n >= 14: morton_qubit_threshold = 8
 *   - n >= 20: morton_qubit_threshold = 6 (more aggressive)
 */
TunedParameters TunedParameters::get_optimal(
    size_t num_qubits,
    size_t circuit_depth,
    double noise_probability
) {
    TunedParameters p;
    p.source = "heuristic";

    // ── Hardware concurrency ──
    unsigned int hw_threads = std::thread::hardware_concurrency();
    if (hw_threads == 0) hw_threads = 4;  // fallback

    // ── Batch size ──
    if (num_qubits < 10) {
        p.batch_size = std::min(circuit_depth, static_cast<size_t>(128));
    } else if (num_qubits < 16) {
        p.batch_size = std::max(static_cast<size_t>(8),
                                std::min(circuit_depth / 2, static_cast<size_t>(64)));
    } else {
        p.batch_size = std::max(static_cast<size_t>(4),
                                std::min(circuit_depth / 4, static_cast<size_t>(32)));
    }
    // Ensure batch_size >= 1
    if (p.batch_size == 0) p.batch_size = 1;

    // ── Truncation threshold ──
    if (noise_probability < 0.01) {
        p.truncation_threshold = 1e-6;
    } else if (noise_probability < 0.05) {
        p.truncation_threshold = 1e-4;
    } else {
        p.truncation_threshold = 1e-3;
    }
    p.noise_threshold_scale = 1.0;

    // ── OpenMP threads ──
    if (num_qubits < 8) {
        p.openmp_threads = 1;
    } else if (num_qubits < 14) {
        p.openmp_threads = std::min(hw_threads, 4u);
    } else {
        p.openmp_threads = std::min(hw_threads, 8u);
    }

    // ── Parallelism thresholds ──
    // Row parallel is good for low rank; column parallel for high rank.
    // These thresholds are crossover points.
    if (num_qubits < 12) {
        p.row_rank_threshold = 64;       // Favour row for smaller systems
        p.column_rank_threshold = 128;
    } else {
        p.row_rank_threshold = 32;
        p.column_rank_threshold = 64;
    }

    // ── Morton parameters ──
    if (num_qubits >= 20) {
        p.morton_qubit_threshold = 6;     // More aggressive for large systems
        p.morton_min_qubits = 14;
        p.morton_min_batch_gates = 1;     // Even a single qualifying gate is worth it
    } else if (num_qubits >= 14) {
        p.morton_qubit_threshold = 8;
        p.morton_min_qubits = 14;
        p.morton_min_batch_gates = 2;
    } else {
        // Morton disabled for small systems
        p.morton_qubit_threshold = 99;    // Effectively disabled
        p.morton_min_qubits = 99;
    }

    // ── Cache tuning ──
    // Prefetch distance: more for large strides, less for small
    p.prefetch_distance = (num_qubits >= 16) ? 8 : 4;

    // Tile size: chosen so one tile fits in L1 cache
    // Each element is complex<double> = 16 bytes
    // Two paired rows in a tile: tile_size × 16 × 2 <= L1_CACHE
    // tile_size <= L1_CACHE / 32
    p.tile_size = std::max(static_cast<size_t>(4),
                           p.l1_cache_bytes / 32);
    // Cap at 256 to avoid excessive register pressure
    p.tile_size = std::min(p.tile_size, static_cast<size_t>(256));

    // ── Max rank limit ──
    // For very large circuits, set a hard cap to prevent OOM
    if (num_qubits >= 18) {
        // At n=18, dim = 262144.  Max rank = dim would be 262144 columns.
        // A practical limit is sqrt(dim) or a fixed cap.
        size_t dim = 1ULL << num_qubits;
        p.max_rank_limit = std::min(static_cast<size_t>(1024),
                                    static_cast<size_t>(std::sqrt(static_cast<double>(dim))));
    }

    return p;
}

//==============================================================================
// Cache-Aware Helpers
//==============================================================================

size_t TunedParameters::batch_size_for_cache(size_t dim, size_t rank) const {
    // Each gate application touches pairs of rows.
    // Working set per batch element: 2 rows × rank columns × 16 bytes/element
    size_t working_set_per_gate = 2 * rank * 16;

    if (working_set_per_gate == 0) return batch_size;  // safety

    // We want total working set <= L2 cache
    size_t max_batch = l2_cache_bytes / working_set_per_gate;

    // Clamp to [1, batch_size]
    max_batch = std::max(max_batch, static_cast<size_t>(1));
    return std::min(max_batch, batch_size);
}

size_t TunedParameters::tile_size_for_cache() const {
    // One tile of two paired rows: tile × 16 bytes × 2
    size_t max_tile = l1_cache_bytes / 32;

    max_tile = std::max(max_tile, static_cast<size_t>(1));
    return std::min(max_tile, tile_size);
}

double TunedParameters::adapted_threshold(double noise_probability) const {
    // For high noise, use a looser threshold to prevent rank explosion.
    // Scale: threshold × (1 + noise_scale × noise_prob)
    //
    // At noise_prob = 0.0:  returns truncation_threshold
    // At noise_prob = 0.1:  returns truncation_threshold × (1 + 1.0 × 0.1) = ×1.1
    // At noise_prob = 0.5:  returns truncation_threshold × (1 + 1.0 × 0.5) = ×1.5
    //
    // This is conservative — the auto_tune.py script may find better scaling.
    return truncation_threshold * (1.0 + noise_threshold_scale * noise_probability);
}

}  // namespace qlret

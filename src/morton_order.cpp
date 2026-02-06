/**
 * @file morton_order.cpp
 * @brief Z-order (Morton) curve implementation for cache-friendly LRET operations
 *
 * Phase 10, Phase 4A: Loop Tiling with Morton Order
 * From CSV #1, Technique #8: "50-80% cache miss reduction, 2-3× speedup for large strides"
 *
 * Implementation details:
 *
 * 1. Bit-interleaving uses the standard "magic numbers" approach for 32-bit
 *    values, which is branch-free and runs in O(1) with ~10 bitwise ops.
 *
 * 2. Permutation tables are precomputed once at construction and stored in
 *    two vectors of size dim.  For n=20 qubits (dim = 1M), this is 8 MB
 *    of permutation data — a small fraction of the L matrix itself.
 *
 * 3. The gate application kernel (apply_single_gate_morton) uses the same
 *    row-pair structure as apply_single_gate_direct() but with Morton-ordered
 *    indices, so paired rows are physically adjacent in memory.
 *
 * 4. OpenMP parallelism is applied when dim > 256 (same threshold as the
 *    standard path).  Static scheduling is used since workload is uniform.
 *
 * @see include/morton_order.h
 */

#include "morton_order.h"
#include "gates_and_noise.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>

#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace qlret {

//==============================================================================
// Bit-Interleaving Helpers (Morton Encoding/Decoding)
//==============================================================================

/**
 * Spread the bits of a 32-bit integer into the even bit positions
 * of a 64-bit integer.
 *
 * Example: 0b1011 → 0b01_00_01_01  (each input bit → even position)
 *
 * Uses the standard 5-step "parallel bits deposit" approach.
 */
uint64_t MortonOrderManager::spread_bits(uint32_t v) {
    uint64_t x = v;
    x = (x | (x << 16)) & 0x0000FFFF0000FFFFULL;
    x = (x | (x <<  8)) & 0x00FF00FF00FF00FFULL;
    x = (x | (x <<  4)) & 0x0F0F0F0F0F0F0F0FULL;
    x = (x | (x <<  2)) & 0x3333333333333333ULL;
    x = (x | (x <<  1)) & 0x5555555555555555ULL;
    return x;
}

/**
 * Compact bits from even positions of a 64-bit integer back into
 * a contiguous 32-bit integer.
 *
 * Inverse of spread_bits.
 */
uint32_t MortonOrderManager::compact_bits(uint64_t v) {
    v = v & 0x5555555555555555ULL;
    v = (v | (v >>  1)) & 0x3333333333333333ULL;
    v = (v | (v >>  2)) & 0x0F0F0F0F0F0F0F0FULL;
    v = (v | (v >>  4)) & 0x00FF00FF00FF00FFULL;
    v = (v | (v >>  8)) & 0x0000FFFF0000FFFFULL;
    v = (v | (v >> 16)) & 0x00000000FFFFFFFFULL;
    return static_cast<uint32_t>(v);
}

/**
 * Encode (x, y) → Morton code z.
 * z = interleave(x into even bits, y into odd bits)
 */
uint64_t MortonOrderManager::encode_morton_2d(uint32_t x, uint32_t y) {
    return spread_bits(x) | (spread_bits(y) << 1);
}

/**
 * Decode Morton code z → (x, y).
 */
std::pair<uint32_t, uint32_t> MortonOrderManager::decode_morton_2d(uint64_t z) {
    uint32_t x = compact_bits(z);
    uint32_t y = compact_bits(z >> 1);
    return {x, y};
}

//==============================================================================
// Permutation Table Construction
//==============================================================================

/**
 * Build forward and inverse permutation tables.
 *
 * Strategy: for an n-qubit system (dim = 2^n), split each row index i
 * into two halves:
 *   - low_half  = bits [0, n/2)     (the "fast" index)
 *   - high_half = bits [n/2, n)     (the "slow" / high-stride index)
 *
 * The Morton code of (low_half, high_half) interleaves these so that
 * rows differing only in high bits (which cause large strides in
 * row-major order) become physically nearby.
 *
 * For odd n, low_half gets ⌈n/2⌉ bits and high_half gets ⌊n/2⌋ bits.
 */
void MortonOrderManager::build_permutation_tables() {
    perm_.resize(dim_);
    inv_perm_.resize(dim_);

    // Determine the number of qubits
    size_t n = 0;
    {
        size_t d = dim_;
        while (d > 1) { d >>= 1; ++n; }
    }

    size_t low_bits  = (n + 1) / 2;  // ⌈n/2⌉
    size_t high_bits = n / 2;        // ⌊n/2⌋
    uint32_t low_mask  = (1u << low_bits) - 1;

    // Build a mapping: row_index → morton_index
    // We need to handle the case where morton indices may exceed dim.
    // Since the Morton encoding maps (low, high) where low < 2^low_bits
    // and high < 2^high_bits, the maximum Morton code is bounded by
    // 2^(2 * max(low_bits, high_bits)).  For balanced splits this
    // equals 2^n = dim, but for unbalanced splits it may exceed dim.
    //
    // To handle this, we use a rank-assignment approach:
    // 1. Compute all Morton codes.
    // 2. Sort them.
    // 3. Assign consecutive positions based on sorted order.
    //
    // However, for balanced or near-balanced splits (|low_bits - high_bits| <= 1),
    // the Morton encoding is a bijection on [0, dim), so we can use it directly.

    if (low_bits == high_bits || low_bits == high_bits + 1) {
        // Balanced split — Morton is a bijection on [0, dim)
        for (size_t i = 0; i < dim_; ++i) {
            uint32_t low_half  = static_cast<uint32_t>(i & low_mask);
            uint32_t high_half = static_cast<uint32_t>(i >> low_bits);
            uint64_t morton_code = encode_morton_2d(low_half, high_half);

            // Clamp to dim_ (safety; should not happen for balanced split)
            size_t morton_idx = static_cast<size_t>(morton_code);
            if (morton_idx >= dim_) {
                // Fallback: identity permutation for this row
                morton_idx = i;
            }

            perm_[i] = morton_idx;
        }

        // Build inverse permutation
        // First check for duplicates (shouldn't happen but safety)
        std::vector<bool> seen(dim_, false);
        bool has_collision = false;
        for (size_t i = 0; i < dim_; ++i) {
            if (seen[perm_[i]]) {
                has_collision = true;
                break;
            }
            seen[perm_[i]] = true;
        }

        if (has_collision) {
            // Collision detected — fall back to identity permutation
            for (size_t i = 0; i < dim_; ++i) {
                perm_[i] = i;
                inv_perm_[i] = i;
            }
        } else {
            for (size_t i = 0; i < dim_; ++i) {
                inv_perm_[perm_[i]] = i;
            }
        }
    } else {
        // Unbalanced split — use sort-based approach
        struct IndexPair {
            uint64_t morton_code;
            size_t original_index;
        };

        std::vector<IndexPair> pairs(dim_);
        for (size_t i = 0; i < dim_; ++i) {
            uint32_t low_half  = static_cast<uint32_t>(i & low_mask);
            uint32_t high_half = static_cast<uint32_t>(i >> low_bits);
            pairs[i] = {encode_morton_2d(low_half, high_half), i};
        }

        // Sort by Morton code to get the Z-order traversal
        std::sort(pairs.begin(), pairs.end(),
                  [](const IndexPair& a, const IndexPair& b) {
                      return a.morton_code < b.morton_code;
                  });

        // Assign consecutive positions
        for (size_t morton_pos = 0; morton_pos < dim_; ++morton_pos) {
            size_t orig = pairs[morton_pos].original_index;
            perm_[orig] = morton_pos;
            inv_perm_[morton_pos] = orig;
        }
    }
}

//==============================================================================
// Constructor
//==============================================================================

MortonOrderManager::MortonOrderManager(size_t dim, size_t rank)
    : dim_(dim), rank_(rank)
{
    if (dim_ == 0) {
        throw std::invalid_argument("MortonOrderManager: dim must be > 0");
    }

    // Verify dim is a power of 2
    if ((dim_ & (dim_ - 1)) != 0) {
        throw std::invalid_argument(
            "MortonOrderManager: dim must be a power of 2, got " + std::to_string(dim_));
    }

    build_permutation_tables();
}

//==============================================================================
// Row Permutation (to/from Morton Layout)
//==============================================================================

MatrixXcd MortonOrderManager::to_morton(const MatrixXcd& L_row_major) const {
    assert(static_cast<size_t>(L_row_major.rows()) == dim_);

    size_t cols = static_cast<size_t>(L_row_major.cols());
    MatrixXcd L_morton(dim_, cols);

    #pragma omp parallel for schedule(static) if(dim_ > 256)
    for (int64_t i = 0; i < static_cast<int64_t>(dim_); ++i) {
        L_morton.row(static_cast<Eigen::Index>(perm_[i])) = L_row_major.row(i);
    }

    return L_morton;
}

MatrixXcd MortonOrderManager::from_morton(const MatrixXcd& L_morton) const {
    assert(static_cast<size_t>(L_morton.rows()) == dim_);

    size_t cols = static_cast<size_t>(L_morton.cols());
    MatrixXcd L_row_major(dim_, cols);

    #pragma omp parallel for schedule(static) if(dim_ > 256)
    for (int64_t i = 0; i < static_cast<int64_t>(dim_); ++i) {
        L_row_major.row(i) = L_morton.row(static_cast<Eigen::Index>(perm_[i]));
    }

    return L_row_major;
}

//==============================================================================
// Gate Application in Morton Layout
//==============================================================================

/**
 * Apply a single-qubit gate to L that is already in Morton layout.
 *
 * The key insight: although the rows are reordered, the *logical*
 * pairing structure is still defined by the target qubit bit.
 * We iterate over logical row indices, use the permutation to find
 * the physical rows in Morton layout, and apply the 2×2 gate in-place.
 *
 * Because paired rows (differing in bit `target`) are nearby in Morton
 * order, the cache performance is significantly better than the
 * standard approach for high target qubits.
 */
MatrixXcd MortonOrderManager::apply_single_gate_morton(
    const MatrixXcd& L_morton,
    const MatrixXcd& gate,
    size_t target,
    size_t num_qubits
) const {
    size_t dim = 1ULL << num_qubits;
    assert(dim == dim_);

    size_t rank = static_cast<size_t>(L_morton.cols());
    MatrixXcd result = L_morton;

    size_t step = 1ULL << target;

    // We iterate over all pairs (i0, i0 + step) where bit `target` of i0 is 0.
    // Then we look up their Morton positions and apply the gate.
    int64_t idim = static_cast<int64_t>(dim);
    int64_t istep = static_cast<int64_t>(step);

    #pragma omp parallel for schedule(static) if(dim > 256 && rank > 2)
    for (int64_t block = 0; block < idim; block += 2 * istep) {
        for (int64_t i = block; i < block + istep && i < idim; ++i) {
            size_t i0 = static_cast<size_t>(i);           // target bit = 0
            size_t i1 = i0 + step;                         // target bit = 1

            if (i1 >= dim) continue;

            // Get Morton-ordered physical row indices
            size_t m0 = perm_[i0];
            size_t m1 = perm_[i1];

            // Apply 2×2 gate to the row pair in Morton layout
            for (size_t r = 0; r < rank; ++r) {
                Complex v0 = L_morton(static_cast<Eigen::Index>(m0), static_cast<Eigen::Index>(r));
                Complex v1 = L_morton(static_cast<Eigen::Index>(m1), static_cast<Eigen::Index>(r));

                result(static_cast<Eigen::Index>(m0), static_cast<Eigen::Index>(r)) =
                    gate(0, 0) * v0 + gate(0, 1) * v1;
                result(static_cast<Eigen::Index>(m1), static_cast<Eigen::Index>(r)) =
                    gate(1, 0) * v0 + gate(1, 1) * v1;
            }
        }
    }

    return result;
}

/**
 * Apply a two-qubit gate to L in Morton layout.
 *
 * Same logic as apply_two_qubit_gate_direct but with permuted row access.
 */
MatrixXcd MortonOrderManager::apply_two_qubit_gate_morton(
    const MatrixXcd& L_morton,
    const MatrixXcd& gate,
    size_t q1,
    size_t q2,
    size_t num_qubits
) const {
    size_t dim = 1ULL << num_qubits;
    assert(dim == dim_);

    size_t rank = static_cast<size_t>(L_morton.cols());
    MatrixXcd result = L_morton;

    size_t step_q1 = 1ULL << q1;
    size_t step_q2 = 1ULL << q2;
    size_t step_min = 1ULL << std::min(q1, q2);
    size_t step_max = 1ULL << std::max(q1, q2);

    for (size_t base = 0; base < dim; ++base) {
        // Only process when both qubit positions are 0
        if ((base & step_min) != 0 || (base & step_max) != 0) continue;

        // Four basis state indices (logical)
        size_t idx[4];
        idx[0] = base;                          // q1=0, q2=0
        idx[1] = base | step_q2;                // q1=0, q2=1
        idx[2] = base | step_q1;                // q1=1, q2=0
        idx[3] = base | step_q1 | step_q2;      // q1=1, q2=1

        // Map to Morton positions
        size_t midx[4];
        for (int k = 0; k < 4; ++k) {
            midx[k] = perm_[idx[k]];
        }

        for (size_t r = 0; r < rank; ++r) {
            Complex v[4];
            for (int k = 0; k < 4; ++k) {
                v[k] = L_morton(static_cast<Eigen::Index>(midx[k]),
                                static_cast<Eigen::Index>(r));
            }

            for (int k = 0; k < 4; ++k) {
                result(static_cast<Eigen::Index>(midx[k]),
                       static_cast<Eigen::Index>(r)) =
                    gate(k, 0) * v[0] + gate(k, 1) * v[1] +
                    gate(k, 2) * v[2] + gate(k, 3) * v[3];
            }
        }
    }

    return result;
}

/**
 * Convenience dispatcher for GateOp in Morton layout.
 */
MatrixXcd MortonOrderManager::apply_gate_morton(
    const MatrixXcd& L_morton,
    const GateOp& gate_op,
    size_t num_qubits
) const {
    if (gate_op.qubits.size() == 1) {
        // Single-qubit gate
        MatrixXcd gate_matrix = get_single_qubit_gate(gate_op.type, gate_op.params);
        if (gate_op.custom_matrix.has_value()) {
            gate_matrix = gate_op.custom_matrix.value();
        }
        return apply_single_gate_morton(L_morton, gate_matrix, gate_op.qubits[0], num_qubits);
    } else if (gate_op.qubits.size() == 2) {
        // Two-qubit gate
        MatrixXcd gate_matrix = get_two_qubit_gate(gate_op.type, gate_op.params);
        if (gate_op.custom_matrix.has_value()) {
            gate_matrix = gate_op.custom_matrix.value();
        }
        return apply_two_qubit_gate_morton(
            L_morton, gate_matrix, gate_op.qubits[0], gate_op.qubits[1], num_qubits);
    }

    // Shouldn't reach here — fall through to identity
    return L_morton;
}

//==============================================================================
// Batched Morton Gate Application
//==============================================================================

/**
 * Convert to Morton, apply all gates, convert back.
 * Amortises the O(dim × rank) permutation cost over many gates.
 */
MatrixXcd MortonOrderManager::apply_gate_batch_morton(
    const MatrixXcd& L,
    const std::vector<GateOp>& gates,
    size_t num_qubits
) const {
    if (gates.empty()) return L;

    // Convert to Morton layout once
    MatrixXcd L_morton = to_morton(L);

    // Apply all gates in Morton layout
    for (const auto& gate_op : gates) {
        L_morton = apply_gate_morton(L_morton, gate_op, num_qubits);
    }

    // Convert back to row-major
    return from_morton(L_morton);
}

//==============================================================================
// Static Heuristics
//==============================================================================

bool MortonOrderManager::should_use_morton(size_t num_qubits, size_t target_qubit) {
    // 1. Circuit must be large enough for permutation cost to be amortised
    if (num_qubits < MIN_QUBITS_FOR_MORTON) return false;

    // 2. Target qubit must create a large stride
    if (target_qubit < MIN_TARGET_FOR_MORTON) return false;

    // 3. Additional heuristic: for very large circuits (n >= 20),
    //    even moderate targets (>= 6) benefit because the stride
    //    is already well beyond L2 cache.
    if (num_qubits >= 20 && target_qubit >= 6) return true;

    return true;
}

bool MortonOrderManager::should_use_morton_batch(
    const std::vector<GateOp>& gates,
    size_t num_qubits,
    size_t min_qualifying
) {
    if (num_qubits < MIN_QUBITS_FOR_MORTON) return false;

    size_t qualifying = 0;
    for (const auto& gate : gates) {
        if (gate.qubits.empty()) continue;

        // For single-qubit gates, check the target
        if (gate.qubits.size() == 1) {
            if (should_use_morton(num_qubits, gate.qubits[0])) {
                ++qualifying;
            }
        } else if (gate.qubits.size() == 2) {
            // For two-qubit gates, check if either qubit is high-indexed
            if (should_use_morton(num_qubits, gate.qubits[0]) ||
                should_use_morton(num_qubits, gate.qubits[1])) {
                ++qualifying;
            }
        }

        if (qualifying >= min_qualifying) return true;
    }

    return false;
}

}  // namespace qlret

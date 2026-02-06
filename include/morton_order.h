#pragma once

/**
 * @file morton_order.h
 * @brief Z-order (Morton) curve for cache-friendly strided access in LRET
 *
 * Phase 10, Phase 4A: Loop Tiling with Morton Order
 * From CSV #1, Technique #8: "50-80% cache miss reduction, 2-3× speedup for large strides"
 *
 * Problem:
 *   In LRET's low-rank density matrix factorisation ρ = L·L†,
 *   single-qubit gate application touches row pairs (i, i + 2^target).
 *   For high-indexed target qubits (target >= 8), stride >= 256 rows,
 *   and the two paired rows will not reside in the same cache line/set.
 *   This causes a cache miss for *every* pair, dominating runtime
 *   at n >= 14 qubits.
 *
 * Solution:
 *   Reorder the rows of L according to a Z-order (Morton) space-filling
 *   curve so that rows that are logically paired by the gate application
 *   (differing only in the target qubit bit) become *physically* adjacent
 *   in memory.  The Morton curve interleaves bit indices of two
 *   coordinates (row_block, pair_index) into a single linear index.
 *
 *   We only reorder for gates on high-indexed qubits (target >= threshold)
 *   where the benefit outweighs the O(dim × rank) permutation cost.
 *
 * API usage:
 *   MortonOrderManager mom(dim, rank);
 *   if (MortonOrderManager::should_use_morton(num_qubits, target)) {
 *       MatrixXcd L_m = mom.to_morton(L);
 *       L_m = mom.apply_gate_morton(L_m, gate, num_qubits);
 *       L = mom.from_morton(L_m);
 *   }
 *
 * Integration point: parallel_modes.cpp row-parallel path, n >= 14.
 *
 * @see ROW_PARALLEL_ADVANCED_OPTIMIZATION_PLAN.md Phase 4A
 */

#include "types.h"
#include <vector>
#include <utility>
#include <cstdint>

namespace qlret {

/**
 * @brief Z-order (Morton) curve manager for cache-friendly row access
 *
 * Maps 2D matrix coordinates to 1D memory using a space-filling curve.
 *
 * Standard row-major traversal for a gate on qubit t visits
 *   (0, 0+2^t), (1, 1+2^t), ...
 * which has stride 2^t.  For t >= 8 this exceeds L1/L2 cache capacity.
 *
 * Morton ordering interleaves the bits of two "coordinates" derived from
 * the row index, grouping rows that differ only in the target qubit bit
 * into contiguous blocks.  The result is that paired rows reside in the
 * same or adjacent cache lines.
 *
 * Morton order encodes (x, y) → z where z's bits are the interleaving of
 * x's and y's bits:  z = ...y2 x2 y1 x1 y0 x0
 */
class MortonOrderManager {
public:
    /**
     * @brief Construct a Morton order manager for a given L matrix shape.
     * @param dim   Number of rows in L (= 2^num_qubits).
     * @param rank  Number of columns in L (current rank).
     */
    MortonOrderManager(size_t dim, size_t rank);

    /**
     * @brief Reorder L matrix rows from row-major to Morton layout.
     *
     * Applies the precomputed permutation table.
     * Complexity: O(dim × rank).
     *
     * @param L_row_major  L in standard row-major layout (dim × rank).
     * @return L with rows permuted to Morton order.
     */
    MatrixXcd to_morton(const MatrixXcd& L_row_major) const;

    /**
     * @brief Reorder L matrix rows from Morton layout back to row-major.
     *
     * Applies the inverse permutation.
     * Complexity: O(dim × rank).
     *
     * @param L_morton  L in Morton order (dim × rank).
     * @return L in standard row-major order.
     */
    MatrixXcd from_morton(const MatrixXcd& L_morton) const;

    /**
     * @brief Apply a single-qubit gate to L that is already in Morton layout.
     *
     * Because the Morton order groups paired rows (differing in the target
     * qubit) into contiguous blocks, the inner loop over pairs has excellent
     * spatial locality.
     *
     * @param L_morton   L in Morton order (dim × rank).
     * @param gate       2×2 unitary gate matrix.
     * @param target     Target qubit index.
     * @param num_qubits Total number of qubits.
     * @return Updated L in Morton order.
     */
    MatrixXcd apply_single_gate_morton(
        const MatrixXcd& L_morton,
        const MatrixXcd& gate,
        size_t target,
        size_t num_qubits
    ) const;

    /**
     * @brief Apply a two-qubit gate to L in Morton layout.
     *
     * Similar locality benefits for two-qubit gates, though the
     * improvement is less dramatic since two-qubit gates already
     * require 4-way grouping.
     *
     * @param L_morton   L in Morton order (dim × rank).
     * @param gate       4×4 unitary gate matrix.
     * @param q1         First qubit (control for CNOT).
     * @param q2         Second qubit (target for CNOT).
     * @param num_qubits Total number of qubits.
     * @return Updated L in Morton order.
     */
    MatrixXcd apply_two_qubit_gate_morton(
        const MatrixXcd& L_morton,
        const MatrixXcd& gate,
        size_t q1,
        size_t q2,
        size_t num_qubits
    ) const;

    /**
     * @brief Apply a GateOp (dispatching single vs two-qubit) in Morton layout.
     *
     * Convenience wrapper that extracts the gate matrix and dispatches.
     *
     * @param L_morton   L in Morton order (dim × rank).
     * @param gate_op    Gate operation to apply.
     * @param num_qubits Total number of qubits.
     * @return Updated L in Morton order.
     */
    MatrixXcd apply_gate_morton(
        const MatrixXcd& L_morton,
        const GateOp& gate_op,
        size_t num_qubits
    ) const;

    /**
     * @brief Apply a batch of gates while L remains in Morton order.
     *
     * Converts to Morton once, applies all gates, converts back once.
     * Amortises the permutation cost over many gates.
     *
     * @param L          L in standard row-major order (dim × rank).
     * @param gates      Vector of gate operations.
     * @param num_qubits Total number of qubits.
     * @return Updated L in standard row-major order.
     */
    MatrixXcd apply_gate_batch_morton(
        const MatrixXcd& L,
        const std::vector<GateOp>& gates,
        size_t num_qubits
    ) const;

    // ────────────────── Static heuristics ──────────────────

    /**
     * @brief Determine whether Morton order is beneficial for a given gate.
     *
     * Heuristic:
     *   1. Require n >= 14 (dim >= 16384) — overhead must be amortised.
     *   2. Require target >= 8 (stride >= 256) — small strides are already
     *      cache-friendly.
     *   3. A batch of >= 4 qualifying gates is even more beneficial
     *      (amortisation of permutation cost).
     *
     * @param num_qubits Total number of qubits.
     * @param target_qubit Target qubit index of the gate.
     * @return true if Morton reordering is expected to help.
     */
    static bool should_use_morton(size_t num_qubits, size_t target_qubit);

    /**
     * @brief Check whether a batch of gates warrants Morton reordering.
     *
     * Counts how many gates in the batch target high-indexed qubits.
     * If at least `min_qualifying` do, Morton is worthwhile.
     *
     * @param gates           Gate batch to inspect.
     * @param num_qubits      Total number of qubits.
     * @param min_qualifying  Minimum number of high-stride gates needed.
     * @return true if the batch benefits from Morton ordering.
     */
    static bool should_use_morton_batch(
        const std::vector<GateOp>& gates,
        size_t num_qubits,
        size_t min_qualifying = 2
    );

    // ────────────────── Accessors ──────────────────

    /** @brief Get the Hilbert space dimension. */
    size_t dim() const { return dim_; }

    /** @brief Get the current rank of L. */
    size_t rank() const { return rank_; }

    /** @brief Get the precomputed Morton permutation table. */
    const std::vector<size_t>& permutation() const { return perm_; }

    /** @brief Get the precomputed inverse permutation table. */
    const std::vector<size_t>& inverse_permutation() const { return inv_perm_; }

    // ────────────────── Tuning constants ──────────────────

    /** Minimum number of qubits for Morton order to be beneficial. */
    static constexpr size_t MIN_QUBITS_FOR_MORTON = 14;

    /** Minimum target qubit index (stride = 2^target >= 256). */
    static constexpr size_t MIN_TARGET_FOR_MORTON = 8;

    /** Cache line size in bytes (typical x86-64). */
    static constexpr size_t CACHE_LINE_BYTES = 64;

    /** L1 data cache size estimate (bytes). */
    static constexpr size_t L1_CACHE_BYTES = 32 * 1024;

    /** L2 cache size estimate (bytes). */
    static constexpr size_t L2_CACHE_BYTES = 256 * 1024;

private:
    size_t dim_;                     ///< Hilbert space dimension (2^n)
    size_t rank_;                    ///< Rank of L matrix
    std::vector<size_t> perm_;       ///< row_major_index → morton_index
    std::vector<size_t> inv_perm_;   ///< morton_index → row_major_index

    // ────────────────── Bit-interleaving helpers ──────────────────

    /**
     * @brief Encode a 2D coordinate (x, y) into a 1D Morton code.
     *
     * Interleaves bits: z = ...y2 x2 y1 x1 y0 x0
     * Result z is at most 2× the bit-width of max(x, y).
     *
     * @param x First coordinate.
     * @param y Second coordinate.
     * @return Morton code z.
     */
    static uint64_t encode_morton_2d(uint32_t x, uint32_t y);

    /**
     * @brief Decode a 1D Morton code back to 2D coordinates.
     *
     * Inverse of encode_morton_2d.
     *
     * @param z Morton code.
     * @return (x, y) pair.
     */
    static std::pair<uint32_t, uint32_t> decode_morton_2d(uint64_t z);

    /**
     * @brief Spread bits of a 32-bit value into even bit positions.
     *
     * Example: 0b1011 → 0b01_00_01_01
     * Uses magic-number bit-interleave approach.
     *
     * @param v Input value.
     * @return Value with bits spread to even positions.
     */
    static uint64_t spread_bits(uint32_t v);

    /**
     * @brief Compact bits from even positions of a 64-bit value.
     *
     * Inverse of spread_bits.
     *
     * @param v Input value with data in even bit positions.
     * @return Compacted 32-bit value.
     */
    static uint32_t compact_bits(uint64_t v);

    /**
     * @brief Build the Morton permutation tables for the current dimension.
     *
     * For each row index i ∈ [0, dim), computes:
     *   - Split i into two halves of n/2 bits each.
     *   - Morton-encode the two halves to get morton_idx.
     *   - perm_[i] = morton_idx; inv_perm_[morton_idx] = i.
     *
     * This permutation groups rows that differ in high bits (large stride)
     * into nearby memory locations.
     */
    void build_permutation_tables();
};

}  // namespace qlret

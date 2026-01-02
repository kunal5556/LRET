#pragma once

/**
 * @file simd_kernels.h
 * @brief SIMD-optimized kernels and CPU feature detection for LRET
 */

#include "types.h"

namespace qlret {

struct SimdCapabilities {
    bool avx2 = false;
    bool avx512 = false;
};

/**
 * @brief Detect SIMD capabilities at runtime (AVX2, AVX-512)
 */
SimdCapabilities detect_simd_capabilities();

/**
 * @brief Apply a single-qubit gate using SIMD-friendly loops with fallback
 *        Automatically dispatches to vectorized path when CPU supports AVX2/AVX-512.
 *        Falls back to scalar implementation otherwise.
 */
MatrixXcd apply_single_qubit_simd(
    const MatrixXcd& L,
    const MatrixXcd& U,
    size_t target,
    size_t num_qubits
);

}  // namespace qlret

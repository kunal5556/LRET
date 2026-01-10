/**
 * @file simd_kernels.cpp
 * @brief SIMD-friendly kernels and CPU feature detection for LRET
 */

#include "simd_kernels.h"

#include <array>
#include <cstdint>
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
#include <cpuid.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace qlret {
namespace {

//------------------------------------------------------------------------------
// CPUID Helpers
//------------------------------------------------------------------------------

inline void cpuid(int info[4], int function_id, int subfunction_id = 0) {
#if defined(_MSC_VER)
    __cpuidex(info, function_id, subfunction_id);
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
    __cpuid_count(function_id, subfunction_id, info[0], info[1], info[2], info[3]);
#else
    // ARM or other architectures - no CPUID
    info[0] = info[1] = info[2] = info[3] = 0;
#endif
}

// Check OS support for XSAVE/XRESTOR (required for AVX)
inline bool os_supports_xsave() {
    int info[4] = {0};
    cpuid(info, 1);
    return (info[2] & (1 << 27)) != 0;  // OSXSAVE
}

// Check if XCR0 enables YMM/ZMM state
inline bool xcr0_supports_avx() {
#if defined(_MSC_VER)
    unsigned long long xcr0 = _xgetbv(0);
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
    uint32_t eax = 0, edx = 0;
    __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    unsigned long long xcr0 = (static_cast<unsigned long long>(edx) << 32) | eax;
#else
    // ARM or other architectures - no AVX
    return false;
#endif
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86) || defined(_MSC_VER)
    // Bit 1 (XMM) and bit 2 (YMM) must be set for AVX
    return (xcr0 & 0x6) == 0x6;
#else
    return false;
#endif
}

inline bool xcr0_supports_zmm() {
#if defined(_MSC_VER)
    unsigned long long xcr0 = _xgetbv(0);
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
    uint32_t eax = 0, edx = 0;
    __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    unsigned long long xcr0 = (static_cast<unsigned long long>(edx) << 32) | eax;
#else
    // ARM or other architectures - no AVX-512
    return false;
#endif
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86) || defined(_MSC_VER)
    // Bits 1 (XMM), 2 (YMM), 5 (opmask), 6 (ZMM_hi256), 7 (ZMM_16_31)
    return (xcr0 & 0xE6) == 0xE6;
#else
    return false;
#endif
}

//------------------------------------------------------------------------------
// Scalar core (with optional #pragma omp simd)
//------------------------------------------------------------------------------

MatrixXcd apply_single_qubit_core(
    const MatrixXcd& L,
    const MatrixXcd& U,
    size_t target,
    size_t num_qubits,
    bool vectorize
) {
    size_t dim = L.rows();
    size_t rank = L.cols();
    MatrixXcd result = L;

    size_t step = 1ULL << target;

#ifdef _OPENMP
    int64_t idim = static_cast<int64_t>(dim);
    int64_t istep = static_cast<int64_t>(step);
    #pragma omp parallel for schedule(static) if(dim > 4096 && rank > 2)
#endif
    for (int64_t block = 0; block < idim; block += 2 * istep) {
        for (size_t i = block; i < block + step && i < dim; ++i) {
            size_t i0 = i;
            size_t i1 = i + step;
            if (i1 >= dim) continue;

            const Complex u00 = U(0, 0);
            const Complex u01 = U(0, 1);
            const Complex u10 = U(1, 0);
            const Complex u11 = U(1, 1);

            if (vectorize) {
#ifdef _OPENMP
                #pragma omp simd
#endif
                for (size_t r = 0; r < rank; ++r) {
                    Complex v0 = L(i0, r);
                    Complex v1 = L(i1, r);
                    result(i0, r) = u00 * v0 + u01 * v1;
                    result(i1, r) = u10 * v0 + u11 * v1;
                }
            } else {
                for (size_t r = 0; r < rank; ++r) {
                    Complex v0 = L(i0, r);
                    Complex v1 = L(i1, r);
                    result(i0, r) = u00 * v0 + u01 * v1;
                    result(i1, r) = u10 * v0 + u11 * v1;
                }
            }
        }
    }

    return result;
}

}  // namespace

//------------------------------------------------------------------------------
// Capability Detection
//------------------------------------------------------------------------------

SimdCapabilities detect_simd_capabilities() {
    SimdCapabilities caps{};

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
    if (!os_supports_xsave() || !xcr0_supports_avx()) {
        return caps;  // AVX not usable
    }

    int info[4] = {0};
    cpuid(info, 0);
    int max_leaf = info[0];
    if (max_leaf < 7) {
        return caps;
    }

    // Leaf 7, subleaf 0
    cpuid(info, 7, 0);
    caps.avx2 = (info[1] & (1 << 5)) != 0;       // EBX bit 5
    bool avx512f = (info[1] & (1 << 16)) != 0;   // EBX bit 16
    bool avx512dq = (info[1] & (1 << 17)) != 0;  // EBX bit 17
    bool avx512cd = (info[1] & (1 << 28)) != 0;  // EBX bit 28

    if (avx512f && avx512dq && avx512cd && xcr0_supports_zmm()) {
        caps.avx512 = true;
    }
#else
    (void)os_supports_xsave;
    (void)xcr0_supports_avx;
    (void)xcr0_supports_zmm;
#endif

    return caps;
}

//------------------------------------------------------------------------------
// Public SIMD-friendly apply
//------------------------------------------------------------------------------

MatrixXcd apply_single_qubit_simd(
    const MatrixXcd& L,
    const MatrixXcd& U,
    size_t target,
    size_t num_qubits
) {
    SimdCapabilities caps = detect_simd_capabilities();

    // Heuristic: only attempt vectorized loop when there's enough work
    // (otherwise scalar loop is faster)
    size_t dim = L.rows();
    size_t rank = L.cols();
    bool enough_work = (dim >= 512) && (rank >= 2);
    bool vectorize = enough_work && (caps.avx512 || caps.avx2);

    return apply_single_qubit_core(L, U, target, num_qubits, vectorize);
}

}  // namespace qlret

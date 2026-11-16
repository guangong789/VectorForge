#include "candidate_scoring_simd.hpp"

#include "numeric_contract.hpp"
#include "vectorforge/embedding_buffer.hpp"

#include <cerrno>
#if VECTORFORGE_HAS_AVX2_SCORING
#include <cfenv>
#include <cpuid.h>
#include <xmmintrin.h>
#endif
#if defined(__linux__) && defined(__GLIBC__)
#include <cstring>
#include <gnu/libc-version.h>
#endif

namespace vectorforge::detail {
bool simd_platform_supported() noexcept {
    // Experimental preconditions only. These version checks do not establish
    // math-provider equivalence and never authorize production SIMD selection.
#if VECTORFORGE_HAS_AVX2_SCORING && defined(__linux__) && defined(__GLIBC__) && \
    __GLIBC__ == 2 && __GLIBC_MINOR__ == 35 && defined(__GNUC__) && !defined(__clang__) && \
    __GNUC__ == 13 && __GNUC_MINOR__ == 4
    static const bool supported = std::strcmp(gnu_get_libc_version(), "2.35") == 0;
    return supported;
#else
    return false;
#endif
}

bool simd_hardware_supported() noexcept {
#if VECTORFORGE_HAS_AVX2_SCORING
    static const bool supported = [] {
        unsigned a, b, c, d;
        if (!__get_cpuid(1, &a, &b, &c, &d) ||
            (c & (bit_AVX | bit_OSXSAVE | bit_FMA)) != (bit_AVX | bit_OSXSAVE | bit_FMA)) { return false; }
        // XGETBV is legal only after OSXSAVE. Require OS-enabled XMM/YMM state.
        unsigned low, high;
        asm volatile("xgetbv" : "=a"(low), "=d"(high) : "c"(0));
        (void)high;
        return (low & 6U) == 6U && __get_cpuid_count(7, 0, &a, &b, &c, &d) && (b & bit_AVX2) != 0;
    }();
    return supported;
#else
    return false;
#endif
}

bool simd_environment_supported() noexcept {
#if VECTORFORGE_HAS_AVX2_SCORING
    const auto csr = _mm_getcsr();
    unsigned short control;
    asm volatile("fnstcw %0" : "=m"(control));
    return std::fegetround() == FE_TONEAREST && (csr & 0xe040U) == 0 &&
        (csr & 0x1f80U) == 0x1f80U && (control & 0x3fU) == 0x3fU;
#else
    return false;
#endif
}

bool simd_scoring_eligible(const EmbeddingBuffer& embeddings, std::span<const float> query,
                           std::span<const std::size_t> rows, std::span<float> output) noexcept {
    if (rows.size() < 8 || rows.size() > candidate_window_capacity || output.size() != rows.size() ||
        query.size() != embeddings.dimension() || !simd_platform_supported() ||
        !simd_hardware_supported() || !simd_environment_supported()) {
        return false;
    }
    for (const auto row : rows) {
        if (row >= embeddings.row_count() || !simd_row_addressable(row, query.size())) { return false; }
    }
    return true;
}

std::size_t score_candidates_auto(const EmbeddingBuffer& embeddings, std::span<const float> query,
                                  std::span<const std::size_t> rows, std::span<float> output) {
#if VECTORFORGE_HAS_AVX2_SCORING
    if (simd_scoring_eligible(embeddings, query, rows, output)) {
        std::size_t first = 0;
        for (; rows.size() - first >= 8; first += 8) {
            const int saved_errno = errno;
            std::fenv_t checkpoint;
            // In supported glibc these operations return zero for valid objects.
            // Save all six x87/MXCSR flags (including denormal operand) and controls.
            (void)std::fegetenv(&checkpoint);
            if (!score_candidates_eight_avx2(embeddings, query, rows.subspan(first, 8), output.subspan(first, 8))) {
                (void)std::fesetenv(&checkpoint);
                errno = saved_errno;
                return first + score_candidates_scalar(embeddings, query, rows.subspan(first, 8), output.subspan(first, 8));
            }
        }
        return first + score_candidates_scalar(embeddings, query, rows.subspan(first), output.subspan(first));
    }
#endif
    return score_candidates_scalar(embeddings, query, rows, output);
}

std::size_t score_candidates(const EmbeddingBuffer& embeddings, std::span<const float> query,
                             std::span<const std::size_t> rows, std::span<float> output) {
    // Final closure policy: unconditional generic scalar, independently of all
    // experiment availability/options/markers. Explicit studies call auto directly.
    return score_candidates_scalar(embeddings, query, rows, output);
}
} // namespace vectorforge::detail

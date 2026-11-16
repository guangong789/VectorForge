#pragma once

#include "numeric_contract.hpp"
#include "vectorforge/embedding_buffer.hpp"

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <xmmintrin.h>

// Private experiment only: no SIMD types or dispatch enter production headers.
namespace vectorforge::benchmark::cross_row {
inline constexpr std::size_t capacity = 32;

inline bool supported() {
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
}

inline unsigned environment() { return _mm_getcsr(); }
inline bool supported_environment() {
    const auto csr = environment();
    // Match the baseline's observed nearest/gradual-underflow environment, with
    // exceptions masked. Sticky flags/errno timing is not an API guarantee.
    return std::fegetround() == FE_TONEAREST && (csr & 0xe040U) == 0 && (csr & 0x1f80U) == 0x1f80U;
}

// Conservative signed BYTE-distance bound, not just signed 32-bit indices.
// Checked without forming row*D first. Every dimension of this row is covered.
inline std::int64_t checked_offset(std::size_t row, std::size_t dimension) {
    static_assert(sizeof(std::size_t) == 8 && sizeof(std::ptrdiff_t) == 8);
    constexpr auto limit = static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(float);
    if (dimension == 0) { throw std::invalid_argument("cross-row dimension must be positive"); }
    if (dimension - 1 > limit || row > (limit - (dimension - 1)) / dimension) {
        throw std::overflow_error("cross-row gather byte offset is not representable");
    }
    return static_cast<std::int64_t>(row * dimension);
}

inline void validate(const EmbeddingBuffer& embeddings, std::span<const float> query,
                     std::span<const std::size_t> rows, std::span<float> output) {
    if (rows.size() > capacity || output.size() != rows.size() || query.size() != embeddings.dimension()) {
        throw std::invalid_argument("invalid cross-row scoring shape");
    }
    for (const auto row : rows) {
        if (row >= embeddings.row_count()) { throw std::out_of_range("cross-row global row is out of range"); }
        (void)checked_offset(row, embeddings.dimension());
    }
}

// Same admitted-finite-input and non-overlapping-output preconditions as the
// production scorer. Repeated global rows are permitted. All indices/shape are
// validated before output writes. Callers must guard ISA AND FP environment.
// Return the finite prefix before the first intermediate overflow; do not write
// its slot or later slots. Float exception traps/side-effect timing are excluded.
using Kernel = std::size_t (*)(const EmbeddingBuffer&, std::span<const float>,
                              std::span<const std::size_t>, std::span<float>);
std::size_t scalar_scores(const EmbeddingBuffer&, std::span<const float>,
                          std::span<const std::size_t>, std::span<float>);
std::size_t avx2_scores(const EmbeddingBuffer&, std::span<const float>,
                        std::span<const std::size_t>, std::span<float>);
} // namespace vectorforge::benchmark::cross_row

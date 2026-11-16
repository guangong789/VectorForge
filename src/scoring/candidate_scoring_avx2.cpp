#include "candidate_scoring_simd.hpp"
#include "vectorforge/embedding_buffer.hpp"

#include "numeric_contract.hpp"

#include <array>
#include <cstdint>
#include <immintrin.h>
#include <limits>

namespace vectorforge::detail {
bool score_candidates_eight_avx2(const EmbeddingBuffer& buffer, std::span<const float> query,
                   std::span<const std::size_t> rows, std::span<float> output) {
    // Exactly eight rows; generic preflight proved bounds and signed byte offsets.
    // Generic dispatch owns environment capture/restoration and ordered replay.
    const auto* base = buffer.row(0).data();
    std::array<std::int64_t, 8> offsets;
    for (std::size_t lane = 0; lane < 8; ++lane) { offsets[lane] = static_cast<std::int64_t>(rows[lane] * query.size()); }
    const auto low = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(offsets.data()));
    const auto high = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(offsets.data() + 4));
    auto sum = _mm256_setzero_ps();
    for (std::size_t d = 0; d < query.size(); ++d) {
        const auto a = _mm256_i64gather_ps(base + d, low, 4);
        const auto b = _mm256_i64gather_ps(base + d, high, 4);
        const auto values = _mm256_insertf128_ps(_mm256_castps128_ps256(a), b, 1);
        sum = _mm256_fmadd_ps(_mm256_set1_ps(query[d]), values, sum);
        const auto magnitude = _mm256_andnot_ps(_mm256_set1_ps(-0.0F), sum);
        if (_mm256_movemask_ps(_mm256_cmp_ps(magnitude,
                _mm256_set1_ps(std::numeric_limits<float>::max()), _CMP_LE_OQ)) != 255) { return false; }
    }
    _mm256_storeu_ps(output.data(), sum);
    return true;
}
} // namespace vectorforge::detail

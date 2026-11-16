#include "cross_row.hpp"

#include <array>
#include <immintrin.h>

namespace vectorforge::benchmark::cross_row {
std::size_t avx2_scores(const EmbeddingBuffer& embeddings, std::span<const float> query,
                        std::span<const std::size_t> rows, std::span<float> output) {
    validate(embeddings, query, rows, output);
    if (rows.empty()) { return 0; } // No row(0)/gather for an empty buffer.
    const float* base = embeddings.row(0).data();
    std::size_t first = 0;
    for (; rows.size() - first >= 8; first += 8) {
        std::array<std::int64_t, 8> offsets;
        for (std::size_t lane = 0; lane < 8; ++lane) {
            offsets[lane] = checked_offset(rows[first + lane], query.size());
        }
        const auto low = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(offsets.data()));
        const auto high = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(offsets.data() + 4));
        auto sum = _mm256_setzero_ps();
        for (std::size_t d = 0; d < query.size(); ++d) {
            // AVX2's 64-bit-index float gather returns FOUR floats. Two gathers
            // cover the eight valid lanes without truncating global offsets.
            const auto a = _mm256_i64gather_ps(base + d, low, sizeof(float));
            const auto b = _mm256_i64gather_ps(base + d, high, sizeof(float));
            const auto values = _mm256_insertf128_ps(_mm256_castps128_ps256(a), b, 1);
            sum = _mm256_fmadd_ps(_mm256_set1_ps(query[d]), values, sum);
            const auto magnitude = _mm256_andnot_ps(_mm256_set1_ps(-0.0F), sum);
            const auto finite = _mm256_cmp_ps(magnitude,
                _mm256_set1_ps(std::numeric_limits<float>::max()), _CMP_LE_OQ);
            if (_mm256_movemask_ps(finite) != 255) {
                // Lane failures can occur in a different dimension order. Replay
                // only this bounded group in candidate order to find the exact
                // scalar prefix (an earlier lane can overflow at a later d).
                return first + scalar_scores(embeddings, query, rows.subspan(first, 8), output.subspan(first, 8));
            }
        }
        _mm256_storeu_ps(output.data() + first, sum);
    }
    // No masked/invalid gather lanes in tails, including group sizes 1..7.
    return first + scalar_scores(embeddings, query, rows.subspan(first), output.subspan(first));
}
} // namespace vectorforge::benchmark::cross_row

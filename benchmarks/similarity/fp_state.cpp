#include "fp_state.hpp"

#include "scoring/candidate_scoring_simd.hpp"

#include <cerrno>
#include <cfenv>
#include <cmath>
#include <stdexcept>

namespace vectorforge::benchmark::fp_state {
float scalar_step(float query, float value, float previous) {
    return std::fma(query, value, previous);
}

std::size_t unisolated_scores(const EmbeddingBuffer& embeddings, std::span<const float> query,
                               std::span<const std::size_t> rows, std::span<float> output) {
    if (!detail::simd_scoring_eligible(embeddings, query, rows, output)) {
        return detail::score_candidates_scalar(embeddings, query, rows, output);
    }
    std::size_t first = 0;
    for (; rows.size() - first >= 8; first += 8) {
        if (!eight_scores(embeddings, query, rows.subspan(first, 8), output.subspan(first, 8))) {
            return first + detail::score_candidates_scalar(embeddings, query, rows.subspan(first, 8), output.subspan(first, 8));
        }
    }
    return first + detail::score_candidates_scalar(embeddings, query, rows.subspan(first), output.subspan(first));
}

std::size_t isolated_scores(const EmbeddingBuffer& embeddings, std::span<const float> query,
                             std::span<const std::size_t> rows, std::span<float> output) {
    if (!detail::simd_scoring_eligible(embeddings, query, rows, output)) {
        return detail::score_candidates_scalar(embeddings, query, rows, output);
    }
    std::size_t first = 0;
    for (; rows.size() - first >= 8; first += 8) {
        const int saved_errno = errno;
        std::fenv_t checkpoint;
        if (std::fegetenv(&checkpoint) != 0) { throw std::runtime_error("study fegetenv failed"); }
        if (!eight_scores(embeddings, query, rows.subspan(first, 8), output.subspan(first, 8))) {
            if (std::fesetenv(&checkpoint) != 0) { throw std::runtime_error("study fesetenv failed"); }
            errno = saved_errno;
            return first + detail::score_candidates_scalar(embeddings, query, rows.subspan(first, 8), output.subspan(first, 8));
        }
    }
    return first + detail::score_candidates_scalar(embeddings, query, rows.subspan(first), output.subspan(first));
}
} // namespace vectorforge::benchmark::fp_state

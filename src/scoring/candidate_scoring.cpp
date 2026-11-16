#include "candidate_scoring.hpp"

#include "numeric_contract.hpp"
#include "vectorforge/embedding_buffer.hpp"

#include <cmath>
#include <stdexcept>

namespace vectorforge::detail {

std::size_t score_candidates(const EmbeddingBuffer& embeddings, std::span<const float> query,
                             std::span<const std::size_t> global_rows, std::span<float> output) {
    if (global_rows.size() > candidate_window_capacity || output.size() != global_rows.size() ||
        query.size() != embeddings.dimension()) {
        throw std::invalid_argument("invalid bounded candidate scoring shape");
    }
    for (std::size_t i = 0; i < global_rows.size(); ++i) {
        const auto embedding = embeddings.row(global_rows[i]);
        float score = 0.0F;
        for (std::size_t feature = 0; feature < query.size(); ++feature) {
            score = std::fma(query[feature], embedding[feature], score);
            if (!std::isfinite(score)) { return i; }
        }
        output[i] = score;
    }
    return global_rows.size();
}

} // namespace vectorforge::detail

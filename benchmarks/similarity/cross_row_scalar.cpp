#include "cross_row.hpp"

namespace vectorforge::benchmark::cross_row {
std::size_t scalar_scores(const EmbeddingBuffer& embeddings, std::span<const float> query,
                          std::span<const std::size_t> rows, std::span<float> output) {
    validate(embeddings, query, rows, output);
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const auto values = embeddings.row(rows[i]);
        float sum = 0.0F;
        for (std::size_t d = 0; d < query.size(); ++d) {
            sum = std::fma(query[d], values[d], sum);
            if (!std::isfinite(sum)) { return i; }
        }
        output[i] = sum;
    }
    return rows.size();
}
} // namespace vectorforge::benchmark::cross_row

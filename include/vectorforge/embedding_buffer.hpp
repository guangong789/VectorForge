#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace vectorforge {

// Owns contiguous float32 values in row-major order. Dimension is always positive.
class EmbeddingBuffer {
public:
    // Requires values.size() == row_count * dimension, including for zero rows.
    EmbeddingBuffer(std::size_t row_count, std::size_t dimension, std::vector<float> values);
    EmbeddingBuffer(const EmbeddingBuffer&) = delete;
    EmbeddingBuffer& operator=(const EmbeddingBuffer&) = delete;
    // Discard all views of the source before moving it. Source retains dimension,
    // but becomes a valid zero-row buffer.
    EmbeddingBuffer(EmbeddingBuffer&& other) noexcept;
    EmbeddingBuffer& operator=(EmbeddingBuffer&&) = delete;

    [[nodiscard]] std::size_t row_count() const noexcept { return row_count_; }
    [[nodiscard]] std::size_t dimension() const noexcept { return dimension_; }

    // Non-owning, read-only span; row index is dataset-global.
    [[nodiscard]] std::span<const float> row(std::size_t index) const &;
    std::span<const float> row(std::size_t) const && = delete;

private:
    std::vector<float> values_;
    std::size_t row_count_;
    std::size_t dimension_;
};

} // namespace vectorforge

#include "vectorforge/embedding_buffer.hpp"

#include <limits>
#include <stdexcept>
#include <utility>

namespace vectorforge {

EmbeddingBuffer::EmbeddingBuffer(std::size_t row_count, std::size_t dimension,
                                 std::vector<float> values)
    : values_(std::move(values)), row_count_(row_count), dimension_(dimension) {
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
                  "VectorForge requires IEEE 754 float32");
    if (dimension_ == 0) {
        throw std::invalid_argument("embedding dimension must be positive");
    }
    if (row_count_ > std::numeric_limits<std::size_t>::max() / dimension_) {
        throw std::overflow_error("embedding row count times dimension overflows size_t");
    }
    if (values_.size() != row_count_ * dimension_) {
        throw std::invalid_argument("embedding size does not match row count and dimension");
    }
}

EmbeddingBuffer::EmbeddingBuffer(EmbeddingBuffer&& other) noexcept
    : values_(std::exchange(other.values_, {})), row_count_(std::exchange(other.row_count_, 0)),
      dimension_(other.dimension_) {}

std::span<const float> EmbeddingBuffer::row(std::size_t index) const & {
    if (index >= row_count_) {
        throw std::out_of_range("embedding row index is out of range");
    }
    // Construction proved row_count * dimension fits; index < row_count.
    return std::span<const float>(values_).subspan(index * dimension_, dimension_);
}

} // namespace vectorforge

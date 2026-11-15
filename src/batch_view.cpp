#include "vectorforge/batch_view.hpp"

#include <algorithm>
#include <stdexcept>

namespace vectorforge {

BatchView::BatchView(const ColumnBatch& columns, const EmbeddingBuffer& embeddings,
                     std::size_t row_offset, std::size_t row_count)
    : columns_(&columns), embeddings_(&embeddings), row_offset_(row_offset), row_count_(row_count) {
    if (columns.row_count() != embeddings.row_count()) {
        throw std::invalid_argument("column and embedding row counts must match");
    }
    // Subtraction avoids overflow from row_offset + row_count.
    if (row_offset_ > columns.row_count() || row_count_ > columns.row_count() - row_offset_) {
        throw std::out_of_range("batch range is outside the dataset");
    }
}

std::size_t BatchView::global_row(std::size_t local_row) const {
    if (local_row >= row_count_) {
        throw std::out_of_range("batch-local row index is out of range");
    }
    // The constructor bounded the entire range by the owners' row count.
    return row_offset_ + local_row;
}

std::span<const float> BatchView::embedding_row(std::size_t local_row) const {
    return embeddings_->row(global_row(local_row));
}

BatchView make_batch(const ColumnBatch& columns, const EmbeddingBuffer& embeddings,
                     std::size_t row_offset, std::size_t batch_size) {
    if (batch_size == 0) {
        throw std::invalid_argument("batch size must be positive");
    }
    if (row_offset > columns.row_count()) {
        throw std::out_of_range("batch offset is outside the dataset");
    }
    return BatchView(columns, embeddings, row_offset,
                     std::min(batch_size, columns.row_count() - row_offset));
}

} // namespace vectorforge

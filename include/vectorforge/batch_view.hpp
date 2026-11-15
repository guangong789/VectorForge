#pragma once

#include "vectorforge/column_batch.hpp"
#include "vectorforge/embedding_buffer.hpp"

namespace vectorforge {

// Borrows both owners. They must outlive this view and all spans obtained from it,
// and must not be moved while borrowed. Copying this view does not copy any data.
class BatchView {
public:
    // Exact half-open dataset range [row_offset, row_offset + row_count).
    // Both owners must have the same total row count, even for an empty view.
    BatchView(const ColumnBatch& columns, const EmbeddingBuffer& embeddings,
              std::size_t row_offset, std::size_t row_count);
    BatchView(const ColumnBatch&&, const EmbeddingBuffer&, std::size_t, std::size_t) = delete;
    BatchView(const ColumnBatch&, const EmbeddingBuffer&&, std::size_t, std::size_t) = delete;

    [[nodiscard]] std::size_t row_offset() const noexcept { return row_offset_; }
    [[nodiscard]] std::size_t row_count() const noexcept { return row_count_; }
    [[nodiscard]] std::size_t column_count() const noexcept { return columns_->column_count(); }
    [[nodiscard]] std::size_t dimension() const noexcept { return embeddings_->dimension(); }

    // Converts a checked batch-local row into its dataset-global row index.
    [[nodiscard]] std::size_t global_row(std::size_t local_row) const;
    [[nodiscard]] std::span<const float> embedding_row(std::size_t local_row) const;

    // Span indices are batch-local, starting at zero.
    template <ColumnValue T>
    [[nodiscard]] std::span<const T> column(std::size_t index) const {
        return columns_->column<T>(index).subspan(row_offset_, row_count_);
    }

private:
    const ColumnBatch* columns_;
    const EmbeddingBuffer* embeddings_;
    std::size_t row_offset_;
    std::size_t row_count_;
};

// Takes up to batch_size rows from a dataset-global offset. A zero batch size
// is invalid. At the dataset end, returns an empty view; past it, throws.
[[nodiscard]] BatchView make_batch(const ColumnBatch& columns, const EmbeddingBuffer& embeddings,
                                   std::size_t row_offset, std::size_t batch_size);
BatchView make_batch(const ColumnBatch&&, const EmbeddingBuffer&, std::size_t, std::size_t) = delete;
BatchView make_batch(const ColumnBatch&, const EmbeddingBuffer&&, std::size_t, std::size_t) = delete;

} // namespace vectorforge

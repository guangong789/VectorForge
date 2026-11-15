#include "vectorforge/column_batch.hpp"

#include <utility>

namespace vectorforge {

ColumnBatch::ColumnBatch(std::vector<Column> columns)
    : columns_(std::move(columns)), row_count_(0) {
    if (!columns_.empty()) {
        row_count_ = std::visit([](const auto& values) { return values.size(); }, columns_.front());
    }
    for (const auto& column : columns_) {
        const auto size = std::visit([](const auto& values) { return values.size(); }, column);
        if (size != row_count_) {
            throw std::invalid_argument("all columns must have the same row count");
        }
    }
}

ColumnBatch::ColumnBatch(ColumnBatch&& other) noexcept
    : columns_(std::exchange(other.columns_, {})), row_count_(std::exchange(other.row_count_, 0)) {}

} // namespace vectorforge

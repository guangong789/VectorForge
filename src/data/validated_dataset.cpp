#include "vectorforge/validated_dataset.hpp"

#include "numeric_contract.hpp"
#include "vectorforge/column_batch.hpp"
#include "vectorforge/embedding_buffer.hpp"

#include <cmath>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace vectorforge {
namespace {

ColumnBatch snapshot_columns(const ColumnBatch& source) {
    std::vector<ColumnBatch::Column> copies;
    copies.reserve(source.column_count());
    for (std::size_t index = 0; index < source.column_count(); ++index) {
        std::visit([&](auto values) {
            if (values.size() != source.row_count()) {
                throw std::invalid_argument("numeric column length does not match row count");
            }
            using T = typename decltype(values)::value_type;
            if constexpr (std::same_as<T, float>) {
                for (const auto value : values) {
                    if (!std::isfinite(value)) {
                        throw std::invalid_argument("dataset float columns must be finite");
                    }
                }
            }
            copies.emplace_back(std::vector<T>(values.begin(), values.end()));
        }, source.column_view(index));
    }
    return ColumnBatch(std::move(copies));
}

EmbeddingBuffer snapshot_embeddings(const EmbeddingBuffer& source) {
    // Source construction already established the positive dimension and checked
    // this product. Reserve one contiguous allocation, never one per row.
    std::vector<float> copy;
    copy.reserve(source.row_count() * source.dimension());
    for (std::size_t row = 0; row < source.row_count(); ++row) {
        const auto values = source.row(row);
        for (const auto value : values) {
            if (!std::isfinite(value)) {
                throw std::invalid_argument("dataset embeddings must be finite");
            }
        }
        copy.insert(copy.end(), values.begin(), values.end());
    }
    return EmbeddingBuffer(source.row_count(), source.dimension(), std::move(copy));
}

} // namespace

struct ValidatedDataset::Storage {
    ColumnBatch columns;
    EmbeddingBuffer embeddings;
    std::vector<RecordID> ids;

    Storage(const ColumnBatch& source_columns, const EmbeddingBuffer& source_embeddings,
            std::span<const RecordID> source_ids)
        : columns(snapshot_columns(source_columns)), embeddings(snapshot_embeddings(source_embeddings)),
          ids(source_ids.begin(), source_ids.end()) {
        std::unordered_set<RecordID> seen;
        for (const auto id : ids) {
            if (!seen.insert(id).second) {
                throw std::invalid_argument("Record IDs must be unique at dataset admission");
            }
        }
    }
};

ValidatedDataset::ValidatedDataset(const ColumnBatch& columns, const EmbeddingBuffer& embeddings,
                                   std::span<const RecordID> record_ids) {
    if (columns.row_count() != embeddings.row_count() || columns.row_count() != record_ids.size()) {
        throw std::invalid_argument("features, embeddings and Record IDs must have equal row counts");
    }
    storage_ = std::make_unique<const Storage>(columns, embeddings, record_ids);
}

ValidatedDataset::~ValidatedDataset() = default;
ValidatedDataset::ValidatedDataset(ValidatedDataset&&) noexcept = default;

const ValidatedDataset::Storage& ValidatedDataset::storage() const {
    if (!storage_) {
        throw std::logic_error("cannot access a moved-from validated dataset");
    }
    return *storage_;
}

const ColumnBatch& ValidatedDataset::columns() const & { return storage().columns; }
const EmbeddingBuffer& ValidatedDataset::embeddings() const & { return storage().embeddings; }
std::span<const RecordID> ValidatedDataset::record_ids() const & { return storage().ids; }

} // namespace vectorforge

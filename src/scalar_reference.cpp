#include "vectorforge/scalar_reference.hpp"

#include "vectorforge/column_batch.hpp"
#include "vectorforge/embedding_buffer.hpp"
#include "vectorforge/validated_dataset.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_set>

namespace vectorforge {
namespace {

void require_finite(std::span<const float> values) {
    for (const float value : values) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("scalar reference inputs must be finite");
        }
    }
}

void validate_comparison(Comparison comparison) {
    switch (comparison) {
    case Comparison::less:
    case Comparison::less_equal:
    case Comparison::greater:
    case Comparison::greater_equal:
    case Comparison::equal:
    case Comparison::not_equal:
        return;
    }
    throw std::invalid_argument("unsupported predicate comparison");
}

template <ColumnValue T>
bool compare(T value, T constant, Comparison comparison) {
    switch (comparison) {
    case Comparison::less:          return value < constant;
    case Comparison::less_equal:    return value <= constant;
    case Comparison::greater:       return value > constant;
    case Comparison::greater_equal: return value >= constant;
    case Comparison::equal:         return value == constant;
    case Comparison::not_equal:     return value != constant;
    }
    throw std::invalid_argument("unsupported predicate comparison");
}

void validate_query_vector(const EmbeddingBuffer& embeddings, std::span<const float> query) {
    if (embeddings.dimension() == 0 || query.size() != embeddings.dimension()) {
        throw std::invalid_argument("query must match the positive embedding dimension");
    }
    require_finite(query);
}

void validate_predicates(const ColumnBatch& columns, std::span<const Predicate> predicates) {
    for (const auto& predicate : predicates) {
        validate_comparison(predicate.comparison);
        std::visit([&]<typename T>(T constant) {
            // Typed access validates both column index and exact type, even with no rows.
            (void)columns.column<T>(predicate.column_index);
            if constexpr (std::same_as<T, float>) {
                if (!std::isfinite(constant)) {
                    throw std::invalid_argument("predicate constant must be finite");
                }
            }
        }, predicate.constant);
    }
}

void validate_inputs(const ColumnBatch& columns, const EmbeddingBuffer& embeddings,
                     std::span<const RecordID> record_ids, std::span<const float> query,
                     std::span<const Predicate> predicates) {
    const auto rows = columns.row_count();
    if (rows != embeddings.row_count() || rows != record_ids.size()) {
        throw std::invalid_argument("features, embeddings and Record IDs must have equal row counts");
    }
    validate_query_vector(embeddings, query);

    for (std::size_t index = 0; index < columns.column_count(); ++index) {
        std::visit([rows](auto values) {
            if (values.size() != rows) {
                throw std::invalid_argument("numeric column length does not match row count");
            }
            if constexpr (std::same_as<typename decltype(values)::value_type, float>) {
                require_finite(values);
            }
        }, columns.column_view(index));
    }
    validate_predicates(columns, predicates);

    std::unordered_set<RecordID> seen;
    for (const auto id : record_ids) {
        if (!seen.insert(id).second) {
            throw std::invalid_argument("Record IDs must be unique");
        }
    }
    for (std::size_t row = 0; row < rows; ++row) {
        require_finite(embeddings.row(row));
    }
}

bool qualifies(const ColumnBatch& columns, std::size_t row, std::span<const Predicate> predicates) {
    for (const auto& predicate : predicates) {
        const bool matches = std::visit([&]<typename T>(T constant) {
            return compare(columns.column<T>(predicate.column_index)[row], constant, predicate.comparison);
        }, predicate.constant);
        if (!matches) {
            return false;
        }
    }
    return true;
}

float dot_product(std::span<const float> query, std::span<const float> embedding) {
    float score = 0.0F;
    for (std::size_t feature = 0; feature < query.size(); ++feature) {
        score = std::fma(query[feature], embedding[feature], score);
        if (!std::isfinite(score)) {
            throw std::overflow_error("scalar dot product produced a non-finite score");
        }
    }
    return score;
}

std::vector<ScoredRecord> execute_query(
    const ColumnBatch& columns, const EmbeddingBuffer& embeddings,
    std::span<const RecordID> record_ids, std::span<const float> query,
    std::span<const Predicate> predicates, std::size_t k) {
    std::vector<ScoredRecord> candidates;
    if (k == 0) {
        return candidates;
    }
    for (std::size_t row = 0; row < columns.row_count(); ++row) {
        if (qualifies(columns, row, predicates)) {
            candidates.push_back({record_ids[row], dot_product(query, embeddings.row(row))});
        }
    }
    std::sort(candidates.begin(), candidates.end(), [](const ScoredRecord& left, const ScoredRecord& right) {
        if (left.score != right.score) {
            return left.score > right.score;
        }
        return left.id < right.id;
    });
    if (candidates.size() > k) {
        candidates.resize(k);
    }
    return candidates;
}

} // namespace

std::vector<ScoredRecord> execute_scalar_reference(
    const ColumnBatch& columns, const EmbeddingBuffer& embeddings,
    std::span<const RecordID> record_ids, std::span<const float> query,
    std::span<const Predicate> predicates, std::size_t k) {
    validate_inputs(columns, embeddings, record_ids, query, predicates);
    return execute_query(columns, embeddings, record_ids, query, predicates, k);
}

std::vector<ScoredRecord> execute_scalar_reference(
    const ValidatedDataset& dataset, std::span<const float> query,
    std::span<const Predicate> predicates, std::size_t k) {
    const auto& columns = dataset.columns();
    const auto& embeddings = dataset.embeddings();
    validate_query_vector(embeddings, query);
    validate_predicates(columns, predicates);
    return execute_query(columns, embeddings, dataset.record_ids(), query, predicates, k);
}

} // namespace vectorforge

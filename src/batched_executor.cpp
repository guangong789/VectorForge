#include "vectorforge/batched_executor.hpp"

#include "vectorforge/batch_view.hpp"
#include "vectorforge/validated_dataset.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_set>

namespace vectorforge {
namespace {

// Deliberately independent validation and execution: this path never calls the
// scalar oracle or shares predicate/scoring/candidate-collection helpers with it.
void validate_floats(std::span<const float> values) {
    for (const auto value : values) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("batched query inputs must be finite");
        }
    }
}

void validate_query_vector(const EmbeddingBuffer& embeddings, std::span<const float> query) {
    if (embeddings.dimension() == 0 || embeddings.dimension() != query.size()) {
        throw std::invalid_argument("query must match the positive embedding dimension");
    }
    validate_floats(query);
}

void validate_predicates(const ColumnBatch& columns, std::span<const Predicate> predicates) {
    for (const auto& predicate : predicates) {
        switch (predicate.comparison) {
        case Comparison::less:
        case Comparison::less_equal:
        case Comparison::greater:
        case Comparison::greater_equal:
        case Comparison::equal:
        case Comparison::not_equal:
            break;
        default:
            throw std::invalid_argument("unsupported predicate comparison");
        }
        std::visit([&]<typename T>(T constant) {
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
                     std::span<const RecordID> ids, std::span<const float> query,
                     std::span<const Predicate> predicates) {
    const auto rows = columns.row_count();
    if (rows != ids.size() || rows != embeddings.row_count()) {
        throw std::invalid_argument("features, embeddings and Record IDs must have equal row counts");
    }
    validate_query_vector(embeddings, query);
    for (std::size_t index = 0; index < columns.column_count(); ++index) {
        std::visit([rows](auto values) {
            if (values.size() != rows) {
                throw std::invalid_argument("numeric column length does not match row count");
            }
            if constexpr (std::same_as<typename decltype(values)::value_type, float>) {
                validate_floats(values);
            }
        }, columns.column_view(index));
    }
    validate_predicates(columns, predicates);

    std::unordered_set<RecordID> seen;
    for (const auto id : ids) {
        if (!seen.insert(id).second) {
            throw std::invalid_argument("Record IDs must be unique");
        }
    }
    for (std::size_t row = 0; row < rows; ++row) {
        validate_floats(embeddings.row(row));
    }
}

bool matches_row(const BatchView& batch, std::size_t local, std::span<const Predicate> predicates) {
    for (const auto& predicate : predicates) {
        const bool matches = std::visit([&]<typename T>(T constant) {
            const T value = batch.column<T>(predicate.column_index)[local];
            switch (predicate.comparison) {
            case Comparison::less:          return value < constant;
            case Comparison::less_equal:    return value <= constant;
            case Comparison::greater:       return value > constant;
            case Comparison::greater_equal: return value >= constant;
            case Comparison::equal:         return value == constant;
            case Comparison::not_equal:     return value != constant;
            }
            throw std::invalid_argument("unsupported predicate comparison");
        }, predicate.constant);
        if (!matches) {
            return false;
        }
    }
    return true;
}

std::vector<ScoredRecord> execute_query(
    const ColumnBatch& columns, const EmbeddingBuffer& embeddings,
    std::span<const RecordID> record_ids, std::span<const float> query,
    std::span<const Predicate> predicates, std::size_t k, std::size_t batch_size) {
    std::vector<ScoredRecord> candidates;
    if (k == 0) {
        return candidates;
    }
    const auto better = [](const ScoredRecord& a, const ScoredRecord& b) {
        return a.score > b.score || (a.score == b.score && a.id < b.id);
    };
    for (std::size_t offset = 0; offset < columns.row_count();) {
        const auto batch = make_batch(columns, embeddings, offset, batch_size);
        for (std::size_t local = 0; local < batch.row_count(); ++local) {
            if (!matches_row(batch, local, predicates)) {
                continue;
            }
            const auto embedding = batch.embedding_row(local);
            float score = 0.0F;
            for (std::size_t feature = 0; feature < query.size(); ++feature) {
                score = std::fma(query[feature], embedding[feature], score);
                if (!std::isfinite(score)) {
                    throw std::overflow_error("batched dot product produced a non-finite score");
                }
            }
            const ScoredRecord candidate{record_ids[batch.global_row(local)], score};
            // With "better" as the heap comparator, the root is the worst
            // retained candidate. Never grow beyond K, even during replacement.
            if (candidates.size() < k) {
                candidates.push_back(candidate);
                std::push_heap(candidates.begin(), candidates.end(), better);
            } else if (better(candidate, candidates.front())) {
                std::pop_heap(candidates.begin(), candidates.end(), better);
                candidates.back() = candidate;
                std::push_heap(candidates.begin(), candidates.end(), better);
            }
        }
        // make_batch bounds this sum by the dataset row count, even for SIZE_MAX
        // batch sizes. Advancing by batch_size instead could overflow or skip rows.
        offset += batch.row_count();
    }
    std::sort_heap(candidates.begin(), candidates.end(), better);
    return candidates;
}

} // namespace

std::vector<ScoredRecord> execute_batched(
    const ColumnBatch& columns, const EmbeddingBuffer& embeddings,
    std::span<const RecordID> record_ids, std::span<const float> query,
    std::span<const Predicate> predicates, std::size_t k, std::size_t batch_size) {
    if (batch_size == 0) {
        throw std::invalid_argument("batch size must be positive");
    }
    validate_inputs(columns, embeddings, record_ids, query, predicates);
    return execute_query(columns, embeddings, record_ids, query, predicates, k, batch_size);
}

std::vector<ScoredRecord> execute_batched(
    const ValidatedDataset& dataset, std::span<const float> query,
    std::span<const Predicate> predicates, std::size_t k, std::size_t batch_size) {
    if (batch_size == 0) {
        throw std::invalid_argument("batch size must be positive");
    }
    const auto& columns = dataset.columns();
    const auto& embeddings = dataset.embeddings();
    validate_query_vector(embeddings, query);
    validate_predicates(columns, predicates);
    return execute_query(columns, embeddings, dataset.record_ids(), query, predicates, k, batch_size);
}

} // namespace vectorforge

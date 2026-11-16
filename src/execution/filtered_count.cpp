#include "vectorforge/filtered_count.hpp"

#include "predicate_evaluation.hpp"
#include "vectorforge/validated_dataset.hpp"

#include <stdexcept>

namespace vectorforge {

std::size_t count_where(const ValidatedDataset& dataset, std::span<const Predicate> predicates,
                        std::size_t batch_size) {
    if (batch_size == 0) { throw std::invalid_argument("batch size must be positive"); }
    const auto& columns = dataset.columns();
    detail::validate_predicates(columns, predicates);
    if (predicates.empty()) { return columns.row_count(); }

    const auto& embeddings = dataset.embeddings(); // Batch alignment metadata only.
    std::size_t count = 0;
    for (std::size_t offset = 0; offset < columns.row_count();) {
        const auto batch = make_batch(columns, embeddings, offset, batch_size);
        for (std::size_t local = 0; local < batch.row_count(); ++local) {
            if (detail::matches_row(batch, local, predicates)) { ++count; }
        }
        offset += batch.row_count(); // Actual length, bounded by dataset row_count.
    }
    return count;
}

} // namespace vectorforge

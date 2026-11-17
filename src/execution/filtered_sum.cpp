#include "vectorforge/filtered_sum.hpp"

#include "checked_sum.hpp"
#include "numeric_column_validation.hpp"
#include "predicate_evaluation.hpp"
#include "vectorforge/column_batch.hpp"

#include <algorithm>
#include <stdexcept>

namespace vectorforge {

std::int64_t sum_where(const ColumnBatch& columns, std::size_t target_column,
                       std::span<const Predicate> predicates, std::size_t batch_size) {
    if (batch_size == 0) { throw std::invalid_argument("batch size must be positive"); }
    const auto rows = columns.row_count();
    for (std::size_t index = 0; index < columns.column_count(); ++index) {
        const auto column = columns.column_view(index);
        detail::validate_column_length(column, rows);
        detail::validate_column_floats(column, "SUM float columns must be finite");
    }
    const auto target = columns.column<std::int32_t>(target_column);
    detail::validate_predicates(columns, predicates);

    std::int64_t total = 0;
    for (std::size_t offset = 0; offset < rows;) {
        // Subtraction/minimum keep the range valid even for SIZE_MAX batch size.
        const auto end = offset + std::min(batch_size, rows - offset);
        for (std::size_t row = offset; row < end; ++row) {
            if (detail::matches_row(columns, row, predicates)) {
                total = detail::checked_sum_add(total, target[row]);
            }
        }
        offset = end;
    }
    return total;
}

} // namespace vectorforge

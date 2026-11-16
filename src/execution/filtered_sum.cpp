#include "vectorforge/filtered_sum.hpp"

#include "checked_sum.hpp"
#include "predicate_evaluation.hpp"
#include "vectorforge/column_batch.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace vectorforge {

std::int64_t sum_where(const ColumnBatch& columns, std::size_t target_column,
                       std::span<const Predicate> predicates, std::size_t batch_size) {
    if (batch_size == 0) { throw std::invalid_argument("batch size must be positive"); }
    const auto rows = columns.row_count();
    for (std::size_t index = 0; index < columns.column_count(); ++index) {
        std::visit([rows](auto values) {
            if (values.size() != rows) {
                throw std::invalid_argument("numeric column length does not match row count");
            }
            if constexpr (std::same_as<typename decltype(values)::value_type, float>) {
                for (const auto value : values) {
                    if (!std::isfinite(value)) {
                        throw std::invalid_argument("SUM float columns must be finite");
                    }
                }
            }
        }, columns.column_view(index));
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

#include "vectorforge/filtered_affine_sum.hpp"

#include "numeric_column_validation.hpp"
#include "predicate_evaluation.hpp"
#include "vectorforge/column_batch.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace vectorforge {
namespace {

// Local int64 boundary; the existing int32 SUM helper and executor stay unchanged.
std::int64_t checked_add(std::int64_t total, std::int64_t value) {
    if ((value > 0 && total > std::numeric_limits<std::int64_t>::max() - value) ||
        (value < 0 && total < std::numeric_limits<std::int64_t>::min() - value)) {
        throw std::overflow_error("filtered affine sum overflows int64");
    }
    return total + value;
}

} // namespace

std::int64_t sum_affine_where(const ColumnBatch& columns, std::size_t source_column,
                             std::int32_t a, std::int32_t b,
                             std::span<const Predicate> predicates, std::size_t batch_size) {
    if (batch_size == 0) { throw std::invalid_argument("batch size must be positive"); }
    const auto rows = columns.row_count();
    for (std::size_t column = 0; column < columns.column_count(); ++column) {
        const auto view = columns.column_view(column);
        detail::validate_column_length(view, rows);
        detail::validate_column_floats(view, "affine SUM float columns must be finite");
    }
    const auto source = columns.column<std::int32_t>(source_column);
    detail::validate_predicates(columns, predicates);

    std::int64_t total = 0;
    for (std::size_t offset = 0; offset < rows;) {
        const auto end = offset + std::min(batch_size, rows - offset);
        for (std::size_t row = offset; row < end; ++row) {
            if (!detail::matches_row(columns, row, predicates)) { continue; }
            // Both operands widen BEFORE multiplication. The full int32-domain
            // product and subsequent bias addition fit int64 by the public bound.
            const auto product = static_cast<std::int64_t>(a) * static_cast<std::int64_t>(source[row]);
            const auto value = product + static_cast<std::int64_t>(b);
            total = checked_add(total, value);
        }
        offset = end;
    }
    return total;
}

} // namespace vectorforge

#pragma once

#include "numeric_contract.hpp"
#include "vectorforge/column_batch.hpp"

#include <cmath>
#include <stdexcept>

namespace vectorforge::detail {

// Small raw-column checks shared by numeric COUNT and SUM. Callers determine
// validation order: COUNT checks all shapes first; SUM keeps its per-column order.
// ColumnView copies only borrowed span metadata; neither check allocates/copies data.
inline void validate_column_length(ColumnBatch::ColumnView column, std::size_t rows) {
    std::visit([rows](auto values) {
        if (values.size() != rows) {
            throw std::invalid_argument("numeric column length does not match row count");
        }
    }, column);
}

inline void validate_column_floats(ColumnBatch::ColumnView column, const char* diagnostic) {
    if (const auto* values = std::get_if<std::span<const float>>(&column)) {
        for (const auto value : *values) {
            if (!std::isfinite(value)) { throw std::invalid_argument(diagnostic); }
        }
    }
}

} // namespace vectorforge::detail

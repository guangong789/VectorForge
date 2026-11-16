#pragma once

#include "numeric_contract.hpp"
#include "vectorforge/batch_view.hpp"
#include "vectorforge/query_types.hpp"

#include <cmath>
#include <stdexcept>

// Private implementation shared only by production Batched Top-K and COUNT.
// Inline to retain the original per-row optimisation/inlining opportunity.
// The independent Scalar Reference must not include this helper.
namespace vectorforge::detail {

inline void validate_predicates(const ColumnBatch& columns, std::span<const Predicate> predicates) {
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

// Precondition: predicates were all validated, and local < batch.row_count().
inline bool matches_row(const BatchView& batch, std::size_t local, std::span<const Predicate> predicates) {
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

} // namespace vectorforge::detail

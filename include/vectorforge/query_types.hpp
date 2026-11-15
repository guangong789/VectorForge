#pragma once

#include <cstddef>
#include <cstdint>
#include <variant>

namespace vectorforge {

using RecordID = std::uint64_t;
using NumericConstant = std::variant<std::int32_t, std::int64_t, float>;

enum class Comparison { less, less_equal, greater, greater_equal, equal, not_equal };

struct Predicate {
    std::size_t column_index;
    Comparison comparison;
    NumericConstant constant;
};

struct ScoredRecord {
    RecordID id;
    float score;
    bool operator==(const ScoredRecord&) const = default;
};

} // namespace vectorforge

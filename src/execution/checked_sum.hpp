#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace vectorforge::detail {

// Private int32 SUM arithmetic boundary, also exercised directly by small tests:
// reaching int64 overflow from zero needs over 2^32 int32 values (>16 GiB).
inline std::int64_t checked_sum_add(std::int64_t total, std::int32_t value) {
    const auto delta = static_cast<std::int64_t>(value);
    if ((delta > 0 && total > std::numeric_limits<std::int64_t>::max() - delta) ||
        (delta < 0 && total < std::numeric_limits<std::int64_t>::min() - delta)) {
        throw std::overflow_error("filtered int32 sum overflows int64");
    }
    return total + delta;
}

} // namespace vectorforge::detail

#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <variant>
#include <vector>

namespace vectorforge {

template <typename T>
concept ColumnValue = std::same_as<T, std::int32_t> ||
                      std::same_as<T, std::int64_t> || std::same_as<T, float>;

// Owns all columns. No structural mutation is exposed after construction.
class ColumnBatch {
public:
    using Column = std::variant<std::vector<std::int32_t>,
                                std::vector<std::int64_t>, std::vector<float>>;

    // Row count is inferred from the first column; no columns means zero rows.
    // Pass std::move(columns) to transfer existing allocations without copying.
    explicit ColumnBatch(std::vector<Column> columns);
    ColumnBatch(const ColumnBatch&) = delete;
    ColumnBatch& operator=(const ColumnBatch&) = delete;
    // Discard all views of the source before moving it. Source becomes empty.
    ColumnBatch(ColumnBatch&& other) noexcept;
    ColumnBatch& operator=(ColumnBatch&&) = delete;

    [[nodiscard]] std::size_t row_count() const noexcept { return row_count_; }
    [[nodiscard]] std::size_t column_count() const noexcept { return columns_.size(); }

    // Non-owning, read-only span. Index and stored type are checked.
    template <ColumnValue T>
    [[nodiscard]] std::span<const T> column(std::size_t index) const & {
        const auto& storage = columns_.at(index);
        const auto* values = std::get_if<std::vector<T>>(&storage);
        if (values == nullptr) {
            throw std::invalid_argument("column access type does not match stored type");
        }
        return *values;
    }

    template <ColumnValue T>
    std::span<const T> column(std::size_t) const && = delete;

private:
    std::vector<Column> columns_;
    std::size_t row_count_;
};

} // namespace vectorforge

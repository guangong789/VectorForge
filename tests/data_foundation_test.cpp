#include "vectorforge/batch_view.hpp"

#include <array>
#include <exception>
#include <iostream>
#include <limits>
#include <source_location>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace {

using vectorforge::BatchView;
using vectorforge::ColumnBatch;
using vectorforge::EmbeddingBuffer;
using vectorforge::make_batch;

template <typename T>
concept AccessibleColumn = requires(const ColumnBatch& columns) { columns.column<T>(0); };

template <typename C, typename E>
concept CanMakeBatch = requires(C&& columns, E&& embeddings) {
    make_batch(std::forward<C>(columns), std::forward<E>(embeddings), 0, 1);
};

template <typename T>
concept CanBorrowColumn = requires(T&& columns) {
    std::forward<T>(columns).template column<float>(0);
};

template <typename T>
concept CanBorrowRow = requires(T&& embeddings) { std::forward<T>(embeddings).row(0); };

static_assert(AccessibleColumn<std::int32_t> && AccessibleColumn<std::int64_t> && AccessibleColumn<float>);
static_assert(!AccessibleColumn<double> && !AccessibleColumn<std::uint32_t> && !AccessibleColumn<bool>);
static_assert(std::same_as<decltype(std::declval<ColumnBatch&>().column<float>(0)), std::span<const float>>);
static_assert(std::same_as<decltype(std::declval<const ColumnBatch&>().column<std::int64_t>(0)),
                           std::span<const std::int64_t>>);
static_assert(std::same_as<decltype(std::declval<EmbeddingBuffer&>().row(0)), std::span<const float>>);
static_assert(std::same_as<decltype(std::declval<BatchView&>().column<float>(0)), std::span<const float>>);
static_assert(std::same_as<decltype(std::declval<BatchView&>().embedding_row(0)), std::span<const float>>);
static_assert(!std::is_assignable_v<decltype(std::declval<EmbeddingBuffer&>().row(0)[0]), float>);
static_assert(!std::is_assignable_v<decltype(std::declval<ColumnBatch&>().column<float>(0)[0]), float>);
static_assert(!std::is_copy_constructible_v<ColumnBatch> && !std::is_copy_constructible_v<EmbeddingBuffer>);
static_assert(!std::is_move_assignable_v<ColumnBatch> && !std::is_move_assignable_v<EmbeddingBuffer>);
static_assert(std::is_nothrow_move_constructible_v<ColumnBatch>);
static_assert(std::is_nothrow_move_constructible_v<EmbeddingBuffer>);
static_assert(std::is_trivially_copyable_v<BatchView>);
static_assert(CanMakeBatch<ColumnBatch&, EmbeddingBuffer&>);
static_assert(!CanMakeBatch<ColumnBatch, EmbeddingBuffer&>);
static_assert(!CanMakeBatch<ColumnBatch&, EmbeddingBuffer>);
static_assert(!CanMakeBatch<ColumnBatch, EmbeddingBuffer>);
static_assert(!std::is_constructible_v<BatchView, ColumnBatch&&, const EmbeddingBuffer&, std::size_t, std::size_t>);
static_assert(!std::is_constructible_v<BatchView, const ColumnBatch&, EmbeddingBuffer&&, std::size_t, std::size_t>);
static_assert(!CanBorrowColumn<ColumnBatch> && !CanBorrowRow<EmbeddingBuffer>);
static_assert(CanBorrowColumn<const ColumnBatch&> && CanBorrowRow<const EmbeddingBuffer&>);

void check(bool condition, std::string_view message,
           const std::source_location location = std::source_location::current()) {
    if (!condition) {
        throw std::runtime_error(std::string(location.file_name()) + ":" +
                                 std::to_string(location.line()) + ": " + std::string(message));
    }
}

template <typename Error, typename Function>
void expect_throw(Function&& function,
                  const std::source_location location = std::source_location::current()) {
    try {
        std::forward<Function>(function)();
    } catch (const Error&) {
        return;
    } catch (const std::exception& error) {
        check(false, std::string("unexpected exception: ") + error.what(), location);
    }
    check(false, "expected exception was not thrown", location);
}

ColumnBatch columns_for(std::size_t count) {
    std::vector<std::int32_t> small;
    std::vector<std::int64_t> large;
    std::vector<float> real;
    for (std::size_t row = 0; row < count; ++row) {
        small.push_back(static_cast<std::int32_t>(row));
        large.push_back(10'000'000'000LL + static_cast<std::int64_t>(row));
        real.push_back(static_cast<float>(row) + 0.5F);
    }
    std::vector<ColumnBatch::Column> columns;
    columns.emplace_back(std::move(small));
    columns.emplace_back(std::move(large));
    columns.emplace_back(std::move(real));
    return ColumnBatch(std::move(columns));
}

EmbeddingBuffer embeddings_for(std::size_t count, std::size_t dimension = 3) {
    std::vector<float> values;
    for (std::size_t row = 0; row < count; ++row) {
        for (std::size_t feature = 0; feature < dimension; ++feature) {
            values.push_back(static_cast<float>(row * 10 + feature));
        }
    }
    return EmbeddingBuffer(count, dimension, std::move(values));
}

void valid_columns() {
    const auto columns = columns_for(3);
    check(columns.row_count() == 3 && columns.column_count() == 3, "column shape");
    check(columns.column<std::int32_t>(0)[2] == 2, "int32 values");
    check(columns.column<std::int64_t>(1)[2] == 10'000'000'002LL, "int64 values retain width");
    check(columns.column<float>(2)[2] == 2.5F, "float values");
    check(columns.column<float>(2).size() == 3, "column span length");

    const ColumnBatch extremes({std::vector<std::int32_t>{std::numeric_limits<std::int32_t>::min()},
                                std::vector<std::int64_t>{std::numeric_limits<std::int64_t>::max()},
                                std::vector<float>{-1.5F}});
    check(extremes.column<std::int32_t>(0)[0] == std::numeric_limits<std::int32_t>::min(), "signed int32");
    check(extremes.column<std::int64_t>(1)[0] == std::numeric_limits<std::int64_t>::max(), "int64 maximum");
    check(extremes.column<float>(2)[0] == -1.5F, "negative float");
}

void invalid_columns() {
    expect_throw<std::invalid_argument>([] {
        const ColumnBatch invalid({std::vector<std::int32_t>{1, 2}, std::vector<float>{1}});
    });
    expect_throw<std::invalid_argument>([] {
        const ColumnBatch invalid({std::vector<float>{}, std::vector<std::int64_t>{1}});
    });
    const auto columns = columns_for(2);
    expect_throw<std::invalid_argument>([&] { (void)columns.column<float>(0); });
    expect_throw<std::invalid_argument>([&] { (void)columns.column<std::int32_t>(1); });
    expect_throw<std::invalid_argument>([&] { (void)columns.column<std::int64_t>(2); });
    expect_throw<std::out_of_range>([&] { (void)columns.column<float>(3); });
    expect_throw<std::out_of_range>([&] {
        (void)columns.column<float>(std::numeric_limits<std::size_t>::max());
    });
}

void empty_inputs() {
    const ColumnBatch columns({});
    const EmbeddingBuffer embeddings(0, 3, {});
    check(columns.row_count() == 0 && columns.column_count() == 0, "no columns means zero rows");
    expect_throw<std::out_of_range>([&] { (void)columns.column<float>(0); });
    expect_throw<std::out_of_range>([&] { (void)embeddings.row(0); });
    const auto batch = make_batch(columns, embeddings, 0, 4);
    check(batch.row_count() == 0 && batch.row_offset() == 0, "empty dataset yields empty batch");
    expect_throw<std::out_of_range>([&] { (void)batch.global_row(0); });
    expect_throw<std::out_of_range>([&] { (void)batch.embedding_row(0); });
    expect_throw<std::out_of_range>([&] { (void)batch.column<float>(0); });
    expect_throw<std::out_of_range>([&] { (void)make_batch(columns, embeddings, 1, 4); });
    expect_throw<std::invalid_argument>([&] { (void)make_batch(columns, embeddings, 0, 0); });

    const auto typed_columns = columns_for(0);
    const auto typed_batch = make_batch(typed_columns, embeddings, 0, 2);
    check(typed_batch.column<std::int32_t>(0).empty(), "empty typed column");
    check(typed_batch.column<std::int64_t>(1).empty(), "empty int64 column");
    check(typed_batch.column<float>(2).empty(), "empty float column");
    check(typed_batch.column_count() == 3 && typed_batch.dimension() == 3, "empty batch retains shape");
    expect_throw<std::invalid_argument>([&] { (void)typed_batch.column<float>(0); });
}

void embedding_layout() {
    const auto embeddings = embeddings_for(4);
    check(embeddings.row_count() == 4 && embeddings.dimension() == 3, "embedding shape");
    for (std::size_t row = 0; row < 4; ++row) {
        const auto values = embeddings.row(row);
        check(values.size() == 3, "fixed embedding dimension");
        check(values.data() == embeddings.row(0).data() + row * 3, "contiguous row-major storage");
        for (std::size_t feature = 0; feature < 3; ++feature) {
            check(values[feature] == static_cast<float>(row * 10 + feature), "embedding feature indexing");
        }
    }
    expect_throw<std::out_of_range>([&] { (void)embeddings.row(4); });
    expect_throw<std::out_of_range>([&] { (void)embeddings.row(std::numeric_limits<std::size_t>::max()); });
    const auto scalar = embeddings_for(2, 1);
    check(scalar.row(1).size() == 1 && scalar.row(1)[0] == 10.0F, "dimension one");
}

void invalid_embeddings() {
    expect_throw<std::invalid_argument>([] { const EmbeddingBuffer invalid(0, 0, {}); });
    expect_throw<std::invalid_argument>([] { const EmbeddingBuffer invalid(2, 0, {}); });
    expect_throw<std::invalid_argument>([] { const EmbeddingBuffer invalid(2, 3, {1, 2, 3, 4, 5}); });
    expect_throw<std::invalid_argument>([] { const EmbeddingBuffer invalid(1, 2, {1, 2, 3}); });
    expect_throw<std::invalid_argument>([] { const EmbeddingBuffer invalid(0, 2, {1}); });
    expect_throw<std::invalid_argument>([] { const EmbeddingBuffer invalid(1, 2, {}); });
}

void overflow_guards() {
    constexpr auto maximum = std::numeric_limits<std::size_t>::max();
    expect_throw<std::overflow_error>([] { const EmbeddingBuffer invalid(maximum, 2, {}); });
    expect_throw<std::overflow_error>([] { const EmbeddingBuffer invalid(2, maximum, {}); });
    expect_throw<std::overflow_error>([] { const EmbeddingBuffer invalid(maximum / 3 + 1, 3, {}); });
    // Large non-overflowing shapes fail for size mismatch, not multiplication overflow.
    expect_throw<std::invalid_argument>([] { const EmbeddingBuffer invalid(maximum, 1, {}); });
    expect_throw<std::invalid_argument>([] { const EmbeddingBuffer invalid(1, maximum, {}); });
    expect_throw<std::invalid_argument>([] { const EmbeddingBuffer invalid(maximum / 2, 2, {}); });
    const EmbeddingBuffer empty(0, maximum, {});
    check(empty.dimension() == maximum && empty.row_count() == 0, "zero product is valid");
    expect_throw<std::out_of_range>([&] { (void)empty.row(maximum); });

    const auto columns = columns_for(3);
    const auto embeddings = embeddings_for(3);
    expect_throw<std::out_of_range>([&] { const BatchView invalid(columns, embeddings, 1, maximum); });
    expect_throw<std::out_of_range>([&] { const BatchView invalid(columns, embeddings, maximum, 2); });
    expect_throw<std::out_of_range>([&] { (void)make_batch(columns, embeddings, maximum, 2); });
    const auto batch = make_batch(columns, embeddings, 1, maximum);
    check(batch.row_count() == 2 && batch.global_row(1) == 2, "huge batch size clamps without addition");
    expect_throw<std::out_of_range>([&] { (void)batch.global_row(maximum); });
    expect_throw<std::out_of_range>([&] { (void)batch.embedding_row(maximum); });
}

void single_row() {
    const auto columns = columns_for(1);
    const auto embeddings = embeddings_for(1);
    for (const auto size : std::array<std::size_t, 3>{1, 2, 100}) {
        const auto batch = make_batch(columns, embeddings, 0, size);
        check(batch.row_count() == 1 && batch.global_row(0) == 0, "single row with any positive batch size");
        check(batch.column<float>(2)[0] == 0.5F, "single-row numeric value");
        check(batch.embedding_row(0)[2] == 2.0F, "single-row embedding value");
        expect_throw<std::out_of_range>([&] { (void)batch.embedding_row(1); });
    }
    const auto end = make_batch(columns, embeddings, 1, 1);
    check(end.row_count() == 0 && end.row_offset() == 1, "single-row end batch");
}

void check_partition(std::size_t rows, std::size_t batch_size, std::size_t expected_batches,
                     std::size_t expected_last_size) {
    const auto columns = columns_for(rows);
    const auto embeddings = embeddings_for(rows);
    std::size_t offset = 0;
    std::size_t batches = 0;
    std::size_t last_size = 0;
    while (offset < rows) {
        const auto batch = make_batch(columns, embeddings, offset, batch_size);
        check(batch.row_offset() == offset, "dataset-global batch offset");
        check(batch.row_count() > 0 && batch.row_count() <= batch_size, "bounded batch size");
        check(batch.column_count() == 3 && batch.dimension() == 3, "view shape");
        const auto small = batch.column<std::int32_t>(0);
        const auto large = batch.column<std::int64_t>(1);
        const auto real = batch.column<float>(2);
        check(small.size() == batch.row_count() && large.size() == small.size() && real.size() == small.size(),
              "all column views have batch length");
        for (std::size_t local = 0; local < batch.row_count(); ++local) {
            const auto global = offset + local;
            check(batch.global_row(local) == global, "local to global row mapping");
            check(small[local] == static_cast<std::int32_t>(global), "int32 row alignment");
            check(large[local] == 10'000'000'000LL + static_cast<std::int64_t>(global), "int64 row alignment");
            check(real[local] == static_cast<float>(global) + 0.5F, "float row alignment");
            for (std::size_t feature = 0; feature < batch.dimension(); ++feature) {
                check(batch.embedding_row(local)[feature] == static_cast<float>(global * 10 + feature),
                      "embedding and numeric rows align");
            }
        }
        expect_throw<std::out_of_range>([&] { (void)batch.global_row(batch.row_count()); });
        last_size = batch.row_count();
        offset += batch.row_count();
        ++batches;
    }
    check(offset == rows && batches == expected_batches && last_size == expected_last_size,
          "partition covers dataset exactly once, with expected tail");
    const auto end = make_batch(columns, embeddings, rows, batch_size);
    check(end.row_count() == 0 && end.row_offset() == rows, "empty terminal batch");
    check(end.column<float>(2).empty(), "empty terminal column span");
    expect_throw<std::out_of_range>([&] { (void)end.global_row(0); });
}

void batch_partitions() {
    check_partition(6, 2, 3, 2);
    check_partition(7, 3, 3, 1);
    check_partition(5, 3, 2, 2);
    check_partition(3, 10, 1, 3);
    check_partition(4, 4, 1, 4);
    check_partition(4, 1, 4, 1);
    check_partition(0, 3, 0, 0);
}

void invalid_batch_ranges() {
    const auto columns = columns_for(4);
    const auto embeddings = embeddings_for(4);
    const auto short_embeddings = embeddings_for(3);
    expect_throw<std::invalid_argument>([&] { const BatchView invalid(columns, short_embeddings, 0, 1); });
    expect_throw<std::invalid_argument>([&] { const BatchView invalid(columns, short_embeddings, 0, 0); });
    expect_throw<std::invalid_argument>([&] { (void)make_batch(columns, short_embeddings, 0, 2); });
    expect_throw<std::invalid_argument>([&] { (void)make_batch(columns, embeddings, 0, 0); });
    expect_throw<std::out_of_range>([&] { (void)make_batch(columns, embeddings, 5, 2); });
    expect_throw<std::out_of_range>([&] { const BatchView invalid(columns, embeddings, 5, 0); });
    expect_throw<std::out_of_range>([&] { const BatchView invalid(columns, embeddings, 3, 2); });
    expect_throw<std::out_of_range>([&] { const BatchView invalid(columns, embeddings, 4, 1); });
    const BatchView middle(columns, embeddings, 2, 0);
    check(middle.column<float>(2).empty(), "empty interior range is valid");
    expect_throw<std::out_of_range>([&] { (void)middle.embedding_row(0); });
    const BatchView slice(columns, embeddings, 1, 2);
    check(slice.global_row(0) == 1 && slice.global_row(1) == 2, "exact subrange");
    expect_throw<std::out_of_range>([&] { (void)slice.global_row(2); });
    expect_throw<std::out_of_range>([&] { (void)slice.column<float>(3); });
    expect_throw<std::invalid_argument>([&] { (void)slice.column<float>(0); });
}

void storage_transfer_and_views() {
    std::vector<std::int32_t> values{5, 6, 7};
    const auto* original_column = values.data();
    std::vector<ColumnBatch::Column> storage;
    storage.emplace_back(std::move(values));
    const ColumnBatch columns(std::move(storage));
    check(columns.column<std::int32_t>(0).data() == original_column, "column allocation is transferred");

    std::vector<float> embedding_values{1, 2, 3, 4, 5, 6};
    const auto* original_embeddings = embedding_values.data();
    const EmbeddingBuffer embeddings(3, 2, std::move(embedding_values));
    check(embeddings.row(0).data() == original_embeddings, "embedding allocation is transferred");
    const auto batch = make_batch(columns, embeddings, 1, 2);
    check(batch.column<std::int32_t>(0).data() == original_column + 1, "column view shares backing storage");
    check(batch.embedding_row(0).data() == original_embeddings + 2, "embedding view shares backing storage");
    const auto copy = batch;
    check(copy.column<std::int32_t>(0).data() == batch.column<std::int32_t>(0).data(), "view copy shares columns");
    check(copy.embedding_row(1).data() == batch.embedding_row(1).data(), "view copy shares embeddings");
    // A span may outlive the view itself when the backing owner remains alive.
    const auto span = make_batch(columns, embeddings, 0, 1).embedding_row(0);
    check(span[1] == 2.0F, "temporary batch view can safely return owner-backed span");
}

void owner_moves() {
    auto source_columns = columns_for(2);
    auto source_embeddings = embeddings_for(2);
    const ColumnBatch columns(std::move(source_columns));
    const EmbeddingBuffer embeddings(std::move(source_embeddings));
    check(columns.row_count() == 2 && columns.column<std::int64_t>(1)[1] == 10'000'000'001LL,
          "column owner move retains data");
    check(embeddings.row_count() == 2 && embeddings.row(1)[2] == 12.0F, "embedding owner move retains data");
    check(source_columns.row_count() == 0 && source_columns.column_count() == 0, "moved-from columns empty");
    check(source_embeddings.row_count() == 0 && source_embeddings.dimension() == 3, "moved-from embeddings valid");
    const auto empty = make_batch(source_columns, source_embeddings, 0, 1);
    check(empty.row_count() == 0, "moved-from owners support empty view");
    expect_throw<std::out_of_range>([&] { (void)source_columns.column<float>(0); });
    expect_throw<std::out_of_range>([&] { (void)source_embeddings.row(0); });
    const auto batch = make_batch(columns, embeddings, 1, 1);
    check(batch.column<std::int32_t>(0)[0] == 1 && batch.embedding_row(0)[0] == 10.0F,
          "moved-to owners support aligned views");
}

} // namespace

int main() {
    const std::array tests{
        std::pair{"valid columns", valid_columns},
        std::pair{"invalid columns", invalid_columns},
        std::pair{"empty inputs", empty_inputs},
        std::pair{"embedding layout", embedding_layout},
        std::pair{"invalid embeddings", invalid_embeddings},
        std::pair{"overflow guards", overflow_guards},
        std::pair{"single row", single_row},
        std::pair{"batch partitions and row alignment", batch_partitions},
        std::pair{"invalid batch ranges", invalid_batch_ranges},
        std::pair{"storage transfer and non-owning views", storage_transfer_and_views},
        std::pair{"owner moves", owner_moves},
    };
    std::size_t failed = 0;
    for (const auto& [name, test] : tests) {
        try {
            test();
            std::cout << "PASS: " << name << '\n';
        } catch (const std::exception& error) {
            ++failed;
            std::cerr << "FAIL: " << name << ": " << error.what() << '\n';
        } catch (...) {
            ++failed;
            std::cerr << "FAIL: " << name << ": unknown exception\n";
        }
    }
    std::cout << (tests.size() - failed) << '/' << tests.size() << " test groups passed\n";
    return failed == 0 ? 0 : 1;
}
